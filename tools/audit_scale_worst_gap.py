#!/usr/bin/env python
"""Decompose cross-scale worst-P_D gaps without treating them as causal A/B."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.run_g2_1a_bridge import SYSTEMS, historical_seeds  # noqa: E402


CONFIG_NAMES = {
    "4x4": "config/exp_800_q4_architecture_v2_maxmin_local_fusion_fullgraph_hold5_gate.yaml",
    "8x8_v3c0": "config/exp_800_k8q8_analytical_l0l1_movement_lex_candidates_blind_lookahead40_indep.yaml",
    "6x6_v3c0": "config/exp_800_k6q6_analytical_l0l1_movement_lex_candidates_v2_blind_indep.yaml",
    "6x6_ce": "config/exp_800_k6q6_analytical_l0l1_movement_lex_candidates_v2_blind_indep.yaml",
}
BANK_METADATA = {
    "4x4": "config/stratified_seeds_800_q4_v2.json",
    "8x8_v3c0": "config/stratified_seeds_1130_k8q8_v2.json",
    "6x6_v3c0": "config/stratified_seeds_980_k6q6_blind.json",
    "6x6_ce": "config/stratified_seeds_980_k6q6_blind.json",
}
CAP_AWARE_ORACLE_FILES = {
    "4x4": "results/_scale_worst_oracle_capaware_q4_seed643/paired_eval.csv",
    "6x6_v3c0": "results/_scale_worst_oracle_capaware_k6_seed237/paired_eval.csv",
    "8x8_v3c0": "results/_scale_worst_oracle_capaware_k8_seed248/paired_eval.csv",
}


def _literal(value: str):
    return ast.literal_eval(value)


def _scalar_list(value: str) -> float:
    parsed = _literal(value)
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError("expected one episode per isolated worker")
    return float(parsed[0])


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    x_mean = math.fsum(float(value) for value in x) / len(x)
    y_mean = math.fsum(float(value) for value in y) / len(y)
    x_centered = [float(value) - x_mean for value in x]
    y_centered = [float(value) - y_mean for value in y]
    x_energy = math.fsum(value * value for value in x_centered)
    y_energy = math.fsum(value * value for value in y_centered)
    if x_energy == 0.0 or y_energy == 0.0:
        return None
    covariance = math.fsum(
        a * b for a, b in zip(x_centered, y_centered))
    return covariance / math.sqrt(x_energy * y_energy)


def poisson_worst_nearest_median(*, density_m2: float, targets: int) -> float:
    """Infinite-plane Poisson approximation: P(max_q R_q <= r)=F_R(r)^Q."""
    if density_m2 <= 0.0 or targets <= 0:
        raise ValueError("density and targets must be positive")
    single_cdf_at_median_max = 0.5 ** (1.0 / targets)
    return math.sqrt(
        -math.log(1.0 - single_cdf_at_median_max) / (math.pi * density_m2))


def _read_workers(name: str) -> list[dict[str, object]]:
    directory = ROOT / SYSTEMS[name].output
    episodes = []
    for worker in sorted(directory.glob("seed_*")):
        path = worker / "paired_eval.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), None)
        if row is None:
            raise ValueError(f"empty worker result: {path}")
        target_power = np.asarray(_literal(row["eval_isac_per_target_power_w"]), dtype=np.float64)
        per_target = np.asarray(_literal(row["eval_per_target"]), dtype=np.float64)
        nearest = np.asarray(_literal(row["eval_per_target_nearest_uav_distance_m"]), dtype=np.float64)
        total_power = float(np.sum(target_power))
        episodes.append({
            "seed": int(_literal(row["eval_episode_seeds"])[0]),
            "worst": float(row["eval_worst_P_D"]),
            "steady": float(row["eval_steady_P_D"]),
            "worst_nearest_m": float(row["eval_worst_nearest_uav_distance_m"]),
            "mean_nearest_m": float(row["eval_mean_nearest_uav_distance_m"]),
            "commitment_coverage": _scalar_list(row["eval_episode_commitment_coverage"]),
            "p0_target_coverage": _scalar_list(row["eval_episode_p0_target_coverage"]),
            "comm_bits_per_frame": float(row["eval_comm_bits_per_frame"]),
            "delivery_rate": float(row["eval_comm_delivery_rate"]),
            "deadline_violation_rate": float(row["eval_comm_deadline_violation_rate"]),
            "target_power_hhi": float(np.sum((target_power / total_power) ** 2)),
            "target_power_max_share": float(np.max(target_power) / total_power),
            "targets_at_worst": int(np.count_nonzero(per_target <= np.min(per_target) + 1e-9)),
            "per_target": per_target.tolist(),
            "per_target_nearest_m": nearest.tolist(),
        })
    if not episodes:
        raise FileNotFoundError(f"no isolated workers in {directory}")
    return sorted(episodes, key=lambda item: int(item["seed"]))


def _system_summary(name: str, episodes: list[dict[str, object]]) -> dict[str, object]:
    cfg = load_config(str(ROOT / CONFIG_NAMES[name]))
    side_x, side_y = map(float, cfg.scenario.region_size)
    k, q = int(cfg.scenario.K), int(cfg.scenario.Q)
    density = k / (side_x * side_y)
    mobility_radius = float(cfg.uav.v_max) * float(cfg.scenario.T) * float(cfg.scenario.dt)

    def values(key: str) -> list[float]:
        return [float(item[key]) for item in episodes]

    worst = values("worst")
    farthest = values("worst_nearest_m")
    all_target_pd = [float(value) for item in episodes for value in item["per_target"]]
    all_target_distance = [float(value) for item in episodes for value in item["per_target_nearest_m"]]
    analytical = bool(cfg.marl.analytical_sensing_power_enabled)
    with (ROOT / BANK_METADATA[name]).open(encoding="utf-8") as handle:
        metadata = json.load(handle)["seed_metadata"]
    bank_seeds = historical_seeds(ROOT / SYSTEMS[name].historical_csv)
    bank_geometry: dict[str, dict[str, float]] = {}
    for key in ("worst_nearest_m", "worst_second_nearest_m", "bottleneck_matching_m"):
        array = np.asarray(
            [metadata[str(seed)][key] for seed in bank_seeds], dtype=np.float64)
        bank_geometry[key] = {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "q90": float(np.quantile(array, 0.90)),
            "max": float(np.max(array)),
            "mean_over_region_side": float(np.mean(array) / max(side_x, side_y)),
        }
    initial_farthest = [
        float(metadata[str(item["seed"])]["worst_nearest_m"])
        for item in episodes]
    return {
        "episodes": len(episodes),
        "config": {
            "K": k,
            "Q": q,
            "region_m": [side_x, side_y],
            "uav_density_per_m2": density,
            "mobility_radius_m": mobility_radius,
            "mobility_radius_over_side": mobility_radius / max(side_x, side_y),
            "analytical_power": analytical,
            "analytical_movement": bool(cfg.marl.analytical_movement_enabled),
            "lookahead_frames": int(cfg.marl.analytical_movement_lookahead_frames),
            "poisson_worst_nearest_median_m": poisson_worst_nearest_median(
                density_m2=density, targets=q),
        },
        "performance": {
            "worst_mean": float(np.mean(worst)),
            "worst_std": float(np.std(worst)),
            "steady_mean": float(np.mean(values("steady"))),
            "worst_nearest_mean_m": float(np.mean(farthest)),
            "mean_nearest_mean_m": float(np.mean(values("mean_nearest_m"))),
            "initial_worst_nearest_mean_m": float(np.mean(initial_farthest)),
            "initial_to_final_worst_nearest_reduction_m": float(
                np.mean(np.asarray(initial_farthest) - np.asarray(farthest))),
        },
        "mechanisms": {
            "episode_corr_worst_vs_farthest_nearest": _correlation(worst, farthest),
            "target_corr_pd_vs_nearest_distance": _correlation(all_target_pd, all_target_distance),
            "commitment_coverage_mean": float(np.mean(values("commitment_coverage"))),
            "p0_target_coverage_mean": float(np.mean(values("p0_target_coverage"))),
            "target_power_hhi_mean": float(np.mean(values("target_power_hhi"))),
            "target_power_max_share_mean": float(np.mean(values("target_power_max_share"))),
            "targets_at_worst_mean": float(np.mean(values("targets_at_worst"))),
            "comm_bits_per_frame_mean": float(np.mean(values("comm_bits_per_frame"))),
            "delivery_rate_mean": float(np.mean(values("delivery_rate"))),
            "deadline_violation_rate_mean": float(np.mean(values("deadline_violation_rate"))),
            "episode_corr_worst_vs_initial_farthest_nearest": _correlation(
                worst, initial_farthest),
        },
        "exact_historical_100_seed_geometry": bank_geometry,
        "episodes_detail": episodes,
    }


def _paired_ce(baseline: list[dict[str, object]], ce: list[dict[str, object]]) -> list[dict[str, float | int]]:
    left = {int(item["seed"]): item for item in baseline}
    right = {int(item["seed"]): item for item in ce}
    if left.keys() != right.keys():
        raise ValueError("paired 6x6 seed sets differ")
    rows = []
    for seed in sorted(left):
        a, b = left[seed], right[seed]
        rows.append({
            "seed": seed,
            "delta_worst": float(b["worst"]) - float(a["worst"]),
            "delta_worst_nearest_m": float(b["worst_nearest_m"]) - float(a["worst_nearest_m"]),
            "delta_p0_target_coverage": float(b["p0_target_coverage"]) - float(a["p0_target_coverage"]),
            "delta_commitment_coverage": float(b["commitment_coverage"]) - float(a["commitment_coverage"]),
            "delta_deadline_violation_rate": float(b["deadline_violation_rate"]) - float(a["deadline_violation_rate"]),
        })
    return rows


def run_audit() -> dict[str, object]:
    raw = {name: _read_workers(name) for name in SYSTEMS}
    systems = {name: _system_summary(name, rows) for name, rows in raw.items()}
    paired = _paired_ce(raw["6x6_v3c0"], raw["6x6_ce"])
    cap_aware_oracles = {}
    for name, relative in CAP_AWARE_ORACLE_FILES.items():
        with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        cap_aware_oracles[name] = {
            "episode_seed": int(_literal(row["eval_episode_seeds"])[0]),
            "sampled_frames": int(float(row["eval_physical_oracle_frame_count"])),
            "episode_worst": float(row["eval_worst_P_D"]),
            "sampled_deployed_worst": float(row["eval_physical_oracle_deployed_worst"]),
            "pair_only_worst": float(row["eval_physical_oracle_pair_only_worst"]),
            "power_only_worst": float(row["eval_physical_oracle_power_only_worst"]),
            "joint_single_role_worst": float(row["eval_physical_oracle_single_worst"]),
            "full_duplex_worst": float(row["eval_physical_oracle_duplex_worst"]),
            "sensing_pa_cap_w_per_uav": 0.0251,
            "scope": "one_episode_five_sampled_frames_mechanism_probe",
        }
    density_values = [systems[name]["config"]["uav_density_per_m2"] for name in ("4x4", "6x6_v3c0", "8x8_v3c0")]
    return {
        "scope": "exploratory_five_episode_scale_audit_not_causal_or_formal",
        "comparability": {
            "K_over_Q_constant": True,
            "density_relative_spread": (max(density_values) - min(density_values)) / np.mean(density_values),
            "strict_scale_only_ab": False,
            "confounders": [
                "different episode seed banks",
                "historical 100-seed banks are not matched by initial geometric difficulty",
                "4x4 disables analytical power and movement while 6x6/8x8 enable them",
                "8x8 uses 40-frame movement lookahead while 6x6 uses zero",
                "different checkpoints and 6x6 CE communication student",
            ],
        },
        "theory": {
            "nearest_order_statistic": "F_Rmax(r)=F_R(r)^Q",
            "symmetric_bistatic_deflection_scaling": "D proportional to p/(R_tx^2 R_rx^2), approximately p/r^4",
            "implication": "constant density and K/Q do not imply a scale-invariant worst target",
        },
        "systems": systems,
        "paired_6x6_ce_minus_v3c0": paired,
        "cap_aware_physical_oracle_probes": cap_aware_oracles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
