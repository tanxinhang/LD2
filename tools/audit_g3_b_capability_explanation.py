#!/usr/bin/env python
"""G3-B: compare cap-aware bistatic capability with distance tail proxies.

This is a development/shadow audit.  Frames within an episode are correlated;
only episode-level summaries are statistical units, while frame correlations
are reported as mechanism diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.seed_stratification import geometry_difficulty  # noqa: E402
from uav_isac.coordination.scale_capability import (  # noqa: E402
    cap_aware_sensing_budget,
    characterize_bistatic_scale_capability,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def _rank_average_ties(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * ((start + 1) + stop)
        for position in range(start, stop):
            ranks[order[position]] = rank
        start = stop
    return ranks


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    x_mean = math.fsum(x) / len(x)
    y_mean = math.fsum(y) / len(y)
    xc = [value - x_mean for value in x]
    yc = [value - y_mean for value in y]
    xx = math.fsum(value * value for value in xc)
    yy = math.fsum(value * value for value in yc)
    if xx == 0.0 or yy == 0.0:
        return None
    return math.fsum(a * b for a, b in zip(xc, yc)) / math.sqrt(xx * yy)


def spearman(x: list[float], y: list[float]) -> float | None:
    return _pearson(_rank_average_ties(x), _rank_average_ties(y))


def audit(trace_path: Path, config_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    seed_order = list(dict.fromkeys(int(seed) for seed in seeds))
    rows: list[dict[str, object]] = []
    max_single_owner_violation = 0.0
    max_owner_relaxed_violation = 0.0
    for index in range(seeds.size):
        coefficient = per_watt_deflection_tensor_from_observables(
            np.asarray(data["privileged_alpha"][index], dtype=np.float64),
            np.asarray(data["privileged_g_dd"][index], dtype=np.float64),
            np.asarray(data["privileged_chi_rep"][index], dtype=np.float64),
            T_sym=float(cfg.otfs.T_sym), M=int(cfg.otfs.M), N=int(cfg.otfs.N),
            kT=float(cfg.channel.kT), bandwidth_hz=float(cfg.otfs.B),
            noise_figure_db=float(cfg.channel.NF),
            g_tx_dbi=float(cfg.otfs.g_tx_dBi),
            g_rx_dbi=float(cfg.otfs.g_rx_dBi),
            n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
            use_swerling=bool(cfg.channel.use_swerling),
        )
        rate = np.asarray(data["outgoing_rate"][index], dtype=np.int64)
        mask = np.asarray(data["outgoing_token_mask"][index], dtype=bool)
        active = (rate > 0) & np.any(mask, axis=1)
        fraction = np.clip(
            np.asarray(data["comm_fraction"][index], dtype=np.float64), 0.0, 1.0)
        comm_power = np.where(active, fraction * float(cfg.uav.P_isac_total), 0.0)
        budget = cap_aware_sensing_budget(
            np.full(int(cfg.scenario.K), float(cfg.uav.P_isac_total)),
            comm_power,
            np.full(int(cfg.scenario.K), float(cfg.uav.P_sense_max)),
        )
        capability = characterize_bistatic_scale_capability(coefficient, budget)
        max_single_owner_violation = max(
            max_single_owner_violation,
            float(np.max(
                capability.best_single_pair_capability
                - capability.owner_consistent_target_capability)),
        )
        max_owner_relaxed_violation = max(
            max_owner_relaxed_violation,
            float(np.max(
                capability.owner_consistent_target_capability
                - capability.relaxed_target_ceiling)),
        )
        geometry = geometry_difficulty(
            np.asarray(data["uav_positions"][index], dtype=np.float64)[:, :2],
            np.asarray(data["target_states"][index], dtype=np.float64)[:, :2],
        )
        rows.append({
            "seed": int(seeds[index]),
            "frame": int(frames[index]),
            "actual_worst_pd": float(np.min(data["physical_pd"][index])),
            "actual_pd_q": np.asarray(
                data["physical_pd"][index], dtype=np.float64).tolist(),
            "single_pair_bottleneck": capability.geometry_bottleneck,
            "owner_capability_bottleneck": capability.owner_consistent_bottleneck,
            "relaxed_capability_bottleneck": capability.relaxed_worst_ceiling,
            "negative_worst_nearest_m": -float(geometry["worst_nearest_m"]),
            "negative_worst_second_nearest_m": -float(geometry["worst_second_nearest_m"]),
            "negative_matching_bottleneck_m": -float(geometry["bottleneck_matching_m"]),
        })

    predictors = [
        "single_pair_bottleneck",
        "owner_capability_bottleneck",
        "relaxed_capability_bottleneck",
        "negative_worst_nearest_m",
        "negative_worst_second_nearest_m",
        "negative_matching_bottleneck_m",
    ]
    frame_outcome = [float(row["actual_worst_pd"]) for row in rows]
    frame_spearman = {
        key: spearman([float(row[key]) for row in rows], frame_outcome)
        for key in predictors
    }

    episode_rows = []
    for seed in seed_order:
        selected = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: int(row["frame"]),
        )[-20:]
        episode_rows.append({
            "seed": seed,
            "actual_worst_pd": float(np.min(np.mean(np.asarray([
                row["actual_pd_q"] for row in selected
            ], dtype=np.float64), axis=0))),
            **{
                key: float(np.mean([float(row[key]) for row in selected]))
                for key in predictors
            },
        })
    episode_outcome = [float(row["actual_worst_pd"]) for row in episode_rows]
    episode_spearman = {
        key: spearman([float(row[key]) for row in episode_rows], episode_outcome)
        for key in predictors
    }
    owner_frame = frame_spearman["owner_capability_bottleneck"]
    distance_frame = max(
        frame_spearman[key] if frame_spearman[key] is not None else -1.0
        for key in (
            "negative_worst_nearest_m",
            "negative_worst_second_nearest_m",
            "negative_matching_bottleneck_m",
        )
    )
    return {
        "gate": "G3-B",
        "scope": "development_shadow_five_exposed_seeds",
        "trace": str(trace_path),
        "config": str(config_path),
        "frames": len(rows),
        "episodes": len(episode_rows),
        "statistical_unit": "episode; frame correlations are diagnostic only",
        "frame_spearman": frame_spearman,
        "episode_spearman": episode_spearman,
        "episode_rows": episode_rows,
        "capability_order_checks": {
            "max_single_minus_owner": max_single_owner_violation,
            "max_owner_minus_relaxed": max_owner_relaxed_violation,
            "pass": bool(
                max_single_owner_violation <= 1.0e-12
                and max_owner_relaxed_violation <= 1.0e-12),
        },
        "directional_falsification": {
            "owner_frame_spearman": owner_frame,
            "best_distance_frame_spearman": distance_frame,
            "owner_exceeds_best_distance": bool(
                owner_frame is not None and owner_frame > distance_frame),
            "note": "exploratory mechanism check, not a model-selection gate",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
