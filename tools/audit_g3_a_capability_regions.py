#!/usr/bin/env python
"""G3-A shadow audit: certified capability-region attribution on trace frames.

The four regions use the deployed three-floor task directly.  A heuristic
joint-oracle miss is never interpreted as an infeasibility certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.scale_capability import (  # noqa: E402
    cap_aware_sensing_budget,
    characterize_bistatic_scale_capability,
    route_task_capability_shadow,
    task_detection_metrics,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import compute_detection_probabilities  # noqa: E402
from uav_isac.physical.feasibility_oracle import (  # noqa: E402
    solve_joint_pair_power_oracle,
)


def _coefficient(data: dict[str, np.ndarray], index: int, cfg) -> np.ndarray:
    return per_watt_deflection_tensor_from_observables(
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


def audit(trace_path: Path, config_path: Path, stride: int) -> dict[str, object]:
    if stride < 1:
        raise ValueError("stride must be positive")
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    floors = tuple(cfg.marl.task_constrained_qos_floors)
    p_fa = float(cfg.detection.P_FA)
    indices = [
        index for index, frame in enumerate(np.asarray(data["frame"], dtype=int))
        if int(frame) % stride == 0
    ]
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index in indices:
        coefficient = _coefficient(data, index, cfg)
        rate = np.asarray(data["outgoing_rate"][index], dtype=np.int64)
        mask = np.asarray(data["outgoing_token_mask"][index], dtype=bool)
        active = (rate > 0) & np.any(mask, axis=1)
        fraction = np.clip(
            np.asarray(data["comm_fraction"][index], dtype=np.float64), 0.0, 1.0)
        comm_power = np.where(active, fraction * float(cfg.uav.P_isac_total), 0.0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm_power, float(cfg.uav.P_sense_max))
        capability = characterize_bistatic_scale_capability(coefficient, budget)
        relaxed_pd = compute_detection_probabilities(
            capability.relaxed_target_ceiling, p_fa)

        # The present hardware PA cap is tighter than joint headroom in the
        # sampled configuration.  If that ceases to hold, the scalar-budget
        # oracle cannot represent heterogeneous residual budgets exactly.
        if not np.allclose(budget, budget[0], rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "joint oracle requires uniform residual budgets for this audit")
        oracle = solve_joint_pair_power_oracle(
            coefficient,
            P_FA=p_fa,
            total_power_w=float(cfg.uav.P_isac_total),
            communication_reserve_w=float(cfg.uav.P_isac_total - budget[0]),
            sensing_power_cap_w=float(cfg.uav.P_sense_max),
            target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
            reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
            full_duplex=False,
            alternating_iterations=6,
            random_starts=2,
            seed=int(data["seed"][index]) * 1000 + int(data["frame"][index]),
            fusion_mode="local_only",
        )
        fixed_pd = np.asarray(data["physical_pd"][index], dtype=np.float64)
        route = route_task_capability_shadow(
            fixed_pd, relaxed_pd, floors, joint_feasible_pd=oracle.P_D_q)
        rows.append({
            "seed": int(data["seed"][index]),
            "frame": int(data["frame"][index]),
            "region": route.region,
            "action": route.action,
            "certified": route.certified,
            "fixed": task_detection_metrics(fixed_pd, floors),
            "joint_witness": task_detection_metrics(oracle.P_D_q, floors),
            "relaxed_upper": task_detection_metrics(relaxed_pd, floors),
        })

    counts = {region: sum(row["region"] == region for row in rows)
              for region in ("I", "II", "III", "U")}
    by_seed = {}
    for seed in dict.fromkeys(int(row["seed"]) for row in rows):
        selected = [row for row in rows if int(row["seed"]) == seed]
        by_seed[str(seed)] = {
            region: sum(row["region"] == region for row in selected)
            for region in ("I", "II", "III", "U")
        }
    return {
        "gate": "G3-A",
        "scope": "development_shadow_exposed_seeds",
        "trace": str(trace_path),
        "config": str(config_path),
        "task_floors": {
            "rho_min": float(floors[0]), "rho_tail": float(floors[1]),
            "rho_avg": float(floors[2]), "k": int(floors[3]),
        },
        "sampling": {"frame_modulo_stride": stride, "frames": len(rows)},
        "region_counts": counts,
        "region_counts_by_seed": by_seed,
        "certification_semantics": {
            "I": "deployed fixed structure is feasible",
            "II": "same-geometry joint heuristic supplies a feasible witness",
            "III": "componentwise relaxed same-geometry upper bound is infeasible",
            "U": "unresolved; heuristic miss is not an infeasibility proof",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.stride)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
