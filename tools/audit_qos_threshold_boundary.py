#!/usr/bin/env python
"""Audit exact same-geometry QoS feasibility on independent trace episodes.

The audit freezes one resolved geometry per episode (the weakest recorded
frame by default) and its recorded U2U communication-power reservation. It
then compares the recorded policy with:

1. exact max--min power repair on the recorded receiver-owner structure; and
2. an exact single-role joint structure/power QoS boundary obtained by MILP
   threshold feasibility and monotone bisection.

The second result is a pre-transport architecture reference.  It preserves the
recorded communication reserve but does not claim that a distributed protocol
can communicate or compute the joint decision within that reserve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _ordered_unique,
    _recorded_power,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.qos_threshold_feasibility import (  # noqa: E402
    solve_maxmin_qos_boundary_bisection,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)


def _coefficient(
    data: dict[str, np.ndarray], row: int, cfg: object,
) -> np.ndarray:
    return per_watt_deflection_tensor_from_observables(
        np.asarray(data["privileged_alpha"][row], dtype=np.float64),
        np.asarray(data["privileged_g_dd"][row], dtype=np.float64),
        np.asarray(data["privileged_chi_rep"][row], dtype=np.float64),
        T_sym=float(cfg.otfs.T_sym),
        M=int(cfg.otfs.M),
        N=int(cfg.otfs.N),
        kT=float(cfg.channel.kT),
        bandwidth_hz=float(cfg.otfs.B),
        noise_figure_db=float(cfg.channel.NF),
        g_tx_dbi=float(cfg.otfs.g_tx_dBi),
        g_rx_dbi=float(cfg.otfs.g_rx_dBi),
        n_cpi=int(cfg.otfs.n_cpi),
        g_min=float(cfg.detection.g_min),
        use_swerling=bool(cfg.channel.use_swerling),
    )


def _episode_bootstrap(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(int(seed))
    means = np.asarray([
        np.mean(data[rng.integers(0, data.size, size=data.size)])
        for _ in range(max(1, int(samples)))
    ])
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def audit(
    trace_path: Path,
    config_path: Path,
    *,
    seed_limit: int,
    probability_tolerance: float,
    max_iterations: int,
    time_limit_s: float,
    bootstrap_samples: int,
    frame_selection: str = "worst_recorded_resolved",
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    pair_limit = int(np.asarray(
        data["target_pair_limit"]).reshape(-1)[0])
    reports_per_receiver = int(np.asarray(
        data["reports_per_receiver"]).reshape(-1)[0])
    if K != int(cfg.scenario.K) or Q != int(cfg.scenario.Q):
        raise ValueError("trace dimensions do not match the supplied config")

    rows: list[dict[str, object]] = []
    skipped: list[tuple[int, int, str]] = []
    selection = str(frame_selection).strip().lower()
    if selection not in {"worst_recorded_resolved", "final_resolved"}:
        raise ValueError(
            "frame_selection must be worst_recorded_resolved or final_resolved")
    started = perf_counter()
    for seed in seed_order:
        candidates = np.flatnonzero(resolved & (seeds == seed))
        if not len(candidates):
            raise ValueError(f"seed {seed} has no resolved frame")
        if selection == "final_resolved":
            row = int(candidates[np.argmax(frames[candidates])])
        else:
            candidate_worst = np.min(np.asarray(
                data["physical_pd"][candidates], dtype=np.float64), axis=1)
            row = int(candidates[np.argmin(candidate_worst)])
        comm_power, sensing_power, power_error = _recorded_power(data, row)
        coefficient = _coefficient(data, row, cfg)
        pair = np.asarray(data["teacher_pair"][row], dtype=bool)
        selected = tuple(
            tuple(int(value) for value in edge)
            for edge in np.argwhere(pair)
        )
        recorded_pd = np.asarray(data["physical_pd"][row], dtype=np.float64)
        fixed_gain, owners = fixed_owner_gain_matrix(coefficient, selected)
        sensing_budget = 1.0 - comm_power
        fixed = solve_fixed_structure_maxmin_power_lp(
            fixed_gain, sensing_budget)
        fixed_pd = compute_detection_probabilities(fixed.deflection, p_fa)
        try:
            boundary = solve_maxmin_qos_boundary_bisection(
                coefficient,
                sensing_budget,
                p_fa=p_fa,
                target_pair_limit=pair_limit,
                reports_per_receiver=reports_per_receiver,
                initial_feasible_pd=max(
                    p_fa, float(np.min(fixed_pd)) - 1.0e-8),
                probability_tolerance=float(probability_tolerance),
                max_iterations=int(max_iterations),
                time_limit_s=float(time_limit_s),
            )
        except RuntimeError as exc:
            # Degenerate frame (e.g. a target with no coverage at the recorded
            # geometry): the exact MILP cannot certify even the initial floor.
            # Record and skip rather than aborting the whole audit.
            skipped.append((int(seed), int(frames[row]), str(exc)))
            continue
        witness = boundary.best_feasible
        rows.append({
            "seed": int(seed),
            "frame": int(frames[row]),
            "recorded_worst": float(np.min(recorded_pd)),
            "recorded_weak3": float(np.mean(np.sort(recorded_pd)[:3])),
            "recorded_steady": float(np.mean(recorded_pd)),
            "fixed_structure_power_worst": float(np.min(fixed_pd)),
            "joint_structure_power_lower": float(boundary.lower_pd),
            "joint_structure_power_upper": float(boundary.upper_pd),
            "joint_boundary_width": float(
                boundary.upper_pd - boundary.lower_pd),
            "joint_exact_infeasible_upper": bool(
                boundary.exact_infeasible_upper),
            "recorded_to_fixed_gain": float(
                np.min(fixed_pd) - np.min(recorded_pd)),
            "fixed_to_joint_gain": float(
                boundary.lower_pd - np.min(fixed_pd)),
            "recorded_to_joint_gain": float(
                boundary.lower_pd - np.min(recorded_pd)),
            "joint_qos_feasible_060": bool(boundary.lower_pd >= 0.60),
            "fixed_qos_feasible_060": bool(np.min(fixed_pd) >= 0.60),
            "recorded_qos_feasible_060": bool(np.min(recorded_pd) >= 0.60),
            "communication_power_w": comm_power.tolist(),
            "recorded_power_balance_error_w": float(power_error),
            "fixed_owner": owners.tolist(),
            "joint_owner": witness.receiver_owner.tolist(),
            "joint_tx_role": witness.tx_role.astype(int).tolist(),
            "joint_rx_role": witness.rx_role.astype(int).tolist(),
            "joint_selected_set": [list(edge) for edge in witness.selected_set],
            "joint_sensing_power_w": witness.sensing_power_w.tolist(),
            "joint_target_pd": witness.target_pd.tolist(),
            "joint_solve_time_s": float(boundary.solve_time_s),
            "joint_iterations": int(boundary.iterations),
        })

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows])

    summary: dict[str, object] = {
        "schema_version": 1,
        "scope": (
            "exact same-geometry single-role local-fusion QoS boundary with "
            "recorded per-UAV communication reserve; pre-transport reference "
            "without distributed commit authority"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "seed_count": int(len(rows)),
        "seed_order": [int(seed) for seed in seed_order],
        "skipped_seed_count": len(skipped),
        "skipped_seeds": skipped,
        "num_uavs": K,
        "num_targets": Q,
        "p_fa": p_fa,
        "target_pair_limit": pair_limit,
        "reports_per_receiver": reports_per_receiver,
        "probability_tolerance": float(probability_tolerance),
        "frame_selection": selection,
        "time_limit_s_per_milp": float(time_limit_s),
        "recorded_mean_worst": float(np.mean(values("recorded_worst"))),
        "fixed_structure_power_mean_worst": float(np.mean(
            values("fixed_structure_power_worst"))),
        "joint_structure_power_mean_lower": float(np.mean(
            values("joint_structure_power_lower"))),
        "joint_structure_power_mean_upper": float(np.mean(
            values("joint_structure_power_upper"))),
        "recorded_qos_feasible_rate": float(np.mean([
            bool(row["recorded_qos_feasible_060"]) for row in rows
        ])),
        "fixed_structure_power_qos_feasible_rate": float(np.mean([
            bool(row["fixed_qos_feasible_060"]) for row in rows
        ])),
        "joint_structure_power_qos_feasible_rate_lower": float(np.mean([
            bool(row["joint_qos_feasible_060"]) for row in rows
        ])),
        "mean_recorded_to_fixed_gain": float(np.mean(
            values("recorded_to_fixed_gain"))),
        "mean_fixed_to_joint_gain": float(np.mean(
            values("fixed_to_joint_gain"))),
        "mean_recorded_to_joint_gain": float(np.mean(
            values("recorded_to_joint_gain"))),
        "recorded_to_joint_gain_bootstrap95": _episode_bootstrap(
            values("recorded_to_joint_gain"),
            samples=int(bootstrap_samples), seed=20260813,
        ),
        "mean_joint_boundary_width": float(np.mean(
            values("joint_boundary_width"))),
        "all_joint_upper_bounds_proven_infeasible": bool(all(
            bool(row["joint_exact_infeasible_upper"]) for row in rows
        )),
        "mean_joint_solve_time_s": float(np.mean(
            values("joint_solve_time_s"))),
        "total_audit_time_s": float(perf_counter() - started),
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-limit", type=int, default=5)
    parser.add_argument("--probability-tolerance", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--time-limit-s", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--frame-selection",
        choices=("worst_recorded_resolved", "final_resolved"),
        default="worst_recorded_resolved",
    )
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.config,
        seed_limit=args.seed_limit,
        probability_tolerance=args.probability_tolerance,
        max_iterations=args.max_iterations,
        time_limit_s=args.time_limit_s,
        bootstrap_samples=args.bootstrap_samples,
        frame_selection=args.frame_selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "seed_count",
            "recorded_mean_worst",
            "fixed_structure_power_mean_worst",
            "joint_structure_power_mean_lower",
            "joint_structure_power_qos_feasible_rate_lower",
            "mean_recorded_to_joint_gain",
            "mean_joint_solve_time_s",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
