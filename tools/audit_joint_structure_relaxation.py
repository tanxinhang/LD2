#!/usr/bin/env python
"""Audit convex routing certificates and dual-guided sparse structure graphs."""

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
from tools.audit_qos_threshold_boundary import _coefficient  # noqa: E402
from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _ordered_unique,
    _recorded_power,
)
from uav_isac.coordination.bottleneck_router import (  # noqa: E402
    route_isac_repair,
)
from uav_isac.coordination.joint_structure_relaxation import (  # noqa: E402
    dual_guided_sparse_candidate_mask,
    owner_pair_relaxed_target_ceiling,
    solve_joint_structure_maxmin_lp_relaxation,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.qos_threshold_feasibility import (  # noqa: E402
    solve_maxmin_qos_boundary_bisection,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)


def _worst_pd(deflection: np.ndarray, p_fa: float) -> float:
    return float(np.min(compute_detection_probabilities(deflection, p_fa)))


def _exact_reference_rows(path: Path | None) -> dict[int, dict[str, object]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in payload["rows"]}


def audit(
    trace_path: Path,
    config_path: Path,
    *,
    seed_limit: int,
    exact_reference_path: Path | None,
    additional_owner_groups: int,
    minimum_owner_groups_per_target: int,
    transmitters_per_owner: int,
    qos_floor: float,
    probability_tolerance: float,
    max_iterations: int,
    time_limit_s: float,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    exact_rows = _exact_reference_rows(exact_reference_path)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    receiver_limit = int(np.asarray(
        data["reports_per_receiver"]).reshape(-1)[0])
    if K != int(cfg.scenario.K) or Q != int(cfg.scenario.Q):
        raise ValueError("trace dimensions do not match supplied config")

    rows: list[dict[str, object]] = []
    started = perf_counter()
    for seed in seed_order:
        candidates = np.flatnonzero(resolved & (seeds == seed))
        if not len(candidates):
            raise ValueError(f"seed {seed} has no resolved event")
        exact_row = exact_rows.get(int(seed))
        if exact_row is None:
            recorded_worst = np.min(np.asarray(
                data["physical_pd"][candidates], dtype=np.float64), axis=1)
            row_id = int(candidates[np.argmin(recorded_worst)])
        else:
            matching = candidates[frames[candidates] == int(exact_row["frame"])]
            if len(matching) != 1:
                raise ValueError(f"exact reference frame missing for seed {seed}")
            row_id = int(matching[0])

        comm_power, _, power_error = _recorded_power(data, row_id)
        budget = 1.0 - comm_power
        coefficient = _coefficient(data, row_id, cfg)
        pair = np.asarray(data["teacher_pair"][row_id], dtype=bool)
        selected = tuple(
            tuple(int(value) for value in edge) for edge in np.argwhere(pair)
        )
        recorded_pd = np.asarray(
            data["physical_pd"][row_id], dtype=np.float64)
        fixed_gain, _ = fixed_owner_gain_matrix(coefficient, selected)
        fixed = solve_fixed_structure_maxmin_power_lp(fixed_gain, budget)
        fixed_pd = _worst_pd(fixed.deflection, p_fa)

        relaxation = solve_joint_structure_maxmin_lp_relaxation(
            coefficient,
            budget,
            target_pair_limit=pair_limit,
            reports_per_receiver=receiver_limit,
        )
        relaxation_pd = float(compute_detection_probabilities(
            np.asarray([relaxation.upper_deflection]), p_fa)[0])
        owner_ceiling = owner_pair_relaxed_target_ceiling(
            coefficient, budget, target_pair_limit=pair_limit)
        owner_ceiling_worst = _worst_pd(owner_ceiling, p_fa)
        sparse = dual_guided_sparse_candidate_mask(
            coefficient,
            budget,
            relaxation,
            target_pair_limit=pair_limit,
            minimum_owner_groups_per_target=int(
                minimum_owner_groups_per_target),
            additional_owner_groups=int(additional_owner_groups),
            transmitters_per_owner=int(transmitters_per_owner),
            incumbent_selected=selected,
        )
        candidate_coefficient = np.where(
            sparse.candidate_mask, coefficient, 0.0)
        candidate_boundary = solve_maxmin_qos_boundary_bisection(
            candidate_coefficient,
            budget,
            p_fa=p_fa,
            target_pair_limit=pair_limit,
            reports_per_receiver=receiver_limit,
            initial_feasible_pd=p_fa,
            probability_tolerance=float(probability_tolerance),
            max_iterations=int(max_iterations),
            time_limit_s=float(time_limit_s),
        )
        recorded_worst = float(np.min(recorded_pd))
        route = route_isac_repair(
            recorded_worst,
            fixed_power_upper=fixed_pd,
            joint_structure_upper=relaxation_pd,
            qos_floor=float(qos_floor),
        )

        exact_lower = (
            float(exact_row["joint_structure_power_lower"])
            if exact_row is not None else float("nan")
        )
        exact_upper = (
            float(exact_row["joint_structure_power_upper"])
            if exact_row is not None else float("nan")
        )
        exact_upper_proven = bool(
            exact_row is not None
            and exact_row["joint_exact_infeasible_upper"]
        )
        structural_headroom = exact_lower - fixed_pd
        restricted_headroom = candidate_boundary.lower_pd - fixed_pd
        recovery = (
            restricted_headroom / structural_headroom
            if structural_headroom > 1.0e-9 else 1.0
        )
        rows.append({
            "seed": int(seed),
            "frame": int(frames[row_id]),
            "recorded_worst": recorded_worst,
            "fixed_power_worst": fixed_pd,
            "owner_pair_ceiling_worst": owner_ceiling_worst,
            "joint_relaxation_upper_worst": relaxation_pd,
            "joint_relaxation_upper_deflection": float(
                relaxation.upper_deflection),
            "joint_relaxation_solve_time_s": float(relaxation.solve_time_s),
            "exact_joint_lower": exact_lower,
            "exact_joint_upper": exact_upper,
            "exact_joint_upper_proven_infeasible": exact_upper_proven,
            "relaxation_integrality_gap_to_exact_lower": float(
                relaxation_pd - exact_lower),
            "candidate_joint_lower": float(candidate_boundary.lower_pd),
            "candidate_joint_upper": float(candidate_boundary.upper_pd),
            "candidate_upper_proven_infeasible": bool(
                candidate_boundary.exact_infeasible_upper),
            "candidate_structural_gain_recovery": float(recovery),
            "candidate_edge_count": int(sparse.candidate_edge_count),
            "full_edge_count": int(sparse.full_edge_count),
            "candidate_edge_fraction": float(
                sparse.candidate_edge_count / max(sparse.full_edge_count, 1)),
            "candidate_owner_group_count": int(
                len(sparse.selected_owner_groups)),
            "minimum_retained_owner_ceiling_ratio": float(
                np.min(sparse.retained_ceiling_ratio)),
            "target_prices": sparse.target_prices.tolist(),
            "route": route.route.value,
            "route_reason": route.reason,
            "safe_geometry_certificate": bool(
                relaxation_pd < float(qos_floor)),
            "exact_geometry_required": bool(
                exact_upper_proven and exact_upper < float(qos_floor)),
            "candidate_qos_feasible": bool(
                candidate_boundary.lower_pd >= float(qos_floor)),
            "exact_qos_feasible": bool(exact_lower >= float(qos_floor)),
            "power_balance_error_w": float(power_error),
        })

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    exact_geometry = [
        bool(row["exact_geometry_required"]) for row in rows
    ]
    safe_geometry = [
        bool(row["safe_geometry_certificate"]) for row in rows
    ]
    summary = {
        "schema_version": 1,
        "scope": (
            "same-geometry joint role-owner-capacity LP upper certificate and "
            "dual-guided sparse candidate MILP on weakest resolved events"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "exact_reference": (
            None if exact_reference_path is None else str(exact_reference_path)),
        "seed_count": len(rows),
        "seed_order": [int(seed) for seed in seed_order],
        "additional_owner_groups": int(additional_owner_groups),
        "minimum_owner_groups_per_target": int(
            minimum_owner_groups_per_target),
        "transmitters_per_owner": int(transmitters_per_owner),
        "qos_floor": float(qos_floor),
        "mean_fixed_power_worst": float(np.mean(values("fixed_power_worst"))),
        "mean_exact_joint_lower": float(np.nanmean(values("exact_joint_lower"))),
        "mean_relaxation_upper_worst": float(np.mean(
            values("joint_relaxation_upper_worst"))),
        "mean_relaxation_integrality_gap": float(np.nanmean(
            values("relaxation_integrality_gap_to_exact_lower"))),
        "mean_candidate_joint_lower": float(np.mean(
            values("candidate_joint_lower"))),
        "candidate_qos_feasible_rate": float(np.mean([
            bool(row["candidate_qos_feasible"]) for row in rows
        ])),
        "exact_qos_feasible_rate": float(np.mean([
            bool(row["exact_qos_feasible"]) for row in rows
        ])),
        "mean_candidate_structural_gain_recovery": float(np.mean(np.clip(
            values("candidate_structural_gain_recovery"), 0.0, 1.0))),
        "mean_candidate_edge_fraction": float(np.mean(
            values("candidate_edge_fraction"))),
        "minimum_retained_owner_ceiling_ratio": float(np.min(
            values("minimum_retained_owner_ceiling_ratio"))),
        "safe_geometry_certificate_rate": float(np.mean(safe_geometry)),
        "exact_geometry_required_rate": float(np.mean(exact_geometry)),
        "unsafe_geometry_route_count": int(sum(
            safe and not exact
            for safe, exact in zip(safe_geometry, exact_geometry)
        )),
        "missed_geometry_certificate_count": int(sum(
            exact and not safe
            for safe, exact in zip(safe_geometry, exact_geometry)
        )),
        "route_counts": {
            route: int(sum(row["route"] == route for row in rows))
            for route in (
                "no_op", "fixed_structure_power", "joint_structure_power",
                "slow_geometry", "hold_unverified",
            )
        },
        "mean_relaxation_solve_time_s": float(np.mean(
            values("joint_relaxation_solve_time_s"))),
        "elapsed_seconds": float(perf_counter() - started),
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exact-reference", type=Path)
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--additional-owner-groups", type=int, default=6)
    parser.add_argument("--minimum-owner-groups-per-target", type=int, default=1)
    parser.add_argument("--transmitters-per-owner", type=int, default=3)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument("--probability-tolerance", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--time-limit-s", type=float, default=5.0)
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.config,
        seed_limit=max(1, int(args.seed_limit)),
        exact_reference_path=args.exact_reference,
        additional_owner_groups=max(0, int(args.additional_owner_groups)),
        minimum_owner_groups_per_target=max(
            1, int(args.minimum_owner_groups_per_target)),
        transmitters_per_owner=max(1, int(args.transmitters_per_owner)),
        qos_floor=float(args.qos_floor),
        probability_tolerance=float(args.probability_tolerance),
        max_iterations=max(1, int(args.max_iterations)),
        time_limit_s=float(args.time_limit_s),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "seed_count",
        "mean_fixed_power_worst",
        "mean_exact_joint_lower",
        "mean_relaxation_upper_worst",
        "mean_candidate_joint_lower",
        "candidate_qos_feasible_rate",
        "mean_candidate_structural_gain_recovery",
        "mean_candidate_edge_fraction",
        "unsafe_geometry_route_count",
        "route_counts",
    )}, indent=2))


if __name__ == "__main__":
    main()
