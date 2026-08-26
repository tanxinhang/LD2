#!/usr/bin/env python
"""G4-B2d audit: oracle-certified conflict cores and hitting-set repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.fixing_conflict_filter import PermissionBlock  # noqa: E402
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix, optimal_maxmin_dual_prices,
)
from uav_isac.coordination.permission_cut_master import (  # noqa: E402
    CoreGuidedIterationLimit, core_guided_minimum_permission_block,
    MonotonePermissionOracle,
    permission_labels,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def audit(
    trace_path: Path,
    config_path: Path,
    g4a_path: Path,
    max_master_calls: int,
    max_frames: int | None,
    shrink_strategy: str,
    deletion_order_name: str,
    objective_mode: str,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    exact = {(int(row["seed"]), int(row["frame"])): row for row in g4a["rows"]}
    rows = []
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in exact:
            continue
        if max_frames is not None and len(rows) >= max_frames:
            break
        coefficient = per_watt_deflection_tensor_from_observables(
            np.asarray(data["privileged_alpha"][index], dtype=np.float64),
            np.asarray(data["privileged_g_dd"][index], dtype=np.float64),
            np.asarray(data["privileged_chi_rep"][index], dtype=np.float64),
            T_sym=float(cfg.otfs.T_sym), M=int(cfg.otfs.M), N=int(cfg.otfs.N),
            kT=float(cfg.channel.kT), bandwidth_hz=float(cfg.otfs.B),
            noise_figure_db=float(cfg.channel.NF),
            g_tx_dbi=float(cfg.otfs.g_tx_dBi), g_rx_dbi=float(cfg.otfs.g_rx_dBi),
            n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
            use_swerling=bool(cfg.channel.use_swerling))
        active = (np.asarray(data["outgoing_rate"][index]) > 0) & np.any(
            np.asarray(data["outgoing_token_mask"][index], dtype=bool), axis=1)
        comm = np.where(active, np.clip(np.asarray(
            data["comm_fraction"][index]), 0, 1) * float(cfg.uav.P_isac_total), 0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        selected = np.asarray(data["teacher_pair"][index], dtype=bool)
        owner = np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64)
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        K, _, Q = coefficient.shape

        deletion_order = None
        if deletion_order_name == "physics":
            # Physics only orders exact-oracle queries; it never creates a cut.
            # All three scores have deflection units after weighting by the
            # simplex max-min price pi.  The normalized probability shortfall
            # is dimensionless and only magnifies target urgency.
            gain, _ = fixed_owner_gain_matrix(
                coefficient, [tuple(edge) for edge in np.argwhere(selected)])
            pi, _ = optimal_maxmin_dual_prices(gain, budget)
            optimistic = np.max(coefficient, axis=1)  # best receiver, (TX,Q)
            floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
            pd = np.asarray(data["physical_pd"][index], dtype=np.float64)
            deficit_factor = 1.0 + np.maximum(floor - pd, 0.0) / max(floor, 1e-12)
            target_score = (
                pi * np.sum(budget[:, None] * optimistic, axis=0)
                * deficit_factor)
            uav_score = budget * np.sum(pi[None, :] * optimistic, axis=1)
            tx_value = uav_score
            rx_value = np.sum(
                budget[:, None, None] * pi[None, None, :] * coefficient,
                axis=(0, 2))
            role_score = np.abs(tx_value - rx_value)
            score = {
                **{f"target:{q}": float(target_score[q]) for q in range(Q)},
                **{f"uav:{i}": float(uav_score[i]) for i in range(K)},
                **{f"role:{i}": float(role_score[i]) for i in range(K)},
            }
            base_order = permission_labels(K, Q)
            deletion_order = tuple(sorted(
                base_order, key=lambda label: (score[label], base_order.index(label))))
        def exact_feasible(block: PermissionBlock) -> bool:
            return solve_minimum_intervention_task_repair_milp(
                coefficient, budget, selected, role == 0, role == 1, owner,
                p_fa=float(cfg.detection.P_FA),
                task_floors=tuple(cfg.marl.task_constrained_qos_floors),
                target_pair_limit=int(np.asarray(data["target_pair_limit"])[0]),
                reports_per_receiver=int(np.asarray(data["reports_per_receiver"])[0]),
                changeable_uavs=block.uavs, changeable_targets=block.targets,
                changeable_role_uavs=block.role_uavs,
                feasibility_only=True, pwl_epsilon=1.0e-3,
                time_limit_s=30.0).feasible

        feasible = MonotonePermissionOracle(exact_feasible)

        try:
            master = core_guided_minimum_permission_block(
                K, Q, feasible, max_iterations=max_master_calls,
                shrink_strategy=shrink_strategy, deletion_order=deletion_order,
                objective_mode=objective_mode)
        except CoreGuidedIterationLimit as error:
            rows.append({
                "seed": key[0], "frame": key[1],
                "master_converged": False,
                "master_oracle_queries": error.master_oracle_calls,
                "shrink_oracle_queries": error.shrink_oracle_calls,
                "logical_oracle_queries": error.oracle_calls,
                "new_exact_feasibility_calls": feasible.exact_calls,
                "dominance_inferred_feasible": feasible.inferred_feasible,
                "dominance_inferred_infeasible": feasible.inferred_infeasible,
                "conflict_cores": [list(core) for core in error.conflict_cores],
                "core_sizes": [len(core) for core in error.conflict_cores],
            })
            continue
        repair = solve_minimum_intervention_task_repair_milp(
            coefficient, budget, selected, role == 0, role == 1, owner,
            p_fa=float(cfg.detection.P_FA),
            task_floors=tuple(cfg.marl.task_constrained_qos_floors),
            target_pair_limit=int(np.asarray(data["target_pair_limit"])[0]),
            reports_per_receiver=int(np.asarray(data["reports_per_receiver"])[0]),
            changeable_uavs=master.block.uavs,
            changeable_targets=master.block.targets,
            changeable_role_uavs=master.block.role_uavs,
            pwl_epsilon=1.0e-3, time_limit_s=30.0)
        rows.append({
            "seed": key[0], "frame": key[1],
            "master_converged": True,
            "master_permission_cardinality": master.permission_cardinality,
            "master_permission_objective": list(master.permission_objective),
            "master_uavs": len(master.block.uavs),
            "master_targets": len(master.block.targets),
            "master_roles": len(master.block.role_uavs),
            "master_oracle_queries": master.master_oracle_calls,
            "shrink_oracle_queries": master.shrink_oracle_calls,
            "logical_oracle_queries": master.oracle_calls,
            "new_exact_feasibility_calls": feasible.exact_calls,
            "dominance_inferred_feasible": feasible.inferred_feasible,
            "dominance_inferred_infeasible": feasible.inferred_infeasible,
            "conflict_cores": [list(core) for core in master.conflict_cores],
            "core_sizes": [len(core) for core in master.conflict_cores],
            "repair_feasible": repair.feasible,
            "repair_closure_cardinality": repair.closure_cardinality,
            "exact_closure_cardinality": int(exact[key]["closure_cardinality"]),
            "minimum_closure_matched": (
                repair.closure_cardinality == int(exact[key]["closure_cardinality"])),
        })
    core_sizes = [size for row in rows for size in row["core_sizes"]]
    converged = sum(row["master_converged"] for row in rows)
    closure_hits = sum(row.get("minimum_closure_matched", False) for row in rows)
    total_exact = sum(row["new_exact_feasibility_calls"] for row in rows)
    return {
        "gate": "G4-B2d-1/2",
        "scope": "five_exposed_development_episodes_correlated_frames",
        "method": f"{shrink_strategy}_oracle_certified_irreducible_conflict_core",
        "deletion_order": deletion_order_name,
        "physics_order_semantics": (
            "ascending dual-weighted deflection opportunity; exact oracle alone certifies cuts"
            if deletion_order_name == "physics" else None),
        "optimality_scope": (
            "global optimum for the declared permission objective, not minimum realized repair closure"),
        "permission_objective": objective_mode,
        "frames": len(rows),
        "master_candidate_cap_per_frame": max_master_calls,
        "master_converged": converged,
        "repair_feasible": sum(row.get("repair_feasible", False) for row in rows),
        "minimum_closure_matched": closure_hits,
        "small_master_blocks": sum(
            row.get("master_uavs", 99) <= 3 and row.get("master_targets", 99) <= 3
            for row in rows),
        "conflict_cores_found": len(core_sizes),
        "median_conflict_core_size": (
            float(np.median(core_sizes)) if core_sizes else None),
        "max_conflict_core_size": max(core_sizes) if core_sizes else None,
        "total_master_oracle_queries": sum(row["master_oracle_queries"] for row in rows),
        "total_shrink_oracle_queries": sum(row["shrink_oracle_queries"] for row in rows),
        "total_logical_oracle_queries": sum(row["logical_oracle_queries"] for row in rows),
        "total_new_exact_feasibility_calls": total_exact,
        "total_dominance_inferences": sum(
            row["dominance_inferred_feasible"] + row["dominance_inferred_infeasible"]
            for row in rows),
        "preregistered_gate": {
            "all_frames_feasible": len(rows) == 21 and converged == 21,
            "at_least_18_within_16_master_candidates": (
                len(rows) == 21 and converged >= 18),
            "minimum_closure_hit_at_least_16": (
                len(rows) == 21 and closure_hits >= 16),
            "exact_feasibility_calls_below_b2b_333": (
                len(rows) == 21 and total_exact < 333),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-master-calls", type=int, default=16)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--shrink-strategy", choices=("sequential", "quickxplain"),
        default="sequential")
    parser.add_argument(
        "--deletion-order", choices=("index", "physics"), default="index")
    parser.add_argument(
        "--objective", choices=("cardinality", "support_lex"),
        default="cardinality")
    args = parser.parse_args()
    result = audit(
        args.trace, args.config, args.g4a, int(args.max_master_calls),
        args.max_frames, args.shrink_strategy, args.deletion_order,
        args.objective)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
