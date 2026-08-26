#!/usr/bin/env python
"""G4-B2e-1: objective-face separation on B2d capped frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.permission_cut_master import (  # noqa: E402
    CertifiedFeasibility, objective_layer_common_zeros,
    permission_block_with_closed_set,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def audit(
    trace_path: Path,
    config_path: Path,
    b2d_path: Path,
    escalate: bool,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    b2d = json.loads(b2d_path.read_text(encoding="utf-8"))
    failed = {
        (int(row["seed"]), int(row["frame"])): row
        for row in b2d["rows"] if not row["master_converged"]}
    rows = []
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in failed:
            continue
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
        cores = tuple(tuple(core) for core in failed[key]["conflict_cores"])
        layer = objective_layer_common_zeros(
            K, Q, cores, objective_mode="support_lex")
        face_block = permission_block_with_closed_set(
            K, Q, layer.common_zero_permissions)
        physical_calls = 0
        status_cache = {}

        def certify(block):
            nonlocal physical_calls
            if block in status_cache:
                return status_cache[block]
            physical_calls += 1
            solved = solve_minimum_intervention_task_repair_milp(
                coefficient, budget, selected, role == 0, role == 1, owner,
                p_fa=float(cfg.detection.P_FA),
                task_floors=tuple(cfg.marl.task_constrained_qos_floors),
                target_pair_limit=int(np.asarray(data["target_pair_limit"])[0]),
                reports_per_receiver=int(np.asarray(data["reports_per_receiver"])[0]),
                changeable_uavs=block.uavs,
                changeable_targets=block.targets,
                changeable_role_uavs=block.role_uavs,
                feasibility_only=True, pwl_epsilon=1.0e-3, time_limit_s=30.0)
            if solved.feasible:
                answer = CertifiedFeasibility.CERT_FEASIBLE
            elif solved.proven_infeasible:
                answer = CertifiedFeasibility.CERT_INFEASIBLE
            else:
                answer = CertifiedFeasibility.UNRESOLVED
            status_cache[block] = (answer, solved)
            return status_cache[block]

        status, result = certify(face_block)
        certified_tau = layer.scalar_cost if (
            status is CertifiedFeasibility.CERT_INFEASIBLE) else None
        certified_zero = layer.common_zero_permissions
        first_feasible_tau = None
        if escalate and status is CertifiedFeasibility.CERT_INFEASIBLE:
            low = layer.scalar_cost
            high = (K + Q) * (K + 1) + K
            high_layer = objective_layer_common_zeros(
                K, Q, cores, objective_mode="support_lex",
                scalar_cost_upper=high)
            high_block = permission_block_with_closed_set(
                K, Q, high_layer.common_zero_permissions)
            high_status, _ = certify(high_block)
            if high_status is CertifiedFeasibility.CERT_FEASIBLE:
                first_feasible_tau = high
                while high - low > 1:
                    middle = (low + high) // 2
                    middle_layer = objective_layer_common_zeros(
                        K, Q, cores, objective_mode="support_lex",
                        scalar_cost_upper=middle)
                    middle_block = permission_block_with_closed_set(
                        K, Q, middle_layer.common_zero_permissions)
                    middle_status, _ = certify(middle_block)
                    if middle_status is CertifiedFeasibility.CERT_INFEASIBLE:
                        low = middle
                        certified_tau = middle
                        certified_zero = middle_layer.common_zero_permissions
                    elif middle_status is CertifiedFeasibility.CERT_FEASIBLE:
                        high = middle
                        first_feasible_tau = middle
                    else:
                        break
        next_cost = None
        if status is CertifiedFeasibility.CERT_INFEASIBLE:
            raised = objective_layer_common_zeros(
                K, Q, cores + (certified_zero,),
                objective_mode="support_lex")
            next_cost = raised.scalar_cost
        rows.append({
            "seed": key[0], "frame": key[1],
            "current_scalar_lower_bound": layer.scalar_cost,
            "current_permission_objective": list(layer.permission_objective),
            "common_zero_permissions": list(layer.common_zero_permissions),
            "common_zero_size": len(layer.common_zero_permissions),
            "face_union_uavs": len(face_block.uavs),
            "face_union_targets": len(face_block.targets),
            "face_union_roles": len(face_block.role_uavs),
            "certificate": status.value,
            "solver_status": result.solver_status,
            "solver_message": result.solver_message,
            "solve_time_s": result.solve_time_s,
            "next_scalar_lower_bound": next_cost,
            "max_certified_infeasible_tau": certified_tau,
            "first_certified_feasible_tau": first_feasible_tau,
            "certified_sublevel_common_zeros": list(certified_zero),
            "physical_oracle_queries": physical_calls,
            "strict_lower_bound_raise": (
                next_cost is not None and next_cost > layer.scalar_cost),
            "master_solves_for_face": layer.master_solves,
        })
    return {
        "gate": "G4-B2e-1",
        "scope": "five_predeclared_b2d_capped_frames",
        "model_semantics": "conservative-PWL-MILP certified, not original-physics infeasibility",
        "frames": len(rows),
        "cert_infeasible": sum(
            row["certificate"] == CertifiedFeasibility.CERT_INFEASIBLE.value
            for row in rows),
        "cert_feasible": sum(
            row["certificate"] == CertifiedFeasibility.CERT_FEASIBLE.value
            for row in rows),
        "unresolved": sum(
            row["certificate"] == CertifiedFeasibility.UNRESOLVED.value
            for row in rows),
        "strict_lower_bound_raises": sum(
            row["strict_lower_bound_raise"] for row in rows),
        "physical_oracle_queries": sum(
            row["physical_oracle_queries"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--b2d", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--escalate", action="store_true")
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.b2d, bool(args.escalate))
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
