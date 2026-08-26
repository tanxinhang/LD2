#!/usr/bin/env python
"""G4-B2e OLCS audit with tri-state conservative-model certificates."""

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
    CertifiedFeasibility, CertifiedMonotonePermissionOracle,
    ObjectiveLayerSeparationLimit,
    objective_layer_separating_permission_block,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def audit(
    trace_path: Path,
    config_path: Path,
    g4a_path: Path,
    max_iterations: int,
    max_bundle_supports: int,
    only_failed_path: Path | None,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    exact = {(int(row["seed"]), int(row["frame"])): row for row in g4a["rows"]}
    only_keys = None
    if only_failed_path is not None:
        prior = json.loads(only_failed_path.read_text(encoding="utf-8"))
        only_keys = {
            (int(row["seed"]), int(row["frame"]))
            for row in prior["rows"] if not row["master_converged"]}
    rows = []
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in exact or (only_keys is not None and key not in only_keys):
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

        def exact_oracle(block):
            result = solve_minimum_intervention_task_repair_milp(
                coefficient, budget, selected, role == 0, role == 1, owner,
                p_fa=float(cfg.detection.P_FA),
                task_floors=tuple(cfg.marl.task_constrained_qos_floors),
                target_pair_limit=int(np.asarray(data["target_pair_limit"])[0]),
                reports_per_receiver=int(np.asarray(data["reports_per_receiver"])[0]),
                changeable_uavs=block.uavs, changeable_targets=block.targets,
                changeable_role_uavs=block.role_uavs,
                feasibility_only=True, pwl_epsilon=1.0e-3, time_limit_s=30.0)
            if result.feasible:
                return CertifiedFeasibility.CERT_FEASIBLE
            if result.proven_infeasible:
                return CertifiedFeasibility.CERT_INFEASIBLE
            return CertifiedFeasibility.UNRESOLVED

        oracle = CertifiedMonotonePermissionOracle(exact_oracle)
        try:
            master = objective_layer_separating_permission_block(
                K, Q, oracle, max_iterations=max_iterations,
                max_bundle_supports=max_bundle_supports)
        except ObjectiveLayerSeparationLimit as error:
            rows.append({
                "seed": key[0], "frame": key[1], "master_converged": False,
                "master_iterations": error.master_iterations,
                "cut_kinds": list(error.cut_kinds),
                "lower_bound_trace": list(error.lower_bound_trace),
                "physical_exact_calls": oracle.exact_calls,
                "dominance_inferences": (
                    oracle.inferred_feasible + oracle.inferred_infeasible),
                "unresolved_queries": oracle.unresolved,
            })
            continue
        except RuntimeError as error:
            rows.append({
                "seed": key[0], "frame": key[1], "master_converged": False,
                "failure": str(error), "physical_exact_calls": oracle.exact_calls,
                "dominance_inferences": (
                    oracle.inferred_feasible + oracle.inferred_infeasible),
                "unresolved_queries": oracle.unresolved,
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
            "seed": key[0], "frame": key[1], "master_converged": True,
            "master_iterations": master.master_iterations,
            "permission_objective": list(master.permission_objective),
            "permission_uavs": len(master.block.uavs),
            "permission_targets": len(master.block.targets),
            "permission_roles": len(master.block.role_uavs),
            "cut_kinds": list(master.cut_kinds),
            "lower_bound_trace": list(master.lower_bound_trace),
            "physical_exact_calls": oracle.exact_calls,
            "dominance_inferences": (
                oracle.inferred_feasible + oracle.inferred_infeasible),
            "unresolved_queries": oracle.unresolved,
            "repair_feasible": repair.feasible,
            "repair_closure_cardinality": repair.closure_cardinality,
            "exact_closure_cardinality": int(exact[key]["closure_cardinality"]),
            "minimum_closure_matched": (
                repair.closure_cardinality == int(exact[key]["closure_cardinality"])),
        })
    converged = sum(row["master_converged"] for row in rows)
    closure_hits = sum(row.get("minimum_closure_matched", False) for row in rows)
    physical_calls = sum(row["physical_exact_calls"] for row in rows)
    return {
        "gate": "G4-B2e-3/4" if only_keys is None else "G4-B2e-2-smoke",
        "scope": ("all_21_region_ii_frames" if only_keys is None
                  else "five_predeclared_b2d_capped_frames"),
        "model_semantics": (
            "tri-state conservative-PWL-MILP certificate; unresolved never cuts"),
        "frames": len(rows), "master_converged": converged,
        "repair_feasible": sum(row.get("repair_feasible", False) for row in rows),
        "minimum_closure_matched": closure_hits,
        "total_physical_exact_calls": physical_calls,
        "total_dominance_inferences": sum(row["dominance_inferences"] for row in rows),
        "total_unresolved_queries": sum(row["unresolved_queries"] for row in rows),
        "sublevel_cuts": sum(
            kind == "sublevel" for row in rows for kind in row.get("cut_kinds", [])),
        "bundle_cuts": sum(
            kind.startswith("bundle:") for row in rows
            for kind in row.get("cut_kinds", [])),
        "preregistered_gate": {
            "all_frames_feasible": len(rows) == 21 and converged == 21,
            "at_least_18_within_16_iterations": len(rows) == 21 and converged >= 18,
            "minimum_closure_hit_at_least_16": len(rows) == 21 and closure_hits >= 16,
            "physical_exact_calls_below_b2b_333": len(rows) == 21 and physical_calls < 333,
            "zero_unresolved": sum(row["unresolved_queries"] for row in rows) == 0,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--max-bundle-supports", type=int, default=8)
    parser.add_argument("--only-b2d-failed", type=Path)
    args = parser.parse_args()
    result = audit(
        args.trace, args.config, args.g4a, int(args.max_iterations),
        int(args.max_bundle_supports), args.only_b2d_failed)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
