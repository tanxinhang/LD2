#!/usr/bin/env python
"""G4-B2c audit: order-invariant backbone and conflict-cut permission master."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.fixing_conflict_filter import (  # noqa: E402
    PermissionBlock, exact_full_block_repair_backbone,
)
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.permission_cut_master import (  # noqa: E402
    PermissionCutIterationLimit, conflict_cut_minimum_permission_block,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def audit(
    trace_path: Path, config_path: Path, g4a_path: Path, max_master_calls: int,
    force_backbone: bool,
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
        cache: dict[PermissionBlock, bool] = {}
        exact_calls = 0

        def feasible(block: PermissionBlock) -> bool:
            nonlocal exact_calls
            if block not in cache:
                exact_calls += 1
                cache[block] = solve_minimum_intervention_task_repair_milp(
                    coefficient, budget, selected, role == 0, role == 1, owner,
                    p_fa=float(cfg.detection.P_FA),
                    task_floors=tuple(cfg.marl.task_constrained_qos_floors),
                    target_pair_limit=int(np.asarray(data["target_pair_limit"])[0]),
                    reports_per_receiver=int(np.asarray(data["reports_per_receiver"])[0]),
                    changeable_uavs=block.uavs, changeable_targets=block.targets,
                    changeable_role_uavs=block.role_uavs,
                    feasibility_only=True, pwl_epsilon=1.0e-3,
                    time_limit_s=30.0).feasible
            return cache[block]

        full = PermissionBlock(tuple(range(K)), tuple(range(Q)), tuple(range(K)))
        backbone = exact_full_block_repair_backbone(full, feasible)
        after_backbone = exact_calls
        try:
            master = conflict_cut_minimum_permission_block(
                K, Q, feasible, max_iterations=max_master_calls,
                forced_permissions=(backbone.backbone if force_backbone else tuple()))
        except PermissionCutIterationLimit as error:
            rows.append({
                "seed": key[0], "frame": key[1],
                "backbone": list(backbone.backbone),
                "backbone_size": len(backbone.backbone),
                "backbone_exact_calls": after_backbone,
                "master_converged": False,
                "master_oracle_queries": error.oracle_calls,
                "master_new_exact_calls_after_cache": exact_calls - after_backbone,
                "conflict_cuts": len(error.conflict_cuts),
            })
            continue
        master_new_calls = exact_calls - after_backbone
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
            "backbone": list(backbone.backbone),
            "backbone_size": len(backbone.backbone),
            "backbone_exact_calls": after_backbone,
            "master_converged": True,
            "master_permission_cardinality": master.permission_cardinality,
            "master_uavs": len(master.block.uavs),
            "master_targets": len(master.block.targets),
            "master_roles": len(master.block.role_uavs),
            "master_oracle_queries": master.oracle_calls,
            "master_new_exact_calls_after_cache": master_new_calls,
            "conflict_cuts": len(master.conflict_cuts),
            "repair_feasible": repair.feasible,
            "repair_closure_cardinality": repair.closure_cardinality,
            "exact_closure_cardinality": int(exact[key]["closure_cardinality"]),
            "minimum_closure_matched": (
                repair.closure_cardinality == int(exact[key]["closure_cardinality"])),
        })
    return {
        "gate": "G4-B2c",
        "scope": "five_exposed_development_episodes_correlated_frames",
        "master_optimality_scope": (
            "global minimum permission cardinality, not global minimum realized repair cost"),
        "frames": len(rows),
        "master_call_cap_per_frame": max_master_calls,
        "backbone_forced_in_master": force_backbone,
        "master_converged": sum(row["master_converged"] for row in rows),
        "repair_feasible": sum(row.get("repair_feasible", False) for row in rows),
        "frames_with_nonempty_backbone": sum(row["backbone_size"] > 0 for row in rows),
        "median_backbone_size": float(np.median([row["backbone_size"] for row in rows])),
        "median_master_permission_cardinality": float(np.median([
            row["master_permission_cardinality"] for row in rows
            if row["master_converged"]])) if any(
                row["master_converged"] for row in rows) else None,
        "small_master_blocks": sum(
            row.get("master_uavs", 99) <= 3 and row.get("master_targets", 99) <= 3
            for row in rows),
        "minimum_closure_matched": sum(
            row.get("minimum_closure_matched", False) for row in rows),
        "total_master_oracle_queries": sum(row["master_oracle_queries"] for row in rows),
        "total_backbone_exact_calls": sum(row["backbone_exact_calls"] for row in rows),
        "total_new_exact_calls_after_backbone_cache": sum(
            row["master_new_exact_calls_after_cache"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-master-calls", type=int, default=16)
    parser.add_argument("--force-backbone", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.trace, args.config, args.g4a, int(args.max_master_calls),
        bool(args.force_backbone))
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
