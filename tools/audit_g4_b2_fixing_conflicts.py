#!/usr/bin/env python
"""G4-B2b exact fixing-deletion conflict audit on G4-A frames."""

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
    PermissionBlock,
    irreducible_feasible_permission_block,
)
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def audit(
    trace_path: Path, config_path: Path, g4a_path: Path, strategy: str,
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
            g_tx_dbi=float(cfg.otfs.g_tx_dBi),
            g_rx_dbi=float(cfg.otfs.g_rx_dBi), n_cpi=int(cfg.otfs.n_cpi),
            g_min=float(cfg.detection.g_min),
            use_swerling=bool(cfg.channel.use_swerling),
        )
        active = (np.asarray(data["outgoing_rate"][index]) > 0) & np.any(
            np.asarray(data["outgoing_token_mask"][index], dtype=bool), axis=1)
        comm = np.where(
            active, np.clip(np.asarray(data["comm_fraction"][index]), 0, 1)
            * float(cfg.uav.P_isac_total), 0.0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        selected = np.asarray(data["teacher_pair"][index], dtype=bool)
        owner = np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64)
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        K, _, Q = coefficient.shape

        def solve(block: PermissionBlock, feasibility_only: bool):
            return solve_minimum_intervention_task_repair_milp(
                coefficient, budget, selected, role == 0, role == 1, owner,
                p_fa=float(cfg.detection.P_FA),
                task_floors=tuple(cfg.marl.task_constrained_qos_floors),
                target_pair_limit=int(np.asarray(data["target_pair_limit"])[0]),
                reports_per_receiver=int(np.asarray(data["reports_per_receiver"])[0]),
                changeable_uavs=block.uavs,
                changeable_targets=block.targets,
                changeable_role_uavs=block.role_uavs,
                feasibility_only=feasibility_only,
                pwl_epsilon=1.0e-3, time_limit_s=30.0,
            )

        filtered = irreducible_feasible_permission_block(
            PermissionBlock(tuple(range(K)), tuple(range(Q)), tuple(range(K))),
            lambda block: solve(block, True).feasible,
            strategy=strategy,
        )
        repair = solve(filtered.block, False)
        exact_closure = int(exact[key]["closure_cardinality"])
        rows.append({
            "seed": key[0], "frame": key[1], "feasible": repair.feasible,
            "permission_uavs": len(filtered.block.uavs),
            "permission_targets": len(filtered.block.targets),
            "permission_roles": len(filtered.block.role_uavs),
            "feasibility_calls": filtered.feasibility_calls,
            "retained_fixing_conflicts": list(filtered.retained_certificates),
            "repair_closure_cardinality": repair.closure_cardinality,
            "exact_closure_cardinality": exact_closure,
            "minimum_closure_matched": repair.closure_cardinality == exact_closure,
        })
    return {
        "gate": "G4-B2b",
        "method": "exact_fixing_deletion_filter_not_IIS_or_Farkas",
        "deletion_strategy": strategy,
        "scope": "five_exposed_development_episodes_correlated_frames",
        "frames": len(rows),
        "feasible": sum(row["feasible"] for row in rows),
        "minimum_closure_matched": sum(row["minimum_closure_matched"] for row in rows),
        "small_permission_blocks": sum(
            row["permission_uavs"] <= 3 and row["permission_targets"] <= 3
            for row in rows),
        "median_permission_uavs": float(np.median([row["permission_uavs"] for row in rows])),
        "median_permission_targets": float(np.median([row["permission_targets"] for row in rows])),
        "median_permission_roles": float(np.median([row["permission_roles"] for row in rows])),
        "total_feasibility_calls": sum(row["feasibility_calls"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--strategy", choices=("sequential", "bisect"), default="sequential")
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.g4a, args.strategy)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
