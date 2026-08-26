#!/usr/bin/env python
"""G4-B shadow audit for task-ranked dependency-closed nested blocks."""

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
from uav_isac.coordination.nested_task_repair import nested_repair_blocks  # noqa: E402
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def audit(
    trace_path: Path, config_path: Path, g4a_path: Path, ranking_mode: str,
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
            g_rx_dbi=float(cfg.otfs.g_rx_dBi),
            n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
            use_swerling=bool(cfg.channel.use_swerling),
        )
        active = (
            np.asarray(data["outgoing_rate"][index], dtype=np.int64) > 0
        ) & np.any(np.asarray(data["outgoing_token_mask"][index], dtype=bool), axis=1)
        fraction = np.clip(np.asarray(data["comm_fraction"][index]), 0.0, 1.0)
        comm = np.where(active, fraction * float(cfg.uav.P_isac_total), 0.0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        selected = np.asarray(data["teacher_pair"][index], dtype=bool)
        owner = np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64)
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        blocks = nested_repair_blocks(
            coefficient, budget, selected, owner,
            role == 0, role == 1,
            np.asarray(data["physical_pd"][index], dtype=np.float64),
            tuple(cfg.marl.task_constrained_qos_floors),
            ranking_mode=ranking_mode,
        )
        attempts = []
        winner = None
        for level, block in enumerate(blocks):
            result = solve_minimum_intervention_task_repair_milp(
                coefficient, budget, selected, role == 0, role == 1, owner,
                p_fa=float(cfg.detection.P_FA),
                task_floors=tuple(cfg.marl.task_constrained_qos_floors),
                target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
                reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
                changeable_uavs=block.uavs, changeable_targets=block.targets,
                changeable_role_uavs=block.role_uavs,
                pwl_epsilon=1.0e-3, time_limit_s=30.0,
            )
            attempts.append({
                "level": level, "block_uavs": len(block.uavs),
                "block_targets": len(block.targets), "feasible": result.feasible,
                "block_role_uavs": len(block.role_uavs),
                "solve_time_s": result.solve_time_s,
            })
            if result.feasible:
                winner = (level, block, result)
                break
        if winner is None:
            rows.append({"seed": key[0], "frame": key[1], "recovered": False,
                         "attempts": attempts})
            continue
        level, block, result = winner
        full_block = len(block.uavs) == coefficient.shape[0] and len(block.targets) == coefficient.shape[2]
        exact_closure = int(exact[key]["closure_cardinality"])
        rows.append({
            "seed": key[0], "frame": key[1], "recovered": True,
            "first_feasible_level": level,
            "first_feasible_block_uavs": len(block.uavs),
            "first_feasible_block_targets": len(block.targets),
            "first_feasible_block_role_uavs": len(block.role_uavs),
            "required_full_block": full_block,
            "repair_closure_cardinality": result.closure_cardinality,
            "exact_closure_cardinality": exact_closure,
            "minimum_closure_matched": result.closure_cardinality == exact_closure,
            "changed_roles": result.changed_roles,
            "changed_owners": result.changed_owners,
            "toggled_edges": result.toggled_edges,
            "attempts": attempts,
        })
    recovered = [row for row in rows if row["recovered"]]
    return {
        "gate": "G4-B",
        "scope": "development_shadow_on_g4a_region_ii_frames",
        "ranking_mode": ranking_mode,
        "statistical_unit": "episode; within-episode frames are correlated diagnostics",
        "frames": len(rows),
        "recovered": len(recovered),
        "recovered_before_full_block": sum(not row["required_full_block"] for row in recovered),
        "minimum_closure_matched": sum(row["minimum_closure_matched"] for row in recovered),
        "median_first_block_uavs": float(np.median([
            row["first_feasible_block_uavs"] for row in recovered])),
        "median_first_block_targets": float(np.median([
            row["first_feasible_block_targets"] for row in recovered])),
        "median_attempts": float(np.median([len(row["attempts"]) for row in recovered])),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--ranking", choices=("gain", "closure_efficiency"), default="gain")
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.g4a, args.ranking)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
