#!/usr/bin/env python
"""G4-A exact minimum-intervention audit on G3 Region-II frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402


def audit(trace_path: Path, config_path: Path, region_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    regions = json.loads(region_path.read_text(encoding="utf-8"))
    region_ii = {(int(row["seed"]), int(row["frame"]))
                 for row in regions["rows"] if row["region"] == "II"}
    cfg = load_config(str(config_path))
    rows = []
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in region_ii:
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
        rate = np.asarray(data["outgoing_rate"][index], dtype=np.int64)
        mask = np.asarray(data["outgoing_token_mask"][index], dtype=bool)
        active = (rate > 0) & np.any(mask, axis=1)
        fraction = np.clip(np.asarray(data["comm_fraction"][index]), 0.0, 1.0)
        comm = np.where(active, fraction * float(cfg.uav.P_isac_total), 0.0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        result = solve_minimum_intervention_task_repair_milp(
            coefficient, budget,
            np.asarray(data["teacher_pair"][index], dtype=bool),
            role == 0, role == 1,
            np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64),
            p_fa=float(cfg.detection.P_FA),
            task_floors=tuple(cfg.marl.task_constrained_qos_floors),
            target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
            reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
            pwl_epsilon=1.0e-3, time_limit_s=30.0,
        )
        rows.append({
            "seed": key[0], "frame": key[1], "feasible": result.feasible,
            "proven_infeasible": result.proven_infeasible,
            "changed_roles": result.changed_roles,
            "changed_owners": result.changed_owners,
            "toggled_edges": result.toggled_edges,
            "affected_targets": len(result.affected_targets),
            "participants": len(result.participants),
            "closure_cardinality": result.closure_cardinality,
            "prepare_bits": result.prepare_bits,
            "total_sensing_power_w": result.total_sensing_power_w,
            "worst_pd": float(np.min(result.target_pd)),
            "bottom_k_pd": float(np.mean(np.partition(
                result.target_pd, int(cfg.marl.task_constrained_qos_floors[3]) - 1
            )[:int(cfg.marl.task_constrained_qos_floors[3])])),
            "average_pd": float(np.mean(result.target_pd)),
            "solve_time_s": result.solve_time_s,
            "solver_status": result.solver_status,
        })
    feasible_rows = [row for row in rows if row["feasible"]]
    def distribution(name: str) -> dict[str, float] | None:
        if not feasible_rows:
            return None
        values = np.asarray([row[name] for row in feasible_rows], dtype=float)
        return {"min": float(np.min(values)), "median": float(np.median(values)),
                "mean": float(np.mean(values)), "max": float(np.max(values))}
    return {
        "gate": "G4-A",
        "scope": "exact_full_information_shadow_on_g3_region_ii_development_frames",
        "statistical_unit": "episode; five within-episode frames are correlated diagnostics",
        "region_ii_frames": len(rows),
        "feasible_witnesses": len(feasible_rows),
        "lexicographic_objective": [
            "affected_targets_plus_participants",
            "current_layout_prepare_payload_bits",
            "total_sensing_power_w",
        ],
        "communication_scope": (
            "prepare_bits uses the existing role/owner/edge/certificate layout; "
            "it excludes any future explicit quantized power-record extension "
            "and is not an L0 over-air admission result"
        ),
        "distributions": {name: distribution(name) for name in (
            "changed_roles", "changed_owners", "toggled_edges",
            "affected_targets", "participants", "closure_cardinality",
            "prepare_bits", "total_sensing_power_w", "solve_time_s")},
        "small_block_counts": {
            "role_flips_le_2": sum(row["changed_roles"] <= 2 for row in feasible_rows),
            "owner_changes_le_2": sum(row["changed_owners"] <= 2 for row in feasible_rows),
            "participants_le_3": sum(row["participants"] <= 3 for row in feasible_rows),
            "all_three": sum(
                row["changed_roles"] <= 2 and row["changed_owners"] <= 2
                and row["participants"] <= 3 for row in feasible_rows),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.regions)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
