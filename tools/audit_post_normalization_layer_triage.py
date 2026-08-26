#!/usr/bin/env python
"""Re-triage exposed G3 frames under the current post-normalization model."""

from __future__ import annotations

import argparse
from collections import Counter
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
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def audit(trace_path: Path, config_path: Path, legacy_g3a_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    legacy = json.loads(legacy_g3a_path.read_text(encoding="utf-8"))
    exposed = {(int(row["seed"]), int(row["frame"])): row
               for row in legacy["rows"]}
    floors = tuple(cfg.marl.task_constrained_qos_floors)
    rows = []
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in exposed:
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
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        repair = solve_minimum_intervention_task_repair_milp(
            coefficient, budget,
            np.asarray(data["teacher_pair"][index], dtype=bool),
            role == 0, role == 1,
            np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64),
            p_fa=float(cfg.detection.P_FA), task_floors=floors,
            target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
            reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
            pwl_epsilon=1.0e-3, time_limit_s=30.0)
        if repair.feasible and repair.closure_cardinality == 0:
            layer = "CURRENT_FIXED_STRUCTURE_FEASIBLE"
        elif repair.feasible:
            layer = "L2_STRUCTURE_REPAIR"
        else:
            layer = "UNRESOLVED"
        rows.append({
            "seed": key[0], "frame": key[1],
            "legacy_region": exposed[key]["region"], "current_layer": layer,
            "current_deployed_status": "UNRESOLVED_TRACE_LACKS_POWER_AND_PHYSICAL_PD",
            "repair_feasible": repair.feasible,
            "repair_closure": repair.closure_cardinality,
            "repair_prepare_bits": repair.prepare_bits,
            "repair_power_w": repair.total_sensing_power_w,
        })
    if len(rows) != len(exposed):
        raise RuntimeError("trace does not contain every legacy exposed frame")
    return {
        "gate": "post-normalization-layer-triage",
        "scope": "same five exposed development seeds and 25 sampled frames",
        "physical_semantics": (
            "current T_sym-cancelled per-watt Deflection, n_cpi=1, c_det=1"),
        "legacy_region_counts": dict(sorted(Counter(
            row["legacy_region"] for row in rows).items())),
        "current_layer_counts": dict(sorted(Counter(
            row["current_layer"] for row in rows).items())),
        "legacy_region_ii_current_fixed_structure_feasible": sum(
            row["legacy_region"] == "II"
            and row["current_layer"] == "CURRENT_FIXED_STRUCTURE_FEASIBLE"
            for row in rows),
        "current_l2_frames": sum(
            row["current_layer"] == "L2_STRUCTURE_REPAIR" for row in rows),
        "provenance_gate_pass": False,
        "deployed_replay_available": False,
        "deployed_replay_blocker": (
            "trace has neither sensing_power_w nor environment-level physical_pd; "
            "receiver-local local_pd is not a post-normalization substitute"),
        "implication": (
            "legacy G4/R1/S0 L2 labels cannot seed M4-D; regenerate only after "
            "a fresh current-model L2 development set is defined"),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--legacy-g3a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.legacy_g3a)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
