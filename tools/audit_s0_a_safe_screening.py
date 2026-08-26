#!/usr/bin/env python
"""S0-A audit of context-free resource-complete safe screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.interaction_width import exact_treewidth  # noqa: E402
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.safe_structural_reduction import (  # noqa: E402
    context_free_zero_gain_screen,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)


def _positive_incidence_width(gain: np.ndarray) -> tuple[int, int, list[int]]:
    K, _, Q = gain.shape
    graph = nx.Graph()
    graph.add_nodes_from(("u", i) for i in range(K))
    graph.add_nodes_from(("q", q) for q in range(Q))
    for i in range(K):
        for q in range(Q):
            if any(gain[i, j, q] > 0.0 or gain[j, i, q] > 0.0
                   for j in range(K) if j != i):
                graph.add_edge(("u", i), ("q", q))
    result = exact_treewidth(graph)
    components = sorted(
        (len(component) for component in nx.connected_components(graph)),
        reverse=True)
    return result.width, graph.number_of_edges(), components


def audit(trace_path: Path, config_path: Path, g4a_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    baseline = {
        (int(row["seed"]), int(row["frame"])): row for row in g4a["rows"]
    }
    rows = []
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in baseline:
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
        selected = np.asarray(data["teacher_pair"][index], dtype=bool)
        rate = np.asarray(data["outgoing_rate"][index], dtype=np.int64)
        mask = np.asarray(data["outgoing_token_mask"][index], dtype=bool)
        active = (rate > 0) & np.any(mask, axis=1)
        comm = np.where(
            active,
            np.clip(np.asarray(data["comm_fraction"][index]), 0.0, 1.0)
            * float(cfg.uav.P_isac_total),
            0.0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        role = np.asarray(data["teacher_role"][index], dtype=np.int8)
        owner = np.asarray(data["teacher_receiver_owner"][index], dtype=np.int64)
        screen = context_free_zero_gain_screen(coefficient, selected)
        common = dict(
            p_fa=float(cfg.detection.P_FA),
            task_floors=tuple(cfg.marl.task_constrained_qos_floors),
            target_pair_limit=int(np.asarray(data["target_pair_limit"]).reshape(-1)[0]),
            reports_per_receiver=int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0]),
            pwl_epsilon=1.0e-3, time_limit_s=30.0)
        d_f = solve_minimum_intervention_task_repair_milp(
            coefficient, budget, selected, role == 0, role == 1, owner,
            forbidden_edges=screen.feasibility_preserving,
            feasibility_only=True, **common)
        d_o = solve_minimum_intervention_task_repair_milp(
            coefficient, budget, selected, role == 0, role == 1, owner,
            forbidden_edges=screen.optimality_preserving, **common)
        reference = baseline[key]
        discrete_equal = bool(
            d_o.feasible
            and d_o.closure_cardinality == int(reference["closure_cardinality"])
            and d_o.prepare_bits == int(reference["prepare_bits"]))
        power_equal = bool(discrete_equal and np.isclose(
            d_o.total_sensing_power_w,
            float(reference["total_sensing_power_w"]),
            rtol=1.0e-6, atol=1.0e-9))
        width, incidence_edges, components = _positive_incidence_width(coefficient)
        rows.append({
            "seed": key[0], "frame": key[1],
            "candidate_edges": screen.candidate_edges,
            "d_f_screened_edges": len(screen.feasibility_preserving),
            "d_o_screened_edges": len(screen.optimality_preserving),
            "selected_zero_gain_edges": len(screen.selected_zero_gain),
            "d_f_feasible": d_f.feasible,
            "d_f_proven_infeasible": d_f.proven_infeasible,
            "d_o_feasible": d_o.feasible,
            "d_o_discrete_lex_equal": discrete_equal,
            "d_o_power_equal": power_equal,
            "d_o_objective": [d_o.closure_cardinality, d_o.prepare_bits,
                              d_o.total_sensing_power_w],
            "baseline_objective": [int(reference["closure_cardinality"]),
                                   int(reference["prepare_bits"]),
                                   float(reference["total_sensing_power_w"])],
            "positive_coefficient_incidence_width": width,
            "positive_coefficient_incidence_edges": incidence_edges,
            "positive_coefficient_component_sizes": components,
            "d_f_solve_time_s": d_f.solve_time_s,
            "d_o_solve_time_s": d_o.solve_time_s,
        })
    total_candidates = sum(row["candidate_edges"] for row in rows)
    total_df = sum(row["d_f_screened_edges"] for row in rows)
    total_do = sum(row["d_o_screened_edges"] for row in rows)
    return {
        "gate": "S0-A-context-free-resource-complete-screening",
        "scope": "21_postG2_region_ii_development_frames_K6_Q6",
        "theorem_scope": (
            "exact-zero coefficients in the conservative model; no epsilon, "
            "no cross-UAV resource pooling, and no hardware-uncertainty claim"),
        "d_f_rule": "fix every exact-zero-gain bistatic incidence to zero",
        "d_o_rule": (
            "fix only exact-zero-gain incidences absent from the reference structure"),
        "cross_uav_context_free_dominance": (
            "structurally impossible under componentwise independent-UAV resources, "
            "because a replacement UAV increases a previously zero resource component"),
        "frames": len(rows),
        "candidate_edges": total_candidates,
        "d_f_screened_edges": total_df,
        "d_o_screened_edges": total_do,
        "d_f_pruning_ratio": total_df / total_candidates if total_candidates else 0.0,
        "d_o_pruning_ratio": total_do / total_candidates if total_candidates else 0.0,
        "selected_zero_gain_edges": sum(
            row["selected_zero_gain_edges"] for row in rows),
        "d_f_feasibility_preserved": sum(row["d_f_feasible"] for row in rows),
        "d_o_discrete_lex_preserved": sum(
            row["d_o_discrete_lex_equal"] for row in rows),
        "d_o_full_lex_preserved": sum(row["d_o_power_equal"] for row in rows),
        "positive_incidence_width_summary": {
            "min": min(row["positive_coefficient_incidence_width"] for row in rows),
            "median": float(np.median([
                row["positive_coefficient_incidence_width"] for row in rows])),
            "max": max(row["positive_coefficient_incidence_width"] for row in rows),
        },
        "gate_pass": bool(
            len(rows) == 21
            and all(row["d_f_feasible"] for row in rows)
            and all(row["d_o_power_equal"] for row in rows)),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.g4a)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
