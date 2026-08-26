#!/usr/bin/env python
"""R1 audit of exact interaction width on the 21 post-G2 Region-II frames."""

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
from uav_isac.coordination.interaction_width import (  # noqa: E402
    exact_treewidth, floor_capable_task_modes, interaction_graphs,
    mode_overlap_hhi,
)
from uav_isac.coordination.scale_capability import cap_aware_sensing_budget  # noqa: E402
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.physical.detection import (  # noqa: E402
    minimum_deflection_for_detection_probability,
)


def audit(trace_path: Path, config_path: Path, g4a_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    region_ii = {(int(row["seed"]), int(row["frame"])) for row in g4a["rows"]}
    d_floor = float(minimum_deflection_for_detection_probability(
        np.asarray([float(cfg.marl.task_constrained_qos_floors[0])]),
        float(cfg.detection.P_FA))[0])
    rows = []
    graph_names = (
        "raw_target_primal", "resource_target_primal",
        "summary_lifted_incidence", "task_mode_clique_primal")
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
            g_tx_dbi=float(cfg.otfs.g_tx_dBi), g_rx_dbi=float(cfg.otfs.g_rx_dBi),
            n_cpi=int(cfg.otfs.n_cpi), g_min=float(cfg.detection.g_min),
            use_swerling=bool(cfg.channel.use_swerling))
        active = (np.asarray(data["outgoing_rate"][index]) > 0) & np.any(
            np.asarray(data["outgoing_token_mask"][index], dtype=bool), axis=1)
        comm = np.where(active, np.clip(np.asarray(
            data["comm_fraction"][index]), 0, 1) * float(cfg.uav.P_isac_total), 0)
        budget = cap_aware_sensing_budget(
            float(cfg.uav.P_isac_total), comm, float(cfg.uav.P_sense_max))
        K, _, Q = coefficient.shape
        pair_limit = int(np.asarray(data["target_pair_limit"])[0])
        modes = floor_capable_task_modes(
            coefficient, budget, pair_limit, d_floor)
        graphs = interaction_graphs(K, Q, modes)
        widths = {}
        separators = {}
        components = {}
        graph_edges = {}
        for name in graph_names:
            graph = graphs[name]
            result = exact_treewidth(graph)
            widths[name] = result.width
            separators[name] = list(result.separator_sizes)
            components[name] = sorted(
                (len(component) for component in nx.connected_components(graph)),
                reverse=True)
            graph_edges[name] = graph.number_of_edges()
        endpoint_target_degree = []
        for i in range(K):
            endpoint_target_degree.append(sum(
                any(i == mode.owner or i in mode.transmitters for mode in target_modes)
                for target_modes in modes))
        rows.append({
            "seed": key[0], "frame": key[1], "K": K, "Q": Q,
            "deflection_floor": d_floor,
            "mode_counts": [len(target_modes) for target_modes in modes],
            "total_modes": sum(map(len, modes)),
            "targets_without_floor_capable_mode": sum(not target_modes for target_modes in modes),
            "mode_overlap_hhi": list(mode_overlap_hhi(K, modes)),
            "uav_endpoint_target_degree": endpoint_target_degree,
            "treewidth": widths, "separator_sizes": separators,
            "component_sizes": components, "graph_edges": graph_edges,
        })
    local_width = [
        max(row["treewidth"]["resource_target_primal"],
            row["treewidth"]["summary_lifted_incidence"])
        for row in rows]
    by_seed = {}
    for seed in sorted({row["seed"] for row in rows}):
        selected = [row for row in rows if row["seed"] == seed]
        by_seed[str(seed)] = {
            "frames": len(selected),
            "median_total_modes": float(np.median([row["total_modes"] for row in selected])),
            "median_effective_width": float(np.median([
                max(row["treewidth"]["resource_target_primal"],
                    row["treewidth"]["summary_lifted_incidence"])
                for row in selected])),
            "max_effective_width": max(
                max(row["treewidth"]["resource_target_primal"],
                    row["treewidth"]["summary_lifted_incidence"])
                for row in selected),
        }
    return {
        "gate": "R1-interaction-width",
        "scope": "21_postG2_region_ii_development_frames_K6_Q6",
        "graph_semantics": {
            "raw_target_primal": "resource scopes plus the unlifted global steady/bottom-k target clique",
            "resource_target_primal": "target clique induced by each UAV endpoint/resource scope",
            "summary_lifted_incidence": "target-UAV incidence; lower graph width may move cost into separator state/frontier",
            "task_mode_clique_primal": "each retained (target, owner, TX-set) mode induces a clique on its target and UAV endpoints",
        },
        "safe_mode_pruning": (
            "discard only if isolated full sensing budgets cannot reach the per-target worst Deflection floor"),
        "frames": len(rows),
        "treewidth_summary": {
            name: {
                "median": float(np.median([row["treewidth"][name] for row in rows])),
                "max": max(row["treewidth"][name] for row in rows),
                "min": min(row["treewidth"][name] for row in rows),
            } for name in graph_names
        },
        "median_total_modes": float(np.median([row["total_modes"] for row in rows])),
        "max_total_modes": max(row["total_modes"] for row in rows),
        "median_mode_overlap_hhi": float(np.median([
            value for row in rows for value in row["mode_overlap_hhi"]])),
        "median_effective_width": float(np.median(local_width)),
        "max_effective_width": max(local_width),
        "local_K6Q6_width_gate": {
            "criterion": "median effective width <=3 and max <=4",
            "pass": float(np.median(local_width)) <= 3.0 and max(local_width) <= 4,
        },
        "scaling_hypothesis_status": (
            "unresolved: current post-G2 Region-II evidence contains only K=Q=6"),
        "by_seed": by_seed,
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
