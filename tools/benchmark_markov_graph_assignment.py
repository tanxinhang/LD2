"""Fixed-seed shadow benchmark for predictive graph movement assignments.

This is a mechanism screen, not formal evidence.  It compares the accepted
physics-KNN joint proposal with the best candidate in the bounded blind swap
neighborhood at the same randomly generated physical state.  No live policy is
modified and all channel/fading effects are evaluated in conditional mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.physical.deflection import DeflectionComputer
from config.params import load_config
from uav_isac.prediction.markov_assignment import (
    build_target_priority_swap_neighborhood,
)
from uav_isac.prediction.markov_graph_assignment import (
    propose_physics_knn_local_search,
)
from uav_isac.prediction.markov_physical_assignment import (
    MarkovPhysicalAssignmentModel,
)


def _percentile(values: list[float], probability: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), probability))


def benchmark(
    *, seed: int, cases: int, cardinality: int, neighbors: int,
    blind_candidates: int, position_std_m: float = 0.0,
    velocity_std_mps: float = 0.0, uncertainty_penalty_std: float = 0.0,
    config_path: str = "config/exp_research_interference_l4_noise_loaded.yaml",
) -> dict:
    rng = np.random.default_rng(int(seed))
    cfg = load_config(config_path)
    K = Q = int(cardinality)
    if K < 4 or K % 2:
        raise ValueError("cardinality must be an even integer at least four")
    half = K // 2
    roles = np.concatenate((
        np.zeros(half, dtype=np.int64),
        np.ones(half, dtype=np.int64),
    ))
    selected = tuple(
        (target % half, half + target % half, target)
        for target in range(Q)
    )
    budget = np.concatenate((
        np.full(half, float(cfg.uav.P_sense_max)), np.zeros(half),
    ))
    position_std = float(position_std_m)
    velocity_std = float(velocity_std_mps)
    uncertainty_penalty = float(uncertainty_penalty_std)
    if (
        not np.isfinite(position_std) or position_std < 0.0
        or not np.isfinite(velocity_std) or velocity_std < 0.0
        or not np.isfinite(uncertainty_penalty) or uncertainty_penalty < 0.0
    ):
        raise ValueError("uncertainty parameters must be finite and non-negative")
    covariance_horizon = None
    if position_std > 0.0 or velocity_std > 0.0:
        covariance_horizon = np.zeros((1, Q, 4, 4), dtype=np.float64)
        covariance_horizon[:, :, 0, 0] = position_std ** 2
        covariance_horizon[:, :, 1, 1] = position_std ** 2
        covariance_horizon[:, :, 2, 2] = velocity_std ** 2
        covariance_horizon[:, :, 3, 3] = velocity_std ** 2
    rows = []
    for case in range(int(cases)):
        dc = DeflectionComputer(
            fc=cfg.otfs.fc, delta_f=cfg.otfs.delta_f,
            T_sym=cfg.otfs.T_sym, M=cfg.otfs.M, N=cfg.otfs.N,
            kT=cfg.channel.kT, B=cfg.otfs.B, NF_dB=cfg.channel.NF,
            P_sense=cfg.uav.P_sense, P_report=cfg.uav.P_report,
            ric_K=cfg.channel.ric_K, rcs=cfg.target.rcs,
            g_min=cfg.detection.g_min,
            rng=np.random.default_rng(int(seed) + 10000 + case),
            g_tx_dBi=cfg.otfs.g_tx_dBi, g_rx_dBi=cfg.otfs.g_rx_dBi,
            use_los_prob=cfg.channel.use_los_prob,
            use_swerling=cfg.channel.use_swerling, use_report_link=True,
            dd_gain_mode=cfg.detection.dd_gain_mode,
            sync_delay_error_bins=cfg.channel.sync_delay_error_bins,
            sync_doppler_error_bins=cfg.channel.sync_doppler_error_bins,
        )
        area_x, area_y = (float(value) for value in cfg.scenario.region_size)
        height = float(cfg.scenario.height)
        model = MarkovPhysicalAssignmentModel(
            dc, selected, budget, roles,
            np.array([area_x / 2.0, area_y / 2.0, 0.0]),
            num_targets=Q, dt_s=float(cfg.scenario.dt),
            movement_step_m=float(cfg.uav.v_max * cfg.scenario.dt),
            area_size_m=(area_x, area_y),
            false_alarm_probability=float(cfg.detection.P_FA),
            safe_distance_m=float(cfg.uav.d_safe),
            weak_count=min(3, Q), weak_weight=0.25,
            target_covariance_horizon=covariance_horizon,
            uncertainty_penalty_std=uncertainty_penalty,
        )
        uav_pos = rng.uniform(
            [0.08 * area_x, 0.08 * area_y, height],
            [0.92 * area_x, 0.92 * area_y, height], size=(K, 3))
        uav_pos[:, 2] = height
        target_pos = rng.uniform(
            [0.10 * area_x, 0.10 * area_y, 0.0],
            [0.90 * area_x, 0.90 * area_y, 0.0], size=(Q, 3))
        target_pos[:, 2] = 0.0
        target_vel = rng.normal(0.0, 8.0, size=(Q, 3))
        target_vel[:, 2] = 0.0
        state = model.pack_state(
            uav_pos, np.zeros((K, 3)), target_pos, target_vel)
        # The mechanism screen starts from a complete, deterministic matching.
        incumbent = np.arange(Q, dtype=np.int64)

        begin = perf_counter()
        graph = propose_physics_knn_local_search(
            model, state, incumbent, stage_index=0,
            k_neighbors=int(neighbors),
            evaluation_budget=int(blind_candidates), max_rounds=3)
        graph_time = perf_counter() - begin

        # Weak-target priority is estimated from the incumbent one-step stage:
        # the fixed assignment vector itself has no target score, so stable
        # target order is used here. A causal dual-price order is the next
        # environment-shadow integration gate.
        blind = build_target_priority_swap_neighborhood(
            incumbent, num_targets=Q,
            target_priority=np.arange(Q, dtype=np.int64),
            max_candidates=int(blind_candidates))
        begin = perf_counter()
        blind_costs = []
        for assignment in blind:
            next_state = model.transition(state, assignment, 0)
            blind_costs.append(float(
                model.evaluate(next_state, stage_index=0).cost))
        blind_time = perf_counter() - begin
        best_blind_cost = float(np.min(blind_costs))
        incumbent_cost = float(blind_costs[0])
        accepted_graph_cost = graph.proposal_cost
        rows.append({
            "case": case,
            "incumbent_cost": incumbent_cost,
            "graph_raw_cost": graph.proposal_cost,
            "graph_accepted_cost": float(accepted_graph_cost),
            "blind_best_cost": best_blind_cost,
            "graph_improvement": incumbent_cost - float(accepted_graph_cost),
            "blind_improvement": incumbent_cost - best_blind_cost,
            "graph_accepted": bool(graph.accepted_swaps),
            "graph_beats_blind": bool(
                float(accepted_graph_cost) < best_blind_cost - 1.0e-9),
            "graph_evaluated_assignments": int(graph.evaluated_assignments),
            "blind_evaluated_candidates": int(len(blind)),
            "graph_time_s": float(graph_time),
            "blind_time_s": float(blind_time),
        })

    graph_improvement = [row["graph_improvement"] for row in rows]
    blind_improvement = [row["blind_improvement"] for row in rows]
    paired_delta = np.asarray(graph_improvement) - np.asarray(blind_improvement)
    bootstrap_rng = np.random.default_rng(int(seed) + 909_091)
    bootstrap_indices = bootstrap_rng.integers(
        0, len(rows), size=(10_000, len(rows)))
    bootstrap_mean = np.mean(paired_delta[bootstrap_indices], axis=1)
    return {
        "schema_version": "markov-graph-assignment-microbenchmark/v3",
        "scope": "fixed-seed conditional-mean physical shadow mechanism screen",
        "seed": int(seed),
        "cases": int(cases),
        "K": K,
        "Q": Q,
        "config": str(config_path),
        "knn_neighbors": int(neighbors),
        "blind_candidate_cap": int(blind_candidates),
        "position_std_m": position_std,
        "velocity_std_mps": velocity_std,
        "uncertainty_penalty_std": uncertainty_penalty,
        "graph_acceptance_rate": float(np.mean([
            row["graph_accepted"] for row in rows])),
        "graph_beats_blind_rate": float(np.mean([
            row["graph_beats_blind"] for row in rows])),
        "graph_mean_improvement": float(np.mean(graph_improvement)),
        "blind_mean_improvement": float(np.mean(blind_improvement)),
        "graph_improvement_p10": _percentile(graph_improvement, 10.0),
        "blind_improvement_p10": _percentile(blind_improvement, 10.0),
        "paired_improvement_delta_mean": float(np.mean(paired_delta)),
        "paired_improvement_delta_median": float(np.median(paired_delta)),
        "paired_improvement_delta_p10": float(np.percentile(paired_delta, 10.0)),
        "paired_improvement_delta_bootstrap95": [
            float(np.percentile(bootstrap_mean, 2.5)),
            float(np.percentile(bootstrap_mean, 97.5)),
        ],
        "paired_graph_wins": int(np.count_nonzero(paired_delta > 1.0e-12)),
        "paired_ties": int(np.count_nonzero(np.abs(paired_delta) <= 1.0e-12)),
        "paired_graph_losses": int(np.count_nonzero(paired_delta < -1.0e-12)),
        "graph_mean_evaluated_assignments": float(np.mean([
            row["graph_evaluated_assignments"] for row in rows])),
        "blind_mean_evaluated_candidates": float(np.mean([
            row["blind_evaluated_candidates"] for row in rows])),
        "graph_mean_time_s": float(np.mean([
            row["graph_time_s"] for row in rows])),
        "blind_mean_time_s": float(np.mean([
            row["blind_time_s"] for row in rows])),
        "rows": rows,
        "formal_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--cardinality", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--blind-candidates", type=int, default=32)
    parser.add_argument("--position-std-m", type=float, default=0.0)
    parser.add_argument("--velocity-std-mps", type=float, default=0.0)
    parser.add_argument("--uncertainty-penalty-std", type=float, default=0.0)
    parser.add_argument(
        "--config",
        default="config/exp_research_interference_l4_noise_loaded.yaml")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    result = benchmark(
        seed=args.seed, cases=args.cases, cardinality=args.cardinality,
        neighbors=args.neighbors, blind_candidates=args.blind_candidates,
        position_std_m=args.position_std_m,
        velocity_std_mps=args.velocity_std_mps,
        uncertainty_penalty_std=args.uncertainty_penalty_std,
        config_path=args.config)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
