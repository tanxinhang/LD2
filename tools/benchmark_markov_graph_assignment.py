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
) -> dict:
    rng = np.random.default_rng(int(seed))
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
        np.full(half, 0.0251), np.zeros(half),
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
            fc=2.8e10, delta_f=1.5625e4, T_sym=6.4e-5, M=64, N=16,
            kT=4.0e-21, B=1.0e6, NF_dB=8.0,
            P_sense=0.0251, P_report=0.25, ric_K=6.0,
            rcs=1.0, g_min=0.5,
            rng=np.random.default_rng(int(seed) + 10000 + case),
            g_tx_dBi=16.0, g_rx_dBi=16.0,
            use_los_prob=True, use_swerling=True, use_report_link=True,
            dd_gain_mode="continuous", sync_delay_error_bins=0.2,
            sync_doppler_error_bins=-0.2,
        )
        model = MarkovPhysicalAssignmentModel(
            dc, selected, budget, roles, np.array([500.0, 500.0, 0.0]),
            num_targets=Q, dt_s=0.1, movement_step_m=2.5,
            area_size_m=(1000.0, 1000.0), false_alarm_probability=0.01,
            safe_distance_m=20.0, weak_count=min(3, Q), weak_weight=0.25,
            target_covariance_horizon=covariance_horizon,
            uncertainty_penalty_std=uncertainty_penalty,
        )
        uav_pos = rng.uniform(80.0, 920.0, size=(K, 3))
        uav_pos[:, 2] = 100.0
        target_pos = rng.uniform(100.0, 900.0, size=(Q, 3))
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
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    result = benchmark(
        seed=args.seed, cases=args.cases, cardinality=args.cardinality,
        neighbors=args.neighbors, blind_candidates=args.blind_candidates,
        position_std_m=args.position_std_m,
        velocity_std_mps=args.velocity_std_mps,
        uncertainty_penalty_std=args.uncertainty_penalty_std)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
