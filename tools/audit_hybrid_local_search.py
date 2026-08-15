#!/usr/bin/env python
"""Audit oracle cold start plus learned warm local repair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.audit_oracle_local_search import _controller_summary  # noqa: E402
from tools.train_factor_graph_coordinator import (  # noqa: E402
    _aligned_data,
    _edge_values,
    _load_npz,
)
from uav_isac.coordination.learned_move_ranker import (  # noqa: E402
    FrozenLocalMoveRanker,
)
from uav_isac.coordination.local_exchange_oracle import (  # noqa: E402
    oracle_best_improvement,
    ranked_first_improvement,
    rebuild_structure,
    role_first_initial_structure,
    role_owner_from_structure,
)
from uav_isac.coordination.local_move_ranker import (  # noqa: E402
    local_move_features,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    data, candidate = _aligned_data(args.trace, args.candidate)
    mask = candidate["candidate_mask"].astype(bool)
    artifact = _load_npz(args.initial_artifact)
    cold_initial = artifact[args.initial_key].astype(bool)
    reference_artifact = _load_npz(args.reference_artifact)
    reference_pair = reference_artifact[args.reference_key].astype(bool)
    if not (mask.shape == cold_initial.shape == reference_pair.shape):
        raise ValueError("candidate/initial/reference shapes do not align")
    edge_value = _edge_values(data, candidate, args.student_checkpoint)
    ranker = FrozenLocalMoveRanker(args.ranker_checkpoint)
    output = np.zeros_like(mask, dtype=bool)
    current = np.zeros_like(mask[0], dtype=bool)
    previous_final = np.zeros_like(current)
    previous_role: np.ndarray | None = None
    previous_episode: int | None = None
    cold_resolves = 0
    warm_resolves = 0
    cold_candidate_labels = 0
    warm_candidate_labels = 0
    warm_verifications = 0
    cold_accepted = 0
    warm_accepted = 0
    warm_positive = []
    warm_top1 = []
    warm_top3 = []
    monotonic = True
    for frame in range(len(mask)):
        episode = int(data["episode"][frame])
        new_episode = episode != previous_episode
        if new_episode:
            current.fill(False)
            previous_final.fill(False)
            previous_role = None
            previous_episode = episode
        if bool(data["p0_resolved"][frame]):
            current.fill(False)
            if np.any(mask[frame]):
                _, fallback_role, _ = role_first_initial_structure(
                    edge_value[frame],
                    mask[frame],
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(
                        data["reports_per_receiver"][0]),
                )
                if previous_role is None or not np.any(previous_final):
                    result = oracle_best_improvement(
                        cold_initial[frame],
                        edge_value[frame],
                        mask[frame],
                        rounds=args.cold_rounds,
                        neighborhoods=("N1", "N2", "N3", "N5"),
                        target_pair_limit=int(data["target_pair_limit"][0]),
                        reports_per_receiver=int(
                            data["reports_per_receiver"][0]),
                        initial_role=fallback_role,
                    )
                    cold_resolves += 1
                    cold_candidate_labels += sum(result.candidate_counts)
                    cold_accepted += len(result.accepted_kinds)
                else:
                    _, owner = role_owner_from_structure(
                        previous_final, fallback_role=previous_role)
                    initial = rebuild_structure(
                        edge_value[frame],
                        mask[frame],
                        previous_role,
                        owner,
                        target_pair_limit=int(data["target_pair_limit"][0]),
                        reports_per_receiver=int(
                            data["reports_per_receiver"][0]),
                    )

                    def score_moves(selected, role, current_owner, moves, value):
                        return ranker.predict(np.stack([
                            local_move_features(
                                selected, role, current_owner, move, value)
                            for move in moves
                        ]))

                    result = ranked_first_improvement(
                        initial,
                        edge_value[frame],
                        mask[frame],
                        rounds=args.warm_rounds,
                        neighborhoods=("N1", "N2", "N3"),
                        target_pair_limit=int(data["target_pair_limit"][0]),
                        reports_per_receiver=int(
                            data["reports_per_receiver"][0]),
                        ranking_method="learned",
                        top_m=args.warm_top_m,
                        initial_role=previous_role,
                        ranking_scorer=score_moves,
                    )
                    warm_resolves += 1
                    warm_candidate_labels += sum(result.candidate_counts)
                    warm_verifications += sum(result.verification_counts)
                    warm_accepted += len(result.accepted_kinds)
                    warm_positive.extend(result.positive_opportunities)
                    warm_top1.extend(result.top1_positive)
                    warm_top3.extend(result.top3_positive)
                current[:] = result.selected
                previous_final[:] = result.selected
                previous_role = result.role.copy()
                monotonic &= all(
                    right > left
                    for left, right in zip(
                        result.objective_history,
                        result.objective_history[1:],
                    )
                )
        output[frame] = current
    summary, rows = _controller_summary(output, data, args)
    reference_summary, _ = _controller_summary(reference_pair, data, args)
    opportunity = np.asarray(warm_positive, dtype=bool)
    opportunity_count = int(np.count_nonzero(opportunity))

    def conditional(values: list[bool]) -> float:
        if not opportunity_count:
            return 1.0
        return float(np.count_nonzero(
            np.asarray(values, dtype=bool) & opportunity) / opportunity_count)

    full_labels = cold_candidate_labels + warm_candidate_labels
    effective = cold_candidate_labels + warm_verifications
    diagnostics = {
        "cold_resolves": cold_resolves,
        "warm_resolves": warm_resolves,
        "cold_fraction": float(cold_resolves / max(
            cold_resolves + warm_resolves, 1)),
        "cold_candidate_labels_mean": float(
            cold_candidate_labels / max(cold_resolves, 1)),
        "warm_candidate_labels_mean": float(
            warm_candidate_labels / max(warm_resolves, 1)),
        "warm_exact_verifications_mean": float(
            warm_verifications / max(warm_resolves, 1)),
        "effective_exact_evaluations_mean": float(
            effective / max(cold_resolves + warm_resolves, 1)),
        "effective_reduction_vs_full_labels": 1.0 - float(
            effective / max(full_labels, 1)),
        "cold_accepted_mean": float(
            cold_accepted / max(cold_resolves, 1)),
        "warm_accepted_mean": float(
            warm_accepted / max(warm_resolves, 1)),
        "warm_top1_positive_rate": conditional(warm_top1),
        "warm_top3_positive_rate": conditional(warm_top3),
        "objective_monotonic": bool(monotonic),
    }
    result = {
        "protocol": "gate_c1_6_oracle_cold_learned_warm_v1",
        "seeds": [int(value) for value in np.unique(data["seed"])],
        "metrics": summary,
        "diagnostics": diagnostics,
        "replicated_teacher_reference": reference_summary,
        "reference_worst_gap": float(reference_summary["worst"])
        - float(summary["worst"]),
        "fresh_test_consumed": False,
        "runtime_s": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "per_frame.npz",
        pair=output.astype(np.uint8),
        episode=data["episode"],
        seed=data["seed"],
        frame=data["frame"],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "episode_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_teacher_trace_selection20/teacher_trace.npz"))
    parser.add_argument(
        "--candidate", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate_a2_holdout10/per_frame.npz"))
    parser.add_argument(
        "--initial-artifact", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_factor_graph_gate_c1_screen/per_frame.npz"))
    parser.add_argument("--initial-key", default="eval_prediction")
    parser.add_argument(
        "--reference-artifact", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_replicated_consensus_holdout10/per_frame.npz"))
    parser.add_argument(
        "--reference-key", default="pairs_replicated_consensus")
    parser.add_argument(
        "--student-checkpoint", type=Path, default=Path(
            "results/architecture_v2_teacher_cleanreset_trace_gate100/frozen_structure_student_endpoint8.pt"))
    parser.add_argument(
        "--ranker-checkpoint", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_move_ranker_gate_c1_6/previous_n123_ranker.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_hybrid_local_search_gate_c1_6"))
    parser.add_argument("--cold-rounds", type=int, default=8)
    parser.add_argument("--warm-rounds", type=int, default=8)
    parser.add_argument("--warm-top-m", type=int, default=3)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    parser.add_argument("--steady-window", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
