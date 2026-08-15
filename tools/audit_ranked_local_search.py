#!/usr/bin/env python
"""Gate C1.6 pre-training audit for ranked feasible local search."""

from __future__ import annotations

import argparse
import csv
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
from uav_isac.coordination.local_exchange_oracle import (  # noqa: E402
    ranked_first_improvement,
    rebuild_structure,
    role_first_initial_structure,
    role_owner_from_structure,
)
from uav_isac.coordination.learned_move_ranker import (  # noqa: E402
    FrozenLocalMoveRanker,
)
from uav_isac.coordination.local_move_ranker import (  # noqa: E402
    local_move_features,
)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 1.0


def _run_ranked_variant(
    initial_name: str,
    ranking_method: str,
    top_m: int,
    repeat: int,
    initial_sequence: np.ndarray,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
    learned_ranker: FrozenLocalMoveRanker | None,
) -> tuple[np.ndarray, dict[str, object]]:
    frames, K, _, _ = candidate_mask.shape
    output = np.zeros_like(candidate_mask, dtype=bool)
    current = np.zeros_like(candidate_mask[0], dtype=bool)
    previous_final = np.zeros_like(current)
    previous_role: np.ndarray | None = None
    previous_episode: int | None = None
    accepted_counts: list[int] = []
    candidate_counts: list[int] = []
    verification_counts: list[int] = []
    positive_opportunities: list[bool] = []
    top1_positive: list[bool] = []
    top3_positive: list[bool] = []
    top1_best: list[bool] = []
    top3_best: list[bool] = []
    primary_regret: list[float] = []
    missed_positive_stops = 0
    move_counts = {kind: 0 for kind in ("N1", "N2", "N3", "N5")}
    monotonic = True
    resolve_count = 0
    for frame in range(frames):
        episode = int(data["episode"][frame])
        if episode != previous_episode:
            current.fill(False)
            previous_final.fill(False)
            previous_role = None
            previous_episode = episode
        if bool(data["p0_resolved"][frame]):
            current.fill(False)
            if np.any(candidate_mask[frame]):
                role_first, fallback_role, _ = role_first_initial_structure(
                    edge_value[frame],
                    candidate_mask[frame],
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(
                        data["reports_per_receiver"][0]),
                )
                if initial_name == "factor_graph":
                    initial = initial_sequence[frame]
                    initial_role = fallback_role
                elif initial_name == "role_first":
                    initial = role_first
                    initial_role = fallback_role
                elif initial_name == "previous":
                    if np.any(previous_final) and previous_role is not None:
                        _, owner = role_owner_from_structure(
                            previous_final, fallback_role=previous_role)
                        initial = rebuild_structure(
                            edge_value[frame],
                            candidate_mask[frame],
                            previous_role,
                            owner,
                            target_pair_limit=int(
                                data["target_pair_limit"][0]),
                            reports_per_receiver=int(
                                data["reports_per_receiver"][0]),
                        )
                        initial_role = previous_role
                    else:
                        initial = role_first
                        initial_role = fallback_role
                else:
                    raise ValueError(f"unknown initial: {initial_name}")
                result = ranked_first_improvement(
                    initial,
                    edge_value[frame],
                    candidate_mask[frame],
                    rounds=args.rounds,
                    neighborhoods=tuple(args.neighborhoods),
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(
                        data["reports_per_receiver"][0]),
                    ranking_method=ranking_method,
                    top_m=top_m,
                    initial_role=initial_role,
                    random_seed=(
                        int(args.random_seed)
                        + 1000003 * int(repeat)
                        + 1009 * int(frame)
                    ),
                    ranking_scorer=(
                        None if learned_ranker is None else
                        lambda selected, role, owner, moves, value: (
                            learned_ranker.predict(np.stack([
                                local_move_features(
                                    selected, role, owner, move, value)
                                for move in moves
                            ]))
                        )
                    ),
                )
                current[:] = result.selected
                previous_final[:] = result.selected
                previous_role = result.role.copy()
                accepted_counts.append(len(result.accepted_kinds))
                candidate_counts.append(sum(result.candidate_counts))
                verification_counts.append(sum(result.verification_counts))
                positive_opportunities.extend(result.positive_opportunities)
                top1_positive.extend(result.top1_positive)
                top3_positive.extend(result.top3_positive)
                top1_best.extend(result.top1_best)
                top3_best.extend(result.top3_best)
                primary_regret.extend(result.primary_regret)
                if (
                    result.positive_opportunities
                    and result.positive_opportunities[-1]
                    and len(result.accepted_kinds)
                    < len(result.positive_opportunities)
                ):
                    missed_positive_stops += 1
                for kind in result.accepted_kinds:
                    move_counts[kind] += 1
                monotonic &= all(
                    right > left
                    for left, right in zip(
                        result.objective_history,
                        result.objective_history[1:],
                    )
                )
                resolve_count += 1
        output[frame] = current
    opportunity = np.asarray(positive_opportunities, dtype=bool)
    opportunity_count = int(np.count_nonzero(opportunity))
    total_candidates = float(np.sum(candidate_counts))
    executor_verifications = float(np.sum(verification_counts))
    full_label_ranking = ranking_method == "oracle"
    effective_exact_evaluations = (
        total_candidates if full_label_ranking else executor_verifications)

    def conditional_rate(values: list[bool]) -> float:
        array = np.asarray(values, dtype=bool)
        return _safe_rate(
            float(np.count_nonzero(array & opportunity)),
            float(opportunity_count),
        )

    diagnostics: dict[str, object] = {
        "resolve_nonempty": resolve_count,
        "accepted_moves_mean": float(np.mean(accepted_counts))
        if accepted_counts else 0.0,
        "accepted_moves_max": int(max(accepted_counts, default=0)),
        "candidate_labels_mean": float(np.mean(candidate_counts))
        if candidate_counts else 0.0,
        "executor_exact_verifications_mean": float(
            np.mean(verification_counts)) if verification_counts else 0.0,
        "ranking_requires_full_exact_labels": full_label_ranking,
        "effective_exact_evaluations_mean": _safe_rate(
            effective_exact_evaluations, float(resolve_count)),
        "effective_exact_evaluation_reduction": 1.0 - _safe_rate(
            effective_exact_evaluations,
            total_candidates,
        ),
        "positive_opportunities": opportunity_count,
        "top1_positive_rate": conditional_rate(top1_positive),
        "top3_positive_rate": conditional_rate(top3_positive),
        "top1_exact_best_rate": conditional_rate(top1_best),
        "top3_exact_best_rate": conditional_rate(top3_best),
        "primary_regret_mean_on_opportunity": float(np.mean(
            np.asarray(primary_regret, dtype=np.float64)[opportunity]
        )) if opportunity_count else 0.0,
        "missed_positive_stop_rate": _safe_rate(
            float(missed_positive_stops), float(resolve_count)),
        "objective_monotonic": bool(monotonic),
        "move_counts": move_counts,
    }
    return output, diagnostics


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    data, candidate = _aligned_data(args.trace, args.candidate)
    initial_artifact = _load_npz(args.initial_artifact)
    initial_sequence = initial_artifact[args.initial_key].astype(bool)
    reference_artifact = _load_npz(args.reference_artifact)
    reference_sequence = reference_artifact[args.reference_key].astype(bool)
    if not (
        initial_sequence.shape
        == reference_sequence.shape
        == candidate["candidate_mask"].shape
    ):
        raise ValueError("initial/reference/candidate shapes do not align")
    edge_value = _edge_values(data, candidate, args.student_checkpoint)
    candidate_mask = candidate["candidate_mask"].astype(bool)
    reference_summary, _ = _controller_summary(
        reference_sequence, data, args)
    learned_ranker = None
    if "learned" in args.methods:
        if args.ranker_checkpoint is None:
            raise ValueError("--ranker-checkpoint is required for learned")
        learned_ranker = FrozenLocalMoveRanker(args.ranker_checkpoint)
    outputs: dict[str, np.ndarray] = {}
    variants: dict[str, dict[str, object]] = {}
    episode_rows: list[dict[str, object]] = []
    specs: list[tuple[str, int, int]] = [("oracle", 1, 0)]
    for method in args.methods:
        repeats = args.random_repeats if method == "random" else 1
        for top_m in args.top_m:
            for repeat in range(repeats):
                specs.append((method, int(top_m), int(repeat)))
    oracle_worst: float | None = None
    for method, top_m, repeat in specs:
        suffix = f"_r{repeat}" if method == "random" else ""
        name = f"{args.initial}_{method}_top{top_m}{suffix}"
        pair, diagnostics = _run_ranked_variant(
            args.initial,
            method,
            top_m,
            repeat,
            initial_sequence,
            edge_value,
            candidate_mask,
            data,
            args,
            learned_ranker,
        )
        summary, rows = _controller_summary(pair, data, args)
        if method == "oracle":
            oracle_worst = float(summary["worst"])
        variants[name] = {
            "initial": args.initial,
            "ranking_method": method,
            "top_m": top_m,
            "repeat": repeat,
            "metrics": summary,
            "diagnostics": diagnostics,
        }
        outputs[f"pairs_{name}"] = pair.astype(np.uint8)
        for row in rows:
            episode_rows.append({"controller": name, **row})
        print(name, json.dumps(variants[name]), flush=True)
    if oracle_worst is None:
        raise RuntimeError("oracle ranking control was not executed")
    for variant in variants.values():
        gap = oracle_worst - float(variant["metrics"]["worst"])
        variant["oracle_local_worst_gap"] = gap
        variant["within_002_oracle"] = bool(gap <= 0.02)
    result = {
        "protocol": "gate_c1_6_ranked_local_search_pretrain_v1",
        "seeds": [int(value) for value in np.unique(data["seed"])],
        "initial": args.initial,
        "neighborhoods": list(args.neighborhoods),
        "rounds": args.rounds,
        "replicated_teacher_reference": reference_summary,
        "oracle_local_worst": oracle_worst,
        "variants": variants,
        "fresh_test_consumed": False,
        "runtime_s": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "per_frame.npz",
        episode=data["episode"],
        seed=data["seed"],
        frame=data["frame"],
        **outputs,
    )
    with (args.output_dir / "episode_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
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
    parser.add_argument("--initial-key", type=str, default="eval_prediction")
    parser.add_argument(
        "--reference-artifact", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_replicated_consensus_holdout10/per_frame.npz"))
    parser.add_argument(
        "--reference-key", type=str, default="pairs_replicated_consensus")
    parser.add_argument(
        "--student-checkpoint", type=Path, default=Path(
            "results/architecture_v2_teacher_cleanreset_trace_gate100/frozen_structure_student_endpoint8.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_ranked_local_search_gate_c1_6_pretrain"))
    parser.add_argument(
        "--initial", choices=("factor_graph", "role_first", "previous"),
        default="previous")
    parser.add_argument(
        "--neighborhoods", nargs="+", default=["N1", "N2", "N3"])
    parser.add_argument(
        "--methods", nargs="+", default=["random", "total_gain", "scarcity"])
    parser.add_argument("--ranker-checkpoint", type=Path)
    parser.add_argument("--top-m", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=8127)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    parser.add_argument("--steady-window", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
