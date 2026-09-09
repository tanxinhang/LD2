#!/usr/bin/env python
"""Train a causal hold-tail graph ranker and replay it on unseen seeds.

The future hold segment is used only to construct supervised training labels.
Inference consumes the same recorded pre-decision local observations and
delivered-token graph available to the frozen distributed structure student.
The exact receiver-owner projection and physical feasibility mask are kept
unchanged.  Results are offline same-state evidence, not closed-loop evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_hold_qos_rank import (  # noqa: E402
    _multiframe_episode_summary,
    hold_aware_score,
)
from tools.audit_oracle_ladder import (  # noqa: E402
    _local_candidate_and_rank,
    _pair_change,
)
from tools.audit_structure_teacher_trace import (  # noqa: E402
    _edge_value_metrics,
    _fit_factorized_edge_value_probe,
    _infer_observation_slices,
    _spearman,
)
from tools.audit_local_candidate_upper_bound import (  # noqa: E402
    _student_features,
)
from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    build_local_candidate_mask,
    delivered_target_tokens,
    episode_detection_summary,
    held_pair_sequence,
    realized_pd_history,
)
from uav_isac.evaluation.pareto_no_regret import (  # noqa: E402
    componentwise_hold_pareto_gate,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)


METRICS = ("steady", "weak3", "worst", "cvar", "qos_feasible")


def seed_split(
    seed_values: np.ndarray,
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> tuple[list[int], list[int], list[int]]:
    """Return deterministic, disjoint episode-seed splits."""
    ordered = list(dict.fromkeys(int(value) for value in seed_values))
    requested = int(train_count) + int(validation_count) + int(test_count)
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("all seed splits must be non-empty")
    if requested > len(ordered):
        raise ValueError(
            f"requested {requested} seeds but trace contains {len(ordered)}")
    train = ordered[:train_count]
    validation = ordered[train_count:train_count + validation_count]
    test = ordered[
        train_count + validation_count:
        train_count + validation_count + test_count
    ]
    return train, validation, test


def _subset(data: dict[str, np.ndarray], seeds: list[int]) -> dict[str, np.ndarray]:
    frame_count = len(data["seed"])
    keep = np.isin(data["seed"], np.asarray(seeds, dtype=np.int64))
    if not np.any(keep):
        raise ValueError("selected seeds have no frames")
    return {
        key: value[keep] if value.ndim > 0 and value.shape[0] == frame_count else value
        for key, value in data.items()
    }


def _resolved_indices(data: dict[str, np.ndarray], seeds: list[int]) -> np.ndarray:
    return np.flatnonzero(
        np.asarray(data["p0_resolved"], dtype=bool)
        & np.isin(data["seed"], np.asarray(seeds, dtype=np.int64))
    )


def _controller_summary(
    data: dict[str, np.ndarray],
    score: np.ndarray,
    admitted: np.ndarray,
    *,
    steady_window: int,
    steady_floor: float,
    weak3_floor: float,
    worst_floor: float,
    frame_duration_s: float,
) -> dict[str, Any]:
    pairs = held_pair_sequence(score, admitted, data)
    return _pair_summary(
        data,
        pairs,
        steady_window=steady_window,
        steady_floor=steady_floor,
        weak3_floor=weak3_floor,
        worst_floor=worst_floor,
        frame_duration_s=frame_duration_s,
    )


def _pair_summary(
    data: dict[str, np.ndarray],
    pairs: np.ndarray,
    *,
    steady_window: int,
    steady_floor: float,
    weak3_floor: float,
    worst_floor: float,
    frame_duration_s: float,
) -> dict[str, Any]:
    realized = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    pd = realized_pd_history(pairs, realized, p_fa)
    summary, episodes = episode_detection_summary(
        pd,
        data["episode"],
        data["seed"],
        steady_window=steady_window,
        qos_thresholds=(steady_floor, weak3_floor, worst_floor),
    )
    receiver_d = np.sum(realized * pairs, axis=1)
    target_d = np.max(receiver_d, axis=1)
    return {
        "summary": summary,
        "episodes": episodes,
        "multiframe_worst": _multiframe_episode_summary(
            target_d,
            data,
            p_fa=p_fa,
            worst_floor=worst_floor,
            frame_duration_s=frame_duration_s,
        ),
    }


def _pair_similarity(
    candidate: np.ndarray,
    reference: np.ndarray,
    resolved: np.ndarray,
) -> dict[str, float]:
    proposed = np.asarray(candidate, dtype=bool)[resolved]
    target = np.asarray(reference, dtype=bool)[resolved]
    intersection = int(np.sum(proposed & target))
    proposed_count = int(np.sum(proposed))
    target_count = int(np.sum(target))
    frame_exact = np.all(proposed == target, axis=(1, 2, 3))
    return {
        "edge_precision": float(intersection / max(proposed_count, 1)),
        "edge_recall": float(intersection / max(target_count, 1)),
        "frame_exact_rate": float(np.mean(frame_exact)),
    }


def _candidate_rank_spearman(
    score: np.ndarray,
    target: np.ndarray,
    admitted: np.ndarray,
    resolved: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(score, dtype=np.float64)
    labels = np.asarray(target, dtype=np.float64)
    mask = np.asarray(admitted, dtype=bool)
    correlations: list[float] = []
    for frame in np.flatnonzero(resolved):
        for target_index in range(values.shape[-1]):
            active = mask[frame, :, :, target_index]
            if int(np.sum(active)) < 2:
                continue
            correlation = _spearman(
                values[frame, :, :, target_index][active],
                labels[frame, :, :, target_index][active],
            )
            if np.isfinite(correlation):
                correlations.append(float(correlation))
    array = np.asarray(correlations, dtype=np.float64)
    return {
        "frame_target_count": int(len(array)),
        "mean": float(np.mean(array)) if len(array) else 0.0,
        "p10": float(np.percentile(array, 10)) if len(array) else 0.0,
        "negative_rate": float(np.mean(array < 0.0)) if len(array) else 0.0,
    }


def _hold_segment_effect(
    data: dict[str, np.ndarray],
    baseline_pairs: np.ndarray,
    candidate_pairs: np.ndarray,
    *,
    steady_window: int,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    realized = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    baseline_pd = realized_pd_history(baseline_pairs, realized, p_fa)
    candidate_pd = realized_pd_history(candidate_pairs, realized, p_fa)
    episodes = np.asarray(data["episode"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    rows: list[dict[str, Any]] = []
    for episode in np.unique(episodes):
        episode_index = np.flatnonzero(episodes == episode)
        resolve_index = episode_index[resolved[episode_index]]
        tail_start = int(episode_index[-min(int(steady_window), len(episode_index))])
        for position, frame in enumerate(resolve_index):
            stop = (
                int(resolve_index[position + 1])
                if position + 1 < len(resolve_index)
                else int(episode_index[-1]) + 1
            )
            segment = np.arange(int(frame), stop, dtype=np.int64)
            baseline_target = np.mean(baseline_pd[segment], axis=0)
            candidate_target = np.mean(candidate_pd[segment], axis=0)
            baseline_ordered = np.sort(baseline_target)
            candidate_ordered = np.sort(candidate_target)
            changed = not bool(np.array_equal(
                baseline_pairs[frame], candidate_pairs[frame]))
            rows.append({
                "episode": int(episode),
                "seed": int(data["seed"][frame]),
                "frame": int(data["frame"][frame]),
                "in_steady_tail": bool(stop > tail_start),
                "changed": changed,
                "worst_delta": float(candidate_ordered[0] - baseline_ordered[0]),
                "weak3_delta": float(
                    np.mean(candidate_ordered[:min(3, len(candidate_ordered))])
                    - np.mean(baseline_ordered[:min(3, len(baseline_ordered))])
                ),
                "steady_delta": float(
                    np.mean(candidate_target) - np.mean(baseline_target)),
            })

    def summarize(selected: list[dict[str, Any]]) -> dict[str, float]:
        changed_rows = [row for row in selected if row["changed"]]
        deltas = np.asarray(
            [row["worst_delta"] for row in changed_rows], dtype=np.float64)
        if not len(deltas):
            return {
                "segments": int(len(selected)),
                "changed_segments": 0,
                "beneficial_rate_among_changed": 0.0,
                "harmful_rate_among_changed": 0.0,
                "neutral_rate_among_changed": 1.0,
                "positive_worst_delta_sum": 0.0,
                "negative_worst_delta_sum": 0.0,
                "mean_worst_delta_among_changed": 0.0,
            }
        beneficial = deltas > float(tolerance)
        harmful = deltas < -float(tolerance)
        return {
            "segments": int(len(selected)),
            "changed_segments": int(len(changed_rows)),
            "beneficial_rate_among_changed": float(np.mean(beneficial)),
            "harmful_rate_among_changed": float(np.mean(harmful)),
            "neutral_rate_among_changed": float(np.mean(~beneficial & ~harmful)),
            "positive_worst_delta_sum": float(np.sum(deltas[beneficial])),
            "negative_worst_delta_sum": float(np.sum(deltas[harmful])),
            "mean_worst_delta_among_changed": float(np.mean(deltas)),
        }

    return {
        "all_segments": summarize(rows),
        "steady_tail_segments": summarize([
            row for row in rows if row["in_steady_tail"]]),
        "by_seed": {
            str(seed): summarize([row for row in rows if row["seed"] == seed])
            for seed in sorted({int(row["seed"]) for row in rows})
        },
    }


def _delta(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    return {
        metric: float(candidate["summary"][metric])
        - float(baseline["summary"][metric])
        for metric in METRICS
    }


def hold_bottleneck_weights(
    hold_target: np.ndarray,
    selected_pair: np.ndarray,
    *,
    p_fa: float,
    qos_floor: float,
    strength: float,
    temperature: float = 0.15,
) -> np.ndarray:
    """Upweight targets whose selected hold-tail evidence misses the floor."""
    receiver_d = np.sum(
        np.asarray(hold_target, dtype=np.float64)
        * np.asarray(selected_pair, dtype=np.float64),
        axis=1,
    )
    target_d = np.max(receiver_d, axis=1)
    target_pd = compute_detection_probabilities(target_d, float(p_fa))
    hardness = np.exp(np.clip(
        (float(qos_floor) - target_pd) / max(float(temperature), 1.0e-6),
        -6.0,
        6.0,
    ))
    hardness /= np.maximum(np.mean(hardness, axis=-1, keepdims=True), 1.0e-8)
    return np.asarray(1.0 + float(strength) * hardness, dtype=np.float32)


def align_proposal_to_baseline_scale(
    baseline: np.ndarray,
    proposal: np.ndarray,
    admitted: np.ndarray,
) -> np.ndarray:
    """Quantile-map proposal order onto baseline values per frame/target.

    This preserves the mature controller's score scale while exposing only
    the learned ordering as a residual.  Non-candidate values are untouched.
    """
    reference = np.asarray(baseline, dtype=np.float64)
    learned = np.asarray(proposal, dtype=np.float64)
    mask = np.asarray(admitted, dtype=bool)
    if reference.shape != learned.shape or reference.shape != mask.shape:
        raise ValueError("baseline, proposal, and admitted shapes must match")
    if reference.ndim != 4:
        raise ValueError("rank tensors must have shape (F,K,K,Q)")
    aligned = reference.copy()
    for frame in range(reference.shape[0]):
        for target in range(reference.shape[-1]):
            active = mask[frame, :, :, target]
            count = int(np.sum(active))
            if count < 2:
                continue
            baseline_values = np.sort(
                reference[frame, :, :, target][active], kind="stable")
            proposal_values = learned[frame, :, :, target][active]
            proposal_order = np.argsort(proposal_values, kind="stable")
            mapped = np.empty(count, dtype=np.float64)
            mapped[proposal_order] = baseline_values
            aligned[frame, :, :, target][active] = mapped
    return aligned


def residual_blend(
    baseline: np.ndarray,
    aligned_proposal: np.ndarray,
    alpha: float,
) -> np.ndarray:
    weight = float(alpha)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("residual blend alpha must be in [0, 1]")
    reference = np.asarray(baseline, dtype=np.float64)
    proposal = np.asarray(aligned_proposal, dtype=np.float64)
    if reference.shape != proposal.shape:
        raise ValueError("baseline and aligned proposal shapes must match")
    return reference + weight * (proposal - reference)


def _blend_selection_key(item: dict[str, Any]) -> tuple[float, ...]:
    summary = item["performance"]["summary"]
    window3 = item["performance"]["multiframe_worst"]["windows"]["3"]
    return (
        float(summary["qos_feasible"]),
        float(summary["worst"]),
        float(window3["worst_mean"]),
        float(summary["cvar"]),
        float(summary["weak3"]),
        float(summary["steady"]),
        -float(item["pair_change"]["changed_frame_rate"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.trace, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    train_seeds, validation_seeds, test_seeds = seed_split(
        data["seed"],
        train_count=args.train_seeds,
        validation_count=args.validation_seeds,
        test_count=args.test_seeds,
    )
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    frozen = FrozenStructureStudent(args.student_checkpoint)

    local_candidate, current_rank, neighbors = _local_candidate_and_rank(
        data,
        np.asarray(data["local_obs"], dtype=np.float32),
        slices,
        frozen,
        neighbor_topk=args.neighbor_topk,
        target_topk=args.target_topk,
        coverage_fraction=args.coverage_fraction,
        refinement_rounds=args.neighbor_refinement_rounds,
    )
    features = _student_features(
        data, slices, neighbor_subset_mask=neighbors)

    # This tensor contains future evidence, but it is sliced only into labels
    # below.  No future-derived quantity is concatenated to model features.
    hold_target = hold_aware_score(
        data["privileged_d_eff"],
        data["episode"],
        data["p0_resolved"],
        tail_weight=1.0,
        quantile=args.quantile,
    )
    train_index = _resolved_indices(data, train_seeds)
    validation_index = _resolved_indices(data, validation_seeds)
    test_index = _resolved_indices(data, test_seeds)
    prediction_index = np.concatenate([
        train_index, validation_index, test_index])
    target_weight = hold_bottleneck_weights(
        hold_target,
        data["teacher_pair"],
        p_fa=float(np.asarray(data["p_fa"]).reshape(-1)[0]),
        qos_floor=float(np.asarray(data["qos_floor"]).reshape(-1)[0]),
        strength=args.bottleneck_weight,
    )

    residual_training = args.training_mode == "baseline_residual"
    if residual_training and (
        args.selected_edge_weight != 0.0
        or args.selected_underprediction_weight != 0.0
        or args.selection_distribution_weight != 0.0
    ):
        raise ValueError(
            "baseline_residual training requires all legacy teacher-pair "
            "weights to be zero")
    selected_label = (
        np.zeros_like(data["teacher_pair"], dtype=np.float32)
        if residual_training
        else data["teacher_pair"]
    )
    base_rank = np.asarray(current_rank, dtype=np.float64)

    predicted = _fit_factorized_edge_value_probe(
        features[train_index],
        hold_target[train_index],
        selected_label[train_index],
        features[validation_index],
        hold_target[validation_index],
        selected_label[validation_index],
        features[prediction_index],
        train_base_d_eff=(
            base_rank[train_index] if residual_training else None),
        validation_base_d_eff=(
            base_rank[validation_index] if residual_training else None),
        test_base_d_eff=(
            base_rank[prediction_index] if residual_training else None),
        train_target_weight=target_weight[train_index],
        validation_target_weight=target_weight[validation_index],
        selected_edge_weight=args.selected_edge_weight,
        selected_underprediction_weight=args.selected_underprediction_weight,
        selection_distribution_weight=args.selection_distribution_weight,
        endpoint_dim=args.endpoint_dim,
        epochs=args.epochs,
        seed=args.seed,
    )
    train_stop = len(train_index)
    validation_stop = train_stop + len(validation_index)
    train_prediction = predicted[:train_stop]
    validation_prediction = predicted[train_stop:validation_stop]
    test_prediction = predicted[validation_stop:]

    train_score = np.zeros_like(data["privileged_d_eff"], dtype=np.float64)
    train_score[train_index] = train_prediction
    train_admitted = (
        np.asarray(local_candidate, dtype=bool)
        & np.asarray(data["privileged_candidate"], dtype=bool)
    )
    train_resolved = (
        np.asarray(data["p0_resolved"], dtype=bool)
        & np.isin(data["seed"], np.asarray(train_seeds, dtype=np.int64))
    )
    fit_diagnostics = {
        "current_rank_vs_hold_target": _candidate_rank_spearman(
            current_rank, hold_target, train_admitted, train_resolved),
        "causal_rank_vs_hold_target": _candidate_rank_spearman(
            train_score, hold_target, train_admitted, train_resolved),
        "current_physics_vs_hold_target": _candidate_rank_spearman(
            data["privileged_d_eff"], hold_target,
            train_admitted, train_resolved),
    }

    def evaluate_split(
        seeds: list[int], indices: np.ndarray, values: np.ndarray
    ) -> dict[str, Any]:
        split = _subset(data, seeds)
        keep = np.isin(data["seed"], np.asarray(seeds, dtype=np.int64))
        score = np.zeros_like(split["privileged_d_eff"], dtype=np.float64)
        split_resolved = np.flatnonzero(split["p0_resolved"])
        if len(split_resolved) != len(indices):
            raise RuntimeError("resolved-frame alignment changed during subsetting")
        score[split_resolved] = values
        physical = np.asarray(split["privileged_candidate"], dtype=bool)
        frozen_candidate = np.asarray(local_candidate[keep], dtype=bool)
        baseline_rank = np.asarray(current_rank[keep], dtype=np.float64)

        token_visible, _ = delivered_target_tokens(split["local_obs"], slices)
        predictor_candidate, _ = build_local_candidate_mask(
            score,
            slices.extract_pd_hist(split["local_obs"]),
            np.asarray(neighbors[keep], dtype=bool),
            token_visible,
            target_topk=args.target_topk,
            qos_floor=float(np.asarray(split["qos_floor"]).reshape(-1)[0]),
            coverage_fraction=args.coverage_fraction,
            require_reciprocal_link=True,
        )
        common = dict(
            steady_window=args.steady_window,
            steady_floor=args.steady_floor,
            weak3_floor=args.weak3_floor,
            worst_floor=args.worst_floor,
            frame_duration_s=args.frame_duration_s,
        )
        oracle_score = hold_aware_score(
            split["privileged_d_eff"],
            split["episode"],
            split["p0_resolved"],
            tail_weight=1.0,
            quantile=args.quantile,
        )
        admitted = frozen_candidate & physical
        baseline_pairs = held_pair_sequence(
            baseline_rank, admitted, split)
        causal_pairs = held_pair_sequence(
            score, admitted, split)
        rebuilt_pairs = held_pair_sequence(
            score, predictor_candidate & physical, split)
        oracle_frozen_pairs = held_pair_sequence(
            oracle_score, admitted, split)
        oracle_pairs = held_pair_sequence(
            oracle_score, physical, split)
        causal_oracle_gated_pairs, causal_oracle_gate = (
            componentwise_hold_pareto_gate(
                split,
                baseline_pairs,
                causal_pairs,
                tolerance=args.pareto_tolerance,
                minimum_gain=args.pareto_minimum_gain,
            )
        )
        baseline = _pair_summary(split, baseline_pairs, **common)
        rank_only = _pair_summary(split, causal_pairs, **common)
        causal_oracle_gated = _pair_summary(
            split, causal_oracle_gated_pairs, **common)
        end_to_end = _pair_summary(split, rebuilt_pairs, **common)
        oracle_frozen_candidate = _pair_summary(
            split, oracle_frozen_pairs, **common)
        oracle = _pair_summary(split, oracle_pairs, **common)
        oracle_selected = oracle_pairs[split_resolved]
        admitted_resolved = (frozen_candidate & physical)[split_resolved]
        oracle_edge_count = int(np.sum(oracle_selected))
        candidate_oracle_recall = float(
            np.sum(admitted_resolved & oracle_selected)
            / max(oracle_edge_count, 1)
        )
        aligned = align_proposal_to_baseline_scale(
            baseline_rank, score, admitted)
        blend_grid: dict[str, dict[str, Any]] = {}
        for alpha in args.residual_alphas:
            blended_score = residual_blend(
                baseline_rank, aligned, float(alpha))
            blended_pairs = held_pair_sequence(
                blended_score, admitted, split)
            performance = _pair_summary(split, blended_pairs, **common)
            key = f"{float(alpha):.6g}"
            blend_grid[key] = {
                "alpha": float(alpha),
                "performance": performance,
                "delta": _delta(performance, baseline),
                "pair_change": _pair_change(
                    baseline_pairs[split_resolved],
                    blended_pairs[split_resolved],
                ),
                "hold_segment_effect": _hold_segment_effect(
                    split,
                    baseline_pairs,
                    blended_pairs,
                    steady_window=args.steady_window,
                ),
            }
        return {
            "seeds": seeds,
            "resolved_frames": int(len(indices)),
            "current_distributed_rank": baseline,
            "causal_hold_rank_frozen_candidate": rank_only,
            "causal_proposal_pareto_oracle_gate": {
                "gate": causal_oracle_gate,
                "performance": causal_oracle_gated,
                "delta": _delta(causal_oracle_gated, baseline),
                "scope": "noncausal acceptance upper bound only",
            },
            "causal_hold_rank_rebuilt_candidate": end_to_end,
            "noncausal_hold_oracle_frozen_candidate": oracle_frozen_candidate,
            "noncausal_hold_oracle_full_candidate": oracle,
            "delta_frozen_candidate": _delta(rank_only, baseline),
            "delta_rebuilt_candidate": _delta(end_to_end, baseline),
            "residual_blend_grid": blend_grid,
            "diagnostics": {
                "current_rank_vs_hold_target": _edge_value_metrics(
                    baseline_rank[split_resolved],
                    oracle_score[split_resolved],
                ),
                "causal_rank_vs_hold_target": _edge_value_metrics(
                    score[split_resolved],
                    oracle_score[split_resolved],
                ),
                "current_physics_vs_hold_target": _edge_value_metrics(
                    split["privileged_d_eff"][split_resolved],
                    oracle_score[split_resolved],
                ),
                "candidate_rank_order": {
                    "current_rank": _candidate_rank_spearman(
                        baseline_rank,
                        oracle_score,
                        admitted,
                        np.asarray(split["p0_resolved"], dtype=bool),
                    ),
                    "causal_rank": _candidate_rank_spearman(
                        score,
                        oracle_score,
                        admitted,
                        np.asarray(split["p0_resolved"], dtype=bool),
                    ),
                    "current_physics": _candidate_rank_spearman(
                        split["privileged_d_eff"],
                        oracle_score,
                        admitted,
                        np.asarray(split["p0_resolved"], dtype=bool),
                    ),
                },
                "causal_pair_change_from_baseline": _pair_change(
                    baseline_pairs[split_resolved],
                    causal_pairs[split_resolved],
                ),
                "local_candidate_recall_of_full_oracle_edges": (
                    candidate_oracle_recall
                ),
                "full_oracle_edges": oracle_edge_count,
                "pair_similarity_to_same_candidate_oracle": {
                    "baseline": _pair_similarity(
                        baseline_pairs, oracle_frozen_pairs,
                        np.asarray(split["p0_resolved"], dtype=bool)),
                    "causal": _pair_similarity(
                        causal_pairs, oracle_frozen_pairs,
                        np.asarray(split["p0_resolved"], dtype=bool)),
                },
                "hold_segment_effect": {
                    "causal_vs_baseline": _hold_segment_effect(
                        split,
                        baseline_pairs,
                        causal_pairs,
                        steady_window=args.steady_window,
                    ),
                    "same_candidate_oracle_vs_baseline": (
                        _hold_segment_effect(
                            split,
                            baseline_pairs,
                            oracle_frozen_pairs,
                            steady_window=args.steady_window,
                        )
                    ),
                },
            },
        }

    validation = evaluate_split(
        validation_seeds, validation_index, validation_prediction)
    test = evaluate_split(test_seeds, test_index, test_prediction)
    selected_alpha = max(
        validation["residual_blend_grid"],
        key=lambda key: _blend_selection_key(
            validation["residual_blend_grid"][key]),
    )
    return {
        "schema": "causal-hold-qos-graph-rank-v1",
        "scope": (
            "offline same-state seed-disjoint audit; future hold evidence is "
            "a training label only; not closed-loop or K16 evidence"
        ),
        "trace": str(args.trace),
        "student_checkpoint": str(args.student_checkpoint),
        "model": {
            "family": "factorized shared-endpoint graph ranker",
            "training_mode": args.training_mode,
            "legacy_teacher_pair_weighting": not residual_training,
            "quantile": float(args.quantile),
            "endpoint_dim": int(args.endpoint_dim),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "bottleneck_weight": float(args.bottleneck_weight),
            "inference_inputs": (
                "current pre-decision local observation, delivered tokens, "
                "current communication/resource context"
            ),
            "training_target": "future hold-segment d_eff lower-tail",
        },
        "split": {
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "test_seeds": test_seeds,
            "train_resolved_frames": int(len(train_index)),
            "validation_resolved_frames": int(len(validation_index)),
            "test_resolved_frames": int(len(test_index)),
        },
        "fit_rank_diagnostics": fit_diagnostics,
        "residual_blend_selection": {
            "selection_split": "validation only",
            "selection_order": [
                "qos_feasible", "worst", "W3_worst_mean", "cvar",
                "weak3", "steady", "minimum_structure_change",
            ],
            "selected_alpha": float(selected_alpha),
            "test_result_key": selected_alpha,
        },
        "validation": validation,
        "test": test,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, default=12)
    parser.add_argument("--validation-seeds", type=int, default=3)
    parser.add_argument("--test-seeds", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument(
        "--training-mode",
        choices=("absolute", "baseline_residual"),
        default="absolute",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--endpoint-dim", type=int, default=32)
    parser.add_argument("--selected-edge-weight", type=float, default=2.0)
    parser.add_argument("--selected-underprediction-weight", type=float, default=0.0)
    parser.add_argument("--selection-distribution-weight", type=float, default=0.05)
    parser.add_argument("--bottleneck-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--neighbor-topk", type=int, default=4)
    parser.add_argument("--target-topk", type=int, default=4)
    parser.add_argument("--coverage-fraction", type=float, default=0.30)
    parser.add_argument("--neighbor-refinement-rounds", type=int, default=1)
    parser.add_argument("--steady-window", type=int, default=20)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    parser.add_argument("--frame-duration-s", type=float, default=0.1)
    parser.add_argument("--pareto-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--pareto-minimum-gain", type=float, default=1.0e-6)
    parser.add_argument(
        "--residual-alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "validation_delta": report["validation"]["delta_frozen_candidate"],
        "test_delta": report["test"]["delta_frozen_candidate"],
        "test_rebuilt_delta": report["test"]["delta_rebuilt_candidate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
