"""Audit the evidence-fusion value currently supplied for free by the env."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from uav_isac.physical.evidence import (
    local_quality_topk_mask,
    receiver_deflection_from_selected,
    scheduled_fusion_owner,
)
from uav_isac.utils.math_utils import compute_PD
from uav_isac.utils.sentinels import OWNER_INDEX_NONE


def lossless_quality_topk_pd(
    receiver_deflection: np.ndarray,
    p_fa: float,
    topk: int,
    fusion_owner: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Lossless evidence-routing capacity for a local quality Top-k rule.

    Each receiver ranks only its own positive per-target deflections before
    observing a stochastic LLR. The target owner is supplied by the scheduled
    hyperedge directory and receives every selected peer statistic without
    quantization, delay, or loss.

    This is intentionally an upper-bound screen for the target-selection rule,
    not a deployable communication result.
    """
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    if receiver_d.ndim != 2:
        raise ValueError("receiver_deflection must have shape (K, Q)")
    if np.any(~np.isfinite(receiver_d)) or np.any(receiver_d < 0.0):
        raise ValueError(
            "receiver_deflection must contain finite non-negative values")
    K, Q = receiver_d.shape
    if K < 1 or Q < 1:
        raise ValueError("receiver_deflection must be non-empty")

    selected = local_quality_topk_mask(receiver_d, topk)

    owner = np.asarray(fusion_owner, dtype=np.int64).reshape(-1)
    if owner.shape != (Q,):
        raise ValueError("fusion_owner must have shape (Q,)")
    if np.any((owner < OWNER_INDEX_NONE) | (owner >= K)):
        raise ValueError("fusion_owner contains an invalid receiver index")
    owned = owner >= 0
    fused_d = np.zeros(Q, dtype=np.float64)
    fused_d[owned] = receiver_d[
        owner[owned], np.arange(Q)[owned]]
    for source in range(K):
        add = selected[source].copy()
        add &= owned & (owner != source)
        fused_d[add] += receiver_d[source, add]
    return {
        "pd": compute_PD(fused_d, p_fa),
        "fused_deflection": fused_d,
        "selected_mask": selected,
        "owner": owner,
    }


def receiver_local_and_global_pd(
    selected_set: Sequence[Tuple[int, int, int]],
    deflection_entries: Iterable,
    num_agents: int,
    num_targets: int,
    p_fa: float,
) -> Dict[str, np.ndarray]:
    """Reconstruct receiver-local and globally fused detection probabilities.

    ``local_best_pd[q]`` is the best detection probability attainable by one
    receiver without exchanging sensing sufficient statistics.  ``global_pd``
    is the current environment semantics: add deflection from every selected
    receiver before applying the detector mapping.
    """
    receiver_d = receiver_deflection_from_selected(
        selected_set,
        deflection_entries,
        num_agents,
        num_targets,
    )
    receiver_pd = np.stack([
        compute_PD(receiver_d[k], p_fa)
        for k in range(num_agents)
    ])
    global_d = np.sum(receiver_d, axis=0)
    owner = scheduled_fusion_owner(
        selected_set, num_agents, num_targets)
    top1 = lossless_quality_topk_pd(
        receiver_d, p_fa, topk=1, fusion_owner=owner)
    top2 = lossless_quality_topk_pd(
        receiver_d, p_fa, topk=2, fusion_owner=owner)
    return {
        "receiver_deflection": receiver_d,
        "receiver_pd": receiver_pd,
        "local_best_pd": np.max(receiver_pd, axis=0),
        "global_pd": compute_PD(global_d, p_fa),
        "lossless_quality_top1_pd": top1["pd"],
        "lossless_quality_top2_pd": top2["pd"],
        "lossless_quality_top1_mask": top1["selected_mask"],
        "lossless_quality_top2_mask": top2["selected_mask"],
        "scheduled_evidence_owner": top1["owner"],
    }


def _episode_metrics(history: np.ndarray, steady_window: int) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64)
    width = min(max(1, int(steady_window)), len(values))
    per_target = np.mean(values[-width:], axis=0)
    ordered = np.sort(per_target)
    return np.asarray([
        float(np.mean(per_target)),
        float(np.mean(ordered[:min(3, ordered.size)])),
        float(ordered[0]),
    ])


def summarize_evidence_oracle(
    global_histories: Sequence[np.ndarray],
    local_best_histories: Sequence[np.ndarray],
    steady_window: int = 20,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260723,
) -> Dict[str, object]:
    """Paired episode-level oracle gap for steady, weak3, and worst."""
    if len(global_histories) != len(local_best_histories):
        raise ValueError("global and local evidence histories must align")
    global_metrics = []
    local_metrics = []
    for global_history, local_history in zip(
            global_histories, local_best_histories):
        if len(global_history) == 0 or len(local_history) == 0:
            continue
        global_metrics.append(
            _episode_metrics(global_history, steady_window))
        local_metrics.append(
            _episode_metrics(local_history, steady_window))
    if not global_metrics:
        return {
            "eval_evidence_oracle_episode_count": 0,
            "eval_evidence_oracle_gate_pass": False,
        }

    global_array = np.asarray(global_metrics)
    local_array = np.asarray(local_metrics)
    delta = global_array - local_array
    metric_names = ("steady", "weak3", "worst")
    summary: Dict[str, object] = {
        "eval_evidence_oracle_episode_count": int(len(delta)),
        "eval_evidence_oracle_note": (
            "same selected edges, movement, and power; local-only is the "
            "best single receiver per target, centralized evidence is the "
            "environment's current unconditional sum of receiver deflection"),
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    episode_count = len(delta)
    sampled_indices = rng.integers(
        0,
        episode_count,
        size=(max(1, int(bootstrap_samples)), episode_count),
    )
    for index, name in enumerate(metric_names):
        summary[f"eval_evidence_oracle_global_{name}"] = float(
            np.mean(global_array[:, index]))
        summary[f"eval_evidence_oracle_local_{name}"] = float(
            np.mean(local_array[:, index]))
        summary[f"eval_evidence_oracle_{name}_delta"] = float(
            np.mean(delta[:, index]))
        bootstrap_means = np.mean(
            delta[sampled_indices, index], axis=1)
        summary[f"eval_evidence_oracle_{name}_delta_ci95"] = [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ]

    worst_delta = float(summary[
        "eval_evidence_oracle_worst_delta"])
    worst_ci = summary[
        "eval_evidence_oracle_worst_delta_ci95"]
    summary["eval_evidence_oracle_gate_pass"] = bool(
        episode_count >= 10
        and worst_delta >= 0.03
        and worst_ci[0] > 0.01
    )
    summary["eval_evidence_oracle_gate_rule"] = (
        "at least 10 paired episodes, centralized-minus-best-local mean "
        "worst >= 0.03, and its episode-bootstrap 95% lower bound > 0.01"
    )
    return summary


def summarize_lossless_topk_capacity(
    global_histories: Sequence[np.ndarray],
    local_best_histories: Sequence[np.ndarray],
    topk_histories: Dict[int, Sequence[np.ndarray]],
    steady_window: int = 20,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260723,
) -> Dict[str, object]:
    """Paired capacity screen for pre-observation local-quality Top-k.

    Recovery is measured relative to the same-frame best-local and central
    evidence bounds.  The screen deliberately excludes quantization and
    transport impairments; failure here means a later packet implementation
    cannot rescue the selected Top-k rule.
    """
    episode_count = len(global_histories)
    if len(local_best_histories) != episode_count:
        raise ValueError("global and local histories must align")
    for topk, histories in topk_histories.items():
        if len(histories) != episode_count:
            raise ValueError(f"Top-{topk} histories must align")
    if episode_count == 0:
        return {
            "eval_evidence_topk_capacity_episode_count": 0,
            "eval_evidence_topk_capacity_gate_pass": False,
        }

    # Audit 2026-08-25: mirror the empty-history guard of
    # summarize_evidence_oracle() -- an empty per-episode history previously
    # flowed into _episode_metrics() and produced NaN/IndexError instead of
    # a clean fail-closed result.
    kept_episodes = 0
    global_metrics = []
    local_metrics = []
    for global_history, local_history in zip(
            global_histories, local_best_histories):
        if len(global_history) == 0 or len(local_history) == 0:
            continue
        kept_episodes += 1
        global_metrics.append(
            _episode_metrics(global_history, steady_window))
        local_metrics.append(
            _episode_metrics(local_history, steady_window))
    topk_metrics = {
        int(topk): np.asarray([
            _episode_metrics(history, steady_window)
            for history, local in zip(histories, local_best_histories)
            if len(history) > 0 and len(local) > 0
        ])
        for topk, histories in topk_histories.items()
    }
    if kept_episodes == 0:
        return {
            "eval_evidence_topk_capacity_episode_count": 0,
            "eval_evidence_topk_capacity_gate_pass": False,
        }
    global_metrics = np.asarray(global_metrics)
    local_metrics = np.asarray(local_metrics)
    metric_names = ("steady", "weak3", "worst")
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = max(1, int(bootstrap_samples))
    sampled_indices = rng.integers(
        0, episode_count, size=(draws, episode_count))

    summary: Dict[str, object] = {
        "eval_evidence_topk_capacity_episode_count": int(episode_count),
        "eval_evidence_topk_capacity_note": (
            "lossless/unquantized upper-bound; each receiver selects targets "
            "by its own positive pre-observation deflection; a deterministic "
            "pre-evidence owner retains local evidence and receives selected "
            "peer evidence"),
    }
    oracle_delta = global_metrics - local_metrics
    for topk, metrics in sorted(topk_metrics.items()):
        delta = metrics - local_metrics
        for index, name in enumerate(metric_names):
            prefix = f"eval_evidence_quality_top{topk}_{name}"
            mean_local = float(np.mean(local_metrics[:, index]))
            mean_oracle = float(np.mean(global_metrics[:, index]))
            mean_topk = float(np.mean(metrics[:, index]))
            denominator = mean_oracle - mean_local
            recovery = (
                (mean_topk - mean_local) / denominator
                if denominator > 1e-12 else 0.0)
            summary[prefix] = mean_topk
            summary[f"{prefix}_delta_vs_local"] = float(
                np.mean(delta[:, index]))
            summary[f"{prefix}_recovery"] = float(recovery)

            bootstrap_delta = np.mean(
                delta[sampled_indices, index], axis=1)
            bootstrap_oracle = np.mean(
                oracle_delta[sampled_indices, index], axis=1)
            bootstrap_recovery = np.divide(
                bootstrap_delta,
                bootstrap_oracle,
                out=np.zeros_like(bootstrap_delta),
                where=bootstrap_oracle > 1e-12,
            )
            summary[f"{prefix}_delta_vs_local_ci95"] = [
                float(np.quantile(bootstrap_delta, 0.025)),
                float(np.quantile(bootstrap_delta, 0.975)),
            ]
            summary[f"{prefix}_recovery_ci95"] = [
                float(np.quantile(bootstrap_recovery, 0.025)),
                float(np.quantile(bootstrap_recovery, 0.975)),
            ]

    top1_recovery = float(summary.get(
        "eval_evidence_quality_top1_worst_recovery", 0.0))
    top1_delta_ci = summary.get(
        "eval_evidence_quality_top1_worst_delta_vs_local_ci95",
        [0.0, 0.0],
    )
    summary["eval_evidence_topk_capacity_gate_pass"] = bool(
        episode_count >= 10
        and top1_recovery >= 0.50
        and float(top1_delta_ci[0]) > 0.0
    )
    summary["eval_evidence_topk_capacity_gate_rule"] = (
        "at least 10 paired episodes; lossless local-quality Top-1 recovers "
        "at least 50% of central-minus-local worst gain; paired bootstrap "
        "95% lower bound for Top-1-minus-local worst is positive"
    )
    return summary
