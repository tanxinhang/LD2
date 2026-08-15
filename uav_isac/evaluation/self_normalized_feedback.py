"""Target-wise self-normalized conformal gates for ISAC feedback control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class TargetwiseFeedbackDecision:
    accept: bool
    candidate_lower: np.ndarray
    noop_lower: np.ndarray
    noop_upper: np.ndarray
    target_safe: np.ndarray
    worst_delta_lower: float
    charged_cost: float


def _vector(values: object, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 1 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite non-empty vector")
    return result


def targetwise_normalized_score(
    *,
    candidate_estimated: np.ndarray,
    candidate_realized: np.ndarray,
    noop_estimated: np.ndarray,
    noop_realized: np.ndarray,
    absolute_uncertainty: np.ndarray,
    noop_upper_uncertainty: np.ndarray,
) -> float:
    """Normalize exactly the residuals consumed by the feedback gate.

    The gate consumes candidate/No-op lower bounds and a No-op upper bound.
    Their target-wise simultaneous envelope implies the worst-target gain
    bound by monotonicity of ``min``; no extra scalar residual is needed.
    """
    candidate_est = _vector(candidate_estimated, "candidate_estimated")
    candidate_true = _vector(candidate_realized, "candidate_realized")
    noop_est = _vector(noop_estimated, "noop_estimated")
    noop_true = _vector(noop_realized, "noop_realized")
    absolute_scale = _vector(absolute_uncertainty, "absolute_uncertainty")
    noop_upper_scale = _vector(
        noop_upper_uncertainty, "noop_upper_uncertainty")
    shape = candidate_est.shape
    if any(value.shape != shape for value in (
        candidate_true, noop_est, noop_true, absolute_scale,
        noop_upper_scale,
    )):
        raise ValueError("target-wise feedback vectors must have equal shape")
    if (
        np.any(absolute_scale <= 0.0)
        or np.any(noop_upper_scale <= 0.0)
    ):
        raise ValueError("uncertainty scales must be finite and positive")

    candidate_score = np.maximum(
        candidate_est - candidate_true, 0.0) / absolute_scale
    noop_score = np.maximum(noop_est - noop_true, 0.0) / absolute_scale
    noop_upper_score = np.maximum(
        noop_true - noop_est, 0.0) / noop_upper_scale
    return float(max(
        np.max(candidate_score),
        np.max(noop_score),
        np.max(noop_upper_score),
        0.0,
    ))


def episode_max_normalized_scores(
    rows: Iterable[Mapping[str, object]],
    *,
    candidate_prefix: str = "causal_routed",
) -> dict[int, float]:
    """Collapse all correlated frames and targets to one score per episode."""
    prefix = str(candidate_prefix)
    grouped: dict[int, float] = {}
    for row in rows:
        if not bool(row.get(f"{prefix}_available", False)):
            continue
        seed = int(row["seed"])
        score = targetwise_normalized_score(
            candidate_estimated=np.asarray(
                row[f"{prefix}_estimated_pd"], dtype=np.float64),
            candidate_realized=np.asarray(
                row[f"{prefix}_realized_pd"], dtype=np.float64),
            noop_estimated=np.asarray(
                row[f"{prefix}_noop_estimated_pd"], dtype=np.float64),
            noop_realized=np.asarray(row["deployed_pd"], dtype=np.float64),
            absolute_uncertainty=np.asarray(
                row[f"{prefix}_absolute_uncertainty"], dtype=np.float64),
            noop_upper_uncertainty=np.asarray(
                row[f"{prefix}_noop_upper_uncertainty"], dtype=np.float64),
        )
        grouped[seed] = max(grouped.get(seed, 0.0), score)
    if not grouped:
        raise ValueError("no target-wise candidate actions were supplied")
    return grouped


def certified_targetwise_feedback_decision(
    *,
    candidate_estimated: np.ndarray,
    noop_estimated: np.ndarray,
    absolute_uncertainty: np.ndarray,
    noop_upper_uncertainty: np.ndarray,
    normalized_margin: float,
    qos_floor: float,
    switch_cost: float = 0.0,
    bit_cost: float = 0.0,
    delay_cost: float = 0.0,
    energy_cost: float = 0.0,
) -> TargetwiseFeedbackDecision:
    """Apply simultaneous per-target preservation and worst-target net gain."""
    candidate = _vector(candidate_estimated, "candidate_estimated")
    noop = _vector(noop_estimated, "noop_estimated")
    scale = _vector(absolute_uncertainty, "absolute_uncertainty")
    upper_scale = _vector(
        noop_upper_uncertainty, "noop_upper_uncertainty")
    if (
        candidate.shape != noop.shape or candidate.shape != scale.shape
        or candidate.shape != upper_scale.shape
    ):
        raise ValueError("candidate, No-op and uncertainty shapes must match")
    margin = float(normalized_margin)
    floor = float(qos_floor)
    costs = np.asarray(
        [switch_cost, bit_cost, delay_cost, energy_cost], dtype=np.float64)
    if (
        not np.isfinite(margin) or margin < 0.0
        or not np.isfinite(floor) or not (0.0 < floor <= 1.0)
        or np.any(~np.isfinite(costs)) or np.any(costs < 0.0)
        or np.any(scale <= 0.0) or np.any(upper_scale <= 0.0)
    ):
        raise ValueError("margin, uncertainty, QoS and costs are invalid")
    if np.any((candidate < 0.0) | (candidate > 1.0)) or np.any(
        (noop < 0.0) | (noop > 1.0)
    ):
        raise ValueError("detection probabilities must lie in [0,1]")

    candidate_lower = np.clip(candidate - margin * scale, 0.0, 1.0)
    noop_lower = np.clip(noop - margin * scale, 0.0, 1.0)
    noop_upper = np.clip(noop + margin * upper_scale, 0.0, 1.0)
    target_safe = candidate_lower + 1.0e-12 >= np.minimum(
        noop_lower, floor)
    worst_delta_lower = float(
        np.min(candidate_lower) - np.min(noop_upper))
    charged = float(np.sum(costs))
    return TargetwiseFeedbackDecision(
        accept=bool(np.all(target_safe) and worst_delta_lower > charged),
        candidate_lower=candidate_lower,
        noop_lower=noop_lower,
        noop_upper=noop_upper,
        target_safe=target_safe,
        worst_delta_lower=worst_delta_lower,
        charged_cost=charged,
    )
