"""Episode-level joint residual calibration for feedback-gated repairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class FeedbackGateDecision:
    accept: bool
    candidate_lower: float
    noop_lower: float
    delta_lower: float
    charged_cost: float


def joint_overprediction_score(
    *,
    candidate_estimated: float,
    candidate_realized: float,
    noop_estimated: float,
    noop_realized: float,
    candidate_component_score: float = 0.0,
    noop_component_score: float = 0.0,
) -> float:
    """One score jointly covers candidate, No-op and paired improvement."""
    values = np.asarray([
        candidate_component_score,
        noop_component_score,
        candidate_estimated - candidate_realized,
        noop_estimated - noop_realized,
        (candidate_estimated - noop_estimated)
        - (candidate_realized - noop_realized),
        0.0,
    ], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("joint residual inputs must be finite")
    return float(np.max(values))


def episode_max_scores(
    rows: Iterable[Mapping[str, object]],
    *,
    candidate_prefix: str = "causal_geometry",
) -> dict[int, float]:
    """Collapse arbitrarily many correlated resolves to one score per episode."""
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if not bool(row.get(f"{candidate_prefix}_available", False)):
            continue
        seed = int(row["seed"])
        score = joint_overprediction_score(
            candidate_estimated=float(
                row[f"{candidate_prefix}_estimated_worst"]),
            candidate_realized=float(row[f"{candidate_prefix}_worst"]),
            noop_estimated=float(
                row[f"{candidate_prefix}_noop_estimated_worst"]),
            noop_realized=float(row["deployed_worst"]),
            candidate_component_score=float(
                row[f"{candidate_prefix}_calibration_score"]),
            noop_component_score=float(
                row[f"{candidate_prefix}_noop_calibration_score"]),
        )
        grouped.setdefault(seed, []).append(score)
    if not grouped or any(not values for values in grouped.values()):
        raise ValueError("no complete episode residuals were supplied")
    return {seed: float(np.max(values)) for seed, values in grouped.items()}


def split_conformal_upper(
    episode_scores: Iterable[float],
    *,
    alpha: float,
) -> tuple[float, float, int]:
    """Return the finite-sample upper order statistic and coverage floor."""
    values = np.sort(np.asarray(list(episode_scores), dtype=np.float64))
    risk = float(alpha)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("episode_scores must be non-empty")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("episode_scores must be finite and non-negative")
    if not (0.0 < risk < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    rank_one_based = min(
        int(np.ceil((values.size + 1) * (1.0 - risk))), values.size)
    margin = float(values[rank_one_based - 1])
    coverage_floor = float(rank_one_based / (values.size + 1))
    return margin, coverage_floor, rank_one_based


def certified_feedback_decision(
    *,
    candidate_estimated: float,
    noop_estimated: float,
    joint_margin: float,
    qos_floor: float,
    switch_cost: float = 0.0,
    bit_cost: float = 0.0,
    delay_cost: float = 0.0,
    energy_cost: float = 0.0,
) -> FeedbackGateDecision:
    """Apply absolute QoS preservation and strict net-improvement tests."""
    values = np.asarray([
        candidate_estimated, noop_estimated, joint_margin, qos_floor,
        switch_cost, bit_cost, delay_cost, energy_cost,
    ], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("feedback gate inputs must be finite")
    if not (0.0 <= candidate_estimated <= 1.0
            and 0.0 <= noop_estimated <= 1.0
            and 0.0 < qos_floor <= 1.0):
        raise ValueError("probability inputs must lie in [0,1]")
    if np.any(values[2:] < 0.0):
        raise ValueError("margin and costs must be non-negative")
    candidate_lower = max(float(candidate_estimated - joint_margin), 0.0)
    noop_lower = max(float(noop_estimated - joint_margin), 0.0)
    delta_lower = float(candidate_estimated - noop_estimated - joint_margin)
    charged = float(switch_cost + bit_cost + delay_cost + energy_cost)
    absolute_safe = candidate_lower >= min(noop_lower, float(qos_floor))
    return FeedbackGateDecision(
        accept=bool(absolute_safe and delta_lower > charged),
        candidate_lower=candidate_lower,
        noop_lower=noop_lower,
        delta_lower=delta_lower,
        charged_cost=charged,
    )
