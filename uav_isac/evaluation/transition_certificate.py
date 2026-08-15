"""Event-level conformal certificate for transition safety losses.

The observed quantity is a *paired safety loss*: positive means that a
candidate is worse than its no-op comparator.  A predictor estimates every
candidate/horizon/tail loss and a strictly positive local scale.  Calibration
collapses all dependent items from one event to one maximum standardized
underestimate, so the resulting upper bound remains valid after candidate
selection within a frozen proposal pipeline.
"""

from __future__ import annotations

import math

import numpy as np


def event_max_standardized_underestimate(
    predicted_loss: np.ndarray,
    observed_loss: np.ndarray,
    uncertainty_scale: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    minimum_scale: float = 1.0e-9,
) -> float:
    """Return one post-selection-safe nonconformity score for an event.

    Arrays may contain any frozen collection of candidates, transition frames,
    and worst-k tail constraints.  They are deliberately reduced to one score
    rather than treated as independent calibration examples.
    """
    predicted = np.asarray(predicted_loss, dtype=np.float64)
    observed = np.asarray(observed_loss, dtype=np.float64)
    scale = np.asarray(uncertainty_scale, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.shape != scale.shape:
        raise ValueError("predicted, observed, and scale shapes must match")
    if valid_mask is None:
        valid = np.ones(predicted.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != predicted.shape:
            raise ValueError("valid_mask shape must match the losses")
    if not np.any(valid):
        raise ValueError("event has no required transition constraints")
    finite = np.isfinite(predicted) & np.isfinite(observed) & np.isfinite(scale)
    if np.any(valid & ~finite):
        # A required post-transition observation that is missing or invalid is
        # a safety failure, not a removable calibration sample.  Explicitly
        # padded/nonexistent items must be excluded by a pre-registered mask.
        return float("inf")
    if np.any(scale[valid] <= 0.0):
        raise ValueError("uncertainty scales must be positive")
    normalized = (
        observed[valid] - predicted[valid]
    ) / np.maximum(scale[valid], float(minimum_scale))
    return float(np.max(normalized))


def split_conformal_upper_multiplier(
    event_scores: np.ndarray,
    *,
    miscoverage: float,
) -> float:
    """Finite-sample split-conformal multiplier for an upper safety bound.

    The order statistic is ``ceil((n+1)*(1-alpha))``.  If the requested
    coverage cannot be resolved with ``n`` calibration events, infinity is
    returned, forcing every non-trivial candidate to fail closed.
    """
    alpha = float(miscoverage)
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")
    scores = np.asarray(event_scores, dtype=np.float64).reshape(-1)
    if np.any(np.isnan(scores)) or np.any(np.isneginf(scores)):
        raise ValueError("event scores cannot contain NaN or negative infinity")
    if not scores.size:
        raise ValueError("at least one event score is required")
    rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def transition_loss_upper_bound(
    predicted_loss: np.ndarray,
    uncertainty_scale: np.ndarray,
    *,
    multiplier: float,
) -> np.ndarray:
    """Joint conformal upper bound on paired transition safety loss."""
    predicted = np.asarray(predicted_loss, dtype=np.float64)
    scale = np.asarray(uncertainty_scale, dtype=np.float64)
    if predicted.shape != scale.shape:
        raise ValueError("predicted loss and uncertainty scale shapes must match")
    if np.any(~np.isfinite(predicted)) or np.any(~np.isfinite(scale)):
        raise ValueError("predicted losses and scales must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("uncertainty scales must be positive")
    beta = float(multiplier)
    if math.isnan(beta):
        raise ValueError("multiplier cannot be NaN")
    return predicted + beta * scale


def transition_candidate_is_certified(
    predicted_loss: np.ndarray,
    uncertainty_scale: np.ndarray,
    *,
    multiplier: float,
    safety_margin: float = 0.0,
) -> bool:
    """Fail-closed decision: every joint upper loss must be non-positive."""
    upper = transition_loss_upper_bound(
        predicted_loss,
        uncertainty_scale,
        multiplier=multiplier,
    )
    return bool(np.all(upper <= -float(safety_margin)))


def safe_reconfiguration_is_certified(
    predicted_loss: np.ndarray,
    uncertainty_scale: np.ndarray,
    *,
    multiplier: float,
    commit_feasible: bool,
    structural_feasible: bool,
    safety_margin: float = 0.0,
) -> bool:
    """Intersection of structural, communication and transition certificates.

    This Boolean conjunction is intentional: quantities in bits, seconds,
    watts and detection probability have no canonical additive conversion.
    An infeasible hard constraint therefore cannot be offset by a larger
    predicted detection gain.
    """
    if not bool(structural_feasible) or not bool(commit_feasible):
        return False
    return transition_candidate_is_certified(
        predicted_loss,
        uncertainty_scale,
        multiplier=multiplier,
        safety_margin=safety_margin,
    )
