"""Deterministic target-wise arbitration between frozen structure experts.

The arbitration score is deliberately a structural support ceiling rather
than a claimed detector probability.  It is used only to decide which public
edge-value expert supplies one target before the existing role/owner/capacity
solver and physical certificate are applied.
"""

from __future__ import annotations

import numpy as np


def target_support_ceiling(
    edge_values: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    pair_limit: int,
) -> np.ndarray:
    """Return the best receiver-local Top-B support for every target.

    Inputs have trailing shape ``(K_tx, K_rx, Q)`` and may contain arbitrary
    leading batch axes.  The score relaxes the cross-target single-role and
    receiver-capacity constraints, so it is an upper support proxy, not a
    feasibility or detection certificate.
    """
    values = np.asarray(edge_values, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if values.shape != mask.shape or values.ndim < 3:
        raise ValueError(
            "edge_values/candidate_mask must share trailing shape (K,K,Q)")
    if values.shape[-3] != values.shape[-2]:
        raise ValueError("transmitter and receiver cardinalities must match")
    if np.any(~np.isfinite(values)):
        raise ValueError("edge_values must be finite")
    limit = int(pair_limit)
    if limit < 1:
        raise ValueError("pair_limit must be positive")
    k_count = values.shape[-3]
    diagonal = np.eye(k_count, dtype=bool)
    diagonal = diagonal.reshape(
        (1,) * (values.ndim - 3) + (k_count, k_count, 1))
    valid = mask & ~diagonal
    nonnegative = np.where(valid, np.maximum(values, 0.0), 0.0)
    take = min(limit, k_count)
    # Sort over transmitters, sum the strongest feasible transmitters for each
    # receiver/target, then take the best receiver-local fusion owner.
    strongest = np.sort(nonnegative, axis=-3)[..., -take:, :, :]
    per_receiver = np.sum(strongest, axis=-3)
    return np.max(per_receiver, axis=-2)


def conservative_target_expert_choice(
    primary_score: np.ndarray,
    alternate_score: np.ndarray,
    *,
    relative_margin: float = 0.0,
    absolute_margin: float = 0.0,
) -> np.ndarray:
    """Choose the alternate only after a strict, scale-aware improvement.

    Ties and unresolved margins stay with the primary expert.  The margins are
    diagnostic pre-certificate guards; deployment margins must be calibrated
    from independent event residuals.
    """
    primary = np.asarray(primary_score, dtype=np.float64)
    alternate = np.asarray(alternate_score, dtype=np.float64)
    if primary.shape != alternate.shape:
        raise ValueError("expert scores must have identical shapes")
    if np.any(~np.isfinite(primary)) or np.any(~np.isfinite(alternate)):
        raise ValueError("expert scores must be finite")
    rel = float(relative_margin)
    absolute = float(absolute_margin)
    if not np.isfinite(rel) or rel < 0.0:
        raise ValueError("relative_margin must be finite and non-negative")
    if not np.isfinite(absolute) or absolute < 0.0:
        raise ValueError("absolute_margin must be finite and non-negative")
    required = np.maximum(absolute, rel * np.maximum(primary, 0.0))
    improvement = alternate - primary
    numerical_tolerance = 8.0 * np.finfo(np.float64).eps * np.maximum.reduce([
        np.ones_like(primary),
        np.abs(primary),
        np.abs(alternate),
        np.abs(required),
    ])
    return improvement > required + numerical_tolerance


def blend_target_edge_values(
    primary: np.ndarray,
    alternate: np.ndarray,
    choose_alternate: np.ndarray,
) -> np.ndarray:
    """Select one complete expert edge field independently per target."""
    first = np.asarray(primary, dtype=np.float64)
    second = np.asarray(alternate, dtype=np.float64)
    choice = np.asarray(choose_alternate, dtype=bool)
    if first.shape != second.shape or first.ndim < 3:
        raise ValueError("expert edge tensors must share shape (...,K,K,Q)")
    if choice.shape != first.shape[:-3] + (first.shape[-1],):
        raise ValueError("target choice must have shape (...,Q)")
    selector = choice[..., None, None, :]
    return np.where(selector, second, first)
