"""Privileged future-outcome audit for a causal ISAC horizon envelope.

The routines in this module deliberately separate two information sets:

* the controller constructs an envelope from information available at the
  intervention frame; and
* a frozen, frame-by-frame trace is consumed only after the decision as an
  audit label.

The future trace therefore has no ranking or commit authority.  In
particular, it represents an open-loop movement-action tape, not a claim that
the counterfactual structure/RF action would leave a feedback movement policy
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix
from uav_isac.physical.detection import compute_detection_probabilities


@dataclass(frozen=True)
class SimultaneousLogEnvelopeScore:
    """One simultaneous coefficient-envelope residual.

    ``joint`` is the maximum residual over every audited horizon step, edge,
    and target.  A positive lower bound covering a zero true coefficient, or
    a zero upper bound covering a positive coefficient, has infinite score;
    no finite multiplicative margin can repair such a support error.
    """

    lower: float
    upper: float
    joint: float
    audited_coefficient_count: int
    lower_zero_support_failure_count: int
    upper_zero_support_failure_count: int


@dataclass(frozen=True)
class HorizonFutureOutcome:
    """Privileged H-step physical outcome of one frozen controller decision."""

    candidate_pd: np.ndarray
    noop_pd: np.ndarray
    coefficient_score: SimultaneousLogEnvelopeScore
    candidate_lower_failure: bool
    noop_upper_failure: bool
    target_no_harm: np.ndarray


def consecutive_horizon_indices(
    seeds: np.ndarray,
    frames: np.ndarray,
    current_index: int,
    steps: int,
) -> np.ndarray | None:
    """Return ``[row, ..., row+H-1]`` only for one contiguous episode.

    Returning ``None`` rather than clipping at the episode boundary is the
    fail-closed behavior required for simultaneous H-step auditing.
    """
    seed_values = np.asarray(seeds, dtype=np.int64).reshape(-1)
    frame_values = np.asarray(frames, dtype=np.int64).reshape(-1)
    if seed_values.shape != frame_values.shape:
        raise ValueError("seeds and frames must have identical 1-D shape")
    row = int(current_index)
    horizon = int(steps)
    if horizon <= 0:
        raise ValueError("steps must be positive")
    if row < 0 or row >= seed_values.size:
        raise IndexError("current_index is outside the trace")
    stop = row + horizon
    if stop > seed_values.size:
        return None
    indices = np.arange(row, stop, dtype=np.int64)
    expected_frames = frame_values[row] + np.arange(horizon, dtype=np.int64)
    if (
        np.any(seed_values[indices] != seed_values[row])
        or not np.array_equal(frame_values[indices], expected_frames)
    ):
        return None
    return indices


def simultaneous_log_envelope_score(
    lower: np.ndarray,
    upper: np.ndarray,
    actual: np.ndarray,
    audit_mask: np.ndarray,
) -> SimultaneousLogEnvelopeScore:
    """Compute the two-sided multiplicative residual over one episode event.

    For every audited coefficient ``a`` and causal interval ``[L,U]``, the
    smallest symmetric log margin is

    ``max([log(L/a)]_+, [log(a/U)]_+)``.

    The maximum over H x edge x target gives one simultaneous event score.
    A later episode score is the maximum over all eligible events, preserving
    the episode as the independent exchangeability unit.
    """
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    truth = np.asarray(actual, dtype=np.float64)
    mask = np.asarray(audit_mask, dtype=bool)
    if low.shape != high.shape or low.shape != truth.shape:
        raise ValueError("lower, upper, and actual must have identical shape")
    try:
        mask = np.broadcast_to(mask, low.shape)
    except ValueError as exc:
        raise ValueError("audit_mask is not broadcast-compatible") from exc
    values = (low, high, truth)
    if any(np.any(~np.isfinite(value)) for value in values):
        raise ValueError("coefficient envelopes and outcomes must be finite")
    if (
        np.any(low < 0.0) or np.any(high < low)
        or np.any(truth < 0.0)
    ):
        raise ValueError("coefficient envelopes and outcomes must be ordered")

    low_m = low[mask]
    high_m = high[mask]
    truth_m = truth[mask]
    lower_zero = (low_m > 0.0) & (truth_m <= 0.0)
    upper_zero = (high_m <= 0.0) & (truth_m > 0.0)

    lower_score = 0.0
    valid_lower = (low_m > 0.0) & (truth_m > 0.0)
    if np.any(valid_lower):
        lower_score = float(max(
            0.0,
            np.max(np.log(low_m[valid_lower] / truth_m[valid_lower])),
        ))
    upper_score = 0.0
    valid_upper = (high_m > 0.0) & (truth_m > 0.0)
    if np.any(valid_upper):
        upper_score = float(max(
            0.0,
            np.max(np.log(truth_m[valid_upper] / high_m[valid_upper])),
        ))
    if np.any(lower_zero):
        lower_score = float("inf")
    if np.any(upper_zero):
        upper_score = float("inf")
    return SimultaneousLogEnvelopeScore(
        lower=lower_score,
        upper=upper_score,
        joint=float(max(lower_score, upper_score)),
        audited_coefficient_count=int(np.sum(mask)),
        lower_zero_support_failure_count=int(np.sum(lower_zero)),
        upper_zero_support_failure_count=int(np.sum(upper_zero)),
    )


def evaluate_horizon_future_outcome(
    actual_coefficient: np.ndarray,
    candidate_selected: np.ndarray,
    noop_selected: np.ndarray,
    candidate_sensing_power_w: np.ndarray,
    noop_sensing_power_w: np.ndarray,
    candidate_lower_pd: np.ndarray,
    noop_upper_pd: np.ndarray,
    coefficient_lower: np.ndarray,
    coefficient_upper: np.ndarray,
    coefficient_audit_mask: np.ndarray,
    *,
    false_alarm_probability: float,
    tolerance: float = 1.0e-12,
) -> HorizonFutureOutcome:
    """Evaluate one decision on a post-decision, frozen movement tape."""
    coefficient = np.asarray(actual_coefficient, dtype=np.float64)
    candidate_power = np.asarray(
        candidate_sensing_power_w, dtype=np.float64)
    noop_power = np.asarray(noop_sensing_power_w, dtype=np.float64)
    candidate_bound = np.asarray(candidate_lower_pd, dtype=np.float64)
    noop_bound = np.asarray(noop_upper_pd, dtype=np.float64)
    if coefficient.ndim != 4:
        raise ValueError("actual_coefficient must have shape (H,K,K,Q)")
    H, K, K2, Q = coefficient.shape
    if K != K2:
        raise ValueError("coefficient endpoint axes must be square")
    expected_power = (H, K, Q)
    expected_pd = (H, Q)
    if candidate_power.shape != expected_power or noop_power.shape != expected_power:
        raise ValueError("sensing-power rollouts must have shape (H,K,Q)")
    if candidate_bound.shape != expected_pd or noop_bound.shape != expected_pd:
        raise ValueError("P_D bounds must have shape (H,Q)")
    if np.asarray(candidate_selected).shape != (K, K, Q):
        raise ValueError("candidate_selected must have shape (K,K,Q)")
    if np.asarray(noop_selected).shape != (K, K, Q):
        raise ValueError("noop_selected must have shape (K,K,Q)")
    p_fa = float(false_alarm_probability)
    eps = float(tolerance)
    if not np.isfinite(p_fa) or not 0.0 < p_fa < 1.0:
        raise ValueError("false_alarm_probability must lie in (0,1)")
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    candidate_edges = [
        tuple(edge) for edge in np.argwhere(candidate_selected)
    ]
    noop_edges = [tuple(edge) for edge in np.argwhere(noop_selected)]
    candidate_pd = []
    noop_pd = []
    for step in range(H):
        candidate_gain, _ = fixed_owner_gain_matrix(
            coefficient[step], candidate_edges)
        noop_gain, _ = fixed_owner_gain_matrix(
            coefficient[step], noop_edges)
        candidate_pd.append(compute_detection_probabilities(
            np.sum(candidate_gain * candidate_power[step], axis=0), p_fa))
        noop_pd.append(compute_detection_probabilities(
            np.sum(noop_gain * noop_power[step], axis=0), p_fa))
    realized_candidate = np.asarray(candidate_pd, dtype=np.float64)
    realized_noop = np.asarray(noop_pd, dtype=np.float64)
    score = simultaneous_log_envelope_score(
        coefficient_lower,
        coefficient_upper,
        coefficient,
        coefficient_audit_mask,
    )
    return HorizonFutureOutcome(
        candidate_pd=realized_candidate,
        noop_pd=realized_noop,
        coefficient_score=score,
        candidate_lower_failure=bool(np.any(
            candidate_bound > realized_candidate + eps)),
        noop_upper_failure=bool(np.any(
            noop_bound + eps < realized_noop)),
        target_no_harm=(realized_candidate + eps >= realized_noop),
    )
