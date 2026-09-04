"""Composable target-level certificates for assembled distributed power.

The certificate is deliberately about the *executed joint allocation*, not
about adding incomparable local LP objectives.  Transmitter ``k`` reports a
component-wise lower bound ``c[k,q] <= a_true[k,q] * p[k,q]``.  A target owner
can therefore add reports from different private information sets without
assuming that the local optimizers shared a common view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComposableTargetCertificate:
    """Certificate reconstructed by deterministic target responsibility."""

    row_contribution_lower: np.ndarray
    target_deflection_lower: np.ndarray
    target_complete: np.ndarray
    source_frame: np.ndarray
    global_optimum_upper: float = float("inf")
    joint_approximation_ratio_lower: float = 0.0

    @property
    def complete_fraction(self) -> float:
        return float(np.mean(self.target_complete))

    @property
    def worst_complete_lower(self) -> float:
        if not np.any(self.target_complete):
            return 0.0
        return float(np.min(
            self.target_deflection_lower[self.target_complete]))


@dataclass(frozen=True)
class RobustPrimalDualBounds:
    """Two-sided max-min value certificate for one stated information view."""

    primal_witness_lower: float
    dual_upper: float

    @property
    def uncertainty_width(self) -> float:
        return float(max(self.dual_upper - self.primal_witness_lower, 0.0))


def robust_primal_dual_bounds(
    gain_lower_per_watt: np.ndarray,
    gain_upper_per_watt: np.ndarray,
    feasible_power_w: np.ndarray,
    sensing_budget_w: np.ndarray,
    target_prices: np.ndarray,
) -> RobustPrimalDualBounds:
    """Certify ``L(P,A-) <= t*(A_true) <= U(lambda,A+)``.

    The function is deliberately algebraic: callers remain responsible for
    constructing physical envelopes ``A- <= A_true <= A+`` from information
    legally available to one node.  No common-view assumption is introduced.
    ``P`` must be row-feasible and ``lambda`` must lie in the target simplex.
    """
    lower = np.asarray(gain_lower_per_watt, dtype=np.float64)
    upper = np.asarray(gain_upper_per_watt, dtype=np.float64)
    power = np.asarray(feasible_power_w, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    prices = np.asarray(target_prices, dtype=np.float64).reshape(-1)
    if (
        lower.ndim != 2 or upper.shape != lower.shape
        or power.shape != lower.shape
        or budget.shape != (lower.shape[0],)
        or prices.shape != (lower.shape[1],)
    ):
        raise ValueError("gain, power, budget and price dimensions disagree")
    values = (lower, upper, power, budget, prices)
    if any(np.any(~np.isfinite(value)) for value in values):
        raise ValueError("certificate inputs must be finite")
    if (
        np.any(lower < 0.0) or np.any(upper < lower)
        or np.any(power < 0.0) or np.any(budget < 0.0)
        or np.any(prices < 0.0)
    ):
        raise ValueError("certificate inputs violate non-negativity/order")
    tolerance = 1.0e-9 * max(1.0, float(np.max(budget, initial=0.0)))
    if np.any(np.sum(power, axis=1) > budget + tolerance):
        raise ValueError("power witness is not row-feasible")
    if abs(float(np.sum(prices)) - 1.0) > 1.0e-9:
        raise ValueError("target prices must lie in the simplex")
    primal_lower = float(np.min(np.sum(lower * power, axis=0)))
    dual_upper = float(np.sum(
        budget * np.max(prices[None, :] * upper, axis=1)))
    # Weak duality plus A- <= A+ implies this ordering.  Close only roundoff;
    # a material violation exposes an invalid physical envelope or witness.
    if primal_lower > dual_upper + 1.0e-9 * max(1.0, abs(dual_upper)):
        raise ValueError("primal/dual movement certificate is inconsistent")
    return RobustPrimalDualBounds(primal_lower, dual_upper)


def robust_movement_dominates(
    candidate: RobustPrimalDualBounds,
    incumbent: RobustPrimalDualBounds,
    *,
    margin: float = 0.0,
) -> bool:
    """Strong robust dominance: candidate worst case beats incumbent best case."""
    threshold = float(margin)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("dominance margin must be finite and non-negative")
    return bool(
        candidate.primal_witness_lower
        > incumbent.dual_upper + threshold)


def conservative_row_contribution(
    executed_power_w: np.ndarray,
    executor_gain_lower_per_watt: np.ndarray,
) -> np.ndarray:
    """Return row-local target contributions under a stated gain lower bound.

    The caller owns the physical assumption ``gain_lower <= gain_true``.  This
    function enforces only the algebraic conditions needed for composition.
    """
    power = np.asarray(executed_power_w, dtype=np.float64)
    gain = np.asarray(executor_gain_lower_per_watt, dtype=np.float64)
    if power.ndim != 2 or gain.shape != power.shape:
        raise ValueError("power and executor gain lower bound must be (K,Q)")
    if (
        np.any(~np.isfinite(power)) or np.any(power < 0.0)
        or np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
    ):
        raise ValueError("power and gain lower bound must be finite/non-negative")
    return power * gain


def quantize_lower_log(
    value: np.ndarray,
    *,
    bits: int,
    scale: float,
    maximum: float,
) -> np.ndarray:
    """Log-quantize non-negative values with a one-sided (downward) error.

    Values beyond ``maximum`` saturate at ``maximum`` and remain lower bounds;
    sub-grid positive values decode to zero.  This is intentional fail-closed
    behaviour, unlike round-to-nearest quantization which can overstate a
    physical contribution.
    """
    values = np.asarray(value, dtype=np.float64)
    levels = (1 << int(bits)) - 1
    scale_value = float(scale)
    maximum_value = float(maximum)
    if int(bits) < 1:
        raise ValueError("bits must be positive")
    if (
        not np.isfinite(scale_value) or scale_value <= 0.0
        or not np.isfinite(maximum_value) or maximum_value <= 0.0
    ):
        raise ValueError("scale and maximum must be finite and positive")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("values must be finite and non-negative")
    log_span = float(np.log1p(maximum_value / scale_value))
    normalized = np.log1p(values / scale_value) / log_span
    code = np.floor(np.clip(normalized, 0.0, 1.0) * levels)
    decoded = scale_value * np.expm1(code / levels * log_span)
    return np.minimum(decoded, values)


def quantize_upper_log(
    value: np.ndarray,
    *,
    bits: int,
    scale: float,
    maximum: float,
) -> np.ndarray:
    """Log-quantize finite upper bounds by rounding away from zero.

    Overflow is rejected rather than saturated: saturating an upper bound would
    silently invalidate the certificate.
    """
    values = np.asarray(value, dtype=np.float64)
    levels = (1 << int(bits)) - 1
    scale_value = float(scale)
    maximum_value = float(maximum)
    if int(bits) < 1:
        raise ValueError("bits must be positive")
    if (
        not np.isfinite(scale_value) or scale_value <= 0.0
        or not np.isfinite(maximum_value) or maximum_value <= 0.0
    ):
        raise ValueError("scale and maximum must be finite and positive")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("values must be finite and non-negative")
    if np.any(values > maximum_value):
        raise OverflowError(
            "upper certificate exceeds configured quantizer maximum")
    log_span = float(np.log1p(maximum_value / scale_value))
    normalized = np.log1p(values / scale_value) / log_span
    code = np.ceil(np.clip(normalized, 0.0, 1.0) * levels)
    decoded = scale_value * np.expm1(code / levels * log_span)
    return np.maximum(decoded, values)


def uniform_price_row_dual_upper(
    row_gain_upper_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> np.ndarray:
    """Return composable row terms for a valid uniform-price LP dual bound.

    For ``pi_q=1/Q``, weak duality gives
    ``t_star <= sum_k b_k max_q pi_q a_upper[k,q]``.
    """
    gain = np.asarray(row_gain_upper_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 2 or gain.shape[0] != budget.size or gain.shape[1] < 1:
        raise ValueError("gain upper must be (K,Q) matching the budget")
    if (
        np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError("gain upper and budget must be finite/non-negative")
    return budget * np.max(gain, axis=1) / float(gain.shape[1])


def aggregate_target_responsibility_certificate(
    row_contribution_lower: np.ndarray,
    report_frame: np.ndarray,
    *,
    target_owner: np.ndarray,
    max_age_frames: int,
    current_frame: int,
    row_dual_upper: np.ndarray | None = None,
) -> ComposableTargetCertificate:
    """Compose row reports at deterministic target owners.

    ``row_contribution_lower[owner,k,q]`` is the report from transmitter ``k``
    stored at responsibility node ``owner``.  A target is complete only when
    its owner has all ``K`` row reports from one identical source frame and the
    report age is within the declared AoI limit.  Incomplete targets receive a
    zero lower bound, so missing communication can never look like success.
    """
    contribution = np.asarray(row_contribution_lower, dtype=np.float64)
    frames = np.asarray(report_frame, dtype=np.int64)
    owners = np.asarray(target_owner, dtype=np.int64).reshape(-1)
    if (
        contribution.ndim != 3
        or contribution.shape[0] != contribution.shape[1]
        or contribution.shape[2] != owners.size
    ):
        raise ValueError("row reports must have shape (K,K,Q)")
    K, _, Q = contribution.shape
    if frames.shape != (K, K):
        raise ValueError("report_frame must have shape (K,K)")
    if np.any(~np.isfinite(contribution)) or np.any(contribution < 0.0):
        raise ValueError("row reports must be finite and non-negative")
    if np.any(owners < 0) or np.any(owners >= K):
        raise ValueError("target owners must be valid node indices")
    if int(max_age_frames) < 0:
        raise ValueError("max_age_frames must be non-negative")

    target_lower = np.zeros(Q, dtype=np.float64)
    target_complete = np.zeros(Q, dtype=bool)
    source_frame = np.full(Q, -1, dtype=np.int64)
    selected_rows = np.zeros((K, Q), dtype=np.float64)
    upper_reports = None
    if row_dual_upper is not None:
        upper_reports = np.asarray(row_dual_upper, dtype=np.float64)
        if (
            upper_reports.shape != (K, K)
            or np.any(~np.isfinite(upper_reports))
            or np.any(upper_reports < 0.0)
        ):
            raise ValueError("row_dual_upper must be finite non-negative (K,K)")
    owner_upper_candidates: list[float] = []
    for q in range(Q):
        owner = int(owners[q])
        owner_frames = frames[owner]
        frame = int(owner_frames[0]) if K else -1
        same_frame = bool(frame >= 0 and np.all(owner_frames == frame))
        fresh = bool(
            same_frame
            and 0 <= int(current_frame) - frame <= int(max_age_frames)
        )
        if not fresh:
            continue
        selected_rows[:, q] = contribution[owner, :, q]
        target_lower[q] = float(np.sum(selected_rows[:, q]))
        target_complete[q] = True
        source_frame[q] = frame
        if upper_reports is not None:
            owner_upper_candidates.append(float(np.sum(
                upper_reports[owner])))
    global_upper = (
        float(np.min(owner_upper_candidates))
        if len(owner_upper_candidates) == Q and Q > 0
        else float("inf")
    )
    joint_lower = (
        float(np.min(target_lower)) if np.all(target_complete) else 0.0)
    ratio = (
        float(np.clip(joint_lower / global_upper, 0.0, 1.0))
        if np.isfinite(global_upper) and global_upper > 0.0 else 0.0
    )
    return ComposableTargetCertificate(
        row_contribution_lower=selected_rows,
        target_deflection_lower=target_lower,
        target_complete=target_complete,
        source_frame=source_frame,
        global_optimum_upper=global_upper,
        joint_approximation_ratio_lower=ratio,
    )
