"""Offline calibration and falsification utilities for soft evidence.

Calibration, validation and runtime are separate roles.  This module estimates
Gaussian evidence statistics from labelled *calibration* traces and evaluates
fixed detectors on held-out traces.  It never selects a subset from validation
P_D and does not expose simulation truth to runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from uav_isac.physical.correlated_soft_evidence import (
    optimal_linear_soft_fusion,
)


@dataclass(frozen=True)
class GaussianEvidenceCalibration:
    mean_h0: np.ndarray
    mean_h1: np.ndarray
    covariance_h0: np.ndarray
    covariance_h1: np.ndarray
    pooled_covariance: np.ndarray
    equal_covariance_relative_error: float
    samples_h0: int
    samples_h1: int

    @property
    def mean_shift(self) -> np.ndarray:
        return np.asarray(self.mean_h1) - np.asarray(self.mean_h0)


@dataclass(frozen=True)
class DetectionCovarianceChoice:
    covariance: np.ndarray
    policy: str
    shrinkage_to_diagonal: float
    condition_number_before: float
    condition_number_after: float


def _validated_trace(trace: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError(f"{name} must have shape (samples>=2, sources>=2)")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def calibrate_gaussian_evidence(
    h0_trace: np.ndarray,
    h1_trace: np.ndarray,
) -> GaussianEvidenceCalibration:
    """Estimate means and unbiased H0/H1/pooled covariance matrices."""
    h0 = _validated_trace(h0_trace, "h0_trace")
    h1 = _validated_trace(h1_trace, "h1_trace")
    if h0.shape[1] != h1.shape[1]:
        raise ValueError("H0 and H1 traces must have equal source count")
    mean_h0 = np.mean(h0, axis=0)
    mean_h1 = np.mean(h1, axis=0)
    covariance_h0 = np.atleast_2d(np.cov(h0, rowvar=False, ddof=1))
    covariance_h1 = np.atleast_2d(np.cov(h1, rowvar=False, ddof=1))
    pooled = (
        (h0.shape[0] - 1) * covariance_h0
        + (h1.shape[0] - 1) * covariance_h1
    ) / float(h0.shape[0] + h1.shape[0] - 2)
    relative_error = float(
        np.linalg.norm(covariance_h1 - covariance_h0, ord="fro")
        / max(np.linalg.norm(covariance_h0, ord="fro"), 1.0e-15)
    )
    return GaussianEvidenceCalibration(
        mean_h0=mean_h0,
        mean_h1=mean_h1,
        covariance_h0=covariance_h0,
        covariance_h1=covariance_h1,
        pooled_covariance=pooled,
        equal_covariance_relative_error=relative_error,
        samples_h0=int(h0.shape[0]),
        samples_h1=int(h1.shape[0]),
    )


def _diagonal_shrinkage_for_condition(
    covariance: np.ndarray,
    maximum_condition_number: float,
) -> tuple[np.ndarray, float, float, float]:
    sigma = np.asarray(covariance, dtype=np.float64)
    diagonal = np.diag(np.diag(sigma))
    before = float(np.linalg.cond(sigma))
    limit = float(maximum_condition_number)
    if not np.isfinite(limit) or limit <= 1.0:
        raise ValueError("maximum_condition_number must exceed one")
    if np.isfinite(before) and before <= limit:
        return sigma.copy(), 0.0, before, before
    if np.any(np.diag(sigma) <= 0.0):
        raise ValueError("covariance diagonal must be positive")
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        candidate = (1.0 - middle) * sigma + middle * diagonal
        if float(np.linalg.cond(candidate)) <= limit:
            high = middle
        else:
            low = middle
    regularized = (1.0 - high) * sigma + high * diagonal
    after = float(np.linalg.cond(regularized))
    return regularized, float(high), before, after


def choose_detection_covariance(
    calibration: GaussianEvidenceCalibration,
    *,
    equal_covariance_tolerance: float,
    maximum_condition_number: float = 1.0e6,
) -> DetectionCovarianceChoice:
    """Use pooled covariance only after passing the equal-covariance gate.

    When the relative H0/H1 covariance mismatch exceeds the declared
    tolerance, H0 covariance is selected as a fixed-P_FA linear fallback.  It
    is not claimed to be QDA-optimal and must still pass held-out ROC ranking.
    """
    tolerance = float(equal_covariance_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("equal_covariance_tolerance must be non-negative")
    if calibration.equal_covariance_relative_error <= tolerance:
        covariance = calibration.pooled_covariance
        policy = "equal_covariance_pooled"
    else:
        covariance = calibration.covariance_h0
        policy = "h0_fixed_pfa_fallback"
    regularized, shrinkage, before, after = _diagonal_shrinkage_for_condition(
        covariance, float(maximum_condition_number))
    try:
        np.linalg.cholesky(regularized)
    except np.linalg.LinAlgError as exc:
        raise ValueError("selected covariance is not positive definite") from exc
    return DetectionCovarianceChoice(
        covariance=regularized,
        policy=policy,
        shrinkage_to_diagonal=shrinkage,
        condition_number_before=before,
        condition_number_after=after,
    )


def covariance_relative_error(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float:
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("covariances must be equal-size matrices")
    return float(
        np.linalg.norm(second - first, ord="fro")
        / max(np.linalg.norm(first, ord="fro"), 1.0e-15)
    )


def fixed_linear_detector_validation(
    calibration_h0: np.ndarray,
    validation_h0: np.ndarray,
    validation_h1: np.ndarray,
    weights: np.ndarray,
    *,
    p_fa: float,
) -> dict[str, float]:
    """Freeze an empirical H0 threshold, then evaluate held-out P_FA/P_D."""
    cal0 = _validated_trace(calibration_h0, "calibration_h0")
    val0 = _validated_trace(validation_h0, "validation_h0")
    val1 = _validated_trace(validation_h1, "validation_h1")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if cal0.shape[1] != w.size or val0.shape[1:] != (w.size,) or val1.shape[1:] != (w.size,):
        raise ValueError("trace source counts must match weights")
    probability = float(p_fa)
    if not 0.0 < probability < 1.0:
        raise ValueError("p_fa must lie in (0,1)")
    calibration_scores = cal0 @ w
    # 'higher' never makes the finite calibration exceed the requested tail
    # probability merely due to interpolation between order statistics.
    threshold = float(np.quantile(
        calibration_scores, 1.0 - probability, method="higher"))
    return {
        "threshold": threshold,
        "validation_pfa": float(np.mean((val0 @ w) > threshold)),
        "validation_pd": float(np.mean((val1 @ w) > threshold)),
    }


def subset_surrogate_rank_validation(
    calibration: GaussianEvidenceCalibration,
    covariance_choice: DetectionCovarianceChoice,
    calibration_h0: np.ndarray,
    validation_h0: np.ndarray,
    validation_h1: np.ndarray,
    *,
    p_fa: float,
) -> dict[str, object]:
    """Compare calibrated D ordering with held-out subset P_D ordering.

    Held-out P_D is used only for this offline falsification statistic and
    never to select the runtime subset.
    """
    source_count = int(np.asarray(calibration.mean_h0).size)
    subsets: list[tuple[int, ...]] = []
    surrogate: list[float] = []
    validation_pd: list[float] = []
    for count in range(1, source_count + 1):
        for subset in combinations(range(source_count), count):
            value, weights = optimal_linear_soft_fusion(
                calibration.mean_shift,
                covariance_choice.covariance,
                subset,
            )
            result = fixed_linear_detector_validation(
                calibration_h0,
                validation_h0,
                validation_h1,
                weights,
                p_fa=float(p_fa),
            )
            subsets.append(subset)
            surrogate.append(float(value))
            validation_pd.append(float(result["validation_pd"]))
    correlation = spearmanr(surrogate, validation_pd)
    return {
        "subset_count": len(subsets),
        "spearman_r": float(correlation.statistic),
        "spearman_pvalue": float(correlation.pvalue),
        "subsets": [list(subset) for subset in subsets],
        "surrogate_deflection": surrogate,
        "validation_pd": validation_pd,
    }
