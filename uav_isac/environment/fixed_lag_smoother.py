"""Causal fixed-lag linear-Gaussian smoothing for research diagnostics.

The implementation is deliberately centralized and exact.  It is the
numerical reference that a later distributed time/UAV decomposition must
reproduce before communication or learned acceleration is considered.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FixedLagSmoothingResult:
    """Filtered and smoothed state sequence for one target/view."""

    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    smoothed_mean: np.ndarray
    smoothed_covariance: np.ndarray
    process_residual: np.ndarray
    innovation_nis: np.ndarray
    observed: np.ndarray


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _require_covariance(
    name: str,
    covariance: np.ndarray,
    dimension: int,
    *,
    positive_definite: bool,
) -> np.ndarray:
    value = np.asarray(covariance, dtype=np.float64)
    if value.shape != (dimension, dimension) or np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must be a finite ({dimension},{dimension}) matrix")
    value = _symmetric(value)
    minimum = float(np.min(np.linalg.eigvalsh(value)))
    tolerance = 1.0e-10 * max(1.0, float(np.linalg.norm(value, ord=2)))
    if positive_definite and minimum <= tolerance:
        raise ValueError(f"{name} must be positive definite")
    if not positive_definite and minimum < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return value


def fixed_lag_linear_gaussian_smoother(
    observations: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    transition: np.ndarray,
    process_covariance: np.ndarray,
    measurement_covariance: np.ndarray,
    *,
    measurement_matrix: Optional[np.ndarray] = None,
    observed: Optional[np.ndarray] = None,
) -> FixedLagSmoothingResult:
    """Run a Kalman forward pass and Rauch--Tung--Striebel backward pass.

    ``observations[t]`` describes state ``t``.  Missing whole observations are
    represented by ``observed[t] == False``; their numeric payload is ignored.
    A time-varying measurement covariance may have shape ``(T,m,m)``.

    This function is causal at the window level: callers may only supply data
    available at the current frame.  Consequently the last smoothed state is
    mathematically identical to the last filtered state.  The useful online
    output is the denoised history and its model residuals, not a fictitious
    future-informed endpoint.
    """
    measurements = np.asarray(observations, dtype=np.float64)
    if measurements.ndim != 2 or measurements.shape[0] < 1:
        raise ValueError("observations must have shape (T,m) with T >= 1")
    steps, measurement_dim = measurements.shape
    initial_mean = np.asarray(prior_mean, dtype=np.float64).reshape(-1)
    if initial_mean.size < 1 or np.any(~np.isfinite(initial_mean)):
        raise ValueError("prior_mean must be a finite non-empty vector")
    state_dim = initial_mean.size
    dynamics = np.asarray(transition, dtype=np.float64)
    if dynamics.shape != (state_dim, state_dim) or np.any(~np.isfinite(dynamics)):
        raise ValueError("transition has incompatible shape or non-finite values")
    observation_operator = (
        np.eye(state_dim, dtype=np.float64)
        if measurement_matrix is None
        else np.asarray(measurement_matrix, dtype=np.float64)
    )
    if (
        observation_operator.shape != (measurement_dim, state_dim)
        or np.any(~np.isfinite(observation_operator))
    ):
        raise ValueError("measurement_matrix has incompatible shape")

    prior_cov = _require_covariance(
        "prior_covariance", prior_covariance, state_dim, positive_definite=True)
    process_cov = _require_covariance(
        "process_covariance", process_covariance, state_dim,
        positive_definite=False)
    measurement_cov = np.asarray(measurement_covariance, dtype=np.float64)
    if measurement_cov.ndim == 2:
        measurement_cov = np.broadcast_to(
            measurement_cov, (steps, measurement_dim, measurement_dim)).copy()
    if measurement_cov.shape != (steps, measurement_dim, measurement_dim):
        raise ValueError("measurement_covariance must have shape (m,m) or (T,m,m)")
    for index in range(steps):
        measurement_cov[index] = _require_covariance(
            f"measurement_covariance[{index}]",
            measurement_cov[index],
            measurement_dim,
            positive_definite=True,
        )

    if observed is None:
        observation_mask = np.all(np.isfinite(measurements), axis=1)
    else:
        observation_mask = np.asarray(observed, dtype=bool)
        if observation_mask.shape != (steps,):
            raise ValueError("observed must have shape (T,)")
    if np.any(~np.isfinite(measurements[observation_mask])):
        raise ValueError("observed measurements must be finite")

    predicted_mean = np.empty((steps, state_dim), dtype=np.float64)
    predicted_covariance = np.empty(
        (steps, state_dim, state_dim), dtype=np.float64)
    filtered_mean = np.empty_like(predicted_mean)
    filtered_covariance = np.empty_like(predicted_covariance)
    innovation_nis = np.full(steps, np.nan, dtype=np.float64)
    identity = np.eye(state_dim, dtype=np.float64)

    for index in range(steps):
        if index == 0:
            mean_minus = initial_mean
            covariance_minus = prior_cov
        else:
            mean_minus = dynamics @ filtered_mean[index - 1]
            covariance_minus = _symmetric(
                dynamics @ filtered_covariance[index - 1] @ dynamics.T
                + process_cov)
        predicted_mean[index] = mean_minus
        predicted_covariance[index] = covariance_minus
        if not observation_mask[index]:
            filtered_mean[index] = mean_minus
            filtered_covariance[index] = covariance_minus
            continue
        innovation = measurements[index] - observation_operator @ mean_minus
        innovation_covariance = _symmetric(
            observation_operator @ covariance_minus @ observation_operator.T
            + measurement_cov[index])
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            observation_operator @ covariance_minus,
        ).T
        filtered_mean[index] = mean_minus + kalman_gain @ innovation
        identity_minus_kh = identity - kalman_gain @ observation_operator
        filtered_covariance[index] = _symmetric(
            identity_minus_kh @ covariance_minus @ identity_minus_kh.T
            + kalman_gain @ measurement_cov[index] @ kalman_gain.T)
        innovation_nis[index] = float(
            innovation @ np.linalg.solve(innovation_covariance, innovation))

    smoothed_mean = filtered_mean.copy()
    smoothed_covariance = filtered_covariance.copy()
    for index in range(steps - 2, -1, -1):
        smoothing_gain = np.linalg.solve(
            predicted_covariance[index + 1],
            dynamics @ filtered_covariance[index],
        ).T
        smoothed_mean[index] += smoothing_gain @ (
            smoothed_mean[index + 1] - predicted_mean[index + 1])
        smoothed_covariance[index] = _symmetric(
            filtered_covariance[index]
            + smoothing_gain
            @ (smoothed_covariance[index + 1]
               - predicted_covariance[index + 1])
            @ smoothing_gain.T)

    process_residual = (
        smoothed_mean[1:]
        - np.einsum("ij,tj->ti", dynamics, smoothed_mean[:-1])
    )
    return FixedLagSmoothingResult(
        filtered_mean=filtered_mean,
        filtered_covariance=filtered_covariance,
        smoothed_mean=smoothed_mean,
        smoothed_covariance=smoothed_covariance,
        process_residual=process_residual,
        innovation_nis=innovation_nis,
        observed=observation_mask.copy(),
    )


def estimate_cv_acceleration_from_residuals(
    process_residual: np.ndarray,
    dt_s: float,
    *,
    lookback: int = 3,
    recency: float = 0.7,
) -> np.ndarray:
    """Estimate recent piecewise-constant acceleration from CV residuals.

    The estimate solves the two equations ``r_p=.5*dt^2*a`` and
    ``r_v=dt*a`` in least squares, then exponentially weights the last
    ``lookback`` transitions.  It is a diagnostic/predictor feature; white
    acceleration has zero predictable mean and should fail a forecast gate.
    """
    residual = np.asarray(process_residual, dtype=np.float64)
    dt = float(dt_s)
    if residual.ndim != 2 or residual.shape[1] != 4 or residual.shape[0] < 1:
        raise ValueError("process_residual must have shape (T-1,4) with T >= 2")
    if np.any(~np.isfinite(residual)) or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("residuals and dt_s must be finite, with dt_s > 0")
    if isinstance(lookback, bool) or int(lookback) != lookback or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    if not np.isfinite(recency) or not 0.0 < recency <= 1.0:
        raise ValueError("recency must lie in (0,1]")
    count = min(int(lookback), residual.shape[0])
    selected = residual[-count:]
    denominator = 0.25 * dt ** 4 + dt ** 2
    acceleration = (
        0.5 * dt * dt * selected[:, :2] + dt * selected[:, 2:4]
    ) / denominator
    weights = recency ** np.arange(count - 1, -1, -1, dtype=np.float64)
    weights /= float(np.sum(weights))
    return weights @ acceleration
