"""Causal multi-step prediction for the canonical linear-Gaussian target model.

The CV state is already a continuous-state Markov process.  These routines
apply its Chapman--Kolmogorov recursion exactly (away from the reflecting
boundary), without introducing an unsupported discrete maneuver model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianMarkovHorizon:
    """Predicted states at steps 0..H for one or more CV beliefs."""

    mean: np.ndarray
    covariance: np.ndarray


def cv_markov_matrices(dt_s: float, acceleration_std_mps2: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact one-step CV transition and acceleration covariance."""

    dt = float(dt_s)
    sigma = float(acceleration_std_mps2)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("acceleration_std_mps2 must be finite and non-negative")
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    transition = np.asarray([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    covariance = sigma * sigma * np.asarray([
        [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
        [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
        [dt3 / 2.0, 0.0, dt2, 0.0],
        [0.0, dt3 / 2.0, 0.0, dt2],
    ])
    return transition, covariance


def predict_cv_markov_horizon(
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    *,
    steps: int,
    dt_s: float,
    acceleration_std_mps2: float,
    process_scale: float | np.ndarray = 1.0,
) -> GaussianMarkovHorizon:
    """Propagate batched CV Gaussian beliefs for ``steps`` transitions."""

    mean = np.asarray(initial_mean, dtype=np.float64)
    covariance = np.asarray(initial_covariance, dtype=np.float64)
    if mean.ndim < 1 or mean.shape[-1] != 4:
        raise ValueError("initial_mean must end in dimension 4")
    if covariance.shape != mean.shape[:-1] + (4, 4):
        raise ValueError("initial_covariance must match mean batch axes and end in (4,4)")
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(covariance)):
        raise ValueError("initial belief must be finite")
    horizon = int(steps)
    if horizon != steps or horizon < 0:
        raise ValueError("steps must be a non-negative integer")
    scale = np.asarray(process_scale, dtype=np.float64)
    try:
        scale = np.broadcast_to(scale, mean.shape[:-1])
    except ValueError as exc:
        raise ValueError("process_scale must broadcast over the belief batch") from exc
    if np.any(~np.isfinite(scale)) or np.any(scale < 0.0):
        raise ValueError("process_scale must be finite and non-negative")

    transition, process = cv_markov_matrices(dt_s, acceleration_std_mps2)
    means = np.empty((horizon + 1,) + mean.shape, dtype=np.float64)
    covariances = np.empty((horizon + 1,) + covariance.shape, dtype=np.float64)
    means[0] = mean
    covariances[0] = covariance
    scaled_process = scale[..., None, None] * process
    for step in range(1, horizon + 1):
        means[step] = np.einsum("ij,...j->...i", transition, means[step - 1])
        propagated = np.einsum(
            "ij,...jk,lk->...il", transition, covariances[step - 1], transition)
        covariances[step] = propagated + scaled_process
        covariances[step] = 0.5 * (
            covariances[step] + np.swapaxes(covariances[step], -1, -2))
    return GaussianMarkovHorizon(mean=means, covariance=covariances)


def predict_reflecting_cv_mean(
    position_xy_m: np.ndarray,
    velocity_xy_mps: np.ndarray,
    *,
    elapsed_s: float,
    area_size_m: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the CV mean under the simulator's axis-aligned wall reflection."""

    position = np.asarray(position_xy_m, dtype=np.float64)
    velocity = np.asarray(velocity_xy_mps, dtype=np.float64)
    if position.shape != velocity.shape or position.shape[-1] != 2:
        raise ValueError("position and velocity must have identical (...,2) shapes")
    elapsed = float(elapsed_s)
    area = np.asarray(area_size_m, dtype=np.float64)
    if (
        np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity))
        or not np.isfinite(elapsed) or elapsed < 0.0
        or area.shape != (2,) or np.any(~np.isfinite(area)) or np.any(area <= 0.0)
    ):
        raise ValueError("reflection inputs must be finite and physically valid")
    free = position + elapsed * velocity
    wrapped = np.mod(free, 2.0 * area)
    reflected_position = area - np.abs(wrapped - area)
    direction = np.where(wrapped < area, 1.0, -1.0)
    reflected_velocity = direction * velocity
    return reflected_position, reflected_velocity
