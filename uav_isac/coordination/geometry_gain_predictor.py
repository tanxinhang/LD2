"""Causal bistatic geometry correction for owner-local per-watt gains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix


@dataclass(frozen=True)
class GeometryGainPrediction:
    gain_per_watt: np.ndarray
    owners: np.ndarray
    edge_gain_per_watt: np.ndarray
    direct_edge_count: int
    fallback_edge_count: int
    calibration_available: bool
    invariant_median: float
    invariant_log_mad: float


@dataclass(frozen=True)
class GeometryCoefficientPrediction:
    coefficient_per_watt: np.ndarray
    direct_edge_count: int
    fallback_edge_count: int
    calibration_available: bool
    invariant_median: float
    invariant_log_mad: float


def _positions3(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3, 4):
        raise ValueError(f"{name} must have shape (n,2), (n,3), or (n,4)")
    # Target traces use [x,y,vx,vy]; only their position is relevant here.
    if array.shape[1] == 4:
        array = array[:, :2]
    if array.shape[1] == 2:
        array = np.concatenate(
            (array, np.zeros((array.shape[0], 1), dtype=np.float64)), axis=1)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _target_ranges(uav_positions: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    delta = uav_positions[:, None, :] - target_positions[None, :, :]
    return np.maximum(np.linalg.norm(delta, axis=-1), 1.0e-6)


def predict_edge_gain_tensor_from_geometry(
    previous_coefficient: np.ndarray,
    previous_uav_positions: np.ndarray,
    current_uav_positions: np.ndarray,
    previous_target_positions: np.ndarray,
    current_target_positions: np.ndarray,
    *,
    current_support: np.ndarray | None = None,
) -> GeometryCoefficientPrediction:
    """Predict every currently supported edge without future observations."""
    previous = np.asarray(previous_coefficient, dtype=np.float64)
    if previous.ndim != 3 or previous.shape[0] != previous.shape[1]:
        raise ValueError("previous_coefficient must have shape (K,K,Q)")
    if np.any(~np.isfinite(previous)) or np.any(previous < 0.0):
        raise ValueError("previous_coefficient must be finite and non-negative")
    K, _, Q = previous.shape
    previous_uav = _positions3(previous_uav_positions, "previous_uav_positions")
    current_uav = _positions3(current_uav_positions, "current_uav_positions")
    previous_target = _positions3(
        previous_target_positions, "previous_target_positions")
    current_target = _positions3(
        current_target_positions, "current_target_positions")
    if previous_uav.shape != (K, 3) or current_uav.shape != (K, 3):
        raise ValueError("UAV position count must match previous_coefficient")
    if previous_target.shape != (Q, 3) or current_target.shape != (Q, 3):
        raise ValueError("target position count must match previous_coefficient")
    if current_support is None:
        support = np.ones_like(previous, dtype=bool)
    else:
        support = np.asarray(current_support, dtype=bool)
        if support.shape != previous.shape:
            raise ValueError("current_support must match previous_coefficient")

    previous_range = _target_ranges(previous_uav, previous_target)
    current_range = _target_ranges(current_uav, current_target)
    previous_product_sq = (
        previous_range[:, None, :] ** 2
        * previous_range[None, :, :] ** 2
    )
    current_product_sq = (
        current_range[:, None, :] ** 2
        * current_range[None, :, :] ** 2
    )
    off_diagonal = ~np.eye(K, dtype=bool)[:, :, None]
    observed = (previous > 0.0) & off_diagonal
    invariant = np.where(observed, previous * previous_product_sq, np.nan)
    finite_invariant = invariant[np.isfinite(invariant) & (invariant > 0.0)]
    calibration_available = bool(finite_invariant.size)
    global_median = (
        float(np.median(finite_invariant)) if calibration_available else 0.0)
    log_mad = (
        float(np.median(np.abs(
            np.log(finite_invariant) - np.median(np.log(finite_invariant))
        )))
        if calibration_available else float("inf")
    )
    target_median = np.zeros(Q, dtype=np.float64)
    for target in range(Q):
        values = invariant[:, :, target]
        values = values[np.isfinite(values) & (values > 0.0)]
        target_median[target] = (
            float(np.median(values)) if values.size else global_median)

    fallback_invariant = np.broadcast_to(
        target_median[None, None, :], previous.shape)
    used_invariant = np.where(observed, invariant, fallback_invariant)
    active = support & off_diagonal & (used_invariant > 0.0)
    predicted = np.where(
        active, used_invariant / current_product_sq, 0.0)
    return GeometryCoefficientPrediction(
        coefficient_per_watt=predicted,
        direct_edge_count=int(np.sum(active & observed)),
        fallback_edge_count=int(np.sum(active & ~observed)),
        calibration_available=calibration_available,
        invariant_median=global_median,
        invariant_log_mad=log_mad,
    )


def predict_fixed_owner_gain_from_geometry(
    previous_coefficient: np.ndarray,
    previous_uav_positions: np.ndarray,
    current_uav_positions: np.ndarray,
    previous_target_positions: np.ndarray,
    current_target_positions: np.ndarray,
    selected: Sequence[tuple[int, int, int]],
    *,
    current_support: np.ndarray | None = None,
) -> GeometryGainPrediction:
    """Transport delayed gains with the bistatic radar range law.

    For an edge that was previously observed, ``a R_tx^2 R_rx^2`` is reused
    directly.  A newly selected edge receives the target-wise median invariant
    from all previously observed ordered pairs, falling back to the global
    median only when that target had no observation.  Missing calibration fails
    closed to zero.  The caller must still calibrate stochastic/model residuals.
    """
    previous = np.asarray(previous_coefficient, dtype=np.float64)
    full = predict_edge_gain_tensor_from_geometry(
        previous,
        previous_uav_positions,
        current_uav_positions,
        previous_target_positions,
        current_target_positions,
        current_support=current_support,
    )
    K, _, Q = previous.shape
    support = (
        np.ones_like(previous, dtype=bool)
        if current_support is None
        else np.asarray(current_support, dtype=bool)
    )
    predicted = np.zeros_like(previous)
    direct = 0
    fallback = 0
    for transmitter, receiver, target in selected:
        i, j, q = int(transmitter), int(receiver), int(target)
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q) or i == j:
            raise ValueError("selected edge is outside coefficient support")
        if not support[i, j, q]:
            continue
        if previous[i, j, q] > 0.0:
            direct += 1
        else:
            fallback += 1
        predicted[i, j, q] = full.coefficient_per_watt[i, j, q]

    gain, owners = fixed_owner_gain_matrix(predicted, selected)
    return GeometryGainPrediction(
        gain_per_watt=gain,
        owners=owners,
        edge_gain_per_watt=predicted,
        direct_edge_count=direct,
        fallback_edge_count=fallback,
        calibration_available=full.calibration_available,
        invariant_median=full.invariant_median,
        invariant_log_mad=full.invariant_log_mad,
    )
