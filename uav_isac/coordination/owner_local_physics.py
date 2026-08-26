"""Owner-local physical sufficient statistics for bistatic repair proposals.

The functions in this module deliberately consume local beliefs and actually
delivered Token masks.  Privileged path gains and DD labels are accepted only
by audit code as outcomes, never as proposal inputs.
"""

# ----------------------------------------------------------------------
# AUDIT/RESEARCH-ONLY MODULE (2026-08-16 audit remediation)
#
# This module is consumed only by tools/ audit scripts and tests. It is
# NOT part of the deployment execution path (env_core / trainer) and its
# results must not be described as deployed behaviour. It exists to keep
# a specific research question reproducible; see
# docs/ARCHITECTURE_V2_RESULTS.md for the associated gate.
# ----------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_isac.environment.observation_slices import ObservationSlices
from uav_isac.physical.geometry import C_LIGHT


@dataclass(frozen=True)
class OwnerLocalKinematicState:
    """Endpoint self-state and receiver-local target beliefs in SI units."""

    uav_position_m: np.ndarray
    uav_velocity_mps: np.ndarray
    target_mean_by_owner: np.ndarray
    target_cov_diag_by_owner: np.ndarray
    target_aoi_frames_by_owner: np.ndarray


@dataclass(frozen=True)
class OwnerLocalCoefficientPrediction:
    """Per-watt edge coefficients reconstructed from lagged feedback."""

    coefficient_per_watt: np.ndarray
    target_invariant: np.ndarray
    direct_edge_count: int
    target_fallback_edge_count: int
    cached_target_fallback_edge_count: int
    unavailable_edge_count: int
    target_invariant_age_frames: np.ndarray
    target_invariant_version: np.ndarray


@dataclass(frozen=True)
class OwnerTargetInvariantCache:
    """Causal per-target bistatic invariant with explicit freshness.

    ``target_invariant[q]`` stores the robust target statistic
    ``median(coefficient * R_tx^2 * R_rx^2)``.  A zero value is unavailable
    and therefore fails closed.  Age and version are target-wise because one
    target can be refreshed while another remains unexcited.
    """

    target_invariant: np.ndarray
    age_frames: np.ndarray
    version: np.ndarray


@dataclass(frozen=True)
class OwnerLocalDDEffectivenessBounds:
    """Point and covariance-robust OTFS DD effectiveness by owner."""

    point: np.ndarray
    lower: np.ndarray
    delay_bin_radius: np.ndarray
    doppler_bin_radius: np.ndarray


@dataclass(frozen=True)
class OwnerLocalHorizonCoefficientBounds:
    """Deterministic geometry map around a calibrated target invariant."""

    lower: np.ndarray
    upper: np.ndarray
    dd_lower: np.ndarray
    dd_certified_support: np.ndarray
    target_position_radius_m: np.ndarray


def advance_owner_local_kinematics(
    state: OwnerLocalKinematicState,
    delta_position_m: np.ndarray,
    *,
    dt_s: float,
    max_speed_mps: float,
    area_size_m: tuple[float, float],
    advance_targets: bool,
) -> OwnerLocalKinematicState:
    """Project an action-time local state to the physical evaluation instant.

    The UAV update exactly mirrors ``UAV.apply_action``: displacement is
    speed-clipped, position is softly reflected at the rectangular boundary,
    and velocity is the applied displacement divided by ``dt``.  Optional
    target propagation uses the owner-local constant-velocity mean.  No true
    future state is consumed.
    """
    position = np.asarray(state.uav_position_m, dtype=np.float64)
    velocity = np.asarray(state.uav_velocity_mps, dtype=np.float64)
    delta = np.asarray(delta_position_m, dtype=np.float64)
    target = np.asarray(state.target_mean_by_owner, dtype=np.float64)
    covariance = np.asarray(
        state.target_cov_diag_by_owner, dtype=np.float64)
    aoi = np.asarray(state.target_aoi_frames_by_owner, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("uav_position_m must have shape (K,3)")
    K = position.shape[0]
    if velocity.shape != (K, 3) or delta.shape != (K, 2):
        raise ValueError("velocity/delta shapes must match the UAV count")
    if (
        target.ndim != 3 or target.shape[0] != K or target.shape[2] != 4
        or covariance.shape != target.shape
        or aoi.shape != target.shape[:2]
    ):
        raise ValueError("target belief shapes are inconsistent")
    dt = float(dt_s)
    speed = float(max_speed_mps)
    width, height = float(area_size_m[0]), float(area_size_m[1])
    values = (position, velocity, delta, target, covariance, aoi)
    if (
        any(np.any(~np.isfinite(value)) for value in values)
        or not np.isfinite(dt) or dt <= 0.0
        or not np.isfinite(speed) or speed <= 0.0
        or not np.isfinite(width) or width <= 0.0
        or not np.isfinite(height) or height <= 0.0
    ):
        raise ValueError("action-alignment inputs must be finite and valid")

    applied = delta.copy()
    norm = np.linalg.norm(applied, axis=1)
    max_displacement = speed * dt
    scale = np.minimum(1.0, max_displacement / np.maximum(norm, 1.0e-15))
    applied *= scale[:, None]
    projected_position = position.copy()
    projected_position[:, :2] += applied
    for axis, bound in ((0, width), (1, height)):
        below = projected_position[:, axis] < 0.0
        projected_position[below, axis] *= -1.0
        above = projected_position[:, axis] > bound
        projected_position[above, axis] = (
            2.0 * bound - projected_position[above, axis])
    projected_velocity = np.zeros_like(velocity)
    projected_velocity[:, :2] = applied / dt
    for axis, bound in ((0, width), (1, height)):
        crossed = (
            (position[:, axis] + applied[:, axis] < 0.0)
            | (position[:, axis] + applied[:, axis] > bound)
        )
        projected_velocity[crossed, axis] *= -1.0

    projected_target = target.copy()
    projected_aoi = aoi.copy()
    if bool(advance_targets):
        projected_target[:, :, :2] += target[:, :, 2:4] * dt
        projected_aoi += 1.0
    return OwnerLocalKinematicState(
        uav_position_m=projected_position,
        uav_velocity_mps=projected_velocity,
        target_mean_by_owner=projected_target,
        target_cov_diag_by_owner=covariance.copy(),
        target_aoi_frames_by_owner=projected_aoi,
    )


def decode_owner_local_kinematics(
    local_obs: np.ndarray,
    slices: ObservationSlices,
    *,
    area_size_m: tuple[float, float],
    height_m: float,
    velocity_scale_mps: float = 25.0,
    aoi_scale_frames: float = 100.0,
) -> OwnerLocalKinematicState:
    """Decode each endpoint's own state and own target beliefs.

    This reverses only the documented normalization in ``ObservationBuilder``.
    It does not fuse observations across UAVs.
    """
    obs = np.asarray(local_obs, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[0] != slices.K:
        raise ValueError("local_obs must have shape (K,D)")
    width, height = (float(area_size_m[0]), float(area_size_m[1]))
    altitude = float(height_m)
    speed = float(velocity_scale_mps)
    aoi_scale = float(aoi_scale_frames)
    scales = np.asarray([width, height, altitude], dtype=np.float64)
    if (
        np.any(~np.isfinite(scales)) or np.any(scales <= 0.0)
        or not np.isfinite(speed) or speed <= 0.0
        or not np.isfinite(aoi_scale) or aoi_scale <= 0.0
    ):
        raise ValueError("normalization scales must be finite and positive")

    self_state = np.asarray(slices.extract_self(obs), dtype=np.float64)
    beliefs = np.asarray(slices.extract_beliefs(obs), dtype=np.float64)
    position = self_state[:, :3] * scales[None, :]
    velocity = self_state[:, 3:6] * speed
    mean_scale = np.asarray([width, height, speed, speed], dtype=np.float64)
    covariance_scale = mean_scale ** 2
    target_mean = beliefs[:, :, :4] * mean_scale[None, None, :]
    target_covariance = (
        beliefs[:, :, 4:8] * covariance_scale[None, None, :]
    )
    target_aoi = beliefs[:, :, 8] * aoi_scale
    values = (position, velocity, target_mean, target_covariance, target_aoi)
    if any(np.any(~np.isfinite(value)) for value in values):
        raise ValueError("decoded local kinematics must be finite")
    if np.any(target_covariance < -1.0e-9) or np.any(target_aoi < -1.0e-9):
        raise ValueError("belief covariance and AoI must be non-negative")
    return OwnerLocalKinematicState(
        uav_position_m=position,
        uav_velocity_mps=velocity,
        target_mean_by_owner=target_mean,
        target_cov_diag_by_owner=np.maximum(target_covariance, 0.0),
        target_aoi_frames_by_owner=np.maximum(target_aoi, 0.0),
    )


def reciprocal_token_candidate_mask(token_visible: np.ndarray) -> np.ndarray:
    """Admit all targets on endpoint pairs with reciprocal delivered Tokens."""
    visible = np.asarray(token_visible, dtype=bool)
    if visible.ndim < 3 or visible.shape[-3] != visible.shape[-2]:
        raise ValueError("token_visible must end in (K,K,Q)")
    peer_link = np.any(visible, axis=-1)
    reciprocal = peer_link & np.swapaxes(peer_link, -1, -2)
    candidate = np.broadcast_to(
        reciprocal[..., :, :, None], visible.shape).copy()
    K = visible.shape[-3]
    off_diagonal = ~np.eye(K, dtype=bool)
    return candidate & off_diagonal.reshape(
        (1,) * (candidate.ndim - 3) + (K, K, 1))


def owner_local_dd_effectiveness(
    state: OwnerLocalKinematicState,
    *,
    carrier_hz: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
) -> np.ndarray:
    """Predict DD effectiveness using receiver-owner target beliefs.

    This is the exact broadcasted form of the scalar bistatic equations.  No
    edge is pruned and no physical approximation is introduced; only the
    Python ``(i,j,q)`` loop is removed from the causal common prefix.
    """
    position = np.asarray(state.uav_position_m, dtype=np.float64)
    velocity = np.asarray(state.uav_velocity_mps, dtype=np.float64)
    target = np.asarray(state.target_mean_by_owner, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("uav_position_m must have shape (K,3)")
    K = position.shape[0]
    if (
        velocity.shape != (K, 3) or target.ndim != 3
        or target.shape[0] != K or target.shape[2] != 4
    ):
        raise ValueError("owner-local kinematic shapes are inconsistent")
    fc = float(carrier_hz)
    delta_f = float(delta_f_hz)
    symbol_period = float(symbol_period_s)
    M, N = int(delay_bins), int(doppler_bins)
    if (
        np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity))
        or np.any(~np.isfinite(target)) or not np.isfinite(fc) or fc <= 0.0
        or not np.isfinite(delta_f) or delta_f <= 0.0
        or not np.isfinite(symbol_period) or symbol_period <= 0.0
        or M < 1 or N < 1
    ):
        raise ValueError("owner-local DD inputs must be finite and valid")
    Q = target.shape[1]
    target_position = np.concatenate((
        target[:, :, :2],
        np.zeros((K, Q, 1), dtype=np.float64),
    ), axis=2)
    target_velocity = np.concatenate((
        target[:, :, 2:4],
        np.zeros((K, Q, 1), dtype=np.float64),
    ), axis=2)
    tx_vector = (
        target_position[None, :, :, :] - position[:, None, None, :])
    rx_vector = position[:, None, :] - target_position
    tx_norm = np.linalg.norm(tx_vector, axis=3)
    rx_norm = np.linalg.norm(rx_vector, axis=2)
    delay = (tx_norm + rx_norm[None, :, :]) / C_LIGHT
    # Match compute_doppler's endpoint-unit-vector convention exactly.
    u_tx = tx_vector / (tx_norm[:, :, :, None] + 1.0e-10)
    u_rx = rx_vector / (rx_norm[:, :, None] + 1.0e-10)
    doppler = fc / C_LIGHT * (
        np.einsum("id,ijqd->ijq", velocity, u_tx)
        + np.einsum(
            "jqd,ijqd->ijq",
            target_velocity,
            u_rx[None, :, :, :] - u_tx,
        )
        + np.einsum("jd,jqd->jq", velocity, u_rx)[None, :, :]
    )
    delay_fraction = delay * M * delta_f
    doppler_fraction = doppler * N * symbol_period
    result = np.abs(
        np.sinc(delay_fraction - np.round(delay_fraction))
        * np.sinc(doppler_fraction - np.round(doppler_fraction)))
    return np.where(
        ~np.eye(K, dtype=bool)[:, :, None], result, 0.0)


def _maximum_fractional_bin_distance(
    center: float,
    radius: float,
) -> float:
    """Maximum distance to an integer over a closed scalar interval."""
    x = float(center)
    r = float(radius)
    if not np.isfinite(x) or not np.isfinite(r) or r < 0.0:
        raise ValueError("bin center/radius must be finite and non-negative")
    if r >= 0.5:
        return 0.5
    left = x - r
    right = x + r
    first_half_integer_index = int(np.ceil(left - 0.5))
    last_half_integer_index = int(np.floor(right - 0.5))
    if first_half_integer_index <= last_half_integer_index:
        return 0.5
    return float(min(0.5, max(
        abs(left - np.round(left)),
        abs(right - np.round(right)),
    )))


def _sinc_alignment_lower(center: float, radius: float) -> float:
    distance = _maximum_fractional_bin_distance(center, radius)
    return float(abs(np.sinc(distance)))


def _sinc_alignment_lower_array(
    center: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """Vectorized form of ``_sinc_alignment_lower`` with identical logic."""
    x = np.asarray(center, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    if x.shape != r.shape:
        raise ValueError("bin center and radius arrays must have equal shape")
    if (
        np.any(~np.isfinite(x)) or np.any(~np.isfinite(r))
        or np.any(r < 0.0)
    ):
        raise ValueError("bin center/radius must be finite and non-negative")
    left = x - r
    right = x + r
    crosses_half_integer = (
        np.ceil(left - 0.5) <= np.floor(right - 0.5))
    endpoint_distance = np.minimum(0.5, np.maximum(
        np.abs(left - np.round(left)),
        np.abs(right - np.round(right)),
    ))
    distance = np.where(
        (r >= 0.5) | crosses_half_integer,
        0.5,
        endpoint_distance,
    )
    return np.abs(np.sinc(distance))


def owner_local_dd_effectiveness_bounds(
    state: OwnerLocalKinematicState,
    *,
    carrier_hz: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    covariance_radius: float,
    uav_position_radius_m: float | np.ndarray = 0.0,
    uav_velocity_radius_mps: float | np.ndarray = 0.0,
) -> OwnerLocalDDEffectivenessBounds:
    """Lower-bound DD alignment over target belief ellipsoids.

    For receiver-owner target covariance ``Sigma_q``, the ellipsoid
    ``delta.T @ Sigma_q^-1 @ delta <= covariance_radius**2`` implies
    Euclidean position/velocity radii bounded by the square root of the
    largest corresponding diagonal variance.  Bistatic delay is
    2-Lipschitz in target position.  Doppler uses the standard unit-vector
    perturbation bound ``min(2, 2 r/(R-r))``.  The final sinc loss is minimized
    exactly over the resulting one-dimensional delay and Doppler bin
    intervals.  Optional UAV position/velocity radii add deterministic
    reachable sets around the nominal platforms.  Statistical coverage of
    the target ellipsoid itself must still be calibrated outside this map.
    """
    beta = float(covariance_radius)
    fc = float(carrier_hz)
    delta_f = float(delta_f_hz)
    symbol_period = float(symbol_period_s)
    M = int(delay_bins)
    N = int(doppler_bins)
    if (
        not np.isfinite(beta) or beta < 0.0
        or not np.isfinite(fc) or fc <= 0.0
        or not np.isfinite(delta_f) or delta_f <= 0.0
        or not np.isfinite(symbol_period) or symbol_period <= 0.0
        or M < 1 or N < 1
    ):
        raise ValueError("DD bound parameters must be finite and valid")
    position = np.asarray(state.uav_position_m, dtype=np.float64)
    velocity = np.asarray(state.uav_velocity_mps, dtype=np.float64)
    target = np.asarray(state.target_mean_by_owner, dtype=np.float64)
    covariance = np.asarray(
        state.target_cov_diag_by_owner, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("uav_position_m must have shape (K,3)")
    K = position.shape[0]
    if (
        velocity.shape != (K, 3) or target.ndim != 3
        or target.shape[0] != K or target.shape[2] != 4
        or covariance.shape != target.shape
    ):
        raise ValueError("owner-local DD state shapes are inconsistent")
    if np.any(~np.isfinite(covariance)) or np.any(covariance < 0.0):
        raise ValueError("target covariance must be finite non-negative")
    uav_position_radius = np.asarray(
        uav_position_radius_m, dtype=np.float64)
    uav_velocity_radius = np.asarray(
        uav_velocity_radius_mps, dtype=np.float64)
    if uav_position_radius.ndim == 0:
        uav_position_radius = np.full(K, float(uav_position_radius))
    else:
        uav_position_radius = uav_position_radius.reshape(-1)
    if uav_velocity_radius.ndim == 0:
        uav_velocity_radius = np.full(K, float(uav_velocity_radius))
    else:
        uav_velocity_radius = uav_velocity_radius.reshape(-1)
    if (
        uav_position_radius.shape != (K,)
        or uav_velocity_radius.shape != (K,)
        or np.any(~np.isfinite(uav_position_radius))
        or np.any(~np.isfinite(uav_velocity_radius))
        or np.any(uav_position_radius < 0.0)
        or np.any(uav_velocity_radius < 0.0)
    ):
        raise ValueError(
            "UAV reachable-set radii must be finite non-negative scalars "
            "or K-vectors")
    Q = target.shape[1]
    target_position = np.concatenate((
        target[:, :, :2],
        np.zeros((K, Q, 1), dtype=np.float64),
    ), axis=2)
    target_velocity = np.concatenate((
        target[:, :, 2:4],
        np.zeros((K, Q, 1), dtype=np.float64),
    ), axis=2)

    # Axes are transmitter i, receiver-owner j, target q, coordinate d.
    tx_vector = (
        target_position[None, :, :, :] - position[:, None, None, :])
    rx_vector = position[:, None, :] - target_position
    tx_norm = np.linalg.norm(tx_vector, axis=3)
    rx_norm = np.linalg.norm(rx_vector, axis=2)
    tx_range = np.maximum(tx_norm, 1.0e-9)
    rx_range = np.maximum(rx_norm, 1.0e-9)
    tau = (tx_range + rx_range[None, :, :]) / C_LIGHT

    # Preserve compute_doppler's 1e-10 denominator convention exactly.
    u_tx = tx_vector / (tx_norm[:, :, :, None] + 1.0e-10)
    u_rx = rx_vector / (rx_norm[:, :, None] + 1.0e-10)
    doppler_tx = np.einsum("id,ijqd->ijq", velocity, u_tx)
    doppler_target = np.einsum(
        "jqd,ijqd->ijq",
        target_velocity,
        u_rx[None, :, :, :] - u_tx,
    )
    doppler_rx = np.einsum("jd,jqd->jq", velocity, u_rx)[None, :, :]
    nu = fc / C_LIGHT * (doppler_tx + doppler_target + doppler_rx)

    delay_fraction = tau * M * delta_f
    doppler_fraction = nu * N * symbol_period
    point = np.abs(
        np.sinc(delay_fraction - np.round(delay_fraction))
        * np.sinc(doppler_fraction - np.round(doppler_fraction)))

    owner_position_radius = beta * np.sqrt(np.max(
        covariance[:, :, :2], axis=2))
    owner_velocity_radius = beta * np.sqrt(np.max(
        covariance[:, :, 2:4], axis=2))
    tx_position_radius = (
        owner_position_radius[None, :, :]
        + uav_position_radius[:, None, None])
    rx_position_radius = (
        owner_position_radius
        + uav_position_radius[:, None])
    delay_radius = (
        2.0 * owner_position_radius[None, :, :]
        + uav_position_radius[:, None, None]
        + uav_position_radius[None, :, None]
    ) / C_LIGHT * M * delta_f
    tx_unit_radius = np.minimum(
        2.0,
        2.0 * tx_position_radius
        / np.maximum(tx_range - tx_position_radius, 1.0e-9),
    )
    rx_unit_radius = np.minimum(
        2.0,
        2.0 * rx_position_radius
        / np.maximum(rx_range - rx_position_radius, 1.0e-9),
    )
    tx_relative_speed = np.linalg.norm(
        velocity[:, None, None, :] - target_velocity[None, :, :, :],
        axis=3,
    )
    rx_relative_speed = np.linalg.norm(
        target_velocity + velocity[:, None, :], axis=2)
    doppler_hz_radius = fc / C_LIGHT * (
        uav_velocity_radius[:, None, None]
        + uav_velocity_radius[None, :, None]
        + 2.0 * owner_velocity_radius[None, :, :]
        + (
            tx_relative_speed
            + uav_velocity_radius[:, None, None]
            + owner_velocity_radius[None, :, :]
        ) * tx_unit_radius
        + (
            rx_relative_speed[None, :, :]
            + uav_velocity_radius[None, :, None]
            + owner_velocity_radius[None, :, :]
        ) * rx_unit_radius[None, :, :]
    )
    doppler_radius = doppler_hz_radius * N * symbol_period
    lower = (
        _sinc_alignment_lower_array(delay_fraction, delay_radius)
        * _sinc_alignment_lower_array(doppler_fraction, doppler_radius))
    off_diagonal = ~np.eye(K, dtype=bool)[:, :, None]
    point = np.where(off_diagonal, point, 0.0)
    lower = np.where(off_diagonal, lower, 0.0)
    delay_radius = np.where(off_diagonal, delay_radius, 0.0)
    doppler_radius = np.where(off_diagonal, doppler_radius, 0.0)
    return OwnerLocalDDEffectivenessBounds(
        point=point,
        lower=np.minimum(lower, point),
        delay_bin_radius=delay_radius,
        doppler_bin_radius=doppler_radius,
    )


def propagate_owner_local_horizon(
    initial_state: OwnerLocalKinematicState,
    future_delta_position_m: np.ndarray,
    *,
    dt_s: float,
    max_speed_mps: float,
    area_size_m: tuple[float, float],
    advance_targets: bool,
    target_acceleration_std_mps2: float = 0.0,
) -> tuple[OwnerLocalKinematicState, ...]:
    """Propagate a causal owner-local belief through deterministic UAV moves.

    The initial state is included as horizon step zero.  Only diagonal target
    covariance is observable in the Token.  For each axis, Minkowski's
    inequality gives the conservative standard-deviation recursion

    ``s_p+ <= s_p + dt*s_v + 0.5*dt^2*s_a`` and
    ``s_v+ <= s_v + dt*s_a``.

    Squaring those radii yields a diagonal envelope that remains valid under
    unknown position--velocity correlation.  It is intentionally wider than
    a Kalman covariance update and consumes no future measurement.
    """
    plan = np.asarray(future_delta_position_m, dtype=np.float64)
    position = np.asarray(initial_state.uav_position_m, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("initial UAV positions must have shape (K,3)")
    K = position.shape[0]
    if plan.ndim != 3 or plan.shape[1:] != (K, 2):
        raise ValueError("future delta plan must have shape (H-1,K,2)")
    sigma_a = float(target_acceleration_std_mps2)
    dt = float(dt_s)
    if (
        np.any(~np.isfinite(plan)) or not np.isfinite(sigma_a)
        or sigma_a < 0.0 or not np.isfinite(dt) or dt <= 0.0
    ):
        raise ValueError("horizon propagation inputs must be finite and valid")
    states = [initial_state]
    for delta in plan:
        prior = states[-1]
        projected = advance_owner_local_kinematics(
            prior,
            delta,
            dt_s=dt,
            max_speed_mps=float(max_speed_mps),
            area_size_m=area_size_m,
            advance_targets=bool(advance_targets),
        )
        if bool(advance_targets):
            covariance = np.asarray(
                prior.target_cov_diag_by_owner, dtype=np.float64)
            standard = np.sqrt(np.maximum(covariance, 0.0))
            next_standard = standard.copy()
            next_standard[:, :, :2] = (
                standard[:, :, :2]
                + dt * standard[:, :, 2:4]
                + 0.5 * dt * dt * sigma_a
            )
            next_standard[:, :, 2:4] = (
                standard[:, :, 2:4] + dt * sigma_a
            )
            projected = OwnerLocalKinematicState(
                uav_position_m=projected.uav_position_m,
                uav_velocity_mps=projected.uav_velocity_mps,
                target_mean_by_owner=projected.target_mean_by_owner,
                target_cov_diag_by_owner=next_standard ** 2,
                target_aoi_frames_by_owner=(
                    projected.target_aoi_frames_by_owner),
            )
        states.append(projected)
    return tuple(states)


def owner_local_horizon_coefficient_bounds(
    states: tuple[OwnerLocalKinematicState, ...],
    target_invariant_lower: np.ndarray,
    target_invariant_upper: np.ndarray,
    provenance_support: np.ndarray,
    *,
    carrier_hz: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    covariance_radius: float,
    dd_support_threshold: float,
    dd_additive_margin: float = 0.0,
    residual_log_margin: float | np.ndarray = 0.0,
    uav_position_radius_m: float | np.ndarray = 0.0,
    uav_velocity_radius_mps: float | np.ndarray = 0.0,
) -> OwnerLocalHorizonCoefficientBounds:
    """Bound horizon coefficients using inverse-range and OTFS support laws.

    The target invariant is the per-watt path/report coefficient multiplied
    by both squared bistatic ranges.  In this simulator ``g_dd`` is a binary
    admission condition rather than a multiplicative amplitude.  Thus, on a
    DD-certified edge,

    ``a_lower = kappa_lower exp(-m) / (R_tx^U^2 R_rx^U^2)``.

    The upper bound uses ``kappa_upper``, the lower range endpoints and
    ``exp(+m)``.  If DD support cannot be certified, the lower coefficient is
    zero while the upper remains non-zero; this is the required fail-closed
    treatment of the discontinuous support threshold.
    """
    items = tuple(states)
    if not items:
        raise ValueError("at least one owner-local horizon state is required")
    invariant_lower = np.asarray(
        target_invariant_lower, dtype=np.float64).reshape(-1)
    invariant_upper = np.asarray(
        target_invariant_upper, dtype=np.float64).reshape(-1)
    support = np.asarray(provenance_support, dtype=bool)
    first_position = np.asarray(items[0].uav_position_m, dtype=np.float64)
    first_target = np.asarray(
        items[0].target_mean_by_owner, dtype=np.float64)
    if first_position.ndim != 2 or first_position.shape[1] != 3:
        raise ValueError("horizon UAV positions must have shape (K,3)")
    K = first_position.shape[0]
    if first_target.ndim != 3 or first_target.shape[0] != K:
        raise ValueError("horizon target means must have shape (K,Q,4)")
    Q = first_target.shape[1]
    if (
        invariant_lower.shape != (Q,) or invariant_upper.shape != (Q,)
        or support.shape != (K, K, Q)
    ):
        raise ValueError("invariant/support dimensions disagree with state")
    if (
        np.any(~np.isfinite(invariant_lower))
        or np.any(~np.isfinite(invariant_upper))
        or np.any(invariant_lower < 0.0)
        or np.any(invariant_upper < invariant_lower)
    ):
        raise ValueError("target invariant interval is invalid")
    beta = float(covariance_radius)
    threshold = float(dd_support_threshold)
    dd_margin = float(dd_additive_margin)
    if (
        not np.isfinite(beta) or beta < 0.0
        or not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0
        or not np.isfinite(dd_margin) or dd_margin < 0.0
    ):
        raise ValueError("DD bound parameters are invalid")
    margin = np.asarray(residual_log_margin, dtype=np.float64)
    if margin.ndim == 0:
        margin = np.full(len(items), float(margin), dtype=np.float64)
    else:
        margin = margin.reshape(-1)
    if (
        margin.shape != (len(items),) or np.any(~np.isfinite(margin))
        or np.any(margin < 0.0)
    ):
        raise ValueError("residual margin must be scalar or have H entries")
    uav_position_radius = np.asarray(
        uav_position_radius_m, dtype=np.float64)
    uav_velocity_radius = np.asarray(
        uav_velocity_radius_mps, dtype=np.float64)
    if uav_position_radius.ndim == 0:
        uav_position_radius = np.full(
            len(items), float(uav_position_radius))
    else:
        uav_position_radius = uav_position_radius.reshape(-1)
    if uav_velocity_radius.ndim == 0:
        uav_velocity_radius = np.full(
            len(items), float(uav_velocity_radius))
    else:
        uav_velocity_radius = uav_velocity_radius.reshape(-1)
    if (
        uav_position_radius.shape != (len(items),)
        or uav_velocity_radius.shape != (len(items),)
        or np.any(~np.isfinite(uav_position_radius))
        or np.any(~np.isfinite(uav_velocity_radius))
        or np.any(uav_position_radius < 0.0)
        or np.any(uav_velocity_radius < 0.0)
    ):
        raise ValueError(
            "UAV reachable-set radii must be scalar or have H entries")

    shape = (len(items), K, K, Q)
    lower = np.zeros(shape, dtype=np.float64)
    upper = np.zeros(shape, dtype=np.float64)
    dd_lower = np.zeros(shape, dtype=np.float64)
    dd_support = np.zeros(shape, dtype=bool)
    position_radius = np.zeros((len(items), K, Q), dtype=np.float64)
    off_diagonal = ~np.eye(K, dtype=bool)[:, :, None]
    for step, state in enumerate(items):
        position = np.asarray(state.uav_position_m, dtype=np.float64)
        target = np.asarray(state.target_mean_by_owner, dtype=np.float64)
        covariance = np.asarray(
            state.target_cov_diag_by_owner, dtype=np.float64)
        if (
            position.shape != (K, 3) or target.shape != (K, Q, 4)
            or covariance.shape != (K, Q, 4)
        ):
            raise ValueError("owner-local horizon state dimensions changed")
        if (
            np.any(~np.isfinite(position)) or np.any(~np.isfinite(target))
            or np.any(~np.isfinite(covariance)) or np.any(covariance < 0.0)
        ):
            raise ValueError("owner-local horizon state is invalid")
        dd = owner_local_dd_effectiveness_bounds(
            state,
            carrier_hz=float(carrier_hz),
            delta_f_hz=float(delta_f_hz),
            symbol_period_s=float(symbol_period_s),
            delay_bins=int(delay_bins),
            doppler_bins=int(doppler_bins),
            covariance_radius=beta,
            uav_position_radius_m=uav_position_radius[step],
            uav_velocity_radius_mps=uav_velocity_radius[step],
        )
        dd_lower[step] = dd.lower
        dd_support[step] = (
            support & off_diagonal
            & (dd.lower - dd_margin > threshold)
        )
        position_radius[step] = beta * np.sqrt(np.max(
            covariance[:, :, :2], axis=2))
        for transmitter, receiver, target_index in np.argwhere(
            support & off_diagonal
        ):
            owner_mean = target[receiver, target_index]
            target_position = np.asarray([
                owner_mean[0], owner_mean[1], 0.0,
            ], dtype=np.float64)
            target_radius = float(
                position_radius[step, receiver, target_index])
            tx_radius = (
                target_radius + float(uav_position_radius[step]))
            rx_radius = (
                target_radius + float(uav_position_radius[step]))
            tx_point = float(np.linalg.norm(
                position[transmitter] - target_position))
            rx_point = float(np.linalg.norm(
                position[receiver] - target_position))
            tx_lower = max(tx_point - tx_radius, 1.0e-6)
            rx_lower = max(rx_point - rx_radius, 1.0e-6)
            tx_upper = tx_point + tx_radius
            rx_upper = rx_point + rx_radius
            q = int(target_index)
            upper[step, transmitter, receiver, q] = (
                invariant_upper[q] * np.exp(margin[step])
                / (tx_lower ** 2 * rx_lower ** 2)
            )
            if dd_support[step, transmitter, receiver, q]:
                lower[step, transmitter, receiver, q] = (
                    invariant_lower[q] * np.exp(-margin[step])
                    / (tx_upper ** 2 * rx_upper ** 2)
                )
    return OwnerLocalHorizonCoefficientBounds(
        lower=lower,
        upper=upper,
        dd_lower=dd_lower,
        dd_certified_support=dd_support,
        target_position_radius_m=position_radius,
    )


def _observed_edge_and_target_invariants(
    coefficient: np.ndarray,
    observed_mask: np.ndarray,
    state: OwnerLocalKinematicState,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover edge and robust target invariants from excited edges only."""
    values = np.asarray(coefficient, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    if observed.shape != values.shape:
        raise ValueError("observed_mask must match coefficient")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("coefficients must be finite non-negative")
    K, _, Q = values.shape
    position = np.asarray(state.uav_position_m, dtype=np.float64)
    target = np.asarray(state.target_mean_by_owner, dtype=np.float64)
    if position.shape != (K, 3) or target.shape != (K, Q, 4):
        raise ValueError("kinematic state does not match coefficient shape")

    off_diagonal = ~np.eye(K, dtype=bool)[:, :, None]
    observed_positive = observed & off_diagonal & (values > 0.0)
    edge_invariant = np.zeros_like(values)
    for transmitter, receiver, target_index in np.argwhere(
        observed_positive
    ):
        target_xy = target[receiver, target_index, :2]
        target_position = np.asarray(
            [target_xy[0], target_xy[1], 0.0], dtype=np.float64)
        tx_range = max(float(np.linalg.norm(
            position[transmitter] - target_position)), 1.0e-6)
        rx_range = max(float(np.linalg.norm(
            position[receiver] - target_position)), 1.0e-6)
        edge_invariant[transmitter, receiver, target_index] = (
            values[transmitter, receiver, target_index]
            * tx_range ** 2 * rx_range ** 2
        )

    target_invariant = np.zeros(Q, dtype=np.float64)
    for target_index in range(Q):
        target_values = edge_invariant[:, :, target_index]
        target_values = target_values[target_values > 0.0]
        if target_values.size:
            target_invariant[target_index] = float(np.median(target_values))
    return edge_invariant, target_invariant


def update_target_invariant_cache(
    previous_coefficient: np.ndarray,
    previous_observed_mask: np.ndarray,
    previous_state: OwnerLocalKinematicState,
    *,
    prior_cache: OwnerTargetInvariantCache | None,
    elapsed_frames: int,
    max_age_frames: int,
) -> OwnerTargetInvariantCache:
    """Advance and refresh a target-invariant cache without imputation.

    An observation made at the previous decision epoch arrives with age
    ``elapsed_frames``.  Missing targets retain only their own prior value;
    information is never copied across targets.  Values older than
    ``max_age_frames`` are set to zero, so expiry is fail-closed.
    """
    values = np.asarray(previous_coefficient, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("previous_coefficient must have shape (K,K,Q)")
    Q = values.shape[2]
    elapsed = int(elapsed_frames)
    max_age = int(max_age_frames)
    if elapsed < 0 or max_age < 0:
        raise ValueError("cache ages must be non-negative")
    _, fresh = _observed_edge_and_target_invariants(
        values, previous_observed_mask, previous_state)

    if prior_cache is None:
        cached = np.zeros(Q, dtype=np.float64)
        age = np.full(Q, max_age + 1, dtype=np.int64)
        version = np.zeros(Q, dtype=np.int64)
    else:
        cached = np.asarray(
            prior_cache.target_invariant, dtype=np.float64).copy()
        age = np.asarray(prior_cache.age_frames, dtype=np.int64).copy()
        version = np.asarray(prior_cache.version, dtype=np.int64).copy()
        if cached.shape != (Q,) or age.shape != (Q,) or version.shape != (Q,):
            raise ValueError("prior cache must have one entry per target")
        if (
            np.any(~np.isfinite(cached)) or np.any(cached < 0.0)
            or np.any(age < 0) or np.any(version < 0)
        ):
            raise ValueError("prior cache contains invalid values")
        age = np.minimum(
            age, np.iinfo(np.int64).max - elapsed) + elapsed

    refreshed = fresh > 0.0
    if np.any(refreshed):
        cached[refreshed] = fresh[refreshed]
        age[refreshed] = elapsed
        version[refreshed] += 1
    expired = age > max_age
    cached[expired] = 0.0
    return OwnerTargetInvariantCache(
        target_invariant=cached,
        age_frames=age,
        version=version,
    )


def predict_coefficients_from_lagged_feedback(
    previous_coefficient: np.ndarray,
    previous_observed_mask: np.ndarray,
    previous_state: OwnerLocalKinematicState,
    current_state: OwnerLocalKinematicState,
    *,
    current_support: np.ndarray,
    target_invariant_cache: OwnerTargetInvariantCache | None = None,
) -> OwnerLocalCoefficientPrediction:
    """Transport selected-edge feedback with the bistatic inverse-range law.

    Each observed edge supplies the sufficient statistic
    ``kappa = coefficient * R_tx^2 * R_rx^2``.  A previously unexcited edge
    uses the median statistic for the same target.  If that target has no
    feedback, prediction fails closed to zero; there is no cross-target RCS
    imputation.
    """
    previous = np.asarray(previous_coefficient, dtype=np.float64)
    observed = np.asarray(previous_observed_mask, dtype=bool)
    support = np.asarray(current_support, dtype=bool)
    if previous.ndim != 3 or previous.shape[0] != previous.shape[1]:
        raise ValueError("previous_coefficient must have shape (K,K,Q)")
    if observed.shape != previous.shape or support.shape != previous.shape:
        raise ValueError("observed mask and support must match coefficients")
    if np.any(~np.isfinite(previous)) or np.any(previous < 0.0):
        raise ValueError("previous coefficients must be finite non-negative")
    K, _, Q = previous.shape
    previous_position = np.asarray(
        previous_state.uav_position_m, dtype=np.float64)
    current_position = np.asarray(
        current_state.uav_position_m, dtype=np.float64)
    previous_target = np.asarray(
        previous_state.target_mean_by_owner, dtype=np.float64)
    current_target = np.asarray(
        current_state.target_mean_by_owner, dtype=np.float64)
    if (
        previous_position.shape != (K, 3)
        or current_position.shape != (K, 3)
        or previous_target.shape != (K, Q, 4)
        or current_target.shape != (K, Q, 4)
    ):
        raise ValueError("kinematic state does not match coefficient shape")

    off_diagonal = ~np.eye(K, dtype=bool)[:, :, None]
    observed_positive = observed & off_diagonal & (previous > 0.0)
    invariant, fresh_target_invariant = (
        _observed_edge_and_target_invariants(
            previous, observed_positive, previous_state)
    )
    target_invariant = fresh_target_invariant.copy()
    target_age = np.full(Q, -1, dtype=np.int64)
    target_version = np.zeros(Q, dtype=np.int64)
    cached_target = np.zeros(Q, dtype=bool)
    if target_invariant_cache is not None:
        cache_value = np.asarray(
            target_invariant_cache.target_invariant, dtype=np.float64)
        cache_age = np.asarray(
            target_invariant_cache.age_frames, dtype=np.int64)
        cache_version = np.asarray(
            target_invariant_cache.version, dtype=np.int64)
        if (
            cache_value.shape != (Q,) or cache_age.shape != (Q,)
            or cache_version.shape != (Q,)
        ):
            raise ValueError("target cache must have one entry per target")
        if (
            np.any(~np.isfinite(cache_value)) or np.any(cache_value < 0.0)
            or np.any(cache_age < 0) or np.any(cache_version < 0)
        ):
            raise ValueError("target cache contains invalid values")
        target_age = cache_age.copy()
        target_version = cache_version.copy()
        cached_target = (target_invariant <= 0.0) & (cache_value > 0.0)
        target_invariant[cached_target] = cache_value[cached_target]

    predicted = np.zeros_like(previous)
    direct_count = 0
    fallback_count = 0
    cached_fallback_count = 0
    unavailable_count = 0
    for transmitter, receiver, target_index in np.argwhere(
        support & off_diagonal
    ):
        used_invariant = invariant[transmitter, receiver, target_index]
        if used_invariant > 0.0:
            direct_count += 1
        else:
            used_invariant = target_invariant[target_index]
            if used_invariant > 0.0:
                fallback_count += 1
                if cached_target[target_index]:
                    cached_fallback_count += 1
            else:
                unavailable_count += 1
                continue
        target_xy = current_target[receiver, target_index, :2]
        target_position = np.asarray(
            [target_xy[0], target_xy[1], 0.0], dtype=np.float64)
        tx_range = max(float(np.linalg.norm(
            current_position[transmitter] - target_position)), 1.0e-6)
        rx_range = max(float(np.linalg.norm(
            current_position[receiver] - target_position)), 1.0e-6)
        predicted[transmitter, receiver, target_index] = (
            used_invariant / (tx_range ** 2 * rx_range ** 2)
        )
    return OwnerLocalCoefficientPrediction(
        coefficient_per_watt=predicted,
        target_invariant=target_invariant,
        direct_edge_count=int(direct_count),
        target_fallback_edge_count=int(fallback_count),
        cached_target_fallback_edge_count=int(cached_fallback_count),
        unavailable_edge_count=int(unavailable_count),
        target_invariant_age_frames=target_age,
        target_invariant_version=target_version,
    )
