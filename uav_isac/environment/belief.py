"""Belief manager: CV prediction + Kalman-lite update for target tracking.

Each frame:
  1. PREDICT:  mean <- F * mean  (CV motion model)
              cov  <- F * cov * F^T + Q  (process noise)
  2. NIS calibration (optional): inflate covariance based on innovation consistency
  3. UPDATE (if target detected): Kalman correction with noisy position measurement.
     AoI resets to 0; undetected targets' AoI increments.

NIS-driven covariance calibration (Layer 1 of Calibrate–Gate–Schedule–Recover):
  - Computes Normalized Innovation Squared (NIS) per (k,q) on each measurement
  - Maintains EMA of NIS/d_z (target = 1.0 for well-calibrated filter)
  - Asymmetric inflation: fast exponential inflate when NIS > 1, slow decay when normal
  - Physical eigenvalue floor prevents collapse to zero uncertainty
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from uav_isac.utils.types import BeliefState
from uav_isac.physical.geometry import C_LIGHT


def bistatic_range_doppler_measurement_and_jacobian(
    state: np.ndarray,
    transmitter_position_m: np.ndarray,
    transmitter_velocity_mps: np.ndarray,
    receiver_position_m: np.ndarray,
    receiver_velocity_mps: np.ndarray,
    carrier_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return physical bistatic range/Doppler and its state Jacobian.

    ``state`` is CV/CA ordered as ``[x,y,vx,vy,(ax,ay)]``; the target is on
    the ground plane.  The measured Doppler convention matches the sensing
    geometry used by the deflection computer.  Acceleration has no direct
    measurement column and becomes observable only through CA dynamics.
    """
    value = np.asarray(state, dtype=np.float64).reshape(-1)
    tx_position = np.asarray(
        transmitter_position_m, dtype=np.float64).reshape(-1)
    rx_position = np.asarray(
        receiver_position_m, dtype=np.float64).reshape(-1)
    tx_velocity = np.asarray(
        transmitter_velocity_mps, dtype=np.float64).reshape(-1)
    rx_velocity = np.asarray(
        receiver_velocity_mps, dtype=np.float64).reshape(-1)
    fc = float(carrier_hz)
    if (
        value.size not in (4, 6)
        or any(item.shape != (3,) for item in (
            tx_position, rx_position, tx_velocity, rx_velocity))
        or any(np.any(~np.isfinite(item)) for item in (
            value, tx_position, rx_position, tx_velocity, rx_velocity))
        or not np.isfinite(fc) or fc <= 0.0
    ):
        raise ValueError('bistatic state/endpoint inputs are invalid')
    target_position = np.asarray([value[0], value[1], 0.0])
    target_velocity = np.asarray([value[2], value[3], 0.0])
    measurement = np.zeros(2, dtype=np.float64)
    jacobian = np.zeros((2, value.size), dtype=np.float64)
    doppler_position_gradient = np.zeros(2, dtype=np.float64)
    doppler_velocity_gradient = np.zeros(2, dtype=np.float64)
    for node_position, node_velocity in (
        (tx_position, tx_velocity),
        (rx_position, rx_velocity),
    ):
        displacement = target_position - node_position
        distance = float(np.linalg.norm(displacement))
        if distance <= 1.0e-9:
            raise ValueError('bistatic endpoint cannot coincide with target')
        direction = displacement / distance
        relative_velocity = node_velocity - target_velocity
        measurement[0] += distance
        measurement[1] += float(relative_velocity @ direction)
        jacobian[0, :2] += direction[:2]
        direction_derivative = (
            np.eye(3, dtype=np.float64)
            - np.outer(direction, direction)
        ) / distance
        doppler_position_gradient += (
            direction_derivative @ relative_velocity)[:2]
        doppler_velocity_gradient -= direction[:2]
    doppler_scale = fc / C_LIGHT
    measurement[1] *= doppler_scale
    jacobian[1, :2] = doppler_scale * doppler_position_gradient
    jacobian[1, 2:4] = doppler_scale * doppler_velocity_gradient
    return measurement, jacobian


def bistatic_range_doppler_crlb(
    effective_deflection: float,
    bandwidth_hz: float,
    coherent_time_s: float,
    *,
    efficiency: float = 1.0,
    minimum_effective_deflection: float = 1.0e-3,
) -> np.ndarray:
    """Ideal known-signal delay/Doppler CRLB with declared practical loss.

    The deployed deflection is signal-energy/noise scaled by report and DD
    effectiveness, so it is the consistent effective-SNR statistic here.
    Rectangular occupied bandwidth/time use RMS spreads B/sqrt(12) and
    T/sqrt(12).  ``efficiency>=1`` prevents calling the ideal bound achieved
    estimator performance without calibration.
    """
    snr = float(effective_deflection)
    bandwidth = float(bandwidth_hz)
    coherent_time = float(coherent_time_s)
    loss = float(efficiency)
    floor = float(minimum_effective_deflection)
    if (
        not all(np.isfinite(item) for item in (
            snr, bandwidth, coherent_time, loss, floor))
        or snr < 0.0 or bandwidth <= 0.0 or coherent_time <= 0.0
        or loss < 1.0 or floor <= 0.0
    ):
        raise ValueError('bistatic CRLB inputs are invalid')
    effective_snr = max(snr, floor)
    beta_rms = bandwidth / np.sqrt(12.0)
    time_rms = coherent_time / np.sqrt(12.0)
    common = 8.0 * np.pi * np.pi * effective_snr
    range_variance = (
        loss * C_LIGHT * C_LIGHT / (common * beta_rms * beta_rms))
    doppler_variance = loss / (common * time_rms * time_rms)
    return np.diag([range_variance, doppler_variance])


def generalized_covariance_intersection(
    means: np.ndarray,
    covariances: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse correlated Gaussian estimates without independence claims.

    For convex weights ``omega_s``, generalized covariance intersection uses

    ``Y = sum_s omega_s P_s^{-1}``,
    ``y = sum_s omega_s P_s^{-1} m_s``.

    It therefore never sums information as if repeated U2U posteriors were
    independent.  Equal weights are permutation invariant and reduce to a
    linear consensus step in information coordinates.
    """
    values = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covariances, dtype=np.float64)
    if values.ndim != 2 or covs.shape != (
            values.shape[0], values.shape[1], values.shape[1]):
        raise ValueError('means/covariances have inconsistent shapes')
    count, dim = values.shape
    if count < 1 or dim < 1:
        raise ValueError('at least one non-empty estimate is required')
    if not (np.all(np.isfinite(values)) and np.all(np.isfinite(covs))):
        raise ValueError('belief estimates must be finite')
    if weights is None:
        omega = np.full(count, 1.0 / count, dtype=np.float64)
    else:
        omega = np.asarray(weights, dtype=np.float64).reshape(-1)
        if omega.shape != (count,) or np.any(~np.isfinite(omega)):
            raise ValueError('weights must match the estimate count')
        if np.any(omega < 0.0) or float(np.sum(omega)) <= 0.0:
            raise ValueError('weights must be non-negative with positive sum')
        omega = omega / float(np.sum(omega))
    information = np.zeros((dim, dim), dtype=np.float64)
    information_mean = np.zeros(dim, dtype=np.float64)
    for index in range(count):
        symmetric = 0.5 * (covs[index] + covs[index].T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        if float(np.min(eigenvalues)) <= 0.0:
            raise ValueError('covariances must be positive definite')
        precision = np.linalg.inv(symmetric)
        information += omega[index] * precision
        information_mean += omega[index] * precision @ values[index]
    fused_covariance = np.linalg.inv(information)
    fused_covariance = 0.5 * (
        fused_covariance + fused_covariance.T)
    fused_mean = fused_covariance @ information_mean
    return fused_mean, fused_covariance


def batched_pair_covariance_intersection(
    local_means: np.ndarray,
    local_covariances: np.ndarray,
    remote_means: np.ndarray,
    remote_covariances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized equal-weight CI for independent receiver/target pairs.

    This is algebraically the two-estimate specialization of
    :func:`generalized_covariance_intersection`.  Batching changes only the
    simulator execution graph: each row remains a separate CI problem and no
    cross-row information is introduced.
    """
    local_values = np.asarray(local_means, dtype=np.float64)
    remote_values = np.asarray(remote_means, dtype=np.float64)
    local_covs = np.asarray(local_covariances, dtype=np.float64)
    remote_covs = np.asarray(remote_covariances, dtype=np.float64)
    if (
        local_values.ndim != 2
        or remote_values.shape != local_values.shape
        or local_covs.shape != (
            local_values.shape[0],
            local_values.shape[1],
            local_values.shape[1],
        )
        or remote_covs.shape != local_covs.shape
    ):
        raise ValueError('batched pair beliefs have inconsistent shapes')
    if local_values.shape[0] == 0:
        return local_values.copy(), local_covs.copy()
    if not all(np.all(np.isfinite(item)) for item in (
        local_values, remote_values, local_covs, remote_covs,
    )):
        raise ValueError('batched pair beliefs must be finite')
    local_symmetric = 0.5 * (
        local_covs + np.swapaxes(local_covs, -1, -2))
    remote_symmetric = 0.5 * (
        remote_covs + np.swapaxes(remote_covs, -1, -2))
    if (
        np.any(np.linalg.eigvalsh(local_symmetric) <= 0.0)
        or np.any(np.linalg.eigvalsh(remote_symmetric) <= 0.0)
    ):
        raise ValueError('covariances must be positive definite')
    local_precision = np.linalg.inv(local_symmetric)
    remote_precision = np.linalg.inv(remote_symmetric)
    information = 0.5 * (local_precision + remote_precision)
    information_mean = 0.5 * (
        np.einsum('nij,nj->ni', local_precision, local_values)
        + np.einsum('nij,nj->ni', remote_precision, remote_values)
    )
    fused_covariance = np.linalg.inv(information)
    fused_covariance = 0.5 * (
        fused_covariance
        + np.swapaxes(fused_covariance, -1, -2)
    )
    fused_mean = np.einsum(
        'nij,nj->ni', fused_covariance, information_mean)
    return fused_mean, fused_covariance


def _cv_transition_matrix(dt: float) -> np.ndarray:
    """Constant-velocity state transition: [x, y, vx, vy]."""
    return np.array([
        [1.0, 0.0,  dt, 0.0],
        [0.0, 1.0, 0.0,  dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _cv_process_noise(dt: float, sigma_a: float) -> np.ndarray:
    """Process noise covariance (piecewise-white acceleration model)."""
    q_p = 0.25 * (dt ** 4) * (sigma_a ** 2)
    q_v = (dt ** 2) * (sigma_a ** 2)
    q_cross = 0.5 * (dt ** 3) * (sigma_a ** 2)
    return np.array([
        [q_p, 0.0, q_cross, 0.0],
        [0.0, q_p, 0.0, q_cross],
        [q_cross, 0.0, q_v, 0.0],
        [0.0, q_cross, 0.0, q_v],
    ], dtype=np.float64)


def _ca_transition_matrix(dt: float) -> np.ndarray:
    """Constant-acceleration state: [x, y, vx, vy, ax, ay]."""
    dt2_2 = 0.5 * dt * dt
    return np.array([
        [1.0, 0.0,  dt, 0.0, dt2_2, 0.0],
        [0.0, 1.0, 0.0,  dt, 0.0, dt2_2],
        [0.0, 0.0, 1.0, 0.0,   dt, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0,   dt],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _ca_process_noise(dt: float, sigma_jerk: float) -> np.ndarray:
    """Process noise for CA model (piecewise-white jerk)."""
    q = sigma_jerk ** 2
    dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt3 * dt; dt5 = dt4 * dt
    return np.array([
        [dt5/20, 0, dt4/8, 0, dt3/6, 0],
        [0, dt5/20, 0, dt4/8, 0, dt3/6],
        [dt4/8, 0, dt3/3, 0, dt2/2, 0],
        [0, dt4/8, 0, dt3/3, 0, dt2/2],
        [dt3/6, 0, dt2/2, 0, dt, 0],
        [0, dt3/6, 0, dt2/2, 0, dt],
    ], dtype=np.float64)


class BeliefManager:
    """Kalman filter: CV or CA motion model + noisy-measurement update.

    CV:  state = [x, y, vx, vy]          (4D)
    CA:  state = [x, y, vx, vy, ax, ay]  (6D)
    Measurement observes [x, y, vx, vy] for both models.
    """

    def __init__(
        self,
        K: int,
        Q: int,
        initial_positions: np.ndarray,   # (Q, 3) true target positions
        initial_velocities: np.ndarray,  # (Q, 3) true target velocities
        initial_position_std: float = 50.0,
        initial_velocity_std: float = 5.0,
        dt: float = 0.1,
        sigma_a: float = 0.5,             # target process noise (m/s²)
        meas_pos_std: float = 15.0,        # measurement noise std (m)
        meas_vel_std: float = 3.0,         # measurement noise std (m/s)
        rng: Optional[np.random.Generator] = None,
        motion_model: str = 'CV',          # 'CV' or 'CA'
        sigma_jerk: float = 1.0,           # CA process noise (m/s³)
        # ── NIS calibration (Layer 1) ──
        nis_enabled: bool = False,
        nis_window: float = 0.05,
        nis_inflate_k: float = 2.0,
        nis_lambda_max: float = 5.0,
        nis_deflate_rate: float = 0.95,
        cov_floor_pos: float = 25.0,
        cov_floor_vel: float = 1.0,
        nis_enter_threshold: float = 1.8,
        nis_exit_threshold: float = 1.2,
        nis_enter_frames: int = 3,
        nis_exit_frames: int = 5,
    ):
        self.K = K
        self.Q = Q
        self.dt = dt
        self.sigma_a = sigma_a
        self.meas_pos_std = meas_pos_std
        self.meas_vel_std = meas_vel_std
        self.initial_position_std = float(initial_position_std)
        self.initial_velocity_std = float(initial_velocity_std)
        if (
            not np.isfinite(self.initial_position_std)
            or not np.isfinite(self.initial_velocity_std)
            or self.initial_position_std < 0.0
            or self.initial_velocity_std < 0.0
        ):
            raise ValueError('initial belief standard deviations must be finite and non-negative')
        self.rng = rng if rng is not None else np.random.default_rng()
        self.motion_model = motion_model

        # Configure model dimensions
        if motion_model == 'CA':
            self.state_dim = 6
            self.meas_dim = 4  # observe [x,y,vx,vy]
            self.F = _ca_transition_matrix(dt)
            self.Q_proc_base = _ca_process_noise(dt, sigma_jerk)
            self.H = np.zeros((4, 6), dtype=np.float64)
            self.H[0, 0] = 1.0; self.H[1, 1] = 1.0
            self.H[2, 2] = 1.0; self.H[3, 3] = 1.0
        else:  # CV
            self.state_dim = 4
            self.meas_dim = 4  # observe [x,y,vx,vy] directly
            self.F = _cv_transition_matrix(dt)
            self.Q_proc_base = _cv_process_noise(dt, sigma_a)
            self.H = np.eye(4)

        self.Q_proc = self.Q_proc_base.copy()
        self.R = np.diag([meas_pos_std**2, meas_pos_std**2,
                          meas_vel_std**2, meas_vel_std**2])

        # ── NIS calibration state ──
        self.nis_enabled = nis_enabled
        self.nis_window = nis_window
        self.nis_inflate_k = nis_inflate_k
        self.nis_lambda_max = nis_lambda_max
        self.nis_deflate_rate = nis_deflate_rate
        floor_vals = [cov_floor_pos, cov_floor_pos, cov_floor_vel, cov_floor_vel]
        if self.state_dim >= 6:
            floor_vals += [0.01, 0.01]  # small floor for ax, ay variance
        self._cov_floor_diag = np.array(floor_vals, dtype=np.float64)
        # Hysteresis thresholds
        self.nis_enter_threshold = nis_enter_threshold
        self.nis_exit_threshold = nis_exit_threshold
        self.nis_enter_frames = nis_enter_frames
        self.nis_exit_frames = nis_exit_frames
        # Per-(k,q) state
        self.nis_ema = np.ones((K, Q), dtype=np.float64)       # NIS/d_z EMA
        self.inflate_factor = np.ones((K, Q), dtype=np.float64) # λ_t
        self._last_nis = np.ones((K, Q), dtype=np.float64)     # most recent NIS
        # State machine: 0=NORMAL, 1=SUSPECT, 2=RECOVERING
        self.nis_state = np.zeros((K, Q), dtype=np.int32)
        self.nis_consecutive = np.zeros((K, Q), dtype=np.int32) # consecutive counter
        # Adaptive-Q: scale factor per (k,q), starts at 1.0
        self.q_scale = np.ones((K, Q), dtype=np.float64)
        self.q_scale_max = 100.0   # max Q scaling
        self.q_scale_alpha = 0.05  # EMA toward target

        # Per-UAV, per-target beliefs
        sd = self.state_dim
        self.mean = np.zeros((K, Q, sd), dtype=np.float64)
        self.cov = np.zeros((K, Q, sd, sd), dtype=np.float64)
        self.aoi = np.zeros((K, Q), dtype=np.int32)

        # Init
        sd = self.state_dim
        for k in range(K):
            for q in range(Q):
                pos_noise = self.rng.normal(0, self.initial_position_std, size=2)
                vel_noise = self.rng.normal(0, self.initial_velocity_std, size=2)
                self.mean[k, q, :] = 0.0
                self.mean[k, q, 0] = initial_positions[q, 0] + pos_noise[0]
                self.mean[k, q, 1] = initial_positions[q, 1] + pos_noise[1]
                self.mean[k, q, 2] = initial_velocities[q, 0] + vel_noise[0]
                self.mean[k, q, 3] = initial_velocities[q, 1] + vel_noise[1]
                cov_init = np.zeros((sd, sd), dtype=np.float64)
                cov_init[0, 0] = self.initial_position_std**2
                cov_init[1, 1] = self.initial_position_std**2
                cov_init[2, 2] = self.initial_velocity_std**2
                cov_init[3, 3] = self.initial_velocity_std**2
                if sd >= 6:
                    cov_init[4, 4] = 1.0; cov_init[5, 5] = 1.0
                self.cov[k, q] = cov_init
                self.aoi[k, q] = 1

    def get_belief(self, uav_id: int, target_id: int) -> BeliefState:
        """Return belief as 4D [x,y,vx,vy] regardless of internal model."""
        if self.motion_model == 'CA':
            # Project 6D → 4D: take first 4 components [x,y,vx,vy]
            mean_4d = self.mean[uav_id, target_id, :4].copy()
            cov_4d = self.cov[uav_id, target_id, :4, :4]
            cov_diag_4d = np.diag(cov_4d).copy()
        else:
            mean_4d = self.mean[uav_id, target_id].copy()
            cov_diag_4d = np.diag(self.cov[uav_id, target_id]).copy()
        return BeliefState(
            mean=mean_4d,
            cov_diag=cov_diag_4d,
            aoi=int(self.aoi[uav_id, target_id]),
        )

    def get_all_beliefs(self, uav_id: int) -> List[BeliefState]:
        return [self.get_belief(uav_id, q) for q in range(self.Q)]

    def step(self) -> None:
        """CV prediction + optional NIS calibration, increment AoI.

        NIS state machine (hysteresis):
          NORMAL  → SUSPECT    when r̄ ≥ τ_enter for M_enter frames
          SUSPECT → RECOVERING when r̄ <  τ_exit  for M_exit  frames
          RECOVERING → NORMAL  when r̄ <  τ_exit  for M_exit  frames
          τ_exit < τ_enter (prevents flickering)

        Inflation (linear multiplicative + additive floor):
          λ_t = 1 + k_λ · max(r̄_t − 1, 0)     clamped to [1, λ_max]
          P_cal = λ_t · P_raw + δI            (δI = diagonal floor)
          λ applied to FRESH prediction → no compounding across frames
        """
        sd = self.state_dim
        for k in range(self.K):
            for q in range(self.Q):
                # Predict with adaptive Q
                self.mean[k, q] = self.F @ self.mean[k, q]
                Q_effective = self.q_scale[k, q] * self.Q_proc_base
                self.cov[k, q] = (self.F @ self.cov[k, q] @ self.F.T
                                  + Q_effective)

                # NIS-driven covariance calibration (Layer 1)
                if self.nis_enabled:
                    r_bar = self.nis_ema[k, q]
                    state = self.nis_state[k, q]
                    count = self.nis_consecutive[k, q]

                    # State machine transitions
                    if state == 0:  # NORMAL
                        if r_bar >= self.nis_enter_threshold:
                            count += 1
                            if count >= self.nis_enter_frames:
                                state = 1  # → SUSPECT
                                count = 0
                        else:
                            count = 0
                    elif state == 1:  # SUSPECT
                        if r_bar < self.nis_exit_threshold:
                            count += 1
                            if count >= self.nis_exit_frames:
                                state = 2  # → RECOVERING
                                count = 0
                        else:
                            count = 0
                    elif state == 2:  # RECOVERING
                        if r_bar < self.nis_exit_threshold:
                            count += 1
                            if count >= self.nis_exit_frames:
                                state = 0  # → NORMAL
                                count = 0
                        else:
                            # Back to SUSPECT if NIS rises again
                            state = 1
                            count = 0

                    self.nis_state[k, q] = state
                    self.nis_consecutive[k, q] = count

                    # Adaptive-Q: scale process noise based on NIS state
                    if state >= 1:  # SUSPECT or RECOVERING
                        # Target: Q should explain observed NIS excess
                        target_q = min(r_bar, self.q_scale_max)
                        self.q_scale[k, q] += self.q_scale_alpha * (target_q - self.q_scale[k, q])
                    elif state == 0 and self.q_scale[k, q] > 1.0:
                        # NORMAL: slowly decay Q_scale toward 1.0
                        self.q_scale[k, q] += self.q_scale_alpha * (1.0 - self.q_scale[k, q])

                    # Linear multiplicative inflation (from fresh prediction)
                    if state >= 1:  # SUSPECT or RECOVERING
                        # λ = 1 + k·max(r̄−1, 0), capped
                        lam = 1.0 + self.nis_inflate_k * max(r_bar - 1.0, 0.0)
                        lam = min(lam, self.nis_lambda_max)
                        self.inflate_factor[k, q] = lam
                    elif r_bar > 1.0:
                        # NORMAL but slight elevation: gentle inflation
                        lam = 1.0 + self.nis_inflate_k * (r_bar - 1.0)
                        lam = min(lam, self.nis_lambda_max)
                        self.inflate_factor[k, q] = lam
                    else:
                        # r̄ ≤ 1: decay toward 1.0
                        self.inflate_factor[k, q] = max(
                            1.0,
                            self.inflate_factor[k, q] * self.nis_deflate_rate,
                        )

                    # Apply: P_cal = λ · P_raw + δI
                    lam = self.inflate_factor[k, q]
                    if lam > 1.0:
                        self.cov[k, q] = lam * self.cov[k, q]
                    # Additive floor (scale-independent, δI)
                    d_idx = np.diag_indices(sd)
                    self.cov[k, q][d_idx] = np.maximum(
                        self.cov[k, q][d_idx], self._cov_floor_diag)

        self.aoi += 1

    def update_after_observation(
        self,
        uav_id: int,
        target_id: int,
        observed: bool,
        true_state: Optional[np.ndarray] = None,  # (4,) [x,y,vx,vy]
        *,
        detection_probability: Optional[float] = None,
        detection_probability_floor: float = 1.0e-3,
    ) -> None:
        """Kalman update if target was observed (detected by this UAV).

        Uses a noisy measurement of TRUE target position/velocity.  The
        historical path uses covariance ``R``.  When a detection probability
        is supplied, the deterministic expected-information approximation
        uses ``R_eff=R/max(p,p_floor)``, because a Bernoulli-available
        measurement contributes expected Fisher information ``p R^-1``.
        Resets AoI to 0 on observation.

        When NIS calibration is enabled, computes the Normalized Innovation
        Squared and updates the per-(k,q) EMA for covariance inflation.

        Args:
            uav_id: UAV index
            target_id: Target index
            observed: Whether this UAV observed this target this frame
            true_state: (4,) true target state [x,y,vx,vy]; if None, skip update
        """
        if not observed or true_state is None:
            return

        effective_R = self.R
        if detection_probability is not None:
            probability = float(detection_probability)
            probability_floor = float(detection_probability_floor)
            if (
                not np.isfinite(probability)
                or not 0.0 <= probability <= 1.0
                or not np.isfinite(probability_floor)
                or not 0.0 < probability_floor <= 1.0
            ):
                raise ValueError(
                    'detection probability/floor must lie in [0,1]/(0,1]')
            effective_R = self.R / max(probability, probability_floor)

        mean = self.mean[uav_id, target_id]  # (sd,)
        cov = self.cov[uav_id, target_id]    # (sd, sd)

        # Noisy measurement of TRUE target state [x,y,vx,vy]
        noise = self.rng.normal(0, np.sqrt(np.diag(effective_R)))
        z = true_state + noise  # (4,)

        # Measurement prediction
        z_pred = self.H @ mean   # (4,)
        # Innovation covariance: S = H P H^T + R
        S = self.H @ cov @ self.H.T + effective_R  # (4, 4)
        # Kalman gain: K = P H^T S^{-1}
        K_gain = cov @ self.H.T @ np.linalg.inv(S)  # (sd, 4)
        innovation = z - z_pred  # (4,)

        # ── NIS computation (before state is updated) ──
        if self.nis_enabled:
            d_z = 4
            nis = float(innovation @ np.linalg.solve(S, innovation))
            self._last_nis[uav_id, target_id] = nis
            # EMA update: r̄ ← (1−ρ)·r̄ + ρ·(NIS/d_z)
            self.nis_ema[uav_id, target_id] = (
                (1.0 - self.nis_window) * self.nis_ema[uav_id, target_id]
                + self.nis_window * (nis / d_z)
            )

        self.mean[uav_id, target_id] = mean + K_gain @ innovation
        # Joseph form: P+ = (I-KH)P-(I-KH)^T + K R K^T
        I_KH = np.eye(self.state_dim) - K_gain @ self.H  # (sd, sd)
        self.cov[uav_id, target_id] = (
            I_KH @ cov @ I_KH.T + K_gain @ effective_R @ K_gain.T
        )
        # Ensure symmetry
        self.cov[uav_id, target_id] = 0.5 * (
            self.cov[uav_id, target_id] + self.cov[uav_id, target_id].T
        )

        # Reset AoI
        self.aoi[uav_id, target_id] = 0

    def update_after_bistatic_observation(
        self,
        uav_id: int,
        target_id: int,
        observed: bool,
        true_state: Optional[np.ndarray],
        *,
        transmitter_position_m: np.ndarray,
        transmitter_velocity_mps: np.ndarray,
        receiver_position_m: np.ndarray,
        receiver_velocity_mps: np.ndarray,
        carrier_hz: float,
        effective_deflection: float,
        bandwidth_hz: float,
        coherent_time_s: float,
        crlb_efficiency: float = 1.0,
        minimum_effective_deflection: float = 1.0e-3,
    ) -> None:
        """EKF update from one selected physical bistatic echo.

        Only the receiver-local filter should consume this update.  Simulator
        truth is used solely to generate a noisy range/Doppler measurement;
        the estimator receives neither Cartesian truth nor target velocity.
        """
        if not observed or true_state is None:
            return
        target_truth = np.asarray(true_state, dtype=np.float64).reshape(-1)
        if target_truth.shape != (4,) or np.any(~np.isfinite(target_truth)):
            raise ValueError('true_state must be finite [x,y,vx,vy]')
        mean = self.mean[uav_id, target_id]
        covariance = self.cov[uav_id, target_id]
        truth_for_model = np.zeros(self.state_dim, dtype=np.float64)
        truth_for_model[:4] = target_truth
        true_measurement, _ = (
            bistatic_range_doppler_measurement_and_jacobian(
                truth_for_model,
                transmitter_position_m,
                transmitter_velocity_mps,
                receiver_position_m,
                receiver_velocity_mps,
                carrier_hz,
            )
        )
        predicted_measurement, jacobian = (
            bistatic_range_doppler_measurement_and_jacobian(
                mean,
                transmitter_position_m,
                transmitter_velocity_mps,
                receiver_position_m,
                receiver_velocity_mps,
                carrier_hz,
            )
        )
        measurement_covariance = bistatic_range_doppler_crlb(
            effective_deflection,
            bandwidth_hz,
            coherent_time_s,
            efficiency=crlb_efficiency,
            minimum_effective_deflection=minimum_effective_deflection,
        )
        measurement = true_measurement + self.rng.normal(
            0.0, np.sqrt(np.diag(measurement_covariance)))
        innovation = measurement - predicted_measurement
        innovation_covariance = (
            jacobian @ covariance @ jacobian.T
            + measurement_covariance
        )
        kalman_gain = (
            covariance @ jacobian.T
            @ np.linalg.inv(innovation_covariance)
        )
        if self.nis_enabled:
            nis = float(
                innovation
                @ np.linalg.solve(innovation_covariance, innovation))
            self._last_nis[uav_id, target_id] = nis
            self.nis_ema[uav_id, target_id] = (
                (1.0 - self.nis_window) * self.nis_ema[uav_id, target_id]
                + self.nis_window * (nis / 2.0)
            )
        self.mean[uav_id, target_id] = mean + kalman_gain @ innovation
        identity_minus_kh = (
            np.eye(self.state_dim) - kalman_gain @ jacobian)
        posterior = (
            identity_minus_kh @ covariance @ identity_minus_kh.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        self.cov[uav_id, target_id] = 0.5 * (
            posterior + posterior.T)
        self.aoi[uav_id, target_id] = 0

    def get_nis_status(self, uav_id: int, target_id: int) -> Dict:
        """Return NIS calibration diagnostics for one (k,q) pair.

        Returns:
            dict with keys: nis_ema, inflate_factor, last_nis, cov_diag_cal,
            nis_state (0=NORMAL,1=SUSPECT,2=RECOVERING), nis_consecutive
        """
        state_names = {0: 'NORMAL', 1: 'SUSPECT', 2: 'RECOVERING'}
        return {
            'nis_ema': float(self.nis_ema[uav_id, target_id]),
            'inflate_factor': float(self.inflate_factor[uav_id, target_id]),
            'last_nis': float(self._last_nis[uav_id, target_id]),
            'cov_diag_cal': np.diag(self.cov[uav_id, target_id]).copy(),
            'nis_state': int(self.nis_state[uav_id, target_id]),
            'nis_state_name': state_names[int(self.nis_state[uav_id, target_id])],
            'nis_consecutive': int(self.nis_consecutive[uav_id, target_id]),
        }

    def get_all_nis_status(self) -> Dict[str, np.ndarray]:
        """Return NIS calibration state for all (k,q) pairs.

        Returns:
            dict with keys: nis_ema (K,Q), inflate_factor (K,Q),
            last_nis (K,Q)
        """
        return {
            'nis_ema': self.nis_ema.copy(),
            'inflate_factor': self.inflate_factor.copy(),
            'last_nis': self._last_nis.copy(),
        }

    def get_calibrated_covariance(self) -> np.ndarray:
        """Return the current (possibly inflated) covariance matrices.

        Returns:
            cov: (K, Q, 4, 4) array of calibrated covariance matrices
        """
        return self.cov.copy()

    def get_cov_diag(self, uav_id: int, target_id: int) -> np.ndarray:
        """Return the diagonal of the calibrated covariance for one (k,q).

        Returns:
            cov_diag: (4,) diagonal of covariance matrix
        """
        return np.diag(self.cov[uav_id, target_id]).copy()

    def reset(
        self,
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
    ) -> None:
        sd = self.state_dim
        for k in range(self.K):
            for q in range(self.Q):
                pos_noise = self.rng.normal(
                    0, self.initial_position_std, size=2)
                vel_noise = self.rng.normal(
                    0, self.initial_velocity_std, size=2)
                self.mean[k, q, :] = 0.0
                self.mean[k, q, 0] = initial_positions[q, 0] + pos_noise[0]
                self.mean[k, q, 1] = initial_positions[q, 1] + pos_noise[1]
                self.mean[k, q, 2] = initial_velocities[q, 0] + vel_noise[0]
                self.mean[k, q, 3] = initial_velocities[q, 1] + vel_noise[1]
                # CA: ax, ay init at 0
                cov_init = np.zeros((sd, sd), dtype=np.float64)
                cov_init[0, 0] = self.initial_position_std**2
                cov_init[1, 1] = self.initial_position_std**2
                cov_init[2, 2] = self.initial_velocity_std**2
                cov_init[3, 3] = self.initial_velocity_std**2
                if sd >= 6:
                    cov_init[4, 4] = 1.0; cov_init[5, 5] = 1.0  # ax,ay variance
                self.cov[k, q] = cov_init
                self.aoi[k, q] = 1
        # Reset NIS state
        self.nis_ema.fill(1.0)
        self.inflate_factor.fill(1.0)
        self._last_nis.fill(1.0)
        self.nis_state.fill(0)
        self.nis_consecutive.fill(0)
        self.q_scale.fill(1.0)
