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
                pos_noise = self.rng.normal(0, initial_position_std, size=2)
                vel_noise = self.rng.normal(0, initial_velocity_std, size=2)
                self.mean[k, q, :] = 0.0
                self.mean[k, q, 0] = initial_positions[q, 0] + pos_noise[0]
                self.mean[k, q, 1] = initial_positions[q, 1] + pos_noise[1]
                self.mean[k, q, 2] = initial_velocities[q, 0] + vel_noise[0]
                self.mean[k, q, 3] = initial_velocities[q, 1] + vel_noise[1]
                cov_init = np.zeros((sd, sd), dtype=np.float64)
                cov_init[0, 0] = initial_position_std**2
                cov_init[1, 1] = initial_position_std**2
                cov_init[2, 2] = initial_velocity_std**2
                cov_init[3, 3] = initial_velocity_std**2
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
    ) -> None:
        """Kalman update if target was observed (detected by this UAV).

        Uses a noisy measurement of TRUE target position/velocity with
        covariance self.R. Resets AoI to 0 on observation.

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

        mean = self.mean[uav_id, target_id]  # (sd,)
        cov = self.cov[uav_id, target_id]    # (sd, sd)

        # Noisy measurement of TRUE target state [x,y,vx,vy]
        noise = self.rng.normal(0, np.sqrt(np.diag(self.R)))
        z = true_state + noise  # (4,)

        # Measurement prediction
        z_pred = self.H @ mean   # (4,)
        # Innovation covariance: S = H P H^T + R
        S = self.H @ cov @ self.H.T + self.R  # (4, 4)
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
            I_KH @ cov @ I_KH.T + K_gain @ self.R @ K_gain.T
        )
        # Ensure symmetry
        self.cov[uav_id, target_id] = 0.5 * (
            self.cov[uav_id, target_id] + self.cov[uav_id, target_id].T
        )

        # Reset AoI
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
                pos_noise = self.rng.normal(0, 50.0, size=2)
                vel_noise = self.rng.normal(0, 5.0, size=2)
                self.mean[k, q, :] = 0.0
                self.mean[k, q, 0] = initial_positions[q, 0] + pos_noise[0]
                self.mean[k, q, 1] = initial_positions[q, 1] + pos_noise[1]
                self.mean[k, q, 2] = initial_velocities[q, 0] + vel_noise[0]
                self.mean[k, q, 3] = initial_velocities[q, 1] + vel_noise[1]
                # CA: ax, ay init at 0
                cov_init = np.zeros((sd, sd), dtype=np.float64)
                cov_init[0, 0] = 2500.0; cov_init[1, 1] = 2500.0
                cov_init[2, 2] = 25.0;   cov_init[3, 3] = 25.0
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
