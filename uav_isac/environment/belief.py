"""Belief manager: CV prediction + Kalman-lite update for target tracking.

Each frame:
  1. PREDICT:  mean <- F * mean  (CV motion model)
              cov  <- F * cov * F^T + Q  (process noise)
  2. UPDATE (if target detected): Kalman correction with noisy position measurement.
     AoI resets to 0; undetected targets' AoI increments.

This gives the actor a dynamically-updated estimate of where each target is,
closing the "open-loop belief" bottleneck identified by the oracle diagnostic.
"""

import numpy as np
from typing import List, Optional
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
    q_p = 0.25 * (dt ** 4) * (sigma_a ** 2)  # position variance
    q_v = (dt ** 2) * (sigma_a ** 2)          # velocity variance
    q_cross = 0.5 * (dt ** 3) * (sigma_a ** 2)
    return np.array([
        [q_p, 0.0, q_cross, 0.0],
        [0.0, q_p, 0.0, q_cross],
        [q_cross, 0.0, q_v, 0.0],
        [0.0, q_cross, 0.0, q_v],
    ], dtype=np.float64)


class BeliefManager:
    """Kalman-lite belief filter: CV predict + noisy-measurement update."""

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
    ):
        self.K = K
        self.Q = Q
        self.dt = dt
        self.sigma_a = sigma_a
        self.meas_pos_std = meas_pos_std
        self.meas_vel_std = meas_vel_std
        self.rng = rng if rng is not None else np.random.default_rng()

        # Pre-compute CV matrices
        self.F = _cv_transition_matrix(dt)
        self.Q_proc = _cv_process_noise(dt, sigma_a)
        self.R = np.diag([meas_pos_std**2, meas_pos_std**2,
                          meas_vel_std**2, meas_vel_std**2])

        # Per-UAV, per-target beliefs
        self.mean = np.zeros((K, Q, 4), dtype=np.float64)
        self.cov = np.zeros((K, Q, 4, 4), dtype=np.float64)
        self.aoi = np.zeros((K, Q), dtype=np.int32)

        # Init
        for k in range(K):
            for q in range(Q):
                pos_noise = self.rng.normal(0, initial_position_std, size=2)
                vel_noise = self.rng.normal(0, initial_velocity_std, size=2)
                self.mean[k, q, 0] = initial_positions[q, 0] + pos_noise[0]
                self.mean[k, q, 1] = initial_positions[q, 1] + pos_noise[1]
                self.mean[k, q, 2] = initial_velocities[q, 0] + vel_noise[0]
                self.mean[k, q, 3] = initial_velocities[q, 1] + vel_noise[1]
                self.cov[k, q] = np.diag([
                    initial_position_std**2, initial_position_std**2,
                    initial_velocity_std**2, initial_velocity_std**2,
                ])
                self.aoi[k, q] = 1

    def get_belief(self, uav_id: int, target_id: int) -> BeliefState:
        return BeliefState(
            mean=self.mean[uav_id, target_id].copy(),
            cov_diag=np.diag(self.cov[uav_id, target_id]).copy(),
            aoi=int(self.aoi[uav_id, target_id]),
        )

    def get_all_beliefs(self, uav_id: int) -> List[BeliefState]:
        return [self.get_belief(uav_id, q) for q in range(self.Q)]

    def step(self) -> None:
        """CV prediction: advance mean and covariance, increment AoI."""
        for k in range(self.K):
            for q in range(self.Q):
                # Predict
                self.mean[k, q] = self.F @ self.mean[k, q]
                self.cov[k, q] = (self.F @ self.cov[k, q] @ self.F.T
                                  + self.Q_proc)
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

        Args:
            uav_id: UAV index
            target_id: Target index
            observed: Whether this UAV observed this target this frame
            true_state: (4,) true target state [x,y,vx,vy]; if None, skip update
        """
        if not observed or true_state is None:
            return

        mean = self.mean[uav_id, target_id]
        cov = self.cov[uav_id, target_id]

        # Noisy measurement of TRUE target state
        noise = self.rng.normal(0, np.sqrt(np.diag(self.R)))
        z = true_state + noise

        # Kalman update
        S = cov + self.R  # innovation covariance
        K_gain = cov @ np.linalg.inv(S)  # Kalman gain (4x4)
        innovation = z - mean
        self.mean[uav_id, target_id] = mean + K_gain @ innovation
        self.cov[uav_id, target_id] = (np.eye(4) - K_gain) @ cov

        # Reset AoI
        self.aoi[uav_id, target_id] = 0

    def reset(
        self,
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
    ) -> None:
        for k in range(self.K):
            for q in range(self.Q):
                pos_noise = self.rng.normal(0, 50.0, size=2)
                vel_noise = self.rng.normal(0, 5.0, size=2)
                self.mean[k, q, 0] = initial_positions[q, 0] + pos_noise[0]
                self.mean[k, q, 1] = initial_positions[q, 1] + pos_noise[1]
                self.mean[k, q, 2] = initial_velocities[q, 0] + vel_noise[0]
                self.mean[k, q, 3] = initial_velocities[q, 1] + vel_noise[1]
                self.cov[k, q] = np.diag([2500.0, 2500.0, 25.0, 25.0])
                self.aoi[k, q] = 1
