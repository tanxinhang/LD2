"""Target motion model: Constant Velocity (CV) with process noise."""

import numpy as np
from typing import Optional
from uav_isac.utils.types import TargetState


class Target:
    """A moving target with CV motion model.

    State: x = [px, py, vx, vy]
    Dynamics: x(t+1) = F * x(t) + w(t), w ~ N(0, Q_w)

    The z-coordinate is fixed at 0 (ground targets).
    """

    def __init__(
        self,
        target_id: int,
        initial_pos: np.ndarray,     # (2,) [px, py]
        initial_vel: np.ndarray,     # (2,) [vx, vy]
        sigma_a: float = 0.5,        # process noise std (m/s^2)
        dt: float = 0.1,             # time step (s)
        area_size: tuple = (1000.0, 1000.0),  # (width, height) for bounds
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            target_id: Unique target identifier
            initial_pos: Initial position [px, py] (m)
            initial_vel: Initial velocity [vx, vy] (m/s)
            sigma_a: Process noise standard deviation (m/s²)
            dt: Frame duration (s)
            area_size: Region bounds (width, height) in meters
            rng: NumPy random generator
        """
        self.id = target_id
        self.dt = dt
        self.sigma_a = sigma_a
        self.area_w, self.area_h = area_size
        self.rng = rng if rng is not None else np.random.default_rng()

        # State: [px, py, vx, vy]
        self.state = np.array([
            initial_pos[0], initial_pos[1],
            initial_vel[0], initial_vel[1]
        ], dtype=np.float64)

        # State transition matrix (CV model)
        self.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Process noise covariance
        # Q = sigma_a^2 * [[dt^4/4, 0, dt^3/2, 0], ...]
        # Standard discrete white noise acceleration model
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        q11 = dt4 / 4.0
        q13 = dt3 / 2.0
        q22 = dt4 / 4.0
        q24 = dt3 / 2.0
        q31 = dt3 / 2.0
        q33 = dt2
        q42 = dt3 / 2.0
        q44 = dt2

        self.Q = (sigma_a ** 2) * np.array([
            [q11, 0.0, q13, 0.0],
            [0.0, q22, 0.0, q24],
            [q31, 0.0, q33, 0.0],
            [0.0, q42, 0.0, q44],
        ], dtype=np.float64)

    def step(self) -> None:
        """Advance target state by one frame with process noise.

        x(t+1) = F * x(t) + w, w ~ N(0, Q)
        Position is reflected at area boundaries.
        """
        # Generate process noise
        w = self.rng.multivariate_normal(
            np.zeros(4), self.Q
        ).astype(np.float64)

        # State transition
        self.state = self.F @ self.state + w

        # Reflect at boundaries (soft bounce)
        if self.state[0] < 0:
            self.state[0] = -self.state[0]
            self.state[2] = abs(self.state[2])  # reverse vx
        elif self.state[0] > self.area_w:
            self.state[0] = 2 * self.area_w - self.state[0]
            self.state[2] = -abs(self.state[2])

        if self.state[1] < 0:
            self.state[1] = -self.state[1]
            self.state[3] = abs(self.state[3])  # reverse vy
        elif self.state[1] > self.area_h:
            self.state[1] = 2 * self.area_h - self.state[1]
            self.state[3] = -abs(self.state[3])

    def get_position(self) -> np.ndarray:
        """Get current position [px, py]."""
        return self.state[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy]."""
        return self.state[2:].copy()

    def get_position_3d(self) -> np.ndarray:
        """Get position as 3D [px, py, 0] (z=0 for ground target)."""
        return np.array([self.state[0], self.state[1], 0.0], dtype=np.float64)

    def get_state_as_target_state(self) -> TargetState:
        """Convert internal state to TargetState namedtuple."""
        return TargetState(
            pos=self.get_position_3d(),
            vel=np.array([self.state[2], self.state[3], 0.0], dtype=np.float64)
        )

    def reset(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Reset target to new initial state.

        Args:
            pos: [px, py] initial position
            vel: [vx, vy] initial velocity
        """
        self.state[0] = pos[0]
        self.state[1] = pos[1]
        self.state[2] = vel[0]
        self.state[3] = vel[1]
