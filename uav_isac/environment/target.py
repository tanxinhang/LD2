"""Target motion models: CV, CT (Coordinated Turn), CA (Constant Acceleration).

CV: Constant Velocity (Kalman assumes this - no mismatch)
CT: Coordinated Turn (Kalman is CV -> model mismatch)
CA: Constant Acceleration (Kalman is CV -> model mismatch)

CT and CA create belief divergence -> larger P0 scheduling errors ->
larger oracle gap -> measurable headroom for advanced methods.
"""
import numpy as np
from typing import Optional
from uav_isac.utils.types import TargetState


class Target:
    """A moving target with configurable motion model.

    CV:  x = [px, py, vx, vy],               F = CV transition
    CT:  x = [px, py, v, theta, omega],       analytic update
    CA:  x = [px, py, vx, vy, ax, ay],       F = CA transition

    Kalman belief filter ALWAYS assumes CV. CT/CA create model mismatch.
    """

    def __init__(
        self,
        target_id: int,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
        sigma_a: float = 0.5,
        dt: float = 0.1,
        area_size: tuple = (1000.0, 1000.0),
        rng: Optional[np.random.Generator] = None,
        motion_model: str = "CV",
        turn_rate: float = 0.3,
    ):
        self.id = target_id
        self.dt = dt
        self.sigma_a = sigma_a
        self.area_w, self.area_h = area_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.model = motion_model.upper()
        self.turn_rate = turn_rate
        self._F = None   # state transition matrix (CV/CA only)
        self._Q = None   # process noise covariance

        if self.model == "CV":
            self._init_cv(pos=initial_pos, vel=initial_vel)
        elif self.model == "CT":
            self._init_ct(pos=initial_pos, vel=initial_vel)
        elif self.model == "CA":
            self._init_ca(pos=initial_pos, vel=initial_vel)
        else:
            raise ValueError(f"Unknown motion model: {motion_model}")

    # ── Init ────────────────────────────────────────────────────────
    def _init_cv(self, pos, vel):
        self.state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)
        dt = self.dt; dt2 = dt**2; dt3 = dt**3; dt4 = dt**4
        self._F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        s2 = self.sigma_a**2
        self._Q = s2 * np.array([[dt4/4,0,dt3/2,0],[0,dt4/4,0,dt3/2],
                                  [dt3/2,0,dt2,0],[0,dt3/2,0,dt2]], dtype=np.float64)

    def _init_ct(self, pos, vel):
        speed = np.linalg.norm(vel)
        theta = np.arctan2(vel[1], vel[0]) if speed > 0.1 else 0.0
        self.state = np.array([pos[0], pos[1], speed, theta, self.turn_rate],
                              dtype=np.float64)
        dt = self.dt
        self._Q = np.diag([1e-4, 1e-4, (self.sigma_a*dt)**2,
                           (0.1*self.sigma_a*dt)**2, 1e-8])

    def _init_ca(self, pos, vel):
        self.state = np.array([pos[0], pos[1], vel[0], vel[1], 0.0, 0.0],
                              dtype=np.float64)
        dt = self.dt; dt2 = dt**2; dt3 = dt**3; dt4 = dt**4
        self._F = np.array([
            [1,0,dt,0,dt2/2,0],[0,1,0,dt,0,dt2/2],
            [0,0,1,0,dt,0],[0,0,0,1,0,dt],
            [0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.float64)
        qj = (0.5*self.sigma_a)**2
        self._Q = qj * np.array([
            [dt4/4,0,dt3/2,0,dt2/2,0],[0,dt4/4,0,dt3/2,0,dt2/2],
            [dt3/2,0,dt2,0,dt,0],[0,dt3/2,0,dt2,0,dt],
            [dt2/2,0,dt,0,1,0],[0,dt2/2,0,dt,0,1]], dtype=np.float64)

    # ── Step ────────────────────────────────────────────────────────
    def step(self) -> None:
        if self.model == "CV":
            self._step_cv()
        elif self.model == "CT":
            self._step_ct()
        elif self.model == "CA":
            self._step_ca()
        self._bounce()

    def _step_cv(self):
        w = self.rng.multivariate_normal(np.zeros(4), self._Q).astype(np.float64)
        self.state = self._F @ self.state + w

    def _step_ct(self):
        px, py, v, theta, omega = self.state
        dt = self.dt
        w = self.rng.multivariate_normal(np.zeros(5), self._Q).astype(np.float64)
        if abs(omega) > 1e-6:
            px_n = px + v/omega*(np.sin(theta+omega*dt)-np.sin(theta))
            py_n = py + v/omega*(np.cos(theta)-np.cos(theta+omega*dt))
        else:
            px_n = px + v*np.cos(theta)*dt
            py_n = py + v*np.sin(theta)*dt
        self.state = np.array([px_n+w[0], py_n+w[1], max(1.0, v+w[2]),
                               (theta+omega*dt+w[3])%(2*np.pi), omega+w[4]],
                              dtype=np.float64)

    def _step_ca(self):
        w = self.rng.multivariate_normal(np.zeros(6), self._Q).astype(np.float64)
        self.state = self._F @ self.state + w

    def _bounce(self):
        px, py = self.state[0], self.state[1]
        if px < 0:
            self.state[0] = -px
            if self.model in ("CV", "CA"): self.state[2] = abs(self.state[2])
        elif px > self.area_w:
            self.state[0] = 2*self.area_w - px
            if self.model in ("CV", "CA"): self.state[2] = -abs(self.state[2])
        if py < 0:
            self.state[1] = -py
            if self.model in ("CV", "CA"): self.state[3] = abs(self.state[3])
        elif py > self.area_h:
            self.state[1] = 2*self.area_h - py
            if self.model in ("CV", "CA"): self.state[3] = -abs(self.state[3])

    # ── Accessors ───────────────────────────────────────────────────
    def get_position(self): return self.state[:2].copy()
    def get_velocity(self):
        if self.model == "CT":
            v, theta = self.state[2], self.state[3]
            return np.array([v*np.cos(theta), v*np.sin(theta)])
        return self.state[2:4].copy()
    def get_position_3d(self):
        return np.array([self.state[0], self.state[1], 0.0], dtype=np.float64)
    def get_state_as_target_state(self):
        v = self.get_velocity()
        return TargetState(pos=self.get_position_3d(),
                           vel=np.array([v[0], v[1], 0.0], dtype=np.float64))
    def reset(self, pos, vel):
        if self.model == "CV": self._init_cv(pos, vel)
        elif self.model == "CT": self._init_ct(pos, vel)
        elif self.model == "CA": self._init_ca(pos, vel)
