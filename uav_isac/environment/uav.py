"""UAV agent: kinematics, energy model, and state management."""

import numpy as np
from uav_isac.utils.types import UAVState


class UAV:
    """Single UAV agent with kinematics and energy dynamics.

    Role convention:
      0 = tx (transmitter — emits ISAC waveform)
      1 = rx (receiver — listens for bistatic returns)
      2 = idle (silent)
    """

    def __init__(
        self,
        uav_id: int,
        initial_pos: np.ndarray,      # (3,) [x, y, z]
        v_max: float = 25.0,           # m/s
        d_safe: float = 20.0,          # m
        B_max: float = 50000.0,        # J initial battery
        P_sense: float = 0.5,          # W sensing tx power
        P_report: float = 0.1,         # W reporting tx power
        P_fly_static: float = 80.0,    # W hover power
        P_fly_coeff: float = 0.05,     # speed-dependent coefficient
        dt: float = 0.1,               # s frame duration
        area_size: tuple = (1000.0, 1000.0),  # m
        height: float = 100.0,          # m fixed altitude
    ):
        """
        Args:
            uav_id: Unique UAV identifier
            initial_pos: Initial position [x, y, z] (m)
            v_max: Maximum flight speed (m/s)
            d_safe: Safe separation distance (m)
            B_max: Maximum battery energy (J)
            P_sense: Sensing transmit power (W)
            P_report: Reporting transmit power (W)
            P_fly_static: Static flight power (hover, W)
            P_fly_coeff: Speed-proportional flight power coefficient
            dt: Frame duration (s)
            area_size: Region bounds (width, height) in meters
            height: Fixed flight altitude (m)
        """
        self.id = uav_id
        self.v_max = v_max
        self.d_safe = d_safe
        self.B_max = B_max
        self.P_sense = P_sense
        self.P_report = P_report
        self.P_fly_static = P_fly_static
        self.P_fly_coeff = P_fly_coeff
        self.dt = dt
        self.area_w, self.area_h = area_size
        self.height = height

        # State
        self.pos = initial_pos.astype(np.float64).copy()
        self.vel = np.zeros(3, dtype=np.float64)
        self.battery = float(B_max)
        self.role = 2  # idle by default

        # Ensure fixed altitude
        self.pos[2] = height

    def apply_action(
        self,
        delta_p: np.ndarray,  # (2,) position increment [dx, dy]
        role: int,             # 0=tx, 1=rx, 2=idle
        account_radio_energy: bool = True,
    ) -> None:
        """Apply one frame's action: move and set role.

        Args:
            delta_p: Position increment [dx, dy] (m), clamped to v_max*dt
            role: New role assignment
        """
        # Clamp position increment to max speed
        dp_norm = np.linalg.norm(delta_p)
        max_dp = self.v_max * self.dt
        if dp_norm > max_dp:
            delta_p = delta_p * (max_dp / dp_norm)

        # Update the endpoint and reflect both position and outgoing velocity
        # at the same physical wall.  Previously the endpoint bounced but the
        # stored velocity retained its incoming sign, so the published state
        # was not a trajectory that the kinematic model could have executed.
        next_xy = self.pos[:2] + delta_p
        reflected = np.zeros(2, dtype=bool)
        # z is fixed.
        if next_xy[0] < 0:
            next_xy[0] = -next_xy[0]
            reflected[0] = True
        elif next_xy[0] > self.area_w:
            next_xy[0] = 2 * self.area_w - next_xy[0]
            reflected[0] = True

        if next_xy[1] < 0:
            next_xy[1] = -next_xy[1]
            reflected[1] = True
        elif next_xy[1] > self.area_h:
            next_xy[1] = 2 * self.area_h - next_xy[1]
            reflected[1] = True
        self.pos[:2] = next_xy
        if self.dt > 0:
            outgoing = np.asarray(delta_p, dtype=np.float64) / self.dt
            outgoing[reflected] *= -1.0
            self.vel = np.array([outgoing[0], outgoing[1], 0.0])

        # Set role
        self.role = int(role)

        # Compute and deduct energy
        flight_energy = self._compute_flight_energy(delta_p)
        self.battery -= flight_energy

        if account_radio_energy:
            if self.role == 0:  # tx
                self.battery -= self.P_sense * self.dt
            elif self.role == 1:  # rx
                self.battery -= self.P_report * self.dt
        # idle: no extra energy beyond flight

        # Floor battery at 0
        if self.battery < 0.0:
            self.battery = 0.0

    def _compute_flight_energy(self, delta_p: np.ndarray) -> float:
        """Compute flight energy for this frame.

        E_fly = (P_static + P_coeff * v^2) * dt
        where v = |delta_p| / dt

        Args:
            delta_p: Position increment [dx, dy]

        Returns:
            Energy consumed (J)
        """
        speed = np.linalg.norm(delta_p) / max(self.dt, 1e-10)
        power = self.P_fly_static + self.P_fly_coeff * (speed ** 2)
        return float(power * self.dt)

    def get_position(self) -> np.ndarray:
        """Get current position [x, y, z]."""
        return self.pos.copy()

    def get_state(self) -> UAVState:
        """Get current state as UAVState namedtuple."""
        return UAVState(
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            battery=self.battery,
            role=self.role,
        )

    def is_alive(self) -> bool:
        """Check if UAV still has battery energy."""
        return self.battery > 0.0

    def reset(self, pos: np.ndarray) -> None:
        """Reset UAV to initial position with full battery.

        Args:
            pos: [x, y, z] initial position
        """
        self.pos = pos.astype(np.float64).copy()
        self.pos[2] = self.height
        self.vel = np.zeros(3, dtype=np.float64)
        self.battery = float(self.B_max)
        self.role = 2  # idle
