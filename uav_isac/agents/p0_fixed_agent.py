"""P0-Fixed baseline: fixed trajectory, only inner P0 solver runs.

This baseline keeps UAVs on fixed trajectories (circular or grid pattern)
and only uses the inner P0 solver to select bistatic reporting assignments.
It represents the lower bound of "no trajectory optimization".
"""

import numpy as np
from typing import Dict, Optional, Tuple
from uav_isac.agents.base_agent import BaseAgent
from uav_isac.environment.action import ActionSpace
from uav_isac.utils.types import Action


class P0FixedAgent(BaseAgent):
    """Fixed trajectory + inner P0 only.

    UAVs follow pre-defined circular trajectories around the area center.
    Roles are fixed: half tx, half rx.
    """

    def __init__(
        self,
        agent_id: int,
        K: int,
        center: Optional[np.ndarray] = None,
        radius: float = 300.0,
        angular_speed: float = 0.05,  # rad/frame
        phase_offset: Optional[float] = None,
        role: Optional[int] = None,
        action_space: Optional[ActionSpace] = None,
        position_scale: float = 1000.0,  # obs denormalization factor
    ):
        """
        Args:
            agent_id: UAV identifier
            K: Total number of UAVs
            center: Center of circular trajectory [x, y]; defaults to region center
            radius: Radius of circle (m)
            angular_speed: Angular speed (rad/frame)
            phase_offset: Phase offset for this UAV on the circle
            role: Fixed role (0=tx, 1=rx, 2=idle); if None, auto-assign
            action_space: ActionSpace for max_dp clamping
            position_scale: Factor to denormalize obs position (region_size)
        """
        super().__init__(agent_id)
        self.K = K
        self.center = center if center is not None else np.array([500.0, 500.0])
        self.radius = radius
        self.angular_speed = angular_speed
        self.position_scale = position_scale

        self.max_dp = action_space.max_dp if action_space is not None else 2.5

        if phase_offset is not None:
            self.phase = phase_offset
        else:
            self.phase = 2 * np.pi * agent_id / K

        if role is not None:
            self.role = role
        else:
            # First half tx, second half rx
            self.role = 0 if agent_id < K // 2 else 1

        self.frame = 0

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[Action, float, float]:
        """Compute fixed trajectory action.

        Uses true tangential motion along the circle: δp = p(θ_{t+1}) - p(θ_t).
        The effective angular speed is capped so that the arc distance per frame
        never exceeds max_dp.

        Returns:
            (Action, log_prob=0, value=0)
        """
        # Effective angular speed: respect max displacement constraint
        effective_omega = min(self.angular_speed, self.max_dp / max(self.radius, 1e-6))

        # Current and next positions on the circle
        theta_now = self.phase + effective_omega * self.frame
        theta_next = self.phase + effective_omega * (self.frame + 1)

        p_now = self.center + self.radius * np.array([np.cos(theta_now), np.sin(theta_now)])
        p_next = self.center + self.radius * np.array([np.cos(theta_next), np.sin(theta_next)])

        delta_p = p_next - p_now

        # Not needed (guaranteed by effective_omega), but keep as safety clamp
        norm = np.linalg.norm(delta_p)
        if norm > self.max_dp:
            delta_p = delta_p * (self.max_dp / norm)

        self.frame += 1
        return Action(delta_p=delta_p, role=self.role), 0.0, 0.0

    def update(self, rollout_data: Dict) -> Dict[str, float]:
        """No learning — baseline."""
        return {}

    def reset(self) -> None:
        self.frame = 0
