"""One-Step Myopic baseline: greedy per-frame action + inner P0.

At each frame, the agent enumerates candidate outer actions (delta_p, role),
evaluates each by running the inner P0 solver, and picks the one that
maximizes immediate single-frame utility. This is "short-sighted" — it
doesn't account for future frames.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from uav_isac.agents.base_agent import BaseAgent
from uav_isac.utils.types import Action
from uav_isac.environment.action import ActionSpace


class OneStepMyopicAgent(BaseAgent):
    """One-step greedy: enumerate candidate actions, pick argmax of immediate utility.

    Evaluates each candidate by:
    1. Simulating the effect of the candidate action on this UAV's position/role
    2. Recomputing deflection entries (when deflection_computer is available)
    3. Running the inner P0 solver
    4. Computing immediate team utility

    When deflection_computer is not available, falls back to a geometry-based
    heuristic that prefers moving toward targets and avoiding idle roles.
    """

    def __init__(
        self,
        agent_id: int,
        action_space: ActionSpace,
        num_dp_candidates: int = 8,     # number of direction candidates
        num_speed_candidates: int = 3,   # number of speed levels
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            agent_id: UAV identifier
            action_space: Action space definition
            num_dp_candidates: Number of direction candidates (uniform angles)
            num_speed_candidates: Number of speed magnitude candidates
            rng: Random generator
        """
        super().__init__(agent_id)
        self.action_space = action_space
        self.num_dp_candidates = num_dp_candidates
        self.num_speed_candidates = num_speed_candidates
        self.rng = rng if rng is not None else np.random.default_rng()

        # Pre-compute candidate action set
        self._build_candidates()

    def _build_candidates(self):
        """Build discrete candidate action set."""
        self.candidates = []

        # Direction angles
        angles = np.linspace(0, 2 * np.pi, self.num_dp_candidates, endpoint=False)

        # Speed levels
        speeds = np.linspace(0, self.action_space.max_dp, self.num_speed_candidates)

        for angle in angles:
            for speed in speeds:
                delta_p = np.array([speed * np.cos(angle), speed * np.sin(angle)])
                for role in range(3):  # tx, rx, idle
                    self.candidates.append(Action(delta_p=delta_p, role=role))

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        *,
        deflection_entries: Optional[list] = None,
        inner_solver: Optional[Any] = None,
        uav_positions: Optional[np.ndarray] = None,
        target_positions: Optional[np.ndarray] = None,
        deflection_computer: Optional[Any] = None,
        uav_velocities: Optional[np.ndarray] = None,
        target_velocities: Optional[np.ndarray] = None,
        current_roles: Optional[np.ndarray] = None,
        fc_position: Optional[np.ndarray] = None,
    ) -> Tuple[Action, float, float]:
        """Select action greedily maximizing immediate utility.

        Args:
            obs: Local observation
            deterministic: If True, use only the first candidate (fast path)
            deflection_entries: Current DeflectionEntry list (from env)
            inner_solver: InnerSolver instance for P0 optimization
            uav_positions: (K, 3) current UAV positions
            target_positions: (Q, 3) current target positions
            deflection_computer: DeflectionComputer instance for recomputing entries
            uav_velocities: (K, 3) current UAV velocities
            target_velocities: (Q, 3) current target velocities
            current_roles: (K,) current role assignments
            fc_position: (3,) fusion center position

        Returns:
            (Action, log_prob=0, value=0)
        """
        # ── Always evaluate candidates; _evaluate_candidate has its own fallback ──
        if not self.candidates:
            return self.action_space.sample(), 0.0, 0.0

        best_action = self.candidates[0]
        best_utility = -np.inf

        # Evaluate ALL candidates
        for candidate in self.candidates:
            utility = self._evaluate_candidate(
                candidate,
                deflection_entries=deflection_entries,
                inner_solver=inner_solver,
                uav_positions=uav_positions,
                target_positions=target_positions,
                deflection_computer=deflection_computer,
                uav_velocities=uav_velocities,
                target_velocities=target_velocities,
                current_roles=current_roles,
                fc_position=fc_position,
                obs=obs,
            )
            if utility > best_utility:
                best_utility = utility
                best_action = candidate

        return best_action, 0.0, 0.0

    def _evaluate_candidate(
        self,
        action: Action,
        deflection_entries: Optional[list],
        inner_solver: Optional[Any],
        uav_positions: Optional[np.ndarray] = None,
        target_positions: Optional[np.ndarray] = None,
        deflection_computer: Optional[Any] = None,
        uav_velocities: Optional[np.ndarray] = None,
        target_velocities: Optional[np.ndarray] = None,
        current_roles: Optional[np.ndarray] = None,
        fc_position: Optional[np.ndarray] = None,
        obs: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluate a candidate action by simulating its effect and running P0.

        Two modes:
        - Full: when deflection_computer + all state info is available.
          Recomputes deflection entries for the candidate, runs P0, returns utility.
        - Heuristic: when deflection_computer is unavailable.
          Uses geometry-based approximation that still differentiates candidates.
        """
        # ── Full evaluation ──
        if (deflection_computer is not None
                and inner_solver is not None
                and uav_positions is not None
                and target_positions is not None
                and uav_velocities is not None
                and target_velocities is not None
                and current_roles is not None
                and fc_position is not None):
            # Simulate new position, velocity, and role for this agent
            new_positions = uav_positions.copy()
            new_positions[self.agent_id, :2] += action.delta_p
            new_velocities = uav_velocities.copy()
            new_velocities[self.agent_id, :2] = action.delta_p / self.action_space.dt
            new_roles = current_roles.copy()
            new_roles[self.agent_id] = action.role

            # Recompute deflection entries
            try:
                new_entries = deflection_computer.compute(
                    uav_positions=new_positions,
                    uav_velocities=new_velocities,
                    target_positions=target_positions,
                    target_velocities=target_velocities,
                    roles=new_roles,
                    fc_position=fc_position,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Myopic candidate evaluation failed for agent {self.agent_id}: {exc}"
                ) from exc

            # Run inner P0
            solution = inner_solver.solve(
                new_entries,
                K=len(new_roles),
                Q=len(target_positions),
            )

            # Compute immediate utility
            utility = (
                solution['detection_utility']
                - 0.1 * solution.get('communication_cost', 0.0)
                - 10.0 * solution.get('constraint_violation', 0.0)
            )
            return float(utility)

        # ── Heuristic fallback (differentiates candidates) ──
        utility = 0.0

        if uav_positions is not None and target_positions is not None:
            # Prefer moving closer to targets
            new_pos = uav_positions[self.agent_id, :2].copy() + action.delta_p
            dists = np.linalg.norm(
                target_positions[:, :2] - new_pos, axis=1
            )
            utility -= 0.01 * np.min(dists)  # closer = better

        # Role: idle is less useful in most configurations
        if action.role == 2:  # idle
            utility -= 1.0
        elif action.role == 0:  # tx — slight bonus (more sensing)
            utility += 0.1

        # Slight preference for movement over standing still
        speed = np.linalg.norm(action.delta_p)
        utility += 0.001 * speed

        # Add d_eff from current entries as a baseline (same for all candidates
        # with same role, but gives a reasonable scale for the utility)
        if deflection_entries is not None:
            for e in deflection_entries:
                if e.i == self.agent_id or e.j == self.agent_id:
                    utility += 0.001 * e.d_eff

        return utility

    def update(self, rollout_data: Dict) -> Dict[str, float]:
        """No learning — baseline."""
        return {}

    def reset(self) -> None:
        pass
