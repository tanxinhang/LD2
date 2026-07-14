"""Constraint checking and penalty computation.

Enforces:
  1. Safety: pairwise UAV distance >= d_safe
  2. Energy: battery >= 0
  3. Detection fairness: P_D^q >= P_D_min per target
  4. Area boundary: UAV within region
"""

import numpy as np
from typing import List, Dict, Tuple


class ConstraintChecker:
    """Checks constraints and computes penalty terms."""

    def __init__(
        self,
        d_safe: float = 20.0,           # m safe separation
        P_D_min: float = 0.8,            # minimum detection probability
        area_size: tuple = (1000.0, 1000.0),  # m
        safety_penalty_weight: float = 1.0,
        energy_penalty_weight: float = 10.0,
        fairness_penalty_weight: float = 5.0,
        boundary_penalty_weight: float = 0.1,
    ):
        """
        Args:
            d_safe: Safe inter-UAV distance (m)
            P_D_min: Minimum detection probability per target
            area_size: Region bounds (width, height) in meters
            safety_penalty_weight: Weight for safety violation penalty
            energy_penalty_weight: Weight for energy depletion penalty
            fairness_penalty_weight: Weight for detection fairness penalty
            boundary_penalty_weight: Weight for boundary violation penalty
        """
        self.d_safe = d_safe
        self.P_D_min = P_D_min
        self.area_w, self.area_h = area_size
        self.w_safety = safety_penalty_weight
        self.w_energy = energy_penalty_weight
        self.w_fairness = fairness_penalty_weight
        self.w_boundary = boundary_penalty_weight

    def check_safety(
        self, uav_positions: np.ndarray  # (K, 3)
    ) -> Tuple[float, int]:
        """Check pairwise safety distances.

        Penalty: sum_{i<j} max(0, d_safe - d_ij)^2

        Args:
            uav_positions: (K, 3) UAV positions

        Returns:
            (total_penalty, num_violations)
        """
        K = uav_positions.shape[0]
        penalty = 0.0
        violations = 0

        for i in range(K):
            for j in range(i + 1, K):
                d_ij = np.linalg.norm(uav_positions[i] - uav_positions[j])
                if d_ij < self.d_safe:
                    violation = self.d_safe - d_ij
                    penalty += violation ** 2
                    violations += 1

        return float(self.w_safety * penalty), violations

    def check_energy(
        self, batteries: np.ndarray  # (K,)
    ) -> Tuple[float, int]:
        """Check energy depletion.

        Penalty: sum_k max(0, -battery_k)

        Args:
            batteries: (K,) remaining battery energy per UAV

        Returns:
            (total_penalty, num_depleted)
        """
        depleted = np.sum(batteries <= 0)
        penalty = np.sum(np.maximum(0.0, -batteries))
        return float(self.w_energy * penalty), int(depleted)

    def check_fairness(
        self, P_D_q: np.ndarray  # (Q,)
    ) -> Tuple[float, int]:
        """Check detection fairness constraint.

        Penalty: sum_q max(0, P_D_min - P_D^q)^2

        Args:
            P_D_q: (Q,) detection probabilities per target

        Returns:
            (total_penalty, num_violations)
        """
        shortfall = np.maximum(0.0, self.P_D_min - P_D_q)
        violations = int(np.sum(shortfall > 0))
        penalty = float(np.sum(shortfall ** 2))
        return float(self.w_fairness * penalty), violations

    def check_boundary(
        self, uav_positions: np.ndarray  # (K, 3)
    ) -> Tuple[float, int]:
        """Check area boundary constraints.

        Penalty: sum_k [max(0, -x_k)^2 + max(0, x_k - area_w)^2 + ...]

        Args:
            uav_positions: (K, 3) UAV positions

        Returns:
            (total_penalty, num_violations)
        """
        K = uav_positions.shape[0]
        penalty = 0.0
        violations = 0

        for k in range(K):
            x, y = uav_positions[k, 0], uav_positions[k, 1]
            if x < 0:
                penalty += x ** 2
                violations += 1
            elif x > self.area_w:
                penalty += (x - self.area_w) ** 2
                violations += 1
            if y < 0:
                penalty += y ** 2
                violations += 1
            elif y > self.area_h:
                penalty += (y - self.area_h) ** 2
                violations += 1

        return float(self.w_boundary * penalty), violations

    def check_all(
        self,
        uav_positions: np.ndarray,   # (K, 3)
        batteries: np.ndarray,       # (K,)
        P_D_q: np.ndarray,           # (Q,)
    ) -> Dict:
        """Check all constraints and return summary.

        Args:
            uav_positions: (K, 3) UAV positions
            batteries: (K,) UAV battery levels
            P_D_q: (Q,) per-target detection probabilities

        Returns:
            Dict with keys: total_penalty, safety_penalty, energy_penalty,
            fairness_penalty, boundary_penalty, any_violation
        """
        safety_p, safety_v = self.check_safety(uav_positions)
        energy_p, energy_v = self.check_energy(batteries)
        fairness_p, fairness_v = self.check_fairness(P_D_q)
        boundary_p, boundary_v = self.check_boundary(uav_positions)

        total = safety_p + energy_p + fairness_p + boundary_p
        any_violation = (safety_v + energy_v + fairness_v + boundary_v) > 0

        return {
            'total_penalty': total,
            'safety_penalty': safety_p,
            'energy_penalty': energy_p,
            'fairness_penalty': fairness_p,
            'boundary_penalty': boundary_p,
            'safety_violations': safety_v,
            'energy_violations': energy_v,
            'fairness_violations': fairness_v,
            'boundary_violations': boundary_v,
            'any_violation': any_violation,
        }
