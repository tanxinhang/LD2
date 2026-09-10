"""Deterministic physical stage model for Markov movement assignments.

This module deliberately remains outside the live environment.  It turns an
assignment vector into a bounded kinematic transition and evaluates the
resulting geometry with conditional-mean channel physics plus the exact
fixed-structure max-min sensing-power LP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uav_isac.coordination.maxmin_power import (
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.hyperedge import project_pairwise_safe_movement
from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.physical.channel import expected_report_link_reliability
from uav_isac.physical.detection import compute_detection_probabilities
from uav_isac.prediction.markov_kinematics import predict_reflecting_cv_mean


@dataclass(frozen=True)
class MarkovPhysicalStageEvaluation:
    """Auditable result of one counterfactual physical stage solve."""

    cost: float
    worst_detection_probability: float
    weak_detection_probability: float
    detection_probability: np.ndarray
    deflection: np.ndarray
    power_w: np.ndarray
    target_price: np.ndarray
    feasible_geometry: bool


def cubature_sigma_points(
    mean: np.ndarray, covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-weight spherical-radial points matching two moments.

    For an ``n``-dimensional Gaussian the ``2n`` points are
    ``mean +/- sqrt(n) * L[:,i]`` with weight ``1/(2n)``.  An eigen square
    root is used so positive-semidefinite (including singular) tracking
    covariances remain valid and deterministic.
    """

    location = np.asarray(mean, dtype=np.float64).reshape(-1)
    matrix = np.asarray(covariance, dtype=np.float64)
    dimension = int(location.size)
    if dimension < 1 or matrix.shape != (dimension, dimension):
        raise ValueError("mean and covariance dimensions do not match")
    if np.any(~np.isfinite(location)) or np.any(~np.isfinite(matrix)):
        raise ValueError("mean and covariance must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]
    offsets = np.sqrt(float(dimension)) * root.T
    points = np.concatenate((location + offsets, location - offsets), axis=0)
    weights = np.full(2 * dimension, 1.0 / (2.0 * dimension))
    return points, weights


class MarkovPhysicalAssignmentModel:
    """Action-conditioned CV transition and expected physical stage cost.

    A continuous state is a ``(K+Q, 6)`` array.  UAV rows come first, target
    rows follow, and each row stores ``[x,y,z,vx,vy,vz]``.  An action contains
    one target index per UAV.  UAVs move at most ``movement_step_m`` toward
    their assigned target during a transition; target means follow the exact
    reflecting constant-velocity Markov transition.

    The discrete sensing/reporting structure is intentionally fixed during a
    plan.  Every stage recomputes expected Rician/Swerling deflection from the
    reached geometry, then solves sensing power exactly.  This separates the
    first testable hypothesis (predictive movement under a valid downstream
    solve) from the later, combinatorially larger joint structure search.
    """

    def __init__(
        self,
        deflection_computer: DeflectionComputer,
        selected_edges: Sequence[tuple[int, int, int]],
        sensing_budget_w: np.ndarray,
        roles: np.ndarray,
        fc_position: np.ndarray,
        *,
        num_targets: int,
        dt_s: float,
        movement_step_m: float,
        area_size_m: tuple[float, float],
        false_alarm_probability: float,
        role_agnostic: bool = False,
        safe_distance_m: float = 0.0,
        weak_count: int = 3,
        weak_weight: float = 0.0,
        quadrature_order: int = 12,
        target_covariance_horizon: np.ndarray | None = None,
        uncertainty_penalty_std: float = 0.0,
    ) -> None:
        self.deflection_computer = deflection_computer
        self.selected_edges = tuple(
            tuple(int(value) for value in edge) for edge in selected_edges)
        self.budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
        self.roles = np.asarray(roles, dtype=np.int64).reshape(-1)
        self.fc_position = np.asarray(fc_position, dtype=np.float64).reshape(-1)
        self.K = int(self.budget.size)
        self.Q = int(num_targets)
        self.dt_s = float(dt_s)
        self.movement_step_m = float(movement_step_m)
        self.area_size_m = tuple(float(value) for value in area_size_m)
        self.p_fa = float(false_alarm_probability)
        self.role_agnostic = bool(role_agnostic)
        self.safe_distance_m = float(safe_distance_m)
        self.weak_count = min(int(weak_count), self.Q)
        self.weak_weight = float(weak_weight)
        self.quadrature_order = int(quadrature_order)
        self.target_covariance_horizon = (
            None if target_covariance_horizon is None
            else np.asarray(target_covariance_horizon, dtype=np.float64).copy()
        )
        self.uncertainty_penalty_std = float(uncertainty_penalty_std)
        if (
            self.K < 1 or self.Q < 1 or self.roles.shape != (self.K,)
            or self.fc_position.shape != (3,)
            or np.any(~np.isfinite(self.fc_position))
        ):
            raise ValueError("budget, roles, and target count have incompatible shapes")
        if np.any(~np.isfinite(self.budget)) or np.any(self.budget < 0.0):
            raise ValueError("sensing budget must be finite and non-negative")
        if (
            not np.isfinite(self.dt_s) or self.dt_s <= 0.0
            or not np.isfinite(self.movement_step_m) or self.movement_step_m < 0.0
            or len(self.area_size_m) != 2
            or np.any(~np.isfinite(self.area_size_m))
            or np.any(np.asarray(self.area_size_m) <= 0.0)
            or not np.isfinite(self.p_fa) or not 0.0 < self.p_fa < 1.0
            or not np.isfinite(self.safe_distance_m) or self.safe_distance_m < 0.0
            or self.weak_count < 1
            or not np.isfinite(self.weak_weight) or self.weak_weight < 0.0
            or self.quadrature_order < 3
            or not np.isfinite(self.uncertainty_penalty_std)
            or self.uncertainty_penalty_std < 0.0
        ):
            raise ValueError("invalid physical assignment model parameter")
        if self.target_covariance_horizon is not None:
            covariance = self.target_covariance_horizon
            if (
                covariance.ndim != 4 or covariance.shape[0] < 1
                or covariance.shape[1:] != (self.Q, 4, 4)
                or np.any(~np.isfinite(covariance))
            ):
                raise ValueError(
                    "target_covariance_horizon must have shape (H,Q,4,4)")
            # The sigma-point constructor performs the numerical PSD check.
            for stage in range(covariance.shape[0]):
                for target in range(self.Q):
                    cubature_sigma_points(np.zeros(4), covariance[stage, target])
        # Validate completeness/unique ownership independently of geometry.
        dummy = np.ones((self.K, self.K, self.Q), dtype=np.float64)
        fixed_owner_gain_matrix(dummy, self.selected_edges)

    def pack_state(
        self,
        uav_positions: np.ndarray,
        uav_velocities: np.ndarray,
        target_positions: np.ndarray,
        target_velocities: np.ndarray,
    ) -> np.ndarray:
        """Pack physical arrays into the scenario-tree state contract."""

        positions = np.concatenate((uav_positions, target_positions), axis=0)
        velocities = np.concatenate((uav_velocities, target_velocities), axis=0)
        state = np.concatenate((positions, velocities), axis=1).astype(
            np.float64, copy=False)
        if state.shape != (self.K + self.Q, 6) or np.any(~np.isfinite(state)):
            raise ValueError("positions and velocities have invalid shapes or values")
        return state.copy()

    def unpack_state(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return UAV/target position and velocity views from a state."""

        values = np.asarray(state, dtype=np.float64)
        if values.shape != (self.K + self.Q, 6) or np.any(~np.isfinite(values)):
            raise ValueError("physical Markov state has invalid shape or values")
        return (
            values[:self.K, :3], values[:self.K, 3:],
            values[self.K:, :3], values[self.K:, 3:],
        )

    def transition(
        self, state: np.ndarray, assignment: np.ndarray, _step: int,
    ) -> np.ndarray:
        """Apply one causal target-CV/UAV-pursuit transition."""

        uav_pos, _uav_vel, target_pos, target_vel = self.unpack_state(state)
        action = np.asarray(assignment, dtype=np.int64).reshape(-1)
        if action.shape != (self.K,) or np.any(action < 0) or np.any(action >= self.Q):
            raise ValueError("assignment must contain one valid target per UAV")

        next_target_xy, next_target_vxy = predict_reflecting_cv_mean(
            target_pos[:, :2], target_vel[:, :2],
            elapsed_s=self.dt_s, area_size_m=self.area_size_m,
        )
        next_target_pos = target_pos.copy()
        next_target_vel = target_vel.copy()
        next_target_pos[:, :2] = next_target_xy
        next_target_vel[:, :2] = next_target_vxy

        direction = next_target_pos[action, :2] - uav_pos[:, :2]
        distance = np.linalg.norm(direction, axis=1)
        travel = np.minimum(distance, self.movement_step_m)
        displacement = np.zeros((self.K, 2), dtype=np.float64)
        moving = distance > 1.0e-12
        displacement[moving] = (
            direction[moving] / distance[moving, None] * travel[moving, None])
        safe_displacement = project_pairwise_safe_movement(
            uav_pos[:, :2],
            displacement,
            minimum_distance_m=self.safe_distance_m,
            maximum_step_m=self.movement_step_m,
            area_size_xy=self.area_size_m,
            independently_composable=True,
        )
        next_uav_pos = uav_pos.copy()
        next_uav_pos[:, :2] = uav_pos[:, :2] + safe_displacement
        next_uav_vel = np.zeros_like(next_uav_pos)
        next_uav_vel[:, :2] = (next_uav_pos[:, :2] - uav_pos[:, :2]) / self.dt_s
        return self.pack_state(
            next_uav_pos, next_uav_vel, next_target_pos, next_target_vel)

    def _expected_coefficient(
        self,
        uav_pos: np.ndarray,
        uav_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_covariance: np.ndarray | None,
    ) -> np.ndarray:
        """Integrate each target's per-watt tensor over its marginal belief."""

        unit_power = np.ones((self.K, self.Q), dtype=np.float64)
        if target_covariance is None:
            return self.deflection_computer.compute_expected_dense(
                uav_pos, uav_vel, target_pos, target_vel,
                self.roles, self.fc_position,
                role_agnostic=self.role_agnostic,
                sensing_power_w=unit_power,
                quadrature_order=self.quadrature_order,
            ).d_eff

        covariance = np.asarray(target_covariance, dtype=np.float64)
        if covariance.shape != (self.Q, 4, 4):
            raise ValueError("target covariance must have shape (Q,4,4)")
        coefficient = np.zeros((self.K, self.K, self.Q), dtype=np.float64)
        squared = np.zeros_like(coefficient)
        report_reliability = None
        if self.deflection_computer.use_report_link:
            if self.role_agnostic:
                receivers = np.arange(self.K)
            else:
                receivers = np.flatnonzero(self.roles == 1)
            report_reliability = np.zeros(self.K, dtype=np.float64)
            dc = self.deflection_computer
            for receiver in receivers:
                report_reliability[int(receiver)] = (
                    expected_report_link_reliability(
                        uav_pos[int(receiver)], self.fc_position,
                        dc.fc, dc.ric_K, dc.noise_power, dc.P_report,
                        use_los_prob=dc.use_los_prob,
                        los_a=dc.los_a, los_b=dc.los_b,
                        eta_los_dB=dc.eta_los_dB,
                        eta_nlos_dB=dc.eta_nlos_dB,
                        quadrature_order=self.quadrature_order,
                    ))
        for target in range(self.Q):
            mean = np.asarray([
                target_pos[target, 0], target_pos[target, 1],
                target_vel[target, 0], target_vel[target, 1],
            ])
            points, weights = cubature_sigma_points(mean, covariance[target])
            for point, weight in zip(points, weights):
                reflected_xy, reflected_velocity = predict_reflecting_cv_mean(
                    point[None, :2], point[None, 2:], elapsed_s=0.0,
                    area_size_m=self.area_size_m,
                )
                # At fixed UAV geometry each target occupies an independent
                # tensor column. Evaluate only this marginal target rather than
                # recomputing all Q columns for every one of its sigma points.
                scenario_pos = target_pos[target:target + 1].copy()
                scenario_vel = target_vel[target:target + 1].copy()
                scenario_pos[0, :2] = reflected_xy[0]
                scenario_vel[0, :2] = reflected_velocity[0]
                dense = self.deflection_computer.compute_expected_dense(
                    uav_pos, uav_vel, scenario_pos, scenario_vel,
                    self.roles, self.fc_position,
                    role_agnostic=self.role_agnostic,
                    sensing_power_w=np.ones((self.K, 1), dtype=np.float64),
                    quadrature_order=self.quadrature_order,
                    expected_report_reliability_by_rx=report_reliability,
                )
                sample = dense.d_eff[:, :, 0]
                coefficient[:, :, target] += float(weight) * sample
                squared[:, :, target] += float(weight) * sample * sample
        if self.uncertainty_penalty_std > 0.0:
            variance = np.maximum(squared - coefficient * coefficient, 0.0)
            coefficient = np.maximum(
                coefficient
                - self.uncertainty_penalty_std * np.sqrt(variance),
                0.0,
            )
        return coefficient

    def evaluate(
        self, state: np.ndarray, *, stage_index: int | None = None,
    ) -> MarkovPhysicalStageEvaluation:
        """Recompute expected physics and exact fixed-structure power."""

        uav_pos, uav_vel, target_pos, target_vel = self.unpack_state(state)
        target_distances = np.linalg.norm(
            uav_pos[:, None, :] - target_pos[None, :, :], axis=2)
        if self.K > 1:
            pairwise = np.linalg.norm(
                uav_pos[:, None, :2] - uav_pos[None, :, :2], axis=2)
            pairwise[np.diag_indices(self.K)] = np.inf
            minimum_uav_distance = float(np.min(pairwise))
        else:
            minimum_uav_distance = float("inf")
        feasible = bool(
            np.min(target_distances) >= self.safe_distance_m - 1.0e-12
            and minimum_uav_distance >= self.safe_distance_m - 1.0e-12
        )
        if not feasible:
            zeros_q = np.zeros(self.Q, dtype=np.float64)
            return MarkovPhysicalStageEvaluation(
                cost=0.0,
                worst_detection_probability=0.0,
                weak_detection_probability=0.0,
                detection_probability=zeros_q,
                deflection=zeros_q.copy(),
                power_w=np.zeros((self.K, self.Q), dtype=np.float64),
                target_price=np.full(
                    self.Q, 1.0 / self.Q, dtype=np.float64),
                feasible_geometry=False,
            )

        target_covariance = None
        if self.target_covariance_horizon is not None:
            if stage_index is None:
                raise ValueError(
                    "stage_index is required with a covariance horizon")
            stage = int(stage_index)
            if stage != stage_index or not 0 <= stage < self.target_covariance_horizon.shape[0]:
                raise ValueError("stage_index is outside the covariance horizon")
            target_covariance = self.target_covariance_horizon[stage]
        coefficient = self._expected_coefficient(
            uav_pos, uav_vel, target_pos, target_vel, target_covariance)
        gain, _owners = fixed_owner_gain_matrix(
            coefficient, self.selected_edges)
        power = solve_fixed_structure_maxmin_power_lp(gain, self.budget)
        probability = compute_detection_probabilities(power.deflection, self.p_fa)
        ordered = np.sort(probability)
        worst = float(ordered[0])
        weak = float(np.mean(ordered[:self.weak_count]))
        return MarkovPhysicalStageEvaluation(
            cost=float(-(worst + self.weak_weight * weak)),
            worst_detection_probability=worst,
            weak_detection_probability=weak,
            detection_probability=probability,
            deflection=power.deflection.copy(),
            power_w=power.power_w.copy(),
            target_price=power.prices.copy(),
            feasible_geometry=True,
        )

    def stage_cost(
        self, state: np.ndarray, _assignment: np.ndarray, step: int,
    ) -> float:
        """Scenario-tree callback returning the minimization cost."""

        return self.evaluate(state, stage_index=step).cost
