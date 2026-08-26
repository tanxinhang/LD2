"""Certified finite-horizon slow-geometry repair for static-target UAV ISAC.

The inverse-range radar law supplies an analytic proposal gradient.  The
gradient never certifies execution: every finite trust-region candidate is
ranked on wire-quantized local statistics and only Top-M is re-evaluated with
the frozen causal coefficient envelope, exact kinematics,
DD support, flight energy, boundary and pairwise-separation constraints, and
an explicit six-stage transport/commit certificate.
"""

# ----------------------------------------------------------------------
# AUDIT/RESEARCH-ONLY MODULE (2026-08-16 audit remediation)
#
# This module is consumed only by tools/ audit scripts and tests. It is
# NOT part of the deployment execution path (env_core / trainer) and its
# results must not be described as deployed behaviour. It exists to keep
# a specific research question reproducible; see
# docs/ARCHITECTURE_V2_RESULTS.md for the associated gate.
# ----------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from uav_isac.coordination.bottleneck_router import RepairRoute
from uav_isac.coordination.geometry_repair_transport import (
    GeometryRepairTransportCertificate,
    GeometryRepairWireLayout,
    certify_geometry_repair_transport,
    quantize_displacement_toward_zero,
)
from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix
from uav_isac.coordination.joint_structure_relaxation import (
    certified_hierarchical_qos_route,
)
from uav_isac.coordination.owner_local_physics import (
    OwnerLocalHorizonCoefficientBounds,
    OwnerLocalKinematicState,
    OwnerTargetInvariantCache,
    advance_owner_local_kinematics,
    owner_local_horizon_coefficient_bounds,
    propagate_owner_local_horizon,
)
from uav_isac.coordination.target_invariant_transport import (
    TargetInvariantWireLayout,
)
from uav_isac.coordination.owner_proposal_transport import (
    quantize_nonnegative_float16_lower,
    quantize_nonnegative_float16_upper,
)
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)
from uav_isac.utils.math_utils import Q_inverse


@dataclass(frozen=True)
class CertifiedGeometryRepairConfig:
    trust_region_fractions: tuple[float, ...] = (1.0, 0.5, 0.25)
    weak_target_count: int = 3
    verification_top_m: int = 4
    learned_verification_top_m: int = 0
    certificate_horizon_steps: int = 3
    minimum_worst_pd_improvement: float = 1.0e-6
    probability_tolerance: float = 1.0e-9
    boundary_margin_m: float = 0.0
    separation_margin_m: float = 0.0
    snr_margin_db: float = 3.0
    latency_margin_s: float = 5.0e-4
    require_pre_reserved_comm_power: bool = True
    target_pair_limit: int = 3
    reports_per_receiver: int = 18

    def __post_init__(self) -> None:
        fractions = np.asarray(self.trust_region_fractions, dtype=np.float64)
        if (
            fractions.ndim != 1 or fractions.size < 1
            or np.any(~np.isfinite(fractions))
            or np.any(fractions <= 0.0) or np.any(fractions > 1.0)
        ):
            raise ValueError("trust-region fractions must lie in (0,1]")
        if int(self.weak_target_count) < 1:
            raise ValueError("weak_target_count must be positive")
        if not 0 <= int(self.verification_top_m) <= 8:
            raise ValueError("verification_top_m must lie in [0,8]")
        if not 0 <= int(self.learned_verification_top_m) <= 8:
            raise ValueError("learned_verification_top_m must lie in [0,8]")
        if (
            int(self.verification_top_m)
            + int(self.learned_verification_top_m) > 8
        ):
            raise ValueError("total geometry verification count exceeds eight")
        if (
            int(self.verification_top_m)
            + int(self.learned_verification_top_m) < 1
        ):
            raise ValueError("at least one geometry candidate must be verified")
        if not 3 <= int(self.certificate_horizon_steps) <= 8:
            raise ValueError("certificate_horizon_steps must lie in [3,8]")
        if int(self.target_pair_limit) < 1 or int(self.reports_per_receiver) < 1:
            raise ValueError("geometry opportunity limits must be positive")
        for name in (
            "minimum_worst_pd_improvement", "probability_tolerance",
            "boundary_margin_m", "separation_margin_m", "snr_margin_db",
            "latency_margin_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class GeometryCandidateCertificate:
    joint_plan_digest: str
    mover: int
    movers: tuple[int, ...]
    baseline_displacement_m: np.ndarray
    baseline_movement_plan_m: np.ndarray
    displacement_m: np.ndarray
    movement_plan_m: np.ndarray
    sensing_power_plan_w: np.ndarray
    communication_power_plan_w: np.ndarray
    proposal_source: str
    proposal_score: float
    active_weak_target: int
    certificate_horizon_steps: int
    next_state: OwnerLocalKinematicState
    baseline_physical_bounds: OwnerLocalHorizonCoefficientBounds
    physical_bounds: OwnerLocalHorizonCoefficientBounds
    candidate_lower_pd: np.ndarray
    candidate_upper_pd: np.ndarray
    wire_baseline_lower_pd: np.ndarray
    wire_baseline_upper_pd: np.ndarray
    wire_candidate_lower_pd: np.ndarray
    coupled_pd_gain: np.ndarray
    coupled_window_pd_gain: np.ndarray
    coupled_window_worst_pd_gain: float
    coupled_first_step_worst_pd_gain: float
    minimum_first_step_worst_pd_gain: float
    coupled_target_safety_margin: np.ndarray
    possible_worst_target: np.ndarray
    flight_energy_j: float
    escrow_flight_energy_plan_j: np.ndarray
    incremental_flight_energy_j: float
    minimum_separation_m: float
    transport: GeometryRepairTransportCertificate


@dataclass(frozen=True)
class CertifiedGeometryRepairDecision:
    accepted: bool
    reason: str
    displacement_m: np.ndarray
    baseline_lower_pd: np.ndarray
    baseline_upper_pd: np.ndarray
    candidate_lower_pd: np.ndarray
    candidate_upper_pd: np.ndarray
    candidate_count: int
    verified_candidate_count: int
    feasible_candidate_count: int
    target_no_harm_failure_count: int
    window_worst_failure_count: int
    fast_opportunity_failure_count: int
    strict_improvement_failure_count: int
    transport_failure_count: int
    energy_failure_count: int
    best_verified_first_step_coupled_gain: float
    best_verified_minimum_coupled_gain: float
    best_verified_window_coupled_gain: float
    best_verified_window_worst_pd_gain: float
    best_candidate: GeometryCandidateCertificate | None

    @property
    def certified_worst_pd_improvement(self) -> float:
        if self.best_candidate is not None:
            return float(np.min(
                self.best_candidate.coupled_target_safety_margin[0]))
        candidate = np.asarray(self.candidate_lower_pd, dtype=np.float64)
        baseline = np.asarray(self.baseline_upper_pd, dtype=np.float64)
        if candidate.ndim == 1:
            candidate = candidate[None, :]
            baseline = baseline[None, :]
        return float(
            np.min(candidate[0]) - np.min(baseline[0]))


def _certified_candidate_rank(
    item: GeometryCandidateCertificate,
) -> tuple[object, ...]:
    """Order only by proved utility, then deterministic cost tie-breakers."""
    return (
        float(item.coupled_window_worst_pd_gain),
        float(item.coupled_first_step_worst_pd_gain),
        float(np.min(item.coupled_window_pd_gain)),
        -float(item.incremental_flight_energy_j),
        int(item.proposal_source == "analytic_maxmin_gradient"),
        float(item.proposal_score),
        -len(item.movers),
        tuple(-mover for mover in item.movers),
    )


def _possible_worst_targets(
    baseline_lower_pd: np.ndarray,
    baseline_upper_pd: np.ndarray,
    probability_tolerance: float,
) -> np.ndarray:
    """Return the frame-wise active set that can attain the global minimum."""
    lower = np.asarray(baseline_lower_pd, dtype=np.float64)
    upper = np.asarray(baseline_upper_pd, dtype=np.float64)
    tolerance = float(probability_tolerance)
    if (
        lower.ndim != 2 or upper.shape != lower.shape
        or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
        or np.any(lower < 0.0) or np.any(upper < lower)
        or not np.isfinite(tolerance) or tolerance < 0.0
    ):
        raise ValueError("worst-target active-set bounds are invalid")
    possible = lower <= np.min(upper, axis=1, keepdims=True) + tolerance
    if not np.all(np.any(possible, axis=1)):
        raise AssertionError("every frame must have a possible worst target")
    return possible


def _normalized_geometry_deflection(
    states: tuple[OwnerLocalKinematicState, ...],
    bounds: OwnerLocalHorizonCoefficientBounds,
    selected: np.ndarray,
    sensing_power_w: np.ndarray,
) -> np.ndarray:
    """Return target Deflection per unit common target invariant."""
    active = np.asarray(selected, dtype=bool)
    sensing = np.asarray(sensing_power_w, dtype=np.float64)
    H, K, K2, Q = bounds.lower.shape
    if (
        K != K2 or len(states) != H or active.shape != (K, K, Q)
        or sensing.shape not in ((K, Q), (H, K, Q))
    ):
        raise ValueError("normalized geometry dimensions disagree")
    result = np.zeros((H, Q), dtype=np.float64)
    for step, state in enumerate(states):
        position = np.asarray(state.uav_position_m, dtype=np.float64)
        target = np.asarray(state.target_mean_by_owner, dtype=np.float64)
        for transmitter, receiver, target_id in np.argwhere(active):
            if not bounds.dd_certified_support[
                step, transmitter, receiver, target_id
            ]:
                continue
            target_position = np.asarray([
                target[receiver, target_id, 0],
                target[receiver, target_id, 1],
                0.0,
            ], dtype=np.float64)
            tx_range_squared = max(float(np.sum(
                (position[transmitter] - target_position) ** 2)), 1.0e-12)
            rx_range_squared = max(float(np.sum(
                (position[receiver] - target_position) ** 2)), 1.0e-12)
            result[step, target_id] += (
                sensing[
                    transmitter, target_id
                ] if sensing.ndim == 2 else sensing[
                    step, transmitter, target_id
                ]
                / (tx_range_squared * rx_range_squared)
            )
    return result


def _minimum_common_scale_pd_gain(
    baseline_factor: float,
    candidate_factor: float,
    scale_lower: float,
    scale_upper: float,
    p_fa: float,
) -> float:
    """Minimize the coupled detector gain over one common positive scale.

    With ``P_D(D)=Q(z-sqrt(D))``, stationary points of
    ``P_D(c x)-P_D(b x)`` satisfy a quadratic in ``sqrt(x)``.  Endpoints and
    its positive roots therefore give the exact minimum on a closed interval.
    """
    baseline = float(baseline_factor)
    candidate = float(candidate_factor)
    lower = float(scale_lower)
    upper = float(scale_upper)
    if (
        not np.isfinite(baseline) or not np.isfinite(candidate)
        or baseline < 0.0 or candidate < 0.0
        or not np.isfinite(lower) or not np.isfinite(upper)
        or lower <= 0.0 or upper < lower
    ):
        raise ValueError("common-scale detector interval is invalid")
    points = [lower, upper]
    if candidate > 0.0 and baseline > 0.0 and candidate != baseline:
        difference = candidate - baseline
        root_difference = np.sqrt(candidate) - np.sqrt(baseline)
        z_value = float(Q_inverse(np.asarray(float(p_fa))))
        discriminant = (
            z_value ** 2 * root_difference ** 2
            + difference * np.log(candidate / baseline)
        )
        if discriminant >= 0.0:
            root = np.sqrt(discriminant)
            for numerator in (
                z_value * root_difference + root,
                z_value * root_difference - root,
            ):
                y_value = numerator / difference
                x_value = y_value ** 2
                if y_value > 0.0 and lower < x_value < upper:
                    points.append(float(x_value))
    gains = []
    for scale in points:
        probabilities = compute_detection_probabilities(
            np.asarray([
                baseline * scale,
                candidate * scale,
            ], dtype=np.float64),
            float(p_fa),
        )
        gains.append(float(probabilities[1] - probabilities[0]))
    return float(min(gains))


def _minimum_common_scale_target_safety_margin(
    baseline_factor: float,
    candidate_factor: float,
    scale_lower: float,
    scale_upper: float,
    p_fa: float,
    qos_floor: float,
) -> float:
    """Minimize ``P_c(x)-min(P_b(x),QoS)`` over one common scale."""
    baseline = float(baseline_factor)
    candidate = float(candidate_factor)
    lower = float(scale_lower)
    upper = float(scale_upper)
    qos = float(qos_floor)
    if not 0.0 < qos < 1.0:
        raise ValueError("QoS floor must lie in (0,1)")
    margins = []
    threshold_deflection = float(
        minimum_deflection_for_detection_probability(
            np.asarray([qos], dtype=np.float64), float(p_fa))[0])
    crossing = (
        float("inf") if baseline <= 0.0
        else threshold_deflection / baseline)
    gain_upper = min(upper, crossing)
    if lower <= gain_upper:
        margins.append(_minimum_common_scale_pd_gain(
            baseline, candidate, lower, gain_upper, p_fa))
    qos_lower = max(lower, crossing)
    if qos_lower <= upper:
        candidate_pd = float(compute_detection_probabilities(
            np.asarray([candidate * qos_lower], dtype=np.float64),
            float(p_fa),
        )[0])
        margins.append(candidate_pd - qos)
    return float(min(margins, default=float("-inf")))


def _minimum_common_scale_clipped_pd_gain(
    baseline_factor: float,
    candidate_factor: float,
    scale_lower: float,
    scale_upper: float,
    p_fa: float,
    qos_floor: float,
) -> float:
    """Minimize QoS-clipped detection gain over one common target scale."""
    baseline = float(baseline_factor)
    candidate = float(candidate_factor)
    lower = float(scale_lower)
    upper = float(scale_upper)
    qos = float(qos_floor)
    if (
        not np.isfinite(baseline) or not np.isfinite(candidate)
        or baseline < 0.0 or candidate < 0.0
        or not np.isfinite(lower) or not np.isfinite(upper)
        or lower <= 0.0 or upper < lower
        or not 0.0 < qos < 1.0
    ):
        raise ValueError("common-scale clipped detector interval is invalid")
    points = [lower, upper]
    threshold = float(minimum_deflection_for_detection_probability(
        np.asarray([qos], dtype=np.float64), float(p_fa))[0])
    for factor in (baseline, candidate):
        crossing = float("inf") if factor <= 0.0 else threshold / factor
        if lower < crossing < upper:
            points.append(float(crossing))
    if candidate > 0.0 and baseline > 0.0 and candidate != baseline:
        difference = candidate - baseline
        root_difference = np.sqrt(candidate) - np.sqrt(baseline)
        z_value = float(Q_inverse(np.asarray(float(p_fa))))
        discriminant = (
            z_value ** 2 * root_difference ** 2
            + difference * np.log(candidate / baseline)
        )
        if discriminant >= 0.0:
            root = np.sqrt(discriminant)
            for numerator in (
                z_value * root_difference + root,
                z_value * root_difference - root,
            ):
                y_value = numerator / difference
                scale = y_value ** 2
                if y_value > 0.0 and lower < scale < upper:
                    points.append(float(scale))
    gains = []
    for scale in points:
        probabilities = compute_detection_probabilities(
            np.asarray([
                baseline * scale,
                candidate * scale,
            ], dtype=np.float64),
            float(p_fa),
        )
        gains.append(float(
            min(probabilities[1], qos) - min(probabilities[0], qos)))
    return float(min(gains))


def _pd(
    coefficient: np.ndarray,
    selected: np.ndarray,
    sensing_power_w: np.ndarray,
    p_fa: float,
) -> np.ndarray:
    gain, _ = fixed_owner_gain_matrix(
        coefficient, [tuple(edge) for edge in np.argwhere(selected)])
    return compute_detection_probabilities(
        np.sum(gain * sensing_power_w, axis=0), float(p_fa))


def inverse_range_deflection_gradient(
    coefficient_per_watt: np.ndarray,
    selected: np.ndarray,
    sensing_power_w: np.ndarray,
    state: OwnerLocalKinematicState,
    target_owner: np.ndarray,
) -> np.ndarray:
    """Return ``d D_q / d p_k`` under the bistatic inverse-range law.

    For ``a_ijq = kappa_q/(R_iq^2 R_jq^2)``, differentiation gives
    ``grad_{p_i} a = -2 a (p_i-y_q)/R_iq^2`` and the analogous receiver term.
    DD admission is held fixed only for proposal ranking; exact DD bounds are
    recomputed before execution.
    """
    coefficient = np.asarray(coefficient_per_watt, dtype=np.float64)
    active = np.asarray(selected, dtype=bool)
    sensing = np.asarray(sensing_power_w, dtype=np.float64)
    position = np.asarray(state.uav_position_m, dtype=np.float64)
    target = np.asarray(state.target_mean_by_owner, dtype=np.float64)
    owner = np.asarray(target_owner, dtype=np.int64).reshape(-1)
    if coefficient.ndim != 3 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = coefficient.shape
    if (
        active.shape != coefficient.shape or sensing.shape != (K, Q)
        or position.shape != (K, 3) or target.shape != (K, Q, 4)
        or owner.shape != (Q,) or np.any(owner < 0) or np.any(owner >= K)
        or np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0)
        or np.any(~np.isfinite(sensing)) or np.any(sensing < 0.0)
    ):
        raise ValueError("gradient inputs are outside physical support")
    gradient = np.zeros((K, Q, 2), dtype=np.float64)
    for transmitter, receiver, target_id in np.argwhere(active):
        power = float(sensing[transmitter, target_id])
        edge_coefficient = float(coefficient[
            transmitter, receiver, target_id])
        if power <= 0.0 or edge_coefficient <= 0.0:
            continue
        target_position = np.asarray([
            target[owner[target_id], target_id, 0],
            target[owner[target_id], target_id, 1],
            0.0,
        ], dtype=np.float64)
        for endpoint in (int(transmitter), int(receiver)):
            offset = position[endpoint] - target_position
            range_squared = max(float(np.dot(offset, offset)), 1.0e-12)
            gradient[endpoint, target_id] += (
                -2.0 * power * edge_coefficient
                * offset[:2] / range_squared
            )
    return gradient


def _minimum_swept_separation(
    positions: np.ndarray,
    displacement_m: np.ndarray,
) -> float:
    """Minimum 3-D separation over the complete linear movement segment."""
    pos = np.asarray(positions, dtype=np.float64)
    displacement = np.asarray(displacement_m, dtype=np.float64)
    if displacement.shape != (pos.shape[0], 2):
        raise ValueError("swept displacement must have shape (K,2)")
    delta = np.concatenate((
        displacement, np.zeros((pos.shape[0], 1), dtype=np.float64),
    ), axis=1)
    distances = []
    for first in range(pos.shape[0]):
        for second in range(first + 1, pos.shape[0]):
            relative = pos[first] - pos[second]
            relative_delta = delta[first] - delta[second]
            denominator = float(np.dot(relative_delta, relative_delta))
            time = (
                0.0 if denominator <= 1.0e-18
                else float(np.clip(
                    -np.dot(relative, relative_delta) / denominator,
                    0.0, 1.0,
                ))
            )
            distances.append(float(np.linalg.norm(
                relative + time * relative_delta)))
    return min(distances, default=float("inf"))


def _candidate_move_groups(selected: np.ndarray) -> tuple[tuple[int, ...], ...]:
    active = np.asarray(selected, dtype=bool)
    K = active.shape[0]
    groups = {(agent,) for agent in range(K)}
    groups.update(
        tuple(sorted((int(transmitter), int(receiver))))
        for transmitter, receiver, _target in np.argwhere(active)
    )
    active_agents = tuple(sorted(set(
        int(agent)
        for transmitter, receiver, _target in np.argwhere(active)
        for agent in (transmitter, receiver)
    )))
    if len(active_agents) > 2:
        groups.add(active_agents)
    return tuple(sorted(groups, key=lambda item: (len(item), item)))


def _maximum_three_step_return_scale(
    baseline_plan: np.ndarray,
    movers: tuple[int, ...],
    direction: np.ndarray,
    maximum_displacement_m: float,
) -> float:
    """Largest common first-step correction with feasible H=3 return.

    Terminal position and velocity require corrections ``(+alpha*d,-alpha*d,0)``.
    Each of the first two actions must remain in its Euclidean speed disk.  The
    positive root of the resulting scalar quadratic is exact for each disk;
    their minimum is the joint feasible step.
    """
    baseline = np.asarray(baseline_plan, dtype=np.float64)
    values = np.asarray(direction, dtype=np.float64)
    maximum = float(maximum_displacement_m)
    if (
        baseline.ndim != 3 or baseline.shape[0] != 3
        or baseline.shape[2] != 2
        or values.shape != (len(movers), 2)
        or not np.isfinite(maximum) or maximum <= 0.0
        or np.any(~np.isfinite(baseline)) or np.any(~np.isfinite(values))
    ):
        raise ValueError("three-step return-scale inputs are invalid")

    upper = maximum
    for mover_index, mover in enumerate(movers):
        correction_direction = values[mover_index]
        quadratic = float(np.dot(
            correction_direction, correction_direction))
        if quadratic <= 1.0e-18:
            continue
        for step, sign in ((0, 1.0), (1, -1.0)):
            action = baseline[step, mover]
            signed_direction = sign * correction_direction
            linear = float(np.dot(action, signed_direction))
            constant = float(np.dot(action, action) - maximum ** 2)
            discriminant = linear ** 2 - quadratic * constant
            if discriminant < -1.0e-10:
                return 0.0
            root = (-linear + np.sqrt(max(discriminant, 0.0))) / quadratic
            upper = min(upper, max(float(root), 0.0))
    return float(upper)


def _joint_box_maxmin_direction(
    gradient_by_target: np.ndarray,
) -> np.ndarray | None:
    """Find a common multi-UAV ascent direction by a conservative LP.

    Each planar UAV correction is restricted to the inscribed box
    ``[-1/sqrt(2),1/sqrt(2)]^2``, which is contained in its unit speed disk.
    Maximizing the minimum target directional derivative is therefore a
    finite-dimensional linear program with no hidden norm approximation in
    the subsequent execution certificate.
    """
    gradient = np.asarray(gradient_by_target, dtype=np.float64)
    if gradient.ndim != 3 or gradient.shape[2] != 2:
        raise ValueError("joint gradient must have shape (R,M,2)")
    if np.any(~np.isfinite(gradient)):
        raise ValueError("joint gradient must be finite")
    targets, movers, _ = gradient.shape
    if targets < 1 or movers < 1:
        return None
    flattened = gradient.reshape(targets, 2 * movers)
    objective = np.zeros(2 * movers + 1, dtype=np.float64)
    objective[-1] = -1.0
    constraints = np.concatenate((
        -flattened,
        np.ones((targets, 1), dtype=np.float64),
    ), axis=1)
    bound = 1.0 / np.sqrt(2.0)
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=np.zeros(targets, dtype=np.float64),
        bounds=[(-bound, bound)] * (2 * movers) + [(None, None)],
        method="highs",
    )
    if not result.success or float(result.x[-1]) <= 1.0e-18:
        return None
    direction = np.asarray(result.x[:-1], dtype=np.float64).reshape(movers, 2)
    maximum_norm = float(np.max(np.linalg.norm(direction, axis=1)))
    if maximum_norm <= 1.0e-18:
        return None
    direction /= maximum_norm
    if float(np.min(flattened @ direction.reshape(-1))) <= 1.0e-18:
        return None
    return direction


def _joint_proposal_directions(
    gradient: np.ndarray,
    movers: tuple[int, ...],
    weak_target_order: np.ndarray,
    weak_target_count: int,
) -> tuple[np.ndarray, ...]:
    """Generate coordinated per-prefix and per-target correction fields."""
    values = np.asarray(gradient, dtype=np.float64)
    order = np.asarray(weak_target_order, dtype=np.int64).reshape(-1)
    count = min(max(1, int(weak_target_count)), order.size)
    mover_index = np.asarray(movers, dtype=np.int64)
    directions: list[np.ndarray] = []
    for prefix_count in range(1, count + 1):
        prefix = np.transpose(
            values[mover_index][:, order[:prefix_count], :],
            (1, 0, 2),
        )
        joint = _joint_box_maxmin_direction(prefix)
        if joint is not None:
            directions.append(joint)
        target_gradient = prefix[-1]
        maximum_norm = float(np.max(np.linalg.norm(
            target_gradient, axis=1)))
        if maximum_norm > 1.0e-18:
            directions.append(target_gradient / maximum_norm)
    unique: list[np.ndarray] = []
    for direction in directions:
        if not any(np.linalg.norm(direction - prior) <= 1.0e-10
                   for prior in unique):
            unique.append(direction)
    return tuple(unique)


def _optimized_recovery_plan(
    baseline_plan: np.ndarray,
    first_displacement: np.ndarray,
    movers: tuple[int, ...],
    gradient_horizon: np.ndarray,
    possible_worst_target: np.ndarray,
    maximum_displacement_m: float,
) -> np.ndarray | None:
    """Close position/velocity while maximizing linearized recovery safety."""
    baseline = np.asarray(baseline_plan, dtype=np.float64)
    first = np.asarray(first_displacement, dtype=np.float64)
    gradient = np.asarray(gradient_horizon, dtype=np.float64)
    possible = np.asarray(possible_worst_target, dtype=bool).reshape(-1)
    H, K, _ = baseline.shape
    M = len(movers)
    if (
        H < 3 or first.shape != (K, 2)
        or gradient.shape != (H, K, possible.size, 2)
        or M < 1 or not np.any(possible)
    ):
        raise ValueError("recovery-plan dimensions are inconsistent")
    recovery_steps = H - 2
    correction = first[np.asarray(movers)] - baseline[0, np.asarray(movers)]
    if recovery_steps == 1:
        result = baseline.copy()
        result[0] = first
        result[1, np.asarray(movers)] -= correction
        return result

    variable_count = recovery_steps * M * 2 + 1
    t_index = variable_count - 1
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[t_index] = -1.0

    def variable(step: int, mover_index: int, axis: int) -> int:
        return (step * M + mover_index) * 2 + axis

    rows: list[np.ndarray] = []
    bounds: list[float] = []
    for state_step in range(1, H - 1):
        for target_id in np.flatnonzero(possible):
            row = np.zeros(variable_count, dtype=np.float64)
            constant = 0.0
            for mover_index, mover in enumerate(movers):
                target_gradient = gradient[
                    state_step, mover, target_id]
                constant += float(np.dot(
                    target_gradient, correction[mover_index]))
                for recovery_step in range(state_step):
                    for axis in range(2):
                        row[variable(
                            recovery_step, mover_index, axis)] -= (
                                target_gradient[axis])
            row[t_index] = 1.0
            rows.append(row)
            bounds.append(constant)

    equality_rows: list[np.ndarray] = []
    equality_values: list[float] = []
    for mover_index in range(M):
        for axis in range(2):
            row = np.zeros(variable_count, dtype=np.float64)
            for recovery_step in range(recovery_steps):
                row[variable(recovery_step, mover_index, axis)] = 1.0
            equality_rows.append(row)
            equality_values.append(float(-correction[mover_index, axis]))

    maximum = float(maximum_displacement_m)
    variable_bounds: list[tuple[float | None, float | None]] = []
    for recovery_step in range(recovery_steps):
        plan_step = recovery_step + 1
        for mover in movers:
            for axis in range(2):
                baseline_component = float(baseline[plan_step, mover, axis])
                variable_bounds.append((
                    -maximum - baseline_component,
                    maximum - baseline_component,
                ))
    variable_bounds.append((None, None))
    result = linprog(
        objective,
        A_ub=np.stack(rows) if rows else None,
        b_ub=np.asarray(bounds) if bounds else None,
        A_eq=np.stack(equality_rows),
        b_eq=np.asarray(equality_values),
        bounds=variable_bounds,
        method="highs",
    )
    if not result.success:
        return None
    plan = baseline.copy()
    plan[0] = first
    for recovery_step in range(recovery_steps):
        for mover_index, mover in enumerate(movers):
            plan[recovery_step + 1, mover] += np.asarray([
                result.x[variable(recovery_step, mover_index, axis)]
                for axis in range(2)
            ])
    return plan


def _maxmin_bundle_direction(gradients: np.ndarray) -> np.ndarray | None:
    """Solve the planar unit-ball max--min linearized direction exactly.

    On the unit circle, the lower envelope of finitely many linear forms can
    attain a positive maximum only at an individual form's stationary
    direction or where two forms intersect.  Enumerating normalized gradients
    and both normals of every pairwise difference therefore covers the exact
    two-dimensional solution; zero is retained when no common ascent exists.
    """
    vector = np.asarray(gradients, dtype=np.float64)
    if vector.ndim != 2 or vector.shape[1] != 2:
        raise ValueError("gradient bundle must have shape (R,2)")
    if np.any(~np.isfinite(vector)):
        raise ValueError("gradient bundle must be finite")
    directions: list[np.ndarray] = []
    for gradient in vector:
        norm = float(np.linalg.norm(gradient))
        if norm > 1.0e-18:
            directions.append(gradient / norm)
    for first in range(vector.shape[0]):
        for second in range(first + 1, vector.shape[0]):
            difference = vector[first] - vector[second]
            normal = np.asarray([-difference[1], difference[0]])
            norm = float(np.linalg.norm(normal))
            if norm > 1.0e-18:
                directions.extend((normal / norm, -normal / norm))
    if not directions:
        return None
    best = max(
        directions,
        key=lambda direction: float(np.min(vector @ direction)),
    )
    if float(np.min(vector @ best)) <= 1.0e-18:
        return None
    return best


def _proposal_directions(
    mover_gradient: np.ndarray,
    weak_target_order: np.ndarray,
    weak_target_count: int,
) -> tuple[np.ndarray, ...]:
    """Generate per-target and exact prefix-bundle ascent directions."""
    gradient = np.asarray(mover_gradient, dtype=np.float64)
    order = np.asarray(weak_target_order, dtype=np.int64).reshape(-1)
    count = min(max(1, int(weak_target_count)), order.size)
    directions: list[np.ndarray] = []
    for prefix_count in range(1, count + 1):
        prefix = gradient[order[:prefix_count]]
        bundle = _maxmin_bundle_direction(prefix)
        if bundle is not None:
            directions.append(bundle)
        target_gradient = gradient[order[prefix_count - 1]]
        norm = float(np.linalg.norm(target_gradient))
        if norm > 1.0e-18:
            directions.append(target_gradient / norm)
    unique: list[np.ndarray] = []
    for direction in directions:
        if not any(np.linalg.norm(direction - prior) <= 1.0e-10
                   for prior in unique):
            unique.append(direction)
    return tuple(unique)


def _future_bounds(
    future_states: tuple[OwnerLocalKinematicState, ...],
    cache: OwnerTargetInvariantCache,
    support: np.ndarray,
    *,
    invariant_layout: TargetInvariantWireLayout,
    carrier_hz: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    covariance_radius: float,
    dd_support_threshold: float,
    dd_additive_margin: float,
    residual_log_margin: float,
) -> OwnerLocalHorizonCoefficientBounds:
    invariant_lower = np.asarray(
        cache.target_invariant, dtype=np.float64).reshape(-1)
    invariant_upper = np.asarray([
        invariant_layout.invariant_cell_upper(value)
        for value in invariant_lower
    ], dtype=np.float64)
    return owner_local_horizon_coefficient_bounds(
        future_states,
        invariant_lower,
        invariant_upper,
        support,
        carrier_hz=float(carrier_hz),
        delta_f_hz=float(delta_f_hz),
        symbol_period_s=float(symbol_period_s),
        delay_bins=int(delay_bins),
        doppler_bins=int(doppler_bins),
        covariance_radius=float(covariance_radius),
        dd_support_threshold=float(dd_support_threshold),
        dd_additive_margin=float(dd_additive_margin),
        residual_log_margin=np.full(
            len(future_states), float(residual_log_margin),
            dtype=np.float64),
        uav_position_radius_m=0.0,
        uav_velocity_radius_mps=0.0,
    )


def certified_trust_region_geometry_repair(
    current_lower_coefficient: np.ndarray,
    current_upper_coefficient: np.ndarray,
    current_state: OwnerLocalKinematicState,
    target_invariant_cache: OwnerTargetInvariantCache,
    support: np.ndarray,
    selected: np.ndarray,
    target_owner: np.ndarray,
    sensing_power_w: np.ndarray,
    communication_power_w: np.ndarray,
    battery_j: np.ndarray,
    *,
    communication_model: InterUAVCommunicationModel,
    control_period_s: float,
    p_fa: float,
    qos_floor: float,
    dt_s: float,
    max_speed_mps: float,
    area_size_m: tuple[float, float],
    safe_separation_m: float,
    static_flight_power_w: float,
    quadratic_flight_power_coeff: float,
    carrier_hz: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    covariance_radius: float,
    dd_support_threshold: float,
    dd_additive_margin: float,
    residual_log_margin: float,
    ground_communication_enabled: bool = False,
    config: CertifiedGeometryRepairConfig | None = None,
    invariant_layout: TargetInvariantWireLayout | None = None,
    action_origin_state: OwnerLocalKinematicState | None = None,
    baseline_displacement_m: np.ndarray | None = None,
    baseline_movement_plan_m: np.ndarray | None = None,
    baseline_sensing_plan_w: np.ndarray | None = None,
    baseline_communication_plan_w: np.ndarray | None = None,
    proposal_movement_plan_m: np.ndarray | None = None,
    joint_plan_digest: str = "",
) -> CertifiedGeometryRepairDecision:
    """Propose by gradient, then accept only by a complete future tube.

    The movement and RF plans are jointly announced for the future horizon.
    Geometry may alter only movement; acceptance binds the unchanged RF plan,
    terminal Actor state, physical bounds and finite-window sensing certificate.
    """
    cfg = config or CertifiedGeometryRepairConfig()
    digest = str(joint_plan_digest)
    if digest and (
        len(digest) != 16
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("joint plan digest must be 64-bit lowercase hex")
    lower = np.asarray(current_lower_coefficient, dtype=np.float64)
    upper = np.asarray(current_upper_coefficient, dtype=np.float64)
    active = np.asarray(selected, dtype=bool)
    provenance = np.asarray(support, dtype=bool)
    sensing = np.asarray(sensing_power_w, dtype=np.float64)
    comm = np.asarray(communication_power_w, dtype=np.float64).reshape(-1)
    battery = np.asarray(battery_j, dtype=np.float64).reshape(-1)
    origin_state = (
        current_state if action_origin_state is None else action_origin_state)
    positions = np.asarray(origin_state.uav_position_m, dtype=np.float64)
    if lower.ndim != 3 or lower.shape[0] != lower.shape[1]:
        raise ValueError("coefficient envelope must have shape (K,K,Q)")
    K, _, Q = lower.shape
    if (
        upper.shape != lower.shape or active.shape != lower.shape
        or provenance.shape != lower.shape or sensing.shape != (K, Q)
        or comm.shape != (K,) or battery.shape != (K,)
        or positions.shape != (K, 3) or np.any(active & ~provenance)
        or np.any(~np.isfinite(lower)) or np.any(lower < 0.0)
        or np.any(~np.isfinite(upper)) or np.any(upper < lower)
        or np.any(~np.isfinite(sensing)) or np.any(sensing < 0.0)
        or np.any(~np.isfinite(comm)) or np.any(comm < 0.0)
        or np.any(comm >= 1.0) or np.any(~np.isfinite(battery))
        or np.any(battery < 0.0)
        or np.any(comm + np.sum(sensing, axis=1) > 1.0 + 1.0e-9)
    ):
        raise ValueError("geometry repair state is outside support")
    if baseline_movement_plan_m is None:
        baseline_plan = np.zeros(
            (int(cfg.certificate_horizon_steps), K, 2), dtype=np.float64)
        if baseline_displacement_m is not None:
            baseline_plan[0] = np.asarray(
                baseline_displacement_m, dtype=np.float64)
    else:
        baseline_plan = np.asarray(
            baseline_movement_plan_m, dtype=np.float64)
    baseline_displacement = baseline_plan[0]
    if (
        baseline_plan.shape != (
            int(cfg.certificate_horizon_steps), K, 2)
        or np.any(~np.isfinite(baseline_plan))
    ):
        raise ValueError("baseline movement plan must have shape (H,K,2)")
    learned_proposal_plan = None
    if proposal_movement_plan_m is not None:
        learned_proposal_plan = np.asarray(
            proposal_movement_plan_m, dtype=np.float64)
        if (
            learned_proposal_plan.shape != baseline_plan.shape
            or np.any(~np.isfinite(learned_proposal_plan))
        ):
            raise ValueError("proposal movement plan must have shape (H,K,2)")
    if np.any(target_invariant_cache.target_invariant <= 0.0):
        raise ValueError("complete target invariants are required")
    if bool(ground_communication_enabled):
        raise ValueError(
            "geometry repair is uncalibrated for position-dependent "
            "ground-report reliability")
    if not bool(cfg.require_pre_reserved_comm_power):
        raise ValueError("only pre-reserved geometry communication is supported")
    if float(control_period_s) <= 0.0 or float(dt_s) <= 0.0:
        raise ValueError("control and dynamics periods must be positive")
    if float(max_speed_mps) <= 0.0 or float(safe_separation_m) < 0.0:
        raise ValueError("speed and separation limits are invalid")
    if (
        float(static_flight_power_w) < 0.0
        or float(quadratic_flight_power_coeff) < 0.0
    ):
        raise ValueError("flight power coefficients must be non-negative")

    horizon = int(cfg.certificate_horizon_steps)
    sensing_plan = (
        np.repeat(sensing[None, ...], horizon, axis=0)
        if baseline_sensing_plan_w is None
        else np.asarray(baseline_sensing_plan_w, dtype=np.float64)
    )
    communication_plan = (
        np.repeat(comm[None, ...], horizon, axis=0)
        if baseline_communication_plan_w is None
        else np.asarray(baseline_communication_plan_w, dtype=np.float64)
    )
    if (
        sensing_plan.shape != (horizon, K, Q)
        or communication_plan.shape != (horizon, K)
        or np.any(~np.isfinite(sensing_plan))
        or np.any(sensing_plan < 0.0)
        or np.any(~np.isfinite(communication_plan))
        or np.any(communication_plan < 0.0)
        or np.any(communication_plan >= 1.0)
        or np.any(
            communication_plan + np.sum(sensing_plan, axis=2)
            > 1.0 + 1.0e-9)
    ):
        raise ValueError("baseline RF plan must have shape (H,K,Q)/(H,K)")
    wire_comm = np.asarray([
        quantize_nonnegative_float16_lower(value)
        for value in comm
    ], dtype=np.float64)
    wire_sensing_lower = np.asarray([
        quantize_nonnegative_float16_lower(value)
        for value in sensing_plan.reshape(-1)
    ], dtype=np.float64).reshape(horizon, K, Q)
    wire_sensing_upper = np.asarray([
        quantize_nonnegative_float16_upper(value)
        for value in sensing_plan.reshape(-1)
    ], dtype=np.float64).reshape(horizon, K, Q)
    wire_communication_plan = np.asarray([
        quantize_nonnegative_float16_lower(value)
        for value in communication_plan.reshape(-1)
    ], dtype=np.float64).reshape(horizon, K)
    maximum_displacement = float(max_speed_mps) * float(dt_s)
    baseline_norm = np.linalg.norm(baseline_displacement, axis=1)
    if np.any(baseline_norm > maximum_displacement + 1.0e-9):
        raise ValueError("baseline mobility action exceeds the speed disk")
    baseline_states = propagate_owner_local_horizon(
        origin_state,
        baseline_plan,
        dt_s=float(dt_s),
        max_speed_mps=float(max_speed_mps),
        area_size_m=area_size_m,
        advance_targets=False,
    )[1:]
    layout = invariant_layout or TargetInvariantWireLayout(K, Q)
    baseline_bounds = _future_bounds(
        baseline_states,
        target_invariant_cache,
        provenance,
        invariant_layout=layout,
        carrier_hz=carrier_hz,
        delta_f_hz=delta_f_hz,
        symbol_period_s=symbol_period_s,
        delay_bins=delay_bins,
        doppler_bins=doppler_bins,
        covariance_radius=covariance_radius,
        dd_support_threshold=dd_support_threshold,
        dd_additive_margin=dd_additive_margin,
        residual_log_margin=residual_log_margin,
    )
    baseline_lower_pd = np.stack([
        _pd(item, active, wire_sensing_lower[step], p_fa)
        for step, item in enumerate(baseline_bounds.lower)
    ])
    baseline_upper_pd = np.stack([
        _pd(item, active, wire_sensing_upper[step], p_fa)
        for step, item in enumerate(baseline_bounds.upper)
    ])
    wire_baseline_lower_pd = np.asarray([
        quantize_nonnegative_float16_lower(value)
        for value in baseline_lower_pd.reshape(-1)
    ], dtype=np.float64).reshape(horizon, Q)
    wire_baseline_upper_pd = np.asarray([
        quantize_nonnegative_float16_upper(value)
        for value in baseline_upper_pd.reshape(-1)
    ], dtype=np.float64).reshape(horizon, Q)
    fast_opportunity_count = 0
    selected_edges = [tuple(edge) for edge in np.argwhere(active)]
    for step in range(horizon):
        opportunity = certified_hierarchical_qos_route(
            baseline_bounds.upper[step],
            selected_edges,
            1.0 - wire_communication_plan[step],
            current_lower_pd=float(np.min(wire_baseline_lower_pd[step])),
            p_fa=float(p_fa),
            qos_floor=float(qos_floor),
            target_pair_limit=int(cfg.target_pair_limit),
            reports_per_receiver=int(cfg.reports_per_receiver),
        )
        fast_opportunity_count += int(
            opportunity.decision.route != RepairRoute.GEOMETRY)
    if fast_opportunity_count:
        return CertifiedGeometryRepairDecision(
            accepted=False,
            reason="baseline_fast_layer_opportunity",
            displacement_m=np.zeros((K, 2), dtype=np.float64),
            baseline_lower_pd=wire_baseline_lower_pd.copy(),
            baseline_upper_pd=wire_baseline_upper_pd.copy(),
            candidate_lower_pd=wire_baseline_lower_pd.copy(),
            candidate_upper_pd=wire_baseline_upper_pd.copy(),
            candidate_count=0,
            verified_candidate_count=0,
            feasible_candidate_count=0,
            target_no_harm_failure_count=0,
            window_worst_failure_count=0,
            fast_opportunity_failure_count=fast_opportunity_count,
            strict_improvement_failure_count=0,
            transport_failure_count=0,
            energy_failure_count=0,
            best_verified_first_step_coupled_gain=float("-inf"),
            best_verified_minimum_coupled_gain=float("-inf"),
            best_verified_window_coupled_gain=float("-inf"),
            best_verified_window_worst_pd_gain=float("-inf"),
            best_candidate=None,
        )
    gradient = inverse_range_deflection_gradient(
        baseline_bounds.lower[0],
        active,
        wire_sensing_lower[0],
        baseline_states[0],
        target_owner,
    )
    wire_gradient = np.asarray(gradient, dtype=np.float32).astype(np.float64)
    if np.any(~np.isfinite(wire_gradient)):
        raise ValueError("geometry gradient exceeds binary32 wire support")
    weak_target_order = np.argsort(
        np.min(wire_baseline_lower_pd, axis=0), kind="stable")
    max_displacement = maximum_displacement
    wire = GeometryRepairWireLayout(K, Q)
    ranked_proposals: list[
        tuple[
            float, tuple[int, ...], np.ndarray, float, np.ndarray, str
        ]
    ] = []
    candidate_count = 0
    width, height = float(area_size_m[0]), float(area_size_m[1])
    boundary_margin = float(cfg.boundary_margin_m)
    protected = np.minimum(wire_baseline_upper_pd, float(qos_floor))
    common_target_scale = bool(
        float(covariance_radius) == 0.0
        and np.all(np.asarray(
            origin_state.target_cov_diag_by_owner,
            dtype=np.float64,
        ) == 0.0)
    )
    invariant_lower = np.asarray(
        target_invariant_cache.target_invariant,
        dtype=np.float64,
    ).reshape(Q)
    invariant_upper = np.asarray([
        layout.invariant_cell_upper(value) for value in invariant_lower
    ], dtype=np.float64)
    baseline_normalized = _normalized_geometry_deflection(
        baseline_states,
        baseline_bounds,
        active,
        wire_sensing_upper,
    )
    proposal_possible_worst = (
        wire_baseline_lower_pd[0]
        <= np.min(wire_baseline_upper_pd[0])
        + float(cfg.probability_tolerance)
    )
    baseline_gradient_horizon = np.stack([
        inverse_range_deflection_gradient(
            baseline_bounds.lower[step],
            active,
            wire_sensing_lower[step],
            baseline_states[step],
            target_owner,
        )
        for step in range(horizon)
    ]).astype(np.float32).astype(np.float64)
    for movers in _candidate_move_groups(active):
        joint_directions = _joint_proposal_directions(
            wire_gradient,
            movers,
            weak_target_order,
            int(cfg.weak_target_count),
        )
        for joint_direction in joint_directions:
            terminal_return_scale = max_displacement
            if horizon == 3:
                terminal_return_scale = _maximum_three_step_return_scale(
                    baseline_plan,
                    movers,
                    joint_direction,
                    max_displacement,
                )
            if terminal_return_scale <= 1.0e-12:
                continue
            for fraction in cfg.trust_region_fractions:
                candidate_count += 1
                displacement = baseline_displacement.copy()
                for mover_index, mover in enumerate(movers):
                    requested_correction = (
                        terminal_return_scale * float(fraction)
                        * joint_direction[mover_index]
                    )
                    correction = quantize_displacement_toward_zero(
                        requested_correction,
                        maximum_component_m=max_displacement,
                        bits_per_axis=wire.displacement_bits_per_axis,
                    )
                    displacement[mover] += correction
                if np.any(np.linalg.norm(
                    displacement, axis=1) > max_displacement + 1.0e-12):
                    continue
                raw_next_xy = positions[:, :2].copy()
                raw_next_xy += displacement
                inside = bool(
                    np.all(raw_next_xy[:, 0] >= boundary_margin)
                    and np.all(
                        raw_next_xy[:, 0] <= width - boundary_margin)
                    and np.all(raw_next_xy[:, 1] >= boundary_margin)
                    and np.all(
                        raw_next_xy[:, 1] <= height - boundary_margin)
                )
                exact_minimum_separation = _minimum_swept_separation(
                    positions, displacement)
                minimum_separation = quantize_nonnegative_float16_lower(
                    exact_minimum_separation)
                collision_safe = bool(
                    minimum_separation + 1.0e-12
                    >= float(safe_separation_m)
                    + float(cfg.separation_margin_m)
                )
                speed = np.linalg.norm(displacement, axis=1) / float(dt_s)
                flight_energy = (
                    float(static_flight_power_w)
                    + float(quadratic_flight_power_coeff) * speed ** 2
                ) * float(dt_s)
                hover_energy = float(static_flight_power_w) * float(dt_s)
                if not inside or not collision_safe:
                    continue
                proposal_score = float(np.min(
                    np.sum([
                        wire_gradient[mover, weak_target_order[
                            :min(int(cfg.weak_target_count), Q)]]
                        @ (
                            displacement[mover]
                            - baseline_displacement[mover])
                        for mover in movers
                    ], axis=0)
                ))
                ranked_proposals.append((
                    proposal_score,
                    movers,
                    displacement.copy(),
                    minimum_separation,
                    flight_energy.copy(),
                    "analytic_maxmin_gradient",
                ))
    ranked_proposals.sort(key=lambda item: (
        -float(item[0]),
        -len(item[1]),
        item[1],
        tuple(float(value) for value in item[2].reshape(-1)),
    ))
    verified_proposals = ranked_proposals[:int(cfg.verification_top_m)]
    learned_ranked_proposals: list[
        tuple[
            float, tuple[int, ...], np.ndarray, float, np.ndarray, str
        ]
    ] = []
    if (
        learned_proposal_plan is not None
        and int(cfg.learned_verification_top_m) > 0
    ):
        cumulative_deviation = np.cumsum(
            learned_proposal_plan - baseline_plan, axis=0)
        learned_direction = cumulative_deviation[
            int(np.argmax(np.linalg.norm(
                cumulative_deviation, axis=2).max(axis=1)))
        ]
        for movers in _candidate_move_groups(active):
            direction = learned_direction[
                np.asarray(movers, dtype=np.int64)].copy()
            maximum_norm = float(np.max(np.linalg.norm(direction, axis=1)))
            if maximum_norm <= 1.0e-12:
                continue
            direction /= maximum_norm
            terminal_return_scale = max_displacement
            if horizon == 3:
                terminal_return_scale = _maximum_three_step_return_scale(
                    baseline_plan, movers, direction, max_displacement)
            if terminal_return_scale <= 1.0e-12:
                continue
            for fraction in cfg.trust_region_fractions:
                candidate_count += 1
                displacement = baseline_displacement.copy()
                for mover_index, mover in enumerate(movers):
                    correction = quantize_displacement_toward_zero(
                        terminal_return_scale * float(fraction)
                        * direction[mover_index],
                        maximum_component_m=max_displacement,
                        bits_per_axis=wire.displacement_bits_per_axis,
                    )
                    displacement[mover] += correction
                if np.any(np.linalg.norm(
                    displacement, axis=1) > max_displacement + 1.0e-12):
                    continue
                raw_next_xy = positions[:, :2] + displacement
                inside = bool(
                    np.all(raw_next_xy[:, 0] >= boundary_margin)
                    and np.all(raw_next_xy[:, 0] <= width - boundary_margin)
                    and np.all(raw_next_xy[:, 1] >= boundary_margin)
                    and np.all(raw_next_xy[:, 1] <= height - boundary_margin)
                )
                minimum_separation = quantize_nonnegative_float16_lower(
                    _minimum_swept_separation(positions, displacement))
                collision_safe = bool(
                    minimum_separation + 1.0e-12
                    >= float(safe_separation_m)
                    + float(cfg.separation_margin_m))
                if not inside or not collision_safe:
                    continue
                speed = np.linalg.norm(displacement, axis=1) / float(dt_s)
                flight_energy = (
                    float(static_flight_power_w)
                    + float(quadratic_flight_power_coeff) * speed ** 2
                ) * float(dt_s)
                proposal_score = float(np.min(np.sum([
                    wire_gradient[mover, weak_target_order[
                        :min(int(cfg.weak_target_count), Q)]]
                    @ (displacement[mover] - baseline_displacement[mover])
                    for mover in movers
                ], axis=0)))
                learned_ranked_proposals.append((
                    proposal_score,
                    movers,
                    displacement.copy(),
                    minimum_separation,
                    flight_energy.copy(),
                    "learned_equivariant_intent",
                ))
        learned_ranked_proposals.sort(key=lambda item: (
            -float(item[0]),
            -len(item[1]),
            item[1],
            tuple(float(value) for value in item[2].reshape(-1)),
        ))
        existing = {
            np.ascontiguousarray(item[2], dtype="<f8").tobytes()
            for item in verified_proposals
        }
        for proposal in learned_ranked_proposals:
            key = np.ascontiguousarray(
                proposal[2], dtype="<f8").tobytes()
            if key in existing:
                continue
            verified_proposals.append(proposal)
            existing.add(key)
            if (
                len(verified_proposals)
                >= int(cfg.verification_top_m)
                + int(cfg.learned_verification_top_m)
            ):
                break
    candidates: list[GeometryCandidateCertificate] = []
    target_no_harm_failure_count = 0
    window_worst_failure_count = 0
    strict_improvement_failure_count = 0
    transport_failure_count = 0
    energy_failure_count = 0
    verified_first_step_gain: list[float] = []
    verified_minimum_gain: list[float] = []
    verified_window_gain: list[float] = []
    verified_window_worst_gain: list[float] = []
    for (
        proposal_score,
        movers,
        displacement,
        minimum_separation,
        flight_energy,
        proposal_source,
    ) in verified_proposals:
            movement_plan = baseline_plan.copy()
            optimized_plan = _optimized_recovery_plan(
                baseline_plan,
                displacement,
                movers,
                baseline_gradient_horizon,
                proposal_possible_worst,
                max_displacement,
            )
            if optimized_plan is None:
                continue
            movement_plan = optimized_plan
            position_cursor = positions.copy()
            full_plan_safe = True
            plan_minimum_separation = float("inf")
            for step_displacement in movement_plan:
                if np.any(np.linalg.norm(
                    step_displacement, axis=1) > max_displacement + 1.0e-12):
                    full_plan_safe = False
                    break
                raw_next_xy = position_cursor[:, :2] + step_displacement
                if not bool(
                    np.all(raw_next_xy[:, 0] >= boundary_margin)
                    and np.all(raw_next_xy[:, 0] <= width - boundary_margin)
                    and np.all(raw_next_xy[:, 1] >= boundary_margin)
                    and np.all(raw_next_xy[:, 1] <= height - boundary_margin)
                ):
                    full_plan_safe = False
                    break
                step_separation = _minimum_swept_separation(
                    position_cursor, step_displacement)
                plan_minimum_separation = min(
                    plan_minimum_separation, step_separation)
                if (
                    step_separation + 1.0e-12
                    < float(safe_separation_m)
                    + float(cfg.separation_margin_m)
                ):
                    full_plan_safe = False
                    break
                position_cursor = position_cursor.copy()
                position_cursor[:, :2] = raw_next_xy
            if not full_plan_safe:
                continue
            future_states = propagate_owner_local_horizon(
                origin_state,
                movement_plan,
                dt_s=float(dt_s),
                max_speed_mps=float(max_speed_mps),
                area_size_m=(width, height),
                advance_targets=False,
            )[1:]
            next_state = future_states[0]
            bounds = _future_bounds(
                future_states,
                target_invariant_cache,
                provenance,
                invariant_layout=layout,
                carrier_hz=carrier_hz,
                delta_f_hz=delta_f_hz,
                symbol_period_s=symbol_period_s,
                delay_bins=delay_bins,
                doppler_bins=doppler_bins,
                covariance_radius=covariance_radius,
                dd_support_threshold=dd_support_threshold,
                dd_additive_margin=dd_additive_margin,
                residual_log_margin=residual_log_margin,
            )
            candidate_lower_pd = np.stack([
                _pd(item, active, wire_sensing_lower[step], p_fa)
                for step, item in enumerate(bounds.lower)
            ])
            candidate_upper_pd = np.stack([
                _pd(item, active, wire_sensing_upper[step], p_fa)
                for step, item in enumerate(bounds.upper)
            ])
            wire_candidate_lower_pd = np.asarray([
                quantize_nonnegative_float16_lower(value)
                for value in candidate_lower_pd.reshape(-1)
            ], dtype=np.float64).reshape(horizon, Q)
            if common_target_scale:
                candidate_normalized = _normalized_geometry_deflection(
                    future_states,
                    bounds,
                    active,
                    wire_sensing_lower,
                )
                coupled_gain = np.zeros((horizon, Q), dtype=np.float64)
                clipped_coupled_gain = np.zeros(
                    (horizon, Q), dtype=np.float64)
                for step in range(horizon):
                    for target_id in range(Q):
                        scale_lower = (
                            invariant_lower[target_id]
                            * np.exp(-float(residual_log_margin)))
                        scale_upper = (
                            invariant_upper[target_id]
                            * np.exp(float(residual_log_margin)))
                        coupled_gain[step, target_id] = (
                            _minimum_common_scale_pd_gain(
                                baseline_normalized[step, target_id],
                                candidate_normalized[step, target_id],
                                scale_lower,
                                scale_upper,
                                p_fa,
                            ))
                        clipped_coupled_gain[step, target_id] = (
                            _minimum_common_scale_clipped_pd_gain(
                                baseline_normalized[step, target_id],
                                candidate_normalized[step, target_id],
                                scale_lower,
                                scale_upper,
                                p_fa,
                                qos_floor,
                            ))
                baseline_pd_lower = compute_detection_probabilities(
                    baseline_normalized * invariant_lower[None, :]
                    * np.exp(-float(residual_log_margin)),
                    p_fa,
                )
                baseline_pd_upper = compute_detection_probabilities(
                    baseline_normalized * invariant_upper[None, :]
                    * np.exp(float(residual_log_margin)),
                    p_fa,
                )
                possible_worst_horizon = _possible_worst_targets(
                    baseline_pd_lower,
                    baseline_pd_upper,
                    float(cfg.probability_tolerance),
                )
                possible_worst = possible_worst_horizon[0]
                coupled_safety = np.zeros((horizon, Q), dtype=np.float64)
                worst_target_margin = np.zeros(
                    (horizon, Q), dtype=np.float64)
                for step in range(horizon):
                    possible_worst_step = possible_worst_horizon[step]
                    baseline_worst_upper = float(np.min(
                        wire_baseline_upper_pd[step]))
                    worst_target_margin[step, possible_worst_step] = (
                        coupled_gain[step, possible_worst_step])
                    worst_target_margin[step, ~possible_worst_step] = (
                        wire_candidate_lower_pd[step, ~possible_worst_step]
                        - baseline_worst_upper)
                    for target_id in range(Q):
                        scale_lower = (
                            invariant_lower[target_id]
                            * np.exp(-float(residual_log_margin)))
                        scale_upper = (
                            invariant_upper[target_id]
                            * np.exp(float(residual_log_margin)))
                        coupled_safety[step, target_id] = (
                            _minimum_common_scale_target_safety_margin(
                                baseline_normalized[step, target_id],
                                candidate_normalized[step, target_id],
                                scale_lower,
                                scale_upper,
                                p_fa,
                                qos_floor,
                            ))
                coupled_window_gain = np.sum(
                    clipped_coupled_gain, axis=0)
                window_worst_gain = float(np.sum(np.min(
                    worst_target_margin, axis=1)))
                target_safe = bool(
                    np.all(
                    coupled_window_gain
                    + float(cfg.probability_tolerance) >= 0.0)
                    and window_worst_gain
                    + float(cfg.probability_tolerance) >= 0.0
                )
                useful = bool(
                    np.min(worst_target_margin[0])
                    > float(cfg.minimum_worst_pd_improvement)
                    + float(cfg.probability_tolerance))
                first_step_worst_gain = float(np.min(
                    worst_target_margin[0]))
            else:
                coupled_gain = (
                    wire_candidate_lower_pd - wire_baseline_upper_pd)
                coupled_safety = coupled_gain.copy()
                possible_worst = np.ones(Q, dtype=bool)
                first_step_worst_margin = np.full(
                    Q,
                    np.min(wire_candidate_lower_pd[0])
                    - np.min(wire_baseline_upper_pd[0]),
                    dtype=np.float64,
                )
                coupled_window_gain = np.sum(
                    np.minimum(wire_candidate_lower_pd, float(qos_floor))
                    - np.minimum(wire_baseline_upper_pd, float(qos_floor)),
                    axis=0,
                )
                window_worst_gain = float(np.sum([
                    np.min(wire_candidate_lower_pd[step])
                    - np.min(wire_baseline_upper_pd[step])
                    for step in range(horizon)
                ]))
                target_safe = bool(
                    np.all(
                    coupled_window_gain
                    + float(cfg.probability_tolerance) >= 0.0)
                    and window_worst_gain
                    + float(cfg.probability_tolerance) >= 0.0
                )
                useful = bool(
                    np.min(first_step_worst_margin)
                    > float(cfg.minimum_worst_pd_improvement)
                    + float(cfg.probability_tolerance))
                first_step_worst_gain = float(np.min(
                    first_step_worst_margin))
            verified_first_step_gain.append(float(np.min(
                coupled_safety[0])))
            verified_minimum_gain.append(float(np.min(coupled_safety)))
            verified_window_gain.append(float(np.min(coupled_window_gain)))
            verified_window_worst_gain.append(window_worst_gain)
            transport_verification_count = len(verified_proposals)
            transport_mover_counts = tuple(
                len(item[1])
                for item in verified_proposals[:transport_verification_count]
            )
            transport = certify_geometry_repair_transport(
                movers,
                positions=positions,
                target_owner=np.asarray(target_owner, dtype=np.int64),
                communication_power_w=wire_comm,
                communication_model=communication_model,
                control_period_s=float(control_period_s),
                layout=wire,
                verification_candidate_count=transport_verification_count,
                verification_mover_counts=transport_mover_counts,
                certificate_horizon_steps=horizon,
                snr_margin_db=float(cfg.snr_margin_db),
                latency_margin_s=float(cfg.latency_margin_s),
            )
            plan_speed = np.linalg.norm(
                movement_plan, axis=2) / float(dt_s)
            plan_flight_energy = (
                float(static_flight_power_w)
                + float(quadratic_flight_power_coeff) * plan_speed ** 2
            ) * float(dt_s)
            flight_by_uav = np.asarray([
                quantize_nonnegative_float16_upper(value)
                for value in np.sum(plan_flight_energy, axis=0)
            ], dtype=np.float64)
            baseline_plan_flight_energy = (
                float(static_flight_power_w)
                + float(quadratic_flight_power_coeff)
                * (np.linalg.norm(
                    baseline_plan, axis=2) / float(dt_s)) ** 2
            ) * float(dt_s)
            universal_step_flight_energy = (
                float(static_flight_power_w)
                + float(quadratic_flight_power_coeff)
                * float(max_speed_mps) ** 2
            ) * float(dt_s)
            escrow_flight_energy_plan = np.full(
                (horizon, K),
                quantize_nonnegative_float16_upper(
                    universal_step_flight_energy),
                dtype=np.float64,
            )
            escrow_flight_by_uav = np.sum(
                escrow_flight_energy_plan, axis=0)
            sensing_energy = (
                np.sum(wire_sensing_upper, axis=(0, 2)) * float(dt_s))
            required_energy = (
                escrow_flight_by_uav + sensing_energy
                + np.asarray(transport.per_uav_energy_j, dtype=np.float64)
            )
            energy_safe = bool(np.all(
                battery + 1.0e-12 >= required_energy))
            target_no_harm_failure_count += int(not target_safe)
            window_worst_failure_count += int(
                window_worst_gain + float(cfg.probability_tolerance) < 0.0)
            strict_improvement_failure_count += int(not useful)
            transport_failure_count += int(not transport.feasible)
            energy_failure_count += int(not energy_safe)
            if (
                not target_safe or not useful or not transport.feasible
                or not energy_safe
            ):
                continue
            candidates.append(GeometryCandidateCertificate(
                joint_plan_digest=digest,
                mover=movers[0],
                movers=movers,
                baseline_displacement_m=baseline_displacement.copy(),
                baseline_movement_plan_m=baseline_plan.copy(),
                displacement_m=displacement,
                movement_plan_m=movement_plan.copy(),
                sensing_power_plan_w=wire_sensing_lower.copy(),
                communication_power_plan_w=wire_communication_plan.copy(),
                proposal_source=proposal_source,
                proposal_score=proposal_score,
                active_weak_target=int(weak_target_order[0]),
                certificate_horizon_steps=horizon,
                next_state=next_state,
                baseline_physical_bounds=baseline_bounds,
                physical_bounds=bounds,
                candidate_lower_pd=candidate_lower_pd,
                candidate_upper_pd=candidate_upper_pd,
                wire_baseline_lower_pd=wire_baseline_lower_pd.copy(),
                wire_baseline_upper_pd=wire_baseline_upper_pd.copy(),
                wire_candidate_lower_pd=wire_candidate_lower_pd,
                coupled_pd_gain=coupled_gain,
                coupled_window_pd_gain=coupled_window_gain,
                coupled_window_worst_pd_gain=window_worst_gain,
                coupled_first_step_worst_pd_gain=first_step_worst_gain,
                minimum_first_step_worst_pd_gain=(
                    float(cfg.minimum_worst_pd_improvement)
                    + float(cfg.probability_tolerance)),
                coupled_target_safety_margin=coupled_safety,
                possible_worst_target=possible_worst.copy(),
                flight_energy_j=float(np.sum(flight_by_uav)),
                escrow_flight_energy_plan_j=(
                    escrow_flight_energy_plan.copy()),
                incremental_flight_energy_j=float(
                    np.sum(np.maximum(
                        escrow_flight_by_uav
                        - np.sum(baseline_plan_flight_energy, axis=0),
                        0.0))),
                minimum_separation_m=(
                    quantize_nonnegative_float16_lower(
                        plan_minimum_separation)),
                transport=transport,
            ))
    if not candidates:
        return CertifiedGeometryRepairDecision(
            accepted=False,
            reason="no_certified_trust_region_move",
            displacement_m=np.zeros((K, 2), dtype=np.float64),
            baseline_lower_pd=baseline_lower_pd,
            baseline_upper_pd=baseline_upper_pd,
            candidate_lower_pd=baseline_lower_pd.copy(),
            candidate_upper_pd=baseline_upper_pd.copy(),
            candidate_count=candidate_count,
            verified_candidate_count=len(verified_proposals),
            feasible_candidate_count=0,
            target_no_harm_failure_count=target_no_harm_failure_count,
            window_worst_failure_count=window_worst_failure_count,
            fast_opportunity_failure_count=0,
            strict_improvement_failure_count=(
                strict_improvement_failure_count),
            transport_failure_count=transport_failure_count,
            energy_failure_count=energy_failure_count,
            best_verified_first_step_coupled_gain=float(max(
                verified_first_step_gain, default=float("-inf"))),
            best_verified_minimum_coupled_gain=float(max(
                verified_minimum_gain, default=float("-inf"))),
            best_verified_window_coupled_gain=float(max(
                verified_window_gain, default=float("-inf"))),
            best_verified_window_worst_pd_gain=float(max(
                verified_window_worst_gain, default=float("-inf"))),
            best_candidate=None,
        )
    best = max(candidates, key=_certified_candidate_rank)
    return CertifiedGeometryRepairDecision(
        accepted=True,
        reason="accepted:certified_atomic_geometry",
        displacement_m=best.displacement_m.copy(),
        baseline_lower_pd=baseline_lower_pd,
        baseline_upper_pd=best.wire_baseline_upper_pd.copy(),
        candidate_lower_pd=best.wire_candidate_lower_pd.copy(),
        candidate_upper_pd=best.candidate_upper_pd.copy(),
        candidate_count=candidate_count,
        verified_candidate_count=len(verified_proposals),
        feasible_candidate_count=len(candidates),
        target_no_harm_failure_count=target_no_harm_failure_count,
        window_worst_failure_count=window_worst_failure_count,
        fast_opportunity_failure_count=0,
        strict_improvement_failure_count=strict_improvement_failure_count,
        transport_failure_count=transport_failure_count,
        energy_failure_count=energy_failure_count,
        best_verified_first_step_coupled_gain=float(max(
            verified_first_step_gain, default=float("-inf"))),
        best_verified_minimum_coupled_gain=float(max(
            verified_minimum_gain, default=float("-inf"))),
        best_verified_window_coupled_gain=float(max(
            verified_window_gain, default=float("-inf"))),
        best_verified_window_worst_pd_gain=float(max(
            verified_window_worst_gain, default=float("-inf"))),
        best_candidate=best,
    )
