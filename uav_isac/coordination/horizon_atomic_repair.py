"""Robust receding-horizon ranking of atomic ISAC structure repairs.

The module deliberately separates interval construction from optimization.
It consumes one simultaneous ``(H,K,K,Q)`` coefficient envelope produced by
an independently calibrated owner-local model.  For each fixed structure and
each horizon step it runs the same finite-round, quantized distributed
max-min protocol as deployment on the lower coefficient tensor.  Optimality
is not required: its feasible RF plan has two monotonic guarantees under the
Gaussian-deflection detector:

* if ``a_true >= a_lower``, its realized candidate detection probability is
  no smaller than the reported candidate lower bound; and
* using the *same No-op RF plan*, ``a_true <= a_upper`` makes the reported
  No-op value an upper bound on realized No-op detection probability.

Consequently ``L_candidate[h,q] >= U_noop[h,q]`` is a sufficient target-wise
no-harm condition.  Only the first atomic move is returned; the full horizon
is recomputed at the next event, so this is receding-horizon control rather
than an open-loop multi-commit sequence.
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
from typing import Sequence
import time

import numpy as np

from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    role_owner_from_structure,
)
from uav_isac.coordination.owner_proposal_transport import (
    quantize_nonnegative_float16_lower,
)
from uav_isac.coordination.maxmin_power import (
    distributed_column_generation_maxmin_power,
    fixed_owner_gain_matrix,
)
from uav_isac.evaluation.horizon_transition_gate import (
    HorizonRouteDecision,
    HorizonTransitionBounds,
    certify_horizon_paired_deflection_transition,
    certify_horizon_transition,
)
from uav_isac.physical.detection import compute_detection_probabilities


@dataclass(frozen=True)
class HorizonCoefficientEnvelope:
    """Simultaneous owner-local edge interval over horizon and targets."""

    lower: np.ndarray
    upper: np.ndarray
    source: str
    miscoverage: float

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if (
            lower.ndim != 4 or lower.shape[0] < 1
            or lower.shape[1] != lower.shape[2]
            or lower.shape[3] < 1 or upper.shape != lower.shape
        ):
            raise ValueError(
                "coefficient bounds must share a non-empty (H,K,K,Q) shape")
        if (
            np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
            or np.any(lower < 0.0) or np.any(upper < lower)
        ):
            raise ValueError(
                "coefficient bounds must be finite with 0 <= lower <= upper")
        diagonal = np.arange(lower.shape[1])
        if np.any(lower[:, diagonal, diagonal, :] > 0.0):
            raise ValueError("self-edge coefficient lower bounds must be zero")
        source = str(self.source).strip().lower()
        if source not in {"feedback", "physics"}:
            raise ValueError("coefficient source must be feedback or physics")
        alpha = float(self.miscoverage)
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("coefficient miscoverage must lie in (0,1)")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "miscoverage", alpha)

    @property
    def horizon(self) -> int:
        return int(self.lower.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.lower.shape[1])

    @property
    def num_targets(self) -> int:
        return int(self.lower.shape[3])


@dataclass(frozen=True)
class HorizonResourceLimits:
    """Hard protocol and RF limits; no objective penalty can relax them."""

    control_period_s: float
    total_power_w: float = 1.0
    max_over_air_bits: int | None = None
    max_control_energy_j: float | None = None

    def __post_init__(self) -> None:
        period = float(self.control_period_s)
        power = float(self.total_power_w)
        if (
            not np.isfinite(period) or period <= 0.0
            or not np.isfinite(power) or power <= 0.0
        ):
            raise ValueError("control period and total power must be positive")
        if self.max_over_air_bits is not None and int(
            self.max_over_air_bits
        ) < 0:
            raise ValueError("bit limit must be non-negative")
        if self.max_control_energy_j is not None and (
            not np.isfinite(float(self.max_control_energy_j))
            or float(self.max_control_energy_j) < 0.0
        ):
            raise ValueError("energy limit must be finite non-negative")


@dataclass(frozen=True)
class HorizonPowerProtocol:
    """Finite-round quantized RF optimizer used by the deployed protocol."""

    rounds: int
    price_bits: int
    feedback_bits: int
    reuse_primal_master_duals: bool = False

    def __post_init__(self) -> None:
        if int(self.rounds) < 1:
            raise ValueError("horizon power protocol requires at least one round")
        if int(self.price_bits) < 0:
            raise ValueError("price bit count must be non-negative")
        if int(self.feedback_bits) not in {0, 16, 32, 64}:
            raise ValueError(
                "feedback bits must be one of {0,16,32,64}")


@dataclass(frozen=True)
class HorizonCandidateResources:
    """Candidate-specific communication reservation and protocol usage."""

    comm_power_w: np.ndarray
    over_air_bits: int = 0
    protocol_latency_s: float = 0.0
    control_energy_j: float = 0.0
    transport_feasible: bool = True
    structural_feasible: bool = True

    def validated(
        self,
        *,
        horizon: int,
        num_agents: int,
        limits: HorizonResourceLimits,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        comm = np.asarray(self.comm_power_w, dtype=np.float64)
        if comm.shape != (int(horizon), int(num_agents)):
            raise ValueError("comm_power_w must have shape (H,K)")
        if (
            np.any(~np.isfinite(comm)) or np.any(comm < 0.0)
            or np.any(comm >= float(limits.total_power_w))
        ):
            raise ValueError(
                "communication power must lie in [0,total_power_w)")
        bits = int(self.over_air_bits)
        latency = float(self.protocol_latency_s)
        energy = float(self.control_energy_j)
        if (
            bits < 0 or not np.isfinite(latency) or latency < 0.0
            or not np.isfinite(energy) or energy < 0.0
        ):
            raise ValueError("protocol usage must be finite non-negative")
        reasons: list[str] = []
        if not bool(self.transport_feasible):
            reasons.append("transport")
        if not bool(self.structural_feasible):
            reasons.append("structure")
        if latency > float(limits.control_period_s) + 1.0e-12:
            reasons.append("deadline")
        if (
            limits.max_over_air_bits is not None
            and bits > int(limits.max_over_air_bits)
        ):
            reasons.append("bits")
        if (
            limits.max_control_energy_j is not None
            and energy > float(limits.max_control_energy_j) + 1.0e-15
        ):
            reasons.append("energy")
        return comm, tuple(reasons)


@dataclass(frozen=True)
class HorizonLagrangePrices:
    """Non-negative resource shadow prices in detection-probability units."""

    per_switched_edge: float = 0.0
    per_over_air_bit: float = 0.0
    per_latency_second: float = 0.0
    per_control_joule: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray([
            self.per_switched_edge,
            self.per_over_air_bit,
            self.per_latency_second,
            self.per_control_joule,
        ], dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("horizon resource prices must be non-negative")

    def penalty(
        self,
        *,
        switched_edges: int,
        resources: HorizonCandidateResources,
    ) -> float:
        return float(
            float(self.per_switched_edge) * int(switched_edges)
            + float(self.per_over_air_bit) * int(resources.over_air_bits)
            + float(self.per_latency_second)
            * float(resources.protocol_latency_s)
            + float(self.per_control_joule)
            * float(resources.control_energy_j)
        )


@dataclass(frozen=True)
class HorizonStructureRollout:
    lower_pd: np.ndarray
    upper_pd: np.ndarray
    sensing_power_w: np.ndarray
    comm_power_w: np.ndarray
    max_power_balance_error_w: float


@dataclass(frozen=True)
class HorizonCommonMixProxy:
    """Feasible coverage-column lower bound for proposal pre-ranking."""

    target_capacity_lower: np.ndarray
    common_deflection_lower: np.ndarray
    worst_pd_lower: np.ndarray
    discounted_worst_pd_lower: float


@dataclass(frozen=True)
class HorizonAtomicCandidateEvaluation:
    candidate_index: int
    move: LocalMove
    switched_edges: int
    resources: HorizonCandidateResources
    resource_reasons: tuple[str, ...]
    objective_penalty: float
    rollout: HorizonStructureRollout
    gate: HorizonRouteDecision
    net_discounted_gain_lower: float
    paired_deflection_lower: np.ndarray | None = None
    rollout_compute_s: float = 0.0


@dataclass(frozen=True)
class HorizonAtomicRepairDecision:
    selected_index: int | None
    selected_move: LocalMove | None
    selected_sensing_power_w: np.ndarray | None
    noop: HorizonStructureRollout
    evaluations: tuple[HorizonAtomicCandidateEvaluation, ...]
    discount: float
    qos_floor: float
    power_protocol: HorizonPowerProtocol
    common_power_across_horizon: bool = False
    noop_rollout_compute_s: float = 0.0

    @property
    def accept(self) -> bool:
        return self.selected_index is not None


def horizon_common_mix_proxy(
    selected: np.ndarray,
    envelope: HorizonCoefficientEnvelope,
    comm_power_w: np.ndarray,
    *,
    total_power_w: float,
    false_alarm_probability: float,
    feedback_bits: int,
    discount: float = 1.0,
) -> HorizonCommonMixProxy:
    """Return a cheap deployable lower bound for owner-proposal ranking.

    The distributed master is initialized with one complete RF column per
    target.  Column q gives target capacity ``c_q``.  Time shares
    ``w_q=(1/c_q)/sum_l(1/c_l)`` attain the common deflection
    ``1/sum_q(1/c_q)`` while preserving every UAV budget.  Capacities are
    rounded downward with the actual feedback scalar codec before computing
    the bound.  Hence the proxy is achievable by the initial protocol column
    set and cannot overstate its lower objective.
    """
    pair = np.asarray(selected, dtype=bool)
    comm = np.asarray(comm_power_w, dtype=np.float64)
    H = envelope.horizon
    K = envelope.num_agents
    Q = envelope.num_targets
    if pair.shape != (K, K, Q) or comm.shape != (H, K):
        raise ValueError("proxy structure/communication dimensions mismatch")
    power = float(total_power_w)
    p_fa = float(false_alarm_probability)
    gamma = float(discount)
    bits = int(feedback_bits)
    if (
        np.any(~np.isfinite(comm)) or np.any(comm < 0.0)
        or np.any(comm >= power) or not np.isfinite(power) or power <= 0.0
        or not np.isfinite(p_fa) or not 0.0 < p_fa < 1.0
        or not np.isfinite(gamma) or not 0.0 < gamma <= 1.0
        or bits not in {0, 16, 32, 64}
    ):
        raise ValueError("proxy physical/protocol inputs are invalid")
    edges = [tuple(edge) for edge in np.argwhere(pair)]
    capacity = np.zeros((H, Q), dtype=np.float64)
    common = np.zeros(H, dtype=np.float64)
    worst_pd = np.full(H, p_fa, dtype=np.float64)
    for step in range(H):
        gain, _ = fixed_owner_gain_matrix(envelope.lower[step], edges)
        raw_capacity = np.sum(
            gain * (power - comm[step])[:, None], axis=0)
        if bits == 16:
            # The wire scalar has finite support.  Saturating an oversized
            # capacity at the largest finite binary16 value is still a valid
            # directional lower bound and avoids encoding +inf.  This only
            # discards resolution in the detector's already saturated regime.
            encodable_capacity = np.minimum(
                raw_capacity, float(np.finfo(np.float16).max))
            capacity[step] = np.asarray([
                quantize_nonnegative_float16_lower(value)
                for value in encodable_capacity
            ], dtype=np.float64)
        elif bits == 32:
            encoded = np.asarray(raw_capacity, dtype=np.float32)
            above = encoded.astype(np.float64) > raw_capacity
            encoded[above] = np.nextafter(
                encoded[above], np.float32(-np.inf), dtype=np.float32)
            capacity[step] = encoded.astype(np.float64)
        else:
            capacity[step] = raw_capacity
        if np.all(capacity[step] > 0.0):
            common[step] = float(
                1.0 / np.sum(1.0 / capacity[step]))
            worst_pd[step] = float(compute_detection_probabilities(
                np.asarray([common[step]]), p_fa)[0])
    weights = gamma ** np.arange(H, dtype=np.float64)
    return HorizonCommonMixProxy(
        target_capacity_lower=capacity,
        common_deflection_lower=common,
        worst_pd_lower=worst_pd,
        discounted_worst_pd_lower=float(np.dot(weights, worst_pd)),
    )


def _structure_rollout(
    selected: np.ndarray,
    envelope: HorizonCoefficientEnvelope,
    comm_power_w: np.ndarray,
    *,
    total_power_w: float,
    false_alarm_probability: float,
    power_protocol: HorizonPowerProtocol,
    incumbent_sensing_power_w: np.ndarray | None = None,
    common_power_across_horizon: bool = False,
) -> HorizonStructureRollout:
    pair = np.asarray(selected, dtype=bool)
    H = envelope.horizon
    K = envelope.num_agents
    Q = envelope.num_targets
    if pair.shape != (K, K, Q):
        raise ValueError("structure must have shape (K,K,Q)")
    if np.any(pair[np.arange(K), np.arange(K), :]):
        raise ValueError("self edges are not valid bistatic structures")
    comm = np.asarray(comm_power_w, dtype=np.float64)
    lower_pd = np.zeros((H, Q), dtype=np.float64)
    upper_pd = np.zeros((H, Q), dtype=np.float64)
    sensing = np.zeros((H, K, Q), dtype=np.float64)
    incumbent = (
        None if incumbent_sensing_power_w is None
        else np.asarray(incumbent_sensing_power_w, dtype=np.float64)
    )
    if incumbent is not None and (
        incumbent.shape != (H, K, Q)
        or np.any(~np.isfinite(incumbent))
        or np.any(incumbent < 0.0)
    ):
        raise ValueError("incumbent sensing power must have shape (H,K,Q)")
    selected_edges = [tuple(edge) for edge in np.argwhere(pair)]
    lower_gains = np.zeros((H, K, Q), dtype=np.float64)
    upper_gains = np.zeros((H, K, Q), dtype=np.float64)
    for step in range(H):
        lower_gain, lower_owner = fixed_owner_gain_matrix(
            envelope.lower[step], selected_edges)
        upper_gain, upper_owner = fixed_owner_gain_matrix(
            envelope.upper[step], selected_edges)
        if not np.array_equal(lower_owner, upper_owner):
            raise AssertionError("coefficient intervals changed graph ownership")
        lower_gains[step] = lower_gain
        upper_gains[step] = upper_gain

    budgets = float(total_power_w) - comm
    if bool(common_power_across_horizon):
        # One causal open-loop weight vector is shared by all predicted steps.
        # Since g_h >= min_h g_h and B_h >= min_h B_h elementwise, the plan
        # solved at the joint minima is a valid lower certificate at every h.
        robust_gain = np.min(lower_gains, axis=0)
        robust_budget = np.min(budgets, axis=0)
        incumbent_step = None
        if incumbent is not None:
            incumbent_mass = np.sum(incumbent[0], axis=1, keepdims=True)
            incumbent_weights = (
                incumbent[0] / np.maximum(incumbent_mass, 1.0e-15))
            zero_mass = incumbent_mass[:, 0] <= 1.0e-15
            if np.any(zero_mass):
                incumbent_weights[zero_mass] = 1.0 / Q
            incumbent_step = robust_budget[:, None] * incumbent_weights
        solved = distributed_column_generation_maxmin_power(
            robust_gain,
            robust_budget,
            rounds=int(power_protocol.rounds),
            price_bits=int(power_protocol.price_bits),
            feedback_bits=int(power_protocol.feedback_bits),
            incumbent_power_w=incumbent_step,
            reuse_primal_master_duals=bool(
                power_protocol.reuse_primal_master_duals),
        )
        robust_weights = solved.power_w / np.maximum(
            robust_budget[:, None], 1.0e-15)
        sensing[:] = budgets[:, :, None] * robust_weights[None, :, :]
    else:
        for step in range(H):
            incumbent_step = None
            if incumbent is not None:
                mass = np.sum(incumbent[step], axis=1, keepdims=True)
                weights = incumbent[step] / np.maximum(mass, 1.0e-15)
                zero_mass = mass[:, 0] <= 1.0e-15
                if np.any(zero_mass):
                    weights[zero_mass] = 1.0 / Q
                incumbent_step = budgets[step, :, None] * weights
            solved = distributed_column_generation_maxmin_power(
                lower_gains[step],
                budgets[step],
                rounds=int(power_protocol.rounds),
                price_bits=int(power_protocol.price_bits),
                feedback_bits=int(power_protocol.feedback_bits),
                incumbent_power_w=incumbent_step,
                reuse_primal_master_duals=bool(
                    power_protocol.reuse_primal_master_duals),
            )
            sensing[step] = solved.power_w

    for step in range(H):
        lower_deflection = np.sum(
            lower_gains[step] * sensing[step], axis=0)
        upper_deflection = np.sum(
            upper_gains[step] * sensing[step], axis=0)
        lower_pd[step] = compute_detection_probabilities(
            lower_deflection, false_alarm_probability)
        upper_pd[step] = compute_detection_probabilities(
            upper_deflection, false_alarm_probability)
    balance = comm + np.sum(sensing, axis=2)
    error = float(np.max(np.abs(balance - float(total_power_w))))
    if error > 1.0e-8:
        raise AssertionError("horizon RF plan violates per-UAV power equality")
    return HorizonStructureRollout(
        lower_pd=lower_pd,
        upper_pd=upper_pd,
        sensing_power_w=sensing,
        comm_power_w=comm.copy(),
        max_power_balance_error_w=error,
    )


def paired_deflection_difference_lower(
    initial_selected: np.ndarray,
    candidate_selected: np.ndarray,
    noop_sensing_power_w: np.ndarray,
    candidate_sensing_power_w: np.ndarray,
    envelope: HorizonCoefficientEnvelope,
) -> np.ndarray:
    """Return the exact common-box lower bound on candidate-minus-Noop D.

    At step h and target q, both actions experience the same coefficient
    tensor.  Writing their linear Deflection difference as ``w*a``, interval
    arithmetic is exact on a box: choose ``L`` for non-negative weights and
    ``U`` for negative weights.  This retains the physical correlation that
    independent candidate-lower/Noop-upper comparisons discard.
    """
    noop_pair = np.asarray(initial_selected, dtype=bool)
    candidate_pair = np.asarray(candidate_selected, dtype=bool)
    noop_power = np.asarray(noop_sensing_power_w, dtype=np.float64)
    candidate_power = np.asarray(
        candidate_sensing_power_w, dtype=np.float64)
    H = envelope.horizon
    K = envelope.num_agents
    Q = envelope.num_targets
    if (
        noop_pair.shape != (K, K, Q)
        or candidate_pair.shape != (K, K, Q)
        or noop_power.shape != (H, K, Q)
        or candidate_power.shape != (H, K, Q)
    ):
        raise ValueError("paired Deflection inputs have inconsistent shape")
    if (
        np.any(~np.isfinite(noop_power))
        or np.any(~np.isfinite(candidate_power))
        or np.any(noop_power < 0.0)
        or np.any(candidate_power < 0.0)
    ):
        raise ValueError("paired sensing powers must be finite non-negative")

    result = np.zeros((H, Q), dtype=np.float64)
    for step in range(H):
        candidate_weight = (
            candidate_pair.astype(np.float64)
            * candidate_power[step, :, None, :]
        )
        noop_weight = (
            noop_pair.astype(np.float64)
            * noop_power[step, :, None, :]
        )
        weight = candidate_weight - noop_weight
        term = np.where(
            weight >= 0.0,
            weight * envelope.lower[step],
            weight * envelope.upper[step],
        )
        result[step] = np.sum(term, axis=(0, 1))
    return result


def rank_atomic_horizon_repairs(
    initial_selected: np.ndarray,
    candidates: Sequence[LocalMove],
    envelope: HorizonCoefficientEnvelope,
    *,
    noop_resources: HorizonCandidateResources,
    candidate_resources: Sequence[HorizonCandidateResources],
    limits: HorizonResourceLimits,
    false_alarm_probability: float,
    qos_floor: float,
    discount: float = 1.0,
    prices: HorizonLagrangePrices | None = None,
    power_protocol: HorizonPowerProtocol,
    noop_incumbent_sensing_power_w: np.ndarray | None = None,
    paired_deflection_gate: bool = False,
    common_power_across_horizon: bool = False,
) -> HorizonAtomicRepairDecision:
    """Rank certified atomic moves and return at most one first action.

    Candidate RF powers are optimized independently at each predicted step
    under the candidate's communication reservation.  No-op upper bounds use
    exactly the No-op lower-optimized powers, which is essential for a valid
    action-versus-No-op comparison rather than an unrelated oracle ceiling.
    """
    pair = np.asarray(initial_selected, dtype=bool)
    items = tuple(candidates)
    resources = tuple(candidate_resources)
    H = envelope.horizon
    K = envelope.num_agents
    Q = envelope.num_targets
    if pair.shape != (K, K, Q):
        raise ValueError("initial structure must have shape (K,K,Q)")
    if len(items) != len(resources):
        raise ValueError("every candidate requires one resource certificate")
    p_fa = float(false_alarm_probability)
    gamma = float(discount)
    floor = float(qos_floor)
    if not np.isfinite(p_fa) or not 0.0 < p_fa < 1.0:
        raise ValueError("false alarm probability must lie in (0,1)")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("discount must lie in (0,1]")
    if not np.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("qos_floor must lie in (0,1]")
    shadow = prices or HorizonLagrangePrices()
    if bool(paired_deflection_gate) and any(
        value > 0.0 for value in (
            shadow.per_switched_edge,
            shadow.per_over_air_bit,
            shadow.per_latency_second,
            shadow.per_control_joule,
        )
    ):
        raise ValueError(
            "paired Deflection gate requires zero probability-unit prices")

    noop_comm, noop_reasons = noop_resources.validated(
        horizon=H, num_agents=K, limits=limits)
    if noop_reasons:
        raise ValueError(
            "No-op resource plan must be hard feasible: "
            + ",".join(noop_reasons))
    noop_started = time.perf_counter()
    noop = _structure_rollout(
        pair,
        envelope,
        noop_comm,
        total_power_w=float(limits.total_power_w),
        false_alarm_probability=p_fa,
        power_protocol=power_protocol,
        incumbent_sensing_power_w=noop_incumbent_sensing_power_w,
        common_power_across_horizon=bool(common_power_across_horizon),
    )
    noop_rollout_compute_s = float(time.perf_counter() - noop_started)

    evaluations: list[HorizonAtomicCandidateEvaluation] = []
    for index, (move, usage) in enumerate(zip(items, resources)):
        proposal = np.asarray(move.selected, dtype=bool)
        if proposal.shape != pair.shape:
            raise ValueError("candidate structure shape is inconsistent")
        move_role = np.asarray(move.role, dtype=np.int8).reshape(-1)
        move_owner = np.asarray(move.owner, dtype=np.int64).reshape(-1)
        if move_role.shape != (K,) or move_owner.shape != (Q,):
            raise ValueError("candidate role/owner shapes are inconsistent")
        derived_role, derived_owner = role_owner_from_structure(proposal)
        active = derived_role >= 0
        if (
            not np.array_equal(move_owner, derived_owner)
            or not np.array_equal(move_role[active], derived_role[active])
        ):
            raise ValueError(
                "candidate role/owner metadata disagrees with its structure")
        comm, resource_reasons = usage.validated(
            horizon=H, num_agents=K, limits=limits)
        rollout_started = time.perf_counter()
        rollout = _structure_rollout(
            proposal,
            envelope,
            comm,
            total_power_w=float(limits.total_power_w),
            false_alarm_probability=p_fa,
            power_protocol=power_protocol,
            incumbent_sensing_power_w=noop.sensing_power_w,
            common_power_across_horizon=bool(common_power_across_horizon),
        )
        rollout_compute_s = float(time.perf_counter() - rollout_started)
        switched = int(np.count_nonzero(pair ^ proposal))
        penalty = shadow.penalty(
            switched_edges=switched, resources=usage)
        bounds = HorizonTransitionBounds(
            source=envelope.source,
            candidate_lower=rollout.lower_pd,
            noop_lower=noop.lower_pd,
            noop_upper=noop.upper_pd,
            miscoverage=envelope.miscoverage,
        )
        paired_lower = None
        structural_feasible = (
            bool(usage.structural_feasible)
            and "structure" not in resource_reasons
        )
        transport_feasible = not any(
            reason != "structure" for reason in resource_reasons)
        if bool(paired_deflection_gate):
            paired_lower = paired_deflection_difference_lower(
                pair,
                proposal,
                noop.sensing_power_w,
                rollout.sensing_power_w,
                envelope,
            )
            gate = certify_horizon_paired_deflection_transition(
                bounds,
                paired_lower,
                qos_floor=floor,
                discount=gamma,
                objective_cost_deflection=0.0,
                structural_feasible=structural_feasible,
                transport_feasible=transport_feasible,
            )
        else:
            gate = certify_horizon_transition(
                bounds,
                qos_floor=floor,
                discount=gamma,
                objective_cost=penalty,
                structural_feasible=structural_feasible,
                transport_feasible=transport_feasible,
            )
        evaluations.append(HorizonAtomicCandidateEvaluation(
            candidate_index=index,
            move=move,
            switched_edges=switched,
            resources=usage,
            resource_reasons=resource_reasons,
            objective_penalty=penalty,
            rollout=rollout,
            gate=gate,
            net_discounted_gain_lower=float(
                gate.discounted_gain_lower - penalty),
            paired_deflection_lower=paired_lower,
            rollout_compute_s=rollout_compute_s,
        ))

    admissible = [item for item in evaluations if item.gate.accept]
    if not admissible:
        return HorizonAtomicRepairDecision(
            selected_index=None,
            selected_move=None,
            selected_sensing_power_w=None,
            noop=noop,
            evaluations=tuple(evaluations),
            discount=gamma,
            qos_floor=floor,
            power_protocol=power_protocol,
            common_power_across_horizon=bool(
                common_power_across_horizon),
            noop_rollout_compute_s=noop_rollout_compute_s,
        )
    best = max(
        admissible,
        key=lambda item: (
            item.net_discounted_gain_lower,
            -item.switched_edges,
            -int(item.resources.over_air_bits),
            -float(item.resources.protocol_latency_s),
            -item.candidate_index,
        ),
    )
    return HorizonAtomicRepairDecision(
        selected_index=int(best.candidate_index),
        selected_move=best.move,
        selected_sensing_power_w=best.rollout.sensing_power_w.copy(),
        noop=noop,
        evaluations=tuple(evaluations),
        discount=gamma,
        qos_floor=floor,
        power_protocol=power_protocol,
        common_power_across_horizon=bool(common_power_across_horizon),
        noop_rollout_compute_s=noop_rollout_compute_s,
    )
