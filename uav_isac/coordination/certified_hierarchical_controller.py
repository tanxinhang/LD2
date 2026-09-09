"""Certificate-routed hierarchical ISAC resource controller.

The controller never treats a centralized optimizer as a deployable action.
It uses genuine upper certificates only for routing, a finite-round fixed-
structure power protocol for the fast layer, and a bounded owner auction plus
Top-1 atomic local repair for the structural layer. Missing certificates,
unreserved RF control power or a failed commit preserve the original state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from uav_isac.coordination.bottleneck_router import (
    BottleneckDecision,
    RepairRoute,
)
from uav_isac.coordination.certified_maxmin_power_controller import (
    CertifiedMaxMinPowerConfig,
    CertifiedMaxMinPowerDecision,
    certified_fixed_structure_maxmin_power_repair,
)
from uav_isac.coordination.dual_guided_structure_repair import (
    DualGuidedStructureRepairResult,
    dual_guided_atomic_structure_repair,
)
from uav_isac.coordination.joint_structure_relaxation import (
    HierarchicalQoSRouteResult,
    certified_hierarchical_qos_route,
)
from uav_isac.coordination.local_exchange_oracle import (
    role_owner_from_structure,
)
from uav_isac.coordination.maxmin_power import fixed_owner_gain_matrix
from uav_isac.coordination.owner_bid_transport import (
    OwnerBidCandidateResult,
    OwnerBidTransportCertificate,
    OwnerBidWireLayout,
    certify_owner_bid_transport,
    owner_bid_sparse_candidate_mask,
)
from uav_isac.coordination.owner_proposal_transport import (
    OwnerProposalWireLayout,
)
from uav_isac.coordination.power_repair_transport import PowerRepairWireLayout
from uav_isac.coordination.structure_sequence_transport import (
    StructureSequenceTransportCertificate,
    certify_top1_structure_sequence_transport,
)
from uav_isac.domain.communication import CommunicationTransport
from uav_isac.physical.detection import compute_detection_probabilities
from uav_isac.physical.detection import (
    minimum_deflection_for_detection_probability,
)


@dataclass(frozen=True)
class CertifiedHierarchicalControllerConfig:
    qos_floor: float = 0.60
    target_pair_limit: int = 3
    reports_per_receiver: int = 18
    owners_per_target: int = 3
    structure_top_m: int = 1
    structure_max_steps: int = 1
    structure_weak_target_count: int = 6
    structure_ranking_rounds: int = 4
    require_pre_reserved_comm_power: bool = True
    minimum_worst_pd_improvement: float = 0.0
    probability_tolerance: float = 1.0e-9
    snr_margin_db: float = 3.0
    latency_margin_s: float = 5.0e-4
    power: CertifiedMaxMinPowerConfig = field(
        default_factory=lambda: CertifiedMaxMinPowerConfig(rounds=6))

    def __post_init__(self) -> None:
        if not 0.0 < float(self.qos_floor) < 1.0:
            raise ValueError("qos_floor must lie in (0,1)")
        if int(self.target_pair_limit) < 1:
            raise ValueError("target_pair_limit must be positive")
        if int(self.reports_per_receiver) < 1:
            raise ValueError("reports_per_receiver must be positive")
        if int(self.owners_per_target) < 1:
            raise ValueError("owners_per_target must be positive")
        if int(self.structure_top_m) != 1:
            raise ValueError("deployable structural verification is Top-1")
        if int(self.structure_max_steps) != 1:
            raise ValueError("one control event permits one atomic structure step")
        if int(self.structure_weak_target_count) < 1:
            raise ValueError("weak target count must be positive")
        if not 1 <= int(self.structure_ranking_rounds) <= int(self.power.rounds):
            raise ValueError(
                "structure ranking rounds must not exceed power rounds")
        if float(self.minimum_worst_pd_improvement) < 0.0:
            raise ValueError("minimum improvement must be non-negative")
        if float(self.probability_tolerance) < 0.0:
            raise ValueError("probability tolerance must be non-negative")


@dataclass(frozen=True)
class HierarchicalProtocolResources:
    feasible: bool
    reasons: tuple[str, ...]
    total_over_air_bits: int
    total_protocol_latency_s: float
    total_energy_j: float
    projected_comm_power_w: np.ndarray
    max_isac_power_balance_error_w: float


@dataclass(frozen=True)
class CertifiedHierarchicalDecision:
    accepted: bool
    route: RepairRoute
    reason: str
    selected: np.ndarray
    role: np.ndarray
    owner: np.ndarray
    sensing_power_w: np.ndarray
    communication_power_w: np.ndarray
    noop_lower_pd: np.ndarray
    noop_upper_pd: np.ndarray
    candidate_lower_pd: np.ndarray
    candidate_upper_pd: np.ndarray
    route_certificate: HierarchicalQoSRouteResult
    power_decision: CertifiedMaxMinPowerDecision | None
    owner_bid_candidate: OwnerBidCandidateResult | None
    owner_bid_transport: OwnerBidTransportCertificate | None
    structure_repair: DualGuidedStructureRepairResult | None
    structure_transport: StructureSequenceTransportCertificate | None
    protocol_resources: HierarchicalProtocolResources | None

    @property
    def certified_worst_pd_improvement(self) -> float:
        return float(
            np.min(self.candidate_lower_pd) - np.min(self.noop_upper_pd))


def _validated_envelope(
    lower_coefficient: np.ndarray,
    upper_coefficient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower_coefficient, dtype=np.float64)
    upper = np.asarray(upper_coefficient, dtype=np.float64)
    if (
        lower.ndim != 3 or lower.shape[0] != lower.shape[1]
        or lower.shape[2] < 1 or upper.shape != lower.shape
        or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
        or np.any(lower < 0.0) or np.any(upper < lower)
    ):
        raise ValueError("coefficient envelope must satisfy finite 0<=lower<=upper")
    return lower, upper


def _noop_decision(
    *,
    route_certificate: HierarchicalQoSRouteResult,
    reason: str,
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    sensing_power_w: np.ndarray,
    communication_power_w: np.ndarray,
    noop_lower_pd: np.ndarray,
    noop_upper_pd: np.ndarray,
    power_decision: CertifiedMaxMinPowerDecision | None = None,
    owner_bid_candidate: OwnerBidCandidateResult | None = None,
    owner_bid_transport: OwnerBidTransportCertificate | None = None,
    structure_repair: DualGuidedStructureRepairResult | None = None,
    structure_transport: StructureSequenceTransportCertificate | None = None,
    protocol_resources: HierarchicalProtocolResources | None = None,
) -> CertifiedHierarchicalDecision:
    return CertifiedHierarchicalDecision(
        accepted=False,
        route=route_certificate.decision.route,
        reason=str(reason),
        selected=selected.copy(),
        role=role.copy(),
        owner=owner.copy(),
        sensing_power_w=sensing_power_w.copy(),
        communication_power_w=communication_power_w.copy(),
        noop_lower_pd=noop_lower_pd.copy(),
        noop_upper_pd=noop_upper_pd.copy(),
        candidate_lower_pd=noop_lower_pd.copy(),
        candidate_upper_pd=noop_upper_pd.copy(),
        route_certificate=route_certificate,
        power_decision=power_decision,
        owner_bid_candidate=owner_bid_candidate,
        owner_bid_transport=owner_bid_transport,
        structure_repair=structure_repair,
        structure_transport=structure_transport,
        protocol_resources=protocol_resources,
    )


def certified_hierarchical_isac_control(
    lower_coefficient: np.ndarray,
    upper_coefficient: np.ndarray,
    initial_selected: np.ndarray,
    initial_role: np.ndarray,
    positions: np.ndarray,
    existing_communication_power_w: np.ndarray,
    existing_sensing_power_w: np.ndarray,
    *,
    communication_model: CommunicationTransport,
    control_period_s: float,
    p_fa: float,
    config: CertifiedHierarchicalControllerConfig,
) -> CertifiedHierarchicalDecision:
    """Run the cheapest certifiably useful layer and otherwise preserve No-op."""
    lower, upper = _validated_envelope(lower_coefficient, upper_coefficient)
    selected = np.asarray(initial_selected, dtype=bool)
    role = np.asarray(initial_role, dtype=np.int8).reshape(-1)
    positions_array = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(
        existing_communication_power_w, dtype=np.float64).reshape(-1)
    sensing = np.asarray(existing_sensing_power_w, dtype=np.float64)
    K, _, Q = lower.shape
    if (
        selected.shape != lower.shape or role.shape != (K,)
        or positions_array.shape != (K, 3) or comm.shape != (K,)
        or sensing.shape != (K, Q)
        or np.any(~np.isfinite(positions_array))
        or np.any(~np.isfinite(comm)) or np.any(~np.isfinite(sensing))
        or np.any(comm < 0.0) or np.any(comm >= 1.0)
        or np.any(sensing < 0.0)
        or np.any(comm + np.sum(sensing, axis=1) > 1.0 + 1.0e-9)
    ):
        raise ValueError("hierarchical controller state is outside support")
    inferred_role, owner = role_owner_from_structure(
        selected, fallback_role=role)
    if not np.array_equal(inferred_role, role) or np.any(owner < 0):
        raise ValueError("initial role/owner state is inconsistent")
    lower_gain, _ = fixed_owner_gain_matrix(
        lower, [tuple(edge) for edge in np.argwhere(selected)])
    upper_gain, _ = fixed_owner_gain_matrix(
        upper, [tuple(edge) for edge in np.argwhere(selected)])
    noop_lower_pd = compute_detection_probabilities(
        np.sum(lower_gain * sensing, axis=0), float(p_fa))
    noop_upper_pd = compute_detection_probabilities(
        np.sum(upper_gain * sensing, axis=0), float(p_fa))
    route_certificate = certified_hierarchical_qos_route(
        upper,
        [tuple(edge) for edge in np.argwhere(selected)],
        1.0 - comm,
        current_lower_pd=float(np.min(noop_lower_pd)),
        p_fa=float(p_fa),
        qos_floor=float(config.qos_floor),
        target_pair_limit=int(config.target_pair_limit),
        reports_per_receiver=int(config.reports_per_receiver),
    )
    route = route_certificate.decision.route
    if route in (RepairRoute.NO_OP, RepairRoute.GEOMETRY, RepairRoute.HOLD_UNVERIFIED):
        return _noop_decision(
            route_certificate=route_certificate,
            reason=route_certificate.decision.reason,
            selected=selected,
            role=role,
            owner=owner,
            sensing_power_w=sensing,
            communication_power_w=comm,
            noop_lower_pd=noop_lower_pd,
            noop_upper_pd=noop_upper_pd,
        )

    if route == RepairRoute.POWER:
        power_decision = certified_fixed_structure_maxmin_power_repair(
            lower,
            upper,
            [tuple(edge) for edge in np.argwhere(selected)],
            positions_array,
            comm,
            sensing,
            communication_model=communication_model,
            control_period_s=float(control_period_s),
            p_fa=float(p_fa),
            config=config.power,
        )
        extra_comm = bool(np.any(
            power_decision.candidate_communication_power_w
            > comm + 1.0e-12))
        if (
            not power_decision.accepted
            or (config.require_pre_reserved_comm_power and extra_comm)
        ):
            reason = (
                "power:unreserved_comm"
                if power_decision.accepted and extra_comm
                else f"power:{power_decision.reason}"
            )
            return _noop_decision(
                route_certificate=route_certificate,
                reason=reason,
                selected=selected,
                role=role,
                owner=owner,
                sensing_power_w=sensing,
                communication_power_w=comm,
                noop_lower_pd=noop_lower_pd,
                noop_upper_pd=noop_upper_pd,
                power_decision=power_decision,
            )
        return CertifiedHierarchicalDecision(
            accepted=True,
            route=route,
            reason="accepted:fixed_structure_power",
            selected=selected.copy(),
            role=role.copy(),
            owner=owner.copy(),
            sensing_power_w=power_decision.sensing_power_w.copy(),
            communication_power_w=power_decision.communication_power_w.copy(),
            noop_lower_pd=noop_lower_pd,
            noop_upper_pd=noop_upper_pd,
            candidate_lower_pd=power_decision.candidate_lower_pd.copy(),
            candidate_upper_pd=power_decision.candidate_upper_pd.copy(),
            route_certificate=route_certificate,
            power_decision=power_decision,
            owner_bid_candidate=None,
            owner_bid_transport=None,
            structure_repair=None,
            structure_transport=None,
            protocol_resources=HierarchicalProtocolResources(
                feasible=True,
                reasons=tuple(),
                total_over_air_bits=int(
                    power_decision.transport.total_over_air_bits),
                total_protocol_latency_s=float(
                    power_decision.transport.total_protocol_latency_s),
                total_energy_j=float(power_decision.transport.total_energy_j),
                projected_comm_power_w=(
                    power_decision.communication_power_w.copy()),
                max_isac_power_balance_error_w=float(np.max(np.abs(
                    power_decision.communication_power_w
                    + np.sum(power_decision.sensing_power_w, axis=1) - 1.0
                ))),
            ),
        )

    if route != RepairRoute.STRUCTURE_POWER:
        raise AssertionError("unhandled hierarchical route")

    bid_layout = OwnerBidWireLayout(
        num_agents=K,
        num_targets=Q,
        owners_per_target=min(int(config.owners_per_target), K),
    )
    bid_transport = certify_owner_bid_transport(
        positions_array,
        comm,
        communication_model=communication_model,
        layout=bid_layout,
        control_period_s=float(control_period_s),
        snr_margin_db=float(config.snr_margin_db),
        latency_margin_s=float(config.latency_margin_s),
    )
    bid_candidate = owner_bid_sparse_candidate_mask(
        lower,
        upper,
        1.0 - bid_transport.projected_comm_power_w,
        target_pair_limit=int(config.target_pair_limit),
        owners_per_target=int(config.owners_per_target),
        incumbent_selected=[tuple(edge) for edge in np.argwhere(selected)],
    )
    bid_extra = bool(np.any(
        bid_transport.projected_comm_power_w > comm + 1.0e-12))
    if (
        not bid_transport.feasible
        or (config.require_pre_reserved_comm_power and bid_extra)
    ):
        reason = (
            "owner_bid:unreserved_comm"
            if bid_transport.feasible and bid_extra
            else "owner_bid:" + ",".join(bid_transport.reasons)
        )
        return _noop_decision(
            route_certificate=route_certificate,
            reason=reason,
            selected=selected,
            role=role,
            owner=owner,
            sensing_power_w=sensing,
            communication_power_w=comm,
            noop_lower_pd=noop_lower_pd,
            noop_upper_pd=noop_upper_pd,
            owner_bid_candidate=bid_candidate,
            owner_bid_transport=bid_transport,
        )

    budget = 1.0 - bid_transport.projected_comm_power_w
    incumbent = sensing.copy()
    for transmitter in range(K):
        total = float(np.sum(incumbent[transmitter]))
        if total > budget[transmitter] and total > 0.0:
            incumbent[transmitter] *= budget[transmitter] / total
    qos_deflection = minimum_deflection_for_detection_probability(
        np.full(Q, float(config.qos_floor), dtype=np.float64), float(p_fa))
    noop_upper_deflection = np.sum(upper_gain * sensing, axis=0)
    target_reserve_deflection = np.minimum(
        noop_upper_deflection, qos_deflection)
    structural = dual_guided_atomic_structure_repair(
        selected,
        lower,
        bid_candidate.candidate_mask,
        role,
        budget,
        target_pair_limit=int(config.target_pair_limit),
        reports_per_receiver=int(config.reports_per_receiver),
        rounds=int(config.power.rounds),
        ranking_rounds=int(config.structure_ranking_rounds),
        price_bits=int(config.power.price_bits),
        feedback_bits=int(config.power.feedback_bits),
        top_m=int(config.structure_top_m),
        max_steps=int(config.structure_max_steps),
        weak_target_count=min(int(config.structure_weak_target_count), Q),
        proxy_mode="dual_interval",
        # Exploit a quantized feasible improvement before using the valid dual
        # upper to explore.  Execution remains pessimistic and reserve-aware.
        interval_policy="certificate_then_upper",
        owner_proposal_mode=True,
        minimum_deflection=target_reserve_deflection,
        incumbent_power_w=incumbent,
    )
    if not structural.accepted:
        return _noop_decision(
            route_certificate=route_certificate,
            reason="structure:no_certified_atomic_improvement",
            selected=selected,
            role=role,
            owner=owner,
            sensing_power_w=sensing,
            communication_power_w=comm,
            noop_lower_pd=noop_lower_pd,
            noop_upper_pd=noop_upper_pd,
            owner_bid_candidate=bid_candidate,
            owner_bid_transport=bid_transport,
            structure_repair=structural,
        )

    final_power = structural.power_result.power_w
    row_mass = np.sum(final_power, axis=1, keepdims=True)
    if np.any(row_mass <= 0.0):
        raise AssertionError("structural power result lacks sensing weights")
    sequence = certify_top1_structure_sequence_transport(
        selected,
        role,
        owner,
        structural.verified_moves,
        structural.accepted_moves,
        positions=positions_array,
        existing_comm_power_w=bid_transport.projected_comm_power_w,
        final_sensing_weights=final_power / row_mass,
        communication_model=communication_model,
        power_layout=PowerRepairWireLayout(
            K,
            Q,
            rounds=int(config.power.rounds),
            price_bits=int(config.power.price_bits),
            deflection_bits=int(config.power.feedback_bits),
        ),
        control_period_s=float(control_period_s),
        snr_margin_db=float(config.snr_margin_db),
        latency_margin_s=float(config.latency_margin_s),
        owner_proposal_rounds=structural.owner_proposal_rounds,
        owner_proposal_layout=OwnerProposalWireLayout(
            K,
            Q,
            target_pair_limit=int(config.target_pair_limit),
            target_budget=2,
            proposal_budget=K,
        ),
    )
    total_latency = float(
        bid_transport.total_protocol_latency_s
        + sequence.total_protocol_latency_s)
    total_bits = int(
        bid_transport.total_over_air_bits + sequence.total_over_air_bits)
    total_energy = float(
        bid_transport.total_energy_j + sequence.total_energy_j)
    final_comm = sequence.projected_comm_power_w.copy()
    final_sensing = sequence.projected_sensing_power_w.copy()
    total_power_error = float(np.max(np.abs(
        final_comm + np.sum(final_sensing, axis=1) - 1.0)))
    reasons = []
    if not sequence.feasible:
        reasons.extend(f"sequence:{reason}" for reason in sequence.reasons)
    if total_latency > float(control_period_s) + 1.0e-12:
        reasons.append("combined:control_period")
    if config.require_pre_reserved_comm_power and np.any(
        final_comm > comm + 1.0e-12
    ):
        reasons.append("combined:unreserved_comm")
    if total_power_error > 1.0e-9:
        reasons.append("combined:rf_budget")

    candidate_lower_gain, _ = fixed_owner_gain_matrix(
        lower, [tuple(edge) for edge in np.argwhere(structural.selected)])
    candidate_upper_gain, _ = fixed_owner_gain_matrix(
        upper, [tuple(edge) for edge in np.argwhere(structural.selected)])
    candidate_lower_pd = compute_detection_probabilities(
        np.sum(candidate_lower_gain * final_sensing, axis=0), float(p_fa))
    candidate_upper_pd = compute_detection_probabilities(
        np.sum(candidate_upper_gain * final_sensing, axis=0), float(p_fa))
    protected = np.minimum(noop_upper_pd, float(config.qos_floor))
    tolerance = float(config.probability_tolerance)
    improvement = float(
        np.min(candidate_lower_pd) - np.min(noop_upper_pd))
    if np.any(candidate_lower_pd + tolerance < protected):
        reasons.append("combined:target_protection")
    if improvement <= float(config.minimum_worst_pd_improvement) + tolerance:
        reasons.append("combined:no_certified_improvement")
    resources = HierarchicalProtocolResources(
        feasible=not reasons,
        reasons=tuple(reasons),
        total_over_air_bits=total_bits,
        total_protocol_latency_s=total_latency,
        total_energy_j=total_energy,
        projected_comm_power_w=final_comm,
        max_isac_power_balance_error_w=total_power_error,
    )
    if reasons:
        return _noop_decision(
            route_certificate=route_certificate,
            reason=",".join(reasons),
            selected=selected,
            role=role,
            owner=owner,
            sensing_power_w=sensing,
            communication_power_w=comm,
            noop_lower_pd=noop_lower_pd,
            noop_upper_pd=noop_upper_pd,
            owner_bid_candidate=bid_candidate,
            owner_bid_transport=bid_transport,
            structure_repair=structural,
            structure_transport=sequence,
            protocol_resources=resources,
        )
    return CertifiedHierarchicalDecision(
        accepted=True,
        route=route,
        reason="accepted:joint_structure_power",
        selected=structural.selected.copy(),
        role=structural.role.copy(),
        owner=structural.owner.copy(),
        sensing_power_w=final_sensing,
        communication_power_w=final_comm,
        noop_lower_pd=noop_lower_pd,
        noop_upper_pd=noop_upper_pd,
        candidate_lower_pd=candidate_lower_pd,
        candidate_upper_pd=candidate_upper_pd,
        route_certificate=route_certificate,
        power_decision=None,
        owner_bid_candidate=bid_candidate,
        owner_bid_transport=bid_transport,
        structure_repair=structural,
        structure_transport=sequence,
        protocol_resources=resources,
    )
