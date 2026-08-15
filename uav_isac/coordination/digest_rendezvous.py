"""Bounded digest rendezvous for set-valued structure consensus.

The digest is only a screening key.  It never authorizes a transition: the
winning owner sends the bounded atomic proposal descriptor and the coordinator
must reconstruct and compare the full ``(selected, role, owner)`` state before
the certificate can be feasible.  Thus a hash collision causes rejection,
not an unsafe commit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_isac.coordination.owner_proposal_transport import (
    OwnerProposalWireLayout,
    RankedOwnerProposal,
    certify_owner_proposal_transport,
)
from uav_isac.coordination.dependency_commit import (
    DependencyCommitLayout,
    minimum_uniform_control_reserve,
)
from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.power_repair_transport import (
    PowerRepairWireLayout,
    _required_power_for_packet,
    certify_power_repair_transport,
)
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.coordination.target_invariant_transport import (
    TargetInvariantToken,
    TargetInvariantWireLayout,
    certify_target_invariant_transport,
)


def _index_bits(cardinality: int) -> int:
    return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))


@dataclass(frozen=True)
class StructureDigestRendezvousLayout:
    """Fixed three-phase request/reply/proposal wire layout."""

    num_agents: int
    num_targets: int
    target_pair_limit: int
    header_bits: int = 64
    epoch_bits: int = 16
    digest_bits: int = 256
    reply_flag_bits: int = 1

    def __post_init__(self) -> None:
        if int(self.num_agents) < 2 or int(self.num_targets) < 1:
            raise ValueError("digest rendezvous requires K>=2 and Q>=1")
        if int(self.target_pair_limit) < 1:
            raise ValueError("target pair limit must be positive")
        if int(self.digest_bits) < 128:
            raise ValueError("rendezvous digest must contain at least 128 bits")
        for name in ("header_bits", "epoch_bits", "reply_flag_bits"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def node_bits(self) -> int:
        return _index_bits(self.num_agents)

    @property
    def request_packet_bits(self) -> int:
        # Coordinator id is explicit so replies are independently routable.
        return int(
            self.header_bits + self.epoch_bits + self.digest_bits
            + self.node_bits
        )

    @property
    def reply_packet_bits(self) -> int:
        # Every non-coordinator replies, including negative matches.  The
        # fixed traffic pattern avoids data-dependent latency accounting.
        return int(
            self.header_bits + self.epoch_bits + self.digest_bits
            + self.node_bits + self.reply_flag_bits
        )

    @property
    def collision_probability_upper(self) -> float:
        return float(2.0 ** (-int(self.digest_bits)))

    def proposal_layout(self) -> OwnerProposalWireLayout:
        return OwnerProposalWireLayout(
            num_agents=int(self.num_agents),
            num_targets=int(self.num_targets),
            target_pair_limit=int(self.target_pair_limit),
            target_budget=2,
            proposal_budget=1,
            header_bits=int(self.header_bits),
            epoch_bits=int(self.epoch_bits),
            digest_bits=int(self.digest_bits),
        )


@dataclass(frozen=True)
class StructureDigestRendezvousCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    coordinator: int
    proposer: int
    full_structure_verified: bool
    digest_bits: int
    collision_probability_upper: float
    request_bits: int
    reply_bits: int
    proposal_bits: int
    total_over_air_bits: int
    request_latency_s: float
    reply_latency_s: float
    proposal_latency_s: float
    total_protocol_latency_s: float
    total_energy_j: float
    min_snr_db: float
    required_comm_power_w: np.ndarray
    projected_comm_power_w: np.ndarray


@dataclass(frozen=True)
class RendezvousCandidateSuffixCertificate:
    """Conservative exact verification plus dependency-commit suffix."""

    feasible: bool
    reasons: tuple[str, ...]
    verification_bits: int
    verification_latency_s: float
    verification_energy_j: float
    dependency_commit_reused: bool
    commit_bits: int
    commit_latency_s: float
    commit_energy_j: float
    total_over_air_bits: int
    total_protocol_latency_s: float
    total_energy_j: float
    projected_comm_power_w: np.ndarray
    projected_sensing_power_w: np.ndarray
    max_isac_power_balance_error_w: float


@dataclass(frozen=True)
class DeferredHorizonPrefixCertificate:
    """Transport required before any candidate-specific horizon commit."""

    feasible: bool
    reasons: tuple[str, ...]
    invariant_bits: int
    invariant_latency_s: float
    invariant_energy_j: float
    baseline_bits: int
    baseline_latency_s: float
    baseline_energy_j: float
    proposal_count: int
    proposal_bits: int
    proposal_latency_s: float
    proposal_energy_j: float
    total_over_air_bits: int
    total_protocol_latency_s: float
    total_energy_j: float
    projected_comm_power_w: np.ndarray


def _candidate_certificate(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    proposal: RankedOwnerProposal,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    *,
    coordinator: int,
    full_structure_verified: bool,
    communication_model: InterUAVCommunicationModel,
    layout: StructureDigestRendezvousLayout,
    control_period_s: float,
    snr_margin_db: float,
    latency_margin_s: float,
) -> StructureDigestRendezvousCertificate:
    K = int(layout.num_agents)
    coordinator_id = int(coordinator)
    proposer = int(proposal.proposer)
    request_bits = int(layout.request_packet_bits)
    reply_packet_bits = int(layout.reply_packet_bits)
    proposal_bits = int(layout.proposal_layout().proposal_bits(
        selected, role, owner, proposal))
    reply_senders = [node for node in range(K) if node != coordinator_id]
    reply_bandwidth = (
        communication_model.bandwidth_hz / len(reply_senders)
        if reply_senders else communication_model.bandwidth_hz
    )

    required = np.zeros(K, dtype=np.float64)
    for receiver in reply_senders:
        required[coordinator_id] = max(
            required[coordinator_id],
            _required_power_for_packet(
                communication_model,
                positions[coordinator_id],
                positions[receiver],
                request_bits,
                communication_model.bandwidth_hz,
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            ),
        )
    for sender in reply_senders:
        required[sender] = max(
            required[sender],
            _required_power_for_packet(
                communication_model,
                positions[sender],
                positions[coordinator_id],
                reply_packet_bits,
                reply_bandwidth,
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            ),
        )
    if proposer != coordinator_id:
        required[proposer] = max(
            required[proposer],
            _required_power_for_packet(
                communication_model,
                positions[proposer],
                positions[coordinator_id],
                proposal_bits,
                communication_model.bandwidth_hz,
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            ),
        )

    projected = np.maximum(existing_comm_power_w, required)
    reasons: list[str] = []
    if np.any(~np.isfinite(projected)) or np.any(projected >= 1.0):
        reasons.append("power:unavailable")
    if not bool(full_structure_verified):
        reasons.append("identity:full_structure_mismatch")

    request_latency = 0.0
    request_serialization = 0.0
    reply_latency = 0.0
    reply_energy = 0.0
    proposal_latency = 0.0
    proposal_energy = 0.0
    min_snr = float("inf")
    for receiver in reply_senders:
        snr, _, serialization, latency = (
            communication_model.robust_link_budget(
                positions[coordinator_id],
                positions[receiver],
                request_bits,
                communication_model.bandwidth_hz,
                float(projected[coordinator_id]),
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
        )
        min_snr = min(min_snr, float(snr))
        request_latency = max(request_latency, float(latency))
        request_serialization = max(
            request_serialization, float(serialization))
        if snr < communication_model.snr_threshold_db:
            reasons.append(
                f"request:snr:{coordinator_id}->{receiver}")
        if latency > communication_model.deadline_s + 1.0e-12:
            reasons.append(
                f"request:deadline:{coordinator_id}->{receiver}")

    for sender in reply_senders:
        snr, _, serialization, latency = (
            communication_model.robust_link_budget(
                positions[sender],
                positions[coordinator_id],
                reply_packet_bits,
                reply_bandwidth,
                float(projected[sender]),
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
        )
        min_snr = min(min_snr, float(snr))
        reply_latency = max(reply_latency, float(latency))
        reply_energy += float(projected[sender]) * float(serialization)
        if snr < communication_model.snr_threshold_db:
            reasons.append(f"reply:snr:{sender}->{coordinator_id}")
        if latency > communication_model.deadline_s + 1.0e-12:
            reasons.append(f"reply:deadline:{sender}->{coordinator_id}")

    if proposer != coordinator_id:
        snr, _, serialization, latency = (
            communication_model.robust_link_budget(
                positions[proposer],
                positions[coordinator_id],
                proposal_bits,
                communication_model.bandwidth_hz,
                float(projected[proposer]),
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
        )
        min_snr = min(min_snr, float(snr))
        proposal_latency = float(latency)
        proposal_energy = float(projected[proposer]) * float(serialization)
        if snr < communication_model.snr_threshold_db:
            reasons.append(f"proposal:snr:{proposer}->{coordinator_id}")
        if latency > communication_model.deadline_s + 1.0e-12:
            reasons.append(f"proposal:deadline:{proposer}->{coordinator_id}")

    total_latency = request_latency + reply_latency + proposal_latency
    if total_latency > float(control_period_s) + 1.0e-12:
        reasons.append("protocol:control_period")
    request_energy = (
        float(projected[coordinator_id]) * request_serialization)
    total_energy = request_energy + reply_energy + proposal_energy
    transmitted_proposal_bits = (
        proposal_bits if proposer != coordinator_id else 0)
    total_bits = (
        request_bits + len(reply_senders) * reply_packet_bits
        + transmitted_proposal_bits
    )
    return StructureDigestRendezvousCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        coordinator=coordinator_id,
        proposer=proposer,
        full_structure_verified=bool(full_structure_verified),
        digest_bits=int(layout.digest_bits),
        collision_probability_upper=layout.collision_probability_upper,
        request_bits=request_bits,
        reply_bits=len(reply_senders) * reply_packet_bits,
        proposal_bits=transmitted_proposal_bits,
        total_over_air_bits=int(total_bits),
        request_latency_s=float(request_latency),
        reply_latency_s=float(reply_latency),
        proposal_latency_s=float(proposal_latency),
        total_protocol_latency_s=float(total_latency),
        total_energy_j=float(total_energy),
        min_snr_db=float(min_snr),
        required_comm_power_w=required,
        projected_comm_power_w=projected,
    )


def certify_structure_digest_rendezvous(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    proposal: RankedOwnerProposal,
    *,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    full_structure_verified: bool,
    communication_model: InterUAVCommunicationModel,
    layout: StructureDigestRendezvousLayout,
    control_period_s: float,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> StructureDigestRendezvousCertificate:
    """Elect the minimum-reserve coordinator for a bounded rendezvous."""
    pair = np.asarray(selected, dtype=bool)
    state_role = np.asarray(role, dtype=np.int8).reshape(-1)
    state_owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(existing_comm_power_w, dtype=np.float64).reshape(-1)
    K, K2, Q = pair.shape
    if (
        K != K2 or K != int(layout.num_agents)
        or Q != int(layout.num_targets)
        or state_role.shape != (K,) or state_owner.shape != (Q,)
        or pos.shape != (K, 3) or comm.shape != (K,)
    ):
        raise ValueError("digest rendezvous dimensions are inconsistent")
    proposer = int(proposal.proposer)
    if proposer < 0 or proposer >= K:
        raise ValueError("proposal sender is outside the UAV set")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(comm < 0.0) or np.any(comm >= 1.0)
        or not np.isfinite(float(control_period_s))
        or float(control_period_s) <= 0.0
    ):
        raise ValueError("digest rendezvous state is outside physical support")
    # Validate the bounded descriptor and dependency closure before evaluating
    # any link.  This also proves that the declared proposer participates.
    layout.proposal_layout().proposal_bits(
        pair, state_role, state_owner, proposal)
    certificates = [
        _candidate_certificate(
            pair,
            state_role,
            state_owner,
            proposal,
            pos,
            comm,
            coordinator=coordinator,
            full_structure_verified=bool(full_structure_verified),
            communication_model=communication_model,
            layout=layout,
            control_period_s=float(control_period_s),
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        for coordinator in range(K)
    ]
    return min(certificates, key=lambda item: (
        not item.feasible,
        float(np.max(item.projected_comm_power_w - comm)),
        float(np.sum(item.projected_comm_power_w - comm)),
        item.total_protocol_latency_s,
        item.total_over_air_bits,
        item.coordinator,
    ))


def certify_rendezvous_candidate_suffix(
    initial_selected: np.ndarray,
    initial_role: np.ndarray,
    initial_owner: np.ndarray,
    candidate: LocalMove,
    *,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    final_sensing_weights: np.ndarray,
    communication_model: InterUAVCommunicationModel,
    power_layout: PowerRepairWireLayout,
    control_period_s: float,
    reserve_upper_w: float = 0.25,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
    commit_layout: DependencyCommitLayout | None = None,
    parallel_committed_move: LocalMove | None = None,
    parallel_commit_feasible: bool = False,
) -> RendezvousCandidateSuffixCertificate:
    """Charge a complete conservative post-rendezvous candidate suffix.

    The fixed-owner horizon-power verification is always repeated.  A prior
    dependency commit may be reused only when the caller supplies its committed
    move, marks that commit feasible, and all selected/role/owner arrays match
    exactly.  This is message reuse for the same discrete transaction, not a
    relaxation of the horizon RF certificate.
    """
    selected = np.asarray(initial_selected, dtype=bool)
    role = np.asarray(initial_role, dtype=np.int8).reshape(-1)
    owner = np.asarray(initial_owner, dtype=np.int64).reshape(-1)
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(existing_comm_power_w, dtype=np.float64).reshape(-1)
    weights = np.asarray(final_sensing_weights, dtype=np.float64)
    K, K2, Q = selected.shape
    if (
        K != K2 or role.shape != (K,) or owner.shape != (Q,)
        or pos.shape != (K, 3) or comm.shape != (K,)
        or weights.shape != (K, Q)
        or power_layout.num_agents != K
        or power_layout.num_targets != Q
    ):
        raise ValueError("rendezvous suffix dimensions are inconsistent")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(~np.isfinite(weights)) or np.any(comm < 0.0)
        or np.any(comm >= 1.0) or np.any(weights < 0.0)
        or not np.isfinite(float(control_period_s))
        or float(control_period_s) <= 0.0
    ):
        raise ValueError("rendezvous suffix state is outside physical support")
    row_mass = np.sum(weights, axis=1, keepdims=True)
    if np.any(row_mass <= 0.0):
        raise ValueError("every UAV requires positive sensing-weight mass")
    weights = weights / row_mass

    verification = certify_power_repair_transport(
        candidate.owner,
        pos,
        comm,
        communication_model=communication_model,
        layout=power_layout,
        control_period_s=float(control_period_s),
        snr_margin_db=float(snr_margin_db),
        latency_margin_s=float(latency_margin_s),
    )
    reasons = [
        f"verification:{reason}" for reason in verification.reasons
    ]
    if verification.feasible:
        comm = np.maximum(comm, verification.projected_comm_power_w)

    commit_bits = 0
    commit_latency = 0.0
    commit_energy = 0.0
    commit_reused = bool(
        parallel_commit_feasible
        and parallel_committed_move is not None
        and np.array_equal(
            parallel_committed_move.selected, candidate.selected)
        and np.array_equal(parallel_committed_move.role, candidate.role)
        and np.array_equal(parallel_committed_move.owner, candidate.owner)
    )
    if bool(parallel_commit_feasible) and not commit_reused:
        raise ValueError(
            "a parallel dependency commit can be reused only for the exact "
            "same selected/role/owner transition")
    if verification.feasible and not commit_reused:
        sensing = (1.0 - comm[:, None]) * weights
        identity = np.zeros(K, dtype=np.int64)
        reserve = minimum_uniform_control_reserve(
            selected,
            role,
            owner,
            candidate,
            positions=pos,
            current_comm_power_w=comm,
            current_sensing_power_w=sensing,
            sensing_weights=weights,
            state_versions=identity,
            certificate_epoch_ids=identity,
            certificate_digests=identity,
            communication_model=communication_model,
            reserve_upper_w=float(reserve_upper_w),
            total_power_w=1.0,
            total_deadline_s=float(control_period_s),
            layout=(commit_layout or DependencyCommitLayout(K, Q)),
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        commit_bits = int(reserve.certificate.total_over_air_bits)
        commit_latency = float(reserve.certificate.total_latency_s)
        commit_energy = float(reserve.certificate.total_energy_j)
        if reserve.feasible:
            comm = np.maximum(
                comm, np.asarray(reserve.comm_power_w, dtype=np.float64))
        else:
            reasons.extend(
                f"commit:{reason}" for reason in reserve.certificate.reasons)
    elif not verification.feasible:
        reasons.append("commit:verification_failed")

    total_latency = (
        float(verification.total_protocol_latency_s) + commit_latency)
    if total_latency > float(control_period_s) + 1.0e-12:
        reasons.append("suffix:control_period")
    projected_sensing = (1.0 - comm[:, None]) * weights
    power_error = float(np.max(np.abs(
        comm + np.sum(projected_sensing, axis=1) - 1.0)))
    verification_bits = int(verification.total_over_air_bits)
    verification_energy = float(verification.total_energy_j)
    return RendezvousCandidateSuffixCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        verification_bits=verification_bits,
        verification_latency_s=float(
            verification.total_protocol_latency_s),
        verification_energy_j=verification_energy,
        dependency_commit_reused=commit_reused,
        commit_bits=commit_bits,
        commit_latency_s=commit_latency,
        commit_energy_j=commit_energy,
        total_over_air_bits=verification_bits + commit_bits,
        total_protocol_latency_s=total_latency,
        total_energy_j=verification_energy + commit_energy,
        projected_comm_power_w=comm,
        projected_sensing_power_w=projected_sensing,
        max_isac_power_balance_error_w=power_error,
    )


def certify_deferred_horizon_prefix(
    initial_selected: np.ndarray,
    initial_role: np.ndarray,
    initial_owner: np.ndarray,
    proposals: tuple[RankedOwnerProposal, ...],
    *,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    communication_model: InterUAVCommunicationModel,
    power_layout: PowerRepairWireLayout,
    proposal_layout: OwnerProposalWireLayout,
    control_period_s: float,
    target_invariant_tokens: tuple[TargetInvariantToken, ...] = (),
    target_invariant_layout: TargetInvariantWireLayout | None = None,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> DeferredHorizonPrefixCertificate:
    """Charge invariant, No-op baseline and owner proposals, but no Top-1.

    These three phases are candidate-independent and therefore must precede
    set reconciliation.  Candidate RF verification and dependency commit are
    deliberately absent and remain in the post-rendezvous suffix.
    """
    selected = np.asarray(initial_selected, dtype=bool)
    role = np.asarray(initial_role, dtype=np.int8).reshape(-1)
    owner = np.asarray(initial_owner, dtype=np.int64).reshape(-1)
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(existing_comm_power_w, dtype=np.float64).reshape(-1)
    K, K2, Q = selected.shape
    if (
        K != K2 or role.shape != (K,) or owner.shape != (Q,)
        or pos.shape != (K, 3) or comm.shape != (K,)
        or power_layout.num_agents != K
        or power_layout.num_targets != Q
        or proposal_layout.num_agents != K
        or proposal_layout.num_targets != Q
    ):
        raise ValueError("deferred horizon prefix dimensions are inconsistent")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(comm < 0.0) or np.any(comm >= 1.0)
        or not np.isfinite(float(control_period_s))
        or float(control_period_s) <= 0.0
    ):
        raise ValueError("deferred horizon prefix is outside physical support")
    tokens = tuple(target_invariant_tokens)
    reasons: list[str] = []
    invariant_bits = 0
    invariant_latency = 0.0
    invariant_energy = 0.0
    if tokens:
        if target_invariant_layout is None:
            raise ValueError("target invariant tokens require a wire layout")
        invariant = certify_target_invariant_transport(
            tokens,
            positions=pos,
            existing_comm_power_w=comm,
            communication_model=communication_model,
            layout=target_invariant_layout,
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        invariant_bits = int(invariant.total_over_air_bits)
        invariant_latency = float(invariant.total_protocol_latency_s)
        invariant_energy = float(invariant.total_energy_j)
        if invariant.feasible:
            comm = np.maximum(comm, invariant.projected_comm_power_w)
        else:
            reasons.extend(f"invariant:{reason}" for reason in invariant.reasons)

    baseline = certify_power_repair_transport(
        owner,
        pos,
        comm,
        communication_model=communication_model,
        layout=power_layout,
        control_period_s=float(control_period_s),
        snr_margin_db=float(snr_margin_db),
        latency_margin_s=float(latency_margin_s),
    )
    if baseline.feasible:
        comm = np.maximum(comm, baseline.projected_comm_power_w)
    else:
        reasons.extend(f"baseline:{reason}" for reason in baseline.reasons)

    proposal_bits = 0
    proposal_latency = 0.0
    proposal_energy = 0.0
    items = tuple(proposals)
    if items and baseline.feasible:
        proposal = certify_owner_proposal_transport(
            selected,
            role,
            owner,
            items,
            positions=pos,
            existing_comm_power_w=comm,
            communication_model=communication_model,
            layout=proposal_layout,
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        proposal_bits = int(proposal.total_over_air_bits)
        proposal_latency = float(proposal.total_protocol_latency_s)
        proposal_energy = float(proposal.total_energy_j)
        if proposal.feasible:
            comm = np.maximum(comm, proposal.projected_comm_power_w)
        else:
            reasons.extend(f"proposal:{reason}" for reason in proposal.reasons)
    elif items:
        reasons.append("proposal:baseline_failed")

    baseline_bits = int(baseline.total_over_air_bits)
    baseline_latency = float(baseline.total_protocol_latency_s)
    baseline_energy = float(baseline.total_energy_j)
    total_latency = invariant_latency + baseline_latency + proposal_latency
    if total_latency > float(control_period_s) + 1.0e-12:
        reasons.append("prefix:control_period")
    return DeferredHorizonPrefixCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        invariant_bits=invariant_bits,
        invariant_latency_s=invariant_latency,
        invariant_energy_j=invariant_energy,
        baseline_bits=baseline_bits,
        baseline_latency_s=baseline_latency,
        baseline_energy_j=baseline_energy,
        proposal_count=len(items),
        proposal_bits=proposal_bits,
        proposal_latency_s=proposal_latency,
        proposal_energy_j=proposal_energy,
        total_over_air_bits=(invariant_bits + baseline_bits + proposal_bits),
        total_protocol_latency_s=total_latency,
        total_energy_j=(invariant_energy + baseline_energy + proposal_energy),
        projected_comm_power_w=comm,
    )
