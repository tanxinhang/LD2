"""End-to-end transport certificate for Top-1 atomic structure search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uav_isac.coordination.dependency_commit import (
    DependencyCommitLayout,
    minimum_uniform_control_reserve,
)
from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.owner_proposal_transport import (
    OwnerProposalWireLayout,
    RankedOwnerProposal,
    certify_owner_proposal_transport,
)
from uav_isac.coordination.power_repair_transport import (
    PowerRepairWireLayout,
    certify_power_repair_transport,
)
from uav_isac.coordination.target_invariant_transport import (
    TargetInvariantToken,
    TargetInvariantWireLayout,
    certify_target_invariant_transport,
)
from uav_isac.domain.communication import CommunicationTransport


@dataclass(frozen=True)
class StructureSequenceStepTransport:
    step: int
    proposal_count: int
    proposal_bits: int
    proposal_latency_s: float
    proposal_energy_j: float
    verification_bits: int
    verification_latency_s: float
    verification_energy_j: float
    committed: bool
    commit_bits: int
    commit_latency_s: float
    commit_energy_j: float
    feasible: bool


@dataclass(frozen=True)
class StructureSequenceTransportCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    invariant_packet_count: int
    invariant_record_count: int
    invariant_bits: int
    invariant_latency_s: float
    invariant_energy_j: float
    proposal_count: int
    proposal_bits: int
    proposal_latency_s: float
    proposal_energy_j: float
    verification_count: int
    commit_count: int
    baseline_verification_bits: int
    baseline_verification_latency_s: float
    baseline_verification_energy_j: float
    verification_bits: int
    commit_bits: int
    total_over_air_bits: int
    max_packet_latency_s: float
    total_protocol_latency_s: float
    total_energy_j: float
    projected_comm_power_w: np.ndarray
    projected_sensing_power_w: np.ndarray
    max_isac_power_balance_error_w: float
    steps: tuple[StructureSequenceStepTransport, ...]


def _same_move(left: LocalMove, right: LocalMove) -> bool:
    return bool(
        np.array_equal(left.selected, right.selected)
        and np.array_equal(left.role, right.role)
        and np.array_equal(left.owner, right.owner)
    )


def certify_top1_structure_sequence_transport(
    initial_selected: np.ndarray,
    initial_role: np.ndarray,
    initial_owner: np.ndarray,
    verified_moves: Sequence[LocalMove],
    accepted_moves: Sequence[LocalMove],
    *,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    final_sensing_weights: np.ndarray,
    communication_model: CommunicationTransport,
    power_layout: PowerRepairWireLayout,
    control_period_s: float,
    reserve_upper_w: float = 0.25,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
    commit_layout: DependencyCommitLayout | None = None,
    owner_proposal_rounds: Sequence[
        Sequence[RankedOwnerProposal]
    ] = (),
    owner_proposal_layout: OwnerProposalWireLayout | None = None,
    target_invariant_tokens: Sequence[TargetInvariantToken] = (),
    target_invariant_layout: TargetInvariantWireLayout | None = None,
) -> StructureSequenceTransportCertificate:
    """Charge every exact verification and every accepted atomic commit.

    This certificate intentionally supports the communication-bounded Top-1
    search contract.  Each accepted move must be the corresponding verified
    move; at most one final verified-but-rejected candidate is allowed.  The
    sequence interleaves one fixed-owner power verification with an optional
    prepare/vote/decision commit and enforces one aggregate control deadline.
    """
    selected = np.asarray(initial_selected, dtype=bool).copy()
    role = np.asarray(initial_role, dtype=np.int8).reshape(-1).copy()
    owner = np.asarray(initial_owner, dtype=np.int64).reshape(-1).copy()
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(existing_comm_power_w, dtype=np.float64).reshape(-1).copy()
    weights = np.asarray(final_sensing_weights, dtype=np.float64)
    K, K2, Q = selected.shape
    if K != K2 or role.shape != (K,) or owner.shape != (Q,):
        raise ValueError("initial structure/role/owner dimensions are inconsistent")
    if pos.shape != (K, 3) or comm.shape != (K,) or weights.shape != (K, Q):
        raise ValueError("positions, communication power or sensing weights mismatch")
    if power_layout.num_agents != K or power_layout.num_targets != Q:
        raise ValueError("power wire layout dimensions must match structure")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(~np.isfinite(weights)) or np.any(comm < 0.0)
        or np.any(comm >= 1.0) or np.any(weights < 0.0)
    ):
        raise ValueError("transport inputs must be finite and physically supported")
    weight_sum = np.sum(weights, axis=1, keepdims=True)
    if np.any(weight_sum <= 0.0):
        raise ValueError("every UAV requires positive sensing-weight mass")
    weights = weights / weight_sum
    verified = tuple(verified_moves)
    accepted = tuple(accepted_moves)
    proposal_rounds = tuple(tuple(round_) for round_ in owner_proposal_rounds)
    if proposal_rounds and len(proposal_rounds) != len(verified):
        raise ValueError("every verified move requires one owner-proposal round")
    if proposal_rounds and owner_proposal_layout is None:
        raise ValueError("owner proposal rounds require an explicit wire layout")
    invariant_tokens = tuple(target_invariant_tokens)
    if invariant_tokens and target_invariant_layout is None:
        raise ValueError(
            "target invariant tokens require an explicit wire layout")
    if target_invariant_layout is not None and (
        target_invariant_layout.num_agents != K
        or target_invariant_layout.num_targets != Q
    ):
        raise ValueError(
            "target invariant wire dimensions must match structure")
    if len(accepted) > len(verified) or len(verified) > len(accepted) + 1:
        raise ValueError(
            "Top-1 sequence permits at most one rejected final verification")
    for index, move in enumerate(accepted):
        if not _same_move(move, verified[index]):
            raise ValueError("accepted Top-1 move must match its verified candidate")

    reasons: list[str] = []
    step_reports: list[StructureSequenceStepTransport] = []
    invariant_packet_count = 0
    invariant_record_count = 0
    invariant_bits = 0
    invariant_latency = 0.0
    invariant_energy = 0.0
    invariant_max_packet_latency = 0.0
    invariant_feasible = True
    if invariant_tokens:
        invariant_transport = certify_target_invariant_transport(
            invariant_tokens,
            positions=pos,
            existing_comm_power_w=comm,
            communication_model=communication_model,
            layout=target_invariant_layout,
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        invariant_packet_count = int(invariant_transport.packet_count)
        invariant_record_count = int(invariant_transport.record_count)
        invariant_bits = int(invariant_transport.total_over_air_bits)
        invariant_latency = float(
            invariant_transport.total_protocol_latency_s)
        invariant_energy = float(invariant_transport.total_energy_j)
        invariant_max_packet_latency = float(
            invariant_transport.max_packet_latency_s)
        invariant_feasible = bool(invariant_transport.feasible)
        if invariant_feasible:
            comm = np.maximum(
                comm, invariant_transport.projected_comm_power_w)
        else:
            reasons.extend(invariant_transport.reasons)
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
    baseline_bits = int(baseline.total_over_air_bits)
    baseline_latency = float(baseline.total_protocol_latency_s)
    baseline_energy = float(baseline.total_energy_j)
    verification_bits = baseline_bits
    proposal_bits = 0
    proposal_latency = 0.0
    proposal_energy = 0.0
    processed_proposals = 0
    commit_bits = 0
    total_latency = invariant_latency + baseline_latency
    max_packet_latency = max(
        invariant_max_packet_latency,
        float(baseline.max_packet_latency_s),
    )
    total_energy = invariant_energy + baseline_energy
    processed_verifications = 1
    processed_commits = 0
    if baseline.feasible:
        comm = np.maximum(comm, baseline.projected_comm_power_w)
    else:
        reasons.extend(f"baseline:{reason}" for reason in baseline.reasons)
    packet_layout = commit_layout or DependencyCommitLayout(K, Q)
    identity = np.zeros(K, dtype=np.int64)
    prefix_feasible = bool(invariant_feasible and baseline.feasible)
    for index, candidate in enumerate(verified if prefix_feasible else ()):
        step_proposal_count = 0
        step_proposal_bits = 0
        step_proposal_latency = 0.0
        step_proposal_energy = 0.0
        if proposal_rounds:
            round_proposals = proposal_rounds[index]
            matching = [
                item for item in round_proposals
                if _same_move(item.move, candidate)
            ]
            if not matching:
                raise ValueError("verified move is absent from its proposal round")
            edge_changes = lambda item: int(np.sum(
                item.move.selected != selected))
            best_key = max(
                (
                    int(bool(item.certified_improvement)),
                    float(item.score), float(item.lower), -edge_changes(item),
                )
                for item in round_proposals
            )
            candidate_key = max(
                (
                    int(bool(item.certified_improvement)),
                    float(item.score), float(item.lower), -edge_changes(item),
                )
                for item in matching
            )
            if candidate_key < best_key:
                raise ValueError("verified move is not the owner-proposal Top-1")
            proposal = certify_owner_proposal_transport(
                selected,
                role,
                owner,
                round_proposals,
                positions=pos,
                existing_comm_power_w=comm,
                communication_model=communication_model,
                layout=owner_proposal_layout,
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
            step_proposal_count = int(proposal.proposal_count)
            step_proposal_bits = int(proposal.total_over_air_bits)
            step_proposal_latency = float(proposal.total_protocol_latency_s)
            step_proposal_energy = float(proposal.total_energy_j)
            processed_proposals += step_proposal_count
            proposal_bits += step_proposal_bits
            proposal_latency += step_proposal_latency
            proposal_energy += step_proposal_energy
            total_latency += step_proposal_latency
            total_energy += step_proposal_energy
            max_packet_latency = max(
                max_packet_latency, float(proposal.max_packet_latency_s))
            if proposal.feasible:
                comm = np.maximum(comm, proposal.projected_comm_power_w)
            else:
                reasons.extend(
                    f"proposal:{index}:{reason}"
                    for reason in proposal.reasons)
                step_reports.append(StructureSequenceStepTransport(
                    step=index,
                    proposal_count=step_proposal_count,
                    proposal_bits=step_proposal_bits,
                    proposal_latency_s=step_proposal_latency,
                    proposal_energy_j=step_proposal_energy,
                    verification_bits=0,
                    verification_latency_s=0.0,
                    verification_energy_j=0.0,
                    committed=False,
                    commit_bits=0,
                    commit_latency_s=0.0,
                    commit_energy_j=0.0,
                    feasible=False,
                ))
                break
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
        verification_bits += int(verification.total_over_air_bits)
        processed_verifications += 1
        max_packet_latency = max(
            max_packet_latency, float(verification.max_packet_latency_s))
        total_latency += float(verification.total_protocol_latency_s)
        total_energy += float(verification.total_energy_j)
        if verification.feasible:
            comm = np.maximum(comm, verification.projected_comm_power_w)
        else:
            reasons.extend(
                f"verification:{index}:{reason}"
                for reason in verification.reasons)

        committed = index < len(accepted)
        step_commit_bits = 0
        step_commit_latency = 0.0
        step_commit_energy = 0.0
        commit_feasible = True
        if committed and verification.feasible:
            sensing = (1.0 - comm[:, None]) * weights
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
                layout=packet_layout,
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
            commit_feasible = bool(reserve.feasible)
            certificate = reserve.certificate
            step_commit_bits = int(certificate.total_over_air_bits)
            step_commit_latency = float(certificate.total_latency_s)
            step_commit_energy = float(certificate.total_energy_j)
            max_packet_latency = max(
                max_packet_latency,
                max((round_.max_link_latency_s for round_ in certificate.rounds),
                    default=0.0),
            )
            commit_bits += step_commit_bits
            total_latency += step_commit_latency
            total_energy += step_commit_energy
            if reserve.feasible:
                processed_commits += 1
                comm = np.maximum(
                    comm, np.asarray(reserve.comm_power_w, dtype=np.float64))
                selected = candidate.selected.copy()
                role = candidate.role.copy()
                owner = candidate.owner.copy()
            else:
                reasons.extend(
                    f"commit:{index}:{reason}"
                    for reason in certificate.reasons)
        elif committed:
            commit_feasible = False
            reasons.append(f"commit:{index}:verification_failed")

        step_reports.append(StructureSequenceStepTransport(
            step=index,
            proposal_count=step_proposal_count,
            proposal_bits=step_proposal_bits,
            proposal_latency_s=step_proposal_latency,
            proposal_energy_j=step_proposal_energy,
            verification_bits=int(verification.total_over_air_bits),
            verification_latency_s=float(
                verification.total_protocol_latency_s),
            verification_energy_j=float(verification.total_energy_j),
            committed=committed,
            commit_bits=step_commit_bits,
            commit_latency_s=step_commit_latency,
            commit_energy_j=step_commit_energy,
            feasible=bool(verification.feasible and commit_feasible),
        ))
        if committed and not commit_feasible:
            break

    if total_latency > float(control_period_s) + 1.0e-12:
        reasons.append("sequence:control_period")
    projected_sensing = (1.0 - comm[:, None]) * weights
    power_error = float(np.max(np.abs(
        comm + np.sum(projected_sensing, axis=1) - 1.0)))
    return StructureSequenceTransportCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        invariant_packet_count=invariant_packet_count,
        invariant_record_count=invariant_record_count,
        invariant_bits=invariant_bits,
        invariant_latency_s=float(invariant_latency),
        invariant_energy_j=float(invariant_energy),
        proposal_count=processed_proposals,
        proposal_bits=proposal_bits,
        proposal_latency_s=float(proposal_latency),
        proposal_energy_j=float(proposal_energy),
        verification_count=processed_verifications,
        commit_count=processed_commits,
        baseline_verification_bits=baseline_bits,
        baseline_verification_latency_s=baseline_latency,
        baseline_verification_energy_j=baseline_energy,
        verification_bits=verification_bits,
        commit_bits=commit_bits,
        total_over_air_bits=int(
            invariant_bits + proposal_bits + verification_bits + commit_bits),
        max_packet_latency_s=float(max_packet_latency),
        total_protocol_latency_s=float(total_latency),
        total_energy_j=float(total_energy),
        projected_comm_power_w=comm,
        projected_sensing_power_w=projected_sensing,
        max_isac_power_balance_error_w=power_error,
        steps=tuple(step_reports),
    )
