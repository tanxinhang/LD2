"""Bounded owner-proposal wire protocol for atomic structure search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uav_isac.coordination.dependency_commit import dependency_closure
from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.power_repair_transport import (
    _required_power_for_packet,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


def _index_bits(cardinality: int) -> int:
    return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))


def quantize_nonnegative_float16_lower(value: float) -> float:
    """Round a supported non-negative scalar downward in binary16."""
    scalar = float(value)
    limit = float(np.finfo(np.float16).max)
    if not np.isfinite(scalar) or not 0.0 <= scalar <= limit:
        raise ValueError("binary16 lower bound is outside finite support")
    encoded = np.float16(scalar)
    if float(encoded) > scalar:
        encoded = np.nextafter(
            encoded, np.float16(-np.inf), dtype=np.float16)
    decoded = float(encoded)
    if decoded < 0.0 or decoded > scalar:
        raise AssertionError("binary16 lower-bound quantizer lost direction")
    return decoded


def quantize_nonnegative_float16_upper(value: float) -> float:
    """Round a supported non-negative scalar upward in binary16."""
    scalar = float(value)
    limit = float(np.finfo(np.float16).max)
    if not np.isfinite(scalar) or not 0.0 <= scalar <= limit:
        raise ValueError("binary16 upper bound is outside finite support")
    encoded = np.float16(scalar)
    if float(encoded) < scalar:
        encoded = np.nextafter(
            encoded, np.float16(np.inf), dtype=np.float16)
    decoded = float(encoded)
    if not np.isfinite(decoded) or decoded < scalar:
        raise AssertionError("binary16 upper-bound quantizer lost direction")
    return decoded


@dataclass(frozen=True)
class RankedOwnerProposal:
    proposer: int
    move: LocalMove
    lower: float
    upper: float
    score: float
    certified_improvement: bool = False


@dataclass(frozen=True)
class OwnerProposalWireLayout:
    num_agents: int
    num_targets: int
    target_pair_limit: int
    target_budget: int = 2
    proposal_budget: int = 2
    header_bits: int = 64
    epoch_bits: int = 16
    digest_bits: int = 64
    score_bits: int = 16
    proof_flag_bits: int = 1

    def __post_init__(self) -> None:
        if self.num_agents < 2 or self.num_targets < 1:
            raise ValueError("proposal layout requires K>=2 and Q>=1")
        if self.target_pair_limit < 1 or not 1 <= self.target_budget <= 2:
            raise ValueError("proposal target and pair budgets are unsupported")
        if not 1 <= int(self.proposal_budget) <= int(self.num_agents):
            raise ValueError(
                "proposal round budget must lie between one and K")
        if int(self.score_bits) != 16:
            raise ValueError("owner proposal intervals use exactly binary16")
        if int(self.proof_flag_bits) != 1:
            raise ValueError("certified-improvement flag uses exactly one bit")
        for name in (
            "header_bits", "epoch_bits", "digest_bits", "score_bits",
            "proof_flag_bits",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def node_bits(self) -> int:
        return _index_bits(self.num_agents)

    @property
    def target_bits(self) -> int:
        return _index_bits(self.num_targets)

    @property
    def role_bits(self) -> int:
        return _index_bits(3)

    @property
    def max_toggled_edges(self) -> int:
        # Per target: at most L old edges off and L new edges on.
        return int(2 * self.target_budget * self.target_pair_limit)

    def proposal_bits(
        self,
        selected: np.ndarray,
        role: np.ndarray,
        owner: np.ndarray,
        proposal: RankedOwnerProposal,
    ) -> int:
        closure = dependency_closure(
            selected, role, owner, proposal.move)
        if len(closure.affected_targets) > self.target_budget:
            raise ValueError("proposal exceeds the atomic target budget")
        if closure.toggled_edges > self.max_toggled_edges:
            raise ValueError("proposal exceeds the atomic edge-toggle bound")
        if closure.changed_roles > 3 or closure.changed_owners > 2:
            raise ValueError("proposal exceeds N5/N6 role or owner bounds")
        if int(proposal.proposer) not in closure.participants:
            raise ValueError("proposal sender must belong to its dependency closure")
        common = (
            int(self.header_bits) + int(self.epoch_bits)
            + int(self.digest_bits) + self.node_bits
        )
        counts = (
            _index_bits(self.target_budget + 1)
            + _index_bits(4)
            + _index_bits(3)
            + _index_bits(self.max_toggled_edges + 1)
        )
        kind_bits = 1  # N5 versus N6.
        target_records = len(closure.affected_targets) * self.target_bits
        role_records = closure.changed_roles * (
            self.node_bits + self.role_bits)
        owner_records = closure.changed_owners * (
            self.target_bits + self.node_bits)
        edge_records = closure.toggled_edges * (
            2 * self.node_bits + self.target_bits + 1)
        interval_record = 2 * int(self.score_bits)
        return int(
            common + counts + kind_bits + target_records + role_records
            + owner_records + edge_records + interval_record
            + int(self.proof_flag_bits))


@dataclass(frozen=True)
class OwnerProposalTransportCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    coordinator: int
    proposal_count: int
    transmitting_owner_count: int
    total_over_air_bits: int
    max_packet_latency_s: float
    total_protocol_latency_s: float
    total_energy_j: float
    min_snr_db: float
    required_comm_power_w: np.ndarray
    projected_comm_power_w: np.ndarray


def _candidate_certificate(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    proposals: Sequence[RankedOwnerProposal],
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    *,
    coordinator: int,
    communication_model: InterUAVCommunicationModel,
    layout: OwnerProposalWireLayout,
    snr_margin_db: float,
    latency_margin_s: float,
) -> OwnerProposalTransportCertificate:
    K = layout.num_agents
    coordinator_id = int(coordinator)
    packets = [
        (int(item.proposer), layout.proposal_bits(
            selected, role, owner, item))
        for item in proposals
        if int(item.proposer) != coordinator_id
    ]
    bandwidth = (
        communication_model.bandwidth_hz / len(packets)
        if packets else communication_model.bandwidth_hz
    )
    required = np.zeros(K, dtype=np.float64)
    for sender, bits in packets:
        power = _required_power_for_packet(
            communication_model,
            positions[sender],
            positions[coordinator_id],
            bits,
            bandwidth,
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        required[sender] = max(required[sender], power)
    projected = np.maximum(existing_comm_power_w, required)
    reasons: list[str] = []
    if np.any(~np.isfinite(projected)) or np.any(projected >= 1.0):
        reasons.append("power:unavailable")
    total_bits = 0
    max_latency = 0.0
    total_energy = 0.0
    min_snr = float("inf")
    for sender, bits in packets:
        snr, _, serialization, latency = (
            communication_model.robust_link_budget(
                positions[sender],
                positions[coordinator_id],
                bits,
                bandwidth,
                float(projected[sender]),
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
        )
        total_bits += int(bits)
        max_latency = max(max_latency, float(latency))
        total_energy += float(projected[sender]) * float(serialization)
        min_snr = min(min_snr, float(snr))
        if snr < communication_model.snr_threshold_db:
            reasons.append(f"proposal:snr:{sender}->{coordinator_id}")
        if latency > communication_model.deadline_s + 1.0e-12:
            reasons.append(f"proposal:deadline:{sender}->{coordinator_id}")
    return OwnerProposalTransportCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        coordinator=coordinator_id,
        proposal_count=len(proposals),
        transmitting_owner_count=len(packets),
        total_over_air_bits=int(total_bits),
        max_packet_latency_s=float(max_latency),
        total_protocol_latency_s=float(max_latency),
        total_energy_j=float(total_energy),
        min_snr_db=float(min_snr),
        required_comm_power_w=required,
        projected_comm_power_w=projected,
    )


def certify_owner_proposal_transport(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    proposals: Sequence[RankedOwnerProposal],
    *,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    communication_model: InterUAVCommunicationModel,
    layout: OwnerProposalWireLayout,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> OwnerProposalTransportCertificate:
    """Elect the feasible coordinator with minimum added proposal reserve."""
    pair = np.asarray(selected, dtype=bool)
    state_role = np.asarray(role, dtype=np.int8).reshape(-1)
    state_owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(existing_comm_power_w, dtype=np.float64).reshape(-1)
    K, K2, Q = pair.shape
    items = tuple(proposals)
    if K != K2 or K != layout.num_agents or Q != layout.num_targets:
        raise ValueError("proposal state and wire dimensions disagree")
    if (
        state_role.shape != (K,) or state_owner.shape != (Q,)
        or pos.shape != (K, 3) or comm.shape != (K,)
    ):
        raise ValueError("proposal transport inputs have inconsistent shapes")
    proposers = [int(item.proposer) for item in items]
    if len(items) > layout.proposal_budget:
        raise ValueError("proposal round exceeds the owner-token budget")
    if len(proposers) != len(set(proposers)):
        raise ValueError("each owner may send at most one proposal per round")
    if any(proposer < 0 or proposer >= K for proposer in proposers):
        raise ValueError("proposal sender is outside the UAV set")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(comm < 0.0) or np.any(comm >= 1.0)
    ):
        raise ValueError("proposal transport state is outside physical support")
    for item in items:
        if (
            not np.isfinite(item.lower) or not np.isfinite(item.upper)
            or not np.isfinite(item.score) or item.lower < 0.0
            or item.upper < item.lower
        ):
            raise ValueError("proposal interval is invalid")
        layout.proposal_bits(pair, state_role, state_owner, item)
    certificates = [
        _candidate_certificate(
            pair,
            state_role,
            state_owner,
            items,
            pos,
            comm,
            coordinator=coordinator,
            communication_model=communication_model,
            layout=layout,
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
