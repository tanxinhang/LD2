"""Physical owner-capacity auction for sparse ISAC structure candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uav_isac.coordination.owner_proposal_transport import (
    quantize_nonnegative_float16_lower,
    quantize_nonnegative_float16_upper,
)
from uav_isac.coordination.owner_gain_ceiling import owner_gain_ceiling
from uav_isac.coordination.power_repair_transport import (
    _required_power_for_packet,
)
from uav_isac.domain.communication import CommunicationTransport


def _index_bits(cardinality: int) -> int:
    return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))


@dataclass(frozen=True)
class OwnerBidWireLayout:
    num_agents: int
    num_targets: int
    owners_per_target: int = 3
    header_bits: int = 64
    epoch_bits: int = 16
    digest_bits: int = 64
    score_bits: int = 16

    def __post_init__(self) -> None:
        if int(self.num_agents) < 2 or int(self.num_targets) < 1:
            raise ValueError("owner auction requires K>=2 and Q>=1")
        if not 1 <= int(self.owners_per_target) <= int(self.num_agents):
            raise ValueError("owners_per_target must lie in [1,K]")
        if int(self.score_bits) != 16:
            raise ValueError("owner bids use directional binary16 scores")
        for name in ("header_bits", "epoch_bits", "digest_bits"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def node_bits(self) -> int:
        return _index_bits(self.num_agents)

    @property
    def target_bits(self) -> int:
        return _index_bits(self.num_targets)

    @property
    def shared_bits(self) -> int:
        return int(
            self.header_bits + self.epoch_bits + self.digest_bits
            + self.node_bits + _index_bits(self.num_targets + 1)
        )

    @property
    def bid_packet_bits(self) -> int:
        record = self.target_bits + 2 * int(self.score_bits)
        return int(self.shared_bits + self.num_targets * record)

    @property
    def decision_packet_bits(self) -> int:
        owner_record = 1 + self.node_bits
        target_record = (
            self.target_bits + self.owners_per_target * owner_record)
        return int(self.shared_bits + self.num_targets * target_record)


@dataclass(frozen=True)
class OwnerBidTransportCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    coordinator: int
    bid_packet_bits: int
    decision_packet_bits: int
    total_over_air_bits: int
    max_packet_latency_s: float
    total_protocol_latency_s: float
    total_energy_j: float
    min_snr_db: float
    required_comm_power_w: np.ndarray
    projected_comm_power_w: np.ndarray


@dataclass(frozen=True)
class OwnerBidCandidateResult:
    candidate_mask: np.ndarray
    selected_set: tuple[tuple[int, int, int], ...]
    selected_owner_groups: tuple[tuple[int, int], ...]
    owner_lower_bid: np.ndarray
    owner_upper_bid: np.ndarray
    full_edge_count: int
    candidate_edge_count: int


def owner_capacity_bid_envelope(
    lower_coefficient: np.ndarray,
    upper_coefficient: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    target_pair_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute receiver-local lower/upper target-capacity bids."""
    lower = np.asarray(lower_coefficient, dtype=np.float64)
    upper = np.asarray(upper_coefficient, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        lower.ndim != 3 or lower.shape[0] != lower.shape[1]
        or lower.shape[0] != budget.size or upper.shape != lower.shape
        or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
        or np.any(lower < 0.0) or np.any(upper < lower)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError("owner bid envelope requires finite 0<=lower<=upper")
    K, _, Q = lower.shape
    pair_limit = min(int(target_pair_limit), K - 1)
    if pair_limit < 1:
        raise ValueError("target_pair_limit must be positive")
    lower_bid = np.zeros((K, Q), dtype=np.float64)
    upper_bid = np.zeros((K, Q), dtype=np.float64)
    for receiver in range(K):
        for target in range(Q):
            lower_value = owner_gain_ceiling(
                lower[:, receiver, target],
                budget_w=budget, pair_limit=pair_limit)
            upper_value = owner_gain_ceiling(
                upper[:, receiver, target],
                budget_w=budget, pair_limit=pair_limit)
            lower_bid[receiver, target] = (
                quantize_nonnegative_float16_lower(lower_value))
            upper_bid[receiver, target] = (
                quantize_nonnegative_float16_upper(upper_value))
    return lower_bid, upper_bid


def owner_bid_sparse_candidate_mask(
    lower_coefficient: np.ndarray,
    upper_coefficient: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    target_pair_limit: int,
    owners_per_target: int,
    incumbent_selected: Sequence[tuple[int, int, int]] = (),
) -> OwnerBidCandidateResult:
    """Select a permutation-equivariant bounded owner graph from wire bids.

    All non-owner transmitters remain candidates for an admitted owner group.
    This preserves role-conflict alternatives while reducing the full
    ``K(K-1)Q`` graph to at most ``m(K-1)Q`` edges for constant ``m``.
    """
    lower = np.asarray(lower_coefficient, dtype=np.float64)
    upper = np.asarray(upper_coefficient, dtype=np.float64)
    K, _, Q = lower.shape
    owner_count = min(int(owners_per_target), K)
    if owner_count < 1:
        raise ValueError("owners_per_target must be positive")
    lower_bid, upper_bid = owner_capacity_bid_envelope(
        lower,
        upper,
        sensing_budget_w,
        target_pair_limit=target_pair_limit,
    )
    groups: set[tuple[int, int]] = set()
    for target in range(Q):
        receivers = sorted(range(K), key=lambda receiver: (
            -float(upper_bid[receiver, target]),
            -float(lower_bid[receiver, target]),
            receiver,
        ))
        groups.update(
            (receiver, target) for receiver in receivers[:owner_count]
            if upper_bid[receiver, target] > 0.0
        )
    candidate = np.zeros_like(lower, dtype=bool)
    for receiver, target in groups:
        for transmitter in range(K):
            if (
                transmitter != receiver
                and upper[transmitter, receiver, target] > 0.0
            ):
                candidate[transmitter, receiver, target] = True
    for transmitter, receiver, target in incumbent_selected:
        edge = (int(transmitter), int(receiver), int(target))
        if (
            0 <= edge[0] < K and 0 <= edge[1] < K and 0 <= edge[2] < Q
            and edge[0] != edge[1] and upper[edge] > 0.0
        ):
            candidate[edge] = True
            groups.add((edge[1], edge[2]))
    selected = tuple(
        tuple(int(value) for value in edge) for edge in np.argwhere(candidate)
    )
    return OwnerBidCandidateResult(
        candidate_mask=candidate,
        selected_set=selected,
        selected_owner_groups=tuple(sorted(groups, key=lambda item: (
            item[1], item[0]))),
        owner_lower_bid=lower_bid,
        owner_upper_bid=upper_bid,
        full_edge_count=int(np.count_nonzero(upper)),
        candidate_edge_count=int(np.count_nonzero(candidate)),
    )


def _candidate_transport(
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    *,
    coordinator: int,
    communication_model: CommunicationTransport,
    layout: OwnerBidWireLayout,
    control_period_s: float,
    snr_margin_db: float,
    latency_margin_s: float,
) -> OwnerBidTransportCertificate:
    K = int(layout.num_agents)
    coordinator_id = int(coordinator)
    senders = [sender for sender in range(K) if sender != coordinator_id]
    bid_bandwidth = communication_model.bandwidth_hz / len(senders)
    required = np.zeros(K, dtype=np.float64)
    for sender in senders:
        required[sender] = _required_power_for_packet(
            communication_model,
            positions[sender],
            positions[coordinator_id],
            layout.bid_packet_bits,
            bid_bandwidth,
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
    for receiver in senders:
        required[coordinator_id] = max(
            required[coordinator_id],
            _required_power_for_packet(
                communication_model,
                positions[coordinator_id],
                positions[receiver],
                layout.decision_packet_bits,
                communication_model.bandwidth_hz,
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            ),
        )
    projected = np.maximum(existing_comm_power_w, required)
    reasons: list[str] = []
    if np.any(~np.isfinite(projected)) or np.any(projected >= 1.0):
        reasons.append("power:unavailable")

    bid_latency = 0.0
    decision_latency = 0.0
    total_energy = 0.0
    min_snr = float("inf")
    for sender in senders:
        snr, _, serialization, latency = communication_model.robust_link_budget(
            positions[sender],
            positions[coordinator_id],
            layout.bid_packet_bits,
            bid_bandwidth,
            float(projected[sender]),
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        min_snr = min(min_snr, float(snr))
        bid_latency = max(bid_latency, float(latency))
        total_energy += float(projected[sender]) * float(serialization)
        if snr < communication_model.snr_threshold_db:
            reasons.append(f"bid:snr:{sender}->{coordinator_id}")
        if latency > communication_model.deadline_s + 1.0e-12:
            reasons.append(f"bid:deadline:{sender}->{coordinator_id}")
    decision_serialization = 0.0
    for receiver in senders:
        snr, _, serialization, latency = communication_model.robust_link_budget(
            positions[coordinator_id],
            positions[receiver],
            layout.decision_packet_bits,
            communication_model.bandwidth_hz,
            float(projected[coordinator_id]),
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        min_snr = min(min_snr, float(snr))
        decision_latency = max(decision_latency, float(latency))
        decision_serialization = max(
            decision_serialization, float(serialization))
        if snr < communication_model.snr_threshold_db:
            reasons.append(f"decision:snr:{coordinator_id}->{receiver}")
        if latency > communication_model.deadline_s + 1.0e-12:
            reasons.append(f"decision:deadline:{coordinator_id}->{receiver}")
    total_energy += (
        float(projected[coordinator_id]) * decision_serialization)
    total_latency = bid_latency + decision_latency
    if total_latency > float(control_period_s) + 1.0e-12:
        reasons.append("protocol:control_period")
    return OwnerBidTransportCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        coordinator=coordinator_id,
        bid_packet_bits=int(layout.bid_packet_bits),
        decision_packet_bits=int(layout.decision_packet_bits),
        total_over_air_bits=int(
            len(senders) * layout.bid_packet_bits
            + layout.decision_packet_bits),
        max_packet_latency_s=float(max(bid_latency, decision_latency)),
        total_protocol_latency_s=float(total_latency),
        total_energy_j=float(total_energy),
        min_snr_db=float(min_snr),
        required_comm_power_w=required,
        projected_comm_power_w=projected,
    )


def certify_owner_bid_transport(
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    *,
    communication_model: CommunicationTransport,
    layout: OwnerBidWireLayout,
    control_period_s: float,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> OwnerBidTransportCertificate:
    """Elect the feasible owner-auction coordinator with least RF reserve."""
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(existing_comm_power_w, dtype=np.float64).reshape(-1)
    K = int(layout.num_agents)
    if pos.shape != (K, 3) or comm.shape != (K,):
        raise ValueError("owner auction state has inconsistent dimensions")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(comm < 0.0) or np.any(comm >= 1.0)
    ):
        raise ValueError("owner auction state is outside physical support")
    candidates = [
        _candidate_transport(
            pos,
            comm,
            coordinator=coordinator,
            communication_model=communication_model,
            layout=layout,
            control_period_s=float(control_period_s),
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        for coordinator in range(K)
    ]
    return min(candidates, key=lambda item: (
        not item.feasible,
        float(np.max(item.projected_comm_power_w - comm)),
        float(np.sum(item.projected_comm_power_w - comm)),
        item.total_protocol_latency_s,
        item.total_over_air_bits,
        item.coordinator,
    ))
