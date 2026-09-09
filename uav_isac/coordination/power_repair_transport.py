"""Physical U2U wire and latency certificate for column-generation repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_isac.domain.communication import CommunicationTransport


def _index_bits(cardinality: int) -> int:
    return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))


@dataclass(frozen=True)
class PowerRepairWireLayout:
    num_agents: int
    num_targets: int
    rounds: int = 4
    header_bits: int = 64
    epoch_bits: int = 16
    digest_bits: int = 64
    price_bits: int = 6
    deflection_bits: int = 16

    def __post_init__(self) -> None:
        if self.num_agents < 2 or self.num_targets < 1 or self.rounds < 1:
            raise ValueError("wire layout requires K>=2, Q>=1 and rounds>=1")
        for name in (
            "header_bits", "epoch_bits", "digest_bits", "price_bits",
            "deflection_bits",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def shared_bits(self) -> int:
        return int(
            self.header_bits + self.epoch_bits + self.digest_bits
            + _index_bits(self.num_agents)
            + _index_bits(self.rounds + 1)
            + _index_bits(self.num_targets + 1)
        )

    @property
    def price_packet_bits(self) -> int:
        return int(self.shared_bits + self.num_targets * self.price_bits)

    def owner_feedback_packet_bits(self, target_count: int) -> int:
        count = int(target_count)
        if not 0 <= count <= self.num_targets:
            raise ValueError("target_count is outside wire layout capacity")
        if count == 0:
            return 0
        entry_bits = _index_bits(self.num_targets) + self.deflection_bits
        return int(self.shared_bits + count * entry_bits)


@dataclass(frozen=True)
class PowerRepairTransportCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    coordinator: int
    required_comm_power_w: np.ndarray
    projected_comm_power_w: np.ndarray
    total_over_air_bits: int
    max_packet_latency_s: float
    total_protocol_latency_s: float
    total_energy_j: float
    min_snr_db: float
    feedback_sender_count: int


def _required_power_for_packet(
    model: CommunicationTransport,
    sender: np.ndarray,
    receiver: np.ndarray,
    bits: int,
    bandwidth_hz: float,
    *,
    snr_margin_db: float,
    latency_margin_s: float,
) -> float:
    available = (
        model.deadline_s - model.processing_delay_s - float(latency_margin_s)
    )
    if int(bits) <= 0:
        return 0.0
    if available <= 0.0 or bandwidth_hz <= 0.0:
        return float("inf")
    rate_required = float(bits) / available
    exponent = rate_required / float(bandwidth_hz)
    if exponent >= 1024.0 or np.isinf(float(snr_margin_db)):
        return float("inf")
    snr_rate = float(np.exp2(exponent) - 1.0)
    snr_threshold = float(10.0 ** (model.snr_threshold_db / 10.0))
    required_robust_snr = max(snr_rate, snr_threshold)
    nominal_snr_db, _, _, _ = model.link_budget(
        sender, receiver, int(bits), float(bandwidth_hz), tx_power_w=1.0)
    robust_snr_per_w = float(
        10.0 ** ((nominal_snr_db - float(snr_margin_db)) / 10.0))
    if robust_snr_per_w <= 0.0:
        return float("inf")
    # Round the analytically minimal RF reserve upward.  Re-evaluating the
    # same Friis/Shannon expression otherwise can land a few ulps below the
    # hard SNR boundary (for example -3e-15 dB at a 0 dB threshold).  This is
    # numerical closure, not a relaxation of the physical requirement.
    required_power = float(required_robust_snr / robust_snr_per_w)
    return float(required_power * (1.0 + 64.0 * np.finfo(np.float64).eps))


def _candidate_certificate(
    owners: np.ndarray,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    *,
    coordinator: int,
    communication_model: CommunicationTransport,
    layout: PowerRepairWireLayout,
    control_period_s: float,
    snr_margin_db: float,
    latency_margin_s: float,
) -> PowerRepairTransportCertificate:
    K = layout.num_agents
    coordinator_id = int(coordinator)
    owner_counts = np.bincount(owners, minlength=K)
    feedback_senders = [
        sender for sender in range(K)
        if sender != coordinator_id and owner_counts[sender] > 0
    ]
    feedback_bandwidth = (
        communication_model.bandwidth_hz / len(feedback_senders)
        if feedback_senders else communication_model.bandwidth_hz
    )
    required = np.zeros(K, dtype=np.float64)
    feedback_bits = 0
    feedback_phase_latency = 0.0
    feedback_phase_energy = 0.0
    min_snr = float("inf")
    reasons: list[str] = []
    for sender in feedback_senders:
        bits = layout.owner_feedback_packet_bits(int(owner_counts[sender]))
        feedback_bits += bits
        power = _required_power_for_packet(
            communication_model,
            positions[sender],
            positions[coordinator_id],
            bits,
            feedback_bandwidth,
            snr_margin_db=snr_margin_db,
            latency_margin_s=latency_margin_s,
        )
        required[sender] = max(required[sender], power)

    price_bits = layout.price_packet_bits
    price_power = 0.0
    for receiver in range(K):
        if receiver == coordinator_id:
            continue
        price_power = max(
            price_power,
            _required_power_for_packet(
                communication_model,
                positions[coordinator_id],
                positions[receiver],
                price_bits,
                communication_model.bandwidth_hz,
                snr_margin_db=snr_margin_db,
                latency_margin_s=latency_margin_s,
            ),
        )
    required[coordinator_id] = max(required[coordinator_id], price_power)
    projected = np.maximum(existing_comm_power_w, required)
    if np.any(~np.isfinite(projected)) or np.any(projected >= 1.0):
        reasons.append("power:unavailable")

    # Evaluate the actual schedule at the projected RF powers, not at a nominal
    # fixed transmit power.  Owner packets share bandwidth orthogonally.
    for sender in feedback_senders:
        bits = layout.owner_feedback_packet_bits(int(owner_counts[sender]))
        snr, _, serialization, latency = communication_model.robust_link_budget(
            positions[sender],
            positions[coordinator_id],
            bits,
            feedback_bandwidth,
            float(projected[sender]),
            snr_margin_db=snr_margin_db,
            latency_margin_s=latency_margin_s,
        )
        min_snr = min(min_snr, float(snr))
        feedback_phase_latency = max(feedback_phase_latency, float(latency))
        feedback_phase_energy += float(projected[sender]) * float(serialization)
        if snr < communication_model.snr_threshold_db:
            reasons.append(f"feedback:snr:{sender}->{coordinator_id}")
        if latency > communication_model.deadline_s + 1.0e-12:
            reasons.append(f"feedback:deadline:{sender}->{coordinator_id}")

    price_phase_latency = 0.0
    price_serialization = 0.0
    for receiver in range(K):
        if receiver == coordinator_id:
            continue
        snr, _, serialization, latency = communication_model.robust_link_budget(
            positions[coordinator_id],
            positions[receiver],
            price_bits,
            communication_model.bandwidth_hz,
            float(projected[coordinator_id]),
            snr_margin_db=snr_margin_db,
            latency_margin_s=latency_margin_s,
        )
        min_snr = min(min_snr, float(snr))
        price_phase_latency = max(price_phase_latency, float(latency))
        price_serialization = max(price_serialization, float(serialization))
        if snr < communication_model.snr_threshold_db:
            reasons.append(f"price:snr:{coordinator_id}->{receiver}")
        if latency > communication_model.deadline_s + 1.0e-12:
            reasons.append(f"price:deadline:{coordinator_id}->{receiver}")

    total_latency = (
        feedback_phase_latency
        + layout.rounds * (price_phase_latency + feedback_phase_latency)
    )
    if total_latency > float(control_period_s) + 1.0e-12:
        reasons.append("protocol:control_period")
    total_bits = int(
        (layout.rounds + 1) * feedback_bits
        + layout.rounds * price_bits
    )
    total_energy = float(
        (layout.rounds + 1) * feedback_phase_energy
        + layout.rounds * projected[coordinator_id] * price_serialization
    )
    return PowerRepairTransportCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        coordinator=coordinator_id,
        required_comm_power_w=required,
        projected_comm_power_w=projected,
        total_over_air_bits=total_bits,
        max_packet_latency_s=float(max(
            feedback_phase_latency, price_phase_latency)),
        total_protocol_latency_s=float(total_latency),
        total_energy_j=total_energy,
        min_snr_db=float(min_snr),
        feedback_sender_count=len(feedback_senders),
    )


def certify_power_repair_transport(
    owners: np.ndarray,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    *,
    communication_model: CommunicationTransport,
    layout: PowerRepairWireLayout,
    control_period_s: float,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> PowerRepairTransportCertificate:
    """Choose the feasible coordinator requiring the least extra RF reserve."""
    owner = np.asarray(owners, dtype=np.int64).reshape(-1)
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(existing_comm_power_w, dtype=np.float64).reshape(-1)
    if owner.shape != (layout.num_targets,):
        raise ValueError("owners must have one entry per target")
    if pos.shape != (layout.num_agents, 3) or comm.shape != (layout.num_agents,):
        raise ValueError("positions/comm power do not match wire layout")
    if (
        np.any(owner < 0) or np.any(owner >= layout.num_agents)
        or np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(comm < 0.0) or np.any(comm >= 1.0)
    ):
        raise ValueError("transport state is outside physical support")
    candidates = [
        _candidate_certificate(
            owner,
            pos,
            comm,
            coordinator=coordinator,
            communication_model=communication_model,
            layout=layout,
            control_period_s=float(control_period_s),
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        for coordinator in range(layout.num_agents)
    ]
    return min(candidates, key=lambda result: (
        not result.feasible,
        float(np.max(result.projected_comm_power_w - comm)),
        float(np.sum(result.projected_comm_power_w - comm)),
        result.total_protocol_latency_s,
        result.total_over_air_bits,
        result.coordinator,
    ))
