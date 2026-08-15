"""Atomic wire certificate for slow-geometry tube repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_isac.coordination.dependency_commit import (
    CommitRoundReport,
    _index_bits,
    _round_report,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


@dataclass(frozen=True)
class GeometryRepairWireLayout:
    """Finite prepare/vote/decision record for an atomic planar move."""

    num_agents: int
    num_targets: int
    header_bits: int = 64
    epoch_bits: int = 16
    digest_bits: int = 64
    position_bits_per_axis: int = 64
    battery_bits: int = 64
    communication_power_bits: int = 16
    gradient_bits_per_axis: int = 32
    displacement_bits_per_axis: int = 16
    bound_bits: int = 16

    def __post_init__(self) -> None:
        if int(self.num_agents) < 2 or int(self.num_targets) < 1:
            raise ValueError("geometry layout requires K>=2 and Q>=1")
        if int(self.displacement_bits_per_axis) < 2:
            raise ValueError("signed displacement needs at least two bits")
        if int(self.position_bits_per_axis) != 64:
            raise ValueError("mover position uses exact binary64 coordinates")
        if int(self.battery_bits) != 64:
            raise ValueError("battery state uses exact binary64 joules")
        if int(self.communication_power_bits) != 16:
            raise ValueError("communication power uses IEEE binary16")
        if int(self.gradient_bits_per_axis) != 32:
            raise ValueError("geometry gradients use IEEE binary32")
        if int(self.bound_bits) != 16:
            raise ValueError("geometry certificate bounds use binary16")
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
    def common_bits(self) -> int:
        return int(self.header_bits + self.epoch_bits + self.digest_bits)

    @property
    def state_bits(self) -> int:
        return self.state_bits_for_horizon(1)

    def state_bits_for_horizon(self, certificate_horizon_steps: int) -> int:
        horizon = int(certificate_horizon_steps)
        if not 1 <= horizon <= 8:
            raise ValueError("geometry state horizon must lie in [1,8]")
        # Each UAV announces its own position, movement tube, battery and RF
        # row.  The two target-wise values are conservative lower/upper
        # binary16 sensing powers, not duplicated target certificates.
        return int(
            self.common_bits + self.node_bits
            + 2 * int(self.position_bits_per_axis)
            + 2 * horizon * int(self.displacement_bits_per_axis)
            + int(self.battery_bits)
            + horizon * int(self.communication_power_bits)
            + 2 * horizon * int(self.num_targets) * int(self.bound_bits)
        )

    def gradient_bits(self, owned_target_count: int) -> int:
        count = int(owned_target_count)
        if not 1 <= count <= int(self.num_targets):
            raise ValueError("owned target count is outside [0,Q]")
        target_record = (
            self.target_bits + int(self.bound_bits)
            + 2 * int(self.num_agents) * int(self.gradient_bits_per_axis)
        )
        return int(
            self.common_bits + self.node_bits
            + _index_bits(self.num_targets + 1)
            + count * target_record
        )

    def verification_bits(
        self,
        owned_target_count: int,
        candidate_count: int,
        certificate_horizon_steps: int = 1,
    ) -> int:
        targets = int(owned_target_count)
        candidates = int(candidate_count)
        horizon = int(certificate_horizon_steps)
        if not 1 <= targets <= int(self.num_targets):
            raise ValueError("owned target count is outside [1,Q]")
        if not 1 <= candidates <= 8:
            raise ValueError("geometry verification supports Top-1 through Top-8")
        if not 1 <= horizon <= 8:
            raise ValueError("geometry verification horizon must lie in [1,8]")
        # One baseline upper factor is shared by all candidates. Each
        # candidate contributes only its lower factor. These are exactly the
        # sufficient statistics used by the no-harm and worst-target tests.
        target_record = (
            self.target_bits
            + horizon * (1 + candidates) * int(self.bound_bits))
        return int(
            self.common_bits + self.node_bits + _index_bits(9)
            + _index_bits(9) + _index_bits(self.num_targets + 1)
            + targets * target_record
        )

    def verification_request_bits(
        self,
        candidate_mover_counts: tuple[int, ...],
        certificate_horizon_steps: int,
    ) -> int:
        mover_counts = tuple(int(value) for value in candidate_mover_counts)
        horizon = int(certificate_horizon_steps)
        if (
            not 1 <= len(mover_counts) <= 8
            or any(not 1 <= value <= self.num_agents for value in mover_counts)
            or not 1 <= horizon <= 8
        ):
            raise ValueError("verification request dimensions are invalid")
        # The baseline tube is already digest-bound. Send only mover-indexed
        # signed corrections, never K dense absolute trajectories.
        return int(
            self.common_bits + self.node_bits + _index_bits(9)
            + sum(
                _index_bits(self.num_agents + 1)
                + movers * (
                    self.node_bits
                    + 2 * horizon * int(self.displacement_bits_per_axis)
                )
                for movers in mover_counts
            )
        )

    @property
    def prepare_bits(self) -> int:
        return self.prepare_bits_for_movers(1)

    def prepare_bits_for_movers(
        self,
        mover_count: int,
        certificate_horizon_steps: int = 1,
    ) -> int:
        # State packets are multicast to the coordinator and every target
        # owner before verification.  The common digest therefore binds a
        # baseline RF/movement cache at every node that needs it.  Prepare
        # sends only the selected sparse correction, its candidate identity,
        # the number of owner attestations, flight energy and swept minimum
        # separation.  Repeating H*K*Q RF values or H*Q owner-returned factors
        # here would add no information to a non-Byzantine atomic vote.
        movers = int(mover_count)
        horizon = int(certificate_horizon_steps)
        if not 1 <= movers <= int(self.num_agents):
            raise ValueError("geometry commit mover count is outside [1,K]")
        if not 1 <= horizon <= 8:
            raise ValueError("geometry commit horizon must lie in [1,8]")
        return int(
            self.common_bits + _index_bits(3)
            + _index_bits(9) + _index_bits(9)
            + _index_bits(self.num_agents + 1)
            + movers * (
                self.node_bits + 2 * int(self.position_bits_per_axis)
                + 2 * horizon * int(self.displacement_bits_per_axis)
            )
            + _index_bits(self.num_targets + 1)
            + 2 * int(self.bound_bits)
        )

    @property
    def vote_bits(self) -> int:
        return int(self.common_bits + self.node_bits + 1)

    @property
    def decision_bits(self) -> int:
        return int(self.common_bits + self.node_bits + 1)


@dataclass(frozen=True)
class GeometryRepairTransportCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    coordinator: int
    proposal_packet_count: int
    proposal_over_air_bits: int
    total_over_air_bits: int
    total_protocol_latency_s: float
    total_energy_j: float
    per_uav_energy_j: tuple[float, ...]
    min_snr_db: float
    rounds: tuple[CommitRoundReport, ...]


def certify_geometry_repair_transport(
    mover: int | tuple[int, ...],
    *,
    positions: np.ndarray,
    target_owner: np.ndarray,
    communication_power_w: np.ndarray,
    communication_model: InterUAVCommunicationModel,
    control_period_s: float,
    layout: GeometryRepairWireLayout,
    verification_candidate_count: int = 1,
    verification_mover_counts: tuple[int, ...] | None = None,
    certificate_horizon_steps: int = 1,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> GeometryRepairTransportCertificate:
    """Certify signed gather, proof-carrying prepare and atomic commit rounds.

    The coordinator generates Top-M candidates from gathered owner-local state
    and gradients. It broadcasts the candidate tubes, target owners return
    candidate-specific bounds, and only then does it select a proof-carrying
    candidate. Its prepare packet carries the chosen
    movement tube and sensing bounds, so every UAV can validate its local
    collision, boundary and owned-target obligations before voting.
    """
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(communication_power_w, dtype=np.float64).reshape(-1)
    owner = np.asarray(target_owner, dtype=np.int64).reshape(-1)
    K = int(layout.num_agents)
    mover_ids = (
        (int(mover),) if isinstance(mover, (int, np.integer))
        else tuple(int(value) for value in mover)
    )
    if (
        pos.shape != (K, 3) or comm.shape != (K,)
        or owner.shape != (int(layout.num_targets),)
        or np.any(owner < 0) or np.any(owner >= K)
        or not 1 <= len(mover_ids) <= K
        or len(set(mover_ids)) != len(mover_ids)
        or any(not 0 <= mover_id < K for mover_id in mover_ids)
        or np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(comm < 0.0) or np.any(comm >= 1.0)
    ):
        raise ValueError("geometry transport state is outside support")
    deadline = float(control_period_s)
    horizon = int(certificate_horizon_steps)
    if not np.isfinite(deadline) or deadline <= 0.0:
        raise ValueError("control period must be finite and positive")
    if not 1 <= horizon <= 8:
        raise ValueError("certificate horizon must lie in [1,8]")

    def heterogeneous_multicast(
        name: str,
        bits_by_sender: dict[int, int],
        receiver_sets: dict[int, tuple[int, ...]],
    ) -> tuple[CommitRoundReport, np.ndarray, tuple[str, ...]]:
        active = tuple(sorted(
            sender for sender in bits_by_sender
            if receiver_sets.get(sender, tuple())))
        energy = np.zeros(K, dtype=np.float64)
        failures: list[str] = []
        total_latency = 0.0
        min_snr = float("inf")
        for sender in active:
            bits = int(bits_by_sender[sender])
            sender_latency = 0.0
            sender_airtime = 0.0
            for receiver in receiver_sets[sender]:
                snr, _rate, serialization, latency = (
                    communication_model.robust_link_budget(
                    pos[sender], pos[receiver], bits,
                    communication_model.bandwidth_hz,
                    float(comm[sender]),
                    snr_margin_db=float(snr_margin_db),
                    latency_margin_s=float(latency_margin_s),
                    )
                )
                min_snr = min(min_snr, float(snr))
                sender_latency = max(sender_latency, float(latency))
                sender_airtime = max(sender_airtime, float(serialization))
                if snr < communication_model.snr_threshold_db:
                    failures.append(f"{name}:snr:{sender}->{receiver}")
                if latency > communication_model.deadline_s + 1.0e-12:
                    failures.append(
                        f"{name}:deadline:{sender}->{receiver}")
            total_latency += sender_latency
            energy[sender] = float(comm[sender]) * sender_airtime
        return CommitRoundReport(
            name=name,
            active_senders=active,
            payload_bits_per_sender=max(
                (bits_by_sender[sender] for sender in active), default=0),
            over_air_bits=int(sum(
                bits_by_sender[sender] for sender in active)),
            duration_s=float(total_latency),
            energy_j=float(np.sum(energy)),
            min_snr_db=float(min_snr),
            max_link_latency_s=float(total_latency),
            feasible=not failures,
        ), energy, tuple(failures)

    state_bits_by_sender = {
        sender: layout.state_bits_for_horizon(horizon) for sender in range(K)
    }
    gradient_bits_by_sender = {
        sender: layout.gradient_bits(int(np.sum(owner == sender)))
        for sender in sorted(set(int(value) for value in owner))
    }
    if not 1 <= int(verification_candidate_count) <= 8:
        raise ValueError("geometry verification supports Top-1 through Top-8")
    mover_counts = (
        tuple(len(mover_ids) for _ in range(int(verification_candidate_count)))
        if verification_mover_counts is None
        else tuple(int(value) for value in verification_mover_counts)
    )
    if len(mover_counts) != int(verification_candidate_count):
        raise ValueError("verification mover counts must match candidates")
    def certificate_for(coordinator: int) -> GeometryRepairTransportCertificate:
        peers = tuple(agent for agent in range(K) if agent != coordinator)
        proposal_reports = []
        proposal_energy = np.zeros(K, dtype=np.float64)
        proposal_reasons: list[str] = []
        owners = tuple(sorted(set(int(value) for value in owner)))
        state_receivers = tuple(sorted(set(owners) | {coordinator}))
        for name, bits_by_sender, receiver_sets in (
            (
                "state",
                state_bits_by_sender,
                {
                    sender: tuple(
                        receiver for receiver in state_receivers
                        if receiver != sender)
                    for sender in range(K)
                },
            ),
            (
                "gradient",
                gradient_bits_by_sender,
                {
                    sender: (() if sender == coordinator else (coordinator,))
                    for sender in gradient_bits_by_sender
                },
            ),
        ):
            report, energy, failures = heterogeneous_multicast(
                name, bits_by_sender, receiver_sets)
            proposal_reports.append(report)
            proposal_energy += energy
            proposal_reasons.extend(failures)
        proposal_bits = int(sum(
            report.over_air_bits for report in proposal_reports))
        proposal_packet_count = int(sum(
            len(report.active_senders) for report in proposal_reports))
        owner_peers = tuple(
            owner_id for owner_id in owners if owner_id != coordinator)
        verification_request, request_energy, request_failures = _round_report(
            name="verification_request",
            senders=(coordinator,),
            receiver_sets={coordinator: owner_peers},
            payload_bits=layout.verification_request_bits(
                mover_counts, horizon),
            positions=pos,
            comm_power_w=comm,
            model=communication_model,
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
        )
        verification_bits_by_sender = {
            owner_id: layout.verification_bits(
                int(np.sum(owner == owner_id)),
                int(verification_candidate_count),
                certificate_horizon_steps=horizon,
            )
            for owner_id in owner_peers
        }
        verification_return, return_energy, return_failures = (
            heterogeneous_multicast(
                "verification_return",
                verification_bits_by_sender,
                {
                    owner_id: (coordinator,)
                    for owner_id in verification_bits_by_sender
                },
            )
        )
        round_inputs = (
            (
                "prepare", (coordinator,), {coordinator: peers},
                layout.prepare_bits_for_movers(
                    len(mover_ids), certificate_horizon_steps=horizon),
            ),
            (
                "vote", peers,
                {peer: (coordinator,) for peer in peers}, layout.vote_bits,
            ),
            (
                "decision", (coordinator,), {coordinator: peers},
                layout.decision_bits,
            ),
        )
        reports: list[CommitRoundReport] = [
            *proposal_reports, verification_request, verification_return]
        reasons: list[str] = [
            *proposal_reasons, *request_failures, *return_failures]
        total_energy = float(np.sum(
            proposal_energy + request_energy + return_energy))
        per_uav_energy = (
            proposal_energy + request_energy + return_energy)
        for name, senders, receivers, bits in round_inputs:
            report, round_energy, failures = _round_report(
                name=name,
                senders=senders,
                receiver_sets=receivers,
                payload_bits=int(bits),
                positions=pos,
                comm_power_w=comm,
                model=communication_model,
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
            reports.append(report)
            reasons.extend(failures)
            total_energy += float(report.energy_j)
            per_uav_energy += round_energy
        total_latency = float(sum(
            report.duration_s for report in reports))
        if total_latency > deadline + 1.0e-12:
            reasons.append("protocol:end_to_end_deadline")
        return GeometryRepairTransportCertificate(
            feasible=not reasons,
            reasons=tuple(reasons),
            coordinator=coordinator,
            proposal_packet_count=proposal_packet_count,
            proposal_over_air_bits=proposal_bits,
            total_over_air_bits=int(sum(
                report.over_air_bits for report in reports)),
            total_protocol_latency_s=total_latency,
            total_energy_j=float(total_energy),
            per_uav_energy_j=tuple(
                float(value) for value in per_uav_energy),
            min_snr_db=float(min(
                (report.min_snr_db for report in reports),
                default=float("inf"),
            )),
            rounds=tuple(reports),
        )

    certificates = [
        certificate_for(coordinator) for coordinator in mover_ids]
    return min(certificates, key=lambda item: (
        not item.feasible,
        item.total_protocol_latency_s,
        item.total_over_air_bits,
        item.coordinator,
    ))


def quantize_displacement_toward_zero(
    displacement_m: np.ndarray,
    *,
    maximum_component_m: float,
    bits_per_axis: int,
) -> np.ndarray:
    """Decode a signed fixed-point displacement without enlarging its norm."""
    value = np.asarray(displacement_m, dtype=np.float64)
    maximum = float(maximum_component_m)
    bits = int(bits_per_axis)
    if (
        value.shape != (2,) or np.any(~np.isfinite(value))
        or not np.isfinite(maximum) or maximum <= 0.0 or bits < 2
    ):
        raise ValueError("displacement quantizer inputs are invalid")
    levels = (1 << (bits - 1)) - 1
    normalized = np.clip(value / maximum, -1.0, 1.0)
    magnitude = np.floor(np.abs(normalized) * levels) / levels
    decoded = np.sign(normalized) * magnitude * maximum
    if np.linalg.norm(decoded) > np.linalg.norm(value) + 1.0e-12:
        raise AssertionError("displacement quantizer enlarged the command")
    return decoded
