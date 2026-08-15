"""Physical certificate for atomic dependency-closure reconfiguration.

The protocol is a fail-closed prepare/vote/decision exchange.  It does not
turn heterogeneous quantities into an arbitrary scalar cost: packet bits,
link SNR, per-packet latency, end-to-end commit time, RF energy and the ISAC
power simplex are audited in their native units.  A candidate is executable
only if every hard constraint is satisfied.

Round 1: one proposer broadcasts the complete prepare record.
Round 2: all other participants return one ACK/NACK concurrently using
orthogonal bandwidth shares.
Round 3: the proposer broadcasts the digest-bound commit/abort decision.

The participant closure contains every changed-role UAV, both endpoints of
every toggled sensing edge, and the old/new owner of each affected target.
Thus no resource whose local state changes can be committed silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.environment.communication import InterUAVCommunicationModel


def _index_bits(cardinality: int) -> int:
    """Bits required for one symbol from a finite non-empty alphabet."""
    return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))


def quantize_unit_interval_lower(
    values: np.ndarray,
    *,
    bits: int,
) -> np.ndarray:
    """Conservative fixed-point quantizer for a probability lower bound."""
    precision = int(bits)
    if precision < 1:
        raise ValueError("bits must be positive")
    value = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(value)):
        raise ValueError("values must be finite")
    clipped = np.clip(value, 0.0, 1.0)
    denominator = float((1 << precision) - 1)
    encoded = np.floor(clipped * denominator) / denominator
    return np.minimum(encoded, clipped)


def quantize_unit_interval_upper(
    values: np.ndarray,
    *,
    bits: int,
) -> np.ndarray:
    """Conservative fixed-point quantizer for non-negative uncertainty."""
    precision = int(bits)
    if precision < 1:
        raise ValueError("bits must be positive")
    value = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(value)):
        raise ValueError("values must be finite")
    clipped = np.clip(value, 0.0, 1.0)
    denominator = float((1 << precision) - 1)
    encoded = np.ceil(clipped * denominator) / denominator
    return np.maximum(encoded, clipped)


@dataclass(frozen=True)
class DependencyCommitLayout:
    """Exact wire layout conditioned on finite K, Q and quantizer widths."""

    num_agents: int
    num_targets: int
    header_bits: int = 64
    epoch_bits: int = 16
    digest_bits: int = 64
    role_cardinality: int = 3
    lower_bound_bits: int = 16
    uncertainty_bits: int = 16

    def __post_init__(self) -> None:
        if int(self.num_agents) < 1 or int(self.num_targets) < 1:
            raise ValueError("num_agents and num_targets must be positive")
        for name in (
            "header_bits", "epoch_bits", "digest_bits",
            "lower_bound_bits", "uncertainty_bits",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.role_cardinality) < 1:
            raise ValueError("role_cardinality must be positive")

    @property
    def node_bits(self) -> int:
        return _index_bits(self.num_agents)

    @property
    def target_bits(self) -> int:
        return _index_bits(self.num_targets)

    @property
    def role_bits(self) -> int:
        return _index_bits(self.role_cardinality)

    @property
    def common_bits(self) -> int:
        return (
            int(self.header_bits) + int(self.epoch_bits)
            + int(self.digest_bits)
        )

    @property
    def _count_bits(self) -> int:
        # Four self-delimiting lists: role, owner, edge, certificate entries.
        max_edges = self.num_agents * max(self.num_agents - 1, 0) * self.num_targets
        return (
            _index_bits(self.num_agents + 1)
            + _index_bits(self.num_targets + 1)
            + _index_bits(max_edges + 1)
            + _index_bits(self.num_targets + 1)
        )

    def prepare_bits(
        self,
        *,
        changed_roles: int,
        changed_owners: int,
        toggled_edges: int,
        certificate_entries: int | None = None,
    ) -> int:
        """Bits in the complete candidate and conservative safety record.

        The safety record carries one lower-bound/uncertainty pair for every
        target by default because worst-k tail dominance is a global order.
        """
        roles = int(changed_roles)
        owners = int(changed_owners)
        edges = int(toggled_edges)
        certs = (
            self.num_targets
            if certificate_entries is None
            else int(certificate_entries)
        )
        if not 0 <= roles <= self.num_agents:
            raise ValueError("changed_roles is outside [0,K]")
        if not 0 <= owners <= self.num_targets:
            raise ValueError("changed_owners is outside [0,Q]")
        max_edges = self.num_agents * max(self.num_agents - 1, 0) * self.num_targets
        if not 0 <= edges <= max_edges:
            raise ValueError("toggled_edges is outside the directed graph")
        if not 0 <= certs <= self.num_targets:
            raise ValueError("certificate_entries is outside [0,Q]")
        role_record = self.node_bits + self.role_bits
        owner_record = self.target_bits + self.node_bits
        edge_record = 2 * self.node_bits + self.target_bits + 1
        certificate_record = (
            self.target_bits + int(self.lower_bound_bits)
            + int(self.uncertainty_bits)
        )
        return int(
            self.common_bits + self.node_bits + self._count_bits
            + roles * role_record
            + owners * owner_record
            + edges * edge_record
            + certs * certificate_record
        )

    @property
    def vote_bits(self) -> int:
        # Sender ID and one ACK/NACK bit are bound to epoch and digest.
        return int(self.common_bits + self.node_bits + 1)

    @property
    def decision_bits(self) -> int:
        # Proposer ID and one commit/abort bit are bound to the same digest.
        return int(self.common_bits + self.node_bits + 1)


@dataclass(frozen=True)
class DependencyClosure:
    affected_targets: tuple[int, ...]
    participants: tuple[int, ...]
    changed_roles: int
    changed_owners: int
    toggled_edges: int


def dependency_closure(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    move: LocalMove,
) -> DependencyClosure:
    """Return the minimal participant closure for an atomic local move."""
    current = np.asarray(selected, dtype=bool)
    proposal = np.asarray(move.selected, dtype=bool)
    current_role = np.asarray(role).reshape(-1)
    proposal_role = np.asarray(move.role).reshape(-1)
    current_owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    proposal_owner = np.asarray(move.owner, dtype=np.int64).reshape(-1)
    if current.ndim != 3 or current.shape != proposal.shape:
        raise ValueError("selected and move.selected must have shape (K,K,Q)")
    K, K2, Q = current.shape
    if K != K2:
        raise ValueError("selected graph must have equal UAV axes")
    if current_role.shape != (K,) or proposal_role.shape != (K,):
        raise ValueError("role vectors must have shape (K,)")
    if current_owner.shape != (Q,) or proposal_owner.shape != (Q,):
        raise ValueError("owner vectors must have shape (Q,)")

    changed_edge_mask = current ^ proposal
    changed_role_ids = np.flatnonzero(current_role != proposal_role)
    changed_owner_targets = np.flatnonzero(current_owner != proposal_owner)
    affected = set(int(q) for q in np.flatnonzero(
        np.any(changed_edge_mask, axis=(0, 1))))
    affected.update(int(q) for q in changed_owner_targets)

    participants = set(int(k) for k in changed_role_ids)
    for tx, rx, _target in np.argwhere(changed_edge_mask):
        participants.add(int(tx))
        participants.add(int(rx))
    for target in affected:
        for candidate_owner in (current_owner[target], proposal_owner[target]):
            if 0 <= int(candidate_owner) < K:
                participants.add(int(candidate_owner))

    return DependencyClosure(
        affected_targets=tuple(sorted(affected)),
        participants=tuple(sorted(participants)),
        changed_roles=int(changed_role_ids.size),
        changed_owners=int(changed_owner_targets.size),
        toggled_edges=int(np.sum(changed_edge_mask)),
    )


@dataclass(frozen=True)
class CommitRoundReport:
    name: str
    active_senders: tuple[int, ...]
    payload_bits_per_sender: int
    over_air_bits: int
    duration_s: float
    energy_j: float
    min_snr_db: float
    max_link_latency_s: float
    feasible: bool


@dataclass(frozen=True)
class DependencyCommitCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    proposer: int | None
    certificate_epoch_id: int | None
    certificate_digest: int | None
    closure: DependencyClosure
    prepare_bits: int
    vote_bits: int
    decision_bits: int
    total_over_air_bits: int
    total_latency_s: float
    total_energy_j: float
    per_uav_energy_j: tuple[float, ...]
    power_excess_w: tuple[float, ...]
    rounds: tuple[CommitRoundReport, ...]


@dataclass(frozen=True)
class MinimumReserveResult:
    feasible: bool
    uniform_comm_floor_w: float | None
    iterations: int
    certificate: DependencyCommitCertificate
    comm_power_w: tuple[float, ...]
    sensing_power_w: tuple[tuple[float, ...], ...]


def _round_report(
    *,
    name: str,
    senders: Iterable[int],
    receiver_sets: dict[int, tuple[int, ...]],
    payload_bits: int,
    positions: np.ndarray,
    comm_power_w: np.ndarray,
    model: InterUAVCommunicationModel,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> tuple[CommitRoundReport, np.ndarray, tuple[str, ...]]:
    active = tuple(sorted(
        int(sender) for sender in senders
        if receiver_sets.get(int(sender), tuple())
    ))
    K = positions.shape[0]
    energy = np.zeros(K, dtype=np.float64)
    if not active:
        return CommitRoundReport(
            name=name,
            active_senders=tuple(),
            payload_bits_per_sender=int(payload_bits),
            over_air_bits=0,
            duration_s=0.0,
            energy_j=0.0,
            min_snr_db=float("inf"),
            max_link_latency_s=0.0,
            feasible=True,
        ), energy, tuple()

    bandwidth = model.bandwidth_hz / len(active)
    max_latency = 0.0
    min_snr = float("inf")
    failures: list[str] = []
    for sender in active:
        sender_airtime = 0.0
        for receiver in receiver_sets.get(sender, tuple()):
            snr_db, _rate, serialization_s, latency_s = (
                model.robust_link_budget(
                positions[sender], positions[receiver], int(payload_bits),
                bandwidth, float(comm_power_w[sender]),
                snr_margin_db=snr_margin_db,
                latency_margin_s=latency_margin_s,
                )
            )
            min_snr = min(min_snr, float(snr_db))
            max_latency = max(max_latency, float(latency_s))
            sender_airtime = max(sender_airtime, float(serialization_s))
            if snr_db < model.snr_threshold_db:
                failures.append(
                    f"{name}:snr:{sender}->{receiver}")
            if latency_s > model.deadline_s:
                failures.append(
                    f"{name}:deadline:{sender}->{receiver}")
        energy[sender] = (
            float("inf")
            if not np.isfinite(sender_airtime)
            else float(comm_power_w[sender]) * sender_airtime
        )
    report = CommitRoundReport(
        name=name,
        active_senders=active,
        payload_bits_per_sender=int(payload_bits),
        over_air_bits=int(len(active) * int(payload_bits)),
        duration_s=float(max_latency),
        energy_j=float(np.sum(energy)),
        min_snr_db=float(min_snr),
        max_link_latency_s=float(max_latency),
        feasible=not failures,
    )
    return report, energy, tuple(failures)


def certify_dependency_commit(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    move: LocalMove,
    *,
    proposer: int,
    positions: np.ndarray,
    comm_power_w: np.ndarray,
    sensing_power_w: np.ndarray,
    state_versions: np.ndarray,
    certificate_epoch_ids: np.ndarray,
    certificate_digests: np.ndarray,
    communication_model: InterUAVCommunicationModel,
    total_power_w: float = 1.0,
    total_deadline_s: float | None = None,
    layout: DependencyCommitLayout | None = None,
    power_tolerance_w: float = 1.0e-12,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> DependencyCommitCertificate:
    """Certify one proposer without relaxing any physical constraint."""
    current = np.asarray(selected, dtype=bool)
    if current.ndim != 3:
        raise ValueError("selected must have shape (K,K,Q)")
    K, K2, Q = current.shape
    if K != K2:
        raise ValueError("selected graph must have equal UAV axes")
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(comm_power_w, dtype=np.float64).reshape(-1)
    versions = np.asarray(state_versions)
    epoch_ids = np.asarray(certificate_epoch_ids)
    digests = np.asarray(certificate_digests)
    sensing = np.asarray(sensing_power_w, dtype=np.float64)
    if pos.shape != (K, 3):
        raise ValueError("positions must have shape (K,3)")
    if comm.shape != (K,):
        raise ValueError("comm_power_w must have shape (K,)")
    if versions.shape != (K,):
        raise ValueError("state_versions must have shape (K,)")
    if epoch_ids.shape != (K,) or digests.shape != (K,):
        raise ValueError("certificate identity vectors must have shape (K,)")

    def integer_tuple(values: np.ndarray, name: str) -> tuple[int, ...]:
        normalized = []
        for raw in values.tolist():
            try:
                value = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must contain integers") from exc
            if isinstance(raw, (float, np.floating)) and (
                not np.isfinite(raw) or float(raw) != float(value)
            ):
                raise ValueError(f"{name} must contain integers")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            normalized.append(value)
        return tuple(normalized)

    version_values = integer_tuple(versions, "state_versions")
    epoch_values = integer_tuple(epoch_ids, "certificate_epoch_ids")
    digest_values = integer_tuple(digests, "certificate_digests")
    if sensing.shape == (K,):
        sensing_total = sensing
    elif sensing.ndim == 2 and sensing.shape[0] == K:
        sensing_total = np.sum(sensing, axis=1)
    else:
        raise ValueError("sensing_power_w must have shape (K,) or (K,Q)")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(~np.isfinite(sensing_total))
        or np.any(comm < 0.0) or np.any(sensing_total < 0.0)
    ):
        raise ValueError("positions and powers must be finite and non-negative")
    budget = float(total_power_w)
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("total_power_w must be finite and positive")
    end_to_end_deadline = (
        communication_model.dt
        if total_deadline_s is None
        else float(total_deadline_s)
    )
    if not np.isfinite(end_to_end_deadline) or end_to_end_deadline < 0.0:
        raise ValueError("total_deadline_s must be finite and non-negative")
    snr_margin = float(snr_margin_db)
    latency_margin = float(latency_margin_s)
    if np.isnan(snr_margin) or snr_margin < 0.0:
        raise ValueError("snr_margin_db must be non-negative and not NaN")
    if np.isnan(latency_margin) or latency_margin < 0.0:
        raise ValueError("latency_margin_s must be non-negative and not NaN")

    closure = dependency_closure(current, role, owner, move)
    packet_layout = layout or DependencyCommitLayout(K, Q)
    if (
        packet_layout.num_agents != K
        or packet_layout.num_targets != Q
    ):
        raise ValueError("layout cardinalities must match selected")
    epoch_limit = 1 << int(packet_layout.epoch_bits)
    digest_limit = 1 << int(packet_layout.digest_bits)
    if any(value >= epoch_limit for value in epoch_values):
        raise ValueError("certificate epoch does not fit the wire layout")
    if any(value >= digest_limit for value in digest_values):
        raise ValueError("certificate digest does not fit the wire layout")
    prepare_bits = packet_layout.prepare_bits(
        changed_roles=closure.changed_roles,
        changed_owners=closure.changed_owners,
        toggled_edges=closure.toggled_edges,
    )
    vote_bits = packet_layout.vote_bits
    decision_bits = packet_layout.decision_bits

    power_excess = comm + sensing_total - budget
    reasons: list[str] = [
        f"power:uav:{k}" for k in np.flatnonzero(
            power_excess > float(power_tolerance_w))
    ]

    participants = closure.participants
    if not participants:
        return DependencyCommitCertificate(
            feasible=not reasons,
            reasons=tuple(reasons),
            proposer=None,
            certificate_epoch_id=None,
            certificate_digest=None,
            closure=closure,
            prepare_bits=0,
            vote_bits=0,
            decision_bits=0,
            total_over_air_bits=0,
            total_latency_s=0.0,
            total_energy_j=0.0,
            per_uav_energy_j=tuple(0.0 for _ in range(K)),
            power_excess_w=tuple(float(value) for value in power_excess),
            rounds=tuple(),
        )
    proposer_id = int(proposer)
    if proposer_id not in participants:
        reasons.append("protocol:proposer_outside_dependency_closure")
        # Keep a valid in-range index for diagnostics without pretending the
        # requested proposer can execute the protocol.
        diagnostic_proposer = participants[0]
    else:
        diagnostic_proposer = proposer_id
    peers = tuple(k for k in participants if k != diagnostic_proposer)
    proposer_version = version_values[diagnostic_proposer]
    proposer_epoch = epoch_values[diagnostic_proposer]
    proposer_digest = digest_values[diagnostic_proposer]
    reasons.extend(
        f"protocol:state_version:uav:{participant}"
        for participant in participants
        if version_values[participant] != proposer_version
    )
    reasons.extend(
        f"protocol:certificate_epoch:uav:{participant}"
        for participant in participants
        if epoch_values[participant] != proposer_epoch
    )
    reasons.extend(
        f"protocol:certificate_digest:uav:{participant}"
        for participant in participants
        if digest_values[participant] != proposer_digest
    )

    rounds_with_energy = []
    prepare = _round_report(
        name="prepare",
        senders=(diagnostic_proposer,),
        receiver_sets={diagnostic_proposer: peers},
        payload_bits=prepare_bits,
        positions=pos,
        comm_power_w=comm,
        model=communication_model,
        snr_margin_db=snr_margin,
        latency_margin_s=latency_margin,
    )
    rounds_with_energy.append(prepare)
    vote = _round_report(
        name="vote",
        senders=peers,
        receiver_sets={peer: (diagnostic_proposer,) for peer in peers},
        payload_bits=vote_bits,
        positions=pos,
        comm_power_w=comm,
        model=communication_model,
        snr_margin_db=snr_margin,
        latency_margin_s=latency_margin,
    )
    rounds_with_energy.append(vote)
    decision = _round_report(
        name="decision",
        senders=(diagnostic_proposer,),
        receiver_sets={diagnostic_proposer: peers},
        payload_bits=decision_bits,
        positions=pos,
        comm_power_w=comm,
        model=communication_model,
        snr_margin_db=snr_margin,
        latency_margin_s=latency_margin,
    )
    rounds_with_energy.append(decision)

    reports = tuple(item[0] for item in rounds_with_energy)
    per_uav_energy = np.sum(
        np.stack([item[1] for item in rounds_with_energy], axis=0), axis=0)
    for item in rounds_with_energy:
        reasons.extend(item[2])
    total_latency = float(sum(report.duration_s for report in reports))
    if total_latency > end_to_end_deadline:
        reasons.append("protocol:end_to_end_deadline")
    return DependencyCommitCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        proposer=proposer_id,
        certificate_epoch_id=proposer_epoch,
        certificate_digest=proposer_digest,
        closure=closure,
        prepare_bits=prepare_bits,
        vote_bits=vote_bits,
        decision_bits=decision_bits,
        total_over_air_bits=int(sum(
            report.over_air_bits for report in reports)),
        total_latency_s=total_latency,
        total_energy_j=float(np.sum(per_uav_energy)),
        per_uav_energy_j=tuple(float(value) for value in per_uav_energy),
        power_excess_w=tuple(float(value) for value in power_excess),
        rounds=reports,
    )


def certify_best_dependency_commit(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    move: LocalMove,
    **kwargs,
) -> DependencyCommitCertificate:
    """Elect the feasible proposer with minimum latency, then RF energy.

    The final UAV index is a deterministic tie-breaker.  If every proposer
    fails, the least-failing diagnostic certificate is returned but remains
    infeasible, so callers still fall back to No-op.
    """
    closure = dependency_closure(selected, role, owner, move)
    if not closure.participants:
        return certify_dependency_commit(
            selected, role, owner, move, proposer=0, **kwargs)
    certificates = [
        certify_dependency_commit(
            selected, role, owner, move, proposer=proposer, **kwargs)
        for proposer in closure.participants
    ]
    return min(
        certificates,
        key=lambda item: (
            not item.feasible,
            len(item.reasons),
            item.total_latency_s,
            item.total_energy_j,
            int(item.proposer),
        ),
    )


def minimum_uniform_control_reserve(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    move: LocalMove,
    *,
    positions: np.ndarray,
    current_comm_power_w: np.ndarray,
    current_sensing_power_w: np.ndarray,
    sensing_weights: np.ndarray,
    state_versions: np.ndarray,
    certificate_epoch_ids: np.ndarray,
    certificate_digests: np.ndarray,
    communication_model: InterUAVCommunicationModel,
    reserve_upper_w: float,
    total_power_w: float = 1.0,
    total_deadline_s: float | None = None,
    layout: DependencyCommitLayout | None = None,
    tolerance_w: float = 1.0e-9,
    max_iterations: int = 60,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> MinimumReserveResult:
    """Find the minimum common participant communication-power floor.

    For a fixed dependency closure, Shannon rate and deadline feasibility are
    monotone in every participant's transmit power.  Bisection therefore
    solves a physically meaningful resource problem without mixing watts with
    detection probability.  Detection must still be replayed after the
    resulting reduction in sensing power.
    """
    current = np.asarray(selected, dtype=bool)
    if current.ndim != 3:
        raise ValueError("selected must have shape (K,K,Q)")
    K, K2, Q = current.shape
    if K != K2:
        raise ValueError("selected graph must have equal UAV axes")
    comm0 = np.asarray(current_comm_power_w, dtype=np.float64).reshape(-1)
    sensing0 = np.asarray(current_sensing_power_w, dtype=np.float64)
    weights = np.asarray(sensing_weights, dtype=np.float64)
    if comm0.shape != (K,):
        raise ValueError("current_comm_power_w must have shape (K,)")
    if sensing0.shape != (K, Q) or weights.shape != (K, Q):
        raise ValueError("sensing powers and weights must have shape (K,Q)")
    if (
        np.any(~np.isfinite(comm0)) or np.any(~np.isfinite(sensing0))
        or np.any(~np.isfinite(weights)) or np.any(comm0 < 0.0)
        or np.any(sensing0 < 0.0) or np.any(weights < 0.0)
    ):
        raise ValueError("powers and sensing weights must be finite/non-negative")
    weight_sum = np.sum(weights, axis=1, keepdims=True)
    if np.any(weight_sum <= 0.0):
        raise ValueError("each UAV must have positive sensing-weight mass")
    weights = weights / weight_sum
    upper = float(reserve_upper_w)
    budget = float(total_power_w)
    tolerance = float(tolerance_w)
    if not 0.0 <= upper <= budget:
        raise ValueError("reserve_upper_w must lie in [0,total_power_w]")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance_w must be finite and positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    closure = dependency_closure(current, role, owner, move)

    def allocation(floor_w: float) -> tuple[np.ndarray, np.ndarray]:
        comm = comm0.copy()
        sensing = sensing0.copy()
        for participant in closure.participants:
            comm[participant] = max(comm[participant], float(floor_w))
            sensing[participant] = (
                budget - comm[participant]
            ) * weights[participant]
        return comm, sensing

    def evaluate(floor_w: float) -> tuple[
        DependencyCommitCertificate, np.ndarray, np.ndarray,
    ]:
        comm, sensing = allocation(floor_w)
        certificate = certify_best_dependency_commit(
            current,
            role,
            owner,
            move,
            positions=positions,
            comm_power_w=comm,
            sensing_power_w=sensing,
            state_versions=state_versions,
            certificate_epoch_ids=certificate_epoch_ids,
            certificate_digests=certificate_digests,
            communication_model=communication_model,
            total_power_w=budget,
            total_deadline_s=total_deadline_s,
            layout=layout,
            snr_margin_db=snr_margin_db,
            latency_margin_s=latency_margin_s,
        )
        return certificate, comm, sensing

    lower_certificate, lower_comm, lower_sensing = evaluate(0.0)
    if lower_certificate.feasible:
        return MinimumReserveResult(
            feasible=True,
            uniform_comm_floor_w=0.0,
            iterations=0,
            certificate=lower_certificate,
            comm_power_w=tuple(float(value) for value in lower_comm),
            sensing_power_w=tuple(
                tuple(float(value) for value in row)
                for row in lower_sensing
            ),
        )
    upper_certificate, upper_comm, upper_sensing = evaluate(upper)
    if not upper_certificate.feasible:
        return MinimumReserveResult(
            feasible=False,
            uniform_comm_floor_w=None,
            iterations=0,
            certificate=upper_certificate,
            comm_power_w=tuple(float(value) for value in upper_comm),
            sensing_power_w=tuple(
                tuple(float(value) for value in row)
                for row in upper_sensing
            ),
        )

    low, high = 0.0, upper
    iterations = 0
    best_certificate = upper_certificate
    best_comm = upper_comm
    best_sensing = upper_sensing
    while high - low > tolerance and iterations < int(max_iterations):
        middle = 0.5 * (low + high)
        certificate, comm, sensing = evaluate(middle)
        iterations += 1
        if certificate.feasible:
            high = middle
            best_certificate = certificate
            best_comm = comm
            best_sensing = sensing
        else:
            low = middle
    return MinimumReserveResult(
        feasible=True,
        uniform_comm_floor_w=float(high),
        iterations=iterations,
        certificate=best_certificate,
        comm_power_w=tuple(float(value) for value in best_comm),
        sensing_power_w=tuple(
            tuple(float(value) for value in row)
            for row in best_sensing
        ),
    )
