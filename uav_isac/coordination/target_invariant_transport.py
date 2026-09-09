"""Quantized transport for causal owner-local target invariants.

The physical cache used by the D0.18 controller stores a target-wise bistatic
coefficient, its age and a monotonically increasing version.  Keeping that
cache outside the wire protocol makes the replay optimistic.  This module
turns those fields into bounded packets, rounds the physical coefficient down
for safety, charges broadcast power/latency, and reconstructs missing or stale
targets as zero support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from uav_isac.coordination.owner_local_physics import (
    OwnerTargetInvariantCache,
)
from uav_isac.coordination.power_repair_transport import (
    _required_power_for_packet,
)
from uav_isac.domain.communication import CommunicationTransport


def _index_bits(cardinality: int) -> int:
    return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))


@dataclass(frozen=True)
class TargetInvariantWireLayout:
    num_agents: int
    num_targets: int
    header_bits: int = 64
    epoch_bits: int = 16
    digest_bits: int = 64
    invariant_bits: int = 16
    invariant_log2_min: float = -64.0
    invariant_log2_max: float = 64.0
    age_bits: int = 8
    version_bits: int = 16

    def __post_init__(self) -> None:
        if int(self.num_agents) < 2 or int(self.num_targets) < 1:
            raise ValueError("target-invariant layout requires K>=2 and Q>=1")
        for name in (
            "header_bits", "epoch_bits", "digest_bits", "invariant_bits",
            "age_bits", "version_bits",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.age_bits) < 1 or int(self.version_bits) < 1:
            raise ValueError("age and version fields require at least one bit")
        if int(self.invariant_bits) < 2:
            raise ValueError("invariant fields require at least two bits")
        if (
            not np.isfinite(self.invariant_log2_min)
            or not np.isfinite(self.invariant_log2_max)
            or float(self.invariant_log2_min)
            >= float(self.invariant_log2_max)
        ):
            raise ValueError("invariant log2 support is invalid")

    @property
    def node_bits(self) -> int:
        return _index_bits(self.num_agents)

    @property
    def target_bits(self) -> int:
        return _index_bits(self.num_targets)

    @property
    def max_age(self) -> int:
        return (1 << int(self.age_bits)) - 1

    @property
    def max_version(self) -> int:
        return (1 << int(self.version_bits)) - 1

    @property
    def record_bits(self) -> int:
        # Explicit validity prevents an all-zero physical value from being
        # confused with a successfully transported lower bound.
        return int(
            self.target_bits + 1 + self.invariant_bits
            + self.age_bits + self.version_bits)

    def packet_bits(self, record_count: int) -> int:
        count = int(record_count)
        if not 0 <= count <= int(self.num_targets):
            raise ValueError("target-invariant record count is invalid")
        return int(
            self.header_bits + self.epoch_bits + self.digest_bits
            + self.node_bits + _index_bits(self.num_targets + 1)
            + count * self.record_bits)

    def quantize_invariant_lower(self, value: float) -> float:
        """Directionally round a positive invariant on a log2 grid."""
        scalar = float(value)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError("target invariant must be finite and positive")
        log_value = float(np.log2(scalar))
        lower = float(self.invariant_log2_min)
        upper = float(self.invariant_log2_max)
        if log_value < lower or log_value > upper:
            raise ValueError("target invariant is outside logarithmic support")
        levels = (1 << int(self.invariant_bits)) - 1
        step = (upper - lower) / levels
        index = int(np.floor((log_value - lower) / step + 1.0e-12))
        index = min(max(index, 0), levels)
        decoded = float(np.exp2(lower + index * step))
        if decoded > scalar:
            index = max(index - 1, 0)
            decoded = float(np.exp2(lower + index * step))
        if decoded <= 0.0 or decoded > scalar * (1.0 + 1.0e-12):
            raise AssertionError("log invariant quantizer lost direction")
        return min(decoded, scalar)

    def invariant_cell_upper(self, encoded_lower: float) -> float:
        """Return the closed conservative upper edge of a quantizer cell."""
        lower_value = float(encoded_lower)
        if not np.isfinite(lower_value) or lower_value <= 0.0:
            raise ValueError("encoded invariant lower bound must be positive")
        support_lower = float(np.exp2(self.invariant_log2_min))
        support_upper = float(np.exp2(self.invariant_log2_max))
        if (
            lower_value < support_lower * (1.0 - 1.0e-12)
            or lower_value > support_upper * (1.0 + 1.0e-12)
        ):
            raise ValueError("encoded invariant is outside logarithmic support")
        levels = (1 << int(self.invariant_bits)) - 1
        step = (
            float(self.invariant_log2_max)
            - float(self.invariant_log2_min)
        ) / levels
        upper = float(np.exp2(np.log2(lower_value) + step))
        return min(max(upper, lower_value), support_upper)


@dataclass(frozen=True)
class TargetInvariantRecord:
    target: int
    valid: bool
    invariant_lower: float
    age_frames: int
    version: int


@dataclass(frozen=True)
class TargetInvariantToken:
    owner: int
    sent_frame: int
    records: tuple[TargetInvariantRecord, ...]
    over_air_bits: int


def encode_owner_target_invariant_tokens(
    cache: OwnerTargetInvariantCache,
    target_owner: np.ndarray,
    *,
    layout: TargetInvariantWireLayout,
    sent_frame: int,
) -> tuple[TargetInvariantToken, ...]:
    """Encode one sparse, safety-rounded packet per active receiver owner."""
    invariant = np.asarray(cache.target_invariant, dtype=np.float64).reshape(-1)
    age = np.asarray(cache.age_frames, dtype=np.int64).reshape(-1)
    version = np.asarray(cache.version, dtype=np.int64).reshape(-1)
    owner = np.asarray(target_owner, dtype=np.int64).reshape(-1)
    Q = int(layout.num_targets)
    if (
        invariant.shape != (Q,) or age.shape != (Q,)
        or version.shape != (Q,) or owner.shape != (Q,)
    ):
        raise ValueError("target-invariant cache and owner shapes disagree")
    if (
        np.any(~np.isfinite(invariant)) or np.any(invariant < 0.0)
        or np.any(age < 0) or np.any(version < 0)
        or np.any(owner < 0) or np.any(owner >= int(layout.num_agents))
    ):
        raise ValueError("target-invariant cache is outside wire support")
    frame = int(sent_frame)
    if frame < 0:
        raise ValueError("sent_frame must be non-negative")
    records_by_owner: dict[int, list[TargetInvariantRecord]] = {}
    for target in range(Q):
        if version[target] <= 0:
            continue
        if invariant[target] > 0.0 and age[target] > layout.max_age:
            raise ValueError("target-invariant age overflows its wire field")
        if version[target] > layout.max_version:
            raise ValueError("target-invariant version overflows its wire field")
        valid = bool(invariant[target] > 0.0)
        record = TargetInvariantRecord(
            target=target,
            valid=valid,
            invariant_lower=(
                layout.quantize_invariant_lower(float(invariant[target]))
                if valid else 0.0
            ),
            age_frames=int(min(age[target], layout.max_age)),
            version=int(version[target]),
        )
        records_by_owner.setdefault(int(owner[target]), []).append(record)
    tokens = []
    for sender in sorted(records_by_owner):
        records = tuple(sorted(
            records_by_owner[sender], key=lambda item: item.target))
        tokens.append(TargetInvariantToken(
            owner=sender,
            sent_frame=frame,
            records=records,
            over_air_bits=layout.packet_bits(len(records)),
        ))
    return tuple(tokens)


def decode_owner_target_invariant_tokens(
    tokens: Sequence[TargetInvariantToken],
    target_owner: np.ndarray,
    *,
    layout: TargetInvariantWireLayout,
    current_frame: int,
    max_age_frames: int,
    minimum_versions: np.ndarray | None = None,
) -> OwnerTargetInvariantCache:
    """Reconstruct owner-authoritative fields; missing/stale data fail closed."""
    owner = np.asarray(target_owner, dtype=np.int64).reshape(-1)
    Q = int(layout.num_targets)
    if owner.shape != (Q,) or np.any(owner < 0) or np.any(
        owner >= int(layout.num_agents)
    ):
        raise ValueError("target owner vector is invalid")
    frame = int(current_frame)
    max_age = int(max_age_frames)
    if frame < 0 or max_age < 0:
        raise ValueError("current frame and cache age must be non-negative")
    minimum = (
        np.zeros(Q, dtype=np.int64)
        if minimum_versions is None
        else np.asarray(minimum_versions, dtype=np.int64).reshape(-1)
    )
    if minimum.shape != (Q,) or np.any(minimum < 0):
        raise ValueError("minimum_versions must contain Q non-negative values")
    by_owner: dict[int, TargetInvariantToken] = {}
    for token in tokens:
        sender = int(token.owner)
        if not 0 <= sender < int(layout.num_agents):
            raise ValueError("target-invariant sender is outside the UAV set")
        if sender in by_owner:
            raise ValueError("each owner may send at most one invariant packet")
        if int(token.sent_frame) > frame or int(token.sent_frame) < 0:
            raise ValueError("target-invariant packet has an invalid timestamp")
        if int(token.over_air_bits) != layout.packet_bits(len(token.records)):
            raise ValueError("target-invariant packet bit count is inconsistent")
        targets = [int(record.target) for record in token.records]
        if len(targets) != len(set(targets)):
            raise ValueError("target-invariant packet repeats a target")
        if any(target < 0 or target >= Q for target in targets):
            raise ValueError(
                "target-invariant packet contains an invalid target")
        by_owner[sender] = token
    invariant = np.zeros(Q, dtype=np.float64)
    age = np.full(Q, max_age + 1, dtype=np.int64)
    version = np.zeros(Q, dtype=np.int64)
    for target in range(Q):
        token = by_owner.get(int(owner[target]))
        if token is None:
            continue
        matches = [
            record for record in token.records
            if int(record.target) == target
        ]
        if not matches:
            continue
        record = matches[0]
        valid = bool(record.valid)
        value = float(record.invariant_lower)
        record_age = int(record.age_frames)
        record_version = int(record.version)
        if (
            not np.isfinite(value) or value < 0.0
            or (valid and value <= 0.0)
            or (not valid and value != 0.0)
            or record_age < 0 or record_age > layout.max_age
            or record_version <= 0 or record_version > layout.max_version
        ):
            raise ValueError("target-invariant record is outside wire support")
        effective_age = record_age + frame - int(token.sent_frame)
        version[target] = record_version
        age[target] = effective_age
        if (
            valid and effective_age <= max_age
            and record_version >= minimum[target]
        ):
            invariant[target] = value
    return OwnerTargetInvariantCache(
        target_invariant=invariant,
        age_frames=age,
        version=version,
    )


@dataclass(frozen=True)
class TargetInvariantTransportCertificate:
    feasible: bool
    reasons: tuple[str, ...]
    packet_count: int
    record_count: int
    total_over_air_bits: int
    max_packet_latency_s: float
    total_protocol_latency_s: float
    total_energy_j: float
    min_snr_db: float
    required_comm_power_w: np.ndarray
    projected_comm_power_w: np.ndarray


def certify_target_invariant_transport(
    tokens: Sequence[TargetInvariantToken],
    *,
    positions: np.ndarray,
    existing_comm_power_w: np.ndarray,
    communication_model: CommunicationTransport,
    layout: TargetInvariantWireLayout,
    snr_margin_db: float = 0.0,
    latency_margin_s: float = 0.0,
) -> TargetInvariantTransportCertificate:
    """Certify parallel owner broadcasts to every other UAV."""
    items = tuple(tokens)
    K = int(layout.num_agents)
    pos = np.asarray(positions, dtype=np.float64)
    comm = np.asarray(existing_comm_power_w, dtype=np.float64).reshape(-1)
    if pos.shape != (K, 3) or comm.shape != (K,):
        raise ValueError("target-invariant transport state has invalid shape")
    if (
        np.any(~np.isfinite(pos)) or np.any(~np.isfinite(comm))
        or np.any(comm < 0.0) or np.any(comm >= 1.0)
    ):
        raise ValueError("target-invariant transport state is unsupported")
    senders = [int(token.owner) for token in items]
    if len(senders) != len(set(senders)):
        raise ValueError("each owner may send at most one invariant packet")
    for token in items:
        if int(token.over_air_bits) != layout.packet_bits(len(token.records)):
            raise ValueError("target-invariant packet bit count is inconsistent")
    if not items:
        zeros = np.zeros(K, dtype=np.float64)
        return TargetInvariantTransportCertificate(
            feasible=True,
            reasons=tuple(),
            packet_count=0,
            record_count=0,
            total_over_air_bits=0,
            max_packet_latency_s=0.0,
            total_protocol_latency_s=0.0,
            total_energy_j=0.0,
            min_snr_db=float("inf"),
            required_comm_power_w=zeros,
            projected_comm_power_w=comm.copy(),
        )
    bandwidth = communication_model.bandwidth_hz / len(items)
    required = np.zeros(K, dtype=np.float64)
    for token in items:
        sender = int(token.owner)
        if not 0 <= sender < K:
            raise ValueError("target-invariant sender is outside the UAV set")
        for receiver in range(K):
            if receiver == sender:
                continue
            required[sender] = max(
                required[sender],
                _required_power_for_packet(
                    communication_model,
                    pos[sender],
                    pos[receiver],
                    int(token.over_air_bits),
                    bandwidth,
                    snr_margin_db=float(snr_margin_db),
                    latency_margin_s=float(latency_margin_s),
                ),
            )
    projected = np.maximum(comm, required)
    reasons: list[str] = []
    if np.any(~np.isfinite(projected)) or np.any(projected >= 1.0):
        reasons.append("invariant:power:unavailable")
    max_latency = 0.0
    min_snr = float("inf")
    total_energy = 0.0
    for token in items:
        sender = int(token.owner)
        packet_serialization = 0.0
        for receiver in range(K):
            if receiver == sender:
                continue
            snr, _, serialization, latency = (
                communication_model.robust_link_budget(
                    pos[sender],
                    pos[receiver],
                    int(token.over_air_bits),
                    bandwidth,
                    float(projected[sender]),
                    snr_margin_db=float(snr_margin_db),
                    latency_margin_s=float(latency_margin_s),
                )
            )
            packet_serialization = max(
                packet_serialization, float(serialization))
            max_latency = max(max_latency, float(latency))
            min_snr = min(min_snr, float(snr))
            if snr < communication_model.snr_threshold_db:
                reasons.append(f"invariant:snr:{sender}->{receiver}")
            if latency > communication_model.deadline_s + 1.0e-12:
                reasons.append(f"invariant:deadline:{sender}->{receiver}")
        total_energy += float(projected[sender]) * packet_serialization
    return TargetInvariantTransportCertificate(
        feasible=not reasons,
        reasons=tuple(reasons),
        packet_count=len(items),
        record_count=sum(len(token.records) for token in items),
        total_over_air_bits=int(sum(
            int(token.over_air_bits) for token in items)),
        max_packet_latency_s=float(max_latency),
        total_protocol_latency_s=float(max_latency),
        total_energy_j=float(total_energy),
        min_snr_db=float(min_snr),
        required_comm_power_w=required,
        projected_comm_power_w=projected,
    )
