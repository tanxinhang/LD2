"""Receiver-local stochastic evidence consistent with the deflection detector.

The environment historically maps a cumulative deflection ``D`` directly to
the analytical detection probability

    P_D = Q(Q^{-1}(P_FA) - sqrt(D)).

For communication-value experiments we need a frame-level random sufficient
statistic rather than the already-fused probability.  Under the Gaussian
shift model used here, the exact log-likelihood ratio (LLR) has distribution

    L | H0 ~ Normal(-D / 2, D)
    L | H1 ~ Normal(+D / 2, D).

Independent receiver LLRs are additive, so local, U2U-distributed, and
central-oracle variants can reuse the same receiver samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from uav_isac.utils.math_utils import Q_inverse, compute_PD
from uav_isac.utils.sentinels import OWNER_INDEX_NONE


DETECTION_FUSION_MODES = frozenset({
    "legacy_global",
    "central_oracle",
    "local_only",
    "u2u_distributed",
})


def receiver_deflection_from_selected(
    selected_set: Sequence[Tuple[int, int, int]],
    deflection_entries: Iterable,
    num_agents: int,
    num_targets: int,
) -> np.ndarray:
    """Reconstruct receiver-local deflection without fusing receivers."""
    K, Q = int(num_agents), int(num_targets)
    if K < 1 or Q < 1:
        raise ValueError("num_agents and num_targets must be positive")
    entry_map = {
        (int(entry.i), int(entry.j), int(entry.q)): float(entry.d_eff)
        for entry in deflection_entries
    }
    receiver_d = np.zeros((K, Q), dtype=np.float64)
    for i_raw, j_raw, q_raw in selected_set:
        i, j, q = int(i_raw), int(j_raw), int(q_raw)
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q):
            raise ValueError("selected_set contains an out-of-range index")
        receiver_d[j, q] += entry_map.get((i, j, q), 0.0)
    return receiver_d


def receiver_deflection_from_broadcast_waveforms(
    selected_set: Sequence[Tuple[int, int, int]],
    deflection_entries: Iterable,
    num_agents: int,
    num_targets: int,
) -> np.ndarray:
    """Expose passive Rx statistics from each selected Tx-target waveform.

    The selected receiver endpoints define the nodes that are in receive mode
    for the sensing sub-slots.  A target-illuminating waveform is radiated once
    by its selected transmitter and can be observed by every such receiver;
    this does not duplicate sensing power.  Any later exchange of those local
    statistics is handled and charged by the physical U2U evidence transport.
    """
    K, Q = int(num_agents), int(num_targets)
    if K < 1 or Q < 1:
        raise ValueError("num_agents and num_targets must be positive")
    waveforms = set()
    receivers = set()
    for i_raw, j_raw, q_raw in selected_set:
        i, j, q = int(i_raw), int(j_raw), int(q_raw)
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q and i != j):
            raise ValueError("selected_set contains an invalid index")
        waveforms.add((i, q))
        receivers.add(j)

    receiver_d = np.zeros((K, Q), dtype=np.float64)
    for entry in deflection_entries:
        i, j, q = int(entry.i), int(entry.j), int(entry.q)
        value = float(entry.d_eff)
        if (
            (i, q) in waveforms
            and j in receivers
            and i != j
            and 0 <= j < K
            and 0 <= q < Q
            and np.isfinite(value)
            and value > 0.0
        ):
            receiver_d[j, q] += value
    return receiver_d


def scheduled_fusion_owner(
    selected_set: Sequence[Tuple[int, int, int]],
    num_agents: int,
    num_targets: int,
) -> np.ndarray:
    """Derive one target owner from the already-broadcast sensing schedule.

    The structure layer has already transported endpoint offers and retained
    mutually endorsed hyperedges before an evidence packet is emitted.  Reuse
    that public schedule instead of reading the global receiver-quality matrix.
    If a legacy schedule contains several receiver endpoints for one target,
    the lowest receiver identifier is the deterministic, quality-free tie
    break that every replica can compute.  Targets absent from the schedule are
    marked ``-1`` and cannot receive peer evidence.

    This is the compact middle level of the broadcast hierarchy:

    1. endpoint state/offer broadcast,
    2. deterministic scheduled-owner directory,
    3. receiver evidence broadcast.

    Level 2 adds no evidence-packet bits because the owner is already encoded
    by the receiver field of the scheduled hyperedge.
    """
    K, Q = int(num_agents), int(num_targets)
    if K < 1 or Q < 1:
        raise ValueError("num_agents and num_targets must be positive")
    owners = np.full(Q, -1, dtype=np.int64)
    receiver_sets = [set() for _ in range(Q)]
    for i_raw, j_raw, q_raw in selected_set:
        i, j, q = int(i_raw), int(j_raw), int(q_raw)
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q and i != j):
            raise ValueError("selected_set contains an invalid index")
        receiver_sets[q].add(j)
    for q, receivers in enumerate(receiver_sets):
        if receivers:
            owners[q] = min(receivers)
    return owners


def quantize_belief_feedback(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    area_size_xy: Tuple[float, float],
    velocity_bound_mps: float,
    mean_bits: int,
    covariance_bits: int,
    acceleration_bound_mps2: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize one CV or CA posterior conservatively.

    Position/velocity coordinates use bounded uniform quantization.  The
    deterministic nearest-level error bound ``Delta^2/4`` is added to the
    decoded covariance (``Delta^2/12`` would require a stochastic uniform-
    noise assumption).  Before log quantization, each covariance diagonal is
    inflated by the absolute off-diagonal row sum.  The resulting diagonal
    matrix dominates the full symmetric covariance by the Gershgorin bound,
    so omitting cross-covariances cannot advertise excess confidence.
    """
    state = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov = np.asarray(covariance, dtype=np.float64)
    if state.shape not in {(4,), (6,)} or cov.shape != (
            state.size, state.size):
        raise ValueError('belief feedback requires a 4D or 6D mean/covariance')
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(cov))):
        raise ValueError('belief feedback must be finite')
    bits_m = max(1, int(mean_bits))
    bits_c = max(1, int(covariance_bits))
    width, height = (max(float(v), 1.0e-9) for v in area_size_xy)
    speed = max(float(velocity_bound_mps), 1.0e-9)
    lower = [0.0, 0.0, -speed, -speed]
    upper = [width, height, speed, speed]
    if state.size == 6:
        acceleration = max(
            float(1.0 if acceleration_bound_mps2 is None
                  else acceleration_bound_mps2),
            1.0e-9,
        )
        lower.extend((-acceleration, -acceleration))
        upper.extend((acceleration, acceleration))
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    levels_m = max(2, 1 << bits_m)
    step = (upper - lower) / float(levels_m - 1)
    clipped = np.clip(state, lower, upper)
    code = np.rint((clipped - lower) / step)
    decoded_mean = lower + code * step
    quantization_variance = step * step / 4.0

    symmetric_cov = 0.5 * (cov + cov.T)
    covariance_row_bound = (
        np.diag(symmetric_cov)
        + np.sum(np.abs(symmetric_cov), axis=1)
        - np.abs(np.diag(symmetric_cov))
    )
    # If a state lies outside the advertised support, conservatively cover
    # the deterministic clipping bias e e^T by a diagonal-dominant bound.
    clipping_error = state - clipped
    clipping_row_bound = (
        np.abs(clipping_error) * np.sum(np.abs(clipping_error)))
    variance = np.maximum(
        covariance_row_bound + clipping_row_bound, 1.0e-9)
    log_min = np.log(1.0e-6)
    log_max = np.log(1.0e8)
    levels_c = max(2, 1 << bits_c)
    log_step = (log_max - log_min) / float(levels_c - 1)
    # Upper-bin decoding is conservative for every in-range variance.
    cov_code = np.ceil((np.log(variance) - log_min) / log_step)
    cov_code = np.clip(cov_code, 0.0, levels_c - 1.0)
    decoded_variance = np.exp(log_min + cov_code * log_step)
    decoded_variance += quantization_variance
    return decoded_mean, np.diag(decoded_variance)


def select_detection_deflection(
    mode: str,
    receiver_deflection: np.ndarray,
    *,
    legacy_global_deflection: Optional[np.ndarray] = None,
    distributed_deflection: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resolve detector evidence under an explicit fusion boundary."""
    normalized = str(mode).strip().lower()
    if normalized not in DETECTION_FUSION_MODES:
        allowed = ", ".join(sorted(DETECTION_FUSION_MODES))
        raise ValueError(
            f"unsupported detection_fusion_mode={mode!r}; expected {allowed}")
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    if receiver_d.ndim != 2 or min(receiver_d.shape) < 1:
        raise ValueError("receiver_deflection must have shape (K, Q)")
    if np.any(~np.isfinite(receiver_d)) or np.any(receiver_d < 0.0):
        raise ValueError(
            "receiver_deflection must contain finite non-negative values")

    if normalized == "local_only":
        return np.max(receiver_d, axis=0)
    if normalized == "central_oracle":
        return np.sum(receiver_d, axis=0)
    if normalized == "legacy_global":
        if legacy_global_deflection is None:
            return np.sum(receiver_d, axis=0)
        legacy = np.asarray(legacy_global_deflection, dtype=np.float64)
        if legacy.shape != (receiver_d.shape[1],):
            raise ValueError(
                "legacy_global_deflection must have shape (Q,)")
        return legacy.copy()

    if distributed_deflection is None:
        raise RuntimeError(
            "u2u_distributed detection requires evidence reconstructed from "
            "delivered structured packets")
    distributed = np.asarray(distributed_deflection, dtype=np.float64)
    if distributed.shape != (receiver_d.shape[1],):
        raise ValueError("distributed_deflection must have shape (Q,)")
    if np.any(~np.isfinite(distributed)) or np.any(distributed < 0.0):
        raise ValueError(
            "distributed_deflection must contain finite non-negative values")
    return distributed.copy()


@dataclass(frozen=True)
class EvidencePacketLayout:
    """Explicit bit accounting for one structured evidence broadcast.

    Source and timestamp are shared by all evidence entries in a broadcast.
    Together with each target ID they form the de-duplication identity, so a
    separate unbounded evidence identifier is neither needed nor free.
    """

    num_agents: int
    num_targets: int
    header_bits: int = 64
    timestamp_bits: int = 16
    confidence_bits: int = 0
    feedback_bits_per_entry: int = 0

    @staticmethod
    def _index_bits(cardinality: int) -> int:
        return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))

    @property
    def source_bits(self) -> int:
        return self._index_bits(self.num_agents)

    @property
    def target_bits(self) -> int:
        return self._index_bits(self.num_targets)

    def broadcast_bits(
        self,
        num_entries: int,
        llr_bits: int,
        belief_entries: Optional[int] = None,
    ) -> int:
        entries = max(0, int(num_entries))
        precision = max(0, int(llr_bits))
        if entries == 0 or precision == 0:
            return 0
        shared = (
            max(0, int(self.header_bits))
            + self.source_bits
            + max(0, int(self.timestamp_bits))
        )
        per_entry = (
            self.target_bits
            + precision
            + max(0, int(self.confidence_bits))
        )
        # Belief feedback is a per-entry payload on top of the structured
        # evidence broadcast.  When ``belief_entries`` is None it rides on
        # every evidence entry (the legacy coupled schedule); an explicit
        # count lets a decoupled freshness schedule charge only the entries
        # that actually carry a quantized posterior.
        belief = (
            entries
            if belief_entries is None
            else max(0, int(belief_entries))
        )
        return shared + entries * per_entry + belief * max(
            0, int(self.feedback_bits_per_entry))


@dataclass(frozen=True)
class EvidencePacket:
    """One structured broadcast generated from receiver-local sensing.

    The evidence identifier is ``(observation_frame, source_rx, target_id)``.
    A broadcast can carry several target entries while source and timestamp
    metadata are paid only once.
    """

    source_rx: int
    observation_frame: int
    target_ids: Tuple[int, ...]
    llr_bits: int
    confidence_bits: int
    payload_bits: int

    @property
    def evidence_ids(self) -> Tuple[Tuple[int, int, int], ...]:
        return tuple(
            (self.observation_frame, self.source_rx, int(target))
            for target in self.target_ids
        )


@dataclass
class EvidenceTransportResult:
    """Routing and physical-delivery result for one sensing frame."""

    packets: Tuple[EvidencePacket, ...]
    owner: np.ndarray
    selected_mask: np.ndarray
    owner_mask: np.ndarray
    delivered_peer_mask: np.ndarray
    delivery_matrix: np.ndarray
    payload_bits_by_sender: np.ndarray
    energy_j_by_sender: np.ndarray
    total_bits: float
    total_energy_j: float
    active_senders: int
    attempted_links: int
    delivered_links: int
    expired_links: int
    deadline_failed_links: int
    snr_failed_links: int
    reliability_failed_links: int
    burst_failed_links: int
    burst_bad_links: int
    service_envelope_filtered_links: int
    useful_unique_entries: int
    mean_latency_s: float
    p95_latency_s: float
    max_latency_s: float
    adaptive_extra_entries: int = 0
    belief_selected_mask: Optional[np.ndarray] = None

    @property
    def delivery_rate(self) -> float:
        return float(
            self.delivered_links / max(self.attempted_links, 1))

    @property
    def deadline_violation_rate(self) -> float:
        return float(
            self.deadline_failed_links / max(self.attempted_links, 1))

    @property
    def delivery_failure_rate(self) -> float:
        return float(self.expired_links / max(self.attempted_links, 1))

    @property
    def evidence_utilization(self) -> float:
        transmitted = int(np.sum(self.selected_mask))
        return float(self.useful_unique_entries / max(transmitted, 1))

    @property
    def belief_schedule_mask(self) -> np.ndarray:
        """Return the belief-fusion schedule actually used by the receiver.

        A decoupled freshness schedule supplies its own mask; the legacy
        coupled behaviour rides the belief on the evidence selection.
        """
        if self.belief_selected_mask is None:
            return np.asarray(self.selected_mask, dtype=bool)
        return np.asarray(self.belief_selected_mask, dtype=bool)

    def as_dict(self) -> Dict[str, object]:
        return {
            "evidence_comm_bits": float(self.total_bits),
            "evidence_comm_energy_j": float(self.total_energy_j),
            "evidence_comm_active_senders": float(self.active_senders),
            "evidence_comm_attempted_links": float(self.attempted_links),
            "evidence_comm_delivered_links": float(self.delivered_links),
            "evidence_comm_delivery_rate": self.delivery_rate,
            "evidence_comm_deadline_violation_rate": (
                self.deadline_violation_rate),
            "evidence_comm_delivery_failure_rate": (
                self.delivery_failure_rate),
            "evidence_comm_snr_violation_rate": float(
                self.snr_failed_links / max(self.attempted_links, 1)),
            "evidence_comm_reliability_violation_rate": float(
                self.reliability_failed_links / max(self.attempted_links, 1)),
            "evidence_comm_burst_failure_rate": float(
                self.burst_failed_links / max(self.attempted_links, 1)),
            "evidence_comm_burst_bad_link_rate": float(
                self.burst_bad_links / max(self.attempted_links, 1)),
            "evidence_comm_mean_latency_s": float(self.mean_latency_s),
            "evidence_comm_p95_latency_s": float(self.p95_latency_s),
            "evidence_comm_max_latency_s": float(self.max_latency_s),
            "evidence_comm_service_envelope_filtered_links": float(
                self.service_envelope_filtered_links),
            "evidence_transmitted_entries": float(
                np.sum(self.selected_mask)),
            "evidence_adaptive_extra_entries": float(
                self.adaptive_extra_entries),
            "evidence_useful_unique_entries": float(
                self.useful_unique_entries),
            "evidence_utilization": self.evidence_utilization,
            "evidence_owner": self.owner.copy(),
            "evidence_owner_source": "scheduled_hyperedge_directory",
            "evidence_owner_unassigned_targets": float(np.sum(
                np.asarray(self.owner, dtype=np.int64) < 0)),
            "evidence_owner_incremental_bits": 0.0,
            "evidence_broadcast_levels": 3.0,
            "evidence_selected_mask": self.selected_mask.copy(),
            "evidence_delivered_peer_mask": (
                self.delivered_peer_mask.copy()),
            "evidence_belief_schedule_entries": float(
                np.sum(self.belief_schedule_mask)),
        }


def route_structured_evidence(
    receiver_deflection: np.ndarray,
    positions: np.ndarray,
    comm_power_w: np.ndarray,
    *,
    observation_frame: int,
    topk: int,
    llr_bits: int,
    layout: EvidencePacketLayout,
    link_model,
    fusion_owner: np.ndarray,
    owner_aware: bool = True,
    ambiguity_second_ratio: Optional[float] = None,
    ambiguity_second_min_deflection: float = 0.0,
    belief_selected_mask: Optional[np.ndarray] = None,
    service_envelope_layout: Optional[EvidencePacketLayout] = None,
) -> EvidenceTransportResult:
    """Select local evidence and transport structured broadcasts.

    Target selection is based only on each receiver's pre-observation local
    quality.  The fusion owner is supplied by the previously agreed sensing
    schedule; this function never elects an owner from the global quality
    matrix. Broadcast delivery is evaluated for every peer, but an entry enters
    the detector only when it reaches that target's scheduled owner.

    ``belief_selected_mask`` optionally decouples the posterior-feedback
    schedule from the evidence selection. The packet carries the union of the
    two target sets, while every selected posterior pays its explicit
    per-entry payload even when its target ID is shared with an evidence entry.
    When it is None the legacy coupled schedule selects a posterior for every
    evidence entry and charges every one.
    """
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    xyz = np.asarray(positions, dtype=np.float64)
    power = np.asarray(comm_power_w, dtype=np.float64)
    if receiver_d.ndim != 2:
        raise ValueError("receiver_deflection must have shape (K, Q)")
    K, Q = receiver_d.shape
    if xyz.shape != (K, 3):
        raise ValueError("positions must have shape (K, 3)")
    if power.shape != (K,):
        raise ValueError("comm_power_w must have shape (K,)")
    if belief_selected_mask is not None:
        belief_mask = np.asarray(belief_selected_mask, dtype=bool)
        if belief_mask.shape != (K, Q):
            raise ValueError(
                "belief_selected_mask must have shape (K, Q)")

    owner = np.asarray(fusion_owner, dtype=np.int64).reshape(-1)
    if owner.shape != (Q,):
        raise ValueError("fusion_owner must have shape (Q,)")
    if np.any((owner < OWNER_INDEX_NONE) | (owner >= K)):
        raise ValueError("fusion_owner contains an invalid receiver index")
    owned_target = owner >= 0
    owner_mask = np.zeros((K, Q), dtype=bool)
    owner_mask[owner[owned_target], np.arange(Q)[owned_target]] = True
    selection_quality = receiver_d.copy()
    selection_quality[:, ~owned_target] = 0.0
    if owner_aware:
        selection_quality[owner_mask] = 0.0
    base_selected = local_quality_topk_mask(selection_quality, topk)
    selected = base_selected
    if ambiguity_second_ratio is not None:
        selected = local_ambiguity_top2_mask(
            selection_quality,
            base_topk=topk,
            second_ratio=float(ambiguity_second_ratio),
            second_min_deflection=float(
                ambiguity_second_min_deflection),
        )
    adaptive_extra_entries = int(
        np.count_nonzero(selected & ~base_selected))

    # Belief schedule semantics: the posterior mask is independent from the
    # evidence mask. Target metadata may be shared in the union packet, but
    # the posterior state itself is always charged.
    if belief_selected_mask is not None:
        belief_selected = belief_mask
        belief_entries_by_sender = [
            int(np.sum(belief_selected[source])) for source in range(K)]
    else:
        belief_selected = np.asarray(selected, dtype=bool)
        belief_entries_by_sender = [
            int(np.sum(selected[source])) for source in range(K)]
    union_selected = np.asarray(selected, dtype=bool) | belief_selected

    payload_bits = np.asarray([
        layout.broadcast_bits(
            int(np.sum(union_selected[source])),
            llr_bits,
            belief_entries=belief_entries_by_sender[source],
        )
        for source in range(K)
    ], dtype=np.int64)
    service_payload_bits = (
        np.asarray([
            service_envelope_layout.broadcast_bits(
                int(np.sum(union_selected[source])),
                llr_bits,
                belief_entries=belief_entries_by_sender[source],
            )
            for source in range(K)
        ], dtype=np.int64)
        if service_envelope_layout is not None else payload_bits
    )
    active = np.flatnonzero(payload_bits > 0)
    delivery = np.zeros((K, K), dtype=bool)
    packets = []
    for source in active:
        targets = tuple(
            int(q) for q in np.flatnonzero(union_selected[source]))
        packets.append(EvidencePacket(
            source_rx=int(source),
            observation_frame=int(observation_frame),
            target_ids=targets,
            llr_bits=int(llr_bits),
            confidence_bits=int(layout.confidence_bits),
            payload_bits=int(payload_bits[source]),
        ))

    # Reuse the exact physical broadcast path used by coordination packets.
    # This applies FBL coding, random SNR, deadline, burst erasure, energy and
    # channel-memory updates once per evidence sub-slot.  Only the explicit
    # payload length matters here; evidence values are decoded separately from
    # the structured packet definition above.
    messages = {
        int(source): np.zeros(link_model.message_dim, dtype=np.float64)
        for source in active
    }
    extra_bits = {}
    evidence_headers = {}
    for source in active:
        # The structured layout is the source of truth for its physical
        # header.  A synchronous fixed-schema evidence sub-slot may use a
        # compact CRC header while retaining explicitly charged source/target
        # indices.  The transport override keeps that header inside the FBL
        # codeword, latency and energy accounting.
        physical_header = max(0, int(layout.header_bits))
        remainder = int(payload_bits[source]) - physical_header
        if remainder <= 0:
            raise ValueError(
                'evidence packet layout must include the physical header')
        extra_bits[int(source)] = remainder
        evidence_headers[int(source)] = physical_header
    deliveries, comm_stats = link_model.transmit(
        messages,
        {int(source): 0 for source in active},
        xyz,
        tx_powers_w={
            int(source): float(max(power[source], 0.0))
            for source in active
        },
        extra_payload_bits=extra_bits,
        base_payload_dimensions={int(source): 0 for source in active},
        suppress_message_payload={int(source): True for source in active},
        header_bits_by_sender=evidence_headers,
        service_envelope_payload_bits_by_sender={
            int(source): int(service_payload_bits[source])
            for source in active
        },
    )
    for item in deliveries:
        delivery[int(item.sender), int(item.receiver)] = True
    energy_by_sender = np.asarray([
        float(comm_stats.per_sender_energy_j.get(source, 0.0))
        for source in range(K)
    ], dtype=np.float64)

    delivered_to_owner = np.zeros((K, Q), dtype=bool)
    for target in np.flatnonzero(owned_target):
        delivered_to_owner[:, target] = delivery[:, owner[target]]
    delivered_peer = selected & ~owner_mask & delivered_to_owner

    # Evidence IDs are unique by construction. Keep an explicit set here so
    # later multi-hop/retransmission support cannot double-count a statistic.
    unique_ids = set()
    useful = 0
    for source, target in zip(*np.nonzero(delivered_peer)):
        evidence_id = (
            int(observation_frame), int(source), int(target))
        if evidence_id not in unique_ids:
            unique_ids.add(evidence_id)
            useful += 1

    return EvidenceTransportResult(
        packets=tuple(packets),
        owner=owner,
        selected_mask=selected,
        owner_mask=owner_mask,
        delivered_peer_mask=delivered_peer,
        delivery_matrix=delivery,
        payload_bits_by_sender=payload_bits,
        energy_j_by_sender=energy_by_sender,
        total_bits=float(comm_stats.total_bits),
        total_energy_j=float(comm_stats.total_energy_j),
        active_senders=int(comm_stats.active_senders),
        attempted_links=int(comm_stats.attempted_links),
        delivered_links=int(comm_stats.delivered_links),
        expired_links=int(comm_stats.expired_links),
        deadline_failed_links=int(comm_stats.deadline_failed_links),
        snr_failed_links=int(comm_stats.snr_failed_links),
        reliability_failed_links=int(comm_stats.reliability_failed_links),
        burst_failed_links=int(comm_stats.burst_failed_links),
        burst_bad_links=int(comm_stats.burst_bad_links),
        service_envelope_filtered_links=int(
            comm_stats.service_envelope_filtered_links),
        useful_unique_entries=int(useful),
        mean_latency_s=float(comm_stats.mean_latency_s),
        p95_latency_s=float(comm_stats.p95_latency_s),
        max_latency_s=float(comm_stats.max_latency_s),
        adaptive_extra_entries=adaptive_extra_entries,
        belief_selected_mask=np.asarray(belief_selected, dtype=bool),
    )


@dataclass(frozen=True)
class DeflectionConfidenceQuantizer:
    """Finite-level decoder for a packet's local evidence quality class."""

    bits: int
    log_boundaries: np.ndarray
    representatives: np.ndarray

    def quantize(self, deflection: np.ndarray) -> np.ndarray:
        values = np.asarray(deflection, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "deflection must contain finite non-negative values")
        result = np.zeros_like(values)
        positive = values > 0.0
        if np.any(positive):
            indices = np.searchsorted(
                self.log_boundaries,
                np.log1p(values[positive]),
                side="right",
            )
            result[positive] = self.representatives[indices]
        return result


def calibrate_deflection_confidence(
    deflection: np.ndarray,
    bits: int = 2,
) -> DeflectionConfidenceQuantizer:
    """Calibrate log-domain quality classes on a separate evidence trace."""
    values = np.asarray(deflection, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0.0)]
    precision = int(bits)
    if precision <= 0:
        raise ValueError("confidence bits must be positive")
    if positive.size == 0:
        raise ValueError("calibration trace has no positive deflection")
    levels = 1 << precision
    log_values = np.log1p(positive)
    probabilities = np.arange(1, levels) / levels
    boundaries = np.quantile(log_values, probabilities)
    bin_index = np.searchsorted(boundaries, log_values, side="right")
    representatives = np.zeros(levels, dtype=np.float64)
    global_median = float(np.median(positive))
    for level in range(levels):
        in_bin = positive[bin_index == level]
        representatives[level] = (
            float(np.median(in_bin))
            if in_bin.size else global_median)
    return DeflectionConfidenceQuantizer(
        bits=precision,
        log_boundaries=np.asarray(boundaries, dtype=np.float64),
        representatives=representatives,
    )


def quantize_llr(
    llr: np.ndarray,
    bits: int,
    clip_max: float,
) -> np.ndarray:
    """Symmetric uniform LLR quantization with an explicit clipping range."""
    values = np.asarray(llr, dtype=np.float64)
    precision = int(bits)
    limit = float(clip_max)
    if precision <= 0:
        return np.zeros_like(values)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("clip_max must be finite and positive")
    clipped = np.clip(values, -limit, limit)
    # Signed mid-tread quantization keeps zero exactly representable.  One of
    # the 2**bits binary patterns remains unused, which is preferable to the
    # systematic non-zero bias of an even-level endpoint quantizer.
    max_code = max(1, (1 << (precision - 1)) - 1)
    codes = np.rint(clipped * max_code / limit)
    return limit * codes / max_code


def detection_probability_to_deflection(
    probability: np.ndarray,
    p_fa: float,
) -> np.ndarray:
    """Map detector probability back to an equivalent deflection.

    This is used only to feed the existing utility implementation after a
    quantized detector has directly estimated ``P_D``. It does not expose
    deflection to a communicating UAV.
    """
    pd = np.asarray(probability, dtype=np.float64)
    floor = float(p_fa)
    if not 0.0 < floor < 1.0:
        raise ValueError("p_fa must lie strictly between zero and one")
    clipped = np.clip(pd, floor, 1.0 - 1e-12)
    root = (
        float(Q_inverse(np.asarray(floor)))
        - Q_inverse(clipped)
    )
    return np.maximum(root, 0.0) ** 2


def estimate_quantized_evidence_detection(
    receiver_deflection: np.ndarray,
    transport: EvidenceTransportResult,
    *,
    llr_bits: int,
    clip_max: float,
    standardized_threshold: float,
    p_fa: float,
    confidence_quantizer: Optional[DeflectionConfidenceQuantizer] = None,
    draws: int = 2048,
    seed: int = 20260725,
    content_mode: str = "normal",
) -> Dict[str, object]:
    """Estimate one frame's P_D/P_FA from delivered quantized LLR packets.

    Owner-local evidence is unquantized. Only unique selected peer entries
    delivered to the predesignated owner are quantized and fused. H0 and H1
    draws are independent; all content controls reuse the same route, packet
    lengths, powers, and delivery events.
    """
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    if receiver_d.ndim != 2:
        raise ValueError("receiver_deflection must have shape (K, Q)")
    K, Q = receiver_d.shape
    if transport.owner_mask.shape != (K, Q):
        raise ValueError("transport shape does not match receiver_deflection")
    peer_mask = np.asarray(
        transport.delivered_peer_mask, dtype=bool)
    owner_mask = np.asarray(transport.owner_mask, dtype=bool)
    sample_count = max(1, int(draws))
    rng = np.random.default_rng(int(seed))
    shape = (sample_count, K, Q)
    noise_h0 = rng.standard_normal(shape)
    noise_h1 = rng.standard_normal(shape)
    normalized_content = str(content_mode).strip().lower()
    if confidence_quantizer is None:
        peer_d_hat = receiver_d
    else:
        peer_d_hat = confidence_quantizer.quantize(receiver_d)

    if normalized_content == "normal":
        # The physical evidence graph is sparse (one owner and only delivered
        # peers per target), while the historical implementation evaluated and
        # quantized every one of the K*Q receiver-target cells.  Preserve both
        # full RNG draws above: compacting the random stream would change the
        # finite-Monte-Carlo detector.  Gather only the active arithmetic, then
        # scatter it into a zero contribution tensor so ``sum(axis=1)`` retains
        # the exact historical receiver order and floating-point reduction.
        peer_i, peer_q = np.nonzero(peer_mask)
        owner_i, owner_q = np.nonzero(owner_mask)

        peer_d = receiver_d[peer_i, peer_q]
        peer_root = np.sqrt(peer_d[None])
        raw_peer_h0 = (
            -0.5 * peer_d[None]
            + peer_root * noise_h0[:, peer_i, peer_q]
        )
        raw_peer_h1 = (
            +0.5 * peer_d[None]
            + peer_root * noise_h1[:, peer_i, peer_q]
        )
        quantized_peer_h0 = quantize_llr(
            raw_peer_h0, llr_bits, clip_max)
        quantized_peer_h1 = quantize_llr(
            raw_peer_h1, llr_bits, clip_max)

        contribution_h0 = np.zeros(shape, dtype=np.float64)
        contribution_h1 = np.zeros(shape, dtype=np.float64)
        contribution_h0[:, peer_i, peer_q] = quantized_peer_h0
        contribution_h1[:, peer_i, peer_q] = quantized_peer_h1

        owner_d = receiver_d[owner_i, owner_q]
        owner_root = np.sqrt(owner_d[None])
        owner_h0 = (
            -0.5 * owner_d[None]
            + owner_root * noise_h0[:, owner_i, owner_q]
        )
        owner_h1 = (
            +0.5 * owner_d[None]
            + owner_root * noise_h1[:, owner_i, owner_q]
        )
        # Owner-local evidence has precedence even for a defensive overlapping
        # mask.  Peer values are nevertheless kept above for diagnostics, just
        # as in the full-tensor reference implementation.
        contribution_h0[:, owner_i, owner_q] = owner_h0
        contribution_h1[:, owner_i, owner_q] = owner_h1
        fused_h0 = np.sum(contribution_h0, axis=1)
        fused_h1 = np.sum(contribution_h1, axis=1)

        # ``np.nonzero`` is row-major, so flattening these (draw, edge) arrays
        # matches boolean indexing of the old (draw, receiver, target) tensor.
        raw_values = np.concatenate([
            raw_peer_h0.reshape(-1),
            raw_peer_h1.reshape(-1),
        ])
        quantized_values = np.concatenate([
            quantized_peer_h0.reshape(-1),
            quantized_peer_h1.reshape(-1),
        ])
    else:
        local_h0 = (
            -0.5 * receiver_d[None]
            + np.sqrt(receiver_d[None]) * noise_h0
        )
        local_h1 = (
            +0.5 * receiver_d[None]
            + np.sqrt(receiver_d[None]) * noise_h1
        )

    if normalized_content == "standardized_score":
        # z=(L+D/2)/sqrt(D) is N(0,1) under H0 for every D>0.  Quantizing
        # this scale-free innovation avoids sacrificing weak-evidence
        # resolution when fleet-scale geometry produces a heavy D tail.
        raw_peer_h0 = np.where(
            receiver_d[None] > 0.0, noise_h0, 0.0)
        raw_peer_h1 = np.where(
            receiver_d[None] > 0.0,
            np.sqrt(receiver_d[None]) + noise_h1,
            0.0,
        )
        quantized_peer_h0 = quantize_llr(
            raw_peer_h0, llr_bits, clip_max)
        quantized_peer_h1 = quantize_llr(
            raw_peer_h1, llr_bits, clip_max)
        quantized_h0 = (
            -0.5 * peer_d_hat[None]
            + np.sqrt(peer_d_hat[None]) * quantized_peer_h0)
        quantized_h1 = (
            -0.5 * peer_d_hat[None]
            + np.sqrt(peer_d_hat[None]) * quantized_peer_h1)
    elif normalized_content != "normal":
        raw_peer_h0 = local_h0
        raw_peer_h1 = local_h1
        quantized_peer_h0 = quantize_llr(
            local_h0, llr_bits, clip_max)
        quantized_peer_h1 = quantize_llr(
            local_h1, llr_bits, clip_max)
        quantized_h0 = quantized_peer_h0
        quantized_h1 = quantized_peer_h1

    if normalized_content == "zero":
        quantized_h0 = np.zeros_like(quantized_h0)
        quantized_h1 = np.zeros_like(quantized_h1)
    elif normalized_content == "value_roll":
        # Target IDs and packet events stay fixed while their values are
        # attached to the previous target. Under target-specific H1, rolled
        # values must come from H0 rather than another present target.
        quantized_h0 = np.roll(quantized_h0, shift=1, axis=-1)
        quantized_h1 = np.roll(
            quantize_llr(local_h0, llr_bits, clip_max),
            shift=1,
            axis=-1,
        )
    elif normalized_content not in {"normal", "standardized_score"}:
        raise ValueError(
            "content_mode must be normal, standardized_score, zero, or "
            "value_roll")

    if normalized_content != "normal":
        fused_h0 = np.sum(np.where(
            owner_mask[None],
            local_h0,
            np.where(peer_mask[None], quantized_h0, 0.0),
        ), axis=1)
        fused_h1 = np.sum(np.where(
            owner_mask[None],
            local_h1,
            np.where(peer_mask[None], quantized_h1, 0.0),
        ), axis=1)
        transmitted = np.broadcast_to(peer_mask[None], local_h0.shape)
        raw_values = np.concatenate([
            raw_peer_h0[transmitted],
            raw_peer_h1[transmitted],
        ])
        quantized_values = np.concatenate([
            quantized_peer_h0[transmitted],
            quantized_peer_h1[transmitted],
        ])

    threshold_d = np.sum(np.where(
        owner_mask,
        receiver_d,
        np.where(peer_mask, peer_d_hat, 0.0),
    ), axis=0)
    threshold = (
        -0.5 * threshold_d
        + np.sqrt(threshold_d) * float(standardized_threshold)
    )
    positive = threshold_d > 0.0
    observed_pfa = np.where(
        positive,
        np.mean(fused_h0 > threshold[None], axis=0),
        float(p_fa),
    )
    pd = np.where(
        positive,
        np.mean(fused_h1 > threshold[None], axis=0),
        float(p_fa),
    )

    if raw_values.size:
        clip_rate = float(np.mean(
            np.abs(raw_values) > float(clip_max)))
        quantization_mse = float(np.mean(
            (quantized_values - raw_values) ** 2))
        nonzero = np.abs(raw_values) > 1e-15
        sign_flip_rate = float(np.mean(
            np.signbit(raw_values[nonzero])
            != np.signbit(quantized_values[nonzero])
        )) if np.any(nonzero) else 0.0
    else:
        clip_rate = quantization_mse = sign_flip_rate = 0.0

    fused_d = np.sum(np.where(
        owner_mask | peer_mask, receiver_d, 0.0), axis=0)
    return {
        "pd": np.asarray(pd, dtype=np.float64),
        "equivalent_deflection": detection_probability_to_deflection(
            pd, p_fa),
        "pfa": np.asarray(observed_pfa, dtype=np.float64),
        "aggregate_pfa": float(np.mean(observed_pfa)),
        "threshold_deflection": threshold_d,
        "available_true_deflection": fused_d,
        "clip_rate": clip_rate,
        "quantization_mse": quantization_mse,
        "sign_flip_rate": sign_flip_rate,
        "draws": int(sample_count),
        "content_mode": normalized_content,
    }


def local_quality_topk_mask(
    receiver_deflection: np.ndarray,
    topk: int,
) -> np.ndarray:
    """Select positive target evidence using only each receiver's local quality.

    Supports either one frame ``(K, Q)`` or a frame batch ``(..., K, Q)``.
    Ties are resolved by target index through a stable descending sort.
    """
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    if receiver_d.ndim < 2:
        raise ValueError(
            "receiver_deflection must end with receiver and target axes")
    if np.any(~np.isfinite(receiver_d)) or np.any(receiver_d < 0.0):
        raise ValueError(
            "receiver_deflection must contain finite non-negative values")
    Q = receiver_d.shape[-1]
    if receiver_d.shape[-2] < 1 or Q < 1:
        raise ValueError("receiver_deflection must be non-empty")
    k_eff = min(max(int(topk), 1), Q)
    order = np.argsort(-receiver_d, axis=-1, kind="stable")
    chosen = np.take_along_axis(receiver_d, order[..., :k_eff], axis=-1)
    mask = np.zeros_like(receiver_d, dtype=bool)
    np.put_along_axis(mask, order[..., :k_eff], chosen > 0.0, axis=-1)
    return mask


def local_ambiguity_top2_mask(
    receiver_deflection: np.ndarray,
    *,
    base_topk: int = 1,
    second_ratio: float = 0.5,
    second_min_deflection: float = 0.0,
) -> np.ndarray:
    """Add one local evidence record only for an ambiguous target ranking.

    Each receiver operates on its own row.  Starting from a bounded Top-k
    packet, it appends the next-ranked target iff

        D_(k+1) >= rho D_(k)  and  D_(k+1) >= D_min.

    Thus no receiver reads another receiver's target-quality vector.  The
    ratio is scale invariant, while the optional floor prevents two nearly
    zero statistics from triggering a packet expansion.
    """
    values = np.asarray(receiver_deflection, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("receiver_deflection must have shape (K, Q)")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            "receiver_deflection must contain finite non-negative values")
    k = int(base_topk)
    rho = float(second_ratio)
    floor = float(second_min_deflection)
    if k < 1:
        raise ValueError("base_topk must be positive")
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("second_ratio must lie in [0,1]")
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("second_min_deflection must be non-negative")
    selected = local_quality_topk_mask(values, k)
    if k >= values.shape[1]:
        return selected
    order = np.argsort(-values, axis=1, kind="stable")
    ranked = np.take_along_axis(values, order, axis=1)
    reference = ranked[:, k - 1]
    candidate = ranked[:, k]
    expand = (
        (reference > 0.0)
        & (candidate > 0.0)
        & (candidate >= rho * reference)
        & (candidate >= floor)
    )
    rows = np.flatnonzero(expand)
    selected[rows, order[rows, k]] = True
    return selected


def calibrate_llr_clip(
    receiver_deflection: np.ndarray,
    *,
    quantile: float = 0.999,
    samples: int = 1_000_000,
    seed: int = 20260723,
) -> float:
    """Calibrate a symmetric LLR clipping range on a separate trace."""
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    positive = receiver_d[
        np.isfinite(receiver_d) & (receiver_d > 0.0)]
    if positive.size == 0:
        raise ValueError("calibration trace has no positive deflection")
    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    count = max(1, int(samples))
    rng = np.random.default_rng(int(seed))
    sampled_d = positive[rng.integers(0, positive.size, size=count)]
    hypotheses = rng.integers(0, 2, size=count)
    noise = rng.standard_normal(count)
    llr = (
        (2.0 * hypotheses - 1.0) * 0.5 * sampled_d
        + np.sqrt(sampled_d) * noise
    )
    return float(np.quantile(np.abs(llr), q))


def sample_gaussian_llr(
    deflection: np.ndarray,
    hypothesis: int,
    rng: Optional[np.random.Generator] = None,
    standard_normal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample a Gaussian LLR for each supplied deflection value.

    Args:
        deflection: Non-negative local deflection values of any shape.
        hypothesis: ``0`` for H0 or ``1`` for H1.
        rng: Random generator used when ``standard_normal`` is not supplied.
        standard_normal: Optional common-random-number draw with the same
            shape as ``deflection``.  Supplying it makes paired variants use
            exactly the same stochastic evidence.

    Returns:
        LLR samples with the same shape as ``deflection``.
    """
    d = np.asarray(deflection, dtype=np.float64)
    if np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("deflection must contain finite non-negative values")
    if int(hypothesis) not in (0, 1):
        raise ValueError("hypothesis must be 0 or 1")

    if standard_normal is None:
        generator = rng if rng is not None else np.random.default_rng()
        noise = generator.standard_normal(size=d.shape)
    else:
        noise = np.asarray(standard_normal, dtype=np.float64)
        if noise.shape != d.shape:
            raise ValueError(
                "standard_normal must have the same shape as deflection")

    mean_sign = 1.0 if int(hypothesis) == 1 else -1.0
    return mean_sign * 0.5 * d + np.sqrt(d) * noise


def llr_threshold(deflection: np.ndarray, p_fa: float) -> np.ndarray:
    """Return the LLR threshold attaining ``p_fa`` for positive deflection.

    At exactly zero deflection the LLR is deterministically zero and no
    deterministic threshold can attain a non-zero false-alarm probability.
    We return ``+inf`` for that degenerate case; the environment's historical
    ``compute_PD(0) ~= P_FA`` should be understood as a randomized/no-evidence
    convention rather than an observable LLR test.
    """
    d = np.asarray(deflection, dtype=np.float64)
    if np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("deflection must contain finite non-negative values")
    if not 0.0 < float(p_fa) < 1.0:
        raise ValueError("p_fa must lie strictly between zero and one")

    threshold = -0.5 * d + np.sqrt(d) * float(
        Q_inverse(np.asarray(p_fa)))
    return np.where(d > 0.0, threshold, np.inf)


def detect_from_llr(
    llr: np.ndarray,
    deflection: np.ndarray,
    p_fa: float,
) -> np.ndarray:
    """Apply the analytically calibrated LLR threshold."""
    values = np.asarray(llr, dtype=np.float64)
    threshold = llr_threshold(deflection, p_fa)
    return values > threshold


def theoretical_detection_probability(
    deflection: np.ndarray,
    p_fa: float,
) -> np.ndarray:
    """Analytical H1 detection probability used by the existing environment."""
    return compute_PD(np.asarray(deflection, dtype=np.float64), p_fa)


def gather_owner_values(
    receiver_values: np.ndarray,
    owner: np.ndarray,
) -> np.ndarray:
    """Gather one receiver value per target for a preselected owner."""
    values = np.asarray(receiver_values)
    owners = np.asarray(owner, dtype=np.int64)
    if values.ndim < 2:
        raise ValueError(
            "receiver_values must end with receiver and target axes")
    if owners.shape != (values.shape[-1],):
        raise ValueError("owner must contain one receiver index per target")
    if np.any(owners < 0) or np.any(owners >= values.shape[-2]):
        raise ValueError("owner contains an out-of-range receiver index")
    target_indices = np.arange(values.shape[-1], dtype=np.int64)
    return values[..., owners, target_indices]
