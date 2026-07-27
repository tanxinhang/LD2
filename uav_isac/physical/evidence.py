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

    @staticmethod
    def _index_bits(cardinality: int) -> int:
        return max(1, int(np.ceil(np.log2(max(int(cardinality), 2)))))

    @property
    def source_bits(self) -> int:
        return self._index_bits(self.num_agents)

    @property
    def target_bits(self) -> int:
        return self._index_bits(self.num_targets)

    def broadcast_bits(self, num_entries: int, llr_bits: int) -> int:
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
        return shared + entries * per_entry


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
    useful_unique_entries: int
    mean_latency_s: float
    p95_latency_s: float
    max_latency_s: float

    @property
    def delivery_rate(self) -> float:
        return float(
            self.delivered_links / max(self.attempted_links, 1))

    @property
    def deadline_violation_rate(self) -> float:
        return float(
            self.expired_links / max(self.attempted_links, 1))

    @property
    def evidence_utilization(self) -> float:
        transmitted = int(np.sum(self.selected_mask))
        return float(self.useful_unique_entries / max(transmitted, 1))

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
            "evidence_comm_mean_latency_s": float(self.mean_latency_s),
            "evidence_comm_p95_latency_s": float(self.p95_latency_s),
            "evidence_comm_max_latency_s": float(self.max_latency_s),
            "evidence_transmitted_entries": float(
                np.sum(self.selected_mask)),
            "evidence_useful_unique_entries": float(
                self.useful_unique_entries),
            "evidence_utilization": self.evidence_utilization,
            "evidence_owner": self.owner.copy(),
            "evidence_selected_mask": self.selected_mask.copy(),
            "evidence_delivered_peer_mask": (
                self.delivered_peer_mask.copy()),
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
    owner_aware: bool = True,
) -> EvidenceTransportResult:
    """Select local evidence and transport structured broadcasts.

    Target selection is based only on each receiver's pre-observation local
    quality. One fusion owner per target is selected before random LLR values
    are generated. Broadcast delivery is evaluated for every peer, but an
    entry enters the detector only when it reaches that target's owner.
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

    owner = pre_evidence_fusion_owner(receiver_d)
    owner_mask = np.zeros((K, Q), dtype=bool)
    owner_mask[owner, np.arange(Q)] = True
    selection_quality = receiver_d.copy()
    if owner_aware:
        selection_quality[owner_mask] = 0.0
    selected = local_quality_topk_mask(selection_quality, topk)

    payload_bits = np.asarray([
        layout.broadcast_bits(int(np.sum(selected[source])), llr_bits)
        for source in range(K)
    ], dtype=np.int64)
    active = np.flatnonzero(payload_bits > 0)
    delivery = np.zeros((K, K), dtype=bool)
    latencies = []
    attempted = delivered = expired = 0
    energy_j = 0.0
    energy_by_sender = np.zeros(K, dtype=np.float64)
    packets = []
    if active.size > 0:
        effective_bandwidth = link_model.bandwidth_hz / active.size
        for source in active:
            targets = tuple(
                int(q) for q in np.flatnonzero(selected[source]))
            packets.append(EvidencePacket(
                source_rx=int(source),
                observation_frame=int(observation_frame),
                target_ids=targets,
                llr_bits=int(llr_bits),
                confidence_bits=int(layout.confidence_bits),
                payload_bits=int(payload_bits[source]),
            ))
            sender_airtime = 0.0
            for receiver in range(K):
                if receiver == source:
                    continue
                snr_db, _rate, serialization_s, latency_s = link_model._link(
                    xyz[source],
                    xyz[receiver],
                    int(payload_bits[source]),
                    effective_bandwidth,
                    float(max(power[source], 0.0)),
                )
                attempted += 1
                latencies.append(float(latency_s))
                # Failed evidence packets are abandoned at the deadline.
                # Preserve raw latency diagnostics while bounding actual RF
                # occupancy and energy by the attempted transmission window.
                billed_airtime = min(
                    float(serialization_s), link_model.deadline_s)
                sender_airtime = max(sender_airtime, billed_airtime)
                success = bool(
                    snr_db >= link_model.snr_threshold_db
                    and latency_s <= link_model.deadline_s
                )
                delivery[source, receiver] = success
                if success:
                    delivered += 1
                else:
                    expired += 1
            sender_energy = (
                float(max(power[source], 0.0)) * sender_airtime)
            energy_by_sender[source] = sender_energy
            energy_j += sender_energy

    source_index = np.arange(K)[:, None]
    owner_index = owner[None, :]
    delivered_to_owner = delivery[source_index, owner_index]
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

    latency_array = np.asarray(latencies, dtype=np.float64)
    return EvidenceTransportResult(
        packets=tuple(packets),
        owner=owner,
        selected_mask=selected,
        owner_mask=owner_mask,
        delivered_peer_mask=delivered_peer,
        delivery_matrix=delivery,
        payload_bits_by_sender=payload_bits,
        energy_j_by_sender=energy_by_sender,
        total_bits=float(np.sum(payload_bits)),
        total_energy_j=float(energy_j),
        active_senders=int(active.size),
        attempted_links=int(attempted),
        delivered_links=int(delivered),
        expired_links=int(expired),
        useful_unique_entries=int(useful),
        mean_latency_s=(
            float(np.mean(latency_array)) if latency_array.size else 0.0),
        p95_latency_s=(
            float(np.percentile(latency_array, 95))
            if latency_array.size else 0.0),
        max_latency_s=(
            float(np.max(latency_array)) if latency_array.size else 0.0),
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
    local_h0 = (
        -0.5 * receiver_d[None]
        + np.sqrt(receiver_d[None]) * noise_h0
    )
    local_h1 = (
        +0.5 * receiver_d[None]
        + np.sqrt(receiver_d[None]) * noise_h1
    )
    quantized_h0 = quantize_llr(local_h0, llr_bits, clip_max)
    quantized_h1 = quantize_llr(local_h1, llr_bits, clip_max)

    normalized_content = str(content_mode).strip().lower()
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
    elif normalized_content != "normal":
        raise ValueError(
            "content_mode must be normal, zero, or value_roll")

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

    if confidence_quantizer is None:
        peer_d_hat = receiver_d
    else:
        peer_d_hat = confidence_quantizer.quantize(receiver_d)
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

    transmitted = np.broadcast_to(peer_mask[None], local_h0.shape)
    raw_values = np.concatenate([
        local_h0[transmitted],
        local_h1[transmitted],
    ])
    quantized_values = np.concatenate([
        quantized_h0[transmitted],
        quantized_h1[transmitted],
    ])
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


def pre_evidence_fusion_owner(
    receiver_deflection: np.ndarray,
) -> np.ndarray:
    """Choose one airborne fusion owner per target before random evidence.

    ``receiver_deflection`` has shape ``(K, Q)``.  Selection depends only on
    predicted physical quality, never on realized frame-level LLR values.
    Ties are resolved deterministically by the lowest receiver index.
    """
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    if receiver_d.ndim != 2:
        raise ValueError("receiver_deflection must have shape (K, Q)")
    if receiver_d.shape[0] == 0:
        raise ValueError("at least one receiver is required")
    if np.any(~np.isfinite(receiver_d)) or np.any(receiver_d < 0.0):
        raise ValueError(
            "receiver_deflection must contain finite non-negative values")
    return np.argmax(receiver_d, axis=0).astype(np.int64)


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
