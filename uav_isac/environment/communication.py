"""Cost-aware emergent inter-UAV communication.

The policy is free to learn the semantic content of a 16-dimensional message.
The environment only assigns physical meaning to the *transport*: a learned
rate action selects silence or a quantization precision, which determines the
number of transmitted bits, link delay, and radio energy.  Messages that miss
the SNR/deadline constraints are not exposed to the receiving actor.

The first implementation uses one-hop broadcast and orthogonal bandwidth
sharing between active senders.  This keeps the transport model explicit and
reproducible without prescribing what the learned message must encode.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class DeliveredMessage:
    sender: int
    receiver: int
    message: np.ndarray
    rate_index: int
    snr_db: float
    latency_s: float
    delay_frames: int
    tx_power_w: float
    token_mask: Optional[np.ndarray] = None


@dataclass
class CommunicationStepStats:
    total_bits: float = 0.0
    total_energy_j: float = 0.0
    active_senders: int = 0
    attempted_links: int = 0
    delivered_links: int = 0
    expired_links: int = 0
    mean_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    max_latency_s: float = 0.0
    per_sender_energy_j: Dict[int, float] = field(default_factory=dict)
    per_sender_bits: Dict[int, float] = field(default_factory=dict)
    per_sender_power_w: Dict[int, float] = field(default_factory=dict)
    per_sender_attempted_links: Dict[int, int] = field(default_factory=dict)
    per_sender_delivered_links: Dict[int, int] = field(default_factory=dict)
    per_sender_expired_links: Dict[int, int] = field(default_factory=dict)

    @property
    def delivery_rate(self) -> float:
        if self.attempted_links <= 0:
            return 1.0
        return float(self.delivered_links / self.attempted_links)

    @property
    def deadline_violation_rate(self) -> float:
        if self.attempted_links <= 0:
            return 0.0
        return float(self.expired_links / self.attempted_links)

    def sender_delivery_rates(self, num_senders: int) -> np.ndarray:
        """Return sender-specific delivery fractions.

        Silent senders use the neutral value one; callers must combine this
        with the actual rate action before applying a failure penalty.
        """
        result = np.ones(max(0, int(num_senders)), dtype=np.float64)
        for sender, attempted in self.per_sender_attempted_links.items():
            if 0 <= int(sender) < result.size and int(attempted) > 0:
                result[int(sender)] = float(
                    self.per_sender_delivered_links.get(sender, 0)
                    / int(attempted))
        return result

    def as_dict(self) -> Dict[str, float]:
        return {
            'learned_comm_bits': float(self.total_bits),
            'learned_comm_energy_j': float(self.total_energy_j),
            'learned_comm_active_senders': float(self.active_senders),
            'learned_comm_attempted_links': float(self.attempted_links),
            'learned_comm_delivered_links': float(self.delivered_links),
            'learned_comm_delivery_rate': self.delivery_rate,
            'learned_comm_deadline_violation_rate': self.deadline_violation_rate,
            'learned_comm_mean_latency_s': float(self.mean_latency_s),
            'learned_comm_p95_latency_s': float(self.p95_latency_s),
            'learned_comm_max_latency_s': float(self.max_latency_s),
            'learned_comm_mean_tx_power_w': float(np.mean(
                list(self.per_sender_power_w.values()) or [0.0])),
            'learned_comm_total_tx_power_w': float(sum(
                self.per_sender_power_w.values())),
        }


class InterUAVCommunicationModel:
    """One-hop broadcast transport for learned UAV messages."""

    def __init__(
        self,
        rate_bits_per_dim: List[int],
        header_bits: int,
        bandwidth_hz: float,
        deadline_s: float,
        processing_delay_s: float,
        snr_threshold_db: float,
        antenna_gain_dbi: float,
        carrier_hz: float,
        tx_power_w: float,
        kT: float,
        noise_figure_db: float,
        dt: float,
        message_dim: int = 16,
    ):
        if not rate_bits_per_dim or int(rate_bits_per_dim[0]) != 0:
            raise ValueError('comm_rate_bits_per_dim must start with 0 (silence)')
        if any(int(b) < 0 for b in rate_bits_per_dim):
            raise ValueError('communication quantization bits must be non-negative')
        self.rate_bits_per_dim = tuple(int(b) for b in rate_bits_per_dim)
        self.header_bits = int(max(0, header_bits))
        self.bandwidth_hz = float(max(bandwidth_hz, 1.0))
        self.deadline_s = float(max(deadline_s, 0.0))
        self.processing_delay_s = float(max(processing_delay_s, 0.0))
        self.snr_threshold_db = float(snr_threshold_db)
        self.antenna_gain_linear = float(10.0 ** (2.0 * antenna_gain_dbi / 10.0))
        self.carrier_hz = float(carrier_hz)
        self.wavelength = 299_792_458.0 / max(self.carrier_hz, 1.0)
        self.tx_power_w = float(max(tx_power_w, 0.0))
        self.kT = float(max(kT, 1e-30))
        self.noise_figure_linear = float(10.0 ** (noise_figure_db / 10.0))
        self.dt = float(max(dt, 1e-9))
        self.message_dim = int(message_dim)

    def payload_bits(
        self, rate_index: int, active_dimensions: Optional[int] = None,
    ) -> int:
        idx = int(np.clip(rate_index, 0, len(self.rate_bits_per_dim) - 1))
        bits_per_dim = self.rate_bits_per_dim[idx]
        if bits_per_dim <= 0:
            return 0
        # ``active_dimensions`` may include an appended in-band control stream
        # (for example the QPD protocol header), so it is not capped at the
        # learned latent message width.
        dims = (
            self.message_dim
            if active_dimensions is None
            else max(0, int(active_dimensions))
        )
        if dims <= 0:
            return 0
        return self.header_bits + dims * bits_per_dim

    def _active_dimensions(
        self, token_mask: Optional[np.ndarray],
    ) -> int:
        if token_mask is None:
            return self.message_dim
        mask = np.asarray(token_mask, dtype=np.float64).reshape(-1)
        if mask.size <= 0 or self.message_dim % mask.size != 0:
            raise ValueError(
                'token mask length must be a positive divisor of message_dim')
        per_token = self.message_dim // mask.size
        return int(np.sum(mask > 0.5) * per_token)

    def quantize(self, message: np.ndarray, rate_index: int) -> np.ndarray:
        """Uniformly quantize a learned message to the selected precision."""
        msg = np.asarray(message, dtype=np.float64).reshape(-1)
        if msg.size != self.message_dim:
            raise ValueError(
                f'expected {self.message_dim}-D communication message, got {msg.size}')
        return self.quantize_values(msg, rate_index)

    def quantize_values(
        self, values: np.ndarray, rate_index: int,
    ) -> np.ndarray:
        """Quantize an arbitrary appended control stream at the packet rate."""
        idx = int(np.clip(rate_index, 0, len(self.rate_bits_per_dim) - 1))
        bits = self.rate_bits_per_dim[idx]
        msg = np.asarray(values, dtype=np.float64)
        if bits <= 0:
            return np.zeros_like(msg)
        clipped = np.clip(msg, -1.0, 1.0)
        levels = max(2, (1 << bits))
        code = np.rint((clipped + 1.0) * 0.5 * (levels - 1))
        return (2.0 * code / (levels - 1) - 1.0).astype(np.float64)

    def _link(self, sender_pos: np.ndarray, receiver_pos: np.ndarray,
              payload_bits: int, effective_bandwidth_hz: float,
              tx_power_w: float = None):
        distance = float(np.linalg.norm(sender_pos - receiver_pos))
        distance = max(distance, 1.0)
        path_gain = (self.wavelength / (4.0 * np.pi * distance)) ** 2
        power_w = self.tx_power_w if tx_power_w is None else max(
            float(tx_power_w), 0.0)
        received_power = power_w * self.antenna_gain_linear * path_gain
        noise_power = self.kT * effective_bandwidth_hz * self.noise_figure_linear
        snr = received_power / max(noise_power, 1e-30)
        snr_db = 10.0 * np.log10(max(snr, 1e-30))
        rate_bps = effective_bandwidth_hz * np.log2(1.0 + max(snr, 0.0))
        serialization_s = payload_bits / max(rate_bps, 1e-12)
        latency_s = serialization_s + self.processing_delay_s
        return snr_db, rate_bps, serialization_s, latency_s

    def transmit(
        self,
        messages: Dict[int, np.ndarray],
        rate_indices: Dict[int, int],
        positions: np.ndarray,
        tx_powers_w: Dict[int, float] = None,
        token_masks: Dict[int, np.ndarray] = None,
        extra_payload_dimensions: Dict[int, int] = None,
    ) -> tuple[List[DeliveredMessage], CommunicationStepStats]:
        """Transport one learned broadcast per active sender.

        Active senders share the configured bandwidth orthogonally.  A
        broadcast is charged once in bits and radio energy, while delivery is
        evaluated independently for every receiving UAV.
        """
        K = int(positions.shape[0])
        masks = token_masks or {}
        extra_dims = extra_payload_dimensions or {}
        active = []
        for k in range(K):
            active_dims = (
                self._active_dimensions(masks.get(k))
                + max(0, int(extra_dims.get(k, 0)))
            )
            if (k in messages and self.payload_bits(
                    rate_indices.get(k, 0), active_dims) > 0):
                active.append(k)
        stats = CommunicationStepStats(active_senders=len(active))
        if not active or K <= 1:
            return [], stats

        effective_bw = self.bandwidth_hz / len(active)
        deliveries: List[DeliveredMessage] = []
        all_latencies: List[float] = []

        for sender in active:
            rate_idx = int(rate_indices.get(sender, 0))
            sender_mask = masks.get(sender)
            active_dims = (
                self._active_dimensions(sender_mask)
                + max(0, int(extra_dims.get(sender, 0)))
            )
            n_bits = self.payload_bits(rate_idx, active_dims)
            quantized = self.quantize(messages[sender], rate_idx)
            if sender_mask is not None:
                mask = np.asarray(sender_mask, dtype=np.float64).reshape(-1)
                per_token = self.message_dim // mask.size
                quantized = quantized.reshape(mask.size, per_token)
                quantized[mask <= 0.5] = 0.0
                quantized = quantized.reshape(-1)
            stats.total_bits += n_bits
            stats.per_sender_bits[sender] = float(n_bits)
            sender_power_w = (
                self.tx_power_w if tx_powers_w is None
                else max(float(tx_powers_w.get(sender, self.tx_power_w)), 0.0)
            )
            stats.per_sender_power_w[sender] = float(sender_power_w)
            stats.per_sender_attempted_links[sender] = 0
            stats.per_sender_delivered_links[sender] = 0
            stats.per_sender_expired_links[sender] = 0

            sender_airtime = 0.0
            for receiver in range(K):
                if receiver == sender:
                    continue
                snr_db, _rate, serialization_s, latency_s = self._link(
                    positions[sender], positions[receiver], n_bits,
                    effective_bw, sender_power_w)
                stats.attempted_links += 1
                stats.per_sender_attempted_links[sender] += 1
                all_latencies.append(float(latency_s))
                # A packet that misses its deadline is aborted rather than
                # occupying the radio for an unbounded Shannon serialization
                # time. Keep raw latency for diagnostics, but bill at most one
                # configured deadline of RF airtime per broadcast.
                billed_airtime = min(
                    float(serialization_s), self.deadline_s)
                sender_airtime = max(sender_airtime, billed_airtime)

                meets_snr = snr_db >= self.snr_threshold_db
                meets_deadline = latency_s <= self.deadline_s
                if meets_snr and meets_deadline:
                    delay_frames = max(1, int(np.ceil(latency_s / self.dt)))
                    deliveries.append(DeliveredMessage(
                        sender=sender,
                        receiver=receiver,
                        message=quantized.copy(),
                        rate_index=rate_idx,
                        snr_db=float(snr_db),
                        latency_s=float(latency_s),
                        delay_frames=delay_frames,
                        tx_power_w=float(sender_power_w),
                        token_mask=(None if sender_mask is None else
                                    np.asarray(sender_mask, dtype=np.float64).copy()),
                    ))
                    stats.delivered_links += 1
                    stats.per_sender_delivered_links[sender] += 1
                else:
                    stats.expired_links += 1
                    stats.per_sender_expired_links[sender] += 1

            sender_energy = sender_power_w * sender_airtime
            stats.per_sender_energy_j[sender] = float(sender_energy)
            stats.total_energy_j += float(sender_energy)

        if all_latencies:
            arr = np.asarray(all_latencies, dtype=np.float64)
            stats.mean_latency_s = float(np.mean(arr))
            stats.p95_latency_s = float(np.percentile(arr, 95))
            stats.max_latency_s = float(np.max(arr))

        return deliveries, stats
