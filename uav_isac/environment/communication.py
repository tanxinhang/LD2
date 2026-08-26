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

from uav_isac.physical.finite_blocklength import (
    minimum_blocklength_normal_approximation,
    normal_approximation_packet_error_probability,
)


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
    packet_error_probability: float = 0.0


@dataclass
class CommunicationStepStats:
    total_bits: float = 0.0
    total_energy_j: float = 0.0
    active_senders: int = 0
    attempted_links: int = 0
    delivered_links: int = 0
    expired_links: int = 0
    deadline_failed_links: int = 0
    snr_failed_links: int = 0
    reliability_failed_links: int = 0
    burst_failed_links: int = 0
    burst_bad_links: int = 0
    mean_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    max_latency_s: float = 0.0
    mean_packet_error_probability: float = 0.0
    max_packet_error_probability: float = 0.0
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
        return float(self.deadline_failed_links / self.attempted_links)

    @property
    def snr_violation_rate(self) -> float:
        if self.attempted_links <= 0:
            return 0.0
        return float(self.snr_failed_links / self.attempted_links)

    @property
    def delivery_failure_rate(self) -> float:
        if self.attempted_links <= 0:
            return 0.0
        return float(self.expired_links / self.attempted_links)

    @property
    def reliability_violation_rate(self) -> float:
        if self.attempted_links <= 0:
            return 0.0
        return float(self.reliability_failed_links / self.attempted_links)

    @property
    def burst_failure_rate(self) -> float:
        if self.attempted_links <= 0:
            return 0.0
        return float(self.burst_failed_links / self.attempted_links)

    @property
    def burst_bad_link_rate(self) -> float:
        if self.attempted_links <= 0:
            return 0.0
        return float(self.burst_bad_links / self.attempted_links)

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
            'learned_comm_snr_violation_rate': self.snr_violation_rate,
            'learned_comm_delivery_failure_rate': self.delivery_failure_rate,
            'learned_comm_reliability_violation_rate': (
                self.reliability_violation_rate),
            'learned_comm_burst_failure_rate': self.burst_failure_rate,
            'learned_comm_burst_bad_link_rate': self.burst_bad_link_rate,
            'learned_comm_mean_latency_s': float(self.mean_latency_s),
            'learned_comm_p95_latency_s': float(self.p95_latency_s),
            'learned_comm_max_latency_s': float(self.max_latency_s),
            'learned_comm_mean_packet_error_probability': float(
                self.mean_packet_error_probability),
            'learned_comm_max_packet_error_probability': float(
                self.max_packet_error_probability),
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
        finite_blocklength_enabled: bool = False,
        finite_blocklength_target_bler: float = 1.0e-5,
        finite_blocklength_max_channel_uses: int = 100_000_000,
        finite_blocklength_sample_errors: bool = True,
        finite_blocklength_coding_snr_margin_db: float = 0.0,
        snr_shadowing_std_db: float = 0.0,
        snr_shadowing_correlation: float = 0.0,
        burst_loss_enabled: bool = False,
        burst_good_to_bad_probability: float = 0.0,
        burst_bad_to_good_probability: float = 1.0,
        burst_good_drop_probability: float = 0.0,
        burst_bad_drop_probability: float = 1.0,
        rng: np.random.Generator | None = None,
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
        self.finite_blocklength_enabled = bool(finite_blocklength_enabled)
        self.finite_blocklength_target_bler = float(
            finite_blocklength_target_bler)
        self.finite_blocklength_max_channel_uses = int(
            finite_blocklength_max_channel_uses)
        self.finite_blocklength_sample_errors = bool(
            finite_blocklength_sample_errors)
        self.finite_blocklength_coding_snr_margin_db = max(
            float(finite_blocklength_coding_snr_margin_db), 0.0)
        self.snr_shadowing_std_db = max(float(snr_shadowing_std_db), 0.0)
        self.snr_shadowing_correlation = float(np.clip(
            snr_shadowing_correlation, 0.0, 0.999999))
        self.burst_loss_enabled = bool(burst_loss_enabled)
        self.burst_good_to_bad_probability = float(
            burst_good_to_bad_probability)
        self.burst_bad_to_good_probability = float(
            burst_bad_to_good_probability)
        self.burst_good_drop_probability = float(
            burst_good_drop_probability)
        self.burst_bad_drop_probability = float(
            burst_bad_drop_probability)
        probabilities = (
            self.burst_good_to_bad_probability,
            self.burst_bad_to_good_probability,
            self.burst_good_drop_probability,
            self.burst_bad_drop_probability,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError('burst transition/drop probabilities must lie in [0,1]')
        self._burst_bad_state = np.zeros((0, 0), dtype=bool)
        self._snr_shadowing_db = np.zeros((0, 0), dtype=np.float64)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        if not 0.0 < self.finite_blocklength_target_bler < 0.5:
            raise ValueError(
                'finite_blocklength_target_bler must lie in (0,0.5)')
        if self.finite_blocklength_max_channel_uses < 1:
            raise ValueError(
                'finite_blocklength_max_channel_uses must be positive')

    def reset_channel_state(self, num_uavs: int) -> None:
        """Reset episode-local Markov erasures and correlated SNR shadowing."""
        K = max(0, int(num_uavs))
        self._burst_bad_state = np.zeros((K, K), dtype=bool)
        self._snr_shadowing_db = np.zeros((K, K), dtype=np.float64)

    def get_channel_state(self) -> Dict[str, np.ndarray]:
        return {
            'burst_bad_state': self._burst_bad_state.copy(),
            'snr_shadowing_db': self._snr_shadowing_db.copy(),
        }

    def set_channel_state(self, state: Dict[str, np.ndarray]) -> None:
        burst = np.asarray(state.get(
            'burst_bad_state', np.zeros((0, 0))), dtype=bool)
        shadowing = np.asarray(state.get(
            'snr_shadowing_db', np.zeros_like(burst, dtype=np.float64)),
            dtype=np.float64)
        if burst.ndim != 2 or burst.shape[0] != burst.shape[1]:
            raise ValueError('burst channel state must be square')
        if shadowing.shape != burst.shape or np.any(~np.isfinite(shadowing)):
            raise ValueError('SNR shadowing state shape/value mismatch')
        self._burst_bad_state = burst.copy()
        self._snr_shadowing_db = shadowing.copy()

    def _advance_channel_state(self, num_uavs: int) -> None:
        K = max(0, int(num_uavs))
        if self._burst_bad_state.shape != (K, K):
            self.reset_channel_state(K)
        off_diagonal = ~np.eye(K, dtype=bool)
        if self.burst_loss_enabled and K > 1:
            draws = self.rng.random((K, K))
            good_to_bad = (
                ~self._burst_bad_state
                & (draws < self.burst_good_to_bad_probability))
            bad_to_good = (
                self._burst_bad_state
                & (draws < self.burst_bad_to_good_probability))
            self._burst_bad_state[good_to_bad & off_diagonal] = True
            self._burst_bad_state[bad_to_good & off_diagonal] = False
            self._burst_bad_state[~off_diagonal] = False
        if self.snr_shadowing_std_db > 0.0 and K > 1:
            rho = self.snr_shadowing_correlation
            innovation = (
                np.sqrt(max(1.0 - rho * rho, 0.0))
                * self.snr_shadowing_std_db
                * self.rng.normal(size=(K, K)))
            self._snr_shadowing_db = rho * self._snr_shadowing_db + innovation
            self._snr_shadowing_db[~off_diagonal] = 0.0
        elif self.snr_shadowing_std_db <= 0.0:
            self._snr_shadowing_db.fill(0.0)

    def packet_error_probability(
        self,
        snr_db: float,
        serialization_s: float,
        payload_bits: int,
        effective_bandwidth_hz: float,
    ) -> float:
        """Return the common learned/evidence-packet BLER approximation."""
        if not self.finite_blocklength_enabled or int(payload_bits) <= 0:
            return 0.0
        if not np.isfinite(serialization_s) or serialization_s < 0.0:
            return 1.0
        blocklength = max(1, int(np.ceil(
            float(serialization_s) * float(effective_bandwidth_hz))))
        return normal_approximation_packet_error_probability(
            10.0 ** (float(snr_db) / 10.0),
            blocklength,
            int(payload_bits),
        )

    def packet_reliability_success(self, error_probability: float) -> bool:
        """Apply the BLER gate and, when enabled, a reproducible erasure."""
        error = float(error_probability)
        if not np.isfinite(error) or not 0.0 <= error <= 1.0:
            raise ValueError('error_probability must lie in [0,1]')
        if not self.finite_blocklength_enabled:
            return True
        if error > self.finite_blocklength_target_bler * (1.0 + 1.0e-9):
            return False
        if not self.finite_blocklength_sample_errors:
            return True
        return bool(self.rng.random() >= error)

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
        return self.quantize_values_at_bits(
            values, self.rate_bits_per_dim[idx])

    @staticmethod
    def quantize_values_at_bits(
        values: np.ndarray,
        bits_per_dimension: int,
    ) -> np.ndarray:
        """Uniformly quantize a separately coded stream at explicit precision."""
        bits = max(0, int(bits_per_dimension))
        msg = np.asarray(values, dtype=np.float64)
        if bits <= 0:
            return np.zeros_like(msg)
        clipped = np.clip(msg, -1.0, 1.0)
        levels = max(2, (1 << bits))
        code = np.rint((clipped + 1.0) * 0.5 * (levels - 1))
        return (2.0 * code / (levels - 1) - 1.0).astype(np.float64)

    def _serialization_budget(
        self, snr: float, payload_bits: int, effective_bandwidth_hz: float,
    ) -> tuple[float, float]:
        bandwidth = float(effective_bandwidth_hz)
        bits = int(payload_bits)
        shannon_rate_bps = bandwidth * np.log2(1.0 + max(snr, 0.0))
        if bits <= 0:
            return float(shannon_rate_bps), 0.0
        if not self.finite_blocklength_enabled:
            return (
                float(shannon_rate_bps),
                float(bits / max(shannon_rate_bps, 1.0e-12)),
            )
        design_snr = (
            max(float(snr), 0.0)
            / 10.0 ** (self.finite_blocklength_coding_snr_margin_db / 10.0))
        blocklength = minimum_blocklength_normal_approximation(
            design_snr, bits,
            self.finite_blocklength_target_bler,
            max_blocklength=self.finite_blocklength_max_channel_uses)
        if blocklength is None:
            return 0.0, float('inf')
        serialization_s = float(blocklength / bandwidth)
        return float(bits / serialization_s), serialization_s

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
        rate_bps, serialization_s = self._serialization_budget(
            snr, payload_bits, effective_bandwidth_hz)
        latency_s = serialization_s + self.processing_delay_s
        return snr_db, rate_bps, serialization_s, latency_s

    def link_budget(
        self,
        sender_pos: np.ndarray,
        receiver_pos: np.ndarray,
        payload_bits: int,
        effective_bandwidth_hz: float,
        tx_power_w: float = None,
    ) -> tuple[float, float, float, float]:
        """Return the physical one-hop budget used by packet transport.

        Coordination certificates use this public, read-only interface so
        their SNR, Shannon rate and latency calculations cannot silently
        diverge from :meth:`transmit`.
        """
        if int(payload_bits) < 0:
            raise ValueError("payload_bits must be non-negative")
        if float(effective_bandwidth_hz) <= 0.0:
            raise ValueError("effective_bandwidth_hz must be positive")
        return self._link(
            np.asarray(sender_pos, dtype=np.float64),
            np.asarray(receiver_pos, dtype=np.float64),
            int(payload_bits),
            float(effective_bandwidth_hz),
            tx_power_w,
        )

    def robust_link_budget(
        self,
        sender_pos: np.ndarray,
        receiver_pos: np.ndarray,
        payload_bits: int,
        effective_bandwidth_hz: float,
        tx_power_w: float = None,
        *,
        snr_margin_db: float = 0.0,
        latency_margin_s: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Return a lower-SNR, upper-latency one-hop budget.

        ``snr_margin_db`` is applied before Shannon rate is evaluated, rather
        than being attached to a nominal latency after the fact.  The separate
        ``latency_margin_s`` must therefore bound only *excess* queue,
        scheduling or processing error after observed-SNR serialization has
        been removed.  This avoids charging the same channel fade twice.

        Infinite non-negative margins are allowed deliberately: an unresolved
        conformal quantile then produces zero rate/infinite latency and forces
        the enclosing protocol certificate to fail closed.
        """
        snr_margin = float(snr_margin_db)
        latency_margin = float(latency_margin_s)
        if np.isnan(snr_margin) or snr_margin < 0.0:
            raise ValueError("snr_margin_db must be non-negative and not NaN")
        if np.isnan(latency_margin) or latency_margin < 0.0:
            raise ValueError(
                "latency_margin_s must be non-negative and not NaN")
        snr_db, _rate, _serialization, _latency = self.link_budget(
            sender_pos,
            receiver_pos,
            payload_bits,
            effective_bandwidth_hz,
            tx_power_w,
        )
        robust_snr_db = float(snr_db - snr_margin)
        if np.isneginf(robust_snr_db):
            robust_snr = 0.0
        else:
            robust_snr = float(10.0 ** (robust_snr_db / 10.0))
        bandwidth = float(effective_bandwidth_hz)
        bits = int(payload_bits)
        rate_bps, serialization_s = self._serialization_budget(
            robust_snr, bits, bandwidth)
        latency_s = (
            float(serialization_s) + self.processing_delay_s + latency_margin
        )
        return (
            robust_snr_db,
            float(rate_bps),
            float(serialization_s),
            float(latency_s),
        )

    def transmit(
        self,
        messages: Dict[int, np.ndarray],
        rate_indices: Dict[int, int],
        positions: np.ndarray,
        tx_powers_w: Dict[int, float] = None,
        token_masks: Dict[int, np.ndarray] = None,
        extra_payload_dimensions: Dict[int, int] = None,
        extra_payload_bits: Dict[int, int] = None,
        base_payload_dimensions: Dict[int, int] = None,
        suppress_message_payload: Dict[int, bool] = None,
    ) -> tuple[List[DeliveredMessage], CommunicationStepStats]:
        """Transport one learned broadcast per active sender.

        Active senders share the configured bandwidth orthogonally.  A
        broadcast is charged once in bits and radio energy, while delivery is
        evaluated independently for every receiving UAV.
        """
        K = int(positions.shape[0])
        # Channel memory evolves once per simulator-frame transport call, not
        # once per receiver, so burst duration is independent of fleet size.
        self._advance_channel_state(K)
        masks = token_masks or {}
        extra_dims = extra_payload_dimensions or {}
        exact_extra_bits = extra_payload_bits or {}
        base_dims = base_payload_dimensions or {}
        suppress_payload = suppress_message_payload or {}

        def packet_bits(sender: int) -> int:
            rate_index = int(rate_indices.get(sender, 0))
            idx = int(np.clip(
                rate_index, 0, len(self.rate_bits_per_dim) - 1))
            bits_per_dim = self.rate_bits_per_dim[idx]
            learned_dimensions = (
                max(0, int(base_dims[sender]))
                if sender in base_dims else
                self._active_dimensions(masks.get(sender))
            ) + max(0, int(extra_dims.get(sender, 0)))
            appended_bits = max(
                0, int(exact_extra_bits.get(sender, 0)))
            payload = learned_dimensions * bits_per_dim + appended_bits
            return self.header_bits + payload if payload > 0 else 0

        active = []
        for k in range(K):
            if k in messages and packet_bits(k) > 0:
                active.append(k)
        stats = CommunicationStepStats(active_senders=len(active))
        if not active or K <= 1:
            return [], stats

        effective_bw = self.bandwidth_hz / len(active)
        deliveries: List[DeliveredMessage] = []
        all_latencies: List[float] = []
        all_blers: List[float] = []

        for sender in active:
            rate_idx = int(rate_indices.get(sender, 0))
            sender_mask = masks.get(sender)
            n_bits = packet_bits(sender)
            quantized = (
                np.zeros_like(np.asarray(messages[sender], dtype=np.float64))
                if bool(suppress_payload.get(sender, False))
                else self.quantize(messages[sender], rate_idx)
            )
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
                snr_db = float(
                    snr_db + self._snr_shadowing_db[sender, receiver])
                stats.attempted_links += 1
                stats.per_sender_attempted_links[sender] += 1
                all_latencies.append(float(latency_s))
                packet_error_probability = self.packet_error_probability(
                    snr_db, serialization_s, n_bits, effective_bw)
                all_blers.append(float(packet_error_probability))
                # A packet that misses its deadline is aborted rather than
                # occupying the radio for an unbounded Shannon serialization
                # time. Keep raw latency for diagnostics, but bill at most one
                # configured deadline of RF airtime per broadcast.
                billed_airtime = min(
                    float(serialization_s), self.deadline_s)
                sender_airtime = max(sender_airtime, billed_airtime)

                meets_snr = snr_db >= self.snr_threshold_db
                meets_deadline = latency_s <= self.deadline_s
                # Match evidence-packet semantics: a link that already fails
                # SNR or deadline does not consume a codeword-erasure RNG draw.
                # This keeps common-seed protocol comparisons call-aligned.
                meets_reliability = (
                    self.packet_reliability_success(
                        packet_error_probability)
                    if meets_snr and meets_deadline else True)
                burst_bad = bool(
                    self.burst_loss_enabled
                    and self._burst_bad_state[sender, receiver])
                stats.burst_bad_links += int(burst_bad)
                meets_burst = True
                if (
                    self.burst_loss_enabled
                    and meets_snr and meets_deadline and meets_reliability
                ):
                    drop_probability = (
                        self.burst_bad_drop_probability
                        if burst_bad else self.burst_good_drop_probability)
                    meets_burst = bool(
                        self.rng.random() >= drop_probability)
                    if not meets_burst:
                        stats.burst_failed_links += 1
                if (meets_snr and meets_deadline
                        and meets_reliability and meets_burst):
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
                        packet_error_probability=float(
                            packet_error_probability),
                    ))
                    stats.delivered_links += 1
                    stats.per_sender_delivered_links[sender] += 1
                else:
                    stats.expired_links += 1
                    stats.per_sender_expired_links[sender] += 1
                    stats.deadline_failed_links += int(not meets_deadline)
                    stats.snr_failed_links += int(not meets_snr)
                    if not meets_reliability:
                        stats.reliability_failed_links += 1

            sender_energy = sender_power_w * sender_airtime
            stats.per_sender_energy_j[sender] = float(sender_energy)
            stats.total_energy_j += float(sender_energy)

        if all_latencies:
            arr = np.asarray(all_latencies, dtype=np.float64)
            stats.mean_latency_s = float(np.mean(arr))
            stats.p95_latency_s = float(np.percentile(arr, 95))
            stats.max_latency_s = float(np.max(arr))
        if all_blers:
            bler = np.asarray(all_blers, dtype=np.float64)
            stats.mean_packet_error_probability = float(np.mean(bler))
            stats.max_packet_error_probability = float(np.max(bler))

        return deliveries, stats
