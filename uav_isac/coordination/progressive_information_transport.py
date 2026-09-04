"""Certified progressive transport for distributed information decisions.

Each refinement layer narrows a valid interval for one candidate's decision
utility.  Transmission stops as soon as one candidate's lower bound dominates
all competing upper bounds.  More bits therefore improve *decision
certifiability*, not the underlying physical sensing information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class InformationRefinementLayer:
    lower_gain: float
    upper_gain: float
    incremental_bits: int
    label: str = ""


@dataclass(frozen=True)
class ProgressiveCandidate:
    candidate_id: int
    layers: tuple[InformationRefinementLayer, ...]


@dataclass(frozen=True)
class ProgressiveTransportResult:
    chosen_candidate_id: int
    certified: bool
    used_incumbent_fallback: bool
    selected_layers: tuple[int, ...]
    transmitted_bits: int
    full_transport_bits: int
    latency_s: float
    rf_energy_j: float
    refinement_count: int
    packet_count: int
    full_transmission_fraction: float
    best_verified_lower_gain: float
    maximum_competing_upper_gain: float


def quantized_psd_logdet_interval(
    gram: np.ndarray,
    *,
    bits_per_entry: int,
) -> tuple[float, float]:
    """Bound ``log det(I+G)`` from a uniformly quantized PSD Gram matrix.

    The encoder sends one shared scale and the upper-triangular entries of the
    symmetric quantized matrix.  If the scalar quantization step is ``h``, the
    entrywise error is at most ``h/2`` and hence the spectral error is at most
    ``n*h/2``.  Weyl's inequality then gives valid lower/upper eigenvalue and
    log-det bounds without using the exact log-det label.
    """
    matrix = np.asarray(gram, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or np.any(~np.isfinite(matrix))):
        raise ValueError("gram must be a finite square matrix")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(matrix), initial=0.0)), 1.0e-12)
    if float(np.min(eigenvalues, initial=np.inf)) < -1.0e-10 * scale:
        raise ValueError("gram must be positive semidefinite")
    bits = int(bits_per_entry)
    if bits < 2 or bits > 52 or bits != bits_per_entry:
        raise ValueError("bits_per_entry must be an integer in [2,52]")
    positive_levels = float((1 << (bits - 1)) - 1)
    step = scale / positive_levels
    quantized = np.round(matrix / step) * step
    quantized = 0.5 * (quantized + quantized.T)
    spectral_error = matrix.shape[0] * step / 2.0
    quantized_eigenvalues = np.linalg.eigvalsh(quantized)
    lower_eigenvalues = np.maximum(
        quantized_eigenvalues - spectral_error, 0.0)
    upper_eigenvalues = np.maximum(
        quantized_eigenvalues + spectral_error, 0.0)
    return (
        float(np.sum(np.log1p(lower_eigenvalues))),
        float(np.sum(np.log1p(upper_eigenvalues))),
    )


def _validate_candidates(
    candidates: Sequence[ProgressiveCandidate],
) -> tuple[ProgressiveCandidate, ...]:
    items = tuple(candidates)
    if not items:
        raise ValueError("at least one progressive candidate is required")
    identifiers: set[int] = set()
    for candidate in items:
        identifier = int(candidate.candidate_id)
        if identifier in identifiers:
            raise ValueError("candidate ids must be unique")
        identifiers.add(identifier)
        if not candidate.layers:
            raise ValueError("every candidate requires at least one layer")
        previous_lower = -np.inf
        previous_upper = np.inf
        for index, layer in enumerate(candidate.layers):
            lower = float(layer.lower_gain)
            upper = float(layer.upper_gain)
            if (not np.isfinite(lower) or not np.isfinite(upper)
                    or lower < 0.0 or lower > upper + 1.0e-12):
                raise ValueError("refinement gains must form a finite interval")
            if (lower + 1.0e-12 < previous_lower
                    or upper > previous_upper + 1.0e-12):
                raise ValueError("refinement intervals must be nested")
            bits = int(layer.incremental_bits)
            if bits != layer.incremental_bits or bits < 1:
                raise ValueError("every refinement layer must add positive bits")
            previous_lower = lower
            previous_upper = upper
    return items


def certified_progressive_transport(
    candidates: Sequence[ProgressiveCandidate],
    *,
    incumbent_candidate_id: int,
    link_rate_bps: float,
    deadline_s: float,
    transmit_power_w: float,
    bit_budget: int | None = None,
    processing_delay_s: float = 0.0,
    refinements_per_packet: int = 1,
    certificate_tolerance: float = 1.0e-12,
    execute_best_lower_on_budget_exhaustion: bool = False,
) -> ProgressiveTransportResult:
    """Transmit only refinements that can still resolve the best candidate.

    All base layers are packed into one initial message.  Later refinements are
    individually requested and charged.  Among feasible refinements, the next
    layer with the largest certified interval-width reduction per bit is sent.
    If the deadline or bit budget expires before a dominance certificate is
    obtained, the default fail-closed mode returns the supplied feasible
    incumbent.  The optional anytime mode executes the candidate with the
    largest received conservative lower bound, provided every base layer was
    received.  It remains explicitly uncertified.
    """
    items = _validate_candidates(candidates)
    id_to_position = {
        int(candidate.candidate_id): position
        for position, candidate in enumerate(items)
    }
    incumbent = int(incumbent_candidate_id)
    if incumbent not in id_to_position:
        raise ValueError("incumbent_candidate_id is absent")
    rate = float(link_rate_bps)
    deadline = float(deadline_s)
    power = float(transmit_power_w)
    processing = float(processing_delay_s)
    if (not np.isfinite(rate) or rate <= 0.0
            or not np.isfinite(deadline) or deadline < 0.0
            or not np.isfinite(power) or power < 0.0
            or not np.isfinite(processing) or processing < 0.0):
        raise ValueError("transport resources are invalid")
    packet_capacity = int(refinements_per_packet)
    if packet_capacity < 1 or packet_capacity != refinements_per_packet:
        raise ValueError("refinements_per_packet must be a positive integer")
    full_bits = int(sum(
        layer.incremental_bits
        for candidate in items for layer in candidate.layers
    ))
    budget = full_bits if bit_budget is None else int(bit_budget)
    if budget < 0 or (bit_budget is not None and budget != bit_budget):
        raise ValueError("bit_budget must be a non-negative integer")

    selected = np.zeros(len(items), dtype=np.int64)
    base_bits = int(sum(candidate.layers[0].incremental_bits for candidate in items))
    transmitted = 0
    latency = 0.0
    energy = 0.0
    packet_count = 0
    if base_bits <= budget:
        base_airtime = base_bits / rate
        if base_airtime + processing <= deadline + 1.0e-15:
            transmitted = base_bits
            latency = base_airtime + processing
            energy = power * base_airtime
            packet_count = 1
        else:
            selected.fill(-1)
    else:
        selected.fill(-1)

    def interval(position: int) -> tuple[float, float]:
        layer_index = int(selected[position])
        if layer_index < 0:
            return 0.0, float("inf")
        layer = items[position].layers[layer_index]
        return float(layer.lower_gain), float(layer.upper_gain)

    def certificate() -> tuple[bool, int, float, float]:
        lowers = np.asarray([interval(i)[0] for i in range(len(items))])
        order = sorted(
            range(len(items)),
            key=lambda i: (-lowers[i], int(items[i].candidate_id)),
        )
        best = int(order[0])
        competing = max(
            (interval(i)[1] for i in range(len(items)) if i != best),
            default=-np.inf,
        )
        proven = bool(
            selected[best] >= 0
            and np.isfinite(competing)
            and competing <= lowers[best] + certificate_tolerance
        )
        if len(items) == 1 and selected[best] >= 0:
            proven = True
        return proven, best, float(lowers[best]), float(competing)

    refinement_count = 0
    while True:
        proven, best, best_lower, competing_upper = certificate()
        if proven:
            break
        options: list[tuple[float, int, int, int]] = []
        for position, candidate in enumerate(items):
            current_index = int(selected[position])
            next_index = current_index + 1
            if current_index < 0 or next_index >= len(candidate.layers):
                continue
            current_lower, current_upper = interval(position)
            # A candidate already dominated by the current best cannot change
            # the decision and receives no additional bits.
            if position != best and current_upper < best_lower - certificate_tolerance:
                continue
            next_layer = candidate.layers[next_index]
            next_width = float(next_layer.upper_gain - next_layer.lower_gain)
            current_width = float(current_upper - current_lower)
            reduction = max(current_width - next_width, 0.0)
            bits = int(next_layer.incremental_bits)
            if transmitted + bits > budget:
                continue
            score = reduction / bits
            options.append((
                score, -int(candidate.candidate_id), position,
                bits,
            ))
        if not options:
            break
        packet_options = sorted(options, reverse=True)
        packet: list[tuple[int, int]] = []
        packet_bits = 0
        for _score, _tie, position, bits in packet_options:
            if len(packet) >= packet_capacity:
                break
            proposed_bits = packet_bits + bits
            proposed_latency = proposed_bits / rate + processing
            if (transmitted + proposed_bits <= budget
                    and latency + proposed_latency
                    <= deadline + 1.0e-15):
                packet.append((position, bits))
                packet_bits = proposed_bits
        if not packet:
            break
        for position, _bits in packet:
            selected[position] += 1
        transmitted += packet_bits
        latency += packet_bits / rate + processing
        energy += power * (packet_bits / rate)
        refinement_count += len(packet)
        packet_count += 1

    proven, best, best_lower, competing_upper = certificate()
    base_received = bool(np.all(selected >= 0))
    execute_best = bool(
        proven
        or (execute_best_lower_on_budget_exhaustion and base_received))
    chosen_position = best if execute_best else id_to_position[incumbent]
    full_fraction = float(np.mean([
        selected[position] == len(candidate.layers) - 1
        for position, candidate in enumerate(items)
    ]))
    return ProgressiveTransportResult(
        chosen_candidate_id=int(items[chosen_position].candidate_id),
        certified=bool(proven),
        used_incumbent_fallback=bool(not proven and not execute_best),
        selected_layers=tuple(int(value) for value in selected),
        transmitted_bits=int(transmitted),
        full_transport_bits=full_bits,
        latency_s=float(latency),
        rf_energy_j=float(energy),
        refinement_count=int(refinement_count),
        packet_count=int(packet_count),
        full_transmission_fraction=full_fraction,
        best_verified_lower_gain=best_lower,
        maximum_competing_upper_gain=competing_upper,
    )
