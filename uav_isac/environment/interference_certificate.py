"""Conservative admission certificates for co-channel U2U transmissions.

The existing :mod:`communication` transport assigns orthogonal bandwidth to
every active sender.  This module is intentionally a read-only *shadow*
diagnostic: it answers whether a proposed set of simultaneous transmitters
would satisfy every required receiver's worst-case SINR.  It does not change
the executed transport until a later, paired experiment explicitly opts in.

The certificate is cumulative rather than pairwise-only.  Pairwise-compatible
links can still fail when three or more interferers are admitted, so every
greedy insertion rechecks the complete candidate group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CertifiedReuseSchedule:
    """Deterministic fail-closed grouping result."""

    groups: tuple[tuple[int, ...], ...]
    pairwise_conflict: np.ndarray
    all_links_certified: bool
    uncertified_links: tuple[tuple[int, int], ...]
    minimum_sinr_margin_db: float
    max_concurrency: int
    reuse_factor: float


def _free_space_gain(
    sender_position: np.ndarray,
    receiver_position: np.ndarray,
    wavelength_m: float,
    antenna_gain_linear: float,
) -> float:
    distance = max(float(np.linalg.norm(
        sender_position - receiver_position)), 1.0)
    return float(
        antenna_gain_linear
        * (wavelength_m / (4.0 * np.pi * distance)) ** 2
    )


def _normalise_receivers(
    active: Sequence[int],
    receiver_sets: Mapping[int, Iterable[int]],
    num_nodes: int,
) -> dict[int, tuple[int, ...]]:
    result: dict[int, tuple[int, ...]] = {}
    for sender in active:
        receivers = tuple(sorted({int(value) for value in receiver_sets.get(
            sender, ()) if int(value) != sender}))
        if not receivers:
            raise ValueError(
                f"active sender {sender} has no required receiver")
        if any(receiver < 0 or receiver >= num_nodes for receiver in receivers):
            raise ValueError("receiver index lies outside positions")
        result[sender] = receivers
    return result


def certified_spatial_reuse_schedule(
    positions: np.ndarray,
    active_senders: Iterable[int],
    receiver_sets: Mapping[int, Iterable[int]],
    tx_powers_w: Mapping[int, float],
    *,
    carrier_hz: float,
    bandwidth_hz: float,
    kT: float,
    noise_figure_db: float,
    antenna_gain_dbi: float,
    required_sinr_db: float,
    desired_gain_margin_db: float = 0.0,
    interference_gain_margin_db: float = 0.0,
    half_duplex: bool = True,
    self_interference_cancellation_db: float = 0.0,
) -> CertifiedReuseSchedule:
    """Build certified co-channel groups using robust cumulative SINR.

    For desired link ``s -> r`` in group ``A`` the admitted lower bound is

    ``p_s g^-_sr / (N + sum_j p_j g^+_jr + I_self) >= gamma_req``.

    ``desired_gain_margin_db`` lowers every desired gain, while
    ``interference_gain_margin_db`` raises every cross-link gain.  Under
    half-duplex operation a required receiver that is transmitting in the
    same group makes that group infeasible.  Full-duplex operation is allowed
    only with an explicit residual self-interference bound.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (K,2) or (K,3)")
    if np.any(~np.isfinite(pos)):
        raise ValueError("positions must be finite")
    active = tuple(sorted({int(sender) for sender in active_senders}))
    if any(sender < 0 or sender >= pos.shape[0] for sender in active):
        raise ValueError("active sender index lies outside positions")
    if not active:
        return CertifiedReuseSchedule(
            groups=(),
            pairwise_conflict=np.zeros((0, 0), dtype=bool),
            all_links_certified=True,
            uncertified_links=(),
            minimum_sinr_margin_db=float("inf"),
            max_concurrency=0,
            reuse_factor=1.0,
        )
    receivers = _normalise_receivers(active, receiver_sets, pos.shape[0])

    scalar_values = (
        carrier_hz,
        bandwidth_hz,
        kT,
        desired_gain_margin_db,
        interference_gain_margin_db,
        self_interference_cancellation_db,
    )
    if any(not np.isfinite(float(value)) for value in scalar_values):
        raise ValueError("physical parameters must be finite")
    if carrier_hz <= 0.0 or bandwidth_hz <= 0.0 or kT <= 0.0:
        raise ValueError("carrier, bandwidth and kT must be positive")
    if desired_gain_margin_db < 0.0 or interference_gain_margin_db < 0.0:
        raise ValueError("robust gain margins must be non-negative")
    if self_interference_cancellation_db < 0.0:
        raise ValueError("self-interference cancellation must be non-negative")

    powers = {
        sender: float(tx_powers_w.get(sender, 0.0)) for sender in active
    }
    if any(not np.isfinite(value) or value < 0.0 for value in powers.values()):
        raise ValueError("transmit powers must be finite and non-negative")

    wavelength = 299_792_458.0 / float(carrier_hz)
    antenna_gain_linear = float(10.0 ** (2.0 * antenna_gain_dbi / 10.0))
    noise_figure_linear = float(10.0 ** (noise_figure_db / 10.0))
    noise_power = float(kT * bandwidth_hz * noise_figure_linear)
    required_sinr_linear = float(10.0 ** (required_sinr_db / 10.0))
    desired_scale = float(10.0 ** (-desired_gain_margin_db / 10.0))
    interference_scale = float(
        10.0 ** (interference_gain_margin_db / 10.0))
    self_residual = float(
        10.0 ** (-self_interference_cancellation_db / 10.0))

    def group_result(
        group: Sequence[int],
    ) -> tuple[bool, float, tuple[tuple[int, int], ...]]:
        group_set = set(group)
        margins: list[float] = []
        failures: list[tuple[int, int]] = []
        for sender in group:
            for receiver in receivers[sender]:
                if half_duplex and receiver in group_set:
                    failures.append((sender, receiver))
                    margins.append(float("-inf"))
                    continue
                desired = (
                    powers[sender]
                    * _free_space_gain(
                        pos[sender], pos[receiver], wavelength,
                        antenna_gain_linear)
                    * desired_scale
                )
                interference = noise_power
                for interferer in group:
                    if interferer == sender:
                        continue
                    if interferer == receiver:
                        interference += powers[interferer] * self_residual
                    else:
                        interference += (
                            powers[interferer]
                            * _free_space_gain(
                                pos[interferer], pos[receiver], wavelength,
                                antenna_gain_linear)
                            * interference_scale
                        )
                sinr = desired / max(interference, 1.0e-300)
                margin_db = float(
                    10.0 * np.log10(max(sinr, 1.0e-300))
                    - required_sinr_db)
                margins.append(margin_db)
                if sinr + 1.0e-15 < required_sinr_linear:
                    failures.append((sender, receiver))
        minimum = min(margins, default=float("inf"))
        return not failures, float(minimum), tuple(sorted(set(failures)))

    # Pairwise conflicts are useful diagnostics, but the final groups below
    # are always checked against aggregate interference.
    conflict = np.zeros((len(active), len(active)), dtype=bool)
    for left in range(len(active)):
        for right in range(left + 1, len(active)):
            feasible, _margin, _failures = group_result(
                (active[left], active[right]))
            conflict[left, right] = conflict[right, left] = not feasible

    degrees = np.sum(conflict, axis=1)
    order = tuple(active[index] for index in sorted(
        range(len(active)), key=lambda idx: (-int(degrees[idx]), active[idx])))
    groups: list[list[int]] = []
    for sender in order:
        inserted = False
        for group in groups:
            candidate = tuple(sorted((*group, sender)))
            feasible, _margin, _failures = group_result(candidate)
            if feasible:
                group[:] = candidate
                inserted = True
                break
        if not inserted:
            groups.append([sender])

    frozen_groups = tuple(tuple(sorted(group)) for group in groups)
    all_failures: list[tuple[int, int]] = []
    group_margins: list[float] = []
    for group in frozen_groups:
        _feasible, margin, failures = group_result(group)
        group_margins.append(margin)
        all_failures.extend(failures)
    uncertified = tuple(sorted(set(all_failures)))
    return CertifiedReuseSchedule(
        groups=frozen_groups,
        pairwise_conflict=conflict,
        all_links_certified=not uncertified,
        uncertified_links=uncertified,
        minimum_sinr_margin_db=float(min(
            group_margins, default=float("inf"))),
        max_concurrency=max((len(group) for group in frozen_groups), default=0),
        reuse_factor=float(len(active) / max(len(frozen_groups), 1)),
    )
