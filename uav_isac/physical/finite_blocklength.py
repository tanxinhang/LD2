"""Finite-blocklength normal approximation for a complex AWGN link.

The implementation follows the second-order approximation

    k ~= n C(gamma) - sqrt(n V(gamma)) Q^{-1}(epsilon),

where ``n`` is the number of complex channel uses, ``k`` the information bits,
``C=log2(1+gamma)`` and
``V=(1-(1+gamma)^-2)(log2(e))^2``.  This is an analytical approximation, not
an error-correcting-code guarantee; callers must preserve that claim boundary.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr, ndtri


def awgn_capacity_bits_per_use(snr_linear: float) -> float:
    snr = float(snr_linear)
    if not np.isfinite(snr) or snr < 0.0:
        raise ValueError("snr_linear must be finite and non-negative")
    return float(np.log2(1.0 + snr))


def awgn_dispersion_bits2_per_use(snr_linear: float) -> float:
    snr = float(snr_linear)
    if not np.isfinite(snr) or snr < 0.0:
        raise ValueError("snr_linear must be finite and non-negative")
    return float(
        (1.0 - (1.0 + snr) ** -2.0) * np.log2(np.e) ** 2.0)


def normal_approximation_packet_error_probability(
    snr_linear: float, blocklength: int, information_bits: int,
) -> float:
    """Approximate BLER for ``k`` bits in ``n`` complex channel uses."""
    n = int(blocklength)
    k = int(information_bits)
    if n < 1:
        raise ValueError("blocklength must be positive")
    if k < 0:
        raise ValueError("information_bits must be non-negative")
    if k == 0:
        return 0.0
    capacity = awgn_capacity_bits_per_use(snr_linear)
    dispersion = awgn_dispersion_bits2_per_use(snr_linear)
    if capacity <= 0.0 or dispersion <= 0.0:
        return 1.0
    z_score = (
        n * capacity - k
    ) / math.sqrt(n * dispersion)
    return float(np.clip(ndtr(-z_score), 0.0, 1.0))


def achievable_information_bits(
    snr_linear: float, blocklength: int, target_bler: float,
) -> float:
    """Second-order achievable-bit approximation at a target BLER."""
    n = int(blocklength)
    epsilon = float(target_bler)
    if n < 1:
        raise ValueError("blocklength must be positive")
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("target_bler must lie strictly between 0 and 0.5")
    capacity = awgn_capacity_bits_per_use(snr_linear)
    dispersion = awgn_dispersion_bits2_per_use(snr_linear)
    q_inverse = float(-ndtri(epsilon))
    return float(max(
        0.0, n * capacity - math.sqrt(n * dispersion) * q_inverse))


def minimum_blocklength_normal_approximation(
    snr_linear: float,
    information_bits: int,
    target_bler: float,
    *,
    max_blocklength: int = 100_000_000,
) -> int | None:
    """Return the smallest integer ``n`` meeting the normal approximation."""
    bits = int(information_bits)
    maximum = int(max_blocklength)
    if bits < 0:
        raise ValueError("information_bits must be non-negative")
    if maximum < 1:
        raise ValueError("max_blocklength must be positive")
    if bits == 0:
        return 1
    # Validate common inputs once before the monotone doubling search.
    achievable_information_bits(snr_linear, 1, target_bler)
    if float(snr_linear) <= 0.0:
        return None
    # The achievable-bit formula is ``n*C - sqrt(n*V)*Qinv``.  For a fixed SNR
    # the capacity C, dispersion V and inverse-Q term Qinv are constants, so
    # precompute them once and evaluate only the n-dependent part inside the
    # doubling + binary search (avoids re-running log2 / ndtri every step).
    capacity = awgn_capacity_bits_per_use(snr_linear)
    dispersion = awgn_dispersion_bits2_per_use(snr_linear)
    q_inverse = float(-ndtri(float(target_bler)))

    def achievable_bits(n: int) -> float:
        return max(
            0.0,
            float(n) * capacity - math.sqrt(float(n) * dispersion) * q_inverse)

    high = 1
    while high < maximum and achievable_bits(high) + 1.0e-12 < bits:
        high = min(maximum, 2 * high)
    if achievable_bits(high) + 1.0e-12 < bits:
        return None
    low = 1
    while low < high:
        middle = (low + high) // 2
        if achievable_bits(middle) + 1.0e-12 >= bits:
            high = middle
        else:
            low = middle + 1
    return int(low)


def minimum_snr_normal_approximation(
    blocklength: int,
    information_bits: int,
    target_bler: float,
    *,
    max_snr_linear: float = 1.0e12,
) -> float | None:
    """Invert the normal approximation for the required linear SNR."""
    n = int(blocklength)
    bits = int(information_bits)
    maximum = float(max_snr_linear)
    if n < 1 or bits < 0:
        raise ValueError("invalid blocklength or information_bits")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_snr_linear must be finite and positive")
    achievable_information_bits(0.0, n, target_bler)
    if bits == 0:
        return 0.0
    high = 1.0
    while high < maximum and achievable_information_bits(
            high, n, target_bler) + 1.0e-12 < bits:
        high = min(maximum, 2.0 * high)
    if achievable_information_bits(
            high, n, target_bler) + 1.0e-12 < bits:
        return None
    low = 0.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if achievable_information_bits(
                middle, n, target_bler) + 1.0e-12 >= bits:
            high = middle
        else:
            low = middle
    return float(high)
