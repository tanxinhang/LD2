"""OTFS DD domain model (simplified analytical).

Instead of full DD grid simulation, we use an analytical model that
maps (tau, nu) to:
  - DD bin indices (l, k) for delay and Doppler
  - DD effectiveness g^DD: how well the signal aligns with the DD grid

This simplified model captures the key effect: targets whose delay-Doppler
fall between grid bins suffer degraded detection performance.
"""

import numpy as np


def compute_dd_bins(
    tau: float,
    nu: float,
    delta_f: float,
    T_sym: float,
    M: int,
    N: int
) -> tuple:
    """Compute delay-Doppler bin indices for given (tau, nu).

    Delay bin: l = round(tau * M * delta_f)
    Doppler bin: k = round(nu * N * T_sym)

    Args:
        tau: Propagation delay (s)
        nu: Doppler shift (Hz)
        delta_f: Subcarrier spacing (Hz)
        T_sym: Symbol period (s)
        M: Number of delay bins
        N: Number of Doppler bins

    Returns:
        (l, k) bin indices (clipped to valid range)
    """
    l = int(np.round(tau * M * delta_f))
    k = int(np.round(nu * N * T_sym))
    l = max(0, min(l, M - 1))
    k = max(-N // 2, min(k, N // 2 - 1))
    return l, k


def compute_dd_misalignment(
    tau: float,
    nu: float,
    delta_f: float,
    T_sym: float,
    M: int,
    N: int
) -> float:
    """Compute DD grid misalignment factor.

    Measures how far (tau, nu) is from the nearest DD grid point.
    Returns a value in [0, 1] where 1 = perfect alignment.

    The misalignment is modeled as the product of sinc-like losses
    in delay and Doppler dimensions:
      misalignment = sinc(delay_offset) * sinc(doppler_offset)

    where the offsets are the fractional bin distances.
    """
    # Fractional bin positions
    l_frac = tau * M * delta_f
    k_frac = nu * N * T_sym

    # Distance to nearest integer bin
    l_offset = l_frac - np.round(l_frac)
    k_offset = k_frac - np.round(k_frac)

    # sinc-based misalignment: sinc(x) = sin(pi*x)/(pi*x)
    def sinc(x):
        if abs(x) < 1e-10:
            return 1.0
        return float(np.sin(np.pi * x) / (np.pi * x))

    return float(sinc(l_offset) * sinc(k_offset))


def compute_dd_effectiveness(
    tau: float,
    nu: float,
    delta_f: float,
    T_sym: float,
    M: int,
    N: int,
    g_min: float = 0.5
) -> float:
    """Compute DD effectiveness g^DD.

    g^DD = misalignment_factor, thresholded by g_min.

    This represents how effectively the bistatic observation contributes
    to detection after DD domain processing.

    Args:
        tau: Delay (s)
        nu: Doppler shift (Hz)
        delta_f: Subcarrier spacing (Hz)
        T_sym: Symbol period (s)
        M: Delay bins
        N: Doppler bins
        g_min: Effectiveness threshold

    Returns:
        g^DD in [0, 1]
    """
    alignment = compute_dd_misalignment(tau, nu, delta_f, T_sym, M, N)
    g_dd = abs(alignment)  # in [0, 1]
    return float(g_dd)


def compute_otfs_snr(
    d_raw: float,
    g_dd: float,
    g_min: float = 0.5
) -> float:
    """Compute effective SNR after OTFS DD processing.

    SNR_eff = d_raw * g_dd if g_dd >= g_min else 0

    This represents the post-processing SNR that feeds into the
    detection statistic.

    Args:
        d_raw: Raw Deflection (pre-processing SNR)
        g_dd: DD effectiveness [0, 1]
        g_min: Threshold below which the observation is discarded

    Returns:
        Effective SNR for detection
    """
    if g_dd < g_min:
        return 0.0
    return float(d_raw * g_dd)
