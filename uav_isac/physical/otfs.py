"""OTFS DD domain model (simplified analytical).

Instead of full DD grid simulation, we use an analytical model that
maps (tau, nu) to:
  - DD bin indices (l, k) for delay and Doppler
  - DD unambiguous support I_support(tau, nu) (post-G2 closure, advice 001)
  - DD continuous ambiguity gain |A(tau, nu)|^2 (post-G2 closure, advice 001)
  - DD effectiveness g^DD (legacy amplitude alignment factor)

Physics contract (audit advice/001.md section 5, 2026-08-26): a target whose
delay-Doppler falls OUTSIDE the unambiguous region aliases back into the grid
and can sit almost exactly on an integer bin, so fractional mismatch alone
must NOT certify support.  The correct deflection scaling is

    D_ijq^phys = I_support * |A(tau,nu)|^2 * D_ijq^(0),

where |A| is the continuous matched-filter amplitude loss (sinc in delay and
Doppler), because deflection is an energy ratio measurable in the mean shift
(|mu1-mu0|^2 / sigma^2).  The legacy ``compute_dd_effectiveness`` returns the
amplitude |A| only and is kept for backward compatibility; callers that build
physics coefficients must use the new support * |A|^2 gain.
"""

import numpy as np


def _sinc_loss(x: float) -> float:
    """Unnormalized sinc magnitude ``|sin(pi x)/(pi x)|`` at ``x=0 -> 1``."""
    if abs(float(x)) < 1e-10:
        return 1.0
    return abs(float(np.sin(np.pi * float(x)) / (np.pi * float(x))))


def _fractional_offsets(
    tau: float,
    nu: float,
    delta_f: float,
    T_sym: float,
    M: int,
    N: int,
) -> tuple[float, float]:
    """Return fractional bin offsets (l_offset, k_offset) for (tau, nu)."""
    l_frac = tau * M * delta_f
    k_frac = nu * N * T_sym
    l_offset = l_frac - np.round(l_frac)
    k_offset = k_frac - np.round(k_frac)
    return float(l_offset), float(k_offset)


def compute_dd_unambiguous_support(
    tau: float,
    nu: float,
    delta_f: float,
    T_sym: float,
    M: int,
    N: int,
) -> float:
    """Unambiguous-support indicator I_support(tau, nu) in {0, 1}.

    A target is inside the DD unambiguous region iff its delay lies in
    ``[0, 1/delta_f)`` (l = tau*M*delta_f in [0, M)) and its (signed)
    Doppler lies in ``(-1/(2*T_sym), 1/(2*T_sym)]`` (k = nu*N*T_sym in
    (-N/2, N/2]).  Outside this region the DD grid aliases the target back
    onto an integer bin and a fractional-only check can falsely yield
    g_dd ~ 1 (audit advice/001.md, P0: "OTFS 只看 fractional bin mismatch，
    没有完整 unambiguous support").

    Args:
        tau: Delay (s), >= 0.
        nu: Doppler shift (Hz), signed.
        delta_f: Subcarrier spacing (Hz).
        T_sym: Symbol period (s).
        M: Delay bins.
        N: Doppler bins.

    Returns:
        1.0 if (tau, nu) is inside the unambiguous region, else 0.0.
    """
    if M < 1 or N < 1:
        raise ValueError("M and N must be positive integers")
    if not np.isfinite(tau) or not np.isfinite(nu):
        raise ValueError("tau and nu must be finite")
    delay_unambiguous = 0.0 <= tau < 1.0 / delta_f
    doppler_limit = 1.0 / (2.0 * T_sym)
    doppler_unambiguous = abs(nu) <= doppler_limit
    return 1.0 if (delay_unambiguous and doppler_unambiguous) else 0.0


def compute_dd_ambiguity_gain(
    tau: float,
    nu: float,
    delta_f: float,
    T_sym: float,
    M: int,
    N: int,
) -> float:
    """Continuous ambiguity magnitude squared ``|A(tau,nu)|^2`` in [0, 1].

    ``|A| = |sinc(l_offset) * sinc(k_offset)|`` under the sinc matched-filter
    model; the squared magnitude is the correct scaling of the deflection
    energy ratio (``D ~ |mu1-mu0|^2/sigma^2 ~ |A|^2 * D0``).  Unlike the old
    binary ``g_dd >= g_min`` gate, this gain degrades deflection continuously
    as the target moves off-bin (audit advice/001.md section 5).

    Args:
        tau: Delay (s).
        nu: Doppler shift (Hz), signed.
        delta_f: Subcarrier spacing (Hz).
        T_sym: Symbol period (s).
        M: Delay bins.
        N: Doppler bins.

    Returns:
        |A|^2 in [0, 1].
    """
    l_offset, k_offset = _fractional_offsets(tau, nu, delta_f, T_sym, M, N)
    amplitude = _sinc_loss(l_offset) * _sinc_loss(k_offset)
    return float(amplitude * amplitude)


def compute_dd_phys_gain(
    tau: float,
    nu: float,
    delta_f: float,
    T_sym: float,
    M: int,
    N: int,
) -> float:
    """Post-G2 physical DD gain ``I_support * |A|^2`` (advice/001 boxed D).

    This is the single quantity that should multiply the raw (support-free)
    deflection ``D_ijq^(0)`` when building physical coefficients:

        ``a_ijq(x) = I_support * |A(tau,nu)|^2 * C_ijq / (R_iq^2 R_jq^2)``.

    Returns 0.0 for out-of-support targets even if the aliased fractional
    mismatch happens to be near an integer bin.
    """
    support = compute_dd_unambiguous_support(
        tau, nu, delta_f, T_sym, M, N)
    if support <= 0.0:
        return 0.0
    return compute_dd_ambiguity_gain(tau, nu, delta_f, T_sym, M, N)


def compute_dd_phys_gain_batch(
    tau: np.ndarray,
    nu: np.ndarray,
    delta_f: float,
    T_sym: float,
    M: int,
    N: int,
) -> np.ndarray:
    """Vectorized ``I_support * |A|^2`` for broadcast-compatible inputs.

    This is numerically equivalent to :func:`compute_dd_phys_gain`, while
    avoiding Python dispatch for every Tx--Rx--target edge.  Fractional-bin
    arithmetic is evaluated only on in-support elements, matching the scalar
    function's early return and avoiding irrelevant overflow off support.
    """
    if M < 1 or N < 1:
        raise ValueError("M and N must be positive integers")

    tau_array, nu_array = np.broadcast_arrays(
        np.asarray(tau, dtype=np.float64),
        np.asarray(nu, dtype=np.float64),
    )
    if not np.all(np.isfinite(tau_array)) or not np.all(np.isfinite(nu_array)):
        raise ValueError("tau and nu must be finite")

    gain = np.zeros(tau_array.shape, dtype=np.float64)
    support = (
        (tau_array >= 0.0)
        & (tau_array < 1.0 / delta_f)
        & (np.abs(nu_array) <= 1.0 / (2.0 * T_sym))
    )
    if not np.any(support):
        return gain

    tau_supported = tau_array[support]
    nu_supported = nu_array[support]
    l_frac = tau_supported * M * delta_f
    k_frac = nu_supported * N * T_sym
    l_offset = l_frac - np.round(l_frac)
    k_offset = k_frac - np.round(k_frac)

    def sinc_magnitude(offset: np.ndarray) -> np.ndarray:
        result = np.ones(offset.shape, dtype=np.float64)
        nonzero = np.abs(offset) >= 1.0e-10
        value = offset[nonzero]
        result[nonzero] = np.abs(np.sin(np.pi * value) / (np.pi * value))
        return result

    amplitude = sinc_magnitude(l_offset) * sinc_magnitude(k_offset)
    gain[support] = amplitude * amplitude
    return gain


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
    g_min: float = 0.5,
) -> float:
    """Compute DD effectiveness g^DD.

    This function returns the continuous DD alignment amplitude in [0, 1] and
    does NOT apply a threshold.  ``g_min`` is retained only for call-signature
    compatibility.  The canonical post-G2 path uses
    :func:`compute_dd_phys_gain` (unambiguous support times amplitude squared);
    only the explicit legacy ``dd_gain_mode=binary`` caller applies
    ``1[g_dd >= g_min]``.

    This represents how effectively the bistatic observation contributes
    to detection after DD domain processing.

    Args:
        tau: Delay (s)
        nu: Doppler shift (Hz)
        delta_f: Subcarrier spacing (Hz)
        T_sym: Symbol period (s)
        M: Delay bins
        N: Doppler bins
        g_min: Legacy binary-mode threshold; ignored by this function.

    Returns:
        g^DD in [0, 1] (continuous alignment factor, not yet thresholded)

    Note (post-G2, advice/001): ``g_dd`` is the sinc AMPLITUDE ``|A|`` (it is
    sign-invariant, so it cannot certify unambiguous support).  Physical
    coefficient reconstruction must use :func:`compute_dd_phys_gain`
    (``I_support * |A|^2``); this function is retained only for legacy
    alignment-gate compatibility.
    """
    alignment = compute_dd_misalignment(tau, nu, delta_f, T_sym, M, N)
    g_dd = abs(alignment)  # in [0, 1]
    return float(g_dd)
