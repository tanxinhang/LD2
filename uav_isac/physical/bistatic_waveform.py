"""Auditable bridge from bistatic geometry to OTFS waveform parameters.

The geometry layer remains the single source of truth for ``tau``, ``nu`` and
the bistatic radar-equation path amplitude.  This module only converts those
quantities into OTFS delay/Doppler bins and applies explicitly supplied sensing
power and antenna gains.

An edge component ``(i,j,q)`` is simulator bookkeeping, not receiver-visible
evidence.  A receiver observes the superposition over all scheduled
transmitters and targets; :func:`aggregate_edge_components_at_receivers`
enforces that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from uav_isac.physical.geometry import compute_all_bistatic_params
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform


@dataclass(frozen=True)
class BistaticWaveformParameters:
    """Physical parameters for every ``(tx, rx, target)`` edge."""

    delay_s: np.ndarray
    doppler_hz: np.ndarray
    path_amplitude: np.ndarray
    received_target_amplitude: np.ndarray
    delay_bin: np.ndarray
    doppler_bin: np.ndarray
    valid_edge_mask: np.ndarray
    unambiguous_edge_mask: np.ndarray


def real_equivalent_complex_noise_variance(noise_power: float) -> float:
    """Convert the frozen real-detector noise power to ``E|n_c|^2``.

    The waveform generator samples ``n_c ~ CN(0, sigma_c^2)`` and reports the
    coherent statistic ``sqrt(2)*Re<s,y>``.  Its H0 variance is therefore
    ``sigma_c^2`` while its squared mean shift is ``2*E_s``.  Setting
    ``sigma_c^2=2*P_noise`` makes its Deflection ``E_s/P_noise``, exactly the
    repository's frozen ``real_gaussian_shift, c_det=1`` convention.
    """
    value = float(noise_power)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("noise_power must be finite and positive")
    return 2.0 * value


def ideal_coherent_h0_deflection(
    received_target_amplitude: np.ndarray | float,
    *,
    waveform: MinimalOTFSWaveform,
    complex_noise_variance: float,
    n_cpi: int = 1,
) -> np.ndarray:
    """Return matched coherent H0-Deflection under the ideal cyclic model."""
    waveform.validate()
    amplitude = np.asarray(received_target_amplitude, dtype=np.float64)
    variance = float(complex_noise_variance)
    looks = int(n_cpi)
    if (
        np.any(~np.isfinite(amplitude))
        or np.any(amplitude < 0.0)
        or not np.isfinite(variance)
        or variance <= 0.0
        or looks < 1
        or float(n_cpi) != float(looks)
    ):
        raise ValueError("amplitude/noise/n_cpi lie outside the physical domain")
    return (
        2.0 * float(waveform.sample_count) * float(looks)
        * np.square(amplitude) / variance
    )


def _target_rcs_vector(rcs_m2: float | np.ndarray, targets: int) -> np.ndarray:
    raw = np.asarray(rcs_m2, dtype=np.float64)
    if raw.ndim == 0:
        result = np.full(int(targets), float(raw), dtype=np.float64)
    elif raw.shape == (int(targets),):
        result = raw.copy()
    else:
        raise ValueError("rcs_m2 must be a scalar or a length-Q vector")
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("rcs_m2 must be finite and non-negative")
    return result


def bistatic_geometry_to_waveform(
    uav_positions: np.ndarray,
    uav_velocities: np.ndarray,
    target_positions: np.ndarray,
    target_velocities: np.ndarray,
    roles: np.ndarray,
    sensing_power_w: np.ndarray,
    *,
    carrier_hz: float,
    waveform: MinimalOTFSWaveform,
    rcs_m2: float | np.ndarray = 1.0,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0,
    role_agnostic: bool = False,
    require_unambiguous: bool = False,
) -> BistaticWaveformParameters:
    """Map physical bistatic paths to one ideal rectangular OTFS grid.

    With ``B=M*Delta_f`` and ``T=1/Delta_f``, the continuous-to-grid map is

    ``delay_bin = tau*B`` and ``doppler_bin = nu*N*T``.

    The received target-path amplitude is

    ``sqrt(P_iq * G_tx * G_rx) * |alpha_ijq|``.

    Thus received power follows the bistatic radar equation and is linear in
    sensing power.  Noise power remains a separate waveform configuration
    quantity; this function does not silently normalize physical amplitudes.
    """
    waveform.validate()
    xyz = np.asarray(uav_positions, dtype=np.float64)
    velocity = np.asarray(uav_velocities, dtype=np.float64)
    targets_xyz = np.asarray(target_positions, dtype=np.float64)
    targets_velocity = np.asarray(target_velocities, dtype=np.float64)
    role = np.asarray(roles, dtype=np.int64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        raise ValueError("uav_positions must have shape (K,3)")
    K = xyz.shape[0]
    if velocity.shape != (K, 3) or role.shape != (K,):
        raise ValueError("UAV velocities/roles do not match uav_positions")
    if targets_xyz.ndim != 2 or targets_xyz.shape[1:] != (3,):
        raise ValueError("target_positions must have shape (Q,3)")
    Q = targets_xyz.shape[0]
    if targets_velocity.shape != (Q, 3):
        raise ValueError("target_velocities must have shape (Q,3)")
    power = np.asarray(sensing_power_w, dtype=np.float64)
    if power.shape != (K, Q):
        raise ValueError("sensing_power_w must have shape (K,Q)")
    if np.any(~np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("sensing_power_w must be finite and non-negative")
    scalars = np.asarray(
        [carrier_hz, tx_gain_dbi, rx_gain_dbi], dtype=np.float64)
    if np.any(~np.isfinite(scalars)) or float(carrier_hz) <= 0.0:
        raise ValueError("carrier and antenna gains must be finite")

    rcs = _target_rcs_vector(rcs_m2, Q)
    delay = np.empty((K, K, Q), dtype=np.float64)
    doppler = np.empty_like(delay)
    path_amplitude = np.empty_like(delay)
    # Reuse the established geometry implementation.  Its public API accepts
    # scalar RCS, so target-specific RCS is applied one target at a time.
    for q in range(Q):
        tau_q, nu_q, alpha_q = compute_all_bistatic_params(
            xyz,
            velocity,
            targets_xyz[q:q + 1],
            targets_velocity[q:q + 1],
            role,
            float(carrier_hz),
            rcs=float(rcs[q]),
            role_agnostic=bool(role_agnostic),
        )
        delay[:, :, q] = tau_q[:, :, 0]
        doppler[:, :, q] = nu_q[:, :, 0]
        path_amplitude[:, :, q] = alpha_q[:, :, 0]

    valid = np.isfinite(delay) & (path_amplitude > 0.0)
    bandwidth_hz = float(waveform.delay_bins) * float(waveform.delta_f_hz)
    symbol_period_s = 1.0 / float(waveform.delta_f_hz)
    delay_bin = delay * bandwidth_hz
    doppler_bin = (
        doppler * float(waveform.doppler_bins) * symbol_period_s)
    unambiguous = (
        valid
        & (delay_bin >= 0.0)
        & (delay_bin < float(waveform.delay_bins))
        & (doppler_bin >= -0.5 * float(waveform.doppler_bins))
        & (doppler_bin < 0.5 * float(waveform.doppler_bins))
    )
    if require_unambiguous and np.any(valid & ~unambiguous):
        raise ValueError("a valid bistatic path aliases on the OTFS grid")

    antenna_power_gain = 10.0 ** (
        (float(tx_gain_dbi) + float(rx_gain_dbi)) / 10.0)
    amplitude_scale = np.sqrt(
        power[:, None, :] * antenna_power_gain)
    received_amplitude = path_amplitude * amplitude_scale
    received_amplitude = np.where(valid, received_amplitude, 0.0)
    return BistaticWaveformParameters(
        delay_s=delay,
        doppler_hz=doppler,
        path_amplitude=path_amplitude,
        received_target_amplitude=received_amplitude,
        delay_bin=delay_bin,
        doppler_bin=doppler_bin,
        valid_edge_mask=valid,
        unambiguous_edge_mask=unambiguous,
    )


def selected_edge_mask(
    shape: tuple[int, int, int],
    selected_edges: Iterable[tuple[int, int, int]],
) -> np.ndarray:
    """Return a validated mask for scheduled physical sensing edges."""
    K_tx, K_rx, Q = (int(value) for value in shape)
    if K_tx < 1 or K_rx != K_tx or Q < 1:
        raise ValueError("shape must be (K,K,Q) with positive K and Q")
    mask = np.zeros((K_tx, K_rx, Q), dtype=bool)
    for i_raw, j_raw, q_raw in selected_edges:
        i, j, q = int(i_raw), int(j_raw), int(q_raw)
        if not (0 <= i < K_tx and 0 <= j < K_rx and 0 <= q < Q and i != j):
            raise ValueError("selected_edges contains an invalid physical edge")
        mask[i, j, q] = True
    return mask


def aggregate_edge_components_at_receivers(
    edge_components: np.ndarray,
    scheduled_edge_mask: np.ndarray,
) -> np.ndarray:
    """Form receiver observations by summing scheduled edge components.

    ``edge_components`` begins with axes ``(tx, rx, target)`` and may contain
    arbitrary trailing sample/grid axes.  Unscheduled components are rejected
    rather than silently exposed, and the result has receiver as its first
    axis.  Detectors must consume this aggregate, never individual edge terms.
    """
    components = np.asarray(edge_components)
    mask = np.asarray(scheduled_edge_mask, dtype=bool)
    if components.ndim < 3 or mask.ndim != 3:
        raise ValueError("components/mask must begin with (K,K,Q)")
    if components.shape[:3] != mask.shape or mask.shape[0] != mask.shape[1]:
        raise ValueError("components and scheduled mask shapes do not agree")
    broadcast_mask = mask[(...,) + (None,) * (components.ndim - 3)]
    if np.any(np.abs(np.where(broadcast_mask, 0.0, components)) > 1.0e-12):
        raise ValueError("unscheduled physical edge contains non-zero signal")
    return np.sum(components, axis=(0, 2))
