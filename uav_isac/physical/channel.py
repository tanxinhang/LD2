"""mmWave channel models: path loss, Rician fading, noise power.

Provides deterministic and stochastic channel components for
both sensing (bistatic radar) and reporting (UAV→FC) links.
"""

import numpy as np
from typing import Tuple


def compute_noise_power(kT: float, B: float, NF_dB: float) -> float:
    """Compute noise power: sigma_z^2 = kT * B * 10^(NF/10).

    Args:
        kT: Thermal noise density (W/Hz), default 4e-21 at 290K
        B: Bandwidth (Hz)
        NF_dB: Noise figure (dB)

    Returns:
        Noise power in linear scale (W)
    """
    NF_linear = 10.0 ** (NF_dB / 10.0)
    return float(kT * B * NF_linear)


def compute_path_loss_dB(distance_m: float, fc: float) -> float:
    """Compute mmWave free-space path loss (Friis equation).

    PL(dB) = 20*log10(4*pi*d/lambda)
    At 28 GHz, PL(1m) ≈ 61.4 dB, PL(100m) ≈ 101.4 dB

    Args:
        distance_m: Distance in meters
        fc: Carrier frequency (Hz)

    Returns:
        Path loss in dB (positive)
    """
    C_LIGHT = 3.0e8
    lam = C_LIGHT / fc
    eps = 1e-6
    d = max(distance_m, eps)
    return float(20.0 * np.log10(4.0 * np.pi * d / lam))


def compute_los_probability(elev_deg: float, a: float = 4.88, b: float = 0.43) -> float:
    """Al-Hourani probabilistic-LoS model for low-altitude air-ground links.

    P_LoS(theta) = 1 / (1 + a*exp(-b*(theta - a))),  theta = elevation in degrees.
    Suburban defaults a=4.88, b=0.43. At H=20 m a point 100 m away horizontally
    is only ~11 deg elevation -> low P_LoS -> blockage matters (pure LoS is optimistic).
    """
    return float(1.0 / (1.0 + a * np.exp(-b * (elev_deg - a))))


def excess_loss_dB(elev_deg: float, a: float, b: float,
                   eta_los_dB: float, eta_nlos_dB: float) -> float:
    """Expected excess loss over free space (probabilistic LoS/NLoS average)."""
    p_los = compute_los_probability(elev_deg, a, b)
    return float(p_los * eta_los_dB + (1.0 - p_los) * eta_nlos_dB)


def generate_rician_channel(
    path_loss_linear: float,
    K_dB: float,
    rng: np.random.Generator
) -> complex:
    """Generate Rician fading channel coefficient.

    h = sqrt(PL) * [ sqrt(K/(K+1)) * LoS + sqrt(1/(K+1)) * NLoS ]
    where LoS has deterministic phase 0, NLoS is complex Gaussian.

    Args:
        path_loss_linear: Path loss in linear scale (1/L, not dB)
        K_dB: Rician K-factor in dB
        rng: NumPy random generator

    Returns:
        Complex channel coefficient
    """
    K_linear = 10.0 ** (K_dB / 10.0)
    los_factor = np.sqrt(K_linear / (K_linear + 1.0))
    nlos_factor = np.sqrt(1.0 / (K_linear + 1.0))

    # LoS component (deterministic phase = 0 for simplicity)
    los = complex(los_factor, 0.0)

    # NLoS component (complex Gaussian, unit variance total)
    nlos_real = rng.normal(0.0, 1.0 / np.sqrt(2.0))
    nlos_imag = rng.normal(0.0, 1.0 / np.sqrt(2.0))
    nlos = complex(nlos_real, nlos_imag)

    h = np.sqrt(path_loss_linear) * (los + nlos)
    return h


def compute_channel_gain_squared(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    fc: float,
    K_dB: float,
    rng: np.random.Generator,
    use_los_prob: bool = False,
    los_a: float = 4.88,
    los_b: float = 0.43,
    eta_los_dB: float = 0.1,
    eta_nlos_dB: float = 21.0,
) -> float:
    """Compute |h|^2 for a communication link (e.g., reporting link).

    Free-space PL + (optional) Al-Hourani probabilistic LoS/NLoS excess loss.
    For UAV->FC at low altitude (H=20 m) blockage at low elevation is significant,
    so use_los_prob=True adds the elevation-dependent excess loss.
    """
    distance = float(np.linalg.norm(rx_pos - tx_pos))
    pl_dB = compute_path_loss_dB(distance, fc)
    if use_los_prob:
        dh = abs(float(tx_pos[2]) - float(rx_pos[2]))
        horiz = float(np.linalg.norm(rx_pos[:2] - tx_pos[:2]))
        elev_deg = float(np.degrees(np.arctan2(dh, max(horiz, 1e-6))))
        pl_dB += excess_loss_dB(elev_deg, los_a, los_b, eta_los_dB, eta_nlos_dB)
    pl_linear = 10.0 ** (-pl_dB / 10.0)
    h = generate_rician_channel(pl_linear, K_dB, rng)
    return float(np.abs(h) ** 2)


def compute_report_link_reliability(
    rx_uav_pos: np.ndarray,
    fc_position: np.ndarray,
    fc: float,
    K_dB: float,
    noise_power: float,
    P_report: float,
    rng: np.random.Generator,
    use_los_prob: bool = False,
    los_a: float = 4.88,
    los_b: float = 0.43,
    eta_los_dB: float = 0.1,
    eta_nlos_dB: float = 21.0,
) -> float:
    """Compute reporting link reliability chi_rep for rx UAV → FC.

    chi_rep = P(reliable | channel) — simplified as:
    SNR_eff / (SNR_eff + 1) bounded to [0, 1], where
    SNR_eff = min(SNR, SNR_max) with a soft reliability curve.

    This is the reliability of the reporting link carrying soft information
    from receiving UAV j to the fusion center.

    Args:
        rx_uav_pos: Position of the receiving UAV
        fc_position: Position of the fusion center
        fc: Carrier frequency
        K_dB: Rician K-factor
        noise_power: Noise power sigma_z^2
        P_report: Reporting power (W)
        rng: Random generator for fading

    Returns:
        chi_rep in [0, 1] — reporting link reliability
    """
    channel_gain = compute_channel_gain_squared(
        rx_uav_pos, fc_position, fc, K_dB, rng,
        use_los_prob=use_los_prob, los_a=los_a, los_b=los_b,
        eta_los_dB=eta_los_dB, eta_nlos_dB=eta_nlos_dB)
    snr = (P_report * channel_gain) / max(noise_power, 1e-15)

    # Sigmoid-like reliability: crosses 0.5 at SNR ~ 0 dB
    # chi_rep ≈ 1 for high SNR, ≈ 0 for low SNR
    chi_rep = snr / (snr + 1.0)
    return float(np.clip(chi_rep, 0.0, 1.0))


def compute_report_link_capacity(
    rx_uav_pos: np.ndarray,
    fc_position: np.ndarray,
    fc: float,
    K_dB: float,
    noise_power: float,
    P_report: float,
    rng: np.random.Generator
) -> float:
    """Compute reporting link capacity (bps) for rx UAV → FC.

    R = B * log2(1 + SNR)

    Args:
        rx_uav_pos: Receiving UAV position
        fc_position: Fusion center position
        fc: Carrier frequency
        K_dB: Rician K-factor
        noise_power: Noise power
        P_report: Reporting power (W)
        rng: Random generator

    Returns:
        Data rate in bits per second
    """
    channel_gain = compute_channel_gain_squared(
        rx_uav_pos, fc_position, fc, K_dB, rng)
    snr = (P_report * channel_gain) / max(noise_power, 1e-15)
    # Use a representative bandwidth for reporting link (e.g., 10 MHz)
    B_report = 10.0e6
    capacity = B_report * np.log2(1.0 + snr)
    return float(max(capacity, 0.0))


def compute_comm_sinr_db(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    all_uav_positions: np.ndarray,
    all_uav_roles: np.ndarray,
    fc: float,
    P_report: float,
    noise_power: float,
    g_tx_dBi: float = 16.0,
    g_rx_dBi: float = 16.0,
) -> float:
    """Compute SINR in dB for UAV-to-UAV communication link.

    SNR_k = P_report * G_tx * G_rx * (λ/(4π·d_k))² / N0
    SINR_k = SNR_k / (1 + Σ_{j≠k} SNR_interferer_j)

    Only UAVs with role==0 (tx) are considered interferers.

    Args:
        tx_pos: Transmitter UAV position (3,)
        rx_pos: Receiver UAV position (3,)
        all_uav_positions: All UAV positions (K, 3)
        all_uav_roles: All UAV roles (K,) int
        fc: Carrier frequency (Hz)
        P_report: Communication power (W)
        noise_power: Noise power sigma_z² (W)
        g_tx_dBi: Tx antenna gain (dBi)
        g_rx_dBi: Rx antenna gain (dBi)

    Returns:
        SINR in dB
    """
    C_LIGHT = 3.0e8
    lam = C_LIGHT / fc
    G_linear = 10.0 ** ((g_tx_dBi + g_rx_dBi) / 10.0)
    eps = 1e-6

    # Signal power from tx to rx
    d_sig = max(float(np.linalg.norm(rx_pos - tx_pos)), eps)
    path_loss_linear = (lam / (4.0 * np.pi * d_sig)) ** 2
    snr_linear = (P_report * G_linear * path_loss_linear) / max(noise_power, 1e-15)

    # Interference from all OTHER transmitting UAVs
    interf_linear = 0.0
    for j in range(all_uav_positions.shape[0]):
        pos_j = all_uav_positions[j]
        # Check if j is transmitting and not the same as tx
        if np.array_equal(pos_j, tx_pos):
            continue
        if all_uav_roles[j] != 0:  # not transmitting
            continue
        d_int = max(float(np.linalg.norm(rx_pos[:2] - pos_j[:2])), eps)
        # Use horizontal distance for interference path loss
        pl_int = (lam / (4.0 * np.pi * d_int)) ** 2
        interf_linear += (P_report * G_linear * pl_int) / max(noise_power, 1e-15)

    # SINR = Signal / (Noise + Interference)
    sinr_linear = snr_linear / (1.0 + interf_linear)
    sinr_db = float(10.0 * np.log10(max(sinr_linear, 1e-15)))
    return sinr_db
