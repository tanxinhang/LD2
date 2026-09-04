"""Bistatic geometry computations.

Computes bistatic ranges, delays, Doppler shifts, and path gains
for all (tx_i, rx_j, target_q) triples given UAV and target positions.
"""

import numpy as np
from typing import Tuple

# Speed of light
C_LIGHT = 3.0e8  # m/s


def compute_bistatic_range(
    tx_pos: np.ndarray,    # (3,) tx UAV position
    rx_pos: np.ndarray,    # (3,) rx UAV position
    target_pos: np.ndarray # (3,) target position
) -> float:
    """Compute bistatic range: R_tx,target + R_target,rx.

    Args:
        tx_pos: Transmitter UAV position [x, y, z]
        rx_pos: Receiver UAV position [x, y, z]
        target_pos: Target position [x, y, z]

    Returns:
        Bistatic range in meters
    """
    r_tx_target = np.linalg.norm(tx_pos - target_pos)
    r_target_rx = np.linalg.norm(target_pos - rx_pos)
    return float(r_tx_target + r_target_rx)


def compute_delay(bistatic_range: float) -> float:
    """Compute propagation delay: tau = R_bistatic / c.

    Args:
        bistatic_range: Bistatic range in meters

    Returns:
        Delay in seconds
    """
    return bistatic_range / C_LIGHT


def compute_doppler(
    tx_pos: np.ndarray, tx_vel: np.ndarray,
    rx_pos: np.ndarray, rx_vel: np.ndarray,
    target_pos: np.ndarray, target_vel: np.ndarray,
    fc: float
) -> float:
    """Compute bistatic Doppler shift.

    With total bistatic range ``R = R_iq + R_jq`` (i = tx, j = rx, q = target):

        u_iq = (x_q - x_i)/R_iq     (tx -> target)
        u_qj = (x_j - x_q)/R_jq     (target -> rx)
        dot_R = (v_q - v_i)^T u_iq + (v_j - v_q)^T u_qj

    Using the convention "total propagation distance shrinking = positive
    Doppler" (audit advice/001.md section 4, 2026-08-26):

        nu = -(f_c/c)*dot_R
           = (f_c/c) * [ v_tx^T u_iq - v_rx^T u_qj + v_target^T (u_qj - u_iq) ].

    Physics contract locked by tests: with Tx/target stationary, an Rx flying
    TOWARD the target yields positive Doppler (and away yields negative).

    Args:
        tx_pos, tx_vel: Transmitter UAV position and velocity
        rx_pos, rx_vel: Receiver UAV position and velocity
        target_pos, target_vel: Target position and velocity
        fc: Carrier frequency (Hz)

    Returns:
        Doppler shift in Hz (signed, this convention)
    """
    eps = 1e-10

    # Unit vectors
    u_tx_target = target_pos - tx_pos
    u_tx_target = u_tx_target / (np.linalg.norm(u_tx_target) + eps)

    u_target_rx = rx_pos - target_pos
    u_target_rx = u_target_rx / (np.linalg.norm(u_target_rx) + eps)

    # Audit 2026-08-26 (P0, advice/001 section 4): the receiver term previously
    # used ``+v_rx^T u_qj``; the derivative of the total bistatic range carries
    # ``(v_j - v_q)^T u_qj``, so with the "shrinking range = positive Doppler"
    # convention the receiver contribution is ``-v_rx^T u_qj``.  The old sign
    # was latent while only |nu| was consumed, but becomes P0 once Phase C
    # uses signed Doppler for direction control.
    doppler_tx = np.dot(tx_vel, u_tx_target)      # + v_i^T u_iq
    doppler_target = np.dot(target_vel, u_target_rx - u_tx_target)  # v_q^T(u_qj - u_iq)
    doppler_rx = -np.dot(rx_vel, u_target_rx)     # - v_j^T u_qj  (audit fix)

    # Total Doppler shift
    nu = (fc / C_LIGHT) * (doppler_tx + doppler_target + doppler_rx)
    return float(nu)


def compute_path_gain(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    target_pos: np.ndarray,
    fc: float,
    rcs: float = 1.0
) -> float:
    """Compute bistatic radar path gain.

    alpha = sqrt( G_tx * G_rx * lambda^2 * sigma_rcs / ((4*pi)^3 * R_tx^2 * R_target_rx^2) )

    Uses Friis transmission for the tx→target→rx path.
    G_tx = G_rx = 1 (isotropic, conservative baseline).

    Args:
        tx_pos: Transmitter UAV position
        rx_pos: Receiver UAV position
        target_pos: Target position
        fc: Carrier frequency (Hz)
        rcs: Radar cross section (m^2)

    Returns:
        Complex-valued path gain magnitude (linear scale)
    """
    lam = C_LIGHT / fc  # wavelength
    r_tx_target = np.linalg.norm(tx_pos - target_pos)
    r_target_rx = np.linalg.norm(target_pos - rx_pos)

    eps = 1e-6
    r_tx_target = max(r_tx_target, eps)
    r_target_rx = max(r_target_rx, eps)

    # Bistatic radar equation: path gain magnitude
    # |alpha|^2 = (lam^2 * rcs) / ((4*pi)^3 * R_tx_target^2 * R_target_rx^2)
    alpha_sq = (lam ** 2 * rcs) / ((4 * np.pi) ** 3 * r_tx_target ** 2 * r_target_rx ** 2)
    return float(np.sqrt(max(alpha_sq, 0.0)))


def compute_all_bistatic_params(
    uav_positions: np.ndarray,    # (K, 3)
    uav_velocities: np.ndarray,   # (K, 3)
    target_positions: np.ndarray, # (Q, 3)
    target_velocities: np.ndarray,# (Q, 3)
    roles: np.ndarray,            # (K,) int: 0=tx, 1=rx, 2=idle
    fc: float,
    rcs: float = 1.0,
    role_agnostic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (tau, nu, alpha) for all valid bistatic pairs.

    A valid pair requires i in tx set, j in rx set, i != j. When
    `role_agnostic` is True the role partition is ignored and EVERY UAV is
    eligible as both tx and rx (all ordered i!=j pairs), so role assignment can
    be delegated to the downstream P0 solver instead of the policy.

    Args:
        uav_positions: (K, 3) UAV positions
        uav_velocities: (K, 3) UAV velocities
        target_positions: (Q, 3) target positions
        target_velocities: (Q, 3) target velocities
        roles: (K,) role assignments (0=tx, 1=rx, 2=idle); ignored if role_agnostic
        fc: Carrier frequency (Hz)
        rcs: Radar cross section (m^2)
        role_agnostic: If True, all UAVs are candidate tx and rx.

    Returns:
        tau: (K, K, Q) delay matrix (inf for invalid pairs)
        nu: (K, K, Q) Doppler matrix (0 for invalid pairs)
        alpha: (K, K, Q) path gain matrix (0 for invalid pairs)
    """
    K = uav_positions.shape[0]
    Q = target_positions.shape[0]

    # Endpoint-target quantities have shape (K,Q).  Bistatic pair quantities
    # then follow from broadcasting the Tx endpoint on axis 0 and the Rx
    # endpoint on axis 1.  This is algebraically identical to the scalar triple
    # loop above but avoids O(K^2 Q) Python calls on every online frame.
    endpoint_vector = (
        target_positions[None, :, :] - uav_positions[:, None, :])
    endpoint_range = np.linalg.norm(endpoint_vector, axis=-1)
    endpoint_unit = endpoint_vector / (
        endpoint_range[..., None] + 1.0e-10)

    tau = (
        endpoint_range[:, None, :] + endpoint_range[None, :, :]
    ) / C_LIGHT
    node_radial = np.sum(
        uav_velocities[:, None, :] * endpoint_unit, axis=-1)
    target_radial = np.sum(
        target_velocities[None, :, :] * endpoint_unit, axis=-1)
    nu = (float(fc) / C_LIGHT) * (
        node_radial[:, None, :] + node_radial[None, :, :]
        - target_radial[:, None, :] - target_radial[None, :, :]
    )

    safe_range = np.maximum(endpoint_range, 1.0e-6)
    wavelength = C_LIGHT / float(fc)
    alpha_sq = (
        wavelength * wavelength * float(rcs) / (4.0 * np.pi) ** 3
    ) / (
        safe_range[:, None, :] ** 2 * safe_range[None, :, :] ** 2
    )
    alpha = np.sqrt(np.maximum(alpha_sq, 0.0))

    if role_agnostic:
        valid_pair = np.ones((K, K), dtype=bool)
    else:
        valid_pair = (
            (np.asarray(roles)[:, None] == 0)
            & (np.asarray(roles)[None, :] == 1)
        )
    valid_pair &= ~np.eye(K, dtype=bool)
    invalid = ~valid_pair[:, :, None]
    tau = np.where(invalid, np.inf, tau)
    nu = np.where(invalid, 0.0, nu)
    alpha = np.where(invalid, 0.0, alpha)
    return tau, nu, alpha
