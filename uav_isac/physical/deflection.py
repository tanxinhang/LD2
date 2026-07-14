"""Deflection computation: the central coupling quantity.

Deflection d_ijq is the deflection coefficient (non-centrality parameter
of the detection statistic) for the bistatic pair (tx_i, rx_j, target_q).

This module bridges geometry/channel/OTFS → detection performance.
"""

import numpy as np
from typing import List, Optional

from uav_isac.physical.geometry import compute_all_bistatic_params
from uav_isac.physical.otfs import compute_dd_effectiveness
from uav_isac.physical.channel import (
    compute_noise_power,
    compute_report_link_reliability
)
from uav_isac.utils.types import DeflectionEntry


# Speed of light
C_LIGHT = 3.0e8


def compute_raw_deflection(
    alpha: float,
    P_sense: float,
    T_sym: float,
    M: int,
    N: int,
    noise_power: float,
    antenna_gain: float = 1.0,
    n_cpi: int = 1
) -> float:
    """Compute raw Deflection coefficient d_ijq.

    d_ijq = (P_sense * |alpha|^2 * T_sym * M * N) / sigma_z^2

    This is the pre-processing SNR accumulated over M*N DD bins.
    It represents the non-centrality parameter of the detection statistic
    under the Gaussian approximation.

    Args:
        alpha: Path gain magnitude (linear)
        P_sense: Sensing transmit power (W)
        T_sym: Symbol period (s)
        M: Delay bins
        N: Doppler bins
        noise_power: Noise power sigma_z^2 (W)

    Returns:
        Raw Deflection (dimensionless, >= 0)
    """
    eps = 1e-15
    # antenna_gain = G_tx*G_rx (linear); n_cpi = coherent frames integrated
    signal_energy = P_sense * (alpha ** 2) * T_sym * M * N * antenna_gain * n_cpi
    return float(signal_energy / max(noise_power, eps))


class DeflectionComputer:
    """Computes the full Deflection matrix for all bistatic pairs."""

    def __init__(
        self,
        fc: float,
        delta_f: float,
        T_sym: float,
        M: int,
        N: int,
        kT: float,
        B: float,
        NF_dB: float,
        P_sense: float,
        P_report: float,
        ric_K: float,
        rcs: float,
        g_min: float,
        rng: np.random.Generator,
        g_tx_dBi: float = 0.0,
        g_rx_dBi: float = 0.0,
        n_cpi: int = 1,
        use_los_prob: bool = False,
        los_a: float = 4.88,
        los_b: float = 0.43,
        eta_los_dB: float = 0.1,
        eta_nlos_dB: float = 21.0,
        use_swerling: bool = False,
    ):
        self.fc = fc
        self.delta_f = delta_f
        self.T_sym = T_sym
        self.M = M
        self.N = N
        self.noise_power = compute_noise_power(kT, B, NF_dB)
        self.P_sense = P_sense
        self.P_report = P_report
        self.ric_K = ric_K
        self.rcs = rcs
        self.g_min = g_min
        self.rng = rng
        # antenna array gain (linear) = G_tx*G_rx, and coherent integration frames
        self.antenna_gain = 10.0 ** ((g_tx_dBi + g_rx_dBi) / 10.0)
        self.n_cpi = int(n_cpi)
        # low-altitude reporting-link blockage (Al-Hourani) and Swerling RCS fading
        self.use_los_prob = use_los_prob
        self.los_a, self.los_b = los_a, los_b
        self.eta_los_dB, self.eta_nlos_dB = eta_los_dB, eta_nlos_dB
        self.use_swerling = use_swerling

    def compute(
        self,
        uav_positions: np.ndarray,     # (K, 3)
        uav_velocities: np.ndarray,    # (K, 3)
        target_positions: np.ndarray,  # (Q, 3)
        target_velocities: np.ndarray, # (Q, 3)
        roles: np.ndarray,             # (K,) int: 0=tx, 1=rx, 2=idle
        fc_position: np.ndarray,       # (3,) fusion center position
        role_agnostic: bool = False,   # if True, any UAV may tx/rx (P0 assigns roles)
    ) -> List[DeflectionEntry]:
        """Compute Deflection entries for all valid bistatic pairs.

        Pipeline:
        1. Compute (tau, nu, alpha) for all (i, j, q)
        2. Compute raw Deflection d_raw
        3. Compute DD effectiveness g_dd
        4. Compute reporting link reliability chi_rep
        5. Compute effective Deflection d_eff = chi_rep * d_raw if g_dd >= g_min

        Args:
            uav_positions: (K, 3) UAV positions
            uav_velocities: (K, 3) UAV velocities
            target_positions: (Q, 3) target positions
            target_velocities: (Q, 3) target velocities
            roles: (K,) role assignments
            fc_position: (3,) fusion center position

        Returns:
            List of DeflectionEntry for all valid bistatic pairs
        """
        K = uav_positions.shape[0]
        Q = target_positions.shape[0]

        # Step 1: Compute geometry
        tau, nu, alpha = compute_all_bistatic_params(
            uav_positions, uav_velocities,
            target_positions, target_velocities,
            roles, self.fc, self.rcs, role_agnostic=role_agnostic
        )

        entries = []
        if role_agnostic:
            tx_indices = np.arange(K)   # any UAV may transmit; P0 picks roles
            rx_indices = np.arange(K)   # any UAV may receive
        else:
            tx_indices = np.where(roles == 0)[0]
            rx_indices = np.where(roles == 1)[0]

        for i in tx_indices:
            for j in rx_indices:
                if i == j:
                    continue

                # Reporting link reliability for rx UAV j → FC
                # (low-altitude blockage via Al-Hourani LoS/NLoS if enabled)
                chi_rep = compute_report_link_reliability(
                    uav_positions[j], fc_position,
                    self.fc, self.ric_K, self.noise_power,
                    self.P_report, self.rng,
                    use_los_prob=self.use_los_prob,
                    los_a=self.los_a, los_b=self.los_b,
                    eta_los_dB=self.eta_los_dB, eta_nlos_dB=self.eta_nlos_dB,
                )

                for q in range(Q):
                    if np.isinf(tau[i, j, q]):
                        continue

                    # Step 2: Raw Deflection
                    d_raw = compute_raw_deflection(
                        alpha[i, j, q], self.P_sense,
                        self.T_sym, self.M, self.N, self.noise_power,
                        antenna_gain=self.antenna_gain, n_cpi=self.n_cpi
                    )

                    # Step 3: DD effectiveness
                    g_dd = compute_dd_effectiveness(
                        tau[i, j, q], nu[i, j, q],
                        self.delta_f, self.T_sym,
                        self.M, self.N, self.g_min
                    )

                    # Optional Swerling-II RCS fading: multiply by exp(1) per look
                    if self.use_swerling:
                        d_raw = d_raw * float(self.rng.exponential(1.0))

                    # Step 4-5: Effective Deflection
                    if g_dd >= self.g_min:
                        d_eff = chi_rep * d_raw
                    else:
                        d_eff = 0.0

                    entry = DeflectionEntry(
                        i=int(i), j=int(j), q=int(q),
                        tau=float(tau[i, j, q]),
                        nu=float(nu[i, j, q]),
                        alpha=float(alpha[i, j, q]),
                        d_raw=float(d_raw),
                        g_dd=float(g_dd),
                        chi_rep=float(chi_rep),
                        d_eff=float(d_eff)
                    )
                    entries.append(entry)

        return entries
