"""Deflection computation: the central coupling quantity.

Deflection d_ijq is the deflection coefficient (non-centrality parameter
of the detection statistic) for the bistatic pair (tx_i, rx_j, target_q).

This module bridges geometry/channel/OTFS → detection performance.
"""

import numpy as np
from dataclasses import dataclass
from itertools import repeat, starmap
from typing import List, Optional

from uav_isac.physical.geometry import compute_all_bistatic_params
from uav_isac.physical.otfs import (
    compute_dd_effectiveness,
    compute_dd_phys_gain,
    compute_dd_phys_gain_batch,
)
from uav_isac.physical.channel import (
    compute_noise_power,
    compute_report_link_reliability
)
from uav_isac.utils.types import DeflectionEntry


# Speed of light
C_LIGHT = 3.0e8


@dataclass(frozen=True)
class DenseDeflection:
    """Tensor form of deterministic U2U deflection.

    The physical quantities remain indexed by ``(tx, rx, target)``.  Keeping
    them dense lets ranking and power allocation operate on contiguous arrays;
    ``DeflectionEntry`` objects are materialized only at an interface that
    still requires the historical list representation.
    """

    tau: np.ndarray
    nu: np.ndarray
    alpha: np.ndarray
    d_raw: np.ndarray
    g_dd: np.ndarray
    d_eff: np.ndarray
    valid: np.ndarray

    def to_entries(
        self,
        power_scale_w: Optional[np.ndarray] = None,
    ) -> List[DeflectionEntry]:
        """Materialize entries in the canonical ``i -> j -> q`` order.

        ``power_scale_w`` exploits the exact fixed-geometry linearity
        ``d(P)=P d(1)``.  It is intended for a tensor computed at unit power.
        Delay, Doppler, path gain and DD effectiveness are power independent.
        """
        raw = self.d_raw
        effective = self.d_eff
        if power_scale_w is not None:
            power = np.asarray(power_scale_w, dtype=np.float64)
            expected = (self.tau.shape[0], self.tau.shape[2])
            if power.shape != expected:
                raise ValueError(
                    f"power_scale_w must have shape {expected}, got "
                    f"{power.shape}")
            if not np.all(np.isfinite(power)) or np.any(power < -1.0e-12):
                raise ValueError(
                    "power_scale_w must be finite and non-negative")
            factor = np.maximum(power, 0.0)[:, None, :]
            raw = raw * factor
            effective = effective * factor

        K, _, _ = self.valid.shape
        selected = np.asarray(self.valid, dtype=bool).copy()
        diagonal = np.arange(K, dtype=np.intp)
        selected[diagonal, diagonal, :] = False
        i, j, q = np.nonzero(selected)
        count = int(i.size)
        if count == 0:
            return []

        # ``np.nonzero`` follows C order, which is the historical canonical
        # i -> j -> q ordering.  Convert columns in bulk instead of performing
        # seven ndarray scalar lookups and conversions inside three Python
        # loops for every valid edge.
        columns = zip(
            i.tolist(),
            j.tolist(),
            q.tolist(),
            self.tau[selected].tolist(),
            self.nu[selected].tolist(),
            self.alpha[selected].tolist(),
            raw[selected].tolist(),
            self.g_dd[selected].tolist(),
            repeat(1.0, count),
            effective[selected].tolist(),
        )
        return list(starmap(DeflectionEntry, columns))


def validate_cpi_schedule(
    n_cpi: int,
    N: int,
    T_sym: float,
    control_frame_s: float,
) -> tuple[float, int]:
    """Validate that explicitly scheduled OTFS looks fit one control frame.

    ``N*T_sym`` is one OTFS-frame duration because the ``M`` subcarriers are
    transmitted in parallel.  This certifies time feasibility only; it does
    not manufacture statistical independence or phase coherence.
    """
    if n_cpi < 1 or N < 1:
        raise ValueError("n_cpi and N must be positive integers")
    values = (float(T_sym), float(control_frame_s))
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("T_sym and control_frame_s must be finite and positive")
    frame_duration_s = int(N) * float(T_sym)
    max_time_feasible_looks = int(np.floor(
        (float(control_frame_s) + 1.0e-15) / frame_duration_s))
    if max_time_feasible_looks < 1:
        raise ValueError("one OTFS frame does not fit in the control frame")
    if int(n_cpi) > max_time_feasible_looks:
        raise ValueError(
            "scheduled CPI looks exceed the control-frame time budget: "
            f"{n_cpi} > {max_time_feasible_looks}")
    return frame_duration_s * int(n_cpi), max_time_feasible_looks


def compute_raw_deflection(
    alpha: float,
    P_sense: float,
    T_sym: float,
    M: int,
    N: int,
    noise_power: float,
    antenna_gain: float = 1.0,
    n_cpi: int = 1,
    c_det: float = 1.0,
) -> float:
    """Compute raw Deflection coefficient d_ijq.

    d_ijq = E_signal / N0
          = (P_sense * |alpha|^2 * N * T_sym)
            / (P_noise / B),  B=M/T_sym

    This is the pre-processing SNR accumulated over M*N DD bins.
    It represents the non-centrality parameter of the detection statistic
    under the Gaussian approximation.

    Args:
        alpha: Path gain magnitude (linear)
        P_sense: Sensing transmit power (W)
        T_sym: Symbol period (s)
        M: Delay bins
        N: Doppler bins
        noise_power: In-band noise power ``k*T*B*NF`` (W)

    Returns:
        Raw Deflection (dimensionless, >= 0)
    """
    eps = 1e-15
    if T_sym <= 0.0:
        raise ValueError("T_sym must be positive")
    if M < 1 or N < 1 or n_cpi < 1:
        raise ValueError("M, N, and n_cpi must be positive integers")
    if noise_power < 0.0:
        raise ValueError("noise_power must be non-negative")
    if not np.isfinite(c_det) or c_det <= 0.0:
        raise ValueError("c_det must be finite and positive")
    # Matched-filter energy convention: E_signal and the (one-sided effective)
    # noise PSD N0=P_noise/B are both measured in joules.  With B=M/T_sym,
    # E_signal/N0 = (P_r/P_noise)*M*N per scheduled OTFS frame.
    # antenna_gain = G_tx*G_rx (linear). n_cpi counts explicitly scheduled
    # OTFS observations; it must not stand for phantom coherent frames.
    observation_time_s = T_sym * N * n_cpi
    signal_energy = P_sense * (alpha ** 2) * antenna_gain * observation_time_s
    implied_bandwidth_hz = M / T_sym
    noise_psd = noise_power / implied_bandwidth_hz
    return float(c_det * signal_energy / max(noise_psd, eps / implied_bandwidth_hz))


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
        c_det: float = 1.0,
        control_frame_s: float | None = None,
        use_los_prob: bool = False,
        los_a: float = 4.88,
        los_b: float = 0.43,
        eta_los_dB: float = 0.1,
        eta_nlos_dB: float = 21.0,
        use_swerling: bool = False,
        use_report_link: bool = True,
        dd_gain_mode: str = "binary",
        sync_delay_error_bins: float = 0.0,
        sync_doppler_error_bins: float = 0.0,
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
        # Post-G2 physics closure (audit advice/001 section 5):
        #   binary      -- legacy ``1[g_dd >= g_min]`` support gate (historical
        #                  certified numbers are all under this convention).
        #   continuous  -- physical gain ``I_support * |A(tau,nu)|^2`` applied
        #                  continuously to raw deflection; out-of-support
        #                  targets contribute exactly 0.
        # The canonical post-G2 manifest pins ``continuous``; ``binary`` is kept
        # so existing pre-G2 result files remain reproducible.
        dd_gain_mode = str(dd_gain_mode)
        if dd_gain_mode not in ("binary", "continuous"):
            raise ValueError(
                f"dd_gain_mode must be 'binary' or 'continuous', got {dd_gain_mode!r}")
        self.dd_gain_mode = dd_gain_mode
        # Antenna array gain (linear) = G_tx*G_rx.  CPI looks are explicit
        # scheduled observations, not an assumed free coherent multiplier.
        self.antenna_gain = 10.0 ** ((g_tx_dBi + g_rx_dBi) / 10.0)
        self.n_cpi = int(n_cpi)
        self.cpi_duration_s = self.n_cpi * self.N * self.T_sym
        self.max_time_feasible_looks: int | None = None
        if control_frame_s is not None:
            self.cpi_duration_s, self.max_time_feasible_looks = (
                validate_cpi_schedule(
                    self.n_cpi, self.N, self.T_sym, control_frame_s))
        self.c_det = float(c_det)
        if not np.isfinite(self.c_det) or self.c_det <= 0.0:
            raise ValueError("c_det must be finite and positive")
        # low-altitude reporting-link blockage (Al-Hourani) and Swerling RCS fading
        self.use_los_prob = use_los_prob
        self.los_a, self.los_b = los_a, los_b
        self.eta_los_dB, self.eta_nlos_dB = eta_los_dB, eta_nlos_dB
        self.use_swerling = use_swerling
        self.use_report_link = bool(use_report_link)
        self.sync_delay_error_bins = float(sync_delay_error_bins)
        self.sync_doppler_error_bins = float(sync_doppler_error_bins)
        for name, value in (
            ("sync_delay_error_bins", self.sync_delay_error_bins),
            ("sync_doppler_error_bins", self.sync_doppler_error_bins),
        ):
            if not np.isfinite(value) or abs(value) > 0.5:
                raise ValueError(f"{name} must be finite and lie in [-0.5, 0.5]")

    def _synchronized_coordinates(
        self, tau: np.ndarray | float, nu: np.ndarray | float,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Return receiver-side coordinates after a fixed fractional-bin error."""
        tau_error_s = self.sync_delay_error_bins / (self.M * self.delta_f)
        nu_error_hz = self.sync_doppler_error_bins / (self.N * self.T_sym)
        return tau + tau_error_s, nu + nu_error_hz

    def compute_dense(
        self,
        uav_positions: np.ndarray,
        uav_velocities: np.ndarray,
        target_positions: np.ndarray,
        target_velocities: np.ndarray,
        roles: np.ndarray,
        fc_position: np.ndarray,
        role_agnostic: bool = False,
        sensing_power_w: Optional[np.ndarray] = None,
    ) -> DenseDeflection:
        """Compute deterministic U2U deflection without per-edge objects.

        This representation is valid only when the receiver-to-fusion report
        channel and Swerling fading are disabled.  Those mechanisms contain
        stateful random draws whose physical ordering must remain explicit.
        """
        if self.use_report_link or self.use_swerling:
            raise ValueError(
                "dense deflection requires deterministic U2U-only execution")
        K = uav_positions.shape[0]
        Q = target_positions.shape[0]
        if sensing_power_w is None:
            power = np.full((K, Q), self.P_sense, dtype=np.float64)
        else:
            power = np.asarray(sensing_power_w, dtype=np.float64)
            if power.shape != (K, Q):
                raise ValueError(
                    f"sensing_power_w must have shape {(K, Q)}, got "
                    f"{power.shape}")
            if not np.all(np.isfinite(power)) or np.any(power < -1.0e-12):
                raise ValueError(
                    "sensing_power_w must be finite and non-negative")
            power = np.maximum(power, 0.0)

        tau, nu, alpha = compute_all_bistatic_params(
            uav_positions, uav_velocities,
            target_positions, target_velocities,
            roles, self.fc, self.rcs, role_agnostic=role_agnostic,
        )
        raw_scale = float(
            self.c_det * self.antenna_gain * self.M * self.N * self.n_cpi
            / max(self.noise_power, 1.0e-15)
        )
        d_raw = raw_scale * alpha * alpha * power[:, None, :]
        finite_tau = np.isfinite(tau)
        finite_nu = np.isfinite(nu)
        tau_eval = np.where(finite_tau, tau, 0.0)
        nu_eval = np.where(finite_nu, nu, 0.0)
        tau_sync, nu_sync = self._synchronized_coordinates(tau_eval, nu_eval)
        delay_fraction = tau_sync * self.M * self.delta_f
        doppler_fraction = nu_sync * self.N * self.T_sym
        delay_offset = delay_fraction - np.round(delay_fraction)
        doppler_offset = doppler_fraction - np.round(doppler_fraction)
        g_dd = np.abs(np.sinc(delay_offset) * np.sinc(doppler_offset))
        if self.dd_gain_mode == "continuous":
            support = (
                finite_tau
                & finite_nu
                & (tau_sync >= 0.0)
                & (tau_sync < 1.0 / self.delta_f)
                & (np.abs(nu_sync) <= 1.0 / (2.0 * self.T_sym))
            )
            d_eff = d_raw * support.astype(np.float64) * g_dd ** 2
        else:
            d_eff = d_raw * (g_dd >= self.g_min).astype(np.float64)

        if role_agnostic:
            endpoint_valid = ~np.eye(K, dtype=bool)
        else:
            role_array = np.asarray(roles)
            endpoint_valid = (
                (role_array[:, None] == 0)
                & (role_array[None, :] == 1)
            )
            endpoint_valid &= ~np.eye(K, dtype=bool)
        valid = np.broadcast_to(endpoint_valid[:, :, None], (K, K, Q)).copy()
        return DenseDeflection(
            tau=tau, nu=nu, alpha=alpha, d_raw=d_raw,
            g_dd=g_dd, d_eff=d_eff, valid=valid,
        )

    def compute(
        self,
        uav_positions: np.ndarray,     # (K, 3)
        uav_velocities: np.ndarray,    # (K, 3)
        target_positions: np.ndarray,  # (Q, 3)
        target_velocities: np.ndarray, # (Q, 3)
        roles: np.ndarray,             # (K,) int: 0=tx, 1=rx, 2=idle
        fc_position: np.ndarray,       # (3,) fusion center position
        role_agnostic: bool = False,   # if True, any UAV may tx/rx (P0 assigns roles)
        sensing_power_w: Optional[np.ndarray] = None,  # (K,Q) target-wise TX power
    ) -> List[DeflectionEntry]:
        """Compute Deflection entries for all valid bistatic pairs.

        Pipeline:
        1. Compute (tau, nu, alpha) for all (i, j, q)
        2. Compute raw Deflection d_raw
        3. Compute DD effectiveness g_dd
        4. Compute reporting link reliability chi_rep
        5. Compute effective Deflection.  Canonical continuous mode uses
           chi_rep*d_raw*I_support*|A|^2; legacy binary mode uses
           chi_rep*d_raw when g_dd >= g_min.

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
        if not self.use_report_link and not self.use_swerling:
            return self.compute_dense(
                uav_positions, uav_velocities,
                target_positions, target_velocities,
                roles, fc_position,
                role_agnostic=role_agnostic,
                sensing_power_w=sensing_power_w,
            ).to_entries()
        if sensing_power_w is not None:
            sensing_power_w = np.asarray(sensing_power_w, dtype=np.float64)
            if sensing_power_w.shape != (K, Q):
                raise ValueError(
                    f'sensing_power_w must have shape {(K, Q)}, got '
                    f'{sensing_power_w.shape}')
            if np.any(sensing_power_w < -1e-12):
                raise ValueError('sensing_power_w must be non-negative')

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

        # V3-C0 (advice 015, 2026-08-17): the report-link reliability chi_rep
        # depends ONLY on the receiver UAV j -> FC channel, but it was previously
        # redrawn (fresh Rician realization) for EVERY transmitter i -- the same
        # physical link got a different reliability per TX, injecting tx-dependent
        # randomness into d_eff and consuming |tx|*|rx| RNG draws per frame.
        # Draw once per receiver and reuse across transmitters.
        if self.use_report_link:
            chi_rep_by_rx = {
                int(j): compute_report_link_reliability(
                    uav_positions[j], fc_position,
                    self.fc, self.ric_K, self.noise_power,
                    self.P_report, self.rng,
                    use_los_prob=self.use_los_prob,
                    los_a=self.los_a, los_b=self.los_b,
                    eta_los_dB=self.eta_los_dB, eta_nlos_dB=self.eta_nlos_dB,
                )
                for j in rx_indices
            }
        else:
            # U2U-only study: sensing quality is determined by the bistatic echo,
            # not by a non-existent ground-report link.
            chi_rep_by_rx = {int(j): 1.0 for j in rx_indices}

        # O2 (roadmap 2026-08-29): canonical-path vectorization.
        # Canonical identity = use_report_link=True + dd_gain_mode=continuous +
        # no Swerling (manifest pins continuous; Swerling off by default).
        # The scalar slow path below is exactly the formula of
        # compute_raw_deflection / compute_dd_effectiveness /
        # compute_dd_phys_gain; the batch branch evaluates the SAME formulas
        # as (K,K,Q) arrays (raw_scale*alpha**2*power is algebraically
        # identical to compute_raw_deflection; compute_dd_phys_gain_batch is
        # bit-for-bit the scalar compute_dd_phys_gain, otfs.py:146-196) and
        # assembles entries in the original (i -> j -> q) order, so results
        # are bit-for-bit identical.  No RNG draws in this branch.
        # Binary legacy mode and ANY Swerling mode keep the original scalar
        # loop below (Swerling draws are RNG-order stateful and must not be
        # reordered; binary is the pre-G2 certified convention).
        if self.dd_gain_mode == "continuous" and not self.use_swerling:
            if sensing_power_w is None:
                power = np.full((K, Q), self.P_sense, dtype=np.float64)
            else:
                power = np.maximum(sensing_power_w, 0.0)
            # Same evaluation order as the scalar compute_raw_deflection
            # (deflection.py 58-107): observation_time=T_sym*N*n_cpi,
            # signal_energy=P*(alpha**2)*antenna_gain*observation_time,
            # noise_psd=noise_power/(M/T_sym), result=c_det*energy/
            # max(noise_psd, 1e-15/implied_bw).  Using the identical
            # expression order keeps the batch path within 1-2 ULP of the
            # scalar path (bit-for-bit not claimed: float, C1).
            implied_bandwidth_hz = self.M / self.T_sym
            noise_psd = self.noise_power / implied_bandwidth_hz
            min_denom = 1.0e-15 / implied_bandwidth_hz
            observation_time_s = self.T_sym * self.N * self.n_cpi
            signal_energy = (
                power[:, None, :] * (alpha ** 2)
                * self.antenna_gain * observation_time_s)
            denom = np.maximum(noise_psd, min_denom)
            d_raw_all = self.c_det * signal_energy / denom
            # compute_dd_phys_gain_batch rejects non-finite tau/nu (otfs.py:
            # it raises before the support mask).  The geometry layer emits
            # inf tau for invalid pairs (same node as tx/rx etc.); the scalar
            # slow path skips those (``if np.isinf(tau): continue``).  Mask
            # them to 0 here (support check then yields gain 0 at exactly the
            # same positions) and keep the skip in the assembly below.
            finite_tau = np.isfinite(tau)
            finite_nu = np.isfinite(nu)
            tau_safe = np.where(finite_tau, tau, 0.0)
            nu_safe = np.where(finite_nu, nu, 0.0)
            tau_sync, nu_sync = self._synchronized_coordinates(
                tau_safe, nu_safe)
            phys_gain_all = compute_dd_phys_gain_batch(
                tau_sync, nu_sync, self.delta_f, self.T_sym, self.M, self.N)
            chi_rep_all = np.zeros((1, K, 1), dtype=np.float64)
            for j in rx_indices:
                chi_rep_all[0, int(j), 0] = float(chi_rep_by_rx[int(j)])
            d_eff_all = chi_rep_all * d_raw_all * phys_gain_all
            return [
                DeflectionEntry(
                    i=int(i), j=int(j), q=int(q),
                    tau=float(tau[i, j, q]),
                    nu=float(nu[i, j, q]),
                    alpha=float(alpha[i, j, q]),
                    d_raw=float(d_raw_all[i, j, q]),
                    g_dd=float(compute_dd_effectiveness(
                        tau[i, j, q], nu[i, j, q],
                        self.delta_f, self.T_sym, self.M, self.N)),
                    chi_rep=float(chi_rep_by_rx[int(j)]),
                    d_eff=float(d_eff_all[i, j, q]),
                )
                for i in tx_indices
                for j in rx_indices
                if i != j
                for q in range(Q)
                if not np.isinf(tau[i, j, q])
            ]

        # Legacy scalar slow path: binary gate and/or Swerling RCS fading.  The
        # RNG draw order below is part of the reproducible random stream.
        for i in tx_indices:
            for j in rx_indices:
                if i == j:
                    continue

                chi_rep = chi_rep_by_rx[int(j)]

                for q in range(Q):
                    if np.isinf(tau[i, j, q]):
                        continue

                    # Step 2: Raw Deflection
                    target_power_w = (
                        self.P_sense if sensing_power_w is None
                        else max(float(sensing_power_w[i, q]), 0.0)
                    )
                    d_raw = compute_raw_deflection(
                        alpha[i, j, q], target_power_w,
                        self.T_sym, self.M, self.N, self.noise_power,
                        antenna_gain=self.antenna_gain, n_cpi=self.n_cpi,
                        c_det=self.c_det,
                    )

                    # Step 3: DD effectiveness
                    tau_sync, nu_sync = self._synchronized_coordinates(
                        tau[i, j, q], nu[i, j, q])
                    g_dd = compute_dd_effectiveness(
                        tau_sync, nu_sync,
                        self.delta_f, self.T_sym,
                        self.M, self.N, self.g_min
                    )

                    # Optional Swerling-II RCS fading: multiply by exp(1) per look
                    if self.use_swerling:
                        d_raw = d_raw * float(self.rng.exponential(1.0))

                    # Step 4-5: Effective Deflection
                    if self.dd_gain_mode == "continuous":
                        # Post-G2 physics (audit advice/001 section 5):
                        # ``d_eff = chi_rep * d_raw * I_support * |A|^2``.
                        # Out-of-support targets get exactly zero even if the
                        # aliased fractional mismatch is near an integer bin.
                        phys_gain = compute_dd_phys_gain(
                            tau_sync, nu_sync,
                            self.delta_f, self.T_sym,
                            self.M, self.N,
                        )
                        d_eff = chi_rep * d_raw * phys_gain
                    else:  # legacy binary gate
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
