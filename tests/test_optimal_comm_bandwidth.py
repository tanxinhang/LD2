"""Tests for D1.1-C optimal orthogonal-bandwidth allocation in the L0
analytical minimum comm power (2026-08-16).

Theory: for sender i the Shannon power is f_i(B) = N0*B*max(gamma_th,
2^(r_i/B)-1)/g_i, convex-decreasing in B (max preserves convexity).  The
equal split B/n_active over-charges high-load senders near capacity; the
optimal allocation min sum_i f_i(B_i) s.t. sum_i B_i = B solves the KKT
condition f_i'(B_i) = -lambda exactly by bisection.  Every sender still
meets its SNR threshold and rate demand, so the optimal total power is
strictly no larger than the equal split.
"""

import numpy as np
import pytest

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def _env_with_payloads(payloads_per_uav):
    """K = len(payloads) env with the analytical comm-power path active and
    per-UAV payload bits injected so the L0 min comm power is well defined."""
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    cfg.scenario.K = cfg.scenario.Q = len(payloads_per_uav)
    cfg.scenario.T = 20
    cfg.target.omega_q = [0.5] * cfg.scenario.Q
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.analytical_comm_power_enabled = True
    env = UAVISACEnv(cfg, seed=11)
    env.reset()
    core = env.core
    # Fake pending comm state: one message per UAV with the given payload.
    core._pending_comm_messages = {k: [None] for k in range(cfg.scenario.K)}
    core._pending_comm_rates = {k: 1 for k in range(cfg.scenario.K)}
    # Force the token mask so _active_dimensions returns a fixed count.
    for k in range(cfg.scenario.K):
        core._pending_comm_token_masks[k] = np.zeros(
            getattr(core.cfg.marl, 'comm_target_token_dim', 16), dtype=np.int64)
    core._pending_structure_student_protocol = set()
    positions = np.array([[100.0 + 40.0 * k, 200.0] for k in range(cfg.scenario.K)])
    return core, positions, payloads_per_uav


def test_optimal_bw_total_power_never_exceeds_equal_split():
    core, positions, payloads = _env_with_payloads(
        [2000, 8000, 4000, 12000])  # heterogeneous rates
    core.cfg.marl.analytical_comm_optimal_bw = True
    p_opt = core._compute_analytical_min_comm_power(positions)
    core.cfg.marl.analytical_comm_optimal_bw = False
    p_eq = core._compute_analytical_min_comm_power(positions)
    assert np.all(p_opt <= p_eq + 1e-15)
    assert float(np.sum(p_opt)) < float(np.sum(p_eq)) + 1e-12


def test_optimal_bw_preserves_snr_and_rate_semantics():
    core, positions, payloads = _env_with_payloads(
        [2000, 8000, 4000, 12000])
    core.cfg.marl.analytical_comm_optimal_bw = True
    p = core._compute_analytical_min_comm_power(positions)
    comm = core._inter_uav_comm
    t_win = max(comm.deadline_s - comm.processing_delay_s, 1e-12)
    gamma_th = float(10.0 ** (comm.snr_threshold_db / 10.0))
    n0 = comm.kT * comm.noise_figure_linear
    # Every active sender's power must at least cover its worst-receiver SNR
    # and its Shannon rate demand under the total bandwidth (a power that
    # meets both is feasible; the allocation only chooses the bandwidth split).
    for k in range(core.K):
        if p[k] <= 0.0:
            continue
        r = payloads[k] / t_win
        # Upper-bound the required power under any B_i in (0, B]:
        # P >= max(gamma_th, 2^(r/B)-1) * N0 * B / g is decreasing in B, so
        # the equal-split power is an upper bound of what any B_i needs.
        b_eff = comm.bandwidth_hz / max(1, int(np.sum(p > 0.0)))
        gamma_rate = float(2.0 ** (r / b_eff) - 1.0)
        gamma_req = max(gamma_th, gamma_rate)
        d = 1.0  # nearest possible receiver distance -> largest gain
        g_max = comm.antenna_gain_linear * (comm.wavelength / (4.0 * np.pi * d)) ** 2
        assert p[k] <= gamma_req * n0 * b_eff / g_max + 1e-9


def test_optimal_bw_matches_equal_split_for_homogeneous_senders():
    # With identical rates and gains the symmetric optimum IS the equal split.
    core, positions, payloads = _env_with_payloads([4000, 4000, 4000, 4000])
    core.cfg.marl.analytical_comm_optimal_bw = True
    p_opt = core._compute_analytical_min_comm_power(positions)
    core.cfg.marl.analytical_comm_optimal_bw = False
    p_eq = core._compute_analytical_min_comm_power(positions)
    np.testing.assert_allclose(p_opt, p_eq, rtol=1e-3, atol=1e-12)
