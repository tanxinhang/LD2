"""Tests for Deflection computation."""

import numpy as np
import pytest
from uav_isac.physical.deflection import (
    compute_raw_deflection,
    DeflectionComputer,
)


class TestRawDeflection:
    def test_increases_with_snr(self):
        """Higher SNR → higher Deflection."""
        alpha_high = 1e-6
        alpha_low = 1e-7
        P_sense = 0.5
        T_sym = 8.33e-6
        M, N = 64, 16
        noise = 1e-12

        d_high = compute_raw_deflection(alpha_high, P_sense, T_sym, M, N, noise)
        d_low = compute_raw_deflection(alpha_low, P_sense, T_sym, M, N, noise)
        assert d_high > d_low

    def test_non_negative(self):
        """Deflection is always non-negative."""
        d = compute_raw_deflection(0.0, 0.5, 8.33e-6, 64, 16, 1e-12)
        assert d >= 0

    def test_proportional_to_coherent_gain(self):
        """Deflection ∝ M * N (coherent integration gain)."""
        alpha = 1e-6
        P_sense = 0.5
        T_sym = 8.33e-6
        noise = 1e-12

        d_64_16 = compute_raw_deflection(alpha, P_sense, T_sym, 64, 16, noise)
        d_32_8 = compute_raw_deflection(alpha, P_sense, T_sym, 32, 8, noise)

        # d(64*16) ≈ 4 * d(32*8) since M*N ratio = 4
        ratio = d_64_16 / max(d_32_8, 1e-15)
        assert ratio == pytest.approx(4.0, rel=0.01)


class TestDeflectionComputer:
    def test_produces_entries(self, default_config, seeded_rng,
                               sample_uav_positions, sample_uav_velocities,
                               sample_target_positions, sample_target_velocities,
                               sample_roles, fc_position):
        """End-to-end: DeflectionComputer produces entries for valid pairs."""
        cfg = default_config
        computer = DeflectionComputer(
            fc=cfg.otfs.fc,
            delta_f=cfg.otfs.delta_f,
            T_sym=cfg.otfs.T_sym,
            M=cfg.otfs.M,
            N=cfg.otfs.N,
            kT=cfg.channel.kT,
            B=cfg.otfs.B,
            NF_dB=cfg.channel.NF,
            P_sense=cfg.uav.P_sense,
            P_report=cfg.uav.P_report,
            ric_K=cfg.channel.ric_K,
            rcs=cfg.target.rcs,
            g_min=cfg.detection.g_min,
            rng=seeded_rng,
        )

        entries = computer.compute(
            sample_uav_positions, sample_uav_velocities,
            sample_target_positions, sample_target_velocities,
            sample_roles, fc_position
        )

        # Should have entries for (tx0, rx2), (tx0, rx3), (tx1, rx2), (tx1, rx3)
        # for Q=2 targets → up to 8 entries
        assert len(entries) > 0
        assert len(entries) <= 8  # 2 tx * 2 rx * 2 targets

        for e in entries:
            assert e.i in (0, 1)  # tx UAVs
            assert e.j in (2, 3)  # rx UAVs
            assert e.q in (0, 1)  # targets
            assert e.d_raw >= 0
            assert 0 <= e.g_dd <= 1
            assert 0 <= e.chi_rep <= 1
            assert e.d_eff >= 0

    def test_deflection_monotonic_with_distance(
        self, default_config, seeded_rng, fc_position):
        """Closer targets produce higher Deflection."""
        cfg = default_config
        computer = DeflectionComputer(
            fc=cfg.otfs.fc, delta_f=cfg.otfs.delta_f,
            T_sym=cfg.otfs.T_sym, M=cfg.otfs.M, N=cfg.otfs.N,
            kT=cfg.channel.kT, B=cfg.otfs.B, NF_dB=cfg.channel.NF,
            P_sense=cfg.uav.P_sense, P_report=cfg.uav.P_report,
            ric_K=cfg.channel.ric_K, rcs=cfg.target.rcs,
            g_min=cfg.detection.g_min, rng=seeded_rng,
        )

        uav_pos = np.array([[100.0, 100.0, 100.0], [900.0, 100.0, 100.0],
                            [100.0, 900.0, 100.0], [900.0, 900.0, 100.0]])
        uav_vel = np.zeros((4, 3))
        target_near = np.array([[400.0, 500.0, 0.0]])
        target_far = np.array([[400.0, 900.0, 0.0]])
        tgt_vel = np.zeros((1, 3))
        roles = np.array([0, 0, 1, 1], dtype=np.int32)

        # Entries for near target
        entries_near = computer.compute(
            uav_pos, uav_vel, target_near, tgt_vel, roles, fc_position)

        # Use same RNG seed for fair comparison
        seeded_rng2 = np.random.default_rng(42)
        computer2 = DeflectionComputer(
            fc=cfg.otfs.fc, delta_f=cfg.otfs.delta_f,
            T_sym=cfg.otfs.T_sym, M=cfg.otfs.M, N=cfg.otfs.N,
            kT=cfg.channel.kT, B=cfg.otfs.B, NF_dB=cfg.channel.NF,
            P_sense=cfg.uav.P_sense, P_report=cfg.uav.P_report,
            ric_K=cfg.channel.ric_K, rcs=cfg.target.rcs,
            g_min=cfg.detection.g_min, rng=seeded_rng2,
        )
        entries_far = computer2.compute(
            uav_pos, uav_vel, target_far, tgt_vel, roles, fc_position)

        # The near target should have higher d_raw on average
        d_near_mean = np.mean([e.d_raw for e in entries_near])
        d_far_mean = np.mean([e.d_raw for e in entries_far])
        assert d_near_mean > d_far_mean
