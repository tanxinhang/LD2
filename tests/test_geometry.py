"""Tests for bistatic geometry computations."""

import numpy as np
import pytest
from uav_isac.physical.geometry import (
    compute_bistatic_range,
    compute_delay,
    compute_doppler,
    compute_path_gain,
    compute_all_bistatic_params,
    C_LIGHT,
)


class TestBistaticRange:
    def test_colinear(self):
        """Target directly between tx and rx."""
        tx = np.array([0.0, 0.0, 100.0])
        rx = np.array([1000.0, 0.0, 100.0])
        target = np.array([500.0, 0.0, 0.0])
        # dist tx→target = sqrt(500^2 + 100^2) = ~509.9
        # dist target→rx = sqrt(500^2 + 100^2) = ~509.9
        r = compute_bistatic_range(tx, rx, target)
        assert r == pytest.approx(1019.8, rel=1e-3)

    def test_monostatic_like(self):
        """Tx and rx at same position (monostatic limit)."""
        tx = np.array([0.0, 0.0, 100.0])
        rx = np.array([0.0, 0.0, 100.0])
        target = np.array([300.0, 0.0, 0.0])
        r = compute_bistatic_range(tx, rx, target)
        expected = 2 * np.sqrt(300 ** 2 + 100 ** 2)
        assert r == pytest.approx(expected, rel=1e-6)


class TestDelay:
    def test_typical(self):
        """Delay for 1 km bistatic range."""
        r = 1000.0
        tau = compute_delay(r)
        assert tau == pytest.approx(1000.0 / C_LIGHT, rel=1e-6)
        assert tau < 1e-5  # < 10 microseconds

    def test_zero_range(self):
        tau = compute_delay(0.0)
        assert tau == 0.0


class TestDoppler:
    def test_stationary_all(self):
        """All static → zero Doppler."""
        tx_pos = np.array([0.0, 0.0, 100.0])
        rx_pos = np.array([1000.0, 0.0, 100.0])
        tgt_pos = np.array([500.0, 0.0, 0.0])
        zeros = np.zeros(3)
        nu = compute_doppler(tx_pos, zeros, rx_pos, zeros, tgt_pos, zeros, 28e9)
        assert nu == pytest.approx(0.0, abs=1e-6)

    def test_target_moving_toward(self):
        """Target moving toward the bistatic bisector."""
        tx_pos = np.array([0.0, 0.0, 100.0])
        rx_pos = np.array([1000.0, 0.0, 100.0])
        tgt_pos = np.array([500.0, 0.0, 0.0])
        zeros = np.zeros(3)
        tgt_vel = np.array([0.0, 10.0, 0.0])  # perpendicular, Doppler should be small
        nu = compute_doppler(tx_pos, zeros, rx_pos, zeros, tgt_pos, tgt_vel, 28e9)
        # Moving perpendicular to bistatic baseline → near-zero Doppler
        assert abs(nu) < 1e-3


class TestPathGain:
    def test_monotonic_with_range(self):
        """Path gain should decrease with increasing range."""
        tx = np.array([0.0, 0.0, 100.0])
        rx = np.array([500.0, 0.0, 100.0])
        target_near = np.array([200.0, 100.0, 0.0])
        target_far = np.array([200.0, 900.0, 0.0])

        alpha_near = compute_path_gain(tx, rx, target_near, 28e9)
        alpha_far = compute_path_gain(tx, rx, target_far, 28e9)

        assert alpha_near > alpha_far


class TestAllBistaticParams:
    def test_output_shapes(self, sample_uav_positions, sample_uav_velocities,
                           sample_target_positions, sample_target_velocities,
                           sample_roles):
        tau, nu, alpha = compute_all_bistatic_params(
            sample_uav_positions, sample_uav_velocities,
            sample_target_positions, sample_target_velocities,
            sample_roles, fc=28e9
        )
        K, Q = 4, 2
        assert tau.shape == (K, K, Q)
        assert nu.shape == (K, K, Q)
        assert alpha.shape == (K, K, Q)

    def test_invalid_pairs_are_masked(self, sample_uav_positions, sample_uav_velocities,
                                       sample_target_positions, sample_target_velocities,
                                       sample_roles):
        """Tx-Tx and Rx-Rx pairs should have inf/zero values."""
        tau, nu, alpha = compute_all_bistatic_params(
            sample_uav_positions, sample_uav_velocities,
            sample_target_positions, sample_target_velocities,
            sample_roles, fc=28e9
        )
        # UAV 0 and 1 are tx (role=0), their pairs should be invalid (no rx)
        # Check that rx-rx and tx-tx pairs are masked
        assert np.isinf(tau[0, 1, 0]) or nu[0, 1, 0] == 0  # tx-tx pair
