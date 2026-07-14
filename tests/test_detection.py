"""Tests for detection probability computation."""

import numpy as np
import pytest
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    compute_target_utilities,
    compute_weighted_utility,
    compute_team_reward,
)
from uav_isac.utils.math_utils import compute_PD, Q_function


class TestPDProperties:
    def test_pd_approaches_pfa_when_d_zero(self):
        """P_D → P_FA as D_q → 0."""
        P_FA = 0.001
        D_q = np.array([0.0])
        P_D = compute_PD(D_q, P_FA)
        assert P_D[0] == pytest.approx(P_FA, rel=0.1)

    def test_pd_approaches_one_when_d_large(self):
        """P_D → 1 as D_q → ∞."""
        P_FA = 0.001
        D_q = np.array([1e6])
        P_D = compute_PD(D_q, P_FA)
        assert P_D[0] > 0.9999

    def test_pd_monotonic_in_d(self):
        """P_D strictly increases with D_q."""
        P_FA = 0.001
        D_small = np.array([1.0])
        D_large = np.array([10.0])
        pd_small = compute_PD(D_small, P_FA)
        pd_large = compute_PD(D_large, P_FA)
        assert pd_large[0] > pd_small[0]

    def test_pd_in_unit_interval(self):
        """P_D is always in [0, 1]."""
        P_FA = 0.001
        for d in [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            pd = compute_PD(np.array([d]), P_FA)
            assert 0.0 <= pd[0] <= 1.0, f"P_D={pd[0]} for D={d}"


class TestDetectionProbabilities:
    def test_output_shape(self):
        D_q = np.array([5.0, 10.0])
        P_D = compute_detection_probabilities(D_q, P_FA=0.001)
        assert P_D.shape == (2,)
        assert np.all(P_D >= 0)
        assert np.all(P_D <= 1)

    def test_multi_target_ordering(self):
        """Target with higher Deflection has higher P_D."""
        D_q = np.array([5.0, 50.0])
        P_D = compute_detection_probabilities(D_q, P_FA=0.001)
        assert P_D[1] > P_D[0]


class TestTargetUtilities:
    def test_utility_monotonic(self):
        """Utility increases with D_q."""
        D_small = np.array([1.0])
        D_large = np.array([10.0])
        U_small = compute_target_utilities(D_small, P_FA=0.001)
        U_large = compute_target_utilities(D_large, P_FA=0.001)
        assert U_large[0] > U_small[0]

    def test_utility_positive(self):
        """Utility is always positive."""
        D_q = np.array([0.0, 5.0, 50.0])
        U = compute_target_utilities(D_q, P_FA=0.001)
        assert np.all(U >= 0)


class TestTeamReward:
    def test_reward_decreases_with_communication(self):
        """Higher communication cost → lower team reward."""
        D_q = np.array([10.0, 10.0])
        omega = np.array([0.5, 0.5])
        r_low_bits = compute_team_reward(D_q, 0.001, omega, total_bits=0, lambda_report=0.01)
        r_high_bits = compute_team_reward(D_q, 0.001, omega, total_bits=500, lambda_report=0.01)
        assert r_high_bits < r_low_bits
