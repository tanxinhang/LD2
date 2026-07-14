"""Tests for reward computation."""

import numpy as np
import pytest
from uav_isac.environment.reward import RewardComputer
from uav_isac.utils.types import DeflectionEntry


class TestTeamReward:
    def test_positive_with_detection(self):
        rc = RewardComputer(
            omega_q=np.array([0.5, 0.5]),
            P_FA=0.001,
            lambda_report=0.001,
        )
        D_q = np.array([10.0, 10.0])
        r = rc.compute_team_reward(D_q, total_bits=0, constraint_penalty=0)
        assert r > 0

    def test_negative_with_penalty(self):
        rc = RewardComputer(omega_q=np.array([0.5, 0.5]), P_FA=0.001)
        D_q = np.array([0.0, 0.0])
        r = rc.compute_team_reward(D_q, total_bits=0, constraint_penalty=1000)
        assert r < 0


class TestMarginalContributions:
    def test_delete_approximation(self):
        """Marginal contribution of a UAV ≈ utility lost when removing its edges."""
        rc = RewardComputer(omega_q=np.array([1.0]), P_FA=0.001)  # Q=1

        entries = [
            DeflectionEntry(i=0, j=2, q=0, tau=1e-6, nu=100.0, alpha=1e-6,
                           d_raw=3.0, g_dd=1.0, chi_rep=1.0, d_eff=3.0),
            DeflectionEntry(i=1, j=2, q=0, tau=1e-6, nu=100.0, alpha=1e-6,
                           d_raw=5.0, g_dd=1.0, chi_rep=1.0, d_eff=5.0),
        ]
        selected = [(0, 2, 0), (1, 2, 0)]  # both tx UAVs contribute
        uav_ids = [0, 1, 2, 3]

        marginal = rc.compute_marginal_contributions(
            uav_ids, selected, entries, Q=1
        )

        # UAV 2 is the rx, removing it removes all edges
        assert marginal[2] > marginal[0]  # rx contributes more
        assert marginal[3] == pytest.approx(0.0, abs=1e-10)  # idle UAV contributes nothing


class TestShapedRewards:
    def test_shaping_is_zero_sum(self):
        """The shaping terms should approximately sum to zero."""
        rc = RewardComputer(omega_q=np.array([0.5, 0.5]), P_FA=0.001, eta_mc=0.5)

        team_reward = 10.0
        marginal = {0: 2.0, 1: 3.0, 2: 5.0, 3: 0.0}

        shaped = rc.compute_shaped_rewards(team_reward, marginal)

        # Sum of shaped rewards
        total = sum(shaped.values())
        # K * team_reward = 4 * 10 = 40
        # sum of shaping = eta * sum(delta_k - mean(delta))
        # mean(delta) = (2+3+5+0)/4 = 2.5
        # delta_k - mean = [-0.5, 0.5, 2.5, -2.5]
        # sum shaping = 0.5 * 0 = 0
        # total = 40
        assert total == pytest.approx(4 * team_reward, rel=1e-10)

    def test_high_contributor_gets_more(self):
        """UAV with higher marginal contribution gets higher shaped reward."""
        rc = RewardComputer(omega_q=np.array([0.5, 0.5]), P_FA=0.001, eta_mc=0.5)

        team_reward = 10.0
        marginal = {0: 1.0, 1: 10.0}

        shaped = rc.compute_shaped_rewards(team_reward, marginal)

        # UAV 1 contributed more, so shaped[1] > shaped[0]
        assert shaped[1] > shaped[0]
