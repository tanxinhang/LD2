"""Unit tests for the max-min aligned reward utilities."""

import numpy as np
import pytest

from uav_isac.environment.maxmin_reward import (
    concave_saturating_deflection_utility,
    maxmin_dual_reward,
    softmin_bottleneck_weights,
)
from uav_isac.coordination.maxmin_power import (
    canonical_maxmin_dual_prices,
    optimal_maxmin_dual_prices,
    solve_fixed_structure_maxmin_power_lp,
)


def test_canonical_dual_is_uniform_on_symmetric_optimal_face():
    gain = np.ones((3, 4), dtype=np.float64)
    budget = np.ones(3, dtype=np.float64)
    prices, value = canonical_maxmin_dual_prices(gain, budget)
    assert value == pytest.approx(0.75)
    np.testing.assert_allclose(prices, np.full(4, 0.25), atol=1e-7)


def test_canonical_dual_is_permutation_equivariant():
    rng = np.random.default_rng(123)
    gain = rng.uniform(0.2, 2.0, size=(4, 5))
    budget = rng.uniform(0.3, 1.0, size=4)
    permutation = np.asarray([2, 4, 0, 1, 3])
    base, value = canonical_maxmin_dual_prices(gain, budget)
    permuted, permuted_value = canonical_maxmin_dual_prices(
        gain[:, permutation], budget)
    assert permuted_value == pytest.approx(value, rel=1e-7, abs=1e-9)
    np.testing.assert_allclose(permuted, base[permutation], atol=2e-6)


def test_concave_utility_monotone_and_concave():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    u = concave_saturating_deflection_utility(d, kappa=1.0)
    assert np.all(u >= 0.0) and np.all(u < 1.0)
    assert np.all(np.diff(u) >= -1e-12), "utility must be non-decreasing"
    increments = np.diff(u)
    assert np.all(np.diff(increments) <= 1e-12), (
        "concave utility must have non-increasing increments")


def test_concave_utility_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        concave_saturating_deflection_utility(np.ones(3), kappa=-1.0)
    with pytest.raises(ValueError):
        concave_saturating_deflection_utility(np.array([-1.0, 0.0]), kappa=1.0)


def test_dual_prices_match_primal_value():
    rng = np.random.default_rng(7)
    for _ in range(20):
        K, Q = 5, 4
        gain = rng.uniform(0.0, 3.0, size=(K, Q))
        budget = rng.uniform(0.1, 1.0, size=K)
        exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
        prices, value = optimal_maxmin_dual_prices(gain, budget)
        assert np.isclose(value, exact.worst_deflection, rtol=1e-6, atol=1e-8)
        assert np.isclose(np.sum(prices), 1.0)
        assert np.all(prices >= 0.0)
        # Dual objective at the returned price must equal the primal value.
        dual_objective = float(np.sum(
            budget * np.max(prices[None, :] * gain, axis=1)))
        assert np.isclose(dual_objective, value, rtol=1e-6, atol=1e-8)


def test_dual_prices_concentrate_on_bottleneck():
    gain = np.array([[10.0, 0.0], [0.0, 1.0]])
    budget = np.array([1.0, 1.0])
    prices, value = optimal_maxmin_dual_prices(gain, budget)
    # Max-min value is min(10, 1) = 1; the bottleneck is target 1.
    assert np.isclose(value, 1.0, atol=1e-8)
    assert prices[1] > prices[0]


def test_dual_prices_degenerate_zero_gain():
    gain = np.zeros((3, 4))
    budget = np.ones(3)
    prices, value = optimal_maxmin_dual_prices(gain, budget)
    assert value == 0.0
    assert np.allclose(prices, 1.0 / 4)


def test_maxmin_dual_reward_linear_and_simplex():
    rng = np.random.default_rng(11)
    K, Q = 4, 4
    gain = rng.uniform(0.1, 2.0, size=(K, Q))
    budget = rng.uniform(0.2, 1.0, size=K)
    d = rng.uniform(0.5, 5.0, size=Q)
    reward, prices = maxmin_dual_reward(d, gain, budget)
    assert np.isclose(reward, float(np.dot(prices, d)))
    assert np.isclose(np.sum(prices), 1.0)


def test_maxmin_dual_reward_concave_variant():
    rng = np.random.default_rng(13)
    gain = rng.uniform(0.1, 2.0, size=(3, 3))
    budget = rng.uniform(0.2, 1.0, size=3)
    d = rng.uniform(0.5, 5.0, size=3)
    reward, prices = maxmin_dual_reward(d, gain, budget, kappa=1.0)
    expected = float(np.dot(
        prices, concave_saturating_deflection_utility(d, 1.0)))
    assert np.isclose(reward, expected)


def test_softmin_weights_simplex_and_concentrate():
    d = np.array([1.0, 5.0, 9.0])
    weights = softmin_bottleneck_weights(d, temperature=0.5)
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[0] > weights[1] > weights[2]


def test_tstar_reward_mode_is_worst_target_pd():
    from uav_isac.environment.reward import RewardComputer

    rc = RewardComputer(
        omega_q=np.array([0.5, 0.5]),
        P_FA=0.001,
        utility_mode="tstar",
    )
    # High deflection for target 0, low for target 1 -> tstar = min P_D.
    D_q = np.array([100.0, 1.0])
    r = rc.compute_team_reward(D_q, total_bits=0, constraint_penalty=0)
    from uav_isac.physical.detection import compute_detection_probabilities
    expected = float(np.min(compute_detection_probabilities(D_q, 0.001)))
    assert np.isclose(r, expected)


def test_tstar_reward_mode_env_integration():
    """tstar + analytical power must run in the full environment."""
    from config.params import get_default_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = get_default_config()
    cfg.scenario.K = 6
    cfg.scenario.Q = 6
    cfg.target.omega_q = [1.0 / 6.0] * 6
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.architecture_v2_enabled = False
    cfg.marl.tracking_enabled = False
    cfg.marl.analytical_sensing_power_enabled = True
    cfg.marl.analytical_sensing_power_reserve_pd = 0.6
    cfg.marl.reward_utility_mode = "tstar"

    env = UAVISACEnv(config=cfg, seed=0)
    env.reset(seed=0)
    for _ in range(5):
        actions = {
            str(k): {"delta_p": np.zeros(2), "role": 2}
            for k in range(env.K)
        }
        _, _, _, _, info = env.step(actions)
        assert np.isfinite(float(info["team_reward"]))
    env.close()


def test_maxmin_dual_reward_env_integration():
    """The maxmin_dual mode must run inside the full environment step."""
    from config.params import get_default_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = get_default_config()
    cfg.scenario.K = 6
    cfg.scenario.Q = 6
    cfg.target.omega_q = [1.0 / 6.0] * 6
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.architecture_v2_enabled = False
    cfg.marl.tracking_enabled = False
    cfg.marl.reward_utility_mode = "maxmin_dual"
    cfg.marl.reward_concave_kappa = 1.0

    env = UAVISACEnv(config=cfg, seed=0)
    env.reset(seed=0)
    for _ in range(5):
        actions = {
            str(k): {"delta_p": np.zeros(2), "role": 2}
            for k in range(env.K)
        }
        _, _, _, _, info = env.step(actions)
    assert np.isfinite(float(info["team_reward"]))
    assert float(info["team_reward"]) >= 0.0
    env.close()
