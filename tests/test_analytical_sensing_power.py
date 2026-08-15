"""Tests for D0.89 analytical inner sensing-power solver."""

import numpy as np
import pytest

from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)


def test_reserve_first_lp_enforces_reserve():
    rng = np.random.default_rng(31)
    K, Q = 5, 4
    gain = rng.uniform(0.1, 3.0, size=(K, Q))
    budget = rng.uniform(0.5, 1.0, size=K)
    # A reserve equal to half the unconstrained max-min value is guaranteed
    # feasible (every target already attains the max-min value jointly).
    base = solve_fixed_structure_maxmin_power_lp(gain, budget)
    reserve = np.full(Q, 0.5 * base.worst_deflection)
    result = solve_fixed_structure_maxmin_power_lp(
        gain, budget, minimum_deflection=reserve)
    assert result.reserve_feasible
    assert np.all(result.deflection + 1e-9 >= reserve), (
        "reserve-first LP must satisfy the per-target reserve")
    assert np.isclose(np.max(np.abs(
        np.sum(result.power_w, axis=1) - budget)), 0.0, atol=1e-12)
    # The reserve is binding, so worst cannot exceed the unconstrained max-min.
    assert result.worst_deflection <= base.worst_deflection + 1e-9


def test_reserve_first_lp_infeasible_raises():
    gain = np.array([[10.0, 0.0], [0.0, 1.0]])
    budget = np.array([1.0, 1.0])
    reserve = np.array([1.0, 100.0])  # target 1 ceiling is 1.0
    with pytest.raises(RuntimeError):
        solve_fixed_structure_maxmin_power_lp(
            gain, budget, minimum_deflection=reserve)


def test_analytical_sensing_power_env_integration():
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
    cfg.marl.analytical_sensing_power_reserve_pd = 0.0

    env = UAVISACEnv(config=cfg, seed=0)
    env.reset(seed=0)
    max_balance = 0.0
    for _ in range(5):
        actions = {
            str(k): {"delta_p": np.zeros(2), "role": 2}
            for k in range(env.K)
        }
        _, _, _, _, info = env.step(actions)
        max_balance = max(max_balance, float(env.core._last_analytical_power_balance_error))
        assert np.isfinite(float(info["team_reward"]))
    assert max_balance < 1e-12, f"power balance error {max_balance}"
    env.close()


def test_per_watt_coefficient_matches_analytic():
    """The live-env coefficient must equal the trace-audit analytic tensor."""
    from config.params import get_default_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = get_default_config()
    cfg.scenario.K = 4
    cfg.scenario.Q = 4
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.architecture_v2_enabled = False
    cfg.marl.tracking_enabled = False
    env = UAVISACEnv(config=cfg, seed=0)
    env.reset(seed=0)
    actions = {str(k): {"delta_p": np.zeros(2), "role": 2} for k in range(env.K)}
    env.step(actions)
    entries = env.core._last_deflection_entries
    coeff = env.core._per_watt_coefficient_from_entries(entries)
    assert coeff.shape == (4, 4, 4)
    assert np.all(coeff >= 0.0)
    assert np.all(np.isfinite(coeff))
    # Diagonal (i == j) must be zero: no self-pairing.
    diag = coeff[np.arange(4), np.arange(4), :]
    assert np.all(diag == 0.0)
    env.close()


def test_analytical_power_does_not_change_structure():
    """The LP override must leave motion/structure/token decisions untouched."""
    from config.params import get_default_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    def build(enabled):
        cfg = get_default_config()
        cfg.scenario.K = 4
        cfg.scenario.Q = 4
        cfg.target.omega_q = [0.25] * 4
        cfg.marl.joint_isac_power_enabled = True
        cfg.marl.architecture_v2_enabled = False
        cfg.marl.tracking_enabled = False
        cfg.marl.analytical_sensing_power_enabled = enabled
        return UAVISACEnv(config=cfg, seed=0)

    off = build(False)
    on = build(True)
    off.reset(seed=0)
    on.reset(seed=0)
    actions = {
        str(k): {"delta_p": np.zeros(2), "role": 2}
        for k in range(4)
    }
    _, _, _, _, off_info = off.step(actions)
    _, _, _, _, on_info = on.step(actions)
    # Structure (roles + owner + selected count) must be identical at frame 0:
    # the LP override runs only after P0 fixes the structure.
    assert np.array_equal(off_info["roles"], on_info["roles"])
    assert np.array_equal(
        off_info["detection_fusion_owner"], on_info["detection_fusion_owner"])
    assert off_info["n_selected"] == on_info["n_selected"]
    off.close()
    on.close()


def test_structure_ranking_requires_analytical_power():
    from config.params import get_default_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = get_default_config()
    cfg.scenario.K = 4
    cfg.scenario.Q = 4
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.analytical_structure_ranking_enabled = True  # without power flag
    with pytest.raises(ValueError):
        UAVISACEnv(config=cfg, seed=0)


def test_structure_ranking_env_integration():
    from config.params import get_default_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = get_default_config()
    cfg.scenario.K = 4
    cfg.scenario.Q = 4
    cfg.target.omega_q = [0.25] * 4
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.architecture_v2_enabled = False
    cfg.marl.tracking_enabled = False
    cfg.marl.analytical_sensing_power_enabled = True
    cfg.marl.analytical_structure_ranking_enabled = True
    env = UAVISACEnv(config=cfg, seed=0)
    env.reset(seed=0)
    for _ in range(3):
        actions = {
            str(k): {"delta_p": np.zeros(2), "role": 2}
            for k in range(env.K)
        }
        _, _, _, _, info = env.step(actions)
    # After the first LP solve, the lagged dual price must be a valid simplex.
    dual = env.core._last_analytical_dual_prices
    assert dual is not None
    assert np.isclose(np.sum(dual), 1.0)
    assert np.all(dual >= 0.0)
    assert float(env.core._last_analytical_power_balance_error) < 1e-12
    env.close()
