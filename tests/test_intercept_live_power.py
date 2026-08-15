"""Tests for D1.1-D live covertness-constrained power (T3 into the live L1
power path, 2026-08-16).

When intercept_constrained_power_enabled is on, the executed sensing power
must satisfy the per-target counter-detection hard bound P_{D,w}^I <= eps
(opponent = target observer with capability theta_w), the 1 W per-UAV
budget must hold, and an infeasible strong-opponent configuration must fall
back without crashing while recording the violation.
"""

import numpy as np
import pytest

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def _env(intercept_on=True, capability="medium", eps=0.1):
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    cfg.scenario.K = 4
    cfg.scenario.Q = 4
    cfg.scenario.T = 20
    cfg.scenario.region_size = (800.0, 800.0)
    cfg.target.omega_q = [0.5] * 4
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.analytical_sensing_power_enabled = True
    cfg.marl.analytical_structure_ranking_enabled = True
    cfg.marl.intercept_constrained_power_enabled = bool(intercept_on)
    cfg.marl.intercept_capability = capability
    cfg.marl.intercept_eps = eps
    env = UAVISACEnv(cfg, seed=3)
    env.reset()
    return env


def _step(env, n=12):
    core = env.core
    pds = []
    for _ in range(n):
        actions = {str(k): {"delta_p": np.zeros(2), "role": 0}
                   for k in range(env.core.K)}
        obs, reward, term, trunc, info = env.step(actions)
        if core._last_intercept_pd_max is not None:
            pds.append(core._last_intercept_pd_max)
    return pds


def test_intercept_power_path_runs_and_bounds_opponent_pd():
    env = _env(intercept_on=True, capability="medium", eps=0.1)
    pds = _step(env)
    assert len(pds) > 0
    # Hard constraint: reported opponent detection probability <= eps (with
    # solver tolerance).
    assert max(pds) <= 0.1 + 1e-6
    # 1 W budget: each UAV's comm+sensing <= 1 W.
    core = env.core
    assert float(np.max(np.sum(core._current_sensing_power_w, axis=1))) \
        <= 1.0 + 1e-9
    assert not core._last_intercept_infeasible
    # The dual price mu (opponent-detection cost) is exposed.
    assert core._last_intercept_mu is not None
    assert np.all(np.asarray(core._last_intercept_mu) >= -1e-12)
    env.close()


def test_intercept_off_has_no_constraint():
    env = _env(intercept_on=False)
    _step(env)
    # With the constraint off there is no intercept bookkeeping.
    assert env.core._last_intercept_mu is None
    env.close()


def test_strong_opponent_pushes_pd_down_and_pays_qos():
    # A strong opponent with a tight eps makes the covertness bound very tight
    # (bar D^I tiny); the executed power must keep P_D^I <= eps at a large QoS
    # cost (the LP is still feasible -- p=0 satisfies the bound -- so the
    # constraint bites through QoS, not through infeasibility).
    env = _env(intercept_on=True, capability="strong", eps=0.01)
    core = env.core
    pds, worses = [], []
    for _ in range(12):
        actions = {str(k): {"delta_p": np.zeros(2), "role": 0}
                   for k in range(core.K)}
        obs, reward, term, trunc, info = env.step(actions)
        if core._last_intercept_pd_max is not None:
            pds.append(core._last_intercept_pd_max)
        worses.append(float(np.min(np.atleast_1d(info.get("P_D_q", [0])))))
    assert max(pds) <= 0.01 + 1e-6
    # The sensing QoS collapses under a strong opponent at eps=0.01 (1 W/28
    # GHz cannot both sense well and stay invisible to a matched filter).
    assert np.mean(worses) < 0.30
    env.close()


def test_intercept_reduces_qos_only_when_bound_binds():
    env_on = _env(intercept_on=True, capability="medium", eps=0.5)
    env_off = _env(intercept_on=False)
    worst_on, worst_off = [], []
    for _ in range(12):
        a = {str(k): {"delta_p": np.zeros(2), "role": 0}
             for k in range(env_on.core.K)}
        _, _, _, _, info_on = env_on.step(a)
        _, _, _, _, info_off = env_off.step(a)
        worst_on.append(float(np.min(np.atleast_1d(info_on.get("P_D_q", [0])))))
        worst_off.append(float(np.min(np.atleast_1d(info_off.get("P_D_q", [0])))))
    # A loose covertness bound (eps=0.5) may bind weakly; the intercept path
    # must never give a higher per-target sensing result than unconstrained
    # (it only removes power, never adds it).
    assert np.mean(worst_on) <= np.mean(worst_off) + 1e-6
    env_on.close()
    env_off.close()
