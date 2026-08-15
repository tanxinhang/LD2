"""Tests for D1.1-B+ dual-pruned multi-candidate L3 (2026-08-16).

Theory: for the fixed-owner max-min power LP, weak duality bounds the
candidate geometry's max-min deflection from above by
    U_lambda(g') = sum_i b_i max_q lambda*_q a'_iq
for any feasible simplex price lambda* (e.g. the current frame's optimal
dual).  P_D is strictly monotone in deflection, so
U_lambda(g') <= best_deflection proves the candidate cannot beat the
incumbent and its exact LP evaluation is provably dominated.

These tests verify (1) the weak-duality bound numerically, (2) that enabling
the pruning never changes the selected candidate, and (3) that it reduces
exact LP evaluations.
"""

import numpy as np
import pytest

from config.params import load_config
from uav_isac.coordination.maxmin_power import (
    fixed_owner_gain_matrix,
    optimal_maxmin_dual_prices,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.environment.env_wrapper import UAVISACEnv


def _small_env():
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    cfg.scenario.K = 2
    cfg.scenario.Q = 2
    cfg.scenario.T = 10
    cfg.target.omega_q = [0.5, 0.5]
    env = UAVISACEnv(cfg, seed=7)
    env.reset()
    return env


def _synthetic_inputs(rng):
    """Deterministic synthetic (coefficient, selected, budget, uav, tgt)."""
    K, Q = 2, 2
    coefficient = np.abs(rng.normal(size=(K, K, Q))) + 0.5
    selected = ((0, 1, 0), (1, 0, 1))  # tx, rx, target (valid half-duplex)
    budget = np.array([0.75, 0.75])
    uav = np.array([[100.0, 100.0], [150.0, 130.0]])
    tgt = np.array([[200.0, 200.0], [180.0, 260.0]])
    return coefficient, selected, budget, uav, tgt


def test_weak_duality_upper_bound_dominates_maxmin():
    rng = np.random.default_rng(20260816)
    for _ in range(20):
        K, Q = 3, 3
        gain = np.abs(rng.normal(size=(K, Q))) + 0.1
        budget = rng.uniform(0.2, 0.9, size=K)
        lam, _ = optimal_maxmin_dual_prices(gain, budget)
        res = solve_fixed_structure_maxmin_power_lp(gain, budget)
        # Weak duality: U_lambda >= t* for the same geometry.
        u_lambda = float(np.sum(budget * np.max(lam[None, :] * gain, axis=1)))
        assert u_lambda >= res.worst_deflection - 1e-9
        # Optimal dual value equals t* (strong duality) at the current geometry.
        _, dual_value = optimal_maxmin_dual_prices(gain, budget)
        assert dual_value == pytest.approx(res.worst_deflection, abs=1e-9)


def test_dual_pruning_never_changes_selected_candidate():
    env = _small_env()
    rng = np.random.default_rng(20260816)
    coefficient, selected, budget, uav, tgt = _synthetic_inputs(rng)
    gain_cur, _ = fixed_owner_gain_matrix(coefficient, selected)
    lam, _ = optimal_maxmin_dual_prices(gain_cur, budget)
    grads = {
        0: np.array([1.0, 0.5]),
        1: np.array([-0.3, 0.9]),
    }

    results = {}
    for prune in (True, False):
        env.core.cfg.marl.analytical_movement_dual_prune = bool(prune)
        out = env.core._select_best_movement_candidate(
            coefficient, selected, budget, uav, tgt, grads, step=2.5)
        results[prune] = out
        assert lam is not None

    # The chosen candidate must be bitwise identical with/without pruning.
    for k in results[False]:
        np.testing.assert_array_equal(
            results[True].get(k, np.zeros(2)),
            results[False].get(k, np.zeros(2)),
            err_msg=f"pruned selection differs at UAV {k}")


def test_dual_pruning_reduces_lp_evaluations(monkeypatch):
    env = _small_env()
    rng = np.random.default_rng(20260816)
    coefficient, selected, budget, uav, tgt = _synthetic_inputs(rng)
    grads = {
        0: np.array([1.0, 0.5]),
        1: np.array([-0.3, 0.9]),
    }

    real_solve = solve_fixed_structure_maxmin_power_lp
    counters = {"calls": 0}

    def counting_solve(*args, **kwargs):
        counters["calls"] += 1
        return real_solve(*args, **kwargs)

    counts = {}
    for prune in (True, False):
        env.core.cfg.marl.analytical_movement_dual_prune = bool(prune)
        counters["calls"] = 0
        monkeypatch.setattr(
            "uav_isac.coordination.maxmin_power."
            "solve_fixed_structure_maxmin_power_lp", counting_solve)
        env.core._select_best_movement_candidate(
            coefficient, selected, budget, uav, tgt, grads, step=2.5)
        counts[prune] = counters["calls"]

    # Pruning must not increase the LP budget; on this synthetic geometry the
    # gradient and radial candidates are usually dominated, so strictly fewer
    # LP evaluations are expected in practice.
    assert counts[True] <= counts[False]
    assert counts[True] < counts[False] or counts[True] == 1
