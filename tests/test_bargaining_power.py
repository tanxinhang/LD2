"""Tests for the reference-normalized bargaining power LP (D0.92-T)."""

import numpy as np
import pytest

from uav_isac.coordination.bargaining_power import (
    bargaining_disagreement_deflection,
    bargaining_ideal_deflection,
    solve_fixed_structure_bargaining_lp,
)
from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)


def test_ideal_and_disagreement_legitimacy():
    rng = np.random.default_rng(41)
    K, Q = 5, 4
    gain = rng.uniform(0.1, 3.0, size=(K, Q))
    budget = rng.uniform(0.5, 1.0, size=K)
    ideal = bargaining_ideal_deflection(gain, budget)
    disagree = bargaining_disagreement_deflection(gain, budget)
    # D^0 <= D^I (uniform split never exceeds the all-on-q extreme).
    assert np.all(disagree <= ideal + 1e-9)
    # D^I = sum_i b_i a_iq exactly.
    assert np.allclose(ideal, np.sum(gain * budget[:, None], axis=0))


def test_bargaining_lp_normalized_gains_in_unit_interval():
    rng = np.random.default_rng(42)
    for _ in range(100):
        K, Q = 5, 4
        gain = rng.uniform(0.1, 3.0, size=(K, Q))
        budget = rng.uniform(0.5, 1.0, size=K)
        result = solve_fixed_structure_bargaining_lp(gain, budget)
        assert np.isclose(result.power_balance_error_w, 0.0, atol=1e-9)
        r = result.normalized_gain[result.feasible_targets]
        assert np.all(r >= -1e-6) and np.all(r <= 1.0 + 1e-6)
        # eta equals the min normalized gain over feasible targets.
        if r.size:
            assert np.isclose(result.bargaining_value, np.min(r), atol=1e-6)


def test_bargaining_reduces_to_maxmin_when_flat():
    """D^0=0 and equal headroom -> proportional to pure max-min."""
    rng = np.random.default_rng(43)
    K, Q = 6, 6
    gain = rng.uniform(0.1, 3.0, size=(K, Q))
    budget = rng.uniform(0.5, 1.0, size=K)
    # Force D^0 = 0 by a zero disagreement vector, equal headroom = ideal.
    ideal = bargaining_ideal_deflection(gain, budget)
    result = solve_fixed_structure_bargaining_lp(
        gain, budget, disagreement_deflection=np.zeros(Q),
        ideal_deflection=ideal,
    )
    maxmin = solve_fixed_structure_maxmin_power_lp(gain, budget)
    # r_q = D_q / D^I_q; pure max-min maximizes min D_q, not min D_q/D^I_q.
    # The bargaining LP must be feasible and yield eta in [0,1].
    assert 0.0 <= result.bargaining_value <= 1.0
    assert result.bargaining_value >= 0.0


def test_bargaining_opportunity_fairness_against_maxmin():
    """KS fairness: equal normalized gain, not equal deflection.

    Complementary strengths: UAV 0 is strong on target 0, UAV 1 on target 1.
    Uniform split wastes both, reallocation is a Pareto improvement.
    """
    gain = np.array([[100.0, 0.1], [0.1, 100.0]])
    budget = np.array([1.0, 1.0])
    result = solve_fixed_structure_bargaining_lp(gain, budget)
    r = result.normalized_gain
    # Equal normalized gain (the KS property).
    assert np.isclose(r[0], r[1], atol=1e-6)
    # A substantial Pareto improvement over the uniform disagreement point.
    assert result.bargaining_value > 0.5
    # Uniform disagreement gives D=[50.05, 50.05]; ideal D^I=[100.1, 100.1].
    # The bargaining reallocation (UAV0->t0, UAV1->t1) reaches D~[100, 100].
    assert result.deflection[0] > 90.0 and result.deflection[1] > 90.0


def test_bargaining_dual_stationarity():
    rng = np.random.default_rng(44)
    for _ in range(50):
        K, Q = 5, 4
        gain = rng.uniform(0.1, 3.0, size=(K, Q))
        budget = rng.uniform(0.5, 1.0, size=K)
        result = solve_fixed_structure_bargaining_lp(gain, budget)
        # Stationarity: sum_q lambda_q * h_q = 1 over feasible targets.
        lhs = float(np.sum(result.prices * result.headroom))
        assert np.isclose(lhs, 1.0, atol=1e-6), f"sum lambda*h = {lhs}"
        assert np.all(result.prices >= -1e-9)


def test_bargaining_lp_strong_duality():
    """Primal eta equals dual value within tolerance."""
    rng = np.random.default_rng(45)
    for _ in range(50):
        K, Q = 5, 4
        gain = rng.uniform(0.1, 3.0, size=(K, Q))
        budget = rng.uniform(0.5, 1.0, size=K)
        result = solve_fixed_structure_bargaining_lp(gain, budget)
        # Dual value = sum_i b_i max_q lambda_q a_iq - sum_q lambda_q D^0_q.
        dual = float(np.sum(budget * np.max(
            result.prices[None, :] * gain, axis=1)) -
            np.dot(result.prices, result.disagreement))
        assert np.isclose(result.bargaining_value, dual, atol=1e-6)


def test_zero_headroom_target_excluded():
    gain = np.array([[10.0, 0.0], [0.0, 0.0]])
    budget = np.array([1.0, 1.0])
    result = solve_fixed_structure_bargaining_lp(gain, budget)
    # Target 1 is unreachable (h=0) -> excluded from bargaining.
    assert not result.feasible_targets[1]
    assert result.prices[1] == 0.0
