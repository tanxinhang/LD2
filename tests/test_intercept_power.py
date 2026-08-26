"""Tests for uav_isac/coordination/intercept_power.py (2026-08-16).

The covertness constraint D_q^I = sum_i a^I[i,q] p_iq <= bar D^I must be a
hard LP constraint (P_{D,w}^I <= eps), the opponent-capability tiers must
be physically ordered (a stronger opponent sees more), and the constrained
max-min LP must reduce to the plain max-min when the bounds do not bind.
"""

import numpy as np
import pytest

from uav_isac.coordination.intercept_power import (
    INTERCEPT_CAPABILITIES,
    constrained_maxmin_lp,
    intercept_coefficients,
    intercept_deflection_limit,
)
from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)


def _basic_geometry():
    uav = np.array([[50.0, 50.0], [80.0, 60.0], [110.0, 70.0]])
    tgt = np.array([[200.0, 200.0], [180.0, 250.0], [220.0, 230.0]])
    return uav, tgt


def test_intercept_coefficients_ordered_by_opponent_capability():
    uav, tgt = _basic_geometry()
    values = {}
    for tier in ("weak", "medium", "strong"):
        a = intercept_coefficients(
            uav, tgt, fc=2.8e10, g_tx_dBi=16.0, height=20.0,
            kt=4.0e-21, theta=INTERCEPT_CAPABILITIES[tier])
        values[tier] = float(np.max(a))
    # Physically: strong >> medium >> weak (15 orders of magnitude span).
    assert values["strong"] > values["medium"] > values["weak"]
    assert values["strong"] / max(values["weak"], 1e-300) > 1e10


def test_intercept_deflection_limit_monotone_in_eps():
    # P_D^I <= eps <=> sqrt(D) <= Q^-1(P_FA) - Q^-1(eps); eps smaller (stricter
    # covertness) means Q^-1(eps) larger, so the allowed Deflection upper
    # bound bar D^I is SMALLER.  loose = 9.55, tight = 0.58 for P_FA=1e-3.
    loose = intercept_deflection_limit(1e-3, 0.5)
    tight = intercept_deflection_limit(1e-3, 0.01)
    assert loose > tight > 0.0
    assert loose == pytest.approx(9.5495, abs=1e-3)
    assert tight == pytest.approx(0.5835, abs=1e-3)


def test_constrained_maxmin_lp_enforces_covertness_bound():
    rng = np.random.default_rng(7)
    K, Q = 3, 3
    gain = np.abs(rng.normal(size=(K, Q))) + 0.5
    budget = np.full(K, 0.8)
    aI = intercept_coefficients(
        np.array([[50.0, 50.0], [80.0, 60.0], [110.0, 70.0]]),
        np.array([[200.0, 200.0], [180.0, 250.0], [220.0, 230.0]]),
        fc=2.8e10, g_tx_dBi=16.0, height=20.0, kt=4.0e-21,
        theta=INTERCEPT_CAPABILITIES["medium"])
    d_bar = intercept_deflection_limit(1e-3, 0.1)
    ub = np.full(Q, d_bar)
    out = constrained_maxmin_lp(gain, budget, aI, ub)
    assert out is not None
    _t, p, lam, beta, mu = out
    # Hard constraint: D^I_q <= bar D^I for every q (with solver tolerance).
    d_intercept = np.sum(aI * p, axis=0)
    assert np.all(d_intercept <= d_bar + 1e-6)
    # Power budgets respected and non-negative.
    assert np.all(p >= -1e-12)
    assert np.all(np.sum(p, axis=1) <= budget + 1e-9)
    # Dual prices: lambda/beta/mu are non-negative prices.
    assert np.all(lam >= -1e-12) and np.all(beta >= -1e-12)
    assert np.all(mu >= -1e-12)


def test_binding_covertness_constraint_can_require_power_underuse():
    """The RF budget is a cap; a hard exposure bound may preclude equality."""
    gain = np.ones((1, 1), dtype=np.float64)
    budget = np.ones(1, dtype=np.float64)
    coefficient = np.ones((1, 1), dtype=np.float64)
    out = constrained_maxmin_lp(
        gain, budget, coefficient, np.asarray([0.2]))
    assert out is not None
    _t, power, _lam, _beta, _mu = out
    assert power[0, 0] == pytest.approx(0.2, abs=1e-9)
    assert float(np.sum(power[0])) < float(budget[0])


def test_constrained_maxmin_lp_reduces_to_pure_maxmin_when_bounds_do_not_bind():
    rng = np.random.default_rng(11)
    K, Q = 3, 3
    gain = np.abs(rng.normal(size=(K, Q))) + 0.5
    budget = np.full(K, 0.8)
    res = solve_fixed_structure_maxmin_power_lp(gain, budget)
    # Huge bounds -> no binding constraint -> identical optimum.
    out = constrained_maxmin_lp(
        gain, budget, np.zeros((K, Q)), np.full(Q, 1e12))
    assert out is not None
    assert out[0] == pytest.approx(res.worst_deflection, abs=1e-6)


def test_tighter_covertness_costs_qos():
    rng = np.random.default_rng(13)
    K, Q = 3, 3
    gain = np.abs(rng.normal(size=(K, Q))) + 0.5
    budget = np.full(K, 0.8)
    aI = np.abs(rng.normal(size=(K, Q))) * 1e-3 + 1e-4
    loose = constrained_maxmin_lp(
        gain, budget, aI, np.full(Q, 1e9))  # effectively unconstrained
    tight = constrained_maxmin_lp(
        gain, budget, aI, np.full(Q, 1e-2))  # strong covertness
    assert loose is not None and tight is not None
    # A tighter covertness bound can only reduce (or keep) the max-min worst.
    assert tight[0] <= loose[0] + 1e-6
