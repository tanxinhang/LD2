"""Tests for uav_isac/coordination/dw_column_generation.py (D1.7).

The Dantzig-Wolfe decomposition of the DC-MM inner LP is LP-exact: the
distributed column generation (local lambda-mu bids + restricted master)
must reproduce the central LP optimum at machine precision, and the local
bid must solve the pricing subproblem.  Physical guarantees: the returned
power satisfies the per-UAV 1 W budget, the covertness bound P_D^I <= eps
(hard), and the sensing floors.
"""

import numpy as np
import pytest

from uav_isac.coordination.dw_column_generation import (
    column_generation_solve,
    local_bid,
    solve_master,
)
from uav_isac.coordination.intercept_power import (
    INTERCEPT_CAPABILITIES,
    constrained_maxmin_lp,
    intercept_coefficients,
    intercept_deflection_limit,
)


def _random_case(seed, K=4, Q=4):
    rng = np.random.default_rng(seed)
    a = np.abs(rng.normal(size=(K, Q))) * 10.0 + 1.0
    aI = np.abs(rng.normal(size=(K, Q))) * 1e-2 + 1e-3
    b = rng.uniform(0.5, 0.95, size=K)
    dbar = np.full(Q, 3.27)  # medium opponent, eps=0.1 (T3 table)
    return a, aI, b, dbar


def test_column_generation_matches_central_lp():
    for seed in range(10):
        a, aI, b, dbar = _random_case(seed)
        central = constrained_maxmin_lp(a, b, aI, dbar)
        cg = column_generation_solve(a, aI, b, dbar)
        assert central is not None and cg is not None
        t_c, p_c, _l, _m, _b = central
        t_cg, p_cg, lam, mu, rounds = cg
        assert abs(t_c - t_cg) < 1e-9
        # The CG power is a feasible point of the central problem.
        assert np.all(np.sum(p_cg, axis=1) <= b + 1e-9)
        assert np.all(np.sum(aI * p_cg, axis=0) <= dbar + 1e-6)
        assert rounds <= 20


def test_cg_power_is_physically_feasible_and_covert():
    a, aI, b, dbar = _random_case(20260825)
    cg = column_generation_solve(a, aI, b, dbar)
    assert cg is not None
    _t, p, lam, mu, _r = cg
    # Per-UAV budget: sum_q p_iq <= b_i (the 1 W sensing budget share).
    assert np.all(np.sum(p, axis=1) <= b + 1e-9)
    assert np.all(p >= -1e-12)
    # Covertness hard bound.
    assert np.all(np.sum(aI * p, axis=0) <= dbar + 1e-6)
    # Prices non-negative (sensing bottleneck and opponent-detection prices).
    assert np.all(lam >= -1e-12) and np.all(mu >= -1e-12)


def test_local_bid_solves_pricing_subproblem():
    a, aI, b, dbar = _random_case(7)
    cg = column_generation_solve(a, aI, b, dbar)
    assert cg is not None
    _t, _p, lam, mu, _r = cg
    for i in range(a.shape[0]):
        s = lam * a[i] - mu * aI[i]
        assert local_bid(lam, mu, a[i], aI[i]) == int(np.argmax(s))


def test_distributed_result_matches_central_power_on_intercept_geometry():
    # End-to-end: use the real intercept geometry coefficients so the CG
    # solves the same covertness problem as the live power path.
    from config.params import load_config
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    uav = np.array([[40.0, 40.0], [60.0, 50.0], [45.0, 70.0], [80.0, 35.0]])
    tgt = np.array([[100.0, 100.0], [90.0, 120.0], [110.0, 85.0],
                    [95.0, 140.0]])
    rng = np.random.default_rng(20260826)
    K, Q = 4, 4
    a = np.abs(rng.normal(size=(K, Q))) * 20.0 + 20.0
    b = np.full(K, 0.8)
    aI = intercept_coefficients(
        uav, tgt, fc=float(cfg.otfs.fc), g_tx_dBi=float(cfg.otfs.g_tx_dBi),
        height=float(cfg.scenario.height), kt=float(cfg.channel.kT),
        theta=INTERCEPT_CAPABILITIES["medium"])
    dbar = np.full(Q, intercept_deflection_limit(1e-3, 0.1))
    central = constrained_maxmin_lp(a, b, aI, dbar)
    cg = column_generation_solve(a, aI, b, dbar)
    assert central is not None and cg is not None
    assert abs(central[0] - cg[0]) < 1e-9
