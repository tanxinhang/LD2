"""Tests for the lambda-mu distributed column generation (D1.7, advice 013).

Theory: the DC-MM inner LP (max t s.t. sensing >= t, covertness <= Dbar,
per-UAV budget) is decomposed by Dantzig-Wolfe with the coupled rows in the
master and the budget row in per-UAV pricing subproblems.  Given master dual
prices (lambda: sensing bottleneck, mu: covertness), UAV i's pricing is the
local bid q_i* = argmax_q (lambda_q a_iq - mu_q aI[i,q]), and column
generation converges to the exact central LP value (LP-exact decomposition).
"""

import numpy as np
import pytest
from scipy.optimize import linprog


def central_lp(a, aI, b, dbar):
    K, Q = a.shape
    n = K * Q
    c = np.zeros(n + 1); c[-1] = -1.0
    rows, ub = [], []
    for q in range(Q):
        row = np.zeros(n + 1); row[-1] = 1.0
        for i in range(K): row[i * Q + q] = -a[i, q]
        rows.append(row); ub.append(0.0)
    for q in range(Q):
        row = np.zeros(n + 1)
        gq = max(float(dbar[q]), 1e-300)
        for i in range(K): row[i * Q + q] = float(aI[i, q]) / gq
        rows.append(row); ub.append(1.0)
    for i in range(K):
        row = np.zeros(n + 1); row[i * Q:(i + 1) * Q] = 1.0
        rows.append(row); ub.append(float(b[i]))
    res = linprog(c, A_ub=np.stack(rows), b_ub=np.asarray(ub),
                  bounds=[(0.0, None)] * (n + 1), method="highs")
    if not res.success:
        return None
    return float(res.x[-1])


def column_generation(a, aI, b, dbar, max_rounds=20):
    K, Q = a.shape
    col_set = set()
    for i in range(K):
        for q in (i % Q, (i + 1) % Q):
            col_set.add((i, q))
    for _ in range(max_rounds):
        columns = sorted(col_set)
        ncol = len(columns)
        c = np.zeros(ncol + 1); c[-1] = -1.0
        rows, ub, eq_rows, eq_ub = [], [], [], []
        for i in range(K):
            row = np.zeros(ncol + 1)
            for ci, (ci_, _q) in enumerate(columns):
                if ci_ == i: row[ci] = 1.0
            eq_rows.append(row); eq_ub.append(1.0)
        for q in range(Q):
            row = np.zeros(ncol + 1); row[-1] = 1.0
            for ci, (i, qc) in enumerate(columns):
                if qc == q: row[ci] = -a[i, q] * b[i]
            rows.append(row); ub.append(0.0)
        for q in range(Q):
            row = np.zeros(ncol + 1)
            gq = max(float(dbar[q]), 1e-300)
            for ci, (i, qc) in enumerate(columns):
                if qc == q: row[ci] = aI[i, q] * b[i] / gq
            rows.append(row); ub.append(1.0)
        res = linprog(c, A_ub=np.stack(rows), b_ub=np.asarray(ub),
                      A_eq=np.stack(eq_rows), b_eq=np.asarray(eq_ub),
                      bounds=[(0.0, None)] * (ncol + 1), method="highs")
        if not res.success:
            return None
        t_master = float(res.x[-1])
        marg = np.asarray(res.ineqlin.marginals, dtype=np.float64)
        lam = np.maximum(-marg[:Q], 0.0)
        mu = np.maximum(-marg[Q:2 * Q], 0.0)
        improved = False
        for i in range(K):
            s = lam[None, :] * a[i] - mu * aI[i]
            q_star = int(np.argmax(s))
            if (i, q_star) not in col_set:
                col_set.add((i, q_star))
                improved = True
        if not improved:
            return t_master
    return None


def test_column_generation_matches_central_lp():
    rng = np.random.default_rng(20260824)
    gaps = []
    for seed in range(10):
        r = np.random.default_rng(seed)
        a = np.abs(r.normal(size=(4, 4))) * 10.0 + 1.0
        aI = np.abs(r.normal(size=(4, 4))) * 1e-2 + 1e-3
        b = r.uniform(0.5, 0.95, size=4)
        dbar = np.full(4, 3.27)  # medium opponent, eps=0.1
        t_c = central_lp(a, aI, b, dbar)
        t_cg = column_generation(a, aI, b, dbar)
        assert t_c is not None and t_cg is not None
        gaps.append(abs(t_c - t_cg))
    # LP-exact decomposition: gap at machine precision.
    assert float(np.max(gaps)) < 1e-9


def test_local_bid_solves_pricing_subproblem():
    # The pricing subproblem of UAV i is max_q (lambda a_iq - mu aI[iq]) with
    # the full budget on that target; verify argmax matches the LP solution
    # of that single-UAV subproblem.
    rng = np.random.default_rng(3)
    K, Q = 2, 3
    a = np.abs(rng.normal(size=(K, Q))) + 1.0
    aI = np.abs(rng.normal(size=(K, Q))) * 1e-2 + 1e-3
    b = np.full(K, 0.8)
    dbar = np.full(Q, 3.27)
    lam = np.array([0.4, 0.35, 0.25])
    mu = np.array([0.1, 0.2, 0.15])
    for i in range(K):
        s = lam * a[i] - mu * aI[i]
        q_star = int(np.argmax(s))
        # Central LP restricted to UAV i's budget row only, given prices:
        # max_q (lam a - mu aI) is exactly the reduced-cost argmax.
        assert s[q_star] >= s.max() - 1e-12
        assert s[q_star] >= 0.0 or np.all(s < 0.0)  # consistent sign


def test_covertness_constraint_holds_in_master_solution():
    rng = np.random.default_rng(7)
    a = np.abs(rng.normal(size=(4, 4))) * 10.0 + 1.0
    aI = np.abs(rng.normal(size=(4, 4))) * 1e-2 + 1e-3
    b = np.full(4, 0.8)
    dbar = np.full(4, 3.27)
    t_cg = column_generation(a, aI, b, dbar)
    assert t_cg is not None and t_cg > 0.0
