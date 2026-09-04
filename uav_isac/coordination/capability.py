"""Task capability gauge and epoch-level capability envelope.

The capability gauge

    gamma_P*(s, xi) = min_{p, gamma}  gamma
        s.t.  D_q = sum_i a_iq p_iq,
              W(D) >= rho_min,  T_k(D) >= rho_tail,  A(D) >= rho_avg,
              sum_q p_iq <= gamma * b_i,  p >= 0,

compresses the multidimensional sensing-QoS task into a single normalized
resource-scaling certificate: gamma* <= 1 iff the fixed-structure power layer
can satisfy the FULL task set.  ``gamma* - 1`` is the fractional extra sensing
budget the structure would need.

Monotonicity
------------
``gamma*(A, b)`` is monotone non-increasing in ``(A, b)`` elementwise: if
``A^(1) <= A^(2)`` and ``b^(1) <= b^(2)`` (elementwise), then

    gamma*(A^(2), b^(2)) <= gamma*(A^(1), b^(1)).

Reason: any power allocation feasible with the smaller gain/budget remains
feasible with the larger one, so the minimum required scaling cannot increase.
This lets an epoch-level envelope be formed: with per-frame coefficient bounds
``A^- <= A_t <= A^+`` and ``b^- <= b_t <= b^+`` over a window,

    gamma^L := gamma*(A^+, b^+)  <=  gamma*_t  <=  gamma^U := gamma*(A^-, b^-).

Bottom-k in O(Q)
----------------
The ``k``-th-order-statistic (sum of k smallest) constraint is linearised with
auxiliary variables ``tau, z_q >= 0``:

    sum_{j=1}^k y_(j) = max_tau [ k*tau - sum_q (tau - y_q)_+ ],

so ``T_k >= rho_tail`` is equivalent to
``z_q >= tau - y_q``, ``z_q >= 0``, ``k*tau - sum_q z_q >= k*rho_tail``,
avoiding the C(Q, k) subset enumeration.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from uav_isac.physical.detection import (
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)


def _bottom_k_sum(pd: np.ndarray, k: int) -> float:
    return float(np.sum(np.sort(np.asarray(pd, dtype=np.float64))[:k]))


def capability_gauge(
    gain: np.ndarray,
    budget: np.ndarray,
    p_fa: float,
    xi: tuple[float, float, float, int],
) -> tuple[float, np.ndarray, np.ndarray, dict, bool] | None:
    """Solve the capability gauge and verify the true QoS constraints.

    Returns ``(gamma*, D*, P_D*, violation_dict, feasible)``, or ``None`` when
    some target's same-feasible-set ceiling is below the worst floor (the power
    layer cannot even attempt the task).
    """
    rho_min, rho_tail, rho_avg, k = xi
    gain = np.asarray(gain, dtype=np.float64)
    budget = np.asarray(budget, dtype=np.float64).reshape(-1)
    K, Q = gain.shape
    if gain.shape[0] != budget.size or Q < 1:
        raise ValueError("gain/budget shapes inconsistent")
    if np.any(~np.isfinite(gain)) or np.any(gain < 0.0):
        raise ValueError("gain must be finite and non-negative")
    if np.any(~np.isfinite(budget)) or np.any(budget < 0.0):
        raise ValueError("budget must be finite and non-negative")

    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([rho_min]), p_fa)[0])
    ceiling = np.sum(gain * budget[:, None], axis=0)
    if np.any(ceiling < d_min - 1e-9):
        return None

    n_p = K * Q

    def pd_of(x):
        p = x[:n_p].reshape(K, Q)
        return compute_detection_probabilities(np.sum(gain * p, axis=0), p_fa)

    def objective(x):
        return float(x[n_p])

    def worst_cons(x):
        return pd_of(x) - rho_min

    def avg_cons(x):
        return float(np.mean(pd_of(x))) - rho_avg

    def z_cons(x):
        return x[n_p + 2:] - (x[n_p + 1] - pd_of(x))

    def tail_cons(x):
        return k * x[n_p + 1] - float(np.sum(x[n_p + 2:])) - k * rho_tail

    def budget_cons(x):
        p = x[:n_p].reshape(K, Q)
        return x[n_p] * budget - np.sum(p, axis=1)

    p0 = np.repeat(budget, Q) / max(Q, 1)
    x0 = np.concatenate([p0, [1.0], [rho_tail], np.zeros(Q)])
    cons = [
        {"type": "ineq", "fun": worst_cons},
        {"type": "ineq", "fun": avg_cons},
        {"type": "ineq", "fun": z_cons},
        {"type": "ineq", "fun": tail_cons},
        {"type": "ineq", "fun": budget_cons},
    ]
    res = minimize(
        objective, x0, method="SLSQP", constraints=cons,
        bounds=[(0.0, None)] * (n_p + 2 + Q),
        options={"maxiter": 4000, "ftol": 1e-11},
    )
    if (
        not bool(getattr(res, "success", False))
        or getattr(res, "x", None) is None
        or np.asarray(res.x).shape != (n_p + 2 + Q,)
        or np.any(~np.isfinite(np.asarray(res.x, dtype=np.float64)))
    ):
        raise RuntimeError(
            "capability gauge solver failed; no certificate may be emitted: "
            f"{getattr(res, 'message', 'unknown optimizer failure')}"
        )
    gamma = float(res.x[n_p])
    p = res.x[:n_p].reshape(K, Q)
    d = np.sum(gain * p, axis=0)
    pd = compute_detection_probabilities(d, p_fa)
    viol = {
        "worst": float(np.min(pd) - rho_min),
        "bottom_k": _bottom_k_sum(pd, k) - k * rho_tail,
        "avg": float(np.mean(pd)) - rho_avg,
        "budget": float(np.max(np.sum(p, axis=1) - gamma * budget)),
    }
    feasible = all(v >= -1e-6 for v in viol.values()) and gamma <= 1.0 + 1e-6
    return gamma, d, pd, viol, feasible


def capability_monotone_holds(
    gain1: np.ndarray, budget1: np.ndarray,
    gain2: np.ndarray, budget2: np.ndarray,
    p_fa: float, xi: tuple[float, float, float, int],
) -> bool:
    """Check the monotonicity relation on one gain/budget pair.

    Requires ``gain1 <= gain2`` and ``budget1 <= budget2`` elementwise (the
    caller guarantees this); returns whether the gauges satisfy
    ``gamma*(A2,b2) <= gamma*(A1,b1) + tol``.
    """
    g1 = capability_gauge(gain1, budget1, p_fa, xi)
    g2 = capability_gauge(gain2, budget2, p_fa, xi)
    if g1 is None and g2 is None:
        return True
    if g1 is None or g2 is None:
        # A better system can only be at least as capable; if the better system
        # is None (target unreachable) the worse one cannot be reachable either.
        return g2 is None or (g2 is not None and g1 is None)
    return float(g2[0]) <= float(g1[0]) + 1e-6


def _capability_gauge_pwl_lp_certificate_raw(
    gain: np.ndarray,
    budget: np.ndarray,
    p_fa: float,
    xi: tuple[float, float, float, int],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    d_min: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    """Solve the PWL gauge and return raw target and budget dual marginals.

    Reformulates the gauge so ``D_q`` is an explicit variable coupled by
    ``D_q - sum_i a_iq p_iq = 0``; the multiplier of that equality is the
    target sensitivity price ``pi_q*``.  By the envelope theorem,

        d gamma* / d a_iq = -pi_q* * p_iq*,

    which is the capability-KKT analogue of the old max-min ``lambda* p`` and is
    the correct shadow price for the FULL task set (worst + bottom-k + average),
    not just the worst target.  Returns ``(gamma*, p*, pi_raw*, eta*)``;
    ``pi_raw`` retains SciPy's equality-marginal sign, while ``eta >= 0`` is
    the canonical multiplier of each normalized per-UAV budget inequality.
    """
    from scipy.optimize import linprog

    rho_min, rho_tail, rho_avg, k = xi
    gain = np.asarray(gain, dtype=np.float64)
    budget = np.asarray(budget, dtype=np.float64).reshape(-1)
    K, Q = gain.shape
    slopes = np.asarray(slopes, dtype=np.float64).reshape(-1)
    intercepts = np.asarray(intercepts, dtype=np.float64).reshape(-1)
    M = slopes.size

    # Variables: p (K*Q), gamma (1), D (Q), y (Q), tau (1), z (Q).
    n_p = K * Q
    o_gamma = n_p
    o_D = o_gamma + 1
    o_y = o_D + Q
    o_tau = o_y + Q
    o_z = o_tau + 1
    n_vars = o_z + Q

    c = np.zeros(n_vars)
    c[o_gamma] = 1.0

    # Equality: D_q - sum_i a_iq p_iq = 0.
    eq_rows = np.zeros((Q, n_vars))
    for q in range(Q):
        eq_rows[q, o_D + q] = 1.0
        for i in range(K):
            eq_rows[q, i * Q + q] = -gain[i, q]

    rows = []
    upper = []
    # Worst: -D_q <= -d_min.
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_D + q] = -1.0
        rows.append(row)
        upper.append(-d_min)
    # PWL: y_q - slope_m D_q <= intercept_m.
    for m in range(M):
        for q in range(Q):
            row = np.zeros(n_vars)
            row[o_D + q] = -slopes[m]
            row[o_y + q] = 1.0
            rows.append(row)
            upper.append(float(intercepts[m]))
    # Steady: -sum_q y_q <= -Q rho_avg.
    row = np.zeros(n_vars)
    row[o_y:o_y + Q] = -1.0
    rows.append(row)
    upper.append(-Q * rho_avg)
    # Bottom-k: -z_q + tau - y_q <= 0.
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_z + q] = -1.0
        row[o_tau] = 1.0
        row[o_y + q] = -1.0
        rows.append(row)
        upper.append(0.0)
    # Bottom-k: -k*tau + sum_q z_q <= -k*rho_tail.
    row = np.zeros(n_vars)
    row[o_tau] = -k
    row[o_z:o_z + Q] = 1.0
    rows.append(row)
    upper.append(-k * rho_tail)
    # Budget: sum_q p_iq - gamma*b_i <= 0.
    for i in range(K):
        row = np.zeros(n_vars)
        row[i * Q:(i + 1) * Q] = 1.0
        row[o_gamma] = -budget[i]
        rows.append(row)
        upper.append(0.0)

    res = linprog(
        c,
        A_ub=np.stack(rows),
        b_ub=np.asarray(upper, dtype=np.float64),
        A_eq=eq_rows,
        b_eq=np.zeros(Q, dtype=np.float64),
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not res.success or res.x is None:
        return None
    gamma = float(res.x[o_gamma])
    p = res.x[:n_p].reshape(K, Q)
    # Envelope theorem: d gamma*/d a_iq = -pi_q * p_iq with pi_q the Lagrange
    # multiplier of D_q - sum_i a_iq p_iq = 0.  scipy's ``eqlin.marginals`` uses
    # the opposite sign convention, so the *empirical* identity validated
    # against finite differences in the test suite is
    #     d gamma*/d a_iq = +marginals_q * p_iq.
    # We return the marginals as-is; callers must apply the (test-validated)
    # sign in the geometry chain rule.
    pi = np.asarray(res.eqlin.marginals, dtype=np.float64).reshape(-1)
    eta = np.maximum(
        -np.asarray(res.ineqlin.marginals[-K:], dtype=np.float64).reshape(-1),
        0.0,
    )
    if gamma > 1.0e-10:
        # KKT stationarity in gamma: 1 - sum_i eta_i b_i = 0.  Keeping this
        # executable prevents a heuristic scarcity score from being reported
        # as the capability gauge's canonical resource dual.
        stationarity = float(np.dot(eta, budget))
        if abs(stationarity - 1.0) > 1.0e-6:
            raise RuntimeError(
                "capability-gauge budget dual violates gamma stationarity")
    return gamma, p, pi, eta


def capability_gauge_pwl_lp_full(
    gain: np.ndarray,
    budget: np.ndarray,
    p_fa: float,
    xi: tuple[float, float, float, int],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    d_min: float,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Return ``(gamma*, p*, raw target equality marginal)``."""
    out = _capability_gauge_pwl_lp_certificate_raw(
        gain, budget, p_fa, xi, slopes, intercepts, d_min)
    return None if out is None else out[:3]


def capability_gauge_pwl_lp_certificate(
    gain: np.ndarray,
    budget: np.ndarray,
    p_fa: float,
    xi: tuple[float, float, float, int],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    d_min: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return a locator-safe canonical L1 certificate.

    Target price uses the positive convention
    ``pi_q=-eqlin.marginal_q``.  Thus ``pi_q*a_ijq-eta_i`` is a reduced
    capability contribution.  This certificate uses no joint-structure
    feasibility query or witness information.
    """
    out = _capability_gauge_pwl_lp_certificate_raw(
        gain, budget, p_fa, xi, slopes, intercepts, d_min)
    if out is None:
        return None
    gamma, power, raw_pi, eta = out
    return gamma, power, np.maximum(-raw_pi, 0.0), eta


def capability_gauge_pwl_lp(
    gain: np.ndarray,
    budget: np.ndarray,
    p_fa: float,
    xi: tuple[float, float, float, int],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    d_min: float,
) -> float | None:
    """Convenience wrapper returning only ``gamma*`` (see ``_full``)."""
    out = capability_gauge_pwl_lp_full(
        gain, budget, p_fa, xi, slopes, intercepts, d_min)
    return None if out is None else out[0]


def minimum_total_power_pwl_lp(
    gain: np.ndarray,
    budget: np.ndarray,
    p_fa: float,
    xi: tuple[float, float, float, int],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    d_min: float,
) -> tuple[float, np.ndarray] | None:
    """Minimum total sensing power for one fixed discrete structure.

    The hard per-UAV budgets and all three task floors are retained.  Unlike
    the capability gauge, this is the third stage of the G4 lexicographic
    objective after the discrete closure and prepare-bit stages are fixed.
    ``None`` means that this fixed structure is infeasible under the supplied
    coefficient realization and physical budgets.
    """
    from scipy.optimize import linprog

    rho_min, rho_tail, rho_avg, k = xi
    value = np.asarray(gain, dtype=np.float64)
    resource = np.asarray(budget, dtype=np.float64).reshape(-1)
    slope = np.asarray(slopes, dtype=np.float64).reshape(-1)
    intercept = np.asarray(intercepts, dtype=np.float64).reshape(-1)
    if value.ndim != 2 or value.shape[0] != resource.size:
        raise ValueError("gain/budget shapes inconsistent")
    if (np.any(~np.isfinite(value)) or np.any(value < 0.0)
            or np.any(~np.isfinite(resource)) or np.any(resource < 0.0)
            or slope.size == 0 or slope.size != intercept.size):
        raise ValueError("invalid fixed-structure power inputs")
    K, Q = value.shape
    if not 1 <= int(k) <= Q:
        raise ValueError("tail order is outside target count")

    # Variables: p(KQ), D(Q), y(Q), tau(1), z(Q).
    n_p = K * Q
    o_D = n_p
    o_y = o_D + Q
    o_tau = o_y + Q
    o_z = o_tau + 1
    n_vars = o_z + Q
    objective = np.zeros(n_vars)
    objective[:n_p] = 1.0

    equality = np.zeros((Q, n_vars))
    for q in range(Q):
        equality[q, o_D + q] = 1.0
        for i in range(K):
            equality[q, i * Q + q] = -value[i, q]

    rows = []
    upper = []
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_D + q] = -1.0
        rows.append(row)
        upper.append(-float(d_min))
    for m in range(slope.size):
        for q in range(Q):
            row = np.zeros(n_vars)
            row[o_D + q] = -slope[m]
            row[o_y + q] = 1.0
            rows.append(row)
            upper.append(float(intercept[m]))
    row = np.zeros(n_vars)
    row[o_y:o_y + Q] = -1.0
    rows.append(row)
    upper.append(-Q * float(rho_avg))
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_z + q] = -1.0
        row[o_tau] = 1.0
        row[o_y + q] = -1.0
        rows.append(row)
        upper.append(0.0)
    row = np.zeros(n_vars)
    row[o_tau] = -int(k)
    row[o_z:o_z + Q] = 1.0
    rows.append(row)
    upper.append(-int(k) * float(rho_tail))
    for i in range(K):
        row = np.zeros(n_vars)
        row[i * Q:(i + 1) * Q] = 1.0
        rows.append(row)
        upper.append(float(resource[i]))

    result = linprog(
        objective, A_ub=np.stack(rows), b_ub=np.asarray(upper),
        A_eq=equality, b_eq=np.zeros(Q),
        bounds=[(0.0, None)] * n_vars, method="highs")
    if not result.success or result.x is None:
        return None
    power = np.asarray(result.x[:n_p], dtype=np.float64).reshape(K, Q)
    return float(np.sum(power)), power


def qos_constrained_maxmin_lp(
    gain: np.ndarray,
    budget: np.ndarray,
    p_fa: float,
    xi: tuple[float, float, float, int],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    d_min: float,
    intercept_coeff: np.ndarray | None = None,
    intercept_ub: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Stage B of lexicographic QoS-constrained max-min (advice 010).

    Maximise the worst target Deflection subject to the FULL QoS floors as hard
    constraints (worst, bottom-k and steady via the PWL chord lower bound):

        max_{p, D, y, tau, z, t}  t
        s.t.  D_q = sum_i a_iq p_iq
              D_q >= t                        (max-min)
              D_q >= d_min                    (worst floor)
              y_q <= slope_m D_q + intercept_m  (P_D >= y, chord lower bound)
              sum_q y_q >= Q * steady_floor     (steady floor)
              z_q >= tau - y_q, z_q >= 0        (bottom-k linearisation)
              k*tau - sum_q z_q >= k * weak3_floor
              sum_q p_iq <= b_i, p >= 0
              [intercept: sum_i aI[i,q] p_iq <= bar D^I_q]   (optional D1.1-E)

    The optional ``intercept_coeff`` (a^I, (K,Q)) and ``intercept_ub`` (bar
    D^I, (Q,)) add the counter-detection hard constraint P_{D,w}^I <= eps to
    the SAME LP, so covertness and QoS floors are jointly enforced (one
    convex LP, one dual).  The capability gauge's own allocation is a feasible
    point of this LP, so the lexicographic optimum is no worse than the gauge
    on BOTH the floors and the max-min worst.  Returns ``(t*, p*, D*)`` or
    ``None`` when infeasible.
    """
    from scipy.optimize import linprog

    rho_min, rho_tail, rho_avg, k = xi
    gain = np.asarray(gain, dtype=np.float64)
    budget = np.asarray(budget, dtype=np.float64).reshape(-1)
    K, Q = gain.shape
    if intercept_coeff is not None:
        intercept_coeff = np.asarray(intercept_coeff, dtype=np.float64)
        intercept_ub = np.asarray(
            intercept_ub, dtype=np.float64).reshape(-1)
        if intercept_coeff.shape != (K, Q) or intercept_ub.size != Q:
            raise ValueError(
                "intercept_coeff must be (K,Q) and intercept_ub length Q")
    n_p = K * Q
    o_D = n_p
    o_y = o_D + Q
    o_tau = o_y + Q
    o_z = o_tau + 1
    o_t = o_z + Q
    n_vars = o_t + 1
    c = np.zeros(n_vars)
    c[o_t] = -1.0  # maximize t

    eq_rows = []
    eq_ub = []
    for q in range(Q):
        row = np.zeros(n_vars)
        row[o_D + q] = 1.0
        for i in range(K):
            row[i * Q + q] = -gain[i, q]
        eq_rows.append(row)
        eq_ub.append(0.0)

    rows = []
    upper = []
    for q in range(Q):  # max-min: D_q >= t
        row = np.zeros(n_vars)
        row[o_D + q] = -1.0
        row[o_t] = 1.0
        rows.append(row)
        upper.append(0.0)
    for q in range(Q):  # worst floor: D_q >= d_min
        row = np.zeros(n_vars)
        row[o_D + q] = -1.0
        rows.append(row)
        upper.append(-d_min)
    for m in range(slopes.size):  # PWL chord: y_q <= slope_m D_q + intercept_m
        for q in range(Q):
            row = np.zeros(n_vars)
            row[o_D + q] = -slopes[m]
            row[o_y + q] = 1.0
            rows.append(row)
            upper.append(float(intercepts[m]))
    row = np.zeros(n_vars)  # steady: sum y_q >= Q * rho_avg
    row[o_y:o_y + Q] = -1.0
    rows.append(row)
    upper.append(-Q * rho_avg)
    for q in range(Q):  # bottom-k: z_q - tau + y_q >= 0
        row = np.zeros(n_vars)
        row[o_z + q] = -1.0
        row[o_tau] = 1.0
        row[o_y + q] = -1.0
        rows.append(row)
        upper.append(0.0)
    row = np.zeros(n_vars)  # bottom-k: -k*tau + sum z_q <= -k*rho_tail
    row[o_tau] = -k
    row[o_z:o_z + Q] = 1.0
    rows.append(row)
    upper.append(-k * rho_tail)
    for i in range(K):  # budget: sum_q p_iq <= b_i
        row = np.zeros(n_vars)
        row[i * Q:(i + 1) * Q] = 1.0
        rows.append(row)
        upper.append(float(budget[i]))
    if intercept_coeff is not None:  # covertness: sum_i aI[i,q] p_iq <= dbar_q
        for q in range(Q):
            row = np.zeros(n_vars)
            gq = max(float(intercept_ub[q]), 1e-300)
            for i in range(K):
                row[i * Q + q] = float(intercept_coeff[i, q]) / gq
            rows.append(row)
            upper.append(1.0)

    res = linprog(
        c,
        A_ub=np.stack(rows),
        b_ub=np.asarray(upper, dtype=np.float64),
        A_eq=np.stack(eq_rows),
        b_eq=np.asarray(eq_ub, dtype=np.float64),
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not res.success or res.x is None:
        return None
    t_star = float(res.x[o_t])
    # Third lexicographic level (advice 010 §8 spirit): while the max-min worst
    # stays at its optimum t*, maximise the SUM of target Deflections, i.e. push
    # the stronger targets with the leftover capability instead of leaving every
    # target equalised at the worst level.  This raises steady/mean without
    # sacrificing worst (not a weighted blend: t* is a hard floor here).
    c2 = np.zeros(n_vars)
    c2[o_D:o_D + Q] = -1.0  # maximise sum_q D_q
    rows2 = []
    upper2 = []
    for q in range(Q):  # worst pinned at t*: D_q >= t_star
        row = np.zeros(n_vars)
        row[o_D + q] = -1.0
        rows2.append(row)
        upper2.append(-t_star)
    for q in range(Q):  # worst floor: D_q >= d_min (redundant, keep for safety)
        row = np.zeros(n_vars)
        row[o_D + q] = -1.0
        rows2.append(row)
        upper2.append(-d_min)
    for m in range(slopes.size):  # PWL chord: y_q <= slope_m D_q + intercept_m
        for q in range(Q):
            row = np.zeros(n_vars)
            row[o_D + q] = -slopes[m]
            row[o_y + q] = 1.0
            rows2.append(row)
            upper2.append(float(intercepts[m]))
    row = np.zeros(n_vars)  # steady: sum y_q >= Q * rho_avg
    row[o_y:o_y + Q] = -1.0
    rows2.append(row)
    upper2.append(-Q * rho_avg)
    for q in range(Q):  # bottom-k: z_q - tau + y_q >= 0
        row = np.zeros(n_vars)
        row[o_z + q] = -1.0
        row[o_tau] = 1.0
        row[o_y + q] = -1.0
        rows2.append(row)
        upper2.append(0.0)
    row = np.zeros(n_vars)  # bottom-k: -k*tau + sum z_q <= -k*rho_tail
    row[o_tau] = -k
    row[o_z:o_z + Q] = 1.0
    rows2.append(row)
    upper2.append(-k * rho_tail)
    for i in range(K):  # budget: sum_q p_iq <= b_i
        row = np.zeros(n_vars)
        row[i * Q:(i + 1) * Q] = 1.0
        rows2.append(row)
        upper2.append(float(budget[i]))
    if intercept_coeff is not None:  # covertness must hold at the optimum too
        for q in range(Q):
            row = np.zeros(n_vars)
            gq = max(float(intercept_ub[q]), 1e-300)
            for i in range(K):
                row[i * Q + q] = float(intercept_coeff[i, q]) / gq
            rows2.append(row)
            upper2.append(1.0)

    res2 = linprog(
        c2,
        A_ub=np.stack(rows2),
        b_ub=np.asarray(upper2, dtype=np.float64),
        A_eq=np.stack(eq_rows),
        b_eq=np.asarray(eq_ub, dtype=np.float64),
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if res2.success and res2.x is not None:
        p = res2.x[:n_p].reshape(K, Q)
    else:
        p = res.x[:n_p].reshape(K, Q)
    # Turn the budget inequalities into the exact per-UAV RF equality (slack to
    # the best-gain target), matching the max-min LP convention.  When covertness
    # rows are present (intercept_coeff), the fill must respect their headroom:
    # the naive best-gain fill could violate sum_i aI[i,q] p_iq <= dbar_q exactly
    # when the LP left slack because covertness binds (audit 2026-08-17, P0).
    p = np.maximum(p, 0.0).copy()
    if intercept_coeff is not None and intercept_ub is not None:
        c_norm = np.asarray(intercept_coeff, dtype=np.float64) / np.maximum(
            np.asarray(intercept_ub, dtype=np.float64)[None, :], 1e-300)
        for i in range(K):
            slack = float(budget[i] - np.sum(p[i]))
            while slack > 1e-10:
                e_q = np.sum(c_norm * p, axis=0)
                room = (1.0 - e_q) / np.maximum(c_norm[i, :], 1e-30)
                q_fill = int(np.argmax(room))
                if room[q_fill] <= 1e-12:
                    break
                add = min(slack, max(float(room[q_fill]), 0.0))
                if add <= 1e-12:
                    break
                p[i, q_fill] += add
                slack -= add
    else:
        for i in range(K):
            slack = float(budget[i] - np.sum(p[i]))
            if slack > 1e-10:
                p[i, int(np.argmax(gain[i]))] += slack
    d = np.sum(gain * p, axis=0)
    return float(t_star), p, d


def capability_geometry_gradient(
    gain: np.ndarray,
    owner: np.ndarray,
    uav_pos: np.ndarray,
    target_pos: np.ndarray,
    power: np.ndarray,
    prices: np.ndarray,
    *,
    height_m: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Gradient of gamma* w.r.t. UAV positions under the Friis gain model.

    The fixed-owner gain is ``a_iq = C_iq / (R_iq^2 * R_owner(q),q^2)``, so a
    UAV influences the gradient through BOTH its transmitter role and its
    receiver/owner role (advice 007 §4):

        d gamma*/d x_k = sum_{i,q} (d gamma*/d a_iq) * d a_iq/d x_k
                       = sum_{i,q} (prices_q * power_iq) *
                           [1[k=i]   * (-2 a_iq (x_i-x_q)/R_iq^2)
                          + 1[k=owner_q] * (-2 a_iq (x_owner_q-x_q)/R_owner_q^2)]

    ``height_m`` supplies the vertical endpoint--target separation when the
    positions contain horizontal coordinates only.  It may be scalar or
    broadcastable to ``(K,Q)``.  Returns a (K, dim) array.  Valid only inside
    the DD-support-stable trust region (the caller must not cross a support
    boundary).
    """
    K, Q = gain.shape
    dim = int(np.asarray(uav_pos).shape[1])
    owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    uav = np.asarray(uav_pos, dtype=np.float64)
    tgt = np.asarray(target_pos, dtype=np.float64)
    delta = uav[:, None, :] - tgt[None, :, :]
    vertical = np.asarray(height_m, dtype=np.float64)
    if np.any(~np.isfinite(vertical)):
        raise ValueError("height_m must be finite")
    if dim >= 3 and np.any(vertical != 0.0):
        raise ValueError(
            "height_m must be zero when positions already contain altitude")
    try:
        vertical_sq = np.broadcast_to(vertical * vertical, (K, Q))
    except ValueError as exc:
        raise ValueError("height_m must be scalar or broadcastable to (K,Q)") from exc
    rtx_sq = np.sum(delta * delta, axis=2) + vertical_sq
    rrx_sq = rtx_sq[owner, np.arange(Q)]
    grad = np.zeros((K, dim), dtype=np.float64)
    for i in range(K):
        for q in range(Q):
            weight = prices[q] * power[i, q]
            if abs(weight) < 1e-15:
                continue
            a = gain[i, q]
            # Tx effect (UAV i).
            grad[i] += (
                weight * (-2.0 * a) * delta[i, q]
                / max(float(rtx_sq[i, q]), 1e-9))
            # Rx/owner effect (UAV owner(q)).
            jq = owner[q]
            grad[jq] += (
                weight * (-2.0 * a) * delta[jq, q]
                / max(float(rrx_sq[q]), 1e-9))
    return grad


def local_capability_gradient_k(
    k: int,
    owner: np.ndarray,
    prices: np.ndarray,
    own_power: np.ndarray,
    own_gain: np.ndarray,
    own_uav_pos: np.ndarray,
    target_pos: np.ndarray,
    support: dict,
    *,
    height_m: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Distributed per-UAV geometry gradient (Tx + Rx/owner), info-boundary safe.

    Computes ``g_k`` from only: the broadcast price ``prices``, UAV k's own power
    row / gain row / position, the target positions (mission-known), the owner
    map, and ``support`` — the delivered transmitter records for targets whose
    owner is k.  ``support[q]`` is a dict of ``i -> (p_iq, a_iq)``; a missing
    record is treated as zero (fail-closed), never reconstructed from truth.

    This is the local decomposition whose concatenation equals the centralized
    ``capability_geometry_gradient`` under ideal communication (T2).
    """
    Q = int(np.asarray(own_gain).size)
    dim = int(np.asarray(own_uav_pos).shape[0])
    owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    prices = np.asarray(prices, dtype=np.float64).reshape(-1)
    own_power = np.asarray(own_power, dtype=np.float64).reshape(-1)
    own_gain = np.asarray(own_gain, dtype=np.float64).reshape(-1)
    uav = np.asarray(own_uav_pos, dtype=np.float64)
    tgt = np.asarray(target_pos, dtype=np.float64)
    vertical = np.asarray(height_m, dtype=np.float64)
    if np.any(~np.isfinite(vertical)):
        raise ValueError("height_m must be finite")
    if dim >= 3 and np.any(vertical != 0.0):
        raise ValueError(
            "height_m must be zero when positions already contain altitude")
    try:
        vertical_sq = np.broadcast_to(vertical * vertical, (Q,))
    except ValueError as exc:
        raise ValueError("height_m must be scalar or broadcastable to (Q,)") from exc
    delta = uav[None, :] - tgt
    range_sq = np.sum(delta * delta, axis=1) + vertical_sq
    g = np.zeros(dim, dtype=np.float64)
    # Tx effect: own gain a_{k,owner_q,q} and own distance R_{k,q}.
    for q in range(Q):
        w = prices[q] * own_power[q]
        if abs(w) < 1e-15:
            continue
        a = own_gain[q]
        jq = owner[q]
        g += (
            w * (-2.0 * a) * delta[q]
            / max(float(range_sq[q]), 1e-9))
    # Rx/owner effect: for targets q owned by k, use delivered transmitter records.
    for q in range(Q):
        if owner[q] != k:
            continue
        for i, (p_iq, a_iq) in support.get(q, {}).items():
            w = prices[q] * p_iq
            if abs(w) < 1e-15:
                continue
            g += (
                w * (-2.0 * a_iq) * delta[q]
                / max(float(range_sq[q]), 1e-9))
    return g
