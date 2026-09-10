"""Covertness-constrained power allocation primitives (advice 012 / T3).

Extracted from the T3 oracle (docs/CURRENT_SYSTEM_MODEL.md) into the
coordination layer so the detection-capability constraint can be reused by
the live power path, not only by the offline oracle audit.

Theory (advice 012): the opponent's detection is a Gaussian-shift test, so
its Deflection from UAV i's sensing power p_iq at observer q is

    D_q^I = sum_i a^I[i,q] p_iq,
    a^I[i,q] = K_w G_tx G_w (lambda/(4 pi d_3d))^2 T_int,w / (kT B_w NF_w),

and the covertness requirement P_{D,w}^I <= eps is exactly the Deflection
hard constraint

    D_q^I <= bar D^I := [Q^{-1}(P_FA^I) - Q^{-1}(eps)]^2.

The constrained max-min LP (inner of the DC-MM master problem) is convex,
and its dual exposes the unified prices (lambda*: sensing bottleneck,
beta*: local RF, mu*: opponent-detection price); the local net value of
UAV i on target q becomes s_iq = lambda_q a_iq - mu_q a^I[i,q].
"""

from __future__ import annotations

import numpy as np

from uav_isac.utils.math_utils import Q_inverse

# Opponent capability tiers (advice 012 SS1.3, bandwidth corrected 2026-08-16:
# in the 1/(kT B_w NF_w) denominator a NARROWER band is MORE sensitive).
INTERCEPT_CAPABILITIES: dict[str, dict[str, float]] = {
    "weak": dict(G_w=1.0, B_w=1e8, NF_w=12.0, T_int_w=1e-5, K_w=1.0),
    "medium": dict(G_w=3.0, B_w=1e7, NF_w=8.0, T_int_w=1e-4, K_w=2.0),
    "strong": dict(G_w=100.0, B_w=1e3, NF_w=3.0, T_int_w=1e-1, K_w=10.0),
}


def intercept_coefficients(
    uav: np.ndarray,
    tgt: np.ndarray,
    *,
    fc: float,
    g_tx_dBi: float,
    height: float,
    kt: float,
    theta: dict,
) -> np.ndarray:
    """a^I[i,q]: per-watt counter-detection Deflection gain at observer q.

    Mirror of the legitimate sensing abstraction: the observer's Deflection
    from UAV i's power at observer q.  d_3d includes the UAV height.
    """
    lam = 299_792_458.0 / max(float(fc), 1.0)
    g_tx_lin = 10.0 ** (float(g_tx_dBi) / 10.0)
    h = float(height)
    d2 = np.linalg.norm(uav[:, None, :] - tgt[None, :, :], axis=2) ** 2
    d3 = np.sqrt(d2 + h * h)
    path = (lam / (4.0 * np.pi * np.maximum(d3, 1e-6))) ** 2
    g_w = float(theta["G_w"])
    b_w = float(theta["B_w"])
    nf_lin = 10.0 ** (float(theta["NF_w"]) / 10.0)
    t_int = float(theta["T_int_w"])
    k_w = float(theta["K_w"])
    noise = max(float(kt) * b_w * nf_lin, 1e-30)
    return k_w * g_tx_lin * g_w * path * t_int / noise


def intercept_deflection_limit(p_fa_i: float, eps: float) -> float:
    """bar D^I = [Q^{-1}(P_FA^I) - Q^{-1}(eps)]^2 (advice 012 SS3)."""
    root = float(Q_inverse(np.asarray(float(p_fa_i)))) - float(
        Q_inverse(np.asarray(float(eps))))
    return root * root


def constrained_maxmin_lp(
    gain: np.ndarray,
    budget: np.ndarray,
    constraint_coeff: np.ndarray,
    constraint_ub: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Inner (I) of the DC-MM master problem: max-min with per-target hard
    constraints (exposure c with Gamma, or counter-detection a^I with bar D^I):

        max_{p,t}  t
        s.t.  sum_i a_iq p_iq >= t,  for all q
              sum_q p_iq <= b_i,     for all i
              sum_i coeff[i,q] p_iq <= ub_q,  for all q   (normalised to unit
                                                           scale for LP solver
                                                           feasibility tolerance)
              p >= 0

    Returns (t*, p*, lambda*, beta*, mu*) — the unified dual prices: lambda =
    sensing bottleneck price, beta = local UAV RF price, mu = constraint
    (exposure / counter-detection) price.  The LP is exact; the local net
    value of UAV i on target q is s_iq = lambda_q a_iq - mu_q coeff[i,q].
    The budget fill respects the remaining constraint headroom (never the raw
    best-gain target), so the constraints stay satisfied after the fill.
    """
    from scipy.optimize import linprog

    gain = np.asarray(gain, dtype=np.float64)
    budget = np.asarray(budget, dtype=np.float64).reshape(-1)
    constraint_coeff = np.asarray(constraint_coeff, dtype=np.float64)
    constraint_ub = np.asarray(constraint_ub, dtype=np.float64).reshape(-1)
    K, Q = gain.shape
    if constraint_coeff.shape != (K, Q):
        raise ValueError("constraint_coeff must have shape (K,Q)")
    if constraint_ub.size != Q:
        raise ValueError("constraint_ub must have length Q")

    n = K * Q
    c_obj = np.zeros(n + 1)
    c_obj[-1] = -1.0  # maximize t
    rows = []
    ub = []
    for q in range(Q):  # sum_i a_iq p_iq >= t  ->  -sum_i a_iq p_iq + t <= 0
        row = np.zeros(n + 1)
        row[-1] = 1.0
        for i in range(K):
            row[i * Q + q] = -gain[i, q]
        rows.append(row)
        ub.append(0.0)
    for i in range(K):  # sum_q p_iq <= b_i
        row = np.zeros(n + 1)
        row[i * Q:(i + 1) * Q] = 1.0
        rows.append(row)
        ub.append(float(budget[i]))
    for q in range(Q):  # normalised hard constraint
        row = np.zeros(n + 1)
        gq = max(float(constraint_ub[q]), 1e-300)
        for i in range(K):
            row[i * Q + q] = float(constraint_coeff[i, q]) / gq
        rows.append(row)
        ub.append(1.0)

    res = linprog(
        c_obj,
        A_ub=np.stack(rows),
        b_ub=np.asarray(ub, dtype=np.float64),
        bounds=[(0.0, None)] * (n + 1),
        method="highs",
    )
    if not res.success or res.x is None:
        return None
    p = res.x[:n].reshape(K, Q)
    t_star = float(res.x[-1])
    marg = np.asarray(res.ineqlin.marginals, dtype=np.float64)
    lam = np.maximum(-marg[:Q], 0.0)
    beta = np.maximum(-marg[Q:Q + K], 0.0)
    mu = np.maximum(-marg[Q + K:], 0.0)
    # Budget fill toward the target with the LARGEST constraint headroom.
    p = np.maximum(p, 0.0).copy()
    c_norm = constraint_coeff / np.maximum(constraint_ub[None, :], 1e-300)
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
    return t_star, p, lam, beta, mu
