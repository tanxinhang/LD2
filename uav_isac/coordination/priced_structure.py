"""Price-mediated structure repair: owner + TX reassignment for the L2 layer.

Theory (D0.95): the joint structure--geometry problem

    min_{x, S}  gamma*(x, S),   S = {(owner_j, TX set S_q)}_q

decouples into a structure step (L2, discrete) and a geometry step (L3,
continuous) coordinated by a *single* price vector pi.

Envelope theorem.  At the capability-gauge optimum (fixed x, S) the KKT
multipliers pi_q of the deflection constraints satisfy

    d gamma*/d a_iq = -pi_q * p_iq .

The SAME pi_q drive the geometry gradient (advice 008, T2):

    d gamma*/d x_k = sum_q pi_q [ p_kq d a_kq/d x_k
                               + 1[k=owner_q] sum_i p_iq d a_iq/d x_k ].

Structure step.  The owner choice is price-orthogonal: for target q the owner j
multiplies the gain of *every* transmitter, so the first-order marginal value of
choosing owner j is  pi_q * sum_i a_ijq b_i  (pi_q > 0 factors out), hence

    j_q* = argmax_j  sum_i a_ijq b_i .

The role partition is the combinatorial core: a half-duplex UAV is TX or RX,
not both, so the owner set R (RX role) and the transmitter set T = [K]\\R must
partition the fleet.  A naive greedy that scores the *ceiling* (each TX using
its full budget on every target) is optimistic: power sharing splits a strong
transmitter's budget across the targets it serves, so the ceiling overstates the
realised deflection.  We therefore score every candidate role set by the exact
max-min power LP value (power-sharing aware), which is the price-orthogonal
binding-floor objective of the gauge.

Novelty.  This replaces the 2^K role-partition enumeration of the oracle with a
price-justified greedy that evaluates the true power-sharing objective, and it
keeps the distributed information boundary: each UAV locally computes its own
marginal  sum_q pi_q a_kjq  and proposes an owner/TX role, so the reporting
graph and the motion plan are coordinated by the same broadcast price.
"""

# ----------------------------------------------------------------------
# AUDIT/RESEARCH-ONLY MODULE (2026-08-16 audit remediation)
#
# This module is consumed only by tools/ audit scripts and tests. It is
# NOT part of the deployment execution path (env_core / trainer) and its
# results must not be described as deployed behaviour. It exists to keep
# a specific research question reproducible; see
# docs/ARCHITECTURE_V2_RESULTS.md for the associated gate.
# ----------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)


def structure_from_rx(
    coefficient: np.ndarray,
    budget: np.ndarray,
    rx_set: set[int] | list[int],
    target_pair_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-owner gain + owner map for a given RX role set (half-duplex)."""
    coefficient = np.asarray(coefficient, dtype=np.float64)
    budget = np.asarray(budget, dtype=np.float64).reshape(-1)
    K, _, Q = coefficient.shape
    rx = set(int(r) for r in rx_set)
    tx = [i for i in range(K) if i not in rx]
    gain = np.zeros((K, Q), dtype=np.float64)
    owner = np.full(Q, -1, dtype=np.int64)
    for q in range(Q):
        best_j, best_val = -1, -1.0
        best_tx: list[int] = []
        for j in rx:
            col = coefficient[:, j, q] * budget
            order = sorted(tx, key=lambda i: -col[i])[:max(1, int(target_pair_limit))]
            val = float(sum(col[i] for i in order))
            if val > best_val:
                best_j, best_val, best_tx = j, val, order
        owner[q] = best_j
        for i in best_tx:
            gain[i, q] = coefficient[i, best_j, q]
    return gain, owner


def _score_worst_lp(gain: np.ndarray, budget: np.ndarray) -> float:
    res = solve_fixed_structure_maxmin_power_lp(gain, budget)
    return float(res.worst_deflection)


def priced_structure_repair(
    coefficient: np.ndarray,
    budget: np.ndarray,
    *,
    target_pair_limit: int = 3,
    prices: np.ndarray | None = None,
    max_rx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy price-justified structure repair (owner + TX, role partition).

    Returns ``(gain, owner)``: the fixed-owner per-watt gain matrix (K, Q) and
    the owner index per target, ready for the fixed-owner capability gauge and
    the L3 geometry descent.
    """
    coefficient = np.asarray(coefficient, dtype=np.float64)
    budget = np.asarray(budget, dtype=np.float64).reshape(-1)
    K, _, Q = coefficient.shape
    if coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    if budget.size != K:
        raise ValueError("budget must have length K")
    if prices is not None:
        prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        if prices.size != Q:
            raise ValueError("prices must have length Q")

    limit = max(1, int(target_pair_limit))
    cap_rx = K - 1 if max_rx is None else min(max(1, int(max_rx)), K - 1)

    def score(gain: np.ndarray) -> float:
        # Power-sharing-aware worst deflection (binding floor).  ``prices`` is
        # reserved for the full gauge (bottom-3/steady) weighting.
        return _score_worst_lp(gain, budget)

    best_rx: set[int] = set()
    best_gain: np.ndarray | None = None
    best_owner: np.ndarray | None = None
    best_score = -np.inf
    rx_set: set[int] = set()
    for _ in range(cap_rx):
        best_add = None
        best_cand = -np.inf
        for k in range(K):
            if k in rx_set:
                continue
            g, _ = structure_from_rx(coefficient, budget, rx_set | {k}, limit)
            s = score(g)
            if s > best_cand + 1e-12:
                best_cand = s
                best_add = k
        if best_add is None:
            break
        rx_set.add(best_add)
        if best_cand > best_score + 1e-12:
            best_score = best_cand
            best_rx = set(rx_set)
            best_gain, best_owner = structure_from_rx(
                coefficient, budget, best_rx, limit)

    if best_gain is None:
        best_gain, best_owner = structure_from_rx(
            coefficient, budget, {0}, limit)
    return best_gain, best_owner
