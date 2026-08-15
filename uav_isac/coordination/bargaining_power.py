"""Reference-normalized bargaining power allocation (Kalai-Smorodinsky).

Upgrades the fixed-owner max-min power LP to a normalized opportunity-fairness
objective.  Instead of ``max min_q D_q`` (which implicitly assumes every target
has equal difficulty and equal reachable ceiling), the bargaining objective is

    max_{p, eta}  eta
    s.t.  D_q(p) >= D_q^0 + eta * (D_q^I - D_q^0)   for every q,

where
    D_q^I = max_{p in F} D_q(p) = sum_i b_i a_iq    (ideal point, same feasible set)
    D_q^0 = sum_i (b_i/Q) a_iq                       (disagreement point, uniform split)

``F`` is the fixed-structure RF feasible set ``{p >= 0 : sum_q p_iq = b_i}``.
The normalized gain ``r_q = (D_q - D_q^0)/(D_q^I - D_q^0)`` is the fraction of
target q's *achievable headroom* actually delivered, so the objective is
"give every target the same fraction of its reachable improvement opportunity".

Theory
------
The problem is a **linear program**: ``D_q(p) = sum_i a_iq p_iq`` is linear, so
``D_q(p) - eta * h_q >= D_q^0`` is linear in ``(p, eta)``.  It is feasible
(``eta = 0`` with the uniform split) and bounded (``eta <= 1`` because
``r_q <= 1``), so strong duality holds and the full Dantzig-Wolfe / staleness /
quantization machinery of the max-min LP carries over unchanged.

The dual is

    min_{lambda, mu}  sum_i b_i mu_i - sum_q lambda_q D_q^0
    s.t.  mu_i >= lambda_q a_iq (all i, q),
          sum_q lambda_q h_q = 1,  lambda >= 0,  mu >= 0,

where ``h_q = D_q^I - D_q^0``.  The stationarity condition
``sum_q lambda_q h_q = 1`` means the price ``lambda_q`` prices "which target's
*unrecovered headroom* is most scarce", not "which target is currently weakest".
This removes the ad-hoc ``lambda + mu`` counterbalance of the reserve-max-min
formulation: the fairness term and the scarcity term come from one primal.

Degeneration: when ``D^0 = 0`` and all ``h_q`` are equal, the objective is
``(1/h) max min_q D_q``, i.e. it reduces to pure max-min.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class BargainingPowerResult:
    power_w: np.ndarray          # (K, Q) sensing power
    deflection: np.ndarray       # (Q,) achieved Deflection
    normalized_gain: np.ndarray  # (Q,) r_q in [0,1]
    bargaining_value: float      # eta* (common normalized gain)
    prices: np.ndarray           # (Q,) bargaining dual lambda (sum lambda*h = 1)
    headroom: np.ndarray         # (Q,) h_q = D^I - D^0
    ideal: np.ndarray            # (Q,) D^I
    disagreement: np.ndarray     # (Q,) D^0
    power_balance_error_w: float
    feasible_targets: np.ndarray  # (Q,) bool, targets with h_q > eps


def _validated(gain: np.ndarray, budget: np.ndarray):
    gain = np.asarray(gain, dtype=np.float64)
    budget = np.asarray(budget, dtype=np.float64).reshape(-1)
    if gain.ndim != 2 or gain.shape[0] != budget.size or gain.shape[1] < 1:
        raise ValueError("gain must have shape (K,Q) matching budget")
    if np.any(~np.isfinite(gain)) or np.any(gain < 0.0):
        raise ValueError("gain must be finite and non-negative")
    if np.any(~np.isfinite(budget)) or np.any(budget < 0.0):
        raise ValueError("budget must be finite and non-negative")
    return gain, budget


def bargaining_ideal_deflection(gain: np.ndarray, budget: np.ndarray) -> np.ndarray:
    """Ideal point D^I_q = max_{p in F} D_q(p) = sum_i b_i a_iq (same feasible set)."""
    g, b = _validated(gain, budget)
    return np.sum(g * b[:, None], axis=0)


def bargaining_disagreement_deflection(
    gain: np.ndarray, budget: np.ndarray,
) -> np.ndarray:
    """Disagreement point D^0_q = sum_i (b_i/Q) a_iq (uniform budget split).

    Achievable (the uniform allocation is feasible) and computable without any
    oracle; it does not depend on the current policy or future geometry.
    """
    g, b = _validated(gain, budget)
    Q = g.shape[1]
    uniform_power = b[:, None] / max(Q, 1)
    return np.sum(g * uniform_power, axis=0)


def solve_fixed_structure_bargaining_lp(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    disagreement_deflection: np.ndarray | None = None,
    ideal_deflection: np.ndarray | None = None,
    *,
    headroom_eps: float = 1e-9,
) -> BargainingPowerResult:
    """Solve the fixed-owner normalized bargaining power LP.

    Args:
        gain_per_watt: Fixed-owner per-watt gain matrix, shape (K, Q).
        sensing_budget_w: Per-UAV sensing budget, shape (K,).
        disagreement_deflection: Optional D^0 (defaults to the uniform split).
        ideal_deflection: Optional D^I (defaults to the same-feasible-set max).
        headroom_eps: Targets with h_q <= eps are treated as zero-headroom and
            excluded from the bargaining (they cannot be improved).

    Returns:
        BargainingPowerResult with the optimal power, the common normalized
        gain ``eta``, the per-target normalized gains, and the dual price.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    ideal = (
        bargaining_ideal_deflection(gain, budget)
        if ideal_deflection is None
        else np.asarray(ideal_deflection, dtype=np.float64).reshape(-1)
    )
    disagreement = (
        bargaining_disagreement_deflection(gain, budget)
        if disagreement_deflection is None
        else np.asarray(disagreement_deflection, dtype=np.float64).reshape(-1)
    )
    if ideal.shape != (Q,) or disagreement.shape != (Q,):
        raise ValueError("ideal/disagreement must be Q-vectors")
    if np.any(~np.isfinite(ideal)) or np.any(~np.isfinite(disagreement)):
        raise ValueError("ideal/disagreement must be finite")
    headroom = ideal - disagreement
    feasible = headroom > float(headroom_eps)

    variables = K * Q + 1  # p (K*Q) + eta (1)
    objective = np.zeros(variables, dtype=np.float64)
    objective[-1] = -1.0  # maximize eta

    rows = []
    upper = []
    # Bargaining constraint: sum_i a_iq p_iq - eta*h_q >= D^0_q
    #  ->  -sum_i a_iq p_iq + eta*h_q <= -D^0_q
    for q in range(Q):
        if not feasible[q]:
            continue
        row = np.zeros(variables, dtype=np.float64)
        for i in range(K):
            row[i * Q + q] = -gain[i, q]
        row[-1] = headroom[q]
        rows.append(row)
        upper.append(-disagreement[q])
    # Per-UAV budget equality.
    equality = np.zeros((K, variables), dtype=np.float64)
    for i in range(K):
        equality[i, i * Q:(i + 1) * Q] = 1.0
    bounds = [(0.0, float(budget[i])) for i in range(K) for _ in range(Q)] + [
        (0.0, 1.0)
    ]
    solved = linprog(
        objective,
        A_ub=np.stack(rows) if rows else None,
        b_ub=np.asarray(upper, dtype=np.float64) if rows else None,
        A_eq=equality,
        b_eq=budget,
        bounds=bounds,
        method="highs",
    )
    if not solved.success or solved.x is None:
        raise RuntimeError(f"bargaining power LP failed: {solved.message}")
    power = np.maximum(solved.x[:K * Q].reshape(K, Q), 0.0)
    # Restore the exact per-UAV budget (numeric slack from the solver).
    for i in range(K):
        slack = budget[i] - np.sum(power[i])
        if abs(slack) > 1e-12:
            target = int(np.argmax(gain[i]))
            power[i, target] += slack
    deflection = np.sum(gain * power, axis=0)
    eta = float(np.clip(solved.x[-1], 0.0, 1.0))
    normalized = np.where(
        feasible,
        np.divide(
            deflection - disagreement,
            headroom,
            out=np.zeros(Q, dtype=np.float64),
            where=feasible,
        ),
        0.0,
    )
    prices = _bargaining_dual_prices(
        gain, budget, disagreement, headroom, feasible)
    return BargainingPowerResult(
        power_w=power,
        deflection=deflection,
        normalized_gain=np.clip(normalized, 0.0, None),
        bargaining_value=eta,
        prices=prices,
        headroom=headroom,
        ideal=ideal,
        disagreement=disagreement,
        power_balance_error_w=float(np.max(np.abs(
            np.sum(power, axis=1) - budget))),
        feasible_targets=feasible,
    )


def _bargaining_dual_prices(
    gain: np.ndarray,
    budget: np.ndarray,
    disagreement: np.ndarray,
    headroom: np.ndarray,
    feasible: np.ndarray,
) -> np.ndarray:
    """Solve the bargaining dual for the optimal price lambda.

    min_{lambda, mu}  sum_i b_i mu_i - sum_q lambda_q D^0_q
    s.t.  lambda_q a_iq - mu_i <= 0  (all i,q),
          sum_q lambda_q h_q = 1,  lambda >= 0,  mu >= 0.
    """
    K, Q = gain.shape
    variables = K + Q  # mu (K) then lambda (Q)
    objective = np.zeros(variables, dtype=np.float64)
    objective[:K] = budget
    objective[K:] = -disagreement

    rows = []
    upper = []
    for i in range(K):
        for q in range(Q):
            row = np.zeros(variables, dtype=np.float64)
            row[i] = -1.0
            row[K + q] = float(gain[i, q])
            rows.append(row)
            upper.append(0.0)
    equality = np.zeros((1, variables), dtype=np.float64)
    equality[0, K:] = np.where(feasible, headroom, 0.0)
    solved = linprog(
        objective,
        A_ub=np.stack(rows),
        b_ub=np.asarray(upper, dtype=np.float64),
        A_eq=equality,
        b_eq=np.ones(1, dtype=np.float64),
        bounds=[(0.0, None)] * variables,
        method="highs",
    )
    if not solved.success or solved.x is None:
        # Numerical degenerate fallback: uniform over feasible targets.
        prices = np.where(feasible, 1.0, 0.0).astype(np.float64)
        total = float(np.sum(prices * headroom))
        return prices / total if total > 0.0 else prices
    lam = np.maximum(solved.x[K:], 0.0)
    lam[~feasible] = 0.0
    return lam
