"""Max-min aligned team reward utilities.

This module implements the reward-side counterpart of the fixed-structure
max-min power LP in ``uav_isac/coordination/maxmin_power.py``.  Its purpose is
to make the *learning* reward curvature-consistent with the *coordination*
objective, closing the convexity misalignment documented in
``docs/KNOWN_ISSUES.md`` (B8).

Theory
------

The deployed detection reward is

    r_log = sum_q omega_q * U(D_q),   U(D) = -log(1 - P_D(D)).

``U(D)`` is monotone increasing but **not concave** in Deflection ``D``
(``U''(D) > 0`` over ~99.6% of the operating range).  A sum of convex
per-target utilities therefore (i) has *increasing* marginal return, which
biases the learner to pour sensing power into already-strong targets
(winner-take-all), and (ii) is not aligned with the max-min QoS gate that the
system is actually evaluated on.

Two orthogonal corrections are provided.

1.  **Concave saturating utility.**  ``U_kappa(D) = 1 - exp(-kappa D)`` is
    concave in ``D`` (``U'' = -kappa^2 e^{-kappa D} < 0``).  Because
    ``D_q = sum_i a_iq p_iq`` is modular in the allocation variables, the team
    objective ``sum_q omega_q U_kappa(D_q)`` is a **monotone submodular** set
    function over additive Deflection, which restores the greedy
    ``(1 - 1/e)`` guarantee that ``-log(1-P_D)`` destroys (B8).

2.  **Dual-price weighting.**  The fixed-owner max-min power LP

        max_{p >= 0} min_q sum_i a_iq p_iq   s.t.  sum_q p_iq = b_i

    has the dual ``min_{lambda in simplex(Q)} sum_i b_i max_q lambda_q a_iq``.
    Its optimum ``lambda*`` is supported on the bottleneck targets and, by
    strong duality and complementary slackness, satisfies

        sum_q lambda*_q D*_q = min_q D*_q

    at the optimum ``D*``.  The linear reward

        r_dual = sum_q lambda*_q D_q

    is (a) **linear**, hence concave and modular, (b) a supporting
    linearisation of the max-min value function, and (c) exactly the shadow
    prices already broadcast by the distributed Dantzig--Wolfe coordinator.
    Training on ``r_dual`` therefore aligns the learner's gradient with the
    Lagrangian the coordination layer can *certify*, rather than with an
    arbitrary convex utility.

The dual price is computed from a *fixed-owner per-watt gain matrix* and a
*sensing budget*; these are team/CTDE quantities (the environment already uses
them for the centralized critic and the teacher), so no new privileged
information is introduced at execution.
"""

from __future__ import annotations

import numpy as np

from uav_isac.coordination.maxmin_power import optimal_maxmin_dual_prices


def concave_saturating_deflection_utility(
    deflection: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Concave, saturating per-target utility in Deflection space.

    ``U_kappa(D) = 1 - exp(-kappa D)``.  Monotone increasing and strictly
    concave in ``D``, so ``sum_q omega_q U_kappa(D_q)`` is monotone submodular
    over additive Deflection (restores the greedy approximation guarantee).

    Args:
        deflection: Per-target cumulative Deflection, shape (Q,).
        kappa: Saturation scale (1/Deflection).  Larger ``kappa`` saturates
            faster and is more concave; must be finite and non-negative.

    Returns:
        Utility per target in ``[0, 1)``.
    """
    values = np.asarray(deflection, dtype=np.float64)
    k = float(kappa)
    if not np.isfinite(k) or k < 0.0:
        raise ValueError("kappa must be finite and non-negative")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("deflection must be finite and non-negative")
    return -np.expm1(-k * values)


def maxmin_dual_reward(
    deflection: np.ndarray,
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    kappa: float | None = None,
) -> tuple[float, np.ndarray]:
    """Max-min aligned team reward from the fixed-owner power LP dual.

    Computes the optimal dual price ``lambda*`` of the fixed-structure max-min
    power LP (see ``optimal_maxmin_dual_prices``) and returns the
    dual-weighted reward.  With ``kappa is None`` the reward is the linear
    form ``sum_q lambda*_q D_q``; with a positive ``kappa`` the concave
    utility is applied per target first, giving
    ``sum_q lambda*_q U_kappa(D_q)`` (submodular *and* bottleneck-focused).

    Args:
        deflection: Realized per-target Deflection, shape (Q,).
        gain_per_watt: Fixed-owner per-watt gain matrix, shape (K, Q).
        sensing_budget_w: Per-UAV sensing budget, shape (K,).
        kappa: Optional saturation scale for the concave utility.

    Returns:
        ``(reward, lambda*)`` with ``lambda*`` a normalized simplex price.
    """
    d = np.asarray(deflection, dtype=np.float64).reshape(-1)
    prices, _value = optimal_maxmin_dual_prices(
        gain_per_watt, sensing_budget_w)
    if d.shape != prices.shape:
        raise ValueError("deflection must be a Q-vector matching the gain")
    if np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("deflection must be finite and non-negative")
    if kappa is None:
        terms = d
    else:
        terms = concave_saturating_deflection_utility(d, float(kappa))
    return float(np.dot(prices, terms)), prices


def softmin_bottleneck_weights(
    deflection: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Smooth bottleneck weights concentrating mass on low-deflection targets.

    ``w_q = exp(-D_q / tau) / sum_q' exp(-D_q' / tau)``.  As ``tau -> 0`` the
    weights converge to a one-hot on ``argmin_q D_q``; for ``tau > 0`` they
    provide a differentiable surrogate for the max-min support without solving
    the LP.  Provided as a cheaper fallback to the exact dual price.
    """
    values = np.asarray(deflection, dtype=np.float64).reshape(-1)
    tau = float(temperature)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("deflection must be finite and non-negative")
    shifted = values - np.min(values)
    logits = -shifted / tau
    logits -= np.max(logits)
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(values.size, 1.0 / values.size, dtype=np.float64)
    return weights / total
