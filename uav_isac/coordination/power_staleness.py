"""Certified staleness bounds for the fixed-owner max-min power LP.

When the fast layer holds a max-min power allocation ``p*`` while the geometry
drifts, the realized worst-target Deflection can fall behind the value that a
fresh re-solve would deliver.  This module gives *certified* (upper) bounds on
that staleness gap that are computable without re-solving the LP, plus the DD
gate-crossing test that forces a re-solve when the support changes.

Theory
------

Fix per-UAV sensing budgets ``b_i`` and a fixed-owner per-watt gain matrix
``a_iq``.  The max-min power LP has value

    t*(A) = max_{p>=0} min_q sum_i a_iq p_iq   s.t.  sum_q p_iq = b_i,

and the equivalent dual

    t*(A) = min_{lambda in simplex(Q)} sum_i b_i max_q lambda_q a_iq.

**Value-function Lipschitz bound.**  For two gain matrices ``A, A'`` with
``Delta a_iq = a'_iq - a_iq``,

    |t*(A') - t*(A)| <= B,   B = sum_i b_i max_q |Delta a_iq|.

Proof: the dual objective ``f_A(lambda) = sum_i b_i max_q lambda_q a_iq`` is
1-Lipschitz in each ``a_iq`` coefficient weighted by ``b_i``, because
``|max_q lambda_q a'_iq - max_q lambda_q a_iq| <= max_q |Delta a_iq|`` for every
``lambda in simplex(Q)``.  Taking the min over ``lambda`` on both sides of the
dual gives the bound.  This is the classical "min of uniformly Lipschitz
functions is Lipschitz" argument and requires no differentiability.

**Held-power staleness bound.**  Let ``p*`` be optimal for ``A`` and
``t_held = min_q sum_i a'_iq p*_iq`` its realized value under the drifted gain
``A'``.  Then

    t*(A') - t_held <= 2B.

Proof: ``t*(A') - t_held = [t*(A') - t*(A)] + [t*(A) - t_held]``.  The first
term is at most ``B`` by the Lipschitz bound; the second equals
``min_q sum_i a_iq p*_iq - min_q sum_i a'_iq p*_iq``, which is at most
``max_q sum_i |Delta a_iq| p*_iq <= sum_i b_i max_q |Delta a_iq| = B`` because
``p*_iq <= b_i``.  Both bounds are one-sided, so the sum is a safe upper bound.

**Geometry-drift Lipschitz.**  The fixed-owner gain is (away from the DD gate)

    a_iq = chi_rep * alpha_iq^2 * C,   alpha_iq^2 ~ 1/(R_iq^2 R_jq^2),

with ``R_iq = |p_i - x_q|``.  First-order differentiation gives

    |Delta a_iq| <= a_iq * (2 delta_i/R_iq + 2 delta_j/R_jq
                            + 2 delta_q (1/R_iq + 1/R_jq)),

where ``delta_i, delta_j, delta_q`` bound the displacement of transmitter,
owner receiver and target.  This lets an event-trigger be formed from
displacement and distances alone, without recomputing the gain tensor.

**DD-gate discontinuity.**  The DD gate ``g_dd >= g_min`` is a *support
discontinuity*: crossing it zeroes ``a_iq``.  The Lipschitz bounds hold only
between crossings; a crossing therefore forces an immediate re-solve.
"""

from __future__ import annotations

import numpy as np


def _validated_gain_budget(
    gain: np.ndarray,
    budget: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(gain, dtype=np.float64)
    b = np.asarray(budget, dtype=np.float64).reshape(-1)
    if g.ndim != 2 or g.shape[0] != b.size or g.shape[1] < 1:
        raise ValueError("gain must have shape (K,Q) matching budget")
    if np.any(~np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("gain must be finite and non-negative")
    if np.any(~np.isfinite(b)) or np.any(b < 0.0):
        raise ValueError("budget must be finite and non-negative")
    return g, b


def maxmin_value_perturbation_bound(
    gain_old: np.ndarray,
    gain_new: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> float:
    """Certified bound on |t*(A') - t*(A)| under a gain perturbation.

    ``B = sum_i b_i max_q |a'_iq - a_iq|``.  A safe upper bound on the change
    of the max-min LP value, valid for arbitrary (non-negative) perturbations.
    """
    old, budget = _validated_gain_budget(gain_old, sensing_budget_w)
    new, _ = _validated_gain_budget(gain_new, sensing_budget_w)
    if new.shape != old.shape:
        raise ValueError("gain_old and gain_new must share shape")
    per_transmitter = np.max(np.abs(new - old), axis=1)
    return float(np.dot(budget, per_transmitter))


def held_power_staleness_bound(
    gain_old: np.ndarray,
    gain_new: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> float:
    """Certified bound on ``t*(A') - t_held`` for a held optimal power ``p*``.

    ``2 * sum_i b_i max_q |a'_iq - a_iq|``.  Safe (never under-estimates) but
    up to a factor two looser than the exact weighted duality gap.
    """
    return 2.0 * maxmin_value_perturbation_bound(
        gain_old, gain_new, sensing_budget_w)


def gain_geometry_drift_bound(
    gain: np.ndarray,
    tx_to_target_m: np.ndarray,
    owner: np.ndarray,
    rx_to_target_m: np.ndarray,
    delta_uav_m: np.ndarray,
    delta_target_m: np.ndarray,
) -> np.ndarray:
    """First-order bound on |Delta a_iq| from per-node displacement.

    ``gain[i, q]`` is the fixed-owner per-watt gain with owner receiver
    ``j = owner[q]``.  ``tx_to_target_m[i, q] = R_iq`` is the transmitter-to-
    target distance; ``rx_to_target_m[q] = R_{owner(q),q}`` is the (single)
    owner-receiver-to-target distance.  ``delta_uav_m[i]`` and
    ``delta_target_m[q]`` bound each node's displacement since the solve.  The
    bound is

        a_iq * (2 du_i/R_iq + 2 du_{owner(q)}/R_{owner(q),q}
                + 2 dt_q (1/R_iq + 1/R_{owner(q),q})).

    It is a *local* bound: it ignores the DD-gate support discontinuity and is
    valid only while ``g_dd`` does not cross ``g_min``.
    """
    g, _ = _validated_gain_budget(gain, np.ones(gain.shape[0]))
    K, Q = g.shape
    rtx = np.asarray(tx_to_target_m, dtype=np.float64)
    owner = np.asarray(owner, dtype=np.int64).reshape(-1)
    rrx = np.asarray(rx_to_target_m, dtype=np.float64).reshape(-1)
    du = np.asarray(delta_uav_m, dtype=np.float64).reshape(-1)
    dtg = np.asarray(delta_target_m, dtype=np.float64).reshape(-1)
    if rtx.shape != (K, Q):
        raise ValueError("tx distance matrix must match gain shape (K,Q)")
    if owner.shape != (Q,) or rrx.shape != (Q,):
        raise ValueError("owner and rx distance must be Q-vectors")
    if du.size != K or dtg.size != Q:
        raise ValueError("delta_uav must be K-vector, delta_target Q-vector")
    if np.any(owner < 0) or np.any(owner >= K):
        raise ValueError("owner indices must lie in [0,K)")
    if np.any(~np.isfinite(rtx)) or np.any(rtx <= 0.0):
        raise ValueError("tx distances must be finite and positive")
    if np.any(~np.isfinite(rrx)) or np.any(rrx <= 0.0):
        raise ValueError("rx distances must be finite and positive")
    if np.any(du < 0.0) or np.any(dtg < 0.0):
        raise ValueError("displacements must be non-negative")
    inv_rtx = 1.0 / rtx
    inv_rrx = 1.0 / rrx[None, :]
    du_owner = du[owner][None, :]
    return g * (
        2.0 * du[:, None] * inv_rtx
        + 2.0 * du_owner * inv_rrx
        + 2.0 * dtg[None, :] * (inv_rtx + inv_rrx)
    )


def dd_gate_crossing(
    g_dd_old: np.ndarray,
    g_dd_new: np.ndarray,
    g_min: float,
) -> np.ndarray:
    """Boolean ``(K,K,Q)`` mask of edges crossing the DD support boundary.

    A crossing means ``a_ijq`` jumps to zero (or back), i.e. the gain support
    changed and the Lipschitz staleness bounds no longer apply; a re-solve must
    be forced.  An edge crosses if it was on one side of ``g_min`` and is now
    on the other.
    """
    old = np.asarray(g_dd_old, dtype=np.float64)
    new = np.asarray(g_dd_new, dtype=np.float64)
    if old.shape != new.shape or old.ndim != 3:
        raise ValueError("g_dd tensors must share shape (K,K,Q)")
    floor = float(g_min)
    return (old >= floor) != (new >= floor)
