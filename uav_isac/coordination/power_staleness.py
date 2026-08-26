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


# ----------------------------------------------------------------------
# C4 / advice 014 (2026-08-17): local active-set theorem for the DD gate.
#
# g_dd = |sinc(l_offset) * sinc(k_offset)| with l_offset, k_offset in
# [-0.5, 0.5] the fractional OTFS delay/Doppler bin offsets, and
# |sinc'| <= 4/pi on that interval.  Within ONE frame's position-only trust
# region (velocities held), a displacement of size r changes
#   |Delta l_offset| <= M*delta_f * 2r/c
#   |Delta k_offset| <= N*T_sym * (fc/c) * r * (|v_tx|+2|v_t|+|v_rx|)/R_min
# (delay via tau = R/c, path change <= 2r; Doppler via the unit-vector
# rotation, |Delta u| <= r/R).  Hence a conservative Lipschitz constant of
# g_dd w.r.t. position displacement (velocity held) is
#   L_g = (4/pi) * [ M*delta_f*2/c
#                    + N*T_sym*(fc/c)*(|v_tx|+2|v_t|+|v_rx|)/R_min ].
# The active-set certificate is then: on every edge with
#   |g_dd,ijq - g_min| > L_g * r
# the indicator 1[g_dd >= g_min] is CONSTANT inside the radius-r trust region,
# hence a_ijq is locally smooth/Lipschitz there (the support is fixed).  Edges
# failing the condition are gate-uncertain -> the certificate fails closed:
# the gain must be re-evaluated exactly at the moved geometry (which the L3
# candidate path already does), never assumed smooth.
#
# Across frames the per-frame velocity model (v = step/dt) lets the Doppler
# alignment swing by O(1) regardless of the step size, so NO across-frame
# certificate exists: the deployment re-solves the tensor exactly every frame,
# which is precisely the fail-closed behaviour the theorem prescribes.
# ----------------------------------------------------------------------


def dd_gate_position_lipschitz_constant(
    M: int = 64,
    delta_f: float = 1.5625e4,
    T_sym: float = 6.4e-5,
    N: int = 16,
    fc: float = 2.8e10,
    c: float = 3.0e8,
    v_max: float = 25.0,
    target_v_max: float = 5.0,
    r_min: float = 100.0,
) -> float:
    """Conservative Lipschitz constant (per meter) of ``g_dd`` w.r.t. position.

    Valid for a *position-only* displacement inside one frame (velocities held,
    ``v_max``/``target_v_max`` bound the projected speeds, ``r_min`` the
    shortest link distance).  ``|g_dd(x') - g_dd(x)| <= L_g * |x' - x|``.
    Across frames (velocity = step/dt) there is no such bound: fail closed and
    re-evaluate exactly (the deployment does this every frame).
    """
    delay_coef = M * delta_f * 2.0 / c
    doppler_coef = (
        N * T_sym * (fc / c) * (2.0 * v_max + 2.0 * target_v_max) / r_min)
    return float((4.0 / np.pi) * (delay_coef + doppler_coef))


def dd_gate_active_set_certificate(
    g_dd: np.ndarray,
    g_min: float,
    r: float,
    L_g: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Local active-set certificate for the DD gate inside a radius-``r`` trust region.

    Returns ``(certified, uncertain, r_max)`` where

    - ``certified[ijq] = |g_dd,ijq - g_min| > L_g * r``: the DD support
      indicator is CONSTANT inside the region (``a_ijq`` locally smooth);
    - ``uncertain[ijq] = ~certified[ijq]``: the gate may flip -> fail-closed
      (re-evaluate exactly at the moved geometry, do not assume smoothness);
    - ``r_max = min_ijq |g_dd,ijq - g_min| / L_g``: the largest trust-region
      radius for which the current support is certified constant everywhere.

    Soundness: if ``L_g`` is a true Lipschitz constant of ``g_dd`` (see
    ``dd_gate_position_lipschitz_constant``), then a certified edge cannot
    cross ``g_min`` under any displacement of size ``<= r``, by the triangle
    inequality ``|g_dd' - g_dd| <= L_g * r < |g_dd - g_min|``.
    """
    dd = np.asarray(g_dd, dtype=np.float64)
    if dd.ndim != 3:
        raise ValueError("g_dd must have shape (K,K,Q)")
    floor = float(g_min)
    rr = float(r)
    ll = float(L_g)
    if rr < 0.0 or ll <= 0.0:
        raise ValueError("r must be non-negative and L_g strictly positive")
    margin = np.abs(dd - floor)
    certified = margin > ll * rr
    r_max = float(np.min(margin)) / ll
    return certified, ~certified, r_max
