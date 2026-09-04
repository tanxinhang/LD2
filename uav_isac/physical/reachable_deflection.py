"""Bistatic reachability ceiling (O1 / T6, roadmap 2026-08-29).

T6 (reachability via the D-gate ceiling).  With the locally reconstructed
public per-watt gain view ``A`` (K,Q) -- the same view the replicated power
solver consumes, i.e. strictly inside the no-simulator-truth boundary (C3) --
and per-UAV sensing budgets ``b`` (joint RF / CPI billing), the maximum
deflection a target ``q`` can receive when EVERY UAV spends ALL of its
sensing budget on ``q`` is

    D_q^max = sum_i A[i,q] * b_i

because deflection is linear in sensing power (manifest ``D_q = sum_edges
a_ijq p_ijq``).  By T0 the detection gate ``P_D >= pd_gate`` requires

    D_c(q) = (Q^{-1}(P_FA) - Q^{-1}(P_D_gate))^2

(the inversion implemented by detection.minimum_deflection_for_detection_
probability).  Target ``q`` is REACHABLE iff ``D_q^max >= D_c(q)``.

Two consequences, both used by the O1 admission design:

1.  If ``q`` is unreachable, NO power allocation (any p >= 0 with
    ``sum_i p_iq <= sum_i b_i``) can make ``q`` pass the gate, because
    ``sum_i A[i,q] p_iq <= D_q^max < D_c(q)``.  Excluding unreachable targets
    from candidate generation is therefore a MONOTONE PRUNE (T2): it never
    removes a feasible solution of the fixed-structure gate task.

2.  The ceiling is monotone in the view: ``A1 <= A2`` elementwise implies
    ``D_q^max(A1) <= D_q^max(A2)``, so the unreachable set shrinks
    monotonically as local views improve -- the admission set is
    frame-consistent (no oscillation between included/excluded frames without
    a genuine gain improvement).

The module is stateless: inputs are the public gain view, sensing budgets and
the detector constants -- no simulator truth, no RNG (C3).
"""

from __future__ import annotations

import numpy as np

from uav_isac.physical.detection import (
    minimum_deflection_for_detection_probability,
)


def target_reachability_ceiling(
    gain_per_watt: np.ndarray,        # (K, Q) locally reconstructed public view
    sensing_budget_w: np.ndarray,     # (K,) per-UAV sensing budget (JRF cap)
    p_fa: float,
    pd_gate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(reachable, d_max)``.

    ``reachable[q]`` = True iff ``D_q^max >= D_c(q)`` (T0/T6); ``d_max`` is the
    all-budget ceiling ``sum_i A[i,q]*b_i``.  ``pd_gate`` must lie in [0,1)
    (the T0 inversion rejects probability one as unbounded).
    """
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        gain.ndim != 2 or gain.shape[0] != budget.size or gain.shape[1] < 1
        or np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError(
            "gain must be finite non-negative (K,Q); budget finite non-negative (K,)")
    ceiling = np.sum(gain * budget[:, None], axis=0)          # (Q,)
    # T0 inversion (existing implementation) -> D_c(q) per target.
    gate_deflection = minimum_deflection_for_detection_probability(
        np.full(gain.shape[1], float(pd_gate), dtype=np.float64), float(p_fa))
    reachable = ceiling >= gate_deflection
    return reachable, ceiling


def admission_mask_and_worst_deficit(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    p_fa: float,
    pd_gate: float,
) -> tuple[np.ndarray, int, float]:
    """Return ``(admission,worst_unreachable_q,worst_deficit)``.

    ``admission`` is the O1 candidate-admission mask (True = target kept),
    ``worst_unreachable_q`` the target with the largest gate deficit among
    unreachable targets (argmax of ``D_c(q) - D_q^max``; -1 if none), and
    ``worst_deficit`` that deficit (0 if all reachable).  The worst-deficit
    target is the O5 movement-layer input (most physically unreachable target).
    """
    reachable, ceiling = target_reachability_ceiling(
        gain_per_watt, sensing_budget_w, p_fa, pd_gate)
    gate_deflection = minimum_deflection_for_detection_probability(
        np.full(ceiling.size, float(pd_gate), dtype=np.float64), float(p_fa))
    deficit = np.maximum(gate_deflection - ceiling, 0.0)
    if np.any(~reachable):
        worst_q = int(np.argmax(deficit))
        return reachable, worst_q, float(deficit[worst_q])
    return reachable, -1, 0.0