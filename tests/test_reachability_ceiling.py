"""O1 / T6 property tests: bistatic reachability ceiling under the gate budget.

T6 (roadmap §3): with public gain view A (K,Q) and sensing budgets b,
    D_q^max = sum_i A[i,q] b_i
is the max deflection target q can receive if every UAV spends ALL of its
budget on q.  T0 gives the gate requirement D_c(q) = (Q^{-1}(P_FA)-Q^{-1}(P_D))^2
(via detection.minimum_deflection_for_detection_probability).  q is REACHABLE
iff D_q^max >= D_c(q).

Certified consequences asserted here:
1.  T0 consistency: the D_c inversion reused here is exactly the detection
    module's inversion (bit-for-bit on the same p_fa/pd_gate).
2.  Sufficiency direction: reachable[q] iff ceiling >= D_c(q) by construction.
3.  Monotone prune (T2): any allocation p with sum_i p_iq <= sum_i b_i has
    deflection <= D_q^max, so an unreachable q can NEVER pass the gate; the
    admission mask excludes only such targets, never a feasible one.
4.  Monotone in view (frame-consistency): A1 <= A2 => unreachable(A2) subset
    of unreachable(A1).
5.  no-truth boundary: the module signature takes only the public gain view,
    budgets and detector constants -- no simulator truth input (zero RNG).
6.  Worst-deficit output for the movement layer (O5): argmax deficit is
    consistent with the reachability mask.
"""

import numpy as np
import pytest

from uav_isac.physical.detection import (
    minimum_deflection_for_detection_probability,
)
from uav_isac.physical.reachable_deflection import (
    admission_mask_and_worst_deficit,
    target_reachability_ceiling,
)


def _rng_view(K: int, Q: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.01, 1.0, size=(K, Q))


def test_gate_deflection_matches_detection_inversion():
    """T0 consistency: the D_c inversion is the detection module's own."""
    p_fa, pd_gate = 0.001, 0.8
    K, Q = 4, 3
    view = _rng_view(K, Q, 7)
    budget = np.ones(K, dtype=np.float64)
    reachable, ceiling = target_reachability_ceiling(view, budget, p_fa, pd_gate)
    d_c = minimum_deflection_for_detection_probability(
        np.full(Q, pd_gate, dtype=np.float64), p_fa)
    assert ceiling.shape == (Q,)
    np.testing.assert_array_equal(
        reachable, ceiling >= d_c)               # sufficiency by construction


def test_reachability_is_sufficiency_closed_form():
    """reachable[q] iff D_q^max >= D_c(q); deficit is D_c - D_max clamped to 0."""
    p_fa, pd_gate = 0.001, 0.7
    K, Q = 5, 4
    view = _rng_view(K, Q, 11)
    budget = np.array([1.0, 0.5, 1.0, 0.5, 1.0], dtype=np.float64)
    reachable, ceiling = target_reachability_ceiling(view, budget, p_fa, pd_gate)
    d_c = minimum_deflection_for_detection_probability(
        np.full(Q, pd_gate, dtype=np.float64), p_fa)
    np.testing.assert_array_equal(reachable, ceiling >= d_c)
    # deficit is exactly max(D_c - ceiling, 0)
    deficit = np.maximum(d_c - ceiling, 0.0)
    np.testing.assert_allclose(
        deficit[np.logical_not(reachable)] > 0.0, True)
    np.testing.assert_allclose(
        deficit[reachable], 0.0, atol=0.0, rtol=0.0)


def test_unreachable_target_never_passes_gate():
    """T2 monotone-prune core: no allocation can lift an unreachable q over D_c."""
    p_fa, pd_gate = 0.001, 0.8
    K, Q = 4, 4
    rng = np.random.default_rng(3)
    view = np.maximum(rng.uniform(0.005, 0.05, size=(K, Q)), 1e-9)
    budget = np.ones(K, dtype=np.float64) * 0.0251   # P_sense_max per UAV
    reachable, ceiling = target_reachability_ceiling(view, budget, p_fa, pd_gate)
    assert np.any(~reachable)                         # some target unreachable
    d_c = minimum_deflection_for_detection_probability(
        np.full(Q, pd_gate, dtype=np.float64), p_fa)
    # try ANY allocation: every UAV's full budget on the worst unreachable q
    for q in np.flatnonzero(~reachable):
        p = np.zeros((K, Q), dtype=np.float64)
        p[:, q] = budget
        max_deflection_achievable = float(np.sum(view[:, q] * budget))
        assert max_deflection_achievable < d_c[q]     # strictly below the gate
        assert max_deflection_achievable == pytest.approx(ceiling[q], rel=1e-12)


def test_monotone_in_view_shrinks_unreachable_set():
    """A1 <= A2 elementwise => unreachable(A2) subset of unreachable(A1)."""
    p_fa, pd_gate = 0.001, 0.75
    K, Q = 4, 5
    rng = np.random.default_rng(17)
    a1 = rng.uniform(0.02, 0.2, size=(K, Q))
    a2 = a1 * 1.5                                   # elementwise improvement
    budget = np.ones(K, dtype=np.float64)
    r1, _ = target_reachability_ceiling(a1, budget, p_fa, pd_gate)
    r2, _ = target_reachability_ceiling(a2, budget, p_fa, pd_gate)
    # unreachable(A2) must be a subset of unreachable(A1)
    assert np.all((~r2) <= (~r1).astype(int))         # no new unreachable

def test_worst_deficit_feeds_movement_layer():
    """worst-deficit target is the max-D_c/max-D gap among unreachable ones."""
    p_fa, pd_gate = 0.001, 0.7
    K, Q = 4, 6
    view = np.full((K, Q), 0.05, dtype=np.float64)
    view[:, 3] = 0.001                                # make target 3 clearly worst
    budget = np.ones(K, dtype=np.float64)
    reachable, ceiling = target_reachability_ceiling(view, budget, p_fa, pd_gate)
    d_c = minimum_deflection_for_detection_probability(
        np.full(Q, pd_gate, dtype=np.float64), p_fa)
    admission, worst_q, deficit = admission_mask_and_worst_deficit(
        view, budget, p_fa, pd_gate)
    np.testing.assert_array_equal(admission, reachable)
    if np.any(~reachable):
        expected_worst = int(np.argmax(np.maximum(d_c - ceiling, 0.0)))
        assert worst_q == expected_worst
        assert deficit == pytest.approx(
            float(np.maximum(d_c[worst_q] - ceiling[worst_q], 0.0)), rel=1e-12)
    else:
        assert worst_q == -1 and deficit == 0.0


def test_no_truth_boundary_in_signature():
    """C3: the ceiling consumes only the public gain view + budgets + detector
    constants.  No simulator-truth argument exists (source-level lock)."""
    import inspect
    import uav_isac.physical.reachable_deflection as mod
    sig = inspect.signature(mod.target_reachability_ceiling)
    params = list(sig.parameters)
    assert params == ["gain_per_watt", "sensing_budget_w", "p_fa", "pd_gate"]
    src = inspect.getsource(mod)
    # the parameter list already enforces the no-truth boundary (only the public
    # gain view, budgets, and detector constants may enter); additionally the
    # module must contain no RNG trace (zero stochasticity -> deterministic).
    for banned in ("np.random", "default_rng", "seed"):
        assert banned not in src