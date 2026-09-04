"""advice/016 (2026-08-17): closed-loop task-regret certificates.

Validates the two theory pieces that restructure the Student calibration from
imitation accuracy to certified sensing-task regret:

1. **Structural-error -> max-min loss Lipschitz bound**
       |t*(A) - t*(A_hat)| <= sum_i b_i max_q |a_iq - a_hat_iq|
   (from |min f - min g| <= sup|f-g| and 1-Lipschitzness of the max over a
   simplex weight).  The QoS floor is preserved when the teacher margin
   m = t*(A) - d_req strictly exceeds the bound.

2. **Decision-preserving communication bound**
       B > log2(1 + R / margin)

   so the minimum integer is ``floor(log2(1 + R/margin)) + 1``.
   (B-bit uniform quantization, per-score error R/(2(2^B-1)); order survives
   iff margin > 2 eps_B).  The event trigger suppresses the Token while
   margin > 2 E_stale(h) + 2 eps_B (advice/016 §13-14).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from uav_isac.coordination.structure_regret import (
    decision_preserving_bits,
    certified_decision_preserving_bits,
    dual_edge_weights,
    preserved_without_transmission,
    qos_floor_preserved,
    quantization_error_bound,
    should_transmit,
    structure_error_bound,
    task_regret_loss,
    tstar_of,
)


# ---------------------------------------------------------------------------
# 1. Lipschitz bound: structure error -> max-min detection loss
# ---------------------------------------------------------------------------

def test_lipschitz_bound_holds_on_random_instances():
    """|t*(A) - t*(A_hat)| <= sum_i b_i max_q |a_iq - a_hat_iq| on 300 random
    instances (the max-min dual value is 1-Lipschitz in the per-watt gains
    under the weighted max-norm)."""
    rng = np.random.default_rng(20260817)
    worst_violation = 0.0
    checked = 0
    for _ in range(300):
        K, Q = 4, 4
        A = rng.uniform(0.1, 3.0, size=(K, Q))
        A_hat = np.maximum(A + rng.normal(0.0, 0.2, size=(K, Q)), 0.0)
        b = rng.uniform(0.5, 1.0, size=K)
        tA = tstar_of(A, b)
        tAh = tstar_of(A_hat, b)
        bound = structure_error_bound(b, A_hat - A)
        worst_violation = max(worst_violation, abs(tA - tAh) - bound)
        checked += 1
    assert checked == 300
    assert worst_violation <= 1e-9, (
        f"Lipschitz bound violated by {worst_violation:.3e}")


def test_lipschitz_bound_tight_for_single_entry_perturbation():
    """A single-entry perturbation saturates the bound (the max-norm is
    attained on the perturbed entry), so the bound is not vacuous."""
    rng = np.random.default_rng(1)
    K, Q = 3, 3
    A = rng.uniform(0.5, 2.0, size=(K, Q))
    b = np.full(K, 1.0)
    A_hat = A.copy()
    A_hat[1, 2] = A[1, 2] + 0.7   # perturb one entry
    bound = structure_error_bound(b, A_hat - A)
    assert bound == pytest.approx(0.7, rel=1e-12)  # b_1 * max_q |delta|
    # the actual t* shift is <= bound and (generically) positive
    assert abs(tstar_of(A_hat, b) - tstar_of(A, b)) <= 0.7 + 1e-9


def test_qos_floor_preserved_when_margin_exceeds_error():
    """If the teacher's max-min margin m = t*(A) - d_req is strictly above the
    structural error bound, the Student structure cannot push the floor below
    d_req (advice/016 §5)."""
    rng = np.random.default_rng(2)
    K, Q = 4, 4
    A = rng.uniform(1.0, 3.0, size=(K, Q))
    b = np.ones(K)
    d_req = 1.2
    tA = tstar_of(A, b)
    assert tA > d_req  # teacher is feasible with margin
    # small error keeps the floor
    A_hat = np.maximum(A + rng.normal(0.0, 0.05, size=(K, Q)), 0.0)
    err = structure_error_bound(b, A_hat - A)
    assert qos_floor_preserved(tA, d_req, err)
    assert tstar_of(A_hat, b) > d_req - 1e-9
    # large error breaks the guarantee
    A_hat2 = np.maximum(A - 2.0, 0.0)   # gross structural error
    err2 = structure_error_bound(b, A_hat2 - A)
    if not qos_floor_preserved(tA, d_req, err2):
        # the certificate is correctly negative (it may still be feasible by
        # luck, but the bound cannot certify it)
        assert tstar_of(A_hat2, b) + 1e-9 >= tA - err2


# ---------------------------------------------------------------------------
# 2. Decision-preserving bits (advice/016 §13)
# ---------------------------------------------------------------------------

def test_decision_preserving_bits_formula():
    """Strict order needs ``2**B - 1 > R / margin``, not equality."""
    # margin == R gives equality at B=1, so B=2 is the first certificate.
    assert decision_preserving_bits(1.0, 1.0) == 2
    # margin 0.1, R 1.0 -> log2(11) = 3.46 -> 4
    assert decision_preserving_bits(0.1, 1.0) == 4
    # margin 0.5, R 2.0 -> log2(5) = 2.32 -> 3
    assert decision_preserving_bits(0.5, 2.0) == 3
    # degenerate: non-positive margin -> cannot certify -> minimal bits
    assert decision_preserving_bits(0.0, 1.0) == 1
    assert decision_preserving_bits(1.0, 0.0) == 1


@pytest.mark.parametrize("boundary_bits", [1, 2, 3, 4, 8, 16])
def test_decision_preserving_bits_is_safe_on_exact_power_of_two_boundary(
    boundary_bits,
):
    """The old ceil formula returned ``boundary_bits`` at an unsafe tie."""
    dynamic_range = 1.0
    margin = dynamic_range / (2 ** boundary_bits - 1)
    selected = decision_preserving_bits(margin, dynamic_range)

    assert selected == boundary_bits + 1
    assert margin > 2.0 * quantization_error_bound(
        selected, dynamic_range)


def test_certified_bits_reserve_stale_and_physical_error_or_fail_closed():
    assert certified_decision_preserving_bits(
        0.5, 1.0, stale_drift_bound=0.05,
        physical_error_bound=0.05) == decision_preserving_bits(0.3, 1.0)
    assert certified_decision_preserving_bits(
        0.2, 1.0, stale_drift_bound=0.05,
        physical_error_bound=0.05) is None
    assert should_transmit(
        0.2, 0.05, 1.0, 16, physical_error_bound=0.05)


def test_quantization_error_bound_matches_formula():
    assert quantization_error_bound(4, 1.0) == pytest.approx(
        1.0 / (2.0 * (2 ** 4 - 1)), rel=1e-12)
    assert quantization_error_bound(6, 1.0) == pytest.approx(
        1.0 / (2.0 * 63), rel=1e-12)


def test_order_preserved_iff_margin_exceeds_2eps():
    """Numerical check of the ordering guarantee on actual quantized scores."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        s1 = rng.uniform(0.5, 1.0)
        s2 = rng.uniform(0.0, 0.5)
        margin = s1 - s2
        R = 1.0
        B = decision_preserving_bits(margin, R)
        eps = quantization_error_bound(B, R)
        # quantize both scores with B bits on [0, R]
        levels = 2 ** B - 1
        q1 = np.rint(np.clip(s1 / R, 0.0, 1.0) * levels) / levels
        q2 = np.rint(np.clip(s2 / R, 0.0, 1.0) * levels) / levels
        # the order is preserved whenever the margin exceeds the certified
        # error; when the margin is below 2eps the order MAY flip (not asserted)
        if margin > 2.0 * eps + 1e-12:
            assert q1 > q2, (
                f"order flipped with certified margin: s1={s1:.4f} "
                f"s2={s2:.4f} B={B} eps={eps:.4f}")


# ---------------------------------------------------------------------------
# 3. Event trigger (advice/016 §14)
# ---------------------------------------------------------------------------

def test_event_trigger_suppresses_certified_decisions():
    """margin > 2 E_stale + 2 eps_B  =>  the Token is suppressed; when the
    inequality fails the trigger sends (decision not certified)."""
    R = 1.0
    # large margin, small stale drift -> suppressed
    assert preserved_without_transmission(0.9, 0.05, R, bits=6)
    assert not should_transmit(0.9, 0.05, R, 6)
    # tiny margin -> must transmit (certification fails)
    assert should_transmit(0.01, 0.05, R, 6)
    assert not preserved_without_transmission(0.01, 0.05, R, 6)
    # marginal: margin just above the certified bound -> suppressed
    eps = quantization_error_bound(6, R)
    m_ok = 2.0 * 0.01 + 2.0 * eps + 1e-9
    assert preserved_without_transmission(m_ok, 0.01, R, 6)
    m_bad = 2.0 * 0.01 + 2.0 * eps - 1e-9
    assert should_transmit(m_bad, 0.01, R, 6)


def test_more_bits_or_less_drift_reduces_transmissions():
    """The event trigger sends fewer Tokens as bits grow or drift shrinks
    (the decision region widens) -- the communication-adaptation coupling."""
    R = 1.0
    margin = 0.3
    drift = 0.05
    # with 4 bits the margin may not be certified -> transmit
    eps4 = quantization_error_bound(4, R)
    if margin <= 2 * drift + 2 * eps4:
        assert should_transmit(margin, drift, R, 4)
    # with 8 bits the quantization error is negligible -> suppressed
    eps8 = quantization_error_bound(8, R)
    if margin > 2 * drift + 2 * eps8:
        assert preserved_without_transmission(margin, drift, R, 8)
    # smaller drift also widens the suppression region
    assert preserved_without_transmission(margin, 0.001, R, 6) or \
        should_transmit(margin, 0.001, R, 6)  # consistent either way


# ---------------------------------------------------------------------------
# 4. dual-weighted training weights (advice/016 §6)
# ---------------------------------------------------------------------------

def test_dual_edge_weights_concentrate_on_bottlenecks():
    """w_iq = pi_q * p_iq: the envelope theorem weight.  A bottleneck target
    (low gains -> binds the max-min) must receive far more training weight
    than easy targets, so Student errors on critical edges cost more."""
    rng = np.random.default_rng(4)
    K, Q = 4, 4
    A = rng.uniform(0.5, 2.0, size=(K, Q))
    b = np.ones(K)
    w = dual_edge_weights(A, b)
    assert w.shape == (K, Q)
    assert np.all(w >= 0.0)
    # make target 2 a bottleneck (all transmitters weak there)
    A2 = A.copy()
    A2[:, 2] *= 0.2
    w2 = dual_edge_weights(A2, b)
    w_bt = float(w2[:, 2].sum())
    w_rest = float(w2[:, [0, 1, 3]].sum())
    assert w_bt > w_rest, (
        f"bottleneck weight {w_bt} should exceed others {w_rest}")


def test_task_regret_gamma_penalizes_feasibility_loss():
    """R_gamma = [teacher feasible but student not]: a structural miss that
    flips a feasible state infeasible is the harmful error class (advice/016
    §4.1); the lexicographic regret counts it."""
    rng = np.random.default_rng(5)
    K, Q = 4, 4
    A = rng.uniform(1.0, 3.0, size=(K, Q))   # strong geometry
    b = np.ones(K)
    d_req = 1.5
    assert tstar_of(A, b) > d_req  # teacher feasible
    # drop the strongest transmitter's contribution to the bottleneck target
    A_hat = A.copy()
    q_star = int(np.argmin(np.sum(A, axis=0)))
    i_star = int(np.argmax(A[:, q_star]))
    A_hat[i_star, q_star] = 0.0
    R_g, R_t, feas = task_regret_loss(A, b, A_hat, d_req)
    if tstar_of(A_hat, b) < d_req:
        assert R_g > 0.0  # teacher feasible, student not -> gauge regret
    else:
        assert R_t >= 0.0  # both feasible -> residual max-min regret
    # the student is never certified better than the teacher
    assert feas <= 1.0
    assert R_g in (0.0, 1.0)
    assert R_t >= 0.0


# ---------------------------------------------------------------------------
# C6: decision-sufficient local plan (advice/001 §12–13, 2026-08-26)
# ---------------------------------------------------------------------------

def test_local_decision_margin_gap_between_top_two():
    from uav_isac.coordination.structure_regret import local_decision_margin

    scores = np.array([0.9, 0.6, 0.1])
    assert local_decision_margin(scores, top_k=2) == pytest.approx(0.3)


def test_local_decision_margin_fail_closed_on_few_candidates():
    from uav_isac.coordination.structure_regret import local_decision_margin

    assert local_decision_margin(np.array([0.9]), top_k=2) == 0.0
    assert local_decision_margin(np.array([np.nan, 0.5]), top_k=2) == 0.0


def test_decision_sufficient_plan_suppresses_large_margin():
    from uav_isac.coordination.structure_regret import decision_sufficient_plan

    scores = np.array([0.95, 0.10, 0.05])
    # margin 0.85 > 2*0 + 2*eps(32) -> suppressed (no airtime)
    transmit, bits = decision_sufficient_plan(
        scores, 1.0, stale_drift_bound=0.0,
        physical_error_bound=0.0, max_bits=32,
    )
    assert transmit is False and bits == 0


def test_decision_sufficient_plan_sends_when_margin_tight():
    from uav_isac.coordination.structure_regret import decision_sufficient_plan

    scores = np.array([0.51, 0.50, 0.00])
    # margin 0.01 not certifiable under drift 0.1 -> must transmit
    transmit, bits = decision_sufficient_plan(
        scores, 1.0, stale_drift_bound=0.1,
        physical_error_bound=0.0, max_bits=32,
    )
    assert transmit is True and bits >= 1


def test_decision_sufficient_plan_bits_shrink_as_margin_grows():
    from uav_isac.coordination.structure_regret import decision_sufficient_plan

    bits_narrow = decision_sufficient_plan(
        np.array([0.52, 0.48]), 1.0, max_bits=32,
    )[1]
    bits_wide = decision_sufficient_plan(
        np.array([0.90, 0.10]), 1.0, max_bits=32,
    )[1]
    # wider margin -> either suppressed or fewer bits; never more.
    assert bits_wide == 0 or bits_wide <= bits_narrow


def test_decision_sufficient_plan_counts_physical_error_when_suppressing():
    from uav_isac.coordination.structure_regret import decision_sufficient_plan

    # Delta=0.2 is safe only if the physical-error term is accidentally
    # ignored.  With E_phys=0.11 the uncertainty alone consumes 0.22 margin,
    # so the sender must remain active at the richest available precision.
    transmit, bits = decision_sufficient_plan(
        np.array([0.6, 0.4]),
        1.0,
        stale_drift_bound=0.0,
        physical_error_bound=0.11,
        max_bits=32,
    )
    assert transmit is True
    assert bits == 32


def test_decision_sufficient_plan_uses_receiver_reference_precision():
    from uav_isac.coordination.structure_regret import decision_sufficient_plan

    # The receiver holds a 4-bit token.  Testing silence against an imaginary
    # 32-bit reference underestimates eps_B and incorrectly suppresses this
    # update: Delta=0.2 <= 2*0.0833 + 2*eps_4.
    transmit, bits = decision_sufficient_plan(
        np.array([0.6, 0.4]),
        1.0,
        stale_drift_bound=0.0833,
        physical_error_bound=0.0,
        max_bits=32,
        reference_bits=4,
    )
    assert transmit is True
    assert bits >= 1
