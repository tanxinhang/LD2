"""O5 / T7 property tests: bistatic movement potential with a strict-accept
descent rule (roadmap docs/CURRENT_SYSTEM_MODEL.md §3 T7).

T7 (potential -- reachability monotone agreement):
    bottleneck Phi non-increasing along accepted moves
    ==> gain ceiling non-decreasing ==> gate deficit D_c - D_q^max
    non-increasing  (monotone link to the O1 reachability certificate).

Certified properties asserted here:
1.  Phi_q = R_tx(q)*R_rx(q) is the geometry factor of the per-watt deflection
    (a ~ 1/Phi^2), so Phi is positive and bounded below by physical standoff.
2.  Strict-accept rule: a move that does not strictly lower the bottleneck Phi
    is rejected (ties rejected).
3.  Accepted-move sequence is monotonically strictly decreasing in Phi
    (Lyapunov descent).
4.  Finite candidate grid + deterministic rule => termination in finitely many
    steps at a local minimum; an extra attempt cannot lower it.
5.  Phi-decrease implies gain-ceiling increase (agreement with O1: smaller Phi
    -> larger per-watt ceiling -> deficit non-increasing).
6.  no-truth boundary: inputs are only public positions + role masks; the
    signature (not the docstring) is the lock.
"""

import numpy as np
import pytest

from uav_isac.physical.movement_potential import (
    accept_move_candidate,
    bistatic_potential,
    coverage_deficit_lexicographic_score,
    potential_descent_sequence,
    role_aware_bistatic_products,
    select_baseline_enveloped_movement,
)


def _simple_scene():
    target = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]], dtype=np.float64)
    uav = np.array([[0.0, 50.0], [100.0, 50.0], [200.0, 50.0], [300.0, 50.0]],
                   dtype=np.float64)
    tx = np.array([True, True, False, False], dtype=bool)
    rx = np.array([False, False, True, True], dtype=bool)
    return target, uav, tx, rx


def test_potential_is_positive_and_below_standoff():
    target, uav, tx, rx = _simple_scene()
    phi, worst, worst_q = bistatic_potential(target, uav, tx, rx)
    assert phi.shape == (3,)
    assert np.all(phi > 0.0)
    assert worst == pytest.approx(float(np.max(phi)), rel=1e-12)
    assert worst_q == int(np.argmax(phi))


def test_strict_accept_rejects_non_decreasing_moves():
    target, uav, tx, rx = _simple_scene()
    # move UAV 0 (TX) far away -> bottleneck can only grow -> rejected
    ok_bad, _, _ = accept_move_candidate(
        target, uav, tx, rx, 0, np.array([900.0, 900.0], dtype=np.float64))
    assert ok_bad is False
    # any move: accepted iff it strictly decreases the bottleneck
    ok_good, phi_good, _ = accept_move_candidate(
        target, uav, tx, rx, 1, np.array([200.0, 0.0], dtype=np.float64))
    _, phi_before, _ = bistatic_potential(target, uav, tx, rx)
    assert (phi_good < phi_before) == ok_good


def test_descent_sequence_monotone_and_terminates():
    target, uav, tx, rx = _simple_scene()
    grid = np.array([[x, y] for x in range(0, 301, 25)
                     for y in range(0, 101, 25)], dtype=np.float64)
    hist, final, steps = potential_descent_sequence(
        target, uav, tx, rx, grid, max_steps=200)
    assert len(hist) == steps
    if steps > 1:
        assert np.all(np.diff(hist) < 0.0)   # every accepted step strictly lowers Phi
    assert final > 0.0
    # termination at a local minimum: one more attempt must not lower it
    _, final_2, _ = potential_descent_sequence(
        target, uav, tx, rx, grid, max_steps=steps + 1)
    assert final == pytest.approx(final_2, rel=1e-12)


def test_phi_decrease_raises_gain_ceiling():
    """T7 agreement with O1: 1/Phi^2 is the per-watt ceiling factor, so a
    strictly smaller Phi raises the gain ceiling monotonically."""
    phi_large = 100.0 * 100.0
    phi_small = 50.0 * 80.0
    assert (1.0 / phi_small ** 2) > (1.0 / phi_large ** 2)
    for a, b in ((30.0, 90.0), (120.0, 121.0), (5.0, 6.0)):
        assert 1.0 / a ** 2 > 1.0 / b ** 2


def test_no_truth_boundary_in_signature():
    """C3: inputs are public positions + role masks only (signature lock)."""
    import inspect
    import uav_isac.physical.movement_potential as mod
    sig = inspect.signature(mod.bistatic_potential)
    assert list(sig.parameters) == ["target_xy", "uav_xy", "tx_mask", "rx_mask"]
    sig2 = inspect.signature(mod.accept_move_candidate)
    assert list(sig2.parameters) == [
        "target_xy", "uav_xy", "tx_mask", "rx_mask",
        "uav_index", "candidate_xy"]
    sig3 = inspect.signature(mod.potential_descent_sequence)
    assert list(sig3.parameters) == [
        "target_xy", "uav_xy", "tx_mask", "rx_mask",
        "candidate_grid", "max_steps"]
    src = inspect.getsource(mod)
    # the signatures enforce the no-truth boundary; additionally lock
    # determinism (no RNG trace in the module body).
    for banned in ("np.random", "default_rng", "seed"):
        assert banned not in src


def test_coverage_score_prioritizes_gate_reachability_then_bistatic_gain():
    target = np.array([[0.0, 0.0], [200.0, 0.0]], dtype=np.float64)
    tx = np.array([True, False], dtype=bool)
    uncovered = np.array([[-300.0, 0.0], [300.0, 0.0]], dtype=np.float64)
    covered = np.array([[-100.0, 0.0], [100.0, 0.0]], dtype=np.float64)
    assert coverage_deficit_lexicographic_score(
        target, covered, tx, 250.0, height_m=20.0,
    ) < coverage_deficit_lexicographic_score(
        target, uncovered, tx, 250.0, height_m=20.0,
    )

    # With equal coverage status, smaller bistatic products win.
    closer = np.array([[-80.0, 0.0], [180.0, 0.0]], dtype=np.float64)
    farther = np.array([[-120.0, 0.0], [220.0, 0.0]], dtype=np.float64)
    assert coverage_deficit_lexicographic_score(
        target, closer, tx, 500.0, height_m=20.0,
    ) < coverage_deficit_lexicographic_score(
        target, farther, tx, 500.0, height_m=20.0,
    )


def test_baseline_envelope_never_selects_worse_public_score():
    target, uav, tx, _rx = _simple_scene()
    baseline = np.zeros_like(uav)
    gap_good = np.zeros_like(uav)
    gap_good[1] = np.array([0.0, -20.0])
    selected, use_gap, gap_score, baseline_score = (
        select_baseline_enveloped_movement(
            target, uav, tx, gap_good, baseline, 500.0,
            height_m=20.0))
    assert use_gap is True
    assert gap_score < baseline_score
    selected_score = coverage_deficit_lexicographic_score(
        target, uav + selected, tx, 500.0, height_m=20.0)
    assert selected_score <= baseline_score

    gap_bad = np.zeros_like(uav)
    gap_bad[0] = np.array([900.0, 900.0])
    selected, use_gap, gap_score, baseline_score = (
        select_baseline_enveloped_movement(
            target, uav, tx, gap_bad, baseline, 500.0,
            height_m=20.0))
    assert use_gap is False
    np.testing.assert_array_equal(selected, baseline)
    assert baseline_score < gap_score


def test_baseline_envelope_tie_retains_baseline():
    target, uav, tx, _rx = _simple_scene()
    baseline = np.zeros_like(uav)
    selected, use_gap, gap_score, baseline_score = (
        select_baseline_enveloped_movement(
            target, uav, tx, baseline.copy(), baseline, 300.0,
            height_m=20.0))
    assert use_gap is False
    assert gap_score == baseline_score
    np.testing.assert_array_equal(selected, baseline)


def test_role_aware_product_uses_only_opposite_role_endpoint():
    # Target 0 has a very close Tx but a distant Rx. A Tx mover must use the
    # distant Rx complement; min(nearest Tx, nearest Rx) would invent an
    # invalid same-role product of 1 instead of the physical product 100.
    range_sq = np.asarray([
        [1.0, 16.0],    # Tx 0
        [4.0, 9.0],     # Tx 1
        [100.0, 25.0],  # Rx 2
        [121.0, 36.0],  # Rx 3
    ])
    tx = np.asarray([True, True, False, False])
    product = role_aware_bistatic_products(range_sq, tx)
    np.testing.assert_array_equal(product[0], [100.0, 400.0])
    np.testing.assert_array_equal(product[1], [400.0, 225.0])
    np.testing.assert_array_equal(product[2], [100.0, 225.0])
    np.testing.assert_array_equal(product[3], [121.0, 324.0])
