"""Dedicated tests for the S0 safe structural screens.

Audit 2026-09-04 §10 B1: ``coordination/safe_structural_reduction.py`` had
no dedicated test file (one function was indirectly exercised by
``test_minimum_intervention_repair.py``).  These tests lock the S0 layer's
core contract: *exact-zero* semantics without numerical tolerance, and the
feasibility / optimality split of the screened edges.
"""

import numpy as np
import pytest

from uav_isac.coordination.safe_structural_reduction import (
    ContextFreeScreen,
    context_free_zero_gain_screen,
)


def _screen(gain, selected):
    return context_free_zero_gain_screen(
        np.asarray(gain, dtype=np.float64),
        np.asarray(selected, dtype=bool),
    )


def test_zero_gain_edges_cover_all_off_diagonal_edges():
    K, Q = 4, 2
    gain = np.zeros((K, K, Q))
    selected = np.zeros_like(gain, dtype=bool)
    screen = _screen(gain, selected)
    assert isinstance(screen, ContextFreeScreen)
    assert screen.candidate_edges == K * (K - 1) * Q
    assert len(screen.feasibility_preserving) == screen.candidate_edges
    assert len(screen.optimality_preserving) == screen.candidate_edges
    assert len(screen.selected_zero_gain) == 0


def test_selected_zero_gain_edges_report_repair_toggle():
    K, Q = 4, 2
    gain = np.zeros((K, K, Q))
    selected = np.zeros_like(gain, dtype=bool)
    selected[0, 1, 0] = True
    selected[2, 3, 1] = True
    screen = _screen(gain, selected)
    assert set(screen.selected_zero_gain) == {(0, 1, 0), (2, 3, 1)}
    assert len(screen.optimality_preserving) == screen.candidate_edges - 2
    assert set(screen.feasibility_preserving) >= set(
        screen.selected_zero_gain)


def test_positive_gain_edges_are_never_screened():
    K, Q = 4, 2
    gain = np.zeros((K, K, Q))
    gain[0, 1, 0] = 1.0
    gain[2, 3, 1] = 0.5
    selected = np.zeros_like(gain, dtype=bool)
    screen = _screen(gain, selected)
    assert (0, 1, 0) not in screen.feasibility_preserving
    assert (0, 1, 0) not in screen.optimality_preserving
    assert (2, 3, 1) not in screen.feasibility_preserving
    assert len(screen.feasibility_preserving) == screen.candidate_edges - 2


def test_exact_zero_semantics_has_no_tolerance():
    # The S0 layer intentionally interprets zero exactly: a tiny positive
    # coefficient remains capable of contributing and must NOT be screened.
    K, Q = 4, 2
    gain = np.zeros((K, K, Q))
    gain[0, 1, 0] = 1.0e-12
    selected = np.zeros_like(gain, dtype=bool)
    screen = _screen(gain, selected)
    assert (0, 1, 0) not in screen.feasibility_preserving
    assert (0, 1, 0) not in screen.optimality_preserving


def test_negative_gain_is_rejected():
    K, Q = 4, 2
    gain = np.zeros((K, K, Q))
    gain[0, 1, 0] = -1.0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _screen(gain, np.zeros_like(gain, dtype=bool))


def test_non_finite_gain_is_rejected():
    K, Q = 4, 2
    gain = np.zeros((K, K, Q))
    gain[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _screen(gain, np.zeros_like(gain, dtype=bool))


def test_shape_validation():
    with pytest.raises(ValueError, match=r"\(K,K,Q\)"):
        _screen(np.zeros((3, 4, 2)), np.zeros((3, 4, 2), dtype=bool))
    with pytest.raises(ValueError, match=r"\(K,K,Q\)"):
        _screen(np.zeros((4, 5)), np.zeros((4, 5), dtype=bool))
    with pytest.raises(ValueError, match="shape must match"):
        _screen(np.zeros((4, 4, 2)), np.zeros((4, 4, 1), dtype=bool))