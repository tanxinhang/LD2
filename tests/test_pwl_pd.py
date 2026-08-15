"""Tests for the two-sided PWL bounds of P_D (D0.93-A2)."""

import numpy as np
import pytest

from uav_isac.coordination.pwl_pd import (
    chord_lower_bound,
    curvature_breakpoints,
    evaluate_pwl,
    p_d,
    tangent_upper_bound,
)

P_FA = 0.001


def test_chord_below_true_and_tangent_above():
    # Working region: P_D >= 0.6 -> D >= (Q^{-1}(PFA)-Q^{-1}(0.6))^2 ~ 8.05.
    d_min, d_max = 8.05, 400.0
    bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=1e-3)
    slopes_c, intc = chord_lower_bound(P_FA, bps)
    slopes_t, intt = tangent_upper_bound(P_FA, bps)
    grid = np.linspace(d_min, d_max, 500)
    true = p_d(grid, P_FA)
    chord = evaluate_pwl(grid, slopes_c, intc)
    tangent = evaluate_pwl(grid, slopes_t, intt)
    assert np.all(chord <= true + 1e-9), "chord must be a lower bound"
    assert np.all(tangent >= true - 1e-9), "tangent must be an upper bound"
    # The bounds are close to the true value (epsilon tightness).
    assert np.max(true - chord) < 0.01
    assert np.max(tangent - true) < 0.01


def test_chord_error_decreases_with_finer_eps():
    d_min, d_max = 8.05, 400.0
    grid = np.linspace(d_min, d_max, 2000)
    true = p_d(grid, P_FA)
    errors = []
    for eps in (1e-2, 1e-3, 1e-4):
        bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=eps)
        slopes, intc = chord_lower_bound(P_FA, bps)
        chord = evaluate_pwl(grid, slopes, intc)
        errors.append(float(np.max(true - chord)))
    # Finer epsilon -> strictly smaller max chord error.
    assert errors[0] > errors[1] > errors[2]
    # The bound is practically tight (sub-1e-2 for eps=1e-4).
    assert errors[2] < 0.01


def test_breakpoints_denser_in_high_curvature():
    d_min, d_max = 8.05, 400.0
    bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=1e-3)
    # The low-D region (near the concavity boundary) has higher curvature, so
    # the first few gaps should be smaller than the last few.
    gaps = np.diff(bps)
    assert np.mean(gaps[:5]) < np.mean(gaps[-5:])
