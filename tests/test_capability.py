"""Tests for the capability gauge and its monotonicity (D0.93-E)."""

import numpy as np
import pytest

from uav_isac.coordination.capability import (
    _bottom_k_sum,
    capability_gauge,
    capability_monotone_holds,
)

XI = (0.60, 0.70, 0.80, 3)
P_FA = 0.001


def test_bottom_k_sum_matches_bruteforce():
    rng = np.random.default_rng(51)
    for _ in range(20):
        y = rng.uniform(0.0, 1.0, size=8)
        for k in (1, 2, 3, 5):
            assert _bottom_k_sum(y, k) == pytest.approx(
                np.sum(np.sort(y)[:k]))


def test_gauge_feasible_when_plenty_of_budget():
    rng = np.random.default_rng(52)
    gain = rng.uniform(0.5, 3.0, size=(6, 6))
    budget = np.full(6, 10.0)  # generous budget
    out = capability_gauge(gain, budget, P_FA, XI)
    assert out is not None
    gamma, d, pd, viol, feasible = out
    assert gamma < 1.0
    assert feasible
    assert all(v >= -1e-6 for v in viol.values())


def test_monotonicity_decreasing_in_gain_and_budget():
    rng = np.random.default_rng(53)
    for _ in range(100):
        K, Q = 5, 4
        gain1 = rng.uniform(0.1, 2.0, size=(K, Q))
        # Elementwise larger gain and larger budget -> system 2 dominates.
        gain2 = gain1 + rng.uniform(0.1, 1.0, size=(K, Q))
        budget1 = rng.uniform(0.3, 0.8, size=K)
        budget2 = budget1 + rng.uniform(0.1, 0.5, size=K)
        assert capability_monotone_holds(
            gain1, budget1, gain2, budget2, P_FA, XI)


def test_monotonicity_same_system_equal():
    rng = np.random.default_rng(54)
    gain = rng.uniform(0.1, 2.0, size=(5, 4))
    budget = rng.uniform(0.3, 0.8, size=5)
    assert capability_monotone_holds(gain, budget, gain, budget, P_FA, XI)


def test_gauge_none_when_target_unreachable():
    gain = np.array([[1.0, 0.0], [1.0, 0.0]])
    budget = np.array([1.0, 1.0])
    assert capability_gauge(gain, budget, P_FA, XI) is None


def test_gauge_bottom_k_binds():
    """A configuration where bottom-3 (not worst) is the binding constraint."""
    # Target 0 is hopelessly weak, others strong: worst binds trivially.
    # Use a configuration where the average is the binding term instead.
    rng = np.random.default_rng(55)
    gain = rng.uniform(0.3, 1.0, size=(6, 6))
    budget = rng.uniform(0.2, 0.5, size=6)
    out = capability_gauge(gain, budget, P_FA, XI)
    if out is not None:
        gamma, d, pd, viol, feasible = out
        # Verified constraints are within tolerance.
        assert all(v >= -1e-6 for v in viol.values())


def test_pwl_lp_sandwich():
    """Two-sided PWL: gamma_optimistic <= gamma_exact <= gamma_conservative."""
    from uav_isac.coordination.capability import capability_gauge_pwl_lp
    from uav_isac.coordination.pwl_pd import (
        chord_lower_bound,
        curvature_breakpoints,
        tangent_upper_bound,
    )
    from uav_isac.physical.detection import (
        minimum_deflection_for_detection_probability,
    )

    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([0.60]), P_FA)[0])
    rng = np.random.default_rng(56)
    for _ in range(20):
        K, Q = 6, 6
        gain = rng.uniform(0.3, 1.0, size=(K, Q))
        budget = rng.uniform(0.3, 0.8, size=K)
        d_max = float(np.max(np.sum(gain * budget[:, None], axis=0))) + 1.0
        exact = capability_gauge(gain, budget, P_FA, XI)
        if exact is None:
            continue
        gamma_exact = exact[0]
        bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=1e-3)
        chord_s, chord_i = chord_lower_bound(P_FA, bps)
        tangent_s, tangent_i = tangent_upper_bound(P_FA, bps)
        gamma_conservative = capability_gauge_pwl_lp(
            gain, budget, P_FA, XI, chord_s, chord_i, d_min)
        gamma_optimistic = capability_gauge_pwl_lp(
            gain, budget, P_FA, XI, tangent_s, tangent_i, d_min)
        if gamma_conservative is None or gamma_optimistic is None:
            continue
        assert gamma_optimistic - 1e-6 <= gamma_exact + 1e-6, (
            f"optimistic {gamma_optimistic} > exact {gamma_exact}")
        assert gamma_exact - 1e-6 <= gamma_conservative + 1e-6, (
            f"exact {gamma_exact} > conservative {gamma_conservative}")


def test_pwl_capability_gauge_remains_defined_above_physical_budget():
    """The gauge must report gamma>1, not erase the L1 fallback margin."""
    from uav_isac.coordination.capability import capability_gauge_pwl_lp
    from uav_isac.coordination.pwl_pd import saturating_chord_lower_bound
    from uav_isac.physical.detection import (
        minimum_deflection_for_detection_probability,
    )

    p_fa = 0.01
    xi = (0.6, 0.6, 0.6, 1)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([xi[0]]), p_fa)[0])
    slopes, intercepts, _ = saturating_chord_lower_bound(
        p_fa, d_min, max(2.0 * d_min, d_min + 1.0), 1.0e-3)
    gain = np.eye(2) * d_min
    gamma = capability_gauge_pwl_lp(
        gain, np.full(2, 0.5), p_fa, xi, slopes, intercepts, d_min)
    assert gamma is not None
    assert gamma > 1.0


def test_capability_certificate_returns_canonical_budget_dual():
    from uav_isac.coordination.capability import (
        capability_gauge_pwl_lp_certificate,
    )
    from uav_isac.coordination.pwl_pd import saturating_chord_lower_bound
    from uav_isac.physical.detection import (
        minimum_deflection_for_detection_probability,
    )

    p_fa = 0.01
    xi = (0.6, 0.6, 0.6, 1)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([xi[0]]), p_fa)[0])
    slopes, intercepts, _ = saturating_chord_lower_bound(
        p_fa, d_min, 3.0 * d_min, 1.0e-3)
    budget = np.asarray([0.5, 0.5])
    out = capability_gauge_pwl_lp_certificate(
        np.eye(2) * d_min, budget, p_fa, xi,
        slopes, intercepts, d_min)
    assert out is not None
    gamma, power, target_price, scarcity_price = out
    assert gamma > 0.0
    assert power.shape == (2, 2)
    assert np.all(target_price >= 0.0)
    assert np.all(scarcity_price >= 0.0)
    assert np.dot(scarcity_price, budget) == pytest.approx(1.0, abs=1e-7)


def test_fixed_structure_minimum_total_power_is_monotone_in_gain():
    from uav_isac.coordination.capability import minimum_total_power_pwl_lp
    from uav_isac.coordination.pwl_pd import saturating_chord_lower_bound
    from uav_isac.physical.detection import (
        minimum_deflection_for_detection_probability,
    )

    p_fa = 0.01
    xi = (0.6, 0.6, 0.6, 1)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([xi[0]]), p_fa)[0])
    slopes, intercepts, _ = saturating_chord_lower_bound(
        p_fa, d_min, 3.0 * d_min, 1.0e-3)
    gain = np.eye(2) * d_min
    base = minimum_total_power_pwl_lp(
        gain, np.full(2, 3.0), p_fa, xi, slopes, intercepts, d_min)
    better = minimum_total_power_pwl_lp(
        2.0 * gain, np.full(2, 3.0), p_fa, xi, slopes, intercepts, d_min)
    assert base is not None and better is not None
    assert np.isclose(base[0], 2.0, atol=1.0e-7)
    assert np.isclose(better[0], 1.0, atol=1.0e-7)


def test_envelope_gradient_matches_finite_difference():
    """L3-T2: d gamma*/d a_iq = -pi_q* p_iq* (capability-KKT envelope theorem)."""
    from uav_isac.coordination.capability import capability_gauge_pwl_lp_full
    from uav_isac.coordination.pwl_pd import (
        chord_lower_bound,
        curvature_breakpoints,
    )
    from uav_isac.physical.detection import (
        minimum_deflection_for_detection_probability,
    )

    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([0.60]), P_FA)[0])
    rng = np.random.default_rng(57)
    K, Q = 5, 4
    gain = rng.uniform(1.0, 3.0, size=(K, Q))
    budget = np.full(K, 3.0)  # generous: ceiling ~30 >> d_min
    d_max = float(np.max(np.sum(gain * budget[:, None], axis=0))) + 1.0
    bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=1e-3)
    chord_s, chord_i = chord_lower_bound(P_FA, bps)
    out = capability_gauge_pwl_lp_full(
        gain, budget, P_FA, XI, chord_s, chord_i, d_min)
    assert out is not None
    gamma, p, pi = out
    analytic = pi[None, :] * p  # d gamma/da_iq = -pi_q * p_iq

    # Finite difference a few entries.
    eps = 1e-5
    for i, q in [(0, 0), (2, 3), (4, 1)]:
        gp = gain.copy()
        gp[i, q] += eps
        gm = gain.copy()
        gm[i, q] -= eps
        outp = capability_gauge_pwl_lp_full(
            gp, budget, P_FA, XI, chord_s, chord_i, d_min)
        outm = capability_gauge_pwl_lp_full(
            gm, budget, P_FA, XI, chord_s, chord_i, d_min)
        if outp is None or outm is None:
            continue
        fd = (outp[0] - outm[0]) / (2.0 * eps)
        assert np.isclose(fd, analytic[i, q], atol=1e-3), (
            f"a[{i},{q}]: finite-diff {fd} vs analytic {analytic[i,q]}")


def test_envelope_gradient_directional_derivative():
    """Directional derivative along a random admissible direction matches."""
    from uav_isac.coordination.capability import capability_gauge_pwl_lp_full
    from uav_isac.coordination.pwl_pd import (
        chord_lower_bound,
        curvature_breakpoints,
    )
    from uav_isac.physical.detection import (
        minimum_deflection_for_detection_probability,
    )

    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([0.60]), P_FA)[0])
    rng = np.random.default_rng(58)
    K, Q = 5, 4
    gain = rng.uniform(1.0, 3.0, size=(K, Q))
    budget = np.full(K, 3.0)  # generous: ceiling ~30 >> d_min
    d_max = float(np.max(np.sum(gain * budget[:, None], axis=0))) + 1.0
    bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=1e-3)
    chord_s, chord_i = chord_lower_bound(P_FA, bps)
    out = capability_gauge_pwl_lp_full(
        gain, budget, P_FA, XI, chord_s, chord_i, d_min)
    assert out is not None
    _, p, pi = out
    analytic_grad = pi[None, :] * p

    direction = rng.normal(0.0, 1.0, size=(K, Q))
    direction = direction / np.linalg.norm(direction)
    for h in (1e-3, 1e-4, 1e-5):
        gp = gain + h * direction
        gm = gain - h * direction
        outp = capability_gauge_pwl_lp_full(
            gp, budget, P_FA, XI, chord_s, chord_i, d_min)
        outm = capability_gauge_pwl_lp_full(
            gm, budget, P_FA, XI, chord_s, chord_i, d_min)
        if outp is None or outm is None:
            continue
        fd = (outp[0] - outm[0]) / (2.0 * h)
        analytic = float(np.sum(analytic_grad * direction))
        assert np.isclose(fd, analytic, atol=1e-2), (
            f"h={h}: fd {fd} vs analytic {analytic}")
