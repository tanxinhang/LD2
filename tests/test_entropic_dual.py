"""Tests for the entropy-regularized (central-path) max-min dual price.

Covers the three theoretical claims used to justify the price plus the
degenerate-face behaviour that the LP vertex gets wrong:

  1. ERROR BOUND: 0 <= f(lambda*_tau) - f(lambda*) bounded, monotone in tau.
  2. DEDICATED TARGETS: a block-diagonal gain makes the whole simplex optimal;
     the LP vertex returns an arbitrary delta while the entropic price is
     strictly interior (lambda > 0).
  3. CLUSTERED BOTTLENECKS: two independent bottleneck clusters; the LP vertex
     concentrates on one cluster, the entropic price covers both (all > 0).
  4. INTERIOR: the entropic price is strictly positive (lambda > 0) even when
     the LP vertex degenerates to a vertex.
  5. CONTINUITY: the entropic price is continuous in the gains (no basis
     switching under small perturbations).
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_isac.coordination.maxmin_power import (
    entropic_maxmin_dual_prices,
    optimal_maxmin_dual_prices,
    canonical_maxmin_dual_prices,
)


def _dual_obj(gain: np.ndarray, budget: np.ndarray, lam: np.ndarray) -> float:
    return float(np.sum(budget * np.max(lam[None, :] * gain, axis=1)))


def test_error_bound_monotone() -> None:
    rng = np.random.default_rng(0)
    K, Q = 6, 6
    gain = rng.lognormal(0.0, 1.2, size=(K, Q))
    budget = np.ones(K)
    _, f_star = optimal_maxmin_dual_prices(gain, budget)
    prev_gap: float | None = None
    for tau in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
        lam, f_tau = entropic_maxmin_dual_prices(gain, budget, tau)
        gap = f_tau - f_star
        assert gap >= -1e-9
        if prev_gap is not None:
            assert gap <= prev_gap + 1e-6
        prev_gap = gap
        np.testing.assert_allclose(np.sum(lam), 1.0, atol=1e-9)
        assert np.all(lam >= -1e-12)


def test_error_bound_holds_when_gain_scale_is_not_dimensionless() -> None:
    """Regression: ``alpha=tau`` mixing violated the claimed physical cap."""
    gain = np.asarray([[1.0, 1000.0]])
    budget = np.ones(1)
    tau = 0.01
    _, f_star = optimal_maxmin_dual_prices(gain, budget)
    lam, f_tau = entropic_maxmin_dual_prices(gain, budget, tau)
    assert f_tau - f_star <= tau * np.log(gain.shape[1]) + 1.0e-9
    assert f_tau == pytest.approx(_dual_obj(gain, budget, lam), abs=1.0e-12)


def test_dual_lp_is_scale_safe_for_small_physical_coefficients() -> None:
    """HiGHS absolute tolerances must not erase radar-scale coefficients."""
    gain = 1.0e-12 * np.asarray([[1.0, 1000.0]])
    budget = np.ones(1)
    lam, value = optimal_maxmin_dual_prices(gain, budget)
    expected = np.asarray([1000.0 / 1001.0, 1.0 / 1001.0])
    np.testing.assert_allclose(lam, expected, rtol=1.0e-9, atol=1.0e-12)
    assert value == pytest.approx(
        _dual_obj(gain, budget, lam), rel=1.0e-10, abs=1.0e-24)


def test_canonical_price_is_equivariant_on_fully_degenerate_face() -> None:
    """An LP vertex must not leak target ordering into a canonical label."""
    gain = np.eye(3)
    budget = np.ones(3)
    permutation = np.asarray([1, 0, 2])
    base, base_value = canonical_maxmin_dual_prices(gain, budget)
    permuted, permuted_value = canonical_maxmin_dual_prices(
        gain[:, permutation], budget)
    np.testing.assert_allclose(base, np.full(3, 1.0 / 3.0), atol=1.0e-9)
    np.testing.assert_allclose(permuted, base[permutation], atol=1.0e-9)
    assert base_value == pytest.approx(_dual_obj(gain, budget, base))
    assert permuted_value == pytest.approx(base_value)


def test_dedicated_targets_strictly_interior() -> None:
    gain = np.eye(3)
    budget = np.ones(3)
    lam_opt, f_star = optimal_maxmin_dual_prices(gain, budget)
    lam_ent, f_ent = entropic_maxmin_dual_prices(gain, budget, 0.0)
    assert f_star == pytest.approx(1.0)
    assert np.min(lam_opt) == pytest.approx(0.0, abs=1e-9)
    assert np.max(lam_opt) == pytest.approx(1.0, abs=1e-9)
    assert np.all(lam_ent > 0.0)
    np.testing.assert_allclose(np.sum(lam_ent), 1.0, atol=1e-9)
    assert f_ent == pytest.approx(1.0, abs=1e-3)


def test_entropic_price_is_strictly_interior() -> None:
    gain = np.eye(4)
    budget = np.ones(4)
    lam, _ = entropic_maxmin_dual_prices(gain, budget, 0.0)
    assert np.all(lam > 0.0)
    np.testing.assert_allclose(np.sum(lam), 1.0, atol=1e-9)


def test_clustered_bottlenecks_cover_both() -> None:
    g = np.zeros((4, 4))
    g[0, 0] = g[1, 0] = 1.0
    g[0, 1] = g[1, 1] = 1.0
    g[2, 2] = g[3, 2] = 1.0
    g[2, 3] = g[3, 3] = 1.0
    lam_opt, _ = optimal_maxmin_dual_prices(g, np.ones(4))
    lam_ent, _ = entropic_maxmin_dual_prices(g, np.ones(4), 0.1)
    first_cluster_mass = float(lam_opt[0] + lam_opt[1])
    assert first_cluster_mass in (pytest.approx(0.0, abs=1e-9),
                                  pytest.approx(1.0, abs=1e-9))
    assert np.all(lam_ent > 0.0)
    ent_first = float(lam_ent[0] + lam_ent[1])
    ent_second = float(lam_ent[2] + lam_ent[3])
    assert ent_first > 0.01
    assert ent_second > 0.01


def test_matching_with_nondegenerate_face() -> None:
    rng = np.random.default_rng(3)
    gain = rng.lognormal(0.0, 1.0, size=(5, 5))
    budget = np.ones(5)
    lam_opt, f_opt = optimal_maxmin_dual_prices(gain, budget)
    lam_can, f_can = canonical_maxmin_dual_prices(gain, budget)
    lam_ent, f_ent = entropic_maxmin_dual_prices(gain, budget, 0.0)
    np.testing.assert_allclose(lam_ent, lam_opt, atol=1e-4)
    assert abs(f_can - f_opt) < 0.01 * max(1.0, abs(f_opt))
    assert abs(f_ent - f_opt) < 0.01 * max(1.0, abs(f_opt))
    assert np.all(lam_can > 0.0)
    assert np.all(lam_ent > 0.0)


def test_continuity_under_perturbation() -> None:
    """Entropic price should not jump under small gain perturbations."""
    rng = np.random.default_rng(42)
    K, Q = 4, 4
    gain = rng.lognormal(0.0, 1.0, size=(K, Q))
    budget = np.ones(K)
    lam_prev, _ = entropic_maxmin_dual_prices(gain, budget, 0.01)
    max_jump = 0.0
    for _ in range(20):
        pert = 1e-4 * rng.standard_normal((K, Q))
        lam_i, _ = entropic_maxmin_dual_prices(gain + pert, budget, 0.01)
        jump = float(np.max(np.abs(lam_i - lam_prev)))
        max_jump = max(max_jump, jump)
        lam_prev = lam_i
    assert max_jump < 0.1
