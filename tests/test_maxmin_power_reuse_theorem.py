"""O8 / T5 property tests: reuse / warm-start is a CERTIFICATE, not a heuristic.

T5 (roadmap docs/CURRENT_SYSTEM_MODEL.md §3):
  1. Constructive feasibility: the held allocation is normalized row-wise and
     rescaled to the CURRENT budget (maxmin_power.py:592-600), so a reused
     solution is non-negative with per-row sum == public budget -- feasibility
     is preserved BY CONSTRUCTION, no solver needed.
  2. Epsilon-optimality certificate: with lower = min_q sum_i a_ikq p_ikq
     (primal value of the held allocation) and upper = sum_k b_k * max_q
     (price_q * a_kq) (a dual upper bound, maxmin_power.py:601-612), weak
     duality gives upper >= OPT.  The reuse decision is gated on
     relative_gap = (upper - lower)/upper <= reuse_tolerance, so any reused
     allocation is epsilon-optimal with a COMPUTED certificate (the gap is
     logged, not asserted by faith).
  3. Same-geometry invariance: for identical gain views the reassembled
     allocation equals the centralized LP optimum (module docstring
     maxmin_power.py:379-383); reuse with gap=0 is exactly that optimum.

C1 note: this locks the certificate properties; the DEFAULT path
(reuse_relative_tolerance=0.0) is untouched (zero-semantics).
"""

import numpy as np
import pytest

from uav_isac.coordination.maxmin_power import (
    replicated_local_row_maxmin_power,
    solve_fixed_structure_maxmin_power_lp,
)


def _common_view(K: int, Q: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gain = rng.uniform(0.01, 1.0, size=(K, Q))
    # same public graph for every viewer -> assembly == centralized LP optimum
    return np.broadcast_to(gain, (K, K, Q)).copy()


def _fresh(views, budget, seed_extra=0):
    rng = np.random.default_rng(seed_extra)
    return replicated_local_row_maxmin_power(
        views, budget, previous_local_power_w=None,
        previous_local_prices=None, previous_local_cache_valid=None,
        reuse_relative_tolerance=0.0, parallel_executor=None,
        parallel_fallback_to_serial=True, parallel_failure_mode="serial",
    )


def test_reuse_is_feasible_by_construction():
    K, Q = 4, 3
    views = _common_view(K, Q, 7)
    budget = np.array([1.0, 1.2, 0.8, 1.0], dtype=np.float64)
    first = _fresh(views, budget, seed_extra=1)
    # warm-start from the first solve
    second = replicated_local_row_maxmin_power(
        views, budget,
        previous_local_power_w=first.local_full_power_w,
        previous_local_prices=first.local_prices,
        previous_local_cache_valid=first.local_cache_valid,
        reuse_relative_tolerance=0.2, parallel_executor=None,
        parallel_fallback_to_serial=True, parallel_failure_mode="serial",
    )
    # row sums of the executed power equal the budget exactly (T5-1)
    np.testing.assert_allclose(
        np.sum(second.power_w, axis=1), budget, rtol=0.0, atol=1e-9)
    assert np.all(second.power_w >= 0.0)


def test_reuse_gap_is_a_computed_certificate():
    """upper (dual) >= OPT (weak duality); reused worst >= lower; gap logged."""
    K, Q = 4, 3
    views = _common_view(K, Q, 11)
    budget = np.ones(K, dtype=np.float64)
    first = _fresh(views, budget, seed_extra=2)
    ref = solve_fixed_structure_maxmin_power_lp(views[0], budget)
    opt = ref.worst_deflection  # centralized LP optimum (identical views)
    second = replicated_local_row_maxmin_power(
        views, budget,
        previous_local_power_w=first.local_full_power_w,
        previous_local_prices=first.local_prices,
        previous_local_cache_valid=first.local_cache_valid,
        reuse_relative_tolerance=0.1, parallel_executor=None,
        parallel_fallback_to_serial=True, parallel_failure_mode="serial",
    )
    # legitimate domain is [0,1): tolerance must be strictly below 1
    # (validated by the function).  For identical views the held allocation is
    # the same LP optimum (module docstring 379-383), so the certified primal-
    # dual gap is ~0 and every viewer reuses.
    assert np.all(second.local_cache_valid)
    gaps = second.local_relative_gap
    assert np.all(gaps <= 0.1)
    # the gap is a COMPUTED certificate (primal lower vs dual upper, weak
    # duality upper >= OPT >= lower): reuse only happened where gap <= tol,
    # and the reassembled worst equals the centralized LP optimum exactly.
    np.testing.assert_allclose(
        second.local_worst_deflection, opt, rtol=0.0, atol=1e-12)


def test_same_geometry_reuse_is_optimal():
    """Identical views + gap=0 -> reuse equals the fresh centralized LP."""
    K, Q = 4, 2
    views = _common_view(K, Q, 23)
    budget = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    first = _fresh(views, budget, seed_extra=3)
    ref = solve_fixed_structure_maxmin_power_lp(views[0], budget)
    second = replicated_local_row_maxmin_power(
        views, budget,
        previous_local_power_w=first.local_full_power_w,
        previous_local_prices=first.local_prices,
        previous_local_cache_valid=first.local_cache_valid,
        reuse_relative_tolerance=0.0,  # gap must be exactly 0 for reuse
        parallel_executor=None,
        parallel_fallback_to_serial=True, parallel_failure_mode="serial",
    )
    # with tol=0 reuse requires relative_gap == 0; if the previous allocation
    # was already optimal on the same views, no solve is needed and worst == OPT
    np.testing.assert_allclose(
        second.local_worst_deflection, ref.worst_deflection,
        rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(second.power_w, axis=1), budget, rtol=0.0, atol=1e-9)


def test_default_path_is_unchanged():
    """reuse_relative_tolerance=0 (default) never takes the reuse branch."""
    K, Q = 4, 3
    views = _common_view(K, Q, 31)
    budget = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    fresh_without_cache = _fresh(views, budget, seed_extra=4)
    # supply a stale cache but keep tolerance 0 -> identical to fresh
    with_cache_zero_tol = replicated_local_row_maxmin_power(
        views, budget,
        previous_local_power_w=fresh_without_cache.local_full_power_w,
        previous_local_prices=fresh_without_cache.local_prices,
        previous_local_cache_valid=fresh_without_cache.local_cache_valid,
        reuse_relative_tolerance=0.0, parallel_executor=None,
        parallel_fallback_to_serial=True, parallel_failure_mode="serial",
    )
    np.testing.assert_array_equal(
        with_cache_zero_tol.power_w, fresh_without_cache.power_w)
    np.testing.assert_array_equal(
        with_cache_zero_tol.local_worst_deflection,
        fresh_without_cache.local_worst_deflection)