import numpy as np
import pytest

from uav_isac.coordination.maxmin_power import (
    NonUniqueFixedOwnerStructureError,
    blend_row_feasible_power_with_inertia,
    distributed_column_generation_maxmin_power,
    distributed_dual_maxmin_power,
    fixed_owner_gain_matrix,
    fixed_owner_gain_matrix_from_selected_values,
    local_transmitter_range_minimax_share,
    optimal_maxmin_dual_prices,
    relaxed_same_geometry_target_ceiling,
    replicated_local_row_maxmin_power,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.maxmin_power import _primal_dual_tolerance


def test_primal_dual_certificate_tolerance_scales_with_physical_objective():
    primal = 383.6030900104997
    dual = 383.6031843422889

    tolerance = _primal_dual_tolerance(primal, dual)

    assert dual - primal < tolerance
    assert tolerance == pytest.approx(dual * 1.0e-6)
    assert 2.0e-6 * dual > tolerance


def test_inertia_blend_preserves_each_local_power_simplex():
    candidate = np.asarray([[0.9, 0.1], [0.0, 0.0]])
    previous = np.asarray([[0.1, 0.9], [0.8, 0.2]])
    budget = np.asarray([0.7, 0.4])

    blended = blend_row_feasible_power_with_inertia(
        candidate, previous, budget, inertia=0.25)

    assert np.all(blended >= 0.0)
    np.testing.assert_allclose(np.sum(blended, axis=1), budget, atol=1.0e-12)
    np.testing.assert_allclose(blended[0], [0.49, 0.21], atol=1.0e-12)
    # A missing current row falls back to the neutral uniform feasible row.
    np.testing.assert_allclose(blended[1], [0.23, 0.17], atol=1.0e-12)


def test_zero_inertia_returns_budget_projected_candidate():
    candidate = np.asarray([[2.0, 1.0]])
    result = blend_row_feasible_power_with_inertia(
        candidate, np.asarray([[0.0, 1.0]]), np.asarray([0.6]), inertia=0.0)
    np.testing.assert_allclose(result, [[0.4, 0.2]], atol=1.0e-12)


def test_distributed_power_projects_machine_precision_simplex_residuals():
    gain = np.asarray([[2.0, 1.0], [1.0, 3.0]])
    budget = np.asarray([1.0, 1.0])
    incumbent = np.asarray([
        [1.0 + 2.0e-16, -2.0e-16],
        [0.4, 0.6 + 1.0e-16],
    ])

    result = distributed_column_generation_maxmin_power(
        gain,
        budget,
        rounds=2,
        incumbent_power_w=incumbent,
    )

    assert np.all(result.power_w >= 0.0)
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), budget, atol=1.0e-12)


def test_exact_lp_recovers_analytic_single_transmitter_split():
    gain = np.asarray([[2.0, 1.0]])
    result = solve_fixed_structure_maxmin_power_lp(gain, np.asarray([1.0]))
    np.testing.assert_allclose(result.power_w, [[1.0 / 3.0, 2.0 / 3.0]], atol=1e-8)
    np.testing.assert_allclose(result.deflection, [2.0 / 3.0, 2.0 / 3.0], atol=1e-8)


@pytest.mark.parametrize("scale", [1.0e-12, 1.0, 1.0e12])
def test_exact_lp_is_invariant_to_global_physical_gain_scale(scale):
    gain = float(scale) * np.asarray([[1.0, 1000.0]])

    result = solve_fixed_structure_maxmin_power_lp(
        gain, np.asarray([1.0]))

    expected = np.asarray([[1000.0 / 1001.0, 1.0 / 1001.0]])
    np.testing.assert_allclose(result.power_w, expected, rtol=1.0e-10)
    np.testing.assert_allclose(
        result.deflection,
        np.full(2, float(scale) * 1000.0 / 1001.0),
        rtol=1.0e-10,
        atol=float(scale) * 1.0e-12,
    )
    np.testing.assert_allclose(
        result.prices, expected[0], rtol=1.0e-10)
    assert result.dual_upper_bound == pytest.approx(
        result.worst_deflection, rel=1.0e-10,
        abs=float(scale) * 1.0e-12)


@pytest.mark.parametrize("strong_gain", [1.0e7, 1.0e9])
def test_exact_lp_preserves_weak_target_across_extreme_gain_range(strong_gain):
    gain = np.asarray([[strong_gain, 1.0]])
    budget = np.asarray([1.0])

    result = solve_fixed_structure_maxmin_power_lp(gain, budget)
    prices, dual_value = optimal_maxmin_dual_prices(gain, budget)

    expected_worst = strong_gain / (strong_gain + 1.0)
    expected_power = np.asarray([
        [1.0 / (strong_gain + 1.0), expected_worst]
    ])
    np.testing.assert_allclose(result.power_w, expected_power, rtol=1.0e-10)
    np.testing.assert_allclose(
        result.deflection, np.full(2, expected_worst), rtol=1.0e-10)
    assert result.worst_deflection == pytest.approx(
        expected_worst, rel=1.0e-10)
    np.testing.assert_allclose(prices, expected_power[0], rtol=1.0e-10)
    assert dual_value == pytest.approx(expected_worst, rel=1.0e-10)


def test_exact_lp_handles_extreme_multitransmitter_range_without_budget_error():
    gain = np.asarray([
        [751595.6985, 80519.9587],
        [3.21e-9, 1.94e-7],
    ])
    budget = np.asarray([0.218, 0.240])

    result = solve_fixed_structure_maxmin_power_lp(gain, budget)

    row_two_target_two = budget[1] * gain[1, 1]
    expected_first_power = (
        gain[0, 1] * budget[0] + row_two_target_two
    ) / (gain[0, 0] + gain[0, 1])
    expected_worst = gain[0, 0] * expected_first_power
    assert np.all(result.power_w >= 0.0)
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), budget, atol=1.0e-12)
    assert result.worst_deflection == pytest.approx(
        expected_worst, rel=1.0e-10)
    assert result.dual_upper_bound == pytest.approx(
        expected_worst, rel=1.0e-10)


def test_exact_lp_uses_finite_reachable_scale_not_cross_product_overflow():
    # max(gain) * sum(budget) overflows, but every physically reachable
    # gain*own-budget product and target ceiling is finite (11 per target).
    gain = np.asarray([
        [1.0e308, 1.0e308],
        [1.0, 1.0],
    ])
    budget = np.asarray([1.0e-308, 10.0])

    result = solve_fixed_structure_maxmin_power_lp(gain, budget)

    assert np.all(np.isfinite(result.deflection))
    np.testing.assert_allclose(result.deflection, [5.5, 5.5], rtol=1.0e-10)
    assert result.worst_deflection == pytest.approx(5.5, rel=1.0e-10)
    assert result.dual_upper_bound == pytest.approx(5.5, rel=1.0e-10)
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), budget, rtol=1.0e-10)


def test_optimal_dual_solver_failure_is_not_reported_as_zero(monkeypatch):
    class FailedSolve:
        success = False
        x = None
        message = "synthetic numerical failure"

    monkeypatch.setattr(
        "uav_isac.coordination.maxmin_power.linprog",
        lambda *args, **kwargs: FailedSolve(),
    )

    with pytest.raises(RuntimeError, match="synthetic numerical failure"):
        optimal_maxmin_dual_prices(
            np.asarray([[2.0, 1.0]]), np.asarray([1.0]))


@pytest.mark.parametrize("scale", [1.0e-12, 1.0, 1.0e12])
def test_exact_lp_scales_reserve_constraints_with_physical_gain(scale):
    gain = float(scale) * np.asarray([
        [2.0, 0.4],
        [0.3, 1.5],
    ])
    budget = np.asarray([0.8, 0.7])
    reserve = float(scale) * np.asarray([0.25, 0.35])

    result = solve_fixed_structure_maxmin_power_lp(
        gain, budget, minimum_deflection=reserve)

    assert result.reserve_feasible
    assert np.all(result.deflection >= reserve - float(scale) * 1.0e-9)
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), budget, atol=1.0e-10)


def test_exact_lp_preserves_every_uav_power_equality():
    gain = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    budget = np.asarray([0.7, 0.8, 0.9])
    result = solve_fixed_structure_maxmin_power_lp(gain, budget)
    np.testing.assert_allclose(np.sum(result.power_w, axis=1), budget, atol=1e-10)
    assert result.worst_deflection == pytest.approx(0.7)


def test_exact_lp_returns_simplex_prices_that_certify_full_dual_value():
    gain = np.asarray([
        [2.0, 0.4, 0.2],
        [0.3, 1.5, 0.8],
        [0.2, 0.5, 1.8],
    ])
    budget = np.asarray([0.8, 0.7, 0.9])
    result = solve_fixed_structure_maxmin_power_lp(gain, budget)

    np.testing.assert_allclose(np.sum(result.prices), 1.0, atol=1.0e-10)
    assert np.all(result.prices >= 0.0)
    lifted_upper = float(np.sum(
        budget * np.max(result.prices[None, :] * gain, axis=1)))
    assert lifted_upper == pytest.approx(
        result.worst_deflection, rel=1.0e-8, abs=1.0e-10)


def test_replicated_local_rows_equal_exact_lp_under_common_view():
    gain = np.asarray([
        [2.0, 0.4, 0.2],
        [0.3, 1.5, 0.8],
        [0.2, 0.5, 1.8],
    ])
    budget = np.asarray([0.8, 0.7, 0.9])
    views = np.repeat(gain[None, :, :], 3, axis=0)
    exact = solve_fixed_structure_maxmin_power_lp(gain, budget)

    result = replicated_local_row_maxmin_power(views, budget)

    assert result.common_view
    assert result.unique_local_problem_count == 1
    assert np.all(result.local_full_coverage)
    np.testing.assert_allclose(result.power_w, exact.power_w, atol=1.0e-10)
    np.testing.assert_allclose(
        result.local_worst_deflection,
        np.full(3, exact.worst_deflection),
        atol=1.0e-10,
    )


def test_history_reserve_has_feasible_witness_and_preserves_local_worst():
    old_gain = np.asarray([
        [2.0, 0.4, 0.2],
        [0.3, 1.5, 0.8],
        [0.2, 0.5, 1.8],
    ])
    new_gain = np.asarray([
        [1.8, 0.6, 0.25],
        [0.45, 1.3, 0.9],
        [0.3, 0.55, 1.6],
    ])
    budget = np.asarray([0.8, 0.7, 0.9])
    old_views = np.repeat(old_gain[None, :, :], 3, axis=0)
    new_views = np.repeat(new_gain[None, :, :], 3, axis=0)
    initial = replicated_local_row_maxmin_power(old_views, budget)

    result = replicated_local_row_maxmin_power(
        new_views,
        budget,
        previous_local_power_w=initial.local_full_power_w,
        previous_local_prices=initial.local_prices,
        previous_local_cache_valid=initial.local_cache_valid,
        history_reserve_deflection_cap=0.8,
    )

    assert result.history_reserve_used_fraction == 1.0
    for viewer in range(3):
        held = initial.local_full_power_w[viewer]
        held_deflection = np.sum(new_gain * held, axis=0)
        reserve = np.minimum(held_deflection, 0.8)
        achieved = np.sum(
            new_gain * result.local_full_power_w[viewer], axis=0)
        np.testing.assert_array_less(reserve - 1.0e-10, achieved)
        assert np.min(achieved) + 1.0e-10 >= np.min(held_deflection)


def test_history_reserve_rejects_non_positive_cap():
    with pytest.raises(ValueError, match="finite and positive"):
        replicated_local_row_maxmin_power(
            np.ones((1, 1, 2)),
            np.ones(1),
            history_reserve_deflection_cap=0.0,
        )


def test_replicated_local_rows_remain_budget_feasible_with_cache_loss():
    gain = np.asarray([
        [2.0, 0.4, 0.2],
        [0.3, 1.5, 0.8],
        [0.2, 0.5, 1.8],
    ])
    views = np.repeat(gain[None, :, :], 3, axis=0)
    views[1, 0] = 0.0
    views[2, 1] = 0.0
    public_budget = np.asarray([0.8, 0.7, 0.9])
    executed_budget = np.asarray([0.6, 0.5, 0.4])

    result = replicated_local_row_maxmin_power(
        views, public_budget, executed_budget)

    assert not result.common_view
    assert result.unique_local_problem_count == 3
    assert np.all(result.power_w >= 0.0)
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), executed_budget, atol=1.0e-12)


def test_replicated_local_power_reuses_only_each_valid_private_certificate():
    gain = np.asarray([
        [2.0, 0.4],
        [0.3, 1.5],
    ])
    budget = np.asarray([0.8, 0.7])
    initial_views = np.repeat(gain[None, :, :], 2, axis=0)
    initial = replicated_local_row_maxmin_power(initial_views, budget)
    assert np.all(initial.local_resolved)
    assert np.all(initial.local_cache_valid)

    changed_views = initial_views.copy()
    changed_views[1] = np.asarray([
        [0.05, 4.0],
        [3.5, 0.05],
    ])
    updated = replicated_local_row_maxmin_power(
        changed_views,
        budget,
        previous_local_power_w=initial.local_full_power_w,
        previous_local_prices=initial.local_prices,
        previous_local_cache_valid=initial.local_cache_valid,
        reuse_relative_tolerance=0.01,
    )

    assert updated.local_resolved.tolist() == [False, True]
    assert updated.local_relative_gap[0] <= 0.01
    np.testing.assert_allclose(
        np.sum(updated.power_w, axis=1), budget, atol=1.0e-12)


@pytest.mark.parametrize("seed", range(8))
def test_replicated_private_reuse_respects_relative_optimality_certificate(seed):
    rng = np.random.default_rng(seed)
    K, Q = 3, 3
    old_views = rng.uniform(0.1, 2.0, size=(K, K, Q))
    new_views = old_views * rng.uniform(0.97, 1.03, size=(K, K, Q))
    budget = rng.uniform(0.4, 1.0, size=K)
    initial = replicated_local_row_maxmin_power(old_views, budget)
    updated = replicated_local_row_maxmin_power(
        new_views,
        budget,
        previous_local_power_w=initial.local_full_power_w,
        previous_local_prices=initial.local_prices,
        previous_local_cache_valid=initial.local_cache_valid,
        reuse_relative_tolerance=0.05,
    )

    for viewer in range(K):
        exact = solve_fixed_structure_maxmin_power_lp(
            new_views[viewer], budget)
        assert updated.local_worst_deflection[viewer] + 1.0e-10 >= (
            0.95 * exact.worst_deflection)


def test_replicated_local_rows_use_uniform_cold_start_not_argmax():
    views = np.asarray([[
        [4.0, 1.0, 0.0],
    ]])
    result = replicated_local_row_maxmin_power(
        views, np.asarray([1.0]))

    assert not result.local_full_coverage[0]
    np.testing.assert_allclose(result.power_w, [[1.0 / 3.0] * 3])


def test_replicated_incomplete_view_preserves_unknown_floor_and_budget():
    views = np.asarray([[[4.0, 1.0, 0.0]]])

    result = replicated_local_row_maxmin_power(
        views,
        np.asarray([1.0]),
        unknown_target_reserve_fraction=0.25,
    )

    assert not result.local_full_coverage[0]
    np.testing.assert_allclose(np.sum(result.power_w, axis=1), [1.0])
    # Every target, including the unknown third one, retains beta/Q.
    assert np.all(result.power_w[0] >= 0.25 / 3.0 - 1.0e-12)
    assert result.power_w[0, 2] == pytest.approx(0.25 / 3.0)


def test_replicated_unknown_reserve_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="lie in"):
        replicated_local_row_maxmin_power(
            np.ones((1, 1, 2)), np.ones(1),
            unknown_target_reserve_fraction=1.1)


def test_local_range_minimax_share_equalizes_inverse_square_proxy():
    transmitters = np.asarray([[0.0, 0.0], [2.0, 1.0]])
    targets = np.asarray([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0]])

    share = local_transmitter_range_minimax_share(transmitters, targets)
    range_squared = np.sum(
        (transmitters[:, None, :] - targets[None, :, :]) ** 2, axis=-1)

    np.testing.assert_allclose(np.sum(share, axis=1), 1.0)
    np.testing.assert_allclose(
        share / range_squared,
        np.repeat((share / range_squared)[:, :1], 3, axis=1),
    )


def test_incomplete_view_can_mix_uniform_floor_with_local_range_prior():
    views = np.asarray([[[1.0, 0.0, 0.0]]])
    prior = np.asarray([[0.1, 0.3, 0.6]])
    result = replicated_local_row_maxmin_power(
        views,
        np.asarray([1.0]),
        unknown_target_reserve_fraction=0.5,
        incomplete_view_prior_share=prior,
    )
    np.testing.assert_allclose(
        result.power_w,
        0.5 * np.full((1, 3), 1.0 / 3.0) + 0.5 * prior,
    )


def test_replicated_local_rows_reject_invalid_view_tensor():
    with pytest.raises(ValueError, match="shape"):
        replicated_local_row_maxmin_power(
            np.ones((2, 3, 2)), np.ones(2))


def test_distributed_dual_converges_toward_exact_lp():
    gain = np.asarray([
        [2.0, 0.4, 0.2],
        [0.3, 1.5, 0.8],
        [0.2, 0.5, 1.8],
    ])
    budget = np.asarray([0.8, 0.7, 0.9])
    exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
    distributed = distributed_dual_maxmin_power(
        gain, budget, rounds=4096)
    assert distributed.worst_deflection >= 0.97 * exact.worst_deflection
    np.testing.assert_allclose(
        np.sum(distributed.power_w, axis=1), budget, atol=1e-10)


def test_quantized_price_protocol_remains_feasible():
    gain = np.asarray([[1.0, 0.5], [0.4, 1.2]])
    budget = np.asarray([0.75, 0.8])
    result = distributed_dual_maxmin_power(
        gain, budget, rounds=64, price_bits=6)
    assert np.all(result.power_w >= 0.0)
    np.testing.assert_allclose(np.sum(result.power_w, axis=1), budget, atol=1e-10)
    assert result.primal_dual_gap >= 0.0


def test_harmonic_safe_start_equalizes_constructive_target_floor():
    gain = np.asarray([
        [8.0, 0.2, 1.0],
        [0.5, 3.0, 0.4],
    ])
    budget = np.asarray([0.7, 0.9])
    ceiling = np.sum(gain * budget[:, None], axis=0)
    expected_floor = 1.0 / np.sum(1.0 / ceiling)

    result = distributed_dual_maxmin_power(
        gain, budget, rounds=1, harmonic_safe_start=True)

    # The one-round response may be rejected, but the safe incumbent cannot
    # be lost and obeys every physical per-transmitter RF budget.
    assert result.worst_deflection + 1.0e-12 >= expected_floor
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), budget, atol=1.0e-12)


def test_harmonic_safe_start_eliminates_large_system_zero_target_transient():
    gain = np.eye(8, dtype=np.float64)
    budget = np.linspace(0.4, 1.0, 8)
    result = distributed_dual_maxmin_power(
        gain, budget, rounds=2, price_bits=8, feedback_bits=16)

    assert result.worst_deflection > 0.0
    assert np.all(result.deflection > 0.0)
    assert result.dual_upper_bound + 1.0e-12 >= result.worst_deflection
    assert result.communication_bits == 8 * (2 * (8 + 16) + 16)


def test_distributed_dual_unreachable_target_has_exact_zero_certificate():
    gain = np.asarray([[1.0, 0.0], [0.3, 0.0]])
    result = distributed_dual_maxmin_power(
        gain, np.asarray([0.5, 0.8]), rounds=4,
        price_bits=8, feedback_bits=16)

    assert result.rounds == 0
    assert result.worst_deflection == 0.0
    assert result.dual_upper_bound == 0.0
    np.testing.assert_allclose(result.prices, [0.0, 1.0])


def test_distributed_dual_rejects_undefined_feedback_codec():
    with pytest.raises(ValueError, match="feedback_bits"):
        distributed_dual_maxmin_power(
            np.ones((2, 2)), np.ones(2), rounds=2, feedback_bits=12)


def test_column_generation_has_safe_target_coverage_from_first_round():
    gain = np.asarray([[2.0, 1.0]])
    result = distributed_column_generation_maxmin_power(
        gain, np.asarray([1.0]), rounds=1)
    np.testing.assert_allclose(
        result.power_w, [[1.0 / 3.0, 2.0 / 3.0]], atol=1e-8)
    assert result.worst_deflection == pytest.approx(2.0 / 3.0)


def test_column_generation_enforces_targetwise_reserve_before_maxmin():
    gain = np.asarray([[2.0, 1.0]])
    result = distributed_column_generation_maxmin_power(
        gain,
        np.asarray([1.0]),
        rounds=4,
        minimum_deflection=np.asarray([0.9, 0.4]),
        incumbent_power_w=np.asarray([[0.5, 0.5]]),
    )
    assert result.reserve_feasible
    np.testing.assert_array_less(
        np.asarray([0.9, 0.4]) - 1.0e-10, result.deflection)
    np.testing.assert_allclose(np.sum(result.power_w, axis=1), 1.0)


def test_quantized_column_generation_never_degrades_feasible_incumbent():
    gain = np.asarray([
        [2.3, 0.2, 0.4],
        [0.1, 1.9, 0.3],
        [0.5, 0.2, 2.1],
    ])
    budget = np.asarray([0.75, 0.75, 0.75])
    incumbent = np.full((3, 3), 0.25)
    incumbent_deflection = np.sum(gain * incumbent, axis=0)
    result = distributed_column_generation_maxmin_power(
        gain,
        budget,
        rounds=6,
        price_bits=6,
        feedback_bits=16,
        minimum_deflection=0.8 * incumbent_deflection,
        incumbent_power_w=incumbent,
    )
    assert result.reserve_feasible
    assert result.worst_deflection + 1.0e-12 >= np.min(
        incumbent_deflection)


@pytest.mark.parametrize("seed", range(6))
def test_more_quantized_column_rounds_preserve_best_checkpoint(seed):
    rng = np.random.default_rng(seed)
    gain = rng.uniform(0.05, 2.5, size=(4, 3))
    budget = rng.uniform(0.4, 0.9, size=4)
    four = distributed_column_generation_maxmin_power(
        gain, budget, rounds=4, price_bits=6, feedback_bits=16)
    six = distributed_column_generation_maxmin_power(
        gain, budget, rounds=6, price_bits=6, feedback_bits=16)
    assert six.worst_deflection + 1.0e-12 >= four.worst_deflection


def test_column_generation_reports_infeasible_targetwise_reserve():
    result = distributed_column_generation_maxmin_power(
        np.asarray([[2.0, 1.0]]),
        np.asarray([1.0]),
        rounds=4,
        minimum_deflection=np.asarray([2.0, 0.5]),
    )
    assert not result.reserve_feasible
    assert result.reserve_shortfall > 0.0


def test_column_generation_converges_with_complementary_transmitters():
    gain = np.asarray([
        [2.0, 0.4, 0.2],
        [0.3, 1.5, 0.8],
        [0.2, 0.5, 1.8],
    ])
    budget = np.asarray([0.8, 0.7, 0.9])
    exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
    result = distributed_column_generation_maxmin_power(
        gain, budget, rounds=12)
    assert result.worst_deflection == pytest.approx(
        exact.worst_deflection, rel=1e-7, abs=1e-9)
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), budget, atol=1e-10)
    assert result.dual_upper_bound + 1e-10 >= result.worst_deflection


def test_quantized_column_generation_is_feasible_and_nonzero():
    gain = np.asarray([[1.0, 0.5], [0.4, 1.2]])
    budget = np.asarray([0.75, 0.8])
    result = distributed_column_generation_maxmin_power(
        gain, budget, rounds=4, price_bits=6)
    assert result.worst_deflection > 0.0
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), budget, atol=1e-10)
    assert result.primal_dual_gap >= 0.0


def test_binary16_feedback_column_generation_preserves_rf_feasibility():
    gain = np.asarray([[1.003, 0.497], [0.401, 1.197]])
    budget = np.asarray([0.75, 0.8])
    result = distributed_column_generation_maxmin_power(
        gain, budget, rounds=4, price_bits=6, feedback_bits=16)
    assert result.worst_deflection > 0.0
    np.testing.assert_allclose(
        np.sum(result.power_w, axis=1), budget, atol=1e-10)
    assert result.dual_upper_bound + 1e-12 >= result.worst_deflection


@pytest.mark.parametrize("seed", range(8))
def test_reused_primal_master_duals_preserve_feasibility_and_certificate(seed):
    """Strong-duality reuse may change a degenerate trajectory, not safety."""
    rng = np.random.default_rng(seed)
    gain = rng.uniform(0.05, 2.0, size=(4, 3))
    budget = rng.uniform(0.25, 1.0, size=4)
    baseline = distributed_column_generation_maxmin_power(
        gain,
        budget,
        rounds=8,
        price_bits=6,
        feedback_bits=16,
    )
    reused = distributed_column_generation_maxmin_power(
        gain,
        budget,
        rounds=8,
        price_bits=6,
        feedback_bits=16,
        reuse_primal_master_duals=True,
    )
    np.testing.assert_allclose(
        np.sum(reused.power_w, axis=1), budget, atol=1.0e-10)
    assert np.all(reused.power_w >= 0.0)
    assert reused.dual_upper_bound + 1.0e-10 >= reused.worst_deflection
    assert reused.primal_dual_gap >= 0.0
    # On a non-degenerate random instance, the two optimal price routes should
    # recover the same finite-column lower value up to wire quantization.
    assert reused.worst_deflection == pytest.approx(
        baseline.worst_deflection, rel=2.0e-3, abs=1.0e-8)


def test_column_generation_rejects_undefined_feedback_codec():
    with pytest.raises(ValueError, match="feedback_bits"):
        distributed_column_generation_maxmin_power(
            np.ones((2, 2)), np.ones(2), rounds=2, feedback_bits=12)


def test_fixed_owner_gain_rejects_multiple_receivers():
    coefficient = np.ones((3, 3, 1), dtype=np.float64)
    with pytest.raises(
        NonUniqueFixedOwnerStructureError, match="one receiver"
    ):
        fixed_owner_gain_matrix(
            coefficient, [(0, 1, 0), (2, 0, 0)])


def test_sparse_batched_fixed_owner_gain_matches_dense_collapse():
    selected = [(0, 1, 0), (2, 1, 0), (1, 3, 1), (3, 2, 2)]
    values = np.asarray([
        [0.2, 0.3, 0.4, 0.5],
        [1.2, 1.3, 1.4, 1.5],
    ])
    sparse_gain, sparse_owner = (
        fixed_owner_gain_matrix_from_selected_values(
            values, selected, num_uavs=4, num_targets=3))

    dense_gains = []
    for view in values:
        coefficient = np.zeros((4, 4, 3), dtype=np.float64)
        for edge, value in zip(selected, view):
            coefficient[edge] = value
        dense_gain, dense_owner = fixed_owner_gain_matrix(
            coefficient, selected)
        dense_gains.append(dense_gain)
        np.testing.assert_array_equal(sparse_owner, dense_owner)

    np.testing.assert_array_equal(sparse_gain, np.stack(dense_gains))


def test_relaxed_geometry_ceiling_dominates_fixed_owner_optimum():
    coefficient = np.zeros((3, 3, 2), dtype=np.float64)
    coefficient[0, 1] = [2.0, 0.5]
    coefficient[1, 2] = [0.4, 1.8]
    coefficient[2, 1] = [0.7, 0.9]
    budget = np.asarray([0.8, 0.7, 0.9])
    gain, _ = fixed_owner_gain_matrix(
        coefficient,
        [(0, 1, 0), (2, 1, 0), (1, 2, 1)],
    )
    exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
    ceiling = relaxed_same_geometry_target_ceiling(coefficient, budget)
    assert np.min(ceiling) + 1e-12 >= exact.worst_deflection
    np.testing.assert_allclose(ceiling, [2.51, 2.47])
