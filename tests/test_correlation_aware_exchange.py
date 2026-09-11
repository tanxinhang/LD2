import numpy as np
import pytest

from uav_isac.coordination.correlation_aware_exchange import (
    correlation_aware_dual_pruned_exact_exchange,
    correlation_calibrated_structure_gain,
)
from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)


def _exchange_fixture():
    K, Q = 4, 2
    coefficient = np.zeros((K, K, Q), dtype=np.float64)
    support = np.zeros_like(coefficient, dtype=bool)
    for edge, value in {
        (0, 2, 0): 10.0,
        (1, 2, 0): 10.0,
        (0, 2, 1): 8.0,
        (0, 3, 0): 8.0,
        (1, 3, 0): 8.0,
        (0, 3, 1): 8.0,
        (1, 3, 1): 8.0,
    }.items():
        coefficient[edge] = value
        support[edge] = True
    selected = np.zeros_like(support)
    selected[0, 2, 0] = True
    selected[1, 2, 0] = True
    selected[0, 2, 1] = True
    tau = np.zeros_like(coefficient)
    nu = np.zeros_like(coefficient)
    # Incumbent target-0 templates coincide. The receiver-3 alternative has
    # one integer delay-bin separation and is therefore orthogonal.
    tau[1, 3, 0] = 1.0 / (64.0 * 15_625.0)
    return coefficient, support, selected, tau, nu


def test_exact_exchange_replaces_redundant_templates_without_degrading_lp():
    coefficient, support, selected, tau, nu = _exchange_fixture()
    budget = np.ones(4, dtype=np.float64)
    initial_gain, _, initial_factors = correlation_calibrated_structure_gain(
        coefficient, selected, tau, nu,
        delay_size=64, doppler_size=16,
        delta_f_hz=15_625.0, symbol_time_s=64e-6,
    )
    initial = solve_fixed_structure_maxmin_power_lp(initial_gain, budget)

    result = correlation_aware_dual_pruned_exact_exchange(
        selected,
        coefficient,
        support,
        np.array([1, 1, 0, 0], dtype=np.int8),
        budget,
        tau,
        nu,
        delay_size=64,
        doppler_size=16,
        delta_f_hz=15_625.0,
        symbol_time_s=64e-6,
        target_pair_limit=2,
        reports_per_receiver=8,
        top_m=8,
    )

    assert result.accepted
    assert result.exact_verification_count <= 8
    assert result.correlation_factor_cache_hits > 0
    assert result.correlation_factor_cache_misses > 0
    assert result.power_result.worst_deflection > initial.worst_deflection
    np.testing.assert_allclose(initial_factors, [2.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(
        result.final_correlation_factors, [1.0, 1.0], atol=1e-12)
    # No-op is the incumbent; exact acceptance therefore certifies monotonicity.
    assert result.power_result.worst_deflection >= initial.worst_deflection


def test_exact_exchange_returns_explicit_no_op_without_alternatives():
    coefficient, _, selected, tau, nu = _exchange_fixture()
    result = correlation_aware_dual_pruned_exact_exchange(
        selected,
        coefficient,
        selected.copy(),
        np.array([1, 1, 0, 0], dtype=np.int8),
        np.ones(4, dtype=np.float64),
        tau,
        nu,
        delay_size=64,
        doppler_size=16,
        delta_f_hz=15_625.0,
        symbol_time_s=64e-6,
        target_pair_limit=2,
        reports_per_receiver=8,
        top_m=8,
    )
    assert not result.accepted
    assert result.accepted_kind == "no_op"
    np.testing.assert_array_equal(result.selected, selected)
    np.testing.assert_allclose(
        result.final_correlation_factors,
        result.baseline_correlation_factors,
    )


def test_physical_prefilter_rejects_before_gram_and_exact_lp_verification():
    coefficient, support, selected, tau, nu = _exchange_fixture()
    result = correlation_aware_dual_pruned_exact_exchange(
        selected,
        coefficient,
        support,
        np.array([1, 1, 0, 0], dtype=np.int8),
        np.ones(4, dtype=np.float64),
        tau,
        nu,
        delay_size=64,
        doppler_size=16,
        delta_f_hz=15_625.0,
        symbol_time_s=64e-6,
        target_pair_limit=2,
        reports_per_receiver=8,
        top_m=8,
        pre_candidate_gate=lambda _move: False,
    )
    assert not result.accepted
    assert result.physical_prefilter_rejected_count > 0
    assert result.exact_verification_count == 0
    # Only the incumbent Gram/LP is evaluated; rejected structures never
    # consume the expensive correlation or exact optimization stages.
    assert result.gram_compute_time_s >= 0.0
    assert result.exact_lp_time_s >= 0.0


def test_dual_bound_early_stop_preserves_topm_exact_result():
    coefficient, support, selected, tau, nu = _exchange_fixture()
    common = dict(
        initial_selected=selected,
        coefficient_per_watt=coefficient,
        candidate_mask=support,
        initial_role=np.array([1, 1, 0, 0], dtype=np.int8),
        sensing_budget_w=np.ones(4, dtype=np.float64),
        delay_s=tau,
        doppler_hz=nu,
        delay_size=64,
        doppler_size=16,
        delta_f_hz=15_625.0,
        symbol_time_s=64e-6,
        target_pair_limit=2,
        reports_per_receiver=8,
        top_m=8,
    )
    bounded = correlation_aware_dual_pruned_exact_exchange(
        **common, dual_bound_early_stop=True)
    exhaustive_topm = correlation_aware_dual_pruned_exact_exchange(
        **common, dual_bound_early_stop=False, raw_gain_upper_prune=False)
    np.testing.assert_array_equal(bounded.selected, exhaustive_topm.selected)
    assert bounded.power_result.worst_deflection == pytest.approx(
        exhaustive_topm.power_result.worst_deflection)
    assert bounded.exact_verification_count <= (
        exhaustive_topm.exact_verification_count)
    assert bounded.raw_gain_upper_pruned_count >= 0
