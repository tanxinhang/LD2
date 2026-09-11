import numpy as np
import pytest

from uav_isac.physical.correlated_soft_evidence import (
    conditional_deflection_gain,
    conditional_information_greedy,
    correlation_unaware_greedy,
    exact_budgeted_selection,
    optimal_linear_soft_fusion,
)


def test_soft_fusion_matches_direct_deflection_and_weights():
    delta = np.array([1.0, 2.0, -0.5])
    covariance = np.array([
        [2.0, 0.3, 0.1],
        [0.3, 1.5, -0.2],
        [0.1, -0.2, 1.0],
    ])
    value, weights = optimal_linear_soft_fusion(
        delta, covariance, selected=(0, 2))
    index = np.array([0, 2])
    expected_weights = np.linalg.solve(
        covariance[np.ix_(index, index)], delta[index])
    assert np.allclose(weights[index], expected_weights)
    assert weights[1] == 0.0
    assert value == pytest.approx(float(delta[index] @ expected_weights))


def test_schur_gain_is_exact_difference_for_random_spd_models():
    rng = np.random.default_rng(20260911)
    for _ in range(20):
        matrix = rng.normal(size=(6, 6))
        covariance = matrix @ matrix.T + 0.5 * np.eye(6)
        delta = rng.normal(size=6)
        selected = (0, 2, 5)
        before, _ = optimal_linear_soft_fusion(
            delta, covariance, selected)
        after, _ = optimal_linear_soft_fusion(
            delta, covariance, selected + (3,))
        gain = conditional_deflection_gain(
            delta, covariance, selected, 3)
        assert gain >= 0.0
        assert after - before == pytest.approx(gain, rel=1e-10, abs=1e-10)


def test_independent_limit_reduces_to_local_deflection():
    delta = np.array([2.0, 3.0, 4.0])
    variance = np.array([2.0, 3.0, 4.0])
    covariance = np.diag(variance)
    for candidate in range(3):
        selected = tuple(index for index in range(3) if index != candidate)
        gain = conditional_deflection_gain(
            delta, covariance, selected, candidate)
        assert gain == pytest.approx(delta[candidate] ** 2 / variance[candidate])


def test_conditional_selection_rejects_high_quality_redundant_peer():
    local_d = np.array([4.0, 3.6, 3.0])
    delta = np.sqrt(local_d)
    covariance = np.array([
        [1.0, 0.95, 0.0],
        [0.95, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    bits = np.array([10, 10, 10])
    proposed = conditional_information_greedy(
        delta, covariance, bits, budget_bits=20)
    unaware = correlation_unaware_greedy(
        delta, covariance, bits, budget_bits=20)
    oracle = exact_budgeted_selection(
        delta, covariance, bits, budget_bits=20)
    assert proposed.selected == (0, 2)
    assert unaware.selected == (0, 1)
    assert proposed.deflection > unaware.deflection
    assert proposed.deflection == pytest.approx(oracle.deflection)


def test_low_correlation_reduces_to_quality_selection():
    delta = np.sqrt(np.array([4.0, 3.6, 3.0, 2.8]))
    covariance = np.eye(4)
    bits = np.full(4, 8)
    proposed = conditional_information_greedy(
        delta, covariance, bits, budget_bits=16)
    unaware = correlation_unaware_greedy(
        delta, covariance, bits, budget_bits=16)
    assert proposed.selected == unaware.selected == (0, 1)
    assert proposed.deflection == pytest.approx(unaware.deflection)


def test_local_initial_evidence_is_free_and_always_retained():
    delta = np.sqrt(np.array([2.0, 4.0, 3.0]))
    covariance = np.eye(3)
    result = conditional_information_greedy(
        delta,
        covariance,
        bits=np.array([0, 8, 8]),
        budget_bits=8,
        initial_selected=(0,),
    )
    assert result.selected == (0, 1)
    assert result.communication_bits == 8
    assert result.deflection == pytest.approx(6.0)


def test_native_deadline_and_reliability_affect_schedule_without_unit_mixing():
    delta = np.sqrt(np.array([4.0, 3.0, 2.0]))
    covariance = np.eye(3)
    result = conditional_information_greedy(
        delta,
        covariance,
        bits=np.array([8, 8, 8]),
        budget_bits=8,
        success_probability=np.array([0.1, 1.0, 1.0]),
        latency_s=np.array([0.001, 0.006, 0.002]),
        deadline_s=0.005,
    )
    # Source 1 has higher quality than source 2 but violates the hard deadline;
    # source 0 is eligible but its delivery-weighted gain is smaller.
    assert result.selected == (2,)


def test_invalid_or_singular_covariance_fails_closed():
    with pytest.raises(ValueError, match="positive definite"):
        optimal_linear_soft_fusion(
            np.ones(2), np.array([[1.0, 1.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        optimal_linear_soft_fusion(
            np.ones(2), np.array([[1.0, 0.2], [0.1, 1.0]]))


def test_exact_oracle_respects_heterogeneous_bit_budget():
    delta = np.sqrt(np.array([5.0, 4.0, 3.0]))
    covariance = np.eye(3)
    result = exact_budgeted_selection(
        delta, covariance, bits=np.array([9, 5, 5]), budget_bits=10)
    assert result.selected == (1, 2)
    assert result.communication_bits == 10
    assert result.deflection == pytest.approx(7.0)


def test_fractional_bit_counts_are_not_silently_truncated():
    with pytest.raises(ValueError, match="integer"):
        conditional_information_greedy(
            np.ones(2), np.eye(2), np.array([4.5, 8.0]), budget_bits=8)
    with pytest.raises(ValueError, match="budget"):
        exact_budgeted_selection(
            np.ones(2), np.eye(2), np.array([4, 8]), budget_bits=8.5)
