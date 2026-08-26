import numpy as np

from uav_isac.coordination.coefficient_structure import (
    balance_bistatic_factor_gauge,
    exact_bistatic_factorization,
    spectral_structure,
)


def test_exact_outer_product_is_rank_one():
    matrix = np.outer(np.array([1.0, 2.0, 4.0]), np.array([3.0, 5.0, 7.0]))
    result = spectral_structure(matrix)
    assert result.rank_90 == result.rank_95 == result.rank_99 == 1
    assert np.isclose(result.stable_rank, 1.0)
    assert np.isclose(result.entropy_effective_rank, 1.0)
    assert result.rank1_relative_frobenius_error < 1.0e-8


def test_architecture_mask_can_destroy_algebraic_rank_one_structure():
    matrix = np.ones((4, 4))
    np.fill_diagonal(matrix, 0.0)
    result = spectral_structure(matrix)
    assert result.rank_90 > 1
    assert result.rank1_relative_frobenius_error > 0.0


def test_zero_matrix_has_zero_effective_rank():
    result = spectral_structure(np.zeros((3, 3)))
    assert result.rank_90 == result.rank_95 == result.rank_99 == 0
    assert result.stable_rank == result.entropy_effective_rank == 0.0


def test_exact_bistatic_outer_product_plus_sparse_dd_reconstruction():
    factor = np.array([1.0, 2.0, 4.0, 5.0])
    path = np.outer(factor, factor)
    np.fill_diagonal(path, 0.0)
    report_vector = np.array([0.7, 0.8, 0.9, 1.0])
    report = np.broadcast_to(report_vector[None, :], path.shape).copy()
    active = np.ones_like(path, dtype=bool)
    active[0, 2] = False
    result = exact_bistatic_factorization(path, report, active)
    expected = path * report * active
    assert np.allclose(result.dense(), expected)
    assert result.inactive_offdiagonal == ((0, 2),)


def test_non_receiver_only_reporting_is_rejected():
    factor = np.array([1.0, 2.0, 3.0])
    path = np.outer(factor, factor)
    np.fill_diagonal(path, 0.0)
    report = np.ones_like(path)
    report[0, 1] = 0.5
    with np.testing.assert_raises(ValueError):
        exact_bistatic_factorization(path, report, np.ones_like(path, dtype=bool))


def test_log_center_gauge_balance_preserves_dense_matrix_and_reduces_range():
    factor = exact_bistatic_factorization(
        np.array([[0.0, 2.0, 3.0], [2.0, 0.0, 6.0], [3.0, 6.0, 0.0]]),
        np.full((3, 3), 1.0e12), np.ones((3, 3), dtype=bool))
    balanced = balance_bistatic_factor_gauge(factor)
    before = np.ptp(np.log(np.concatenate([factor.tx_factor, factor.rx_factor])))
    after = np.ptp(np.log(np.concatenate([balanced.tx_factor, balanced.rx_factor])))
    assert np.allclose(balanced.dense(), factor.dense())
    assert after < before
