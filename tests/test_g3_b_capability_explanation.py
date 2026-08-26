import pytest

from tools.audit_g3_b_capability_explanation import spearman


def test_spearman_handles_ties_and_monotone_direction_without_blas():
    assert spearman([1, 2, 2, 4], [10, 20, 20, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
