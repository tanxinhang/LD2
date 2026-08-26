import pytest

from tools.audit_scale_worst_gap import _correlation, poisson_worst_nearest_median


def test_poisson_worst_nearest_increases_with_target_count_at_fixed_density():
    values = [poisson_worst_nearest_median(density_m2=6.25e-6, targets=q) for q in (4, 6, 8)]
    assert values[0] < values[1] < values[2]


def test_correlation_handles_degenerate_and_detects_direction():
    assert _correlation([1, 1, 1], [1, 2, 3]) is None
    assert _correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
