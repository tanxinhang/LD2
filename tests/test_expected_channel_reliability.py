import numpy as np
import pytest

from uav_isac.physical.channel import (
    expected_report_link_reliability,
    expected_rician_soft_reliability,
)


@pytest.mark.parametrize("mean_snr", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_gauss_hermite_rician_expectation_matches_monte_carlo(mean_snr):
    k_db = 6.0
    expected = expected_rician_soft_reliability(
        mean_snr, k_db, quadrature_order=20)
    rng = np.random.default_rng(20260910)
    count = 500_000
    k_linear = 10.0 ** (k_db / 10.0)
    mean_component = np.sqrt(k_linear / (k_linear + 1.0))
    component_std = np.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
    real = mean_component + component_std * rng.normal(size=count)
    imag = component_std * rng.normal(size=count)
    instantaneous = mean_snr * (real * real + imag * imag)
    monte_carlo = np.mean(instantaneous / (instantaneous + 1.0))
    assert expected == pytest.approx(monte_carlo, abs=1.5e-3)


def test_expected_reliability_is_bounded_and_monotone():
    values = [expected_rician_soft_reliability(value, 6.0)
              for value in (0.0, 0.1, 1.0, 10.0)]
    assert values[0] == 0.0
    assert all(0.0 <= value <= 1.0 for value in values)
    assert np.all(np.diff(values) > 0.0)


def test_expected_report_link_is_deterministic_and_rng_free():
    args = dict(
        rx_uav_pos=np.asarray([100.0, 150.0, 20.0]),
        fc_position=np.asarray([200.0, 200.0, 0.0]),
        fc=28.0e9,
        K_dB=6.0,
        noise_power=1.0e-14,
        P_report=0.25,
        use_los_prob=True,
    )
    first = expected_report_link_reliability(**args)
    second = expected_report_link_reliability(**args)
    assert first == second
    assert 0.0 < first < 1.0


def test_expected_reliability_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="mean_snr"):
        expected_rician_soft_reliability(-1.0, 6.0)
    with pytest.raises(ValueError, match="quadrature"):
        expected_rician_soft_reliability(1.0, 6.0, quadrature_order=2)
    with pytest.raises(ValueError, match="quadrature"):
        expected_rician_soft_reliability(1.0, 6.0, quadrature_order=3.5)
