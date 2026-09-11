import numpy as np
import pytest

from uav_isac.physical.correlation_calibration import (
    apply_target_correlation_factors,
    calibrate_selected_entries,
    otfs_target_correlation_factor_from_arrays,
    otfs_template_gram,
    selected_otfs_correlation_factors,
    selected_otfs_correlation_factors_from_arrays,
)
from uav_isac.utils.types import DeflectionEntry


def _entry(i, j, q, tau, nu, d_eff=1.0):
    return DeflectionEntry(i, j, q, tau, nu, 1.0, d_eff, 1.0, 1.0, d_eff)


def test_otfs_template_gram_is_hermitian_psd_with_unit_diagonal():
    gram = otfs_template_gram(
        np.array([0.1, 0.35, 3.7]), np.array([-0.2, 0.4, 1.1]),
        delay_size=64, doppler_size=16,
    )
    np.testing.assert_allclose(gram, gram.conj().T, atol=1e-14)
    np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-14)
    assert np.min(np.linalg.eigvalsh(gram)) >= -1e-12


def test_identical_templates_reduce_additive_deflection_to_one_look():
    entries = [
        _entry(0, 2, 0, 1e-6, 100.0, 4.0),
        _entry(1, 2, 0, 1e-6, 100.0, 6.0),
        _entry(2, 1, 1, 2e-6, -50.0, 3.0),
    ]
    selected = [(0, 2, 0), (1, 2, 0), (2, 1, 1)]
    factors = selected_otfs_correlation_factors(
        selected, entries, num_targets=2,
        delay_size=64, doppler_size=16,
        delta_f_hz=15_625.0, symbol_time_s=64e-6,
    )
    np.testing.assert_allclose(factors, [2.0, 1.0], atol=1e-12)
    gain = apply_target_correlation_factors(
        np.array([[4.0, 0.0], [6.0, 0.0], [0.0, 3.0]]), factors)
    np.testing.assert_allclose(np.sum(gain, axis=0), [5.0, 3.0])
    calibrated = calibrate_selected_entries(entries, selected, factors)
    assert sum(entry.d_eff for entry in calibrated if entry.q == 0) == pytest.approx(5.0)


def test_well_separated_integer_bin_templates_have_no_penalty():
    # One delay-bin separation is orthogonal on the length-M finite grid.
    delta_tau = 1.0 / (64 * 15_625.0)
    entries = [
        _entry(0, 2, 0, 0.0, 0.0),
        _entry(1, 2, 0, delta_tau, 0.0),
    ]
    factors = selected_otfs_correlation_factors(
        [(0, 2, 0), (1, 2, 0)], entries, num_targets=1,
        delay_size=64, doppler_size=16,
        delta_f_hz=15_625.0, symbol_time_s=64e-6,
    )
    np.testing.assert_allclose(factors, [1.0], atol=1e-12)


def test_zero_effective_edge_does_not_create_phantom_correlation():
    entries = [
        _entry(0, 2, 0, 1e-6, 0.0, 1.0),
        _entry(1, 2, 0, 1e-6, 0.0, 0.0),
    ]
    factors = selected_otfs_correlation_factors(
        [(0, 2, 0), (1, 2, 0)], entries, num_targets=1,
        delay_size=64, doppler_size=16,
        delta_f_hz=15_625.0, symbol_time_s=64e-6,
    )
    np.testing.assert_array_equal(factors, [1.0])


def test_duplicate_selected_edge_is_rejected():
    entry = _entry(0, 1, 0, 0.0, 0.0)
    with pytest.raises(ValueError, match="duplicate"):
        selected_otfs_correlation_factors(
            [(0, 1, 0), (0, 1, 0)], [entry], num_targets=1,
            delay_size=64, doppler_size=16,
            delta_f_hz=15_625.0, symbol_time_s=64e-6,
        )


def test_array_native_factors_use_unit_coefficient_not_current_power():
    selected = np.zeros((3, 3, 1), dtype=bool)
    coefficient = np.zeros_like(selected, dtype=np.float64)
    tau = np.zeros_like(coefficient)
    nu = np.zeros_like(coefficient)
    selected[0, 2, 0] = True
    selected[1, 2, 0] = True
    coefficient[0, 2, 0] = 4.0
    coefficient[1, 2, 0] = 6.0
    factors = selected_otfs_correlation_factors_from_arrays(
        selected, coefficient, tau, nu,
        delay_size=64, doppler_size=16,
        delta_f_hz=15_625.0, symbol_time_s=64e-6,
    )
    np.testing.assert_allclose(factors, [2.0], atol=1e-12)
    target_factor = otfs_target_correlation_factor_from_arrays(
        selected[:, :, 0], tau[:, :, 0], nu[:, :, 0],
        delay_size=64, doppler_size=16,
        delta_f_hz=15_625.0, symbol_time_s=64e-6,
    )
    assert target_factor == pytest.approx(factors[0], abs=1e-12)
