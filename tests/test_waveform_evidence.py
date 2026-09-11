import numpy as np
import pytest

from uav_isac.physical.evidence_calibration import (
    calibrate_gaussian_evidence,
    choose_detection_covariance,
    covariance_relative_error,
    fixed_linear_detector_validation,
    subset_surrogate_rank_validation,
)
from uav_isac.physical.waveform_evidence import (
    MinimalOTFSWaveform,
    WaveformEvidenceScenario,
    apply_fractional_dd_path,
    generate_local_evidence_trace,
    otfs_demodulate,
    otfs_modulate,
)


def _scenario(*, fluctuation: float = 0.0) -> WaveformEvidenceScenario:
    count = 4
    return WaveformEvidenceScenario(
        target_delay_bin=np.array([2.0, 2.3, 6.4, 10.1]),
        target_doppler_bin=np.array([1.0, 1.2, -2.1, 2.7]),
        target_amplitude=np.array([0.16, 0.15, 0.14, 0.13]),
        common_clutter_delay_bin=np.full(count, 2.15),
        common_clutter_doppler_bin=np.full(count, 1.1),
        common_clutter_loading=np.array([1.0, 0.9, 0.15, 0.05]),
        common_clutter_std=0.25,
        local_clutter_delay_bin=np.array([3.1, 3.3, 7.2, 9.4]),
        local_clutter_doppler_bin=np.array([0.2, 0.3, -1.3, 3.4]),
        local_clutter_std=np.full(count, 0.04),
        target_secondary_relative_gain=0.25 + 0.1j,
        target_secondary_delay_offset_bin=0.7,
        target_secondary_doppler_offset_bin=-0.35,
        target_fluctuation_std=fluctuation,
    )


def test_otfs_modulation_demodulation_is_unitary_round_trip():
    rng = np.random.default_rng(11)
    dd = rng.normal(size=(3, 8, 16)) + 1j * rng.normal(size=(3, 8, 16))
    time = otfs_modulate(dd)
    recovered = otfs_demodulate(time, doppler_bins=8, delay_bins=16)
    assert np.allclose(recovered, dd, atol=1e-12, rtol=1e-12)
    assert np.allclose(
        np.sum(np.abs(time) ** 2, axis=-1),
        np.sum(np.abs(dd) ** 2, axis=(-2, -1)),
    )


def test_integer_delay_and_fractional_dd_path_preserve_energy():
    rng = np.random.default_rng(12)
    samples = rng.normal(size=128) + 1j * rng.normal(size=128)
    integer = apply_fractional_dd_path(
        samples, delay_bin=3.0, doppler_bin=0.0)
    fractional = apply_fractional_dd_path(
        samples, delay_bin=3.4, doppler_bin=-1.7)
    assert np.allclose(integer, np.roll(samples, 3), atol=1e-12)
    assert np.linalg.norm(fractional) == pytest.approx(
        np.linalg.norm(samples), rel=1e-12)


def test_waveform_trace_is_local_and_exhibits_common_clutter_correlation():
    config = MinimalOTFSWaveform()
    scenario = _scenario()
    h0 = generate_local_evidence_trace(
        config, scenario, trials=5000, hypothesis=0, seed=101)
    h1 = generate_local_evidence_trace(
        config, scenario, trials=5000, hypothesis=1, seed=102)
    calibration = calibrate_gaussian_evidence(h0, h1)
    assert np.all(calibration.mean_shift > 0.5)
    correlation = np.corrcoef(h0, rowvar=False)
    assert correlation[0, 1] > correlation[0, 3] + 0.2
    assert calibration.equal_covariance_relative_error < 0.15


def test_equal_covariance_gate_uses_h0_fallback_for_fluctuating_target():
    config = MinimalOTFSWaveform()
    scenario = _scenario(fluctuation=0.25)
    h0 = generate_local_evidence_trace(
        config, scenario, trials=4000, hypothesis=0, seed=201)
    h1 = generate_local_evidence_trace(
        config, scenario, trials=4000, hypothesis=1, seed=202)
    calibration = calibrate_gaussian_evidence(h0, h1)
    choice = choose_detection_covariance(
        calibration, equal_covariance_tolerance=0.15)
    assert calibration.equal_covariance_relative_error > 0.15
    assert choice.policy == "h0_fixed_pfa_fallback"


def test_bad_conditioning_triggers_diagonal_shrinkage():
    rng = np.random.default_rng(301)
    common0 = rng.normal(size=(3000, 1))
    common1 = rng.normal(size=(3000, 1))
    h0 = common0 + 1.0e-5 * rng.normal(size=(3000, 3))
    h1 = 1.0 + common1 + 1.0e-5 * rng.normal(size=(3000, 3))
    calibration = calibrate_gaussian_evidence(h0, h1)
    choice = choose_detection_covariance(
        calibration,
        equal_covariance_tolerance=0.2,
        maximum_condition_number=100.0,
    )
    assert choice.shrinkage_to_diagonal > 0.0
    assert choice.condition_number_after <= 100.0 * (1.0 + 1e-10)


def test_calibration_threshold_and_subset_surrogate_use_held_out_data_only():
    config = MinimalOTFSWaveform()
    scenario = _scenario()
    cal0 = generate_local_evidence_trace(
        config, scenario, trials=6000, hypothesis=0, seed=401)
    cal1 = generate_local_evidence_trace(
        config, scenario, trials=6000, hypothesis=1, seed=402)
    val0 = generate_local_evidence_trace(
        config, scenario, trials=6000, hypothesis=0, seed=403)
    val1 = generate_local_evidence_trace(
        config, scenario, trials=6000, hypothesis=1, seed=404)
    calibration = calibrate_gaussian_evidence(cal0, cal1)
    choice = choose_detection_covariance(
        calibration, equal_covariance_tolerance=0.15)
    weights = np.linalg.solve(choice.covariance, calibration.mean_shift)
    detector = fixed_linear_detector_validation(
        cal0, val0, val1, weights, p_fa=0.01)
    assert abs(detector["validation_pfa"] - 0.01) < 0.01
    ranking = subset_surrogate_rank_validation(
        calibration, choice, cal0, val0, val1, p_fa=0.01)
    assert ranking["subset_count"] == 15
    assert ranking["spearman_r"] > 0.8
    validation_calibration = calibrate_gaussian_evidence(val0, val1)
    assert covariance_relative_error(
        calibration.covariance_h0,
        validation_calibration.covariance_h0,
    ) < 0.15
