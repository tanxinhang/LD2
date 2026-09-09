import copy

import numpy as np

from uav_isac.physical.deflection import DeflectionComputer


def _computer(*, report: bool, swerling: bool, seed: int = 19):
    return DeflectionComputer(
        fc=2.8e10, delta_f=1.5625e4, T_sym=6.4e-5, M=64, N=16,
        kT=4.0e-21, B=1.0e6, NF_dB=4.0,
        P_sense=0.0251, P_report=0.25, ric_K=6.0,
        rcs=1.0, g_min=0.5, rng=np.random.default_rng(seed),
        g_tx_dBi=16.0, g_rx_dBi=16.0,
        use_los_prob=True, use_swerling=swerling,
        use_report_link=report, dd_gain_mode="continuous",
        sync_delay_error_bins=0.13, sync_doppler_error_bins=-0.21,
    )


def _state():
    uav_positions = np.array([
        [100.0, 100.0, 100.0], [800.0, 150.0, 100.0],
        [200.0, 750.0, 100.0], [850.0, 800.0, 100.0],
    ])
    uav_velocities = np.array([
        [2.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [-1.0, 0.5, 0.0], [0.0, -2.0, 0.0],
    ])
    target_positions = np.array([
        [400.0, 450.0, 0.0], [650.0, 300.0, 0.0],
    ])
    target_velocities = np.array([
        [5.0, -2.0, 0.0], [-3.0, 4.0, 0.0],
    ])
    roles = np.array([0, 0, 1, 1], dtype=np.int64)
    fusion = np.array([500.0, 500.0, 0.0])
    return (
        uav_positions, uav_velocities, target_positions,
        target_velocities, roles, fusion,
    )


def test_expected_dense_matches_deterministic_dense_when_random_effects_off():
    dc = _computer(report=False, swerling=False)
    args = _state()
    deterministic = dc.compute_dense(*args)
    expected = dc.compute_expected_dense(*args)
    for field in ("tau", "nu", "alpha", "d_raw", "g_dd", "d_eff"):
        np.testing.assert_allclose(
            getattr(expected, field), getattr(deterministic, field),
            rtol=1.0e-14, atol=0.0,
        )
    np.testing.assert_array_equal(expected.valid, deterministic.valid)


def test_expected_dense_with_report_and_swerling_is_rng_free_and_repeatable():
    dc = _computer(report=True, swerling=True)
    args = _state()
    rng_before = copy.deepcopy(dc.rng.bit_generator.state)
    first = dc.compute_expected_dense(*args, quadrature_order=20)
    rng_after_first = copy.deepcopy(dc.rng.bit_generator.state)
    second = dc.compute_expected_dense(*args, quadrature_order=20)
    rng_after_second = copy.deepcopy(dc.rng.bit_generator.state)

    assert rng_before == rng_after_first == rng_after_second
    np.testing.assert_array_equal(first.d_eff, second.d_eff)
    assert np.all(first.d_eff[first.valid] >= 0.0)
    assert np.all(first.d_eff[~first.valid] == 0.0)
    assert np.all(first.d_eff[first.valid] <= first.d_raw[first.valid])


def test_expected_swerling_mean_is_the_no_fading_raw_deflection():
    plain = _computer(report=False, swerling=False)
    faded = _computer(report=False, swerling=True)
    args = _state()
    plain_dense = plain.compute_dense(*args)
    expected_faded = faded.compute_expected_dense(*args)
    np.testing.assert_array_equal(expected_faded.d_raw, plain_dense.d_raw)
    np.testing.assert_allclose(
        expected_faded.d_eff, plain_dense.d_eff,
        rtol=1.0e-14, atol=0.0,
    )
