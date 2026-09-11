import numpy as np
import pytest

from uav_isac.physical.bistatic_waveform import (
    aggregate_edge_components_at_receivers,
    bistatic_geometry_to_waveform,
    ideal_coherent_h0_deflection,
    real_equivalent_complex_noise_variance,
    selected_edge_mask,
)
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform


def _physical_case(range_scale: float = 1.0, power_w: float = 0.025):
    uav = range_scale * np.array([
        [-100.0, 0.0, 20.0],
        [100.0, 0.0, 20.0],
    ])
    target = np.array([[0.0, 0.0, 0.0]])
    velocity = np.zeros_like(uav)
    target_velocity = np.zeros_like(target)
    roles = np.array([0, 1])
    power = np.array([[power_w], [0.0]])
    waveform = MinimalOTFSWaveform(
        delay_bins=64, doppler_bins=16, delta_f_hz=15_625.0)
    return bistatic_geometry_to_waveform(
        uav,
        velocity,
        target,
        target_velocity,
        roles,
        power,
        carrier_hz=28.0e9,
        waveform=waveform,
        rcs_m2=1.0,
        tx_gain_dbi=16.0,
        rx_gain_dbi=16.0,
        require_unambiguous=True,
    )


def test_physical_bridge_obeys_bistatic_range_and_power_laws():
    near = _physical_case(range_scale=1.0, power_w=0.025)
    far = _physical_case(range_scale=2.0, power_w=0.025)
    double_power = _physical_case(range_scale=1.0, power_w=0.05)
    edge = (0, 1, 0)
    # Both bistatic legs scale approximately by two (the fixed target height is
    # negligible but retained), so compare the exact radar-equation ratio.
    expected_ratio = far.path_amplitude[edge] / near.path_amplitude[edge]
    assert expected_ratio < 0.26
    assert expected_ratio > 0.24
    assert (
        far.received_target_amplitude[edge]
        / near.received_target_amplitude[edge]
    ) == pytest.approx(expected_ratio)
    assert (
        double_power.received_target_amplitude[edge] ** 2
        / near.received_target_amplitude[edge] ** 2
    ) == pytest.approx(2.0)


def test_delay_and_doppler_bins_follow_otfs_resolution():
    result = _physical_case()
    edge = (0, 1, 0)
    assert result.delay_bin[edge] == pytest.approx(
        result.delay_s[edge] * 64 * 15_625.0)
    assert result.doppler_bin[edge] == pytest.approx(
        result.doppler_hz[edge] * 16 / 15_625.0)
    assert result.valid_edge_mask[edge]
    assert result.unambiguous_edge_mask[edge]


def test_complex_waveform_noise_matches_frozen_real_detector_convention():
    result = _physical_case()
    edge = (0, 1, 0)
    real_noise_power = 3.2e-14
    complex_variance = real_equivalent_complex_noise_variance(real_noise_power)
    waveform = MinimalOTFSWaveform(
        delay_bins=64, doppler_bins=16, delta_f_hz=15_625.0)
    waveform_d0 = ideal_coherent_h0_deflection(
        result.received_target_amplitude[edge],
        waveform=waveform,
        complex_noise_variance=complex_variance,
    )
    expected = (
        waveform.sample_count
        * result.received_target_amplitude[edge] ** 2
        / real_noise_power
    )
    assert float(waveform_d0) == pytest.approx(expected)


def test_receiver_only_sees_sum_of_scheduled_edge_components():
    mask = selected_edge_mask((3, 3, 2), [(0, 1, 0), (2, 1, 1)])
    components = np.zeros((3, 3, 2, 4), dtype=np.complex128)
    components[0, 1, 0] = 1.0 + 2.0j
    components[2, 1, 1] = np.arange(4)
    received = aggregate_edge_components_at_receivers(components, mask)
    assert received.shape == (3, 4)
    assert np.allclose(received[1], 1.0 + 2.0j + np.arange(4))
    assert np.all(received[[0, 2]] == 0.0)


def test_unscheduled_edge_cannot_leak_into_receiver_observation():
    mask = selected_edge_mask((2, 2, 1), [(0, 1, 0)])
    components = np.zeros((2, 2, 1, 3))
    components[1, 0, 0] = 1.0
    with pytest.raises(ValueError, match="unscheduled"):
        aggregate_edge_components_at_receivers(components, mask)
