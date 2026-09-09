import numpy as np
import pytest

from uav_isac.acceleration.hyperedge import (
    HyperedgeAccelerationService,
    NumpyHyperedgeAccelerationService,
    available_hyperedge_acceleration_backends,
    create_hyperedge_acceleration_service,
    register_hyperedge_acceleration_backend,
)
from uav_isac.coordination.hyperedge import (
    reconstruct_bistatic_coefficient_from_public_state,
)


def _dense_inputs():
    positions = np.asarray([
        [[100.0, 120.0], [130.0, 150.0]],
        [[220.0, 210.0], [250.0, 260.0]],
        [[320.0, 300.0], [340.0, 360.0]],
    ])
    velocities = np.zeros_like(positions)
    targets = np.asarray([[180.0, 190.0, 0.0], [280.0, 290.0, 0.0]])
    target_velocities = np.zeros_like(targets)
    visible = np.ones((3, 2), dtype=bool)
    parameters = dict(
        uav_height_m=20.0,
        fc_hz=28.0e9,
        rcs_m2=1.0,
        delta_f_hz=15625.0,
        symbol_period_s=6.4e-5,
        delay_bins=64,
        doppler_bins=16,
        dd_gate_min=0.5,
        coefficient_scale=1.0e15,
        position_uncertainty_m=np.zeros(3),
        robust_dd_uncertainty=False,
        dd_gain_mode="continuous",
    )
    return positions, velocities, targets, target_velocities, visible, parameters


def test_numpy_service_matches_reference_and_reports_resettable_timing():
    service = NumpyHyperedgeAccelerationService(collect_timing=True)
    assert isinstance(service, HyperedgeAccelerationService)
    assert "numpy" in available_hyperedge_acceleration_backends()
    args = _dense_inputs()

    actual = service.reconstruct_dense(*args[:-1], **args[-1])
    expected = reconstruct_bistatic_coefficient_from_public_state(
        *args[:-1], **args[-1])
    np.testing.assert_array_equal(actual, expected)

    stats = service.stats()
    assert stats["backend"] == "numpy"
    assert stats["dense_calls"] == 1
    assert stats["total_calls"] == 1
    assert stats["timing_enabled"] == 1
    assert stats["total_seconds"] >= 0.0
    service.reset_stats()
    assert service.stats()["total_calls"] == 0


def test_service_registry_and_external_factory_are_drop_in():
    register_hyperedge_acceleration_backend(
        "test_numpy_service",
        NumpyHyperedgeAccelerationService,
        replace=True,
    )
    registered = create_hyperedge_acceleration_service("TEST_NUMPY_SERVICE")
    external = create_hyperedge_acceleration_service(
        "uav_isac.acceleration.hyperedge:NumpyHyperedgeAccelerationService")
    assert isinstance(registered, HyperedgeAccelerationService)
    assert isinstance(external, HyperedgeAccelerationService)


def test_service_factory_rejects_unknown_or_invalid_backend():
    with pytest.raises(ValueError, match="unknown hyperedge acceleration"):
        create_hyperedge_acceleration_service("does_not_exist")
    with pytest.raises(ValueError, match="cannot load"):
        create_hyperedge_acceleration_service("does.not.exist:create")
