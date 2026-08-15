import numpy as np

from uav_isac.coordination.certified_maxmin_power_controller import (
    CertifiedMaxMinPowerConfig,
    certified_fixed_structure_maxmin_power_repair,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


def _model() -> InterUAVCommunicationModel:
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4],
        header_bits=64,
        bandwidth_hz=100_000.0,
        deadline_s=0.005,
        processing_delay_s=0.0002,
        snr_threshold_db=0.0,
        antenna_gain_dbi=0.0,
        carrier_hz=28.0e9,
        tx_power_w=0.25,
        kT=4.0e-21,
        noise_figure_db=4.0,
        dt=0.1,
    )


def _problem() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    coefficient = np.zeros((3, 3, 2), dtype=np.float64)
    coefficient[0, 2, 0] = 20.0
    coefficient[0, 2, 1] = 5.0
    coefficient[1, 2, 0] = 5.0
    coefficient[1, 2, 1] = 20.0
    selected = [(0, 2, 0), (0, 2, 1), (1, 2, 0), (1, 2, 1)]
    return coefficient, selected


def test_controller_commits_certified_qos_safe_improvement():
    coefficient, selected = _problem()
    positions = np.asarray([
        [0.0, 0.0, 20.0],
        [40.0, 0.0, 20.0],
        [20.0, 30.0, 20.0],
    ])
    comm = np.full(3, 0.1)
    sensing = np.asarray([
        [0.9, 0.0],
        [0.9, 0.0],
        [0.9, 0.0],
    ])
    result = certified_fixed_structure_maxmin_power_repair(
        coefficient,
        coefficient,
        selected,
        positions,
        comm,
        sensing,
        communication_model=_model(),
        control_period_s=0.1,
        p_fa=1.0e-3,
    )
    assert result.accepted
    assert result.reason == "accepted"
    assert result.certified_worst_pd_improvement > 0.0
    assert np.all(result.candidate_lower_pd + 1.0e-9 >= result.protected_pd)
    assert np.all(
        np.sum(result.sensing_power_w, axis=1)
        + result.communication_power_w <= 1.0 + 1.0e-12)
    assert result.transport.feasible
    assert result.optimizer is not None
    assert result.optimizer.dual_upper_bound >= result.optimizer.worst_deflection


def test_controller_fails_closed_when_protocol_misses_control_period():
    coefficient, selected = _problem()
    positions = np.asarray([
        [0.0, 0.0, 20.0],
        [40.0, 0.0, 20.0],
        [20.0, 30.0, 20.0],
    ])
    comm = np.full(3, 0.1)
    sensing = np.full((3, 2), 0.45)
    result = certified_fixed_structure_maxmin_power_repair(
        coefficient,
        coefficient,
        selected,
        positions,
        comm,
        sensing,
        communication_model=_model(),
        control_period_s=1.0e-6,
        p_fa=1.0e-3,
    )
    assert not result.accepted
    assert "transport" in result.reason
    np.testing.assert_array_equal(result.sensing_power_w, sensing)
    np.testing.assert_array_equal(result.communication_power_w, comm)


def test_controller_rejects_improvement_not_proven_under_uncertainty():
    coefficient, selected = _problem()
    lower = 0.25 * coefficient
    upper = 4.0 * coefficient
    positions = np.asarray([
        [0.0, 0.0, 20.0],
        [40.0, 0.0, 20.0],
        [20.0, 30.0, 20.0],
    ])
    comm = np.full(3, 0.1)
    sensing = np.full((3, 2), 0.45)
    result = certified_fixed_structure_maxmin_power_repair(
        lower,
        upper,
        selected,
        positions,
        comm,
        sensing,
        communication_model=_model(),
        control_period_s=0.1,
        p_fa=1.0e-3,
        config=CertifiedMaxMinPowerConfig(qos_floor=0.60),
    )
    assert not result.accepted
    assert (
        "qos_reserve" in result.reason
        or "no_certified_improvement" in result.reason
    )
    np.testing.assert_array_equal(result.sensing_power_w, sensing)
