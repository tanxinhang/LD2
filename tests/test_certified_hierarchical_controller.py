import numpy as np

from uav_isac.coordination.bottleneck_router import RepairRoute
from uav_isac.coordination.certified_hierarchical_controller import (
    CertifiedHierarchicalControllerConfig,
    certified_hierarchical_isac_control,
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


def _positions(count: int) -> np.ndarray:
    return np.asarray([
        [30.0 * (index % 2), 30.0 * (index // 2), 20.0]
        for index in range(count)
    ])


def test_hierarchical_default_separates_four_round_ranking_from_six_round_execution():
    config = CertifiedHierarchicalControllerConfig()
    assert config.structure_ranking_rounds == 4
    assert config.power.rounds == 6


def test_controller_routes_fixed_structure_power_and_commits():
    coefficient = np.zeros((3, 3, 2), dtype=np.float64)
    coefficient[0, 2] = [20.0, 5.0]
    coefficient[1, 2] = [5.0, 20.0]
    selected = np.zeros_like(coefficient, dtype=bool)
    selected[0, 2] = True
    selected[1, 2] = True
    sensing = np.asarray([
        [0.75, 0.0],
        [0.75, 0.0],
        [0.75, 0.0],
    ])
    result = certified_hierarchical_isac_control(
        coefficient,
        coefficient,
        selected,
        np.asarray([1, 1, 0], dtype=np.int8),
        _positions(3),
        np.full(3, 0.25),
        sensing,
        communication_model=_model(),
        control_period_s=0.1,
        p_fa=0.1,
        config=CertifiedHierarchicalControllerConfig(
            target_pair_limit=2,
            reports_per_receiver=4,
        ),
    )
    assert result.route == RepairRoute.POWER
    assert result.accepted
    assert result.certified_worst_pd_improvement > 0.0
    assert result.protocol_resources is not None
    assert result.protocol_resources.feasible


def test_controller_routes_structure_through_owner_bid_and_atomic_commit():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    coefficient[0, 2, 0] = 10.0
    coefficient[1, 2, 1] = 1.0
    coefficient[1, 3, 1] = 10.0
    selected = np.zeros_like(coefficient, dtype=bool)
    selected[0, 2, 0] = True
    selected[1, 2, 1] = True
    sensing = np.full((4, 2), 0.375)
    result = certified_hierarchical_isac_control(
        coefficient,
        coefficient,
        selected,
        np.asarray([1, 1, 0, 0], dtype=np.int8),
        _positions(4),
        np.full(4, 0.25),
        sensing,
        communication_model=_model(),
        control_period_s=0.1,
        p_fa=0.1,
        config=CertifiedHierarchicalControllerConfig(
            target_pair_limit=1,
            reports_per_receiver=4,
            owners_per_target=3,
            structure_weak_target_count=2,
        ),
    )
    assert result.route == RepairRoute.STRUCTURE_POWER
    assert result.accepted
    assert result.owner[1] == 3
    assert result.owner_bid_transport is not None
    assert result.owner_bid_transport.feasible
    assert result.structure_transport is not None
    assert result.structure_transport.feasible
    assert result.protocol_resources is not None
    assert result.protocol_resources.total_protocol_latency_s <= 0.1
    assert result.protocol_resources.max_isac_power_balance_error_w <= 1e-12


def test_controller_routes_to_geometry_without_unsafe_structure_commit():
    coefficient = np.zeros((3, 3, 2), dtype=np.float64)
    coefficient[0, 2, 0] = 0.1
    coefficient[1, 2, 1] = 0.1
    selected = coefficient > 0.0
    sensing = np.full((3, 2), 0.375)
    result = certified_hierarchical_isac_control(
        coefficient,
        coefficient,
        selected,
        np.asarray([1, 1, 0], dtype=np.int8),
        _positions(3),
        np.full(3, 0.25),
        sensing,
        communication_model=_model(),
        control_period_s=0.1,
        p_fa=0.1,
        config=CertifiedHierarchicalControllerConfig(
            target_pair_limit=1,
            reports_per_receiver=2,
        ),
    )
    assert result.route == RepairRoute.GEOMETRY
    assert not result.accepted
    np.testing.assert_array_equal(result.selected, selected)
    np.testing.assert_array_equal(result.sensing_power_w, sensing)


def test_controller_fails_closed_when_control_power_was_not_reserved():
    coefficient = np.zeros((3, 3, 2), dtype=np.float64)
    coefficient[0, 2] = [20.0, 5.0]
    coefficient[1, 2] = [5.0, 20.0]
    selected = coefficient > 0.0
    comm = np.full(3, 1.0e-12)
    sensing = np.asarray([
        [1.0 - 1.0e-12, 0.0],
        [1.0 - 1.0e-12, 0.0],
        [1.0 - 1.0e-12, 0.0],
    ])
    result = certified_hierarchical_isac_control(
        coefficient,
        coefficient,
        selected,
        np.asarray([1, 1, 0], dtype=np.int8),
        _positions(3),
        comm,
        sensing,
        communication_model=_model(),
        control_period_s=0.1,
        p_fa=0.1,
        config=CertifiedHierarchicalControllerConfig(
            target_pair_limit=2,
            reports_per_receiver=4,
            require_pre_reserved_comm_power=True,
        ),
    )
    assert not result.accepted
    assert "unreserved_comm" in result.reason
    np.testing.assert_array_equal(result.communication_power_w, comm)
