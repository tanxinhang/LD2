import numpy as np

from uav_isac.coordination.geometry_repair_transport import (
    GeometryRepairWireLayout,
    certify_geometry_repair_transport,
    quantize_displacement_toward_zero,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


def _model(deadline_s=0.005):
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4],
        header_bits=64,
        bandwidth_hz=100_000.0,
        deadline_s=deadline_s,
        processing_delay_s=0.0002,
        snr_threshold_db=0.0,
        antenna_gain_dbi=0.0,
        carrier_hz=28.0e9,
        tx_power_w=0.25,
        kT=4.0e-21,
        noise_figure_db=4.0,
        dt=0.1,
    )


def test_geometry_layout_charges_complete_atomic_record():
    layout = GeometryRepairWireLayout(2, 1)
    assert layout.prepare_bits == 350
    assert layout.vote_bits == 146
    assert layout.decision_bits == 146
    assert layout.verification_request_bits((1, 2), 3) < (
        layout.verification_request_bits((2, 2), 3))
    assert layout.verification_bits(1, 4, 3) < (
        layout.common_bits + 1 + 3 * 4 * 2 * layout.bound_bits + 32)


def test_digest_bound_prepare_does_not_repeat_dense_rf_plan():
    small = GeometryRepairWireLayout(4, 4)
    large = GeometryRepairWireLayout(8, 8)
    assert large.prepare_bits_for_movers(2, 3) - (
        small.prepare_bits_for_movers(2, 3)
    ) <= 8
    assert large.state_bits_for_horizon(3) > small.state_bits_for_horizon(3)


def test_geometry_transport_is_seven_round_and_physically_feasible():
    positions = np.asarray([
        [100.0, 200.0, 20.0],
        [300.0, 200.0, 20.0],
    ])
    result = certify_geometry_repair_transport(
        0,
        positions=positions,
        target_owner=np.asarray([1]),
        communication_power_w=np.full(2, 0.25),
        communication_model=_model(),
        control_period_s=0.1,
        layout=GeometryRepairWireLayout(2, 1),
        snr_margin_db=3.0,
        latency_margin_s=5.0e-4,
    )
    assert result.feasible
    assert tuple(report.name for report in result.rounds) == (
        "state", "gradient", "verification_request",
        "verification_return", "prepare", "vote", "decision")
    assert result.proposal_over_air_bits == 2 * 417 + 291
    assert result.total_over_air_bits > 1445
    assert result.total_protocol_latency_s <= 0.1
    assert result.total_energy_j > 0.0
    assert np.isclose(sum(result.per_uav_energy_j), result.total_energy_j)


def test_displacement_codec_never_enlarges_the_trust_region():
    requested = np.asarray([2.5 / np.sqrt(2.0), 2.5 / np.sqrt(2.0)])
    decoded = quantize_displacement_toward_zero(
        requested, maximum_component_m=2.5, bits_per_axis=16)
    assert np.linalg.norm(decoded) <= np.linalg.norm(requested)
    assert np.linalg.norm(requested - decoded) < 2.0e-4


def test_geometry_transport_fails_closed_without_reserved_power():
    positions = np.asarray([
        [100.0, 200.0, 20.0],
        [300.0, 200.0, 20.0],
    ])
    result = certify_geometry_repair_transport(
        0,
        positions=positions,
        target_owner=np.asarray([1]),
        communication_power_w=np.zeros(2),
        communication_model=_model(),
        control_period_s=0.1,
        layout=GeometryRepairWireLayout(2, 1),
    )
    assert not result.feasible
    assert any("snr" in reason for reason in result.reasons)


def test_geometry_transport_charges_both_mover_records():
    positions = np.asarray([
        [100.0, 200.0, 20.0],
        [300.0, 200.0, 20.0],
    ])
    layout = GeometryRepairWireLayout(2, 1)
    result = certify_geometry_repair_transport(
        (0, 1),
        positions=positions,
        target_owner=np.asarray([1]),
        communication_power_w=np.full(2, 0.25),
        communication_model=_model(),
        control_period_s=0.1,
        layout=layout,
    )
    assert result.feasible
    assert result.coordinator == 1
    assert layout.prepare_bits_for_movers(2) - layout.prepare_bits == 161
    assert result.total_over_air_bits == 1220


def test_geometry_transport_charges_every_certificate_step():
    positions = np.asarray([
        [100.0, 200.0, 20.0],
        [300.0, 200.0, 20.0],
    ])
    layout = GeometryRepairWireLayout(2, 1)
    one_step = certify_geometry_repair_transport(
        (0, 1), positions=positions, target_owner=np.asarray([1]),
        communication_power_w=np.full(2, 0.25),
        communication_model=_model(), control_period_s=0.1, layout=layout,
        certificate_horizon_steps=1,
    )
    three_step = certify_geometry_repair_transport(
        (0, 1), positions=positions, target_owner=np.asarray([1]),
        communication_power_w=np.full(2, 0.25),
        communication_model=_model(), control_period_s=0.1, layout=layout,
        certificate_horizon_steps=3,
    )
    assert three_step.total_over_air_bits > one_step.total_over_air_bits
    assert three_step.total_protocol_latency_s > one_step.total_protocol_latency_s
