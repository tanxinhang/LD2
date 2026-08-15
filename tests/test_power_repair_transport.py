import numpy as np

from uav_isac.coordination.power_repair_transport import (
    PowerRepairWireLayout,
    certify_power_repair_transport,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


def _model():
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


def test_wire_layout_charges_headers_per_sender_and_round():
    layout = PowerRepairWireLayout(num_agents=6, num_targets=6, rounds=4)
    assert layout.shared_bits == 153
    assert layout.price_packet_bits == 189
    assert layout.owner_feedback_packet_bits(1) == 172


def test_transport_preserves_physical_power_support_and_deadlines():
    positions = np.asarray([
        [0.0, 0.0, 20.0], [40.0, 0.0, 20.0],
        [80.0, 0.0, 20.0], [0.0, 40.0, 20.0],
        [40.0, 40.0, 20.0], [80.0, 40.0, 20.0],
    ])
    layout = PowerRepairWireLayout(num_agents=6, num_targets=6, rounds=4)
    result = certify_power_repair_transport(
        np.arange(6),
        positions,
        np.zeros(6),
        communication_model=_model(),
        layout=layout,
        control_period_s=0.1,
    )
    assert result.feasible
    assert np.all(result.projected_comm_power_w < 1.0)
    assert result.max_packet_latency_s <= 0.005 + 1e-12
    assert result.total_protocol_latency_s <= 0.1
    # One coordinator owns its target locally: five over-air owner packets.
    assert result.total_over_air_bits == 5 * 5 * 172 + 4 * 189


def test_more_existing_comm_power_never_increases_required_projection():
    positions = np.asarray([
        [0.0, 0.0, 20.0], [100.0, 0.0, 20.0],
        [0.0, 100.0, 20.0], [100.0, 100.0, 20.0],
        [50.0, 0.0, 20.0], [50.0, 100.0, 20.0],
    ])
    layout = PowerRepairWireLayout(num_agents=6, num_targets=6, rounds=4)
    zero = certify_power_repair_transport(
        np.arange(6), positions, np.zeros(6),
        communication_model=_model(), layout=layout, control_period_s=0.1)
    existing = certify_power_repair_transport(
        np.arange(6), positions, np.full(6, 0.1),
        communication_model=_model(), layout=layout, control_period_s=0.1)
    assert np.max(existing.projected_comm_power_w - 0.1) <= (
        np.max(zero.projected_comm_power_w) + 1e-12)
