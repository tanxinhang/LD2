import numpy as np

from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.owner_local_physics import (
    OwnerTargetInvariantCache,
)
from uav_isac.coordination.power_repair_transport import PowerRepairWireLayout
from uav_isac.coordination.structure_sequence_transport import (
    certify_top1_structure_sequence_transport,
)
from uav_isac.coordination.target_invariant_transport import (
    TargetInvariantWireLayout,
    certify_target_invariant_transport,
    decode_owner_target_invariant_tokens,
    encode_owner_target_invariant_tokens,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


def _model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4], header_bits=64,
        bandwidth_hz=100_000.0, deadline_s=0.005,
        processing_delay_s=0.0002, snr_threshold_db=0.0,
        antenna_gain_dbi=0.0, carrier_hz=28.0e9,
        tx_power_w=0.25, kT=4.0e-21, noise_figure_db=4.0,
        dt=0.1,
    )


def _positions():
    return np.asarray([
        [0.0, 0.0, 20.0], [30.0, 0.0, 20.0],
        [0.0, 30.0, 20.0], [30.0, 30.0, 20.0],
    ])


def _cache():
    return OwnerTargetInvariantCache(
        target_invariant=np.asarray([2.0e8, 3.0e7]),
        age_frames=np.asarray([5, 7], dtype=np.int64),
        version=np.asarray([3, 4], dtype=np.int64),
    )


def test_target_invariant_wire_is_directional_versioned_and_fail_closed():
    layout = TargetInvariantWireLayout(4, 2)
    owner = np.asarray([2, 3], dtype=np.int64)
    tokens = encode_owner_target_invariant_tokens(
        _cache(), owner, layout=layout, sent_frame=10)

    assert len(tokens) == 2
    decoded = decode_owner_target_invariant_tokens(
        tokens,
        owner,
        layout=layout,
        current_frame=12,
        max_age_frames=20,
    )
    assert np.all(decoded.target_invariant > 0.0)
    assert np.all(decoded.target_invariant <= _cache().target_invariant)
    assert decoded.age_frames.tolist() == [7, 9]
    assert decoded.version.tolist() == [3, 4]
    for original, encoded in zip(
        _cache().target_invariant, decoded.target_invariant
    ):
        assert encoded <= original
        assert original <= layout.invariant_cell_upper(encoded)

    stale = decode_owner_target_invariant_tokens(
        tokens,
        owner,
        layout=layout,
        current_frame=30,
        max_age_frames=20,
    )
    assert stale.target_invariant.tolist() == [0.0, 0.0]

    wrong_owner = decode_owner_target_invariant_tokens(
        tokens,
        np.asarray([3, 2], dtype=np.int64),
        layout=layout,
        current_frame=12,
        max_age_frames=20,
    )
    assert wrong_owner.target_invariant.tolist() == [0.0, 0.0]


def test_expired_invalid_record_preserves_version_and_fails_closed():
    layout = TargetInvariantWireLayout(4, 2)
    cache = OwnerTargetInvariantCache(
        target_invariant=np.asarray([0.0, 3.0e7]),
        age_frames=np.asarray([300, 7], dtype=np.int64),
        version=np.asarray([8, 4], dtype=np.int64),
    )
    owner = np.asarray([2, 3], dtype=np.int64)
    tokens = encode_owner_target_invariant_tokens(
        cache, owner, layout=layout, sent_frame=10)
    decoded = decode_owner_target_invariant_tokens(
        tokens,
        owner,
        layout=layout,
        current_frame=10,
        max_age_frames=150,
    )

    assert decoded.target_invariant[0] == 0.0
    assert decoded.age_frames[0] == layout.max_age
    assert decoded.version[0] == 8


def test_target_invariant_transport_charges_broadcast_bits_and_power():
    layout = TargetInvariantWireLayout(4, 2)
    tokens = encode_owner_target_invariant_tokens(
        _cache(), np.asarray([2, 3]), layout=layout, sent_frame=10)
    result = certify_target_invariant_transport(
        tokens,
        positions=_positions(),
        existing_comm_power_w=np.zeros(4),
        communication_model=_model(),
        layout=layout,
    )

    assert result.feasible
    assert result.packet_count == 2
    assert result.record_count == 2
    assert result.total_over_air_bits == sum(
        token.over_air_bits for token in tokens)
    assert result.max_packet_latency_s <= _model().deadline_s
    assert np.all(result.projected_comm_power_w < 1.0)


def test_structure_sequence_includes_target_invariant_wire_cost():
    selected = np.zeros((4, 4, 2), dtype=bool)
    selected[0, 2, 0] = True
    selected[1, 3, 1] = True
    role = np.asarray([1, 1, 0, 0], dtype=np.int8)
    owner = np.asarray([2, 3], dtype=np.int64)
    proposal = selected.copy()
    proposal[1, 3, 1] = False
    proposal[0, 3, 1] = True
    move = LocalMove(
        kind="N6",
        selected=proposal,
        role=role.copy(),
        owner=owner.copy(),
    )
    invariant_layout = TargetInvariantWireLayout(4, 2)
    tokens = encode_owner_target_invariant_tokens(
        _cache(), owner, layout=invariant_layout, sent_frame=10)
    result = certify_top1_structure_sequence_transport(
        selected,
        role,
        owner,
        [move],
        [],
        positions=_positions(),
        existing_comm_power_w=np.zeros(4),
        final_sensing_weights=np.full((4, 2), 0.5),
        communication_model=_model(),
        power_layout=PowerRepairWireLayout(4, 2, rounds=4),
        control_period_s=0.1,
        target_invariant_tokens=tokens,
        target_invariant_layout=invariant_layout,
    )

    assert result.feasible
    assert result.invariant_packet_count == 2
    assert result.invariant_record_count == 2
    assert result.invariant_bits > 0
    assert result.total_over_air_bits >= result.invariant_bits
    assert result.total_protocol_latency_s >= result.invariant_latency_s
