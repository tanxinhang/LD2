import numpy as np
import pytest

from config.params import get_default_config
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.physical.evidence import (
    EvidencePacketLayout,
    route_structured_evidence,
)


def _model(
    *, fbl: bool, deadline_s: float = 0.005, target_bler: float = 1.0e-5,
    **channel_kwargs,
):
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 8],
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
        message_dim=16,
        finite_blocklength_enabled=fbl,
        finite_blocklength_target_bler=target_bler,
        **channel_kwargs,
    )


def test_finite_blocklength_latency_exceeds_shannon_for_same_link():
    sender = np.asarray([0.0, 0.0, 100.0])
    receiver = np.asarray([250.0, 0.0, 100.0])
    bits, bandwidth = 512, 100_000.0
    shannon = _model(fbl=False).link_budget(
        sender, receiver, bits, bandwidth)
    finite = _model(fbl=True).link_budget(
        sender, receiver, bits, bandwidth)
    assert finite[2] > shannon[2]
    assert finite[1] < shannon[1]


def test_finite_blocklength_coding_margin_uses_longer_codeword():
    sender = np.asarray([0.0, 0.0, 100.0])
    receiver = np.asarray([250.0, 0.0, 100.0])
    nominal = _model(fbl=True).link_budget(
        sender, receiver, 512, 100_000.0)
    robust = _model(
        fbl=True,
        finite_blocklength_coding_snr_margin_db=6.0,
    ).link_budget(sender, receiver, 512, 100_000.0)
    assert robust[2] > nominal[2]
    assert robust[1] < nominal[1]


def test_finite_blocklength_transport_reports_bler_and_delivers():
    model = _model(fbl=True)
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    delivered, stats = model.transmit(
        {0: np.zeros(16)}, {0: 1}, positions)
    assert len(delivered) == 1
    assert delivered[0].packet_error_probability <= 1.0e-5 * (1.0 + 1.0e-9)
    assert 0.0 < stats.mean_packet_error_probability <= 1.0e-5
    assert stats.reliability_violation_rate == 0.0
    assert stats.delivery_rate == 1.0


def test_protocol_only_packet_charges_exact_bits_and_hides_latent_payload():
    model = _model(fbl=False)
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    delivered, stats = model.transmit(
        {0: np.ones(16)}, {0: 1}, positions,
        extra_payload_bits={0: 32},
        base_payload_dimensions={0: 0},
        suppress_message_payload={0: True},
    )
    assert stats.total_bits == 64 + 32
    assert len(delivered) == 1
    np.testing.assert_array_equal(delivered[0].message, np.zeros(16))


def test_fixed_schema_header_override_is_charged_exactly():
    model = _model(fbl=False)
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    delivered, stats = model.transmit(
        {0: np.ones(16)}, {0: 1}, positions,
        extra_payload_bits={0: 32},
        base_payload_dimensions={0: 0},
        suppress_message_payload={0: True},
        header_bits_by_sender={0: 16},
    )
    assert len(delivered) == 1
    assert stats.total_bits == 16 + 32
    assert stats.per_sender_bits[0] == 16 + 32


def test_service_envelope_keeps_policy_latency_metadata_legacy():
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    legacy_model = _model(fbl=True)
    compact_model = _model(fbl=True)
    legacy_delivery, legacy_stats = legacy_model.transmit(
        {0: np.zeros(16)}, {0: 0}, positions,
        extra_payload_bits={0: 28},
        base_payload_dimensions={0: 0},
        suppress_message_payload={0: True},
    )
    compact_delivery, compact_stats = compact_model.transmit(
        {0: np.zeros(16)}, {0: 0}, positions,
        extra_payload_bits={0: 12},
        base_payload_dimensions={0: 0},
        suppress_message_payload={0: True},
        header_bits_by_sender={0: 16},
        service_envelope_payload_bits_by_sender={0: 92},
    )
    assert len(legacy_delivery) == len(compact_delivery) == 1
    assert compact_stats.mean_latency_s < legacy_stats.mean_latency_s
    assert compact_delivery[0].latency_s == pytest.approx(
        legacy_delivery[0].latency_s)


def test_finite_blocklength_deadline_failure_is_fail_closed():
    model = _model(fbl=True, deadline_s=0.00021)
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [1000.0, 0.0, 100.0],
    ])
    delivered, stats = model.transmit(
        {0: np.zeros(16)}, {0: 1}, positions)
    assert not delivered
    assert stats.expired_links == 1
    assert stats.deadline_failed_links == 1
    assert stats.deadline_violation_rate == 1.0
    assert stats.delivery_rate == 0.0


def test_deadline_failure_does_not_consume_erasure_rng():
    failed = _model(fbl=True, deadline_s=1.0e-8, target_bler=0.49)
    fresh = _model(fbl=True, deadline_s=1.0e-8, target_bler=0.49)
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    failed.transmit({0: np.zeros(16)}, {0: 1}, positions)
    assert failed.packet_reliability_success(0.2) == (
        fresh.packet_reliability_success(0.2))


def test_finite_blocklength_erasure_sampling_is_reproducible():
    first = _model(fbl=True, target_bler=0.49)
    second = _model(fbl=True, target_bler=0.49)
    sequence_a = [first.packet_reliability_success(0.2) for _ in range(8)]
    sequence_b = [second.packet_reliability_success(0.2) for _ in range(8)]
    assert sequence_a == sequence_b
    assert any(sequence_a) and not all(sequence_a)


def test_gilbert_elliott_burst_erasure_and_recovery_are_stateful():
    model = _model(
        fbl=False,
        burst_loss_enabled=True,
        burst_good_to_bad_probability=1.0,
        burst_bad_to_good_probability=1.0,
        burst_good_drop_probability=0.0,
        burst_bad_drop_probability=1.0,
    )
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    first, first_stats = model.transmit(
        {0: np.zeros(16)}, {0: 1}, positions)
    second, second_stats = model.transmit(
        {0: np.zeros(16)}, {0: 1}, positions)
    assert not first
    assert first_stats.burst_failed_links == 1
    assert first_stats.burst_bad_link_rate == 1.0
    assert first_stats.deadline_violation_rate == 0.0
    assert first_stats.delivery_failure_rate == 1.0
    assert len(second) == 1
    assert second_stats.burst_failed_links == 0


def test_correlated_snr_shadowing_is_replayable_from_channel_snapshot():
    model = _model(
        fbl=False,
        snr_shadowing_std_db=4.0,
        snr_shadowing_correlation=0.8,
    )
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    rng_state = model.rng.bit_generator.state
    channel_state = model.get_channel_state()
    first, _ = model.transmit({0: np.zeros(16)}, {0: 1}, positions)
    first_snr = first[0].snr_db
    model.rng.bit_generator.state = rng_state
    model.set_channel_state(channel_state)
    replay, _ = model.transmit({0: np.zeros(16)}, {0: 1}, positions)
    assert replay[0].snr_db == pytest.approx(first_snr)


def test_evidence_transport_uses_same_reliability_gate():
    model = _model(fbl=True)
    # Force a deterministic codeword erasure while retaining the real FBL
    # serialization path.  This isolates routing semantics from RNG luck.
    model.packet_reliability_success = lambda _error: False
    receiver_d = np.asarray([[4.0], [2.0]])
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    result = route_structured_evidence(
        receiver_d,
        positions,
        np.full(2, 0.25),
        observation_frame=3,
        topk=1,
        llr_bits=8,
        layout=EvidencePacketLayout(2, 1, confidence_bits=2),
        link_model=model,
        fusion_owner=np.asarray([0]),
        owner_aware=True,
    )
    assert result.attempted_links == 1
    assert result.delivered_links == 0
    assert not np.any(result.delivered_peer_mask)


def test_synchronous_evidence_header_is_charged_inside_transport():
    model = _model(fbl=False)
    receiver_d = np.asarray([[4.0], [2.0]])
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    layout = EvidencePacketLayout(
        2, 1,
        header_bits=16,
        timestamp_bits=0,
        confidence_bits=2,
    )
    result = route_structured_evidence(
        receiver_d,
        positions,
        np.full(2, 0.25),
        observation_frame=3,
        topk=1,
        llr_bits=8,
        layout=layout,
        link_model=model,
        fusion_owner=np.asarray([0]),
        owner_aware=True,
    )
    # One non-owner source sends: CRC16 + source1 + target1 + LLR8 + conf2.
    assert result.active_senders == 1
    assert result.total_bits == 28
    assert result.payload_bits_by_sender[1] == 28


def test_compact_evidence_service_envelope_preserves_legacy_delivery():
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [100.0, 0.0, 100.0],
    ])
    receiver_d = np.asarray([[4.0], [2.0]])
    legacy_layout = EvidencePacketLayout(
        2, 1, header_bits=64, timestamp_bits=16, confidence_bits=2)
    compact_layout = EvidencePacketLayout(
        2, 1, header_bits=16, timestamp_bits=0, confidence_bits=2)
    common = dict(
        receiver_deflection=receiver_d,
        positions=positions,
        comm_power_w=np.full(2, 0.25),
        observation_frame=3,
        topk=1,
        llr_bits=8,
        fusion_owner=np.asarray([0]),
        owner_aware=True,
    )
    legacy = route_structured_evidence(
        **common, layout=legacy_layout, link_model=_model(fbl=True))
    compact = route_structured_evidence(
        **common,
        layout=compact_layout,
        service_envelope_layout=legacy_layout,
        link_model=_model(fbl=True),
    )
    np.testing.assert_array_equal(
        compact.delivery_matrix, legacy.delivery_matrix)
    assert compact.total_bits < legacy.total_bits
    assert compact.mean_latency_s < legacy.mean_latency_s
    assert compact.total_energy_j == pytest.approx(legacy.total_energy_j)


def test_four_bit_hyperedge_target_code_is_lossless_for_q12():
    q = 12
    normalized = np.arange(q + 1, dtype=np.float64) / q
    encoded = InterUAVCommunicationModel.quantize_values_at_bits(
        2.0 * normalized - 1.0, 4)
    decoded = np.rint(0.5 * (encoded + 1.0) * q).astype(np.int64)
    np.testing.assert_array_equal(decoded, np.arange(q + 1))
    legacy = InterUAVCommunicationModel.quantize_values_at_bits(
        2.0 * decoded.astype(np.float64) / q - 1.0, 8)
    reference = InterUAVCommunicationModel.quantize_values_at_bits(
        2.0 * normalized - 1.0, 8)
    np.testing.assert_array_equal(legacy, reference)


def test_crc10_closes_configured_undetected_error_budget():
    target_bler = 1.0e-3
    undetected_budget = 1.0e-6
    assert target_bler * 2.0 ** -10 <= undetected_budget
    assert target_bler * 2.0 ** -9 > undetected_budget


def test_two_bit_aoi_is_lossless_under_three_frame_guard():
    aoi = np.arange(4, dtype=np.int64)
    encoded = np.minimum(aoi, (1 << 2) - 1)
    np.testing.assert_array_equal(encoded, aoi)


def test_fbl_rejects_shannon_optimal_bandwidth_l0():
    cfg = get_default_config()
    cfg.scenario.K = 2
    cfg.scenario.Q = 2
    cfg.target.omega_q = [0.5, 0.5]
    cfg.marl.learned_comm_mode = 'cost_aware'
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.analytical_comm_power_enabled = True
    cfg.marl.analytical_comm_optimal_bw = True
    cfg.marl.comm_finite_blocklength_enabled = True
    with pytest.raises(ValueError, match='Shannon KKT'):
        UAVISACEnv(config=cfg, seed=0)
