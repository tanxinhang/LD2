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
