"""Detection evidence must respect the configured execution boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.evaluation.quantized_evidence_audit import (
    simulate_quantized_detection,
)
from uav_isac.physical.evidence import (
    DeflectionConfidenceQuantizer,
    EvidencePacketLayout,
    detection_probability_to_deflection,
    estimate_quantized_evidence_detection,
    quantize_belief_feedback,
    quantize_llr,
    receiver_deflection_from_broadcast_waveforms,
    receiver_deflection_from_selected,
    route_structured_evidence,
    select_detection_deflection,
)
from uav_isac.utils.math_utils import compute_PD
from uav_isac.utils.types import DeflectionEntry


def _entry(i, j, q, d_eff):
    return DeflectionEntry(
        i=i, j=j, q=q, tau=0.0, nu=0.0, alpha=1.0,
        d_raw=d_eff, g_dd=1.0, chi_rep=1.0, d_eff=d_eff,
    )


def _zero_actions(K):
    return {
        str(k): {"delta_p": np.zeros(2), "role": 0}
        for k in range(K)
    }


def _link_model(message_dim=1, **overrides):
    params = dict(
        rate_bits_per_dim=[0, 8],
        header_bits=64,
        bandwidth_hz=1.0e5,
        deadline_s=0.005,
        processing_delay_s=2.0e-4,
        snr_threshold_db=0.0,
        antenna_gain_dbi=0.0,
        carrier_hz=3.0e9,
        tx_power_w=0.25,
        kT=4.0e-21,
        noise_figure_db=5.0,
        dt=0.1,
        message_dim=message_dim,
    )
    params.update(overrides)
    return InterUAVCommunicationModel(**params)


def _full_normal_evidence_reference(
    receiver_deflection,
    transport,
    *,
    llr_bits,
    clip_max,
    standardized_threshold,
    p_fa,
    confidence_quantizer,
    draws,
    seed,
):
    """Pre-sparse full-tensor reference for the normal evidence payload."""
    receiver_d = np.asarray(receiver_deflection, dtype=np.float64)
    K, Q = receiver_d.shape
    peer_mask = np.asarray(transport.delivered_peer_mask, dtype=bool)
    owner_mask = np.asarray(transport.owner_mask, dtype=bool)
    sample_count = max(1, int(draws))
    rng = np.random.default_rng(int(seed))
    shape = (sample_count, K, Q)
    noise_h0 = rng.standard_normal(shape)
    noise_h1 = rng.standard_normal(shape)
    local_h0 = (
        -0.5 * receiver_d[None]
        + np.sqrt(receiver_d[None]) * noise_h0
    )
    local_h1 = (
        +0.5 * receiver_d[None]
        + np.sqrt(receiver_d[None]) * noise_h1
    )
    peer_d_hat = (
        receiver_d
        if confidence_quantizer is None
        else confidence_quantizer.quantize(receiver_d)
    )
    quantized_peer_h0 = quantize_llr(local_h0, llr_bits, clip_max)
    quantized_peer_h1 = quantize_llr(local_h1, llr_bits, clip_max)
    fused_h0 = np.sum(np.where(
        owner_mask[None],
        local_h0,
        np.where(peer_mask[None], quantized_peer_h0, 0.0),
    ), axis=1)
    fused_h1 = np.sum(np.where(
        owner_mask[None],
        local_h1,
        np.where(peer_mask[None], quantized_peer_h1, 0.0),
    ), axis=1)
    threshold_d = np.sum(np.where(
        owner_mask,
        receiver_d,
        np.where(peer_mask, peer_d_hat, 0.0),
    ), axis=0)
    threshold = (
        -0.5 * threshold_d
        + np.sqrt(threshold_d) * float(standardized_threshold)
    )
    positive = threshold_d > 0.0
    observed_pfa = np.where(
        positive,
        np.mean(fused_h0 > threshold[None], axis=0),
        float(p_fa),
    )
    pd = np.where(
        positive,
        np.mean(fused_h1 > threshold[None], axis=0),
        float(p_fa),
    )
    transmitted = np.broadcast_to(peer_mask[None], local_h0.shape)
    raw_values = np.concatenate([
        local_h0[transmitted],
        local_h1[transmitted],
    ])
    quantized_values = np.concatenate([
        quantized_peer_h0[transmitted],
        quantized_peer_h1[transmitted],
    ])
    if raw_values.size:
        clip_rate = float(np.mean(np.abs(raw_values) > float(clip_max)))
        quantization_mse = float(np.mean(
            (quantized_values - raw_values) ** 2))
        nonzero = np.abs(raw_values) > 1e-15
        sign_flip_rate = float(np.mean(
            np.signbit(raw_values[nonzero])
            != np.signbit(quantized_values[nonzero])
        )) if np.any(nonzero) else 0.0
    else:
        clip_rate = quantization_mse = sign_flip_rate = 0.0
    fused_d = np.sum(np.where(
        owner_mask | peer_mask, receiver_d, 0.0), axis=0)
    return {
        "pd": np.asarray(pd, dtype=np.float64),
        "equivalent_deflection": detection_probability_to_deflection(
            pd, p_fa),
        "pfa": np.asarray(observed_pfa, dtype=np.float64),
        "aggregate_pfa": float(np.mean(observed_pfa)),
        "threshold_deflection": threshold_d,
        "available_true_deflection": fused_d,
        "clip_rate": clip_rate,
        "quantization_mse": quantization_mse,
        "sign_flip_rate": sign_flip_rate,
        "draws": int(sample_count),
        "content_mode": "normal",
    }


def test_pure_local_and_central_fusion_are_distinct():
    entries = [
        _entry(0, 1, 0, 1.0),
        _entry(2, 3, 0, 3.0),
        _entry(1, 0, 1, 2.0),
    ]
    selected = [(0, 1, 0), (2, 3, 0), (1, 0, 1)]
    receiver_d = receiver_deflection_from_selected(
        selected, entries, num_agents=4, num_targets=2)

    np.testing.assert_allclose(
        select_detection_deflection("local_only", receiver_d),
        [3.0, 2.0],
    )
    np.testing.assert_allclose(
        select_detection_deflection("central_oracle", receiver_d),
        [4.0, 2.0],
    )


def test_selected_waveform_is_observed_by_all_reserved_receivers():
    entries = [
        _entry(0, 1, 0, 4.0),
        _entry(0, 3, 0, 2.0),
        _entry(2, 1, 1, 3.0),
        _entry(2, 3, 1, 1.0),
        _entry(1, 3, 0, 99.0),  # transmitter-target waveform not selected
    ]
    selected = [(0, 1, 0), (2, 3, 1)]
    receiver_d = receiver_deflection_from_broadcast_waveforms(
        selected, entries, num_agents=4, num_targets=2)
    np.testing.assert_allclose(receiver_d, [
        [0.0, 0.0],
        [4.0, 3.0],
        [0.0, 0.0],
        [2.0, 1.0],
    ])


def test_distributed_mode_cannot_fall_back_to_oracle():
    with pytest.raises(RuntimeError, match="delivered structured packets"):
        select_detection_deflection(
            "u2u_distributed", np.ones((2, 2), dtype=np.float64))


def test_legacy_and_explicit_central_environment_match_one_step():
    config_path = (
        "config/exp_800_q4_u2u_hierarchical_multistatic_"
        "distributed_matching_hybrid50_paper_top1_eval.yaml"
    )
    legacy_cfg = load_config(config_path)
    central_cfg = load_config(config_path)
    legacy_cfg.marl.detection_fusion_mode = "legacy_global"
    central_cfg.marl.detection_fusion_mode = "central_oracle"
    legacy = UAVISACEnv(config=legacy_cfg, seed=73)
    central = UAVISACEnv(config=central_cfg, seed=73)
    legacy.reset(seed=73)
    central.reset(seed=73)

    _, _, _, _, legacy_info = legacy.step(_zero_actions(legacy.K))
    _, _, _, _, central_info = central.step(_zero_actions(central.K))

    np.testing.assert_allclose(
        legacy_info["P_D_q"], central_info["P_D_q"], rtol=0.0, atol=1e-12)
    legacy.close()
    central.close()


def test_local_environment_uses_best_single_receiver():
    config_path = (
        "config/exp_800_q4_u2u_hierarchical_multistatic_"
        "distributed_matching_hybrid50_paper_top1_local_only_eval.yaml"
    )
    cfg = load_config(config_path)
    env = UAVISACEnv(config=cfg, seed=91)
    env.reset(seed=91)
    _, _, _, _, info = env.step(_zero_actions(env.K))

    receiver_d = np.asarray(
        info["detection_receiver_deflection"], dtype=np.float64)
    expected = compute_PD(
        np.max(receiver_d, axis=0), cfg.detection.P_FA)
    np.testing.assert_allclose(
        info["P_D_q"], expected, rtol=0.0, atol=1e-12)
    assert info["detection_fusion_mode"] == "local_only"
    env.close()


def test_distributed_mode_requires_explicit_calibration_profile():
    cfg = load_config("config/default.yaml")
    cfg.marl.detection_fusion_mode = "u2u_distributed"
    cfg.marl.use_difference_reward = False
    with pytest.raises(ValueError, match="clip_max"):
        UAVISACEnv(config=cfg, seed=3)


def test_online_quantized_frame_matches_offline_gate_core():
    receiver_d = np.asarray([
        [4.0, 0.5],
        [2.0, 3.0],
        [1.0, 1.5],
    ])
    positions = np.asarray([
        [0.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [0.0, 20.0, 20.0],
    ])
    layout = EvidencePacketLayout(
        num_agents=3,
        num_targets=2,
        confidence_bits=2,
    )
    transport = route_structured_evidence(
        receiver_d,
        positions,
        np.full(3, 0.25),
        observation_frame=7,
        topk=1,
        llr_bits=8,
        layout=layout,
        link_model=_link_model(),
        fusion_owner=np.asarray([0, 1]),
        owner_aware=True,
    )
    quantizer = DeflectionConfidenceQuantizer(
        bits=2,
        log_boundaries=np.asarray([1.0, 2.0, 3.0]),
        representatives=np.asarray([1.0, 2.5, 6.0, 12.0]),
    )
    online = estimate_quantized_evidence_detection(
        receiver_d,
        transport,
        llr_bits=8,
        clip_max=20.0,
        standardized_threshold=3.05,
        p_fa=0.001,
        confidence_quantizer=quantizer,
        draws=4096,
        seed=1234,
    )
    offline = simulate_quantized_detection(
        receiver_d[None],
        fusion_owner=transport.owner[None],
        topk=1,
        bits=8,
        clip_max=20.0,
        standardized_threshold=3.05,
        p_fa=0.001,
        owner_aware=True,
        delivery_matrix=transport.delivery_matrix[None],
        peer_deflection_estimate=quantizer.quantize(
            receiver_d)[None],
        draws_per_frame=4096,
        batch_frames=1,
        seed=1234,
    )
    np.testing.assert_array_equal(
        online["pd"], offline["pd"][0])
    np.testing.assert_array_equal(
        online["pfa"], offline["pfa_by_frame_target"][0])

    online_standardized = estimate_quantized_evidence_detection(
        receiver_d,
        transport,
        llr_bits=8,
        clip_max=5.0,
        standardized_threshold=3.05,
        p_fa=0.001,
        confidence_quantizer=quantizer,
        draws=4096,
        seed=1234,
        content_mode="standardized_score",
    )
    offline_standardized = simulate_quantized_detection(
        receiver_d[None],
        fusion_owner=transport.owner[None],
        topk=1,
        bits=8,
        clip_max=5.0,
        standardized_threshold=3.05,
        p_fa=0.001,
        owner_aware=True,
        delivery_matrix=transport.delivery_matrix[None],
        peer_deflection_estimate=quantizer.quantize(receiver_d)[None],
        draws_per_frame=4096,
        batch_frames=1,
        seed=1234,
        content_mode="standardized_score",
    )
    np.testing.assert_array_equal(
        online_standardized["pd"], offline_standardized["pd"][0])
    np.testing.assert_array_equal(
        online_standardized["pfa"],
        offline_standardized["pfa_by_frame_target"][0],
    )


@pytest.mark.parametrize(
    "owner_mask,peer_mask,use_confidence",
    [
        (
            np.eye(4, dtype=bool),
            np.asarray([
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
                [True, False, False, False],
            ]),
            True,
        ),
        (np.eye(4, dtype=bool), np.zeros((4, 4), dtype=bool), False),
        # Defensive overlap: owner-local evidence wins fusion, while the peer
        # record remains represented in packet-value diagnostics.
        (np.eye(4, dtype=bool), np.eye(4, dtype=bool), True),
        (
            np.asarray([
                [True, False, False, False],
                [False, False, False, False],
                [False, False, True, False],
                [False, False, False, False],
            ]),
            np.asarray([
                [False, False, False, False],
                [True, False, False, True],
                [False, False, False, False],
                [False, True, False, False],
            ]),
            False,
        ),
    ],
)
def test_sparse_normal_evidence_is_bit_exact_to_full_tensor_reference(
    owner_mask,
    peer_mask,
    use_confidence,
):
    receiver_d = np.asarray([
        [0.0, 0.25, 2.0, 12.0],
        [1.0e-12, 0.5, 3.0, 15.0],
        [0.1, 0.75, 4.0, 20.0],
        [0.2, 1.0, 5.0, 25.0],
    ])
    transport = SimpleNamespace(
        owner_mask=np.asarray(owner_mask, dtype=bool),
        delivered_peer_mask=np.asarray(peer_mask, dtype=bool),
    )
    quantizer = (
        DeflectionConfidenceQuantizer(
            bits=2,
            log_boundaries=np.asarray([0.2, 1.0, 2.5]),
            representatives=np.asarray([0.1, 0.8, 4.0, 16.0]),
        )
        if use_confidence else None
    )
    kwargs = dict(
        llr_bits=8,
        clip_max=12.5,
        standardized_threshold=3.05,
        p_fa=0.001,
        confidence_quantizer=quantizer,
        draws=257,
        seed=20260831,
    )
    expected = _full_normal_evidence_reference(
        receiver_d, transport, **kwargs)
    actual = estimate_quantized_evidence_detection(
        receiver_d, transport, content_mode="normal", **kwargs)

    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def test_evidence_route_obeys_scheduled_owner_even_when_not_quality_argmax():
    receiver_d = np.asarray([[9.0], [4.0], [1.0]])
    positions = np.asarray([
        [0.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [0.0, 20.0, 20.0],
    ])
    transport = route_structured_evidence(
        receiver_d,
        positions,
        np.full(3, 0.25),
        observation_frame=8,
        topk=1,
        llr_bits=8,
        layout=EvidencePacketLayout(3, 1, confidence_bits=2),
        link_model=_link_model(),
        fusion_owner=np.asarray([2]),
        owner_aware=True,
    )
    assert transport.owner[0] == 2
    assert transport.owner_mask[2, 0]
    assert not transport.owner_mask[0, 0]
    assert transport.as_dict()["evidence_owner_incremental_bits"] == 0.0


def test_evidence_transport_uses_same_burst_channel_as_coordination():
    receiver_d = np.asarray([
        [4.0, 1.0],
        [2.0, 3.0],
        [1.0, 2.0],
    ])
    positions = np.asarray([
        [0.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [0.0, 20.0, 20.0],
    ])
    transport = route_structured_evidence(
        receiver_d,
        positions,
        np.full(3, 0.25),
        observation_frame=9,
        topk=1,
        llr_bits=8,
        layout=EvidencePacketLayout(
            num_agents=3, num_targets=2, confidence_bits=2),
        link_model=_link_model(
            burst_loss_enabled=True,
            burst_good_to_bad_probability=1.0,
            burst_bad_to_good_probability=0.0,
            burst_bad_drop_probability=1.0,
        ),
        fusion_owner=np.asarray([0, 1]),
        owner_aware=True,
    )
    assert transport.total_bits > 0.0
    assert transport.attempted_links > 0
    assert transport.delivered_links == 0
    assert transport.delivery_failure_rate == pytest.approx(1.0)
    assert transport.burst_failed_links == transport.attempted_links
    assert transport.deadline_failed_links == 0


def test_u2u_environment_consumes_delivered_evidence_packets():
    config_path = (
        "config/exp_800_q4_u2u_hierarchical_multistatic_"
        "distributed_matching_hybrid50_paper_top1_u2u_evidence_eval.yaml"
    )
    cfg = load_config(config_path)
    cfg.marl.evidence_packet_mc_draws = 512
    env = UAVISACEnv(config=cfg, seed=19)
    env.reset(seed=19)
    _, _, _, _, info = env.step(_zero_actions(env.K))

    assert info["detection_fusion_mode"] == "u2u_distributed"
    assert info["evidence_packet_calibration_profile"].startswith("gate1c")
    assert info["evidence_comm_bits"] >= 0.0
    assert info["evidence_useful_unique_entries"] <= (
        info["evidence_transmitted_entries"])
    np.testing.assert_allclose(
        info["P_D_q"],
        compute_PD(
            info["detection_deflection_q"],
            cfg.detection.P_FA,
        ),
        atol=1e-12,
    )
    env.close()


def test_belief_feedback_quantization_is_conservative_and_charged():
    mean = np.asarray([321.25, 678.75, 4.2, -3.7])
    covariance = np.asarray([
        [40.0, 18.0, 2.0, 0.0],
        [18.0, 55.0, 0.0, -3.0],
        [2.0, 0.0, 5.0, 1.5],
        [0.0, -3.0, 1.5, 7.0],
    ])
    decoded_mean, decoded_covariance = quantize_belief_feedback(
        mean,
        covariance,
        area_size_xy=(1200.0, 1200.0),
        velocity_bound_mps=25.0,
        mean_bits=12,
        covariance_bits=8,
    )
    steps = np.asarray([1200.0, 1200.0, 50.0, 50.0]) / (2**12 - 1)
    assert np.all(np.abs(decoded_mean - mean) <= 0.5 * steps + 1.0e-12)
    # The decoded diagonal covariance must dominate the transmitted full
    # covariance in PSD order despite omitted cross-covariances.
    assert np.min(np.linalg.eigvalsh(
        decoded_covariance - covariance)) >= -1.0e-9

    base = EvidencePacketLayout(
        num_agents=12, num_targets=12, confidence_bits=2)
    feedback = EvidencePacketLayout(
        num_agents=12,
        num_targets=12,
        confidence_bits=2,
        feedback_bits_per_entry=4 * 12 + 4 * 8 + 8,
    )
    assert feedback.broadcast_bits(3, 8) - base.broadcast_bits(3, 8) == 3 * 88


def test_ca_belief_feedback_quantization_preserves_six_dimensional_contract():
    mean = np.asarray([321.25, 678.75, 4.2, -3.7, 0.8, -0.6])
    covariance = np.diag([40.0, 55.0, 5.0, 7.0, 0.5, 0.7])
    covariance[0, 4] = covariance[4, 0] = 0.8
    decoded_mean, decoded_covariance = quantize_belief_feedback(
        mean,
        covariance,
        area_size_xy=(1200.0, 1200.0),
        velocity_bound_mps=25.0,
        acceleration_bound_mps2=2.0,
        mean_bits=12,
        covariance_bits=8,
    )

    assert decoded_mean.shape == (6,)
    assert decoded_covariance.shape == (6, 6)
    assert np.min(np.linalg.eigvalsh(
        decoded_covariance - covariance)) >= -1.0e-9


def test_dynamic_u2u_feedback_is_enabled_and_payload_is_physically_charged():
    cfg = load_config(
        'config/exp_800_k12q12_distributed_v2_dynamic_u2u_pilot.yaml')
    cfg.marl.evidence_packet_mc_draws = 64
    env = UAVISACEnv(config=cfg, seed=29)
    env.reset(seed=29)
    _, _, _, _, info = env.step(_zero_actions(env.K))

    assert info['belief_feedback_enabled'] == 1.0
    assert env.core._evidence_packet_layout.feedback_bits_per_entry == 88
    assert info['belief_feedback_fused_entries'] >= 0.0
    assert np.isfinite(info['belief_feedback_contraction_ratio'])
    assert np.isfinite(info['belief_position_rmse_m'])
    env.close()
