"""Detection evidence must respect the configured execution boundary."""

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
    estimate_quantized_evidence_detection,
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


def _link_model(message_dim=1):
    return InterUAVCommunicationModel(
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
