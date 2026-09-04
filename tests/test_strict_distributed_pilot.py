"""Strict distributed pilot identity and information-boundary tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config.params import load_config
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.environment.action import ActionSpace
from tools.run_strict_distributed_pilot import (
    _algorithm_version,
    _episode,
    validate_formal_system_identity,
    validate_strict_config,
)
from tools.run_strict_distributed_sweep import SweepCase, _aggregate
from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.environment.env_wrapper import UAVISACEnv


PILOT = "config/exp_strict_distributed_no_truth_pilot.yaml"


def test_formal_identity_rejects_a_clean_looking_noncanonical_override(tmp_path):
    canonical = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "exp_strict_distributed_k16q16.yaml"
    )
    profile = tmp_path / "binary-dd.yaml"
    profile.write_text(
        f"extends: {canonical.as_posix()}\n"
        "detection:\n  dd_gain_mode: binary\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="dd_gain_mode"):
        validate_formal_system_identity(str(profile))


def test_strict_pilot_activates_real_protocol_stack():
    cfg = load_config(PILOT)
    validate_strict_config(cfg)
    assert cfg.scenario.region_size == (1130, 1130)
    assert cfg.scenario.K == cfg.scenario.Q == 8
    assert cfg.marl.hyperedge_safety_fallback_enabled is False
    assert cfg.marl.distributed_replicated_power_enabled is True
    assert cfg.marl.distributed_owner_posterior_enabled is True
    assert cfg.marl.distributed_bottleneck_matching_movement_enabled is False
    assert cfg.marl.distributed_bistatic_bottleneck_movement_enabled is True
    assert cfg.marl.distributed_greedy_matching_hold_frames == 20
    assert cfg.marl.comm_bandwidth_hz == pytest.approx(500_000.0)
    assert cfg.marl.use_difference_reward is False
    assert cfg.marl.analytical_power_geometry_reuse_enabled is True


def test_strict_pilot_constructs_run_mappo_structured_actor():
    """Exercise architecture-v2 prerequisites as run_mappo wires them."""
    cfg = load_config(PILOT)
    env = UAVISACEnv(config=cfg, seed=7)
    try:
        observations, _ = env.reset(seed=7)
        action_space = ActionSpace(
            v_max=cfg.uav.v_max,
            dt=cfg.scenario.dt,
            seed=7,
            learn_roles=cfg.marl.learn_roles,
            dp_parameterization=cfg.marl.dp_parameterization,
        )
        action_space.num_targets = cfg.scenario.Q
        action_space.structured_actor = True
        action_space.structured_entity_dim = 64
        target_tokens = cfg.marl.comm_payload_mode == "target_tokens"
        token_dim = cfg.marl.comm_target_token_dim
        agent = MAPPOAgent(
            agent_id=0,
            obs_dim=observations["0"].shape[0],
            global_state_dim=env.core.obs_builder.get_global_state_dim(),
            action_space=action_space,
            num_agents=cfg.scenario.K,
            num_targets=cfg.scenario.Q,
            hidden_layers=cfg.marl.hidden_layers,
            device="cpu",
            use_comm_cross_attention=cfg.marl.comm_cross_attention_enabled,
            comm_token_dim=token_dim + 6 if target_tokens else 21,
            comm_tokens_per_sender=cfg.scenario.Q if target_tokens else 1,
            comm_payload_dim=(
                cfg.scenario.Q * token_dim if target_tokens else 16),
            comm_target_token_enabled=target_tokens,
            comm_target_token_dim=token_dim,
            use_target_allocation=(
                cfg.marl.target_allocation_enabled
                or cfg.marl.target_allocation_teacher_enabled),
            architecture_v2_enabled=cfg.marl.architecture_v2_enabled,
        )
        assert agent.actor._architecture_v2_enabled is True
        assert agent.actor._comm_target_token_enabled is True
    finally:
        env.close()


def test_robust_power_candidate_requires_positive_uncertainty_radius():
    cfg = load_config(PILOT)
    cfg.marl.distributed_replicated_power_robust_gain_mix = 0.25
    cfg.marl.distributed_target_position_uncertainty_sigma = 0.0
    with pytest.raises(ValueError, match="positive.*uncertainty"):
        EnvironmentCore(cfg)


def test_risk_mix_changes_algorithm_identity_without_renaming_baseline():
    cfg = load_config(
        "config/exp_strict_distributed_k16q16_ca_cabelief_process4.yaml")
    baseline = _algorithm_version(cfg)
    assert "v5-ca6" in baseline
    assert "riskmix" not in baseline
    cfg.marl.distributed_replicated_power_robust_gain_mix = 0.25
    assert _algorithm_version(cfg) == (
        baseline + "-nominal-robust-riskmix-0p25")


def test_expected_information_changes_identity_and_excludes_random_gating():
    cfg = load_config(PILOT)
    baseline = _algorithm_version(cfg)
    cfg.marl.belief_expected_detection_information_enabled = True
    assert _algorithm_version(cfg) == (
        baseline + "-expected-detection-information")
    cfg.marl.belief_detection_sampling = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        EnvironmentCore(cfg)


def test_bistatic_tracker_changes_identity_and_rejects_double_information():
    cfg = load_config(PILOT)
    baseline = _algorithm_version(cfg)
    cfg.marl.belief_measurement_model = "bistatic_range_doppler"
    assert _algorithm_version(cfg) == baseline + "-bistatic-range-doppler-ekf"
    cfg.marl.belief_expected_detection_information_enabled = True
    with pytest.raises(ValueError, match="double count"):
        EnvironmentCore(cfg)
    cfg.marl.belief_expected_detection_information_enabled = False
    cfg.marl.hyperedge_bistatic_information_ranking_enabled = True
    assert _algorithm_version(cfg) == (
        baseline
        + "-bistatic-range-doppler-ekf"
        + "-submodular-information-ranking")


def test_robust_power_candidate_executes_certificate_lower_gain_views():
    def run(gain_mix: float):
        cfg = load_config(PILOT)
        cfg.marl.distributed_replicated_power_robust_gain_mix = gain_mix
        env = UAVISACEnv(config=cfg)
        observations, _ = env.reset(seed=7)
        core = env.core
        # Populate the physically delivered endpoint-state cache through the
        # same zero-content carrier used by the strict analytical runner.
        for _frame in range(3):
            carrier_fraction = float(
                cfg.marl.comm_tx_power_w / cfg.uav.P_isac_total)
            core.submit_learned_communications(
                messages={
                    sender: np.zeros(core._comm_payload_dim)
                    for sender in range(core.K)
                },
                rate_indices={sender: 0 for sender in range(core.K)},
                comm_power_fractions={
                    sender: carrier_fraction for sender in range(core.K)
                },
                sensing_target_weights={
                    sender: np.ones(core.Q) for sender in range(core.K)
                },
                token_masks={
                    sender: np.ones(core.Q) for sender in range(core.K)
                },
            )
            actions = {
                agent: {"delta_p": np.zeros(2), "role": 2}
                for agent in observations
            }
            observations, *_ = env.step(actions)
        return core

    nominal = run(0.0)
    core = run(0.25)
    assert np.any(core._composable_certificate_gain_views > 0.0)
    expected = (
        0.75 * nominal._hyperedge_public_gain_views
        + 0.25 * core._composable_certificate_gain_views
    )
    np.testing.assert_allclose(
        core._hyperedge_public_gain_views,
        expected,
    )


def test_belief_model_can_be_pinned_for_maneuver_mismatch():
    cfg = load_config(PILOT)
    cfg.target.motion_model = "CA"
    cfg.marl.belief_motion_model = "CV"
    core = EnvironmentCore(cfg)
    core.reset()

    assert all(target.model == "CA" for target in core.targets)
    assert core.belief_mgr.motion_model == "CV"


def test_ca_belief_exposes_only_decision_sufficient_prefix_to_coordination():
    cfg = load_config(PILOT)
    cfg.target.motion_model = "CA"
    cfg.marl.belief_motion_model = "CA"
    core = EnvironmentCore(cfg)
    core.reset()

    assert core.belief_mgr.mean.shape[-1] == 6
    position, velocity = core._coordination_target_state_for_viewer(0)
    assert position.shape == (core.Q, 3)
    assert velocity.shape == (core.Q, 3)
    np.testing.assert_allclose(
        position[:, :2], core.belief_mgr.mean[0, :, :2])
    np.testing.assert_allclose(
        velocity[:, :2], core.belief_mgr.mean[0, :, 2:4])


def test_role_capacity_diagnostic_tracks_executed_worst_first_target():
    cfg = load_config(PILOT)
    core = EnvironmentCore(cfg)
    core.reset()
    public_xy = np.zeros((core.K, 2), dtype=np.float64)
    target_xy = np.zeros((core.Q, 2), dtype=np.float64)
    target_xy[0] = [1.0, 0.0]
    target_xy[1] = [10.0, 0.0]
    tx_resp = np.zeros((core.K, core.Q), dtype=np.int8)
    rx_resp = np.zeros_like(tx_resp)
    tx_resp[0, 0] = 1
    tx_resp[0, 1] = 1

    desired, states, chosen = core._role_capacity_desired_movement(
        public_xy, target_xy, tx_resp, rx_resp,
        movement_step=2.0, strategy_phase=False, standoff=0.0)

    assert chosen[0] == 1
    assert states[0] == "rolecap_radial"
    np.testing.assert_allclose(desired[0], [2.0, 0.0])


def test_strict_validator_rejects_no_protocol_pilot():
    cfg = load_config(PILOT)
    cfg.marl.hyperedge_negotiation_enabled = False
    with pytest.raises(ValueError, match="hyperedge_negotiation_enabled"):
        validate_strict_config(cfg)


def test_strict_online_validator_rejects_training_only_difference_reward():
    cfg = load_config(PILOT)
    cfg.marl.use_difference_reward = True
    with pytest.raises(ValueError, match="training-only difference reward"):
        validate_strict_config(cfg)


def test_hyperedge_submission_is_invariant_to_hidden_target_truth():
    """Only the fixed local beliefs may influence a sender's public beacon."""
    cfg = load_config(PILOT)
    core = EnvironmentCore(cfg)
    core.reset()
    core._prepare_hyperedge_submission()
    before = {
        sender: payload.copy()
        for sender, payload in core._pending_hyperedge_protocol.items()
    }

    # Perturb simulator truth without changing any viewer's local belief.
    for target in core.targets:
        target.state[0] = (target.state[0] + 317.0) % cfg.scenario.region_size[0]
        target.state[1] = (target.state[1] + 191.0) % cfg.scenario.region_size[1]
    core._prepare_hyperedge_submission()
    after = core._pending_hyperedge_protocol

    assert before.keys() == after.keys()
    for sender in before:
        np.testing.assert_array_equal(before[sender], after[sender])


def test_strict_pilot_control_protocol_gets_on_air():
    cfg = load_config(PILOT)
    cfg.scenario.T = 2
    episode = _episode(cfg, seed=7, tail_window=2)
    assert episode["bits_per_frame"] > 0.0
    assert episode["active_senders_per_frame"] > 0.0
    assert 0.0 <= episode["movement_target_coverage"] <= 1.0


def test_periodic_control_carrier_reduces_airtime_without_disabling_protocol():
    cfg = load_config(PILOT)
    # Isolate the periodic state beacon.  The composable certificate becomes
    # available only after the first executed allocation and therefore has a
    # deliberate one-carrier warm-up rather than a stationary 1/N bit ratio.
    cfg.marl.distributed_composable_certificate_enabled = False
    cfg.marl.distributed_owner_posterior_enabled = False
    cfg.scenario.T = 6
    every_frame = _episode(cfg, seed=7, tail_window=2, carrier_period=1)
    periodic = _episode(cfg, seed=7, tail_window=2, carrier_period=3)

    assert periodic["bits_per_frame"] == pytest.approx(
        every_frame["bits_per_frame"] / 3.0)
    assert periodic["active_senders_per_frame"] == pytest.approx(
        every_frame["active_senders_per_frame"] / 3.0)
    assert periodic["final_hyperedge_coverage"] == pytest.approx(1.0)


def test_composable_certificate_is_physically_billed_after_warmup():
    cfg = load_config(PILOT)
    cfg.scenario.T = 3
    enabled = _episode(cfg, seed=7, tail_window=2, carrier_period=1)
    cfg.marl.distributed_composable_certificate_enabled = False
    disabled = _episode(cfg, seed=7, tail_window=2, carrier_period=1)

    expected_per_sender = (
        (cfg.scenario.Q + 1)
        * cfg.marl.distributed_composable_certificate_bits_per_target
        + cfg.marl.distributed_composable_certificate_frame_bits
    )
    # Frame one has no prior allocation to certify; frames two and three do.
    expected_mean_delta = (
        cfg.scenario.K * expected_per_sender * 2.0 / 3.0)
    assert enabled["bits_per_frame"] - disabled["bits_per_frame"] == (
        pytest.approx(expected_mean_delta))
    assert enabled["certificate_payload_bits_per_sender"] == pytest.approx(
        expected_per_sender)


def test_owner_posterior_is_truth_invariant_and_uses_receiver_owner():
    cfg = load_config(PILOT)
    core = EnvironmentCore(cfg)
    core.reset()
    selected = tuple(
        (int((q + 1) % core.K), int(q % core.K), int(q))
        for q in range(core.Q)
    )
    core._prepare_owner_posterior_submission(selected)
    before_mean = core._owner_posterior_pending_mean.copy()
    before_cov = core._owner_posterior_pending_cov.copy()
    before_valid = core._owner_posterior_pending_valid.copy()

    for target in core.targets:
        target.state[0] += 317.0
        target.state[1] -= 191.0
    core._prepare_owner_posterior_submission(selected)

    np.testing.assert_array_equal(
        core._owner_posterior_pending_valid, before_valid)
    np.testing.assert_array_equal(
        core._owner_posterior_pending_mean, before_mean)
    np.testing.assert_array_equal(
        core._owner_posterior_pending_cov, before_cov)
    for q in range(core.Q):
        assert core._owner_posterior_pending_valid[q % core.K, q]


def test_owner_posterior_ci_fuses_each_delivered_frame_once():
    cfg = load_config(PILOT)
    core = EnvironmentCore(cfg)
    core.reset()
    core.t = 2
    receiver, sender, target = 0, 1, 0
    core.belief_mgr.mean[receiver, target] = np.zeros(4)
    core.belief_mgr.cov[receiver, target] = 100.0 * np.eye(4)
    packet = {
        "targets": np.asarray([target]),
        "means": np.asarray([[10.0, 5.0, 1.0, 0.0]]),
        "covariances": np.asarray([25.0 * np.eye(4)]),
        "aoi": np.asarray([0]),
        "frames": np.asarray([1]),
    }
    core._merge_owner_posterior_packet(receiver, sender, packet)
    core._fuse_delivered_owner_posteriors()

    assert core._owner_posterior_metrics[
        "owner_posterior_fused_entries"] == 1.0
    assert np.trace(core.belief_mgr.cov[receiver, target]) < 400.0
    fused_mean = core.belief_mgr.mean[receiver, target].copy()
    fused_cov = core.belief_mgr.cov[receiver, target].copy()

    core._fuse_delivered_owner_posteriors()
    assert core._owner_posterior_metrics[
        "owner_posterior_fused_entries"] == 0.0
    np.testing.assert_array_equal(
        core.belief_mgr.mean[receiver, target], fused_mean)
    np.testing.assert_array_equal(
        core.belief_mgr.cov[receiver, target], fused_cov)


def test_owner_posterior_payload_is_physically_billed_after_warmup():
    cfg = load_config(PILOT)
    cfg.marl.distributed_composable_certificate_enabled = False
    cfg.scenario.T = 3
    enabled = _episode(cfg, seed=7, tail_window=2, carrier_period=1)
    cfg.marl.distributed_owner_posterior_enabled = False
    disabled = _episode(cfg, seed=7, tail_window=2, carrier_period=1)

    target_id_bits = max(1, int(np.ceil(np.log2(cfg.scenario.Q))))
    entry_bits = (
        4 * cfg.marl.distributed_owner_posterior_mean_bits
        + 4 * cfg.marl.distributed_owner_posterior_cov_bits
        + cfg.marl.distributed_owner_posterior_aoi_bits
        + target_id_bits
    )
    expected_packet_bits = cfg.scenario.Q * entry_bits
    assert enabled["owner_posterior_payload_bits_per_frame"] == (
        pytest.approx(2.0 * expected_packet_bits / 3.0))
    assert enabled["bits_per_frame"] - disabled["bits_per_frame"] == (
        pytest.approx(2.0 * expected_packet_bits / 3.0))
    assert enabled["owner_posterior_fused_entries_per_frame"] > 0.0


def test_ca_owner_posterior_uses_six_dimensions_and_bills_them():
    cfg = load_config(PILOT)
    cfg.target.motion_model = "CA"
    cfg.marl.belief_motion_model = "CA"
    cfg.marl.distributed_composable_certificate_enabled = False
    cfg.scenario.T = 3
    enabled = _episode(cfg, seed=7, tail_window=2, carrier_period=1)

    target_id_bits = max(1, int(np.ceil(np.log2(cfg.scenario.Q))))
    entry_bits = (
        6 * cfg.marl.distributed_owner_posterior_mean_bits
        + 6 * cfg.marl.distributed_owner_posterior_cov_bits
        + cfg.marl.distributed_owner_posterior_aoi_bits
        + target_id_bits
    )
    expected_packet_bits = cfg.scenario.Q * entry_bits
    assert enabled["owner_posterior_payload_bits_per_frame"] == (
        pytest.approx(2.0 * expected_packet_bits / 3.0))
    assert enabled["owner_posterior_fused_entries_per_frame"] > 0.0


def test_scale_protocol_does_not_require_online_optimality_certificate():
    episode = {
        "qos_success": True,
        "steady": 0.9,
        "weak3": 0.8,
        "worst": 0.7,
        "bits_per_frame": 100.0,
        "active_senders_per_frame": 1.0,
        "delivery_rate": 1.0,
        "deadline_violation_rate": 0.0,
        "hyperedge_coverage": 1.0,
        "movement_target_coverage": 1.0,
        "belief_position_rmse_m": 1.0,
        "step_time_ms": 10.0,
        "step_time_p95_ms": 20.0,
        "step_time_max_ms": 25.0,
        "online_budget_ms": 100.0,
        "online_deadline_miss_rate": 0.0,
        "closed_loop_critical_path_p95_ms": 15.0,
    }
    result = _aggregate([episode], SweepCase("k8_q8", "scale_baseline"))

    assert result["passes_all_gates"]
    assert set(result["hard_and_service_gates"]) == {
        "qos_rate_ge_0_80",
        "delivery_rate_ge_0_99",
        "radio_deadline_violation_rate_le_0_01",
        "online_deadline_miss_rate_le_0_01",
        "p95_closed_loop_critical_path_within_control_period",
    }
