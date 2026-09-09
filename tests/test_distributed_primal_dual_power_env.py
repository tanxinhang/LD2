"""Environment gate tests for the bounded distributed power block."""

import numpy as np
import pytest

from config.params import get_default_config, load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def _zero_actions(env):
    return {
        str(k): {"delta_p": np.zeros(2), "role": 2}
        for k in range(env.K)
    }


def test_distributed_primal_dual_smoke_is_hard_safe_and_warm_started():
    cfg = load_config("config/exp_distributed_primal_dual_power_smoke.yaml")
    cfg.validate_runtime_boundary()
    env = UAVISACEnv(config=cfg, seed=1)
    env.reset(seed=1)
    infos = []
    try:
        for _ in range(2):
            _, _, _, _, info = env.step(_zero_actions(env))
            infos.append(info)
            assert info["distributed_primal_dual_power_enabled"] == 1.0
            assert info["isac_max_power_budget_violation_w"] < 1.0e-12
            comm = np.asarray(info["isac_per_uav_comm_power_w"])
            sense = np.asarray(info["isac_per_uav_sensing_power_w"])
            assert np.all(comm + sense <= cfg.uav.P_isac_total + 1.0e-12)
            assert np.all(sense <= cfg.uav.P_sense_max + 1.0e-12)
            assert info["distributed_primal_dual_power_budget_violation_w"] == 0.0
            assert info["distributed_primal_dual_power_consensus_residual"] <= 1.0e-12
            assert (
                info["distributed_primal_dual_power_candidate_violation"]
                <= info["distributed_primal_dual_power_baseline_violation"]
                + 1.0e-10
            )
        assert infos[0]["distributed_primal_dual_power_warm_started"] == 0.0
        assert infos[1]["distributed_primal_dual_power_warm_started"] == 1.0
        # With this deterministic seed the reserve is reachable; even when
        # the fixed iteration budget stops before stationarity, the selected
        # iterate must satisfy the target residual gate.
        assert infos[0]["distributed_primal_dual_power_selected_violation"] <= 2.0e-4
    finally:
        env.close()


def test_distributed_primal_dual_config_is_fail_closed():
    cfg = get_default_config()
    cfg.marl.distributed_primal_dual_power_enabled = True
    with pytest.raises(ValueError, match="analytical_sensing_power_enabled"):
        cfg.validate_runtime_boundary()

    cfg = get_default_config()
    cfg.marl.analytical_sensing_power_enabled = True
    cfg.marl.distributed_primal_dual_power_enabled = True
    with pytest.raises(ValueError, match="joint_isac_power_enabled"):
        cfg.validate_runtime_boundary()

    cfg = get_default_config()
    cfg.marl.joint_isac_power_enabled = True
    cfg.marl.analytical_sensing_power_enabled = True
    cfg.marl.analytical_sensing_power_reserve_pd = 0.6
    cfg.marl.distributed_primal_dual_power_enabled = True
    cfg.marl.distributed_replicated_power_enabled = True
    with pytest.raises(ValueError, match="replicated_power_enabled"):
        cfg.validate_runtime_boundary()


def test_distributed_power_diagnostics_reach_one_ppo_update():
    """The trainer records bounded-power health without changing PPO inputs."""
    from uav_isac.agents.mappo_agent import MAPPOAgent
    from uav_isac.agents.trainer import MAPPTrainer
    from uav_isac.environment.action import ActionSpace

    cfg = load_config("config/exp_distributed_primal_dual_power_smoke.yaml")
    cfg.scenario.T = 8
    cfg.marl.rollout_steps = 8
    cfg.marl.minibatch_size = 8
    cfg.marl.ppo_epochs = 1
    env = UAVISACEnv(config=cfg, seed=1)
    action_space = ActionSpace(
        cfg.uav.v_max,
        cfg.scenario.dt,
        rng=np.random.default_rng(1),
        learn_roles=False,
    )
    action_space.num_targets = cfg.scenario.Q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 64
    obs, _ = env.reset(seed=1)
    agent = MAPPOAgent(
        0,
        obs["0"].shape[-1],
        env.core.obs_builder.get_global_state_dim(),
        action_space,
        num_agents=cfg.scenario.K,
        num_targets=cfg.scenario.Q,
        comm_num_rate_levels=len(cfg.marl.comm_rate_bits_per_dim),
        use_comm_cross_attention=True,
        comm_token_dim=cfg.marl.comm_target_token_dim + 6,
        comm_tokens_per_sender=cfg.scenario.Q,
        comm_payload_dim=cfg.scenario.Q * cfg.marl.comm_target_token_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=cfg.marl.comm_target_token_dim,
        isac_power_log_std_init=cfg.marl.isac_power_log_std_init,
        sensing_allocation_log_std_init=(
            cfg.marl.sensing_allocation_log_std_init),
        hidden_layers=[32, 32],
        device="cpu",
    )
    trainer = MAPPTrainer(
        env, [agent] * cfg.scenario.K, cfg, device="cpu")
    try:
        trainer.collect_rollout()
        metrics = trainer.update()
    finally:
        env.close()

    required = (
        "distributed_primal_dual_power_selected_violation",
        "distributed_primal_dual_power_candidate_stationarity",
        "distributed_primal_dual_power_candidate_dual_residual",
        "distributed_primal_dual_power_consensus_residual",
        "distributed_primal_dual_power_target_price_max",
        "distributed_primal_dual_power_fallback",
        "distributed_primal_dual_power_rounds",
        "isac_max_power_budget_violation_w",
    )
    assert all(np.isfinite(metrics[name]) for name in required)
    assert metrics["distributed_primal_dual_power_selected_violation"] <= 2.0e-4
    assert metrics["distributed_primal_dual_power_consensus_residual"] <= 1.0e-12
    assert metrics["isac_max_power_budget_violation_w"] <= 1.0e-9


def test_distributed_power_warm_state_roundtrip_is_reproducible():
    cfg = load_config("config/exp_distributed_primal_dual_power_smoke.yaml")
    env = UAVISACEnv(config=cfg, seed=1)
    env.reset(seed=1)
    actions = _zero_actions(env)
    try:
        env.step(actions)
        checkpoint = env.get_state()
        _, _, _, _, first = env.step(actions)
        first_power = env.core._current_sensing_power_w.copy()

        env.set_state(checkpoint)
        _, _, _, _, replay = env.step(actions)
        replay_power = env.core._current_sensing_power_w.copy()
    finally:
        env.close()

    np.testing.assert_allclose(replay_power, first_power, rtol=0.0, atol=1.0e-12)
    for name in (
            "distributed_primal_dual_power_selected_violation",
            "distributed_primal_dual_power_candidate_dual_residual",
            "distributed_primal_dual_power_target_price_max",
            "distributed_primal_dual_power_warm_started"):
        assert replay[name] == first[name]
