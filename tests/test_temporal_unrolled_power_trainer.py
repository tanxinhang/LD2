"""End-to-end gates for the multi-frame differentiable optimization path."""

import numpy as np
import pytest
import torch

from config.params import get_default_config, load_config


def test_temporal_unroll_is_default_off_and_requires_the_complete_stack():
    cfg = get_default_config()
    assert cfg.marl.temporal_unrolled_power_enabled is False

    cfg.marl.temporal_unrolled_power_enabled = True
    with pytest.raises(ValueError, match="distributed_primal_dual_power_enabled"):
        cfg.validate_runtime_boundary()

    cfg = load_config("config/exp_distributed_primal_dual_power_smoke.yaml")
    cfg.marl.temporal_unrolled_power_enabled = True
    with pytest.raises(ValueError, match="constrained_cvar_ppo_enabled"):
        cfg.validate_runtime_boundary()

    cfg = load_config("config/exp_temporal_unrolled_power_smoke.yaml")
    assert cfg.marl.temporal_feasible_structure_enabled is False
    assert cfg.marl.temporal_feasible_structure_hard_forward is True
    cfg.marl.temporal_feasible_structure_enabled = True
    cfg.marl.analytical_structure_ranking_enabled = True
    with pytest.raises(ValueError, match="analytical_structure_ranking_enabled"):
        cfg.validate_runtime_boundary()

    cfg = load_config("config/exp_temporal_unrolled_power_smoke.yaml")
    cfg.marl.temporal_feasible_structure_enabled = True
    cfg.marl.target_allocation_teacher_enabled = True
    with pytest.raises(ValueError, match="teacher-free"):
        cfg.validate_runtime_boundary()


def test_one_real_ppo_update_uses_unrolled_pareto_cvar_and_kl_transaction():
    from uav_isac.agents.mappo_agent import MAPPOAgent
    from uav_isac.agents.trainer import MAPPTrainer
    from uav_isac.environment.action import ActionSpace
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = load_config("config/exp_temporal_unrolled_power_smoke.yaml")
    cfg.scenario.T = 8
    cfg.marl.rollout_steps = 8
    cfg.marl.minibatch_size = 8
    cfg.marl.ppo_epochs = 1
    cfg.marl.temporal_unrolled_power_inner_iterations = 2
    cfg.marl.temporal_unrolled_power_pareto_max_iterations = 16
    cfg.validate_runtime_boundary()
    with torch.random.fork_rng():
        torch.manual_seed(31415)
        env = UAVISACEnv(config=cfg, seed=31415)
        action_space = ActionSpace(
            cfg.uav.v_max,
            cfg.scenario.dt,
            rng=np.random.default_rng(31415),
            learn_roles=False,
        )
        action_space.num_targets = cfg.scenario.Q
        action_space.structured_actor = True
        action_space.structured_entity_dim = 64
        obs, _ = env.reset(seed=31415)
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
            comm_payload_dim=(
                cfg.scenario.Q * cfg.marl.comm_target_token_dim),
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
        "temporal_unrolled_power_applied",
        "temporal_unrolled_power_windows",
        "temporal_unrolled_power_frames",
        "temporal_unrolled_power_cvar_residual_max",
        "temporal_unrolled_power_budget_violation_w",
        "temporal_unrolled_power_applied_gradient_norm",
        "temporal_unrolled_power_pareto_weight_negative_worst_detection",
        "temporal_unrolled_power_pareto_weight_negative_mean_detection",
        "temporal_unrolled_power_pareto_weight_communication_load",
        "temporal_unrolled_power_pareto_weight_sensing_energy",
        "temporal_unrolled_power_pareto_weight_temporal_switching",
        "post_update_attempted_approx_kl",
        "isac_comm_power_w_per_frame",
        "isac_sensing_power_w_per_frame",
        "isac_unused_power_w_per_frame",
    )
    assert all(np.isfinite(metrics[name]) for name in required)
    assert metrics["temporal_unrolled_power_applied"] == 1.0
    assert metrics["temporal_unrolled_power_windows"] >= 1.0
    assert metrics["temporal_unrolled_power_frames"] >= 2.0
    assert metrics["temporal_unrolled_power_budget_violation_w"] <= 1e-7
    assert metrics["temporal_unrolled_power_applied_gradient_norm"] > 0.0
    weight_sum = sum(
        metrics[f"temporal_unrolled_power_pareto_weight_{name}"]
        for name in (
            "negative_worst_detection", "negative_mean_detection",
            "communication_load", "sensing_energy", "temporal_switching"))
    assert weight_sum == pytest.approx(1.0, abs=1e-5)
    if not metrics["actor_update_rejected"]:
        assert metrics["post_update_approx_kl"] <= 1.5 * cfg.marl.target_kl
