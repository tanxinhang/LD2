import numpy as np

from config.params import get_default_config, load_config


def test_constrained_cvar_trainer_path_is_default_off():
    cfg = get_default_config()
    assert cfg.marl.constrained_cvar_ppo_enabled is False
    assert cfg.marl.cvar_tau == 0.0


def test_constrained_cvar_smoke_profile_is_explicit_and_valid():
    cfg = load_config("config/exp_constrained_cvar_ppo_smoke.yaml")
    assert cfg.marl.constrained_cvar_ppo_enabled is True
    assert cfg.marl.cvar_tau == 0.0
    assert cfg.marl.target_kl > 0.0
    assert 0.0 < cfg.marl.constrained_cvar_tail_fraction <= 1.0
    rollback_cfg = load_config(
        "config/exp_constrained_cvar_ppo_rollback_smoke.yaml")
    assert rollback_cfg.marl.target_kl == 0.0001


def test_constrained_cvar_one_real_ppo_update_is_finite_and_budget_safe():
    import torch

    from uav_isac.agents.mappo_agent import MAPPOAgent
    from uav_isac.agents.trainer import MAPPTrainer
    from uav_isac.environment.action import ActionSpace
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = load_config("config/exp_constrained_cvar_ppo_smoke.yaml")
    cfg.marl.rollout_steps = 8
    cfg.scenario.T = 8
    env = UAVISACEnv(cfg, seed=31415)
    action_space = ActionSpace(
        cfg.uav.v_max, cfg.scenario.dt,
        rng=np.random.default_rng(31415), learn_roles=False)
    action_space.num_targets = cfg.scenario.Q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 64
    obs, _ = env.reset(seed=31415)
    agent = MAPPOAgent(
        0, obs['0'].shape[-1],
        env.core.obs_builder.get_global_state_dim(), action_space,
        num_agents=cfg.scenario.K, num_targets=cfg.scenario.Q,
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
        hidden_layers=[64, 64], device='cpu')
    trainer = MAPPTrainer(env, [agent, agent], cfg, device='cpu')
    try:
        trainer.collect_rollout()
        metrics = trainer.update()
    finally:
        env.close()

    required_finite = (
        'constrained_cvar_loss',
        'constrained_cvar_gradient_norm',
        'constrained_cvar_primal_violation',
        'constrained_cvar_dual_residual',
        'post_update_attempted_approx_kl',
        'isac_max_power_budget_violation_w',
    )
    assert all(np.isfinite(metrics[name]) for name in required_finite)
    assert metrics['constrained_cvar_gradient_finite'] == 1.0
    assert metrics['isac_max_power_budget_violation_w'] <= 1.0e-9
    if not metrics['actor_update_rejected']:
        assert metrics['post_update_approx_kl'] <= 1.5 * cfg.marl.target_kl
    assert torch.isfinite(trainer._constrained_cvar_multipliers).all()
