import numpy as np
import torch

from config.params import load_config
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.networks import CausalContributionPredictor
from uav_isac.agents.trainer import (
    MAPPTrainer, mask_sender_from_token_observations)
from uav_isac.environment.action import ActionSpace
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.environment.env_core import (
    filter_deflection_by_local_commitments,
)
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.observation_slices import ObservationSlices
from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.utils.types import DeflectionEntry


def _comm_model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4, 8], header_bits=64,
        bandwidth_hz=1.0e5, deadline_s=0.1,
        processing_delay_s=2.0e-4, snr_threshold_db=-100.0,
        antenna_gain_dbi=0.0, carrier_hz=28.0e9, tx_power_w=0.25,
        kT=4.0e-21, noise_figure_db=4.0, dt=0.1,
    )


def test_variable_communication_power_changes_link_and_energy():
    model = _comm_model()
    pos = np.array([[0.0, 0.0, 20.0], [300.0, 0.0, 20.0]])
    msg = {0: np.linspace(-1.0, 1.0, 16)}
    rate = {0: 2}
    low_delivery, low = model.transmit(
        msg, rate, pos, tx_powers_w={0: 0.01})
    high_delivery, high = model.transmit(
        msg, rate, pos, tx_powers_w={0: 0.20})

    assert low_delivery and high_delivery
    assert high_delivery[0].snr_db > low_delivery[0].snr_db
    assert high_delivery[0].latency_s < low_delivery[0].latency_s
    assert high.per_sender_power_w[0] == 0.20
    assert low.per_sender_power_w[0] == 0.01
    assert high.total_energy_j > low.total_energy_j


def test_deflection_uses_per_uav_per_target_sensing_power():
    dc = DeflectionComputer(
        fc=28.0e9, delta_f=1.5625e4, T_sym=6.4e-5,
        M=64, N=16, kT=4.0e-21, B=1.0e6, NF_dB=4.0,
        P_sense=0.0251, P_report=0.0, ric_K=6.0, rcs=1.0,
        g_min=0.0, rng=np.random.default_rng(3), use_report_link=False,
    )
    uav_pos = np.array([[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]])
    uav_vel = np.zeros((2, 3))
    # Duplicate geometry isolates the target-wise power scaling.
    tgt_pos = np.array([[50.0, 20.0, 0.0], [50.0, 20.0, 0.0]])
    tgt_vel = np.zeros((2, 3))
    sensing_power = np.array([[0.20, 0.05], [0.10, 0.10]])
    entries = dc.compute(
        uav_pos, uav_vel, tgt_pos, tgt_vel, np.array([0, 1]),
        np.zeros(3), role_agnostic=True, sensing_power_w=sensing_power)
    d = {(e.i, e.j, e.q): e.d_raw for e in entries}

    assert np.isclose(d[(0, 1, 0)] / d[(0, 1, 1)], 4.0)
    assert np.isclose(d[(1, 0, 0)] / d[(1, 0, 1)], 1.0)


def _entry(i, j, q):
    return DeflectionEntry(
        i=i, j=j, q=q, tau=0.0, nu=0.0, alpha=1.0,
        d_raw=1.0, g_dd=1.0, chi_rep=1.0, d_eff=1.0)


def test_local_commitment_filter_limits_p0_to_endpoint_choices():
    entries = [_entry(i, j, q)
               for i in range(3) for j in range(3) if i != j
               for q in range(3)]
    power = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.7, 0.2, 0.1],
    ])

    sender_only, sender_metrics = filter_deflection_by_local_commitments(
        entries, power, topk=1, require_receiver=False)
    assert sender_only
    assert all(e.q == int(np.argmax(power[e.i])) for e in sender_only)
    assert sender_metrics['learned_comm_commitment_target_coverage'] == 2 / 3

    agreed, agreed_metrics = filter_deflection_by_local_commitments(
        entries, power, topk=1, require_receiver=True)
    assert {(e.i, e.j, e.q) for e in agreed} == {(0, 2, 0), (2, 0, 0)}
    assert agreed_metrics['learned_comm_commitment_target_coverage'] == 1 / 3
    mask = agreed_metrics['learned_comm_commitment_mask']
    assert np.array_equal(mask.argmax(axis=1), np.array([0, 1, 0]))


def test_local_commitment_filter_topk_relaxes_rendezvous_without_fallback():
    entries = [_entry(0, 1, q) for q in range(3)]
    power = np.array([[0.6, 0.3, 0.1], [0.1, 0.5, 0.4]])
    strict, _ = filter_deflection_by_local_commitments(
        entries, power, topk=1, require_receiver=True)
    relaxed, _ = filter_deflection_by_local_commitments(
        entries, power, topk=2, require_receiver=True)
    assert strict == []
    assert [e.q for e in relaxed] == [1]


def test_explicit_token_commitment_mask_overrides_sensing_power_topk():
    entries = [_entry(0, 1, q) for q in range(3)]
    power = np.array([
        [0.9, 0.08, 0.02],
        [0.8, 0.15, 0.05],
    ])
    token_mask = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    ranked, metrics = filter_deflection_by_local_commitments(
        entries,
        power,
        topk=1,
        require_receiver=True,
        commitment_mask=token_mask,
    )
    assert [entry.q for entry in ranked] == [2]
    np.testing.assert_array_equal(
        metrics['learned_comm_commitment_mask'],
        token_mask.astype(bool),
    )
    assert metrics['learned_comm_commitment_claims_per_uav'] == 1.0


def test_token_commitment_hold_and_handover_are_persistent_and_sparse():
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    cfg.marl.distributed_target_commitment_source = 'sent_token'
    cfg.marl.distributed_target_commitment_topk = 1
    cfg.marl.distributed_target_commitment_min_hold_frames = 2
    cfg.marl.distributed_target_commitment_handover_frames = 2
    cfg.marl.distributed_target_commitment_max_age_frames = 5
    env = UAVISACEnv(cfg, seed=17)
    env.reset(seed=17)
    core = env.core
    core._current_sensing_power_w = np.tile(
        np.array([0.7, 0.2, 0.08, 0.02]), (core.K, 1))

    first = np.array([1.0, 0.0, 0.0, 0.0])
    second = np.array([0.0, 1.0, 0.0, 0.0])
    core.t = 0
    core._last_sent_comm_token_masks = {0: first}
    mask0 = core._resolve_persistent_token_commitments()
    np.testing.assert_array_equal(mask0[0], first.astype(bool))

    core.t = 1
    core._last_sent_comm_token_masks = {0: second}
    held = core._resolve_persistent_token_commitments()
    np.testing.assert_array_equal(held[0], first.astype(bool))

    core.t = 2
    handed = core._resolve_persistent_token_commitments()
    np.testing.assert_array_equal(
        handed[0], np.logical_or(first, second))
    assert core._persistent_commitment_metrics[
        'learned_comm_commitment_switch_rate'] == 1.0 / core.K

    core.t = 3
    grace = core._resolve_persistent_token_commitments()
    np.testing.assert_array_equal(
        grace[0], np.logical_or(first, second))

    core.t = 4
    settled = core._resolve_persistent_token_commitments()
    np.testing.assert_array_equal(settled[0], second.astype(bool))
    assert settled[0].sum() == 1


def test_soft_local_commitment_keeps_graph_and_relaxes_uncertain_claims():
    entries = [_entry(0, 1, q) for q in range(3)]
    uniform = np.full((2, 3), 1.0 / 3.0)
    ranked, metrics = filter_deflection_by_local_commitments(
        entries, uniform, topk=1, require_receiver=True,
        mode='soft', soft_floor=0.25, uncertainty_relief=1.0)
    assert len(ranked) == len(entries)
    assert all(np.isclose(entry.d_eff, 1.0) for entry in ranked)
    assert metrics['learned_comm_commitment_candidate_fraction'] == 1.0
    assert metrics['learned_comm_commitment_mode'] == 'soft'

    sharp = np.array([[0.98, 0.01, 0.01], [0.98, 0.01, 0.01]])
    ranked, metrics = filter_deflection_by_local_commitments(
        entries, sharp, topk=1, require_receiver=True,
        mode='soft', soft_floor=0.25, uncertainty_relief=1.0)
    scores = np.array([entry.d_eff for entry in ranked])
    assert scores[0] == 1.0
    assert np.all(scores[1:] >= 0.25)
    assert np.all(scores[1:] < scores[0])
    assert metrics['learned_comm_commitment_hard_target_coverage'] == 1 / 3


def test_joint_budget_is_exact_and_communication_repeats_each_frame():
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    assert cfg.uav.P_isac_total == 1.0
    assert cfg.marl.comm_power_fraction_min == 0.0
    assert cfg.marl.comm_power_fraction_max == 1.0
    cfg.scenario.T = 3
    env = UAVISACEnv(cfg, seed=11)
    env.reset(seed=11)
    K, Q = cfg.scenario.K, cfg.scenario.Q
    actions = {
        str(k): {'delta_p': np.zeros(2), 'role': 2} for k in range(K)}

    sent_frames = []
    for round_idx in range(2):
        messages = {
            k: np.full(Q * cfg.marl.comm_target_token_dim,
                       0.1 * (round_idx + 1) * (k + 1))
            for k in range(K)}
        rates = {k: 2 for k in range(K)}
        fractions = {k: 0.10 + 0.05 * k for k in range(K)}
        weights = {
            k: np.roll(np.array([0.55, 0.25, 0.15, 0.05]), k)
            for k in range(K)}
        env.core.submit_learned_communications(
            messages, rates, fractions, weights)
        next_obs, _, _, _, info = env.step(actions)
        assert info['learned_comm_bits'] > 0.0
        assert info['isac_max_power_balance_error_w'] < 1e-12
        combined = (
            info['isac_per_uav_comm_power_w']
            + info['isac_per_uav_sensing_power_w'])
        assert np.allclose(combined, cfg.uav.P_isac_total)
        assert info['isac_target_power_w'].shape == (Q,)
        assert np.all(info['isac_target_power_w'] > 0.0)
        metadata = env.core._received_comm_meta.get(1, {}).get(0, {})
        sent_frames.append(metadata.get('sent_frame'))

    slices = ObservationSlices.from_config(
        K=K, Q=Q, use_comm_tokens=True,
        comm_token_dim=cfg.marl.comm_target_token_dim + 6,
        comm_tokens_per_sender=Q)
    received_tokens = slices.extract_comm_tokens(next_obs['1'])
    received_mask = slices.extract_comm_mask(next_obs['1'])
    assert received_tokens.shape == (
        (K - 1) * Q, cfg.marl.comm_target_token_dim + 6)
    assert received_mask.shape == ((K - 1) * Q,)
    assert np.all(received_mask > 0.5)
    # Metadata target-id column must identify all Q target tokens per sender.
    assert np.allclose(
        received_tokens[:Q, cfg.marl.comm_target_token_dim + 1],
        np.linspace(0.0, 1.0, Q))

    assert sent_frames == [1, 2]


def test_joint_resource_actions_preserve_ppo_log_probability():
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    cfg.scenario.K = 2
    cfg.scenario.Q = 2
    cfg.target.omega_q = [0.5, 0.5]
    env = UAVISACEnv(cfg, seed=19)
    obs, _ = env.reset(seed=19)
    obs_np = np.stack([obs[str(k)] for k in range(2)])
    action_space = ActionSpace(
        cfg.uav.v_max, cfg.scenario.dt, rng=np.random.default_rng(19),
        learn_roles=False)
    action_space.structured_actor = True
    action_space.structured_entity_dim = 64
    agent = MAPPOAgent(
        0, obs_np.shape[-1], env.core.obs_builder.get_global_state_dim(),
        action_space, num_agents=2, num_targets=2,
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
    )
    with torch.no_grad():
        agent.actor.comm_rate_head.bias.fill_(-10.0)
        agent.actor.comm_rate_head.bias[2] = 10.0

    obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
    with torch.no_grad():
        dp_mean, dp_log_std, role_logits, comm_mean, _, _ = agent.actor(obs_t)
        assert comm_mean.shape == (2, 2 * cfg.marl.comm_target_token_dim)
        comm, rate, comm_lp, _ = agent.sample_communication(comm_mean)
        power_raw, sensing_raw, _, _, resource_lp, _ = (
            agent.sample_isac_resources(comm_mean, rate))

    dp = np.zeros((2, 2))
    roles = np.zeros(2, dtype=np.int64)
    move_lp = np.zeros(2)
    for k in range(2):
        action, lp = action_space.decode(
            dp_mean[k].numpy(), dp_log_std.numpy(), role_logits[k].numpy())
        dp[k] = action.delta_p
        roles[k] = action.role
        move_lp[k] = lp
    old_lp = torch.as_tensor(move_lp) + comm_lp.double() + resource_lp.double()
    passed, max_diff = agent.verify_old_log_prob_consistency(
        obs_t, torch.as_tensor(dp, dtype=torch.float32),
        torch.as_tensor(roles), old_lp.float(),
        actions_comm=comm,
        actions_comm_rate=rate,
        actions_isac_power_raw=power_raw,
        actions_sensing_raw=sensing_raw,
    )
    assert passed, max_diff

    gs = torch.as_tensor(np.repeat(
        env.core.get_global_state()[None, :], 2, axis=0), dtype=torch.float32)
    agent.actor.zero_grad(set_to_none=True)
    new_lp, *_ = agent.evaluate_actions(
        obs_t, gs, torch.as_tensor(dp, dtype=torch.float32),
        torch.as_tensor(roles), actions_comm=comm,
        actions_comm_rate=rate, actions_isac_power_raw=power_raw,
        actions_sensing_raw=sensing_raw)
    (-new_lp.mean()).backward()
    power_grad = agent.actor.isac_power_mean_head.weight.grad
    sensing_grad = agent.actor.isac_sensing_mean_head.weight.grad
    assert power_grad is not None and torch.any(power_grad != 0)
    assert sensing_grad is not None and torch.any(sensing_grad != 0)


def _small_joint_agent(cfg, env, k=2, q=2):
    action_space = ActionSpace(
        cfg.uav.v_max, cfg.scenario.dt, rng=np.random.default_rng(23),
        learn_roles=False)
    action_space.num_targets = q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 64
    obs, _ = env.reset(seed=23)
    obs_dim = obs['0'].shape[-1]
    return MAPPOAgent(
        0, obs_dim, env.core.obs_builder.get_global_state_dim(),
        action_space, num_agents=k, num_targets=q,
        comm_num_rate_levels=len(cfg.marl.comm_rate_bits_per_dim),
        use_comm_cross_attention=True,
        comm_token_dim=cfg.marl.comm_target_token_dim + 6,
        comm_tokens_per_sender=q,
        comm_payload_dim=q * cfg.marl.comm_target_token_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=cfg.marl.comm_target_token_dim,
        isac_power_log_std_init=cfg.marl.isac_power_log_std_init,
        sensing_allocation_log_std_init=(
            cfg.marl.sensing_allocation_log_std_init),
        hidden_layers=[64, 64], device='cpu')


def test_held_movement_mask_keeps_communication_and_resource_log_prob():
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    cfg.scenario.K = cfg.scenario.Q = 2
    cfg.target.omega_q = [0.5, 0.5]
    env = UAVISACEnv(cfg, seed=23)
    agent = _small_joint_agent(cfg, env)
    obs, _ = env.reset(seed=23)
    obs_t = torch.as_tensor(
        np.stack([obs['0'], obs['1']]), dtype=torch.float32)

    with torch.no_grad():
        dp_mean, dp_log_std, role_logits, comm_mean, _, _ = agent.actor(obs_t)
        comm, rate, comm_lp, _ = agent.sample_communication(comm_mean)
        power_raw, sensing_raw, _, _, resource_lp, _ = (
            agent.sample_isac_resources(comm_mean, rate))
    actions_dp = np.zeros((2, 2), dtype=np.float32)
    actions_role = np.zeros(2, dtype=np.int64)
    for idx in range(2):
        action, _ = agent.action_space.decode(
            dp_mean[idx].numpy(), dp_log_std.numpy(), role_logits[idx].numpy())
        actions_dp[idx] = action.delta_p

    movement_mask = torch.zeros(2)
    old_lp = comm_lp + resource_lp
    passed, max_diff = agent.verify_old_log_prob_consistency(
        obs_t, torch.as_tensor(actions_dp), torch.as_tensor(actions_role),
        old_lp, actions_comm=comm, actions_comm_rate=rate,
        actions_isac_power_raw=power_raw, actions_sensing_raw=sensing_raw,
        movement_action_mask=movement_mask)
    assert passed, max_diff

    gs = torch.as_tensor(np.repeat(
        env.core.get_global_state()[None, :], 2, axis=0), dtype=torch.float32)
    agent.actor.zero_grad(set_to_none=True)
    new_lp, *_ = agent.evaluate_actions(
        obs_t, gs, torch.as_tensor(actions_dp), torch.as_tensor(actions_role),
        actions_comm=comm, actions_comm_rate=rate,
        actions_isac_power_raw=power_raw, actions_sensing_raw=sensing_raw,
        movement_action_mask=movement_mask)
    (-new_lp.mean()).backward()
    dp_grad = agent.actor.dp_head.weight.grad
    assert dp_grad is None or torch.all(dp_grad == 0)
    assert torch.any(agent.actor.isac_sensing_mean_head.weight.grad != 0)
    env.close()


def test_fast_token_rounds_hold_only_movement():
    cfg = load_config('config/exp_800_q4_u2u_joint_isac.yaml')
    cfg.scenario.K = cfg.scenario.Q = 2
    cfg.scenario.T = 4
    cfg.target.omega_q = [0.5, 0.5]
    cfg.marl.num_envs = 1
    cfg.marl.rollout_steps = 4
    cfg.marl.ppo_epochs = 1
    cfg.marl.minibatch_size = 4
    # The 800_q4 selection/confirmation splits still contain quarantined
    # seeds 795/747; use the clean test split and disable checkpoint
    # confirmation so trainer construction does not fail closed on the legacy
    # bank (see tests/test_quarantined_seeds.py).
    cfg.marl.eval_seed_split = "test"
    cfg.marl.checkpoint_confirmation_enabled = False
    assert cfg.marl.actor_decision_interval == 1
    assert cfg.marl.movement_decision_interval == 2

    env = UAVISACEnv(cfg, seed=29)
    shared = _small_joint_agent(cfg, env)
    agents = [shared, shared]
    with torch.no_grad():
        shared.actor.comm_rate_head.weight.zero_()
        shared.actor.comm_rate_head.bias.fill_(-20.0)
        shared.actor.comm_rate_head.bias[2] = 20.0
    trainer = MAPPTrainer(env, agents, cfg, device='cpu')

    submit_calls = 0
    original_submit = env.core.submit_learned_communications

    def counted_submit(*args, **kwargs):
        nonlocal submit_calls
        submit_calls += 1
        return original_submit(*args, **kwargs)

    env.core.submit_learned_communications = counted_submit
    trainer.collect_rollout()

    assert submit_calls == 4
    np.testing.assert_array_equal(
        trainer.buffer.movement_action_masks[:4, 0], [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(
        trainer.buffer.comm_round_phases[:4, 0], [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        trainer.buffer.actions_dp[0], trainer.buffer.actions_dp[1])
    np.testing.assert_allclose(
        trainer.buffer.actions_dp[2], trainer.buffer.actions_dp[3])
    assert all(bits > 0 for bits in trainer._rollout_learned_comm_bits)
    env.close()


def test_headwise_log_probs_sum_to_exact_joint_density():
    cfg = load_config('config/exp_800_q4_u2u_headwise_credit.yaml')
    cfg.scenario.K = cfg.scenario.Q = 2
    cfg.target.omega_q = [0.5, 0.5]
    env = UAVISACEnv(cfg, seed=31)
    agent = _small_joint_agent(cfg, env)
    obs, _ = env.reset(seed=31)
    obs_t = torch.as_tensor(
        np.stack([obs['0'], obs['1']]), dtype=torch.float32)

    with torch.no_grad():
        dp_mean, dp_log_std, role_logits, comm_mean, _, _ = agent.actor(obs_t)
        comm, rate, comm_lp, _, sampled_parts = agent.sample_communication(
            comm_mean, return_components=True)
        power_raw, sensing_raw, _, _, resource_lp, _ = (
            agent.sample_isac_resources(comm_mean, rate))
    actions_dp = np.zeros((2, 2), dtype=np.float32)
    actions_role = np.zeros(2, dtype=np.int64)
    movement_lp = np.zeros(2, dtype=np.float32)
    for idx in range(2):
        action, lp = agent.action_space.decode(
            dp_mean[idx].numpy(), dp_log_std.numpy(), role_logits[idx].numpy())
        actions_dp[idx] = action.delta_p
        movement_lp[idx] = lp

    gs = torch.as_tensor(np.repeat(
        env.core.get_global_state()[None, :], 2, axis=0), dtype=torch.float32)
    evaluated = agent.evaluate_actions(
        obs_t, gs, torch.as_tensor(actions_dp),
        torch.as_tensor(actions_role), actions_comm=comm,
        actions_comm_rate=rate, actions_isac_power_raw=power_raw,
        actions_sensing_raw=sensing_raw,
        return_log_prob_components=True)
    new_joint = evaluated[0]
    parts = evaluated[-1]['log_probs']
    sampled_joint = (
        torch.as_tensor(movement_lp) + comm_lp + resource_lp)

    torch.testing.assert_close(parts.sum(dim=-1), new_joint, atol=3e-6, rtol=0)
    torch.testing.assert_close(new_joint, sampled_joint, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        sampled_parts['message'] + sampled_parts['rate'], comm_lp,
        atol=1e-6, rtol=0)
    assert evaluated[-1]['credit_values'].shape == (2, 4)
    env.close()


def test_headwise_credit_rollout_builds_four_gae_streams_and_updates():
    cfg = load_config('config/exp_800_q4_u2u_headwise_credit.yaml')
    cfg.scenario.K = cfg.scenario.Q = 2
    cfg.scenario.T = 4
    cfg.target.omega_q = [0.5, 0.5]
    cfg.marl.num_envs = 1
    cfg.marl.rollout_steps = 4
    cfg.marl.ppo_epochs = 1
    cfg.marl.minibatch_size = 4
    cfg.marl.early_stop = False
    # 800_q4 selection/confirmation still contain quarantined seeds 795/747;
    # use the clean test split and disable confirmation (see
    # tests/test_quarantined_seeds.py).
    cfg.marl.eval_seed_split = "test"
    cfg.marl.checkpoint_confirmation_enabled = False

    env = UAVISACEnv(cfg, seed=37)
    shared = _small_joint_agent(cfg, env)
    agents = [shared, shared]
    with torch.no_grad():
        shared.actor.comm_rate_head.weight.zero_()
        shared.actor.comm_rate_head.bias.fill_(-20.0)
        shared.actor.comm_rate_head.bias[2] = 20.0
    trainer = MAPPTrainer(env, agents, cfg, device='cpu')
    trainer.collect_rollout()
    data = trainer.buffer.get_training_data()

    assert data['old_head_log_probs'].shape == (8, 4)
    assert data['credit_advantages'].shape == (8, 4)
    assert data['credit_returns'].shape == (8, 4)
    torch.testing.assert_close(
        data['old_head_log_probs'].sum(dim=-1), data['old_log_probs'],
        atol=1e-5, rtol=0)
    # Held movement has zero movement density, while communication/resource
    # remain genuine samples with their own advantages.
    held = data['movement_action_masks'] == 0
    assert torch.all(data['old_head_log_probs'][held, 0] == 0)
    assert torch.any(data['old_head_log_probs'][held, 1:] != 0)
    # The last token has no next transition inside this rollout, so it must not
    # receive same-frame sensing credit. Earlier rows are back-filled only by
    # the following frame's coordination result.
    assert np.allclose(trainer.buffer.credit_rewards[3, :, 1], 0.0)

    metrics = trainer.update()
    for name in ('movement', 'message', 'rate', 'resource'):
        assert np.isfinite(metrics[f'actor_loss_{name}'])
        assert np.isfinite(metrics[f'approx_kl_{name}'])
    assert np.isfinite(metrics['credit_critic_loss'])
    env.close()


def test_headwise_rate_loss_does_not_rewrite_token_encoder():
    cfg = load_config('config/exp_800_q4_u2u_headwise_credit.yaml')
    cfg.scenario.K = cfg.scenario.Q = 2
    cfg.target.omega_q = [0.5, 0.5]
    env = UAVISACEnv(cfg, seed=41)
    agent = _small_joint_agent(cfg, env)
    obs, _ = env.reset(seed=41)
    obs_t = torch.as_tensor(
        np.stack([obs['0'], obs['1']]), dtype=torch.float32)
    with torch.no_grad():
        dp_mean, dp_log_std, role_logits, comm_mean, _, _ = agent.actor(obs_t)
        actions_dp = []
        for idx in range(2):
            action, _ = agent.action_space.decode(
                dp_mean[idx].numpy(), dp_log_std.numpy(),
                role_logits[idx].numpy())
            actions_dp.append(action.delta_p)
    actions_dp = torch.as_tensor(np.asarray(actions_dp), dtype=torch.float32)
    actions_role = torch.zeros(2, dtype=torch.long)
    actions_rate = torch.tensor([1, 2], dtype=torch.long)
    actions_comm = comm_mean.detach().clone()
    gs = torch.as_tensor(np.repeat(
        env.core.get_global_state()[None, :], 2, axis=0), dtype=torch.float32)

    agent.actor.zero_grad(set_to_none=True)
    evaluated = agent.evaluate_actions(
        obs_t, gs, actions_dp, actions_role,
        actions_comm=actions_comm, actions_comm_rate=actions_rate,
        return_log_prob_components=True)
    rate_loss = -evaluated[-1]['log_probs'][:, 2].mean()
    rate_loss.backward()

    assert torch.any(agent.actor.comm_rate_head.weight.grad != 0)
    token_grad = agent.actor.comm_target_token_head.weight.grad
    assert token_grad is None or torch.all(token_grad == 0)
    env.close()


def test_sender_virtual_intervention_clears_content_metadata_and_mask():
    slices = ObservationSlices.from_config(
        K=3, Q=2, use_comm_tokens=True,
        comm_token_dim=6, comm_tokens_per_sender=2)
    observations = {}
    for receiver in range(3):
        row = np.zeros(slices.total_dim, dtype=np.float64)
        token_count = (slices.K - 1) * slices.comm_tokens_per_sender
        token_len = token_count * slices.comm_token_per_sender
        row[slices.comm_token_start:
            slices.comm_token_start + token_len] = np.arange(
                1, token_len + 1)
        row[slices.comm_mask_start:
            slices.comm_mask_start + token_count] = 1.0
        observations[str(receiver)] = row

    masked, changed = mask_sender_from_token_observations(
        observations, sender=1, slices=slices)
    assert changed
    for receiver in (0, 2):
        peers = [peer for peer in range(3) if peer != receiver]
        slot = peers.index(1)
        token0 = (slices.comm_token_start
                  + slot * 2 * slices.comm_token_per_sender)
        token1 = token0 + 2 * slices.comm_token_per_sender
        mask0 = slices.comm_mask_start + slot * 2
        assert np.all(masked[str(receiver)][token0:token1] == 0.0)
        assert np.all(masked[str(receiver)][mask0:mask0 + 2] == 0.0)
    np.testing.assert_array_equal(masked['1'], observations['1'])


def test_causal_contribution_predictor_starts_neutral_and_learns_effects():
    predictor = CausalContributionPredictor(
        obs_dim=12, comm_dim=8, num_targets=3,
        num_rate_levels=4, hidden_dim=32)
    obs = torch.randn(6, 12)
    message = torch.randn(6, 8)
    rate = torch.tensor([1, 2, 3, 1, 2, 3])
    initial = predictor(obs, message, rate)
    torch.testing.assert_close(initial, torch.zeros_like(initial))
    target = torch.full_like(initial, 0.05)
    loss = torch.nn.functional.smooth_l1_loss(initial, target)
    loss.backward()
    assert torch.any(predictor.effect_head.weight.grad != 0)


def test_causal_ccp_rollout_collects_paired_interventions_and_updates():
    cfg = load_config('config/exp_800_q4_u2u_causal_ccp.yaml')
    cfg.scenario.K = cfg.scenario.Q = 2
    cfg.scenario.T = 12
    cfg.target.omega_q = [0.5, 0.5]
    cfg.marl.num_envs = 1
    cfg.marl.rollout_steps = 8
    cfg.marl.ppo_epochs = 1
    cfg.marl.minibatch_size = 8
    cfg.marl.early_stop = False
    cfg.marl.causal_ccp_intervention_stride = 1
    cfg.marl.causal_ccp_min_samples = 4
    cfg.marl.causal_ccp_epochs = 2
    # 800_q4 selection/confirmation still contain quarantined seeds 795/747;
    # use the clean test split and disable confirmation (see
    # tests/test_quarantined_seeds.py).
    cfg.marl.eval_seed_split = "test"
    cfg.marl.checkpoint_confirmation_enabled = False

    env = UAVISACEnv(cfg, seed=47)
    shared = _small_joint_agent(cfg, env)
    with torch.no_grad():
        shared.actor.comm_rate_head.weight.zero_()
        shared.actor.comm_rate_head.bias.fill_(-20.0)
        shared.actor.comm_rate_head.bias[2] = 20.0
    trainer = MAPPTrainer(env, [shared, shared], cfg, device='cpu')
    trainer.collect_rollout()
    data = trainer.buffer.get_training_data()

    assert 'causal_teacher_effects' in data
    assert int(data['causal_teacher_masks'].sum().item()) >= 4
    assert data['causal_teacher_effects'].shape == (16, 2)
    metrics = trainer.update()
    assert metrics['causal_ccp_samples'] >= 4
    assert np.isfinite(metrics['causal_ccp_loss'])
    assert np.isfinite(metrics['causal_ccp_val_corr'])
    env.close()
