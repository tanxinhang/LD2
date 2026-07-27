"""Regression tests for tracking-free, cost-aware emergent U2U communication."""

import numpy as np
import torch

from config.params import get_default_config, load_config
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.networks import (
    StructuredActorNetwork,
    capacity_sinkhorn_normalize,
    sinkhorn_normalize,
)
from uav_isac.agents.trainer import (
    MAPPTrainer,
    build_comm_qos_checkpoint_key,
    build_differentiable_u2u_inbox,
    compute_balanced_assignment_teacher,
    compute_qos_bistatic_assignment_teacher,
    compute_comm_encouragement_bonuses,
    compute_comm_rate_exploration_bonuses,
    compute_comm_qos_metrics,
    compute_comm_qos_rate_penalties,
    compute_sender_delivery_penalties,
    compute_target_allocation_regularizer,
    compute_sparse_endpoint_load_loss,
    compute_target_allocation_temporal_loss,
    update_long_silence_penalties,
)
from uav_isac.environment.action import ActionSpace
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.observation_slices import ObservationSlices


def _transport(deadline_s: float = 0.005):
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4, 8],
        header_bits=64,
        bandwidth_hz=1.0e5,
        deadline_s=deadline_s,
        processing_delay_s=2.0e-4,
        snr_threshold_db=0.0,
        antenna_gain_dbi=0.0,
        carrier_hz=28.0e9,
        tx_power_w=0.25,
        kT=4.0e-21,
        noise_figure_db=4.0,
        dt=0.1,
        message_dim=16,
    )


def test_medium_u2u_config_inherits_tracking_free_communication_defaults():
    cfg = load_config('config/exp_800_q4_u2u_teacher.yaml')
    assert tuple(cfg.scenario.region_size) == (800, 800)
    assert cfg.scenario.Q == 4
    assert cfg.marl.tracking_enabled is False
    assert cfg.marl.ground_communication_enabled is False
    assert cfg.marl.learned_comm_mode == 'cost_aware'
    assert cfg.marl.comm_only_neighbor_information is True
    assert cfg.marl.comm_cross_attention_enabled is True
    assert cfg.marl.comm_rate_bits_per_dim == [0, 4, 8, 16, 32]
    assert cfg.marl.target_allocation_teacher_enabled is True


def test_canonical_medium_u2u_config_uses_soft_communication_objective():
    cfg = load_config('config/exp_800_q4_u2u.yaml')
    assert tuple(cfg.scenario.region_size) == (800, 800)
    assert cfg.scenario.Q == 4
    assert cfg.marl.tracking_enabled is False
    assert cfg.marl.ground_communication_enabled is False
    assert cfg.marl.learned_comm_mode == 'cost_aware'
    assert cfg.marl.comm_encouragement_enabled is True
    assert cfg.marl.comm_qos_min_rate_bits == 0
    assert cfg.marl.target_allocation_enabled is False
    assert cfg.marl.target_allocation_teacher_enabled is False
    assert cfg.marl.target_allocation_sinkhorn_enabled is False

    silence_cfg = load_config('config/exp_800_q4_u2u_silence.yaml')
    assert silence_cfg.marl.learned_comm_mode == 'cost_aware'
    assert silence_cfg.marl.comm_eval_force_silence is True

    zero_cfg = load_config('config/exp_800_q4_u2u_zero_message.yaml')
    permute_cfg = load_config('config/exp_800_q4_u2u_permute_message.yaml')
    assert zero_cfg.marl.comm_eval_message_ablation == 'zero'
    assert permute_cfg.marl.comm_eval_message_ablation == 'permute'

    central_cfg = load_config(
        'config/exp_800_q4_u2u_central_assignment.yaml')
    assert central_cfg.marl.eval_centralized_assignment_movement is True
    assert central_cfg.marl.eval_qos_bistatic_assignment_movement is False

    commitment_cfg = load_config(
        'config/exp_800_q4_u2u_commitment.yaml')
    assert commitment_cfg.marl.target_allocation_enabled is True
    assert commitment_cfg.marl.target_allocation_teacher_enabled is False
    assert commitment_cfg.marl.target_allocation_differentiable_comm is True
    assert commitment_cfg.marl.target_allocation_straight_through is True
    assert commitment_cfg.marl.target_allocation_movement_blend == 0.25
    sinkhorn_cfg = load_config(
        'config/exp_800_q4_u2u_commitment_sinkhorn.yaml')
    assert sinkhorn_cfg.marl.target_allocation_sinkhorn_enabled is True
    commitment_teacher_cfg = load_config(
        'config/exp_800_q4_u2u_commitment_teacher.yaml')
    assert commitment_teacher_cfg.marl.target_allocation_enabled is True
    assert commitment_teacher_cfg.marl.target_allocation_teacher_enabled is True
    assert (commitment_teacher_cfg.marl
            .target_allocation_teacher_differentiable_comm is True)
    pretrain_cfg = load_config(
        'config/exp_800_q4_u2u_commitment_pretrain.yaml')
    assert pretrain_cfg.marl.ppo_epochs == 0
    assert pretrain_cfg.marl.target_allocation_movement_blend == 0.05
    assert pretrain_cfg.marl.target_allocation_teacher_epochs == 32


def test_qos_rate_barrier_requires_precision_only_until_floors_are_met():
    levels = [0, 4, 8, 16, 32]
    rates = np.arange(len(levels))
    targets = np.array([0.80, 0.70, 0.60])
    infeasible = compute_comm_qos_rate_penalties(
        rates, levels, np.array([0.70, 0.55, 0.30]), targets,
        min_rate_bits=8, penalty_scale=0.10)
    assert infeasible[0] > infeasible[1] > 0.0
    np.testing.assert_allclose(infeasible[2:], 0.0)

    feasible = compute_comm_qos_rate_penalties(
        rates, levels, targets, targets,
        min_rate_bits=8, penalty_scale=0.10)
    np.testing.assert_allclose(feasible, 0.0)


def test_soft_comm_encouragement_rewards_successful_activity_not_bitrate():
    targets = np.array([0.80, 0.70, 0.60])
    bonuses = compute_comm_encouragement_bonuses(
        rate_indices=np.array([0, 1, 2, 4]),
        qos_values=np.array([0.70, 0.55, 0.30]),
        qos_targets=targets,
        delivery_rate=1.0,
        bonus_weight=0.05,
        floor_ratio=0.10,
    )
    assert bonuses[0] == 0.0
    assert bonuses[1] > 0.0
    np.testing.assert_allclose(bonuses[1:], bonuses[1])


def test_soft_comm_encouragement_is_delivery_and_qos_gated():
    targets = np.array([0.80, 0.70, 0.60])
    failed = compute_comm_encouragement_bonuses(
        np.array([1, 1]), targets, targets,
        delivery_rate=0.0, bonus_weight=0.05, floor_ratio=0.10)
    np.testing.assert_allclose(failed, 0.0)

    feasible = compute_comm_encouragement_bonuses(
        np.array([1, 1]), targets, targets,
        delivery_rate=1.0, bonus_weight=0.05, floor_ratio=0.10)
    np.testing.assert_allclose(feasible, 0.005)


def test_soft_rate_bonus_increases_to_target_then_saturates():
    targets = np.array([0.80, 0.70, 0.60])
    bonuses = compute_comm_rate_exploration_bonuses(
        rate_indices=np.arange(5),
        rate_bits_per_dim=[0, 4, 8, 16, 32],
        qos_values=np.array([0.70, 0.55, 0.30]),
        qos_targets=targets,
        delivery_rate=1.0,
        bonus_weight=0.03,
        target_bits=16,
    )
    assert bonuses[0] == 0.0
    assert 0.0 < bonuses[1] < bonuses[2] < bonuses[3]
    assert bonuses[3] == bonuses[4]

    feasible = compute_comm_rate_exploration_bonuses(
        np.array([3]), [0, 4, 8, 16, 32], targets, targets,
        delivery_rate=1.0, bonus_weight=0.03, target_bits=16)
    np.testing.assert_allclose(feasible, 0.0)


def test_long_silence_penalty_has_grace_ramp_cap_and_active_reset():
    streaks, penalties = update_long_silence_penalties(
        rate_indices=np.array([0, 0, 1, 0]),
        previous_streaks=np.array([2, 3, 8, 20]),
        grace_decisions=3,
        penalty_per_decision=0.01,
        max_penalty=0.05,
    )
    np.testing.assert_array_equal(streaks, np.array([3, 4, 0, 21]))
    np.testing.assert_allclose(penalties, np.array([0.0, 0.01, 0.0, 0.05]))


def test_qos_checkpoint_selection_prioritizes_feasibility_then_worst():
    targets = np.array([0.80, 0.70, 0.60])
    # Higher steady and much lower traffic cannot compensate for lower worst.
    low_worst = build_comm_qos_checkpoint_key(
        np.array([0.79, 0.69, 0.30]), targets, bits_per_frame=1.0)
    high_worst = build_comm_qos_checkpoint_key(
        np.array([0.70, 0.60, 0.40]), targets, bits_per_frame=500.0)
    assert high_worst > low_worst

    # Once feasible, worst remains ahead of bandwidth in the ordering.
    feasible_cheap = build_comm_qos_checkpoint_key(
        np.array([0.81, 0.71, 0.61]), targets, bits_per_frame=1.0)
    feasible_fair = build_comm_qos_checkpoint_key(
        np.array([0.81, 0.71, 0.65]), targets, bits_per_frame=500.0)
    assert feasible_fair > feasible_cheap
    assert feasible_cheap > high_worst


def test_default_comm_objective_is_soft_encouragement_plus_cost():
    cfg = load_config('config/default.yaml')
    assert cfg.marl.comm_encouragement_enabled is True
    assert cfg.marl.comm_encouragement_weight > 0.0
    assert cfg.marl.comm_rate_bonus_enabled is True
    assert cfg.marl.comm_rate_bonus_target_bits == 8
    assert cfg.marl.comm_rate_bonus_aux_coef > 0.0
    assert cfg.marl.comm_rate_bonus_aux_lr > 0.0
    assert cfg.marl.comm_silence_penalty_enabled is True
    assert cfg.marl.comm_silence_grace_decisions == 3
    assert cfg.marl.comm_silence_penalty_per_decision > 0.0
    assert cfg.marl.comm_bit_cost_weight > 0.0
    assert cfg.marl.comm_energy_cost_weight > 0.0
    assert cfg.marl.comm_delay_cost_weight > 0.0
    assert cfg.marl.comm_qos_min_rate_bits == 0
    assert cfg.marl.comm_qos_rate_shortfall_penalty == 0.0
    assert cfg.marl.comm_qos_rate_aux_coef == 0.0


def test_silence_has_no_transport_cost():
    model = _transport()
    positions = np.array([[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]])
    deliveries, stats = model.transmit(
        {0: np.ones(16)}, {0: 0}, positions)

    assert deliveries == []
    assert stats.total_bits == 0.0
    assert stats.total_energy_j == 0.0
    assert stats.active_senders == 0


def test_active_message_is_quantized_delivered_and_charged():
    model = _transport()
    positions = np.array([[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]])
    message = np.linspace(-0.97, 0.91, 16)
    deliveries, stats = model.transmit({0: message}, {0: 1}, positions)

    assert stats.total_bits == 64 + 16 * 4
    assert stats.total_energy_j > 0.0
    assert stats.attempted_links == 1
    assert stats.delivered_links == 1
    assert stats.mean_latency_s > 0.0
    assert len(deliveries) == 1
    assert deliveries[0].sender == 0 and deliveries[0].receiver == 1
    assert not np.array_equal(deliveries[0].message, message)


def test_missed_deadline_still_consumes_bits_and_energy():
    model = _transport(deadline_s=1.0e-6)
    positions = np.array([[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]])
    deliveries, stats = model.transmit(
        {0: np.zeros(16)}, {0: 2}, positions)

    assert deliveries == []
    assert stats.total_bits == 64 + 16 * 8
    assert stats.total_energy_j > 0.0
    assert stats.total_energy_j <= model.tx_power_w * model.deadline_s
    assert stats.expired_links == 1
    assert stats.deadline_violation_rate == 1.0
    np.testing.assert_allclose(
        stats.sender_delivery_rates(num_senders=2), [0.0, 1.0])


def test_sender_delivery_penalty_is_attributable_and_silence_neutral():
    penalties = compute_sender_delivery_penalties(
        rate_indices=np.array([1, 2, 0, 1]),
        sender_delivery_rates=np.array([1.0, 1.0 / 3.0, 0.0, 0.5]),
        penalty_weight=0.06,
    )
    np.testing.assert_allclose(penalties, [0.0, 0.04, 0.0, 0.03])


def test_sparse_target_packet_charges_only_transmitted_token_dimensions():
    model = InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4, 8], header_bits=64,
        bandwidth_hz=1.0e5, deadline_s=0.05,
        processing_delay_s=2.0e-4, snr_threshold_db=0.0,
        antenna_gain_dbi=0.0, carrier_hz=28.0e9, tx_power_w=0.25,
        kT=4.0e-21, noise_figure_db=4.0, dt=0.1,
        message_dim=64)
    positions = np.array([[0.0, 0.0, 20.0], [100.0, 0.0, 20.0]])
    message = np.linspace(-0.9, 0.9, 64)
    mask = np.array([1.0, 0.0, 1.0, 1.0])

    deliveries, stats = model.transmit(
        {0: message}, {0: 2}, positions, token_masks={0: mask})

    assert stats.total_bits == 64 + 3 * 16 * 8
    assert len(deliveries) == 1
    np.testing.assert_allclose(deliveries[0].token_mask, mask)
    delivered_tokens = deliveries[0].message.reshape(4, 16)
    np.testing.assert_allclose(delivered_tokens[1], 0.0)
    assert np.any(delivered_tokens[0] != 0.0)


def test_qos_metrics_match_publication_threshold_definitions():
    metrics = compute_comm_qos_metrics(np.array([0.9, 0.8, 0.7, 0.5]))
    np.testing.assert_allclose(metrics, [0.725, 2.0 / 3.0, 0.5])


def test_target_allocation_regularizer_prefers_balanced_committed_teams():
    balanced = torch.eye(4)
    collapsed = torch.zeros(4, 4)
    collapsed[:, 0] = 1.0
    uniform = torch.full((4, 4), 0.25)

    balanced_loss, balanced_entropy = compute_target_allocation_regularizer(
        balanced, num_agents=4)
    collapsed_loss, _ = compute_target_allocation_regularizer(
        collapsed, num_agents=4)
    _, uniform_entropy = compute_target_allocation_regularizer(
        uniform, num_agents=4)

    torch.testing.assert_close(balanced_loss, torch.tensor(0.0))
    assert collapsed_loss > balanced_loss
    assert uniform_entropy > balanced_entropy


def test_straight_through_balance_escapes_uniform_softmax_stationary_point():
    logits = torch.zeros(4, 4, requires_grad=True)
    soft = torch.softmax(logits, dim=-1)
    hard = torch.nn.functional.one_hot(
        soft.argmax(dim=-1), num_classes=4).to(soft.dtype)
    straight_through = hard + soft - soft.detach()
    balance_loss, _ = compute_target_allocation_regularizer(
        straight_through, num_agents=4)
    balance_loss.backward()

    assert balance_loss.item() > 0.1
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0.0


def test_sparse_endpoint_loss_accepts_two_endpoints_and_rejects_collapse():
    ideal = torch.tensor([
        [0.55, 0.40, 0.03, 0.02],
        [0.52, 0.02, 0.44, 0.02],
        [0.02, 0.51, 0.03, 0.44],
        [0.02, 0.03, 0.50, 0.45],
    ])
    under, over = compute_sparse_endpoint_load_loss(
        ideal, num_agents=4, commitment_topk=2, desired_endpoints=2)
    torch.testing.assert_close(under, torch.tensor(0.0))
    torch.testing.assert_close(over, torch.tensor(0.0))

    logits = torch.tensor(
        [[2.0, 1.0, -1.0, -2.0]] * 4, requires_grad=True)
    collapsed = torch.softmax(logits, dim=-1)
    under, over = compute_sparse_endpoint_load_loss(
        collapsed, num_agents=4, commitment_topk=2,
        desired_endpoints=2)
    loss = under + over
    assert under.item() > 0.0 and over.item() > 0.0
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0.0


def test_macro_action_transmits_one_packet_not_one_per_micro_frame():
    cfg = get_default_config().to_small_config()
    cfg.scenario.K, cfg.scenario.Q, cfg.scenario.T = 2, 1, 6
    cfg.target.omega_q = [1.0]
    cfg.marl.tracking_enabled = False
    cfg.marl.ground_communication_enabled = False
    cfg.marl.learned_comm_mode = 'cost_aware'
    cfg.marl.comm_only_neighbor_information = True
    cfg.marl.comm_cross_attention_enabled = True
    cfg.marl.actor_decision_interval = 3
    cfg.marl.num_envs = 1
    cfg.marl.rollout_steps = 3
    cfg.marl.ppo_epochs = 1
    cfg.marl.minibatch_size = 2

    env = UAVISACEnv(config=cfg, seed=31)
    action_space = ActionSpace(
        v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
        learn_roles=cfg.marl.learn_roles)
    action_space.num_targets = 1
    action_space.structured_actor = True
    action_space.structured_entity_dim = 32
    obs_dim = env.core.obs_builder.get_obs_dim()
    global_dim = env.core.obs_builder.get_global_state_dim()
    agents = [MAPPOAgent(
        k, obs_dim, global_dim, action_space, 2, num_targets=1,
        hidden_layers=[32, 32], device='cpu',
        comm_num_rate_levels=len(cfg.marl.comm_rate_bits_per_dim),
        use_comm_cross_attention=True,
    ) for k in range(2)]
    with torch.no_grad():
        agents[0].actor.comm_rate_head.weight.zero_()
        agents[0].actor.comm_rate_head.bias.copy_(
            torch.tensor([-10.0, 10.0, -10.0, -10.0]))

    trainer = MAPPTrainer(env, agents, cfg, device='cpu')
    calls = 0
    original_submit = env.core.submit_learned_communications

    def counted_submit(messages, rates):
        nonlocal calls
        calls += 1
        return original_submit(messages, rates)

    env.core.submit_learned_communications = counted_submit
    trainer.collect_rollout()

    assert calls == 1
    assert trainer._rollout_learned_comm_bits == [2 * (64 + 16 * 4)]
    env.close()


def test_tracking_free_u2u_environment_closes_free_neighbor_channel():
    cfg = get_default_config().to_small_config()
    cfg.scenario.K = 2
    cfg.scenario.Q = 1
    cfg.scenario.T = 3
    cfg.target.omega_q = [1.0]
    cfg.marl.tracking_enabled = False
    cfg.marl.ground_communication_enabled = False
    cfg.marl.learned_comm_mode = 'cost_aware'
    cfg.marl.comm_only_neighbor_information = True
    cfg.marl.comm_cross_attention_enabled = True

    env = UAVISACEnv(config=cfg, seed=7)
    obs, _ = env.reset(seed=7)
    initial_target_state = env.core.targets[0].state.copy()

    slices = ObservationSlices.from_config(
        K=2, Q=1, use_p0=False, use_rel_features=cfg.marl.rel_features,
        use_comm_tokens=True)
    assert obs['0'].shape == (slices.total_dim,)
    np.testing.assert_allclose(slices.extract_neighbors(obs['0']), 0.0)

    message = np.linspace(-0.9, 0.9, 16)
    env.core.submit_learned_communications({0: message}, {0: 1})
    actions = {
        '0': {'delta_p': np.zeros(2), 'role': 2},
        '1': {'delta_p': np.zeros(2), 'role': 2},
    }
    next_obs, _, _, _, info = env.step(actions)

    np.testing.assert_allclose(env.core.targets[0].state, initial_target_state)
    assert info['total_bits'] == 0.0  # no receiver-to-ground reporting
    assert info['learned_comm_bits'] == 64 + 16 * 4
    assert info['learned_comm_energy_j'] > 0.0
    assert info['learned_comm_delivery_rate'] == 1.0
    np.testing.assert_allclose(slices.extract_comm(next_obs['1']), 0.0)
    np.testing.assert_allclose(slices.extract_comm(next_obs['0']), 0.0)
    receiver_tokens = slices.extract_comm_tokens(next_obs['1'])
    receiver_mask = slices.extract_comm_mask(next_obs['1'])
    np.testing.assert_allclose(
        receiver_tokens[0, :16],
        env.core._inter_uav_comm.quantize(message, 1),
    )
    np.testing.assert_allclose(receiver_mask, [1.0])
    np.testing.assert_allclose(slices.extract_comm_mask(next_obs['0']), [0.0])

    # No new transmission: the last delivered token remains available with AoI.
    aged_obs, _, _, _, aged_info = env.step(actions)
    assert aged_info['learned_comm_bits'] == 0.0
    np.testing.assert_allclose(slices.extract_comm_mask(aged_obs['1']), [1.0])
    assert slices.extract_comm_tokens(aged_obs['1'])[0, -1] > 0.0


def test_episode_channel_profiles_are_seeded_and_update_both_streams():
    cfg = get_default_config().to_small_config()
    cfg.scenario.K = 2
    cfg.scenario.Q = 1
    cfg.target.omega_q = [1.0]
    cfg.marl.learned_comm_mode = 'cost_aware'
    cfg.marl.comm_channel_randomization_enabled = True
    cfg.marl.comm_channel_randomization_snr_threshold_db_values = [
        0.0, 30.0, 35.0]
    cfg.marl.comm_channel_randomization_deadline_s_values = [
        0.005, 0.005, 0.0008]
    cfg.marl.comm_channel_randomization_profile_weights = [0.2, 0.3, 0.5]

    env = UAVISACEnv(config=cfg, seed=91)
    _, first = env.reset(seed=91)
    _, replay = env.reset(seed=91)
    assert first['comm_channel_profile'] == replay['comm_channel_profile']
    assert (first['comm_channel_snr_threshold_db']
            == replay['comm_channel_snr_threshold_db'])
    assert (first['comm_channel_deadline_s']
            == replay['comm_channel_deadline_s'])

    idx = int(first['comm_channel_profile'])
    expected_snr = (
        cfg.marl.comm_channel_randomization_snr_threshold_db_values[idx])
    expected_deadline = (
        cfg.marl.comm_channel_randomization_deadline_s_values[idx])
    assert env.core._inter_uav_comm.snr_threshold_db == expected_snr
    assert env.core._inter_uav_comm.deadline_s == expected_deadline
    assert env.core.obs_builder.comm_deadline_s == expected_deadline
    randomized_uav_positions = np.asarray(
        [u.pos.copy() for u in env.core.uavs])
    randomized_target_states = np.asarray(
        [t.state.copy() for t in env.core.targets])

    actions = {
        '0': {'delta_p': np.zeros(2), 'role': 2},
        '1': {'delta_p': np.zeros(2), 'role': 2},
    }
    _, _, _, _, info = env.step(actions)
    assert info['learned_comm_channel_profile'] == idx
    assert info['learned_comm_channel_snr_threshold_db'] == expected_snr
    assert info['learned_comm_channel_deadline_s'] == expected_deadline

    fixed_cfg = get_default_config().to_small_config()
    fixed_cfg.scenario.K = 2
    fixed_cfg.scenario.Q = 1
    fixed_cfg.target.omega_q = [1.0]
    fixed_cfg.marl.learned_comm_mode = 'cost_aware'
    fixed_env = UAVISACEnv(config=fixed_cfg, seed=91)
    fixed_env.reset(seed=91)
    np.testing.assert_allclose(
        randomized_uav_positions,
        np.asarray([u.pos for u in fixed_env.core.uavs]),
    )
    np.testing.assert_allclose(
        randomized_target_states,
        np.asarray([t.state for t in fixed_env.core.targets]),
    )


def test_environment_subtracts_explicit_communication_cost():
    cfg = get_default_config().to_small_config()
    cfg.scenario.K = 2
    cfg.scenario.Q = 1
    cfg.scenario.T = 3
    cfg.target.omega_q = [1.0]
    cfg.marl.tracking_enabled = False
    cfg.marl.ground_communication_enabled = False
    cfg.marl.learned_comm_mode = 'cost_aware'
    cfg.marl.comm_only_neighbor_information = True
    env = UAVISACEnv(config=cfg, seed=11)
    env.reset(seed=11)
    state = env.core.get_state()
    actions = {
        '0': {'delta_p': np.zeros(2), 'role': 2},
        '1': {'delta_p': np.zeros(2), 'role': 2},
    }

    env.core.submit_learned_communications({0: np.zeros(16)}, {0: 0})
    _, _, _, _, silent_info = env.step(actions)

    env.core.set_state(state)
    env.core.submit_learned_communications({0: np.zeros(16)}, {0: 1})
    _, _, _, _, active_info = env.step(actions)

    expected_cost = (
        cfg.marl.comm_bit_cost_weight * active_info['learned_comm_bits']
        + cfg.marl.comm_energy_cost_weight * active_info['learned_comm_energy_j']
        + cfg.marl.comm_delay_cost_weight
        * (active_info['learned_comm_mean_latency_s']
           + active_info['learned_comm_deadline_violation_rate'])
    )
    assert silent_info['learned_comm_bits'] == 0.0
    np.testing.assert_allclose(
        active_info['team_reward'],
        silent_info['team_reward'] - expected_cost,
        rtol=1e-10,
        atol=1e-10,
    )


def test_joint_movement_and_communication_log_prob_is_reproducible():
    torch.manual_seed(4)
    k, q = 2, 1
    obs_dim = ObservationSlices.from_config(k, q, False, True).total_dim
    action_space = ActionSpace(v_max=25.0, dt=0.1, learn_roles=False)
    action_space.num_targets = q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 32
    agent = MAPPOAgent(
        agent_id=0,
        obs_dim=obs_dim,
        global_state_dim=20,
        action_space=action_space,
        num_agents=k,
        num_targets=q,
        hidden_layers=[32, 32],
        device='cpu',
    )

    obs = torch.randn(k, obs_dim)
    with torch.no_grad():
        dp_mean, dp_log_std, role_logits, comm_mean, _, _ = agent.actor(obs)
        _, initial_rate_logits = agent.actor.communication_parameters(comm_mean)
        _, deterministic_rate, _, _ = agent.sample_communication(
            comm_mean, deterministic=True)
        torch.testing.assert_close(initial_rate_logits, torch.zeros_like(initial_rate_logits))
        torch.testing.assert_close(deterministic_rate, torch.zeros_like(deterministic_rate))
        comm_action, comm_rate, comm_lp, _ = agent.sample_communication(comm_mean)

    actions_dp = np.zeros((k, 2), dtype=np.float64)
    actions_role = np.zeros(k, dtype=np.int64)
    movement_lp = np.zeros(k, dtype=np.float64)
    for idx in range(k):
        action, lp = action_space.decode(
            dp_mean[idx].numpy(), dp_log_std.numpy(), role_logits[idx].numpy())
        actions_dp[idx] = action.delta_p
        actions_role[idx] = action.role
        movement_lp[idx] = lp

    old_lp = torch.as_tensor(movement_lp, dtype=torch.float32) + comm_lp
    passed, max_diff = agent.verify_old_log_prob_consistency(
        obs,
        torch.as_tensor(actions_dp, dtype=torch.float32),
        torch.as_tensor(actions_role, dtype=torch.long),
        old_lp,
        actions_comm=comm_action,
        actions_comm_rate=comm_rate,
    )
    assert passed, max_diff


def test_old_four_rate_checkpoint_expands_to_optional_fifth_rate():
    k, q = 2, 1
    obs_dim = ObservationSlices.from_config(k, q, False, True).total_dim
    action_space = ActionSpace(v_max=25.0, dt=0.1, learn_roles=False)
    action_space.num_targets = q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 32
    old_agent = MAPPOAgent(
        0, obs_dim, 20, action_space, k, num_targets=q,
        hidden_layers=[32, 32], comm_num_rate_levels=4, device='cpu')
    new_agent = MAPPOAgent(
        0, obs_dim, 20, action_space, k, num_targets=q,
        hidden_layers=[32, 32], comm_num_rate_levels=5, device='cpu')
    with torch.no_grad():
        old_agent.actor.comm_rate_head.weight.fill_(0.25)
        old_agent.actor.comm_rate_head.bias.copy_(torch.arange(4.0))
    old_state = old_agent.actor.state_dict()

    new_agent.load_actor_state_dict_compatible(old_state)

    torch.testing.assert_close(
        new_agent.actor.comm_rate_head.weight[:4],
        old_agent.actor.comm_rate_head.weight)
    torch.testing.assert_close(
        new_agent.actor.comm_rate_head.bias[:4],
        old_agent.actor.comm_rate_head.bias)
    torch.testing.assert_close(
        new_agent.actor.comm_rate_head.weight[4],
        torch.zeros_like(new_agent.actor.comm_rate_head.weight[4]))
    torch.testing.assert_close(
        new_agent.actor.comm_rate_head.bias[4],
        torch.zeros_like(new_agent.actor.comm_rate_head.bias[4]))


def test_target_conditioned_cross_attention_uses_only_valid_sender_tokens():
    torch.manual_seed(17)
    k, q = 3, 2
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True, use_comm_tokens=True)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim,
        K=k,
        Q=q,
        entity_dim=64,
        use_comm_cross_attention=True,
        comm_token_dim=21,
    )
    actor.eval()

    base = torch.randn(1, slices.total_dim)
    base[:, slices.comm_mask_start:slices.comm_mask_start + slices.comm_mask_len] = 0.0
    altered_masked = base.clone()
    token_len = (k - 1) * slices.comm_token_per_sender
    altered_masked[:, slices.comm_token_start:slices.comm_token_start + token_len] += 50.0

    out_masked = actor(base)[0]
    out_altered_masked = actor(altered_masked)[0]
    torch.testing.assert_close(out_masked, out_altered_masked)

    one_valid = base.clone()
    one_valid[:, slices.comm_token_start:slices.comm_token_start + 16] = 1.0
    one_valid[:, slices.comm_mask_start] = 1.0
    out_valid = actor(one_valid)[0]
    assert not torch.allclose(out_masked, out_valid, atol=1e-7, rtol=1e-7)

    weights = actor.last_comm_attention
    assert weights is not None
    assert weights.shape[-2:] == (q, k - 1)
    torch.testing.assert_close(weights[..., 1], torch.zeros_like(weights[..., 1]))

    actor.zero_grad(set_to_none=True)
    actor(one_valid)[0].sum().backward()
    assert actor.comm_cross_attn.in_proj_weight.grad is not None
    assert actor.comm_cross_attn.in_proj_weight.grad.abs().sum() > 0.0


def test_sparse_claim_suppresses_full_target_and_omits_low_token():
    torch.manual_seed(19)
    k, q, content_dim = 4, 4, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_corrected_parser=True, use_comm_cross_attention=True,
        comm_token_dim=token_dim, comm_tokens_per_sender=q,
        comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True,
        sparse_claim_enabled=True,
        sparse_claim_share_topk=3,
        sparse_claim_desired_endpoints=2,
        sparse_claim_full_penalty=4.0,
        sparse_claim_vacant_bonus=1.0,
        sparse_claim_temperature=0.1)
    torch.nn.init.zeros_(actor.target_assignment_head[-1].weight)
    torch.nn.init.zeros_(actor.target_assignment_head[-1].bias)

    obs = torch.zeros(1, slices.total_dim)
    # Two peers claim target 0; all remaining targets are vacant.
    obs[:, slices.comm_mask_start] = 1.0
    obs[:, slices.comm_mask_start + q] = 1.0
    comm_mean = actor(obs)[3]
    assignment = actor.last_target_assignment[0]
    outgoing_mask = actor.last_outgoing_token_mask[0]

    assert assignment[1] > assignment[0]
    torch.testing.assert_close(outgoing_mask.sum(), torch.tensor(3.0))
    assert outgoing_mask[0].item() == 0.0
    torch.testing.assert_close(
        comm_mean.reshape(q, content_dim)[0],
        torch.zeros(content_dim))


def test_endpoint_deficit_semantic_field_attracts_missing_and_ignores_full_target():
    torch.manual_seed(21)
    k, q, content_dim = 3, 2, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_corrected_parser=True, use_comm_cross_attention=True,
        comm_token_dim=token_dim, comm_tokens_per_sender=q,
        comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True,
        sparse_claim_enabled=True,
        sparse_claim_share_topk=1,
        sparse_claim_desired_endpoints=2,
        sparse_claim_full_penalty=0.0,
        sparse_claim_vacant_bonus=0.0,
        semantic_kinematic_field_enabled=True,
        semantic_kinematic_field_gain=0.5)
    torch.nn.init.zeros_(actor.target_assignment_head[-1].weight)
    torch.nn.init.zeros_(actor.target_assignment_head[-1].bias)
    torch.nn.init.zeros_(actor.dp_head.weight)
    torch.nn.init.zeros_(actor.dp_head.bias)

    obs = torch.zeros(1, slices.total_dim)
    geometry = obs[:, slices.geom_start:
                   slices.geom_start + q * slices.geom_per_target]
    geometry = geometry.reshape(1, q, slices.geom_per_target)
    geometry[:, 0, :2] = torch.tensor([1.0, 0.0])  # full target: east
    geometry[:, 1, :2] = torch.tensor([0.0, 1.0])  # vacant target: north
    # Both peers claim target 0, while target 1 has an endpoint deficit.
    obs[:, slices.comm_mask_start] = 1.0
    obs[:, slices.comm_mask_start + q] = 1.0

    movement = torch.tanh(actor(obs)[0])[0]
    weights = actor.last_semantic_field_weights[0]
    torch.testing.assert_close(weights[0], torch.tensor(0.0))
    assert weights[1] > 0.0
    torch.testing.assert_close(movement[0], torch.tensor(0.0))
    assert movement[1] > 0.0  # attracted to the vacant northern target

    actor.zero_grad(set_to_none=True)
    movement_logits = actor(obs)[0]
    movement_logits[:, 1].sum().backward()
    grad = actor.target_assignment_head[-1].weight.grad
    assert grad is not None and torch.any(grad != 0)

    silent = obs.detach().clone()
    silent[:, slices.comm_token_start:
           slices.comm_mask_start + slices.comm_mask_len] = 0.0
    silent_movement = torch.tanh(actor(silent)[0])
    torch.testing.assert_close(silent_movement, torch.zeros_like(silent_movement))
    torch.testing.assert_close(
        actor.last_semantic_field_delta,
        torch.zeros_like(actor.last_semantic_field_delta))


def test_endpoint_deficit_field_preserves_cancellation_confidence():
    torch.manual_seed(22)
    k, q, content_dim = 3, 2, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_corrected_parser=True, use_comm_cross_attention=True,
        comm_token_dim=token_dim, comm_tokens_per_sender=q,
        comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True,
        sparse_claim_enabled=True,
        sparse_claim_share_topk=1,
        sparse_claim_desired_endpoints=2,
        sparse_claim_full_penalty=0.01,
        sparse_claim_vacant_bonus=0.0,
        sparse_claim_temperature=0.1,
        semantic_kinematic_field_enabled=True,
        semantic_kinematic_field_gain=0.5)
    torch.nn.init.zeros_(actor.target_assignment_head[-1].weight)
    torch.nn.init.zeros_(actor.target_assignment_head[-1].bias)
    torch.nn.init.zeros_(actor.dp_head.weight)
    torch.nn.init.zeros_(actor.dp_head.bias)

    obs = torch.zeros(1, slices.total_dim)
    geometry = obs[:, slices.geom_start:
                   slices.geom_start + q * slices.geom_per_target]
    geometry = geometry.reshape(1, q, slices.geom_per_target)
    # Only the vacant target contributes. Its soft responsibility is below one,
    # so the intervention must remain below the full configured gain.
    geometry[:, :, 0] = 1.0
    obs[:, slices.comm_mask_start] = 1.0
    obs[:, slices.comm_mask_start + q] = 1.0

    actor(obs)
    semantic_mass = actor.last_semantic_field_weights.abs().sum().item()
    assert 0.0 < semantic_mass < 1.0
    assert (actor.last_semantic_field_delta.norm().item()
            <= 0.5 * semantic_mass + 1e-6)


def test_target_conditioned_movement_is_neutral_then_uses_peer_claims():
    torch.manual_seed(24)
    k, q, content_dim = 3, 2, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_corrected_parser=True, use_comm_cross_attention=True,
        comm_token_dim=token_dim, comm_tokens_per_sender=q,
        comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True,
        sparse_claim_enabled=True,
        sparse_claim_share_topk=1,
        sparse_claim_desired_endpoints=2,
        sparse_claim_full_penalty=4.0,
        sparse_claim_vacant_bonus=1.0,
        sparse_claim_temperature=0.1,
        target_conditioned_movement_enabled=True,
        target_conditioned_movement_gain=0.5)
    torch.nn.init.zeros_(actor.target_assignment_head[-1].weight)
    torch.nn.init.zeros_(actor.target_assignment_head[-1].bias)
    torch.nn.init.zeros_(actor.dp_head.weight)
    torch.nn.init.zeros_(actor.dp_head.bias)

    obs = torch.zeros(1, slices.total_dim)
    geometry = obs[:, slices.geom_start:
                   slices.geom_start + q * slices.geom_per_target]
    geometry = geometry.reshape(1, q, slices.geom_per_target)
    geometry[:, 0, :2] = torch.tensor([1.0, 0.0])  # target 0: east
    geometry[:, 1, :2] = torch.tensor([0.0, 1.0])  # target 1: north

    # Zero-init makes a newly introduced head exactly neutral for an existing
    # checkpoint before it receives any PPO update.
    neutral_movement = torch.tanh(actor(obs)[0])
    assert torch.equal(neutral_movement, torch.zeros(1, 2))
    torch.testing.assert_close(
        actor.last_target_movement_delta,
        torch.zeros_like(actor.last_target_movement_delta))
    actor.zero_grad(set_to_none=True)
    actor(obs)[0].sum().backward()
    neutral_grad = actor.target_movement_head[-1].bias.grad
    assert neutral_grad is not None and torch.any(neutral_grad != 0)

    # Make every target candidate radial for a deterministic causal test.
    with torch.no_grad():
        actor.target_movement_head[-1].weight.zero_()
        actor.target_movement_head[-1].bias.copy_(torch.tensor([2.0, 0.0]))
    silent_movement = torch.tanh(actor(obs)[0])[0]

    claimed = obs.clone()
    # Both peers claim target 0. The receiver should redirect its learned
    # target-conditioned movement toward the vacant northern target.
    claimed[:, slices.comm_mask_start] = 1.0
    claimed[:, slices.comm_mask_start + q] = 1.0
    claimed_movement = torch.tanh(actor(claimed)[0])[0]
    assert claimed_movement[1] > silent_movement[1]
    assert claimed_movement[0] < silent_movement[0]

    actor.zero_grad(set_to_none=True)
    movement_logits = actor(claimed)[0]
    (movement_logits[:, 1] - movement_logits[:, 0]).sum().backward()
    assignment_grad = actor.target_assignment_head[-1].weight.grad
    movement_grad = actor.target_movement_head[-1].bias.grad
    assert assignment_grad is not None and torch.any(assignment_grad != 0)
    assert movement_grad is not None and torch.any(movement_grad != 0)


def test_message_aware_target_allocation_controls_actor_residual():
    torch.manual_seed(23)
    k, q = 4, 4
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=22,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_corrected_parser=True,
        use_comm_cross_attention=True,
        use_target_allocation=True,
    )
    obs = torch.randn(2, slices.total_dim)
    obs[:, slices.comm_mask_start:slices.comm_mask_start + k - 1] = 1.0

    movement = actor(obs)[0]
    assignment = actor.last_target_assignment

    assert assignment.shape == (2, q)
    torch.testing.assert_close(
        assignment.sum(dim=-1), torch.ones(2), rtol=1e-5, atol=1e-6)
    actor.zero_grad(set_to_none=True)
    (movement.sum() + assignment[:, 0].sum()).backward()
    assert actor.target_assignment_head[-1].weight.grad is not None
    assert actor.target_assignment_head[-1].weight.grad.abs().sum() > 0.0
    assert actor.allocation_proj.weight.grad is not None
    assert actor.allocation_proj.weight.grad.abs().sum() > 0.0


def test_negotiated_allocation_directly_controls_sensing_resource_logits():
    torch.manual_seed(25)
    k, q = 4, 4
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_corrected_parser=True,
        use_comm_cross_attention=True,
        use_target_allocation=True,
        target_allocation_resource_blend=2.0,
    )
    obs = torch.randn(2, slices.total_dim)
    obs[:, slices.comm_mask_start:slices.comm_mask_start + k - 1] = 1.0

    comm_mean = actor(obs)[3]
    _, _, sensing_mean, _ = actor.isac_resource_parameters(comm_mean)
    base_mean = actor.isac_sensing_mean_head(comm_mean)
    assert not torch.allclose(sensing_mean, base_mean)

    actor.zero_grad(set_to_none=True)
    sensing_mean[:, 0].mean().backward()
    grad = actor.target_assignment_head[-1].weight.grad
    assert grad is not None and grad.abs().sum() > 0.0


def test_received_tokens_directly_control_executed_sensing_logits():
    torch.manual_seed(26)
    k, q, content_dim = 3, 2, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_corrected_parser=True, use_comm_cross_attention=True,
        comm_token_dim=token_dim, comm_tokens_per_sender=q,
        comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True,
        comm_aided_sensing_enabled=True,
        comm_aided_sensing_blend=1.0)
    with torch.no_grad():
        actor.comm_sensing_gate.bias.fill_(10.0)
        actor.comm_sensing_head[-1].weight.fill_(0.1)

    obs = torch.randn(2, slices.total_dim)
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + slices.comm_mask_len] = 1.0
    comm_mean = actor(obs)[3]
    _, _, sensing_mean, _ = actor.isac_resource_parameters(comm_mean)
    residual = actor.last_comm_sensing_logits
    assert residual is not None and torch.any(residual.abs() > 1e-6)

    actor.zero_grad(set_to_none=True)
    sensing_mean[:, 0].mean().backward()
    grad = actor.comm_sensing_head[-1].weight.grad
    assert grad is not None and torch.any(grad != 0)

    no_comm = obs.detach().clone()
    no_comm[:, slices.comm_token_start:
            slices.comm_mask_start + slices.comm_mask_len] = 0.0
    actor(no_comm)
    torch.testing.assert_close(
        actor.last_comm_sensing_logits,
        torch.zeros_like(actor.last_comm_sensing_logits))


def test_round_negotiation_is_phase_and_inbox_causal():
    torch.manual_seed(27)
    k, q, token_dim = 3, 2, 22
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_corrected_parser=True, use_comm_cross_attention=True,
        comm_token_dim=token_dim, comm_tokens_per_sender=q,
        comm_payload_dim=q * 16, comm_target_token_enabled=True,
        comm_target_token_dim=16, use_target_allocation=True,
        round_negotiation_enabled=True,
        target_allocation_movement_blend=0.15)
    obs_a = torch.randn(k, slices.total_dim)
    obs_a[:, slices.comm_mask_start:
          slices.comm_mask_start + (k - 1) * q] = 1.0
    obs_b = obs_a.clone()
    token_end = slices.comm_token_start + (k - 1) * q * token_dim
    obs_b[:, slices.comm_token_start:token_end] *= -1.0

    phase0 = torch.zeros(k)
    phase1 = torch.ones(k)
    proposal = actor(obs_a, comm_round_phase=phase0)[3]
    identified_proposal = actor(
        obs_a, comm_round_phase=phase0,
        agent_identity=torch.arange(k))[3]
    response_a = actor(obs_a, comm_round_phase=phase1)[3]
    assignment_a = actor.last_target_assignment.detach().clone()
    response_b = actor(obs_b, comm_round_phase=phase1)[3]
    assignment_b = actor.last_target_assignment.detach().clone()

    assert proposal.shape == (k, q * 16)
    assert not torch.allclose(
        proposal, identified_proposal, atol=1e-8, rtol=1e-8)
    assert not torch.allclose(proposal, response_a, atol=1e-8, rtol=1e-8)
    assert not torch.allclose(response_a, response_b, atol=1e-7, rtol=1e-7)
    assert not torch.allclose(assignment_a, assignment_b, atol=1e-7, rtol=1e-7)

    actor.zero_grad(set_to_none=True)
    actor(obs_a, comm_round_phase=phase1)
    (-torch.log(actor.last_target_assignment[:, 0].clamp_min(1e-8)).mean()
     ).backward()
    claim_grad = actor.round_peer_claim_head[-1].weight.grad
    phase_grad = actor.round_phase_enc[0].weight.grad
    assert claim_grad is not None and torch.any(claim_grad != 0)
    assert phase_grad is not None and torch.any(phase_grad != 0)


def test_straight_through_commitment_directly_guides_movement():
    torch.manual_seed(29)
    k, q = 4, 4
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True,
        use_target_allocation=True,
        target_allocation_temperature=0.15,
        target_allocation_straight_through=True,
        target_allocation_movement_blend=1.0,
    )
    obs = torch.zeros(k, slices.total_dim)
    directions = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
    ])
    geometry = obs[:, slices.geom_start:
                   slices.geom_start + q * slices.geom_per_target]
    geometry = geometry.reshape(k, q, slices.geom_per_target)
    geometry[:, :, :2] = directions

    dp_mean = actor(obs)[0]
    choice = actor.last_target_assignment.argmax(dim=-1)
    expected = directions[choice]
    torch.testing.assert_close(
        torch.tanh(dp_mean), expected, rtol=2e-3, atol=2e-3)


def test_uncertain_commitment_has_zero_confidence_gated_authority():
    torch.manual_seed(31)
    k, q = 4, 4
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True,
        use_target_allocation=True,
        target_allocation_straight_through=True,
        target_allocation_movement_blend=1.0,
        target_allocation_movement_confidence_gating_enabled=True,
        target_allocation_movement_confidence_floor=0.0,
        target_allocation_movement_confidence_power=2.0,
    )
    for parameter in actor.target_assignment_head.parameters():
        parameter.data.zero_()
    actor(torch.zeros(k, slices.total_dim))
    torch.testing.assert_close(
        actor.last_movement_confidence, torch.zeros(k))
    torch.testing.assert_close(
        actor.last_effective_movement_blend, torch.zeros(k, 1))


def test_zero_initialized_movement_adapter_preserves_policy_and_learns():
    torch.manual_seed(37)
    k, q = 4, 4
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=22,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True,
        comm_target_token_enabled=True,
        comm_tokens_per_sender=q,
        comm_token_dim=22,
        use_target_allocation=True,
        capacity_matching_enabled=True,
        hierarchical_dual_assignment_enabled=True,
    )
    with torch.no_grad():
        actor.movement_feature_adapter.weight.zero_()
        actor.movement_feature_adapter.bias.zero_()
    obs = torch.randn(k, slices.total_dim)
    actor(obs, agent_identity=torch.arange(k))
    before = actor.last_movement_assignment.detach().clone()

    # The zero residual is exactly behaviour-preserving, yet the single
    # linear adapter receives a non-zero gradient through the trained head.
    loss = -torch.log(
        actor.last_movement_assignment[:, 0].clamp_min(1e-8)).mean()
    loss.backward()
    grad = actor.movement_feature_adapter.weight.grad
    assert grad is not None and torch.any(grad != 0)
    with torch.no_grad():
        actor.movement_feature_adapter.weight.zero_()
        actor.movement_feature_adapter.bias.zero_()
        actor(obs, agent_identity=torch.arange(k))
    torch.testing.assert_close(actor.last_movement_assignment, before)


def test_temporal_commitment_loss_respects_interleaved_environment_lag():
    # Six stored teams, two interleaved environments and two UAVs per team.
    base = torch.tensor([
        [0.95, 0.05], [0.05, 0.95],
    ])
    stable = base.repeat(6, 1)
    changed = stable.clone().reshape(6, 2, 2)
    changed[2:, :, :] = changed[2:, :, :].flip(-1)
    changed = changed.reshape(-1, 2)
    masks = torch.ones(12)

    stable_loss = compute_target_allocation_temporal_loss(
        stable, num_agents=2, delay_teams=2, transition_masks=masks)
    changed_loss = compute_target_allocation_temporal_loss(
        changed, num_agents=2, delay_teams=2, transition_masks=masks)
    assert stable_loss.item() == 0.0
    assert changed_loss.item() > 0.1


def test_balanced_assignment_teacher_produces_unique_targets_and_bounded_motion():
    k, q = 4, 4
    state = np.zeros(8 * k + 6 * q, dtype=np.float32)
    uav_xy = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
    ], dtype=np.float32)
    # Target indices are deliberately permuted relative to the UAV order.
    target_xy = np.array([
        [0.05, 0.95], [0.05, 0.05], [0.95, 0.95], [0.95, 0.05],
    ], dtype=np.float32)
    for agent_id in range(k):
        state[8 * agent_id:8 * agent_id + 2] = uav_xy[agent_id]
    target_offset = 8 * k
    for target_id in range(q):
        start = target_offset + 6 * target_id
        state[start:start + 2] = target_xy[target_id]

    labels, movement = compute_balanced_assignment_teacher(
        torch.as_tensor(state[None, :]),
        num_agents=k,
        num_targets=q,
        region_size=(800.0, 800.0),
        max_dp=2.5,
    )

    torch.testing.assert_close(labels, torch.tensor([1, 3, 0, 2]))
    assert len(torch.unique(labels)) == q
    torch.testing.assert_close(
        torch.linalg.vector_norm(movement, dim=-1),
        torch.full((k,), 2.5),
        rtol=1e-5,
        atol=1e-5,
    )


def test_balanced_assignment_teacher_hysteresis_prevents_label_chatter():
    k = q = 2
    states = np.zeros((2, 8 * k + 6 * q), dtype=np.float32)
    targets = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    uavs = [
        np.array([[0.49, 0.0], [0.51, 0.0]], dtype=np.float32),
        np.array([[0.51, 0.0], [0.49, 0.0]], dtype=np.float32),
    ]
    for step in range(2):
        for agent_id in range(k):
            states[step, 8 * agent_id:8 * agent_id + 2] = uavs[step][agent_id]
        for target_id in range(q):
            start = 8 * k + 6 * target_id
            states[step, start:start + 2] = targets[target_id]

    no_hysteresis, _ = compute_balanced_assignment_teacher(
        torch.as_tensor(states), k, q, (800.0, 800.0), 2.5)
    sticky, _ = compute_balanced_assignment_teacher(
        torch.as_tensor(states), k, q, (800.0, 800.0), 2.5,
        num_envs=1, transition_masks=torch.ones(2 * k),
        switching_penalty_m=75.0)

    torch.testing.assert_close(
        no_hysteresis.reshape(2, k),
        torch.tensor([[0, 1], [1, 0]]))
    torch.testing.assert_close(
        sticky.reshape(2, k),
        torch.tensor([[0, 1], [0, 1]]))


def test_qos_bistatic_teacher_redirects_best_geometry_to_weak_target():
    k = q = 4
    uavs = np.array([
        [0.49372072, 0.65902042],
        [0.10472244, 0.55003651],
        [0.29430644, 0.84168606],
        [0.10779299, 0.66126338],
    ], dtype=np.float32)
    targets = np.array([
        [0.83307965, 0.25458667],
        [0.85590342, 0.83497592],
        [0.06666550, 0.68674601],
        [0.05107972, 0.50302757],
    ], dtype=np.float32)

    def make_state(pd):
        state = np.zeros(8 * k + 6 * q + 1 + 2 * q, dtype=np.float32)
        for agent_id in range(k):
            state[8 * agent_id:8 * agent_id + 2] = uavs[agent_id]
        target_offset = 8 * k
        for target_id in range(q):
            start = target_offset + 6 * target_id
            state[start:start + 2] = targets[target_id]
        pd_offset = target_offset + 6 * q + 1 + q
        state[pd_offset:pd_offset + q] = pd
        return torch.as_tensor(state[None, :])

    neutral, _ = compute_qos_bistatic_assignment_teacher(
        make_state(np.zeros(q)), k, q, (800.0, 800.0), 2.5,
        commitment_frames=20, qos_weight=8.0)
    weak_target, movement = compute_qos_bistatic_assignment_teacher(
        make_state(np.array([0.9, 0.0, 0.9, 0.9])),
        k, q, (800.0, 800.0), 2.5,
        commitment_frames=20, qos_weight=8.0)

    torch.testing.assert_close(neutral, torch.tensor([0, 3, 1, 2]))
    torch.testing.assert_close(weak_target, torch.tensor([1, 3, 0, 2]))
    assert len(torch.unique(weak_target)) == q
    assert torch.all(torch.linalg.vector_norm(movement, dim=-1) <= 2.5 + 1e-6)


def test_qos_teacher_projects_near_equivalent_labels_toward_student():
    k = q = 2
    state = torch.zeros(1, 8 * k + 6 * q + 1 + 2 * q)
    state[0, 0:2] = torch.tensor([0.25, 0.50])
    state[0, 8:10] = torch.tensor([0.75, 0.50])
    target_offset = 8 * k
    state[0, target_offset:target_offset + 2] = torch.tensor([0.25, 0.50])
    state[0, target_offset + 6:target_offset + 8] = torch.tensor([0.75, 0.50])
    pd_offset = target_offset + 6 * q + 1 + q
    state[0, pd_offset:pd_offset + q] = 0.5
    policy = torch.tensor([[[0.01, 0.99], [0.99, 0.01]]])
    labels, _ = compute_qos_bistatic_assignment_teacher(
        state, k, q, (800.0, 800.0), 2.5,
        commitment_frames=5,
        policy_assignment_probs=policy,
        policy_alignment_m=1000.0,
    )
    assert labels.tolist() == [1, 0]


def test_differentiable_quantized_inbox_trains_sender_message_content():
    torch.manual_seed(31)
    k, q = 3, 2
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True,
        use_target_allocation=True,
    )
    obs = torch.randn(2 * k, slices.total_dim)
    raw_messages = torch.randn(2 * k, 16, requires_grad=True)
    inbox_obs = build_differentiable_u2u_inbox(
        obs, torch.tanh(raw_messages), actor,
        num_agents=k, rate_index=2, rate_bits=8,
        num_rate_levels=5,
    )

    actor(inbox_obs)
    assignment = actor.last_target_assignment
    loss = -torch.log(assignment[:, 0].clamp_min(1e-8)).mean()
    loss.backward()

    assert raw_messages.grad is not None
    assert raw_messages.grad.abs().sum() > 0.0
    # The injected metadata matches the physical token layout.
    np.testing.assert_allclose(
        inbox_obs.detach().numpy()[:, slices.comm_mask_start:], 1.0)


def test_differentiable_inbox_supports_target_token_payloads():
    torch.manual_seed(33)
    k, q, teams, content_dim = 3, 2, 2, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True, comm_target_token_dim=content_dim,
        use_target_allocation=True, round_negotiation_enabled=True)
    # Distinct local target geometry is required to break the intentionally
    # symmetric all-zero assignment problem.
    obs = torch.randn(teams * k, slices.total_dim)
    messages = torch.randn(
        teams * k, q * content_dim, requires_grad=True)
    inbox = build_differentiable_u2u_inbox(
        obs, torch.tanh(messages), actor, num_agents=k,
        rate_index=2, rate_bits=8, num_rate_levels=5)

    tokens = inbox[:, slices.comm_token_start:slices.comm_mask_start]
    tokens = tokens.reshape(teams * k, (k - 1) * q, token_dim)
    torch.testing.assert_close(
        tokens[0, :q, content_dim + 1], torch.tensor([0.0, 1.0]))
    actor(
        inbox, comm_round_phase=torch.ones(teams * k),
        agent_identity=torch.arange(teams * k) % k)
    loss = -torch.log(
        actor.last_target_assignment[:, 0].clamp_min(1e-8)).mean()
    loss.backward()
    assert messages.grad is not None and torch.any(messages.grad != 0)


def test_differentiable_inbox_preserves_sparse_sender_token_masks():
    torch.manual_seed(331)
    k, q, content_dim = 3, 2, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True)
    obs = torch.zeros(k, slices.total_dim)
    messages = torch.randn(k, q * content_dim, requires_grad=True)
    sender_masks = torch.tensor([
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
    ])
    inbox = build_differentiable_u2u_inbox(
        obs, torch.tanh(messages), actor, num_agents=k,
        rate_index=2, rate_bits=8, num_rate_levels=5,
        sender_token_masks=sender_masks)

    receiver0_mask = inbox[
        0, slices.comm_mask_start:
        slices.comm_mask_start + slices.comm_mask_len]
    torch.testing.assert_close(
        receiver0_mask, torch.tensor([0.0, 1.0, 1.0, 0.0]))
    receiver0_tokens = inbox[
        0, slices.comm_token_start:slices.comm_mask_start
    ].reshape((k - 1) * q, token_dim)
    torch.testing.assert_close(
        receiver0_tokens[0], torch.zeros(token_dim))
    torch.testing.assert_close(
        receiver0_tokens[3], torch.zeros(token_dim))


def test_target_token_sinkhorn_uses_one_row_per_sender():
    torch.manual_seed(34)
    k, q, content_dim = 4, 4, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True, comm_target_token_dim=content_dim,
        use_target_allocation=True, use_team_sinkhorn=True)
    obs = torch.randn(k, slices.total_dim)
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + (k - 1) * q] = 1.0

    actor(obs)
    assignment = actor.last_target_assignment
    assert assignment.shape == (k, q)
    torch.testing.assert_close(
        assignment.sum(dim=-1), torch.ones(k), rtol=1e-5, atol=1e-6)
    # The old bug produced 1+(K-1)Q Sinkhorn rows. The corrected target-token
    # branch must construct exactly own + K-1 peer rows.
    with torch.no_grad():
        parsed = actor._parse_obs(obs)
        comm_tokens = parsed[7]
        msg_entities = actor.comm_token_enc(comm_tokens)
        assert msg_entities.reshape(k, k - 1, q, -1).shape[1] == k - 1


def test_differentiable_inbox_uses_previous_interleaved_team_messages():
    torch.manual_seed(35)
    k, q, teams = 2, 2, 3
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True,
        use_target_allocation=True,
    )
    obs = torch.zeros(teams * k, slices.total_dim)
    messages = torch.randn(teams * k, 16, requires_grad=True)
    inbox = build_differentiable_u2u_inbox(
        obs, torch.tanh(messages), actor,
        num_agents=k, rate_index=2, rate_bits=8,
        num_rate_levels=5, delay_teams=1,
        transition_masks=torch.ones(teams * k),
    )
    # Team 2 receiver 0 reads sender 1 from team 1, not the current team 2.
    loss = inbox[2 * k, slices.comm_token_start]
    loss.backward()
    assert messages.grad[k + 1, 0].abs() > 0.0
    assert messages.grad[2 * k + 1, 0] == 0.0


def test_sinkhorn_team_assignment_is_doubly_stochastic_and_differentiable():
    torch.manual_seed(37)
    logits = torch.randn(3, 4, 4, requires_grad=True)
    assignment = sinkhorn_normalize(logits, iterations=48)

    torch.testing.assert_close(
        assignment.sum(dim=-1), torch.ones(3, 4),
        rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        assignment.sum(dim=-2), torch.ones(3, 4),
        rtol=5e-3, atol=5e-3)
    assignment[:, 0, 0].sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0.0


def test_capacity_sinkhorn_respects_two_endpoint_marginals_and_gradient():
    """Each UAV serves two targets and each target receives two endpoints."""
    torch.manual_seed(371)
    logits = torch.randn(3, 4, 4, requires_grad=True)
    assignment = capacity_sinkhorn_normalize(
        logits, row_capacity=2, column_capacity=2,
        iterations=96, temperature=0.35)

    torch.testing.assert_close(
        assignment.sum(dim=-1), torch.full((3, 4), 2.0),
        rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        assignment.sum(dim=-2), torch.full((3, 4), 2.0),
        rtol=1e-3, atol=1e-3)
    assert torch.all((assignment >= 0.0) & (assignment <= 1.0))
    assignment[:, 0, 0].sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0.0


def test_compatible_warmstart_zeroes_new_capacity_bid_layers():
    torch.manual_seed(3711)
    cfg = load_config('config/exp_800_q4_u2u_sparse_claim_top2.yaml')
    env = UAVISACEnv(config=cfg, seed=1)
    try:
        action_space = ActionSpace(
            v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
        action_space.num_targets = cfg.scenario.Q
        action_space.structured_actor = True
        obs_dim = env.core.obs_builder.get_obs_dim()
        global_dim = env.core.obs_builder.get_global_state_dim()
        target_dim = cfg.marl.comm_target_token_dim
        receiver_dim = target_dim + 6
        old_agent = MAPPOAgent(
            0, obs_dim, global_dim, action_space, cfg.scenario.K,
            num_targets=cfg.scenario.Q, hidden_layers=[64, 64],
            use_comm_cross_attention=True,
            comm_token_dim=receiver_dim,
            comm_tokens_per_sender=cfg.scenario.Q,
            comm_payload_dim=cfg.scenario.Q * target_dim,
            comm_target_token_enabled=True,
            comm_target_token_dim=target_dim,
            use_target_allocation=True,
        )
        new_agent = MAPPOAgent(
            0, obs_dim, global_dim, action_space, cfg.scenario.K,
            num_targets=cfg.scenario.Q, hidden_layers=[64, 64],
            use_comm_cross_attention=True,
            comm_token_dim=receiver_dim,
            comm_tokens_per_sender=cfg.scenario.Q,
            comm_payload_dim=cfg.scenario.Q * target_dim,
            comm_target_token_enabled=True,
            comm_target_token_dim=target_dim,
            use_target_allocation=True,
            capacity_matching_enabled=True,
        )
        result = new_agent.load_actor_state_dict_compatible(
            old_agent.actor.state_dict())
        assert any('neighbor_bid' in key for key in result.missing_keys)
        for name, parameter in new_agent.actor.named_parameters():
            if 'neighbor_bid_msg_proj' in name or name.endswith(
                    'neighbor_bid_target_proj.bias'):
                torch.testing.assert_close(
                    parameter, torch.zeros_like(parameter))
        target_weight = new_agent.actor.neighbor_bid_target_proj.weight
        torch.testing.assert_close(
            target_weight, torch.eye(target_weight.shape[0]))
    finally:
        env.close()


def test_sparse_target_tokens_activate_capacity_matching():
    """Capacity matching must work when top-k communication masks some tokens."""
    torch.manual_seed(372)
    k, q, content_dim = 4, 4, 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True, comm_target_token_dim=content_dim,
        use_target_allocation=True, capacity_matching_enabled=True,
        capacity_matching_row_capacity=2,
        capacity_matching_column_capacity=2,
        capacity_matching_iterations=8, capacity_matching_blend=1.0)
    obs = torch.randn(k, slices.total_dim)
    # Every sender exposes exactly two of its four target claims.  The masks
    # differ by receiver but every peer still contributes at least one claim.
    sparse_mask = torch.tensor([
        [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    ], dtype=obs.dtype)
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + (k - 1) * q] = sparse_mask

    actor(obs)
    assert actor.last_capacity_assignment is not None
    # Each decentralized actor instance builds one local team view.
    assert actor.last_capacity_assignment.shape == (k, k, q)
    torch.testing.assert_close(
        actor.last_capacity_assignment.sum(dim=-1),
        torch.full((k, k), 2.0), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        actor.last_capacity_assignment.sum(dim=-2),
        torch.full((k, q), 2.0), rtol=2e-3, atol=2e-3)


def test_distributed_movement_matching_is_one_to_one_and_separate():
    torch.manual_seed(3721)
    k = q = 4
    content_dim = 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True, comm_target_token_dim=content_dim,
        use_target_allocation=True, capacity_matching_enabled=True,
        hierarchical_dual_assignment_enabled=True,
        movement_team_matching_enabled=True,
        movement_team_matching_blend=1.0,
        movement_team_matching_iterations=16,
        capacity_matching_row_capacity=2,
        capacity_matching_column_capacity=2)
    obs = torch.randn(k, slices.total_dim)
    global_claims = torch.tensor([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=obs.dtype)
    global_scores = torch.tensor([
        [0.9, 0.2, 0.0, 0.0],
        [0.0, 0.8, 0.5, 0.0],
        [0.0, 0.0, 0.7, 0.6],
        [0.4, 0.0, 0.0, 0.9],
    ], dtype=obs.dtype)
    for receiver in range(k):
        obs[receiver, slices.comm_start:slices.comm_start + q] = (
            global_scores[receiver])
        obs[receiver, slices.comm_start + q:
            slices.comm_start + 2 * q] = global_claims[receiver]
        peer_ids = [sender for sender in range(k) if sender != receiver]
        peer_tokens = torch.zeros(k - 1, q, token_dim)
        peer_tokens[..., 0] = global_scores[peer_ids]
        obs[receiver, slices.comm_token_start:
            slices.comm_token_start + (k - 1) * q * token_dim] = (
                peer_tokens.reshape(-1))
        obs[receiver, slices.comm_mask_start:
            slices.comm_mask_start + (k - 1) * q] = (
                global_claims[peer_ids].reshape(-1))
    identities = torch.arange(k)
    actor(obs, agent_identity=identities)
    movement = actor.last_movement_team_assignment
    capacity = actor.last_capacity_assignment
    assert movement is not None and capacity is not None
    torch.testing.assert_close(
        movement.sum(dim=-1), torch.ones(k, k),
        rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        movement.sum(dim=-2), torch.ones(k, q),
        rtol=2e-3, atol=2e-3)
    for receiver in range(1, k):
        torch.testing.assert_close(movement[receiver], movement[0])
    torch.testing.assert_close(
        capacity.sum(dim=-1), torch.full((k, k), 2.0),
        rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        actor.last_movement_assignment,
        movement[torch.arange(k), identities, :],
        rtol=2e-3, atol=2e-3)


def test_hierarchical_assignment_separates_motion_from_endpoint_capacity():
    """A one-target motion label must not supervise the capacity-2 row."""
    torch.manual_seed(373)
    k = q = 4
    content_dim = 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True, comm_target_token_dim=content_dim,
        use_target_allocation=True, capacity_matching_enabled=True,
        capacity_matching_row_capacity=2,
        capacity_matching_column_capacity=2,
        capacity_matching_iterations=8, capacity_matching_blend=1.0,
        target_allocation_straight_through=True,
        hierarchical_dual_assignment_enabled=True)
    obs = torch.randn(k, slices.total_dim)
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + (k - 1) * q] = 1.0

    actor(obs)
    movement = actor.last_movement_assignment
    resource = actor.last_target_assignment
    assert movement is not None and resource is not None
    assert movement.shape == resource.shape == (k, q)
    torch.testing.assert_close(movement.sum(dim=-1), torch.ones(k))
    torch.testing.assert_close(resource.sum(dim=-1), torch.ones(k))
    # Capacity projection deliberately spreads a row over two endpoint bids;
    # the motion branch retains the pre-projection one-target distribution.
    assert not torch.allclose(movement, resource, rtol=1e-4, atol=1e-4)
    hard_motion = actor.last_target_assignment_st
    torch.testing.assert_close(
        hard_motion.detach().sum(dim=-1), torch.ones(k))
