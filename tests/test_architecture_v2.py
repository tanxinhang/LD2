from __future__ import annotations

import torch
import pytest

from uav_isac.environment.belief_fusion import MultiHeadNeighborAttention
from uav_isac.agents.networks import CriticNetwork, StructuredActorNetwork
from uav_isac.environment.observation_slices import ObservationSlices


@pytest.mark.parametrize(
    ("dimension", "heads"),
    [(0, 1), (8, 0), (10, 4), (True, 1), (8, False)],
)
def test_neighbor_attention_rejects_invalid_head_geometry(dimension, heads):
    with pytest.raises(ValueError, match="positive integers"):
        MultiHeadNeighborAttention(D=dimension, num_heads=heads)


def _actor(
    k: int,
    q: int,
    dim: int = 32,
    **actor_kwargs,
) -> StructuredActorNetwork:
    content_dim = 4
    receiver_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=True,
        comm_token_dim=receiver_dim,
        comm_tokens_per_sender=q,
    )
    return StructuredActorNetwork(
        obs_dim=slices.total_dim,
        K=k,
        Q=q,
        entity_dim=dim,
        single_frame_dim=slices.total_dim,
        use_corrected_parser=True,
        use_p0=False,
        use_comm_cross_attention=True,
        comm_token_dim=receiver_dim,
        comm_tokens_per_sender=q,
        comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True,
        round_negotiation_enabled=True,
        architecture_v2_enabled=True,
        **actor_kwargs,
    )


def _permute_targets(
    obs: torch.Tensor,
    slices: ObservationSlices,
    permutation: torch.Tensor,
) -> torch.Tensor:
    result = obs.clone()
    batch = obs.shape[0]
    q = slices.Q

    belief_len = q * slices.belief_per_target
    result[:, slices.belief_start:slices.belief_start + belief_len] = (
        obs[:, slices.belief_start:slices.belief_start + belief_len]
        .reshape(batch, q, slices.belief_per_target)[:, permutation]
        .reshape(batch, -1)
    )
    geom_len = q * slices.geom_per_target
    result[:, slices.geom_start:slices.geom_start + geom_len] = (
        obs[:, slices.geom_start:slices.geom_start + geom_len]
        .reshape(batch, q, slices.geom_per_target)[:, permutation]
        .reshape(batch, -1)
    )
    result[:, slices.pd_hist_start:slices.pd_hist_start + q] = (
        obs[:, slices.pd_hist_start:slices.pd_hist_start + q][
            :, permutation]
    )
    return result


def _critic_state(batch: int, k: int, q: int) -> torch.Tensor:
    base_dim = 8 * k + 8 * q + 1
    return torch.randn(batch, base_dim + k + 16)


def _equivariant_critic(k: int, q: int) -> CriticNetwork:
    return CriticNetwork(
        state_dim=8 * k + 8 * q + 1,
        hidden_layers=[32, 32],
        num_agents=k,
        comm_dim=16,
        num_targets=q,
        equivariant_value_critic_enabled=True,
    )


def test_v2_actor_parameter_shapes_do_not_depend_on_k_or_q() -> None:
    actor_4 = _actor(4, 4)
    actor_6 = _actor(6, 6)
    shapes_4 = {
        name: tuple(value.shape)
        for name, value in actor_4.state_dict().items()
    }
    shapes_6 = {
        name: tuple(value.shape)
        for name, value in actor_6.state_dict().items()
    }

    assert shapes_4 == shapes_6
    assert tuple(actor_4.comm_log_std.shape) == (4,)
    assert tuple(actor_4.isac_sensing_log_std.shape) == (1,)
    assert not hasattr(actor_4, 'intent_head')
    assert not hasattr(actor_4, 'isac_sensing_mean_head')
    assert not hasattr(actor_4, 'round_phase_enc')


def test_v2_shared_action_scales_expand_without_learned_q_dimension() -> None:
    actor = _actor(4, 4)
    obs = torch.zeros(2, actor._obs_slices.total_dim)
    token = actor(obs)[3]
    message_log_std, _ = actor.communication_parameters(token)
    _, _, sensing_mean, sensing_log_std = actor.isac_resource_parameters(
        token)

    assert message_log_std.shape == (16,)
    assert actor.comm_log_std.shape == (4,)
    assert sensing_mean.shape == (2, 4)
    assert sensing_log_std.shape == (4,)
    assert actor.isac_sensing_log_std.shape == (1,)


def test_v2_physics_prior_prefers_weak_reachable_target() -> None:
    actor = _actor(4, 4).eval()
    slices = actor._obs_slices
    obs = torch.zeros(1, slices.total_dim)
    geometry = obs[
        :, slices.geom_start:
        slices.geom_start + slices.Q * slices.geom_per_target
    ].reshape(1, slices.Q, slices.geom_per_target)
    geometry[0, :, 2] = torch.tensor([0.8, 0.2, 0.5, 0.9])
    obs[0, slices.pd_hist_start:
        slices.pd_hist_start + slices.Q] = torch.tensor(
            [0.8, 0.2, 0.6, 0.9])

    token = actor(obs)[3]
    sensing = actor.isac_resource_parameters(token)[2]

    assert actor.last_v2_physics_prior.argmax(dim=-1).item() == 1
    assert actor.last_target_assignment.argmax(dim=-1).item() == 1
    assert sensing.argmax(dim=-1).item() == 1


def test_v2_crisis_prior_prevents_greedy_silence_collapse() -> None:
    actor = _actor(4, 4).eval()
    slices = actor._obs_slices
    crisis_obs = torch.zeros(2, slices.total_dim)
    crisis_token = actor(crisis_obs)[3]
    crisis_rate = actor.communication_parameters(crisis_token)[1]

    safe_obs = crisis_obs.clone()
    safe_obs[:, slices.pd_hist_start:
             slices.pd_hist_start + slices.Q] = 1.0
    safe_token = actor(safe_obs)[3]
    safe_rate = actor.communication_parameters(safe_token)[1]

    assert torch.all(crisis_rate[:, 1] > crisis_rate[:, 0])
    assert torch.all(safe_rate[:, 0] > safe_rate[:, 1])


def test_v2_token_header_carries_comparable_local_bid() -> None:
    actor = _actor(4, 4).eval()
    slices = actor._obs_slices
    obs = torch.zeros(1, slices.total_dim)
    geometry = obs[
        :, slices.geom_start:
        slices.geom_start + slices.Q * slices.geom_per_target
    ].reshape(1, slices.Q, slices.geom_per_target)
    geometry[0, :, 2] = torch.tensor([0.1, 0.3, 0.5, 0.7])

    token = actor(obs)[3].reshape(1, slices.Q, -1)

    torch.testing.assert_close(token[..., 0], actor.last_v2_local_bids)
    assert token[..., 0].std() > 0.0


def test_v2_sensing_aligned_claim_mask_matches_executed_sensing_topk() -> None:
    k, q, content_dim = 4, 4, 4
    receiver_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=True,
        comm_token_dim=receiver_dim,
        comm_tokens_per_sender=q,
    )
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim,
        K=k,
        Q=q,
        entity_dim=32,
        single_frame_dim=slices.total_dim,
        use_corrected_parser=True,
        use_comm_cross_attention=True,
        comm_token_dim=receiver_dim,
        comm_tokens_per_sender=q,
        comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True,
        sparse_claim_enabled=True,
        sparse_claim_share_topk=2,
        comm_aided_sensing_enabled=True,
        architecture_v2_enabled=True,
        architecture_v2_sensing_aligned_claims_enabled=True,
    ).eval()
    obs = torch.randn(3, slices.total_dim)
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + slices.comm_mask_len] = 0.0

    token = actor(obs)[3]
    sensing_logits = actor.isac_resource_parameters(token)[2]
    expected = torch.zeros_like(sensing_logits)
    expected.scatter_(
        1, torch.topk(sensing_logits, k=2, dim=-1).indices, 1.0)

    torch.testing.assert_close(actor.last_outgoing_token_mask, expected)


def test_v2_peer_bid_softly_suppresses_lost_target_competition() -> None:
    actor = _actor(4, 4).eval()
    slices = actor._obs_slices
    obs = torch.zeros(1, slices.total_dim)
    received_len = (
        (slices.K - 1) * slices.Q * actor.comm_token_dim)
    received = obs[
        :, slices.comm_token_start:
        slices.comm_token_start + received_len
    ].reshape(1, (slices.K - 1) * slices.Q, -1)
    received[..., 0] = -1.0
    received[:, 0::slices.Q, 0] = 1.0
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + slices.comm_mask_len] = 1.0
    obs[:, slices.comm_start:slices.comm_start + slices.Q] = torch.tensor(
        [[-0.6, -0.2, 0.2, 0.6]])
    obs[:, slices.comm_start + slices.Q:
        slices.comm_start + 2 * slices.Q] = 1.0

    actor(obs, comm_round_phase=torch.ones(1))

    agreement = actor.last_v2_bid_agreement
    assert agreement is not None
    assert agreement[0, 0] < agreement[0, 1:].min()
    movement = actor.last_v2_movement_consensus
    endpoints = actor.last_v2_endpoint_consensus
    assert movement is not None
    assert endpoints is not None
    torch.testing.assert_close(
        actor.last_movement_assignment,
        movement[:, 0, :],
    )
    torch.testing.assert_close(
        movement.sum(dim=-1), torch.ones(1, slices.K))
    torch.testing.assert_close(
        movement.sum(dim=-2), torch.ones(1, slices.Q))
    torch.testing.assert_close(
        endpoints.sum(dim=-1),
        torch.full((1, slices.K), 2.0),
        atol=2e-4,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        endpoints.sum(dim=-2),
        torch.full((1, slices.Q), 2.0),
        atol=1e-3,
        rtol=1e-3,
    )


def test_v2_target_permutation_equivariance_and_invariant_team_actions() -> None:
    torch.manual_seed(9301)
    k, q, batch = 4, 4, 3
    actor = _actor(k, q).eval()
    slices = actor._obs_slices
    obs = torch.randn(batch, slices.total_dim)
    # Isolate the target-set property: silence is a neutral inbox.
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + slices.comm_mask_len] = 0.0
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_obs = _permute_targets(obs, slices, permutation)

    original = actor(obs, comm_round_phase=torch.zeros(batch))
    original_assignment = actor.last_target_assignment.clone()
    original_sensing = actor.isac_resource_parameters(original[3])[2]
    original_rate = actor.communication_parameters(original[3])[1]
    original_power = actor.isac_resource_parameters(original[3])[0]

    permuted = actor(
        permuted_obs, comm_round_phase=torch.zeros(batch))
    permuted_assignment = actor.last_target_assignment.clone()
    permuted_sensing = actor.isac_resource_parameters(permuted[3])[2]
    permuted_rate = actor.communication_parameters(permuted[3])[1]
    permuted_power = actor.isac_resource_parameters(permuted[3])[0]

    torch.testing.assert_close(permuted[0], original[0])
    torch.testing.assert_close(permuted[2], original[2])
    torch.testing.assert_close(
        permuted[3].reshape(batch, q, -1),
        original[3].reshape(batch, q, -1)[:, permutation],
    )
    torch.testing.assert_close(
        permuted_assignment, original_assignment[:, permutation])
    torch.testing.assert_close(
        permuted_sensing, original_sensing[:, permutation])
    torch.testing.assert_close(permuted_rate, original_rate)
    torch.testing.assert_close(permuted_power, original_power)


def test_v2_actor_has_finite_end_to_end_gradients() -> None:
    torch.manual_seed(9302)
    actor = _actor(4, 4)
    obs = torch.randn(5, actor._obs_slices.total_dim)
    obs[:, actor._obs_slices.comm_mask_start:
        actor._obs_slices.comm_mask_start
        + actor._obs_slices.comm_mask_len] = 1.0

    dp, _, role, token, _, _ = actor(
        obs, comm_round_phase=torch.ones(5))
    _, rate = actor.communication_parameters(token)
    power, _, sensing, _ = actor.isac_resource_parameters(token)
    loss = (
        dp.square().mean()
        + role.square().mean()
        + token.square().mean()
        + rate.square().mean()
        + power.square().mean()
        + sensing.square().mean()
    )
    loss.backward()

    required = (
        actor.v2_target_policy[0].weight.grad,
        actor.v2_assignment_head.weight.grad,
        actor.v2_sensing_head.weight.grad,
        actor.v2_movement_head.weight.grad,
        actor.comm_target_token_head.weight.grad,
    )
    assert all(gradient is not None for gradient in required)
    assert all(torch.isfinite(gradient).all() for gradient in required)


def test_v2_local_bid_logits_support_direct_teacher_credit() -> None:
    torch.manual_seed(9304)
    actor = _actor(4, 4)
    obs = torch.randn(8, actor._obs_slices.total_dim)
    obs[:, actor._obs_slices.comm_mask_start:
        actor._obs_slices.comm_mask_start
        + actor._obs_slices.comm_mask_len] = 0.0
    actor(obs, comm_round_phase=torch.zeros(8))
    logits = actor.last_v2_local_bid_logits
    assert logits is not None and logits.shape == (8, 4)

    labels = torch.arange(8) % 4
    torch.nn.functional.cross_entropy(logits, labels).backward()
    gradient = actor.v2_assignment_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0


def test_v2_modular_coordination_is_neutral_after_checkpoint_migration() -> None:
    torch.manual_seed(9305)
    baseline = _actor(4, 4).eval()
    torch.manual_seed(9305)
    modular = _actor(
        4, 4,
        architecture_v2_modular_coordination_enabled=True,
        architecture_v2_modular_num_experts=3,
    ).eval()
    baseline_state = baseline.state_dict()
    modular_state = modular.state_dict()
    for name, value in baseline_state.items():
        torch.testing.assert_close(modular_state[name], value)
    obs = torch.randn(5, baseline._obs_slices.total_dim)
    obs[:, baseline._obs_slices.comm_mask_start:
        baseline._obs_slices.comm_mask_start
        + baseline._obs_slices.comm_mask_len] = 0.0

    baseline_out = baseline(obs, comm_round_phase=torch.zeros(5))
    baseline_assignment = baseline.last_target_assignment.clone()
    baseline_bid = baseline.last_v2_local_bid_logits.clone()
    modular_out = modular(obs, comm_round_phase=torch.zeros(5))

    torch.testing.assert_close(modular_out[0], baseline_out[0])
    torch.testing.assert_close(modular_out[3], baseline_out[3])
    torch.testing.assert_close(
        modular.last_target_assignment, baseline_assignment)
    torch.testing.assert_close(modular.last_v2_local_bid_logits, baseline_bid)
    torch.testing.assert_close(
        modular.last_v2_module_routing,
        torch.full((5, 3), 1.0 / 3.0),
    )
    torch.testing.assert_close(
        modular.last_v2_module_residual_norm, torch.zeros(5))


def test_v2_modular_router_is_target_permutation_invariant() -> None:
    torch.manual_seed(9306)
    actor = _actor(
        4, 4,
        architecture_v2_modular_coordination_enabled=True,
        architecture_v2_modular_num_experts=3,
    ).eval()
    with torch.no_grad():
        torch.nn.init.normal_(actor.v2_module_router[-1].weight, std=0.05)
    slices = actor._obs_slices
    obs = torch.randn(4, slices.total_dim)
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + slices.comm_mask_len] = 0.0
    permutation = torch.tensor([2, 0, 3, 1])

    actor(obs, comm_round_phase=torch.zeros(4))
    original_route = actor.last_v2_module_routing.clone()
    original_bid = actor.last_v2_local_bid_logits.clone()
    actor(
        _permute_targets(obs, slices, permutation),
        comm_round_phase=torch.zeros(4),
    )

    torch.testing.assert_close(actor.last_v2_module_routing, original_route)
    torch.testing.assert_close(
        actor.last_v2_local_bid_logits,
        original_bid[:, permutation],
    )


def test_v2_direct_bid_credit_can_train_identity_free_router() -> None:
    torch.manual_seed(9307)
    actor = _actor(
        4, 4,
        architecture_v2_modular_coordination_enabled=True,
        architecture_v2_modular_num_experts=3,
    )
    with torch.no_grad():
        torch.nn.init.normal_(actor.v2_assignment_head.weight, std=0.1)
    obs = torch.randn(12, actor._obs_slices.total_dim)
    obs[:, actor._obs_slices.comm_mask_start:
        actor._obs_slices.comm_mask_start
        + actor._obs_slices.comm_mask_len] = 0.0
    actor(obs, comm_round_phase=torch.zeros(12))
    labels = torch.arange(12) % 4
    torch.nn.functional.cross_entropy(
        actor.last_v2_local_bid_logits, labels).backward()

    gradient = actor.v2_module_router[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0


def test_equivariant_value_critic_shapes_are_cardinality_independent() -> None:
    critic_4 = _equivariant_critic(4, 4)
    critic_6 = _equivariant_critic(6, 6)
    shapes_4 = {
        name: tuple(value.shape)
        for name, value in critic_4.state_dict().items()
    }
    shapes_6 = {
        name: tuple(value.shape)
        for name, value in critic_6.state_dict().items()
    }
    assert shapes_4 == shapes_6


def test_equivariant_value_critic_respects_uav_and_target_permutations() -> None:
    torch.manual_seed(9303)
    k, q = 4, 4
    critic = _equivariant_critic(k, q).eval()
    state = _critic_state(3, k, q)
    permutation = torch.tensor([2, 0, 3, 1])

    uav_permuted = state.clone()
    uav_permuted[:, :8 * k] = (
        state[:, :8 * k].reshape(3, k, 8)[:, permutation].reshape(3, -1)
    )
    base_dim = 8 * k + 8 * q + 1
    uav_permuted[:, base_dim:base_dim + k] = state[
        :, base_dim:base_dim + k][:, permutation]

    target_permuted = state.clone()
    motion_start = 8 * k
    motion_end = motion_start + 6 * q
    target_permuted[:, motion_start:motion_end] = (
        state[:, motion_start:motion_end]
        .reshape(3, q, 6)[:, permutation].reshape(3, -1)
    )
    uncertainty_start = motion_end + 1
    pd_start = uncertainty_start + q
    target_permuted[
        :, uncertainty_start:uncertainty_start + q
    ] = state[:, uncertainty_start:uncertainty_start + q][:, permutation]
    target_permuted[:, pd_start:pd_start + q] = state[
        :, pd_start:pd_start + q][:, permutation]

    value, target_value = critic.forward_with_targets(state)
    uav_value, uav_target_value = critic.forward_with_targets(uav_permuted)
    permuted_value, permuted_target_value = critic.forward_with_targets(
        target_permuted)

    torch.testing.assert_close(uav_value, value)
    torch.testing.assert_close(uav_target_value, target_value)
    torch.testing.assert_close(permuted_value, value)
    torch.testing.assert_close(
        permuted_target_value, target_value[:, permutation])
