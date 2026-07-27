from __future__ import annotations

import torch

from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.environment.action import ActionSpace
from uav_isac.environment.observation_slices import ObservationSlices


def _agent(
    k: int,
    q: int,
    *,
    scale_heads: bool,
    equivariant_round: bool = False,
) -> MAPPOAgent:
    content_dim = 4
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    action_space = ActionSpace(v_max=25.0, dt=0.1, learn_roles=False)
    action_space.num_targets = q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 32
    return MAPPOAgent(
        agent_id=0,
        obs_dim=slices.total_dim,
        global_state_dim=4 * (k + q) + 1,
        action_space=action_space,
        num_agents=k,
        num_targets=q,
        hidden_layers=[32, 32],
        device='cpu',
        use_comm_cross_attention=True,
        comm_token_dim=token_dim,
        comm_tokens_per_sender=q,
        comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True,
        comm_target_token_dim=content_dim,
        use_target_allocation=True,
        round_negotiation_enabled=True,
        scale_equivariant_comm_heads_enabled=scale_heads,
        permutation_equivariant_round_encoding_enabled=equivariant_round,
    )


def test_local_set_rate_and_power_are_target_permutation_invariant() -> None:
    torch.manual_seed(9101)
    agent = _agent(4, 4, scale_heads=True)
    actor = agent.actor
    with torch.no_grad():
        actor.comm_set_attention.weight.normal_(0.0, 0.2)
        actor.comm_set_rate_head.weight.normal_(0.0, 0.2)
        actor.isac_set_power_head.weight.normal_(0.0, 0.2)
    tokens = torch.randn(3, 4, 4)
    permuted = tokens[:, torch.tensor([2, 0, 3, 1])]

    _, rate = actor.communication_parameters(tokens.reshape(3, -1))
    power = actor.isac_resource_parameters(tokens.reshape(3, -1))[0]
    _, permuted_rate = actor.communication_parameters(
        permuted.reshape(3, -1))
    permuted_power = actor.isac_resource_parameters(
        permuted.reshape(3, -1))[0]

    torch.testing.assert_close(rate, permuted_rate)
    torch.testing.assert_close(power, permuted_power)


def test_legacy_flat_heads_project_to_same_output_across_cardinality() -> None:
    torch.manual_seed(9102)
    old = _agent(4, 4, scale_heads=False)
    new = _agent(
        6, 6, scale_heads=True, equivariant_round=True)
    with torch.no_grad():
        old.actor.comm_rate_head.weight.normal_(0.0, 0.2)
        old.actor.comm_rate_head.bias.normal_(0.0, 0.2)
        old.actor.isac_power_mean_head.weight.normal_(0.0, 0.2)
        old.actor.isac_power_mean_head.bias.normal_(0.0, 0.2)
    new.load_actor_state_dict_compatible(old.actor.state_dict())

    token = torch.randn(5, 4)
    old_payload = token[:, None, :].expand(-1, 4, -1).reshape(5, -1)
    new_payload = token[:, None, :].expand(-1, 6, -1).reshape(5, -1)
    old_rate = old.actor.communication_parameters(old_payload)[1]
    old_power = old.actor.isac_resource_parameters(old_payload)[0]
    new_rate = new.actor.communication_parameters(new_payload)[1]
    new_power = new.actor.isac_resource_parameters(new_payload)[0]

    torch.testing.assert_close(old_rate, new_rate, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(old_power, new_power, rtol=1e-5, atol=1e-6)


def test_equivariant_round_encoding_does_not_use_absolute_agent_id() -> None:
    torch.manual_seed(9103)
    old = _agent(4, 4, scale_heads=False)
    new = _agent(
        4, 4, scale_heads=True, equivariant_round=True)
    new.load_actor_state_dict_compatible(old.actor.state_dict())
    obs_dim = new.actor._obs_slices.total_dim
    repeated_obs = torch.randn(1, obs_dim).expand(4, -1).clone()
    phase = torch.ones(4)

    output_a = new.actor(
        repeated_obs, comm_round_phase=phase,
        agent_identity=torch.arange(4))[3]
    output_b = new.actor(
        repeated_obs, comm_round_phase=phase,
        agent_identity=torch.tensor([2, 0, 3, 1]))[3]

    torch.testing.assert_close(output_a, output_b)
