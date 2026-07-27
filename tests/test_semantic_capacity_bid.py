import torch

from uav_isac.agents.networks import (
    StructuredActorNetwork,
    apply_semantic_capacity_bid_correction,
)
from uav_isac.environment.observation_slices import ObservationSlices


def test_semantic_capacity_correction_is_row_neutral_and_mask_safe():
    logits = torch.zeros(1, 2, 4)
    evidence = torch.tensor([[[0.9, 0.3, 0.8, 0.2],
                              [0.1, 0.7, 0.4, 0.9]]])
    pd = torch.tensor([[[0.8, 0.5, 0.2, 0.9],
                        [0.6, 0.8, 0.3, 0.4]]])
    valid = torch.tensor([[[1, 1, 0, 0],
                           [0, 1, 1, 0]]], dtype=torch.bool)

    corrected, bias = apply_semantic_capacity_bid_correction(
        logits, evidence, pd, valid, gain=0.5)

    torch.testing.assert_close(
        (bias * valid).sum(dim=-1), torch.zeros(1, 2))
    torch.testing.assert_close(
        bias[~valid], torch.zeros_like(bias[~valid]))
    torch.testing.assert_close(corrected, 0.5 * bias)
    unchanged, _ = apply_semantic_capacity_bid_correction(
        logits, evidence, pd, valid, gain=0.0)
    torch.testing.assert_close(unchanged, logits)


def _build_actor(enabled: bool, extra_token: bool = False) -> StructuredActorNetwork:
    k = q = 4
    content_dim = 16
    token_dim = content_dim + 6
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    return StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, comm_payload_dim=q * content_dim,
        comm_target_token_enabled=True, comm_target_token_dim=content_dim,
        use_target_allocation=True, capacity_matching_enabled=True,
        capacity_matching_row_capacity=2,
        capacity_matching_column_capacity=2,
        capacity_matching_iterations=16, capacity_matching_blend=1.0,
        hierarchical_dual_assignment_enabled=True,
        sparse_claim_enabled=extra_token,
        sparse_claim_share_topk=2,
        sparse_claim_desired_endpoints=2,
        comm_semantic_decoder_enabled=True,
        comm_semantic_capacity_bid_enabled=enabled,
        comm_semantic_capacity_bid_gain=1.0,
        comm_semantic_extra_token_enabled=extra_token,
        comm_semantic_extra_token_threshold=0.01,
    )


def test_semantic_capacity_bid_changes_endpoints_not_movement():
    torch.manual_seed(3801)
    base = _build_actor(False)
    corrected = _build_actor(True)
    corrected.load_state_dict(base.state_dict())

    # Make the decoder monotone in latent token dimension one so the test does
    # not depend on random semantic-head initialization.
    for actor in (base, corrected):
        decoder = actor.comm_semantic_decoder
        for parameter in decoder.parameters():
            parameter.data.zero_()
        decoder[0].weight.data[0, 0] = 1.0
        decoder[2].weight.data[0, 0] = 1.0
        decoder[4].weight.data[:, 0] = torch.tensor([2.0, 1.0])

    k = q = 4
    token_dim = 22
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    obs = torch.zeros(k, slices.total_dim)
    tokens = torch.zeros(k, k - 1, q, token_dim)
    tokens[..., 1] = torch.tensor([0.2, 0.8, 1.4, 2.0])
    obs[:, slices.comm_token_start:slices.comm_mask_start] = tokens.reshape(k, -1)
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + (k - 1) * q] = 1.0
    identities = torch.arange(k)

    with torch.inference_mode():
        base(obs, agent_identity=identities)
        corrected(obs, agent_identity=identities)

    torch.testing.assert_close(
        corrected.last_movement_assignment,
        base.last_movement_assignment,
        rtol=0.0,
        atol=0.0,
    )
    assert corrected.last_comm_semantic_capacity_bias is not None
    assert not torch.allclose(
        corrected.last_target_assignment, base.last_target_assignment)
    torch.testing.assert_close(
        corrected.last_capacity_assignment.sum(dim=-1),
        torch.full((k, k), 2.0), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        corrected.last_capacity_assignment.sum(dim=-2),
        torch.full((k, q), 2.0), rtol=2e-3, atol=2e-3)


def test_semantic_extra_token_adds_at_most_one_underloaded_edge():
    torch.manual_seed(3802)
    actor = _build_actor(False, extra_token=True)
    k = q = 4
    token_dim = 22
    slices = ObservationSlices.from_config(
        k, q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q)
    obs = torch.zeros(k, slices.total_dim)
    # Set exp(-distance/150 m) to one for every local target.
    for target in range(q):
        obs[:, slices.geom_start + target * slices.geom_per_target + 6] = 1.0
    # Peers claim only target zero, leaving three targets under capacity.
    peer_mask = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(k, k - 1)
    obs[:, slices.comm_mask_start:
        slices.comm_mask_start + (k - 1) * q] = peer_mask

    with torch.inference_mode():
        actor(obs, agent_identity=torch.arange(k))

    extra = actor.last_comm_semantic_extra_token_mask
    outgoing = actor.last_outgoing_token_mask
    assert extra is not None and outgoing is not None
    torch.testing.assert_close(extra.sum(dim=-1), torch.ones(k))
    torch.testing.assert_close(outgoing.sum(dim=-1), torch.full((k,), 3.0))
