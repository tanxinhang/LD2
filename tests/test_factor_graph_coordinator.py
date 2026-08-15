import numpy as np
import torch

from uav_isac.coordination.factor_graph_coordinator import (
    FiniteRoundFactorGraphCoordinator,
    assert_hard_structure_invariants,
    decode_lightweight_feasible,
    factor_graph_edge_features,
)


def _example(batch=2, K=4, Q=3):
    generator = torch.Generator().manual_seed(7)
    value = torch.rand(batch, K, K, Q, generator=generator)
    mask = torch.rand(batch, K, K, Q, generator=generator) > 0.35
    diagonal = torch.arange(K)
    mask[:, diagonal, diagonal, :] = False
    return value, mask


def test_factor_graph_shapes_and_masking():
    value, mask = _example()
    features = factor_graph_edge_features(value, mask)
    model = FiniteRoundFactorGraphCoordinator(
        hidden_dim=16, rounds=2, coupling_rounds=2,
        use_global_context=True)
    output = model(features, mask)
    assert output.edge_logits.shape == value.shape
    assert output.owner_logits.shape == (2, 3, 5)
    assert output.role_logits.shape == (2, 4, 3)
    assert output.stop_logit.shape == (2,)
    assert len(output.round_edge_logits) == 2
    assert torch.all(output.edge_logits[~mask] < -10)
    assert torch.all(torch.isfinite(output.owner_logits))
    assert torch.all(torch.isfinite(output.role_logits))


def test_factor_graph_is_uav_permutation_equivariant():
    value, mask = _example(batch=1, K=4, Q=3)
    model = FiniteRoundFactorGraphCoordinator(hidden_dim=16, rounds=2).eval()
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.inference_mode():
        original = model(factor_graph_edge_features(value, mask), mask)
        permuted_value = value[:, permutation][:, :, permutation]
        permuted_mask = mask[:, permutation][:, :, permutation]
        permuted = model(
            factor_graph_edge_features(permuted_value, permuted_mask),
            permuted_mask,
        )
    expected_edge = original.edge_logits[:, permutation][:, :, permutation]
    expected_role = original.role_logits[:, permutation]
    expected_owner = torch.cat(
        (original.owner_logits[:, :, :4][:, :, permutation],
         original.owner_logits[:, :, 4:]), dim=-1)
    assert torch.allclose(permuted.edge_logits, expected_edge, atol=1e-5)
    assert torch.allclose(permuted.role_logits, expected_role, atol=1e-5)
    assert torch.allclose(permuted.owner_logits, expected_owner, atol=1e-5)


def test_lightweight_projection_enforces_hard_invariants():
    K, Q = 4, 3
    mask = np.ones((K, K, Q), dtype=bool)
    mask[np.arange(K), np.arange(K), :] = False
    edge = np.arange(K * K * Q, dtype=float).reshape(K, K, Q)
    roles = np.array([
        [0, 5, 0], [0, 4, 0], [0, 0, 5], [0, 0, 4]], dtype=float)
    owners = np.zeros((Q, K + 1), dtype=float)
    owners[:, 2] = 3.0
    result = decode_lightweight_feasible(
        edge, owners, roles, mask,
        target_pair_limit=2,
        reports_per_receiver=4,
        edge_value=edge,
    )
    assert_hard_structure_invariants(
        result.selected, reports_per_receiver=4)
    assert np.any(result.selected)
    assert set(np.flatnonzero(np.any(result.selected, axis=(1, 2)))) <= {0, 1}


def test_lightweight_projection_fails_closed_without_role_partition():
    K, Q = 3, 2
    mask = np.ones((K, K, Q), dtype=bool)
    mask[np.arange(K), np.arange(K), :] = False
    roles = np.tile(np.array([0.0, 5.0, 0.0]), (K, 1))
    owners = np.zeros((Q, K + 1), dtype=float)
    result = decode_lightweight_feasible(
        np.ones_like(mask, dtype=float), owners, roles, mask,
        target_pair_limit=2,
        reports_per_receiver=4,
    )
    assert result.role_fallback
    assert not np.any(result.selected)
    assert_hard_structure_invariants(
        result.selected, reports_per_receiver=4)


def test_proposal_already_respects_predicted_roles_and_owner():
    K, Q = 4, 2
    mask = np.ones((K, K, Q), dtype=bool)
    mask[np.arange(K), np.arange(K), :] = False
    roles = np.array([
        [0, 5, 0], [0, 4, 0], [0, 0, 5], [5, 0, 0]], dtype=float)
    owners = np.full((Q, K + 1), -3.0)
    owners[:, 2] = 3.0
    result = decode_lightweight_feasible(
        np.ones_like(mask, dtype=float), owners, roles, mask,
        target_pair_limit=2, reports_per_receiver=8)
    assert np.array_equal(result.proposal, result.selected)
    assert result.edge_change_rate == 0.0
