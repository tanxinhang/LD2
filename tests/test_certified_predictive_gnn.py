import numpy as np
import pytest
import torch

from uav_isac.prediction.certified_gnn import (
    CertifiedBipartiteGNN,
    PREDICTIVE_CHECKPOINT_SCHEMA,
    PredictionPrefetchCache,
    append_causal_feature_residual,
    audit_predictive_checkpoint,
    augment_temporal_protocol_features,
    build_endpoint_features,
    require_compatible_predictive_checkpoint,
    validate_selected_prediction,
)


def test_predictive_checkpoint_audit_rejects_legacy_missing_heads():
    model = CertifiedBipartiteGNN(hidden_dim=8, message_rounds=1)
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith((
            "power_head.", "detection_head.", "detection_delta_head."))
    }
    payload = {
        "schema_version": "predictive-gnn-shadow/v0",
        "state_dict": legacy_state,
    }

    audit = audit_predictive_checkpoint(payload, model)

    assert audit["compatible"] is False
    assert audit["checkpoint_schema"] == "predictive-gnn-shadow/v0"
    assert "power_head.weight" in audit["missing_tensors"]
    with pytest.raises(ValueError, match="unsupported schema.*missing model"):
        require_compatible_predictive_checkpoint(payload, model)


def test_predictive_checkpoint_audit_accepts_current_exact_schema():
    model = CertifiedBipartiteGNN(hidden_dim=8, message_rounds=1)
    payload = {
        "schema_version": PREDICTIVE_CHECKPOINT_SCHEMA,
        "state_dict": model.state_dict(),
    }

    audit = require_compatible_predictive_checkpoint(payload, model)

    assert audit["compatible"] is True
    assert audit["issues"] == []


def test_predictive_gnn_shapes_and_node_permutation_equivariance():
    torch.manual_seed(7)
    batch, nodes, targets = 3, 5, 4
    model = CertifiedBipartiteGNN(hidden_dim=16, message_rounds=2).eval()
    features = torch.randn(batch, nodes, targets, 9)
    visible = torch.ones(batch, nodes, targets, dtype=torch.bool)
    permutation = torch.tensor([2, 4, 0, 3, 1])

    with torch.inference_mode():
        expected = model(features, visible)
        permuted = model(features[:, permutation], visible[:, permutation])

    torch.testing.assert_close(
        permuted.owner_logits, expected.owner_logits[:, permutation])
    torch.testing.assert_close(
        permuted.transmitter_logits,
        expected.transmitter_logits[:, permutation],
    )
    torch.testing.assert_close(
        permuted.dual_warm_start, expected.dual_warm_start)
    torch.testing.assert_close(permuted.risk_radius, expected.risk_radius)
    torch.testing.assert_close(
        permuted.detection_logits, expected.detection_logits)
    torch.testing.assert_close(
        permuted.detection_delta, expected.detection_delta)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(nodes)
    receiver = torch.tensor([[0, 1, 2, 3]]).expand(batch, -1)
    permuted_receiver = inverse[receiver]
    with torch.inference_mode():
        expected_conditional = model.conditional_transmitter_logits(
            expected, receiver)
        permuted_conditional = model.conditional_transmitter_logits(
            permuted, permuted_receiver)
    torch.testing.assert_close(
        permuted_conditional,
        expected_conditional[:, permutation],
    )


def test_endpoint_features_and_prefetch_contract():
    b, k, q = 2, 3, 4
    features = build_endpoint_features(
        np.zeros((b, k, q, 2)),
        np.zeros((b, k, q, 2)),
        np.ones((b, q, 3)),
        np.zeros((b, q, 3)),
        np.zeros((b, q)),
        np.zeros((b, q)),
        np.ones((b, k, q), dtype=bool),
        distance_scale_m=1000.0,
        velocity_scale_mps=50.0,
    )
    model = CertifiedBipartiteGNN(hidden_dim=8).eval()
    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(features),
            torch.ones((b, k, q), dtype=torch.bool),
        )
    cache = PredictionPrefetchCache()
    cache.publish(11, prediction)
    assert cache.consume(11) is None
    assert cache.consume(12) is not None
    assert cache.consume(12) is None


def test_structural_validator_fails_closed():
    valid, reasons = validate_selected_prediction(
        [(0, 1, 0), (2, 1, 0), (1, 0, 1)],
        num_uavs=3,
        num_targets=2,
        target_pair_limit=2,
    )
    assert valid
    assert reasons == ()

    valid, reasons = validate_selected_prediction(
        [(0, 1, 0), (2, 0, 0)],
        num_uavs=3,
        num_targets=2,
        target_pair_limit=1,
    )
    assert not valid
    assert "non_unique_owner" in reasons
    assert "incomplete_target_coverage" in reasons


def test_temporal_protocol_features_are_causal_and_fixed_shape():
    frames, viewers, nodes, targets = 2, 3, 4, 2
    base = np.zeros((frames, viewers, nodes, targets, 9), dtype=np.float32)
    edges = np.full((frames, 3, 3), -1, dtype=np.int32)
    mask = np.zeros((frames, 3), dtype=bool)
    edges[0, 0] = (1, 2, 0)
    mask[0, 0] = True
    tensor = augment_temporal_protocol_features(
        base, edges, mask, np.array([5, 6]),
        np.zeros((frames, viewers, nodes, targets)),
        np.zeros((frames, viewers, nodes, targets)),
        np.zeros((frames, viewers, nodes, targets)),
        np.zeros((frames, viewers, nodes, targets)),
        np.zeros((frames, viewers, targets)),
    )
    assert tensor.shape == (frames, viewers, nodes, targets, 18)
    assert np.all(tensor[0, :, 2, 0, 9] == 1.0)
    assert np.all(tensor[0, :, 1, 0, 10] == 1.0)
    assert np.all(tensor[0, ..., 11] == 1.0)
    assert np.all(tensor[1, ..., 11] == 0.0)


def test_boundary_residual_head_only_changes_boundary_frames():
    torch.manual_seed(13)
    model = CertifiedBipartiteGNN(
        feature_dim=18, hidden_dim=8, message_rounds=1,
        boundary_feature_index=11,
    ).eval()
    features = torch.randn(2, 3, 2, 18)
    features[..., 11] = 0.0
    visible = torch.ones(2, 3, 2, dtype=torch.bool)
    with torch.inference_mode():
        hold = model(features, visible)
        features[..., 11] = 1.0
        boundary = model(features, visible)
    assert not torch.equal(hold.owner_logits, boundary.owner_logits)
    assert not torch.equal(hold.power_logits, boundary.power_logits)


def test_causal_feature_residual_clears_invalid_episode_history():
    current = np.full((2, 1, 2, 1, 3), 5.0, dtype=np.float32)
    previous = np.full_like(current, 2.0)
    augmented = append_causal_feature_residual(
        current, previous, np.array([0.0, 1.0], dtype=np.float32))
    assert augmented.shape == (2, 1, 2, 1, 7)
    np.testing.assert_array_equal(augmented[0, ..., 3:6], 0.0)
    np.testing.assert_array_equal(augmented[0, ..., 6], 0.0)
    np.testing.assert_array_equal(augmented[1, ..., 3:6], 3.0)
    np.testing.assert_array_equal(augmented[1, ..., 6], 1.0)
