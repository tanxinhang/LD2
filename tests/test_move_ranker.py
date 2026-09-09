"""Behavior tests for the local-move rankers (feature + learned scorer).

Audit 2026-09-04 §10 B1: ``coordination/local_move_ranker.py`` and
``coordination/learned_move_ranker.py`` had no dedicated tests (indirect
coverage only via ``test_safe_checkpoint_loading.py`` /
``test_local_exchange_oracle.py``).  These tests lock the feature contract:
fixed width, permutation-invariant descriptor, one-hot move kinds, and the
normalized MLP scorer.
"""

import numpy as np
import pytest
import torch

from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.local_move_ranker import (
    FEATURE_NAMES,
    local_move_features,
)
from uav_isac.coordination.learned_move_ranker import LocalMoveRanker


def _graph(K=4, Q=2):
    value = np.ones((K, K, Q), dtype=float)
    mask = np.ones_like(value, dtype=bool)
    mask[np.arange(K), np.arange(K), :] = False
    selected = mask.copy()
    selected[0, 1, 0] = True
    selected[1, 2, 1] = True
    selected[0, 3, 1] = True
    for i in range(K):
        for j in range(K):
            for q in range(Q):
                if mask[i, j, q]:
                    value[i, j, q] += 0.1 * i + 0.2 * j + 0.3 * q
    return value, selected


def _uniform_move(kind, selected, role, owner):
    return LocalMove(kind=kind, selected=selected.copy(),
                     role=role.copy(), owner=owner.copy())


def _kind_index(kind):
    # FEATURE_NAMES uses lowercase kinds ("kind_n1") while LocalMove kinds
    # are uppercase ("N1").
    return FEATURE_NAMES.index(f"kind_{kind.lower()}")


def test_feature_width_matches_declared_names():
    value, selected = _graph()
    role = np.arange(4, dtype=np.int64)
    owner = np.array([0, 3], dtype=np.int64)
    move = _uniform_move("N1", selected, role, owner)
    features = local_move_features(selected, role, owner, move, value)
    assert features.shape == (len(FEATURE_NAMES),)
    assert features.dtype == np.float32


def test_features_are_deterministic():
    value, selected = _graph()
    role = np.arange(4, dtype=np.int64)
    owner = np.array([0, 3], dtype=np.int64)
    move = _uniform_move("N1", selected, role, owner)
    first = local_move_features(selected, role, owner, move, value)
    second = local_move_features(selected, role, owner, move, value)
    assert np.array_equal(first, second)


def test_features_are_finite_for_valid_inputs():
    value, selected = _graph()
    role = np.arange(4, dtype=np.int64)
    owner = np.array([0, 3], dtype=np.int64)
    for kind in ("N1", "N2", "N3", "N5"):
        move = _uniform_move(kind, selected, role, owner)
        features = local_move_features(selected, role, owner, move, value)
        assert np.all(np.isfinite(features)), kind


def test_kind_is_one_hot_in_declared_order():
    value, selected = _graph()
    role = np.arange(4, dtype=np.int64)
    owner = np.array([0, 3], dtype=np.int64)
    for kind in ("N1", "N2", "N3", "N5"):
        move = _uniform_move(kind, selected, role, owner)
        features = local_move_features(selected, role, owner, move, value)
        hot = _kind_index(kind)
        assert features[hot] == 1.0
        for other in ("N1", "N2", "N3", "N5"):
            if other != kind:
                assert features[_kind_index(other)] == 0.0


def test_owner_change_fraction_reflects_owner_delta():
    value, selected = _graph()
    role = np.arange(4, dtype=np.int64)
    owner = np.array([0, 3], dtype=np.int64)

    def owner_fraction(move_owner):
        move = _uniform_move("N1", selected, role, move_owner)
        features = local_move_features(selected, role, owner, move, value)
        return features[FEATURE_NAMES.index("owner_change_fraction")]

    assert owner_fraction(owner) == 0.0
    changed = np.array([0, 2], dtype=np.int64)
    assert owner_fraction(changed) == pytest.approx(1.0 / 2.0)


def test_added_and_removed_edge_fractions_are_congruent():
    value, selected = _graph()
    role = np.arange(4, dtype=np.int64)
    owner = np.array([0, 3], dtype=np.int64)
    # N2-style move: drop two selected edges, add none.
    proposed = selected.copy()
    proposed[0, 1, 0] = False
    proposed[1, 2, 1] = False
    move = LocalMove(kind="N2", selected=proposed,
                     role=role.copy(), owner=owner.copy())
    features = local_move_features(selected, role, owner, move, value)
    added_frac = features[FEATURE_NAMES.index("added_edge_fraction")]
    removed_frac = features[FEATURE_NAMES.index("removed_edge_fraction")]
    changed_scale = 4 * 2  # K * Q
    assert added_frac == 0.0
    assert removed_frac == pytest.approx(2.0 / changed_scale)


def test_features_reject_shape_mismatch():
    value, selected = _graph()
    role = np.arange(4, dtype=np.int64)
    owner = np.array([0, 3], dtype=np.int64)
    bad = np.zeros((3, 3, 2), dtype=bool)
    move = _uniform_move("N1", bad, role, owner)
    with pytest.raises(ValueError, match="shapes must match"):
        local_move_features(selected, role, owner, move, value)


def test_features_reject_role_owner_shape_mismatch():
    value, selected = _graph()
    move = _uniform_move("N1", selected, np.arange(4), np.array([0, 3]))
    with pytest.raises(ValueError, match="role/owner shapes"):
        local_move_features(selected, np.arange(3), np.array([0, 1]),
                            move, value)
    with pytest.raises(ValueError, match="role/owner shapes"):
        local_move_features(selected, np.arange(4), np.array([0]),
                            move, value)


def test_ranker_rejects_normalization_width_mismatch():
    width = len(FEATURE_NAMES)
    with pytest.raises(ValueError, match="width mismatch"):
        LocalMoveRanker(np.zeros(width - 1), np.ones(width - 1))
    with pytest.raises(ValueError, match="width mismatch"):
        LocalMoveRanker(np.zeros(width), np.ones(width - 1))


def test_ranker_forward_returns_rank_and_positive_logits():
    width = len(FEATURE_NAMES)
    ranker = LocalMoveRanker(np.zeros(width), np.ones(width), hidden_dim=8)
    batch = torch.zeros((5, width))
    rank_score, positive_logit = ranker(batch)
    assert rank_score.shape == (5,)
    assert positive_logit.shape == (5,)
    assert torch.isfinite(rank_score).all()
    assert torch.isfinite(positive_logit).all()


def test_ranker_clamps_zero_std_and_stays_finite():
    width = len(FEATURE_NAMES)
    # All-zero normalization std must not divide by zero: the module clamps
    # std to >= 1e-4.
    ranker = LocalMoveRanker(np.zeros(width), np.zeros(width), hidden_dim=8)
    features = torch.randn((3, width))
    rank_score, positive_logit = ranker(features)
    assert torch.isfinite(rank_score).all()
    assert torch.isfinite(positive_logit).all()