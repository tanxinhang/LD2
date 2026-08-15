"""Cardinality-agnostic features for feasible local-move ranking."""

from __future__ import annotations

import numpy as np

from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    local_move_ranking_key,
)


FEATURE_NAMES = (
    "kind_n1",
    "kind_n2",
    "kind_n3",
    "kind_n5",
    "affected_target_fraction",
    "changed_edge_fraction",
    "role_change_fraction",
    "owner_change_fraction",
    "current_log_min",
    "current_log_mean",
    "current_log_max",
    "current_log_std",
    "affected_current_log_min",
    "affected_current_log_mean",
    "affected_current_log_max",
    "delta_signed_log_min",
    "delta_signed_log_mean",
    "delta_signed_log_max",
    "delta_signed_log_sum",
    "added_edge_fraction",
    "added_log_mean",
    "added_log_max",
    "added_log_sum",
    "removed_edge_fraction",
    "removed_log_mean",
    "removed_log_max",
    "removed_log_sum",
    "scarcity_signed_log",
    "total_gain_signed_log",
)


def _signed_log(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value, dtype=np.float64)
    transformed = np.sign(array) * np.log1p(np.abs(array))
    if transformed.ndim == 0:
        return float(transformed)
    return transformed


def _positive_log_stats(values: np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return 0.0, 0.0, 0.0
    logged = np.log1p(np.maximum(array, 0.0))
    return (
        float(np.mean(logged)),
        float(np.max(logged)),
        float(np.log1p(np.sum(np.maximum(array, 0.0)))),
    )


def local_move_features(
    selected: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    move: LocalMove,
    edge_value: np.ndarray,
) -> np.ndarray:
    """Build a fixed-width, permutation-invariant move descriptor.

    The descriptor uses the frozen local edge estimates and structural deltas,
    but deliberately excludes the post-move global minimum and exact
    lexicographic key used by the verifier.
    """
    current = np.asarray(selected, dtype=bool)
    proposal = np.asarray(move.selected, dtype=bool)
    value = np.asarray(edge_value, dtype=np.float64)
    current_role = np.asarray(role, dtype=np.int8)
    current_owner = np.asarray(owner, dtype=np.int64)
    if not (current.shape == proposal.shape == value.shape):
        raise ValueError("selected, move and edge_value shapes must match")
    K, _, Q = current.shape
    if current_role.shape != (K,) or current_owner.shape != (Q,):
        raise ValueError("role/owner shapes do not match selected")
    positive_value = np.maximum(value, 0.0)
    current_target = np.sum(
        np.where(current, positive_value, 0.0), axis=(0, 1))
    proposal_target = np.sum(
        np.where(proposal, positive_value, 0.0), axis=(0, 1))
    delta = proposal_target - current_target
    changed = current ^ proposal
    owner_changed = current_owner != np.asarray(move.owner, dtype=np.int64)
    affected = np.any(changed, axis=(0, 1)) | owner_changed
    current_log = np.log1p(np.maximum(current_target, 0.0))
    affected_current = current_log[affected]
    affected_delta = np.asarray(_signed_log(delta[affected]), dtype=np.float64)
    added = positive_value[proposal & ~current]
    removed = positive_value[current & ~proposal]
    added_stats = _positive_log_stats(added)
    removed_stats = _positive_log_stats(removed)
    scarcity = local_move_ranking_key(
        current, move, value, method="scarcity")[0]
    total_gain = float(np.sum(delta))
    kind = [float(move.kind == name) for name in ("N1", "N2", "N3", "N5")]
    changed_scale = float(max(K * max(Q, 1), 1))
    features = np.asarray([
        *kind,
        float(np.count_nonzero(affected)) / float(max(Q, 1)),
        float(np.count_nonzero(changed)) / changed_scale,
        float(np.count_nonzero(
            current_role != np.asarray(move.role, dtype=np.int8)))
        / float(max(K, 1)),
        float(np.count_nonzero(owner_changed)) / float(max(Q, 1)),
        float(np.min(current_log)) if Q else 0.0,
        float(np.mean(current_log)) if Q else 0.0,
        float(np.max(current_log)) if Q else 0.0,
        float(np.std(current_log)) if Q else 0.0,
        float(np.min(affected_current)) if affected_current.size else 0.0,
        float(np.mean(affected_current)) if affected_current.size else 0.0,
        float(np.max(affected_current)) if affected_current.size else 0.0,
        float(np.min(affected_delta)) if affected_delta.size else 0.0,
        float(np.mean(affected_delta)) if affected_delta.size else 0.0,
        float(np.max(affected_delta)) if affected_delta.size else 0.0,
        float(np.sum(affected_delta)) if affected_delta.size else 0.0,
        float(added.size) / changed_scale,
        *added_stats,
        float(removed.size) / changed_scale,
        *removed_stats,
        float(_signed_log(float(scarcity))),
        float(_signed_log(total_gain)),
    ], dtype=np.float32)
    if features.shape != (len(FEATURE_NAMES),):
        raise AssertionError("local move feature width drifted")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("local move features contain NaN/Inf")
    return features

