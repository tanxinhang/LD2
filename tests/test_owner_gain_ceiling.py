"""Unit tests for the shared owner-gain ceiling primitive (P2-2 first sub-step)."""

import numpy as np
import pytest

from uav_isac.coordination.owner_gain_ceiling import owner_gain_ceiling


def _reference_topk_sum(row, budget, pair_limit):
    """Byte-equivalent reference: scale row by budget, stable sort, keep last
    ``pair_limit`` values, sum — the exact inline formula that used to live in
    ``owner_bid_transport.owner_capacity_bid_envelope``."""
    scaled = np.asarray(row, dtype=np.float64) * float(budget)
    k = min(int(pair_limit), scaled.size)
    if k < 1:
        return 0.0
    return float(np.sum(np.sort(scaled)[-k:]))


def test_ceiling_matches_reference_fixed():
    row = np.array([0.1, 0.5, 0.2, 0.8, 0.3], dtype=np.float64)
    for budget in (0.0, 0.5, 1.0, 2.5):
        for limit in (1, 3, 5, 10):
            assert owner_gain_ceiling(
                row, budget_w=budget, pair_limit=limit
            ) == _reference_topk_sum(row, budget, limit)


def test_ceiling_matches_reference_random():
    rng = np.random.default_rng(7)
    for _ in range(60):
        row = rng.random(9)
        budget = float(rng.random())
        limit = int(rng.integers(1, 12))
        assert owner_gain_ceiling(
            row, budget_w=budget, pair_limit=limit
        ) == _reference_topk_sum(row, budget, limit)


def test_ceiling_edge_cases():
    row = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    # pair_limit 1 -> single max scaled; pair_limit >= size -> scaled sum.
    assert owner_gain_ceiling(row, budget_w=2.0, pair_limit=1) == 6.0
    assert owner_gain_ceiling(row, budget_w=2.0, pair_limit=3) == 12.0
    # zero budget -> zero.
    assert owner_gain_ceiling(row, budget_w=0.0, pair_limit=1) == 0.0
    # empty row -> zero.
    assert owner_gain_ceiling(np.empty(0), budget_w=1.0, pair_limit=1) == 0.0
    # pair_limit < 1 -> zero.
    assert owner_gain_ceiling(row, budget_w=1.0, pair_limit=0) == 0.0


def test_ceiling_matches_reference_vector_budget():
    # envelope-converged path: budget_w is a (K,) per-transmitter vector.
    row = np.array([0.1, 0.5, 0.2, 0.8, 0.3], dtype=np.float64)
    budget_vec = np.array([0.25, 1.0, 0.5, 2.0, 0.75], dtype=np.float64)
    for limit in (1, 2, 3, 5):
        scaled = row * budget_vec
        k = min(limit, scaled.size)
        expected = 0.0 if k < 1 else float(np.sum(np.sort(scaled)[-k:]))
        assert owner_gain_ceiling(
            row, budget_w=budget_vec, pair_limit=limit) == expected


def test_ceiling_vector_budget_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="vector must match gain_row"):
        owner_gain_ceiling(
            np.array([1.0, 2.0]),
            budget_w=np.array([1.0, 2.0, 3.0]),
            pair_limit=1)


def test_ceiling_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="finite and >= 0"):
        owner_gain_ceiling(np.array([1.0, -1.0]), budget_w=1.0, pair_limit=1)
    with pytest.raises(ValueError, match="must be 1-D"):
        owner_gain_ceiling(np.zeros((2, 2)), budget_w=1.0, pair_limit=1)
    with pytest.raises(ValueError, match="budget_w must be finite and >= 0"):
        owner_gain_ceiling(np.array([1.0]), budget_w=-1.0, pair_limit=1)
    with pytest.raises(ValueError, match="budget_w must be finite and >= 0"):
        owner_gain_ceiling(
            np.array([1.0, 2.0]),
            budget_w=np.array([np.nan, 1.0]),
            pair_limit=1)