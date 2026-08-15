"""Unit tests for the price-justified structure repair (L2 layer)."""

import numpy as np
import pytest

from uav_isac.coordination.priced_structure import (
    priced_structure_repair,
    structure_from_rx,
)


def _random_coefficient(K, Q, seed):
    rng = np.random.default_rng(seed)
    coeff = rng.uniform(0.1, 3.0, size=(K, K, Q))
    for i in range(K):
        coeff[i, i, :] = 0.0  # no self-pair
    return coeff


def test_structure_enforces_role_partition():
    """Owner (RX) and transmitter (TX) sets must be disjoint (half-duplex)."""
    K, Q = 8, 8
    coeff = _random_coefficient(K, Q, 0)
    budget = np.full(K, 0.7)
    gain, owner = priced_structure_repair(coeff, budget, target_pair_limit=3)
    tx_set = set(np.where(gain.sum(axis=1) > 1e-12)[0].tolist())
    rx_set = set(int(o) for o in owner if o >= 0)
    assert tx_set.isdisjoint(rx_set)
    assert len(tx_set) >= 1, "need at least one transmitter"
    assert len(rx_set) >= 1, "need at least one receiver/owner"


def test_repair_never_worse_than_single_rx():
    """The greedy returns a structure at least as good as the best single RX
    under the SAME power-sharing-aware objective (max-min LP value)."""
    K, Q = 8, 8
    coeff = _random_coefficient(K, Q, 1)
    budget = np.full(K, 0.7)
    from uav_isac.coordination.maxmin_power import (
        solve_fixed_structure_maxmin_power_lp)
    gain, owner = priced_structure_repair(coeff, budget, target_pair_limit=3)
    got = float(solve_fixed_structure_maxmin_power_lp(gain, budget).worst_deflection)
    best_single = -np.inf
    for j in range(K):
        g, _ = structure_from_rx(coeff, budget, {j}, 3)
        v = float(solve_fixed_structure_maxmin_power_lp(g, budget).worst_deflection)
        best_single = max(best_single, v)
    assert got >= best_single - 1e-9


def test_owner_is_best_receiver_for_its_targets():
    """With receiver 3 forced into the RX set, it owns the target it dominates."""
    K, Q = 8, 8
    coeff = _random_coefficient(K, Q, 2)
    coeff[:, 3, 0] *= 10.0  # receiver 3 dominates target 0
    budget = np.full(K, 0.7)
    _, owner = structure_from_rx(coeff, budget, {3}, 3)
    assert owner[0] == 3


def test_budget_matters():
    """A zero-budget UAV should never be selected as a transmitter."""
    K, Q = 8, 8
    coeff = _random_coefficient(K, Q, 3)
    budget = np.full(K, 0.7)
    budget[2] = 0.0
    gain, _ = priced_structure_repair(coeff, budget, target_pair_limit=3)
    assert gain[2].sum() <= 1e-12
