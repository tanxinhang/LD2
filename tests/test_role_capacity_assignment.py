"""Tests for the role-capacitated Tx/Rx responsibility assignment (audit P0 #2).

The assignment is the pure, public-view reconstruction used by the
role-capacity distributed movement mode.  Every target must receive exactly one
Tx and one Rx geometric responsibility, every node at most its role capacity,
and identical inputs must reproduce identical responsibilities (no optimizer
certificate is exchanged).
"""

from itertools import combinations

import numpy as np
import pytest

from uav_isac.coordination.hyperedge import (
    role_capacity_bottleneck_assignment,
)


def _random_geometry(seed, K, Q, area=1386.0):
    rng = np.random.default_rng(seed)
    nodes = rng.uniform(0.0, area, (K, 2))
    targets = rng.uniform(0.0, area, (Q, 2))
    return nodes, targets


def test_every_target_gets_one_tx_and_one_rx():
    K, Q = 12, 12
    nodes, targets = _random_geometry(7, K, Q)
    roles = np.arange(K) % 2 == 0
    tx, rx, _cost, feasible = role_capacity_bottleneck_assignment(
        nodes, targets, roles, height_m=20.0, capacity=2)
    assert feasible
    np.testing.assert_array_equal(tx.sum(axis=0), np.ones(Q, dtype=np.int8))
    np.testing.assert_array_equal(rx.sum(axis=0), np.ones(Q, dtype=np.int8))


def test_per_node_capacity_is_respected():
    K, Q = 12, 12
    nodes, targets = _random_geometry(11, K, Q)
    roles = np.arange(K) % 2 == 0
    tx, rx, _cost, feasible = role_capacity_bottleneck_assignment(
        nodes, targets, roles, height_m=20.0, capacity=2)
    assert feasible
    assert int(tx.sum(axis=1).max()) <= 2
    assert int(rx.sum(axis=1).max()) <= 2
    # With K == Q and a 6/6 role split every node serves exactly two targets.
    np.testing.assert_array_equal(
        tx.sum(axis=1)[roles], np.full(int(roles.sum()), 2, dtype=np.int64))
    np.testing.assert_array_equal(
        rx.sum(axis=1)[~roles], np.full(int((~roles).sum()), 2, dtype=np.int64))


def test_role_exclusivity():
    K, Q = 12, 12
    nodes, targets = _random_geometry(3, K, Q)
    roles = np.arange(K) % 2 == 0
    tx, rx, _cost, feasible = role_capacity_bottleneck_assignment(
        nodes, targets, roles, height_m=20.0, capacity=2)
    assert feasible
    assert not tx[~roles].any()
    assert not rx[roles].any()


def test_deterministic_reconstruction():
    K, Q = 12, 12
    nodes, targets = _random_geometry(42, K, Q)
    roles = np.arange(K) % 2 == 0
    tx_a, rx_a, cost_a, feasible_a = role_capacity_bottleneck_assignment(
        nodes, targets, roles, height_m=20.0, capacity=2)
    tx_b, rx_b, cost_b, feasible_b = role_capacity_bottleneck_assignment(
        nodes, targets, roles, height_m=20.0, capacity=2)
    assert feasible_a and feasible_b
    assert np.array_equal(tx_a, tx_b)
    assert np.array_equal(rx_a, rx_b)
    assert cost_a == cost_b


def test_asymmetric_roles_capacity_tuple():
    K, Q = 10, 12
    nodes, targets = _random_geometry(5, K, Q)
    roles = np.arange(K) % 2 == 0  # 5 Tx / 5 Rx, capacity 3 each
    tx, rx, _cost, feasible = role_capacity_bottleneck_assignment(
        nodes, targets, roles, height_m=20.0, capacity=(3, 3))
    assert feasible
    np.testing.assert_array_equal(tx.sum(axis=0), np.ones(Q, dtype=np.int8))
    np.testing.assert_array_equal(rx.sum(axis=0), np.ones(Q, dtype=np.int8))
    assert int(tx.sum(axis=1).max()) <= 3
    assert int(rx.sum(axis=1).max()) <= 3


def test_insufficient_role_capacity_returns_explicit_infeasible_result():
    nodes, targets = _random_geometry(17, 4, 5, area=400.0)
    roles = np.asarray([True, True, False, False])

    tx, rx, cost, feasible = role_capacity_bottleneck_assignment(
        nodes, targets, roles, height_m=20.0, capacity=2)

    assert feasible is False
    assert np.isinf(cost)
    assert not tx.any()
    assert not rx.any()


def test_bottleneck_optimality_small_case():
    """Exhaustive check that the returned worst range equals the true min-max.

    K = Q = 4, roles 2/2, capacity 2: each role assigns its two nodes to the
    four targets with every node taking exactly two targets.  Enumerate all
    3 partitions of 4 targets into two 2-sets per node and compare.
    """
    nodes, targets = _random_geometry(99, 4, 4, area=400.0)
    roles = np.array([True, False, True, False])
    tx, rx, cost, feasible = role_capacity_bottleneck_assignment(
        nodes, targets, roles, height_m=10.0, capacity=2)
    assert feasible
    height = 10.0
    range_sq = height * height + np.sum(
        (nodes[:, None, :] - targets[None, :, :]) ** 2, axis=-1)

    def role_minmax(node_list):
        best = float('inf')
        node_list = list(node_list)
        for first_pair in combinations(range(4), 2):
            second_pair = tuple(q for q in range(4) if q not in first_pair)
            # node0 takes first_pair, node1 takes second_pair (and swapped)
            for (a, b) in [(first_pair, second_pair), (second_pair, first_pair)]:
                worst = max(
                    range_sq[node_list[0], a[0]],
                    range_sq[node_list[0], a[1]],
                    range_sq[node_list[1], b[0]],
                    range_sq[node_list[1], b[1]],
                )
                best = min(best, float(worst))
        return best

    expected = max(
        role_minmax(np.flatnonzero(roles)),
        role_minmax(np.flatnonzero(~roles)),
    )
    assert abs(cost - expected) < 1.0e-6
    # The executed responsibilities must be feasible at the returned cost.
    tx_worst = max(float(range_sq[int(n), int(q)])
                   for n, q in zip(*np.nonzero(tx)))
    rx_worst = max(float(range_sq[int(n), int(q)])
                   for n, q in zip(*np.nonzero(rx)))
    assert max(tx_worst, rx_worst) <= expected + 1.0e-9


def test_invalid_inputs():
    nodes, targets = _random_geometry(1, 12, 12)
    roles = np.arange(12) % 2 == 0
    with pytest.raises(ValueError):
        role_capacity_bottleneck_assignment(
            nodes[:, :1], targets, roles, height_m=20.0, capacity=2)
    with pytest.raises(ValueError):
        role_capacity_bottleneck_assignment(
            nodes, targets, np.ones(12, dtype=bool),
            height_m=20.0, capacity=2)
    with pytest.raises(ValueError):
        role_capacity_bottleneck_assignment(
            nodes, targets, np.zeros(12, dtype=bool),
            height_m=20.0, capacity=2)
