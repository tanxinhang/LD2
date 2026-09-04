"""Tests for the gap-coverage movement (three-phase policy, Phase A/B).

The geometric objective asks for a Tx and Rx endpoint within the configured
design radius; it is not itself a P_D guarantee.  The deterministic
assignment must give recovery responsibility to uncovered targets, respect
per-node capacity, be
reproducible from identical public views, and never leave an uncovered target
without a responsibility when a suitable endpoint exists.
"""

import numpy as np
import pytest

from uav_isac.coordination.hyperedge import (
    gap_coverage_bottleneck_assignment,
)


def _geometry(seed, K, Q, area=1386.0):
    rng = np.random.default_rng(seed)
    nodes = rng.uniform(0.0, area, (K, 2))
    targets = rng.uniform(0.0, area, (Q, 2))
    return nodes, targets


def test_every_uncovered_target_gets_missing_role():
    K, Q = 12, 12
    nodes, targets = _geometry(7, K, Q)
    roles = np.arange(K) % 2 == 0
    r_crit = 320.0
    tx, rx, uncovered = gap_coverage_bottleneck_assignment(
        nodes, targets, roles, critical_radius_m=r_crit, capacity=2)
    # For every uncovered target, a Tx and/or Rx responsibility exists.
    for q in np.flatnonzero(uncovered):
        nearest_tx = np.linalg.norm(nodes[roles] - targets[q], axis=1).min()
        nearest_rx = np.linalg.norm(nodes[~roles] - targets[q], axis=1).min()
        if nearest_tx > r_crit:
            assert tx[:, q].sum() >= 1
        if nearest_rx > r_crit:
            assert rx[:, q].sum() >= 1


def test_capacity_respected():
    K, Q = 12, 12
    nodes, targets = _geometry(11, K, Q)
    roles = np.arange(K) % 2 == 0
    tx, rx, _u = gap_coverage_bottleneck_assignment(
        nodes, targets, roles, critical_radius_m=320.0, capacity=2)
    assert int((tx + rx).sum(axis=1).max()) <= 2


def test_deterministic_reconstruction():
    K, Q = 12, 12
    nodes, targets = _geometry(42, K, Q)
    roles = np.arange(K) % 2 == 0
    tx_a, rx_a, u_a = gap_coverage_bottleneck_assignment(
        nodes, targets, roles, critical_radius_m=320.0, capacity=2)
    tx_b, rx_b, u_b = gap_coverage_bottleneck_assignment(
        nodes, targets, roles, critical_radius_m=320.0, capacity=2)
    assert np.array_equal(tx_a, tx_b)
    assert np.array_equal(rx_a, rx_b)
    assert np.array_equal(u_a, u_b)


def test_role_exclusivity():
    K, Q = 12, 12
    nodes, targets = _geometry(3, K, Q)
    roles = np.arange(K) % 2 == 0
    tx, rx, _u = gap_coverage_bottleneck_assignment(
        nodes, targets, roles, critical_radius_m=320.0, capacity=2)
    assert not tx[~roles].any()
    assert not rx[roles].any()


def test_all_within_critical_radius_is_covered():
    # Compact geometry: everything inside the critical radius -> no uncovered.
    K, Q = 6, 6
    nodes = np.array([[0., 0.], [100., 0.], [0., 100.],
                      [100., 100.], [50., 200.], [200., 50.]])
    targets = np.array([[60., 60.], [120., 120.], [40., 140.],
                        [140., 40.], [80., 180.], [180., 80.]])
    roles = np.arange(K) % 2 == 0
    tx, rx, uncovered = gap_coverage_bottleneck_assignment(
        nodes, targets, roles, critical_radius_m=320.0, capacity=2)
    assert not np.any(uncovered)


def test_deficit_priority_preserves_coverage_invariant():
    # With K=Q=6 and capacity 2 the slot budget exactly matches the 12
    # Tx/Rx needs; the deficit ordering must not break the coverage
    # invariant (every target keeps a Tx and an Rx responsibility).
    K, Q = 6, 6
    rng = np.random.default_rng(123)
    nodes = rng.uniform(0.0, 1386.0, (K, 2))
    targets = rng.uniform(0.0, 1386.0, (Q, 2))
    roles = np.arange(K) % 2 == 0
    deficit = np.zeros(Q)
    deficit[0] = 1.0   # weakest target
    deficit[1] = 0.0   # saturated target
    tx, rx, _u = gap_coverage_bottleneck_assignment(
        nodes, targets, roles, critical_radius_m=320.0, capacity=2,
        target_deficit=deficit)
    assert np.all(tx.sum(axis=0) >= 1)
    assert np.all(rx.sum(axis=0) >= 1)
    # The weakest target keeps both roles when the budget is tight.
    assert tx[:, 0].sum() + rx[:, 0].sum() >= 2


def test_invalid_inputs():
    nodes, targets = _geometry(1, 12, 12)
    roles = np.arange(12) % 2 == 0
    with pytest.raises(ValueError):
        gap_coverage_bottleneck_assignment(
            nodes[:, :1], targets, roles,
            critical_radius_m=320.0, capacity=2)
    with pytest.raises(ValueError):
        gap_coverage_bottleneck_assignment(
            nodes, targets, np.ones(12, dtype=bool),
            critical_radius_m=320.0, capacity=2)
    with pytest.raises(ValueError):
        gap_coverage_bottleneck_assignment(
            nodes, targets, np.zeros(12, dtype=bool),
            critical_radius_m=320.0, capacity=2)


def test_uncovered_metric_consistent_with_geometry():
    K, Q = 8, 8
    rng = np.random.default_rng(99)
    nodes = rng.uniform(0.0, 1386.0, (K, 2))
    targets = rng.uniform(0.0, 1386.0, (Q, 2))
    roles = np.arange(K) % 2 == 0
    for r_crit in (200.0, 320.0, 500.0):
        _tx, _rx, uncovered = gap_coverage_bottleneck_assignment(
            nodes, targets, roles, critical_radius_m=r_crit, capacity=2)
        nearest_tx = np.min(np.linalg.norm(
            nodes[roles, None, :] - targets[None, :, :], axis=-1), axis=0)
        nearest_rx = np.min(np.linalg.norm(
            nodes[~roles, None, :] - targets[None, :, :], axis=-1), axis=0)
        expected = (nearest_tx > r_crit) | (nearest_rx > r_crit)
        assert np.array_equal(uncovered, expected)
