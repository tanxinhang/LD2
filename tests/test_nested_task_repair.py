import numpy as np

from uav_isac.coordination.nested_task_repair import (
    dependency_closed_block,
    nested_repair_blocks,
    task_violation_subgradient_weights,
)


def test_task_subgradient_selects_the_maximally_violated_branch():
    pd = np.array([0.5, 0.7, 0.9, 0.9])
    weight = task_violation_subgradient_weights(pd, (0.6, 0.6, 0.6, 2))
    np.testing.assert_allclose(weight, [1.0, 0.0, 0.0, 0.0])


def test_dependency_closure_propagates_only_changed_role_incidence():
    selected = np.zeros((4, 4, 3), dtype=bool)
    selected[0, 2, 0] = True
    selected[1, 2, 1] = True
    selected[1, 3, 2] = True
    uavs, targets = dependency_closed_block(
        selected, np.array([2, 2, 3]), {0}, {0}, {0})
    assert uavs == (0, 2)
    assert targets == (0,)


def test_nested_blocks_are_monotone_and_end_at_full_problem():
    gain = np.ones((4, 4, 3))
    selected = np.zeros_like(gain, dtype=bool)
    blocks = nested_repair_blocks(
        gain, np.ones(4), selected, np.array([1, 2, 3]),
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([0, 0, 1, 1], dtype=bool),
        np.array([0.4, 0.6, 0.8]), (0.6, 0.7, 0.8, 2))
    for left, right in zip(blocks, blocks[1:]):
        assert set(left.uavs) <= set(right.uavs)
        assert set(left.targets) <= set(right.targets)
    assert blocks[-1].uavs == (0, 1, 2, 3)
    assert blocks[-1].targets == (0, 1, 2)


def test_role_compatible_edge_does_not_expand_all_incident_targets():
    gain = np.ones((4, 4, 3))
    selected = np.zeros_like(gain, dtype=bool)
    selected[0, 2, :] = True
    blocks = nested_repair_blocks(
        gain, np.ones(4), selected, np.array([2, 2, 2]),
        np.array([1, 0, 0, 0], dtype=bool),
        np.array([0, 0, 1, 1], dtype=bool),
        np.array([0.4, 0.8, 0.9]), (0.6, 0.6, 0.6, 1))
    assert len(blocks[0].targets) == 1


def test_unknown_nested_ranking_mode_is_rejected():
    with np.testing.assert_raises(ValueError):
        nested_repair_blocks(
            np.ones((2, 2, 1)), np.ones(2),
            np.zeros((2, 2, 1), dtype=bool), np.array([1]),
            np.array([1, 0], dtype=bool), np.array([0, 1], dtype=bool),
            np.array([0.5]), (0.6, 0.6, 0.6, 1), ranking_mode="unknown")
