import numpy as np
import pytest

from uav_isac.prediction.markov_assignment import (
    assignment_switch_distance,
    build_target_priority_swap_neighborhood,
    solve_markov_assignment_path,
    solve_markov_assignment_scenario_tree,
)


def test_switch_distance_is_normalized_hamming_distance():
    assert assignment_switch_distance([0, 1, 2, 3], [0, 2, 1, 3]) == 0.5


def test_priority_swap_neighborhood_is_bounded_and_count_preserving():
    incumbent = np.array([0, 1, 2, 3])
    candidates = build_target_priority_swap_neighborhood(
        incumbent, num_targets=4, target_priority=np.array([2, 0]),
        max_candidates=5)
    np.testing.assert_array_equal(candidates[0], incumbent)
    assert candidates.shape == (5, 4)
    for candidate in candidates:
        np.testing.assert_array_equal(
            np.bincount(candidate, minlength=4), np.ones(4, dtype=np.int64))
    # The first non-incumbent candidates all reassign priority target 2.
    assert all(np.argmax(candidate == 2) != 2 for candidate in candidates[1:4])


def test_priority_swap_neighborhood_is_deterministic_and_deduplicated():
    kwargs = dict(
        incumbent_assignment=np.array([0, 0, 1, 2, 2]),
        num_targets=3,
        target_priority=np.array([2]),
        max_candidates=20,
    )
    first = build_target_priority_swap_neighborhood(**kwargs)
    second = build_target_priority_swap_neighborhood(**kwargs)
    np.testing.assert_array_equal(first, second)
    assert np.unique(first, axis=0).shape[0] == first.shape[0]
    baseline_counts = np.bincount(first[0], minlength=3)
    for candidate in first:
        np.testing.assert_array_equal(
            np.bincount(candidate, minlength=3), baseline_counts)


def test_k16_q16_priority_neighborhood_respects_research_compute_cap():
    incumbent = np.arange(16, dtype=np.int64)
    candidates = build_target_priority_swap_neighborhood(
        incumbent, num_targets=16,
        target_priority=np.array([15, 7, 3]), max_candidates=32)
    assert candidates.shape == (32, 16)
    np.testing.assert_array_equal(candidates[0], incumbent)
    assert np.unique(candidates, axis=0).shape[0] == 32


def test_zero_penalty_selects_each_stage_minimum():
    candidates = np.asarray([[0, 1], [1, 0]])
    costs = np.asarray([[0.0, 2.0], [3.0, 0.0], [0.0, 4.0]])
    plan = solve_markov_assignment_path(
        candidates, costs, candidates[0], switch_penalty=0.0)
    assert plan.state_indices == (0, 1, 0)
    assert plan.total_cost == 0.0
    assert plan.switching_cost == 0.0


def test_switch_penalty_rejects_transient_one_step_gain():
    candidates = np.asarray([[0, 1], [1, 0]])
    costs = np.asarray([[1.0, 0.6], [1.0, 1.0], [1.0, 1.0]])
    plan = solve_markov_assignment_path(
        candidates, costs, candidates[0], switch_penalty=0.5)
    assert plan.state_indices == (0, 0, 0)
    np.testing.assert_array_equal(plan.first_assignment, candidates[0])


def test_persistent_gain_pays_switch_cost_once():
    candidates = np.asarray([[0, 1], [1, 0]])
    costs = np.asarray([[1.0, 0.6], [1.0, 0.6], [1.0, 0.6]])
    plan = solve_markov_assignment_path(
        candidates, costs, candidates[0], switch_penalty=0.5)
    assert plan.state_indices == (1, 1, 1)
    assert plan.stage_cost == pytest.approx(1.8)
    assert plan.switching_cost == pytest.approx(0.5)
    assert plan.total_cost == pytest.approx(2.3)


def test_ties_are_resolved_by_stable_candidate_order():
    candidates = np.asarray([[0, 1], [1, 0]])
    costs = np.ones((2, 2))
    plan = solve_markov_assignment_path(
        candidates, costs, candidates[0], switch_penalty=0.0)
    assert plan.state_indices == (0, 0)


@pytest.mark.parametrize("penalty", [-1.0, float("nan")])
def test_invalid_switch_penalty_is_rejected(penalty):
    with pytest.raises(ValueError, match="switch_penalty"):
        solve_markov_assignment_path(
            np.asarray([[0, 1]]), np.asarray([[1.0]]), np.asarray([0, 1]),
            switch_penalty=penalty)


def test_scenario_tree_cost_depends_on_preceding_geometry_path():
    candidates = np.asarray([[0], [1]])

    def transition(state, assignment, _step):
        return state + float(assignment[0])

    def stage_cost(state, _assignment, _step):
        return float((3.0 - state[0]) ** 2)

    plan = solve_markov_assignment_scenario_tree(
        candidates,
        incumbent_assignment=np.asarray([0]),
        initial_state=np.asarray([0.0]),
        horizon_steps=3,
        transition=transition,
        stage_cost=stage_cost,
        switch_penalty=0.1,
        beam_width=8,
    )
    assert plan.action_indices == (1, 1, 1)
    np.testing.assert_allclose([state[0] for state in plan.states], [1, 2, 3])
    assert plan.stage_cost == pytest.approx(5.0)
    assert plan.switching_cost == pytest.approx(0.1)


def test_scenario_tree_is_deterministic_and_fails_on_invalid_callback_state():
    candidates = np.asarray([[0, 1], [1, 0]])
    kwargs = dict(
        candidate_assignments=candidates,
        incumbent_assignment=candidates[0],
        initial_state=np.zeros(2),
        horizon_steps=2,
        transition=lambda state, _assignment, _step: state,
        stage_cost=lambda _state, _assignment, _step: 1.0,
        switch_penalty=0.0,
        beam_width=4,
    )
    first = solve_markov_assignment_scenario_tree(**kwargs)
    second = solve_markov_assignment_scenario_tree(**kwargs)
    assert first.action_indices == second.action_indices == (0, 0)

    kwargs["transition"] = lambda _state, _assignment, _step: np.asarray([np.nan, 0.0])
    with pytest.raises(ValueError, match="transition"):
        solve_markov_assignment_scenario_tree(**kwargs)
