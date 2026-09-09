import numpy as np
import pytest

from uav_isac.prediction.markov_assignment import (
    assignment_switch_distance,
    solve_markov_assignment_path,
)


def test_switch_distance_is_normalized_hamming_distance():
    assert assignment_switch_distance([0, 1, 2, 3], [0, 2, 1, 3]) == 0.5


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
