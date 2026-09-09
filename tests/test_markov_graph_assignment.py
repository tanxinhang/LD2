from types import SimpleNamespace
import copy

import numpy as np

from uav_isac.prediction.markov_graph_assignment import (
    propose_physics_knn_assignment,
    propose_physics_knn_local_search,
)
from uav_isac.physical.deflection import DeflectionComputer
from uav_isac.prediction.markov_physical_assignment import (
    MarkovPhysicalAssignmentModel,
)


class _GraphFixture:
    K = 3
    Q = 3

    def __init__(self, edge_value, joint_penalty=None):
        self.edge_value = np.asarray(edge_value, dtype=np.float64)
        self.joint_penalty = joint_penalty or {}
        self.uav = np.array([
            [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [20.0, 0.0, 10.0]])
        # Each desired cycle edge is among the two nearest UAVs.
        self.target = np.array([
            [18.0, 0.0, 0.0], [2.0, 0.0, 0.0], [11.0, 0.0, 0.0]])

    def unpack_state(self, state):
        return self.uav, np.zeros_like(self.uav), self.target, np.zeros_like(self.target)

    def transition(self, _state, assignment, _step):
        return np.asarray(assignment, dtype=np.float64)

    def evaluate(self, state, *, stage_index=None):
        assignment = np.asarray(state, dtype=np.int64)
        value = float(np.sum(self.edge_value[np.arange(self.K), assignment]))
        value -= float(self.joint_penalty.get(tuple(assignment), 0.0))
        return SimpleNamespace(cost=-value)


def test_physics_knn_graph_finds_global_cycle_from_edge_marginals():
    value = np.array([
        [0.0, 3.0, 0.0], [0.0, 0.0, 4.0], [5.0, 0.0, 0.0],
    ])
    model = _GraphFixture(value)
    proposal = propose_physics_knn_assignment(
        model, np.zeros(1), np.array([0, 1, 2]),
        stage_index=0, k_neighbors=2)
    np.testing.assert_array_equal(proposal.raw_assignment, [1, 2, 0])
    np.testing.assert_array_equal(proposal.assignment, [1, 2, 0])
    assert proposal.accepted
    assert proposal.proposal_cost < proposal.incumbent_cost
    assert proposal.evaluated_edges <= 3 * 2 + 3


def test_joint_physical_rescore_rejects_harmful_additivity_error():
    value = np.array([
        [0.0, 3.0, 0.0], [0.0, 0.0, 4.0], [5.0, 0.0, 0.0],
    ])
    model = _GraphFixture(value, joint_penalty={(1, 2, 0): 20.0})
    incumbent = np.array([0, 1, 2])
    proposal = propose_physics_knn_assignment(
        model, np.zeros(1), incumbent, stage_index=0, k_neighbors=2)
    np.testing.assert_array_equal(proposal.raw_assignment, [1, 2, 0])
    np.testing.assert_array_equal(proposal.assignment, incumbent)
    assert not proposal.accepted


def test_knn_graph_is_deterministic_and_preserves_incumbent_edges():
    model = _GraphFixture(np.eye(3))
    incumbent = np.array([0, 1, 2])
    first = propose_physics_knn_assignment(
        model, np.zeros(1), incumbent, stage_index=0, k_neighbors=1)
    second = propose_physics_knn_assignment(
        model, np.zeros(1), incumbent, stage_index=0, k_neighbors=1)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    assert np.all(first.adjacency[np.arange(3), incumbent])
    assert first.evaluated_edges <= 6


def test_every_finite_graph_edge_is_scored_by_a_count_preserving_exchange():
    class RecordingFixture(_GraphFixture):
        def __init__(self):
            super().__init__(np.eye(3))
            self.actions = []

        def transition(self, state, assignment, step):
            self.actions.append(np.asarray(assignment, dtype=np.int64).copy())
            return super().transition(state, assignment, step)

    model = RecordingFixture()
    propose_physics_knn_assignment(
        model, np.zeros(1), np.array([0, 1, 2]),
        stage_index=0, k_neighbors=2)
    for action in model.actions:
        np.testing.assert_array_equal(
            np.bincount(action, minlength=3), np.ones(3, dtype=np.int64))


def test_knn_proposal_runs_on_expected_physics_without_advancing_rng():
    dc = DeflectionComputer(
        fc=2.8e10, delta_f=1.5625e4, T_sym=6.4e-5, M=64, N=16,
        kT=4.0e-21, B=1.0e6, NF_dB=4.0,
        P_sense=0.0251, P_report=0.25, ric_K=6.0,
        rcs=1.0, g_min=0.5, rng=np.random.default_rng(91),
        g_tx_dBi=16.0, g_rx_dBi=16.0, use_los_prob=True,
        use_swerling=True, use_report_link=True, dd_gain_mode="continuous",
    )
    model = MarkovPhysicalAssignmentModel(
        dc, ((0, 1, 0), (0, 1, 1)), np.array([0.0251, 0.0]),
        np.array([0, 1]), np.array([500.0, 500.0, 0.0]),
        num_targets=2, dt_s=0.1, movement_step_m=2.5,
        area_size_m=(1000.0, 1000.0), false_alarm_probability=0.01,
    )
    state = model.pack_state(
        np.array([[100.0, 100.0, 100.0], [800.0, 800.0, 100.0]]),
        np.zeros((2, 3)),
        np.array([[300.0, 350.0, 0.0], [700.0, 650.0, 0.0]]),
        np.array([[3.0, 1.0, 0.0], [-2.0, -1.0, 0.0]]),
    )
    before = copy.deepcopy(dc.rng.bit_generator.state)
    proposal = propose_physics_knn_assignment(
        model, state, np.array([0, 1]), stage_index=0, k_neighbors=2)
    after = copy.deepcopy(dc.rng.bit_generator.state)
    assert before == after
    assert sorted(proposal.assignment.tolist()) == [0, 1]
    assert np.isfinite(proposal.incumbent_cost)
    assert np.isfinite(proposal.proposal_cost)


def test_knn_local_search_accepts_only_exact_improvements_within_budget():
    value = np.array([
        [0.0, 3.0, 0.0], [0.0, 0.0, 4.0], [5.0, 0.0, 0.0],
    ])
    model = _GraphFixture(value)
    proposal = propose_physics_knn_local_search(
        model, np.zeros(1), np.array([0, 1, 2]),
        stage_index=0, k_neighbors=2, evaluation_budget=6,
        max_rounds=3)
    assert proposal.proposal_cost < proposal.incumbent_cost
    assert proposal.evaluated_assignments <= 6
    assert proposal.accepted_swaps
    np.testing.assert_array_equal(
        np.bincount(proposal.assignment, minlength=3),
        np.ones(3, dtype=np.int64))


def test_knn_local_search_returns_incumbent_when_no_swap_improves():
    model = _GraphFixture(np.eye(3))
    incumbent = np.array([0, 1, 2])
    proposal = propose_physics_knn_local_search(
        model, np.zeros(1), incumbent,
        stage_index=0, k_neighbors=2, evaluation_budget=6)
    np.testing.assert_array_equal(proposal.assignment, incumbent)
    assert proposal.accepted_swaps == ()
    assert proposal.proposal_cost == proposal.incumbent_cost


def test_knn_local_search_validates_target_priority():
    model = _GraphFixture(np.eye(3))
    with np.testing.assert_raises_regex(ValueError, "target_priority"):
        propose_physics_knn_local_search(
            model, np.zeros(1), np.array([0, 1, 2]),
            stage_index=0, target_priority=np.array([1.0, 0.0, 1.0]))
