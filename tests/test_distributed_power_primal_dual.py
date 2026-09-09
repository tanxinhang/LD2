import numpy as np
import pytest

from uav_isac.optimization.distributed_power_primal_dual import (
    BoundedTemporalPowerController,
    FixedStructurePowerPrimalDual,
    project_masked_row_power_budget,
)
from tools.benchmark_distributed_power_primal_dual import run_benchmark


def test_masked_projection_enforces_structure_nonnegativity_and_row_budget():
    proposed = np.array([[2.0, -1.0, 1.0], [0.2, 0.7, 0.8]])
    mask = np.array([[True, False, True], [False, True, True]])
    projected = project_masked_row_power_budget(
        proposed, np.array([1.0, 0.5]), mask)
    np.testing.assert_allclose(projected.sum(axis=1), [1.0, 0.5])
    assert np.all(projected >= 0.0)
    assert np.all(projected[~mask] == 0.0)
    np.testing.assert_allclose(projected[0], [1.0, 0.0, 0.0])


def test_fixed_structure_primal_dual_converges_without_budget_violation():
    solver = FixedStructurePowerPrimalDual(
        np.eye(2), np.array([0.5, 0.5]), 1.0,
        np.eye(2, dtype=bool),
        power_cost_per_watt=0.01,
        quadratic_regularization=0.01,
    )
    result = solver.run(
        100,
        primal_tolerance=1.0e-4,
        stationarity_tolerance=1.0e-4,
        dual_tolerance=1.0e-4,
        convergence_patience=3,
    )
    assert result.converged
    assert result.iteration < 100
    assert result.primal_violation <= 1.0e-4
    assert result.power_budget_violation_w == 0.0
    assert result.consensus_residual == 0.0
    np.testing.assert_allclose(result.power_w, np.diag([0.5, 0.5]), atol=1e-4)


def test_shared_target_prices_move_only_supported_local_rows():
    gain = np.array([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0]])
    mask = gain > 0.0
    solver = FixedStructurePowerPrimalDual(
        gain, np.array([0.6, 0.8]), np.array([0.5, 0.5, 0.5]), mask,
        power_cost_per_watt=0.001,
        quadratic_regularization=0.01,
    )
    result = solver.run(
        500,
        primal_tolerance=2.0e-3,
        stationarity_tolerance=2.0e-3,
        dual_tolerance=2.0e-3,
        convergence_patience=3,
    )
    assert result.converged
    assert np.all(result.power_w[~mask] == 0.0)
    assert np.all(result.power_w.sum(axis=1) <= 0.5 + 1e-12)
    assert np.all(result.aggregate_target_contribution >= np.array([0.6, 0.8]) - 2e-3)
    np.testing.assert_allclose(
        result.aggregate_target_contribution,
        result.local_target_contribution.sum(axis=0),
    )


def test_infeasible_requirement_fails_convergence_but_keeps_hard_constraints():
    solver = FixedStructurePowerPrimalDual(
        np.ones((2, 1)), np.array([3.0]), np.ones(2),
        np.ones((2, 1), dtype=bool),
    )
    result = solver.run(50)
    assert not result.converged
    assert result.primal_violation == pytest.approx(1.0)
    assert result.power_budget_violation_w == 0.0
    np.testing.assert_allclose(result.power_w.sum(axis=1), np.ones(2))


def test_uav_and_target_permutations_are_equivariant():
    gain = np.array([[1.0, 0.2, 0.0], [0.1, 1.2, 0.8]])
    mask = gain > 0.0
    demand = np.array([0.4, 0.5, 0.3])
    budget = np.array([0.7, 0.8])
    base = FixedStructurePowerPrimalDual(gain, demand, budget, mask)
    base_result = base.run(1000, primal_tolerance=1e-4,
                           stationarity_tolerance=1e-4,
                           dual_tolerance=1e-4)
    row_order = np.array([1, 0])
    target_order = np.array([2, 0, 1])
    permuted = FixedStructurePowerPrimalDual(
        gain[row_order][:, target_order], demand[target_order],
        budget[row_order], mask[row_order][:, target_order])
    permuted_result = permuted.run(
        1000, primal_tolerance=1e-4,
        stationarity_tolerance=1e-4, dual_tolerance=1e-4)
    restored = np.empty_like(permuted_result.power_w)
    restored[row_order[:, None], target_order[None, :]] = (
        permuted_result.power_w)
    np.testing.assert_allclose(restored, base_result.power_w, atol=1e-8)


def test_checkpoint_resume_matches_uninterrupted_iterations():
    args = (
        np.array([[1.0, 0.4], [0.2, 1.0]]),
        np.array([0.6, 0.7]),
        np.array([0.8, 0.8]),
        np.ones((2, 2), dtype=bool),
    )
    uninterrupted = FixedStructurePowerPrimalDual(*args)
    for _ in range(25):
        expected = uninterrupted.step(convergence_patience=1000)
    prefix = FixedStructurePowerPrimalDual(*args)
    for _ in range(10):
        prefix.step(convergence_patience=1000)
    resumed = FixedStructurePowerPrimalDual(*args)
    resumed.load_state_dict(prefix.state_dict())
    for _ in range(15):
        actual = resumed.step(convergence_patience=1000)
    np.testing.assert_allclose(actual.power_w, expected.power_w)
    np.testing.assert_allclose(actual.target_prices, expected.target_prices)
    assert actual.iteration == expected.iteration == 25


def test_fixed_seed_random_feasible_benchmark_has_no_hard_violations():
    summary = run_benchmark(seed=20260908, cases=10, max_iterations=2000)
    assert summary["convergence_rate"] == 1.0
    assert summary["maximum_primal_violation"] <= 2.0e-4
    assert summary["maximum_power_budget_violation_w"] == 0.0
    assert summary["maximum_consensus_residual"] == 0.0


def test_bounded_temporal_warm_start_never_regresses_selected_violation():
    controller = BoundedTemporalPowerController(
        maximum_iterations_per_frame=20,
        solver_options={
            "power_cost_per_watt": 0.001,
            "quadratic_regularization": 0.01,
        },
        step_options={"convergence_patience": 3},
    )
    gain = np.array([[1.0, 0.2], [0.2, 1.0]])
    mask = np.ones_like(gain, dtype=bool)
    previous = None
    for scale in np.linspace(0.8, 1.0, 12):
        result = controller.solve(
            gain, scale * np.array([0.45, 0.45]), np.ones(2), mask)
        assert result.candidate_accepted
        assert result.selected_primal_violation <= (
            result.baseline_primal_violation + 1e-10)
        assert result.power_budget_violation_w == 0.0
        assert np.all(result.power_w >= 0.0)
        if previous is not None:
            assert result.warm_started
        previous = result.power_w


def test_bounded_temporal_controller_falls_back_on_primal_regression():
    controller = BoundedTemporalPowerController(
        maximum_iterations_per_frame=1,
        acceptance_tolerance=0.0,
        solver_options={"quadratic_regularization": 0.0},
    )
    controller.previous_power_w = np.array([[0.5, 0.5]])
    controller.previous_target_prices = np.array([100.0, 0.0])
    result = controller.solve(
        np.ones((1, 2)), np.array([0.5, 0.5]), 1.0,
        np.ones((1, 2), dtype=bool))
    assert not result.candidate_accepted
    assert result.fallback_reason == "primal_violation_regression"
    assert result.selected_primal_violation == 0.0
    np.testing.assert_allclose(result.power_w, [[0.5, 0.5]])


def test_actor_proposal_is_the_real_bounded_solver_initialization():
    gain = np.eye(2)
    demand = np.array([0.2, 0.2])
    mask = np.eye(2, dtype=bool)
    first = BoundedTemporalPowerController(
        maximum_iterations_per_frame=1)
    second = BoundedTemporalPowerController(
        maximum_iterations_per_frame=1)
    first_result = first.solve(
        gain, demand, 1.0, mask,
        actor_proposal_power_w=np.array([[0.8, 0.0], [0.0, 0.1]]),
        actor_proposal_mix=0.5)
    second_result = second.solve(
        gain, demand, 1.0, mask,
        actor_proposal_power_w=np.array([[0.1, 0.0], [0.0, 0.8]]),
        actor_proposal_mix=0.5)
    assert first_result.actor_proposal_used
    assert second_result.actor_proposal_used
    assert first_result.actor_proposal_mix == 0.5
    assert first_result.power_budget_violation_w == 0.0
    assert second_result.power_budget_violation_w == 0.0
    assert not np.allclose(first_result.power_w, second_result.power_w)


def test_actor_proximal_center_changes_bounded_iterate_without_budget_violation():
    gain = np.ones((1, 1))
    demand = np.zeros(1)
    mask = np.ones((1, 1), dtype=bool)
    low = BoundedTemporalPowerController(
        maximum_iterations_per_frame=1,
        solver_options={"proximal_regularization": 1.0},
    )
    high = BoundedTemporalPowerController(
        maximum_iterations_per_frame=1,
        solver_options={"proximal_regularization": 1.0},
    )
    low_result = low.solve(
        gain, demand, 1.0, mask,
        actor_proposal_power_w=np.array([[0.1]]),
    )
    high_result = high.solve(
        gain, demand, 1.0, mask,
        actor_proposal_power_w=np.array([[0.8]]),
    )
    assert high_result.actor_proposal_used
    assert high_result.power_w[0, 0] > low_result.power_w[0, 0]
    assert high_result.power_budget_violation_w == 0.0
    assert low_result.power_budget_violation_w == 0.0
    assert high_result.candidate_accepted


def test_bounded_temporal_checkpoint_preserves_warm_start_and_counters():
    gain = np.eye(2)
    demand = np.array([0.4, 0.4])
    mask = np.eye(2, dtype=bool)
    source = BoundedTemporalPowerController(maximum_iterations_per_frame=10)
    source.solve(gain, demand, 1.0, mask)
    restored = BoundedTemporalPowerController(maximum_iterations_per_frame=10)
    restored.load_state_dict(source.state_dict())
    expected = source.solve(gain, demand, 1.0, mask)
    actual = restored.solve(gain, demand, 1.0, mask)
    assert actual.warm_started
    np.testing.assert_allclose(actual.power_w, expected.power_w)
    np.testing.assert_allclose(actual.target_prices, expected.target_prices)
    assert restored.frame_count == source.frame_count
