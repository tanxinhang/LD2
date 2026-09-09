import pytest
import torch

from uav_isac.optimization.alternating_controller import (
    AlternatingConstrainedController,
    team_aligned_minibatches,
)
from uav_isac.optimization.constrained_pareto import (
    adaptive_constraint_projection,
    assign_constrained_pareto_gradients,
    collapse_repeated_team_detection,
    constrained_pareto_status,
    empirical_detection_cvar_residual,
    flattened_team_detection_cvar_residual,
    importance_weighted_detection_cvar_residual,
    inequality_augmented_lagrangian,
    joint_policy_detection_cvar_residual,
    minimum_norm_pareto_direction,
    projected_target_dual_update,
)


def test_minimum_norm_pareto_cancels_opposing_gradients():
    result = minimum_norm_pareto_direction(torch.tensor([
        [1.0, 0.0],
        [-1.0, 0.0],
    ]))
    torch.testing.assert_close(result.weights, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(result.gradient, torch.zeros(2), atol=1e-7, rtol=0)
    assert float(result.gradient_norm) <= 1e-7
    assert result.converged


def test_minimum_norm_pareto_finds_balanced_common_direction():
    result = minimum_norm_pareto_direction(torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
    ]))
    torch.testing.assert_close(result.weights, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(result.gradient, torch.tensor([0.5, 0.5]))
    assert result.converged


def test_detection_cvar_keeps_margin_and_selected_tail_gradient():
    probability = torch.tensor([
        [0.90, 0.80],
        [0.70, 0.65],
        [0.50, 0.55],
        [0.10, 0.45],
    ], requires_grad=True)
    residual = empirical_detection_cvar_residual(
        probability, 0.60, tail_fraction=0.5)
    torch.testing.assert_close(residual, torch.tensor([0.30, 0.10]))
    residual.sum().backward()
    assert probability.grad is not None
    assert int(torch.count_nonzero(probability.grad)) == 4


def test_importance_weighted_cvar_equals_empirical_at_rollout_policy():
    probability = torch.tensor([
        [0.90, 0.80],
        [0.70, 0.65],
        [0.50, 0.55],
        [0.10, 0.45],
    ])
    empirical = empirical_detection_cvar_residual(
        probability, 0.60, tail_fraction=0.5)
    surrogate = importance_weighted_detection_cvar_residual(
        probability,
        0.60,
        torch.zeros(probability.shape[0]),
        tail_fraction=0.5,
    )
    torch.testing.assert_close(surrogate, empirical)


def test_importance_weighted_cvar_backpropagates_only_selected_tail():
    probability = torch.tensor([
        [0.90, 0.80],
        [0.70, 0.65],
        [0.50, 0.55],
        [0.10, 0.45],
    ])
    log_ratio = torch.zeros(4, requires_grad=True)
    residual = importance_weighted_detection_cvar_residual(
        probability, 0.60, log_ratio, tail_fraction=0.5)
    residual.sum().backward()
    # Each sample receives the sum of the target shortfalls for which it lies
    # in the frozen rollout tail. High-performing sample 0 is never selected.
    assert log_ratio.grad is not None
    assert float(log_ratio.grad[0]) == pytest.approx(0.0)
    assert float(log_ratio.grad[2]) == pytest.approx(0.075)
    assert float(log_ratio.grad[3]) == pytest.approx(0.325)


def test_joint_policy_cvar_uses_team_probability_not_duplicated_agent_rows():
    probability = torch.tensor([
        [0.90, 0.80],
        [0.50, 0.55],
        [0.10, 0.45],
    ])
    new_log_probability = torch.zeros((3, 2), requires_grad=True)
    old_log_probability = torch.zeros((3, 2))
    residual = joint_policy_detection_cvar_residual(
        probability,
        0.60,
        new_log_probability,
        old_log_probability,
        tail_fraction=1.0 / 3.0,
    )
    torch.testing.assert_close(residual, torch.tensor([0.50, 0.15]))
    residual.sum().backward()
    assert new_log_probability.grad is not None
    # Only the single worst team transition is selected. Both agents receive
    # the same joint-action constraint gradient for that physical outcome.
    torch.testing.assert_close(
        new_log_probability.grad,
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.65, 0.65]]),
    )


def test_flattened_team_bridge_collapses_only_consistent_physical_outcomes():
    unique_probability = torch.tensor([
        [0.90, 0.80],
        [0.50, 0.55],
        [0.10, 0.45],
    ])
    flattened = unique_probability.repeat_interleave(2, dim=0)
    torch.testing.assert_close(
        collapse_repeated_team_detection(flattened, 2),
        unique_probability,
    )
    new_log_probability = torch.zeros(6, requires_grad=True)
    residual = flattened_team_detection_cvar_residual(
        flattened,
        0.60,
        new_log_probability,
        torch.zeros(6),
        num_agents=2,
        tail_fraction=1.0 / 3.0,
    )
    torch.testing.assert_close(residual, torch.tensor([0.50, 0.15]))
    residual.sum().backward()
    torch.testing.assert_close(
        new_log_probability.grad,
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.65, 0.65]),
    )


def test_flattened_team_bridge_rejects_inconsistent_or_incomplete_teams():
    inconsistent = torch.tensor([
        [0.7, 0.8], [0.7, 0.6],
        [0.5, 0.4], [0.5, 0.4],
    ])
    with pytest.raises(ValueError, match="consistent team"):
        collapse_repeated_team_detection(inconsistent, 2)
    with pytest.raises(ValueError, match="complete teams"):
        collapse_repeated_team_detection(torch.rand(5, 2), 2)


def test_target_dual_update_is_projected_per_target():
    updated = projected_target_dual_update(
        torch.tensor([0.2, 0.1, 0.0]),
        torch.tensor([0.3, -0.4, 2.0]),
        step_size=0.5,
        maximum=0.75,
    )
    torch.testing.assert_close(updated, torch.tensor([0.35, 0.0, 0.75]))
    assert not updated.requires_grad


def test_convergence_requires_all_independent_conditions():
    accepted = constrained_pareto_status(
        5e-4, torch.tensor([-0.1, 8e-4]), torch.tensor([2e-4]), 3e-4,
        stationarity_tolerance=1e-3,
        primal_tolerance=1e-3,
        dual_tolerance=1e-3,
        consensus_tolerance=1e-3,
    )
    assert accepted.converged
    rejected = constrained_pareto_status(
        5e-4, torch.tensor([2e-3]), torch.tensor([2e-4]), 3e-4,
        stationarity_tolerance=1e-3,
        primal_tolerance=1e-3,
        dual_tolerance=1e-3,
        consensus_tolerance=1e-3,
    )
    assert not rejected.converged
    assert rejected.primal_violation == pytest.approx(2e-3)


def test_pareto_backward_reaches_stationary_compromise_without_loss_weights():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.10)
    for _ in range(80):
        first = (parameter[0] - 1.0).square() + parameter[1].square()
        second = parameter[0].square() + (parameter[1] - 1.0).square()
        optimizer.zero_grad(set_to_none=True)
        assigned = assign_constrained_pareto_gradients(
            (first, second), [parameter])
        optimizer.step()
    torch.testing.assert_close(
        parameter.detach(), torch.tensor([0.5, 0.5]), atol=2e-3, rtol=0)
    assert float(assigned.pareto.gradient_norm) <= 2e-3


def test_constraint_gradient_is_not_traded_against_pareto_objectives():
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    first = parameter[0]
    second = -parameter[0]
    constraint = -2.0 * parameter[1]
    assigned = assign_constrained_pareto_gradients(
        (first, second), [parameter], constraint_loss=constraint)
    torch.testing.assert_close(parameter.grad, torch.tensor([0.0, -2.0]))
    assert float(assigned.constraint_gradient_norm) == pytest.approx(2.0)
    assert float(assigned.applied_gradient_norm) == pytest.approx(2.0)


def test_feasibility_first_uses_scale_invariant_constraint_direction():
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    objective = -parameter[0]
    constraint = 7.0 * (parameter[0] + parameter[1])
    assigned = assign_constrained_pareto_gradients(
        (objective,), [parameter], constraint_loss=constraint,
        feasibility_first=True)
    expected = torch.tensor([1.0, 1.0]) / torch.sqrt(torch.tensor(2.0))
    torch.testing.assert_close(parameter.grad, expected)
    assert assigned.update_mode == "feasibility_first"
    assert float(assigned.applied_gradient_norm) == pytest.approx(1.0)


def test_feasibility_first_rejects_zero_constraint_gradient():
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    with pytest.raises(ValueError, match="non-zero constraint gradient"):
        assign_constrained_pareto_gradients(
            (parameter.sum(),), [parameter],
            constraint_loss=parameter.sum() * 0.0,
            feasibility_first=True)


def test_objective_gradient_normalization_removes_physical_unit_scale():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    detection = parameter[0]
    communication_bits = 1000.0 * parameter[1]
    assigned = assign_constrained_pareto_gradients(
        (detection, communication_bits), [parameter])
    torch.testing.assert_close(
        assigned.pareto.weights, torch.tensor([0.5, 0.5]),
        atol=1e-6, rtol=0)
    torch.testing.assert_close(
        parameter.grad, torch.tensor([0.5, 0.5]), atol=1e-6, rtol=0)


def test_batched_pareto_vjp_is_equivalent_to_reference_reverse_passes():
    def evaluate(*, batched):
        parameter = torch.nn.Parameter(torch.tensor([1.2, -0.4, 0.7]))
        first = (parameter[0] - 1.0).square() + 2.0 * parameter[1]
        second = 3.0 * parameter[0] + parameter[2].square()
        constraint = 0.7 * parameter[1] - 0.2 * parameter[2]
        result = assign_constrained_pareto_gradients(
            (first, second), [parameter], constraint_loss=constraint,
            batched_vjp=batched)
        return parameter.grad.detach().clone(), result

    batched_gradient, batched = evaluate(batched=True)
    scalar_gradient, scalar = evaluate(batched=False)
    torch.testing.assert_close(batched_gradient, scalar_gradient)
    torch.testing.assert_close(batched.pareto.weights, scalar.pareto.weights)
    torch.testing.assert_close(
        batched.pareto.gradient, scalar.pareto.gradient)
    torch.testing.assert_close(
        batched.objective_gradient_norms, scalar.objective_gradient_norms)
    torch.testing.assert_close(
        batched.objective_gradient_cosine, scalar.objective_gradient_cosine)


def test_active_equality_projects_pareto_direction_to_resource_tangent():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    first = parameter[0] + 2.0 * parameter[1]
    second = parameter[0] + 3.0 * parameter[1]
    active_rf_budget = parameter[0]
    assigned = assign_constrained_pareto_gradients(
        (first, second), [parameter],
        constraint_loss=2.0 * parameter[1],
        tangent_constraint_losses=(active_rf_budget,))
    # The active budget equality fixes the first coordinate.  The applied
    # Pareto direction must therefore lie in its tangent (second) coordinate.
    torch.testing.assert_close(parameter.grad[0], torch.tensor(0.0), atol=1e-7, rtol=0)
    assert float(parameter.grad[1]) > 0.0
    assert assigned.equality_constraint_gradient_norm is not None
    assert assigned.tangent_gradient_norm is not None
    assert float(assigned.equality_constraint_gradient_norm) == pytest.approx(1.0)


def test_adaptive_projection_switches_to_rank_aware_multi_constraint_mode():
    objective = torch.tensor([[1.0, 2.0, 3.0]])
    # The second equality is dependent on the first and must not create a
    # second numerical normal direction.
    equalities = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = adaptive_constraint_projection(objective, equalities)
    assert result.mode == "multi"
    assert result.rank == 1
    torch.testing.assert_close(
        result.gradients, torch.tensor([[0.0, 2.0, 3.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        result.norm_ratio, torch.sqrt(torch.tensor(13.0 / 14.0)),
                               atol=1e-6, rtol=0)


def test_alternating_cvar_primal_dual_reaches_constraint_boundary():
    logits = torch.nn.Parameter(torch.full((3,), -0.4))
    optimizer = torch.optim.Adam([logits], lr=0.02)
    multipliers = torch.zeros(3)
    penalty = 0.05
    noise = torch.tensor([
        [-0.4, -0.3, -0.5],
        [-0.2, -0.1, -0.3],
        [0.0, 0.0, -0.1],
        [0.1, 0.2, 0.1],
        [0.3, 0.4, 0.2],
    ])
    for _ in range(600):
        probability = torch.sigmoid(logits[None, :] + noise)
        residual = empirical_detection_cvar_residual(
            probability, 0.60, tail_fraction=0.40)
        # Resource use is the sole objective; QoS is an optimization condition.
        objective = torch.nn.functional.softplus(logits).mean()
        loss = objective + inequality_augmented_lagrangian(
            residual, multipliers, penalty=penalty)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probability = torch.sigmoid(logits[None, :] + noise)
            residual = empirical_detection_cvar_residual(
                probability, 0.60, tail_fraction=0.40)
            multipliers = projected_target_dual_update(
                multipliers, residual, step_size=penalty)
    assert float(torch.max(torch.abs(residual))) <= 1.0e-3
    assert torch.all(multipliers > 0.0)


def test_controller_cycles_small_objective_blocks_and_updates_conditions():
    controller = AlternatingConstrainedController(
        (("detection",), ("communication", "latency")),
        constraint_count=2,
        penalty=0.2,
        dual_maximum=0.5,
    )
    assert controller.current_objectives == ("detection",)
    selected = controller.select_losses({
        "detection": torch.tensor(1.0),
        "communication": torch.tensor(2.0),
        "latency": torch.tensor(3.0),
    })
    assert len(selected) == 1
    step = controller.finish_step(torch.tensor([0.5, -0.25]))
    torch.testing.assert_close(step.multipliers, torch.tensor([0.1, 0.0]))
    assert step.maximum_violation == pytest.approx(0.5)
    assert controller.current_objectives == ("communication", "latency")
    controller.finish_step(torch.tensor([3.0, 0.5]))
    torch.testing.assert_close(controller.multipliers, torch.tensor([0.5, 0.1]))
    assert controller.current_objectives == ("detection",)


def test_team_aligned_minibatches_preserve_complete_shuffled_teams():
    batches = team_aligned_minibatches(
        total_rows=18,
        num_agents=3,
        maximum_rows_per_batch=7,
        team_order=torch.tensor([4, 1, 5, 0, 3, 2]),
    )
    assert [batch.numel() for batch in batches] == [6, 6, 6]
    torch.testing.assert_close(batches[0], torch.tensor([12, 13, 14, 3, 4, 5]))
    all_rows = torch.cat(batches)
    torch.testing.assert_close(torch.sort(all_rows).values, torch.arange(18))
    for batch in batches:
        teams = batch.reshape(-1, 3)
        torch.testing.assert_close(
            teams[:, 1] - teams[:, 0], torch.ones_like(teams[:, 0]))
        torch.testing.assert_close(
            teams[:, 2] - teams[:, 1], torch.ones_like(teams[:, 0]))


def test_team_aligned_minibatches_reject_invalid_layout_or_permutation():
    with pytest.raises(ValueError, match="complete teams"):
        team_aligned_minibatches(10, 3, 6)
    with pytest.raises(ValueError, match="at least one complete team"):
        team_aligned_minibatches(12, 3, 2)
    with pytest.raises(ValueError, match="permutation"):
        team_aligned_minibatches(
            12, 3, 6, team_order=torch.tensor([0, 0, 2, 3]))


def test_controller_checkpoint_resume_preserves_schedule_and_duals():
    original = AlternatingConstrainedController(
        (("detection",), ("power", "refresh"), ("communication",)),
        constraint_count=3,
        penalty=0.1,
    )
    original.finish_step(torch.tensor([0.3, -0.2, 0.1]))
    original.finish_step(torch.tensor([0.2, 0.4, -0.1]))
    restored = AlternatingConstrainedController(
        original.objective_groups,
        constraint_count=3,
        penalty=0.1,
    )
    restored.load_state_dict(original.state_dict())
    assert restored.current_objectives == ("communication",)
    assert restored.update_count == 2
    torch.testing.assert_close(restored.multipliers, original.multipliers)
    torch.testing.assert_close(restored.last_residual, original.last_residual)
    next_step = restored.finish_step(torch.tensor([-0.1, 0.2, 0.3]))
    expected = original.finish_step(torch.tensor([-0.1, 0.2, 0.3]))
    torch.testing.assert_close(next_step.multipliers, expected.multipliers)
    assert restored.current_objectives == original.current_objectives


def test_controller_checkpoint_fails_closed_on_incompatible_conditions():
    source = AlternatingConstrainedController(
        (("detection",),), constraint_count=2, penalty=0.1)
    target = AlternatingConstrainedController(
        (("detection",),), constraint_count=3, penalty=0.1)
    with pytest.raises(ValueError, match="constraint count"):
        target.load_state_dict(source.state_dict())


def test_controller_convergence_uses_latest_condition_and_dual_residuals():
    controller = AlternatingConstrainedController(
        (("detection",),), constraint_count=2, penalty=0.1)
    controller.finish_step(torch.tensor([-0.2, 5.0e-4]))
    status = controller.convergence_status(
        4.0e-4,
        stationarity_tolerance=1.0e-3,
        primal_tolerance=1.0e-3,
        dual_tolerance=1.0e-3,
    )
    assert status.converged
    controller.finish_step(torch.tensor([0.02, -0.1]))
    status = controller.convergence_status(
        4.0e-4,
        stationarity_tolerance=1.0e-3,
        primal_tolerance=1.0e-3,
        dual_tolerance=1.0e-3,
    )
    assert not status.converged
    assert status.primal_violation == pytest.approx(0.02)
