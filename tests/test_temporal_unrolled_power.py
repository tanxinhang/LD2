import numpy as np
import pytest
import torch

from uav_isac.optimization.constrained_pareto import (
    assign_constrained_pareto_gradients,
    empirical_detection_cvar_residual,
    inequality_augmented_lagrangian,
)
from uav_isac.optimization.distributed_power_primal_dual import (
    project_masked_row_power_budget,
)
from uav_isac.optimization.temporal_unrolled_power import (
    DifferentiableTemporalPowerUnroll,
    actor_sensing_budget,
    project_masked_row_power_budget_torch,
    temporal_power_objectives,
)


def _problem(*, requires_grad=True):
    generator = torch.Generator().manual_seed(20260908)
    horizon, agents, targets, rates = 4, 2, 2, 3
    communication = torch.zeros(
        horizon, agents, requires_grad=requires_grad)
    sensing = torch.randn(
        horizon, agents, targets, generator=generator).requires_grad_(
            requires_grad)
    rate_logits = torch.zeros(
        horizon, agents, rates, requires_grad=requires_grad)
    gain = torch.tensor([
        [[1.20, 0.00], [0.00, 1.00]],
        [[1.10, 0.00], [0.00, 1.20]],
        [[1.00, 0.00], [0.00, 1.10]],
        [[1.20, 0.00], [0.00, 1.00]],
    ])
    requirement = torch.full((horizon, targets), 0.35)
    return (
        communication, sensing, rate_logits, gain, requirement, gain > 0.0)


def test_torch_projection_matches_execution_projection_and_retains_gradient():
    proposed = torch.tensor(
        [[[0.8, -0.2, 0.7], [0.1, 0.4, 0.9]],
         [[0.2, 0.3, 0.4], [1.2, 0.1, -0.4]]],
        dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[[True, False, True], [True, True, False]],
         [[True, True, True], [False, True, True]]])
    budget = torch.tensor([[1.0, 0.3], [2.0, 0.05]], dtype=torch.float64)
    projected = project_masked_row_power_budget_torch(
        proposed, budget, mask)
    expected = np.stack([
        project_masked_row_power_budget(
            proposed.detach().numpy()[frame], budget.numpy()[frame],
            mask.numpy()[frame])
        for frame in range(proposed.shape[0])
    ])
    np.testing.assert_allclose(projected.detach().numpy(), expected, atol=1e-12)
    assert torch.all(projected[~mask] == 0.0)
    assert torch.all(projected.sum(dim=-1) <= budget + 1e-12)
    projected.square().sum().backward()
    assert proposed.grad is not None
    assert torch.isfinite(proposed.grad).all()
    assert float(proposed.grad.abs().sum()) > 0.0


def test_actor_rate_and_power_heads_jointly_determine_the_feasible_budget():
    power = torch.zeros(2, 2, requires_grad=True)
    rates = torch.zeros(2, 2, 3, requires_grad=True)
    budget, fraction, active = actor_sensing_budget(
        power, rates, total_rf_budget_w=1.0, sensing_power_cap_w=1.0,
        communication_fraction_min=0.0,
        communication_fraction_max=1.0)
    assert torch.allclose(active, torch.full_like(active, 2.0 / 3.0))
    assert torch.allclose(fraction, torch.full_like(fraction, 1.0 / 3.0))
    assert torch.allclose(budget, torch.full_like(budget, 2.0 / 3.0))
    budget.sum().backward()
    assert float(power.grad.abs().sum()) > 0.0
    assert float(rates.grad.abs().sum()) > 0.0


def test_unrolled_horizon_is_hard_safe_and_all_outer_losses_share_one_graph():
    communication, sensing, rates, gain, requirement, mask = _problem()
    layer = DifferentiableTemporalPowerUnroll(
        inner_iterations=3, warm_start_mix=0.4)
    result = layer(
        communication, sensing, rates, gain, requirement, mask,
        total_rf_budget_w=1.0, sensing_power_cap_w=1.0, p_fa=0.01)
    assert result.power_w.shape == (4, 2, 2)
    assert result.target_prices.shape == (4, 2)
    assert result.detection_probability.shape == (4, 2)
    assert float(result.budget_violation_w.detach().max()) <= 1e-7
    assert torch.all(result.power_w[~mask] == 0.0)

    objectives = temporal_power_objectives(result)
    cvar = empirical_detection_cvar_residual(
        result.detection_probability, 0.06, tail_fraction=0.5)
    constraint = inequality_augmented_lagrangian(
        cvar, torch.ones_like(cvar), penalty=0.1)
    diagnostic = assign_constrained_pareto_gradients(
        objectives.as_tuple(), (communication, sensing, rates),
        constraint_loss=constraint)
    assert torch.isfinite(diagnostic.applied_gradient_norm)
    assert float(diagnostic.applied_gradient_norm) > 0.0
    assert pytest.approx(float(diagnostic.pareto.weights.sum()), abs=1e-6) == 1.0
    # One update direction receives information from communication load,
    # sensing energy/switching, detection, and the target-wise CVaR condition.
    for tensor in (communication, sensing, rates):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert float(tensor.grad.abs().sum()) > 0.0


def test_cross_frame_state_creates_bptt_gradient_and_detach_is_explicit():
    problem = _problem()
    communication, sensing, rates, gain, requirement, mask = problem
    coupled = DifferentiableTemporalPowerUnroll(
        inner_iterations=1, warm_start_mix=0.4,
        detach_between_frames=False)
    coupled_result = coupled(
        communication, sensing, rates, gain, requirement, mask,
        1.0, 1.0, p_fa=0.01)
    coupled_gradient = torch.autograd.grad(
        -coupled_result.detection_probability[-1].mean(), sensing,
        retain_graph=True)[0]
    assert float(coupled_gradient[0].abs().sum()) > 0.0

    detached = DifferentiableTemporalPowerUnroll(
        inner_iterations=1, warm_start_mix=0.4,
        detach_between_frames=True)
    detached_result = detached(
        communication, sensing, rates, gain, requirement, mask,
        1.0, 1.0, p_fa=0.01)
    detached_gradient = torch.autograd.grad(
        -detached_result.detection_probability[-1].mean(), sensing)[0]
    assert float(detached_gradient[:-1].abs().sum()) == 0.0
    assert float(detached_gradient[-1].abs().sum()) > 0.0


def test_more_inner_iterations_reduce_target_residual_on_same_actor_plan():
    communication, sensing, rates, gain, requirement, mask = _problem()
    short = DifferentiableTemporalPowerUnroll(
        inner_iterations=1, warm_start_mix=0.4)
    long = DifferentiableTemporalPowerUnroll(
        inner_iterations=12, warm_start_mix=0.4)
    short_result = short(
        communication, sensing, rates, gain, requirement, mask,
        1.0, 1.0, p_fa=0.01)
    long_result = long(
        communication, sensing, rates, gain, requirement, mask,
        1.0, 1.0, p_fa=0.01)
    assert float(long_result.primal_violation.detach().max()) <= (
        float(short_result.primal_violation.detach().max()) + 1e-7)


def test_actor_proximal_curvature_is_shared_by_unroll_and_keeps_projection_safe():
    communication, sensing, rates, gain, requirement, mask = _problem()
    reference = DifferentiableTemporalPowerUnroll(
        inner_iterations=2, warm_start_mix=0.4,
        actor_proximal_regularization=0.0)
    anchored = DifferentiableTemporalPowerUnroll(
        inner_iterations=2, warm_start_mix=0.4,
        actor_proximal_regularization=0.1)
    reference_result = reference(
        communication, sensing, rates, gain, requirement, mask,
        1.0, 1.0, p_fa=0.01)
    anchored_result = anchored(
        communication, sensing, rates, gain, requirement, mask,
        1.0, 1.0, p_fa=0.01)
    assert torch.isfinite(anchored_result.power_w).all()
    assert float(anchored_result.budget_violation_w.detach().max()) <= 1e-7
    assert not torch.allclose(
        reference_result.power_w, anchored_result.power_w)
    loss = -anchored_result.detection_probability.mean()
    loss.backward()
    assert sensing.grad is not None
    assert torch.isfinite(sensing.grad).all()
