import numpy as np
import pytest
import torch

from uav_isac.optimization.temporal_feasible_structure import (
    enumerate_feasible_single_role_structures,
    straight_through_structure,
    temporal_feasible_structure_mixture,
)
from uav_isac.optimization.temporal_unrolled_power import (
    project_capped_row_power_budget_torch,
)


def _assert_hard_structure(structure, pair_limit, receiver_limit):
    z = np.asarray(structure, dtype=bool)
    k_count, _, q_count = z.shape
    assert not np.any(z[np.arange(k_count), np.arange(k_count), :])
    used_tx = set(np.argwhere(z)[:, 0])
    used_rx = set(np.argwhere(z)[:, 1])
    assert used_tx.isdisjoint(used_rx)
    for q in range(q_count):
        edges = np.argwhere(z[:, :, q])
        assert 1 <= len(edges) <= pair_limit
        assert len(set(edges[:, 1])) == 1
    assert np.all(np.sum(z, axis=(0, 2)) <= receiver_limit)


def test_exact_schedule_enumeration_obeys_all_hard_constraints():
    support = np.ones((4, 4, 3), dtype=bool)
    support[np.arange(4), np.arange(4), :] = False
    schedules = enumerate_feasible_single_role_structures(
        support,
        target_pair_limit=2,
        reports_per_receiver=4,
        max_structures=20000,
    )
    assert schedules.ndim == 4
    assert schedules.shape[1:] == (4, 4, 3)
    assert len(schedules) > 1
    for schedule in schedules:
        _assert_hard_structure(schedule, pair_limit=2, receiver_limit=4)


def test_schedule_bound_fails_loudly_instead_of_silent_pruning():
    support = np.ones((3, 3, 2), dtype=bool)
    support[np.arange(3), np.arange(3), :] = False
    with pytest.raises(RuntimeError, match="max_structures"):
        enumerate_feasible_single_role_structures(
            support,
            target_pair_limit=1,
            reports_per_receiver=2,
            max_structures=1,
        )


def test_temporal_mixture_is_convex_and_cross_frame_differentiable():
    dtype = torch.float64
    logits = torch.tensor(
        [
            [[2.0, -1.0], [-0.5, 1.0], [0.2, -0.1]],
            [[0.3, -0.2], [0.1, 0.4], [-0.1, 0.2]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    coefficient = torch.ones((2, 3, 3, 2), dtype=dtype)
    diagonal = torch.arange(3)
    coefficient[:, diagonal, diagonal, :] = 0.0
    result = temporal_feasible_structure_mixture(
        logits,
        coefficient,
        target_pair_limit=1,
        reports_per_receiver=2,
        temperature=0.6,
        inertia=2.0,
        max_structures=500,
    )
    assert torch.all(result.mixture >= 0.0)
    assert torch.all(result.mixture <= 1.0)
    assert torch.allclose(
        result.mixture.sum(dim=(1, 2)),
        torch.ones((2, 2), dtype=dtype),
        atol=1.0e-10,
    )
    for hard in result.hard_schedule.detach().cpu().numpy():
        _assert_hard_structure(hard, pair_limit=1, receiver_limit=2)
    # The second-frame objective reaches the first-frame actor only through
    # the explicit temporal proximal state.
    result.effective_gain_per_watt[1, 0, 0].backward()
    assert float(torch.linalg.vector_norm(logits.grad[0])) > 1.0e-10


def test_straight_through_schedule_is_hard_forward_but_differentiable():
    logits = torch.tensor(
        [[[1.2, -0.2], [-0.1, 0.8], [0.0, 0.1]],
         [[0.3, -0.4], [0.2, 0.6], [-0.2, 0.1]]],
        dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(
        [[[[0.0, 0.0], [0.2, 1.0], [0.9, 0.1]],
          [[0.3, 0.7], [0.0, 0.0], [1.1, 0.4]],
          [[0.8, 0.2], [0.6, 1.3], [0.0, 0.0]]],
         [[[0.0, 0.0], [0.5, 0.9], [0.2, 1.2]],
          [[0.7, 0.1], [0.0, 0.0], [1.0, 0.3]],
          [[0.4, 0.8], [0.9, 0.2], [0.0, 0.0]]]],
        dtype=torch.float64)
    diagonal = torch.arange(3)
    coefficient[:, diagonal, diagonal, :] = 0.0
    result = temporal_feasible_structure_mixture(
        logits, coefficient, target_pair_limit=1,
        reports_per_receiver=2, temperature=0.4, inertia=1.0,
        max_structures=500)
    schedule = straight_through_structure(result)
    assert torch.allclose(schedule, result.hard_schedule)
    loss = (coefficient * schedule).sum()
    loss.backward()
    assert float(torch.linalg.vector_norm(logits.grad)) > 1.0e-10


def test_detached_structure_state_removes_cross_frame_gradient():
    logits = torch.tensor(
        [
            [[1.0, -0.2], [-0.3, 0.8], [0.2, -0.1]],
            [[0.1, -0.1], [0.2, 0.0], [-0.2, 0.3]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    coefficient = torch.ones((2, 3, 3, 2), dtype=torch.float64)
    diagonal = torch.arange(3)
    coefficient[:, diagonal, diagonal, :] = 0.0
    result = temporal_feasible_structure_mixture(
        logits,
        coefficient,
        target_pair_limit=1,
        reports_per_receiver=2,
        temperature=0.6,
        inertia=2.0,
        max_structures=500,
        detach_between_frames=True,
    )
    result.effective_gain_per_watt[1, 0, 0].backward()
    assert torch.allclose(logits.grad[0], torch.zeros_like(logits.grad[0]))


def test_capped_projection_enforces_budget_and_structure_power_link():
    proposal = torch.tensor(
        [[1.4, 0.8, 0.3], [0.9, 0.6, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    budget = torch.tensor([1.0, 0.7], dtype=torch.float64)
    upper = torch.tensor(
        [[0.1, 0.8, 0.7], [0.0, 0.2, 0.7]], dtype=torch.float64)
    projected = project_capped_row_power_budget_torch(
        proposal, budget, upper)
    assert torch.all(projected >= 0.0)
    assert torch.all(projected <= upper + 1.0e-10)
    assert torch.all(projected.sum(dim=-1) <= budget + 1.0e-10)
    assert projected[1, 0].item() == 0.0
    projected.square().sum().backward()
    assert torch.all(torch.isfinite(proposal.grad))
