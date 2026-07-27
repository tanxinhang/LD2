from __future__ import annotations

import itertools
import numpy as np
import torch

from uav_isac.agents.networks import CriticNetwork
from uav_isac.agents.trainer import (
    binary_risk_calibration_metrics,
    linear_sum_assignment_numpy,
    quantile_calibration_metrics,
    quantile_huber_loss,
)


def _critic(
    k: int,
    q: int,
    *,
    monotonic_quantiles: bool = False,
) -> CriticNetwork:
    return CriticNetwork(
        state_dim=8 * k + 8 * q + 1,
        hidden_layers=[32, 32],
        num_agents=k,
        comm_dim=16,
        num_targets=q,
        set_risk_critic_enabled=True,
        risk_hidden_dim=32,
        risk_num_quantiles=8,
        risk_cvar_alpha=0.25,
        risk_monotonic_quantiles_enabled=monotonic_quantiles,
    )


def _state(batch: int, k: int, q: int) -> torch.Tensor:
    base_dim = 8 * k + 8 * q + 1
    return torch.randn(batch, base_dim + k + 16)


def _permute_uavs(
    state: torch.Tensor,
    k: int,
    q: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    result = state.clone()
    result[:, :8 * k] = state[:, :8 * k].reshape(
        state.shape[0], k, 8)[:, permutation].reshape(state.shape[0], -1)
    # Also change the ignored absolute-agent one-hot to ensure it cannot leak.
    base_dim = 8 * k + 8 * q + 1
    result[:, base_dim:base_dim + k] = state[
        :, base_dim:base_dim + k][:, permutation]
    return result


def _permute_targets(
    state: torch.Tensor,
    k: int,
    q: int,
    permutation: torch.Tensor,
) -> torch.Tensor:
    result = state.clone()
    motion_start = 8 * k
    motion_end = motion_start + 6 * q
    result[:, motion_start:motion_end] = state[
        :, motion_start:motion_end
    ].reshape(state.shape[0], q, 6)[:, permutation].reshape(
        state.shape[0], -1)
    uncertainty_start = motion_end + 1
    pd_start = uncertainty_start + q
    result[:, uncertainty_start:uncertainty_start + q] = state[
        :, uncertainty_start:uncertainty_start + q][:, permutation]
    result[:, pd_start:pd_start + q] = state[
        :, pd_start:pd_start + q][:, permutation]
    return result


def test_set_risk_critic_shapes_and_finite_gradients() -> None:
    torch.manual_seed(9201)
    critic = _critic(6, 6)
    state = _state(5, 6, 6)
    quantiles, violation_logits = critic.forward_risk(state)

    assert quantiles.shape == (5, 6, 8)
    assert violation_logits.shape == (5, 6)
    assert critic.risk_cvar(state).shape == (5, 6)

    returns = torch.randn(5, 6)
    violations = torch.randint(0, 2, (5, 6)).float()
    loss = (
        quantile_huber_loss(quantiles, returns)
        + torch.nn.functional.binary_cross_entropy_with_logits(
            violation_logits, violations)
    )
    loss.backward()
    gradients = [
        parameter.grad for parameter in critic.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_set_risk_critic_is_uav_permutation_invariant() -> None:
    torch.manual_seed(9202)
    k, q = 4, 4
    critic = _critic(k, q).eval()
    state = _state(3, k, q)
    permutation = torch.tensor([2, 0, 3, 1])

    original = critic.forward_risk(state)
    permuted = critic.forward_risk(
        _permute_uavs(state, k, q, permutation))

    torch.testing.assert_close(original[0], permuted[0])
    torch.testing.assert_close(original[1], permuted[1])


def test_set_risk_critic_is_target_permutation_equivariant() -> None:
    torch.manual_seed(9203)
    k, q = 4, 4
    critic = _critic(k, q).eval()
    state = _state(3, k, q)
    permutation = torch.tensor([2, 0, 3, 1])

    original = critic.forward_risk(state)
    permuted = critic.forward_risk(
        _permute_targets(state, k, q, permutation))

    torch.testing.assert_close(permuted[0], original[0][:, permutation])
    torch.testing.assert_close(permuted[1], original[1][:, permutation])


def test_monotonic_risk_quantiles_are_bounded_and_ordered() -> None:
    torch.manual_seed(9205)
    critic = _critic(4, 4, monotonic_quantiles=True)
    quantiles, _ = critic.forward_risk(_state(7, 4, 4))

    assert bool(torch.all(quantiles > 0.0))
    assert bool(torch.all(quantiles < 1.0))
    assert bool(torch.all(quantiles[..., 1:] >= quantiles[..., :-1]))


def test_risk_critic_rejects_noncentralized_state_layout() -> None:
    try:
        CriticNetwork(
            state_dim=31,
            hidden_layers=[32, 32],
            num_agents=4,
            comm_dim=16,
            num_targets=4,
            set_risk_critic_enabled=True,
        )
    except ValueError as exc:
        assert 'centralized global-state layout' in str(exc)
    else:
        raise AssertionError('invalid risk-critic state layout was accepted')


def test_binary_risk_calibration_metrics_recognize_perfect_ranking() -> None:
    logits = torch.tensor([-8.0, -4.0, 4.0, 8.0]).numpy()
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0]).numpy()
    metrics = binary_risk_calibration_metrics(logits, labels)

    assert metrics['risk_violation_auroc'] == 1.0
    assert metrics['risk_violation_auprc'] == 1.0
    assert metrics['risk_violation_balanced_accuracy'] == 1.0
    assert metrics['risk_violation_brier'] < 0.01


def test_quantile_calibration_metrics_detect_crossing() -> None:
    target = torch.tensor([[0.2, 0.8], [0.4, 0.6]]).numpy()
    quantiles = torch.tensor([
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.7, 0.8, 0.9]],
        [[0.2, 0.3, 0.5, 0.6], [0.8, 0.7, 0.6, 0.5]],
    ]).numpy()
    metrics = quantile_calibration_metrics(quantiles, target)

    assert metrics['risk_quantile_pinball'] >= 0.0
    assert 0.0 <= metrics['risk_quantile_calibration_error'] <= 1.0
    assert metrics['risk_quantile_crossing_rate'] > 0.0


def test_numpy_hungarian_matches_bruteforce_rectangular_cost() -> None:
    rng = np.random.default_rng(9204)
    for shape in ((4, 4), (3, 5), (5, 3)):
        cost = rng.normal(size=shape)
        rows, columns = linear_sum_assignment_numpy(cost)
        observed = float(cost[rows, columns].sum())
        if shape[0] <= shape[1]:
            expected = min(
                sum(cost[row, column] for row, column in enumerate(choice))
                for choice in itertools.permutations(
                    range(shape[1]), shape[0]))
        else:
            expected = min(
                sum(cost[row, column] for column, row in enumerate(choice))
                for choice in itertools.permutations(
                    range(shape[0]), shape[1]))
        assert abs(observed - expected) < 1e-10
