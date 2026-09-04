"""Equivalence locks for zero-semantic performance optimizations."""

import numpy as np
import pytest
import torch

from uav_isac.agents.networks import CriticNetwork
from uav_isac.agents.trainer import MAPPTrainer
from uav_isac.environment.belief import BeliefState
from uav_isac.environment.observation import ObservationBuilder
from uav_isac.utils.types import UAVState
from uav_isac.utils.math_utils import symmetric_2x2_max_eigenvalue


def test_symmetric_2x2_closed_form_matches_general_eigensolver() -> None:
    rng = np.random.default_rng(20260829)
    raw = rng.normal(size=(17, 23, 2, 2))
    symmetric = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    expected = np.linalg.eigvalsh(symmetric)[..., -1]
    actual = symmetric_2x2_max_eigenvalue(symmetric)
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("equivariant", [False, True])
@pytest.mark.parametrize("include_credit", [False, True])
def test_single_pass_critic_matches_separate_heads(
    equivariant: bool,
    include_credit: bool,
) -> None:
    torch.manual_seed(20260829)
    k, q = 4, 3
    base_dim = 8 * k + 8 * q + 1 if equivariant else 41
    critic = CriticNetwork(
        state_dim=base_dim,
        hidden_layers=[32, 32],
        num_agents=k,
        comm_dim=16,
        num_targets=q,
        equivariant_value_critic_enabled=equivariant,
    ).eval()
    state = torch.randn(7, base_dim + k + 16)

    expected_scalar, expected_target = critic.forward_with_targets(state)
    expected_credit = None
    if include_credit:
        expected_scalar, expected_credit = critic.forward_with_credit(state)
    actual_scalar, actual_credit, actual_target = (
        critic.forward_with_auxiliaries(
            state, include_credit=include_credit))

    torch.testing.assert_close(actual_scalar, expected_scalar)
    torch.testing.assert_close(actual_target, expected_target)
    if include_credit:
        torch.testing.assert_close(actual_credit, expected_credit)
    else:
        assert actual_credit is None


def test_rollout_is_decorated_with_inference_mode() -> None:
    # torch's context-decorator keeps __wrapped__; the production rollout tests
    # exercise the wrapped body and this lock prevents the no-grad boundary
    # from being accidentally narrowed back to tensor copies only.
    assert hasattr(MAPPTrainer.collect_rollout, "__wrapped__")


def test_dense_belief_observation_matches_object_interface() -> None:
    rng = np.random.default_rng(9301)
    k, q = 4, 3
    builder = ObservationBuilder(
        k, q, area_size=(800.0, 600.0), height=20.0,
        use_relative_features=True, expose_neighbor_state=True)
    states = [
        UAVState(
            pos=np.asarray([50.0 + 20.0 * node, 80.0, 20.0]),
            vel=np.asarray([1.0, -0.5, 0.0]),
            battery=40000.0,
            role=node % 3,
        )
        for node in range(k)
    ]
    mean = rng.normal(size=(k, q, 4))
    mean[:, :, 0] = 300.0 + 20.0 * mean[:, :, 0]
    mean[:, :, 1] = 250.0 + 20.0 * mean[:, :, 1]
    cov_diag = rng.uniform(0.5, 50.0, size=(k, q, 4))
    aoi = rng.integers(0, 20, size=(k, q))
    beliefs = [[
        BeliefState(
            mean=mean[node, target].copy(),
            cov_diag=cov_diag[node, target].copy(),
            aoi=int(aoi[node, target]),
        )
        for target in range(q)
    ] for node in range(k)]

    for node in range(k):
        object_obs = builder.build_local_obs(
            node, states, beliefs, np.zeros(q))
        dense_obs = builder.build_local_obs(
            node, states, None, np.zeros(q),
            belief_mean=mean,
            belief_cov_diag=cov_diag,
            belief_aoi=aoi,
        )
        np.testing.assert_array_equal(dense_obs, object_obs)
