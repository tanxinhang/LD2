import numpy as np
import pytest

from config.params import load_config
from uav_isac.prediction.markov_kinematics import (
    cv_markov_matrices,
    predict_cv_markov_horizon,
    predict_reflecting_cv_mean,
)


def test_cv_horizon_matches_repeated_chapman_kolmogorov_recursion():
    mean = np.asarray([[10.0, 20.0, 3.0, -2.0], [50.0, 40.0, -1.0, 4.0]])
    covariance = np.stack([np.eye(4), 2.0 * np.eye(4)])
    result = predict_cv_markov_horizon(
        mean, covariance, steps=7, dt_s=0.1,
        acceleration_std_mps2=1.5, process_scale=np.asarray([1.0, 2.0]))
    transition, process = cv_markov_matrices(0.1, 1.5)
    expected_mean = mean.copy()
    expected_covariance = covariance.copy()
    for _ in range(7):
        expected_mean = np.einsum("ij,...j->...i", transition, expected_mean)
        expected_covariance = np.einsum(
            "ij,...jk,lk->...il", transition, expected_covariance, transition)
        expected_covariance += np.asarray([1.0, 2.0])[:, None, None] * process
    np.testing.assert_allclose(result.mean[-1], expected_mean, atol=1e-13)
    np.testing.assert_allclose(result.covariance[-1], expected_covariance, atol=1e-13)


def test_cv_horizon_preserves_psd_and_grows_position_uncertainty():
    result = predict_cv_markov_horizon(
        np.zeros(4), np.eye(4), steps=20, dt_s=0.1,
        acceleration_std_mps2=1.5)
    assert np.all(np.linalg.eigvalsh(result.covariance) >= -1e-12)
    assert result.covariance[-1, 0, 0] > result.covariance[0, 0, 0]
    assert result.covariance[-1, 1, 1] > result.covariance[0, 1, 1]


def test_reflecting_cv_mean_matches_wall_dynamics_and_multiple_crossings():
    position, velocity = predict_reflecting_cv_mean(
        np.asarray([[9.0, 1.0], [2.0, 8.0]]),
        np.asarray([[3.0, -4.0], [23.0, 17.0]]),
        elapsed_s=1.0,
        area_size_m=(10.0, 10.0),
    )
    np.testing.assert_allclose(position[0], [8.0, 3.0])
    np.testing.assert_allclose(velocity[0], [-3.0, 4.0])
    np.testing.assert_allclose(position[1], [5.0, 5.0])
    np.testing.assert_allclose(velocity[1], [23.0, 17.0])


@pytest.mark.parametrize("steps", [-1, 1.5])
def test_cv_horizon_rejects_invalid_step_count(steps):
    with pytest.raises(ValueError, match="steps"):
        predict_cv_markov_horizon(
            np.zeros(4), np.eye(4), steps=steps, dt_s=0.1,
            acceleration_std_mps2=1.0)


def test_l4_markov_profile_enables_only_causal_mean_prediction():
    cfg = load_config("config/exp_research_l4_markov_prediction_h10.yaml")
    assert cfg.marl.distributed_greedy_matching_prediction_frames == 10
    assert cfg.marl.distributed_greedy_matching_hold_frames == 20
    assert cfg.marl.distributed_bistatic_bottleneck_movement_enabled is True
