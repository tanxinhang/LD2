import numpy as np
import pytest

from uav_isac.environment.fixed_lag_smoother import (
    estimate_cv_acceleration_from_residuals,
    fixed_lag_linear_gaussian_smoother,
)
from uav_isac.prediction.markov_kinematics import cv_markov_matrices


def _problem(seed: int = 7):
    rng = np.random.default_rng(seed)
    transition, process = cv_markov_matrices(0.2, 0.8)
    truth = np.zeros((18, 4), dtype=np.float64)
    truth[0] = [10.0, 20.0, 2.0, -1.0]
    for index in range(1, truth.shape[0]):
        acceleration = np.asarray([0.6, -0.3]) if 5 <= index < 13 else np.zeros(2)
        truth[index] = transition @ truth[index - 1]
        truth[index, :2] += 0.5 * 0.2 ** 2 * acceleration
        truth[index, 2:] += 0.2 * acceleration
    measurement_covariance = np.diag([4.0, 4.0, 1.0, 1.0])
    observations = truth + rng.multivariate_normal(
        np.zeros(4), measurement_covariance, size=truth.shape[0])
    observed = np.ones(truth.shape[0], dtype=bool)
    observed[[4, 9, 10]] = False
    observations[~observed] = np.nan
    result = fixed_lag_linear_gaussian_smoother(
        observations,
        prior_mean=np.asarray([8.0, 22.0, 1.0, 0.0]),
        prior_covariance=np.diag([9.0, 9.0, 4.0, 4.0]),
        transition=transition,
        process_covariance=process,
        measurement_covariance=measurement_covariance,
        observed=observed,
    )
    return truth, result


def test_fixed_lag_smoother_preserves_causal_endpoint_and_psd():
    _, result = _problem()
    np.testing.assert_allclose(
        result.smoothed_mean[-1], result.filtered_mean[-1], atol=0.0)
    np.testing.assert_allclose(
        result.smoothed_covariance[-1], result.filtered_covariance[-1], atol=0.0)
    assert np.min(np.linalg.eigvalsh(result.smoothed_covariance)) >= -1.0e-10
    assert np.isnan(result.innovation_nis[[4, 9, 10]]).all()


def test_smoothing_reduces_historical_error_and_uncertainty_on_maneuver_case():
    truth, result = _problem()
    historical = slice(0, -1)
    filtered_rmse = np.sqrt(np.mean(
        (result.filtered_mean[historical, :2] - truth[historical, :2]) ** 2))
    smoothed_rmse = np.sqrt(np.mean(
        (result.smoothed_mean[historical, :2] - truth[historical, :2]) ** 2))
    assert smoothed_rmse < filtered_rmse
    assert np.mean(np.trace(result.smoothed_covariance[historical], axis1=1, axis2=2)) < np.mean(
        np.trace(result.filtered_covariance[historical], axis1=1, axis2=2))


def test_cv_acceleration_estimator_recovers_exact_constant_acceleration():
    dt = 0.1
    acceleration = np.asarray([1.2, -0.7])
    one_residual = np.r_[0.5 * dt * dt * acceleration, dt * acceleration]
    residuals = np.repeat(one_residual[None], 5, axis=0)
    np.testing.assert_allclose(
        estimate_cv_acceleration_from_residuals(residuals, dt, lookback=4),
        acceleration,
        atol=1.0e-14,
    )


@pytest.mark.parametrize("recency", [0.0, 1.1])
def test_cv_acceleration_estimator_rejects_invalid_recency(recency):
    with pytest.raises(ValueError, match="recency"):
        estimate_cv_acceleration_from_residuals(
            np.zeros((2, 4)), 0.1, recency=recency)
