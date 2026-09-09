"""Mechanism screen for fixed-lag smoothing and residual acceleration.

The ``white`` case matches the current L4 CV target: acceleration is sampled
independently every frame.  The ``persistent`` case is an explicit alternative
research mechanism, not a claim about the current simulator.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.environment.fixed_lag_smoother import (
    estimate_cv_acceleration_from_residuals,
    fixed_lag_linear_gaussian_smoother,
)
from uav_isac.prediction.markov_kinematics import cv_markov_matrices


def _advance(state: np.ndarray, acceleration: np.ndarray, dt: float) -> np.ndarray:
    result = state.copy()
    result[:2] += dt * state[2:] + 0.5 * dt * dt * acceleration
    result[2:] += dt * acceleration
    return result


def _forecast(
    state: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
    steps: int,
    decay: float,
) -> np.ndarray:
    result = state.copy()
    current_acceleration = acceleration.copy()
    for _ in range(steps):
        result = _advance(result, current_acceleration, dt)
        current_acceleration *= decay
    return result


def _trial(rng: np.random.Generator, mode: str) -> Dict[str, float]:
    dt = 0.1
    history_steps = 30
    horizon = 10
    sigma_a = 1.5
    transition, process_covariance = cv_markov_matrices(dt, sigma_a)
    measurement_covariance = np.diag([15.0 ** 2, 15.0 ** 2, 3.0 ** 2, 3.0 ** 2])
    truth = np.empty((history_steps + horizon, 4), dtype=np.float64)
    truth[0] = np.asarray([400.0, 500.0, 6.0, -3.0])
    acceleration = np.zeros(2, dtype=np.float64)
    for index in range(1, truth.shape[0]):
        if mode == "white":
            acceleration = rng.normal(0.0, sigma_a, size=2)
        elif mode == "persistent":
            acceleration = (
                0.9 * acceleration
                + rng.normal(0.0, sigma_a * np.sqrt(1.0 - 0.9 ** 2), size=2)
            )
        else:
            raise ValueError("mode must be white or persistent")
        truth[index] = _advance(truth[index - 1], acceleration, dt)

    observations = truth[:history_steps] + rng.multivariate_normal(
        np.zeros(4), measurement_covariance, size=history_steps)
    observed = rng.random(history_steps) < 0.75
    observed[-1] = True
    observations[~observed] = np.nan
    result = fixed_lag_linear_gaussian_smoother(
        observations,
        prior_mean=np.asarray([400.0, 500.0, 6.0, -3.0]),
        prior_covariance=np.diag([25.0 ** 2, 25.0 ** 2, 5.0 ** 2, 5.0 ** 2]),
        transition=transition,
        process_covariance=process_covariance,
        measurement_covariance=measurement_covariance,
        observed=observed,
    )
    acceleration_estimate = estimate_cv_acceleration_from_residuals(
        result.process_residual, dt, lookback=5, recency=0.8)
    endpoint = result.filtered_mean[-1]
    cv_prediction = _forecast(endpoint, np.zeros(2), dt, horizon, 1.0)
    residual_prediction = _forecast(
        endpoint, acceleration_estimate, dt, horizon, decay=0.9)
    destination = truth[history_steps + horizon - 1]
    filtered_history_error = float(np.mean(np.sum(
        (result.filtered_mean[:-1, :2] - truth[:history_steps - 1, :2]) ** 2,
        axis=1)))
    smoothed_history_error = float(np.mean(np.sum(
        (result.smoothed_mean[:-1, :2] - truth[:history_steps - 1, :2]) ** 2,
        axis=1)))
    return {
        "filtered_history_mse": filtered_history_error,
        "smoothed_history_mse": smoothed_history_error,
        "cv_forecast_error": float(np.linalg.norm(cv_prediction[:2] - destination[:2])),
        "residual_forecast_error": float(
            np.linalg.norm(residual_prediction[:2] - destination[:2])),
    }


def _bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int = 4096,
) -> list[float]:
    indices = rng.integers(0, values.size, size=(samples, values.size))
    bootstrap_means = np.mean(values[indices], axis=1)
    return [float(item) for item in np.quantile(bootstrap_means, [0.025, 0.975])]


def benchmark(seed_count: int = 256) -> Dict[str, object]:
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    report: Dict[str, object] = {
        "scope": "mechanism_screen_not_model_selection_evidence",
        "seed_count": int(seed_count),
        "current_l4_acceleration_model": "white",
    }
    for mode, seed_offset in (("white", 1000), ("persistent", 2000)):
        rows = [_trial(np.random.default_rng(seed_offset + seed), mode)
                for seed in range(seed_count)]
        keys = rows[0].keys()
        means = {key: float(np.mean([row[key] for row in rows])) for key in keys}
        forecast_delta = np.asarray([
            row["cv_forecast_error"] - row["residual_forecast_error"]
            for row in rows
        ])
        history_delta = np.asarray([
            row["filtered_history_mse"] - row["smoothed_history_mse"]
            for row in rows
        ])
        report[mode] = {
            **means,
            "smoothing_history_mse_ratio": (
                means["smoothed_history_mse"] / means["filtered_history_mse"]),
            "smoothing_history_mean_improvement": float(np.mean(history_delta)),
            "smoothing_history_mean_improvement_ci95": _bootstrap_mean_interval(
                history_delta, np.random.default_rng(seed_offset + 91_000)),
            "residual_forecast_mean_improvement": float(np.mean(forecast_delta)),
            "residual_forecast_mean_improvement_ci95": _bootstrap_mean_interval(
                forecast_delta, np.random.default_rng(seed_offset + 92_000)),
            "residual_forecast_win_rate": float(np.mean(forecast_delta > 0.0)),
        }
    report["decision"] = (
        "retain_smoother_as_history_diagnostic; gate residual forecasting by "
        "measured temporal persistence"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.seed_count)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
