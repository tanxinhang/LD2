"""Episode-level calibration of complete controller compute critical paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RuntimeLatencyEventObservation:
    episode_id: str
    event_id: str
    observed_complete_compute_latency_s: float | None
    timer_resolution_s: float | None
    timer_difference_uncertainty_s: float | None = None
    window_censored: bool = False


@dataclass(frozen=True)
class EpisodeRuntimeLatencyScore:
    episode_id: str
    complete_compute_latency_upper_s: float
    event_count: int


@dataclass(frozen=True)
class FrozenRuntimeLatencyEpoch:
    epoch_id: str
    miscoverage: float
    calibration_episode_ids: tuple[str, ...]
    training_episode_ids: tuple[str, ...]
    episode_scores_s: tuple[float, ...]
    complete_compute_latency_bound_s: float
    rank_one_based: int
    coverage_floor: float

    @property
    def finite(self) -> bool:
        return math.isfinite(self.complete_compute_latency_bound_s)


@dataclass(frozen=True)
class RuntimeLatencyValidation:
    validation_episode_ids: tuple[str, ...]
    failure_episode_ids: tuple[str, ...]

    @property
    def episode_count(self) -> int:
        return len(self.validation_episode_ids)

    @property
    def failure_count(self) -> int:
        return len(self.failure_episode_ids)


def _normalize_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def event_complete_compute_latency_upper_s(
    observation: RuntimeLatencyEventObservation,
) -> float:
    """Return a timer/metrology-safe complete compute critical-path window."""
    if not isinstance(observation.window_censored, (bool, np.bool_)):
        raise ValueError("window_censored must be boolean")
    if bool(observation.window_censored):
        return float("inf")
    if (
        observation.observed_complete_compute_latency_s is None
        or observation.timer_resolution_s is None
    ):
        raise ValueError("uncensored runtime windows require latency/resolution")
    observed = float(observation.observed_complete_compute_latency_s)
    resolution = float(observation.timer_resolution_s)
    uncertainty = (
        0.0 if observation.timer_difference_uncertainty_s is None
        else float(observation.timer_difference_uncertainty_s))
    if (
        not np.isfinite(observed) or observed < 0.0
        or not np.isfinite(resolution) or resolution <= 0.0
        or not np.isfinite(uncertainty) or uncertainty < 0.0
    ):
        raise ValueError("runtime latency/metrology values are outside support")
    return float(observed + resolution + uncertainty)


def episode_runtime_latency_scores(
    observations: Iterable[RuntimeLatencyEventObservation],
    *,
    expected_episode_ids: Sequence[str],
) -> tuple[EpisodeRuntimeLatencyScore, ...]:
    expected = _normalize_ids(expected_episode_ids, "expected_episode_ids")
    expected_set = set(expected)
    grouped: dict[str, list[float]] = {value: [] for value in expected}
    event_keys: set[tuple[str, str]] = set()
    for item in observations:
        episode_id = str(item.episode_id)
        event_id = str(item.event_id)
        if episode_id not in expected_set:
            raise ValueError(
                f"runtime observation is outside frozen split: {episode_id}")
        key = (episode_id, event_id)
        if key in event_keys:
            raise ValueError(f"duplicate runtime event: {key}")
        event_keys.add(key)
        grouped[episode_id].append(
            event_complete_compute_latency_upper_s(item))
    missing = [key for key, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"frozen episodes have no runtime observations: {missing}")
    return tuple(
        EpisodeRuntimeLatencyScore(
            episode_id=episode_id,
            complete_compute_latency_upper_s=float(max(grouped[episode_id])),
            event_count=len(grouped[episode_id]),
        )
        for episode_id in expected
    )


def _split_upper(
    scores: Sequence[float], *, miscoverage: float,
) -> tuple[float, float, int]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    alpha = float(miscoverage)
    if not values.size:
        raise ValueError("at least one runtime calibration episode is required")
    if (
        np.any(np.isnan(values)) or np.any(np.isneginf(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("runtime scores must be non-negative or infinity")
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    if rank > values.size:
        return float("inf"), 1.0, values.size + 1
    return (
        float(np.partition(values, rank - 1)[rank - 1]),
        float(rank / (values.size + 1)),
        rank,
    )


def calibrate_frozen_runtime_latency_epoch(
    observations: Iterable[RuntimeLatencyEventObservation],
    *,
    calibration_episode_ids: Sequence[str],
    training_episode_ids: Sequence[str] = (),
    miscoverage: float,
    epoch_id: str,
) -> FrozenRuntimeLatencyEpoch:
    calibration_ids = _normalize_ids(
        calibration_episode_ids, "calibration_episode_ids")
    training_ids = tuple(str(value) for value in training_episode_ids)
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("training_episode_ids must contain unique IDs")
    overlap = set(training_ids) & set(calibration_ids)
    if overlap:
        raise ValueError(
            f"training/runtime-calibration episode leakage: {sorted(overlap)}")
    scores = episode_runtime_latency_scores(
        observations, expected_episode_ids=calibration_ids)
    raw_scores = tuple(
        score.complete_compute_latency_upper_s for score in scores)
    bound, coverage, rank = _split_upper(
        raw_scores, miscoverage=float(miscoverage))
    return FrozenRuntimeLatencyEpoch(
        epoch_id=str(epoch_id),
        miscoverage=float(miscoverage),
        calibration_episode_ids=calibration_ids,
        training_episode_ids=training_ids,
        episode_scores_s=raw_scores,
        complete_compute_latency_bound_s=float(bound),
        rank_one_based=int(rank),
        coverage_floor=float(coverage),
    )


def validate_frozen_runtime_latency_epoch(
    epoch: FrozenRuntimeLatencyEpoch,
    observations: Iterable[RuntimeLatencyEventObservation],
    *,
    validation_episode_ids: Sequence[str],
) -> RuntimeLatencyValidation:
    validation_ids = _normalize_ids(
        validation_episode_ids, "validation_episode_ids")
    overlap = (
        set(epoch.training_episode_ids) | set(epoch.calibration_episode_ids)
    ) & set(validation_ids)
    if overlap:
        raise ValueError(
            f"runtime train/calibration/validation leakage: {sorted(overlap)}")
    scores = episode_runtime_latency_scores(
        observations, expected_episode_ids=validation_ids)
    failures = tuple(
        score.episode_id for score in scores
        if score.complete_compute_latency_upper_s
        > epoch.complete_compute_latency_bound_s
    )
    return RuntimeLatencyValidation(
        validation_episode_ids=validation_ids,
        failure_episode_ids=failures,
    )
