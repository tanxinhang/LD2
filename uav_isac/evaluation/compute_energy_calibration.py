"""Episode-level calibration of complete controller package-energy windows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ComputeEnergyEventObservation:
    episode_id: str
    event_id: str
    counter_start_j: float | None
    counter_end_j: float | None
    counter_resolution_j: float | None
    counter_wrap_count: int = 0
    counter_modulus_j: float | None = None
    window_censored: bool = False
    counter_difference_uncertainty_j: float | None = None


@dataclass(frozen=True)
class EpisodeComputeEnergyScore:
    episode_id: str
    package_energy_upper_j: float
    event_count: int


@dataclass(frozen=True)
class FrozenComputeEnergyEpoch:
    epoch_id: str
    miscoverage: float
    calibration_episode_ids: tuple[str, ...]
    training_episode_ids: tuple[str, ...]
    episode_scores_j: tuple[float, ...]
    package_energy_bound_j: float
    rank_one_based: int
    coverage_floor: float

    @property
    def finite(self) -> bool:
        return math.isfinite(self.package_energy_bound_j)


@dataclass(frozen=True)
class ComputeEnergyValidation:
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


def event_package_energy_upper_j(
    observation: ComputeEnergyEventObservation,
) -> float:
    """Return a counter-resolution-safe package-energy window.

    The caller must record the exact counter wrap count.  Guessing a wrap from
    ``end < start`` is not accepted because resets and multiple wraps are
    observationally different.  Adding one declared counter resolution is a
    conservative quantization bound on the difference of two readings.  A
    separately declared complete-window metrology uncertainty is added once;
    counter resolution alone is not treated as instrument accuracy.
    """
    if not isinstance(observation.window_censored, (bool, np.bool_)):
        raise ValueError("window_censored must be boolean")
    wraps_raw = observation.counter_wrap_count
    if isinstance(wraps_raw, (bool, np.bool_)):
        raise ValueError("counter_wrap_count must be an integer, not bool")
    wraps = int(wraps_raw)
    if wraps != wraps_raw or wraps < 0:
        raise ValueError("counter_wrap_count must be a non-negative integer")
    if bool(observation.window_censored):
        return float("inf")
    if any(value is None for value in (
        observation.counter_start_j,
        observation.counter_end_j,
        observation.counter_resolution_j,
    )):
        raise ValueError("uncensored energy windows require both readings/resolution")
    start = float(observation.counter_start_j)
    end = float(observation.counter_end_j)
    resolution = float(observation.counter_resolution_j)
    if (
        not np.isfinite(start) or start < 0.0
        or not np.isfinite(end) or end < 0.0
        or not np.isfinite(resolution) or resolution <= 0.0
    ):
        raise ValueError(
            "energy readings must be finite non-negative and resolution positive")
    modulus = observation.counter_modulus_j
    if wraps > 0:
        if modulus is None:
            raise ValueError("positive wrap count requires counter_modulus_j")
        modulus_value = float(modulus)
        if (
            not np.isfinite(modulus_value) or modulus_value <= 0.0
            or start >= modulus_value or end >= modulus_value
        ):
            raise ValueError(
                "wrapped readings must lie in a finite positive modulus")
    else:
        modulus_value = 0.0
        if modulus is not None:
            supplied_modulus = float(modulus)
            if not np.isfinite(supplied_modulus) or supplied_modulus <= 0.0:
                raise ValueError("counter_modulus_j must be finite positive")
            if start >= supplied_modulus or end >= supplied_modulus:
                raise ValueError("counter readings exceed the supplied modulus")
        if end < start:
            raise ValueError(
                "decreasing counter requires an explicit positive wrap count")
    delta = end - start + wraps * modulus_value
    if delta < 0.0 or not np.isfinite(delta):
        raise ValueError("counter delta is invalid")
    uncertainty = (
        0.0 if observation.counter_difference_uncertainty_j is None
        else float(observation.counter_difference_uncertainty_j)
    )
    if not np.isfinite(uncertainty) or uncertainty < 0.0:
        raise ValueError(
            "counter_difference_uncertainty_j must be finite non-negative")
    return float(delta + resolution + uncertainty)


def episode_compute_energy_scores(
    observations: Iterable[ComputeEnergyEventObservation],
    *,
    expected_episode_ids: Sequence[str],
) -> tuple[EpisodeComputeEnergyScore, ...]:
    expected = _normalize_ids(expected_episode_ids, "expected_episode_ids")
    expected_set = set(expected)
    grouped: dict[str, list[float]] = {
        episode_id: [] for episode_id in expected
    }
    event_keys: set[tuple[str, str]] = set()
    for item in observations:
        episode_id = str(item.episode_id)
        event_id = str(item.event_id)
        if episode_id not in expected_set:
            raise ValueError(
                f"energy observation is outside frozen split: {episode_id}")
        key = (episode_id, event_id)
        if key in event_keys:
            raise ValueError(f"duplicate energy event: {key}")
        event_keys.add(key)
        grouped[episode_id].append(event_package_energy_upper_j(item))
    missing = [episode_id for episode_id, values in grouped.items() if not values]
    if missing:
        raise ValueError(
            f"frozen episodes have no energy observations: {missing}")
    return tuple(
        EpisodeComputeEnergyScore(
            episode_id=episode_id,
            package_energy_upper_j=float(max(grouped[episode_id])),
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
        raise ValueError("at least one energy calibration episode is required")
    if (
        np.any(np.isnan(values)) or np.any(np.isneginf(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("energy scores must be non-negative or positive infinity")
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


def calibrate_frozen_compute_energy_epoch(
    observations: Iterable[ComputeEnergyEventObservation],
    *,
    calibration_episode_ids: Sequence[str],
    training_episode_ids: Sequence[str] = (),
    miscoverage: float,
    epoch_id: str,
) -> FrozenComputeEnergyEpoch:
    calibration_ids = _normalize_ids(
        calibration_episode_ids, "calibration_episode_ids")
    training_ids = tuple(str(value) for value in training_episode_ids)
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("training_episode_ids must contain unique IDs")
    overlap = set(training_ids) & set(calibration_ids)
    if overlap:
        raise ValueError(
            f"training/energy-calibration episode leakage: {sorted(overlap)}")
    episode_scores = episode_compute_energy_scores(
        observations, expected_episode_ids=calibration_ids)
    raw_scores = tuple(score.package_energy_upper_j for score in episode_scores)
    bound, coverage, rank = _split_upper(
        raw_scores, miscoverage=float(miscoverage))
    return FrozenComputeEnergyEpoch(
        epoch_id=str(epoch_id),
        miscoverage=float(miscoverage),
        calibration_episode_ids=calibration_ids,
        training_episode_ids=training_ids,
        episode_scores_j=raw_scores,
        package_energy_bound_j=float(bound),
        rank_one_based=int(rank),
        coverage_floor=float(coverage),
    )


def validate_frozen_compute_energy_epoch(
    epoch: FrozenComputeEnergyEpoch,
    observations: Iterable[ComputeEnergyEventObservation],
    *,
    validation_episode_ids: Sequence[str],
) -> ComputeEnergyValidation:
    validation_ids = _normalize_ids(
        validation_episode_ids, "validation_episode_ids")
    overlap = (
        set(epoch.training_episode_ids) | set(epoch.calibration_episode_ids)
    ) & set(validation_ids)
    if overlap:
        raise ValueError(
            f"energy train/calibration/validation leakage: {sorted(overlap)}")
    scores = episode_compute_energy_scores(
        observations, expected_episode_ids=validation_ids)
    failures = tuple(
        score.episode_id for score in scores
        if score.package_energy_upper_j > epoch.package_energy_bound_j
    )
    return ComputeEnergyValidation(
        validation_episode_ids=validation_ids,
        failure_episode_ids=failures,
    )
