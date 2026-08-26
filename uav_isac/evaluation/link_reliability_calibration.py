"""Episode-level calibration for bounded U2U erasures and excess queueing.

The physical link model must first remove observed-SNR Shannon serialization
and known processing from the measured complete-protocol latency.  This module
then calibrates only two orthogonal episode maxima:

``S_e``: erased copies before first delivery of any required logical packet;
``S_q``: positive complete-protocol excess queue/scheduling latency.

Separate split-conformal upper bounds with risks ``alpha_e`` and ``alpha_q``
give simultaneous episode coverage at least ``1-alpha_e-alpha_q`` by the
union bound.  No independence between erasures and queueing is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LinkReliabilityEventObservation:
    episode_id: str
    event_id: str
    erased_copies_before_success: int
    observed_complete_protocol_latency_s: float | None
    observed_snr_modeled_protocol_latency_s: float | None
    delivery_censored: bool = False


@dataclass(frozen=True)
class EpisodeLinkReliabilityScore:
    episode_id: str
    erasure_score: float
    excess_queue_score_s: float
    event_count: int


@dataclass(frozen=True)
class FrozenLinkReliabilityEpoch:
    epoch_id: str
    total_miscoverage: float
    erasure_miscoverage: float
    queue_miscoverage: float
    calibration_episode_ids: tuple[str, ...]
    training_episode_ids: tuple[str, ...]
    erasure_episode_scores: tuple[float, ...]
    queue_episode_scores_s: tuple[float, ...]
    erasure_bound: float
    queue_bound_s: float
    erasure_rank_one_based: int
    queue_rank_one_based: int
    erasure_coverage_floor: float
    queue_coverage_floor: float

    @property
    def allocated_miscoverage(self) -> float:
        return float(self.erasure_miscoverage + self.queue_miscoverage)

    @property
    def joint_coverage_floor(self) -> float:
        return float(max(0.0, 1.0 - self.allocated_miscoverage))

    @property
    def finite(self) -> bool:
        return bool(
            math.isfinite(self.erasure_bound)
            and math.isfinite(self.queue_bound_s)
        )

    @property
    def recommended_repetition_count(self) -> int | None:
        if not self.finite:
            return None
        return int(self.erasure_bound) + 1


@dataclass(frozen=True)
class LinkReliabilityValidation:
    validation_episode_ids: tuple[str, ...]
    erasure_failure_episode_ids: tuple[str, ...]
    queue_failure_episode_ids: tuple[str, ...]
    joint_failure_episode_ids: tuple[str, ...]

    @property
    def episode_count(self) -> int:
        return len(self.validation_episode_ids)

    @property
    def joint_failure_count(self) -> int:
        return len(self.joint_failure_episode_ids)


def _normalize_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def episode_link_reliability_scores(
    observations: Iterable[LinkReliabilityEventObservation],
    *,
    expected_episode_ids: Sequence[str],
) -> tuple[EpisodeLinkReliabilityScore, ...]:
    """Reduce every correlated event collection to one score per episode.

    ``observed_snr_modeled_protocol_latency_s`` must already contain the
    complete nominal protocol latency recomputed from observed SNR.  Hence the
    queue score cannot double-charge a fade already represented by Shannon
    serialization.  A censored delivery makes both components infinite:
    neither required repetitions nor complete latency are then observed.
    """
    expected = _normalize_ids(expected_episode_ids, "expected_episode_ids")
    expected_set = set(expected)
    grouped: dict[str, list[tuple[float, float]]] = {
        episode_id: [] for episode_id in expected
    }
    event_keys: set[tuple[str, str]] = set()
    for item in observations:
        episode_id = str(item.episode_id)
        event_id = str(item.event_id)
        if episode_id not in expected_set:
            raise ValueError(
                f"observation episode is outside the frozen split: {episode_id}")
        key = (episode_id, event_id)
        if key in event_keys:
            raise ValueError(f"duplicate link event: {key}")
        event_keys.add(key)
        erased = item.erased_copies_before_success
        if isinstance(erased, (bool, np.bool_)):
            raise ValueError("erased copy count must be an integer, not bool")
        erased_count = int(erased)
        if erased_count != erased or erased_count < 0:
            raise ValueError(
                "erased_copies_before_success must be a non-negative integer")
        if not isinstance(item.delivery_censored, (bool, np.bool_)):
            raise ValueError("delivery_censored must be boolean")
        if bool(item.delivery_censored):
            grouped[episode_id].append((float("inf"), float("inf")))
            continue
        if (
            item.observed_complete_protocol_latency_s is None
            or item.observed_snr_modeled_protocol_latency_s is None
        ):
            raise ValueError(
                "uncensored events require observed and modeled latency")
        observed = float(item.observed_complete_protocol_latency_s)
        modeled = float(item.observed_snr_modeled_protocol_latency_s)
        if (
            not np.isfinite(observed) or observed < 0.0
            or not np.isfinite(modeled) or modeled < 0.0
        ):
            raise ValueError(
                "uncensored observed/modeled latency must be finite non-negative")
        grouped[episode_id].append((
            float(erased_count),
            float(max(observed - modeled, 0.0)),
        ))
    missing = [episode_id for episode_id, values in grouped.items() if not values]
    if missing:
        raise ValueError(
            f"frozen episodes have no link observations: {missing}")
    return tuple(
        EpisodeLinkReliabilityScore(
            episode_id=episode_id,
            erasure_score=float(max(value[0] for value in grouped[episode_id])),
            excess_queue_score_s=float(max(
                value[1] for value in grouped[episode_id])),
            event_count=len(grouped[episode_id]),
        )
        for episode_id in expected
    )


def _split_conformal_upper_with_infinity(
    scores: Sequence[float], *, miscoverage: float,
) -> tuple[float, float, int]:
    """One-sided split-conformal bound with an explicit infinity sentinel."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    alpha = float(miscoverage)
    if not values.size:
        raise ValueError("at least one calibration episode is required")
    if (
        np.any(np.isnan(values)) or np.any(np.isneginf(values))
        or np.any(values < 0.0)
    ):
        raise ValueError(
            "calibration scores must be non-negative or positive infinity")
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    if rank > values.size:
        return float("inf"), 1.0, values.size + 1
    bound = float(np.partition(values, rank - 1)[rank - 1])
    return bound, float(rank / (values.size + 1)), rank


def calibrate_frozen_link_reliability_epoch(
    observations: Iterable[LinkReliabilityEventObservation],
    *,
    calibration_episode_ids: Sequence[str],
    training_episode_ids: Sequence[str] = (),
    total_miscoverage: float,
    erasure_miscoverage: float,
    queue_miscoverage: float,
    epoch_id: str,
) -> FrozenLinkReliabilityEpoch:
    """Freeze episode-level repetition and complete-path queue bounds."""
    calibration_ids = _normalize_ids(
        calibration_episode_ids, "calibration_episode_ids")
    training_ids = tuple(str(value) for value in training_episode_ids)
    if len(set(training_ids)) != len(training_ids):
        raise ValueError("training_episode_ids must contain unique IDs")
    overlap = set(training_ids) & set(calibration_ids)
    if overlap:
        raise ValueError(
            f"training/calibration episode leakage: {sorted(overlap)}")
    total_alpha = float(total_miscoverage)
    erasure_alpha = float(erasure_miscoverage)
    queue_alpha = float(queue_miscoverage)
    if not 0.0 < total_alpha < 1.0:
        raise ValueError("total_miscoverage must lie in (0,1)")
    if not 0.0 < erasure_alpha < 1.0 or not 0.0 < queue_alpha < 1.0:
        raise ValueError("component miscoverages must lie in (0,1)")
    if erasure_alpha + queue_alpha > total_alpha + 1.0e-15:
        raise ValueError(
            "erasure plus queue risk exceeds the total link risk budget")
    episode_scores = episode_link_reliability_scores(
        observations, expected_episode_ids=calibration_ids)
    erasure_scores = tuple(score.erasure_score for score in episode_scores)
    queue_scores = tuple(
        score.excess_queue_score_s for score in episode_scores)
    erasure_bound, erasure_coverage, erasure_rank = (
        _split_conformal_upper_with_infinity(
            erasure_scores, miscoverage=erasure_alpha)
    )
    queue_bound, queue_coverage, queue_rank = (
        _split_conformal_upper_with_infinity(
            queue_scores, miscoverage=queue_alpha)
    )
    if math.isfinite(erasure_bound) and erasure_bound != int(erasure_bound):
        raise RuntimeError("erasure conformal bound lost integer support")
    return FrozenLinkReliabilityEpoch(
        epoch_id=str(epoch_id),
        total_miscoverage=total_alpha,
        erasure_miscoverage=erasure_alpha,
        queue_miscoverage=queue_alpha,
        calibration_episode_ids=calibration_ids,
        training_episode_ids=training_ids,
        erasure_episode_scores=erasure_scores,
        queue_episode_scores_s=queue_scores,
        erasure_bound=float(erasure_bound),
        queue_bound_s=float(queue_bound),
        erasure_rank_one_based=int(erasure_rank),
        queue_rank_one_based=int(queue_rank),
        erasure_coverage_floor=float(erasure_coverage),
        queue_coverage_floor=float(queue_coverage),
    )


def validate_frozen_link_reliability_epoch(
    epoch: FrozenLinkReliabilityEpoch,
    observations: Iterable[LinkReliabilityEventObservation],
    *,
    validation_episode_ids: Sequence[str],
) -> LinkReliabilityValidation:
    """Evaluate the already frozen bounds on disjoint complete episodes."""
    validation_ids = _normalize_ids(
        validation_episode_ids, "validation_episode_ids")
    forbidden = (
        set(epoch.training_episode_ids) | set(epoch.calibration_episode_ids)
    )
    overlap = forbidden & set(validation_ids)
    if overlap:
        raise ValueError(
            f"training/calibration/validation episode leakage: {sorted(overlap)}")
    scores = episode_link_reliability_scores(
        observations, expected_episode_ids=validation_ids)
    # Audit 2026-08-17: non-finite bounds (rank > n, or censored events storing
    # inf scores) made the comparisons never True -> zero failures.  Fail closed:
    # an uncalibratable epoch fails every validation episode.
    erasure_bound = epoch.erasure_bound
    queue_bound = epoch.queue_bound_s
    if not math.isfinite(erasure_bound) or not math.isfinite(queue_bound):
        erasure_failures = tuple(score.episode_id for score in scores)
        queue_failures = erasure_failures
    else:
        erasure_failures = tuple(
            score.episode_id for score in scores
            if score.erasure_score > erasure_bound
        )
        queue_failures = tuple(
            score.episode_id for score in scores
            if score.excess_queue_score_s > queue_bound
        )
    joint_failures = tuple(
        episode_id for episode_id in validation_ids
        if episode_id in set(erasure_failures) | set(queue_failures)
    )
    return LinkReliabilityValidation(
        validation_episode_ids=validation_ids,
        erasure_failure_episode_ids=erasure_failures,
        queue_failure_episode_ids=queue_failures,
        joint_failure_episode_ids=joint_failures,
    )
