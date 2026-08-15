"""Finite-sample feasibility primitives for frozen ISAC certificates."""

from __future__ import annotations

from dataclasses import dataclass
import math

from scipy.stats import beta as beta_distribution


def _miscoverage(value: float) -> float:
    alpha = float(value)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie in (0,1)")
    return alpha


def split_conformal_rank(
    calibration_episode_count: int, miscoverage: float,
) -> int:
    """Return the one-based split-conformal upper-quantile rank."""

    count = int(calibration_episode_count)
    if count <= 0:
        raise ValueError("calibration episode count must be positive")
    alpha = _miscoverage(miscoverage)
    return int(math.ceil((count + 1) * (1.0 - alpha)))


def minimum_split_conformal_calibration_episodes(
    miscoverage: float,
) -> int:
    """Smallest episode count for which the conformal rank is finite."""

    alpha = _miscoverage(miscoverage)
    count = max(1, int(math.ceil((1.0 - alpha) / alpha)))
    while split_conformal_rank(count, alpha) > count:
        count += 1
    while (
        count > 1
        and split_conformal_rank(count - 1, alpha) <= count - 1
    ):
        count -= 1
    return count


def clopper_pearson_one_sided_upper(
    failures: int,
    episode_count: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Exact one-sided binomial upper bound for an episode failure rate."""

    failed = int(failures)
    count = int(episode_count)
    level = float(confidence)
    if count <= 0 or failed < 0 or failed > count:
        raise ValueError("invalid failure count for exact binomial bound")
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if failed == count:
        return 1.0
    return float(beta_distribution.ppf(level, failed + 1, count - failed))


def minimum_zero_failure_validation_episodes(
    miscoverage: float,
    *,
    confidence: float = 0.95,
) -> int:
    """Best-case validation size needed for an exact upper <= risk.

    With zero failures the exact upper is ``1-(1-confidence)**(1/n)``.
    This is a necessary sample-size condition; observed failures can require
    more data or make the frozen epoch fail validation entirely.
    """

    alpha = _miscoverage(miscoverage)
    level = float(confidence)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    count = max(1, int(math.ceil(
        math.log1p(-level) / math.log1p(-alpha))))
    while clopper_pearson_one_sided_upper(
        0, count, confidence=level,
    ) > alpha + 1.0e-15:
        count += 1
    while (
        count > 1
        and clopper_pearson_one_sided_upper(
            0, count - 1, confidence=level,
        ) <= alpha + 1.0e-15
    ):
        count -= 1
    return count


@dataclass(frozen=True)
class FiniteSampleRequirement:
    label: str
    stage: str
    miscoverage: float
    observed_episode_count: int
    minimum_episode_count: int
    attainable_in_best_case: bool
    rank_one_based: int | None
    confidence: float | None


def conformal_sample_requirement(
    label: str,
    *,
    miscoverage: float,
    calibration_episode_count: int,
) -> FiniteSampleRequirement:
    count = int(calibration_episode_count)
    if count <= 0:
        raise ValueError("calibration episode count must be positive")
    alpha = _miscoverage(miscoverage)
    minimum = minimum_split_conformal_calibration_episodes(alpha)
    rank = split_conformal_rank(count, alpha)
    return FiniteSampleRequirement(
        label=str(label),
        stage="split_conformal_calibration",
        miscoverage=alpha,
        observed_episode_count=count,
        minimum_episode_count=minimum,
        attainable_in_best_case=bool(rank <= count),
        rank_one_based=rank,
        confidence=None,
    )


def zero_failure_validation_requirement(
    label: str,
    *,
    miscoverage: float,
    validation_episode_count: int,
    confidence: float = 0.95,
) -> FiniteSampleRequirement:
    count = int(validation_episode_count)
    if count <= 0:
        raise ValueError("validation episode count must be positive")
    alpha = _miscoverage(miscoverage)
    minimum = minimum_zero_failure_validation_episodes(
        alpha, confidence=confidence)
    return FiniteSampleRequirement(
        label=str(label),
        stage="independent_exact_binomial_validation_zero_failure_best_case",
        miscoverage=alpha,
        observed_episode_count=count,
        minimum_episode_count=minimum,
        attainable_in_best_case=bool(count >= minimum),
        rank_one_based=None,
        confidence=float(confidence),
    )
