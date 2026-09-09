"""Framework-neutral values returned by an episode use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class EpisodeSpec:
    """Inputs that determine one bounded simulation episode."""

    seed: int
    max_frames: int

    def __post_init__(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")


@dataclass(frozen=True)
class FrameResult:
    """One immutable application-level frame result.

    Array payloads intentionally remain opaque here. Shape and unit validation
    belongs to domain-specific contracts introduced as each subsystem migrates.
    """

    index: int
    observations: Mapping[str, Any]
    rewards: Mapping[str, float]
    terminated: Mapping[str, bool]
    truncated: Mapping[str, bool]
    info: Mapping[str, Any]


@dataclass(frozen=True)
class EpisodeResult:
    """Result of the canonical episode application use case."""

    seed: int
    frames: Tuple[FrameResult, ...]
    initial_info: Mapping[str, Any]

    @property
    def completed(self) -> bool:
        if not self.frames:
            return False
        last = self.frames[-1]
        return bool(
            last.terminated.get("__all__", False)
            or last.truncated.get("__all__", False)
        )

