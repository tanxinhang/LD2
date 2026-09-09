"""Stable domain-facing contracts for architecture V2."""

from .episode import EpisodeResult, EpisodeSpec, FrameResult
from .artifacts import ArtifactRecord, RunManifest
from .communication import CommunicationStepStatsLike, CommunicationTransport
from .observation import ObservationLayout
from .action import ActionSpace, DP_PARAM_RADIAL_CLIP, DP_PARAM_SMOOTH_DISK
from .observation_slices import ObservationSlices

__all__ = [
    "ArtifactRecord",
    "EpisodeResult",
    "EpisodeSpec",
    "FrameResult",
    "RunManifest",
    "CommunicationStepStatsLike",
    "CommunicationTransport",
    "ObservationLayout",
    "ActionSpace",
    "DP_PARAM_RADIAL_CLIP",
    "DP_PARAM_SMOOTH_DISK",
    "ObservationSlices",
]
