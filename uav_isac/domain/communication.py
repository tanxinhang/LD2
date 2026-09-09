"""Domain ports for inter-UAV transport used by coordination primitives."""

from __future__ import annotations

from typing import Any, Protocol


class CommunicationStepStatsLike(Protocol):
    """Marker port for transport statistics returned by legacy adapters."""


class CommunicationTransport(Protocol):
    """Structural interface consumed by distributed coordination."""

    bandwidth_hz: float
    deadline_s: float
    processing_delay_s: float
    snr_threshold_db: float
    dt: float
    message_dim: int

    def link_budget(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def robust_link_budget(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def transmit(self, *args: Any, **kwargs: Any) -> Any:
        ...

