"""Domain port for extracting local state from a flat observation layout."""

from __future__ import annotations

from typing import Any, Protocol


class ObservationLayout(Protocol):
    K: int
    Q: int

    def extract_self(self, observation: Any) -> Any:
        ...

    def extract_beliefs(self, observation: Any) -> Any:
        ...

