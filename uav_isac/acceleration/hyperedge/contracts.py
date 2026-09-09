"""Public contract for replaceable hyperedge coefficient acceleration."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from uav_isac.coordination.hyperedge import SelectedBistaticCoefficients


@runtime_checkable
class HyperedgeAccelerationService(Protocol):
    """Backend-neutral coefficient reconstruction service.

    Positional and keyword arguments intentionally mirror the audited pure
    physical kernels.  A replacement backend must preserve their array shapes,
    visibility masking and conservative lower/upper-bound semantics.
    """

    @property
    def name(self) -> str:
        """Stable backend identity used in manifests and diagnostics."""

    def reconstruct_dense(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return one dense nominal or robust ``(K,K,Q)`` coefficient view."""

    def reconstruct_upper(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return one dense conservative upper coefficient view."""

    def reconstruct_selected(
        self, *args: Any, **kwargs: Any,
    ) -> SelectedBistaticCoefficients:
        """Return joint nominal/lower/upper values for selected edges."""

    def reconstruct_selected_batch(
        self, *args: Any, **kwargs: Any,
    ) -> SelectedBistaticCoefficients:
        """Return selected values for all viewer-local public states."""

    def reset_stats(self) -> None:
        """Reset cumulative service counters and wall times."""

    def stats(self) -> dict[str, float | int | str]:
        """Return a serializable cumulative diagnostic snapshot."""
