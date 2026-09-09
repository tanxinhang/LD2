"""Contract for replaceable dense-deflection materialization."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from uav_isac.physical.deflection import DenseDeflection
from uav_isac.utils.types import DeflectionEntry


@runtime_checkable
class DeflectionMaterializationService(Protocol):
    """Backend-neutral conversion from dense tensors to legacy entries."""

    @property
    def name(self) -> str:
        """Stable backend identity."""

    def materialize(
        self,
        dense: DenseDeflection,
        power_scale_w: np.ndarray | None = None,
    ) -> list[DeflectionEntry]:
        """Return entries in canonical ``i -> j -> q`` order."""
