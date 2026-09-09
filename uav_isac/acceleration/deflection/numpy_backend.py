"""Audited NumPy dense-deflection materialization backend."""

from __future__ import annotations

from uav_isac.physical.deflection import DenseDeflection


class NumpyDeflectionMaterializationService:
    """Delegate to the vectorized canonical ``DenseDeflection`` converter."""

    name = "numpy"
    # Static binding keeps the service boundary at construction/configuration
    # time while making the normal call path identical to invoking the audited
    # converter directly.
    materialize = staticmethod(DenseDeflection.to_entries)
