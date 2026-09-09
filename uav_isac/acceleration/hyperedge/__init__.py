"""Drop-in hyperedge coefficient acceleration service package.

Typical use::

    from uav_isac.acceleration.hyperedge import (
        create_hyperedge_acceleration_service,
    )
    service = create_hyperedge_acceleration_service("numpy")

Custom backends register a zero-argument factory, then select its name through
``marl.hyperedge_acceleration_backend``.
"""

from .contracts import HyperedgeAccelerationService
from .numpy_backend import NumpyHyperedgeAccelerationService
from .registry import (
    available_hyperedge_acceleration_backends,
    create_hyperedge_acceleration_service,
    register_hyperedge_acceleration_backend,
)

__all__ = [
    "HyperedgeAccelerationService",
    "NumpyHyperedgeAccelerationService",
    "available_hyperedge_acceleration_backends",
    "create_hyperedge_acceleration_service",
    "register_hyperedge_acceleration_backend",
]
