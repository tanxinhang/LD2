"""Replaceable computation backends for simulation hot paths."""

from .hyperedge import (
    HyperedgeAccelerationService,
    NumpyHyperedgeAccelerationService,
    available_hyperedge_acceleration_backends,
    create_hyperedge_acceleration_service,
    register_hyperedge_acceleration_backend,
)
from .deflection import (
    DeflectionMaterializationService,
    NumpyDeflectionMaterializationService,
    available_deflection_materialization_backends,
    create_deflection_materialization_service,
    register_deflection_materialization_backend,
)

__all__ = [
    "HyperedgeAccelerationService",
    "NumpyHyperedgeAccelerationService",
    "available_hyperedge_acceleration_backends",
    "create_hyperedge_acceleration_service",
    "register_hyperedge_acceleration_backend",
    "DeflectionMaterializationService",
    "NumpyDeflectionMaterializationService",
    "available_deflection_materialization_backends",
    "create_deflection_materialization_service",
    "register_deflection_materialization_backend",
]
