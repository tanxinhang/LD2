"""Replaceable dense-deflection materialization service."""

from .contracts import DeflectionMaterializationService
from .numpy_backend import NumpyDeflectionMaterializationService
from .registry import (
    available_deflection_materialization_backends,
    create_deflection_materialization_service,
    register_deflection_materialization_backend,
)

__all__ = [
    "DeflectionMaterializationService",
    "NumpyDeflectionMaterializationService",
    "available_deflection_materialization_backends",
    "create_deflection_materialization_service",
    "register_deflection_materialization_backend",
]
