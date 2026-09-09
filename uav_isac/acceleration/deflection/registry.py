"""Registry for dense-deflection materialization services."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from .contracts import DeflectionMaterializationService
from .numpy_backend import NumpyDeflectionMaterializationService


BackendFactory = Callable[[], DeflectionMaterializationService]
_BACKENDS: dict[str, BackendFactory] = {
    "numpy": NumpyDeflectionMaterializationService,
}


def register_deflection_materialization_backend(
    name: str,
    factory: BackendFactory,
    *,
    replace: bool = False,
) -> None:
    key = str(name).strip().lower()
    if not key or not callable(factory):
        raise ValueError("backend name and factory must be valid")
    if key in _BACKENDS and not replace:
        raise ValueError(
            f"deflection materialization backend already exists: {key}")
    _BACKENDS[key] = factory


def available_deflection_materialization_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def create_deflection_materialization_service(
    name: str = "numpy",
) -> DeflectionMaterializationService:
    requested = str(name).strip()
    key = requested.lower()
    if key in _BACKENDS:
        factory = _BACKENDS[key]
    elif ":" in requested:
        module_name, attribute_name = requested.rsplit(":", 1)
        if not module_name or not attribute_name:
            raise ValueError(
                "external deflection backend must be 'module:factory'")
        try:
            factory = getattr(import_module(module_name), attribute_name)
        except (ImportError, AttributeError) as error:
            raise ValueError(
                f"cannot load deflection materialization backend {requested!r}"
            ) from error
        if not callable(factory):
            raise TypeError(
                f"deflection materialization factory {requested!r} "
                "is not callable")
    else:
        available = ", ".join(
            available_deflection_materialization_backends())
        raise ValueError(
            f"unknown deflection materialization backend {key!r}; "
            f"available: {available}")
    service = factory()
    if not isinstance(service, DeflectionMaterializationService):
        raise TypeError(
            f"backend {requested!r} does not implement "
            "DeflectionMaterializationService")
    return service
