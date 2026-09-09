"""Registry and factory for hyperedge acceleration services."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from .contracts import HyperedgeAccelerationService
from .numpy_backend import NumpyHyperedgeAccelerationService


BackendFactory = Callable[[], HyperedgeAccelerationService]
_BACKENDS: dict[str, BackendFactory] = {
    "numpy": NumpyHyperedgeAccelerationService,
}


def register_hyperedge_acceleration_backend(
    name: str,
    factory: BackendFactory,
    *,
    replace: bool = False,
) -> None:
    """Register an in-process backend factory under a stable lowercase name."""
    key = str(name).strip().lower()
    if not key or not callable(factory):
        raise ValueError("backend name and factory must be valid")
    if key in _BACKENDS and not replace:
        raise ValueError(f"hyperedge acceleration backend already exists: {key}")
    _BACKENDS[key] = factory


def available_hyperedge_acceleration_backends() -> tuple[str, ...]:
    """Return registered backend names in deterministic order."""
    return tuple(sorted(_BACKENDS))


def create_hyperedge_acceleration_service(
    name: str = "numpy",
) -> HyperedgeAccelerationService:
    """Create and validate one configured acceleration service.

    Registered names are case-insensitive.  An external zero-argument factory
    may also be selected without modifying this package by using
    ``"package.module:factory"``.
    """
    requested = str(name).strip()
    key = requested.lower()
    if key in _BACKENDS:
        factory = _BACKENDS[key]
    elif ":" in requested:
        module_name, attribute_name = requested.rsplit(":", 1)
        if not module_name or not attribute_name:
            raise ValueError(
                "external hyperedge backend must be 'module:factory'")
        try:
            factory = getattr(import_module(module_name), attribute_name)
        except (ImportError, AttributeError) as error:
            raise ValueError(
                f"cannot load hyperedge acceleration backend {requested!r}"
            ) from error
        if not callable(factory):
            raise TypeError(
                f"hyperedge acceleration factory {requested!r} is not callable")
    else:
        available = ", ".join(available_hyperedge_acceleration_backends())
        raise ValueError(
            f"unknown hyperedge acceleration backend {key!r}; "
            f"available: {available}")
    service = factory()
    if not isinstance(service, HyperedgeAccelerationService):
        raise TypeError(
            f"backend {requested!r} does not implement "
            "HyperedgeAccelerationService")
    return service
