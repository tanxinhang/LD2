"""Restricted loading for deployment-facing PyTorch checkpoints.

``torch.load`` can execute pickle payloads when ``weights_only=False``.  The
loaders in this project accept only PyTorch's restricted weights-only format.
Some historical model artifacts contain NumPy float32 normalization arrays;
callers may opt into the small, explicit allowlist needed to reconstruct those
arrays without enabling arbitrary pickle globals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
import math
from os import fstat, PathLike
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch


class SafeCheckpointLoadError(RuntimeError):
    """Raised when a checkpoint is incompatible with restricted loading."""


_MINIMUM_SAFE_TORCH = (2, 10, 0)
DEFAULT_MAX_CHECKPOINT_BYTES = 1_073_741_824
DEFAULT_MAX_CHECKPOINT_TENSOR_ELEMENTS = 100_000_000


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _checkpoint_path(parent: str, key: Any) -> str:
    rendered = repr(key)
    if len(rendered) > 80:
        rendered = rendered[:77] + "..."
    return f"{parent}[{rendered}]"


def _finite_tensor_values(value: torch.Tensor, path: str) -> torch.Tensor:
    """Return stored values suitable for a finite-value scan.

    Sparse optimizer states are valid checkpoint content.  Inspect only their
    stored values instead of densifying them; the logical ``numel`` is still
    charged against the resource budget below.
    """
    if bool(getattr(value, "is_meta", False)):
        raise SafeCheckpointLoadError(
            f"Checkpoint tensor at {path} is on the meta device and cannot "
            "be validated.")
    if value.layout == torch.strided:
        return value
    try:
        if value.layout == torch.sparse_coo:
            return value.coalesce().values()
        return value.values()
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        raise SafeCheckpointLoadError(
            f"Checkpoint tensor at {path} has unsupported layout "
            f"{value.layout!s}; finite values cannot be verified.") from exc


def validate_state_dict(
    state_dict: Any,
    *,
    description: str = "model state_dict",
) -> Mapping[str, torch.Tensor]:
    """Validate the structural contract expected by ``load_state_dict``.

    This deliberately applies only to model state dictionaries.  Optimizer
    state dictionaries legitimately contain nested mappings, lists, scalar
    hyperparameters, and integer parameter identifiers and must not be passed
    to this helper.
    """
    if not isinstance(state_dict, Mapping):
        raise SafeCheckpointLoadError(
            f"{description} must be a mapping, got "
            f"{type(state_dict).__name__}.")
    if not state_dict:
        raise SafeCheckpointLoadError(f"{description} must not be empty.")
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise SafeCheckpointLoadError(
                f"{description} contains a non-string parameter key "
                f"{key!r}.")
        if not isinstance(value, torch.Tensor):
            raise SafeCheckpointLoadError(
                f"{description}[{key!r}] must be a tensor, got "
                f"{type(value).__name__}.")
    return state_dict


def validate_checkpoint_payload(
    payload: Any,
    *,
    description: str = "checkpoint",
    required_keys: Iterable[str] = (),
    mapping_keys: Iterable[str] = (),
    optional_mapping_keys: Iterable[str] = (),
    state_dict_keys: Iterable[str] = (),
    optional_state_dict_keys: Iterable[str] = (),
    max_tensor_elements: int = DEFAULT_MAX_CHECKPOINT_TENSOR_ELEMENTS,
) -> Mapping[Any, Any]:
    """Validate a restricted-load result before a consumer uses it.

    The traversal is intentionally permissive about nested containers so that
    canonical PyTorch optimizer state remains supported.  It enforces only the
    deployment invariants common to every consumer: a mapping root, declared
    schema keys, a cumulative tensor/array element ceiling, and finite numeric
    values.  Declared model state dictionaries receive the stricter flat
    string-to-tensor check required by ``nn.Module.load_state_dict``.
    """
    if not isinstance(payload, Mapping):
        raise SafeCheckpointLoadError(
            f"{description} root must be a mapping, got "
            f"{type(payload).__name__}.")
    _require_positive_integer(max_tensor_elements, "max_tensor_elements")

    required = tuple(required_keys)
    mapped = tuple(mapping_keys)
    optional_mapped = tuple(optional_mapping_keys)
    model_states = tuple(state_dict_keys)
    optional_model_states = tuple(optional_state_dict_keys)
    for key in (
        *required, *mapped, *optional_mapped,
        *model_states, *optional_model_states,
    ):
        if not isinstance(key, str) or not key:
            raise ValueError("checkpoint schema keys must be non-empty strings")
    for key in (*required, *mapped, *model_states):
        if key not in payload:
            raise SafeCheckpointLoadError(
                f"{description} is missing required key {key!r}.")
    for key in mapped:
        if not isinstance(payload[key], Mapping):
            raise SafeCheckpointLoadError(
                f"{description}[{key!r}] must be a mapping, got "
                f"{type(payload[key]).__name__}.")
    for key in optional_mapped:
        if key in payload and not isinstance(payload[key], Mapping):
            raise SafeCheckpointLoadError(
                f"{description}[{key!r}] must be a mapping, got "
                f"{type(payload[key]).__name__}.")
    for key in model_states:
        validate_state_dict(
            payload[key], description=f"{description}[{key!r}]")
    for key in optional_model_states:
        if key in payload:
            validate_state_dict(
                payload[key], description=f"{description}[{key!r}]")

    total_elements = 0
    seen: set[int] = set()
    stack: list[tuple[str, Any]] = [("$", payload)]
    while stack:
        path, value = stack.pop()
        if isinstance(value, torch.Tensor):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            # Device validity is independent of dtype.  In particular,
            # integer and boolean state tensors on ``meta`` have no backing
            # storage just like floating meta tensors and must not bypass the
            # finite-value branch below.
            if bool(getattr(value, "is_meta", False)):
                raise SafeCheckpointLoadError(
                    f"Checkpoint tensor at {path} is on the meta device and "
                    "cannot be validated.")
            try:
                elements = int(value.numel())
            except RuntimeError as exc:
                raise SafeCheckpointLoadError(
                    f"Checkpoint tensor size at {path} cannot be verified.") \
                    from exc
            total_elements += elements
            if total_elements > max_tensor_elements:
                raise SafeCheckpointLoadError(
                    f"{description} contains {total_elements:,} tensor/array "
                    f"elements, exceeding the limit of "
                    f"{max_tensor_elements:,}.")
            if value.is_floating_point() or value.is_complex():
                inspected = _finite_tensor_values(value, path)
                try:
                    finite = bool(torch.isfinite(inspected).all().item())
                except (NotImplementedError, RuntimeError, TypeError) as exc:
                    raise SafeCheckpointLoadError(
                        f"Checkpoint tensor values at {path} cannot be "
                        "verified as finite.") from exc
                if not finite:
                    raise SafeCheckpointLoadError(
                        f"{description} contains a non-finite tensor value at "
                        f"{path}.")
            continue
        if isinstance(value, np.ndarray):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            total_elements += int(value.size)
            if total_elements > max_tensor_elements:
                raise SafeCheckpointLoadError(
                    f"{description} contains {total_elements:,} tensor/array "
                    f"elements, exceeding the limit of "
                    f"{max_tensor_elements:,}.")
            if (
                np.issubdtype(value.dtype, np.floating)
                or np.issubdtype(value.dtype, np.complexfloating)
            ) and not bool(np.isfinite(value).all()):
                raise SafeCheckpointLoadError(
                    f"{description} contains a non-finite array value at "
                    f"{path}.")
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise SafeCheckpointLoadError(
                    f"{description} contains non-finite numeric metadata at "
                    f"{path}.")
            continue
        if isinstance(value, complex):
            if not (
                math.isfinite(value.real) and math.isfinite(value.imag)
            ):
                raise SafeCheckpointLoadError(
                    f"{description} contains non-finite numeric metadata at "
                    f"{path}.")
            continue
        if isinstance(value, np.generic):
            if (
                np.issubdtype(value.dtype, np.floating)
                or np.issubdtype(value.dtype, np.complexfloating)
            ) and not bool(np.isfinite(value)):
                raise SafeCheckpointLoadError(
                    f"{description} contains non-finite numeric metadata at "
                    f"{path}.")
            continue
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            stack.extend(
                (_checkpoint_path(path, key), item)
                for key, item in value.items()
            )
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            stack.extend(
                (f"{path}[{index}]", item)
                for index, item in enumerate(value)
            )

    return payload


def _require_safe_torch_version() -> None:
    """Refuse versions with known weights-only unpickler vulnerabilities."""
    raw = str(getattr(torch, "__version__", ""))
    match = re.fullmatch(
        r"(\d+)\.(\d+)(?:\.(\d+))?(?:\+[A-Za-z0-9_.-]+)?",
        raw,
    )
    if match is None:
        raise SafeCheckpointLoadError(
            f"Cannot verify the PyTorch checkpoint-loader version {raw!r}; "
            "install a stable PyTorch >=2.10 release before loading artifacts."
        )
    release = tuple(int(value or 0) for value in match.groups())
    if release < _MINIMUM_SAFE_TORCH:
        raise SafeCheckpointLoadError(
            f"PyTorch {raw} has known weights-only checkpoint-loader "
            "vulnerabilities; upgrade to PyTorch >=2.10 before loading "
            "artifacts."
        )


def _legacy_numpy_float32_context():
    """Allow only the globals needed by NumPy float32 ndarray pickles."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - NumPy is a core dependency.
        raise SafeCheckpointLoadError(
            "Legacy NumPy checkpoint compatibility requires NumPy.") from exc

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if safe_globals is None:
        raise SafeCheckpointLoadError(
            "This PyTorch version cannot safely load legacy NumPy checkpoint "
            "arrays; upgrade PyTorch or re-export the checkpoint using tensors."
        )

    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:  # NumPy < 2
        from numpy import core as numpy_core
    reconstruct = numpy_core.multiarray._reconstruct
    # NumPy 2 moved the private module from ``numpy.core`` to ``numpy._core``.
    # Register both serialized names so trusted historical float32 arrays can
    # migrate across that boundary.  No user-defined class/function is added.
    allowed: list[Any] = [
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (reconstruct, "numpy._core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
    ]
    return safe_globals(allowed)


def _concise_rejection_detail(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    for marker in ("Unsupported global:", "Can only build"):
        for line in lines:
            if marker in line:
                # PyTorch's longer message suggests allowlisting the rejected
                # global.  Do not repeat that advice at a deployment boundary:
                # a user-defined allowlisted callable can itself execute code.
                return line.split("Please use", 1)[0].rstrip(". ")
    return lines[0] if lines else type(exc).__name__


def safe_torch_load(
    checkpoint: str | PathLike[str] | Path,
    *,
    map_location: Any = "cpu",
    description: str = "checkpoint",
    allow_legacy_numpy_float32: bool = False,
    required_keys: Iterable[str] = (),
    mapping_keys: Iterable[str] = (),
    optional_mapping_keys: Iterable[str] = (),
    state_dict_keys: Iterable[str] = (),
    optional_state_dict_keys: Iterable[str] = (),
    max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    max_tensor_elements: int = DEFAULT_MAX_CHECKPOINT_TENSOR_ELEMENTS,
) -> Mapping[Any, Any]:
    """Load a checkpoint without permitting arbitrary pickle execution.

    There is deliberately no ``weights_only=False`` fallback.  Incompatible
    legacy files fail with migration guidance instead.  The byte ceiling is a
    pre-deserialization guard.  The element/finite scan necessarily happens
    after PyTorch materializes tensors, so it is defense in depth rather than
    an absolute compressed-payload memory sandbox.
    """
    _require_safe_torch_version()
    _require_positive_integer(max_checkpoint_bytes, "max_checkpoint_bytes")
    _require_positive_integer(max_tensor_elements, "max_tensor_elements")
    try:
        checkpoint_stream = Path(checkpoint).open("rb")
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        raise
    with checkpoint_stream:
        checkpoint_bytes = fstat(checkpoint_stream.fileno()).st_size
        if checkpoint_bytes > max_checkpoint_bytes:
            raise SafeCheckpointLoadError(
                f"Safe loading refused {description} at {str(checkpoint)!r}: "
                f"file size {checkpoint_bytes:,} bytes exceeds the limit of "
                f"{max_checkpoint_bytes:,} bytes.")
        context = (
            _legacy_numpy_float32_context()
            if allow_legacy_numpy_float32 else nullcontext()
        )
        try:
            with context:
                payload = torch.load(
                    checkpoint_stream,
                    map_location=map_location,
                    weights_only=True,
                )
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            raise
        except Exception as exc:
            detail = _concise_rejection_detail(exc)
            raise SafeCheckpointLoadError(
                f"Safe loading refused {description} at {str(checkpoint)!r}: "
                f"{detail}. No unsafe pickle fallback was attempted. Re-export "
                "the trusted legacy artifact with tensor values and primitive "
                "metadata."
            ) from None
    return validate_checkpoint_payload(
        payload,
        description=description,
        required_keys=required_keys,
        mapping_keys=mapping_keys,
        optional_mapping_keys=optional_mapping_keys,
        state_dict_keys=state_dict_keys,
        optional_state_dict_keys=optional_state_dict_keys,
        max_tensor_elements=max_tensor_elements,
    )
