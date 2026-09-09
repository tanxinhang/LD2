"""Deterministic semantic fingerprints for migration characterization."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Callable


def is_runtime_telemetry_key(key: str) -> bool:
    """Return whether a field is nondeterministic wall-clock telemetry."""

    return (
        key == "p0_solve_time_s"
        or key.startswith("timing_")
        or key.startswith("movement_safety_solve_time_s")
        or key.endswith("_warmup_time_s")
    )


def _normalize(value: Any, exclude_key: Callable[[str], bool]) -> Any:
    if dataclasses.is_dataclass(value):
        return _normalize(dataclasses.asdict(value), exclude_key)
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item, exclude_key)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not exclude_key(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, exclude_key) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize(item, exclude_key) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "tolist"):
        return _normalize(value.tolist(), exclude_key)
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def semantic_fingerprint(
    value: Any,
    exclude_key: Callable[[str], bool] = is_runtime_telemetry_key,
) -> str:
    """Hash canonical JSON after removing explicitly nondeterministic fields."""

    normalized = _normalize(value, exclude_key)
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash every field of a JSON-compatible value without exclusions."""

    return semantic_fingerprint(value, exclude_key=lambda key: False)


def semantic_differences(first: Any, second: Any) -> tuple[str, ...]:
    """Return paths whose normalized semantic values differ."""

    left = _normalize(first, is_runtime_telemetry_key)
    right = _normalize(second, is_runtime_telemetry_key)
    differences: list[str] = []

    def compare(a: Any, b: Any, path: str) -> None:
        if type(a) is not type(b):
            differences.append(path)
            return
        if isinstance(a, dict):
            if set(a) != set(b):
                differences.append(path)
                return
            for key in a:
                compare(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, list):
            if len(a) != len(b):
                differences.append(path)
                return
            for index, (a_item, b_item) in enumerate(zip(a, b)):
                compare(a_item, b_item, f"{path}[{index}]")
            return
        if a != b:
            differences.append(path)

    compare(left, right, "$")
    return tuple(differences)
