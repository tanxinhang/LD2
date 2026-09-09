"""Audited NumPy implementation of the hyperedge acceleration contract."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from uav_isac.coordination.hyperedge import (
    SelectedBistaticCoefficients,
    reconstruct_bistatic_coefficient_from_public_state,
    reconstruct_bistatic_coefficient_upper_from_public_state,
    reconstruct_selected_bistatic_coefficients_from_public_state,
    reconstruct_selected_bistatic_coefficients_batch_from_public_state,
)


class NumpyHyperedgeAccelerationService:
    """Default exact backend; delegates to the tested NumPy kernels."""

    name = "numpy"

    def __init__(self, *, collect_timing: bool = False) -> None:
        """Create the backend.

        Per-call instrumentation is intentionally opt-in: even a small counter
        branch on every sparse edge refresh distorts the latency this service is
        meant to reduce.  The default path therefore binds the audited kernels
        directly on the instance.
        """
        self.collect_timing = bool(collect_timing)
        self.reset_stats()
        if not self.collect_timing:
            # Fast path: preserve the service boundary while eliminating per
            # call dispatch/counter overhead in normal simulation runs.
            self.reconstruct_dense = (  # type: ignore[method-assign]
                reconstruct_bistatic_coefficient_from_public_state)
            self.reconstruct_upper = (  # type: ignore[method-assign]
                reconstruct_bistatic_coefficient_upper_from_public_state)
            self.reconstruct_selected = (  # type: ignore[method-assign]
                reconstruct_selected_bistatic_coefficients_from_public_state)
            self.reconstruct_selected_batch = (  # type: ignore[method-assign]
                reconstruct_selected_bistatic_coefficients_batch_from_public_state)

    def reconstruct_dense(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self._calls["dense"] += 1
        if not self.collect_timing:
            return reconstruct_bistatic_coefficient_from_public_state(
                *args, **kwargs)
        started = time.perf_counter()
        try:
            return reconstruct_bistatic_coefficient_from_public_state(
                *args, **kwargs)
        finally:
            self._seconds["dense"] += time.perf_counter() - started

    def reconstruct_upper(self, *args: Any, **kwargs: Any) -> np.ndarray:
        self._calls["upper"] += 1
        if not self.collect_timing:
            return reconstruct_bistatic_coefficient_upper_from_public_state(
                *args, **kwargs)
        started = time.perf_counter()
        try:
            return reconstruct_bistatic_coefficient_upper_from_public_state(
                *args, **kwargs)
        finally:
            self._seconds["upper"] += time.perf_counter() - started

    def reconstruct_selected(
        self, *args: Any, **kwargs: Any,
    ) -> SelectedBistaticCoefficients:
        self._calls["selected"] += 1
        if not self.collect_timing:
            return reconstruct_selected_bistatic_coefficients_from_public_state(
                *args, **kwargs)
        started = time.perf_counter()
        try:
            return reconstruct_selected_bistatic_coefficients_from_public_state(
                *args, **kwargs)
        finally:
            self._seconds["selected"] += time.perf_counter() - started

    def reconstruct_selected_batch(
        self, *args: Any, **kwargs: Any,
    ) -> SelectedBistaticCoefficients:
        self._calls["selected_batch"] += 1
        if not self.collect_timing:
            return reconstruct_selected_bistatic_coefficients_batch_from_public_state(
                *args, **kwargs)
        started = time.perf_counter()
        try:
            return reconstruct_selected_bistatic_coefficients_batch_from_public_state(
                *args, **kwargs)
        finally:
            self._seconds["selected_batch"] += time.perf_counter() - started

    def reset_stats(self) -> None:
        operations = ("dense", "upper", "selected", "selected_batch")
        self._calls = {operation: 0 for operation in operations}
        self._seconds = {operation: 0.0 for operation in operations}

    def stats(self) -> dict[str, float | int | str]:
        result: dict[str, float | int | str] = {
            "backend": self.name,
            "timing_enabled": int(self.collect_timing),
        }
        total_calls = 0
        total_seconds = 0.0
        for operation in ("dense", "upper", "selected", "selected_batch"):
            calls = int(self._calls[operation])
            seconds = float(self._seconds[operation])
            result[f"{operation}_calls"] = calls
            result[f"{operation}_seconds"] = seconds
            result[f"{operation}_mean_ms"] = (
                1000.0 * seconds / calls if calls else 0.0)
            total_calls += calls
            total_seconds += seconds
        result["total_calls"] = total_calls
        result["total_seconds"] = total_seconds
        return result
