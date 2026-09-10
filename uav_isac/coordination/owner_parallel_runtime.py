"""Measured concurrent execution of independent owner-local ranking jobs."""

# ----------------------------------------------------------------------
# AUDIT/RESEARCH-ONLY MODULE (2026-08-16 audit remediation)
#
# This module is consumed only by tools/ audit scripts and tests. It is
# NOT part of the deployment execution path (env_core / trainer) and its
# results must not be described as deployed behaviour. It exists to keep
# a specific research question reproducible; see
# docs/EXPERIMENT_LOG.md for the associated gate.
# ----------------------------------------------------------------------

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
from typing import Callable, Generic, Mapping, TypeVar

import numpy as np


T = TypeVar("T")


@dataclass(frozen=True)
class OwnerJobMeasurement(Generic[T]):
    owner: int
    result: T
    wall_latency_s: float
    thread_cpu_s: float


@dataclass(frozen=True)
class ParallelOwnerExecution(Generic[T]):
    measurements: tuple[OwnerJobMeasurement[T], ...]
    wall_latency_s: float
    process_cpu_s: float
    sum_worker_wall_s: float
    sum_worker_thread_cpu_s: float
    worker_count: int

    @property
    def results(self) -> Mapping[int, T]:
        return {
            measurement.owner: measurement.result
            for measurement in self.measurements
        }

    def charged_compute_energy_j(self, logical_cpu_power_w: float) -> float:
        """Charge energy from measured process CPU time and a frozen W/CPU model.

        This is accounting, not a hardware power-meter observation.  Process
        CPU time is additive across concurrent threads, whereas wall time is
        the deadline quantity.
        """
        power = float(logical_cpu_power_w)
        if not np.isfinite(power) or power < 0.0:
            raise ValueError("logical_cpu_power_w must be finite non-negative")
        return float(power * self.process_cpu_s)


def execute_owner_jobs_concurrently(
    jobs: Mapping[int, Callable[[], T]],
    *,
    max_workers: int | None = None,
) -> ParallelOwnerExecution[T]:
    """Execute independent owner jobs concurrently and preserve owner order."""
    normalized = {int(owner): job for owner, job in jobs.items()}
    if not normalized:
        raise ValueError("at least one owner job is required")
    if len(normalized) != len(jobs):
        raise ValueError("owner identifiers must be unique integers")
    if any(not callable(job) for job in normalized.values()):
        raise TypeError("every owner job must be callable")
    workers = len(normalized) if max_workers is None else int(max_workers)
    if workers < 1:
        raise ValueError("max_workers must be positive")
    workers = min(workers, len(normalized))

    def measured(owner: int, job: Callable[[], T]) -> OwnerJobMeasurement[T]:
        wall_started = time.perf_counter()
        cpu_started = time.thread_time()
        result = job()
        return OwnerJobMeasurement(
            owner=owner,
            result=result,
            wall_latency_s=float(time.perf_counter() - wall_started),
            thread_cpu_s=float(time.thread_time() - cpu_started),
        )

    wall_started = time.perf_counter()
    process_started = time.process_time()
    measurements: list[OwnerJobMeasurement[T]] = []
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="isac-owner",
    ) as executor:
        futures = {
            executor.submit(measured, owner, job): owner
            for owner, job in normalized.items()
        }
        for future in as_completed(futures):
            measurements.append(future.result())
    process_cpu_s = float(time.process_time() - process_started)
    wall_latency_s = float(time.perf_counter() - wall_started)
    ordered = tuple(sorted(measurements, key=lambda item: item.owner))
    return ParallelOwnerExecution(
        measurements=ordered,
        wall_latency_s=wall_latency_s,
        process_cpu_s=process_cpu_s,
        sum_worker_wall_s=float(sum(
            item.wall_latency_s for item in ordered)),
        sum_worker_thread_cpu_s=float(sum(
            item.thread_cpu_s for item in ordered)),
        worker_count=workers,
    )
