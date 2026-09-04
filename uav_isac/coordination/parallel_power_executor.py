"""Persistent isolated-process executor for independent private power LPs."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from time import perf_counter
from typing import Sequence

import numpy as np

from uav_isac.coordination.maxmin_power import (
    MaxMinPowerResult,
    solve_fixed_structure_maxmin_power_lp,
)


_WORKER_THREADPOOL_LIMITER = None


def _initialize_private_lp_worker() -> None:
    """Limit every isolated LP worker to one native numerical thread.

    Four process-level LP shards already expose the available parallelism.
    Allowing BLAS/OpenMP to create another pool inside each process causes
    recursive oversubscription and invalidates tail-latency attribution.
    ``threadpoolctl`` also applies the limit when NumPy/SciPy was imported by
    the Windows spawn bootstrap before this initializer ran.
    """
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    from threadpoolctl import threadpool_limits

    global _WORKER_THREADPOOL_LIMITER
    _WORKER_THREADPOOL_LIMITER = threadpool_limits(limits=1)


def _solve_private_lp(payload) -> tuple[MaxMinPowerResult, float]:
    """Top-level pickle-safe worker entry point."""
    gain, budget = payload
    started = perf_counter()
    result = solve_fixed_structure_maxmin_power_lp(gain, budget)
    return result, float(perf_counter() - started)


def _solve_private_lp_batch(payload) -> tuple[
    tuple[int, MaxMinPowerResult, float], ...
]:
    """Solve one deterministic shard without per-LP process round trips.

    The LPs are independent.  Grouping several immutable inputs in one worker
    task changes only their execution schedule; the original problem index is
    returned explicitly so the parent can restore canonical viewer order.
    """
    indexed_gains, budget = payload
    solved = []
    for problem_index, gain, minimum_deflection in indexed_gains:
        started = perf_counter()
        result = solve_fixed_structure_maxmin_power_lp(
            gain,
            budget,
            minimum_deflection=minimum_deflection,
        )
        solved.append((
            int(problem_index),
            result,
            float(perf_counter() - started),
        ))
    return tuple(solved)


@dataclass(frozen=True)
class ParallelPowerBatch:
    results: tuple[MaxMinPowerResult, ...]
    node_compute_time_s: tuple[float, ...]
    batch_wall_time_s: float
    worker_count: int


class ReplicatedPowerProcessExecutor:
    """Lazy persistent process pool with deterministic input/output ordering.

    Processes exchange only immutable gain matrices and the public RF budget.
    No environment, target truth, random generator, message cache, or executed
    power row is shared with a worker.
    """

    def __init__(self, worker_count: int = 4, batch_timeout_s: float = 0.1):
        workers = int(worker_count)
        if workers < 1:
            raise ValueError("worker_count must be positive")
        timeout = float(batch_timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("batch_timeout_s must be finite and positive")
        self.worker_count = workers
        self.batch_timeout_s = timeout
        self._pool: ProcessPoolExecutor | None = None
        self._closed = False
        self._warmed = False
        self.warmup_time_s = 0.0

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._closed:
            raise RuntimeError("parallel power executor is closed")
        if self._pool is None:
            # Spawn avoids inheriting mutable simulator/RNG state.  It is more
            # expensive than fork, hence the pool persists across frames.
            self._pool = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=get_context("spawn"),
                initializer=_initialize_private_lp_worker,
            )
        return self._pool

    def solve_many(
        self,
        gain_matrices: Sequence[np.ndarray],
        public_budget_w: np.ndarray,
        *,
        minimum_deflections: Sequence[np.ndarray | None] | None = None,
    ) -> ParallelPowerBatch:
        gains = tuple(
            np.asarray(gain, dtype=np.float64).copy()
            for gain in gain_matrices
        )
        if not gains:
            return ParallelPowerBatch((), (), 0.0, self.worker_count)
        if not self._warmed:
            raise RuntimeError(
                "parallel power executor must be warmed before online solve")
        budget = np.asarray(public_budget_w, dtype=np.float64).copy()
        if minimum_deflections is None:
            reserves: tuple[np.ndarray | None, ...] = tuple(
                None for _ in gains
            )
        else:
            supplied = tuple(minimum_deflections)
            if len(supplied) != len(gains):
                raise ValueError(
                    "minimum_deflections must match gain_matrices")
            reserves = tuple(
                None if value is None else np.asarray(
                    value, dtype=np.float64).copy()
                for value in supplied
            )
        for gain, reserve in zip(gains, reserves):
            if reserve is not None and (
                reserve.shape != (gain.shape[1],)
                or np.any(~np.isfinite(reserve))
                or np.any(reserve < 0.0)
            ):
                raise ValueError(
                    "each minimum deflection must be a finite non-negative "
                    "target vector matching its gain matrix")
        # At K=Q=12, one private LP takes only about 1--2 ms.  Dispatching all
        # K LPs as separate Windows process tasks makes serialization and
        # future scheduling a material fraction of the critical path.  Use at
        # most one deterministic round-robin shard per worker.  Equal LP
        # dimensions make this balanced, while round-robin placement is less
        # sensitive to a locally difficult consecutive group than contiguous
        # chunking.  There is no algebraic coupling between shards.
        task_count = min(self.worker_count, len(gains))
        shards: list[list[
            tuple[int, np.ndarray, np.ndarray | None]
        ]] = [
            [] for _ in range(task_count)
        ]
        for problem_index, (gain, reserve) in enumerate(
            zip(gains, reserves)
        ):
            shards[problem_index % task_count].append((
                problem_index,
                gain,
                reserve,
            ))
        payloads = tuple(
            (tuple(shard), budget) for shard in shards if shard
        )
        started = perf_counter()
        returned_shards = tuple(self._ensure_pool().map(
            _solve_private_lp_batch,
            payloads,
            timeout=self.batch_timeout_s,
            chunksize=1,
        ))
        wall = float(perf_counter() - started)
        ordered: list[tuple[MaxMinPowerResult, float] | None] = [
            None for _ in gains
        ]
        for returned_shard in returned_shards:
            for problem_index, result, elapsed in returned_shard:
                if (
                    problem_index < 0
                    or problem_index >= len(ordered)
                    or ordered[problem_index] is not None
                ):
                    raise RuntimeError(
                        "parallel power worker returned an invalid index")
                ordered[problem_index] = (result, float(elapsed))
        if any(item is None for item in ordered):
            raise RuntimeError(
                "parallel power worker returned an incomplete batch")
        canonical = tuple(item for item in ordered if item is not None)
        return ParallelPowerBatch(
            results=tuple(item[0] for item in canonical),
            node_compute_time_s=tuple(float(item[1]) for item in canonical),
            batch_wall_time_s=wall,
            worker_count=self.worker_count,
        )

    def warm_up(self, num_transmitters: int, num_targets: int) -> float:
        """Start every worker and initialize the LP backend before mission time.

        The returned one-time cost belongs to controller initialization, not a
        control frame. It remains explicitly reportable and is never erased.
        """
        if self._warmed:
            return 0.0
        K, Q = int(num_transmitters), int(num_targets)
        if K < 1 or Q < 1:
            raise ValueError("warm-up dimensions must be positive")
        gain = np.ones((K, Q), dtype=np.float64)
        budget = np.ones(K, dtype=np.float64)
        payloads = tuple(
            (gain * (1.0 + 1.0e-6 * worker), budget)
            for worker in range(self.worker_count)
        )
        started = perf_counter()
        returned = tuple(self._ensure_pool().map(
            _solve_private_lp,
            payloads,
            # Process creation and imports are initialization work, not the
            # online batch deadline.
            timeout=max(10.0, 10.0 * self.batch_timeout_s),
            chunksize=1,
        ))
        elapsed = float(perf_counter() - started)
        if len(returned) != self.worker_count:
            raise RuntimeError("parallel power warm-up returned incomplete data")
        self._warmed = True
        self.warmup_time_s = elapsed
        return elapsed

    def close(self, wait: bool = True) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=bool(wait), cancel_futures=True)
            self._pool = None
        self._closed = True
        self._warmed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
