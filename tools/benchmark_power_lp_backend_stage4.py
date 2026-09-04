#!/usr/bin/env python
"""Stage-4 benchmark of exact LP construction and SciPy/HiGHS backends.

This isolates fixed call/setup overhead after the learned active-set experiment
showed that halving variables did not reduce wall time.  Every candidate solves
the same full max-min LP and is checked against the current implementation.
No production solver is changed by this benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_coarse_to_fine_power_stage1 import (  # noqa: E402
    _load_instances,
    _quantiles,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    solve_fixed_structure_maxmin_power_lp,
)


class PreparedLayout:
    def __init__(self, num_uavs: int, num_targets: int) -> None:
        self.K = int(num_uavs)
        self.Q = int(num_targets)
        self.V = self.K * self.Q
        self.objective = np.zeros(self.V + 1, dtype=np.float64)
        self.objective[-1] = -1.0
        self.upper = np.zeros(self.Q + self.K, dtype=np.float64)
        self.template = np.zeros(
            (self.Q + self.K, self.V + 1), dtype=np.float64)
        self.template[:self.Q, -1] = 1.0
        for transmitter in range(self.K):
            begin = transmitter * self.Q
            self.template[self.Q + transmitter, begin:begin + self.Q] = 1.0
        self.target_rows = np.repeat(np.arange(self.Q), self.K)
        self.target_columns = (
            np.arange(self.K)[:, None] * self.Q
            + np.arange(self.Q)[None, :]
        ).T.reshape(-1)

        budget_rows = np.repeat(np.arange(self.K) + self.Q, self.Q)
        budget_columns = np.arange(self.V)
        self.sparse_rows = np.concatenate((
            self.target_rows,
            np.arange(self.Q),
            budget_rows,
        ))
        self.sparse_columns = np.concatenate((
            self.target_columns,
            np.full(self.Q, self.V),
            budget_columns,
        ))
        self.sparse_tail = np.concatenate((
            np.ones(self.Q),
            np.ones(self.V),
        ))

    def dense(self, gain: np.ndarray, budget: np.ndarray) -> tuple:
        matrix = self.template.copy()
        matrix[self.target_rows, self.target_columns] = -gain.T.reshape(-1)
        upper = self.upper.copy()
        upper[self.Q:] = budget
        return matrix, upper

    def sparse(self, gain: np.ndarray, budget: np.ndarray) -> tuple:
        values = np.concatenate((-gain.T.reshape(-1), self.sparse_tail))
        matrix = coo_array(
            (values, (self.sparse_rows, self.sparse_columns)),
            shape=(self.Q + self.K, self.V + 1),
        ).tocsr()
        upper = self.upper.copy()
        upper[self.Q:] = budget
        return matrix, upper


def _solve(
    layout: PreparedLayout,
    gain: np.ndarray,
    budget: np.ndarray,
    *,
    matrix_kind: str,
    method: str,
    presolve: bool,
) -> tuple[float, np.ndarray]:
    matrix, upper = (
        layout.dense(gain, budget)
        if matrix_kind == "dense"
        else layout.sparse(gain, budget)
    )
    result = linprog(
        layout.objective,
        A_ub=matrix,
        b_ub=upper,
        bounds=(0.0, None),
        method=method,
        options={"presolve": bool(presolve)},
    )
    if not result.success or result.x is None:
        raise RuntimeError(result.message)
    power = np.maximum(result.x[:layout.V], 0.0).reshape(
        layout.K, layout.Q)
    return float(np.min(np.sum(gain * power, axis=0))), power


def benchmark(
    trace_path: Path,
    config_path: Path,
    *,
    max_samples: int,
) -> dict[str, object]:
    instances = _load_instances(
        trace_path, config_path, max_samples=int(max_samples))
    K, Q = instances[0][0].shape
    layout = PreparedLayout(K, Q)
    gain0, budget0, _seed, _frame = instances[0]
    solve_fixed_structure_maxmin_power_lp(gain0, budget0)
    for kind, method, presolve in (
        ("dense", "highs", True),
        ("dense", "highs-ds", True),
        ("dense", "highs-ipm", True),
        ("dense", "highs-ds", False),
        ("sparse", "highs-ds", True),
        ("sparse", "highs-ds", False),
    ):
        _solve(
            layout, gain0, budget0,
            matrix_kind=kind, method=method, presolve=presolve)

    baseline_time: list[float] = []
    baseline_worst: list[float] = []
    for gain, budget, _seed, _frame in instances:
        started = perf_counter_ns()
        result = solve_fixed_structure_maxmin_power_lp(gain, budget)
        baseline_time.append((perf_counter_ns() - started) / 1.0e6)
        baseline_worst.append(float(result.worst_deflection))

    candidates = {}
    for kind, method, presolve in (
        ("dense", "highs", True),
        ("dense", "highs-ds", True),
        ("dense", "highs-ipm", True),
        ("dense", "highs-ds", False),
        ("sparse", "highs-ds", True),
        ("sparse", "highs-ds", False),
    ):
        label = f"{kind}_{method}_{'presolve' if presolve else 'no_presolve'}"
        timing: list[float] = []
        worst: list[float] = []
        feasible: list[bool] = []
        for gain, budget, _seed, _frame in instances:
            started = perf_counter_ns()
            value, power = _solve(
                layout,
                gain,
                budget,
                matrix_kind=kind,
                method=method,
                presolve=presolve,
            )
            timing.append((perf_counter_ns() - started) / 1.0e6)
            worst.append(value)
            feasible.append(bool(
                np.all(power >= -1.0e-10)
                and np.all(np.sum(power, axis=1) <= budget + 1.0e-8)
            ))
        timing_array = np.asarray(timing, dtype=np.float64)
        baseline_array = np.asarray(baseline_time, dtype=np.float64)
        ratio = np.asarray(worst) / np.maximum(
            np.asarray(baseline_worst), 1.0e-300)
        candidates[label] = {
            "time_ms": _quantiles(timing_array),
            "paired_speedup": _quantiles(baseline_array / timing_array),
            "objective_ratio_to_current": _quantiles(ratio),
            "rf_feasible_rate": float(np.mean(feasible)),
            "exact_match_rate": float(np.mean(np.abs(ratio - 1.0) <= 1.0e-7)),
        }
    return {
        "schema_version": 1,
        "scope": "exact full LP backend/construction shadow benchmark",
        "trace": str(trace_path),
        "config": str(config_path),
        "sample_count": len(instances),
        "num_uavs": int(K),
        "num_targets": int(Q),
        "current_solver_time_ms": _quantiles(baseline_time),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.trace, args.config, max_samples=int(args.max_samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "current_solver_time_ms": result["current_solver_time_ms"],
        "candidates": result["candidates"],
    }, indent=2))


if __name__ == "__main__":
    main()
