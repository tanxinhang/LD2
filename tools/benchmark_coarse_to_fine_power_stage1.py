#!/usr/bin/env python
"""Stage-1 gate for coarse-to-fine sensing-power acceleration.

This benchmark deliberately contains no learned model.  It first establishes
whether the existing analytic safe start and finite-round distributed dual
solver leave a useful, learnable residual relative to the exact fixed-structure
LP on recorded physical traces.  AI should only be added if this gate shows a
quality/time trade-off that a learned warm start could plausibly improve.

Only solver time is measured.  Trace loading and physical gain reconstruction
are kept outside the timed region so the result answers the narrow question:
can a coarse power layer replace the repeated per-view LP calls?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.audit_d087_power_deployment import (  # noqa: E402
    _fixed_owner_gain,
    _recorded_power,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    distributed_dual_maxmin_power,
    solve_fixed_structure_maxmin_power_lp,
)


def _harmonic_safe_start(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return the zero-message constructive primal and a uniform dual bound."""
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    ceiling = np.sum(gain * budget[:, None], axis=0)
    if np.any(ceiling <= 0.0):
        power = np.zeros_like(gain)
        power[:, int(np.argmin(ceiling))] = budget
        return power, 0.0, 0.0
    inverse = 1.0 / ceiling
    share = inverse / float(np.sum(inverse))
    power = budget[:, None] * share[None, :]
    worst = float(np.min(np.sum(gain * power, axis=0)))
    prices = np.full(gain.shape[1], 1.0 / gain.shape[1])
    upper = float(np.sum(
        budget * np.max(prices[None, :] * gain, axis=1)))
    return power, worst, max(upper, worst)


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _load_instances(
    trace_path: Path,
    config_path: Path,
    *,
    max_samples: int,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    cfg = load_config(str(config_path))
    with np.load(trace_path, allow_pickle=False) as loaded:
        resolved = np.asarray(loaded["p0_resolved"], dtype=bool)
        candidates = np.flatnonzero(resolved)
        if candidates.size == 0:
            candidates = np.arange(int(loaded["frame"].shape[0]))
        take = min(int(max_samples), int(candidates.size))
        # Deterministic coverage of the entire trace, rather than a favorable
        # contiguous window or a random subset that changes between runs.
        selected = candidates[np.linspace(
            0, candidates.size - 1, take, dtype=np.int64)]
        instances: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        for row in selected:
            try:
                gain, _owners = _fixed_owner_gain(loaded, int(row), cfg)
            except ValueError:
                continue
            _comm, _sensing, budget, _error = _recorded_power(
                loaded, int(row))
            if np.all(np.sum(gain * budget[:, None], axis=0) > 0.0):
                instances.append((
                    gain,
                    budget,
                    int(np.asarray(loaded["seed"])[row]),
                    int(np.asarray(loaded["frame"])[row]),
                ))
    if not instances:
        raise RuntimeError("trace yielded no fully reachable power instances")
    return instances


def benchmark(
    trace_path: Path,
    config_path: Path,
    *,
    max_samples: int,
    rounds: tuple[int, ...],
    price_bits: int,
    feedback_bits: int,
) -> dict[str, object]:
    instances = _load_instances(
        trace_path, config_path, max_samples=int(max_samples))

    # One unreported warm-up per solver family removes library initialization
    # from the first measured sample without hiding steady-state Python costs.
    warm_gain, warm_budget, _seed, _frame = instances[0]
    solve_fixed_structure_maxmin_power_lp(warm_gain, warm_budget)
    distributed_dual_maxmin_power(
        warm_gain,
        warm_budget,
        rounds=max(rounds),
        price_bits=int(price_bits),
        feedback_bits=int(feedback_bits),
    )

    exact_worst: list[float] = []
    exact_time_ms: list[float] = []
    exact_power: list[np.ndarray] = []
    for gain, budget, _seed, _frame in instances:
        started = perf_counter_ns()
        result = solve_fixed_structure_maxmin_power_lp(gain, budget)
        exact_time_ms.append((perf_counter_ns() - started) / 1.0e6)
        exact_worst.append(float(result.worst_deflection))
        exact_power.append(result.power_w)

    methods: dict[str, dict[str, object]] = {}

    harmonic_time: list[float] = []
    harmonic_worst: list[float] = []
    harmonic_upper: list[float] = []
    harmonic_feasible: list[bool] = []
    for gain, budget, _seed, _frame in instances:
        started = perf_counter_ns()
        power, worst, upper = _harmonic_safe_start(gain, budget)
        harmonic_time.append((perf_counter_ns() - started) / 1.0e6)
        harmonic_worst.append(worst)
        harmonic_upper.append(upper)
        harmonic_feasible.append(bool(
            np.all(power >= -1.0e-12)
            and np.all(np.sum(power, axis=1) <= budget + 1.0e-10)
        ))
    methods["harmonic_r0"] = _method_summary(
        harmonic_worst,
        harmonic_upper,
        exact_worst,
        harmonic_time,
        exact_time_ms,
        harmonic_feasible,
        communication_bits=0,
    )

    for count in rounds:
        timing: list[float] = []
        worst_values: list[float] = []
        upper_values: list[float] = []
        feasible: list[bool] = []
        communication: list[int] = []
        for gain, budget, _seed, _frame in instances:
            started = perf_counter_ns()
            result = distributed_dual_maxmin_power(
                gain,
                budget,
                rounds=int(count),
                price_bits=int(price_bits),
                feedback_bits=int(feedback_bits),
            )
            timing.append((perf_counter_ns() - started) / 1.0e6)
            worst_values.append(float(result.worst_deflection))
            upper_values.append(float(result.dual_upper_bound))
            communication.append(int(result.communication_bits))
            feasible.append(bool(
                np.all(result.power_w >= -1.0e-12)
                and np.all(
                    np.sum(result.power_w, axis=1) <= budget + 1.0e-10)
            ))
        methods[f"dual_r{count}"] = _method_summary(
            worst_values,
            upper_values,
            exact_worst,
            timing,
            exact_time_ms,
            feasible,
            communication_bits=int(np.median(communication)),
        )

    exact_timing = _quantiles(exact_time_ms)
    return {
        "schema_version": 1,
        "scope": (
            "stage-1 shadow benchmark; recorded KxQ physical gains; solver "
            "only; no AI and no closed-loop claim"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "sample_count": len(instances),
        "num_uavs": int(instances[0][0].shape[0]),
        "num_targets": int(instances[0][0].shape[1]),
        "price_bits": int(price_bits),
        "feedback_bits": int(feedback_bits),
        "exact_lp": {
            "time_ms": exact_timing,
            "worst_deflection": _quantiles(exact_worst),
            "rf_feasible_rate": float(np.mean([
                np.all(power >= -1.0e-12)
                and np.all(np.sum(power, axis=1) <= budget + 1.0e-10)
                for power, (_gain, budget, _seed, _frame)
                in zip(exact_power, instances)
            ])),
        },
        "methods": methods,
        "stage2_gate": {
            "target_utility_ratio": 0.995,
            "target_p95_time_ms": 0.5,
            "target_rf_feasible_rate": 1.0,
            "target_certificate_gap": 0.005,
            "interpretation": (
                "Train a residual warm-start model only if no existing "
                "finite-round method passes all four targets."
            ),
        },
    }


def _method_summary(
    worst_values: list[float],
    upper_values: list[float],
    exact_worst: list[float],
    timing_ms: list[float],
    exact_timing_ms: list[float],
    feasible: list[bool],
    *,
    communication_bits: int,
) -> dict[str, object]:
    worst = np.asarray(worst_values, dtype=np.float64)
    upper = np.asarray(upper_values, dtype=np.float64)
    exact = np.asarray(exact_worst, dtype=np.float64)
    ratio = np.divide(
        worst,
        exact,
        out=np.ones_like(worst),
        where=exact > 1.0e-300,
    )
    relative_gap = np.divide(
        np.maximum(upper - worst, 0.0),
        np.maximum(upper, 1.0e-300),
    )
    time_array = np.asarray(timing_ms, dtype=np.float64)
    exact_time = np.asarray(exact_timing_ms, dtype=np.float64)
    return {
        "utility_ratio_to_exact": _quantiles(ratio),
        "ratio_at_least_0_995_rate": float(np.mean(ratio >= 0.995)),
        "relative_primal_dual_gap": _quantiles(relative_gap),
        "gap_at_most_0_005_rate": float(np.mean(relative_gap <= 0.005)),
        "time_ms": _quantiles(time_array),
        "paired_speedup": _quantiles(exact_time / time_array),
        "rf_feasible_rate": float(np.mean(feasible)),
        "communication_bits_per_solve": int(communication_bits),
        "passes_stage2_gate": bool(
            np.quantile(ratio, 0.05) >= 0.995
            and np.quantile(time_array, 0.95) <= 0.5
            and np.mean(feasible) >= 1.0
            and np.quantile(relative_gap, 0.95) <= 0.005
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--price-bits", type=int, default=6)
    parser.add_argument("--feedback-bits", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.trace,
        args.config,
        max_samples=int(args.max_samples),
        rounds=tuple(sorted(set(int(v) for v in args.rounds))),
        price_bits=int(args.price_bits),
        feedback_bits=int(args.feedback_bits),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "sample_count": result["sample_count"],
        "exact_lp_time_ms": result["exact_lp"]["time_ms"],
        "methods": {
            name: {
                "ratio_p05": row["utility_ratio_to_exact"]["p05"],
                "gap_p95": row["relative_primal_dual_gap"]["p95"],
                "time_p95_ms": row["time_ms"]["p95"],
                "passes": row["passes_stage2_gate"],
            }
            for name, row in result["methods"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
