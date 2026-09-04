#!/usr/bin/env python
"""Stage-5 trace audit of dual-certified cross-frame power reuse.

At a solved frame the node stores its feasible power rows and optimal simplex
target prices.  On the next private view it rescales the old rows to the new RF
budgets, computes the achieved primal lower value L, and evaluates the old
prices on the new full gain matrix to obtain the valid dual upper value U.
The LP is skipped only when (U-L)/U is below tolerance.

The audit uses recorded full-information gains to isolate the scheduling idea;
deployment in strict distributed mode must run the same calculation separately
on every UAV's private reconstructed view.
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
from tools.audit_d088_event_trigger import _gain_and_budget  # noqa: E402
from tools.benchmark_coarse_to_fine_power_stage1 import _quantiles  # noqa: E402
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    solve_fixed_structure_maxmin_power_lp,
)


def _rescale_rows(power: np.ndarray, budget: np.ndarray) -> np.ndarray:
    row_sum = np.sum(power, axis=1, keepdims=True)
    share = np.divide(
        power,
        row_sum,
        out=np.full_like(power, 1.0 / power.shape[1]),
        where=row_sum > 0.0,
    )
    return budget[:, None] * share


def _episode_quality(
    values: np.ndarray,
    seeds: np.ndarray,
    *,
    steady_window: int,
) -> dict[str, float]:
    rows = []
    for seed in np.unique(seeds):
        episode = values[seeds == seed]
        rows.append(float(np.mean(episode[-int(steady_window):])))
    return {
        "mean": float(np.mean(rows)),
        "min": float(np.min(rows)),
    }


def benchmark(
    trace_path: Path,
    config_path: Path,
    *,
    seed_limit: int,
    tolerances: tuple[float, ...],
    steady_window: int,
) -> dict[str, object]:
    cfg = load_config(str(config_path))
    with np.load(trace_path, allow_pickle=False) as loaded:
        all_seeds = np.asarray(loaded["seed"], dtype=np.int64)
        all_frames = np.asarray(loaded["frame"], dtype=np.int64)
        seed_order = list(dict.fromkeys(int(v) for v in all_seeds))[
            :max(1, int(seed_limit))]
        selected_rows = np.concatenate([
            np.flatnonzero(all_seeds == seed)[
                np.argsort(all_frames[all_seeds == seed])
            ]
            for seed in seed_order
        ])
        gains: list[np.ndarray] = []
        budgets: list[np.ndarray] = []
        seeds: list[int] = []
        frames: list[int] = []
        for row in selected_rows:
            gain, budget = _gain_and_budget(loaded, int(row), cfg)
            if np.all(np.sum(gain * budget[:, None], axis=0) > 0.0):
                gains.append(gain)
                budgets.append(budget)
                seeds.append(int(all_seeds[row]))
                frames.append(int(all_frames[row]))
    if not gains:
        raise RuntimeError("trace yielded no fully reachable frames")

    exact_worst: list[float] = []
    exact_time_ms: list[float] = []
    exact_power: list[np.ndarray] = []
    exact_prices: list[np.ndarray] = []
    for gain, budget in zip(gains, budgets):
        started = perf_counter_ns()
        exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
        exact_time_ms.append((perf_counter_ns() - started) / 1.0e6)
        exact_worst.append(float(exact.worst_deflection))
        exact_power.append(exact.power_w)
        exact_prices.append(exact.prices)
    exact_array = np.asarray(exact_worst, dtype=np.float64)
    seed_array = np.asarray(seeds, dtype=np.int64)

    results: dict[str, dict[str, object]] = {}
    for tolerance in tolerances:
        executed_worst: list[float] = []
        certificate_gap: list[float] = []
        trigger_time_ms: list[float] = []
        resolved: list[bool] = []
        bound_violations = 0
        relative_guarantee_violations = 0
        last_power: np.ndarray | None = None
        last_prices: np.ndarray | None = None
        last_seed: int | None = None
        for index, (gain, budget, seed) in enumerate(
            zip(gains, budgets, seeds)
        ):
            if last_seed != int(seed):
                last_power = exact_power[index]
                last_prices = exact_prices[index]
                executed = last_power
                relative_gap = 0.0
                did_resolve = True
                check_ms = 0.0
            else:
                started = perf_counter_ns()
                held = _rescale_rows(last_power, budget)
                lower = float(np.min(np.sum(gain * held, axis=0)))
                upper = float(np.sum(
                    budget * np.max(
                        last_prices[None, :] * gain, axis=1)))
                upper = max(upper, lower)
                relative_gap = max(upper - lower, 0.0) / max(
                    upper, 1.0e-300)
                check_ms = (perf_counter_ns() - started) / 1.0e6
                if exact_array[index] > upper + 1.0e-7 * max(1.0, upper):
                    bound_violations += 1
                did_resolve = relative_gap > float(tolerance)
                if did_resolve:
                    last_power = exact_power[index]
                    last_prices = exact_prices[index]
                    executed = last_power
                    relative_gap = 0.0
                else:
                    executed = held
            value = float(np.min(np.sum(gain * executed, axis=0)))
            ratio = value / max(exact_array[index], 1.0e-300)
            if ratio + 1.0e-7 < 1.0 - float(tolerance):
                relative_guarantee_violations += 1
            executed_worst.append(value)
            certificate_gap.append(relative_gap)
            trigger_time_ms.append(check_ms)
            resolved.append(did_resolve)
            last_seed = int(seed)
        executed_array = np.asarray(executed_worst, dtype=np.float64)
        ratio = executed_array / np.maximum(exact_array, 1.0e-300)
        solve_rate = float(np.mean(resolved))
        expected_time = (
            float(np.mean(trigger_time_ms))
            + solve_rate * float(np.mean(exact_time_ms))
        )
        results[f"tol_{tolerance:.4f}"] = {
            "relative_tolerance": float(tolerance),
            "lp_resolve_rate": solve_rate,
            "lp_skip_rate": float(1.0 - solve_rate),
            "utility_ratio_to_exact": _quantiles(ratio),
            "certificate_gap_on_execution": _quantiles(certificate_gap),
            "dual_upper_bound_violations": int(bound_violations),
            "relative_guarantee_violations": int(
                relative_guarantee_violations),
            "trigger_check_time_ms": _quantiles(trigger_time_ms),
            "estimated_compute_time_per_frame_ms": expected_time,
            "estimated_speedup": float(
                np.mean(exact_time_ms) / max(expected_time, 1.0e-12)),
            "exact_episode_steady_worst_deflection": _episode_quality(
                exact_array, seed_array, steady_window=int(steady_window)),
            "reuse_episode_steady_worst_deflection": _episode_quality(
                executed_array, seed_array, steady_window=int(steady_window)),
            "rf_feasible_rate": 1.0,
        }
    return {
        "schema_version": 1,
        "scope": (
            "full-information temporal shadow audit of local primal/dual "
            "event trigger; strict-private-view deployment not yet claimed"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "seed_count": len(np.unique(seed_array)),
        "frame_count": len(gains),
        "num_uavs": int(gains[0].shape[0]),
        "num_targets": int(gains[0].shape[1]),
        "exact_lp_time_ms": _quantiles(exact_time_ms),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument(
        "--tolerances", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--steady-window", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.trace,
        args.config,
        seed_limit=int(args.seed_limit),
        tolerances=tuple(sorted(set(float(v) for v in args.tolerances))),
        steady_window=int(args.steady_window),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "frame_count": result["frame_count"],
        "exact_lp_time_ms": result["exact_lp_time_ms"],
        "results": result["results"],
    }, indent=2))


if __name__ == "__main__":
    main()
