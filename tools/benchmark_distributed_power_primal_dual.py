"""Fixed-seed property benchmark for the fixed-structure power block."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_isac.optimization.distributed_power_primal_dual import (
    BoundedTemporalPowerController,
    FixedStructurePowerPrimalDual,
)


def run_benchmark(seed: int, cases: int, max_iterations: int) -> dict:
    rng = np.random.default_rng(seed)
    records = []
    start = time.perf_counter()
    for case in range(cases):
        uavs = int(rng.integers(3, 9))
        targets = int(rng.integers(2, 7))
        mask = rng.random((uavs, targets)) < 0.65
        # Ensure every target is structurally supported.
        for target in range(targets):
            if not np.any(mask[:, target]):
                mask[int(rng.integers(uavs)), target] = True
        gain = np.where(mask, rng.uniform(0.25, 2.0, (uavs, targets)), 0.0)
        budget = rng.uniform(0.5, 1.2, uavs)
        witness_raw = rng.uniform(0.0, 1.0, (uavs, targets))
        witness_raw[~mask] = 0.0
        row_sum = witness_raw.sum(axis=1)
        witness = witness_raw * (
            (0.55 * budget / np.maximum(row_sum, 1.0e-12))[:, None])
        requirement = np.sum(gain * witness, axis=0) * rng.uniform(
            0.65, 0.90, targets)
        solver = FixedStructurePowerPrimalDual(
            gain, requirement, budget, mask,
            power_cost_per_watt=1.0e-4,
            quadratic_regularization=1.0e-3,
        )
        result = solver.run(
            max_iterations,
            primal_tolerance=2.0e-4,
            stationarity_tolerance=2.0e-4,
            dual_tolerance=2.0e-4,
            convergence_patience=3,
        )
        records.append({
            "case": case,
            "uavs": uavs,
            "targets": targets,
            "converged": result.converged,
            "iterations": result.iteration,
            "primal_violation": result.primal_violation,
            "power_budget_violation_w": result.power_budget_violation_w,
            "consensus_residual": result.consensus_residual,
        })
    elapsed = time.perf_counter() - start
    return {
        "seed": seed,
        "cases": cases,
        "max_iterations": max_iterations,
        "converged_cases": sum(record["converged"] for record in records),
        "convergence_rate": float(np.mean([
            record["converged"] for record in records])),
        "median_iterations": float(np.median([
            record["iterations"] for record in records])),
        "p95_iterations": float(np.quantile([
            record["iterations"] for record in records], 0.95)),
        "maximum_primal_violation": max(
            record["primal_violation"] for record in records),
        "maximum_power_budget_violation_w": max(
            record["power_budget_violation_w"] for record in records),
        "maximum_consensus_residual": max(
            record["consensus_residual"] for record in records),
        "elapsed_s": elapsed,
        "mean_case_time_ms": 1000.0 * elapsed / cases,
        "records": records,
    }


def run_temporal_benchmark(
    seed: int,
    sequences: int,
    frames: int,
    iterations_per_frame: int,
) -> dict:
    """Compare bounded warm starts with equal-budget cold starts."""
    rng = np.random.default_rng(seed)
    warm_violations = []
    cold_violations = []
    warm_power = []
    cold_power = []
    accepted = 0
    fallbacks = 0
    start = time.perf_counter()
    options = {
        "power_cost_per_watt": 1.0e-4,
        "quadratic_regularization": 1.0e-3,
    }
    step_options = {
        "primal_tolerance": 2.0e-4,
        "stationarity_tolerance": 2.0e-4,
        "dual_tolerance": 2.0e-4,
        "convergence_patience": 3,
    }
    for _ in range(sequences):
        uavs, targets = 6, 4
        mask = rng.random((uavs, targets)) < 0.70
        for target in range(targets):
            if not np.any(mask[:, target]):
                mask[int(rng.integers(uavs)), target] = True
        base_gain = np.where(
            mask, rng.uniform(0.4, 1.6, (uavs, targets)), 0.0)
        budget = rng.uniform(0.7, 1.0, uavs)
        witness = rng.uniform(0.0, 1.0, (uavs, targets))
        witness[~mask] = 0.0
        witness *= (
            0.5 * budget / np.maximum(witness.sum(axis=1), 1e-12)
        )[:, None]
        base_requirement = np.sum(base_gain * witness, axis=0) * 0.8
        warm = BoundedTemporalPowerController(
            maximum_iterations_per_frame=iterations_per_frame,
            solver_options=options,
            step_options=step_options,
        )
        for _frame in range(frames):
            gain = np.where(mask, base_gain * np.clip(
                1.0 + rng.normal(0.0, 0.01, base_gain.shape), 0.9, 1.1), 0.0)
            requirement = base_requirement * np.clip(
                1.0 + rng.normal(0.0, 0.01, targets), 0.9, 1.1)
            warm_result = warm.solve(gain, requirement, budget, mask)
            cold = BoundedTemporalPowerController(
                maximum_iterations_per_frame=iterations_per_frame,
                solver_options=options,
                step_options=step_options,
            )
            cold_result = cold.solve(gain, requirement, budget, mask)
            warm_violations.append(warm_result.selected_primal_violation)
            cold_violations.append(cold_result.selected_primal_violation)
            warm_power.append(float(warm_result.power_w.sum()))
            cold_power.append(float(cold_result.power_w.sum()))
            accepted += int(warm_result.candidate_accepted)
            fallbacks += int(not warm_result.candidate_accepted)
    elapsed = time.perf_counter() - start
    warm_array = np.asarray(warm_violations)
    cold_array = np.asarray(cold_violations)
    warm_power_array = np.asarray(warm_power)
    cold_power_array = np.asarray(cold_power)
    return {
        "seed": seed,
        "sequences": sequences,
        "frames": frames,
        "iterations_per_frame": iterations_per_frame,
        "warm_mean_primal_violation": float(warm_array.mean()),
        "cold_mean_primal_violation": float(cold_array.mean()),
        "mean_violation_reduction": float(
            cold_array.mean() - warm_array.mean()),
        "warm_better_rate": float(np.mean(warm_array < cold_array - 1e-12)),
        "warm_mean_power_w": float(warm_power_array.mean()),
        "cold_mean_power_w": float(cold_power_array.mean()),
        "mean_power_reduction_w": float(
            cold_power_array.mean() - warm_power_array.mean()),
        "accepted_updates": accepted,
        "fallback_updates": fallbacks,
        "maximum_warm_primal_violation": float(warm_array.max()),
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--output", default="")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--temporal-sequences", type=int, default=0)
    parser.add_argument("--temporal-frames", type=int, default=20)
    parser.add_argument("--iterations-per-frame", type=int, default=20)
    args = parser.parse_args()
    summary = run_benchmark(args.seed, args.cases, args.max_iterations)
    if args.temporal_sequences > 0:
        summary["temporal_warm_start"] = run_temporal_benchmark(
            args.seed,
            args.temporal_sequences,
            args.temporal_frames,
            args.iterations_per_frame,
        )
    payload = json.dumps(summary, indent=2)
    printed = dict(summary)
    if args.summary_only:
        printed.pop("records", None)
    print(json.dumps(printed, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
