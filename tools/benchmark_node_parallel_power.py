#!/usr/bin/env python
"""Benchmark serial versus node-parallel private LP execution.

This is a development benchmark, not a deployment latency claim.  It captures
the strict simulator's private gain views after one causal frame, then solves
the same immutable per-node LP inputs serially and with the production
persistent process/shard executor. Radio transport remains orthogonal and is
not part of this benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Prevent each node worker from recursively creating a native BLAS pool.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.parallel_power_executor import (
    ReplicatedPowerProcessExecutor,
)


def _capture_private_views(config_path: str, seed: int):
    # Keep the environment/Torch stack out of spawned LP-only workers.
    from config.params import load_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = load_config(config_path)
    env = UAVISACEnv(config=cfg)
    observations, _ = env.reset(seed=int(seed))
    try:
        core = env.core
        carrier_fraction = float(np.clip(
            float(cfg.marl.comm_tx_power_w)
            / max(float(cfg.uav.P_isac_total), 1.0e-12),
            0.0,
            1.0,
        ))
        core.submit_learned_communications(
            messages={
                sender: np.zeros(core._comm_payload_dim, dtype=np.float64)
                for sender in range(core.K)
            },
            rate_indices={sender: 0 for sender in range(core.K)},
            comm_power_fractions={
                sender: carrier_fraction for sender in range(core.K)
            },
            sensing_target_weights={
                sender: np.ones(core.Q, dtype=np.float64)
                for sender in range(core.K)
            },
            token_masks={
                sender: np.ones(core.Q, dtype=np.float64)
                for sender in range(core.K)
            },
        )
        actions = {
            agent: {"delta_p": np.zeros(2), "role": 2}
            for agent in observations
        }
        env.step(actions)
        views = np.asarray(
            core._hyperedge_public_gain_views, dtype=np.float64).copy()
        budget = np.full(
            core.K, float(core._sensing_power_cap_w), dtype=np.float64)
        if views.shape != (core.K, core.K, core.Q):
            raise RuntimeError("strict private gain views were not constructed")
        return views, budget
    finally:
        env.close()


def _solve_payload(payload):
    """Pickle-safe isolated worker entry point."""
    viewer, view, budget = payload
    solved = solve_fixed_structure_maxmin_power_lp(view, budget)
    return int(viewer), solved.power_w, float(solved.worst_deflection)


def benchmark(
    config_path: str,
    seed: int,
    repeats: int,
    worker_counts: tuple[int, ...],
) -> dict:
    views, budget = _capture_private_views(config_path, seed)

    payloads = tuple(
        (viewer, views[viewer], budget) for viewer in range(views.shape[0]))

    def solve_serial():
        return tuple(_solve_payload(payload) for payload in payloads)

    # One untimed pass pays imports/HiGHS initialization and defines the exact
    # deterministic reference before any concurrent execution.
    reference = solve_serial()
    records = []
    for requested_workers in worker_counts:
        workers = max(1, min(int(requested_workers), views.shape[0]))
        samples_ms = []
        latest = None
        warmup_ms = 0.0
        if workers == 1:
            for _ in range(repeats):
                started = time.perf_counter()
                latest = solve_serial()
                samples_ms.append(1000.0 * (time.perf_counter() - started))
        else:
            with ReplicatedPowerProcessExecutor(
                worker_count=workers,
                # Development benchmark: do not confuse an intentionally
                # tight online deployment deadline with benchmark failure.
                batch_timeout_s=10.0,
            ) as executor:
                warmup_ms = 1000.0 * executor.warm_up(
                    views.shape[1], views.shape[2])
                for _ in range(repeats):
                    batch = executor.solve_many(tuple(views), budget)
                    latest = tuple(
                        (viewer, result.power_w, result.worst_deflection)
                        for viewer, result in enumerate(batch.results)
                    )
                    samples_ms.append(1000.0 * batch.batch_wall_time_s)
        assert latest is not None
        max_power_error = max(float(np.max(np.abs(
            expected[1] - actual[1])))
            for expected, actual in zip(reference, latest))
        max_worst_error = max(
            abs(float(expected[2]) - float(actual[2]))
            for expected, actual in zip(reference, latest)
        )
        records.append({
            "workers": workers,
            "mean_ms": float(np.mean(samples_ms)),
            "p50_ms": float(np.percentile(samples_ms, 50)),
            "p95_ms": float(np.percentile(samples_ms, 95)),
            "max_ms": float(np.max(samples_ms)),
            "one_time_worker_warmup_ms": float(warmup_ms),
            "max_power_abs_error": max_power_error,
            "max_worst_deflection_abs_error": max_worst_error,
            "numerically_equivalent": bool(
                max_power_error <= 1.0e-12
                and max_worst_error <= 1.0e-12),
        })
    serial_mean = records[0]["mean_ms"]
    for record in records:
        record["mean_speedup_over_first_case"] = float(
            serial_mean / max(record["mean_ms"], 1.0e-12))
    return {
        "status": "development_benchmark_only",
        "config": str(config_path),
        "seed": int(seed),
        "nodes": int(views.shape[0]),
        "targets": int(views.shape[2]),
        "repeats": int(repeats),
        "radio_transport_changed": False,
        "parallel_backend": "isolated_processes",
        "native_threads_per_worker_requested": 1,
        "results": records,
        "deployment_latency_claim_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/exp_strict_distributed_k16q16.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.workers):
        parser.error("repeats and worker counts must be positive")
    print(json.dumps(benchmark(
        args.config,
        args.seed,
        args.repeats,
        tuple(args.workers),
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
