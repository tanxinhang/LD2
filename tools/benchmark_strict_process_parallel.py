#!/usr/bin/env python
"""Paired strict-run replay test for persistent private-LP processes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from tools.run_strict_distributed_pilot import _episode, validate_strict_config


TRACE_FIELDS = (
    "detection",
    "belief_rmse_m",
    "uav_positions",
    "sensing_power_w",
    "bits",
    "delivery",
)


def _run(config_path: str, seed: int, frames: int, workers: int, enabled: bool):
    cfg = load_config(config_path)
    validate_strict_config(cfg)
    cfg.scenario.T = int(frames)
    cfg.marl.distributed_replicated_power_process_parallel_enabled = bool(enabled)
    cfg.marl.distributed_replicated_power_process_workers = int(workers)
    return _episode(
        cfg, seed=int(seed), tail_window=min(50, int(frames)),
        carrier_period=3, include_trace=True)


def paired_benchmark(
    config_path: str,
    seeds: tuple[int, ...],
    frames: int,
    workers: int,
) -> dict:
    pairs = []
    for seed in seeds:
        serial = _run(config_path, seed, frames, workers, False)
        parallel = _run(config_path, seed, frames, workers, True)
        field_errors = {}
        field_exact = {}
        for field in TRACE_FIELDS:
            left = np.asarray(serial["trace"][field], dtype=np.float64)
            right = np.asarray(parallel["trace"][field], dtype=np.float64)
            field_exact[field] = bool(np.array_equal(left, right))
            field_errors[field] = float(np.max(np.abs(left - right)))
        pairs.append({
            "seed": int(seed),
            "all_traces_exact": bool(all(field_exact.values())),
            "trace_exact": field_exact,
            "trace_max_abs_error": field_errors,
            "serial": {
                "step_mean_ms": serial["step_time_ms"],
                "step_p95_ms": serial["step_time_p95_ms"],
                "power_batch_mean_ms": serial["power_process_batch_wall_ms"],
                "worst": serial["worst"],
                "belief_rmse_m": serial["belief_position_rmse_m"],
            },
            "parallel": {
                "step_mean_ms": parallel["step_time_ms"],
                "step_p95_ms": parallel["step_time_p95_ms"],
                "power_batch_mean_ms": parallel["power_process_batch_wall_ms"],
                "worker_warmup_ms": parallel["power_process_worker_warmup_ms"],
                "fallback_fraction": parallel[
                    "power_process_fallback_fraction"],
                "worst": parallel["worst"],
                "belief_rmse_m": parallel["belief_position_rmse_m"],
            },
            "step_mean_speedup": float(
                serial["step_time_ms"] / max(parallel["step_time_ms"], 1.0e-12)),
        })
    return {
        "status": "paired_development_diagnostic_only",
        "config": str(config_path),
        "frames": int(frames),
        "workers": int(workers),
        "seeds": list(seeds),
        "radio_transport_changed": False,
        "all_pairs_exact": bool(all(
            pair["all_traces_exact"] for pair in pairs)),
        "no_parallel_fallback": bool(all(
            pair["parallel"]["fallback_fraction"] == 0.0 for pair in pairs)),
        "mean_step_speedup": float(np.mean([
            pair["step_mean_speedup"] for pair in pairs])),
        "pairs": pairs,
        "formal_result_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/exp_strict_distributed_k16q16.yaml")
    parser.add_argument("--seeds", default="7,19,43")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or args.frames < 1 or args.workers < 1:
        parser.error("seeds, frames and workers must be non-empty/positive")
    report = paired_benchmark(args.config, seeds, args.frames, args.workers)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
