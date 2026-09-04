#!/usr/bin/env python
"""Blind-bank runner for the strict distributed pilot.

The statistical unit is one complete seeded episode. Frames and targets are
never counted as extra independent samples. Diagnostic checkpoints are written
atomically after every completed episode and may resume without changing seed
order or effective configuration. Formal runs reject resume and recompute all
100 frozen episodes from immutable inputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from tools.run_strict_distributed_pilot import (
    DEFAULT_CARRIER_PERIOD,
    _episode,
    validate_formal_system_identity,
    validate_strict_config,
)
from uav_isac.utils.reproducibility import (
    _canonical_json,
    _strict_json_loads,
    build_run_manifest,
    validate_formal_run,
)


def wilson_lower_bound(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> float:
    """One-sided Wilson lower confidence bound for episode success."""
    n = int(total)
    x = int(successes)
    if n <= 0 or not 0 <= x <= n:
        raise ValueError("successes/total must satisfy 0 <= successes <= total")
    level = float(confidence)
    if not 0.5 < level < 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    z = float(NormalDist().inv_cdf(level))
    p = x / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    radius = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return float(np.clip((centre - radius) / denominator, 0.0, 1.0))


def _run_seed(arguments: tuple[Any, int, int, int]) -> dict[str, Any]:
    cfg, seed, tail_window, carrier_period = arguments
    # Each episode receives an immutable-by-convention snapshot instead of
    # reopening a YAML file that may change after the formal preflight.
    cfg = copy.deepcopy(cfg)
    validate_strict_config(cfg)
    return _episode(cfg, int(seed), int(tail_window), int(carrier_period))


def validate_resume_policy(*, resume: bool, formal: bool) -> None:
    """Formal evidence is never assembled from a mutable prior JSON file."""
    if bool(resume) and bool(formal):
        raise ValueError(
            "formal execution cannot resume from an editable diagnostic "
            "checkpoint; restart the frozen bank from seed 1")


def validate_formal_protocol_arguments(
    *,
    formal: bool,
    split: str,
    workers: int,
    tail_window: int,
    carrier_period: int,
) -> None:
    if not formal:
        return
    expected = {
        "split": (str(split), "test"),
        "workers": (int(workers), 1),
        "tail_window": (int(tail_window), 50),
        "carrier_period": (int(carrier_period), DEFAULT_CARRIER_PERIOD),
    }
    mismatches = [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            "formal execution protocol arguments are frozen: "
            + "; ".join(mismatches))


def bind_run_spec(
    manifest: dict[str, Any],
    *,
    formal: bool,
    split: str,
    workers: int,
    tail_window: int,
    carrier_period: int,
) -> None:
    """Attach one canonical protocol identity to the run manifest."""
    spec = {
        "schema_version": "strict-distributed-run-spec/v1",
        "execution_mode": "formal" if formal else "diagnostic",
        "algorithm_version": manifest.get("algorithm_version"),
        "git_commit": manifest.get("git", {}).get("commit"),
        "source_tree_sha256": manifest.get("source_tree", {}).get("sha256"),
        "config_path": manifest.get("config", {}).get("path"),
        "config_source_sha256": manifest.get("config", {}).get(
            "source_sha256"),
        "config_effective_sha256": manifest.get("config", {}).get(
            "effective_sha256"),
        "seed_bank_path": manifest.get("seeds", {}).get("library_path"),
        "seed_bank_sha256": manifest.get("seeds", {}).get("library_sha256"),
        "seed_bank_metadata": manifest.get("seeds", {}).get(
            "library_metadata"),
        "seeds": manifest.get("seeds", {}).get("values"),
        "seed_split": str(split),
        "workers": int(workers),
        "tail_window": int(tail_window),
        "carrier_period": int(carrier_period),
        "runtime_packages": manifest.get("runtime", {}).get("packages"),
        "thread_environment": manifest.get("runtime", {}).get(
            "thread_environment"),
        "acceptance_thresholds": {
            "steady_min": 0.80,
            "weak3_min": 0.70,
            "worst_min": 0.60,
            "qos_wilson_lower_min": 0.80,
            "delivery_rate_min": 0.99,
            "deadline_violation_rate_max": 0.01,
            "closed_loop_p95_ms_max": 100.0,
        },
    }
    manifest["run_spec"] = spec
    manifest["run_spec_sha256"] = hashlib.sha256(
        _canonical_json(spec)).hexdigest()


def validate_worker_topology(cfg: Any, workers: int) -> bool:
    """Reject nested episode/node process pools; return internal mode."""
    internal_parallel = bool(getattr(
        cfg.marl,
        "distributed_replicated_power_process_parallel_enabled",
        False,
    ))
    if internal_parallel and int(workers) != 1:
        raise ValueError(
            "internal node-process parallelism requires --workers 1; "
            "nested episode and node pools invalidate timing and can "
            "oversubscribe the host")
    return internal_parallel


def _continuous_summary(
    episodes: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    values = np.asarray([float(item[key]) for item in episodes])
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def summarize(
    episodes: list[dict[str, Any]],
    *,
    timing_claim_eligible: bool = True,
) -> dict[str, Any]:
    """Summarize independent episodes without pseudo-replication."""
    if not episodes:
        return {"completed_episodes": 0}
    successes = int(sum(bool(item["qos_success"]) for item in episodes))
    count = len(episodes)
    delivery = float(np.mean([item["delivery_rate"] for item in episodes]))
    deadline = float(np.mean([
        item["deadline_violation_rate"] for item in episodes]))
    worst_closed_loop_p95 = float(np.max([
        item["closed_loop_critical_path_p95_ms"] for item in episodes]))
    return {
        "completed_episodes": count,
        "qos_successes": successes,
        "qos_rate": successes / count,
        "qos_rate_wilson_lower_95_one_sided": wilson_lower_bound(
            successes, count, 0.95),
        "steady": _continuous_summary(episodes, "steady"),
        "weak3": _continuous_summary(episodes, "weak3"),
        "worst": _continuous_summary(episodes, "worst"),
        "belief_position_rmse_m": _continuous_summary(
            episodes, "belief_position_rmse_m"),
        "movement_target_coverage": _continuous_summary(
            episodes, "movement_target_coverage"),
        "step_time_p95_ms": _continuous_summary(
            episodes, "step_time_p95_ms"),
        "closed_loop_critical_path_p95_ms": _continuous_summary(
            episodes, "closed_loop_critical_path_p95_ms"),
        "delivery_rate_mean": delivery,
        "deadline_violation_rate_mean": deadline,
        "timing_claim_eligible": bool(timing_claim_eligible),
        "timing_claim_note": (
            "single-worker execution; per-episode wall timing is attributable"
            if timing_claim_eligible else
            "parallel workers contend for CPU; use a separate single-worker "
            "run for latency claims"
        ),
        "gates": {
            "qos_wilson_lower_ge_0_80": (
                wilson_lower_bound(successes, count, 0.95) >= 0.80),
            "delivery_rate_ge_0_99": delivery >= 0.99,
            "deadline_violation_rate_le_0_01": deadline <= 0.01,
            "every_seed_closed_loop_p95_le_100ms": (
                (worst_closed_loop_p95 <= 100.0)
                if timing_claim_eligible else None),
        },
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def validate_resume_identity(
    previous: dict[str, Any],
    current_manifest: dict[str, Any],
    seeds: list[int],
    *,
    workers: int,
    tail_window: int,
    carrier_period: int,
) -> None:
    """Reject a checkpoint assembled under any different experiment identity."""
    prior_manifest = previous.get("run_manifest", {})
    checks = (
        ("run-manifest schema", prior_manifest.get("schema_version"),
         current_manifest.get("schema_version")),
        ("algorithm version", prior_manifest.get("algorithm_version"),
         current_manifest.get("algorithm_version")),
        ("formal-result eligibility",
         prior_manifest.get("formal_result_eligible"),
         current_manifest.get("formal_result_eligible")),
        ("Git commit",
         prior_manifest.get("git", {}).get("commit"),
         current_manifest.get("git", {}).get("commit")),
        ("Git dirty state",
         prior_manifest.get("git", {}).get("dirty"),
         current_manifest.get("git", {}).get("dirty")),
        ("source-tree hash",
         prior_manifest.get("source_tree", {}).get("sha256"),
         current_manifest.get("source_tree", {}).get("sha256")),
        ("effective configuration hash",
         prior_manifest.get("config", {}).get("effective_sha256"),
         current_manifest.get("config", {}).get("effective_sha256")),
        ("seed-bank hash",
         prior_manifest.get("seeds", {}).get("library_sha256"),
         current_manifest.get("seeds", {}).get("library_sha256")),
        ("Python runtime",
         prior_manifest.get("runtime", {}).get("python_version"),
         current_manifest.get("runtime", {}).get("python_version")),
        ("platform",
         prior_manifest.get("runtime", {}).get("platform"),
         current_manifest.get("runtime", {}).get("platform")),
        ("package set",
         prior_manifest.get("runtime", {}).get("packages"),
         current_manifest.get("runtime", {}).get("packages")),
        ("native thread environment",
         prior_manifest.get("runtime", {}).get("thread_environment"),
         current_manifest.get("runtime", {}).get("thread_environment")),
        ("worker count", previous.get("workers"), int(workers)),
        ("tail window", previous.get("tail_window"), int(tail_window)),
        ("carrier period", previous.get("carrier_period"), int(carrier_period)),
    )
    for label, prior, current in checks:
        if prior != current:
            raise RuntimeError(f"resume refused: {label} changed")
    previous_seeds = [int(seed) for seed in previous.get("seeds", [])]
    if previous_seeds != [int(seed) for seed in seeds]:
        raise RuntimeError("resume refused: frozen seed order changed")


def run_bank(
    config_path: str,
    output_path: str,
    *,
    split: str = "test",
    workers: int = 1,
    tail_window: int = 50,
    carrier_period: int = DEFAULT_CARRIER_PERIOD,
    resume: bool = False,
    formal: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    validate_resume_policy(resume=resume, formal=formal)
    validate_formal_protocol_arguments(
        formal=formal,
        split=split,
        workers=workers,
        tail_window=tail_window,
        carrier_period=carrier_period,
    )
    cfg = load_config(config_path)
    validate_strict_config(cfg)
    if formal:
        validate_formal_system_identity(config_path)
    internal_parallel = validate_worker_topology(cfg, workers)
    bank_path = Path(cfg.marl.eval_seed_bank_path)
    if not bank_path.is_absolute():
        bank_path = ROOT / bank_path
    bank = _strict_json_loads(bank_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in bank.get("splits", {}).get(split, [])]
    if not seeds:
        raise ValueError(f"seed-bank split {split!r} is empty")
    if int(limit) > 0:
        if formal:
            raise ValueError("formal execution cannot limit the frozen split")
        seeds = seeds[:int(limit)]
    algorithm_version = (
        "strict-distributed-owner-posterior-bistatic-v4-process-parallel"
        if internal_parallel else
        "strict-distributed-owner-posterior-bistatic-v3")
    manifest = build_run_manifest(
        cfg,
        config_path=config_path,
        seeds=seeds,
        algorithm_version=algorithm_version,
        root=ROOT,
    )
    bind_run_spec(
        manifest,
        formal=formal,
        split=split,
        workers=workers,
        tail_window=tail_window,
        carrier_period=carrier_period,
    )
    if formal:
        validate_formal_run(cfg, seeds, manifest)

    output = Path(output_path)
    completed: dict[int, dict[str, Any]] = {}
    if resume and output.is_file():
        previous = _strict_json_loads(output.read_text(encoding="utf-8"))
        validate_resume_identity(
            previous,
            manifest,
            seeds,
            workers=workers,
            tail_window=tail_window,
            carrier_period=carrier_period,
        )
        completed = {
            int(item["seed"]): item
            for item in previous.get("episodes", [])
            if int(item["seed"]) in set(seeds)
        }

    def checkpoint(status: str) -> dict[str, Any]:
        ordered = [completed[seed] for seed in seeds if seed in completed]
        payload = {
            "status": status,
            "execution_mode": "formal" if formal else "diagnostic",
            "config": os.path.normpath(config_path),
            "seed_bank": str(bank_path),
            "seed_split": split,
            "seeds": seeds,
            "workers": int(workers),
            "tail_window": int(tail_window),
            "carrier_period": int(carrier_period),
            "run_manifest": manifest,
            "summary": summarize(
                ordered,
                timing_claim_eligible=(int(workers) == 1),
            ),
            "episodes": ordered,
        }
        _atomic_write(output, payload)
        return payload

    pending = [seed for seed in seeds if seed not in completed]
    in_progress_status = (
        "FORMAL_IN_PROGRESS" if formal else "DIAGNOSTIC_IN_PROGRESS")
    complete_status = "FORMAL_COMPLETE" if formal else "DIAGNOSTIC_ONLY"
    checkpoint(in_progress_status if pending else complete_status)
    arguments = [
        (cfg, seed, int(tail_window), int(carrier_period))
        for seed in pending
    ]
    if max(1, int(workers)) == 1:
        for argument in arguments:
            episode = _run_seed(argument)
            completed[int(episode["seed"])] = episode
            checkpoint(in_progress_status)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(_run_seed, argument): int(argument[1])
                for argument in arguments
            }
            for future in as_completed(futures):
                episode = future.result()
                completed[int(episode["seed"])] = episode
                checkpoint(in_progress_status)
    if formal:
        # Close the preflight/execution TOCTOU window: code, Git state,
        # effective config and the frozen bank must still match immediately
        # before the artifact receives a formal completion state.
        validate_formal_system_identity(config_path)
        validate_formal_run(cfg, seeds, manifest)
    return checkpoint(complete_status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/exp_strict_distributed_k16q16.yaml")
    parser.add_argument(
        "--output", default=(
            "results/_strict_distributed_sweep/"
            "k16q16_blind100_owner_posterior_v3.json"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tail-window", type=int, default=50)
    parser.add_argument(
        "--carrier-period", type=int, default=DEFAULT_CARRIER_PERIOD)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    result = run_bank(
        args.config,
        args.output,
        split=args.split,
        workers=args.workers,
        tail_window=args.tail_window,
        carrier_period=args.carrier_period,
        resume=args.resume,
        formal=args.formal,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
