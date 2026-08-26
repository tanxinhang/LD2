#!/usr/bin/env python
"""P1-5 (advice 014 §9): crash-isolated per-seed certification evaluation.

The Windows/MKL ``numpy.linalg.eigvalsh`` native abort (KNOWN_ISSUES #4) can
kill the whole certification process when it fires inside a merged-belief
eigen-decomposition.  This wrapper runs the final evaluation ONE SEED PER
SUBPROCESS with ``MKL_NUM_THREADS=1`` / ``OMP_NUM_THREADS=1``, so a native
library crash terminates only that seed's worker; the parent records the seed
as crashed (fail-closed) and continues with the rest.  A single-seed library
crash can therefore never kill an entire certification run.

Usage::

  python tools/crash_isolated_seed_eval.py \\
      --config config/exp_800_k8q8_..._indep.yaml \\
      --warm-start results/.../best_restored.pt \\
      [--structure-student-checkpoint ... --structure-student-channel
       --structure-student-allow-scale-migration
       --structure-student-send-mode p0_resolve] \\
      --final-eval-seeds 615,298,613 --out-dir results/_crash_iso_ab

The merged ``paired_eval.csv`` keeps the same one-row-per-columns-of-arrays
schema the in-process path writes, so downstream aggregators
(assert_formal_gates.py, report_blind_certification.py) work unchanged.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUN_MAPPO = ROOT / "scripts" / "run_mappo.py"


def _parse(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return min(1.0, max(0.0, (center - radius) / denominator))


def _merge_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    """Merge seed rows and recompute aggregates from episode arrays.

    Copying the first worker's scalar summaries is statistically wrong for a
    multi-seed certification.  Arrays are the source of truth; corresponding
    ``eval_*`` means and the QoS/Wilson statistics are rebuilt after merging.
    """
    if not rows:
        return {}
    merged: dict[str, str] = {}
    columns = list(rows[0].keys())
    max_columns = {
        "eval_isac_max_power_balance_error_w",
        "eval_isac_max_power_budget_violation_w",
        "eval_isac_max_sensing_power_cap_violation_w",
        "eval_evidence_oracle_reconstruction_max_error",
    }
    min_columns = {
        "eval_inter_uav_min_distance_m",
        "eval_inter_uav_continuous_min_distance_m",
    }
    sum_columns = {
        "eval_movement_safety_fail_closed_calls",
        "eval_movement_safety_fail_closed_outside_invariant_calls",
    }
    for col in columns:
        values = [_parse(row.get(col, "")) for row in rows]
        if all(isinstance(v, list) for v in values):
            merged[col] = repr([item for v in values for item in v])
        elif all(isinstance(v, (int, float)) for v in values):
            numeric = [float(value) for value in values]
            if col in max_columns:
                aggregate = max(numeric)
            elif col in min_columns:
                aggregate = min(numeric)
            elif col in sum_columns:
                aggregate = sum(numeric)
            else:
                aggregate = sum(numeric) / len(numeric)
            merged[col] = repr(aggregate)
        else:
            merged[col] = rows[0].get(col, "")
    for col, raw in list(merged.items()):
        if not col.startswith("eval_episode_") or col == "eval_episode_seeds":
            continue
        values = _parse(raw)
        summary_col = "eval_" + col[len("eval_episode_"):]
        if (summary_col in merged and isinstance(values, list) and values
                and all(isinstance(value, (int, float)) for value in values)):
            merged[summary_col] = repr(sum(float(v) for v in values) / len(values))
    steady = _parse(merged.get("eval_episode_steady_P_D", ""))
    weak3 = _parse(merged.get("eval_episode_weak3_P_D", ""))
    worst = _parse(merged.get("eval_episode_worst_P_D", ""))
    if (all(isinstance(values, list) for values in (steady, weak3, worst))
            and len(steady) == len(weak3) == len(worst) and steady):
        tolerance = float(_parse(merged.get("eval_qos_tol", "0")) or 0.0)
        feasible = sum(
            float(s) >= 0.80 - tolerance
            and float(w) >= 0.70 - tolerance
            and float(wo) >= 0.60 - tolerance
            for s, w, wo in zip(steady, weak3, worst))
        merged["eval_qos_feasible_rate"] = repr(feasible / len(steady))
        merged["eval_qos_feasible_wilson_lcb"] = repr(
            _wilson_lower(feasible, len(steady)))
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--final-eval-seeds", required=True,
                    help="comma-separated seed list")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed per-seed CSVs in the same out-dir")
    ap.add_argument("--structure-student-checkpoint", default=None)
    ap.add_argument("--structure-student-channel", action="store_true")
    ap.add_argument("--structure-student-allow-scale-migration",
                    action="store_true")
    ap.add_argument("--structure-student-send-mode", default="every_frame")
    ap.add_argument("--structure-student-bits-per-dim", type=int, default=8)
    ap.add_argument("--structure-student-adaptive-min-bits-per-dim",
                    type=int, default=0)
    ap.add_argument("--episodes", type=int, default=0,
                    help="training episodes per worker; 0 = eval-only "
                         "(the certification protocol)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.final_eval_seeds.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [PYTHON, str(RUN_MAPPO), "--config", args.config]
    if args.warm_start:
        base_cmd += ["--warm-start", args.warm_start]
    if args.episodes >= 0:
        base_cmd += ["--episodes", str(args.episodes)]
    if args.structure_student_checkpoint:
        base_cmd += ["--structure-student-checkpoint",
                     args.structure_student_checkpoint]
        if args.structure_student_channel:
            base_cmd += ["--structure-student-channel"]
        if args.structure_student_allow_scale_migration:
            base_cmd += ["--structure-student-allow-scale-migration"]
        base_cmd += ["--structure-student-send-mode",
                     args.structure_student_send_mode,
                     "--structure-student-bits-per-dim",
                     str(args.structure_student_bits_per_dim)]
        if args.structure_student_adaptive_min_bits_per_dim > 0:
            base_cmd += ["--structure-student-adaptive-min-bits-per-dim",
                         str(args.structure_student_adaptive_min_bits_per_dim)]

    crashed: list[int] = []
    rows: list[dict[str, str]] = []
    started = time.perf_counter()
    queue = list(seeds)
    running: dict[int, subprocess.Popen] = {}
    running_logs: dict[int, object] = {}
    seed_dirs: dict[int, Path] = {}

    def _launch(seed: int) -> None:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        env = dict(os.environ)
        env["MKL_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        cmd = base_cmd + [
            "--final-eval-seeds", str(seed),
            "--out-dir", str(seed_dir),
        ]
        log = open(seed_dir / "worker.log", "w", encoding="utf-8")
        running[seed] = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        running_logs[seed] = log
        seed_dirs[seed] = seed_dir

    if args.resume:
        pending = []
        for seed in queue:
            csv_path = out_dir / f"seed_{seed}" / "paired_eval.csv"
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as fh:
                    row = next(csv.DictReader(fh), None)
                parsed_seeds = (
                    _parse(row.get("eval_episode_seeds", ""))
                    if row is not None else None)
                if parsed_seeds == [seed]:
                    rows.append(row)
                    seed_dirs[seed] = csv_path.parent
                    print(f"seed {seed}: RESUMED")
                    continue
            pending.append(seed)
        queue = pending

    # launch initial workers
    while queue and len(running) < max(1, args.max_workers):
        _launch(queue.pop(0))

    while running:
        for seed in list(running):
            ret = running[seed].poll()
            if ret is None:
                continue
            running[seed].wait()
            running.pop(seed)
            running_logs.pop(seed).close()
            csv_path = seed_dirs[seed] / "paired_eval.csv"
            if ret == 0 and csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as fh:
                    row = next(csv.DictReader(fh), None)
                if row is not None:
                    rows.append(row)
                    print(f"seed {seed}: OK")
                else:
                    crashed.append(seed)
                    print(f"seed {seed}: empty paired_eval (fail-closed)")
            else:
                crashed.append(seed)
                print(f"seed {seed}: CRASHED (exit {ret}, fail-closed)")
            if queue:
                _launch(queue.pop(0))

    elapsed = time.perf_counter() - started

    if rows:
        rows.sort(key=lambda row: int(
            _parse(row.get("eval_episode_seeds", "[0]"))[0]))
        merged = _merge_rows(rows)
        with open(out_dir / "paired_eval.csv", "w", newline="",
                  encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(merged.keys()))
            writer.writeheader()
            writer.writerow(merged)
    manifest = {
        "schema_version": 1,
        "tool": "tools/crash_isolated_seed_eval.py (P1-5)",
        "config": args.config,
        "warm_start": args.warm_start,
        "episodes": args.episodes,
        "structure_student": {
            "checkpoint": args.structure_student_checkpoint,
            "channel": bool(args.structure_student_channel),
            "allow_scale_migration": bool(
                args.structure_student_allow_scale_migration),
            "send_mode": args.structure_student_send_mode,
            "bits_per_dim": args.structure_student_bits_per_dim,
            "adaptive_min_bits_per_dim": (
                args.structure_student_adaptive_min_bits_per_dim),
        },
        "requested_seeds": seeds,
        "completed_seeds": sorted({
            int(s)
            for row in rows
            for s in _parse(row.get("eval_episode_seeds", ""))
            if isinstance(_parse(row.get("eval_episode_seeds", "")), list)
        }),
        "crashed_seeds_fail_closed": sorted(crashed),
        "worker_env": {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        "resume_enabled": bool(args.resume),
        "elapsed_seconds": float(elapsed),
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({
        "completed": len(rows),
        "crashed_fail_closed": sorted(crashed),
        "elapsed_seconds": round(elapsed, 1),
    }, indent=2))
    return 0 if not crashed else 1


if __name__ == "__main__":
    sys.exit(main())
