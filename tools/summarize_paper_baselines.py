#!/usr/bin/env python
"""Summarize the matched 100-scenario paper baselines.

The script validates that every run contains the same number of per-episode
metrics and computes a paired bootstrap interval for the strict worst-target
difference relative to the full DCB configuration. It is intentionally
read-only: Markdown or JSON is written to stdout for audit or manuscript use.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np


RUNS = OrderedDict([
    ("Pre-consensus", "qos_pretrained_soft_gated_move50_test100"),
    ("No U2U communication", "baseline_silence_test100"),
    ("No capacity-two projection", "baseline_no_capacity_test100"),
    ("No direct comm-to-sensing residual", "baseline_no_comm_sensing_test100"),
    ("Zero token values", "baseline_zero_payload_test100"),
    ("Permuted sender semantics", "baseline_permute_identity_test100"),
    ("Projected-only bid (beta=0)", "distributed_consensus_bid_token_frozen_test100"),
    ("Intrinsic-only bid (beta=1)", "distributed_consensus_intrinsic_bid_frozen_test100"),
    ("Full DCB (beta=0.5)", "distributed_consensus_hybrid50_bid_frozen_test100"),
    ("Central motion oracle", "baseline_central_movement_oracle_test100"),
])
FULL_NAME = "Full DCB (beta=0.5)"


def _load_run(csv_path: Path) -> Dict[str, object]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    arrays = {
        key: np.asarray(ast.literal_eval(row[key]), dtype=np.float64)
        for key in (
            "eval_episode_steady_P_D",
            "eval_episode_weak3_P_D",
            "eval_episode_worst_P_D",
        )
    }
    lengths = {array.size for array in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
        raise ValueError(f"invalid episode arrays in {csv_path}: {lengths}")
    return {
        "arrays": arrays,
        "n": next(iter(lengths)),
        "steady": float(row["eval_steady_P_D"]),
        "weak3": float(row["eval_weak3_P_D"]),
        "worst": float(row["eval_worst_P_D"]),
        "worst_lcb": float(row["eval_worst_lcb"]),
        "worst_cvar": float(row["eval_worst_cvar"]),
        "feasible": float(row["eval_qos_feasible_rate"]),
        "wilson": float(row["eval_qos_feasible_wilson_lcb"]),
        "bits": float(row["eval_comm_bits_per_frame"]),
        # Use the executed-action diagnostic so movement-oracle interventions
        # are measured after, rather than before, action replacement.
        "collision": float(row["eval_executed_move_collision_frame_rate"]),
        "distance": float(row["eval_mean_nearest_uav_distance_m"]),
    }


def summarize(
    results_root: Path,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20_260_722,
) -> OrderedDict[str, Dict[str, object]]:
    loaded = OrderedDict(
        (name, _load_run(results_root / directory / "paired_eval.csv"))
        for name, directory in RUNS.items()
    )
    sample_counts = {int(run["n"]) for run in loaded.values()}
    if len(sample_counts) != 1:
        raise ValueError(f"baseline episode counts differ: {sample_counts}")
    n = next(iter(sample_counts))
    full_worst = loaded[FULL_NAME]["arrays"]["eval_episode_worst_P_D"]
    rng = np.random.default_rng(int(bootstrap_seed))

    summary: OrderedDict[str, Dict[str, object]] = OrderedDict()
    for name, run in loaded.items():
        item = {key: value for key, value in run.items()
                if key not in {"arrays", "n"}}
        item["episodes"] = n
        if name != FULL_NAME:
            difference = (
                full_worst - run["arrays"]["eval_episode_worst_P_D"])
            indices = rng.integers(
                0, n, size=(max(1, int(bootstrap_samples)), n))
            bootstrap_mean = difference[indices].mean(axis=1)
            item["full_minus_variant_worst"] = float(difference.mean())
            item["paired_ci95"] = [
                float(value)
                for value in np.quantile(bootstrap_mean, [0.025, 0.975])
            ]
        summary[name] = item
    return summary


def _markdown(summary: OrderedDict[str, Dict[str, object]]) -> str:
    def format_bound(value: float) -> str:
        return f"{value:.1e}" if 0.0 < abs(value) < 5e-5 else f"{value:.4f}"

    lines = [
        "| Variant | steady | weak3 | worst | Full - variant worst [95% CI] | feasible | bit/frame |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary.items():
        if "paired_ci95" in item:
            low, high = item["paired_ci95"]
            delta = (
                f"{item['full_minus_variant_worst']:.4f} "
                f"[{format_bound(low)}, {format_bound(high)}]")
        else:
            delta = "--"
        lines.append(
            f"| {name} | {item['steady']:.4f} | {item['weak3']:.4f} | "
            f"{item['worst']:.4f} | {delta} | {item['feasible']:.2f} | "
            f"{item['bits']:.0f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_722)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = summarize(
        args.results_root,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(_markdown(summary))


if __name__ == "__main__":
    main()
