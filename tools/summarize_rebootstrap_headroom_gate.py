#!/usr/bin/env python
"""Summarize the Gate C1.7b periodic-N5 rebootstrap headroom test."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from pathlib import Path


METRICS = (
    "eval_steady_P_D",
    "eval_weak3_P_D",
    "eval_worst_P_D",
    "eval_worst_cvar",
    "eval_qos_feasible_rate",
    "eval_local_search_exact_verifications_per_resolve",
    "eval_local_search_rebootstrap_attempt_rate",
    "eval_local_search_rebootstrap_accept_rate",
    "eval_local_search_rebootstrap_accept_given_attempt",
    "eval_local_search_rebootstrap_candidates_per_resolve",
    "eval_p0_solve_time_s_per_resolve",
    "eval_isac_max_power_balance_error_w",
)


def _read(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    result: dict[str, object] = {
        key: float(row.get(key, 0.0) or 0.0) for key in METRICS
    }
    result["episode_seeds"] = [
        int(value) for value in ast.literal_eval(row["eval_episode_seeds"])]
    result["episode_worst"] = [
        float(value)
        for value in ast.literal_eval(row["eval_episode_worst_values"])]
    return result


def _bootstrap_mean_ci(
        values: list[float], *, samples: int = 20_000,
        seed: int = 17) -> list[float]:
    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples))
    return [
        means[int(0.025 * (samples - 1))],
        means[int(0.975 * (samples - 1))],
    ]


def run(args: argparse.Namespace) -> dict[str, object]:
    controllers = {
        "local_only_top5": _read(args.local),
        "periodic_n5_h5": _read(args.periodic_h5),
        "replicated_global_reference": _read(args.replicated),
    }
    seeds = controllers["local_only_top5"]["episode_seeds"]
    if any(value["episode_seeds"] != seeds for value in controllers.values()):
        raise ValueError("controller seed order does not match")

    local = controllers["local_only_top5"]
    h5 = controllers["periodic_n5_h5"]
    replicated = controllers["replicated_global_reference"]
    episodes = []
    for seed, local_worst, h5_worst, global_worst in zip(
            seeds,
            local["episode_worst"],
            h5["episode_worst"],
            replicated["episode_worst"]):
        episodes.append({
            "seed": seed,
            "local_worst": local_worst,
            "periodic_h5_worst": h5_worst,
            "replicated_global_worst": global_worst,
            "periodic_minus_local": h5_worst - local_worst,
            "global_minus_periodic": global_worst - h5_worst,
        })

    # Episode differences below 1e-4 are numerical/trajectory tie noise rather
    # than a meaningful controller change.
    tolerance = 1.0e-4
    improved = sum(
        item["periodic_minus_local"] > tolerance for item in episodes)
    tied = sum(
        abs(item["periodic_minus_local"]) <= tolerance for item in episodes)
    harmed = sum(
        item["periodic_minus_local"] < -tolerance for item in episodes)
    paired_worst_gains = [
        item["periodic_minus_local"] for item in episodes]
    reaches_target = h5["eval_worst_P_D"] >= args.min_mean_worst
    report = {
        "protocol": "gate_c1_7b_global_rebootstrap_headroom_v1",
        "scope": (
            "10-seed development split; frozen Actor and Student; Hold=5; "
            "learned Top-5 local maintenance followed by an exact N5 check "
            "on every eligible warm resolve; only strictly positive public-"
            "proxy N5 moves are accepted"),
        "seeds": seeds,
        "controllers": controllers,
        "episode_comparison": episodes,
        "periodic_h5_relative_to_local": {
            "steady_gain": (
                h5["eval_steady_P_D"] - local["eval_steady_P_D"]),
            "weak3_gain": (
                h5["eval_weak3_P_D"] - local["eval_weak3_P_D"]),
            "mean_worst_gain": (
                h5["eval_worst_P_D"] - local["eval_worst_P_D"]),
            "mean_worst_gain_bootstrap_ci95": _bootstrap_mean_ci(
                paired_worst_gains),
            "cvar_gain": (
                h5["eval_worst_cvar"] - local["eval_worst_cvar"]),
            "improved_episode_count": improved,
            "tied_episode_count": tied,
            "harmed_episode_count": harmed,
        },
        "remaining_gap_to_replicated_global": {
            "steady": (
                replicated["eval_steady_P_D"] - h5["eval_steady_P_D"]),
            "weak3": (
                replicated["eval_weak3_P_D"] - h5["eval_weak3_P_D"]),
            "mean_worst": (
                replicated["eval_worst_P_D"] - h5["eval_worst_P_D"]),
            "cvar": (
                replicated["eval_worst_cvar"] - h5["eval_worst_cvar"]),
        },
        "gate": {
            "target": {"min_mean_worst": args.min_mean_worst},
            "pass": reaches_target,
            "run_h10_h20": reaches_target,
        },
        "decision": {
            "periodic_n5": (
                "reject as the final reconfiguration mechanism: it improves "
                "the mean but misses the registered 0.60 mean-worst gate, "
                "materially harms three episodes, and requires near-full "
                "warm-frame N5 "
                "enumeration"),
            "h10_h20": (
                "stop: lower-frequency variants cannot establish more "
                "headroom than the every-eligible-resolve H=5 test"),
            "learned_trigger": (
                "defer: trigger labels are not trustworthy until accepted "
                "single-frame proxy gains align with multi-frame physical "
                "worst gains"),
            "next_gate": (
                "common-state proxy-alignment audit, followed only if needed "
                "by exact block-LNS over jointly selected UAV/target blocks"),
        },
        "fresh_test_consumed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "episode_comparison.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=episodes[0].keys())
        writer.writeheader()
        writer.writerows(episodes)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--periodic-h5", type=Path, required=True)
    parser.add_argument("--replicated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-mean-worst", type=float, default=0.60)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
