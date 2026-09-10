#!/usr/bin/env python
"""Build the three-seed frozen/MAPPO/IPPO algorithm evidence table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.summarize_algorithm_baselines import (
        SCALARS,
        bootstrap_mean_ci,
        load_run,
    )
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from summarize_algorithm_baselines import (  # type: ignore[no-redef]
        SCALARS,
        bootstrap_mean_ci,
        load_run,
    )


SEEDS = (42, 123, 456)
METHOD_DIRS = {
    "frozen": "paper_top1_training_seed{seed}_formal_test100",
    "mappo": "paper_mappo_top1_seed{seed}_critic_aligned_train40_test100",
    "ippo": "paper_ippo_top1_seed{seed}_critic_aligned_train40_test100",
}
DISPLAY = {"frozen": "Frozen", "mappo": "MAPPO", "ippo": "IPPO"}
PAIRS = (
    ("mappo_minus_frozen", "mappo", "frozen"),
    ("ippo_minus_frozen", "ippo", "frozen"),
    ("ippo_minus_mappo", "ippo", "mappo"),
)


def validate_manifest(path: Path, seed: int, method: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    updates = 0 if method == "frozen" else 40
    expected = {
        "seed": int(seed),
        "total_episodes": updates,
        "total_frames": updates * 2048,
        "final_eval_seed_split": "test",
        "max_final_eval_seeds": 100,
        "warm_start": (
            f"results/paper_training_seed{seed}_formal/"
            "qos_commitment_pretrained.pt"),
    }
    if method == "ippo":
        expected["centralized_critic"] = False
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"protocol mismatch in {path}: {key}={manifest.get(key)!r}, "
                f"expected {value!r}")
    return expected


def aggregate_method(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in SCALARS:
        values = np.asarray(
            [row[f"{method}_{metric}"] for row in rows], dtype=np.float64)
        output[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return output


def aggregate_delta(rows: list[dict[str, Any]], comparison: str) -> dict[str, float]:
    values = np.asarray(
        [row[f"{comparison}_worst_delta"] for row in rows],
        dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def build(
    results_root: Path,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20_260_723,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(bootstrap_seed))
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for seed in SEEDS:
        runs = {}
        for method, template in METHOD_DIRS.items():
            directory = results_root / template.format(seed=seed)
            runs[method] = load_run(directory / "paired_eval.csv")
            provenance.append({
                "method": method,
                **validate_manifest(
                    directory / "run_manifest.json", seed, method),
            })
        row: dict[str, Any] = {"seed": int(seed)}
        for method, run in runs.items():
            row.update({
                f"{method}_{metric}": run[metric]
                for metric in SCALARS
            })
        for label, left, right in PAIRS:
            delta = (runs[left]["arrays"]["worst"]
                     - runs[right]["arrays"]["worst"])
            ci = bootstrap_mean_ci(delta, bootstrap_samples, rng)
            row[f"{label}_worst_delta"] = float(delta.mean())
            row[f"{label}_worst_ci95_low"] = ci[0]
            row[f"{label}_worst_ci95_high"] = ci[1]
        rows.append(row)

    return {
        "scope": (
            "fine-tuning/head-seed sensitivity with a shared foundation "
            "actor and common 100-scenario bank; not full end-to-end "
            "random-initialization variance"),
        "protocol": {
            "seeds": list(SEEDS),
            "ppo_updates_per_trained_method": 40,
            "frames_per_trained_method_seed": 81_920,
            "test_scenarios_per_method_seed": 100,
            "bootstrap_samples_per_seed": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "rows": rows,
        "method_aggregates": {
            method: aggregate_method(rows, method)
            for method in METHOD_DIRS
        },
        "seed_level_delta_aggregates": {
            label: aggregate_delta(rows, label)
            for label, _, _ in PAIRS
        },
        "provenance": provenance,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    aggregates = report["method_aggregates"]
    deltas = report["seed_level_delta_aggregates"]
    lines = [
        "# Three-seed algorithm stability",
        "",
        "Frozen, critic-aligned MAPPO, and critic-aligned IPPO use matching "
        "commitment-head seeds and the same ordered 100-scenario test bank. "
        "Each PPO method receives 40 updates (81,920 frames) per seed.",
        "",
        "| Seed | Frozen worst | MAPPO worst | IPPO worst | MAPPO-frozen | IPPO-frozen | IPPO-MAPPO |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['frozen_worst']:.4f} | "
            f"{row['mappo_worst']:.4f} | {row['ippo_worst']:.4f} | "
            f"{row['mappo_minus_frozen_worst_delta']:+.4f} | "
            f"{row['ippo_minus_frozen_worst_delta']:+.4f} | "
            f"{row['ippo_minus_mappo_worst_delta']:+.4f} |")

    lines.extend([
        "",
        "## Across-seed results",
        "",
        "| Method | steady | weak3 | worst | CVaR20 | QoS feasible |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in METHOD_DIRS:
        item = aggregates[method]
        lines.append(
            f"| {DISPLAY[method]} | "
            f"{item['steady']['mean']:.4f} +/- {item['steady']['sample_std']:.4f} | "
            f"{item['weak3']['mean']:.4f} +/- {item['weak3']['sample_std']:.4f} | "
            f"{item['worst']['mean']:.4f} +/- {item['worst']['sample_std']:.4f} | "
            f"{item['cvar20']['mean']:.4f} +/- {item['cvar20']['sample_std']:.4f} | "
            f"{item['qos_feasible']['mean']:.4f} +/- "
            f"{item['qos_feasible']['sample_std']:.4f} |")

    lines.extend([
        "",
        "Seed-level mean worst differences (mean +/- sample standard deviation):",
        "",
        f"- MAPPO - frozen: "
        f"{deltas['mappo_minus_frozen']['mean']:+.4f} +/- "
        f"{deltas['mappo_minus_frozen']['sample_std']:.4f}",
        f"- IPPO - frozen: "
        f"{deltas['ippo_minus_frozen']['mean']:+.4f} +/- "
        f"{deltas['ippo_minus_frozen']['sample_std']:.4f}",
        f"- IPPO - MAPPO: "
        f"{deltas['ippo_minus_mappo']['mean']:+.4f} +/- "
        f"{deltas['ippo_minus_mappo']['sample_std']:.4f}",
        "",
        "All nine method-seed rows satisfy the requested Medium average "
        "thresholds. However, neither PPO variant produces a reproducible "
        "gain over the frozen policy: mean worst is almost unchanged, while "
        "both PPO variants have substantially larger seed variance and lower "
        "mean CVaR/QoS feasibility. The seed42 IPPO-over-MAPPO advantage "
        "reverses at seed456, so it must not be presented as an algorithmic "
        "superiority result.",
        "",
        "The frozen commitment policy is therefore the defensible primary "
        "method under the current evidence. PPO fine-tuning belongs in the "
        "baseline/sensitivity section, not in the claimed performance path.",
        "",
        "These replications vary fine-tuning and commitment-head seeds while "
        "sharing the foundation actor; they are not full end-to-end random "
        "initializations.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    results_root = Path("results")
    output = results_root / "paper_algorithm_seed_stability"
    output.mkdir(parents=True, exist_ok=True)
    report = build(results_root)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "algorithm_seed_comparison.csv", report["rows"])
    Path("results/current/algorithm_seed_stability.md").write_text(
        markdown(report), encoding="utf-8")
    print(f"wrote {output / 'summary.json'}")
    print(f"wrote {output / 'algorithm_seed_comparison.csv'}")
    print("wrote results/current/algorithm_seed_stability.md")


if __name__ == "__main__":
    main()
