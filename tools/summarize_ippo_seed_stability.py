#!/usr/bin/env python
"""Summarize three critic-aligned IPPO fine-tuning seeds.

Each IPPO run is paired with the frozen commitment policy trained with the
same commitment-head seed.  All policies share the same foundation actor and
100-scenario test bank, so this measures fine-tuning/head-seed sensitivity,
not full end-to-end random-initialization variance.
"""

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
        exact_mcnemar_p,
        load_run,
    )
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from summarize_algorithm_baselines import (  # type: ignore[no-redef]
        SCALARS,
        bootstrap_mean_ci,
        exact_mcnemar_p,
        load_run,
    )


SEEDS = (42, 123, 456)
IPPO_DIR = "paper_ippo_top1_seed{seed}_critic_aligned_train40_test100"
FROZEN_DIR = "paper_top1_training_seed{seed}_formal_test100"


def validate_ippo_manifest(path: Path, seed: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "seed": int(seed),
        "total_episodes": 40,
        "total_frames": 81_920,
        "final_eval_seed_split": "test",
        "max_final_eval_seeds": 100,
        "warm_start": (
            f"results/paper_training_seed{seed}_formal/"
            "qos_commitment_pretrained.pt"),
        "centralized_critic": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"protocol mismatch in {path}: {key}={manifest.get(key)!r}, "
                f"expected {value!r}")
    return expected


def aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metrics = list(SCALARS) + ["worst_delta_vs_frozen"]
    for metric in metrics:
        values = np.asarray(
            [row[f"{prefix}_{metric}"] if prefix else row[metric]
             for row in rows], dtype=np.float64)
        output[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return output


def build(
    results_root: Path,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20_260_723,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(bootstrap_seed))
    rows = []
    provenance = []
    for seed in SEEDS:
        ippo_dir = results_root / IPPO_DIR.format(seed=seed)
        frozen_dir = results_root / FROZEN_DIR.format(seed=seed)
        ippo = load_run(ippo_dir / "paired_eval.csv")
        frozen = load_run(frozen_dir / "paired_eval.csv")
        delta = ippo["arrays"]["worst"] - frozen["arrays"]["worst"]
        ippo_only = int(np.sum(ippo["qos_mask"] & ~frozen["qos_mask"]))
        frozen_only = int(np.sum(~ippo["qos_mask"] & frozen["qos_mask"]))
        row: dict[str, Any] = {
            "seed": int(seed),
            **{f"ippo_{name}": ippo[name] for name in SCALARS},
            **{f"frozen_{name}": frozen[name] for name in SCALARS},
            "ippo_worst_delta_vs_frozen": float(delta.mean()),
        }
        ci = bootstrap_mean_ci(delta, bootstrap_samples, rng)
        row["ippo_worst_delta_ci95_low"] = ci[0]
        row["ippo_worst_delta_ci95_high"] = ci[1]
        row["ippo_qos_delta_vs_frozen"] = float(
            ippo["qos_feasible"] - frozen["qos_feasible"])
        row["ippo_only_qos_successes"] = ippo_only
        row["frozen_only_qos_successes"] = frozen_only
        row["qos_exact_mcnemar_p"] = exact_mcnemar_p(
            ippo_only, frozen_only)
        rows.append(row)
        provenance.append(validate_ippo_manifest(
            ippo_dir / "run_manifest.json", seed))

    # A convenience field lets aggregate() treat the frozen rows uniformly.
    frozen_rows = []
    for row in rows:
        item = {f"frozen_{name}": row[f"frozen_{name}"] for name in SCALARS}
        item["frozen_worst_delta_vs_frozen"] = 0.0
        frozen_rows.append(item)

    return {
        "scope": (
            "IPPO fine-tuning/head-seed sensitivity with a shared foundation "
            "actor; not end-to-end random-initialization variance"),
        "protocol": {
            "seeds": list(SEEDS),
            "ppo_updates": 40,
            "frames_per_seed": 81_920,
            "test_scenarios_per_seed": 100,
            "bootstrap_samples_per_seed": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "rows": rows,
        "ippo_aggregate": aggregate(rows, "ippo"),
        "frozen_aggregate": aggregate(frozen_rows, "frozen"),
        "provenance": provenance,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(report: dict[str, Any]) -> str:
    rows = report["rows"]
    ippo = report["ippo_aggregate"]
    frozen = report["frozen_aggregate"]
    lines = [
        "# IPPO fine-tuning seed stability",
        "",
        "Each critic-aligned IPPO run uses 40 PPO updates (81,920 frames) and "
        "is compared with the frozen commitment policy carrying the same "
        "commitment-head seed. All rows use the same 100 fixed test scenarios.",
        "",
        "| Seed | IPPO steady | IPPO weak3 | IPPO worst | worst LCB | CVaR20 | QoS | frozen worst | IPPO-frozen worst (95% paired CI) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['ippo_steady']:.4f} | "
            f"{row['ippo_weak3']:.4f} | {row['ippo_worst']:.4f} | "
            f"{row['ippo_worst_lcb']:.4f} | {row['ippo_cvar20']:.4f} | "
            f"{row['ippo_qos_feasible']:.2f} | "
            f"{row['frozen_worst']:.4f} | "
            f"{row['ippo_worst_delta_vs_frozen']:+.4f} "
            f"[{row['ippo_worst_delta_ci95_low']:+.4f}, "
            f"{row['ippo_worst_delta_ci95_high']:+.4f}] |")
    lines.extend([
        "",
        "Across the three seeds:",
        "",
        f"- IPPO worst: {ippo['worst']['mean']:.4f} +/- "
        f"{ippo['worst']['sample_std']:.4f}",
        f"- frozen worst: {frozen['worst']['mean']:.4f} +/- "
        f"{frozen['worst']['sample_std']:.4f}",
        f"- seed-level IPPO-frozen worst delta: "
        f"{ippo['worst_delta_vs_frozen']['mean']:+.4f} +/- "
        f"{ippo['worst_delta_vs_frozen']['sample_std']:.4f}",
        f"- IPPO CVaR20: {ippo['cvar20']['mean']:.4f} +/- "
        f"{ippo['cvar20']['sample_std']:.4f}",
        f"- frozen CVaR20: {frozen['cvar20']['mean']:.4f} +/- "
        f"{frozen['cvar20']['sample_std']:.4f}",
        "",
        "All IPPO seeds satisfy the Medium average thresholds, but the seed42 "
        "gain does not replicate: seeds 123 and 456 both reduce worst and tail "
        "performance relative to their frozen counterparts. IPPO fine-tuning "
        "therefore adds variance without a reproducible mean gain under the "
        "current protocol. The frozen policy remains the defensible primary "
        "method until a more stable fine-tuning rule is demonstrated.",
        "",
        "These are fine-tuning/head seeds with a shared foundation actor, not "
        "full end-to-end random-initialization seeds.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    results_root = Path("results")
    output = results_root / "paper_ippo_training_stability"
    output.mkdir(parents=True, exist_ok=True)
    report = build(results_root)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "ippo_training_seeds.csv", report["rows"])
    Path("results/current/ippo_training_seed_stability.md").write_text(
        markdown(report), encoding="utf-8")
    print(f"wrote {output / 'summary.json'}")
    print(f"wrote {output / 'ippo_training_seeds.csv'}")
    print("wrote results/current/ippo_training_seed_stability.md")


if __name__ == "__main__":
    main()
