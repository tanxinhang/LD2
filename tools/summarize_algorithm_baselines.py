#!/usr/bin/env python
"""Summarize the critic-aligned frozen/MAPPO/IPPO paper comparison.

The three policies are evaluated on the same ordered 100-scenario test bank.
Episode arrays stored inside ``paired_eval.csv`` are therefore compared with
paired bootstrap intervals.  Earlier MAPPO/IPPO runs are retained in the
machine-readable report only as invalid diagnostics: their PPO critic inputs
did not match rollout-time field order, and the legacy IPPO update also zeroed
the communication-summary field.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FORMAL_RUNS = {
    "Frozen commitment (0 PPO)":
        "paper_top1_training_seed42_formal_test100",
    "MAPPO fine-tune (critic aligned)":
        "paper_mappo_top1_seed42_critic_aligned_train40_test100",
    "IPPO fine-tune (critic aligned)":
        "paper_ippo_top1_seed42_critic_aligned_train40_test100",
}
LEGACY_INVALID_RUNS = {
    "Legacy MAPPO (excluded)": "paper_mappo_top1_seed42_train40_test100",
    "Legacy IPPO (excluded)": "paper_ippo_top1_seed42_train40_test100",
}
COMPARISONS = (
    ("MAPPO minus frozen",
     "MAPPO fine-tune (critic aligned)", "Frozen commitment (0 PPO)"),
    ("IPPO minus frozen",
     "IPPO fine-tune (critic aligned)", "Frozen commitment (0 PPO)"),
    ("IPPO minus MAPPO",
     "IPPO fine-tune (critic aligned)",
     "MAPPO fine-tune (critic aligned)"),
)

SCALARS = {
    "steady": "eval_steady_P_D",
    "weak3": "eval_weak3_P_D",
    "worst": "eval_worst_P_D",
    "worst_lcb": "eval_worst_lcb",
    "cvar20": "eval_worst_cvar",
    "qos_feasible": "eval_qos_feasible_rate",
    "qos_wilson_lcb": "eval_qos_feasible_wilson_lcb",
    "bits_per_frame": "eval_comm_bits_per_frame",
}

QOS_FLOORS = {"steady": 0.80, "weak3": 0.70, "worst": 0.60}


def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    indices = rng.integers(0, values.size,
                           size=(max(1, int(samples)), values.size))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    """Return the exact two-sided binomial McNemar p value."""
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    tail = min(int(left_only), int(right_only))
    probability = sum(math.comb(discordant, k)
                      for k in range(tail + 1)) / (2 ** discordant)
    return float(min(1.0, 2.0 * probability))


def qos_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.logical_and.reduce([
        arrays[name] >= threshold
        for name, threshold in QOS_FLOORS.items()
    ])


def load_run(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one aggregate row in {path}, got {len(rows)}")
    raw = rows[0]
    arrays = {
        name: np.asarray(ast.literal_eval(raw[f"eval_episode_{name}_P_D"]),
                         dtype=np.float64)
        for name in ("steady", "weak3", "worst")
    }
    if {values.size for values in arrays.values()} != {100}:
        raise ValueError(f"{path} is not a matched 100-scenario evaluation")
    feasible = qos_mask(arrays)
    recorded_feasible = float(raw[SCALARS["qos_feasible"]])
    if not np.isclose(feasible.mean(), recorded_feasible, atol=1e-12):
        raise ValueError(
            f"QoS mask mismatch in {path}: derived={feasible.mean()}, "
            f"recorded={recorded_feasible}")
    return {
        "source": str(path),
        "arrays": arrays,
        "qos_mask": feasible,
        **{name: float(raw[column]) for name, column in SCALARS.items()},
    }


def validate_manifest(path: Path, expected_episodes: int) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "seed": 42,
        "total_episodes": expected_episodes,
        "total_frames": expected_episodes * 2048,
        "final_eval_seed_split": "test",
        "max_final_eval_seeds": 100,
        "warm_start": (
            "results/paper_training_seed42_formal/"
            "qos_commitment_pretrained.pt"),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"protocol mismatch in {path}: {key}={manifest.get(key)!r}, "
                f"expected {value!r}")
    return {key: manifest[key] for key in expected}


def scalar_row(name: str, run: dict[str, Any], episodes: int) -> dict[str, Any]:
    return {
        "method": name,
        "ppo_updates": int(episodes),
        "frames": int(episodes * 2048),
        **{metric: run[metric] for metric in SCALARS},
    }


def compare(
    label: str,
    left_name: str,
    right_name: str,
    runs: dict[str, dict[str, Any]],
    samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    left = runs[left_name]
    right = runs[right_name]
    metric_deltas: dict[str, Any] = {}
    for metric in ("steady", "weak3", "worst"):
        delta = left["arrays"][metric] - right["arrays"][metric]
        metric_deltas[metric] = {
            "mean_delta": float(delta.mean()),
            "bootstrap_ci95": bootstrap_mean_ci(delta, samples, rng),
            "left_better": int(np.sum(delta > 1e-12)),
            "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "left_worse": int(np.sum(delta < -1e-12)),
        }
    left_only = int(np.sum(left["qos_mask"] & ~right["qos_mask"]))
    right_only = int(np.sum(~left["qos_mask"] & right["qos_mask"]))
    return {
        "comparison": label,
        "left": left_name,
        "right": right_name,
        "metrics": metric_deltas,
        "qos_feasible_delta": float(
            left["qos_feasible"] - right["qos_feasible"]),
        "qos_left_only_successes": left_only,
        "qos_right_only_successes": right_only,
        "qos_exact_mcnemar_p": exact_mcnemar_p(left_only, right_only),
    }


def build(
    results_root: Path,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20_260_723,
) -> dict[str, Any]:
    formal = {
        name: load_run(results_root / directory / "paired_eval.csv")
        for name, directory in FORMAL_RUNS.items()
    }
    manifests = {}
    rows = []
    for name, directory in FORMAL_RUNS.items():
        episodes = 0 if name.startswith("Frozen") else 40
        manifests[name] = validate_manifest(
            results_root / directory / "run_manifest.json", episodes)
        rows.append(scalar_row(name, formal[name], episodes))

    rng = np.random.default_rng(int(bootstrap_seed))
    comparisons = [
        compare(label, left, right, formal, bootstrap_samples, rng)
        for label, left, right in COMPARISONS
    ]

    invalid = []
    for name, directory in LEGACY_INVALID_RUNS.items():
        run = load_run(results_root / directory / "paired_eval.csv")
        invalid.append({
            "name": name,
            "source": run["source"],
            "recorded_worst_for_diagnosis_only": run["worst"],
            "status": "excluded",
            "reason": (
                "PPO critic update fields did not match rollout order; legacy "
                "IPPO additionally zeroed the communication-summary field."),
        })

    return {
        "scope": (
            "single commitment-head seed (42), common foundation actor and "
            "matched 100-scenario test bank; not a multi-seed end-to-end "
            "algorithm comparison"),
        "protocol": {
            "training_seed": 42,
            "ppo_updates": 40,
            "frames_per_trained_method": 81_920,
            "test_scenarios": 100,
            "qos_floors": QOS_FLOORS,
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "critic_update_order": "base/local observation, agent one-hot, communication summary",
        },
        "formal_rows": rows,
        "formal_manifests": manifests,
        "paired_comparisons": comparisons,
        "invalid_legacy_diagnostics": invalid,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison in report["paired_comparisons"]:
        row: dict[str, Any] = {
            "comparison": comparison["comparison"],
            "qos_feasible_delta": comparison["qos_feasible_delta"],
            "qos_left_only_successes":
                comparison["qos_left_only_successes"],
            "qos_right_only_successes":
                comparison["qos_right_only_successes"],
            "qos_exact_mcnemar_p": comparison["qos_exact_mcnemar_p"],
        }
        for metric, values in comparison["metrics"].items():
            row[f"{metric}_mean_delta"] = values["mean_delta"]
            row[f"{metric}_ci95_low"] = values["bootstrap_ci95"][0]
            row[f"{metric}_ci95_high"] = values["bootstrap_ci95"][1]
            row[f"{metric}_left_better"] = values["left_better"]
            row[f"{metric}_ties"] = values["ties"]
            row[f"{metric}_left_worse"] = values["left_worse"]
        rows.append(row)
    return rows


def markdown(report: dict[str, Any]) -> str:
    rows = report["formal_rows"]
    comparisons = report["paired_comparisons"]
    by_name = {item["comparison"]: item for item in comparisons}
    ippo_frozen = by_name["IPPO minus frozen"]
    ippo_mappo = by_name["IPPO minus MAPPO"]
    mappo_frozen = by_name["MAPPO minus frozen"]

    lines = [
        "# Critic-aligned algorithm baseline results",
        "",
        "The formal comparison uses commitment-head seed 42, a shared "
        "foundation actor, identical Top-1 configuration, 40 PPO updates "
        "(81,920 frames) for each trained method, and the same ordered 100 "
        "test scenarios. The frozen row receives zero PPO updates.",
        "",
        "| Method | updates | steady | weak3 | worst | worst LCB | CVaR20 | QoS feasible | bit/frame |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['ppo_updates']} | "
            f"{row['steady']:.4f} | {row['weak3']:.4f} | "
            f"{row['worst']:.4f} | {row['worst_lcb']:.4f} | "
            f"{row['cvar20']:.4f} | {row['qos_feasible']:.2f} | "
            f"{row['bits_per_frame']:.0f} |")

    lines.extend([
        "",
        "All three rows meet the requested Medium average thresholds "
        "(steady >= 0.80, weak3 >= 0.70, mean worst >= 0.60). That threshold "
        "result is distinct from proving that one learning algorithm is "
        "superior.",
        "",
        "## Matched test-bank differences",
        "",
        "| Left minus right | worst delta (95% paired bootstrap CI) | steady delta | weak3 delta | QoS delta | exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for comparison in comparisons:
        worst = comparison["metrics"]["worst"]
        steady = comparison["metrics"]["steady"]
        weak3 = comparison["metrics"]["weak3"]
        lines.append(
            f"| {comparison['comparison']} | {worst['mean_delta']:+.4f} "
            f"[{worst['bootstrap_ci95'][0]:+.4f}, "
            f"{worst['bootstrap_ci95'][1]:+.4f}] | "
            f"{steady['mean_delta']:+.4f} | {weak3['mean_delta']:+.4f} | "
            f"{comparison['qos_feasible_delta']:+.2f} | "
            f"{comparison['qos_exact_mcnemar_p']:.4f} |")

    ippo_frozen_worst = ippo_frozen["metrics"]["worst"]
    ippo_mappo_worst = ippo_mappo["metrics"]["worst"]
    mappo_frozen_worst = mappo_frozen["metrics"]["worst"]
    lines.extend([
        "",
        "On this one training seed, aligned IPPO improves mean worst over "
        f"the frozen policy by {ippo_frozen_worst['mean_delta']:+.4f} and "
        f"over aligned MAPPO by {ippo_mappo_worst['mean_delta']:+.4f}. "
        "Aligned MAPPO is effectively tied with the frozen policy: its "
        f"difference is {mappo_frozen_worst['mean_delta']:+.4f}, with a "
        "paired interval that must be consulted above. The result does not "
        "support a claim that centralized-critic MAPPO is currently better "
        "than IPPO at this budget.",
        "",
        "This seed42 observation is retained for auditability, but subsequent "
        "seed123/456 replications show that it is not stable. Across three "
        "seeds, MAPPO, IPPO, and frozen mean worst are nearly identical, while "
        "both PPO variants have larger seed variance and lower mean CVaR/QoS. "
        "See `results/current/algorithm_seed_stability.md` for the formal multi-seed "
        "conclusion.",
        "",
        "## Excluded legacy runs",
        "",
        "Earlier 40-update MAPPO/IPPO outputs are diagnostic only and are "
        "excluded from every formal table. During PPO update, critic fields "
        "were assembled in a different order from rollout; legacy IPPO also "
        "replaced the communication summary with zeros. Regression tests now "
        "enforce rollout/update alignment for both critic variants.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    results_root = Path("results")
    output = results_root / "paper_algorithm_baselines"
    output.mkdir(parents=True, exist_ok=True)
    report = build(results_root)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "algorithm_comparison.csv", report["formal_rows"])
    write_csv(output / "paired_differences.csv", paired_rows(report))
    Path("results/current/algorithm_baseline_results.md").write_text(
        markdown(report), encoding="utf-8")
    print(f"wrote {output / 'summary.json'}")
    print(f"wrote {output / 'algorithm_comparison.csv'}")
    print(f"wrote {output / 'paired_differences.csv'}")
    print("wrote results/current/algorithm_baseline_results.md")


if __name__ == "__main__":
    main()
