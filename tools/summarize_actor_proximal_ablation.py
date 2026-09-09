#!/usr/bin/env python
"""Audit the paired multi-seed actor--execution proximal ablation.

The independent unit is the training seed.  Evaluation episodes are paired
within each trained policy and are retained for diagnostics, but they are not
treated as independent samples for the promotion decision.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RUN_PAIRS = {
    31415: ("train-a569bdfdf44376dad496", "train-4a1339fba1bb15108023"),
    31416: ("train-69633e8a702b11b7d1b1", "train-7c279598c8feac9dbcc7"),
    31417: ("train-a5b1a0b3d260af1824ae", "train-88491f28372d2c2cf73e"),
    31418: ("train-8c15b7b4c6b1dd973415", "train-1ed42a2b0855241f5f91"),
    31419: ("train-01b672d6b080228835a3", "train-d6ee09b5fceac7a42d7c"),
}
EVAL_SEEDS = tuple(range(30001, 30011))
METRICS = ("steady", "weak3", "worst")
NUMERICAL_EQUIVALENCE_TOLERANCE = 1e-8
PROXIMAL_KEY = (
    "distributed_primal_dual_power_actor_proximal_regularization")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def _float_array(raw: str, *, field: str) -> np.ndarray:
    values = np.asarray(ast.literal_eval(raw), dtype=np.float64)
    if values.ndim != 1 or values.size != len(EVAL_SEEDS):
        raise ValueError(f"{field} is not a {len(EVAL_SEEDS)}-episode vector")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field} contains non-finite values")
    return values


def _max_training_field(path: Path, field: str) -> float:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or field not in rows[0]:
        raise ValueError(f"missing training field {field!r} in {path}")
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite training field {field!r} in {path}")
    return float(values.max())


def load_run(
    runs_root: Path, run_id: str, training_seed: int, expected_rho: float,
) -> dict[str, Any]:
    root = runs_root / run_id
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (root / "completion.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or manifest.get("state") != "created":
        raise ValueError(f"invalid creation manifest for {run_id}")
    if manifest.get("seeds") != [training_seed]:
        raise ValueError(f"training-seed mismatch for {run_id}")
    if completion.get("run_id") != run_id or completion.get("state") != "completed":
        raise ValueError(f"run is not completed: {run_id}")

    required = (
        "raw/output/paired_eval.csv",
        "raw/output/run_manifest.json",
        "raw/output/train_metrics.csv",
    )
    for relative in required:
        path = root / relative
        recorded = completion.get("artifacts", {}).get(relative)
        if not isinstance(recorded, str) or _sha256(path) != recorded:
            raise ValueError(f"completion hash mismatch for {run_id}/{relative}")

    run_manifest = json.loads(
        (root / "raw/output/run_manifest.json").read_text(encoding="utf-8"))
    marl = run_manifest["provenance"]["resolved_config"]["marl"]
    rho = float(marl[PROXIMAL_KEY])
    if not np.isclose(rho, expected_rho, rtol=0.0, atol=1e-15):
        raise ValueError(
            f"proximal coefficient mismatch for {run_id}: {rho} != {expected_rho}")

    evaluation = _single_csv_row(root / "raw/output/paired_eval.csv")
    seeds = tuple(int(value) for value in ast.literal_eval(
        evaluation["eval_episode_seeds"]))
    if seeds != EVAL_SEEDS:
        raise ValueError(f"evaluation-seed mismatch for {run_id}: {seeds}")
    arrays = {
        name: _float_array(
            evaluation[f"eval_episode_{name}_values"],
            field=f"{run_id}:{name}",
        )
        for name in METRICS
    }
    training_path = root / "raw/output/train_metrics.csv"
    return {
        "run_id": run_id,
        "training_seed": training_seed,
        "rho": rho,
        "code_identity": manifest["code_identity"],
        "config_sha256": manifest["config_sha256"],
        "artifact_sha256": {
            relative: completion["artifacts"][relative] for relative in required
        },
        "arrays": arrays,
        "qos_feasible_rate": float(evaluation["eval_qos_feasible_rate"]),
        "rf_budget_violation_w": float(
            evaluation["eval_isac_max_power_budget_violation_w"]),
        "comm_delivery_rate": float(evaluation["eval_comm_delivery_rate"]),
        "max_attempted_kl": _max_training_field(
            training_path, "post_update_attempted_approx_kl"),
        "max_cvar_residual": _max_training_field(
            training_path, "constrained_cvar_residual_max"),
        "max_temporal_rf_violation_w": _max_training_field(
            training_path, "temporal_unrolled_power_budget_violation_w"),
    }


def bootstrap_mean_ci(
    values: Iterable[float], *, samples: int, seed: int,
) -> list[float]:
    vector = np.asarray(tuple(values), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or samples <= 0:
        raise ValueError("bootstrap requires a non-empty vector and samples > 0")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, vector.size, size=(samples, vector.size))
    means = vector[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def exact_one_sided_sign_p(values: Iterable[float], tolerance: float) -> float:
    vector = np.asarray(tuple(values), dtype=np.float64)
    positive = int(np.sum(vector > tolerance))
    negative = int(np.sum(vector < -tolerance))
    non_ties = positive + negative
    if non_ties == 0:
        return 1.0
    return float(sum(comb(non_ties, k) for k in range(positive, non_ties + 1))
                 / (2 ** non_ties))


def build(
    runs_root: Path,
    *,
    bootstrap_samples: int = 100_000,
    bootstrap_seed: int = 9308,
) -> dict[str, Any]:
    pairs = []
    code_identities: set[str] = set()
    for training_seed, (baseline_id, strong_id) in RUN_PAIRS.items():
        baseline = load_run(runs_root, baseline_id, training_seed, 0.0)
        strong = load_run(runs_root, strong_id, training_seed, 0.1)
        code_identities.update((baseline["code_identity"], strong["code_identity"]))
        metric_rows = {}
        for name in METRICS:
            delta = strong["arrays"][name] - baseline["arrays"][name]
            metric_rows[name] = {
                "baseline_mean": float(baseline["arrays"][name].mean()),
                "strong_mean": float(strong["arrays"][name].mean()),
                "delta_mean": float(delta.mean()),
                "delta_median": float(np.median(delta)),
                "positive_episodes": int(np.sum(delta > 1e-12)),
                "negative_episodes": int(np.sum(delta < -1e-12)),
                "tied_episodes": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        pairs.append({
            "training_seed": training_seed,
            "baseline_run": baseline_id,
            "strong_run": strong_id,
            "metrics": metric_rows,
            "safety": {
                arm: {
                    key: run[key] for key in (
                        "qos_feasible_rate", "rf_budget_violation_w",
                        "comm_delivery_rate", "max_attempted_kl",
                        "max_cvar_residual", "max_temporal_rf_violation_w",
                    )
                }
                for arm, run in (("baseline", baseline), ("strong", strong))
            },
            "provenance": {
                "baseline": {
                    "config_sha256": baseline["config_sha256"],
                    "artifact_sha256": baseline["artifact_sha256"],
                },
                "strong": {
                    "config_sha256": strong["config_sha256"],
                    "artifact_sha256": strong["artifact_sha256"],
                },
            },
        })
    if len(code_identities) != 1:
        raise ValueError(f"runs do not share one code identity: {code_identities}")

    aggregate = {}
    for index, name in enumerate(METRICS):
        deltas = np.asarray(
            [pair["metrics"][name]["delta_mean"] for pair in pairs],
            dtype=np.float64,
        )
        inference_deltas = np.where(
            np.abs(deltas) <= NUMERICAL_EQUIVALENCE_TOLERANCE, 0.0, deltas)
        aggregate[name] = {
            "independent_unit": "training_seed",
            "n": int(deltas.size),
            "delta_mean": float(deltas.mean()),
            "delta_median": float(np.median(deltas)),
            "delta_sample_std": float(deltas.std(ddof=1)),
            "cluster_bootstrap_ci95": bootstrap_mean_ci(
                inference_deltas, samples=bootstrap_samples,
                seed=bootstrap_seed + index),
            "numerical_equivalence_tolerance": NUMERICAL_EQUIVALENCE_TOLERANCE,
            "positive_clusters_at_1e-8": int(np.sum(
                deltas > NUMERICAL_EQUIVALENCE_TOLERANCE)),
            "negative_clusters_at_1e-8": int(np.sum(
                deltas < -NUMERICAL_EQUIVALENCE_TOLERANCE)),
            "tied_clusters_at_1e-8": int(np.sum(
                np.abs(deltas) <= NUMERICAL_EQUIVALENCE_TOLERANCE)),
            "one_sided_exact_sign_p_at_1e-8": exact_one_sided_sign_p(
                deltas, NUMERICAL_EQUIVALENCE_TOLERANCE),
        }

    worst = aggregate["worst"]
    safety_thresholds = {
        "qos_feasible_rate": 1.0,
        "eval_rf_budget_violation_w_max": 1e-12,
        "training_rf_roundoff_w_max": 1e-7,
        "attempted_kl_max": 0.03,
        "cvar_primal_residual_max": 0.01,
    }
    candidate_safety_pass = all(
        pair["safety"]["strong"]["qos_feasible_rate"]
        >= safety_thresholds["qos_feasible_rate"]
        and pair["safety"]["strong"]["rf_budget_violation_w"]
        <= safety_thresholds["eval_rf_budget_violation_w_max"]
        and pair["safety"]["strong"]["max_temporal_rf_violation_w"]
        <= safety_thresholds["training_rf_roundoff_w_max"]
        and pair["safety"]["strong"]["max_attempted_kl"]
        <= safety_thresholds["attempted_kl_max"]
        and pair["safety"]["strong"]["max_cvar_residual"]
        <= safety_thresholds["cvar_primal_residual_max"]
        for pair in pairs
    )
    # This is a conservative, explicitly post-hoc promotion rule.  It is a
    # decision aid only; it must not be represented as a preregistered test.
    meaningful_clusters = sum(
        pair["metrics"]["worst"]["delta_mean"] >= 1e-4 for pair in pairs)
    promotion_pass = bool(
        worst["delta_mean"] >= 1e-3
        and worst["cluster_bootstrap_ci95"][0] > 0.0
        and meaningful_clusters >= 4
        and candidate_safety_pass
    )
    return {
        "schema_version": "actor-proximal-ablation/v1",
        "scope": (
            "exploratory K=2/Q=2, three-update training runs with ten fixed "
            "paired evaluation seeds per policy"),
        "independence_note": (
            "Training seeds are independent units; the ten common evaluation "
            "seeds are nested paired diagnostics and are not pooled as n=50."),
        "code_identity": next(iter(code_identities)),
        "baseline_rho": 0.0,
        "candidate_rho": 0.1,
        "training_seeds": list(RUN_PAIRS),
        "evaluation_seeds": list(EVAL_SEEDS),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "method": "percentile bootstrap over training-seed means",
        },
        "pairs": pairs,
        "aggregate": aggregate,
        "decision": {
            "rule_status": "post_hoc_conservative_not_preregistered",
            "minimum_mean_worst_delta": 1e-3,
            "minimum_meaningful_clusters": 4,
            "meaningful_cluster_delta": 1e-4,
            "meaningful_clusters_observed": meaningful_clusters,
            "safety_thresholds": safety_thresholds,
            "candidate_safety_pass": candidate_safety_pass,
            "promote_to_default": promotion_pass,
            "conclusion": (
                "retain_opt_in; improvement is concentrated in one training "
                "seed and does not establish a general algorithmic gain"
                if not promotion_pass else
                "eligible_for_preregistered_confirmation; not yet formal"),
        },
    }


def main() -> None:
    report = build(Path("artifacts/runs"))
    output = Path("artifacts/research/actor_proximal_ablation_v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(json.dumps(report["aggregate"]["worst"], indent=2))
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
