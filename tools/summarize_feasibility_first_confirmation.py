#!/usr/bin/env python
"""Evaluate the frozen non-saturated feasibility-first confirmation."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from tools.summarize_actor_proximal_ablation import bootstrap_mean_ci
except ModuleNotFoundError:  # Direct ``python tools/<script>.py`` execution.
    from summarize_actor_proximal_ablation import bootstrap_mean_ci


RUN_PAIRS = {
    41001: ("train-26142f3be9fde46d7011", "train-31135ef74e4eda4a435f"),
    41002: ("train-06c31dadee7f8020bd94", "train-7460c6ca4fd8449499d5"),
    41003: ("train-1f86c3b0736b81c6b7ae", "train-6576bccb5fdb446aebcd"),
    41004: ("train-e7f67b88355abd1697fe", "train-18d4ae9cefc31823ab41"),
    41005: ("train-74e13f23c758be71aa16", "train-ddc163b2218b05b6816f"),
}
METRICS = ("steady", "weak3", "worst")
SWITCH = "temporal_unrolled_power_feasibility_first_enabled"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _load_run(
    root: Path, run_id: str, training_seed: int, expected_switch: bool,
    evaluation_seeds: tuple[int, ...],
) -> dict[str, Any]:
    run = root / run_id
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (run / "completion.json").read_text(encoding="utf-8"))
    if manifest.get("state") != "created" or manifest.get("seeds") != [training_seed]:
        raise ValueError(f"invalid creation manifest for {run_id}")
    if completion.get("state") != "completed" or completion.get("run_id") != run_id:
        raise ValueError(f"incomplete run: {run_id}")
    required = (
        "raw/output/paired_eval.csv", "raw/output/run_manifest.json",
        "raw/output/train_metrics.csv",
    )
    for relative in required:
        expected = completion.get("artifacts", {}).get(relative)
        if expected != _sha256(run / relative):
            raise ValueError(f"completion hash mismatch: {run_id}/{relative}")

    provenance = json.loads(
        (run / "raw/output/run_manifest.json").read_text(encoding="utf-8"))
    resolved = provenance["provenance"]["resolved_config"]
    actual_switch = bool(resolved["marl"][SWITCH])
    if actual_switch is not expected_switch:
        raise ValueError(f"arm switch mismatch for {run_id}")
    evaluation_rows = _rows(run / "raw/output/paired_eval.csv")
    if len(evaluation_rows) != 1:
        raise ValueError(f"expected one evaluation row for {run_id}")
    evaluation = evaluation_rows[0]
    actual_eval_seeds = tuple(int(value) for value in ast.literal_eval(
        evaluation["eval_episode_seeds"]))
    if actual_eval_seeds != evaluation_seeds:
        raise ValueError(f"evaluation seed mismatch for {run_id}")
    arrays = {}
    for metric in METRICS:
        values = np.asarray(ast.literal_eval(
            evaluation[f"eval_episode_{metric}_values"]), dtype=np.float64)
        if values.shape != (len(evaluation_seeds),) or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid {metric} vector for {run_id}")
        arrays[metric] = values

    training = _rows(run / "raw/output/train_metrics.csv")
    if len(training) != 3:
        raise ValueError(f"expected three training updates for {run_id}")
    numeric = lambda field: np.asarray(
        [float(row[field]) for row in training], dtype=np.float64)
    cvar = numeric("constrained_cvar_residual_max")
    return {
        "run_id": run_id,
        "code_identity": manifest["code_identity"],
        "config_sha256": manifest["config_sha256"],
        "resolved_config": resolved,
        "artifact_sha256": {
            relative: completion["artifacts"][relative] for relative in required
        },
        "arrays": arrays,
        "qos_feasible_rate": float(evaluation["eval_qos_feasible_rate"]),
        "eval_rf_violation_w": float(
            evaluation["eval_isac_max_power_budget_violation_w"]),
        "positive_cvar_residual_auc": float(np.maximum(cvar, 0.0).sum()),
        "max_attempted_kl": float(numeric(
            "post_update_attempted_approx_kl").max()),
        "max_training_rf_violation_w": float(numeric(
            "temporal_unrolled_power_budget_violation_w").max()),
        "actor_update_rejections": int(numeric("actor_update_rejected").sum()),
        "feasibility_first_active_updates": int(numeric(
            "temporal_unrolled_power_feasibility_first_active").sum()),
    }


def _different_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        return [
            path for key in keys
            for path in _different_paths(
                left.get(key), right.get(key),
                f"{prefix}.{key}" if prefix else str(key))
        ]
    return [] if left == right else [prefix]


def build(
    runs_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("schema_version") != "research-preregistration/v1"
            or protocol.get("status") != "frozen_before_execution"):
        raise ValueError("protocol is not a frozen v1 preregistration")
    training_seeds = tuple(int(seed) for seed in protocol["seeds"]["training"])
    evaluation_seeds = tuple(int(seed) for seed in protocol["seeds"]["evaluation"])
    if training_seeds != tuple(RUN_PAIRS):
        raise ValueError("run mapping does not match preregistered training seeds")

    pairs = []
    identities = set()
    for seed, (control_id, candidate_id) in RUN_PAIRS.items():
        control = _load_run(
            runs_root, control_id, seed, False, evaluation_seeds)
        candidate = _load_run(
            runs_root, candidate_id, seed, True, evaluation_seeds)
        identities.update((control["code_identity"], candidate["code_identity"]))
        differences = _different_paths(
            control["resolved_config"], candidate["resolved_config"])
        expected_path = f"marl.{protocol['arms']['allowed_difference']}"
        if differences != [expected_path]:
            raise ValueError(
                f"arms differ outside preregistration for seed {seed}: {differences}")
        metrics = {}
        for name in METRICS:
            delta = candidate["arrays"][name] - control["arrays"][name]
            metrics[name] = {
                "control_mean": float(control["arrays"][name].mean()),
                "candidate_mean": float(candidate["arrays"][name].mean()),
                "delta_mean": float(delta.mean()),
                "delta_median": float(np.median(delta)),
            }
        pairs.append({
            "training_seed": seed,
            "control_run": control_id,
            "candidate_run": candidate_id,
            "metrics": metrics,
            "qos_feasible_rate": {
                "control": control["qos_feasible_rate"],
                "candidate": candidate["qos_feasible_rate"],
            },
            "positive_cvar_residual_auc": {
                "control": control["positive_cvar_residual_auc"],
                "candidate": candidate["positive_cvar_residual_auc"],
                "delta": candidate["positive_cvar_residual_auc"]
                - control["positive_cvar_residual_auc"],
            },
            "candidate_active_updates": candidate[
                "feasibility_first_active_updates"],
            "safety": {
                arm: {
                    key: run[key] for key in (
                        "eval_rf_violation_w", "max_training_rf_violation_w",
                        "max_attempted_kl", "actor_update_rejections",
                    )
                }
                for arm, run in (("control", control), ("candidate", candidate))
            },
            "provenance": {
                arm: {
                    "config_sha256": run["config_sha256"],
                    "artifact_sha256": run["artifact_sha256"],
                }
                for arm, run in (("control", control), ("candidate", candidate))
            },
        })
    if len(identities) != 1:
        raise ValueError(f"confirmation runs span code identities: {identities}")

    samples = int(protocol["statistics"]["bootstrap_samples"])
    bootstrap_seed = int(protocol["statistics"]["bootstrap_seed"])
    tolerance = float(protocol["statistics"]["numerical_equivalence_tolerance"])
    aggregate = {}
    for index, name in enumerate(METRICS):
        deltas = np.asarray(
            [pair["metrics"][name]["delta_mean"] for pair in pairs])
        censored = np.where(np.abs(deltas) <= tolerance, 0.0, deltas)
        aggregate[name] = {
            "n_training_seeds": len(pairs),
            "delta_mean": float(deltas.mean()),
            "delta_median": float(np.median(deltas)),
            "delta_sample_std": float(deltas.std(ddof=1)),
            "cluster_bootstrap_ci95": bootstrap_mean_ci(
                censored, samples=samples, seed=bootstrap_seed + index),
            "clusters_above_0_001": int(np.sum(deltas >= 0.001)),
            "positive_clusters": int(np.sum(deltas > tolerance)),
            "negative_clusters": int(np.sum(deltas < -tolerance)),
            "tied_clusters": int(np.sum(np.abs(deltas) <= tolerance)),
        }

    gate = protocol["promotion_gate"]
    safety_gate = protocol["safety_gate"]
    primary = aggregate["worst"]
    qos_control = np.mean([
        pair["qos_feasible_rate"]["control"] for pair in pairs])
    qos_candidate = np.mean([
        pair["qos_feasible_rate"]["candidate"] for pair in pairs])
    auc_deltas = np.asarray([
        pair["positive_cvar_residual_auc"]["delta"] for pair in pairs])
    conditions = {
        "primary_mean_effect": bool(primary["delta_mean"]
        >= float(gate["minimum_primary_mean_delta"])),
        "primary_lcb_positive": bool(
            primary["cluster_bootstrap_ci95"][0] > 0.0),
        "cluster_replication": bool(primary["clusters_above_0_001"]
        >= int(gate["minimum_clusters_above_0_001"])),
        "qos_not_lower": bool(qos_candidate >= qos_control),
        "cvar_auc_cluster_control": bool(
            int(np.sum(auc_deltas <= tolerance)) >= int(gate[
                "candidate_positive_CVaR_residual_AUC_not_higher_in_minimum_clusters"])),
        "cvar_auc_mean_control": bool(float(auc_deltas.mean())
        <= float(gate["candidate_mean_positive_CVaR_residual_AUC_delta_max"])),
        "safety": all(
            pair["safety"]["candidate"]["eval_rf_violation_w"]
            <= float(safety_gate["eval_RF_budget_violation_w_max"])
            and pair["safety"]["candidate"]["max_training_rf_violation_w"]
            <= float(safety_gate["training_RF_roundoff_w_max"])
            and pair["safety"]["candidate"]["max_attempted_kl"]
            <= float(safety_gate["attempted_KL_max"])
            and pair["safety"]["candidate"]["actor_update_rejections"]
            <= int(safety_gate["actor_update_rejections_max"])
            for pair in pairs),
    }
    promote = all(conditions.values())
    return {
        "schema_version": "feasibility-first-confirmation/v1",
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "code_identity": next(iter(identities)),
        "independent_unit": "training_seed",
        "pairs": pairs,
        "aggregate": aggregate,
        "constraint_control": {
            "mean_qos_control": float(qos_control),
            "mean_qos_candidate": float(qos_candidate),
            "mean_positive_cvar_residual_auc_delta": float(auc_deltas.mean()),
            "candidate_active_updates_total": int(sum(
                pair["candidate_active_updates"] for pair in pairs)),
        },
        "decision": {
            "conditions": conditions,
            "promote_to_default": promote,
            "conclusion": (
                "eligible for independent replication" if promote else
                "falsified under the preregistered confirmation protocol; "
                "do not promote and retire this active research direction"),
        },
    }


def main() -> None:
    protocol = Path(
        "docs/research_protocols/"
        "preregistered_feasibility_first_nonsaturated_v1.yaml")
    report = build(Path("artifacts/runs"), protocol)
    output = Path(
        "artifacts/research/feasibility_first_confirmation_v1.json")
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"wrote {output}")
    print(json.dumps(report["aggregate"]["worst"], indent=2))
    print(json.dumps(report["constraint_control"], indent=2))
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
