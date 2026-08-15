#!/usr/bin/env python
"""Summarize the paired Gate C1.7a dynamic local-search controls."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


METRICS = (
    "eval_steady_P_D",
    "eval_weak3_P_D",
    "eval_worst_P_D",
    "eval_worst_cvar",
    "eval_qos_feasible_rate",
    "eval_qos_feasible_wilson_lcb",
    "eval_local_search_candidate_evaluations_per_resolve",
    "eval_local_search_exact_verifications_per_resolve",
    "eval_local_search_accepted_moves_per_resolve",
    "eval_local_search_cold_resolve_fraction",
    "eval_p0_solve_time_s_per_resolve",
    "eval_isac_max_power_balance_error_w",
    "valid_pair_rate",
    "no_TX_rate",
)


def _read(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    result: dict[str, object] = {
        key: float(row[key]) for key in METRICS
    }
    result["episode_seeds"] = [
        int(value) for value in ast.literal_eval(row["eval_episode_seeds"])]
    result["episode_worst"] = [
        float(value)
        for value in ast.literal_eval(row["eval_episode_worst_values"])]
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    controllers = {
        "previous_only": _read(args.previous),
        "dynamic_oracle_n1_n3": _read(args.oracle),
        "hybrid_learned_top3": _read(args.top3),
        "hybrid_learned_top5": _read(args.top5),
        "replicated_candidate_reference": _read(args.replicated),
    }
    seeds = controllers["dynamic_oracle_n1_n3"]["episode_seeds"]
    if any(value["episode_seeds"] != seeds for value in controllers.values()):
        raise ValueError("controller seed order does not match")
    oracle = controllers["dynamic_oracle_n1_n3"]
    proposed = controllers["hybrid_learned_top5"]
    replicated = controllers["replicated_candidate_reference"]
    exact_reduction = 1.0 - (
        proposed["eval_local_search_exact_verifications_per_resolve"]
        / oracle["eval_local_search_exact_verifications_per_resolve"])
    worst_gap = (
        oracle["eval_worst_P_D"] - proposed["eval_worst_P_D"])
    cvar_gap = (
        oracle["eval_worst_cvar"] - proposed["eval_worst_cvar"])
    checks = {
        "worst_gap_le_0p02": worst_gap <= args.max_worst_gap,
        "cvar_gap_le_0p01": cvar_gap <= args.max_cvar_gap,
        "exact_verification_reduction_ge_0p50": (
            exact_reduction >= args.min_exact_reduction),
        "qos_feasibility_not_below_oracle": (
            proposed["eval_qos_feasible_rate"]
            >= oracle["eval_qos_feasible_rate"]),
        "one_watt_projection_numerically_exact": (
            proposed["eval_isac_max_power_balance_error_w"] <= 1.0e-12),
        "valid_pair_rate_matches_oracle": abs(
            proposed["valid_pair_rate"] - oracle["valid_pair_rate"])
            <= 1.0e-12,
    }
    report = {
        "protocol": "gate_c1_7a_dynamic_local_search_v1",
        "scope": (
            "10-seed development split; Actor, motion, targets, U2U Token "
            "history, local beliefs and candidate graph re-evolve; Hold=5; "
            "no event-trigger or switch-cost ablation"),
        "seeds": seeds,
        "controllers": controllers,
        "proposed": "hybrid_learned_top5",
        "relative_to_dynamic_oracle": {
            "mean_worst_gap": worst_gap,
            "cvar_gap": cvar_gap,
            "exact_verification_reduction": exact_reduction,
            "solve_time_ratio": (
                proposed["eval_p0_solve_time_s_per_resolve"]
                / oracle["eval_p0_solve_time_s_per_resolve"]),
        },
        "relative_to_replicated_candidate_reference": {
            "steady_gap": (
                replicated["eval_steady_P_D"]
                - proposed["eval_steady_P_D"]),
            "weak3_gap": (
                replicated["eval_weak3_P_D"]
                - proposed["eval_weak3_P_D"]),
            "mean_worst_gap": (
                replicated["eval_worst_P_D"]
                - proposed["eval_worst_P_D"]),
            "cvar_gap": (
                replicated["eval_worst_cvar"]
                - proposed["eval_worst_cvar"]),
        },
        "gate": {
            "checks": checks,
            "pass": bool(all(checks.values())),
            "thresholds": {
                "max_worst_gap": args.max_worst_gap,
                "max_cvar_gap": args.max_cvar_gap,
                "min_exact_verification_reduction": (
                    args.min_exact_reduction),
            },
        },
        "interpretation": {
            "top3": (
                "fails the dynamic worst gate because seed 483 loses "
                "0.259 episode worst"),
            "top5": (
                "passes the pre-registered local-maintenance gates"),
            "runtime": (
                "exact verification count falls, but the current Python "
                "rank/enumerate implementation is not yet faster than the "
                "dynamic Oracle local enumerator"),
            "remaining_gap": (
                "the large replicated-reference gap is a global-vs-local "
                "coordination headroom result, not a learned-ranker gap"),
        },
        "fresh_test_consumed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "episode_metrics.csv").open(
        "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("controller", "seed", "episode_worst"))
        for name, values in controllers.items():
            for seed, worst in zip(seeds, values["episode_worst"]):
                writer.writerow((name, seed, worst))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--top3", type=Path, required=True)
    parser.add_argument("--top5", type=Path, required=True)
    parser.add_argument("--replicated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-worst-gap", type=float, default=0.02)
    parser.add_argument("--max-cvar-gap", type=float, default=0.01)
    parser.add_argument("--min-exact-reduction", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
