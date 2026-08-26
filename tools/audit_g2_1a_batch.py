#!/usr/bin/env python
"""Audit an in-progress or formal G2-1A bridge without tuning on it."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT_HINT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_HINT))

from tools.crash_isolated_seed_eval import _wilson_lower
from tools.run_g2_1a_bridge import ROOT, SYSTEMS


UAV_COUNT = {"4x4": 4, "8x8_v3c0": 8, "6x6_v3c0": 6, "6x6_ce": 6}
SENSING_PA_CAP_W = 0.0251
QOS_TARGETS = (0.80, 0.70, 0.60)


def _list(row: dict[str, str], key: str, cast=float) -> list:
    values = ast.literal_eval(row[key])
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a list")
    return [cast(value) for value in values]


def _finite_probability(values: list[float], key: str) -> None:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{key} contains a non-probability")


def audit_row(name: str, row: dict[str, str]) -> dict[str, object]:
    seeds = _list(row, "eval_episode_seeds", int)
    steady = _list(row, "eval_episode_steady_P_D")
    weak3 = _list(row, "eval_episode_weak3_P_D")
    worst = _list(row, "eval_episode_worst_P_D")
    n = len(seeds)
    if n == 0 or len(set(seeds)) != n:
        raise ValueError(f"{name}: episode seeds must be nonempty and unique")
    if not (len(steady) == len(weak3) == len(worst) == n):
        raise ValueError(f"{name}: episode arrays are misaligned")
    for key, values in (("steady", steady), ("weak3", weak3), ("worst", worst)):
        _finite_probability(values, key)
    tol = float(row["eval_qos_tol"])
    feasible = (
        (np.asarray(steady) >= QOS_TARGETS[0] - tol)
        & (np.asarray(weak3) >= QOS_TARGETS[1] - tol)
        & (np.asarray(worst) >= QOS_TARGETS[2] - tol))
    successes = int(np.count_nonzero(feasible))
    expected_sensing = UAV_COUNT[name] * SENSING_PA_CAP_W
    observed_sensing = float(row["eval_isac_sensing_power_w_per_frame"])
    sensing_error = abs(observed_sensing - expected_sensing)
    checks = {
        "episode_arrays_valid": True,
        "reported_steady_recomputed": math.isclose(float(row["eval_steady_P_D"]), float(np.mean(steady)), abs_tol=1e-12),
        "reported_weak3_recomputed": math.isclose(float(row["eval_weak3_P_D"]), float(np.mean(weak3)), abs_tol=1e-12),
        "reported_worst_recomputed": math.isclose(float(row["eval_worst_P_D"]), float(np.mean(worst)), abs_tol=1e-12),
        "reported_qos_recomputed": math.isclose(float(row["eval_qos_feasible_rate"]), successes / n, abs_tol=1e-12),
        "reported_wilson_recomputed": math.isclose(float(row["eval_qos_feasible_wilson_lcb"]), _wilson_lower(successes, n), abs_tol=1e-12),
        "sensing_total_matches_per_uav_cap": sensing_error <= 1e-12,
        "comm_delivery_probability_valid": 0.0 <= float(row["eval_comm_delivery_rate"]) <= 1.0,
        "comm_deadline_probability_valid": 0.0 <= float(row["eval_comm_deadline_violation_rate"]) <= 1.0,
        "comm_rate_conservation": math.isclose(float(row["eval_comm_delivery_rate"]) + float(row["eval_comm_deadline_violation_rate"]), 1.0, abs_tol=1e-9),
    }
    return {
        "name": name,
        "status": "FORMAL" if n == 100 else "IN_PROGRESS",
        "episodes": n,
        "seeds": seeds,
        "metrics": {
            "steady_mean": float(np.mean(steady)),
            "weak3_mean": float(np.mean(weak3)),
            "worst_mean": float(np.mean(worst)),
            "qos_successes": successes,
            "qos_rate": successes / n,
            "qos_wilson_lcb": _wilson_lower(successes, n),
            "steady_saturation_fraction_ge_0_999": float(np.mean(np.asarray(steady) >= 0.999)),
            "comm_bits_per_frame": float(row["eval_comm_bits_per_frame"]),
            "comm_mean_latency_s": float(row["eval_comm_mean_latency_s"]),
            "comm_delivery_rate": float(row["eval_comm_delivery_rate"]),
            "comm_deadline_violation_rate": float(row["eval_comm_deadline_violation_rate"]),
            "sensing_power_w_per_frame": observed_sensing,
            "expected_sensing_power_w_per_frame": expected_sensing,
            "comm_power_w_per_frame": float(row["eval_isac_comm_power_w_per_frame"]),
            "max_power_balance_residual_w": float(row["eval_isac_max_power_balance_error_w"]),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def paired_delta(left_row: dict[str, str], right_row: dict[str, str]) -> dict[str, object]:
    left_seeds = _list(left_row, "eval_episode_seeds", int)
    right_seeds = _list(right_row, "eval_episode_seeds", int)
    if left_seeds != right_seeds:
        raise ValueError("6x6 paired systems do not have identical ordered seeds")
    result: dict[str, object] = {"seeds": left_seeds, "direction": "6x6_ce_minus_6x6_v3c0"}
    for label, column in (("steady", "eval_episode_steady_P_D"), ("weak3", "eval_episode_weak3_P_D"), ("worst", "eval_episode_worst_P_D")):
        delta = np.asarray(_list(right_row, column)) - np.asarray(_list(left_row, column))
        result[label] = {
            "mean_delta": float(np.mean(delta)),
            "min_delta": float(np.min(delta)),
            "max_delta": float(np.max(delta)),
            "ce_wins": int(np.count_nonzero(delta > 0.0)),
            "ties": int(np.count_nonzero(delta == 0.0)),
            "episodes": int(delta.size),
        }
    return result


def run_audit() -> dict[str, object]:
    rows: dict[str, dict[str, str]] = {}
    systems = []
    for name, system in SYSTEMS.items():
        path = ROOT / system.output / "paired_eval.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle), None)
        if row is None:
            raise ValueError(f"empty result: {path}")
        rows[name] = row
        systems.append(audit_row(name, row))
    return {
        "gate": "G2-1A",
        "interpretation": "diagnostic_only_until_all_systems_have_100_episodes",
        "formal_complete": all(item["status"] == "FORMAL" for item in systems),
        "all_checks_pass": all(item["all_checks_pass"] for item in systems),
        "systems": systems,
        "paired_6x6": paired_delta(rows["6x6_v3c0"], rows["6x6_ce"]),
        "power_balance_note": "max_power_balance_residual_w is an allocation identity residual/unused headroom diagnostic, not by itself a cap violation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-checks", action="store_true")
    args = parser.parse_args()
    result = run_audit()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.assert_checks and not result["all_checks_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
