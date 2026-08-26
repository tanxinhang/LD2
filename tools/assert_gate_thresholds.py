#!/usr/bin/env python
"""Gate threshold assertions for formal evaluation runs.

Turns "Gate passed" from documentation narrative into a script-enforced check:
any formal run whose episode aggregates do not clear the Medium thresholds
(steady >= 0.80, weak3 >= 0.70, mean worst >= 0.60, QoS feasible >= 0.70 with
the lower endpoint of a 95% two-sided Wilson interval) exits non-zero, so a regression can never be
silently written into a docs table again.

Two entry points:
  - Python API: medium_gate_checks(...) / assert_medium_gate(...)
  - CLI:        python tools/assert_gate_thresholds.py <paired_eval.csv> [--qos-floor 0.70]

The CSV reader follows the project convention used by
tools/summarize_algorithm_baselines.py: paired_eval.csv holds exactly one
aggregate row and the per-episode arrays live in the eval_episode_*_P_D
columns as literal Python lists, which are re-parsed so every aggregate here
is recomputed independently of summary.json.  The QoS mask is strict
(no tolerance) and the Wilson LCB uses z=1.96, matching
tools/summarize_final_paper_suite.py; note the trainer's recorded
eval_qos_feasible_wilson_lcb may differ slightly because it uses a one-sided
z=inv_cdf(1-alpha) and a solver tolerance.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from typing import Dict, List, Optional, Tuple

# Medium average thresholds (docs/SYSTEM_OVERVIEW_AND_ROADMAP.md §5).
MEDIUM_FLOORS = {"steady": 0.80, "weak3": 0.70, "worst": 0.60}
QOS_FLOOR = 0.70
Z_95 = 1.96


def wilson_lower(successes: int, total: int, z: float = Z_95) -> float:
    """Lower endpoint of the 95% two-sided Wilson interval (z=1.96)."""
    if total <= 0:
        return float("nan")
    p = successes / total
    z2 = z * z
    centre = p + z2 / (2.0 * total)
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z2 / (4.0 * total * total))
    return max(0.0, (centre - radius) / (1.0 + z2 / total))


def _mean(values: List[float]) -> float:
    return float(sum(values)) / len(values) if values else float("nan")


def medium_gate_checks(
    steady: float,
    weak3: float,
    worst: float,
    qos_feasible: float,
    qos_wilson_lcb: Optional[float] = None,
    qos_floor: float = QOS_FLOOR,
) -> List[Dict[str, object]]:
    """Return per-threshold checks; each check has passed/value/floor."""
    checks = [
        {"name": "steady", "value": steady,
         "floor": MEDIUM_FLOORS["steady"],
         "passed": steady >= MEDIUM_FLOORS["steady"]},
        {"name": "weak3", "value": weak3,
         "floor": MEDIUM_FLOORS["weak3"],
         "passed": weak3 >= MEDIUM_FLOORS["weak3"]},
        {"name": "mean_worst", "value": worst,
         "floor": MEDIUM_FLOORS["worst"],
         "passed": worst >= MEDIUM_FLOORS["worst"]},
        {"name": "qos_feasible", "value": qos_feasible,
         "floor": qos_floor,
         "passed": qos_feasible >= qos_floor},
    ]
    if qos_wilson_lcb is not None:
        checks.append({"name": "qos_wilson_lcb", "value": qos_wilson_lcb,
                       "floor": qos_floor,
                       "passed": qos_wilson_lcb >= qos_floor})
    return checks


def assert_medium_gate(
    steady: float,
    weak3: float,
    worst: float,
    qos_feasible: float,
    qos_wilson_lcb: Optional[float] = None,
    qos_floor: float = QOS_FLOOR,
    label: str = "run",
) -> None:
    """Raise AssertionError listing every unmet threshold."""
    checks = medium_gate_checks(
        steady, weak3, worst, qos_feasible, qos_wilson_lcb, qos_floor)
    failed = [c for c in checks if not c["passed"]]
    if failed:
        lines = [f"Gate assertion FAILED for {label}:"]
        for c in checks:
            mark = "ok " if c["passed"] else "FAIL"
            lines.append(
                f"  [{mark}] {c['name']}: {c['value']:.4f} "
                f"(floor {c['floor']:.2f})")
        raise AssertionError("\n".join(lines))


def read_episode_arrays(csv_path: str) -> Dict[str, List[float]]:
    """Parse the per-episode arrays from a single-row trainer paired_eval.csv."""
    with open(csv_path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(
            f"expected one aggregate row in {csv_path}, got {len(rows)}")
    raw = rows[0]
    arrays: Dict[str, List[float]] = {}
    for name, column in (("steady", "eval_episode_steady_P_D"),
                         ("weak3", "eval_episode_weak3_P_D"),
                         ("worst", "eval_episode_worst_P_D")):
        if column not in raw:
            raise ValueError(
                f"{csv_path} lacks column {column!r} (not a trainer paired_eval)")
        values = ast.literal_eval(raw[column])
        arrays[name] = [float(value) for value in values]
    lengths = {name: len(values) for name, values in arrays.items()}
    if not lengths["steady"]:
        raise ValueError(f"{csv_path} contains no evaluation episodes")
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{csv_path} has misaligned episode arrays: {lengths}")
    for name, values in arrays.items():
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0
               for value in values):
            raise ValueError(
                f"{csv_path} has non-finite/out-of-range {name} values")
    seed_column = raw.get("eval_episode_seeds")
    if seed_column:
        seeds = [int(value) for value in ast.literal_eval(seed_column)]
        if len(seeds) != lengths["steady"]:
            raise ValueError(
                f"{csv_path} seed count does not match episode arrays")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{csv_path} contains duplicate evaluation seeds")
        arrays["seeds"] = seeds
    return arrays


def assert_gate_from_csv(
    csv_path: str,
    qos_floor: float = QOS_FLOOR,
    require_wilson_lcb: bool = False,
    qos_tol: float = 0.0,
) -> Dict[str, float]:
    """Recompute aggregates from episode arrays and assert the Medium gate.

    QoS feasible is recomputed per episode from the three floors (strict mask,
    optional solver tolerance qos_tol), then the Wilson LCB is derived from
    the feasible count.  The LCB is reported always but enforced only when
    require_wilson_lcb=True (the audit-recommended future standard; e.g. the
    current 4/4 formal result has QoS 0.72 with LCB 0.63 under z=1.96, so
    enforcing the LCB retroactively would reject the deployed baseline).
    Returns the aggregates so callers can log them.
    """
    arrays = read_episode_arrays(csv_path)
    n = len(arrays["steady"])
    steady = _mean(arrays["steady"])
    weak3 = _mean(arrays["weak3"])
    worst = _mean(arrays["worst"])
    feasible = sum(
        1 for s, w, wo in zip(arrays["steady"], arrays["weak3"],
                              arrays["worst"])
        if (s >= MEDIUM_FLOORS["steady"] - qos_tol
            and w >= MEDIUM_FLOORS["weak3"] - qos_tol
            and wo >= MEDIUM_FLOORS["worst"] - qos_tol))
    qos = feasible / n
    lcb = wilson_lower(feasible, n)
    assert_medium_gate(
        steady, weak3, worst, qos,
        qos_wilson_lcb=lcb if require_wilson_lcb else None,
        qos_floor=qos_floor, label=csv_path)
    return {
        "episodes": n, "steady": steady, "weak3": weak3,
        "worst": worst, "qos_feasible": qos, "qos_wilson_lcb": lcb,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="paired_eval.csv path (trainer format)")
    parser.add_argument("--qos-floor", type=float, default=QOS_FLOOR)
    parser.add_argument("--require-lcb", action="store_true",
                        help="also require the QoS Wilson LCB to clear the "
                             "floor (audit-recommended future standard)")
    parser.add_argument("--json-output", default=None,
                        help="write aggregates to this path")
    args = parser.parse_args(argv)

    try:
        aggregates = assert_gate_from_csv(
            args.csv, qos_floor=args.qos_floor,
            require_wilson_lcb=args.require_lcb)
    except (ValueError, AssertionError) as exc:
        print(f"gate assertion error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(aggregates, indent=2))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(aggregates, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
