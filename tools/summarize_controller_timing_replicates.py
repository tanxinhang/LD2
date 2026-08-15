"""Summarize repeated workstation timing without claiming deployment jitter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_exact_equivalent_retiming import (
    EQUIVALENCE_FIELDS,
    _canonical,
)
from tools.summarize_paired_horizon_confirmation import _sha256


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) < 2:
        raise ValueError("at least two complete timing audits are required")
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference_rows = list(audits[0].get("rows", []))
    differences: list[dict[str, object]] = []
    for run_index, audit in enumerate(audits[1:], start=1):
        rows = list(audit.get("rows", []))
        if len(rows) != len(reference_rows):
            raise ValueError("timing replicate changed row count")
        for row_index, (before, after) in enumerate(zip(reference_rows, rows)):
            if (
                int(before.get("seed", -1)) != int(after.get("seed", -1))
                or int(before.get("frame", -1)) != int(after.get("frame", -1))
            ):
                differences.append({
                    "run": run_index,
                    "row": row_index,
                    "field": "seed/frame",
                })
                continue
            for field in EQUIVALENCE_FIELDS:
                if _canonical(before.get(field)) != _canonical(after.get(field)):
                    differences.append({
                        "run": run_index,
                        "row": row_index,
                        "field": field,
                    })
    if differences:
        raise ValueError(
            f"timing replicate changed certified behavior: {differences[:10]}")

    attempt_keys = [
        (int(row["seed"]), int(row["frame"]))
        for row in reference_rows
        if bool(row.get("horizon_digest_rendezvous_attempted", False))
    ]
    latency_runs: list[np.ndarray] = []
    for audit in audits:
        by_key = {
            (int(row["seed"]), int(row["frame"])): float(
                row["horizon_digest_rendezvous_total_latency_s"])
            for row in audit["rows"]
            if bool(row.get("horizon_digest_rendezvous_attempted", False))
        }
        if set(by_key) != set(attempt_keys):
            raise ValueError("timing replicate changed the attempt set")
        latency_runs.append(np.asarray([
            by_key[key] for key in attempt_keys], dtype=np.float64))
    latency = np.stack(latency_runs)
    event_range = np.ptp(latency, axis=0)
    event_worst = np.max(latency, axis=0)
    run_means = np.mean(latency, axis=1)
    run_maxima = np.max(latency, axis=1)
    deadline = 0.1
    return {
        "schema_version": 1,
        "status": "workstation_repeat_timing_pass_not_deployment_jitter",
        "run_count": len(paths),
        "attempt_count_per_run": len(attempt_keys),
        "behavior_equivalence": {
            "comparison_count": (
                (len(paths) - 1) * len(reference_rows)
                * len(EQUIVALENCE_FIELDS)),
            "difference_count": 0,
        },
        "timing": {
            "deadline_s": deadline,
            "deadline_pass_count_across_all_runs": int(np.sum(
                latency <= deadline)),
            "deadline_check_count_across_all_runs": int(latency.size),
            "deadline_failure_count_across_all_runs": int(np.sum(
                latency > deadline)),
            "run_mean_latency_s": run_means.tolist(),
            "run_max_latency_s": run_maxima.tolist(),
            "all_run_latency_s_mean": float(np.mean(latency)),
            "all_run_latency_s_max": float(np.max(latency)),
            "per_event_worst_latency_s_quantiles": {
                str(q): float(np.quantile(event_worst, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "per_event_worst_latency_s_max": float(np.max(event_worst)),
            "per_event_run_range_s_mean": float(np.mean(event_range)),
            "per_event_run_range_s_p95": float(np.quantile(
                event_range, 0.95)),
            "per_event_run_range_s_max": float(np.max(event_range)),
        },
        "scope": {
            "host": "current Windows development workstation",
            "deployment_scheduler_jitter_measured": False,
            "network_contention_retransmission_measured": False,
            "commit_authority": False,
            "interpretation": (
                "Repeated local process timing measures software/host "
                "reproducibility only and cannot replace distributed UAV "
                "scheduler, RF-network or energy measurements."
            ),
        },
        "provenance": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(list(args.audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
