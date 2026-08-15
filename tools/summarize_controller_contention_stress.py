"""Verify deadline fail-closed behavior in a controller contention screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_exact_equivalent_retiming import (
    EQUIVALENCE_FIELDS,
    _canonical,
)
from tools.summarize_paired_horizon_confirmation import _sha256


def summarize(
    reference_path: Path,
    stressed_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    stressed = json.loads(stressed_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("audit_return_code") != 0:
        raise ValueError("stressed audit did not complete successfully")
    before_rows = list(reference.get("rows", []))
    after_rows = list(stressed.get("rows", []))
    if len(before_rows) != len(after_rows):
        raise ValueError("stressed audit changed the row count")
    differences: list[dict[str, object]] = []
    for index, (before, after) in enumerate(zip(before_rows, after_rows)):
        key_before = (int(before["seed"]), int(before["frame"]))
        key_after = (int(after["seed"]), int(after["frame"]))
        if key_before != key_after:
            differences.append({"row": index, "field": "seed/frame"})
            continue
        for field in EQUIVALENCE_FIELDS:
            if _canonical(before.get(field)) != _canonical(after.get(field)):
                differences.append({"row": index, "field": field})
    if differences:
        raise ValueError(
            f"contention changed non-timing behavior: {differences[:10]}")

    attempts = [
        row for row in after_rows
        if bool(row.get("horizon_digest_rendezvous_attempted", False))
    ]
    late = [
        row for row in attempts
        if not bool(row["horizon_digest_rendezvous_deadline_pass"])
    ]
    admitted_without_deadline = [
        row for row in attempts
        if bool(row.get("horizon_digest_rendezvous_additional_eligible", False))
        and not bool(row["horizon_digest_rendezvous_deadline_pass"])
    ]
    admitted_without_complete_gate = [
        row for row in attempts
        if bool(row.get("horizon_digest_rendezvous_additional_eligible", False))
        and not all((
            bool(row.get("horizon_digest_rendezvous_exact_accept", False)),
            bool(row.get("horizon_digest_rendezvous_prefix_feasible", False)),
            bool(row.get("horizon_digest_rendezvous_transport_feasible", False)),
            bool(row.get("horizon_digest_rendezvous_suffix_feasible", False)),
            bool(row.get("horizon_digest_rendezvous_comm_bound_valid", False)),
            bool(row.get("horizon_digest_rendezvous_deadline_pass", False)),
        ))
    ]
    stressed_summary = dict(stressed["summary"])
    reference_summary = dict(reference["summary"])
    deadline_fail_closed = not (
        admitted_without_deadline or admitted_without_complete_gate)
    return {
        "schema_version": 1,
        "status": (
            "workstation_contention_fail_closed_pass_not_deployment_validation"
            if deadline_fail_closed else
            "workstation_contention_fail_closed_violation"
        ),
        "stress_design": {
            "logical_cpu_count_reported": metadata.get(
                "logical_cpu_count_reported"),
            "affinity_logical_cpus": metadata.get("affinity_logical_cpus"),
            "contention_worker_count": metadata.get(
                "contention_worker_count"),
            "same_affinity_for_audit_and_workers": True,
        },
        "behavior_equivalence": {
            "row_count": len(after_rows),
            "field_count_per_row": len(EQUIVALENCE_FIELDS),
            "comparison_count": len(after_rows) * len(EQUIVALENCE_FIELDS),
            "difference_count": 0,
            "fields": list(EQUIVALENCE_FIELDS),
        },
        "timing": {
            "deadline_s": 0.1,
            "attempt_count": len(attempts),
            "reference_deadline_pass_count": int(reference_summary[
                "horizon_digest_rendezvous_deadline_pass_count"]),
            "stressed_deadline_pass_count": int(stressed_summary[
                "horizon_digest_rendezvous_deadline_pass_count"]),
            "stressed_deadline_failure_count": len(late),
            "reference_latency_s_mean": float(reference_summary[
                "horizon_digest_rendezvous_mean_total_latency_s"]),
            "stressed_latency_s_mean": float(stressed_summary[
                "horizon_digest_rendezvous_mean_total_latency_s"]),
            "reference_latency_s_max": float(reference_summary[
                "horizon_digest_rendezvous_max_total_latency_s"]),
            "stressed_latency_s_max": float(stressed_summary[
                "horizon_digest_rendezvous_max_total_latency_s"]),
            "late_events": [{
                "seed": int(row["seed"]),
                "frame": int(row["frame"]),
                "latency_s": float(
                    row["horizon_digest_rendezvous_total_latency_s"]),
                "physical_accept": bool(
                    row["horizon_digest_rendezvous_exact_accept"]),
                "admitted": bool(
                    row["horizon_digest_rendezvous_additional_eligible"]),
            } for row in late],
        },
        "hard_gate": {
            "rule": (
                "admit = physical_accept AND prefix_feasible AND "
                "transport_feasible AND suffix_feasible AND "
                "comm_bound_valid AND latency<=deadline"
            ),
            "admitted_without_deadline_count": len(
                admitted_without_deadline),
            "admitted_without_complete_gate_count": len(
                admitted_without_complete_gate),
            "physical_accept_count": int(stressed_summary[
                "horizon_digest_rendezvous_exact_accept_count"]),
            "timely_admitted_count": int(stressed_summary[
                "horizon_digest_rendezvous_additional_eligible_count"]),
            "future_failure_count": int(stressed_summary[
                "horizon_digest_rendezvous_future_failure_count"]),
            "max_isac_power_balance_error_w": float(stressed_summary[
                "horizon_digest_rendezvous_max_power_balance_error_w"]),
            "fail_closed_pass": deadline_fail_closed,
        },
        "scope": {
            "workstation_contention_only": True,
            "deployment_scheduler_jitter_measured": False,
            "network_contention_measured": False,
            "energy_measured": False,
            "commit_authority": False,
        },
        "provenance": {
            "reference": {
                "path": str(reference_path),
                "sha256": _sha256(reference_path),
            },
            "stressed": {
                "path": str(stressed_path),
                "sha256": _sha256(stressed_path),
            },
            "metadata": {
                "path": str(metadata_path),
                "sha256": _sha256(metadata_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--stressed", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.reference, args.stressed, args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not bool(result["hard_gate"]["fail_closed_pass"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
