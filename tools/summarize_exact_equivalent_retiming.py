"""Certify action equivalence and summarize post-validation retiming.

This tool does not create a new independent validation claim.  It compares a
frozen validation audit with a later implementation-equivalent replay and
fails unless every listed non-timing decision/certificate field is identical
row by row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_paired_horizon_confirmation import _sha256


EQUIVALENCE_FIELDS = (
    "causal_route",
    "structure_accepted",
    "structure_candidate_count",
    "structure_candidate_discrete_digest",
    "structure_candidate_action_digest",
    "horizon_ranked_candidate_count",
    "horizon_digest_rendezvous_attempted",
    "horizon_digest_rendezvous_full_structure_verified",
    "horizon_digest_rendezvous_exact_accept",
    "horizon_digest_rendezvous_exact_target_safe",
    "horizon_digest_rendezvous_exact_net_gain_lower",
    "horizon_digest_rendezvous_future_target_no_harm",
)


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def summarize(reference_path: Path, retimed_path: Path) -> dict[str, object]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    retimed = json.loads(retimed_path.read_text(encoding="utf-8"))
    if reference.get("seed_order") != retimed.get("seed_order"):
        raise ValueError("retimed audit changed the episode order")
    reference_rows = list(reference.get("rows", []))
    retimed_rows = list(retimed.get("rows", []))
    if len(reference_rows) != len(retimed_rows):
        raise ValueError("retimed audit changed the row count")
    differences: list[dict[str, object]] = []
    for index, (before, after) in enumerate(zip(
        reference_rows, retimed_rows
    )):
        if (
            int(before.get("seed", -1)) != int(after.get("seed", -1))
            or int(before.get("frame", -1)) != int(after.get("frame", -1))
        ):
            differences.append({"row": index, "field": "seed/frame"})
            continue
        for field in EQUIVALENCE_FIELDS:
            if _canonical(before.get(field)) != _canonical(after.get(field)):
                differences.append({"row": index, "field": field})
    if differences:
        raise ValueError(
            f"retimed implementation changed certified behavior: "
            f"{differences[:10]}")

    old = dict(reference["summary"])
    new = dict(retimed["summary"])
    return {
        "schema_version": 1,
        "status": "retrospective_exact_equivalent_timing_diagnostic",
        "independent_validation_reused": False,
        "equivalence": {
            "row_count": len(reference_rows),
            "field_count_per_row": len(EQUIVALENCE_FIELDS),
            "comparison_count": len(reference_rows) * len(EQUIVALENCE_FIELDS),
            "difference_count": 0,
            "fields": list(EQUIVALENCE_FIELDS),
            "golden_candidate_order_test_seed_count": 10,
            "interpretation": (
                "The optimization changed neither route/action digests nor "
                "the exact H-step and future no-harm certificate outputs."
            ),
        },
        "timing": {
            "reference_common_instrumented_mean_s": float(
                old["controller_common_preprocessing_mean_s"]),
            "retimed_common_causal_net_mean_s": float(
                new["controller_common_preprocessing_mean_s"]),
            "retimed_common_instrumented_mean_s": float(
                new["controller_common_instrumented_wall_mean_s"]),
            "retimed_privileged_audit_mean_s": float(
                new["controller_privileged_audit_mean_s"]),
            "reference_horizon_branch_mean_s": float(
                old["horizon_complete_branch_mean_compute_s"]),
            "retimed_horizon_branch_mean_s": float(
                new["horizon_complete_branch_mean_compute_s"]),
            "retimed_horizon_stage_mean_s": {
                "propagation": float(
                    new["horizon_propagation_mean_compute_s"]),
                "physics_envelope": float(
                    new["horizon_envelope_mean_compute_s"]),
                "exact_candidate_set": float(
                    new["horizon_candidate_set_mean_compute_s"]),
            },
            "reference_complete_path_mean_s": float(
                old["horizon_digest_rendezvous_mean_total_latency_s"]),
            "retimed_complete_path_mean_s": float(
                new["horizon_digest_rendezvous_mean_total_latency_s"]),
            "reference_complete_path_max_s": float(
                old["horizon_digest_rendezvous_max_total_latency_s"]),
            "retimed_complete_path_max_s": float(
                new["horizon_digest_rendezvous_max_total_latency_s"]),
            "reference_deadline_pass_count": int(
                old["horizon_digest_rendezvous_deadline_pass_count"]),
            "retimed_deadline_pass_count": int(
                new["horizon_digest_rendezvous_deadline_pass_count"]),
            "attempt_count": int(
                new["horizon_digest_rendezvous_attempt_event_count"]),
        },
        "safety_unchanged": {
            "physical_accept_count": int(
                new["horizon_digest_rendezvous_exact_accept_count"]),
            "eligible_episode_count": int(
                new[
                    "horizon_digest_rendezvous_additional_eligible_episode_count"
                ]),
            "future_failure_count": int(
                new["horizon_digest_rendezvous_future_failure_count"]),
            "max_power_balance_error_w": float(
                new["horizon_digest_rendezvous_max_power_balance_error_w"]),
        },
        "authority": {
            "resource_accounting_complete": False,
            "commit_authority": False,
        },
        "provenance": {
            "reference_audit": {
                "path": str(reference_path),
                "sha256": _sha256(reference_path),
            },
            "retimed_audit": {
                "path": str(retimed_path),
                "sha256": _sha256(retimed_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--retimed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.reference, args.retimed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
