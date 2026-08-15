#!/usr/bin/env python
"""Create a machine-readable strict repair/baseline execution certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rows_by_key(summary: dict[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    rows = summary["rows"]
    return {
        (int(row["seed"]), int(row["frame"])): row
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zoh-reference", type=Path, default=None)
    parser.add_argument("--diagnostic-upper-reference", type=Path, default=None)
    args = parser.parse_args()
    repair = json.loads(args.repair.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    repair_rows = _rows_by_key(repair)
    baseline_rows = _rows_by_key(baseline)
    if repair_rows.keys() != baseline_rows.keys():
        raise ValueError("repair and baseline event keys differ")
    accepted_repair = {
        (int(row["seed"]), int(row["frame"]), str(
            row["geometry_joint_plan_digest"]))
        for row in repair["rows"] if bool(row["geometry_shadow_accepted"])
    }
    accepted_baseline = {
        (int(row["seed"]), int(row["frame"]), str(
            row["geometry_joint_plan_digest"]))
        for row in baseline["rows"] if bool(row["geometry_shadow_accepted"])
    }
    if accepted_repair != accepted_baseline:
        raise ValueError("accepted commitment epochs or digests differ")
    seeds = list(dict.fromkeys(
        int(row["seed"]) for row in repair["rows"]))
    per_seed_delta: dict[str, float] = {}
    for seed in seeds:
        keys = [key for key in repair_rows if key[0] == seed]
        per_seed_delta[str(seed)] = float(np.mean([
            float(repair_rows[key]["candidate_worst_pd"])
            - float(baseline_rows[key]["candidate_worst_pd"])
            for key in keys
        ]))
    tube_window_delta: dict[str, float] = {}
    accepted_window_keys: set[tuple[int, int]] = set()
    for seed, decision_frame, _ in sorted(accepted_repair):
        accepted_window_keys.add((seed, decision_frame))
        keys = [
            key for key, row in repair_rows.items()
            if key[0] == seed
            and bool(row["geometry_tube_executed"])
            and int(row["geometry_tube_decision_frame"]) == decision_frame
        ]
        accepted_window_keys.update(keys)
        tube_window_delta[f"{seed}:{decision_frame}"] = float(np.sum([
            float(repair_rows[key]["candidate_worst_pd"])
            - float(baseline_rows[key]["candidate_worst_pd"])
            for key in keys
        ]))
    seed_values = np.asarray(list(per_seed_delta.values()), dtype=np.float64)
    window_values = np.asarray(
        list(tube_window_delta.values()), dtype=np.float64)
    outside_runtime_digest_mismatch = [
        key for key in repair_rows
        if key not in accepted_window_keys
        and repair_rows[key].get("post_event_runtime_state_digest")
        != baseline_rows[key].get("post_event_runtime_state_digest")
    ]
    outside_policy_delta = np.asarray([
        float(repair_rows[key]["candidate_worst_pd"])
        - float(baseline_rows[key]["candidate_worst_pd"])
        for key in repair_rows if key not in accepted_window_keys
    ], dtype=np.float64)
    paired_tube_keys = [
        key for key, row in repair_rows.items()
        if bool(row["geometry_tube_proof_available"])
    ]
    paired_label_errors = np.asarray([
        max(
            abs(
                float(repair_rows[key]["geometry_tube_repair_worst_pd"])
                - float(baseline_rows[key][
                    "geometry_tube_repair_worst_pd"])),
            abs(
                float(repair_rows[key][
                    "geometry_tube_paired_baseline_worst_pd"])
                - float(baseline_rows[key][
                    "geometry_tube_paired_baseline_worst_pd"])),
            abs(
                float(repair_rows[key][
                    "geometry_tube_global_worst_pd_gain"])
                - float(baseline_rows[key][
                    "geometry_tube_global_worst_pd_gain"])),
            float(np.max(np.abs(
                np.asarray(repair_rows[key][
                    "geometry_tube_clipped_pd_gain_by_target"],
                    dtype=np.float64)
                - np.asarray(baseline_rows[key][
                    "geometry_tube_clipped_pd_gain_by_target"],
                    dtype=np.float64)
            ))),
        )
        for key in paired_tube_keys
    ], dtype=np.float64)
    active_pair_errors = np.asarray([
        abs(
            float(repair_rows[key]["candidate_worst_pd"])
            - float(baseline_rows[key]["candidate_worst_pd"])
            - float(repair_rows[key][
                "geometry_tube_global_worst_pd_gain"])
        )
        for key in paired_tube_keys
    ], dtype=np.float64)
    policy_delta = float(
        repair["policy_mean_worst"] - baseline["policy_mean_worst"])
    proposal_source_counts: dict[str, int] = {}
    for row in repair["rows"]:
        if not bool(row["geometry_shadow_accepted"]):
            continue
        source = str(row.get("geometry_proposal_source"))
        proposal_source_counts[source] = proposal_source_counts.get(source, 0) + 1
    certificate: dict[str, object] = {
        "schema_version": 2,
        "method": (
            "same seeds, event keys, accepted commitment epochs, quantized "
            "joint-plan digests and RF plans; only certified repair versus "
            "exact committed baseline movement differs inside each H-step tube"
        ),
        "repair_summary": str(args.repair),
        "baseline_summary": str(args.baseline),
        "seed_count": len(seeds),
        "event_count": len(repair_rows),
        "accepted_commitment_count": len(accepted_repair),
        "accepted_commitment_identity_match": True,
        "accepted_proposal_source_counts": proposal_source_counts,
        "policy_mean_worst_repair": float(repair["policy_mean_worst"]),
        "policy_mean_worst_committed_baseline": float(
            baseline["policy_mean_worst"]),
        "paired_policy_mean_worst_delta": policy_delta,
        "policy_qos_rate_repair": float(repair["policy_qos_rate"]),
        "policy_qos_rate_committed_baseline": float(
            baseline["policy_qos_rate"]),
        "paired_policy_qos_rate_delta": float(
            repair["policy_qos_rate"] - baseline["policy_qos_rate"]),
        "per_seed_mean_delta": per_seed_delta,
        "positive_seed_count": int(np.sum(seed_values > 1.0e-15)),
        "zero_seed_count": int(np.sum(np.abs(seed_values) <= 1.0e-15)),
        "negative_seed_count": int(np.sum(seed_values < -1.0e-15)),
        "minimum_seed_delta": float(np.min(seed_values)),
        "maximum_seed_delta": float(np.max(seed_values)),
        "tube_window_worst_pd_delta": tube_window_delta,
        "positive_tube_window_count": int(np.sum(window_values > 0.0)),
        "nonpositive_tube_window_count": int(np.sum(window_values <= 0.0)),
        "minimum_tube_window_delta": float(np.min(window_values)),
        "mean_tube_window_delta": float(np.mean(window_values)),
        "maximum_tube_window_delta": float(np.max(window_values)),
        "outside_accepted_windows_runtime_state_digest_mismatch_count": (
            len(outside_runtime_digest_mismatch)),
        "runtime_state_digest_floating_mantissa_bits": int(
            repair.get("runtime_state_digest_floating_mantissa_bits", 0)),
        "maximum_absolute_policy_delta_outside_accepted_windows": float(
            np.max(np.abs(outside_policy_delta), initial=0.0)),
        "paired_tube_label_maximum_absolute_mismatch": float(
            np.max(paired_label_errors, initial=0.0)),
        "active_pair_delta_maximum_absolute_mismatch": float(
            np.max(active_pair_errors, initial=0.0)),
        "repair_envelope_coverage_rate": float(
            repair["geometry_tube_envelope_coverage_rate"]),
        "repair_certified_window_target_no_harm_failure_count": int(
            repair[
                "geometry_tube_certified_window_target_no_harm_failure_count"
            ]),
        "repair_certified_window_global_worst_no_harm_failure_count": int(
            repair[
                "geometry_tube_certified_window_global_worst_no_harm_failure_count"
            ]),
        "repair_certified_first_step_strict_improvement_failure_count": int(
            repair[
                "geometry_tube_certified_first_step_strict_improvement_failure_count"
            ]),
        "repair_tube_realized_paired_window_target_no_harm_violation_count": int(
            repair[
                "geometry_tube_realized_paired_window_target_no_harm_violation_count"
            ]),
        "repair_tube_realized_paired_window_global_worst_no_harm_violation_count": int(
            repair[
                "geometry_tube_realized_paired_window_global_worst_no_harm_violation_count"
            ]),
        "repair_tube_realized_paired_first_step_strict_improvement_failure_count": int(
            repair[
                "geometry_tube_realized_paired_first_step_strict_improvement_failure_count"
            ]),
        "repair_tube_maximum_active_branch_coefficient_match_error": float(
            repair[
                "geometry_tube_max_active_branch_coefficient_match_error"]),
        "repair_realized_window_target_no_harm_violation_count": int(
            repair[
                "geometry_shadow_realized_window_target_no_harm_violation_count"
            ]),
        "repair_realized_window_global_worst_no_harm_violation_count": int(
            repair[
                "geometry_shadow_realized_window_global_worst_no_harm_violation_count"
            ]),
        "repair_all_protocol_feasible": bool(
            repair["geometry_shadow_all_protocol_feasible"]),
        "geometry_protocol_rounds": [
            "state",
            "gradient",
            "verification_request",
            "verification_return",
            "prepare",
            "vote",
            "decision",
        ],
        "repair_mean_geometry_transport_bits": float(
            repair["geometry_shadow_mean_transport_bits"]),
        "repair_max_geometry_transport_latency_s": float(
            repair["geometry_shadow_max_transport_latency_s"]),
        "repair_mean_combined_transaction_energy_j": float(
            repair["geometry_shadow_mean_combined_energy_j"]),
        "repair_atomic_transaction_complete_execution": bool(
            repair["geometry_atomic_transaction_complete_execution"]),
        "repair_max_terminal_position_error_m": float(
            repair["geometry_shadow_max_terminal_position_error_m"]),
        "repair_max_terminal_velocity_error_mps": float(
            repair["geometry_shadow_max_terminal_velocity_error_mps"]),
        "repair_minimum_separation_m": float(
            repair["geometry_shadow_minimum_separation_m"]),
        "strict_paired_improvement": bool(
            policy_delta > 0.0
            and np.all(seed_values >= -1.0e-15)
            and np.all(window_values > 0.0)
            and not outside_runtime_digest_mismatch
            and np.max(paired_label_errors, initial=0.0) <= 1.0e-12
            and np.max(active_pair_errors, initial=0.0) <= 1.0e-12
            and float(repair["geometry_tube_envelope_coverage_rate"]) == 1.0
            and int(repair[
                "geometry_tube_certified_window_target_no_harm_failure_count"
            ]) == 0
            and int(repair[
                "geometry_tube_certified_window_global_worst_no_harm_failure_count"
            ]) == 0
            and int(repair[
                "geometry_tube_certified_first_step_strict_improvement_failure_count"
            ]) == 0
            and int(repair[
                "geometry_tube_realized_paired_window_target_no_harm_violation_count"
            ]) == 0
            and int(repair[
                "geometry_tube_realized_paired_window_global_worst_no_harm_violation_count"
            ]) == 0
            and int(repair[
                "geometry_tube_realized_paired_first_step_strict_improvement_failure_count"
            ]) == 0
            and float(repair[
                "geometry_tube_max_active_branch_coefficient_match_error"
            ]) <= 1.0e-9
            and bool(repair["geometry_shadow_all_protocol_feasible"])
            and bool(repair[
                "geometry_atomic_transaction_complete_execution"])
        ),
    }
    for field, path in (
        ("zoh_reference", args.zoh_reference),
        ("diagnostic_upper_reference", args.diagnostic_upper_reference),
    ):
        if path is None:
            continue
        reference = json.loads(path.read_text(encoding="utf-8"))
        certificate[field] = str(path)
        certificate[f"policy_mean_worst_delta_vs_{field}"] = float(
            repair["policy_mean_worst"] - reference["policy_mean_worst"])
        certificate[f"policy_qos_rate_delta_vs_{field}"] = float(
            repair["policy_qos_rate"] - reference["policy_qos_rate"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
