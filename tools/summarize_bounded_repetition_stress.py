"""Verify a fixed-repetition U2U stress replay against its R=1 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_exact_equivalent_retiming import _canonical
from tools.summarize_paired_horizon_confirmation import _sha256, _wilson


PHYSICAL_EQUIVALENCE_FIELDS = (
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
)

NOMINAL_NETWORK_FIELDS = (
    "structure_sequence_total_protocol_latency_s",
    "horizon_digest_rendezvous_prefix_latency_s",
    "horizon_digest_rendezvous_latency_s",
    "horizon_digest_rendezvous_suffix_latency_s",
    "horizon_network_nominal_protocol_bits",
    "horizon_network_nominal_rf_energy_j",
)


def _key(row: dict[str, object]) -> tuple[int, int]:
    return int(row["seed"]), int(row["frame"])


def summarize(baseline_path: Path, repeated_path: Path) -> dict[str, object]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    repeated = json.loads(repeated_path.read_text(encoding="utf-8"))
    before_rows = list(baseline.get("rows", []))
    after_rows = list(repeated.get("rows", []))
    if len(before_rows) != len(after_rows):
        raise ValueError("repetition replay changed the row count")
    physical_differences: list[dict[str, object]] = []
    network_differences: list[dict[str, object]] = []
    for index, (before, after) in enumerate(zip(before_rows, after_rows)):
        if _key(before) != _key(after):
            physical_differences.append({"row": index, "field": "seed/frame"})
            continue
        for field in PHYSICAL_EQUIVALENCE_FIELDS:
            if _canonical(before.get(field)) != _canonical(after.get(field)):
                physical_differences.append({"row": index, "field": field})
        for field in NOMINAL_NETWORK_FIELDS:
            if _canonical(before.get(field)) != _canonical(after.get(field)):
                network_differences.append({"row": index, "field": field})
    if physical_differences:
        raise ValueError(
            f"repetition changed physical decisions: {physical_differences[:10]}")
    if network_differences:
        raise ValueError(
            f"repetition changed nominal network physics: "
            f"{network_differences[:10]}")

    baseline_attempts = {
        _key(row): row for row in before_rows
        if bool(row.get("horizon_digest_rendezvous_attempted", False))
    }
    repeated_attempts = {
        _key(row): row for row in after_rows
        if bool(row.get("horizon_digest_rendezvous_attempted", False))
    }
    if baseline_attempts.keys() != repeated_attempts.keys():
        raise ValueError("repetition changed the rendezvous attempt set")
    repetition_counts = {
        int(row["horizon_network_repetition_count"])
        for row in repeated_attempts.values()
    }
    if len(repetition_counts) != 1:
        raise ValueError("repetition replay does not use one frozen R")
    repetitions = repetition_counts.pop()
    if repetitions <= 1:
        raise ValueError("stress replay must use R>1")

    resource_violations: list[dict[str, object]] = []
    gate_violations: list[dict[str, object]] = []
    for key, row in repeated_attempts.items():
        nominal_bits = int(row["horizon_network_nominal_protocol_bits"])
        worst_bits = int(row["horizon_network_worst_protocol_bits"])
        nominal_energy = float(row["horizon_network_nominal_rf_energy_j"])
        worst_energy = float(row["horizon_network_worst_rf_energy_j"])
        if worst_bits != repetitions * nominal_bits:
            resource_violations.append({"key": key, "field": "bits"})
        if not np.isclose(
            worst_energy,
            repetitions * nominal_energy,
            rtol=4.0 * np.finfo(np.float64).eps,
            atol=0.0,
        ):
            resource_violations.append({"key": key, "field": "rf_energy"})
        admitted = bool(
            row.get("horizon_digest_rendezvous_additional_eligible", False))
        complete_gate = all((
            bool(row.get("horizon_digest_rendezvous_exact_accept", False)),
            bool(row.get("horizon_digest_rendezvous_prefix_feasible", False)),
            bool(row.get("horizon_digest_rendezvous_transport_feasible", False)),
            bool(row.get("horizon_digest_rendezvous_suffix_feasible", False)),
            bool(row.get("horizon_digest_rendezvous_comm_bound_valid", False)),
            bool(row.get("horizon_network_repetition_feasible", False)),
            bool(row.get("horizon_digest_rendezvous_deadline_pass", False)),
        ))
        if admitted and not complete_gate:
            gate_violations.append({"key": key, "field": "complete_gate"})
    if resource_violations:
        raise ValueError(
            f"repetition resource law failed: {resource_violations[:10]}")
    if gate_violations:
        raise ValueError(
            f"repetition authorization failed closed: {gate_violations[:10]}")

    baseline_eligible = {
        key for key, row in baseline_attempts.items()
        if bool(row.get("horizon_digest_rendezvous_additional_eligible", False))
    }
    repeated_eligible = {
        key for key, row in repeated_attempts.items()
        if bool(row.get("horizon_digest_rendezvous_additional_eligible", False))
    }
    if not repeated_eligible.issubset(baseline_eligible):
        raise ValueError("repetition created an action absent from R=1 authority")
    physical_accepts = {
        key for key, row in repeated_attempts.items()
        if bool(row.get("horizon_digest_rendezvous_exact_accept", False))
    }
    deadline_closed_accepts = sorted(physical_accepts - repeated_eligible)
    for key in repeated_eligible:
        row = repeated_attempts[key]
        if (
            row.get("horizon_digest_rendezvous_future_target_no_harm") is not True
            or bool(row.get(
                "horizon_digest_rendezvous_future_candidate_bound_failure",
                False))
            or bool(row.get(
                "horizon_digest_rendezvous_future_noop_bound_failure", False))
        ):
            raise ValueError(f"repeated eligible action failed future audit: {key}")

    summary = dict(repeated["summary"])
    eligible_seeds = {seed for seed, _ in repeated_eligible}
    episode_count = len(repeated.get("seed_order", []))
    return {
        "schema_version": 1,
        "status": "bounded_repetition_stress_safe_fail_closed_shadow_only",
        "independent_validation_reused": True,
        "new_independent_validation_claim_created": False,
        "stress_design": {
            "repetition_count": repetitions,
            "tolerated_erasures_per_logical_packet": repetitions - 1,
            "excess_queue_bound_s": float(summary[
                "horizon_network_excess_queue_bound_s"]),
            "guarantee": (
                "delivery support if every logical packet loses at most "
                "R-1 fixed copies; no stochastic independence assumed"
            ),
        },
        "equivalence": {
            "row_count": len(after_rows),
            "physical_field_count_per_row": len(
                PHYSICAL_EQUIVALENCE_FIELDS),
            "physical_comparison_count": (
                len(after_rows) * len(PHYSICAL_EQUIVALENCE_FIELDS)),
            "physical_difference_count": 0,
            "nominal_network_field_count_per_row": len(
                NOMINAL_NETWORK_FIELDS),
            "nominal_network_comparison_count": (
                len(after_rows) * len(NOMINAL_NETWORK_FIELDS)),
            "nominal_network_difference_count": 0,
        },
        "resources": {
            "nominal_protocol_bits": int(summary[
                "horizon_network_nominal_protocol_bits"]),
            "worst_case_protocol_bits": int(summary[
                "horizon_network_worst_protocol_bits"]),
            "nominal_rf_energy_j": float(summary[
                "horizon_network_nominal_rf_energy_j"]),
            "worst_case_rf_energy_j": float(summary[
                "horizon_network_worst_rf_energy_j"]),
            "per_event_repetition_law_violation_count": 0,
            "compute_energy_included": False,
        },
        "timing_and_authority": {
            "deadline_s": 0.1,
            "attempt_count": len(repeated_attempts),
            "network_feasible_count": int(summary[
                "horizon_network_repetition_feasible_count"]),
            "deadline_pass_count": int(summary[
                "horizon_digest_rendezvous_deadline_pass_count"]),
            "nominal_latency_s_mean": float(summary[
                "horizon_digest_rendezvous_mean_nominal_total_latency_s"]),
            "worst_case_latency_s_mean": float(summary[
                "horizon_digest_rendezvous_mean_total_latency_s"]),
            "worst_case_latency_s_max": float(summary[
                "horizon_digest_rendezvous_max_total_latency_s"]),
            "physical_accept_count": len(physical_accepts),
            "timely_admitted_count": len(repeated_eligible),
            "timely_admitted_episode_count": len(eligible_seeds),
            "timely_admitted_episode_wilson95": _wilson(
                len(eligible_seeds), episode_count),
            "deadline_closed_physical_accept_count": len(
                deadline_closed_accepts),
            "deadline_closed_physical_accepts": [{
                "seed": seed,
                "frame": frame,
                "worst_case_latency_s": float(repeated_attempts[(seed, frame)][
                    "horizon_digest_rendezvous_total_latency_s"]),
            } for seed, frame in deadline_closed_accepts],
            "admitted_outside_r1_count": 0,
            "admitted_without_complete_gate_count": 0,
        },
        "safety": {
            "future_failure_count": int(summary[
                "horizon_digest_rendezvous_future_failure_count"]),
            "max_isac_power_balance_error_w": float(summary[
                "horizon_digest_rendezvous_max_power_balance_error_w"]),
        },
        "scope": {
            "post_hoc_stress_on_existing_independent_episodes": True,
            "new_independent_validation": False,
            "empirical_packet_loss_calibration": False,
            "deployment_network_jitter_measured": False,
            "hardware_cpu_energy_measured": False,
            "commit_authority": False,
        },
        "provenance": {
            "r1_baseline": {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
            },
            "repeated_stress": {
                "path": str(repeated_path),
                "sha256": _sha256(repeated_path),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--repeated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.baseline, args.repeated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
