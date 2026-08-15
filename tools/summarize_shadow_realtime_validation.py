"""Summarize a preregistered D0.21 real-time shadow validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_paired_horizon_confirmation import (
    _bootstrap_mean_ci,
    _sha256,
    _two_sided_sign_p,
    _wilson,
)


def summarize(
    audit_path: Path,
    seed_manifest_path: Path,
    trace_path: Path,
    *,
    bootstrap_samples: int = 100000,
    bootstrap_seed: int = 20260811,
) -> dict[str, object]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    seed_manifest = json.loads(
        Path(seed_manifest_path).read_text(encoding="utf-8"))
    splits = list(seed_manifest.get("splits", {}).values())
    if len(splits) != 1:
        raise ValueError("D0.21 seed manifest must contain exactly one split")
    seeds = [int(seed) for seed in splits[0]]
    if [int(seed) for seed in audit.get("seed_order", [])] != seeds:
        raise ValueError("audit seed order differs from preregistration")

    design = audit.get("horizon_diagnostic", {})
    expected = {
        "steps": 3,
        "proposal_ranking": "distributed_screened_margin",
        "transition_gate": "common_box_paired_deflection",
        "power_plan": "common_robust_weights",
        "weak_target_count": 3,
        "local_shortlist_per_owner": 1,
        "concurrent_owner_workers": 4,
        "deadline_accounting": "complete_ranking_wall_plus_transport",
    }
    mismatches = {
        key: {"expected": value, "observed": design.get(key)}
        for key, value in expected.items()
        if design.get(key) != value
    }
    if mismatches:
        raise ValueError(f"D0.21 frozen design mismatch: {mismatches}")

    rows = list(audit.get("rows", []))
    evaluated = [
        row for row in rows
        if bool(row.get("horizon_diagnostic_evaluated", False))
    ]
    accepted = [
        row for row in evaluated
        if bool(row.get("horizon_diagnostic_accept", False))
    ]
    if not evaluated or not accepted:
        raise ValueError("D0.21 audit has no evaluated/accepted event")
    if any(not bool(row.get("horizon_owner_concurrent_measured", False))
           for row in evaluated):
        raise ValueError("an evaluated event lacks concurrent-owner timing")
    if any(not bool(row.get("horizon_future_outcome_valid", False))
           for row in accepted):
        raise ValueError("an accepted event lacks a true-future audit")

    summary = audit["summary"]
    coefficient_failures = int(
        summary["horizon_future_lower_zero_support_failure_count"]
    ) + int(summary["horizon_future_upper_zero_support_failure_count"])
    coefficient_failures += int(round(
        float(summary["horizon_future_candidate_bound_failure_rate"])
        * int(summary["horizon_future_outcome_event_count"])))
    coefficient_failures += int(round(
        float(summary["horizon_future_noop_bound_failure_rate"])
        * int(summary["horizon_future_outcome_event_count"])))

    tolerance = 1.0e-12
    target_deltas: list[float] = []
    for row in accepted:
        candidate = np.asarray(
            row["horizon_future_realized_candidate_pd"], dtype=np.float64)
        noop = np.asarray(
            row["horizon_future_realized_noop_pd"], dtype=np.float64)
        target_deltas.extend((candidate - noop).reshape(-1).tolist())
    no_harm_failures = int(np.sum(
        np.asarray(target_deltas, dtype=np.float64) < -tolerance))
    accepted_deadline = np.asarray([
        float(row["horizon_diagnostic_end_to_end_latency_s"])
        <= 0.1 + tolerance
        for row in accepted
    ], dtype=bool)
    accepted_comm_failures = int(sum(
        row.get("horizon_ranking_comm_bound_valid") is False
        for row in accepted
    ))

    rows_by_seed = {
        seed: [row for row in rows if int(row["seed"]) == seed]
        for seed in seeds
    }
    unsafe_episode_count = int(sum(
        any(
            int(row.get(
                "horizon_future_lower_zero_support_failure_count", 0)) > 0
            or int(row.get(
                "horizon_future_upper_zero_support_failure_count", 0)) > 0
            or bool(row.get("horizon_future_candidate_bound_failure", False))
            or bool(row.get("horizon_future_noop_bound_failure", False))
            or (
                bool(row.get("horizon_diagnostic_accept", False))
                and (
                    row.get("horizon_future_realized_target_no_harm") is False
                    or float(row["horizon_diagnostic_end_to_end_latency_s"])
                    > 0.1 + tolerance
                    or row.get("horizon_ranking_comm_bound_valid") is False
                )
            )
            for row in rows_by_seed[seed]
        )
        for seed in seeds
    ))
    physical_realtime_pass = bool(
        coefficient_failures == 0
        and no_harm_failures == 0
        and accepted_comm_failures == 0
        and np.all(accepted_deadline)
    )

    baseline_counts = np.asarray([
        sum(bool(row.get("structure_accepted", False))
            for row in rows_by_seed[seed])
        for seed in seeds
    ], dtype=np.int64)
    horizon_counts = np.asarray([
        sum(bool(row.get("horizon_diagnostic_accept", False))
            for row in rows_by_seed[seed])
        for seed in seeds
    ], dtype=np.int64)
    union_counts = np.asarray([
        sum(
            bool(row.get("structure_accepted", False))
            or bool(row.get("horizon_diagnostic_accept", False))
            for row in rows_by_seed[seed]
        )
        for seed in seeds
    ], dtype=np.int64)
    trigger_counts = np.asarray([
        sum(bool(row.get("structure_triggered", False))
            for row in rows_by_seed[seed])
        for seed in seeds
    ], dtype=np.int64)
    gains = union_counts - baseline_counts
    rate_gains = gains / np.maximum(trigger_counts, 1)
    positive = int(np.sum(gains > 0))
    negative = int(np.sum(gains < 0))

    latencies = np.asarray([
        float(row["horizon_diagnostic_end_to_end_latency_s"])
        for row in evaluated
    ], dtype=np.float64)
    accepted_latencies = np.asarray([
        float(row["horizon_diagnostic_end_to_end_latency_s"])
        for row in accepted
    ], dtype=np.float64)
    # With no calibrated CPU wattage, energy remains the affine model
    # E(P_cpu)=E_protocol + P_cpu * t_process_cpu.  Reporting both terms avoids
    # inventing a hardware measurement.
    energy_intercepts = np.asarray([
        float(row["shadow_router_total_control_energy_j"])
        for row in evaluated
    ], dtype=np.float64)
    cpu_coefficients = np.asarray([
        float(row["structure_ranking_process_cpu_s"])
        + float(row["horizon_ranking_process_cpu_s"])
        for row in evaluated
    ], dtype=np.float64)

    return {
        "schema_version": 1,
        "status": (
            "validation_physical_realtime_pass_shadow_only"
            if physical_realtime_pass else "validation_did_not_pass"
        ),
        "frozen_design": {
            **expected,
            "miscoverage": float(design["miscoverage"]),
            "transition_residual_log_margin": float(
                design["frozen_transition_residual_log_margin"]),
            "commit_authority": False,
        },
        "validation": {
            "episode_count": len(seeds),
            "evaluated_event_count": len(evaluated),
            "accepted_event_count": len(accepted),
            "accepted_rate": float(len(accepted) / len(evaluated)),
            "future_outcome_event_count": int(
                summary["horizon_future_outcome_event_count"]),
            "audited_coefficient_count": int(
                summary["horizon_future_coefficient_audit_count"]),
            "coefficient_envelope_failure_count": coefficient_failures,
            "accepted_target_step_count": len(target_deltas),
            "accepted_target_no_harm_failure_count": no_harm_failures,
            "accepted_min_realized_target_pd_delta": float(
                min(target_deltas)),
            "accepted_deadline_rate": float(np.mean(accepted_deadline)),
            "accepted_communication_failure_count": accepted_comm_failures,
            "episode_any_failure_count": unsafe_episode_count,
            "episode_any_failure_rate_wilson95": _wilson(
                unsafe_episode_count, len(seeds)),
            "physical_realtime_pass": physical_realtime_pass,
        },
        "timing": {
            "complete_end_to_end_latency_s_quantiles": {
                str(q): float(np.quantile(latencies, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "complete_end_to_end_latency_s_mean": float(np.mean(latencies)),
            "complete_end_to_end_latency_s_max": float(np.max(latencies)),
            "evaluated_deadline_rate": float(np.mean(
                latencies <= 0.1 + tolerance)),
            "accepted_latency_s_mean": float(np.mean(accepted_latencies)),
            "accepted_latency_s_max": float(np.max(accepted_latencies)),
            "actual_concurrent_owner_measurement": True,
            "deployment_jitter_measured": False,
        },
        "incremental_route": {
            "one_step_accept_count": int(np.sum(baseline_counts)),
            "horizon_accept_count": int(np.sum(horizon_counts)),
            "overlap_count": int(sum(
                bool(row.get("structure_accepted", False))
                and bool(row.get("horizon_diagnostic_accept", False))
                for row in rows
            )),
            "horizon_only_count": int(
                np.sum(union_counts) - np.sum(baseline_counts)),
            "union_accept_count": int(np.sum(union_counts)),
            "episode_positive_count": positive,
            "episode_negative_count": negative,
            "episode_tie_count": int(len(seeds) - positive - negative),
            "episode_exact_two_sided_sign_p": _two_sided_sign_p(
                positive, negative),
            "mean_incremental_accept_count_per_episode": float(
                np.mean(gains)),
            "mean_incremental_accept_count_bootstrap95": _bootstrap_mean_ci(
                gains,
                samples=int(bootstrap_samples),
                seed=int(bootstrap_seed),
            ),
            "mean_incremental_trigger_rate_per_episode": float(
                np.mean(rate_gains)),
            "mean_incremental_trigger_rate_bootstrap95": _bootstrap_mean_ci(
                rate_gains,
                samples=int(bootstrap_samples),
                seed=int(bootstrap_seed) + 1,
            ),
        },
        "shadow_authority": {
            "full_action_identity_match_count": int(sum(
                row.get("shadow_router_candidate_identity_match") is True
                for row in evaluated
            )),
            "consensus_candidate_count": int(sum(
                row.get("shadow_router_action") == "consensus_candidate"
                for row in evaluated
            )),
            "resource_accounting_complete": False,
            "commit_authority": False,
        },
        "compute_energy_model": {
            "hardware_power_metered": False,
            "formula": (
                "E_total(P_cpu)=E_protocol+P_cpu*t_process_cpu"),
            "evaluated_mean_protocol_intercept_j": float(np.mean(
                energy_intercepts)),
            "evaluated_mean_process_cpu_coefficient_s": float(np.mean(
                cpu_coefficients)),
            "evaluated_p95_process_cpu_coefficient_s": float(np.quantile(
                cpu_coefficients, 0.95)),
            "calibrated_logical_cpu_power_w": None,
        },
        "provenance": {
            "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "seed_manifest": {
                "path": str(seed_manifest_path),
                "sha256": _sha256(seed_manifest_path),
            },
            "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        },
        "system_certificate_ready": False,
        "remaining_blockers": [
            "shadow route has no commit authority",
            "one-step and horizon full action digests never agree",
            "common preprocessing and power-only compute are not fully metered",
            "CPU wattage is not hardware calibrated",
            "thread timing is measured on one workstation, not distributed UAV hardware",
            "future outcomes use a frozen open-loop movement-action tape",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.audit,
        args.seed_manifest,
        args.trace,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "validation": result["validation"],
        "timing": result["timing"],
        "incremental_route": result["incremental_route"],
    }, indent=2))


if __name__ == "__main__":
    main()
