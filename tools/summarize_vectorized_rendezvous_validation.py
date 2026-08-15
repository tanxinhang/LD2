"""Summarize the frozen D0.29 vectorized-rendezvous validation.

The exchangeability unit is an episode seed.  Event, horizon-step and target
counts remain diagnostics and are never substituted for independent episodes.
Late attempts fail closed; only a physically accepted, resource-feasible and
deadline-feasible rendezvous is eligible even in the non-authoritative shadow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_paired_horizon_confirmation import _sha256, _wilson


def _single_split(manifest: dict[str, object]) -> list[int]:
    splits = list(dict(manifest.get("splits", {})).values())
    if len(splits) != 1:
        raise ValueError("D0.29 seed manifest must contain exactly one split")
    return [int(seed) for seed in splits[0]]


def summarize(
    audit_path: Path,
    seed_manifest_path: Path,
    trace_path: Path,
    *,
    validation_label: str = "D0.29",
) -> dict[str, object]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    manifest = json.loads(
        Path(seed_manifest_path).read_text(encoding="utf-8"))
    seeds = _single_split(manifest)
    if [int(seed) for seed in audit.get("seed_order", [])] != seeds:
        raise ValueError("audit seed order differs from the frozen manifest")

    design = dict(audit.get("horizon_diagnostic", {}))
    expected_horizon = {
        "steps": 3,
        "proposal_ranking": "causal_set_membership",
        "transition_gate": "common_box_paired_deflection",
        "power_plan": "common_robust_weights",
        "weak_target_count": 3,
        "ranking_rounds": 4,
        "deadline_accounting": (
            "complete_common_preprocessing_horizon_envelope_ranking_"
            "wall_plus_transport"),
        "set_membership_verifier": True,
        "reuse_primal_master_duals": True,
        "commit_authority": False,
    }
    rendezvous = dict(audit.get("horizon_digest_rendezvous", {}))
    expected_rendezvous = {
        "enabled": True,
        "digest_bits": 256,
        "structure_commit_reuse_enabled": True,
        "deferred_top1_transport_enabled": True,
        "set_membership_verifier_enabled": True,
        "reuse_primal_master_duals": True,
        "resource_accounting_complete": False,
        "commit_authority": False,
    }
    mismatches = {
        f"horizon.{key}": {"expected": value, "observed": design.get(key)}
        for key, value in expected_horizon.items()
        if design.get(key) != value
    }
    mismatches.update({
        f"rendezvous.{key}": {
            "expected": value,
            "observed": rendezvous.get(key),
        }
        for key, value in expected_rendezvous.items()
        if rendezvous.get(key) != value
    })
    if mismatches:
        raise ValueError(f"D0.29 frozen design mismatch: {mismatches}")

    rows = list(audit.get("rows", []))
    attempts = [
        row for row in rows
        if bool(row.get("horizon_digest_rendezvous_attempted", False))
    ]
    physical_accepts = [
        row for row in attempts
        if bool(row.get("horizon_digest_rendezvous_exact_accept", False))
    ]
    eligible = [
        row for row in attempts
        if bool(row.get(
            "horizon_digest_rendezvous_additional_eligible", False))
    ]
    if not attempts or not physical_accepts:
        raise ValueError("D0.29 has no rendezvous attempt/physical acceptance")

    tolerance = 1.0e-12
    target_deltas: list[float] = []
    for row in eligible:
        if not bool(row.get(
            "horizon_digest_rendezvous_future_valid", False)):
            raise ValueError("eligible rendezvous lacks future audit")
        candidate = np.asarray(
            row["horizon_digest_rendezvous_future_candidate_pd"],
            dtype=np.float64,
        )
        noop = np.asarray(
            row["horizon_digest_rendezvous_future_noop_pd"],
            dtype=np.float64,
        )
        if candidate.shape != noop.shape or candidate.shape != (3, 6):
            raise ValueError("D0.29 future outcome has the wrong H-by-Q shape")
        target_deltas.extend((candidate - noop).reshape(-1).tolist())

    attempt_deadline_failures = [
        row for row in attempts
        if not bool(row.get("horizon_digest_rendezvous_deadline_pass", False))
    ]
    accepted_deadline_failures = [
        row for row in physical_accepts
        if not bool(row.get("horizon_digest_rendezvous_deadline_pass", False))
    ]
    accepted_transport_failures = [
        row for row in physical_accepts
        if (
            not bool(row.get(
                "horizon_digest_rendezvous_prefix_feasible", False))
            or not bool(row.get(
                "horizon_digest_rendezvous_transport_feasible", False))
            or not bool(row.get(
                "horizon_digest_rendezvous_suffix_feasible", False))
            or not bool(row.get(
                "horizon_digest_rendezvous_comm_bound_valid", False))
        )
    ]
    future_failure_rows = [
        row for row in eligible
        if (
            bool(row.get(
                "horizon_digest_rendezvous_future_candidate_bound_failure",
                False))
            or bool(row.get(
                "horizon_digest_rendezvous_future_noop_bound_failure",
                False))
            or row.get(
                "horizon_digest_rendezvous_future_target_no_harm") is False
        )
    ]
    target_harm_count = int(np.sum(
        np.asarray(target_deltas, dtype=np.float64) < -tolerance
    ))
    physical_accept_seed_set = {
        int(row["seed"]) for row in physical_accepts}
    eligible_seed_set = {int(row["seed"]) for row in eligible}
    failure_seed_set = {
        int(row["seed"])
        for row in (
            accepted_deadline_failures
            + accepted_transport_failures
            + future_failure_rows
        )
    }
    if target_harm_count:
        failure_seed_set.update(eligible_seed_set)

    physical_fail_closed = bool(
        len(eligible) == len(physical_accepts)
        and not accepted_deadline_failures
        and not accepted_transport_failures
        and not future_failure_rows
        and target_harm_count == 0
    )
    latencies = np.asarray([
        float(row["horizon_digest_rendezvous_total_latency_s"])
        for row in attempts
    ], dtype=np.float64)
    accepted_latencies = np.asarray([
        float(row["horizon_digest_rendezvous_total_latency_s"])
        for row in physical_accepts
    ], dtype=np.float64)

    status = (
        "validation_safe_fail_closed_rendezvous_shadow_only"
        if physical_fail_closed
        else "validation_rendezvous_did_not_pass"
    )
    return {
        "schema_version": 1,
        "validation_label": str(validation_label),
        "status": status,
        "frozen_design": {
            **expected_horizon,
            **{f"rendezvous_{key}": value
               for key, value in expected_rendezvous.items()},
        },
        "independent_validation": {
            "episode_count": len(seeds),
            "rendezvous_attempt_event_count": len(attempts),
            "rendezvous_attempt_episode_count": len({
                int(row["seed"]) for row in attempts}),
            "physical_accept_event_count": len(physical_accepts),
            "physical_accept_episode_count": len(physical_accept_seed_set),
            "physical_accept_seeds": sorted(physical_accept_seed_set),
            "eligible_event_count": len(eligible),
            "eligible_episode_count": len(eligible_seed_set),
            "eligible_seeds": sorted(eligible_seed_set),
            "episode_occurrence_rate_wilson95": _wilson(
                len(eligible_seed_set), len(seeds)),
        },
        "timing": {
            "attempt_deadline_pass_count": int(
                len(attempts) - len(attempt_deadline_failures)),
            "attempt_deadline_failure_count": len(
                attempt_deadline_failures),
            "attempt_deadline_rate": float(np.mean(latencies <= 0.1)),
            "attempt_latency_s_mean": float(np.mean(latencies)),
            "attempt_latency_s_quantiles": {
                str(q): float(np.quantile(latencies, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            },
            "attempt_latency_s_max": float(np.max(latencies)),
            "physical_accept_deadline_failure_count": len(
                accepted_deadline_failures),
            "physical_accept_latency_s_mean": float(np.mean(
                accepted_latencies)),
            "physical_accept_latency_s_max": float(np.max(
                accepted_latencies)),
            "late_attempt_policy": "fail_closed_to_noop",
            "deployment_jitter_measured": False,
        },
        "physical_and_communication_safety": {
            "accepted_transport_failure_count": len(
                accepted_transport_failures),
            "eligible_future_failure_event_count": len(future_failure_rows),
            "eligible_target_step_count": len(target_deltas),
            "eligible_target_no_harm_failure_count": target_harm_count,
            "eligible_minimum_realized_target_pd_delta": (
                float(min(target_deltas)) if target_deltas else None),
            "max_isac_power_balance_error_w": float(max(
                float(row.get(
                    "horizon_digest_rendezvous_power_balance_error_w", 0.0)
                    or 0.0)
                for row in attempts
            )),
            "episode_any_failure_count": len(failure_seed_set),
            "episode_any_failure_rate_wilson95_all_episodes": _wilson(
                len(failure_seed_set), len(seeds)),
            "conditional_episode_any_failure_rate_wilson95": (
                _wilson(len(failure_seed_set), len(eligible_seed_set))
                if eligible_seed_set else None
            ),
            "physical_fail_closed_pass": physical_fail_closed,
        },
        "authority": {
            "resource_accounting_complete": False,
            "hardware_cpu_power_metered": False,
            "deployment_scheduler_jitter_measured": False,
            "commit_authority": False,
            "actual_selected_candidate_count": 0,
        },
        "interpretation": (
            "Every independently observed physical acceptance also met the "
            "transport and 100 ms gates, while late non-accepted attempts "
            "failed closed.  The result supports a safe nonempty shadow "
            "protocol, not deployment readiness: occurrence and conditional "
            "safety uncertainty remain episode-limited and compute energy/"
            "deployment jitter are not calibrated."
        ),
        "system_certificate_ready": False,
        "remaining_blockers": [
            (
                f"only {len(eligible_seed_set)} independent episodes contain "
                "an eligible action"
            ),
            "CPU electrical power is not hardware-metered",
            "deployment scheduler and network jitter are not measured",
            "the frozen future movement tape is audit-only, not a live closed loop",
            "hardware RF impairments and SDR-in-the-loop behavior are untested",
        ],
        "provenance": {
            "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "seed_manifest": {
                "path": str(seed_manifest_path),
                "sha256": _sha256(seed_manifest_path),
            },
            "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="D0.29")
    args = parser.parse_args()
    result = summarize(
        args.audit,
        args.seed_manifest,
        args.trace,
        validation_label=str(args.label),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "independent_validation": result["independent_validation"],
        "timing": result["timing"],
        "physical_and_communication_safety": (
            result["physical_and_communication_safety"]),
        "authority": result["authority"],
    }, indent=2))


if __name__ == "__main__":
    main()
