"""Summarize a frozen paired-horizon confirmation audit without retuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _two_sided_sign_p(positive: int, negative: int) -> float:
    n = int(positive) + int(negative)
    if n == 0:
        return 1.0
    tail = min(int(positive), int(negative))
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * probability))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size < 1 or samples < 1:
        raise ValueError("bootstrap inputs must be non-empty and positive")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    # Chunking bounds memory while preserving a fixed, reproducible RNG tape.
    offset = 0
    while offset < int(samples):
        count = min(4096, int(samples) - offset)
        indices = rng.integers(0, data.size, size=(count, data.size))
        means[offset:offset + count] = np.mean(data[indices], axis=1)
        offset += count
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def summarize(
    audit_path: Path,
    calibration_path: Path,
    seed_manifest_path: Path,
    trace_path: Path,
    *,
    bootstrap_samples: int = 100000,
    bootstrap_seed: int = 20260811,
) -> dict[str, object]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    calibration = json.loads(
        Path(calibration_path).read_text(encoding="utf-8"))
    seed_manifest = json.loads(
        Path(seed_manifest_path).read_text(encoding="utf-8"))
    split_values = list(seed_manifest.get("splits", {}).values())
    if len(split_values) != 1:
        raise ValueError("confirmation seed manifest must contain one split")
    registered_seeds = [int(seed) for seed in split_values[0]]
    audit_seeds = [int(seed) for seed in audit.get("seed_order", [])]
    if audit_seeds != registered_seeds:
        raise ValueError("audit seeds differ from the preregistered order")
    horizon = audit.get("horizon_diagnostic", {})
    if horizon.get("transition_gate") != "common_box_paired_deflection":
        raise ValueError("audit did not use the frozen paired Deflection gate")
    if not bool(calibration.get("envelope_calibration_ready", False)):
        raise ValueError("horizon coefficient envelope is not calibrated")
    if not np.isclose(
        float(horizon.get("frozen_transition_residual_log_margin", np.nan)),
        float(calibration.get(
            "frozen_transition_residual_log_margin", np.nan)),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("audit and calibration transition margins differ")
    if not np.isclose(
        float(horizon.get("miscoverage", np.nan)),
        float(calibration.get("alpha", np.nan)),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("audit and calibration risk levels differ")

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
        raise ValueError("confirmation audit has no evaluated/accepted events")
    if any(not bool(row.get("horizon_future_outcome_valid", False))
           for row in accepted):
        raise ValueError("an accepted event lacks a complete future outcome")

    future_target_deltas = []
    accepted_sum_pd_gain = []
    for row in accepted:
        candidate = np.asarray(
            row["horizon_future_realized_candidate_pd"], dtype=np.float64)
        noop = np.asarray(
            row["horizon_future_realized_noop_pd"], dtype=np.float64)
        delta = candidate - noop
        future_target_deltas.extend(delta.reshape(-1).tolist())
        accepted_sum_pd_gain.append(float(np.sum(delta)))
    tolerance = 1.0e-12
    # Schema-6 audits originally stored transport plus max(owner compute),
    # which is only a modeled owner critical path.  The already-recorded
    # horizon_ranking_compute_s is the actual end-to-end wall time of the
    # complete ranking call in this implementation and includes candidate
    # generation/coordinator overhead.  Recompute the deadline from it rather
    # than legitimizing an incomplete latency field.
    def actual_end_to_end_latency(row: dict[str, object]) -> float:
        return float(row["horizon_ranking_compute_s"]) + float(
            row["horizon_diagnostic_transport_latency_s"])

    accepted_deadline = np.asarray([
        actual_end_to_end_latency(row) <= 0.1 + tolerance
        for row in accepted
    ], dtype=bool)
    late_evaluated = [
        row for row in evaluated
        if actual_end_to_end_latency(row) > 0.1 + tolerance
    ]

    seed_rows = {
        seed: [row for row in rows if int(row["seed"]) == seed]
        for seed in registered_seeds
    }
    myopic_counts = np.asarray([
        sum(bool(row.get("structure_accepted", False))
            for row in seed_rows[seed])
        for seed in registered_seeds
    ], dtype=np.int64)
    horizon_counts = np.asarray([
        sum(bool(row.get("horizon_diagnostic_accept", False))
            for row in seed_rows[seed])
        for seed in registered_seeds
    ], dtype=np.int64)
    union_counts = np.asarray([
        sum(
            bool(row.get("structure_accepted", False))
            or bool(row.get("horizon_diagnostic_accept", False))
            for row in seed_rows[seed]
        )
        for seed in registered_seeds
    ], dtype=np.int64)
    trigger_counts = np.asarray([
        sum(bool(row.get("structure_triggered", False))
            for row in seed_rows[seed])
        for seed in registered_seeds
    ], dtype=np.int64)
    count_gain = union_counts - myopic_counts
    rate_gain = count_gain / np.maximum(trigger_counts, 1)
    positive = int(np.sum(count_gain > 0))
    negative = int(np.sum(count_gain < 0))
    ties = int(np.sum(count_gain == 0))
    both = int(sum(
        bool(row.get("structure_accepted", False))
        and bool(row.get("horizon_diagnostic_accept", False))
        for row in rows
    ))
    myopic_total = int(np.sum(myopic_counts))
    horizon_total = int(np.sum(horizon_counts))
    union_total = int(np.sum(union_counts))
    horizon_only = union_total - myopic_total
    summary = audit.get("summary", {})
    coefficient_failures = int(summary.get(
        "horizon_future_lower_zero_support_failure_count", 0
    )) + int(summary.get(
        "horizon_future_upper_zero_support_failure_count", 0
    ))
    coefficient_failures += int(round(
        float(summary.get("horizon_future_candidate_bound_failure_rate", 0.0))
        * int(summary.get("horizon_future_outcome_event_count", 0))))
    coefficient_failures += int(round(
        float(summary.get("horizon_future_noop_bound_failure_rate", 0.0))
        * int(summary.get("horizon_future_outcome_event_count", 0))))
    no_harm_violations = int(np.sum(
        np.asarray(future_target_deltas) < -tolerance))
    unsafe_episode_count = int(sum(
        any(
            (
                int(row.get(
                    "horizon_future_lower_zero_support_failure_count", 0))
                + int(row.get(
                    "horizon_future_upper_zero_support_failure_count", 0))
                > 0
            )
            or bool(row.get("horizon_future_candidate_bound_failure", False))
            or bool(row.get("horizon_future_noop_bound_failure", False))
            or (
                bool(row.get("horizon_diagnostic_accept", False))
                and row.get("horizon_future_realized_target_no_harm") is False
            )
            for row in seed_rows[seed]
        )
        for seed in registered_seeds
    ))
    safety_pass = bool(
        coefficient_failures == 0
        and no_harm_violations == 0
        and np.all(accepted_deadline)
        and not any(bool(row.get("horizon_ranking_comm_bound_valid")) is False
                    for row in accepted)
    )

    return {
        "schema_version": 1,
        "status": (
            "confirmation_safety_and_incremental_performance_pass_"
            "diagnostic_only"
            if safety_pass and positive > negative
            else "confirmation_did_not_pass"
        ),
        "frozen_design": {
            "horizon_steps": int(horizon["steps"]),
            "transition_gate": horizon["transition_gate"],
            "proposal_ranking": horizon["proposal_ranking"],
            "weak_target_count": int(horizon["weak_target_count"]),
            "local_shortlist_per_owner": int(
                horizon["local_shortlist_per_owner"]),
            "miscoverage": float(horizon["miscoverage"]),
            "transition_residual_log_margin": float(
                horizon["frozen_transition_residual_log_margin"]),
            "uav_reachable_position_radius_m": horizon[
                "uav_reachable_position_radius_m"],
            "uav_reachable_velocity_radius_mps": horizon[
                "uav_reachable_velocity_radius_mps"],
            "commit_authority": False,
        },
        "calibration": {
            "episode_count": int(calibration["calibration_episode_count"]),
            "rank_one_based": int(calibration["rank_one_based"]),
            "finite_sample_coverage_floor": float(
                calibration["finite_sample_coverage_floor"]),
            "audited_coefficient_count": int(
                calibration["audited_coefficient_count"]),
        },
        "confirmation": {
            "episode_count": len(registered_seeds),
            "evaluated_event_count": len(evaluated),
            "horizon_accept_count": len(accepted),
            "horizon_accept_rate": float(len(accepted) / len(evaluated)),
            "future_outcome_event_count": int(summary[
                "horizon_future_outcome_event_count"]),
            "audited_coefficient_count": int(summary[
                "horizon_future_coefficient_audit_count"]),
            "coefficient_envelope_failure_count": coefficient_failures,
            "accepted_target_step_count": len(future_target_deltas),
            "accepted_target_no_harm_violation_count": no_harm_violations,
            "accepted_min_realized_target_pd_delta": float(
                min(future_target_deltas)),
            "accepted_mean_sum_pd_gain": float(np.mean(
                accepted_sum_pd_gain)),
            "accepted_median_sum_pd_gain": float(np.median(
                accepted_sum_pd_gain)),
            "accepted_actual_single_process_deadline_rate": float(np.mean(
                accepted_deadline)),
            "accepted_max_actual_single_process_end_to_end_latency_s": (
                float(max(actual_end_to_end_latency(row) for row in accepted))
            ),
            "accepted_mean_actual_single_process_end_to_end_latency_s": (
                float(np.mean([
                    actual_end_to_end_latency(row) for row in accepted
                ]))
            ),
            "legacy_incomplete_latency_field": (
                "horizon_diagnostic_end_to_end_latency_s omitted candidate "
                "generation/coordinator overhead in this frozen audit"
            ),
            "late_evaluated_event_count": len(late_evaluated),
            "late_accepted_event_count": int(sum(
                bool(row.get("horizon_diagnostic_accept", False))
                for row in late_evaluated)),
            "communication_bound_failure_count": int(summary[
                "horizon_ranking_comm_bound_failure_count"]),
            "episode_any_envelope_or_accepted_harm_failure_count": (
                unsafe_episode_count),
            "episode_any_failure_rate_wilson95": _wilson(
                unsafe_episode_count, len(registered_seeds)),
            "safety_pass": safety_pass,
        },
        "incremental_route": {
            "myopic_accept_count": myopic_total,
            "horizon_accept_count": horizon_total,
            "overlap_count": both,
            "horizon_only_accept_count": horizon_only,
            "union_accept_count": union_total,
            "union_relative_count_gain": float(
                (union_total - myopic_total) / max(myopic_total, 1)),
            "union_trigger_rate_gain": float(
                (union_total - myopic_total) / max(np.sum(trigger_counts), 1)),
            "episode_positive_count": positive,
            "episode_negative_count": negative,
            "episode_tie_count": ties,
            "episode_exact_two_sided_sign_p": _two_sided_sign_p(
                positive, negative),
            "mean_incremental_accept_count_per_episode": float(np.mean(
                count_gain)),
            "mean_incremental_accept_count_cluster_bootstrap95": (
                _bootstrap_mean_ci(
                    count_gain,
                    samples=int(bootstrap_samples),
                    seed=int(bootstrap_seed),
                )),
            "mean_incremental_trigger_rate_per_episode": float(np.mean(
                rate_gain)),
            "mean_incremental_trigger_rate_cluster_bootstrap95": (
                _bootstrap_mean_ci(
                    rate_gain,
                    samples=int(bootstrap_samples),
                    seed=int(bootstrap_seed) + 1,
                )),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "provenance": {
            "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "calibration": {
                "path": str(calibration_path),
                "sha256": _sha256(calibration_path),
            },
            "seed_manifest": {
                "path": str(seed_manifest_path),
                "sha256": _sha256(seed_manifest_path),
            },
            "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        },
        "system_certificate_ready": False,
        "remaining_blockers": [
            "paired horizon remains a diagnostic route without live commit authority",
            "owner-parallel latency is modeled as max serial owner-group time; "
            "actual concurrent scheduling and jitter are not measured",
            "owner-local ranking compute energy is not charged",
            "future validation uses a frozen open-loop movement-action tape",
            "split-conformal coverage requires episode exchangeability",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.audit,
        args.calibration,
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
        "confirmation": result["confirmation"],
        "incremental_route": result["incremental_route"],
    }, indent=2))


if __name__ == "__main__":
    main()
