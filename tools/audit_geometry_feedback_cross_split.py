#!/usr/bin/env python
"""Calibrate on independent episodes and validate geometry feedback elsewhere."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.evaluation.episode_joint_conformal import (  # noqa: E402
    certified_feedback_decision,
    episode_max_scores,
    split_conformal_upper,
)


def audit(
    calibration_path: Path | Sequence[Path],
    validation_path: Path,
    *,
    alpha: float,
    qos_floor: float,
    switch_weight: float,
    bit_weight: float,
    payload_bits: int,
    delay_weight: float,
    energy_weight: float,
    candidate_prefix: str,
) -> dict[str, object]:
    calibration_paths = (
        [calibration_path]
        if isinstance(calibration_path, Path)
        else [Path(path) for path in calibration_path]
    )
    if not calibration_paths:
        raise ValueError("at least one calibration report is required")
    calibrations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in calibration_paths
    ]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    full_structure_transport = bool(
        all(report.get("structure_sequence_transport_enabled", False)
            for report in calibrations)
        and validation.get("structure_sequence_transport_enabled", False)
    )
    owner_proposal_transport = bool(
        all(report.get("owner_proposal_transport_enabled", False)
            for report in calibrations)
        and validation.get("owner_proposal_transport_enabled", False)
    )
    calibration_seeds: set[int] = set()
    calibration_rows = []
    for report in calibrations:
        report_seeds = {int(seed) for seed in report["seed_order"]}
        if calibration_seeds & report_seeds:
            raise ValueError("calibration reports contain overlapping episodes")
        calibration_seeds.update(report_seeds)
        calibration_rows.extend(report["rows"])
    validation_seeds = {int(seed) for seed in validation["seed_order"]}
    if calibration_seeds & validation_seeds:
        raise ValueError("calibration and validation episode IDs must be disjoint")
    prefix = str(candidate_prefix)
    episode_scores = episode_max_scores(
        calibration_rows, candidate_prefix=prefix)
    calibration_row_seeds = {
        int(row["seed"]) for row in calibration_rows
    }
    if not calibration_seeds.issubset(calibration_row_seeds):
        raise ValueError("calibration report is missing complete episode rows")
    # For the fixed routed policy, an episode with no proposed action has an
    # empty violation set and therefore the non-negative maximum score zero.
    # This is distinct from a missing episode, which still fails above.
    for seed in calibration_seeds:
        episode_scores.setdefault(seed, 0.0)
    margin, coverage_floor, rank = split_conformal_upper(
        episode_scores.values(), alpha=float(alpha))

    records = []
    for row in validation["rows"]:
        if not bool(row.get(f"{prefix}_available", False)):
            continue
        switch_cost = float(
            switch_weight * float(row.get(
                f"{prefix}_power_total_variation", 0.0)))
        event_bits = int(row.get("transport_total_bits", payload_bits))
        event_max_packet_delay = float(row.get(
            "transport_max_packet_latency_s", 0.0))
        event_delay = float(row.get(
            "transport_total_protocol_latency_s",
            event_max_packet_delay,
        ))
        event_energy = float(row.get("transport_total_energy_j", 0.0))
        bit_cost = float(bit_weight * event_bits)
        delay_cost = float(delay_weight * event_delay)
        energy_cost = float(energy_weight * event_energy)
        decision = certified_feedback_decision(
            candidate_estimated=float(
                row[f"{prefix}_estimated_worst"]),
            noop_estimated=float(
                row[f"{prefix}_noop_estimated_worst"]),
            joint_margin=margin,
            qos_floor=float(qos_floor),
            switch_cost=switch_cost,
            bit_cost=bit_cost,
            delay_cost=delay_cost,
            energy_cost=energy_cost,
        )
        transport_feasible = bool(row.get("transport_feasible", True))
        accept = bool(decision.accept and transport_feasible)
        candidate_true = float(row[f"{prefix}_worst"])
        noop_true = float(row["deployed_worst"])
        realized_delta = candidate_true - noop_true
        records.append({
            "seed": int(row["seed"]),
            "frame": int(row["frame"]),
            "accept": accept,
            "transport_feasible": transport_feasible,
            "candidate_true_worst": candidate_true,
            "noop_true_worst": noop_true,
            "policy_true_worst": (
                candidate_true if accept else noop_true),
            "candidate_lower": float(decision.candidate_lower),
            "noop_lower": float(decision.noop_lower),
            "delta_lower": float(decision.delta_lower),
            "charged_cost": float(decision.charged_cost),
            "switch_cost": switch_cost,
            "bit_cost": bit_cost,
            "delay_cost": delay_cost,
            "energy_cost": energy_cost,
            "transport_total_bits": event_bits,
            "transport_max_packet_latency_s": event_max_packet_delay,
            "transport_decision_latency_s": event_delay,
            "transport_total_energy_j": event_energy,
            "constraint_violation": bool(
                accept
                and candidate_true + 1.0e-12 < min(noop_true, qos_floor)
            ),
            "net_gain_violation": bool(
                accept
                and realized_delta <= decision.charged_cost + 1.0e-12
            ),
        })
    episode_violation = {}
    episode_net_violation = {}
    for seed in sorted(validation_seeds):
        selected = [row for row in records if int(row["seed"]) == seed]
        episode_violation[str(seed)] = bool(any(
            row["constraint_violation"] for row in selected))
        episode_net_violation[str(seed)] = bool(any(
            row["net_gain_violation"] for row in selected))
    accepted = [row for row in records if bool(row["accept"])]
    record_lookup = {
        (int(row["seed"]), int(row["frame"])): row for row in records
    }
    overall_policy = []
    overall_noop = []
    for row in validation["rows"]:
        if "deployed_worst" not in row:
            continue
        key = (int(row["seed"]), int(row["frame"]))
        decision_row = record_lookup.get(key)
        overall_policy.append(float(
            decision_row["policy_true_worst"]
            if decision_row is not None else row["deployed_worst"]))
        overall_noop.append(float(row["deployed_worst"]))
    return {
        "schema_version": 1,
        "scope": (
            f"{len(episode_scores)} disjoint training-pool calibration "
            f"episodes to {len(validation_seeds)} development validation "
            "episodes; one maximum score per episode; "
            f"candidate prefix {prefix}"
        ),
        "candidate_prefix": prefix,
        "calibration_report": (
            str(calibration_paths[0]) if len(calibration_paths) == 1 else None),
        "calibration_reports": [str(path) for path in calibration_paths],
        "validation_report": str(validation_path),
        "alpha": float(alpha),
        "calibration_episode_count": len(episode_scores),
        "validation_episode_count": len(validation_seeds),
        "conformal_rank_one_based": int(rank),
        "finite_sample_episode_coverage_floor": float(coverage_floor),
        "joint_margin": float(margin),
        "calibration_episode_scores": {
            str(seed): float(score) for seed, score in episode_scores.items()
        },
        "event_count": len(records),
        "event_accept_rate": float(np.mean([
            bool(row["accept"]) for row in records
        ])),
        "accepted_event_count": len(accepted),
        "policy_event_mean_worst": float(np.mean([
            float(row["policy_true_worst"]) for row in records
        ])),
        "noop_event_mean_worst": float(np.mean([
            float(row["noop_true_worst"]) for row in records
        ])),
        "candidate_event_mean_worst": float(np.mean([
            float(row["candidate_true_worst"]) for row in records
        ])),
        "overall_validation_event_count": len(overall_policy),
        "overall_policy_event_mean_worst": float(np.mean(overall_policy)),
        "overall_noop_event_mean_worst": float(np.mean(overall_noop)),
        "overall_policy_event_qos_rate": float(np.mean(
            np.asarray(overall_policy) >= float(qos_floor))),
        "accepted_constraint_violation_rate": float(np.mean([
            bool(row["constraint_violation"]) for row in accepted
        ])) if accepted else 0.0,
        "accepted_net_gain_violation_rate": float(np.mean([
            bool(row["net_gain_violation"]) for row in accepted
        ])) if accepted else 0.0,
        "episode_any_constraint_violation_rate": float(np.mean(
            list(episode_violation.values()))),
        "episode_any_net_gain_violation_rate": float(np.mean(
            list(episode_net_violation.values()))),
        "episode_constraint_violation": episode_violation,
        "episode_net_gain_violation": episode_net_violation,
        "cost_model": {
            "switch_weight": float(switch_weight),
            "bit_weight": float(bit_weight),
            "fallback_payload_bits_per_resolve": int(payload_bits),
            "delay_weight": float(delay_weight),
            "energy_weight": float(energy_weight),
            "event_specific_transport_cost_used": bool(any(
                "transport_total_bits" in row for row in validation["rows"])),
            "delay_cost_included": bool(any(
                "transport_total_protocol_latency_s" in row
                for row in validation["rows"])),
            "delay_cost_uses_full_decision_protocol": bool(any(
                "transport_total_protocol_latency_s" in row
                for row in validation["rows"])),
            "energy_cost_included": bool(any(
                "transport_total_energy_j" in row
                for row in validation["rows"])),
            "headers_and_orthogonal_feedback_contention_included": bool(any(
                "transport_total_bits" in row for row in validation["rows"])),
            "retransmission_included": False
        },
        "certificate_ready": False,
        "certificate_blockers": (
            ([
                "deterministic U2U replay has no empirical packet-loss residual",
                "the SNR and latency engineering margins are not statistically calibrated",
                "slow-geometry actions are routed but not implemented",
            ] + ([] if full_structure_transport else [
                "multi-step structural dependency-commit bits, delay and energy are not charged",
            ]) + ([] if owner_proposal_transport else [
                "owner proposal descriptors and ranking competition are not transported",
            ]) + ([
                "owner-local cache provenance from delivered tokens remains to be replayed",
            ] if owner_proposal_transport else [
            ]))
            if prefix == "causal_routed"
            else [
                "deterministic U2U replay has no empirical packet-loss residual",
                "the SNR and latency engineering margins are not statistically calibrated",
                "the current sensing model does not test Swerling/report-link residuals",
                "the structural graph remains the frozen trace structure",
            ]
        ),
        "fresh_test_consumed": False,
        "records": records
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, type=Path, nargs="+")
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument("--switch-weight", type=float, default=0.02)
    parser.add_argument("--bit-weight", type=float, default=1.0e-5)
    parser.add_argument("--payload-bits", type=int, default=624)
    parser.add_argument("--delay-weight", type=float, default=10.0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument(
        "--candidate-prefix", default="causal_geometry")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.calibration,
        args.validation,
        alpha=float(args.alpha),
        qos_floor=float(args.qos_floor),
        switch_weight=float(args.switch_weight),
        bit_weight=float(args.bit_weight),
        payload_bits=int(args.payload_bits),
        delay_weight=float(args.delay_weight),
        energy_weight=float(args.energy_weight),
        candidate_prefix=str(args.candidate_prefix),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "schema_version",
            "calibration_episode_count",
            "validation_episode_count",
            "finite_sample_episode_coverage_floor",
            "joint_margin",
            "event_accept_rate",
            "policy_event_mean_worst",
            "noop_event_mean_worst",
            "accepted_constraint_violation_rate",
            "accepted_net_gain_violation_rate",
            "episode_any_constraint_violation_rate",
            "episode_any_net_gain_violation_rate",
            "certificate_ready",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
