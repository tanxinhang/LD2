#!/usr/bin/env python
"""Cross-split audit of the target-wise self-normalized ISAC gate."""

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
    split_conformal_upper,
)
from uav_isac.evaluation.self_normalized_feedback import (  # noqa: E402
    certified_targetwise_feedback_decision,
    episode_max_normalized_scores,
)


def _paths(value: Path | Sequence[Path]) -> list[Path]:
    return [value] if isinstance(value, Path) else [Path(path) for path in value]


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
    candidate_prefix: str = "causal_routed",
) -> dict[str, object]:
    calibration_paths = _paths(calibration_path)
    if not calibration_paths:
        raise ValueError("at least one calibration report is required")
    calibrations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in calibration_paths
    ]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    calibration_rows: list[dict[str, object]] = []
    calibration_seeds: set[int] = set()
    for report in calibrations:
        report_seeds = {int(seed) for seed in report["seed_order"]}
        if calibration_seeds & report_seeds:
            raise ValueError("calibration reports contain overlapping episodes")
        calibration_seeds.update(report_seeds)
        calibration_rows.extend(report["rows"])
    validation_seeds = {int(seed) for seed in validation["seed_order"]}
    if calibration_seeds & validation_seeds:
        raise ValueError("calibration and validation episodes must be disjoint")
    if not calibration_seeds.issubset({
        int(row["seed"]) for row in calibration_rows
    }):
        raise ValueError("calibration report is missing episode rows")

    prefix = str(candidate_prefix)
    episode_scores = episode_max_normalized_scores(
        calibration_rows, candidate_prefix=prefix)
    # An episode with no routed candidate has an empty error set, hence score
    # zero.  Missing episode rows were rejected above and are not imputed.
    for seed in calibration_seeds:
        episode_scores.setdefault(seed, 0.0)
    margin, coverage_floor, rank = split_conformal_upper(
        episode_scores.values(), alpha=float(alpha))

    records: list[dict[str, object]] = []
    for row in validation["rows"]:
        if not bool(row.get(f"{prefix}_available", False)):
            continue
        switch_cost = float(switch_weight) * float(
            row.get(f"{prefix}_power_total_variation", 0.0))
        event_bits = int(row.get("transport_total_bits", payload_bits))
        max_packet_delay = float(
            row.get("transport_max_packet_latency_s", 0.0))
        protocol_delay = float(row.get(
            "transport_total_protocol_latency_s", max_packet_delay))
        event_energy = float(row.get("transport_total_energy_j", 0.0))
        decision = certified_targetwise_feedback_decision(
            candidate_estimated=np.asarray(
                row[f"{prefix}_estimated_pd"], dtype=np.float64),
            noop_estimated=np.asarray(
                row[f"{prefix}_noop_estimated_pd"], dtype=np.float64),
            absolute_uncertainty=np.asarray(
                row[f"{prefix}_absolute_uncertainty"], dtype=np.float64),
            noop_upper_uncertainty=np.asarray(
                row[f"{prefix}_noop_upper_uncertainty"], dtype=np.float64),
            normalized_margin=float(margin),
            qos_floor=float(qos_floor),
            switch_cost=switch_cost,
            bit_cost=float(bit_weight) * event_bits,
            delay_cost=float(delay_weight) * protocol_delay,
            energy_cost=float(energy_weight) * event_energy,
        )
        transport_feasible = bool(row.get("transport_feasible", True))
        accept = bool(decision.accept and transport_feasible)
        candidate_true = np.asarray(
            row[f"{prefix}_realized_pd"], dtype=np.float64)
        noop_true = np.asarray(row["deployed_pd"], dtype=np.float64)
        candidate_worst = float(np.min(candidate_true))
        noop_worst = float(np.min(noop_true))
        realized_worst_delta = candidate_worst - noop_worst
        bound_failure = bool(
            np.any(candidate_true + 1.0e-12 < decision.candidate_lower)
            or np.any(noop_true + 1.0e-12 < decision.noop_lower)
            or realized_worst_delta + 1.0e-12
            < decision.worst_delta_lower
        )
        # The stated lower-bound constraint preserves the certified service
        # level.  Realized per-target no-harm is stricter and is reported as a
        # diagnostic rather than silently conflated with that guarantee.
        certified_level_violation = bool(
            accept and np.any(
                candidate_true + 1.0e-12
                < np.minimum(decision.noop_lower, float(qos_floor))))
        realized_target_no_harm_violation = bool(
            accept and np.any(
                candidate_true + 1.0e-12
                < np.minimum(noop_true, float(qos_floor))))
        net_gain_violation = bool(
            accept and realized_worst_delta
            <= decision.charged_cost + 1.0e-12)
        records.append({
            "seed": int(row["seed"]),
            "frame": int(row["frame"]),
            "accept": accept,
            "transport_feasible": transport_feasible,
            "candidate_true_pd": candidate_true.tolist(),
            "noop_true_pd": noop_true.tolist(),
            "candidate_true_worst": candidate_worst,
            "noop_true_worst": noop_worst,
            "policy_true_worst": candidate_worst if accept else noop_worst,
            "candidate_lower": decision.candidate_lower.tolist(),
            "noop_lower": decision.noop_lower.tolist(),
            "noop_upper": decision.noop_upper.tolist(),
            "target_safe": decision.target_safe.tolist(),
            "worst_delta_lower": float(decision.worst_delta_lower),
            "realized_worst_delta": realized_worst_delta,
            "charged_cost": float(decision.charged_cost),
            "switch_cost": switch_cost,
            "bit_cost": float(bit_weight) * event_bits,
            "delay_cost": float(delay_weight) * protocol_delay,
            "energy_cost": float(energy_weight) * event_energy,
            "transport_total_bits": event_bits,
            "transport_max_packet_latency_s": max_packet_delay,
            "transport_decision_latency_s": protocol_delay,
            "transport_total_energy_j": event_energy,
            "simultaneous_bound_failure": bool(accept and bound_failure),
            "certified_level_violation": certified_level_violation,
            "realized_target_no_harm_violation": (
                realized_target_no_harm_violation),
            "net_gain_violation": net_gain_violation,
        })

    accepted = [record for record in records if bool(record["accept"])]
    lookup = {
        (int(record["seed"]), int(record["frame"])): record
        for record in records
    }
    overall_policy: list[float] = []
    overall_noop: list[float] = []
    for row in validation["rows"]:
        if "deployed_worst" not in row:
            continue
        noop_worst = float(row["deployed_worst"])
        record = lookup.get((int(row["seed"]), int(row["frame"])))
        overall_noop.append(noop_worst)
        overall_policy.append(float(
            record["policy_true_worst"] if record is not None else noop_worst))

    def episode_rate(key: str) -> tuple[float, dict[str, bool]]:
        flags = {
            str(seed): any(
                bool(record[key]) for record in records
                if int(record["seed"]) == seed)
            for seed in sorted(validation_seeds)
        }
        return float(np.mean(list(flags.values()))), flags

    bound_rate, episode_bound = episode_rate("simultaneous_bound_failure")
    level_rate, episode_level = episode_rate("certified_level_violation")
    target_rate, episode_target = episode_rate(
        "realized_target_no_harm_violation")
    net_rate, episode_net = episode_rate("net_gain_violation")
    return {
        "schema_version": 1,
        "scope": (
            f"{len(episode_scores)} calibration episodes to "
            f"{len(validation_seeds)} disjoint development episodes; "
            "one joint maximum score per episode and all targets"),
        "candidate_prefix": prefix,
        "calibration_reports": [str(path) for path in calibration_paths],
        "validation_report": str(validation_path),
        "alpha": float(alpha),
        "calibration_episode_count": len(episode_scores),
        "validation_episode_count": len(validation_seeds),
        "conformal_rank_one_based": int(rank),
        "finite_sample_episode_coverage_floor": float(coverage_floor),
        "normalized_margin": float(margin),
        "calibration_episode_scores": {
            str(seed): float(score) for seed, score in episode_scores.items()
        },
        "event_count": len(records),
        "accepted_event_count": len(accepted),
        "event_accept_rate": float(np.mean([
            bool(record["accept"]) for record in records])) if records else 0.0,
        "overall_validation_event_count": len(overall_policy),
        "overall_policy_event_mean_worst": float(np.mean(overall_policy)),
        "overall_noop_event_mean_worst": float(np.mean(overall_noop)),
        "overall_policy_event_qos_rate": float(np.mean(
            np.asarray(overall_policy) >= float(qos_floor))),
        "overall_noop_event_qos_rate": float(np.mean(
            np.asarray(overall_noop) >= float(qos_floor))),
        "accepted_simultaneous_bound_failure_rate": float(np.mean([
            bool(record["simultaneous_bound_failure"])
            for record in accepted])) if accepted else 0.0,
        "accepted_certified_level_violation_rate": float(np.mean([
            bool(record["certified_level_violation"])
            for record in accepted])) if accepted else 0.0,
        "accepted_realized_target_no_harm_violation_rate": float(np.mean([
            bool(record["realized_target_no_harm_violation"])
            for record in accepted])) if accepted else 0.0,
        "accepted_net_gain_violation_rate": float(np.mean([
            bool(record["net_gain_violation"])
            for record in accepted])) if accepted else 0.0,
        "episode_any_simultaneous_bound_failure_rate": bound_rate,
        "episode_any_certified_level_violation_rate": level_rate,
        "episode_any_realized_target_no_harm_violation_rate": target_rate,
        "episode_any_net_gain_violation_rate": net_rate,
        "episode_bound_failure": episode_bound,
        "episode_certified_level_violation": episode_level,
        "episode_realized_target_no_harm_violation": episode_target,
        "episode_net_gain_violation": episode_net,
        "cost_model": {
            "switch_weight": float(switch_weight),
            "bit_weight": float(bit_weight),
            "fallback_payload_bits_per_resolve": int(payload_bits),
            "delay_weight": float(delay_weight),
            "energy_weight": float(energy_weight),
            "full_protocol_delay_charged": True,
            "headers_and_contention_bits_charged": True,
            "retransmission_included": False,
        },
        "certificate_ready": False,
        "certificate_blockers": [
            "packet loss and link-model residual risk are not calibrated",
            "slow-geometry actions are routed but not implemented",
            "the current replay does not include Swerling/model-shift residuals",
        ],
        "fresh_test_consumed": False,
        "records": records,
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
    parser.add_argument("--candidate-prefix", default="causal_routed")
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
    keys = (
        "calibration_episode_count",
        "validation_episode_count",
        "finite_sample_episode_coverage_floor",
        "normalized_margin",
        "event_accept_rate",
        "overall_policy_event_mean_worst",
        "overall_noop_event_mean_worst",
        "overall_policy_event_qos_rate",
        "accepted_simultaneous_bound_failure_rate",
        "accepted_certified_level_violation_rate",
        "accepted_realized_target_no_harm_violation_rate",
        "accepted_net_gain_violation_rate",
        "certificate_ready",
    )
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
