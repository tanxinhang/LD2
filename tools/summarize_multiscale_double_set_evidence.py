#!/usr/bin/env python
"""Summarize cardinality-transfer evidence without merging proof levels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k8-transfer", type=Path, required=True)
    parser.add_argument("--k8-v1-transfer", type=Path, required=True)
    parser.add_argument("--k6-repair", type=Path, required=True)
    parser.add_argument("--k6-paired", type=Path, required=True)
    parser.add_argument("--historical-k6-repair", type=Path, required=True)
    parser.add_argument("--historical-k6-paired", type=Path, required=True)
    parser.add_argument("--k8-calibration", type=Path, required=True)
    parser.add_argument("--k8-repair", type=Path, required=True)
    parser.add_argument("--k8-paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = _load(args.training_report)
    transfer = _load(args.k8_transfer)
    v1_transfer = _load(args.k8_v1_transfer)
    repair = _load(args.k6_repair)
    paired = _load(args.k6_paired)
    historical_repair = _load(args.historical_k6_repair)
    historical_paired = _load(args.historical_k6_paired)
    k8_calibration = _load(args.k8_calibration)
    k8_repair = _load(args.k8_repair)
    k8_paired = _load(args.k8_paired)
    metadata = training["metadata"]
    domains = training["domains"]
    checkpoint_sha256 = _sha256(args.checkpoint)
    if checkpoint_sha256 != str(transfer["checkpoint_sha256"]):
        raise ValueError("8/8 transfer used a different checkpoint")
    runtime_checkpoint = Path(str(repair["causal_plan_checkpoint"]))
    runtime_checkpoint_sha256 = _sha256(runtime_checkpoint)
    if runtime_checkpoint_sha256 != str(
        repair["causal_plan_checkpoint_sha256"]
    ):
        raise ValueError("6/6 runtime checkpoint fingerprint is invalid")
    if not bool(training["all_domains_selection_admitted"]):
        raise ValueError("multi-scale development admission failed")
    if not bool(transfer["transfer_admitted"]):
        raise ValueError("independent 8/8 forecast transfer failed")
    if not bool(paired["strict_paired_improvement"]):
        raise ValueError("6/6 strict physical pair failed")
    if not bool(k8_calibration["envelope_calibration_ready"]):
        raise ValueError("8/8 coefficient-envelope calibration failed")
    if not bool(k8_paired["strict_paired_improvement"]):
        raise ValueError("8/8 strict physical pair failed")
    if str(k8_repair["causal_plan_checkpoint_sha256"]) != checkpoint_sha256:
        raise ValueError("8/8 physical audit used a different checkpoint")
    if str(k8_repair["causal_plan_architecture"]) != str(
        metadata["architecture"]
    ):
        raise ValueError("8/8 physical audit used a different architecture")
    if Path(str(k8_repair["calibration"])).resolve() != (
        args.k8_calibration.resolve()
    ):
        raise ValueError("8/8 physical audit used a different calibration")
    if not bool(k8_repair["physical_design_matches_calibration"]):
        raise ValueError("8/8 physical design does not match calibration")
    if int(k8_repair["calibration_validation_seed_overlap_count"]) != 0:
        raise ValueError("8/8 audit reports calibration/test leakage")
    calibration_seeds = set(k8_calibration["episode_scores"])
    test_seeds = set(k8_repair["episode_scores"])
    if calibration_seeds & test_seeds:
        raise ValueError("8/8 calibration and test episode seeds overlap")

    current_bits = float(repair["geometry_shadow_mean_transport_bits"])
    historical_bits = float(
        historical_repair["geometry_shadow_mean_transport_bits"])
    current_absolute = float(repair["policy_mean_worst"])
    historical_absolute = float(historical_repair["policy_mean_worst"])
    current_pair = float(paired["paired_policy_mean_worst_delta"])
    historical_pair = float(
        historical_paired["paired_policy_mean_worst_delta"])
    result: dict[str, object] = {
        "schema_version": 2,
        "status": (
            "cross_cardinality_forecast_and_layered_k6_k8_atomic_physics_"
            "passed_arbitrary_cardinality_open"),
        "architecture": metadata["architecture"],
        "forecast_checkpoint": str(args.checkpoint),
        "forecast_checkpoint_sha256": checkpoint_sha256,
        "forecast_checkpoint_bytes": int(args.checkpoint.stat().st_size),
        "physical_runtime_checkpoint": str(runtime_checkpoint),
        "physical_runtime_checkpoint_sha256": runtime_checkpoint_sha256,
        "physical_runtime_checkpoint_parameter_count": int(
            repair["causal_plan_parameter_count"]),
        "evidence_levels": {
            "k4q4_development_holdout_forecast": {
                "admitted": bool(domains["k4q4"]["selection_admitted"]),
                "relative_composite_improvement": float(
                    domains["k4q4"]["relative_composite_improvement"]),
                "action_mae_baseline_m": float(
                    domains["k4q4"]["baseline"]["action_mae_m"]),
                "action_mae_learned_m": float(
                    domains["k4q4"]["learned"]["action_mae_m"]),
                "endpoint_mae_baseline_m": float(
                    domains["k4q4"]["baseline"]["endpoint_mae_m"]),
                "endpoint_mae_learned_m": float(
                    domains["k4q4"]["learned"]["endpoint_mae_m"]),
            },
            "k6q6_development_holdout_forecast": {
                "admitted": bool(domains["k6q6"]["selection_admitted"]),
                "relative_composite_improvement": float(
                    domains["k6q6"]["relative_composite_improvement"]),
                "action_mae_baseline_m": float(
                    domains["k6q6"]["baseline"]["action_mae_m"]),
                "action_mae_learned_m": float(
                    domains["k6q6"]["learned"]["action_mae_m"]),
                "endpoint_mae_baseline_m": float(
                    domains["k6q6"]["baseline"]["endpoint_mae_m"]),
                "endpoint_mae_learned_m": float(
                    domains["k6q6"]["learned"]["endpoint_mae_m"]),
            },
            "k8q8_independent_zero_shot_forecast": {
                "admitted": bool(transfer["transfer_admitted"]),
                "scope": transfer["scope"],
                "episode_count": int(transfer["independent_episode_count"]),
                "evaluated_horizon_count": int(
                    transfer["evaluated_horizon_count"]),
                "relative_composite_improvement": float(
                    transfer["relative_composite_improvement"]),
                "action_mae_baseline_m": float(
                    transfer["baseline"]["action_mae_m"]),
                "action_mae_learned_m": float(
                    transfer["learned"]["action_mae_m"]),
                "endpoint_mae_baseline_m": float(
                    transfer["baseline"]["endpoint_mae_m"]),
                "endpoint_mae_learned_m": float(
                    transfer["learned"]["endpoint_mae_m"]),
                "uniform_region_scale": float(
                    transfer["uniform_region_scale"]),
                "speed_disk_satisfied": bool(
                    transfer["speed_disk_satisfied"]),
            },
            "k6q6_independent_atomic_physical": {
                "strict_paired_improvement": True,
                "seed_count": int(paired["seed_count"]),
                "event_count": int(paired["event_count"]),
                "accepted_commitment_count": int(
                    paired["accepted_commitment_count"]),
                "policy_mean_worst": current_absolute,
                "policy_qos_rate": float(repair["policy_qos_rate"]),
                "paired_policy_mean_worst_delta": current_pair,
                "paired_policy_qos_rate_delta": float(
                    paired["paired_policy_qos_rate_delta"]),
                "negative_seed_count": int(paired["negative_seed_count"]),
                "positive_tube_window_count": int(
                    paired["positive_tube_window_count"]),
                "nonpositive_tube_window_count": int(
                    paired["nonpositive_tube_window_count"]),
                "minimum_tube_window_delta": float(
                    paired["minimum_tube_window_delta"]),
                "envelope_coverage_rate": float(
                    paired["repair_envelope_coverage_rate"]),
                "all_protocol_feasible": bool(
                    paired["repair_all_protocol_feasible"]),
            },
            "k8q8_independent_envelope_calibration": {
                "path": str(args.k8_calibration),
                "sha256": _sha256(args.k8_calibration),
                "calibration_episode_count": int(
                    k8_calibration["calibration_episode_count"]),
                "future_event_count": int(
                    k8_calibration["future_event_count"]),
                "audited_coefficient_count": int(
                    k8_calibration["audited_coefficient_count"]),
                "target_coverage": float(k8_calibration["target_coverage"]),
                "finite_sample_coverage_floor": float(
                    k8_calibration["finite_sample_coverage_floor"]),
                "frozen_transition_residual_log_margin": float(
                    k8_calibration[
                        "frozen_transition_residual_log_margin"]),
                "test_seed_overlap_count": len(calibration_seeds & test_seeds),
            },
            "k8q8_independent_atomic_physical": {
                "repair_summary": str(args.k8_repair),
                "repair_summary_sha256": _sha256(args.k8_repair),
                "paired_certificate": str(args.k8_paired),
                "paired_certificate_sha256": _sha256(args.k8_paired),
                "strict_paired_improvement": True,
                "seed_count": int(k8_paired["seed_count"]),
                "event_count": int(k8_paired["event_count"]),
                "accepted_commitment_count": int(
                    k8_paired["accepted_commitment_count"]),
                "policy_mean_worst": float(k8_repair["policy_mean_worst"]),
                "policy_qos_rate": float(k8_repair["policy_qos_rate"]),
                "paired_policy_mean_worst_delta": float(
                    k8_paired["paired_policy_mean_worst_delta"]),
                "paired_policy_qos_rate_delta": float(
                    k8_paired["paired_policy_qos_rate_delta"]),
                "positive_seed_count": int(
                    k8_paired["positive_seed_count"]),
                "negative_seed_count": int(
                    k8_paired["negative_seed_count"]),
                "positive_tube_window_count": int(
                    k8_paired["positive_tube_window_count"]),
                "nonpositive_tube_window_count": int(
                    k8_paired["nonpositive_tube_window_count"]),
                "minimum_tube_window_delta": float(
                    k8_paired["minimum_tube_window_delta"]),
                "envelope_coverage_rate": float(
                    k8_paired["repair_envelope_coverage_rate"]),
                "all_protocol_feasible": bool(
                    k8_paired["repair_all_protocol_feasible"]),
                "mean_geometry_transport_bits": float(
                    k8_paired["repair_mean_geometry_transport_bits"]),
                "max_geometry_transport_latency_s": float(
                    k8_paired["repair_max_geometry_transport_latency_s"]),
                "uniform_region_scale": float(
                    k8_repair["causal_plan_uniform_region_scale"]),
                "runtime_planner_architecture": str(
                    k8_repair["causal_plan_architecture"]),
                "runtime_planner_checkpoint_sha256": str(
                    k8_repair["causal_plan_checkpoint_sha256"]),
            },
        },
        "architecture_comparisons": {
            "v1_to_v2_k8_relative_composite_improvement_delta": float(
                transfer["relative_composite_improvement"]
                - v1_transfer["relative_composite_improvement"]),
            "k6_absolute_policy_mean_worst_delta_vs_d078": float(
                current_absolute - historical_absolute),
            "k6_absolute_policy_qos_rate_delta_vs_d078": float(
                repair["policy_qos_rate"]
                - historical_repair["policy_qos_rate"]),
            "k6_paired_gain_ratio_vs_d078": float(
                current_pair / historical_pair),
            "geometry_transport_mean_bits_reduction_fraction_vs_d078": float(
                1.0 - current_bits / historical_bits),
            "geometry_transport_max_latency_reduction_s_vs_d078": float(
                historical_repair["geometry_shadow_max_transport_latency_s"]
                - repair["geometry_shadow_max_transport_latency_s"]),
            "runtime_planner_architecture": str(
                repair["causal_plan_architecture"]),
            "runtime_planner_checkpoint_sha256": str(
                repair["causal_plan_checkpoint_sha256"]),
            "runtime_candidate_ranking": (
                "all physically feasible analytic and learned candidates are "
                "ranked by certified H-window global-worst gain, certified "
                "first-step global-worst gain and certified per-target "
                "window gain; source type is only a tie-breaker"
            ),
        },
        "claim_boundary": {
            "cross_cardinality_movement_forecast_supported": True,
            "k6q6_atomic_physical_certificate_supported": True,
            "k8q8_atomic_physical_certificate_supported": True,
            "universal_kq_physical_theorem_supported": False,
            "reason": (
                "6/6 and 8/8 now have layered independent atomic physical "
                "certificates, but only finite cardinalities, static targets, "
                "fixed RCS, deterministic links and the audited region scales "
                "are covered; arbitrary K/Q remains unproved"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
