#!/usr/bin/env python
"""Run staged scale and maneuver screens for the strict distributed pilot.

The scale screen keeps the 1130 m square fixed so a changed K or Q measures
resource/load scaling rather than a simultaneous density change.  The motion
screen keeps K=Q=8 and pins every local tracker to CV: CV is matched, while CT
and CA are explicit maneuver-model mismatch tests.  These are diagnostic
screens; formal claims require a frozen independent seed bank per condition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from tools.run_strict_distributed_pilot import (
    DEFAULT_CARRIER_PERIOD,
    _algorithm_version,
    _episode,
    _execution_audit,
    validate_formal_system_identity,
    validate_strict_config,
)
from uav_isac.utils.reproducibility import (
    build_run_manifest,
    validate_formal_run,
)


@dataclass(frozen=True)
class SweepCase:
    name: str
    family: str
    K: int = 8
    Q: int = 8
    speed_min: float = 0.0
    speed_max: float = 5.0
    motion_model: str = "CV"
    belief_model: str = "CV"


def scale_cases(sizes: tuple[int, ...]) -> list[SweepCase]:
    cases = [SweepCase(name="k8_q8", family="scale_baseline")]
    for size in sizes:
        if size <= 8:
            continue
        cases.extend((
            SweepCase(
                name=f"k8_q{size}", family="fixed_uav_more_targets", Q=size),
            SweepCase(
                name=f"k{size}_q8", family="more_uav_fixed_targets", K=size),
            SweepCase(
                name=f"k{size}_q{size}", family="both_increase", K=size, Q=size),
        ))
    return cases


def motion_cases() -> list[SweepCase]:
    bands = ((0.0, 5.0), (5.0, 10.0), (10.0, 20.0))
    return [
        SweepCase(
            name=f"{model.lower()}_{int(lo)}_{int(hi)}mps",
            family="motion",
            speed_min=lo,
            speed_max=hi,
            motion_model=model,
            belief_model="CV",
        )
        for model in ("CV", "CT", "CA")
        for lo, hi in bands
    ]


def _configure(
    base_config: str,
    case: SweepCase,
    frames: int,
    movement_mode: str,
    reuse_relative_tolerance: float = 0.0,
) -> Any:
    cfg = load_config(base_config)
    cfg.scenario.K = int(case.K)
    cfg.scenario.Q = int(case.Q)
    cfg.scenario.T = int(frames)
    cfg.target.speed_range = (
        float(case.speed_min), float(case.speed_max))
    cfg.target.motion_model = str(case.motion_model)
    cfg.target.omega_q = [1.0 / float(case.Q)] * int(case.Q)
    cfg.marl.belief_motion_model = str(case.belief_model)
    # Online latency excludes training-only fixed-assignment counterfactual
    # rewards.  They do not feed the policy or physical execution and otherwise
    # add K+1 full deflection evaluations to every measured control frame.
    cfg.marl.use_difference_reward = False
    cfg.marl.distributed_replicated_power_reuse_relative_tolerance = float(
        reuse_relative_tolerance)
    cfg.marl.distributed_bottleneck_matching_movement_enabled = bool(
        movement_mode == "bottleneck")
    cfg.marl.distributed_bistatic_bottleneck_movement_enabled = bool(
        movement_mode == "bistatic_bottleneck")
    cfg.marl.distributed_role_capacity_movement_enabled = bool(
        movement_mode == "role_capacity")
    if movement_mode == "role_capacity":
        cfg.marl.distributed_movement_safety_projection_enabled = True
        cfg.marl.distributed_movement_execute_projected_public_action = True
    validate_strict_config(cfg)
    return cfg


def _aggregate(episodes: list[dict[str, Any]], case: SweepCase) -> dict[str, Any]:
    mean = lambda key: float(np.mean([episode[key] for episode in episodes]))
    mean_optional = lambda key, default: float(np.mean([
        episode.get(key, default) for episode in episodes
    ]))
    finite_certificate_upper = np.asarray([
        episode.get("certificate_global_optimum_upper")
        for episode in episodes
        if episode.get("certificate_global_optimum_upper") is not None
    ], dtype=np.float64)
    finite_certificate_upper = finite_certificate_upper[
        np.isfinite(finite_certificate_upper)]
    bits = mean("bits_per_frame")
    result = {
        "case": case.name,
        "family": case.family,
        "K": case.K,
        "Q": case.Q,
        "speed_range_mps": [case.speed_min, case.speed_max],
        "target_motion_model": case.motion_model,
        "belief_motion_model": case.belief_model,
        "qos_rate": mean("qos_success"),
        "steady": mean("steady"),
        "weak3": mean("weak3"),
        "worst": mean("worst"),
        "bits_per_frame": bits,
        "bits_per_uav_frame": bits / float(case.K),
        "bits_per_target_frame": bits / float(case.Q),
        "active_senders_per_frame": mean("active_senders_per_frame"),
        "delivery_rate": mean("delivery_rate"),
        "deadline_violation_rate": mean("deadline_violation_rate"),
        "hyperedge_coverage": mean("hyperedge_coverage"),
        "movement_target_coverage": mean("movement_target_coverage"),
        "belief_position_rmse_m": mean("belief_position_rmse_m"),
        "certificate_complete_fraction": mean_optional(
            "certificate_complete_fraction", 0.0),
        "certificate_worst_pd_lower": mean_optional(
            "certificate_worst_pd_lower", 0.0),
        "certificate_global_optimum_upper": (
            float(np.mean(finite_certificate_upper))
            if finite_certificate_upper.size else None),
        "certificate_global_optimum_upper_unavailable_reason": (
            None if finite_certificate_upper.size else
            "no finite global certificate upper bound was produced"),
        "certificate_joint_approximation_ratio_lower": mean_optional(
            "certificate_joint_approximation_ratio_lower", 0.0),
        "certificate_payload_bits_per_sender": mean_optional(
            "certificate_payload_bits_per_sender", 0.0),
        "owner_posterior_payload_bits_per_frame": mean_optional(
            "owner_posterior_payload_bits_per_frame", 0.0),
        "owner_posterior_fused_entries_per_frame": mean_optional(
            "owner_posterior_fused_entries_per_frame", 0.0),
        "owner_posterior_cov_trace_ratio": mean_optional(
            "owner_posterior_cov_trace_ratio", 1.0),
        "power_resolve_fraction": mean_optional(
            "power_resolve_fraction", 1.0),
        "power_reuse_fraction": mean_optional(
            "power_reuse_fraction", 0.0),
        "power_solve_time_ms": mean_optional(
            "power_solve_time_ms", 0.0),
        "power_parallel_critical_path_ms": mean_optional(
            "power_parallel_critical_path_ms", 0.0),
        "controller_compute_critical_path_ms": mean_optional(
            "controller_compute_critical_path_ms", 0.0),
        "controller_compute_critical_path_p95_ms": mean_optional(
            "controller_compute_critical_path_p95_ms", 0.0),
        "radio_critical_path_ms": mean_optional(
            "radio_critical_path_ms", 0.0),
        "closed_loop_critical_path_ms": mean_optional(
            "closed_loop_critical_path_ms", 0.0),
        "closed_loop_critical_path_p95_ms": mean_optional(
            "closed_loop_critical_path_p95_ms", 0.0),
        "deployment_step_time_estimate_ms": mean_optional(
            "deployment_step_time_estimate_ms",
            mean("step_time_ms"),
        ),
        "step_time_ms": mean("step_time_ms"),
        "step_time_p95_ms": mean("step_time_p95_ms"),
        "step_time_max_ms": mean("step_time_max_ms"),
        "online_budget_ms": mean("online_budget_ms"),
        "online_deadline_miss_rate": mean("online_deadline_miss_rate"),
        "episodes": episodes,
    }
    gate = {
        "qos_rate_ge_0_80": result["qos_rate"] >= 0.80,
        "delivery_rate_ge_0_99": result["delivery_rate"] >= 0.99,
        "radio_deadline_violation_rate_le_0_01": (
            result["deadline_violation_rate"] <= 0.01),
        "online_deadline_miss_rate_le_0_01": (
            result["online_deadline_miss_rate"] <= 0.01),
        "p95_closed_loop_critical_path_within_control_period": (
            result["closed_loop_critical_path_p95_ms"]
            <= result["online_budget_ms"]),
    }
    result["hard_and_service_gates"] = gate
    result["passes_all_gates"] = bool(all(gate.values()))
    return result


def run_sweep(
    config: str,
    cases: list[SweepCase],
    seeds: list[int],
    frames: int,
    tail_window: int,
    carrier_period: int,
    movement_mode: str = "bottleneck",
    reuse_relative_tolerance: float = 0.0,
    formal: bool = False,
) -> dict[str, Any]:
    if formal:
        validate_formal_system_identity(config)
    results = []
    for case in cases:
        cfg = _configure(
            config,
            case,
            frames,
            movement_mode,
            reuse_relative_tolerance=float(reuse_relative_tolerance),
        )
        manifest = build_run_manifest(
            cfg,
            config_path=config,
            seeds=seeds,
            algorithm_version=_algorithm_version(cfg),
            root=ROOT,
        )
        if formal:
            validate_formal_run(cfg, seeds, manifest)
        modules_before_episode = set(sys.modules)
        episodes = [
            _episode(cfg, seed, tail_window, carrier_period)
            for seed in seeds
        ]
        execution_audit = _execution_audit(cfg, modules_before_episode)
        manifest["execution_audit"] = execution_audit
        case_result = _aggregate(episodes, case)
        case_result["run_manifest"] = manifest
        case_result["execution_audit"] = execution_audit
        results.append(case_result)
    return {
        "config": os.path.normpath(config),
        "status": "FORMAL_COMPLETE" if formal else "DIAGNOSTIC_ONLY",
        "region_policy": "fixed_1130m_square",
        "carrier_period": int(carrier_period),
        "movement_mode": movement_mode,
        "power_reuse_relative_tolerance": float(
            reuse_relative_tolerance),
        "frames": int(frames),
        "seeds": [int(seed) for seed in seeds],
        "evaluation_protocol": {
            "execution_gate": "hard physical and deadline feasibility only",
            "performance_certificate": "not required online",
            "exact_optimality": "not run at scale; small-scale audit only",
            "service_thresholds": {
                "qos_seed_rate": 0.80,
                "delivery_rate": 0.99,
                "radio_deadline_violation_rate": 0.01,
                "online_deadline_miss_rate": 0.01,
            },
        },
        "passes_all_cases": bool(all(
            case_result["passes_all_gates"] for case_result in results)),
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    from uav_isac.governance import assert_operation_allowed
    assert_operation_allowed("full_result_refresh")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/exp_strict_distributed_no_truth_pilot.yaml")
    parser.add_argument("--suite", choices=("scale", "motion", "all"),
                        default="all")
    parser.add_argument("--sizes", default="8,10,12")
    parser.add_argument(
        "--case-names", default="",
        help="optional comma-separated exact case names after suite expansion")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--tail-window", type=int, default=10)
    parser.add_argument("--carrier-period", type=int,
                        default=DEFAULT_CARRIER_PERIOD)
    parser.add_argument(
        "--movement-mode",
        choices=("bottleneck", "bistatic_bottleneck", "role_capacity"),
        default="bottleneck")
    parser.add_argument(
        "--power-reuse-relative-tolerance", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--formal", action="store_true",
        help="fail closed unless workspace, 100-seed bank and threads are frozen")
    args = parser.parse_args(argv)

    sizes = tuple(sorted({int(value) for value in args.sizes.split(",")}))
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not sizes or min(sizes) < 1:
        parser.error("--sizes must contain positive integers")
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    if args.frames < 1 or args.tail_window < 1 or args.carrier_period < 1:
        parser.error("frames, tail-window, and carrier-period must be positive")
    if not 0.0 <= args.power_reuse_relative_tolerance < 1.0:
        parser.error("power reuse relative tolerance must lie in [0,1)")

    cases: list[SweepCase] = []
    if args.suite in ("scale", "all"):
        cases.extend(scale_cases(sizes))
    if args.suite in ("motion", "all"):
        cases.extend(motion_cases())
    selected_names = {
        value.strip() for value in args.case_names.split(",") if value.strip()
    }
    if selected_names:
        cases = [case for case in cases if case.name in selected_names]
        missing = selected_names.difference(case.name for case in cases)
        if missing:
            parser.error("unknown --case-names: " + ", ".join(sorted(missing)))
    if not cases:
        parser.error("suite/case selection produced no cases")
    report = run_sweep(
        args.config, cases, seeds, args.frames,
        args.tail_window, args.carrier_period, args.movement_mode,
        args.power_reuse_relative_tolerance, args.formal)
    rendered = json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
