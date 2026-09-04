#!/usr/bin/env python
"""Run the deterministic strict-distributed analytical pilot.

This is a kernel/protocol baseline, not a learned-policy result.  Agents submit
hold actions while the strict local-belief hyperedge, physical U2U transport,
replicated L1 power, and distributed movement stack execute normally.  The
runner intentionally rejects configs that can fall back to centralized
structure or simulator truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.utils.reproducibility import (
    build_run_manifest,
    validate_formal_run,
)


DEFAULT_CARRIER_PERIOD = 3


def _algorithm_version(cfg: Any) -> str:
    """Return an identity that changes with executed algorithmic variants."""
    process_harmonic = (
        bool(getattr(
            cfg.marl,
            "distributed_replicated_power_process_parallel_enabled",
            False,
        ))
        and str(getattr(
            cfg.marl,
            "distributed_replicated_power_parallel_failure_mode",
            "serial",
        )).strip().lower() == "cached_or_harmonic"
    )
    if process_harmonic:
        version = (
            "strict-distributed-composable-owner-posterior-v5-"
            "ca6-process-deadline-sparse-harmonic"
            if str(getattr(
                cfg.marl,
                "belief_motion_model",
                "TARGET",
            )).strip().upper() == "CA"
            else (
                "strict-distributed-composable-owner-posterior-v4-"
                "process-deadline-sparse-harmonic"
            )
        )
    elif bool(getattr(
        cfg.marl,
        "distributed_replicated_power_process_parallel_enabled",
        False,
    )):
        version = (
            "strict-distributed-composable-owner-posterior-v3-"
            "process-parallel"
        )
    else:
        version = "strict-distributed-composable-owner-posterior-v2"
    robust_mix = float(getattr(
        cfg.marl, "distributed_replicated_power_robust_gain_mix", 0.0))
    if robust_mix > 0.0:
        encoded = f"{robust_mix:.6f}".rstrip("0").rstrip(".").replace(".", "p")
        version += f"-nominal-robust-riskmix-{encoded}"
    if bool(getattr(
        cfg.marl, "belief_expected_detection_information_enabled", False)):
        version += "-expected-detection-information"
    if str(getattr(
        cfg.marl, "belief_measurement_model", "cartesian"
    )).strip().lower() == "bistatic_range_doppler":
        version += "-bistatic-range-doppler-ekf"
    if bool(getattr(
        cfg.marl, "hyperedge_bistatic_information_ranking_enabled", False)):
        version += "-submodular-information-ranking"
    return version


def validate_strict_config(cfg: Any) -> None:
    """Reject a pilot that is no-truth in name but not in execution."""
    ma = cfg.marl
    required_true = {
        "tracking_enabled": ma.tracking_enabled,
        "joint_isac_power_enabled": ma.joint_isac_power_enabled,
        "distributed_coordination_use_local_belief_targets": (
            ma.distributed_coordination_use_local_belief_targets),
        "distributed_no_truth_fail_closed": ma.distributed_no_truth_fail_closed,
        "hyperedge_negotiation_enabled": ma.hyperedge_negotiation_enabled,
        "hyperedge_state_stream_enabled": ma.hyperedge_state_stream_enabled,
        "hyperedge_protocol_only_enabled": ma.hyperedge_protocol_only_enabled,
        "distributed_replicated_power_enabled": (
            ma.distributed_replicated_power_enabled),
        "distributed_composable_certificate_enabled": (
            ma.distributed_composable_certificate_enabled),
        "distributed_owner_posterior_enabled": (
            ma.distributed_owner_posterior_enabled),
        "analytical_sensing_power_enabled": ma.analytical_sensing_power_enabled,
    }
    missing = [name for name, value in required_true.items() if not bool(value)]
    if missing:
        raise ValueError(
            "strict distributed pilot requires enabled flags: "
            + ", ".join(missing))
    if bool(ma.ground_communication_enabled):
        raise ValueError("strict distributed pilot forbids ground communication")
    if str(ma.detection_fusion_mode) != "local_only":
        raise ValueError("strict distributed pilot requires local_only fusion")
    if str(ma.hyperedge_pair_score_mode) != "budget_reconstructable":
        raise ValueError(
            "strict distributed pilot requires budget_reconstructable hyperedges")
    if bool(ma.hyperedge_safety_fallback_enabled):
        raise ValueError("strict distributed pilot forbids centralized safety fallback")
    if bool(ma.use_difference_reward):
        raise ValueError(
            "strict distributed online pilot forbids training-only difference reward")


def validate_formal_system_identity(config_path: str) -> None:
    """Require a formal profile to satisfy the canonical executable identity."""
    # Keep diagnostic worker imports light; only a formal parent process needs
    # the repository-level identity gate.
    from tools.check_system_identity import collect_checks

    failures, _passes = collect_checks(config_path, strict=True)
    if failures:
        raise RuntimeError(
            "formal run system-identity gate failed: " + "; ".join(failures)
        )


def _episode(
    cfg: Any,
    seed: int,
    tail_window: int,
    carrier_period: int = DEFAULT_CARRIER_PERIOD,
    include_trace: bool = False,
    fault_inject_power_timeout_frame: int | None = None,
    fault_inject_power_timeout_duration: int = 1,
) -> dict[str, Any]:
    if int(carrier_period) < 1:
        raise ValueError("carrier_period must be positive")
    env = UAVISACEnv(config=cfg)
    observations, reset_info = env.reset(seed=int(seed))
    worker_warmup_ms = 1000.0 * float(reset_info.get(
        "distributed_replicated_power_worker_warmup_time_s", 0.0))
    detection: list[np.ndarray] = []
    bits: list[float] = []
    active_senders: list[float] = []
    delivery: list[float] = []
    deadline: list[float] = []
    coverage: list[float] = []
    movement_coverage: list[float] = []
    belief_rmse: list[float] = []
    power_resolve_fraction: list[float] = []
    power_reuse_fraction: list[float] = []
    power_solve_time_ms: list[float] = []
    power_parallel_critical_path_ms: list[float] = []
    power_process_parallel_used: list[float] = []
    power_process_batch_wall_ms: list[float] = []
    power_process_fallback: list[float] = []
    power_deadline_incumbent: list[float] = []
    power_deadline_incumbent_certified: list[float] = []
    power_deadline_uniform: list[float] = []
    power_deadline_harmonic: list[float] = []
    power_deadline_composable_floor: list[float] = []
    power_deadline_composable_pd_floor: list[float] = []
    power_deadline_safe_target_coverage: list[float] = []
    power_deadline_safe_min_contributors: list[float] = []
    power_deadline_harmonic_shadow_pd_floor: list[float] = []
    power_deadline_shadow_reserve_fraction: list[float] = []
    power_deadline_shadow_reserve_feasible: list[float] = []
    power_deadline_shadow_global_lp_pd_floor: list[float] = []
    power_deadline_shadow_global_lp_time_ms: list[float] = []
    power_deadline_shadow_harmonic_ratio: list[float] = []
    certificate_complete: list[float] = []
    certificate_worst_pd_lower: list[float] = []
    certificate_global_upper: list[float] = []
    certificate_approximation_ratio: list[float] = []
    certificate_payload_bits: list[float] = []
    owner_posterior_payload_bits: list[float] = []
    owner_posterior_fused_entries: list[float] = []
    owner_posterior_cov_trace_ratio: list[float] = []
    owner_posterior_packet_aoi: list[float] = []
    owner_posterior_state_dim: list[float] = []
    bistatic_information_gain: list[float] = []
    bistatic_full_rank_fraction: list[float] = []
    controller_compute_ms: list[float] = []
    radio_critical_path_ms: list[float] = []
    closed_loop_critical_path_ms: list[float] = []
    step_seconds: list[float] = []
    uav_position_trace: list[np.ndarray] = []
    sensing_power_trace: list[np.ndarray] = []
    certificate_safe_gain_trace: list[np.ndarray] = []
    try:
        for frame in range(int(cfg.scenario.T)):
            # The analytical pilot has no learned actor, but the distributed
            # control protocol still needs an explicit physical carrier.  A
            # zero-rate/zero-content base message keeps learned semantics out
            # of the baseline; hyperedge_protocol_only appends its fixed-bit
            # endpoint-state payload and the ordinary transport charges the
            # configured power, serialization delay, loss, and energy.
            core = env.core
            if frame % int(carrier_period) == 0:
                carrier_fraction = float(np.clip(
                    float(cfg.marl.comm_tx_power_w)
                    / max(float(cfg.uav.P_isac_total), 1.0e-12),
                    0.0,
                    1.0,
                ))
                core.submit_learned_communications(
                    messages={
                        sender: np.zeros(
                            core._comm_payload_dim, dtype=np.float64)
                        for sender in range(core.K)
                    },
                    rate_indices={sender: 0 for sender in range(core.K)},
                    comm_power_fractions={
                        sender: carrier_fraction for sender in range(core.K)
                    },
                    sensing_target_weights={
                        sender: np.ones(core.Q, dtype=np.float64)
                        for sender in range(core.K)
                    },
                    token_masks={
                        sender: np.ones(core.Q, dtype=np.float64)
                        for sender in range(core.K)
                    },
                )
            actions = {
                agent: {
                    "delta_p": np.zeros(2, dtype=np.float64),
                    "role": 2,
                }
                for agent in observations
            }
            injected_timeout_original = None
            if (
                fault_inject_power_timeout_frame is not None
                and int(fault_inject_power_timeout_frame) <= frame
                < int(fault_inject_power_timeout_frame)
                + int(fault_inject_power_timeout_duration)
                and core._distributed_replicated_power_executor is not None
            ):
                # Development-only deterministic fault injection. Warm-up has
                # already completed; an effectively zero online timeout tests
                # the incumbent path without changing physics or RNG state.
                injected_timeout_original = float(
                    core._distributed_replicated_power_executor.batch_timeout_s)
                core._distributed_replicated_power_executor.batch_timeout_s = 1.0e-9
            step_started = time.perf_counter()
            observations, _rewards, terminated, _truncated, info = env.step(
                actions)
            if (
                injected_timeout_original is not None
                and core._distributed_replicated_power_executor is not None
            ):
                core._distributed_replicated_power_executor.batch_timeout_s = (
                    injected_timeout_original)
            step_seconds.append(time.perf_counter() - step_started)
            detection.append(np.asarray(info["P_D_q"], dtype=np.float64))
            if include_trace:
                uav_position_trace.append(np.asarray(
                    info["uav_positions"], dtype=np.float64).copy())
                sensing_power_trace.append(np.asarray(
                    env.core._current_sensing_power_w,
                    dtype=np.float64,
                ).copy())
                certificate_safe_gain_trace.append(np.asarray([
                    env.core._composable_certificate_gain_views[node, node]
                    for node in range(env.core.K)
                ], dtype=np.float64))
            bits.append(float(info.get(
                "total_bits_all", info.get("total_bits", 0.0))))
            active_senders.append(float(info.get(
                "learned_comm_active_senders", 0.0)))
            delivery.append(float(info.get("learned_comm_delivery_rate", 1.0)))
            deadline.append(float(info.get(
                "learned_comm_deadline_violation_rate", 0.0)))
            coverage.append(float(info.get("hyperedge_target_coverage", 0.0)))
            movement_coverage.append(float(info.get(
                "movement_executed_assignment_target_coverage", 0.0)))
            belief_rmse.append(float(info.get(
                "belief_position_rmse_m", float("nan"))))
            power_resolve_fraction.append(float(info.get(
                "distributed_replicated_power_resolve_fraction", 1.0)))
            power_reuse_fraction.append(float(info.get(
                "distributed_replicated_power_reuse_fraction", 0.0)))
            power_solve_time_ms.append(1000.0 * float(info.get(
                "distributed_replicated_power_solve_time_s", 0.0)))
            power_parallel_critical_path_ms.append(1000.0 * float(info.get(
                "distributed_replicated_power_parallel_critical_path_s", 0.0)))
            power_process_parallel_used.append(float(info.get(
                "distributed_replicated_power_process_parallel_used", 0.0)))
            power_process_batch_wall_ms.append(1000.0 * float(info.get(
                "distributed_replicated_power_parallel_batch_wall_s", 0.0)))
            power_process_fallback.append(float(info.get(
                "distributed_replicated_power_parallel_fallback", 0.0)))
            power_deadline_incumbent.append(float(info.get(
                "distributed_replicated_power_deadline_incumbent_fraction", 0.0)))
            power_deadline_incumbent_certified.append(float(info.get(
                "distributed_replicated_power_deadline_incumbent_certificate_fraction",
                0.0)))
            power_deadline_uniform.append(float(info.get(
                "distributed_replicated_power_deadline_uniform_fraction", 0.0)))
            power_deadline_harmonic.append(float(info.get(
                "distributed_replicated_power_deadline_harmonic_fraction", 0.0)))
            power_deadline_composable_floor.append(float(info.get(
                "distributed_replicated_power_deadline_composable_deflection_floor",
                0.0)))
            power_deadline_composable_pd_floor.append(float(info.get(
                "distributed_replicated_power_deadline_composable_pd_floor",
                0.0)))
            power_deadline_safe_target_coverage.append(float(info.get(
                "distributed_replicated_power_deadline_safe_target_coverage",
                0.0)))
            power_deadline_safe_min_contributors.append(float(info.get(
                "distributed_replicated_power_deadline_safe_min_contributors",
                0.0)))
            power_deadline_harmonic_shadow_pd_floor.append(float(info.get(
                "distributed_replicated_power_deadline_sparse_harmonic_shadow_pd_floor",
                0.0)))
            power_deadline_shadow_reserve_fraction.append(float(info.get(
                "distributed_replicated_power_deadline_shadow_minimum_reserve_fraction",
                float("inf"))))
            power_deadline_shadow_reserve_feasible.append(float(info.get(
                "distributed_replicated_power_deadline_shadow_reserve_feasible",
                0.0)))
            power_deadline_shadow_global_lp_pd_floor.append(float(info.get(
                "distributed_replicated_power_deadline_shadow_global_lp_pd_floor",
                0.0)))
            power_deadline_shadow_global_lp_time_ms.append(1000.0 * float(
                info.get(
                    "distributed_replicated_power_deadline_shadow_global_lp_time_s",
                    0.0)))
            power_deadline_shadow_harmonic_ratio.append(float(info.get(
                "distributed_replicated_power_deadline_shadow_harmonic_approximation_ratio",
                0.0)))
            certificate_complete.append(float(info.get(
                "composable_certificate_complete_fraction", 0.0)))
            certificate_worst_pd_lower.append(float(info.get(
                "composable_certificate_worst_pd_lower", 0.0)))
            certificate_global_upper.append(float(info.get(
                "composable_certificate_global_optimum_upper", float("inf"))))
            certificate_approximation_ratio.append(float(info.get(
                "composable_certificate_joint_approximation_ratio_lower", 0.0)))
            certificate_payload_bits.append(float(info.get(
                "composable_certificate_payload_bits_per_sender", 0.0)))
            owner_posterior_payload_bits.append(float(info.get(
                "owner_posterior_payload_bits", 0.0)))
            owner_posterior_fused_entries.append(float(info.get(
                "owner_posterior_fused_entries", 0.0)))
            owner_posterior_cov_trace_ratio.append(float(info.get(
                "owner_posterior_cov_trace_ratio", 1.0)))
            owner_posterior_packet_aoi.append(float(info.get(
                "owner_posterior_packet_aoi_mean_frames", 0.0)))
            owner_posterior_state_dim.append(float(info.get(
                "owner_posterior_state_dim", 0.0)))
            bistatic_information_gain.append(float(info.get(
                "bistatic_tracker_information_gain_mean", 0.0)))
            bistatic_full_rank_fraction.append(float(info.get(
                "bistatic_tracker_full_rank_target_fraction", 0.0)))
            controller_compute_ms.append(1000.0 * float(info.get(
                "timing_controller_compute_critical_path_s", 0.0)))
            radio_critical_path_ms.append(1000.0 * float(info.get(
                "timing_radio_serialization_critical_path_s", 0.0)))
            closed_loop_critical_path_ms.append(1000.0 * float(info.get(
                "timing_closed_loop_critical_path_upper_s", 0.0)))
            if bool(terminated.get("__all__", False)):
                break
    finally:
        env.close()

    values = np.asarray(detection, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise RuntimeError("strict pilot produced no detection trajectory")
    tail = values[-min(int(tail_window), values.shape[0]):]
    target_mean = np.mean(tail, axis=0)
    bottom = np.sort(target_mean)[:min(3, target_mean.size)]
    steady = float(np.mean(tail))
    weak3 = float(np.mean(bottom))
    worst = float(np.min(target_mean))
    finite_certificate_upper = np.asarray(
        certificate_global_upper, dtype=np.float64)
    finite_certificate_upper = finite_certificate_upper[
        np.isfinite(finite_certificate_upper)]
    result = {
        "seed": int(seed),
        "carrier_period": int(carrier_period),
        "frames": int(values.shape[0]),
        "fault_inject_power_timeout_frame": (
            None if fault_inject_power_timeout_frame is None
            else int(fault_inject_power_timeout_frame)),
        "fault_inject_power_timeout_duration": int(
            fault_inject_power_timeout_duration),
        "steady": steady,
        "weak3": weak3,
        "worst": worst,
        "qos_success": bool(
            steady >= 0.80 and weak3 >= 0.70 and worst >= 0.60),
        "bits_per_frame": float(np.mean(bits)),
        "active_senders_per_frame": float(np.mean(active_senders)),
        "delivery_rate": float(np.mean(delivery)),
        "deadline_violation_rate": float(np.mean(deadline)),
        "hyperedge_coverage": float(np.mean(coverage)),
        "final_hyperedge_coverage": float(coverage[-1]),
        "movement_target_coverage": float(np.mean(movement_coverage)),
        "belief_position_rmse_m": float(np.nanmean(belief_rmse)),
        "power_resolve_fraction": float(np.mean(power_resolve_fraction)),
        "power_reuse_fraction": float(np.mean(power_reuse_fraction)),
        "power_solve_time_ms": float(np.mean(power_solve_time_ms)),
        "power_parallel_critical_path_ms": float(np.mean(
            power_parallel_critical_path_ms)),
        "power_process_parallel_used_fraction": float(np.mean(
            power_process_parallel_used)),
        "power_process_batch_wall_ms": float(np.mean(
            power_process_batch_wall_ms)),
        "power_process_fallback_fraction": float(np.mean(
            power_process_fallback)),
        "power_process_worker_warmup_ms": worker_warmup_ms,
        "power_deadline_incumbent_fraction": float(np.mean(
            power_deadline_incumbent)),
        "power_deadline_incumbent_certificate_fraction": float(np.mean(
            power_deadline_incumbent_certified)),
        "power_deadline_uniform_fraction": float(np.mean(
            power_deadline_uniform)),
        "power_deadline_harmonic_fraction": float(np.mean(
            power_deadline_harmonic)),
        "power_deadline_composable_deflection_floor": float(np.mean(
            power_deadline_composable_floor)),
        "power_deadline_composable_pd_floor": float(np.mean(
            power_deadline_composable_pd_floor)),
        "power_deadline_composable_pd_floor_p05": float(np.percentile(
            power_deadline_composable_pd_floor, 5)),
        "power_deadline_composable_pd_floor_min": float(np.min(
            power_deadline_composable_pd_floor)),
        "power_deadline_safe_target_coverage": float(np.mean(
            power_deadline_safe_target_coverage)),
        "power_deadline_safe_min_contributors": float(np.mean(
            power_deadline_safe_min_contributors)),
        "power_deadline_sparse_harmonic_shadow_pd_floor": float(np.mean(
            power_deadline_harmonic_shadow_pd_floor)),
        "power_deadline_shadow_reserve_feasible_fraction": float(np.mean(
            power_deadline_shadow_reserve_feasible)),
        "power_deadline_shadow_global_lp_pd_floor": float(np.mean(
            power_deadline_shadow_global_lp_pd_floor)),
        "power_deadline_shadow_global_lp_time_ms": float(np.mean(
            power_deadline_shadow_global_lp_time_ms)),
        "power_deadline_shadow_harmonic_approximation_ratio": float(np.mean(
            power_deadline_shadow_harmonic_ratio)),
        "power_deadline_shadow_minimum_reserve_fraction_mean": float(np.mean(
            np.asarray(power_deadline_shadow_reserve_fraction)[
                np.isfinite(power_deadline_shadow_reserve_fraction)]))
        if np.any(np.isfinite(power_deadline_shadow_reserve_fraction))
        else float("inf"),
        "certificate_complete_fraction": float(np.mean(
            certificate_complete)),
        "certificate_worst_pd_lower": float(np.mean(
            certificate_worst_pd_lower)),
        "certificate_global_optimum_upper": float(
            np.mean(finite_certificate_upper)
            if finite_certificate_upper.size else float("inf")),
        "certificate_joint_approximation_ratio_lower": float(np.mean(
            certificate_approximation_ratio)),
        "certificate_payload_bits_per_sender": float(np.max(
            certificate_payload_bits, initial=0.0)),
        "owner_posterior_payload_bits_per_frame": float(np.mean(
            owner_posterior_payload_bits)),
        "owner_posterior_fused_entries_per_frame": float(np.mean(
            owner_posterior_fused_entries)),
        "owner_posterior_cov_trace_ratio": float(np.mean(
            owner_posterior_cov_trace_ratio)),
        "owner_posterior_packet_aoi_mean_frames": float(np.mean(
            owner_posterior_packet_aoi)),
        "owner_posterior_state_dim": float(np.max(
            owner_posterior_state_dim, initial=0.0)),
        "bistatic_tracker_information_gain_mean": float(np.mean(
            bistatic_information_gain)),
        "bistatic_tracker_full_rank_target_fraction": float(np.mean(
            bistatic_full_rank_fraction)),
        "controller_compute_critical_path_ms": float(np.mean(
            controller_compute_ms)),
        "controller_compute_critical_path_p95_ms": float(np.percentile(
            controller_compute_ms, 95)),
        "radio_critical_path_ms": float(np.mean(radio_critical_path_ms)),
        "closed_loop_critical_path_ms": float(np.mean(
            closed_loop_critical_path_ms)),
        "closed_loop_critical_path_p95_ms": float(np.percentile(
            closed_loop_critical_path_ms, 95)),
        "closed_loop_critical_path_max_ms": float(np.max(
            closed_loop_critical_path_ms)),
        # Backward-compatible alias. Unlike the old subtract/add wall-clock
        # estimate, this is now the explicitly scoped conservative path.
        "deployment_step_time_estimate_ms": float(np.mean(
            closed_loop_critical_path_ms)),
        "step_time_ms": float(1000.0 * np.mean(step_seconds)),
        "step_time_p95_ms": float(
            1000.0 * np.percentile(step_seconds, 95)),
        "step_time_max_ms": float(1000.0 * np.max(step_seconds)),
        "online_budget_ms": float(1000.0 * cfg.scenario.dt),
        "online_deadline_miss_rate": float(np.mean(
            np.asarray(closed_loop_critical_path_ms)
            > 1000.0 * float(cfg.scenario.dt))),
    }
    if include_trace:
        result["trace"] = {
            "detection": values.tolist(),
            "belief_rmse_m": list(belief_rmse),
            "uav_positions": np.asarray(
                uav_position_trace, dtype=np.float64).tolist(),
            "sensing_power_w": np.asarray(
                sensing_power_trace, dtype=np.float64).tolist(),
            "certificate_safe_gain_per_watt": np.asarray(
                certificate_safe_gain_trace, dtype=np.float64).tolist(),
            "bits": list(bits),
            "delivery": list(delivery),
            "hyperedge_coverage": list(coverage),
            "actual_worst_pd": np.min(values, axis=1).tolist(),
            "certificate_complete_fraction": list(certificate_complete),
            "certificate_worst_pd_lower": list(certificate_worst_pd_lower),
            "power_deadline_composable_pd_floor": list(
                power_deadline_composable_pd_floor),
            "power_deadline_safe_target_coverage": list(
                power_deadline_safe_target_coverage),
            "power_deadline_safe_min_contributors": list(
                power_deadline_safe_min_contributors),
            "power_deadline_sparse_harmonic_shadow_pd_floor": list(
                power_deadline_harmonic_shadow_pd_floor),
            "power_deadline_shadow_minimum_reserve_fraction": list(
                power_deadline_shadow_reserve_fraction),
            "power_deadline_shadow_reserve_feasible": list(
                power_deadline_shadow_reserve_feasible),
            "power_deadline_shadow_global_lp_pd_floor": list(
                power_deadline_shadow_global_lp_pd_floor),
            "power_deadline_shadow_global_lp_time_ms": list(
                power_deadline_shadow_global_lp_time_ms),
            "power_deadline_shadow_harmonic_approximation_ratio": list(
                power_deadline_shadow_harmonic_ratio),
        }
    return result


def run(
    config: str,
    seeds: list[int],
    tail_window: int,
    carrier_period: int = DEFAULT_CARRIER_PERIOD,
    formal: bool = False,
    frames: int | None = None,
    fault_inject_power_timeout_frame: int | None = None,
    fault_inject_power_timeout_duration: int = 1,
    include_trace: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config)
    if frames is not None:
        if formal:
            raise ValueError(
                "formal runs forbid --frames; freeze scenario.T in config")
        if int(frames) < 1:
            raise ValueError("frames must be positive")
        cfg.scenario.T = int(frames)
    if fault_inject_power_timeout_frame is not None:
        if formal:
            raise ValueError("formal runs forbid fault injection")
        if not 0 <= int(fault_inject_power_timeout_frame) < int(cfg.scenario.T):
            raise ValueError("fault injection frame lies outside the episode")
        if int(fault_inject_power_timeout_duration) < 1:
            raise ValueError("fault injection duration must be positive")
        if (
            int(fault_inject_power_timeout_frame)
            + int(fault_inject_power_timeout_duration)
            > int(cfg.scenario.T)
        ):
            raise ValueError("fault injection interval lies outside the episode")
    validate_strict_config(cfg)
    if formal:
        validate_formal_system_identity(config)
    manifest = build_run_manifest(
        cfg,
        config_path=config,
        seeds=seeds,
        algorithm_version=_algorithm_version(cfg),
        root=ROOT,
    )
    if formal:
        validate_formal_run(cfg, seeds, manifest)
    episodes = [
        _episode(
            cfg,
            seed,
            tail_window,
            carrier_period,
            include_trace=include_trace,
            fault_inject_power_timeout_frame=(
                fault_inject_power_timeout_frame),
            fault_inject_power_timeout_duration=(
                fault_inject_power_timeout_duration),
        )
        for seed in seeds
    ]
    return {
        "config": os.path.normpath(config),
        "run_manifest": manifest,
        "status": "FORMAL_COMPLETE" if formal else "DIAGNOSTIC_ONLY",
        "policy": "hold_action_plus_periodic_control_carrier_and_strict_stack",
        "carrier_period": int(carrier_period),
        "episodes": episodes,
        "summary": {
            "seeds": len(episodes),
            "qos_rate": float(np.mean([e["qos_success"] for e in episodes])),
            "steady_mean": float(np.mean([e["steady"] for e in episodes])),
            "weak3_mean": float(np.mean([e["weak3"] for e in episodes])),
            "worst_mean": float(np.mean([e["worst"] for e in episodes])),
            "bits_per_frame": float(np.mean([
                e["bits_per_frame"] for e in episodes])),
            "deadline_violation_rate": float(np.mean([
                e["deadline_violation_rate"] for e in episodes])),
            "hyperedge_coverage": float(np.mean([
                e["hyperedge_coverage"] for e in episodes])),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/exp_strict_distributed_no_truth_pilot.yaml")
    parser.add_argument(
        "--seeds", default="7",
        help="comma-separated diagnostic seeds (not a formal blind evaluation)")
    parser.add_argument("--tail-window", type=int, default=50)
    parser.add_argument(
        "--frames", type=int, default=None,
        help="diagnostic frame override; formal runs should use frozen config")
    parser.add_argument(
        "--fault-inject-power-timeout-frame", type=int, default=None,
        help="development-only: force the node LP pool to time out at frame N")
    parser.add_argument(
        "--fault-inject-power-timeout-duration", type=int, default=1,
        help="development-only: number of consecutive forced-timeout frames")
    parser.add_argument(
        "--include-trace", action="store_true",
        help="include frame-level diagnostic traces in the result JSON")
    parser.add_argument(
        "--carrier-period", type=int, default=DEFAULT_CARRIER_PERIOD,
        help="transmit the strict control carrier once every N frames")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--formal", action="store_true",
        help="fail closed unless workspace, 100-seed bank and threads are frozen")
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        parser.error("--seeds must contain at least one integer")
    if args.tail_window < 1:
        parser.error("--tail-window must be positive")
    if args.carrier_period < 1:
        parser.error("--carrier-period must be positive")
    report = run(
        args.config,
        seeds,
        args.tail_window,
        args.carrier_period,
        args.formal,
        args.frames,
        args.fault_inject_power_timeout_frame,
        args.fault_inject_power_timeout_duration,
        args.include_trace,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
