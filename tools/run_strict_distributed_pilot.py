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
from uav_isac.evaluation.temporal_performance import (
    summarize_closed_loop_convergence,
    summarize_multiframe_detection,
)
from uav_isac.utils.reproducibility import (
    build_run_manifest,
    validate_formal_run,
)


DEFAULT_CARRIER_PERIOD = 3
MULTIFRAME_QOS_FLOORS = (0.80, 0.70, 0.75)


def _minimum_pairwise_distance(positions_xy: np.ndarray) -> float:
    """Return the minimum distinct-node Euclidean distance."""

    positions = np.asarray(positions_xy, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_xy must have shape (K,2)")
    if positions.shape[0] < 2:
        return float("inf")
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    distance[np.diag_indices(positions.shape[0])] = np.inf
    return float(np.min(distance))


def _minimum_swept_pairwise_distance(
    before_xy: np.ndarray, after_xy: np.ndarray,
) -> float:
    """Return exact minimum separation along simultaneous linear segments."""

    before = np.asarray(before_xy, dtype=np.float64)
    after = np.asarray(after_xy, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 2:
        raise ValueError("before_xy and after_xy must have identical (K,2) shapes")
    minimum = float("inf")
    velocity = after - before
    for left in range(before.shape[0]):
        for right in range(left + 1, before.shape[0]):
            relative = before[left] - before[right]
            relative_velocity = velocity[left] - velocity[right]
            denominator = float(relative_velocity @ relative_velocity)
            fraction = 0.0 if denominator <= 1.0e-18 else float(np.clip(
                -(relative @ relative_velocity) / denominator, 0.0, 1.0))
            separation = relative + fraction * relative_velocity
            minimum = min(minimum, float(np.linalg.norm(separation)))
    return minimum


_AUDITED_RUNTIME_MODULES = (
    "uav_isac.environment.env_core",
    "uav_isac.coordination.hyperedge",
    "uav_isac.coordination.maxmin_power",
    "uav_isac.agents.trainer",
    "uav_isac.agents.frozen_structure_student",
    "uav_isac.prediction.certified_gnn",
    "uav_isac.prediction.refresh_gate",
    "uav_isac.optimization.constrained_pareto",
    "uav_isac.optimization.temporal_feasible_structure",
)


def _execution_audit(
    cfg: Any,
    modules_before_episode: set[str] | None = None,
) -> dict[str, Any]:
    """Describe controller ownership and observe optional runtime imports.

    Configuration hashes alone cannot distinguish an analytical baseline from
    a learned controller.  Keep this audit deliberately declarative: it names
    the component that owns each executed decision and separately records
    whether research/legacy modules were imported into the process.
    """
    ma = cfg.marl
    if bool(getattr(
        ma, "distributed_primal_dual_power_enabled", False
    )):
        power_controller = "distributed_primal_dual_power"
    elif bool(getattr(ma, "distributed_replicated_power_enabled", False)):
        power_controller = "distributed_replicated_analytical_power"
    else:
        power_controller = "environment_default_power"

    if bool(getattr(
        ma, "distributed_bistatic_bottleneck_movement_enabled", False
    )):
        movement_controller = "distributed_bistatic_bottleneck"
    elif bool(getattr(
        ma, "distributed_bottleneck_matching_movement_enabled", False
    )):
        movement_controller = "distributed_bottleneck_matching"
    else:
        movement_controller = "submitted_hold_action"

    return {
        "schema_version": "strict-controller-execution-audit/v1",
        "controller_family": "analytical_distributed_baseline",
        "learned_policy_executed": False,
        "checkpoint_loaded": False,
        "decision_owners": {
            "submitted_agent_action": "deterministic_hold",
            "communication_semantics": (
                "zero_learned_payload_plus_physical_protocol"),
            "structure_proposal": "local_belief_hyperedge_negotiation",
            "structure_selection": "distributed_hyperedge_stack",
            "sensing_power": (
                "analytical" if bool(getattr(
                    ma, "analytical_sensing_power_enabled", False
                )) else "submitted_actor_split"),
            "power_allocation": power_controller,
            "movement": movement_controller,
        },
        "extension_flags": {
            "constrained_cvar_ppo": bool(getattr(
                ma, "constrained_cvar_ppo_enabled", False)),
            "distributed_primal_dual_power": bool(getattr(
                ma, "distributed_primal_dual_power_enabled", False)),
            "temporal_feasible_structure": bool(getattr(
                ma, "temporal_feasible_structure_enabled", False)),
        },
        "observed_module_imports": {
            module: {
                "present_before_episode": (
                    module in modules_before_episode
                    if modules_before_episode is not None else None),
                "imported_during_episode": (
                    module not in modules_before_episode and module in sys.modules
                    if modules_before_episode is not None else None),
                "present_after_episode": module in sys.modules,
            }
            for module in _AUDITED_RUNTIME_MODULES
        },
    }


def _finite_values_or_null(values: list[float]) -> list[float | None]:
    """Return strict-JSON values without encoding NaN/Infinity tokens."""
    return [float(value) if np.isfinite(value) else None for value in values]


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
    if bool(getattr(
        cfg.marl,
        "distributed_replicated_power_common_model_certificate",
        False,
    )):
        version += "-common-model-certified"
    if bool(getattr(
        cfg.marl,
        "distributed_common_model_packet_reconstruction_enabled",
        False,
    )):
        version += "-packet-model-rendezvous"
    if bool(getattr(
        cfg.marl,
        "distributed_atomic_decision_epoch_enabled",
        False,
    )):
        version += "-atomic-epoch"
    if bool(getattr(
        cfg.marl,
        "distributed_movement_preexecution_swept_certificate",
        False,
    )):
        version += "-aoi-swept-certified"
    if bool(getattr(
        cfg.marl,
        "distributed_movement_analytic_composable_projection_enabled",
        False,
    )):
        version += "-separable-safe-qp"
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
    if not bool(getattr(
        cfg.marl, "distributed_composable_certificate_enabled", False
    )):
        version += "-certificate-transport-off"
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
        "distributed_replicated_power_common_model_certificate": (
            ma.distributed_replicated_power_common_model_certificate),
        "distributed_common_model_packet_reconstruction_enabled": (
            ma.distributed_common_model_packet_reconstruction_enabled),
        "distributed_atomic_decision_epoch_enabled": (
            ma.distributed_atomic_decision_epoch_enabled),
        "distributed_movement_preexecution_swept_certificate": (
            ma.distributed_movement_preexecution_swept_certificate),
        "distributed_movement_analytic_composable_projection_enabled": (
            ma.distributed_movement_analytic_composable_projection_enabled),
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
    required_age_margin = float(
        2.0 * cfg.uav.v_max * cfg.scenario.dt)
    if float(ma.distributed_movement_safety_margin_per_age_m) + 1.0e-12 < (
        required_age_margin
    ):
        raise ValueError(
            "strict distributed pilot requires two-endpoint vmax*AoI "
            "movement safety margin")
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
    include_timing_trace: bool = False,
    fault_inject_power_timeout_frame: int | None = None,
    fault_inject_power_timeout_duration: int = 1,
) -> dict[str, Any]:
    if int(carrier_period) < 1:
        raise ValueError("carrier_period must be positive")
    cfg.marl.acceleration_golden_trace_enabled = bool(include_trace)
    env = UAVISACEnv(config=cfg)
    observations, reset_info = env.reset(seed=int(seed))
    worker_warmup_ms = 1000.0 * float(reset_info.get(
        "distributed_replicated_power_worker_warmup_time_s", 0.0))
    detection: list[np.ndarray] = []
    detection_deflection: list[np.ndarray] = []
    fusion_correlation_factor_mean: list[float] = []
    fusion_correlation_factor_max: list[float] = []
    fusion_correlation_factor_trace: list[np.ndarray] = []
    correlation_shadow_attempted: list[float] = []
    correlation_shadow_exact_accepted: list[float] = []
    correlation_shadow_would_commit: list[float] = []
    correlation_shadow_replica_agreement: list[float] = []
    correlation_shadow_active_unchanged: list[float] = []
    correlation_shadow_improvement: list[float] = []
    correlation_shadow_protocol_bits: list[float] = []
    correlation_shadow_protocol_latency_ms: list[float] = []
    correlation_shadow_gram_time_ms: list[float] = []
    correlation_shadow_lp_time_ms: list[float] = []
    correlation_shadow_cache_hits: list[float] = []
    correlation_shadow_cache_misses: list[float] = []
    correlation_shadow_dual_early_stop: list[float] = []
    correlation_shadow_rebuild_critical_ms: list[float] = []
    correlation_shadow_record_bytes: list[float] = []
    correlation_shadow_failure_reasons: list[str] = []
    correlation_shadow_input_scopes: list[str] = []
    bits: list[float] = []
    active_senders: list[float] = []
    delivery: list[float] = []
    deadline: list[float] = []
    coverage: list[float] = []
    movement_coverage: list[float] = []
    belief_rmse: list[float] = []
    power_resolve_fraction: list[float] = []
    power_reuse_fraction: list[float] = []
    power_common_model_certificate: list[float] = []
    power_common_model_fallback: list[float] = []
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
    hyperedge_acceleration_total_calls: list[float] = []
    hyperedge_acceleration_total_seconds: list[float] = []
    hyperedge_acceleration_backend: list[str] = []
    deflection_materialization_backend: list[str] = []
    controller_compute_ms: list[float] = []
    controller_movement_compute_ms: list[float] = []
    controller_structure_compute_ms: list[float] = []
    movement_projection_solve_ms: list[float] = []
    movement_projection_calls: list[float] = []
    movement_reduced_qp_rate: list[float] = []
    radio_critical_path_ms: list[float] = []
    closed_loop_critical_path_ms: list[float] = []
    step_seconds: list[float] = []
    inter_uav_min_distance_m: list[float] = []
    inter_uav_swept_min_distance_m: list[float] = []
    preexecution_swept_min_distance_m: list[float] = []
    preexecution_swept_fail_closed: list[float] = []
    isac_power_budget_violation_w: list[float] = []
    minimum_battery_j: list[float] = []
    energy_causality_violation_j: list[float] = []
    uav_position_trace: list[np.ndarray] = []
    sensing_power_trace: list[np.ndarray] = []
    certificate_safe_gain_trace: list[np.ndarray] = []
    acceleration_golden_trace: list[dict[str, Any]] = []
    try:
        for frame in range(int(cfg.scenario.T)):
            # The analytical pilot has no learned actor, but the distributed
            # control protocol still needs an explicit physical carrier.  A
            # zero-rate/zero-content base message keeps learned semantics out
            # of the baseline; hyperedge_protocol_only appends its fixed-bit
            # endpoint-state payload and the ordinary transport charges the
            # configured power, serialization delay, loss, and energy.
            core = env.core
            acquisition_carrier = bool(
                getattr(
                    core,
                    "_distributed_atomic_decision_epoch_enabled",
                    False,
                )
                and not core._hyperedge_selected_set
            )
            if frame % int(carrier_period) == 0 or acquisition_carrier:
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
            positions_before = np.asarray(
                [uav.pos[:2] for uav in core.uavs], dtype=np.float64)
            observations, _rewards, terminated, _truncated, info = env.step(
                actions)
            if (
                injected_timeout_original is not None
                and core._distributed_replicated_power_executor is not None
            ):
                core._distributed_replicated_power_executor.batch_timeout_s = (
                    injected_timeout_original)
            step_seconds.append(time.perf_counter() - step_started)
            positions_after = np.asarray(
                [uav.pos[:2] for uav in core.uavs], dtype=np.float64)
            inter_uav_min_distance_m.append(
                _minimum_pairwise_distance(positions_after))
            inter_uav_swept_min_distance_m.append(
                _minimum_swept_pairwise_distance(
                    positions_before, positions_after))
            preexecution_swept_min_distance_m.append(float(info.get(
                "movement_preexecution_swept_minimum_m", float("inf"))))
            preexecution_swept_fail_closed.append(float(info.get(
                "movement_preexecution_swept_fail_closed", 0.0)))
            isac_power_budget_violation_w.append(float(info.get(
                "isac_max_power_budget_violation_w", 0.0)))
            minimum_battery_j.append(float(min(
                uav.battery for uav in core.uavs)))
            energy_causality_violation_j.append(float(max(
                uav.max_energy_deficit_j for uav in core.uavs)))
            detection.append(np.asarray(info["P_D_q"], dtype=np.float64))
            detection_deflection.append(np.asarray(
                info["detection_deflection_q"], dtype=np.float64))
            fusion_correlation_factor_mean.append(float(info.get(
                "fusion_correlation_factor_mean", 1.0)))
            fusion_correlation_factor_max.append(float(info.get(
                "fusion_correlation_factor_max", 1.0)))
            correlation_shadow_attempted.append(float(info.get(
                "correlation_exchange_shadow_attempted", 0.0)))
            correlation_shadow_exact_accepted.append(float(info.get(
                "correlation_exchange_shadow_exact_accepted", 0.0)))
            correlation_shadow_would_commit.append(float(info.get(
                "correlation_exchange_shadow_would_commit", 0.0)))
            correlation_shadow_replica_agreement.append(float(info.get(
                "correlation_exchange_shadow_replica_agreement", 0.0)))
            correlation_shadow_active_unchanged.append(float(info.get(
                "correlation_exchange_shadow_active_unchanged", 1.0)))
            correlation_shadow_improvement.append(float(info.get(
                "correlation_exchange_shadow_improvement", 0.0)))
            correlation_shadow_protocol_bits.append(float(info.get(
                "correlation_exchange_shadow_protocol_bits", 0.0)))
            correlation_shadow_protocol_latency_ms.append(1000.0 * float(
                info.get("correlation_exchange_shadow_protocol_latency_s", 0.0)))
            correlation_shadow_gram_time_ms.append(1000.0 * float(info.get(
                "correlation_exchange_shadow_gram_time_s", 0.0)))
            correlation_shadow_lp_time_ms.append(1000.0 * float(info.get(
                "correlation_exchange_shadow_lp_time_s", 0.0)))
            correlation_shadow_cache_hits.append(float(info.get(
                "correlation_exchange_shadow_factor_cache_hits", 0.0)))
            correlation_shadow_cache_misses.append(float(info.get(
                "correlation_exchange_shadow_factor_cache_misses", 0.0)))
            correlation_shadow_dual_early_stop.append(float(info.get(
                "correlation_exchange_shadow_dual_early_stop", 0.0)))
            correlation_shadow_rebuild_critical_ms.append(1000.0 * float(
                info.get(
                    "correlation_exchange_shadow_rebuild_critical_path_s", 0.0)))
            correlation_shadow_record_bytes.append(float(info.get(
                "correlation_exchange_shadow_record_bytes", 0.0)))
            shadow_failure = str(info.get(
                "correlation_exchange_shadow_failure_reason", ""))
            if shadow_failure:
                correlation_shadow_failure_reasons.append(shadow_failure)
            shadow_scope = str(info.get(
                "correlation_exchange_shadow_input_scope", ""))
            if shadow_scope:
                correlation_shadow_input_scopes.append(shadow_scope)
            if include_trace:
                fusion_correlation_factor_trace.append(np.asarray(
                    info.get(
                        "fusion_correlation_factor_q",
                        np.ones(int(cfg.scenario.Q), dtype=np.float64),
                    ),
                    dtype=np.float64,
                ).copy())
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
                snapshot = env.core._hyperedge_golden_snapshot
                acceleration_golden_trace.append({
                    "frame": int(snapshot.get("frame", frame)),
                    "hold_active": bool(snapshot.get(
                        "hold_active", False)),
                    "received_last_seen": np.asarray(snapshot.get(
                        "received_last_seen", []), dtype=np.int64).tolist(),
                    "visible_views": np.asarray(snapshot.get(
                        "visible_views", []), dtype=bool).tolist(),
                    "endpoint_position_views": np.asarray(snapshot.get(
                        "endpoint_position_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "endpoint_velocity_views": np.asarray(snapshot.get(
                        "endpoint_velocity_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "target_position_views": np.asarray(snapshot.get(
                        "target_position_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "target_velocity_views": np.asarray(snapshot.get(
                        "target_velocity_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "target_position_uncertainty_views": np.asarray(
                        snapshot.get(
                            "target_position_uncertainty_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "target_velocity_uncertainty_views": np.asarray(
                        snapshot.get(
                            "target_velocity_uncertainty_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "local_plans": [{
                        "selected": [list(edge) for edge in plan["selected"]],
                        "proxy_target_value": np.asarray(
                            plan["proxy_target_value"],
                            dtype=np.float64,
                        ).tolist(),
                        "proxy_scores": [
                            [list(edge), float(score)]
                            for edge, score in plan["proxy_scores"]
                        ],
                    } for plan in snapshot.get("local_plans", ())],
                    "mutual": [
                        list(edge) for edge in snapshot.get("mutual", ())],
                    "stable": [
                        list(edge) for edge in snapshot.get("stable", ())],
                    "selected": [
                        list(edge) for edge in snapshot.get("selected", ())],
                    "public_full_views": np.asarray(snapshot.get(
                        "public_full_views", []), dtype=bool).tolist(),
                    "public_gain_views": np.asarray(snapshot.get(
                        "public_gain_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "certificate_gain_views": np.asarray(snapshot.get(
                        "certificate_gain_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "certificate_gain_upper_views": np.asarray(snapshot.get(
                        "certificate_gain_upper_views", []),
                        dtype=np.float64,
                    ).tolist(),
                    "consensus_streak": np.asarray(snapshot.get(
                        "consensus_streak", []), dtype=np.int64).tolist(),
                    "executed_power_w": np.asarray(
                        env.core._current_sensing_power_w,
                        dtype=np.float64,
                    ).tolist(),
                    "local_power_cache_w": np.asarray(
                        env.core._distributed_replicated_local_power_cache,
                        dtype=np.float64,
                    ).tolist(),
                    "local_prices": np.asarray(
                        env.core._distributed_replicated_local_price_cache,
                        dtype=np.float64,
                    ).tolist(),
                    "local_cache_valid": np.asarray(
                        env.core._distributed_replicated_local_cache_valid,
                        dtype=bool,
                    ).tolist(),
                })
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
            power_common_model_certificate.append(float(info.get(
                "distributed_replicated_power_common_model_certificate",
                0.0)))
            power_common_model_fallback.append(float(info.get(
                "distributed_replicated_power_common_model_fallback_fraction",
                0.0)))
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
            hyperedge_acceleration_total_calls.append(float(info.get(
                "hyperedge_acceleration_total_calls", 0.0)))
            hyperedge_acceleration_total_seconds.append(float(info.get(
                "hyperedge_acceleration_total_seconds", 0.0)))
            hyperedge_acceleration_backend.append(str(info.get(
                "hyperedge_acceleration_backend", "unknown")))
            deflection_materialization_backend.append(str(info.get(
                "deflection_materialization_backend", "unknown")))
            controller_compute_ms.append(1000.0 * float(info.get(
                "timing_controller_compute_critical_path_s", 0.0)))
            controller_movement_compute_ms.append(1000.0 * float(info.get(
                "timing_controller_movement_compute_s", 0.0)))
            controller_structure_compute_ms.append(1000.0 * float(info.get(
                "timing_controller_structure_compute_s", 0.0)))
            movement_projection_solve_ms.append(1000.0 * float(info.get(
                "movement_safety_solve_time_s", 0.0)))
            movement_projection_calls.append(float(info.get(
                "movement_safety_projection_calls", 0.0)))
            movement_reduced_qp_rate.append(float(info.get(
                "movement_safety_reduced_linear_qp_rate", 0.0)))
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
    deflection_values = np.asarray(detection_deflection, dtype=np.float64)
    if deflection_values.shape != values.shape:
        raise RuntimeError("strict pilot produced incompatible Deflection history")
    tail = values[-min(int(tail_window), values.shape[0]):]
    target_mean = np.mean(tail, axis=0)
    bottom = np.sort(target_mean)[:min(3, target_mean.size)]
    steady = float(np.mean(tail))
    weak3 = float(np.mean(bottom))
    worst = float(np.min(target_mean))
    multiframe = summarize_multiframe_detection(
        deflection_values,
        float(cfg.detection.P_FA),
        windows=(1, 2, 3, 5, 10, 20),
        tail_window=int(tail_window),
        floors=MULTIFRAME_QOS_FLOORS,
        frame_duration_s=float(cfg.scenario.dt),
    )
    convergence = summarize_closed_loop_convergence(
        values,
        rolling_window=min(10, values.shape[0]),
        terminal_window=min(20, values.shape[0]),
        patience=3,
    )
    finite_certificate_upper = np.asarray(
        certificate_global_upper, dtype=np.float64)
    finite_certificate_upper = finite_certificate_upper[
        np.isfinite(finite_certificate_upper)]
    finite_reserve = np.asarray(
        power_deadline_shadow_reserve_fraction, dtype=np.float64)
    finite_reserve = finite_reserve[np.isfinite(finite_reserve)]
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
        "multiframe_detection": multiframe,
        "closed_loop_convergence": convergence,
        "bits_per_frame": float(np.mean(bits)),
        "active_senders_per_frame": float(np.mean(active_senders)),
        "delivery_rate": float(np.mean(delivery)),
        "deadline_violation_rate": float(np.mean(deadline)),
        "hyperedge_coverage": float(np.mean(coverage)),
        "final_hyperedge_coverage": float(coverage[-1]),
        "movement_target_coverage": float(np.mean(movement_coverage)),
        "inter_uav_min_distance_m": float(np.min(
            inter_uav_min_distance_m, initial=float("inf"))),
        "inter_uav_swept_min_distance_m": float(np.min(
            inter_uav_swept_min_distance_m, initial=float("inf"))),
        "preexecution_swept_min_distance_m": float(np.min(
            preexecution_swept_min_distance_m, initial=float("inf"))),
        "preexecution_swept_fail_closed_fraction": float(np.mean(
            preexecution_swept_fail_closed)),
        "isac_max_power_budget_violation_w": float(np.max(
            isac_power_budget_violation_w, initial=0.0)),
        "minimum_battery_j": float(np.min(
            minimum_battery_j, initial=float("inf"))),
        "energy_causality_violation_j": float(np.max(
            energy_causality_violation_j, initial=0.0)),
        "belief_position_rmse_m": float(np.nanmean(belief_rmse)),
        "power_resolve_fraction": float(np.mean(power_resolve_fraction)),
        "power_reuse_fraction": float(np.mean(power_reuse_fraction)),
        "power_common_model_certificate_fraction": float(np.mean(
            power_common_model_certificate)),
        "power_common_model_fallback_fraction": float(np.mean(
            power_common_model_fallback)),
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
        "power_deadline_shadow_minimum_reserve_fraction_mean": (
            float(np.mean(finite_reserve)) if finite_reserve.size else None),
        "power_deadline_shadow_minimum_reserve_fraction_unavailable_reason": (
            None if finite_reserve.size else
            "no finite reserve value was produced by the shadow diagnostic"),
        "certificate_complete_fraction": float(np.mean(
            certificate_complete)),
        "certificate_worst_pd_lower": float(np.mean(
            certificate_worst_pd_lower)),
        "certificate_global_optimum_upper": (
            float(np.mean(finite_certificate_upper))
            if finite_certificate_upper.size else None),
        "certificate_global_optimum_upper_unavailable_reason": (
            None if finite_certificate_upper.size else
            "no finite global certificate upper bound was produced"),
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
        "fusion_correlation_mode": str(
            cfg.detection.fusion_correlation_mode),
        "fusion_correlation_factor_mean": float(np.mean(
            fusion_correlation_factor_mean)),
        "fusion_correlation_factor_max": float(np.max(
            fusion_correlation_factor_max)),
        "correlation_exchange_shadow_enabled": bool(getattr(
            cfg.marl, "correlation_exchange_shadow_enabled", False)),
        "correlation_exchange_shadow_attempt_rate": float(np.mean(
            correlation_shadow_attempted)),
        "correlation_exchange_shadow_exact_accept_rate_per_attempt": float(
            np.sum(correlation_shadow_exact_accepted)
            / max(np.sum(correlation_shadow_attempted), 1.0)),
        "correlation_exchange_shadow_would_commit_rate_per_attempt": float(
            np.sum(correlation_shadow_would_commit)
            / max(np.sum(correlation_shadow_attempted), 1.0)),
        "correlation_exchange_shadow_replica_agreement_rate_per_accept": float(
            np.sum(correlation_shadow_replica_agreement)
            / max(np.sum(correlation_shadow_exact_accepted), 1.0)),
        "correlation_exchange_shadow_active_unchanged_fraction": float(
            np.mean(correlation_shadow_active_unchanged)),
        "correlation_exchange_shadow_improvement_mean_accepted": float(
            np.sum(correlation_shadow_improvement)
            / max(np.sum(correlation_shadow_exact_accepted), 1.0)),
        "correlation_exchange_shadow_protocol_bits_mean_accepted": float(
            np.sum(correlation_shadow_protocol_bits)
            / max(np.sum(correlation_shadow_exact_accepted), 1.0)),
        "correlation_exchange_shadow_protocol_latency_ms_mean_accepted": float(
            np.sum(correlation_shadow_protocol_latency_ms)
            / max(np.sum(correlation_shadow_exact_accepted), 1.0)),
        "correlation_exchange_shadow_gram_time_ms_mean_attempt": float(
            np.sum(correlation_shadow_gram_time_ms)
            / max(np.sum(correlation_shadow_attempted), 1.0)),
        "correlation_exchange_shadow_lp_time_ms_mean_attempt": float(
            np.sum(correlation_shadow_lp_time_ms)
            / max(np.sum(correlation_shadow_attempted), 1.0)),
        "correlation_exchange_shadow_factor_cache_hit_rate": float(
            np.sum(correlation_shadow_cache_hits)
            / max(
                np.sum(correlation_shadow_cache_hits)
                + np.sum(correlation_shadow_cache_misses),
                1.0,
            )),
        "correlation_exchange_shadow_dual_early_stop_rate_per_attempt": float(
            np.sum(correlation_shadow_dual_early_stop)
            / max(np.sum(correlation_shadow_attempted), 1.0)),
        "correlation_exchange_shadow_rebuild_critical_ms_mean_accepted": float(
            np.sum(correlation_shadow_rebuild_critical_ms)
            / max(np.sum(correlation_shadow_exact_accepted), 1.0)),
        "correlation_exchange_shadow_record_bytes_mean_accepted": float(
            np.sum(correlation_shadow_record_bytes)
            / max(np.sum(correlation_shadow_exact_accepted), 1.0)),
        "correlation_exchange_shadow_failure_reasons": sorted(set(
            correlation_shadow_failure_reasons)),
        "correlation_exchange_shadow_input_scopes": sorted(set(
            correlation_shadow_input_scopes)),
        "hyperedge_acceleration_backend": (
            hyperedge_acceleration_backend[-1]
            if hyperedge_acceleration_backend else "unknown"),
        "hyperedge_acceleration_total_calls": float(
            hyperedge_acceleration_total_calls[-1]
            if hyperedge_acceleration_total_calls else 0.0),
        "hyperedge_acceleration_total_seconds": float(
            hyperedge_acceleration_total_seconds[-1]
            if hyperedge_acceleration_total_seconds else 0.0),
        "deflection_materialization_backend": (
            deflection_materialization_backend[-1]
            if deflection_materialization_backend else "unknown"),
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
            "detection_deflection": deflection_values.tolist(),
            "fusion_correlation_factor_q": np.asarray(
                fusion_correlation_factor_trace,
                dtype=np.float64,
            ).tolist(),
            "belief_rmse_m": list(belief_rmse),
            "uav_positions": np.asarray(
                uav_position_trace, dtype=np.float64).tolist(),
            "sensing_power_w": np.asarray(
                sensing_power_trace, dtype=np.float64).tolist(),
            "certificate_safe_gain_per_watt": np.asarray(
                certificate_safe_gain_trace, dtype=np.float64).tolist(),
            "acceleration_golden": acceleration_golden_trace,
            "bits": list(bits),
            "delivery": list(delivery),
            "hyperedge_coverage": list(coverage),
            "power_common_model_certificate": list(
                power_common_model_certificate),
            "power_common_model_fallback_fraction": list(
                power_common_model_fallback),
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
            "power_deadline_shadow_minimum_reserve_fraction": (
                _finite_values_or_null(
                    power_deadline_shadow_reserve_fraction)),
            "power_deadline_shadow_reserve_feasible": list(
                power_deadline_shadow_reserve_feasible),
            "power_deadline_shadow_global_lp_pd_floor": list(
                power_deadline_shadow_global_lp_pd_floor),
            "power_deadline_shadow_global_lp_time_ms": list(
                power_deadline_shadow_global_lp_time_ms),
            "power_deadline_shadow_harmonic_approximation_ratio": list(
                power_deadline_shadow_harmonic_ratio),
        }
    if include_timing_trace:
        result["timing_trace_ms"] = {
            "movement_compute": list(controller_movement_compute_ms),
            "structure_compute": list(controller_structure_compute_ms),
            "movement_projection_solve": list(
                movement_projection_solve_ms),
            "movement_projection_calls": list(movement_projection_calls),
            "movement_reduced_qp_rate": list(movement_reduced_qp_rate),
            "power_parallel": list(power_parallel_critical_path_ms),
            "controller_critical_path": list(controller_compute_ms),
            "radio_serialization_critical_path": list(
                radio_critical_path_ms),
            "closed_loop_critical_path": list(
                closed_loop_critical_path_ms),
            "simulator_step_wall": [
                1000.0 * value for value in step_seconds
            ],
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
    include_timing_trace: bool = False,
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
    modules_before_episode = set(sys.modules)
    episodes = [
        _episode(
            cfg,
            seed,
            tail_window,
            carrier_period,
            include_trace=include_trace,
            include_timing_trace=include_timing_trace,
            fault_inject_power_timeout_frame=(
                fault_inject_power_timeout_frame),
            fault_inject_power_timeout_duration=(
                fault_inject_power_timeout_duration),
        )
        for seed in seeds
    ]
    execution_audit = _execution_audit(cfg, modules_before_episode)
    manifest["execution_audit"] = execution_audit
    common_multiframe_windows = sorted(
        set.intersection(*[
            set(episode["multiframe_detection"]["windows"])
            for episode in episodes
        ]),
        key=int,
    )
    multiframe_summary = {
        width: {
            "steady_mean": float(np.mean([
                episode["multiframe_detection"]["windows"][width]["steady"]
                for episode in episodes
            ])),
            "weak3_mean": float(np.mean([
                episode["multiframe_detection"]["windows"][width]["weak3"]
                for episode in episodes
            ])),
            "worst_mean": float(np.mean([
                episode["multiframe_detection"]["windows"][width]["worst"]
                for episode in episodes
            ])),
            "worst_min": float(np.min([
                episode["multiframe_detection"]["windows"][width]["worst"]
                for episode in episodes
            ])),
            "worst_floor_pass_rate": float(np.mean([
                episode["multiframe_detection"]["windows"][width]["worst"]
                >= MULTIFRAME_QOS_FLOORS[2]
                for episode in episodes
            ])),
            "qos_rate": float(np.mean([
                episode["multiframe_detection"]["windows"][width]["qos_success"]
                for episode in episodes
            ])),
            "rolling_qos_rate_mean": float(np.mean([
                episode["multiframe_detection"]["windows"][width]["rolling_qos_rate"]
                for episode in episodes
            ])),
        }
        for width in common_multiframe_windows
    }
    all_episode_pass_windows = [
        int(width) for width, item in multiframe_summary.items()
        if item["worst_min"] >= MULTIFRAME_QOS_FLOORS[2]
    ]
    return {
        "config": os.path.normpath(config),
        "run_manifest": manifest,
        "execution_audit": execution_audit,
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
            "multiframe_detection": multiframe_summary,
            "multiframe_qos_floors": {
                "steady": MULTIFRAME_QOS_FLOORS[0],
                "weak3": MULTIFRAME_QOS_FLOORS[1],
                "worst": MULTIFRAME_QOS_FLOORS[2],
            },
            "minimum_multiframe_window_with_all_episode_worst_pass": (
                min(all_episode_pass_windows)
                if all_episode_pass_windows else None
            ),
            "convergence": {
                "steady_converged_rate": float(np.mean([
                    e["closed_loop_convergence"]["steady"]["converged"]
                    for e in episodes
                ])),
                "weak3_converged_rate": float(np.mean([
                    e["closed_loop_convergence"]["weak3"]["converged"]
                    for e in episodes
                ])),
                "worst_converged_rate": float(np.mean([
                    e["closed_loop_convergence"]["worst"]["converged"]
                    for e in episodes
                ])),
                "joint_qos_terminal_hold_rate_mean": float(np.mean([
                    e["closed_loop_convergence"]["joint_qos"]["terminal_hold_rate"]
                    for e in episodes
                ])),
                "joint_qos_ever_sustained_rate": float(np.mean([
                    e["closed_loop_convergence"]["joint_qos"]["ever_sustained"]
                    for e in episodes
                ])),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    from uav_isac.governance import assert_managed_executor
    assert_managed_executor("pilot")

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
        "--include-timing-trace", action="store_true",
        help="include lightweight per-frame timing without golden-state traces")
    parser.add_argument(
        "--teacher-output", default=None,
        help="directly export a native-objective teacher NPZ in memory")
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress stdout JSON (use with --output for large runs)")
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
        carrier_period=args.carrier_period,
        formal=args.formal,
        frames=args.frames,
        fault_inject_power_timeout_frame=(
            args.fault_inject_power_timeout_frame),
        fault_inject_power_timeout_duration=(
            args.fault_inject_power_timeout_duration),
        include_trace=bool(args.include_trace or args.teacher_output),
        include_timing_trace=bool(args.include_timing_trace),
    )
    if args.teacher_output:
        from tools.export_predictive_teacher_dataset import (
            export_predictive_teacher_payload,
        )
        teacher_summary = export_predictive_teacher_payload(
            report, args.teacher_output)
        # The compact NPZ is now the trace artifact. Keep the JSON report
        # operationally useful without duplicating hundreds of MB.
        for episode in report["episodes"]:
            episode.pop("trace", None)
        report["teacher_dataset"] = teacher_summary
    rendered = json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False)
    if not args.quiet:
        print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
