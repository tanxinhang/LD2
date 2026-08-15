#!/usr/bin/env python
"""Audit causally routed B=2 dual-guided structural repair on frozen traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_power_repair_transport_trace import (  # noqa: E402
    _coefficient_from_trace,
    _communication_model,
    _episode_mean,
    _worst_pd,
)
from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _ordered_unique,
    _recorded_power,
)
from tools.audit_structure_teacher_trace import (  # noqa: E402
    _infer_observation_slices,
)
from uav_isac.coordination.bottleneck_router import route_isac_repair  # noqa: E402
from uav_isac.coordination.bounded_repetition import (  # noqa: E402
    certify_bounded_repetition_path,
)
from uav_isac.coordination.protocol_fingerprint import (  # noqa: E402
    fingerprint_controller_implementation,
    fingerprint_protocol_implementation,
)
from uav_isac.coordination.dual_guided_structure_repair import (  # noqa: E402
    dual_guided_atomic_structure_repair,
    enumerate_dual_guided_atomic_candidates,
)
from uav_isac.coordination.geometry_gain_predictor import (  # noqa: E402
    predict_edge_gain_tensor_from_geometry,
)
from uav_isac.coordination.local_exchange_oracle import (  # noqa: E402
    role_owner_from_structure,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    distributed_column_generation_maxmin_power,
    fixed_owner_gain_matrix,
    relaxed_same_geometry_target_ceiling,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.power_repair_transport import (  # noqa: E402
    PowerRepairWireLayout,
    certify_power_repair_transport,
)
from uav_isac.coordination.owner_proposal_transport import (  # noqa: E402
    OwnerProposalWireLayout,
    RankedOwnerProposal,
)
from uav_isac.coordination.digest_rendezvous import (  # noqa: E402
    StructureDigestRendezvousLayout,
    certify_deferred_horizon_prefix,
    certify_rendezvous_candidate_suffix,
    certify_structure_digest_rendezvous,
)
from uav_isac.coordination.owner_parallel_runtime import (  # noqa: E402
    execute_owner_jobs_concurrently,
)
from uav_isac.coordination.owner_local_physics import (  # noqa: E402
    advance_owner_local_kinematics,
    decode_owner_local_kinematics,
    owner_local_dd_effectiveness,
    owner_local_dd_effectiveness_bounds,
    owner_local_horizon_coefficient_bounds,
    predict_coefficients_from_lagged_feedback,
    propagate_owner_local_horizon,
    reciprocal_token_candidate_mask,
    update_target_invariant_cache,
)
from uav_isac.coordination.horizon_atomic_repair import (  # noqa: E402
    HorizonCandidateResources,
    HorizonCoefficientEnvelope,
    HorizonLagrangePrices,
    HorizonPowerProtocol,
    HorizonResourceLimits,
    horizon_common_mix_proxy,
    rank_atomic_horizon_repairs,
)
from uav_isac.coordination.structure_sequence_transport import (  # noqa: E402
    certify_top1_structure_sequence_transport,
)
from uav_isac.coordination.target_invariant_transport import (  # noqa: E402
    TargetInvariantWireLayout,
    decode_owner_target_invariant_tokens,
    encode_owner_target_invariant_tokens,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)
from uav_isac.evaluation.episode_joint_conformal import (  # noqa: E402
    split_conformal_upper,
)
from uav_isac.evaluation.self_normalized_feedback import (  # noqa: E402
    certified_targetwise_feedback_decision,
)
from uav_isac.evaluation.physics_interval_gate import (  # noqa: E402
    certified_physics_interval_decision,
)
from uav_isac.evaluation.horizon_transition_gate import (  # noqa: E402
    CertificateRiskBudget,
    risk_budgeted_certificate_union,
    validate_resource_epoch_risk_budget,
)
from uav_isac.evaluation.horizon_future_audit import (  # noqa: E402
    consecutive_horizon_indices,
    evaluate_horizon_future_outcome,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    delivered_target_tokens,
)
from uav_isac.evaluation.compute_energy_epoch_wiring import (  # noqa: E402
    bind_compute_energy_epoch,
)
from uav_isac.evaluation.runtime_latency_epoch_wiring import (  # noqa: E402
    bind_runtime_latency_epoch,
)
from uav_isac.evaluation.finite_sample_feasibility import (  # noqa: E402
    clopper_pearson_one_sided_upper,
    conformal_sample_requirement,
    zero_failure_validation_requirement,
)
from uav_isac.evaluation.shadow_horizon_router import (  # noqa: E402
    ShadowBranchObservation,
    arbitrate_shadow_horizon,
    isac_action_digest,
    isac_structure_digest,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    trace_path: Path,
    config_path: Path,
    *,
    seed_limit: int,
    rounds: int,
    price_bits: int,
    feedback_bits: int,
    top_m: int,
    structure_steps: int,
    qos_floor: float,
    snr_margin_db: float,
    latency_margin_s: float,
    certify_structure_sequence: bool = False,
    proxy_mode: str = "feasible_mix",
    optimism_weight: float = 0.0,
    owner_proposal_transport: bool = False,
    observability_mode: str = "analytic_full",
    unknown_edge_lower_log_margin: float | None = None,
    unknown_edge_upper_log_margin: float | None = None,
    feedback_normalized_margin: float | None = None,
    slow_geometry_fast_fallback: bool = False,
    persistent_closed_loop: bool = False,
    controller_normalized_margin: float | None = None,
    target_invariant_cache_max_age_frames: int = 0,
    persistent_rf_recourse_from_trace: bool = False,
    owner_dd_covariance_radius: float = 0.0,
    owner_dd_additive_margin: float | None = None,
    action_aligned_owner_state: bool = False,
    controller_certificate_mode: str = "feedback",
    target_invariant_token_transport: bool = False,
    controller_total_miscoverage: float | None = None,
    controller_feedback_miscoverage: float | None = None,
    controller_physics_miscoverage: float | None = None,
    controller_link_miscoverage: float | None = None,
    controller_runtime_miscoverage: float | None = None,
    controller_energy_miscoverage: float | None = None,
    horizon_diagnostic_steps: int = 0,
    horizon_miscoverage: float | None = None,
    horizon_discount: float = 1.0,
    horizon_target_acceleration_std_mps2: float = 0.0,
    horizon_switch_price: float = 0.0,
    horizon_bit_price: float = 0.0,
    horizon_latency_price: float = 0.0,
    horizon_energy_price: float = 0.0,
    horizon_proposal_ranking: str = "myopic",
    horizon_ranking_comm_reserve_w: float = 0.25,
    horizon_weak_target_count: int = 2,
    horizon_local_shortlist_per_owner: int = 1,
    horizon_oracle_diagnostics: bool = False,
    horizon_ranking_rounds: int = 0,
    horizon_residual_log_margin: float = 0.0,
    horizon_paired_deflection_gate: bool = False,
    horizon_common_power_plan: bool = False,
    horizon_concurrent_owner_workers: int = 0,
    shadow_logical_cpu_power_w: float | None = None,
    shadow_max_control_energy_j: float | None = None,
    shadow_reconcile_horizon_power: bool = False,
    horizon_digest_rendezvous: bool = False,
    horizon_digest_rendezvous_reuse_structure_commit: bool = False,
    horizon_digest_rendezvous_defer_top1_transport: bool = False,
    horizon_set_membership_verifier: bool = False,
    horizon_reuse_primal_master_duals: bool = False,
    horizon_network_repetition_count: int = 1,
    horizon_network_excess_queue_bound_s: float = 0.0,
    horizon_network_calibration_epoch: Mapping[str, object] | None = None,
    shadow_compute_energy_calibration_epoch: (
        Mapping[str, object] | None) = None,
    shadow_runtime_latency_calibration_epoch: (
        Mapping[str, object] | None) = None,
    shadow_current_hardware_id: str | None = None,
    shadow_current_runtime_id: str | None = None,
) -> dict[str, object]:
    if bool(certify_structure_sequence) and int(top_m) != 1:
        raise ValueError(
            "full structure-sequence transport currently requires Top-1")
    if bool(owner_proposal_transport) and not bool(certify_structure_sequence):
        raise ValueError(
            "owner proposal transport requires full structure-sequence transport")
    if bool(horizon_digest_rendezvous) and (
        int(horizon_diagnostic_steps) <= 0
        or not bool(shadow_reconcile_horizon_power)
        or not bool(certify_structure_sequence)
    ):
        raise ValueError(
            "digest rendezvous requires the horizon, structure sequence and "
            "structure-consensus shadow route")
    if (
        bool(horizon_digest_rendezvous_reuse_structure_commit)
        and not bool(horizon_digest_rendezvous)
    ):
        raise ValueError(
            "structure-commit reuse requires digest rendezvous")
    if (
        bool(horizon_digest_rendezvous_defer_top1_transport)
        and not bool(horizon_digest_rendezvous)
    ):
        raise ValueError("deferred Top-1 transport requires digest rendezvous")
    if bool(horizon_set_membership_verifier) and (
        not bool(horizon_digest_rendezvous)
        or not bool(horizon_digest_rendezvous_defer_top1_transport)
    ):
        raise ValueError(
            "set-membership verification requires deferred digest rendezvous")
    network_repetitions = int(horizon_network_repetition_count)
    network_queue_bound = float(horizon_network_excess_queue_bound_s)
    if network_repetitions < 1:
        raise ValueError("network repetition count must be positive")
    if np.isnan(network_queue_bound) or network_queue_bound < 0.0:
        raise ValueError(
            "network excess queue bound must be non-negative and not NaN")
    if (
        network_repetitions != 1 or network_queue_bound != 0.0
    ) and not bool(horizon_digest_rendezvous):
        raise ValueError(
            "network repetition envelope requires digest rendezvous")
    network_calibration_metadata = None
    if horizon_network_calibration_epoch is not None:
        calibration = horizon_network_calibration_epoch
        recommended = calibration.get("recommended_controller_parameters")
        epoch_metadata = calibration.get("epoch")
        provenance = calibration.get("provenance")
        authority = calibration.get("authority")
        if not all(isinstance(value, Mapping) for value in (
            recommended, epoch_metadata, provenance, authority,
        )):
            raise ValueError("network calibration epoch has invalid structure")
        if not bool(recommended.get("usable_for_shadow_controller", False)):
            raise ValueError("network calibration epoch is not finite/usable")
        calibrated_repetitions = int(recommended[
            "horizon_network_repetition_count"])
        calibrated_queue = float(recommended[
            "horizon_network_excess_queue_bound_s"])
        if calibrated_repetitions != network_repetitions or not np.isclose(
            calibrated_queue, network_queue_bound, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError(
                "explicit network R/J disagree with the frozen calibration")
        if (
            not bool(epoch_metadata.get("finite", False))
            or not np.isfinite(calibrated_queue)
            or calibrated_queue < 0.0
        ):
            raise ValueError("network calibration parameters are not finite")
        validation = calibration.get("validation")
        validation_present = isinstance(validation, Mapping)
        validation_ids: list[str] = []
        validation_count = 0
        validation_failure_count = 0
        exact_validation_upper = None
        validation_supports_declared_risk = False
        if validation_present:
            validation_ids = [
                str(value) for value in validation.get(
                    "validation_episode_ids", ())]
            validation_count = int(validation.get("episode_count", -1))
            failure_ids = [
                str(value) for value in validation.get(
                    "joint_failure_episode_ids", ())]
            validation_failure_count = int(validation.get(
                "joint_failure_count", -1))
            if (
                validation_count <= 0
                or validation_count != len(validation_ids)
                or len(set(validation_ids)) != len(validation_ids)
                or validation_failure_count != len(failure_ids)
                or len(set(failure_ids)) != len(failure_ids)
                or not set(failure_ids).issubset(set(validation_ids))
            ):
                raise ValueError(
                    "network validation episode/failure manifest is invalid")
            exact_validation_upper = clopper_pearson_one_sided_upper(
                validation_failure_count, validation_count)
            reported_exact_upper = float(validation.get(
                "joint_failure_rate_clopper_pearson_one_sided95_upper", -1.0))
            if not np.isclose(
                exact_validation_upper,
                reported_exact_upper,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "network validation exact upper is inconsistent")
            validation_supports_declared_risk = bool(
                exact_validation_upper
                <= float(epoch_metadata["total_miscoverage"]) + 1.0e-15)
            if bool(validation.get(
                "supports_declared_miscoverage_at_one_sided95", False
            )) != validation_supports_declared_risk:
                raise ValueError(
                    "network validation support flag is inconsistent")
        empirical_hardware_source = bool(
            str(provenance.get("source_kind")) == "hardware_u2u")
        link_certificate_candidate = bool(
            bool(epoch_metadata.get("finite", False))
            and empirical_hardware_source
            and validation_present
            and validation_supports_declared_risk
        )
        authority_checks = (
            ("independent_validation_present", validation_present),
            ("validation_supports_declared_risk",
             validation_supports_declared_risk),
            ("empirical_hardware_source", empirical_hardware_source),
            ("link_certificate_candidate", link_certificate_candidate),
        )
        for field, expected in authority_checks:
            if bool(authority.get(field, False)) != bool(expected):
                raise ValueError(
                    f"network authority field {field} is inconsistent")
        network_calibration_metadata = {
            "epoch_id": str(epoch_metadata.get("epoch_id")),
            "source_kind": str(provenance.get("source_kind")),
            "protocol_implementation_sha256": str(
                provenance.get("protocol_implementation_sha256", "")),
            "input": provenance.get("input"),
            "artifact": provenance.get("calibration_artifact"),
            "joint_coverage_floor_union_bound": float(
                epoch_metadata["joint_coverage_floor_union_bound"]),
            "total_miscoverage": float(
                epoch_metadata["total_miscoverage"]),
            "erasure_miscoverage": float(
                epoch_metadata["erasure_miscoverage"]),
            "queue_miscoverage": float(
                epoch_metadata["queue_miscoverage"]),
            "calibration_episode_count": int(
                epoch_metadata["calibration_episode_count"]),
            "calibration_episode_ids": list(
                epoch_metadata.get("calibration_episode_ids", ())),
            "training_episode_ids": list(
                epoch_metadata.get("training_episode_ids", ())),
            "validation_episode_ids": validation_ids,
            "validation_episode_count": int(validation_count),
            "validation_failure_count": int(validation_failure_count),
            "validation_failure_rate_exact_one_sided95_upper": (
                exact_validation_upper),
            "validation_supports_declared_risk": bool(
                validation_supports_declared_risk),
            "independent_validation_present": bool(validation_present),
            "empirical_hardware_source": bool(empirical_hardware_source),
            "link_certificate_candidate": bool(link_certificate_candidate),
        }
    if (
        shadow_compute_energy_calibration_epoch is not None
        and shadow_logical_cpu_power_w is not None
    ):
        raise ValueError(
            "hardware package-energy calibration and logical CPU-power "
            "sensitivity model are mutually exclusive")
    compute_energy_calibration_metadata = None
    shared_package_energy_bound_j = 0.0
    runtime_latency_calibration_metadata = None
    shared_complete_compute_latency_bound_s = 0.0
    current_shadow_hardware_id = (
        None if shadow_current_hardware_id is None
        else str(shadow_current_hardware_id).strip())
    current_shadow_runtime_id = (
        None if shadow_current_runtime_id is None
        else str(shadow_current_runtime_id).strip())
    if (
        shadow_compute_energy_calibration_epoch is not None
        or shadow_runtime_latency_calibration_epoch is not None
    ) and (
        not current_shadow_hardware_id or not current_shadow_runtime_id
    ):
        raise ValueError(
            "calibrated shadow resources require explicit current hardware "
            "and runtime IDs")
    observation_mode = str(observability_mode)
    if observation_mode not in (
        "analytic_full", "selected_lag1", "selected_lag1_conformal",
        "owner_local_conformal",
    ):
        raise ValueError(
            "unsupported observability_mode")
    lower_log_margin = (
        0.0 if unknown_edge_lower_log_margin is None
        else float(unknown_edge_lower_log_margin)
    )
    upper_log_margin = (
        0.0 if unknown_edge_upper_log_margin is None
        else float(unknown_edge_upper_log_margin)
    )
    if observation_mode in (
        "selected_lag1_conformal", "owner_local_conformal",
    ) and (
        unknown_edge_lower_log_margin is None
        or unknown_edge_upper_log_margin is None
    ):
        raise ValueError(
            "conformal observability requires frozen lower and upper margins")
    if (
        not np.isfinite(lower_log_margin) or lower_log_margin < 0.0
        or not np.isfinite(upper_log_margin) or upper_log_margin < 0.0
    ):
        raise ValueError("unknown-edge log margins must be finite non-negative")
    normalized_margin = (
        None if feedback_normalized_margin is None
        else float(feedback_normalized_margin)
    )
    if normalized_margin is not None and (
        not np.isfinite(normalized_margin) or normalized_margin < 0.0
    ):
        raise ValueError(
            "feedback_normalized_margin must be finite and non-negative")
    if normalized_margin is not None and observation_mode != "owner_local_conformal":
        raise ValueError(
            "target reserve requires owner_local_conformal observability")
    controller_margin = (
        None if controller_normalized_margin is None
        else float(controller_normalized_margin)
    )
    if bool(persistent_closed_loop) and (
        observation_mode != "owner_local_conformal"
        or controller_margin is None
        or not np.isfinite(controller_margin)
        or controller_margin < 0.0
    ):
        raise ValueError(
            "persistent closed loop requires owner-local observability and "
            "a finite non-negative controller margin")
    if bool(persistent_rf_recourse_from_trace) and not bool(
        persistent_closed_loop
    ):
        raise ValueError("RF recourse requires persistent closed-loop replay")
    certificate_mode = str(controller_certificate_mode)
    if certificate_mode not in (
        "feedback", "feedback_or_physics", "risk_budgeted_union",
    ):
        raise ValueError("unsupported controller certificate mode")
    if certificate_mode != "feedback" and not bool(persistent_closed_loop):
        raise ValueError(
            "physical controller certificate requires persistent replay")
    cache_max_age = int(target_invariant_cache_max_age_frames)
    if cache_max_age < 0:
        raise ValueError(
            "target invariant cache max age must be non-negative")
    if cache_max_age > 0 and observation_mode != "owner_local_conformal":
        raise ValueError(
            "target invariant cache requires owner-local observability")
    if bool(target_invariant_token_transport) and (
        cache_max_age <= 0
        or cache_max_age > 255
        or observation_mode != "owner_local_conformal"
        or not bool(certify_structure_sequence)
    ):
        raise ValueError(
            "target-invariant token transport requires an enabled owner-local "
            "cache representable in 8 age bits and full structure-sequence "
            "transport")
    controller_risk_budget = None
    if certificate_mode == "risk_budgeted_union":
        components = (
            controller_total_miscoverage,
            controller_feedback_miscoverage,
            controller_physics_miscoverage,
            controller_link_miscoverage,
            controller_runtime_miscoverage,
            controller_energy_miscoverage,
        )
        if any(value is None for value in components):
            raise ValueError(
                "risk-budgeted union requires explicit total, feedback, "
                "physics, link, runtime and energy miscoverage allocations")
        controller_risk_budget = CertificateRiskBudget(
            total=float(controller_total_miscoverage),
            feedback=float(controller_feedback_miscoverage),
            physics=float(controller_physics_miscoverage),
            link=float(controller_link_miscoverage),
            runtime=float(controller_runtime_miscoverage),
            energy=float(controller_energy_miscoverage),
        )
    horizon_steps = int(horizon_diagnostic_steps)
    horizon_residual_margin = float(horizon_residual_log_margin)
    if (
        not np.isfinite(horizon_residual_margin)
        or horizon_residual_margin < 0.0
    ):
        raise ValueError(
            "horizon residual log margin must be finite and non-negative")
    if bool(horizon_paired_deflection_gate) and any(
        value > 0.0 for value in (
            float(horizon_switch_price),
            float(horizon_bit_price),
            float(horizon_latency_price),
            float(horizon_energy_price),
        )
    ):
        raise ValueError(
            "paired Deflection gate requires all probability-unit prices "
            "to be zero")
    horizon_ranking_mode = str(horizon_proposal_ranking)
    if horizon_ranking_mode not in (
        "myopic", "common_mix_sum", "noharm_slack",
        "finite_horizon_margin", "screened_finite_margin",
        "distributed_screened_margin",
    ):
        raise ValueError(
            "horizon proposal ranking must be myopic, common_mix_sum, "
            "noharm_slack, finite_horizon_margin or "
            "screened_finite_margin/distributed_screened_margin")
    horizon_ranking_reserve = float(horizon_ranking_comm_reserve_w)
    if (
        not np.isfinite(horizon_ranking_reserve)
        or not 0.0 <= horizon_ranking_reserve < 1.0
    ):
        raise ValueError(
            "horizon ranking communication reserve must lie in [0,1)")
    horizon_weak_targets = int(horizon_weak_target_count)
    if horizon_weak_targets < 1:
        raise ValueError("horizon weak-target count must be positive")
    horizon_local_shortlist = int(horizon_local_shortlist_per_owner)
    if horizon_local_shortlist < 1:
        raise ValueError("horizon local shortlist must be positive")
    horizon_rank_rounds = int(horizon_ranking_rounds)
    if horizon_rank_rounds < 0:
        raise ValueError("horizon ranking rounds must be non-negative")
    if horizon_rank_rounds == 0:
        horizon_rank_rounds = int(rounds)
    horizon_owner_workers = int(horizon_concurrent_owner_workers)
    if horizon_owner_workers < 0:
        raise ValueError("concurrent owner worker count must be non-negative")
    if (
        horizon_owner_workers > 0
        and horizon_ranking_mode != "distributed_screened_margin"
    ):
        raise ValueError(
            "concurrent owner workers require distributed_screened_margin")
    shadow_cpu_power = (
        None if shadow_logical_cpu_power_w is None
        else float(shadow_logical_cpu_power_w)
    )
    if shadow_cpu_power is not None and (
        not np.isfinite(shadow_cpu_power) or shadow_cpu_power < 0.0
    ):
        raise ValueError(
            "shadow logical CPU power must be finite non-negative")
    shadow_energy_limit = (
        None if shadow_max_control_energy_j is None
        else float(shadow_max_control_energy_j)
    )
    if shadow_energy_limit is not None and (
        not np.isfinite(shadow_energy_limit) or shadow_energy_limit < 0.0
    ):
        raise ValueError(
            "shadow control-energy limit must be finite non-negative")
    if horizon_steps < 0:
        raise ValueError("horizon diagnostic steps must be non-negative")
    horizon_alpha = None
    if horizon_steps > 0:
        if (
            observation_mode != "owner_local_conformal"
            or cache_max_age <= 0
            or not bool(target_invariant_token_transport)
            or not bool(certify_structure_sequence)
            or int(structure_steps) != 1
            or horizon_miscoverage is None
        ):
            raise ValueError(
                "horizon diagnostics require owner-local observability, "
                "transported target invariants, full sequence transport and "
                "an explicit miscoverage; only one receding-horizon atomic "
                "step may be committed")
        horizon_alpha = float(horizon_miscoverage)
        if not np.isfinite(horizon_alpha) or not 0.0 < horizon_alpha < 1.0:
            raise ValueError("horizon miscoverage must lie in (0,1)")
        if (
            controller_risk_budget is not None
            and horizon_alpha
            > float(controller_risk_budget.physics) + 1.0e-15
        ):
            raise ValueError(
                "horizon physics miscoverage exceeds its risk allocation")
    horizon_prices = HorizonLagrangePrices(
        per_switched_edge=float(horizon_switch_price),
        per_over_air_bit=float(horizon_bit_price),
        per_latency_second=float(horizon_latency_price),
        per_control_joule=float(horizon_energy_price),
    )
    horizon_gamma = float(horizon_discount)
    horizon_sigma_a = float(horizon_target_acceleration_std_mps2)
    if (
        not np.isfinite(horizon_gamma) or not 0.0 < horizon_gamma <= 1.0
        or not np.isfinite(horizon_sigma_a) or horizon_sigma_a < 0.0
    ):
        raise ValueError(
            "horizon discount/target acceleration must be finite and valid")
    dd_covariance_radius = float(owner_dd_covariance_radius)
    dd_support_margin = (
        lower_log_margin
        if owner_dd_additive_margin is None
        else float(owner_dd_additive_margin)
    )
    if (
        not np.isfinite(dd_covariance_radius) or dd_covariance_radius < 0.0
        or not np.isfinite(dd_support_margin) or dd_support_margin < 0.0
    ):
        raise ValueError("owner-local DD radius/margin must be non-negative")
    if (
        (dd_covariance_radius > 0.0 or owner_dd_additive_margin is not None)
        and observation_mode != "owner_local_conformal"
    ):
        raise ValueError(
            "owner-local DD uncertainty options require owner-local mode")
    if (
        observation_mode in (
            "selected_lag1_conformal", "owner_local_conformal",
        )
        and not np.isclose(lower_log_margin, upper_log_margin, rtol=0.0,
                           atol=1.0e-15)
    ):
        raise ValueError(
            "conformal observability requires one joint two-sided margin")
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg, comm_model = _communication_model(config_path)
    if network_calibration_metadata is not None:
        current_protocol_fingerprint = fingerprint_protocol_implementation(
            config_path, workspace_root=ROOT)
        frozen_protocol_fingerprint = str(
            network_calibration_metadata[
                "protocol_implementation_sha256"])
        if current_protocol_fingerprint.sha256 != frozen_protocol_fingerprint:
            raise ValueError(
                "network epoch protocol/config fingerprint mismatch: "
                f"frozen={frozen_protocol_fingerprint}, "
                f"current={current_protocol_fingerprint.sha256}")
        network_calibration_metadata[
            "current_protocol_implementation_sha256"] = (
                current_protocol_fingerprint.sha256)
        network_calibration_metadata["current_config_chain"] = list(
            current_protocol_fingerprint.config_chain)
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    if network_calibration_metadata is not None:
        controller_episode_ids = {str(int(seed)) for seed in seed_order}
        calibration_source_ids = set(str(value) for value in (
            network_calibration_metadata["training_episode_ids"]
            + network_calibration_metadata["calibration_episode_ids"]
            + network_calibration_metadata["validation_episode_ids"]
        ))
        overlap = controller_episode_ids & calibration_source_ids
        if overlap:
            raise ValueError(
                "network epoch/controller evaluation episode leakage: "
                f"{sorted(overlap)}")
    if shadow_compute_energy_calibration_epoch is not None:
        current_controller_fingerprint = fingerprint_controller_implementation(
            config_path, workspace_root=ROOT)
        energy_binding = bind_compute_energy_epoch(
            shadow_compute_energy_calibration_epoch,
            current_controller_sha256=current_controller_fingerprint.sha256,
            controller_episode_ids=tuple(str(int(seed)) for seed in seed_order),
        )
        compute_energy_calibration_metadata = dict(energy_binding.metadata)
        if (
            compute_energy_calibration_metadata["hardware_id"]
            != current_shadow_hardware_id
            or compute_energy_calibration_metadata["runtime_id"]
            != current_shadow_runtime_id
        ):
            raise ValueError(
                "compute-energy epoch hardware/runtime identity mismatch")
        compute_energy_calibration_metadata["current_hardware_id"] = (
            current_shadow_hardware_id)
        compute_energy_calibration_metadata["current_runtime_id"] = (
            current_shadow_runtime_id)
        compute_energy_calibration_metadata["current_config_chain"] = list(
            current_controller_fingerprint.config_chain)
        shared_package_energy_bound_j = float(
            energy_binding.package_energy_bound_j)
    if shadow_runtime_latency_calibration_epoch is not None:
        current_runtime_fingerprint = fingerprint_controller_implementation(
            config_path, workspace_root=ROOT)
        runtime_binding = bind_runtime_latency_epoch(
            shadow_runtime_latency_calibration_epoch,
            current_controller_sha256=current_runtime_fingerprint.sha256,
            controller_episode_ids=tuple(str(int(seed)) for seed in seed_order),
        )
        runtime_latency_calibration_metadata = dict(runtime_binding.metadata)
        if (
            runtime_latency_calibration_metadata["hardware_id"]
            != current_shadow_hardware_id
            or runtime_latency_calibration_metadata["runtime_id"]
            != current_shadow_runtime_id
        ):
            raise ValueError(
                "runtime epoch hardware/runtime identity mismatch")
        runtime_latency_calibration_metadata["current_hardware_id"] = (
            current_shadow_hardware_id)
        runtime_latency_calibration_metadata["current_runtime_id"] = (
            current_shadow_runtime_id)
        runtime_latency_calibration_metadata["current_config_chain"] = list(
            current_runtime_fingerprint.config_chain)
        shared_complete_compute_latency_bound_s = float(
            runtime_binding.complete_compute_latency_bound_s)
    resource_risk_ledger = None
    resource_risk_budget_complete = False
    if controller_risk_budget is not None:
        resource_risk_ledger = validate_resource_epoch_risk_budget(
            controller_risk_budget,
            link_miscoverage=(
                None if network_calibration_metadata is None
                else float(network_calibration_metadata["total_miscoverage"])
            ),
            runtime_miscoverage=(
                None if runtime_latency_calibration_metadata is None
                else float(runtime_latency_calibration_metadata["miscoverage"])
            ),
            energy_miscoverage=(
                None if compute_energy_calibration_metadata is None
                else float(compute_energy_calibration_metadata["miscoverage"])
            ),
        )
        resource_risk_budget_complete = bool(resource_risk_ledger.complete)
    resource_sample_requirements = []
    if network_calibration_metadata is not None:
        link_calibration_count = int(
            network_calibration_metadata["calibration_episode_count"])
        resource_sample_requirements.extend((
            conformal_sample_requirement(
                "link_erasure",
                miscoverage=float(network_calibration_metadata[
                    "erasure_miscoverage"]),
                calibration_episode_count=link_calibration_count,
            ),
            conformal_sample_requirement(
                "link_queue",
                miscoverage=float(network_calibration_metadata[
                    "queue_miscoverage"]),
                calibration_episode_count=link_calibration_count,
            ),
        ))
        if int(network_calibration_metadata[
            "validation_episode_count"
        ]) > 0:
            resource_sample_requirements.append(
                zero_failure_validation_requirement(
                    "link_joint",
                    miscoverage=float(network_calibration_metadata[
                        "total_miscoverage"]),
                    validation_episode_count=int(network_calibration_metadata[
                        "validation_episode_count"]),
                ))
    if runtime_latency_calibration_metadata is not None:
        runtime_alpha = float(
            runtime_latency_calibration_metadata["miscoverage"])
        resource_sample_requirements.append(
            conformal_sample_requirement(
                "runtime",
                miscoverage=runtime_alpha,
                calibration_episode_count=len(
                    runtime_latency_calibration_metadata[
                        "calibration_episode_ids"]),
            ))
        runtime_validation_count = len(
            runtime_latency_calibration_metadata["validation_episode_ids"])
        if runtime_validation_count > 0:
            resource_sample_requirements.append(
                zero_failure_validation_requirement(
                    "runtime",
                    miscoverage=runtime_alpha,
                    validation_episode_count=runtime_validation_count,
                ))
    if compute_energy_calibration_metadata is not None:
        energy_alpha = float(
            compute_energy_calibration_metadata["miscoverage"])
        resource_sample_requirements.append(
            conformal_sample_requirement(
                "energy",
                miscoverage=energy_alpha,
                calibration_episode_count=len(
                    compute_energy_calibration_metadata[
                        "calibration_episode_ids"]),
            ))
        energy_validation_count = len(
            compute_energy_calibration_metadata["validation_episode_ids"])
        if energy_validation_count > 0:
            resource_sample_requirements.append(
                zero_failure_validation_requirement(
                    "energy",
                    miscoverage=energy_alpha,
                    validation_episode_count=energy_validation_count,
                ))
    resource_finite_sample_plan_complete = bool(
        network_calibration_metadata is not None
        and runtime_latency_calibration_metadata is not None
        and compute_energy_calibration_metadata is not None
        and len(resource_sample_requirements) == 7
        and all(requirement.attainable_in_best_case
                for requirement in resource_sample_requirements)
    )
    shadow_resource_accounting_complete = bool(
        network_calibration_metadata is not None
        and bool(network_calibration_metadata["link_certificate_candidate"])
        and compute_energy_calibration_metadata is not None
        and bool(compute_energy_calibration_metadata[
            "compute_energy_certificate_candidate"])
        and runtime_latency_calibration_metadata is not None
        and bool(runtime_latency_calibration_metadata[
            "runtime_latency_certificate_candidate"])
        and shadow_energy_limit is not None
        and resource_risk_budget_complete
        and resource_finite_sample_plan_complete
    )
    indices = np.flatnonzero(resolved & np.isin(seeds, seed_order))
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    reports = int(np.asarray(data["reports_per_receiver"]).reshape(-1)[0])
    layout = PowerRepairWireLayout(
        num_agents=K,
        num_targets=Q,
        rounds=int(rounds),
        header_bits=int(cfg.marl.comm_header_bits),
        price_bits=int(price_bits),
        deflection_bits=int(feedback_bits),
    )
    proposal_layout = OwnerProposalWireLayout(
        num_agents=K,
        num_targets=Q,
        target_pair_limit=pair_limit,
        target_budget=2,
        proposal_budget=(
            max(2, horizon_weak_targets)
            if horizon_steps > 0 else 2),
        header_bits=int(cfg.marl.comm_header_bits),
    )
    rendezvous_layout = StructureDigestRendezvousLayout(
        num_agents=K,
        num_targets=Q,
        target_pair_limit=pair_limit,
        header_bits=int(cfg.marl.comm_header_bits),
        digest_bits=256,
    )
    invariant_layout = TargetInvariantWireLayout(
        num_agents=K,
        num_targets=Q,
        header_bits=int(cfg.marl.comm_header_bits),
    )
    previous_coefficient: dict[int, np.ndarray] = {}
    previous_uav_positions: dict[int, np.ndarray] = {}
    previous_target_states: dict[int, np.ndarray] = {}
    previous_observed_mask: dict[int, np.ndarray] = {}
    previous_owner_local_state: dict[int, object] = {}
    previous_event_frame: dict[int, int] = {}
    target_invariant_cache: dict[int, object] = {}
    deployed_selected: dict[int, np.ndarray] = {}
    deployed_comm_power: dict[int, np.ndarray] = {}
    deployed_sensing_power: dict[int, np.ndarray] = {}
    deployed_feedback_ema: dict[int, np.ndarray] = {}
    deployed_last_row: dict[int, int] = {}
    deployed_state_version: dict[int, int] = {}
    owner_local_slices = None
    owner_local_candidate = None
    if observation_mode == "owner_local_conformal":
        local_obs = np.asarray(data["local_obs"], dtype=np.float32)
        owner_local_slices = _infer_observation_slices(
            local_obs.shape[-1], K, Q)
        token_visible, _ = delivered_target_tokens(
            local_obs, owner_local_slices)
        owner_local_candidate = reciprocal_token_candidate_mask(token_visible)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index in indices:
        row_id = int(index)
        seed = int(seeds[row_id])
        trace_comm_power, trace_sensing_power, _ = _recorded_power(
            data, row_id)
        trace_selected = np.asarray(
            data["teacher_pair"][row_id], dtype=bool)
        if bool(persistent_closed_loop) and seed in deployed_selected:
            selected = deployed_selected[seed].copy()
            if bool(persistent_rf_recourse_from_trace):
                # Two-timescale replay: the structural graph is the slow
                # persistent state, whereas the local continuous RF proposal
                # is refreshed at every event and remains hard-feasible.
                comm_power = trace_comm_power
                sensing_power = trace_sensing_power
            else:
                comm_power = deployed_comm_power[seed].copy()
                sensing_power = deployed_sensing_power[seed].copy()
        else:
            comm_power = trace_comm_power
            sensing_power = trace_sensing_power
            selected = trace_selected
            if bool(persistent_closed_loop):
                deployed_selected[seed] = selected.copy()
                deployed_comm_power[seed] = comm_power.copy()
                deployed_sensing_power[seed] = sensing_power.copy()
                deployed_state_version[seed] = 0
        coefficient = _coefficient_from_trace(data, row_id, cfg)
        physical_support = (
            np.asarray(data["privileged_candidate"][row_id], dtype=bool)
            & (
                np.asarray(data["privileged_g_dd"][row_id], dtype=np.float64)
                >= float(cfg.detection.g_min)
            )
        )
        support = physical_support
        selected_edges = [tuple(edge) for edge in np.argwhere(selected)]
        true_gain, owners = fixed_owner_gain_matrix(
            coefficient, selected_edges)
        deployed_pd = compute_detection_probabilities(
            np.sum(true_gain * sensing_power, axis=0), p_fa)
        candidate_selected = selected.copy()
        candidate_comm_power = comm_power.copy()
        candidate_sensing_power = sensing_power.copy()
        candidate_realized_pd = deployed_pd.copy()
        if bool(persistent_closed_loop):
            if seed not in deployed_feedback_ema:
                trace_feedback_valid = bool(np.asarray(
                    data["coord_pd_ema_valid"][row_id]).reshape(-1)[0])
                deployed_feedback_ema[seed] = (
                    np.clip(np.asarray(
                        data["coord_pd_ema"][row_id], dtype=np.float64),
                        0.0,
                        1.0,
                    )
                    if trace_feedback_valid else deployed_pd.copy()
                )
            elif seed in deployed_last_row:
                # Replay every intervening simulator frame under the held
                # action.  Motion remains the frozen exogenous trace; only
                # structure, RF allocation and the resulting feedback close
                # the counterfactual loop.
                ema = deployed_feedback_ema[seed].copy()
                alpha = float(getattr(
                    cfg.marl, "coord_reward_ema_alpha", 0.20))
                for intermediate in range(
                    int(deployed_last_row[seed]) + 1, row_id + 1
                ):
                    if int(seeds[intermediate]) != seed:
                        continue
                    intermediate_coefficient = _coefficient_from_trace(
                        data, intermediate, cfg)
                    intermediate_gain, _ = fixed_owner_gain_matrix(
                        intermediate_coefficient,
                        [tuple(edge) for edge in np.argwhere(selected)],
                    )
                    intermediate_sensing_power = sensing_power
                    if bool(persistent_rf_recourse_from_trace):
                        _, intermediate_sensing_power, _ = _recorded_power(
                            data, intermediate)
                    intermediate_pd = compute_detection_probabilities(
                        np.sum(
                            intermediate_gain * intermediate_sensing_power,
                            axis=0),
                        p_fa,
                    )
                    ema = alpha * intermediate_pd + (1.0 - alpha) * ema
                deployed_feedback_ema[seed] = ema
        report: dict[str, object] = {
            "seed": seed,
            "frame": int(frames[row_id]),
            "deployed_worst": float(np.min(deployed_pd)),
            "deployed_pd": deployed_pd.tolist(),
            "causal_available": seed in previous_coefficient,
            "persistent_closed_loop": bool(persistent_closed_loop),
            "persistent_rf_recourse_from_trace": bool(
                persistent_rf_recourse_from_trace),
            "deployed_state_version": int(
                deployed_state_version.get(seed, 0)),
            "trace_structure_matches_deployed": bool(np.array_equal(
                trace_selected, selected)),
        }
        controller_common_started = time.perf_counter()
        controller_common_cpu_started = time.process_time()
        controller_common_preprocessing_s = 0.0
        controller_common_preprocessing_cpu_s = 0.0
        controller_common_instrumented_wall_s = 0.0
        controller_common_instrumented_cpu_s = 0.0
        controller_privileged_audit_s = 0.0
        controller_privileged_audit_cpu_s = 0.0
        current_owner_local_state = None
        cache_elapsed_frames = 0
        cache_available_target_count = 0
        cache_refreshed_target_count = 0
        cache_retained_target_count = 0
        cache_expired_target_count = 0
        cache_max_available_age_frames = 0
        prediction_cached_fallback_count = 0
        current_cache = None
        target_invariant_tokens = ()
        if observation_mode == "owner_local_conformal":
            if owner_local_slices is None or owner_local_candidate is None:
                raise RuntimeError("owner-local observations were not initialized")
            current_owner_local_state = decode_owner_local_kinematics(
                np.asarray(data["local_obs"][row_id], dtype=np.float32),
                owner_local_slices,
                area_size_m=tuple(float(value) for value in (
                    cfg.scenario.region_size)),
                height_m=float(cfg.scenario.height),
            )
            if bool(action_aligned_owner_state):
                current_owner_local_state = advance_owner_local_kinematics(
                    current_owner_local_state,
                    np.asarray(data["delta_p"][row_id], dtype=np.float64),
                    dt_s=float(cfg.scenario.dt),
                    max_speed_mps=float(cfg.uav.v_max),
                    area_size_m=tuple(float(value) for value in (
                        cfg.scenario.region_size)),
                    advance_targets=bool(getattr(
                        cfg.marl, "tracking_enabled", True)),
                )
        if seed in previous_coefficient:
            directly_observed = (
                previous_coefficient[seed] > 0.0
                if observation_mode == "analytic_full"
                else previous_observed_mask[seed]
            )
            if observation_mode == "owner_local_conformal":
                if current_owner_local_state is None:
                    raise RuntimeError("owner-local state is unavailable")
                if cache_max_age > 0:
                    cache_elapsed_frames = (
                        int(frames[row_id])
                        - int(previous_event_frame[seed])
                    )
                    if cache_elapsed_frames < 0:
                        raise ValueError(
                            "per-seed event frames must be non-decreasing")
                    prior_cache = target_invariant_cache.get(seed)
                    prior_version = (
                        np.zeros(Q, dtype=np.int64)
                        if prior_cache is None
                        else np.asarray(
                            prior_cache.version, dtype=np.int64)
                    )
                    current_cache = update_target_invariant_cache(
                        previous_coefficient[seed],
                        previous_observed_mask[seed],
                        previous_owner_local_state[seed],
                        prior_cache=prior_cache,
                        elapsed_frames=cache_elapsed_frames,
                        max_age_frames=cache_max_age,
                    )
                    if bool(target_invariant_token_transport):
                        target_invariant_tokens = (
                            encode_owner_target_invariant_tokens(
                                current_cache,
                                owners,
                                layout=invariant_layout,
                                sent_frame=int(frames[row_id]),
                            )
                        )
                        current_cache = (
                            decode_owner_target_invariant_tokens(
                                target_invariant_tokens,
                                owners,
                                layout=invariant_layout,
                                current_frame=int(frames[row_id]),
                                max_age_frames=cache_max_age,
                                minimum_versions=prior_version,
                            )
                        )
                    target_invariant_cache[seed] = current_cache
                    cache_value = np.asarray(
                        current_cache.target_invariant, dtype=np.float64)
                    cache_age = np.asarray(
                        current_cache.age_frames, dtype=np.int64)
                    cache_version = np.asarray(
                        current_cache.version, dtype=np.int64)
                    cache_available = cache_value > 0.0
                    cache_refreshed = cache_version > prior_version
                    cache_available_target_count = int(np.sum(
                        cache_available))
                    cache_refreshed_target_count = int(np.sum(
                        cache_refreshed))
                    cache_retained_target_count = int(np.sum(
                        cache_available & ~cache_refreshed))
                    cache_expired_target_count = int(np.sum(
                        (cache_value <= 0.0)
                        & (cache_version > 0)
                        & (cache_age > cache_max_age)
                    ))
                    cache_max_available_age_frames = int(max(
                        cache_age[cache_available].tolist(), default=0))
                    # A direct lagged edge is also stale after expiry.  The
                    # target cache follows the same fail-closed age rule.
                    if cache_elapsed_frames > cache_max_age:
                        directly_observed = np.zeros_like(
                            directly_observed, dtype=bool)
                local_prediction = predict_coefficients_from_lagged_feedback(
                    previous_coefficient[seed],
                    directly_observed,
                    previous_owner_local_state[seed],
                    current_owner_local_state,
                    current_support=(
                        np.asarray(owner_local_candidate[row_id], dtype=bool)
                        | selected
                    ),
                    target_invariant_cache=current_cache,
                )
                prediction_coefficient = local_prediction.coefficient_per_watt
                prediction_direct_count = local_prediction.direct_edge_count
                prediction_fallback_count = (
                    local_prediction.target_fallback_edge_count)
                prediction_cached_fallback_count = (
                    local_prediction.cached_target_fallback_edge_count)
                if dd_covariance_radius > 0.0:
                    dd_bounds = owner_local_dd_effectiveness_bounds(
                        current_owner_local_state,
                        carrier_hz=float(cfg.otfs.fc),
                        delta_f_hz=float(cfg.otfs.delta_f),
                        symbol_period_s=float(cfg.otfs.T_sym),
                        delay_bins=int(cfg.otfs.M),
                        doppler_bins=int(cfg.otfs.N),
                        covariance_radius=dd_covariance_radius,
                    )
                    dd_prediction = dd_bounds.point
                    dd_geometry_lower = dd_bounds.lower
                    dd_max_delay_bin_radius = float(np.max(
                        dd_bounds.delay_bin_radius))
                    dd_max_doppler_bin_radius = float(np.max(
                        dd_bounds.doppler_bin_radius))
                else:
                    dd_prediction = owner_local_dd_effectiveness(
                        current_owner_local_state,
                        carrier_hz=float(cfg.otfs.fc),
                        delta_f_hz=float(cfg.otfs.delta_f),
                        symbol_period_s=float(cfg.otfs.T_sym),
                        delay_bins=int(cfg.otfs.M),
                        doppler_bins=int(cfg.otfs.N),
                    )
                    dd_geometry_lower = dd_prediction
                    dd_max_delay_bin_radius = 0.0
                    dd_max_doppler_bin_radius = 0.0
                dd_safe = (
                    dd_geometry_lower - dd_support_margin
                    > float(cfg.detection.g_min)
                )
                dd_admitted = (
                    np.asarray(owner_local_candidate[row_id], dtype=bool)
                    & dd_safe
                )
                selected_provenance_renewal = (
                    selected & directly_observed & dd_safe
                )
                support_provenance = (
                    np.asarray(owner_local_candidate[row_id], dtype=bool)
                    | (selected & directly_observed)
                )
                privileged_started = time.perf_counter()
                privileged_cpu_started = time.process_time()
                dd_false_support_score = float(np.max(np.where(
                    support_provenance
                    & ~physical_support,
                    np.maximum(
                        dd_geometry_lower - float(cfg.detection.g_min), 0.0),
                    0.0,
                )))
                controller_privileged_audit_s += float(
                    time.perf_counter() - privileged_started)
                controller_privileged_audit_cpu_s += float(
                    time.process_time() - privileged_cpu_started)
            else:
                predictor_coefficient = np.where(
                    directly_observed,
                    previous_coefficient[seed],
                    0.0,
                )
                prediction = predict_edge_gain_tensor_from_geometry(
                    predictor_coefficient,
                    previous_uav_positions[seed],
                    np.asarray(
                        data["uav_positions"][row_id], dtype=np.float64),
                    previous_target_states[seed],
                    np.asarray(
                        data["target_states"][row_id], dtype=np.float64),
                    current_support=support,
                )
                prediction_coefficient = prediction.coefficient_per_watt
                prediction_direct_count = prediction.direct_edge_count
                prediction_fallback_count = prediction.fallback_edge_count
                dd_false_support_score = 0.0
            privileged_started = time.perf_counter()
            privileged_cpu_started = time.process_time()
            residual_support = (
                physical_support
                & np.asarray(owner_local_candidate[row_id], dtype=bool)
                if observation_mode == "owner_local_conformal"
                else support
            )
            # Structural search is confined to provenance-bearing lag-1 edge
            # measurements unless a frozen conformal interval is supplied.
            # In conformal mode the same episode-level interval is applied to
            # direct and fallback predictions: geometry transport is a model,
            # not an exact observation, even on a previously excited edge.
            if observation_mode in (
                "selected_lag1_conformal", "owner_local_conformal",
            ):
                coefficient_lower = (
                    prediction_coefficient
                    * np.exp(-lower_log_margin)
                )
                coefficient_upper = (
                    prediction_coefficient
                    * np.exp(upper_log_margin)
                )
                if observation_mode == "owner_local_conformal":
                    structural_support = (
                        (dd_admitted | selected_provenance_renewal)
                        & (coefficient_lower > 0.0)
                    )
                else:
                    structural_support = support & (
                        (coefficient_lower > 0.0) | selected)
                structural_coefficient = np.where(
                    structural_support, coefficient_lower, 0.0)
                coefficient_upper = np.where(
                    structural_support, coefficient_upper, 0.0)
            else:
                structural_support = support & (directly_observed | selected)
                structural_coefficient = np.where(
                    structural_support,
                    prediction_coefficient,
                    0.0,
                )
                coefficient_upper = structural_coefficient
            unknown_prediction_mask = (
                residual_support
                & ~directly_observed
                & (prediction_coefficient > 0.0)
                & (coefficient > 0.0)
            )
            unknown_log_overprediction = (
                float(np.max(np.maximum(np.log(
                    prediction_coefficient[
                        unknown_prediction_mask]
                    / coefficient[unknown_prediction_mask]
                ), 0.0)))
                if np.any(unknown_prediction_mask) else 0.0
            )
            all_prediction_mask = (
                residual_support
                & (prediction_coefficient > 0.0)
                & (coefficient > 0.0)
            )
            all_log_overprediction = (
                float(np.max(np.maximum(np.log(
                    prediction_coefficient[all_prediction_mask]
                    / coefficient[all_prediction_mask]
                ), 0.0)))
                if np.any(all_prediction_mask) else 0.0
            )
            all_log_underprediction = (
                float(np.max(np.maximum(np.log(
                    coefficient[all_prediction_mask]
                    / prediction_coefficient[all_prediction_mask]
                ), 0.0)))
                if np.any(all_prediction_mask) else 0.0
            )
            unpredicted_positive_edge_count = int(np.sum(
                residual_support
                & (coefficient > 0.0)
                & (prediction_coefficient <= 0.0)
            ))
            unknown_log_underprediction = (
                float(np.max(np.maximum(np.log(
                    coefficient[unknown_prediction_mask]
                    / prediction_coefficient[
                        unknown_prediction_mask]
                ), 0.0)))
                if np.any(unknown_prediction_mask) else 0.0
            )
            controller_privileged_audit_s += float(
                time.perf_counter() - privileged_started)
            controller_privileged_audit_cpu_s += float(
                time.process_time() - privileged_cpu_started)
            decision_coefficient = (
                structural_coefficient
                if observation_mode == "owner_local_conformal"
                else prediction_coefficient
            )
            predicted_gain, _ = fixed_owner_gain_matrix(
                decision_coefficient, selected_edges)
            predicted_upper_gain, _ = fixed_owner_gain_matrix(
                coefficient_upper, selected_edges)
            transport = certify_power_repair_transport(
                owners,
                np.asarray(data["uav_positions"][row_id], dtype=np.float64),
                comm_power,
                communication_model=comm_model,
                layout=layout,
                control_period_s=float(cfg.scenario.dt),
                snr_margin_db=float(snr_margin_db),
                latency_margin_s=float(latency_margin_s),
            )
            if not transport.feasible:
                report.update({
                    "causal_route": "hold_unverified",
                    "true_diagnostic_route": "hold_unverified",
                    "structure_triggered": False,
                    "structure_accepted": False,
                    "transport_feasible": False,
                    "transport_reasons": list(transport.reasons),
                })
            else:
                budget = 1.0 - transport.projected_comm_power_w
                physics_noop_pd = compute_detection_probabilities(
                    np.sum(
                        decision_coefficient
                        * selected * sensing_power[:, None, :],
                        axis=(0, 1),
                    ),
                    p_fa,
                )
                feedback_available = bool(
                    observation_mode == "owner_local_conformal"
                    and (
                        seed in deployed_feedback_ema
                        if bool(persistent_closed_loop)
                        else np.asarray(
                            data["coord_pd_ema_valid"][row_id]
                        ).reshape(-1)[0]
                    )
                )
                noop_estimated_pd = (
                    np.clip(
                        deployed_feedback_ema[seed]
                        if bool(persistent_closed_loop)
                        else np.asarray(
                            data["coord_pd_ema"][row_id],
                            dtype=np.float64,
                        ),
                        0.0,
                        1.0,
                    )
                    if feedback_available else physics_noop_pd
                )
                physics_noop_upper_pd = compute_detection_probabilities(
                    np.sum(
                        predicted_upper_gain
                        * sensing_power,
                        axis=0,
                    ),
                    p_fa,
                )
                uncertainty_floor = 1.0e-6
                absolute_uncertainty = (
                    np.maximum(noop_estimated_pd - physics_noop_pd, 0.0)
                    + uncertainty_floor
                )
                target_reserve_pd = None
                target_reserve_deflection = None
                target_reserve_possible = True
                if normalized_margin is not None:
                    noop_lower_for_reserve = np.clip(
                        noop_estimated_pd
                        - normalized_margin * absolute_uncertainty,
                        0.0,
                        1.0,
                    )
                    certified_target_level = np.minimum(
                        noop_lower_for_reserve, float(qos_floor))
                    required_candidate_estimate = (
                        certified_target_level
                        + normalized_margin * absolute_uncertainty
                    )
                    target_reserve_possible = bool(np.all(
                        required_candidate_estimate < 1.0))
                    required_physics_pd = (
                        required_candidate_estimate
                        - noop_estimated_pd
                        + physics_noop_pd
                    )
                    target_reserve_pd = np.clip(
                        required_physics_pd, p_fa, 1.0 - 1.0e-12)
                    target_reserve_deflection = (
                        minimum_deflection_for_detection_probability(
                            target_reserve_pd, p_fa))
                predicted_fixed_exact = solve_fixed_structure_maxmin_power_lp(
                    predicted_upper_gain, budget)
                predicted_relaxed = relaxed_same_geometry_target_ceiling(
                    coefficient_upper, budget)
                fixed_upper_pd = compute_detection_probabilities(
                    predicted_fixed_exact.deflection, p_fa)
                relaxed_upper_pd = compute_detection_probabilities(
                    predicted_relaxed, p_fa)
                if observation_mode == "owner_local_conformal":
                    fixed_route_upper_pd = np.clip(
                        noop_estimated_pd + fixed_upper_pd - physics_noop_pd,
                        0.0, 1.0)
                    relaxed_route_upper_pd = np.clip(
                        noop_estimated_pd + relaxed_upper_pd - physics_noop_pd,
                        0.0, 1.0)
                else:
                    fixed_route_upper_pd = fixed_upper_pd
                    relaxed_route_upper_pd = relaxed_upper_pd
                causal_route = route_isac_repair(
                    float(np.min(noop_estimated_pd)),
                    fixed_power_upper=float(np.min(fixed_route_upper_pd)),
                    joint_structure_upper=float(np.min(
                        relaxed_route_upper_pd)),
                    qos_floor=float(qos_floor),
                )
                privileged_started = time.perf_counter()
                privileged_cpu_started = time.process_time()
                true_fixed_exact = solve_fixed_structure_maxmin_power_lp(
                    true_gain, budget)
                true_relaxed = relaxed_same_geometry_target_ceiling(
                    coefficient, budget)
                true_route = route_isac_repair(
                    float(np.min(deployed_pd)),
                    fixed_power_upper=_worst_pd(
                        true_fixed_exact.deflection, p_fa),
                    joint_structure_upper=_worst_pd(true_relaxed, p_fa),
                    qos_floor=float(qos_floor),
                )
                controller_privileged_audit_s += float(
                    time.perf_counter() - privileged_started)
                controller_privileged_audit_cpu_s += float(
                    time.process_time() - privileged_cpu_started)
                power_only = distributed_column_generation_maxmin_power(
                    predicted_gain,
                    budget,
                    rounds=int(rounds),
                    price_bits=int(price_bits),
                    feedback_bits=int(feedback_bits),
                    minimum_deflection=target_reserve_deflection,
                    incumbent_power_w=(
                        budget[:, None]
                        * sensing_power
                        / np.maximum(
                            np.sum(sensing_power, axis=1, keepdims=True),
                            1.0e-12,
                        )
                        if target_reserve_deflection is not None else None
                    ),
                )
                privileged_started = time.perf_counter()
                privileged_cpu_started = time.process_time()
                realized_power_pd = compute_detection_probabilities(
                    np.sum(true_gain * power_only.power_w, axis=0), p_fa)
                controller_privileged_audit_s += float(
                    time.perf_counter() - privileged_started)
                controller_privileged_audit_cpu_s += float(
                    time.process_time() - privileged_cpu_started)
                physics_power_pd = compute_detection_probabilities(
                    power_only.deflection, p_fa)
                physics_power_upper_pd = compute_detection_probabilities(
                    np.sum(predicted_upper_gain * power_only.power_w, axis=0),
                    p_fa,
                )
                estimated_power_pd = (
                    np.clip(
                        noop_estimated_pd + physics_power_pd - physics_noop_pd,
                        0.0, 1.0)
                    if observation_mode == "owner_local_conformal"
                    else physics_power_pd
                )
                structure_triggered = bool(
                    causal_route.route.value == "joint_structure_power"
                    or (
                        bool(slow_geometry_fast_fallback)
                        and causal_route.route.value == "slow_geometry"
                    )
                )
                structural = None
                horizon_ranked_structural = None
                horizon_membership_candidate_set = None
                horizon_envelope = None
                horizon_ranking_comm_power = None
                structure_transport = None
                horizon_repair = None
                horizon_candidate_transport = None
                horizon_candidate_comm_power = None
                horizon_owner_proposal_oracle = None
                horizon_raw_candidate_oracle = None
                horizon_rendezvous_transport = None
                horizon_rendezvous_prefix = None
                horizon_rendezvous_suffix = None
                horizon_rendezvous_repair = None
                horizon_rendezvous_evaluation = None
                horizon_rendezvous_future = None
                horizon_rendezvous_match_move = None
                horizon_rendezvous_match_proposer = None
                horizon_rendezvous_compute_s = 0.0
                horizon_rendezvous_process_cpu_s = 0.0
                horizon_rendezvous_total_latency_s = None
                horizon_rendezvous_nominal_total_latency_s = None
                horizon_rendezvous_repetition_certificate = None
                horizon_rendezvous_comm_bound_valid = None
                horizon_rendezvous_additional_eligible = False
                horizon_rank_batch_cache = None
                horizon_ranking_compute_s = 0.0
                horizon_ranking_process_cpu_s = 0.0
                horizon_complete_branch_compute_s = 0.0
                horizon_complete_branch_process_cpu_s = 0.0
                horizon_propagation_compute_s = 0.0
                horizon_envelope_compute_s = 0.0
                horizon_candidate_set_compute_s = 0.0
                horizon_ranking_parallel_compute_s = 0.0
                horizon_ranking_owner_compute_s = []
                horizon_owner_concurrent_measured = False
                horizon_owner_worker_count = 0
                horizon_owner_sum_thread_cpu_s = 0.0
                structure_ranking_compute_s = 0.0
                structure_ranking_process_cpu_s = 0.0
                horizon_top1_matches_myopic = None
                ranking_comm_bound_valid = None
                horizon_diagnostic_reason = (
                    "disabled" if horizon_steps <= 0 else "not_triggered")
                structure_selected = selected
                structure_power = power_only.power_w
                if structure_triggered:
                    if bool(persistent_closed_loop):
                        initial_role, _ = role_owner_from_structure(
                            selected,
                            fallback_role=(1 - np.asarray(
                                data["teacher_role"][row_id],
                                dtype=np.int8)),
                        )
                    else:
                        initial_role = 1 - np.asarray(
                            data["teacher_role"][row_id], dtype=np.int8)
                    controller_common_instrumented_wall_s = float(
                        time.perf_counter() - controller_common_started)
                    controller_common_instrumented_cpu_s = float(
                        time.process_time() - controller_common_cpu_started)
                    controller_common_preprocessing_s = max(
                        controller_common_instrumented_wall_s
                        - controller_privileged_audit_s,
                        0.0,
                    )
                    controller_common_preprocessing_cpu_s = max(
                        controller_common_instrumented_cpu_s
                        - controller_privileged_audit_cpu_s,
                        0.0,
                    )
                    structure_ranking_started = time.perf_counter()
                    structure_ranking_cpu_started = time.process_time()
                    structural = dual_guided_atomic_structure_repair(
                        selected,
                        structural_coefficient,
                        structural_support,
                        # Trace labels use 0=Tx/1=Rx; the local-exchange
                        # feasibility layer uses the explicit 1=Tx/0=Rx code.
                        initial_role,
                        budget,
                        target_pair_limit=pair_limit,
                        reports_per_receiver=reports,
                        rounds=int(rounds),
                        price_bits=int(price_bits),
                        feedback_bits=int(feedback_bits),
                        top_m=int(top_m),
                        max_steps=int(structure_steps),
                        weak_target_count=2,
                        proxy_mode=str(proxy_mode),
                        optimism_weight=float(optimism_weight),
                        owner_proposal_mode=bool(owner_proposal_transport),
                        minimum_deflection=target_reserve_deflection,
                        incumbent_power_w=(
                            budget[:, None]
                            * sensing_power
                            / np.maximum(
                                np.sum(sensing_power, axis=1, keepdims=True),
                                1.0e-12,
                            )
                            if target_reserve_deflection is not None else None
                        ),
                    )
                    structure_ranking_compute_s = float(
                        time.perf_counter() - structure_ranking_started)
                    structure_ranking_process_cpu_s = float(
                        time.process_time() - structure_ranking_cpu_started)
                    if (
                        horizon_steps > 0
                        and current_cache is not None
                        and horizon_alpha is not None
                        and not np.any(selected & ~support_provenance)
                        and np.all(np.asarray(
                            current_cache.target_invariant,
                            dtype=np.float64) > 0.0)
                    ):
                        horizon_membership_started = time.perf_counter()
                        horizon_membership_cpu_started = time.process_time()
                        horizon_propagation_started = time.perf_counter()
                        horizon_states = propagate_owner_local_horizon(
                            current_owner_local_state,
                            np.zeros((
                                max(horizon_steps - 1, 0), K, 2,
                            ), dtype=np.float64),
                            dt_s=float(cfg.scenario.dt),
                            max_speed_mps=float(cfg.uav.v_max),
                            area_size_m=tuple(float(value) for value in (
                                cfg.scenario.region_size)),
                            advance_targets=bool(getattr(
                                cfg.marl, "tracking_enabled", True)),
                            target_acceleration_std_mps2=horizon_sigma_a,
                        )
                        horizon_propagation_compute_s = float(
                            time.perf_counter() - horizon_propagation_started)
                        horizon_envelope_started = time.perf_counter()
                        # Future movement decisions are not observable at the
                        # current event.  Around the nominal zero-move plan,
                        # speed clipping gives the deterministic reachable
                        # set ||p_h-p_hat_h|| <= h*v_max*dt and, for h>=1,
                        # ||v_h-v_hat_h|| <= v_max.  These radii protect both
                        # inverse-range coefficients and the discontinuous DD
                        # admission condition.
                        horizon_uav_position_radius_m = (
                            np.arange(
                                horizon_steps, dtype=np.float64)
                            * float(cfg.uav.v_max)
                            * float(cfg.scenario.dt)
                        )
                        horizon_uav_velocity_radius_mps = np.concatenate((
                            np.zeros(1, dtype=np.float64),
                            np.full(
                                max(horizon_steps - 1, 0),
                                float(cfg.uav.v_max),
                                dtype=np.float64,
                            ),
                        ))
                        invariant_lower = np.asarray(
                            current_cache.target_invariant,
                            dtype=np.float64)
                        invariant_upper = np.asarray([
                            invariant_layout.invariant_cell_upper(value)
                            for value in invariant_lower
                        ], dtype=np.float64)
                        horizon_physics = (
                            owner_local_horizon_coefficient_bounds(
                                horizon_states,
                                invariant_lower,
                                invariant_upper,
                                support_provenance,
                                carrier_hz=float(cfg.otfs.fc),
                                delta_f_hz=float(cfg.otfs.delta_f),
                                symbol_period_s=float(cfg.otfs.T_sym),
                                delay_bins=int(cfg.otfs.M),
                                doppler_bins=int(cfg.otfs.N),
                                covariance_radius=dd_covariance_radius,
                                dd_support_threshold=float(
                                    cfg.detection.g_min),
                                dd_additive_margin=dd_support_margin,
                                # The first term protects the lagged/current
                                # coefficient reconstruction.  The second is
                                # a separately frozen simultaneous H x edge x
                                # target transition margin.  It is calibrated
                                # from whole-episode future residual maxima;
                                # future rows never enter this computation.
                                residual_log_margin=(
                                    max(
                                        lower_log_margin,
                                        upper_log_margin,
                                    ) + horizon_residual_margin
                                ),
                                uav_position_radius_m=(
                                    horizon_uav_position_radius_m),
                                uav_velocity_radius_mps=(
                                    horizon_uav_velocity_radius_mps),
                            )
                        )
                        horizon_envelope = HorizonCoefficientEnvelope(
                            lower=horizon_physics.lower,
                            upper=horizon_physics.upper,
                            source="physics",
                            miscoverage=horizon_alpha,
                        )
                        horizon_envelope_compute_s = float(
                            time.perf_counter() - horizon_envelope_started)
                        if bool(horizon_set_membership_verifier):
                            horizon_candidate_set_started = time.perf_counter()
                            horizon_membership_candidate_set = (
                                enumerate_dual_guided_atomic_candidates(
                                    selected,
                                    structural_coefficient,
                                    structural_support,
                                    initial_role,
                                    budget,
                                    target_pair_limit=pair_limit,
                                    reports_per_receiver=reports,
                                    weak_target_count=horizon_weak_targets,
                                    minimum_deflection=(
                                        target_reserve_deflection),
                                )
                            )
                            horizon_candidate_set_compute_s = float(
                                time.perf_counter()
                                - horizon_candidate_set_started)
                            horizon_ranking_compute_s = float(
                                time.perf_counter()
                                - horizon_membership_started)
                            horizon_ranking_process_cpu_s = float(
                                time.process_time()
                                - horizon_membership_cpu_started)
                            horizon_complete_branch_compute_s = (
                                horizon_ranking_compute_s)
                            horizon_complete_branch_process_cpu_s = (
                                horizon_ranking_process_cpu_s)
                            horizon_ranking_parallel_compute_s = (
                                horizon_ranking_compute_s)
                            horizon_ranking_owner_compute_s = [
                                horizon_ranking_compute_s]
                        elif horizon_ranking_mode == "myopic":
                            horizon_ranked_structural = structural
                            horizon_complete_branch_compute_s = float(
                                time.perf_counter()
                                - horizon_membership_started)
                            horizon_complete_branch_process_cpu_s = float(
                                time.process_time()
                                - horizon_membership_cpu_started)
                        else:
                            # Ranking reserves at least 0.25 W for control.
                            # A selected candidate is invalidated if its full
                            # transported protocol later needs more.  Thus the
                            # common-mix value remains a feasible lower bound.
                            horizon_ranking_comm_power = np.broadcast_to(
                                np.maximum(
                                    comm_power, horizon_ranking_reserve),
                                (horizon_steps, K),
                            ).copy()
                            horizon_noop_upper_pd = []
                            for horizon_step in range(horizon_steps):
                                horizon_noop_gain, _ = (
                                    fixed_owner_gain_matrix(
                                        horizon_envelope.upper[horizon_step],
                                        [tuple(edge) for edge in np.argwhere(
                                            selected)],
                                    )
                                )
                                horizon_noop_upper_pd.append(
                                    compute_detection_probabilities(
                                        np.sum(
                                            horizon_noop_gain * sensing_power,
                                            axis=0,
                                        ),
                                        p_fa,
                                    )
                                )
                            horizon_noop_upper_pd = np.asarray(
                                horizon_noop_upper_pd, dtype=np.float64)

                            def horizon_interval_ranker(move):
                                proxy = horizon_common_mix_proxy(
                                    move.selected,
                                    horizon_envelope,
                                    horizon_ranking_comm_power,
                                    total_power_w=1.0,
                                    false_alarm_probability=p_fa,
                                    feedback_bits=int(feedback_bits),
                                    discount=horizon_gamma,
                                )
                                if horizon_ranking_mode == "common_mix_sum":
                                    value = float(
                                        proxy.discounted_worst_pd_lower)
                                else:
                                    # Both P_D terms lie in [0,1], so the +1
                                    # shift gives an exactly order-preserving
                                    # non-negative scalar for binary16 wire
                                    # transport.  This score is aligned with
                                    # the strict H-by-Q no-harm gate.
                                    value = float(1.0 + np.min(
                                        proxy.worst_pd_lower[:, None]
                                        - horizon_noop_upper_pd
                                    ))
                                    value = float(np.clip(value, 0.0, 2.0))
                                return value, value, value

                            def finite_horizon_batch_ranker(moves):
                                nonlocal horizon_rank_batch_cache
                                nonlocal horizon_ranking_parallel_compute_s
                                nonlocal horizon_ranking_owner_compute_s
                                nonlocal horizon_owner_concurrent_measured
                                nonlocal horizon_owner_worker_count
                                nonlocal horizon_owner_sum_thread_cpu_s
                                evaluated_indices = list(range(len(moves)))
                                selected_owner_groups = None
                                if horizon_ranking_mode in (
                                    "screened_finite_margin",
                                    "distributed_screened_margin",
                                ):
                                    cheap_scores = []
                                    proposer_groups: dict[int, list[int]] = {}
                                    for move_index, move in enumerate(moves):
                                        proxy = horizon_common_mix_proxy(
                                            move.selected,
                                            horizon_envelope,
                                            horizon_ranking_comm_power,
                                            total_power_w=1.0,
                                            false_alarm_probability=p_fa,
                                            feedback_bits=int(feedback_bits),
                                            discount=horizon_gamma,
                                        )
                                        cheap_value = float(1.0 + np.min(
                                            proxy.worst_pd_lower[:, None]
                                            - horizon_noop_upper_pd
                                        ))
                                        cheap_scores.append(float(np.clip(
                                            cheap_value, 0.0, 2.0)))
                                        affected = np.flatnonzero(
                                            np.any(
                                                move.selected != selected,
                                                axis=(0, 1),
                                            )
                                            | (move.owner != owners)
                                        )
                                        proposer = int(np.min(
                                            owners[affected]))
                                        proposer_groups.setdefault(
                                            proposer, []).append(move_index)
                                    selected_owner_groups = {
                                        proposer: sorted(
                                            group,
                                            key=lambda index: (
                                                cheap_scores[index],
                                                -int(np.sum(
                                                    moves[index].selected
                                                    != selected)),
                                                -index,
                                            ),
                                            reverse=True,
                                        )[:horizon_local_shortlist]
                                        for proposer, group in
                                        proposer_groups.items()
                                    }
                                    evaluated_indices = sorted(
                                        move_index
                                        for group in
                                        selected_owner_groups.values()
                                        for move_index in group
                                    )
                                if horizon_ranking_mode == (
                                    "distributed_screened_margin"
                                ):
                                    intervals = [
                                        (0.0, 0.0, 0.0) for _ in moves]
                                    owner_groups = {
                                        int(owner): tuple(group)
                                        for owner, group in (
                                            selected_owner_groups or {}
                                        ).items()
                                    }

                                    def rank_owner_group(group):
                                        group_moves = tuple(
                                            moves[index] for index in group)
                                        group_result = (
                                            rank_atomic_horizon_repairs(
                                                selected,
                                                group_moves,
                                                horizon_envelope,
                                                noop_resources=(
                                                    HorizonCandidateResources(
                                                        comm_power_w=(
                                                            np.broadcast_to(
                                                                comm_power,
                                                                (
                                                                    horizon_steps,
                                                                    K,
                                                                ),
                                                            ).copy()
                                                        ),
                                                    )
                                                ),
                                                candidate_resources=tuple(
                                                    HorizonCandidateResources(
                                                        comm_power_w=(
                                                            horizon_ranking_comm_power
                                                            .copy()
                                                        ),
                                                    )
                                                    for _ in group_moves
                                                ),
                                                limits=HorizonResourceLimits(
                                                    control_period_s=float(
                                                        cfg.scenario.dt),
                                                    total_power_w=1.0,
                                                ),
                                                false_alarm_probability=p_fa,
                                                qos_floor=float(qos_floor),
                                                discount=horizon_gamma,
                                                prices=horizon_prices,
                                                power_protocol=(
                                                    HorizonPowerProtocol(
                                                        rounds=int(
                                                            horizon_rank_rounds),
                                                        price_bits=int(
                                                            price_bits),
                                                        feedback_bits=int(
                                                            feedback_bits),
                                                        reuse_primal_master_duals=bool(
                                                            horizon_reuse_primal_master_duals),
                                                    )
                                                ),
                                                noop_incumbent_sensing_power_w=(
                                                    np.broadcast_to(
                                                        sensing_power,
                                                        (
                                                            horizon_steps,
                                                            K,
                                                            Q,
                                                        ),
                                                    ).copy()
                                                ),
                                                paired_deflection_gate=bool(
                                                    horizon_paired_deflection_gate),
                                                common_power_across_horizon=bool(
                                                    horizon_common_power_plan),
                                            )
                                        )
                                        return tuple(group), group_result

                                    if horizon_owner_workers > 0:
                                        parallel_execution = (
                                            execute_owner_jobs_concurrently(
                                                {
                                                    owner: (
                                                        lambda group=group:
                                                        rank_owner_group(group)
                                                    )
                                                    for owner, group in
                                                    owner_groups.items()
                                                },
                                                max_workers=(
                                                    horizon_owner_workers),
                                            )
                                        )
                                        owner_results = [
                                            measurement.result
                                            for measurement in
                                            parallel_execution.measurements
                                        ]
                                        owner_times = [
                                            float(measurement.wall_latency_s)
                                            for measurement in
                                            parallel_execution.measurements
                                        ]
                                        horizon_ranking_parallel_compute_s = (
                                            float(parallel_execution
                                                  .wall_latency_s)
                                        )
                                        horizon_owner_concurrent_measured = True
                                        horizon_owner_worker_count = int(
                                            parallel_execution.worker_count)
                                        horizon_owner_sum_thread_cpu_s = float(
                                            parallel_execution
                                            .sum_worker_thread_cpu_s)
                                    else:
                                        owner_results = []
                                        owner_times = []
                                        for group in owner_groups.values():
                                            group_started = time.perf_counter()
                                            owner_results.append(
                                                rank_owner_group(group))
                                            owner_times.append(float(
                                                time.perf_counter()
                                                - group_started))
                                        horizon_ranking_parallel_compute_s = (
                                            float(max(
                                                owner_times, default=0.0))
                                        )

                                    for group, group_result in owner_results:
                                        discount_mass = float(np.sum(
                                            horizon_gamma ** np.arange(
                                                horizon_steps,
                                                dtype=np.float64,
                                            )
                                        ))
                                        for move_index, finite_eval in zip(
                                            group,
                                            group_result.evaluations,
                                        ):
                                            target_margin = float(np.min(
                                                finite_eval.rollout.lower_pd
                                                - group_result.noop.upper_pd
                                            ))
                                            service_margin = float(np.min(
                                                np.min(
                                                    finite_eval.rollout.lower_pd,
                                                    axis=1,
                                                ) - float(qos_floor)
                                            ))
                                            objective_margin = float(
                                                finite_eval
                                                .net_discounted_gain_lower
                                                / discount_mass
                                            )
                                            value = float(1.0 + np.clip(
                                                min(
                                                    target_margin,
                                                    service_margin,
                                                    objective_margin,
                                                ),
                                                -1.0,
                                                1.0,
                                            ))
                                            intervals[move_index] = (
                                                value, value, value)
                                    horizon_ranking_owner_compute_s = (
                                        owner_times)
                                    horizon_rank_batch_cache = None
                                    return tuple(intervals)
                                evaluated_moves = tuple(
                                    moves[index]
                                    for index in evaluated_indices
                                )
                                finite_margin = rank_atomic_horizon_repairs(
                                    selected,
                                    evaluated_moves,
                                    horizon_envelope,
                                    noop_resources=HorizonCandidateResources(
                                        comm_power_w=np.broadcast_to(
                                            comm_power,
                                            (horizon_steps, K),
                                        ).copy(),
                                    ),
                                    candidate_resources=tuple(
                                        HorizonCandidateResources(
                                            comm_power_w=(
                                                horizon_ranking_comm_power
                                                .copy()
                                            ),
                                        )
                                        for _ in evaluated_moves
                                    ),
                                    limits=HorizonResourceLimits(
                                        control_period_s=float(
                                            cfg.scenario.dt),
                                        total_power_w=1.0,
                                    ),
                                    false_alarm_probability=p_fa,
                                    qos_floor=float(qos_floor),
                                    discount=horizon_gamma,
                                    prices=horizon_prices,
                                    power_protocol=HorizonPowerProtocol(
                                        rounds=int(horizon_rank_rounds),
                                        price_bits=int(price_bits),
                                        feedback_bits=int(feedback_bits),
                                        reuse_primal_master_duals=bool(
                                            horizon_reuse_primal_master_duals),
                                    ),
                                    noop_incumbent_sensing_power_w=(
                                        np.broadcast_to(
                                            sensing_power,
                                            (horizon_steps, K, Q),
                                        ).copy()
                                    ),
                                    paired_deflection_gate=bool(
                                        horizon_paired_deflection_gate),
                                    common_power_across_horizon=bool(
                                        horizon_common_power_plan),
                                )
                                horizon_rank_batch_cache = finite_margin
                                discount_mass = float(np.sum(
                                    horizon_gamma ** np.arange(
                                        horizon_steps,
                                        dtype=np.float64,
                                    )
                                ))
                                intervals = [
                                    (0.0, 0.0, 0.0) for _ in moves]
                                for move_index, finite_eval in zip(
                                    evaluated_indices,
                                    finite_margin.evaluations,
                                ):
                                    target_margin = float(np.min(
                                        finite_eval.rollout.lower_pd
                                        - finite_margin.noop.upper_pd
                                    ))
                                    service_margin = float(np.min(
                                        np.min(
                                            finite_eval.rollout.lower_pd,
                                            axis=1,
                                        ) - float(qos_floor)
                                    ))
                                    objective_margin = float(
                                        finite_eval.net_discounted_gain_lower
                                        / discount_mass
                                    )
                                    value = float(1.0 + np.clip(
                                        min(
                                            target_margin,
                                            service_margin,
                                            objective_margin,
                                        ),
                                        -1.0,
                                        1.0,
                                    ))
                                    intervals[move_index] = (
                                        value, value, value)
                                return tuple(intervals)

                            ranking_started = time.perf_counter()
                            ranking_cpu_started = time.process_time()
                            horizon_ranked_structural = (
                                dual_guided_atomic_structure_repair(
                                    selected,
                                    structural_coefficient,
                                    structural_support,
                                    initial_role,
                                    budget,
                                    target_pair_limit=pair_limit,
                                    reports_per_receiver=reports,
                                    rounds=int(rounds),
                                    price_bits=int(price_bits),
                                    feedback_bits=int(feedback_bits),
                                    top_m=int(top_m),
                                    max_steps=1,
                                    weak_target_count=horizon_weak_targets,
                                    proxy_mode=str(proxy_mode),
                                    optimism_weight=float(optimism_weight),
                                    owner_proposal_mode=bool(
                                        owner_proposal_transport),
                                    minimum_deflection=(
                                        target_reserve_deflection),
                                    incumbent_power_w=(
                                        budget[:, None]
                                        * sensing_power
                                        / np.maximum(
                                            np.sum(
                                                sensing_power,
                                                axis=1,
                                                keepdims=True,
                                            ),
                                            1.0e-12,
                                        )
                                        if target_reserve_deflection is not None
                                        else None
                                    ),
                                    candidate_interval_ranker=(
                                        None
                                        if horizon_ranking_mode ==
                                        "finite_horizon_margin"
                                        or horizon_ranking_mode ==
                                        "screened_finite_margin"
                                        or horizon_ranking_mode ==
                                        "distributed_screened_margin"
                                        else horizon_interval_ranker),
                                    candidate_batch_interval_ranker=(
                                        finite_horizon_batch_ranker
                                        if horizon_ranking_mode in (
                                            "finite_horizon_margin",
                                            "screened_finite_margin",
                                            "distributed_screened_margin",
                                        )
                                        else None),
                                )
                            )
                            horizon_ranking_compute_s = float(
                                time.perf_counter() - ranking_started)
                            horizon_ranking_process_cpu_s = float(
                                time.process_time() - ranking_cpu_started)
                            horizon_complete_branch_compute_s = float(
                                time.perf_counter()
                                - horizon_membership_started)
                            horizon_complete_branch_process_cpu_s = float(
                                time.process_time()
                                - horizon_membership_cpu_started)
                            if horizon_ranking_mode != (
                                "distributed_screened_margin"
                            ):
                                horizon_ranking_parallel_compute_s = (
                                    horizon_ranking_compute_s)
                                horizon_ranking_owner_compute_s = [
                                    horizon_ranking_compute_s]
                    elif horizon_steps > 0:
                        if current_cache is None:
                            horizon_diagnostic_reason = (
                                "incomplete_target_invariant")
                        elif np.any(selected & ~support_provenance):
                            horizon_diagnostic_reason = (
                                "incomplete_noop_provenance")
                        elif np.any(np.asarray(
                            current_cache.target_invariant,
                            dtype=np.float64,
                        ) <= 0.0):
                            horizon_diagnostic_reason = (
                                "incomplete_target_invariant")
                    structure_selected = structural.selected
                    structure_power = structural.power_result.power_w
                    if bool(certify_structure_sequence):
                        weight_mass = np.sum(
                            structure_power, axis=1, keepdims=True)
                        structure_weights = structure_power / weight_mass
                        if (
                            bool(target_invariant_token_transport)
                            and current_cache is None
                        ):
                            raise RuntimeError(
                                "target-invariant cache is unavailable")
                        structure_transport = (
                            certify_top1_structure_sequence_transport(
                                selected,
                                initial_role,
                                owners,
                                structural.verified_moves,
                                structural.accepted_moves,
                                positions=np.asarray(
                                    data["uav_positions"][row_id],
                                    dtype=np.float64),
                                existing_comm_power_w=comm_power,
                                final_sensing_weights=structure_weights,
                                communication_model=comm_model,
                                power_layout=layout,
                                control_period_s=float(cfg.scenario.dt),
                                snr_margin_db=float(snr_margin_db),
                                latency_margin_s=float(latency_margin_s),
                                owner_proposal_rounds=(
                                    structural.owner_proposal_rounds
                                    if bool(owner_proposal_transport) else ()
                                ),
                                owner_proposal_layout=(
                                    proposal_layout
                                    if bool(owner_proposal_transport) else None
                                ),
                                target_invariant_tokens=(
                                    target_invariant_tokens),
                                target_invariant_layout=(
                                    invariant_layout
                                    if bool(target_invariant_token_transport)
                                    else None
                                ),
                            )
                        )
                        if (
                            horizon_steps > 0
                            and bool(horizon_oracle_diagnostics)
                            and horizon_ranked_structural is not None
                            and horizon_ranked_structural.owner_proposal_rounds
                            and horizon_envelope is not None
                        ):
                            # Optimistic diagnostic upper bound only: every
                            # already-broadcast owner proposal receives a
                            # finite-round physics rollout, but its additional
                            # verification/commit link cost is deliberately
                            # omitted.  It can expose ranking regret; it can
                            # never authorize a commit.
                            oracle_moves = tuple(
                                proposal.move for proposal in
                                horizon_ranked_structural
                                .owner_proposal_rounds[0]
                            )
                            oracle_comm = (
                                horizon_ranking_comm_power
                                if horizon_ranking_comm_power is not None
                                else np.broadcast_to(
                                    comm_power, (horizon_steps, K)).copy()
                            )
                            horizon_owner_proposal_oracle = (
                                rank_atomic_horizon_repairs(
                                    selected,
                                    oracle_moves,
                                    horizon_envelope,
                                    noop_resources=(
                                        HorizonCandidateResources(
                                            comm_power_w=np.broadcast_to(
                                                comm_power,
                                                (horizon_steps, K),
                                            ).copy(),
                                        )
                                    ),
                                    candidate_resources=tuple(
                                        HorizonCandidateResources(
                                            comm_power_w=oracle_comm.copy(),
                                        )
                                        for _ in oracle_moves
                                    ),
                                    limits=HorizonResourceLimits(
                                        control_period_s=float(
                                            cfg.scenario.dt),
                                        total_power_w=1.0,
                                    ),
                                    false_alarm_probability=p_fa,
                                    qos_floor=float(qos_floor),
                                    discount=horizon_gamma,
                                    prices=horizon_prices,
                                    power_protocol=HorizonPowerProtocol(
                                        rounds=int(rounds),
                                        price_bits=int(price_bits),
                                        feedback_bits=int(feedback_bits),
                                        reuse_primal_master_duals=bool(
                                            horizon_reuse_primal_master_duals),
                                    ),
                                    noop_incumbent_sensing_power_w=(
                                        np.broadcast_to(
                                            sensing_power,
                                            (horizon_steps, K, Q),
                                        ).copy()
                                    ),
                                    paired_deflection_gate=bool(
                                        horizon_paired_deflection_gate),
                                    common_power_across_horizon=bool(
                                        horizon_common_power_plan),
                                )
                            )
                            raw_oracle_moves = tuple(
                                horizon_ranked_structural
                                .ranked_candidate_moves
                            )
                            if (
                                horizon_ranking_mode ==
                                "finite_horizon_margin"
                                and horizon_rank_batch_cache is not None
                                and len(
                                    horizon_rank_batch_cache.evaluations
                                ) == len(raw_oracle_moves)
                            ):
                                horizon_raw_candidate_oracle = (
                                    horizon_rank_batch_cache)
                            else:
                                horizon_raw_candidate_oracle = (
                                    rank_atomic_horizon_repairs(
                                    selected,
                                    raw_oracle_moves,
                                    horizon_envelope,
                                    noop_resources=(
                                        HorizonCandidateResources(
                                            comm_power_w=np.broadcast_to(
                                                comm_power,
                                                (horizon_steps, K),
                                            ).copy(),
                                        )
                                    ),
                                    candidate_resources=tuple(
                                        HorizonCandidateResources(
                                            comm_power_w=oracle_comm.copy(),
                                        )
                                        for _ in raw_oracle_moves
                                    ),
                                    limits=HorizonResourceLimits(
                                        control_period_s=float(
                                            cfg.scenario.dt),
                                        total_power_w=1.0,
                                    ),
                                    false_alarm_probability=p_fa,
                                    qos_floor=float(qos_floor),
                                    discount=horizon_gamma,
                                    prices=horizon_prices,
                                    power_protocol=HorizonPowerProtocol(
                                        rounds=int(rounds),
                                        price_bits=int(price_bits),
                                        feedback_bits=int(feedback_bits),
                                        reuse_primal_master_duals=bool(
                                            horizon_reuse_primal_master_duals),
                                    ),
                                    noop_incumbent_sensing_power_w=(
                                        np.broadcast_to(
                                            sensing_power,
                                            (horizon_steps, K, Q),
                                        ).copy()
                                    ),
                                        paired_deflection_gate=bool(
                                            horizon_paired_deflection_gate),
                                        common_power_across_horizon=bool(
                                            horizon_common_power_plan),
                                    )
                                )
                        if (
                            horizon_steps > 0
                            and horizon_ranked_structural is not None
                            and horizon_ranked_structural.verified_moves
                            and not np.any(selected & ~support_provenance)
                        ):
                            horizon_diagnostic_reason = "evaluated"
                            horizon_move = (
                                horizon_ranked_structural.verified_moves[0])
                            if structural.verified_moves:
                                myopic_move = structural.verified_moves[0]
                                horizon_top1_matches_myopic = bool(
                                    horizon_move.kind == myopic_move.kind
                                    and np.array_equal(
                                        horizon_move.selected,
                                        myopic_move.selected,
                                    )
                                    and np.array_equal(
                                        horizon_move.role,
                                        myopic_move.role,
                                    )
                                    and np.array_equal(
                                        horizon_move.owner,
                                        myopic_move.owner,
                                    )
                                )
                            horizon_candidate_transport = (
                                certify_top1_structure_sequence_transport(
                                    selected,
                                    initial_role,
                                    owners,
                                    (horizon_move,),
                                    (horizon_move,),
                                    positions=np.asarray(
                                        data["uav_positions"][row_id],
                                        dtype=np.float64),
                                    existing_comm_power_w=comm_power,
                                    final_sensing_weights=structure_weights,
                                    communication_model=comm_model,
                                    power_layout=layout,
                                    control_period_s=float(cfg.scenario.dt),
                                    snr_margin_db=float(snr_margin_db),
                                    latency_margin_s=float(latency_margin_s),
                                    owner_proposal_rounds=(
                                        horizon_ranked_structural
                                        .owner_proposal_rounds[:1]
                                        if bool(owner_proposal_transport)
                                        else ()
                                    ),
                                    owner_proposal_layout=(
                                        proposal_layout
                                        if bool(owner_proposal_transport)
                                        else None
                                    ),
                                    target_invariant_tokens=(
                                        target_invariant_tokens),
                                    target_invariant_layout=(
                                        invariant_layout),
                                )
                            )
                            if current_cache is None or horizon_alpha is None:
                                raise RuntimeError(
                                    "horizon invariant/risk state is unavailable")
                            if horizon_envelope is None:
                                raise RuntimeError(
                                    "horizon envelope was not initialized")
                            candidate_comm = np.asarray(
                                horizon_candidate_transport
                                .projected_comm_power_w,
                                dtype=np.float64)
                            if (
                                np.any(~np.isfinite(candidate_comm))
                                or np.any(candidate_comm >= 1.0)
                            ):
                                candidate_comm = comm_power.copy()
                            horizon_candidate_comm_power = (
                                candidate_comm.copy())
                            if horizon_ranking_comm_power is not None:
                                ranking_comm_bound_valid = bool(np.all(
                                    candidate_comm
                                    <= horizon_ranking_comm_power[0]
                                    + 1.0e-12
                                ))
                            if ranking_comm_bound_valid is False:
                                horizon_diagnostic_reason = (
                                    "ranking_comm_bound_failure")
                            horizon_repair = rank_atomic_horizon_repairs(
                                selected,
                                (horizon_move,),
                                horizon_envelope,
                                noop_resources=HorizonCandidateResources(
                                    comm_power_w=np.broadcast_to(
                                        comm_power,
                                        (horizon_steps, K),
                                    ).copy(),
                                ),
                                candidate_resources=(
                                    HorizonCandidateResources(
                                        comm_power_w=np.broadcast_to(
                                            candidate_comm,
                                            (horizon_steps, K),
                                        ).copy(),
                                        over_air_bits=int(
                                            horizon_candidate_transport
                                            .total_over_air_bits),
                                        protocol_latency_s=float(
                                            horizon_candidate_transport
                                            .total_protocol_latency_s
                                            + (
                                                shared_complete_compute_latency_bound_s
                                                if runtime_latency_calibration_metadata
                                                is not None else
                                                horizon_complete_branch_compute_s
                                                + controller_common_preprocessing_s
                                            )),
                                        control_energy_j=float(
                                            horizon_candidate_transport
                                            .total_energy_j),
                                        transport_feasible=bool(
                                            horizon_candidate_transport
                                            .feasible
                                            and ranking_comm_bound_valid
                                            is not False),
                                    ),
                                ),
                                limits=HorizonResourceLimits(
                                    control_period_s=float(cfg.scenario.dt),
                                    total_power_w=1.0,
                                ),
                                false_alarm_probability=p_fa,
                                qos_floor=float(qos_floor),
                                discount=horizon_gamma,
                                prices=horizon_prices,
                                power_protocol=HorizonPowerProtocol(
                                    rounds=int(rounds),
                                    price_bits=int(price_bits),
                                    feedback_bits=int(feedback_bits),
                                    reuse_primal_master_duals=bool(
                                        horizon_reuse_primal_master_duals),
                                ),
                                noop_incumbent_sensing_power_w=(
                                    np.broadcast_to(
                                        sensing_power,
                                        (horizon_steps, K, Q),
                                    ).copy()
                                ),
                                paired_deflection_gate=bool(
                                    horizon_paired_deflection_gate),
                                common_power_across_horizon=bool(
                                    horizon_common_power_plan),
                            )
                            if (
                                horizon_repair.evaluations
                                and "deadline" in horizon_repair
                                .evaluations[0].resource_reasons
                            ):
                                horizon_diagnostic_reason = (
                                    "end_to_end_deadline_failure")
                        elif horizon_steps > 0:
                            if horizon_diagnostic_reason in (
                                "not_triggered", "evaluated",
                            ):
                                horizon_diagnostic_reason = (
                                    "incomplete_noop_provenance"
                                    if np.any(selected & ~support_provenance)
                                    else "no_verified_atomic_candidate"
                                )
                        if structure_transport.feasible:
                            structure_power = (
                                structure_transport.projected_sensing_power_w)
                        else:
                            # The infeasible sequence is known before actuation;
                            # fail closed to the deployed No-op state.
                            structure_selected = selected
                            structure_power = sensing_power
                predicted_structure_gain, _ = fixed_owner_gain_matrix(
                    structural_coefficient,
                    [tuple(edge) for edge in np.argwhere(structure_selected)],
                )
                true_structure_gain, _ = fixed_owner_gain_matrix(
                    coefficient,
                    [tuple(edge) for edge in np.argwhere(structure_selected)],
                )
                physics_structure_pd = compute_detection_probabilities(
                    np.sum(predicted_structure_gain * structure_power, axis=0),
                    p_fa,
                )
                upper_structure_gain, _ = fixed_owner_gain_matrix(
                    coefficient_upper,
                    [tuple(edge) for edge in np.argwhere(structure_selected)],
                )
                physics_structure_upper_pd = compute_detection_probabilities(
                    np.sum(upper_structure_gain * structure_power, axis=0),
                    p_fa,
                )
                estimated_structure_pd = (
                    np.clip(
                        noop_estimated_pd
                        + physics_structure_pd - physics_noop_pd,
                        0.0, 1.0)
                    if observation_mode == "owner_local_conformal"
                    else physics_structure_pd
                )
                realized_structure_pd = compute_detection_probabilities(
                    np.sum(true_structure_gain * structure_power, axis=0),
                    p_fa,
                )
                structure_sequence_feasible = bool(
                    structure_transport is None
                    or structure_transport.feasible)
                structure_accepted = bool(
                    structural is not None and structural.accepted
                    and structure_sequence_feasible)
                route_name = causal_route.route.value
                fixed_reserve_feasible = bool(
                    target_reserve_possible and power_only.reserve_feasible)
                structure_reserve_feasible = bool(
                    target_reserve_possible
                    and structural is not None
                    and structural.power_result.reserve_feasible)
                executed_route_name = route_name
                if (
                    bool(slow_geometry_fast_fallback)
                    and route_name == "slow_geometry"
                ):
                    # Geometry evolves on a slower timescale.  While its plan
                    # is pending, execute the best reserve-feasible fast
                    # repair instead of interpreting "cannot hit the floor"
                    # as "No-op".  The slow route remains recorded and is not
                    # falsely relabelled as a fast QoS certificate.
                    if (
                        structural is not None
                        and structural.accepted
                        and structure_sequence_feasible
                        and structure_reserve_feasible
                    ):
                        executed_route_name = "joint_structure_power"
                    elif fixed_reserve_feasible:
                        executed_route_name = "fixed_structure_power"
                    else:
                        executed_route_name = "no_op"
                routed_action = bool(
                    (
                        executed_route_name == "fixed_structure_power"
                        and fixed_reserve_feasible
                    )
                    or (
                        executed_route_name == "joint_structure_power"
                        and structure_sequence_feasible
                        and structure_reserve_feasible
                    )
                )
                if (
                    executed_route_name == "fixed_structure_power"
                    and routed_action
                ):
                    routed_estimated_pd = estimated_power_pd
                    routed_realized_pd = realized_power_pd
                    routed_physics_lower_pd = physics_power_pd
                    routed_physics_upper_pd = physics_power_upper_pd
                    routed_comm_power = transport.projected_comm_power_w
                    routed_sensing_power = power_only.power_w
                    routed_transport_bits = transport.total_over_air_bits
                    routed_transport_max_packet_latency = (
                        transport.max_packet_latency_s)
                    routed_transport_protocol_latency = (
                        transport.total_protocol_latency_s)
                    routed_transport_energy = transport.total_energy_j
                    routed_power_error = float(np.max(np.abs(
                        routed_comm_power
                        + np.sum(routed_sensing_power, axis=1) - 1.0)))
                    candidate_selected = selected.copy()
                elif (
                    executed_route_name == "joint_structure_power"
                    and routed_action
                ):
                    routed_estimated_pd = estimated_structure_pd
                    routed_realized_pd = realized_structure_pd
                    routed_physics_lower_pd = physics_structure_pd
                    routed_physics_upper_pd = physics_structure_upper_pd
                    if structure_transport is not None:
                        routed_comm_power = (
                            structure_transport.projected_comm_power_w)
                        routed_transport_bits = (
                            structure_transport.total_over_air_bits)
                        routed_transport_max_packet_latency = (
                            structure_transport.max_packet_latency_s)
                        routed_transport_protocol_latency = (
                            structure_transport.total_protocol_latency_s)
                        routed_transport_energy = (
                            structure_transport.total_energy_j)
                        routed_power_error = (
                            structure_transport.max_isac_power_balance_error_w)
                    else:
                        routed_comm_power = transport.projected_comm_power_w
                        routed_transport_bits = transport.total_over_air_bits
                        routed_transport_max_packet_latency = (
                            transport.max_packet_latency_s)
                        routed_transport_protocol_latency = (
                            transport.total_protocol_latency_s)
                        routed_transport_energy = transport.total_energy_j
                        routed_power_error = float(np.max(np.abs(
                            routed_comm_power
                            + np.sum(structure_power, axis=1) - 1.0)))
                    routed_sensing_power = structure_power
                    candidate_selected = structure_selected.copy()
                else:
                    routed_estimated_pd = noop_estimated_pd
                    routed_realized_pd = deployed_pd
                    routed_physics_lower_pd = physics_noop_pd
                    routed_physics_upper_pd = physics_noop_upper_pd
                    routed_comm_power = comm_power
                    routed_sensing_power = sensing_power
                    routed_transport_bits = 0
                    routed_transport_max_packet_latency = 0.0
                    routed_transport_protocol_latency = 0.0
                    routed_transport_energy = 0.0
                    routed_power_error = float(np.max(np.abs(
                        comm_power + np.sum(sensing_power, axis=1) - 1.0)))
                    candidate_selected = selected.copy()
                candidate_comm_power = routed_comm_power.copy()
                candidate_sensing_power = routed_sensing_power.copy()
                candidate_realized_pd = routed_realized_pd.copy()
                noop_interval_width = np.maximum(
                    physics_noop_upper_pd - physics_noop_pd, 0.0)
                routed_interval_width = np.maximum(
                    routed_physics_upper_pd - routed_physics_lower_pd, 0.0)
                # Let F be the delayed owner feedback, and [L0, U0] and
                # [Lc, Uc] the monotone physics intervals for No-op and the
                # candidate.  With Ec=clip(F+Lc-L0), on the simultaneous
                # physics event both Ec-Pc and F-P0 are upper-bounded by
                # A=max(F-L0, 0).  Moreover,
                #   (min Ec-min F)-(min Pc-min P0)
                #       <= max_q A_q + max_q max(U0_q-F_q, 0).
                # These state-adaptive scales remain causal even when the
                # feedback is stale; staleness widens the gate instead of
                # being silently treated as an independent sample.
                absolute_uncertainty = (
                    np.maximum(noop_estimated_pd - physics_noop_pd, 0.0)
                    + uncertainty_floor
                )
                noop_upper_uncertainty = (
                    np.maximum(
                        physics_noop_upper_pd - noop_estimated_pd, 0.0)
                    + uncertainty_floor
                )
                # Retained as an interpretable interval diagnostic.  It is
                # not part of the conformal score because the deployed gate
                # never consumes a per-target delta bound.
                delta_uncertainty = (
                    noop_interval_width
                    + routed_interval_width
                    + uncertainty_floor
                )
                worst_delta_uncertainty = float(
                    np.max(absolute_uncertainty)
                    + np.max(noop_upper_uncertainty)
                )
                routed_power_total_variation = 0.5 * float(
                    np.sum(np.abs(routed_comm_power - comm_power))
                    + np.sum(np.abs(routed_sensing_power - sensing_power)))
                noop_incomplete_target = np.any(
                    selected & ~structural_support, axis=(0, 1))
                physics_noop_upper_certificate = (
                    physics_noop_upper_pd.copy())
                physics_noop_upper_certificate[
                    noop_incomplete_target] = 1.0
                horizon_evaluation = (
                    None
                    if horizon_repair is None
                    or not horizon_repair.evaluations
                    else horizon_repair.evaluations[0]
                )
                if (
                    bool(horizon_digest_rendezvous)
                    and bool(structure_accepted)
                    and structural is not None
                    and horizon_envelope is not None
                    and (
                        (
                            horizon_membership_candidate_set is not None
                            and horizon_membership_candidate_set.moves
                        )
                        or (
                            horizon_ranked_structural is not None
                            and horizon_ranked_structural
                            .ranked_candidate_moves
                        )
                    )
                ):
                    structure_key = isac_structure_digest(
                        selected=structural.selected,
                        role=structural.role,
                        owner=structural.owner,
                    )
                    current_top_is_reconcilable = bool(
                        horizon_evaluation is not None
                        and horizon_evaluation.gate.accept
                        and isac_structure_digest(
                            selected=horizon_evaluation.move.selected,
                            role=horizon_evaluation.move.role,
                            owner=horizon_evaluation.move.owner,
                        ) == structure_key
                    )
                    if horizon_membership_candidate_set is not None:
                        candidate_pairs = tuple(zip(
                            horizon_membership_candidate_set.moves,
                            horizon_membership_candidate_set.proposers,
                        ))
                    else:
                        candidate_pairs = tuple(
                            (move, None)
                            for move in horizon_ranked_structural
                            .ranked_candidate_moves
                        )
                    raw_matches = tuple(
                        (move, proposer)
                        for move, proposer in candidate_pairs
                        if isac_structure_digest(
                            selected=move.selected,
                            role=move.role,
                            owner=move.owner,
                        ) == structure_key
                    )
                    if raw_matches and not current_top_is_reconcilable:
                        (
                            horizon_rendezvous_match_move,
                            membership_proposer,
                        ) = raw_matches[0]
                        affected_targets = np.flatnonzero(
                            np.any(
                                horizon_rendezvous_match_move.selected
                                != selected,
                                axis=(0, 1),
                            )
                            | (
                                horizon_rendezvous_match_move.owner
                                != owners
                            )
                        )
                        if affected_targets.size:
                            horizon_rendezvous_match_proposer = (
                                int(membership_proposer)
                                if membership_proposer is not None else
                                int(np.min(owners[affected_targets]))
                            )
                            rendezvous_proposal = RankedOwnerProposal(
                                proposer=(
                                    horizon_rendezvous_match_proposer),
                                move=horizon_rendezvous_match_move,
                                lower=0.0,
                                upper=0.0,
                                score=0.0,
                            )
                            if bool(
                                horizon_digest_rendezvous_defer_top1_transport
                            ):
                                prefix_proposals = (
                                    ()
                                    if horizon_membership_candidate_set
                                    is not None else (
                                        tuple(horizon_ranked_structural
                                              .owner_proposal_rounds[0])
                                        if horizon_ranked_structural
                                        .owner_proposal_rounds
                                        else ()
                                    )
                                )
                                horizon_rendezvous_prefix = (
                                    certify_deferred_horizon_prefix(
                                        selected,
                                        initial_role,
                                        owners,
                                        prefix_proposals,
                                        positions=np.asarray(
                                            data["uav_positions"][row_id],
                                            dtype=np.float64),
                                        existing_comm_power_w=comm_power,
                                        communication_model=comm_model,
                                        power_layout=layout,
                                        proposal_layout=proposal_layout,
                                        control_period_s=float(
                                            cfg.scenario.dt),
                                        target_invariant_tokens=tuple(
                                            target_invariant_tokens),
                                        target_invariant_layout=(
                                            invariant_layout),
                                        snr_margin_db=float(snr_margin_db),
                                        latency_margin_s=float(
                                            latency_margin_s),
                                    )
                                )
                            parallel_comm = comm_power.copy()
                            branch_transports = [structure_transport]
                            if horizon_rendezvous_prefix is not None:
                                branch_transports.append(
                                    horizon_rendezvous_prefix)
                            else:
                                branch_transports.append(
                                    horizon_candidate_transport)
                            for branch_transport in branch_transports:
                                if (
                                    branch_transport is not None
                                    and np.all(np.isfinite(
                                        branch_transport
                                        .projected_comm_power_w))
                                    and np.all(
                                        branch_transport
                                        .projected_comm_power_w < 1.0)
                                ):
                                    parallel_comm = np.maximum(
                                        parallel_comm,
                                        branch_transport
                                        .projected_comm_power_w,
                                    )
                            horizon_rendezvous_transport = (
                                certify_structure_digest_rendezvous(
                                    selected,
                                    initial_role,
                                    owners,
                                    rendezvous_proposal,
                                    positions=np.asarray(
                                        data["uav_positions"][row_id],
                                        dtype=np.float64),
                                    existing_comm_power_w=parallel_comm,
                                    full_structure_verified=bool(
                                        np.array_equal(
                                            structural.selected,
                                            horizon_rendezvous_match_move
                                            .selected,
                                        )
                                        and np.array_equal(
                                            structural.role,
                                            horizon_rendezvous_match_move
                                            .role,
                                        )
                                        and np.array_equal(
                                            structural.owner,
                                            horizon_rendezvous_match_move
                                            .owner,
                                        )
                                    ),
                                    communication_model=comm_model,
                                    layout=rendezvous_layout,
                                    control_period_s=float(cfg.scenario.dt),
                                    snr_margin_db=float(snr_margin_db),
                                    latency_margin_s=float(
                                        latency_margin_s),
                                )
                            )
                            horizon_rendezvous_suffix = (
                                certify_rendezvous_candidate_suffix(
                                    selected,
                                    initial_role,
                                    owners,
                                    horizon_rendezvous_match_move,
                                    positions=np.asarray(
                                        data["uav_positions"][row_id],
                                        dtype=np.float64),
                                    existing_comm_power_w=(
                                        horizon_rendezvous_transport
                                        .projected_comm_power_w),
                                    final_sensing_weights=structure_weights,
                                    communication_model=comm_model,
                                    power_layout=layout,
                                    control_period_s=float(cfg.scenario.dt),
                                    snr_margin_db=float(snr_margin_db),
                                    latency_margin_s=float(
                                        latency_margin_s),
                                    parallel_committed_move=(
                                        structural.accepted_moves[-1]
                                        if bool(
                                            horizon_digest_rendezvous_reuse_structure_commit)
                                        and structural.accepted_moves
                                        else None
                                    ),
                                    parallel_commit_feasible=bool(
                                        horizon_digest_rendezvous_reuse_structure_commit
                                        and structure_transport is not None
                                        and structure_transport.feasible
                                        and structure_transport.commit_count > 0
                                    ),
                                )
                            )
                            rendezvous_comm = np.asarray(
                                horizon_rendezvous_suffix
                                .projected_comm_power_w,
                                dtype=np.float64,
                            )
                            horizon_rendezvous_comm_bound_valid = bool(
                                np.all(np.isfinite(rendezvous_comm))
                                and np.all(rendezvous_comm < 1.0)
                            )
                            rendezvous_started = time.perf_counter()
                            rendezvous_cpu_started = time.process_time()
                            horizon_rendezvous_repair = (
                                rank_atomic_horizon_repairs(
                                    selected,
                                    (horizon_rendezvous_match_move,),
                                    horizon_envelope,
                                    noop_resources=(
                                        HorizonCandidateResources(
                                            comm_power_w=np.broadcast_to(
                                                comm_power,
                                                (horizon_steps, K),
                                            ).copy(),
                                        )
                                    ),
                                    candidate_resources=(
                                        HorizonCandidateResources(
                                            comm_power_w=np.broadcast_to(
                                                rendezvous_comm,
                                                (horizon_steps, K),
                                            ).copy(),
                                            over_air_bits=int(
                                                (
                                                    horizon_rendezvous_prefix
                                                    .total_over_air_bits
                                                    if horizon_rendezvous_prefix
                                                    is not None else 0
                                                )
                                                + horizon_rendezvous_transport
                                                .total_over_air_bits
                                                + horizon_rendezvous_suffix
                                                .total_over_air_bits),
                                            protocol_latency_s=float(
                                                (
                                                    horizon_rendezvous_prefix
                                                    .total_protocol_latency_s
                                                    if horizon_rendezvous_prefix
                                                    is not None else 0.0
                                                )
                                                + horizon_rendezvous_transport
                                                .total_protocol_latency_s
                                                + horizon_rendezvous_suffix
                                                .total_protocol_latency_s),
                                            control_energy_j=float(
                                                (
                                                    horizon_rendezvous_prefix
                                                    .total_energy_j
                                                    if horizon_rendezvous_prefix
                                                    is not None else 0.0
                                                )
                                                + horizon_rendezvous_transport
                                                .total_energy_j
                                                + horizon_rendezvous_suffix
                                                .total_energy_j),
                                            transport_feasible=bool(
                                                (
                                                    horizon_rendezvous_prefix
                                                    is None
                                                    or horizon_rendezvous_prefix
                                                    .feasible
                                                )
                                                and
                                                horizon_rendezvous_transport
                                                .feasible
                                                and horizon_rendezvous_suffix
                                                .feasible
                                                and horizon_rendezvous_comm_bound_valid),
                                        ),
                                    ),
                                    limits=HorizonResourceLimits(
                                        control_period_s=float(
                                            cfg.scenario.dt),
                                        total_power_w=1.0,
                                    ),
                                    false_alarm_probability=p_fa,
                                    qos_floor=float(qos_floor),
                                    discount=horizon_gamma,
                                    prices=horizon_prices,
                                    power_protocol=HorizonPowerProtocol(
                                        rounds=int(rounds),
                                        price_bits=int(price_bits),
                                        feedback_bits=int(feedback_bits),
                                        reuse_primal_master_duals=bool(
                                            horizon_reuse_primal_master_duals),
                                    ),
                                    noop_incumbent_sensing_power_w=(
                                        np.broadcast_to(
                                            sensing_power,
                                            (horizon_steps, K, Q),
                                        ).copy()
                                    ),
                                    paired_deflection_gate=bool(
                                        horizon_paired_deflection_gate),
                                    common_power_across_horizon=bool(
                                        horizon_common_power_plan),
                                )
                            )
                            horizon_rendezvous_compute_s = float(
                                time.perf_counter() - rendezvous_started)
                            horizon_rendezvous_process_cpu_s = float(
                                time.process_time()
                                - rendezvous_cpu_started)
                            if horizon_rendezvous_repair.evaluations:
                                horizon_rendezvous_evaluation = (
                                    horizon_rendezvous_repair.evaluations[0])
                            structure_protocol_latency = float(
                                structure_transport.total_protocol_latency_s
                                if structure_transport is not None else 0.0)
                            horizon_branch_transport = (
                                horizon_rendezvous_prefix
                                if horizon_rendezvous_prefix is not None
                                else horizon_candidate_transport
                            )
                            horizon_protocol_latency = float(
                                horizon_branch_transport
                                .total_protocol_latency_s
                                if horizon_branch_transport is not None
                                else 0.0
                            )
                            nominal_protocol_bits = int(sum((
                                int(structure_transport.total_over_air_bits)
                                if structure_transport is not None else 0,
                                int(horizon_branch_transport
                                    .total_over_air_bits)
                                if horizon_branch_transport is not None else 0,
                                int(horizon_rendezvous_transport
                                    .total_over_air_bits),
                                int(horizon_rendezvous_suffix
                                    .total_over_air_bits),
                            )))
                            nominal_protocol_energy = float(sum((
                                float(structure_transport.total_energy_j)
                                if structure_transport is not None else 0.0,
                                float(horizon_branch_transport.total_energy_j)
                                if horizon_branch_transport is not None else 0.0,
                                float(horizon_rendezvous_transport
                                      .total_energy_j),
                                float(horizon_rendezvous_suffix.total_energy_j),
                            )))
                            nominal_transport_feasible = bool(
                                structure_transport is not None
                                and structure_transport.feasible
                                and horizon_branch_transport is not None
                                and horizon_branch_transport.feasible
                                and horizon_rendezvous_transport.feasible
                                and horizon_rendezvous_suffix.feasible
                                and horizon_rendezvous_comm_bound_valid
                            )
                            horizon_rendezvous_repetition_certificate = (
                                certify_bounded_repetition_path(
                                    common_compute_s=(
                                        shared_complete_compute_latency_bound_s
                                        if runtime_latency_calibration_metadata
                                        is not None else
                                        controller_common_preprocessing_s),
                                    parallel_branch_compute_s=(
                                        (0.0 if runtime_latency_calibration_metadata
                                         is not None else
                                         structure_ranking_compute_s),
                                        (0.0 if runtime_latency_calibration_metadata
                                         is not None else
                                         horizon_complete_branch_compute_s),
                                    ),
                                    parallel_branch_protocol_latency_s=(
                                        structure_protocol_latency,
                                        horizon_protocol_latency,
                                    ),
                                    serial_compute_s=(
                                        0.0
                                        if runtime_latency_calibration_metadata
                                        is not None else
                                        horizon_rendezvous_compute_s),
                                    serial_protocol_latency_s=(
                                        horizon_rendezvous_transport
                                        .total_protocol_latency_s,
                                        horizon_rendezvous_suffix
                                        .total_protocol_latency_s,
                                    ),
                                    nominal_over_air_bits=(
                                        nominal_protocol_bits),
                                    nominal_rf_energy_j=(
                                        nominal_protocol_energy),
                                    repetition_count=network_repetitions,
                                    excess_queue_bound_s=network_queue_bound,
                                    deadline_s=float(cfg.scenario.dt),
                                    nominal_transport_feasible=(
                                        nominal_transport_feasible),
                                )
                            )
                            horizon_rendezvous_nominal_total_latency_s = float(
                                horizon_rendezvous_repetition_certificate
                                .nominal_total_latency_s)
                            horizon_rendezvous_total_latency_s = float(
                                horizon_rendezvous_repetition_certificate
                                .worst_case_total_latency_s)
                            horizon_rendezvous_additional_eligible = bool(
                                horizon_rendezvous_evaluation is not None
                                and horizon_rendezvous_evaluation.gate.accept
                                and (
                                    horizon_rendezvous_prefix is None
                                    or horizon_rendezvous_prefix.feasible
                                )
                                and horizon_rendezvous_transport.feasible
                                and horizon_rendezvous_suffix.feasible
                                and horizon_rendezvous_comm_bound_valid
                                and horizon_rendezvous_repetition_certificate
                                .feasible
                                and horizon_rendezvous_total_latency_s
                                <= float(cfg.scenario.dt) + 1.0e-12
                            )
                structure_action_digest = None
                horizon_action_digest = None
                structure_discrete_digest = None
                horizon_discrete_digest = None
                shadow_route_decision = None
                structure_protocol_energy_j = float(
                    structure_transport.total_energy_j
                    if structure_transport is not None else 0.0
                )
                horizon_protocol_energy_j = float(
                    horizon_candidate_transport.total_energy_j
                    if horizon_candidate_transport is not None else 0.0
                )
                structure_compute_energy_j = float(
                    shadow_cpu_power * (
                        controller_common_preprocessing_cpu_s
                        + structure_ranking_process_cpu_s)
                    if shadow_cpu_power is not None else 0.0
                )
                horizon_compute_energy_j = float(
                    shadow_cpu_power * horizon_complete_branch_process_cpu_s
                    if shadow_cpu_power is not None else 0.0
                )
                if structural is not None:
                    structure_discrete_digest = isac_structure_digest(
                        selected=structural.selected,
                        role=structural.role,
                        owner=structural.owner,
                    )
                    structure_action_digest = isac_action_digest(
                        selected=structural.selected,
                        role=structural.role,
                        owner=structural.owner,
                        sensing_power_w=structure_power,
                        comm_power_w=(
                            structure_transport.projected_comm_power_w
                            if structure_transport is not None
                            and structure_transport.feasible
                            else comm_power
                        ),
                    )
                if (
                    horizon_evaluation is not None
                    and horizon_candidate_comm_power is not None
                ):
                    horizon_discrete_digest = isac_structure_digest(
                        selected=horizon_evaluation.move.selected,
                        role=horizon_evaluation.move.role,
                        owner=horizon_evaluation.move.owner,
                    )
                    horizon_action_digest = isac_action_digest(
                        selected=horizon_evaluation.move.selected,
                        role=horizon_evaluation.move.role,
                        owner=horizon_evaluation.move.owner,
                        sensing_power_w=(
                            horizon_evaluation.rollout.sensing_power_w[0]),
                        comm_power_w=horizon_candidate_comm_power,
                    )
                if horizon_steps > 0:
                    shadow_route_decision = arbitrate_shadow_horizon(
                        ShadowBranchObservation(
                            evaluated=bool(structural is not None),
                            accept=bool(structure_accepted),
                            action_digest=structure_action_digest,
                            structure_digest=structure_discrete_digest,
                            latency_s=float(
                                (0.0
                                 if runtime_latency_calibration_metadata
                                 is not None else
                                 controller_common_preprocessing_s
                                 + structure_ranking_compute_s)
                                + (
                                    structure_transport
                                    .total_protocol_latency_s
                                    if structure_transport is not None
                                    else 0.0
                                )
                            ),
                            control_energy_j=float(
                                structure_protocol_energy_j
                                + structure_compute_energy_j),
                            hard_feasible=bool(
                                structure_sequence_feasible),
                            # Wall/CPU timing includes the common causal
                            # preprocessing and power-only prefix.  Electrical
                            # power and deployment jitter remain unmeasured.
                            resource_accounting_complete=(
                                shadow_resource_accounting_complete),
                        ),
                        ShadowBranchObservation(
                            evaluated=bool(horizon_evaluation is not None),
                            accept=bool(
                                horizon_evaluation is not None
                                and horizon_evaluation.gate.accept),
                            action_digest=horizon_action_digest,
                            structure_digest=horizon_discrete_digest,
                            latency_s=float(
                                (0.0
                                 if runtime_latency_calibration_metadata
                                 is not None else
                                 controller_common_preprocessing_s
                                 + horizon_complete_branch_compute_s)
                                + (
                                    horizon_candidate_transport
                                    .total_protocol_latency_s
                                    if horizon_candidate_transport is not None
                                    else 0.0
                                )
                            ),
                            control_energy_j=float(
                                horizon_protocol_energy_j
                                + horizon_compute_energy_j),
                            hard_feasible=bool(
                                horizon_evaluation is not None
                                and horizon_candidate_transport is not None
                                and horizon_candidate_transport.feasible
                                and ranking_comm_bound_valid is not False),
                            resource_accounting_complete=(
                                shadow_resource_accounting_complete),
                        ),
                        control_period_s=float(cfg.scenario.dt),
                        max_control_energy_j=shadow_energy_limit,
                        shared_control_latency_s=float(
                            shared_complete_compute_latency_bound_s),
                        shared_control_energy_j=float(
                            shared_package_energy_bound_j),
                        allow_horizon_power_on_structure_consensus=bool(
                            shadow_reconcile_horizon_power),
                    )
                horizon_owner_oracle_accept_count = int(sum(
                    evaluation.gate.accept
                    for evaluation in (
                        horizon_owner_proposal_oracle.evaluations
                        if horizon_owner_proposal_oracle is not None else ()
                    )
                ))
                # Set-valued consensus diagnostic.  The one-step branch and
                # the horizon branch remain independently ranked; after both
                # finish, ask whether the already-broadcast owner proposal
                # set contains the one-step branch's discrete structure and
                # whether that member passes the unchanged horizon gate.
                # This is only an upper bound on a deployable reconciliation:
                # per-candidate verification/commit transport and the scan's
                # compute cost are intentionally not charged here.
                owner_oracle_structure_matches = tuple(
                    evaluation
                    for evaluation in (
                        horizon_owner_proposal_oracle.evaluations
                        if horizon_owner_proposal_oracle is not None else ()
                    )
                    if structural is not None
                    and isac_structure_digest(
                        selected=evaluation.move.selected,
                        role=evaluation.move.role,
                        owner=evaluation.move.owner,
                    ) == isac_structure_digest(
                        selected=structural.selected,
                        role=structural.role,
                        owner=structural.owner,
                    )
                )
                owner_oracle_structure_match_accept_count = int(sum(
                    evaluation.gate.accept
                    for evaluation in owner_oracle_structure_matches
                ))
                horizon_raw_oracle_accept_count = int(sum(
                    evaluation.gate.accept
                    for evaluation in (
                        horizon_raw_candidate_oracle.evaluations
                        if horizon_raw_candidate_oracle is not None else ()
                    )
                ))
                raw_oracle_structure_matches = tuple(
                    evaluation
                    for evaluation in (
                        horizon_raw_candidate_oracle.evaluations
                        if horizon_raw_candidate_oracle is not None else ()
                    )
                    if structural is not None
                    and isac_structure_digest(
                        selected=evaluation.move.selected,
                        role=evaluation.move.role,
                        owner=evaluation.move.owner,
                    ) == isac_structure_digest(
                        selected=structural.selected,
                        role=structural.role,
                        owner=structural.owner,
                    )
                )
                raw_oracle_structure_match_accept_count = int(sum(
                    evaluation.gate.accept
                    for evaluation in raw_oracle_structure_matches
                ))
                horizon_hold_outcome_valid = bool(
                    horizon_evaluation is not None
                    and not bool(getattr(
                        cfg.marl, "tracking_enabled", True))
                    and not bool(cfg.channel.use_swerling)
                )
                horizon_realized_candidate_pd = None
                horizon_realized_noop_pd = None
                horizon_candidate_bound_failure = None
                horizon_noop_bound_failure = None
                horizon_realized_target_no_harm = None
                if horizon_hold_outcome_valid:
                    horizon_candidate_gain, _ = fixed_owner_gain_matrix(
                        coefficient,
                        [tuple(edge) for edge in np.argwhere(
                            horizon_evaluation.move.selected)],
                    )
                    horizon_realized_candidate_pd = np.stack([
                        compute_detection_probabilities(
                            np.sum(
                                horizon_candidate_gain
                                * horizon_evaluation.rollout
                                .sensing_power_w[step],
                                axis=0,
                            ),
                            p_fa,
                        )
                        for step in range(horizon_steps)
                    ])
                    horizon_realized_noop_pd = np.stack([
                        compute_detection_probabilities(
                            np.sum(
                                true_gain
                                * horizon_repair.noop
                                .sensing_power_w[step],
                                axis=0,
                            ),
                            p_fa,
                        )
                        for step in range(horizon_steps)
                    ])
                    horizon_candidate_bound_failure = bool(np.any(
                        horizon_evaluation.rollout.lower_pd
                        > horizon_realized_candidate_pd + 1.0e-12))
                    horizon_noop_bound_failure = bool(np.any(
                        horizon_repair.noop.upper_pd + 1.0e-12
                        < horizon_realized_noop_pd))
                    horizon_realized_target_no_harm = bool(np.all(
                        horizon_realized_candidate_pd + 1.0e-12
                        >= horizon_realized_noop_pd))

                # Privileged future audit.  Unlike the static hold check
                # above, step h consumes the physical coefficient at the
                # consecutive trace frame row+h.  These rows are read only
                # after ranking and acceptance are complete.  They represent
                # a frozen open-loop movement-action tape; they are never a
                # feature of the controller or its candidate ranker.
                horizon_future_outcome_valid = False
                horizon_future_outcome_reason = "not_evaluated"
                horizon_future_trace_indices = None
                horizon_future = None
                if (
                    horizon_envelope is not None
                    and (
                        horizon_evaluation is not None
                        or horizon_rendezvous_evaluation is not None
                    )
                ):
                    if bool(getattr(cfg.marl, "tracking_enabled", True)):
                        horizon_future_outcome_reason = (
                            "tracking_feedback_couples_future_movement")
                    elif bool(cfg.channel.use_swerling):
                        horizon_future_outcome_reason = (
                            "swerling_channel_not_frozen_exogenous")
                    else:
                        horizon_future_trace_indices = (
                            consecutive_horizon_indices(
                                seeds,
                                frames,
                                row_id,
                                horizon_steps,
                            )
                        )
                        if horizon_future_trace_indices is None:
                            horizon_future_outcome_reason = (
                                "incomplete_consecutive_episode_horizon")
                        else:
                            future_coefficients = np.stack([
                                _coefficient_from_trace(
                                    data, int(future_row), cfg)
                                for future_row in horizon_future_trace_indices
                            ])
                            if horizon_evaluation is not None:
                                horizon_future = (
                                    evaluate_horizon_future_outcome(
                                        future_coefficients,
                                        horizon_evaluation.move.selected,
                                        selected,
                                        horizon_evaluation.rollout
                                        .sensing_power_w,
                                        horizon_repair.noop.sensing_power_w,
                                        horizon_evaluation.rollout.lower_pd,
                                        horizon_repair.noop.upper_pd,
                                        horizon_envelope.lower,
                                        horizon_envelope.upper,
                                        np.broadcast_to(
                                            support_provenance,
                                            horizon_envelope.lower.shape,
                                        ),
                                        false_alarm_probability=p_fa,
                                    )
                                )
                                horizon_future_outcome_valid = True
                                horizon_future_outcome_reason = "evaluated"
                            if (
                                horizon_rendezvous_evaluation is not None
                                and horizon_rendezvous_repair is not None
                            ):
                                horizon_rendezvous_future = (
                                    evaluate_horizon_future_outcome(
                                        future_coefficients,
                                        horizon_rendezvous_evaluation
                                        .move.selected,
                                        selected,
                                        horizon_rendezvous_evaluation
                                        .rollout.sensing_power_w,
                                        horizon_rendezvous_repair
                                        .noop.sensing_power_w,
                                        horizon_rendezvous_evaluation
                                        .rollout.lower_pd,
                                        horizon_rendezvous_repair
                                        .noop.upper_pd,
                                        horizon_envelope.lower,
                                        horizon_envelope.upper,
                                        np.broadcast_to(
                                            support_provenance,
                                            horizon_envelope.lower.shape,
                                        ),
                                        false_alarm_probability=p_fa,
                                    )
                                )
                report.update({
                    "causal_route": causal_route.route.value,
                    "causal_executed_route": executed_route_name,
                    "true_diagnostic_route": true_route.route.value,
                    "route_matches_true_diagnostic": bool(
                        causal_route.route == true_route.route),
                    "transport_feasible": True,
                    "structure_sequence_transport_enabled": bool(
                        certify_structure_sequence),
                    "structure_sequence_transport_feasible": bool(
                        structure_sequence_feasible),
                    "structure_sequence_transport_reasons": (
                        list(structure_transport.reasons)
                        if structure_transport is not None else []),
                    "structure_sequence_verification_count": int(
                        structure_transport.verification_count
                        if structure_transport is not None else 0),
                    "structure_sequence_owner_proposal_count": int(
                        structure_transport.proposal_count
                        if structure_transport is not None else 0),
                    "structure_sequence_owner_proposal_bits": int(
                        structure_transport.proposal_bits
                        if structure_transport is not None else 0),
                    "structure_sequence_owner_proposal_latency_s": float(
                        structure_transport.proposal_latency_s
                        if structure_transport is not None else 0.0),
                    "structure_sequence_owner_proposal_energy_j": float(
                        structure_transport.proposal_energy_j
                        if structure_transport is not None else 0.0),
                    "structure_sequence_target_invariant_packet_count": int(
                        structure_transport.invariant_packet_count
                        if structure_transport is not None else 0),
                    "structure_sequence_target_invariant_record_count": int(
                        structure_transport.invariant_record_count
                        if structure_transport is not None else 0),
                    "structure_sequence_target_invariant_bits": int(
                        structure_transport.invariant_bits
                        if structure_transport is not None else 0),
                    "structure_sequence_target_invariant_latency_s": float(
                        structure_transport.invariant_latency_s
                        if structure_transport is not None else 0.0),
                    "structure_sequence_target_invariant_energy_j": float(
                        structure_transport.invariant_energy_j
                        if structure_transport is not None else 0.0),
                    "horizon_diagnostic_enabled": bool(horizon_steps > 0),
                    "horizon_diagnostic_steps": int(horizon_steps),
                    "horizon_diagnostic_reason": horizon_diagnostic_reason,
                    "horizon_diagnostic_evaluated": bool(
                        horizon_repair is not None),
                    "horizon_diagnostic_accept": bool(
                        horizon_repair.accept
                        if horizon_repair is not None else False),
                    "horizon_diagnostic_candidate_count": int(
                        len(horizon_repair.evaluations)
                        if horizon_repair is not None else 0),
                    "horizon_owner_proposal_oracle_candidate_count": int(
                        len(horizon_owner_proposal_oracle.evaluations)
                        if horizon_owner_proposal_oracle is not None else 0),
                    "horizon_owner_proposal_oracle_accept_count": int(
                        horizon_owner_oracle_accept_count),
                    "horizon_owner_proposal_oracle_any_accept": bool(
                        horizon_owner_oracle_accept_count > 0),
                    "horizon_owner_proposal_oracle_top1_regret": bool(
                        horizon_owner_oracle_accept_count > 0
                        and not bool(
                            horizon_repair.accept
                            if horizon_repair is not None else False)),
                    "horizon_owner_proposal_oracle_structure_match_count": (
                        len(owner_oracle_structure_matches)),
                    "horizon_owner_proposal_oracle_structure_match_accept_count": (
                        owner_oracle_structure_match_accept_count),
                    "horizon_owner_proposal_oracle_structure_match_any_accept": bool(
                        owner_oracle_structure_match_accept_count > 0),
                    "horizon_owner_proposal_oracle_structure_match_resource_certified": False,
                    "horizon_raw_candidate_oracle_candidate_count": int(
                        len(horizon_raw_candidate_oracle.evaluations)
                        if horizon_raw_candidate_oracle is not None else 0),
                    "horizon_raw_candidate_oracle_accept_count": int(
                        horizon_raw_oracle_accept_count),
                    "horizon_raw_candidate_oracle_any_accept": bool(
                        horizon_raw_oracle_accept_count > 0),
                    "horizon_raw_candidate_oracle_owner_compression_regret": (
                        bool(
                            horizon_raw_oracle_accept_count > 0
                            and horizon_owner_oracle_accept_count == 0)),
                    "horizon_raw_candidate_oracle_structure_match_count": (
                        len(raw_oracle_structure_matches)),
                    "horizon_raw_candidate_oracle_structure_match_accept_count": (
                        raw_oracle_structure_match_accept_count),
                    "horizon_raw_candidate_oracle_structure_match_any_accept": bool(
                        raw_oracle_structure_match_accept_count > 0),
                    "horizon_raw_candidate_oracle_structure_match_resource_certified": False,
                    "horizon_ranked_candidate_count": int(
                        len(horizon_membership_candidate_set.moves)
                        if horizon_membership_candidate_set is not None
                        else (
                            horizon_ranked_structural.candidate_count
                            if horizon_ranked_structural is not None else 0
                        )),
                    "horizon_set_membership_verifier": bool(
                        horizon_set_membership_verifier),
                    "horizon_reuse_primal_master_duals": bool(
                        horizon_reuse_primal_master_duals),
                    "horizon_ranked_owner_proposal_count": int(
                        horizon_ranked_structural.owner_proposal_count
                        if horizon_ranked_structural is not None else 0),
                    "horizon_ranking_proxy_lower": (
                        None if horizon_ranked_structural is None else float(
                            horizon_ranked_structural.proxy_best_lower)),
                    "horizon_ranking_compute_s": float(
                        horizon_ranking_compute_s),
                    "horizon_ranking_process_cpu_s": float(
                        horizon_ranking_process_cpu_s),
                    "horizon_complete_branch_compute_s": float(
                        horizon_complete_branch_compute_s),
                    "horizon_complete_branch_process_cpu_s": float(
                        horizon_complete_branch_process_cpu_s),
                    "horizon_propagation_compute_s": float(
                        horizon_propagation_compute_s),
                    "horizon_envelope_compute_s": float(
                        horizon_envelope_compute_s),
                    "horizon_candidate_set_compute_s": float(
                        horizon_candidate_set_compute_s),
                    "horizon_ranking_parallel_compute_s": float(
                        horizon_ranking_parallel_compute_s),
                    "horizon_ranking_owner_compute_s": list(
                        horizon_ranking_owner_compute_s),
                    "horizon_owner_concurrent_measured": bool(
                        horizon_owner_concurrent_measured),
                    "horizon_owner_worker_count": int(
                        horizon_owner_worker_count),
                    "horizon_owner_sum_thread_cpu_s": float(
                        horizon_owner_sum_thread_cpu_s),
                    "horizon_ranked_top1_matches_myopic": (
                        horizon_top1_matches_myopic),
                    "horizon_ranking_comm_upper_w": (
                        None if horizon_ranking_comm_power is None else
                        horizon_ranking_comm_power[0].tolist()),
                    "horizon_ranking_comm_bound_valid": (
                        ranking_comm_bound_valid),
                    "horizon_candidate_comm_power_w": (
                        None if horizon_candidate_comm_power is None else
                        horizon_candidate_comm_power.tolist()),
                    "horizon_candidate_max_comm_power_w": (
                        None if horizon_candidate_comm_power is None else
                        float(np.max(horizon_candidate_comm_power))),
                    "horizon_diagnostic_net_discounted_gain_lower": (
                        None if horizon_evaluation is None else float(
                            horizon_evaluation.net_discounted_gain_lower)),
                    "horizon_diagnostic_candidate_lower_pd": (
                        None if horizon_evaluation is None else
                        horizon_evaluation.rollout.lower_pd.tolist()),
                    "horizon_diagnostic_gate_mode": (
                        None if horizon_evaluation is None else
                        horizon_evaluation.gate.certificate_mode),
                    "horizon_diagnostic_gain_unit": (
                        None if horizon_evaluation is None else
                        horizon_evaluation.gate.gain_unit),
                    "horizon_diagnostic_paired_deflection_lower": (
                        None
                        if horizon_evaluation is None
                        or horizon_evaluation.paired_deflection_lower is None
                        else horizon_evaluation
                        .paired_deflection_lower.tolist()),
                    "horizon_diagnostic_noop_upper_pd": (
                        None if horizon_repair is None else
                        horizon_repair.noop.upper_pd.tolist()),
                    "horizon_diagnostic_target_safe": (
                        None if horizon_evaluation is None else
                        horizon_evaluation.gate.target_safe.tolist()),
                    "horizon_diagnostic_transport_bits": int(
                        horizon_candidate_transport.total_over_air_bits
                        if horizon_candidate_transport is not None else 0),
                    "horizon_diagnostic_transport_latency_s": float(
                        horizon_candidate_transport.total_protocol_latency_s
                        if horizon_candidate_transport is not None else 0.0),
                    "horizon_diagnostic_transport_energy_j": float(
                        horizon_candidate_transport.total_energy_j
                        if horizon_candidate_transport is not None else 0.0),
                    "horizon_diagnostic_ranking_compute_latency_s": float(
                        horizon_ranking_parallel_compute_s),
                    "horizon_diagnostic_end_to_end_latency_s": float(
                        (
                            horizon_candidate_transport
                            .total_protocol_latency_s
                            if horizon_candidate_transport is not None
                            else 0.0
                        ) + horizon_complete_branch_compute_s
                        + controller_common_preprocessing_s),
                    "controller_common_preprocessing_s": float(
                        controller_common_preprocessing_s),
                    "controller_common_preprocessing_cpu_s": float(
                        controller_common_preprocessing_cpu_s),
                    "controller_common_instrumented_wall_s": float(
                        controller_common_instrumented_wall_s),
                    "controller_common_instrumented_cpu_s": float(
                        controller_common_instrumented_cpu_s),
                    "controller_privileged_audit_s": float(
                        controller_privileged_audit_s),
                    "controller_privileged_audit_cpu_s": float(
                        controller_privileged_audit_cpu_s),
                    "horizon_digest_rendezvous_enabled": bool(
                        horizon_digest_rendezvous),
                    "horizon_digest_rendezvous_attempted": bool(
                        horizon_rendezvous_transport is not None),
                    "horizon_digest_rendezvous_deferred_top1_transport": bool(
                        horizon_digest_rendezvous_defer_top1_transport),
                    "horizon_digest_rendezvous_prefix_feasible": (
                        None if horizon_rendezvous_prefix is None else
                        bool(horizon_rendezvous_prefix.feasible)),
                    "horizon_digest_rendezvous_prefix_reasons": (
                        [] if horizon_rendezvous_prefix is None else
                        list(horizon_rendezvous_prefix.reasons)),
                    "horizon_digest_rendezvous_prefix_bits": (
                        0 if horizon_rendezvous_prefix is None else
                        int(horizon_rendezvous_prefix.total_over_air_bits)),
                    "horizon_digest_rendezvous_prefix_latency_s": (
                        0.0 if horizon_rendezvous_prefix is None else
                        float(horizon_rendezvous_prefix
                              .total_protocol_latency_s)),
                    "horizon_digest_rendezvous_prefix_energy_j": (
                        0.0 if horizon_rendezvous_prefix is None else
                        float(horizon_rendezvous_prefix.total_energy_j)),
                    "horizon_digest_rendezvous_match_proposer": (
                        horizon_rendezvous_match_proposer),
                    "horizon_digest_rendezvous_transport_feasible": (
                        None if horizon_rendezvous_transport is None else
                        bool(horizon_rendezvous_transport.feasible)),
                    "horizon_digest_rendezvous_transport_reasons": (
                        [] if horizon_rendezvous_transport is None else
                        list(horizon_rendezvous_transport.reasons)),
                    "horizon_digest_rendezvous_full_structure_verified": (
                        None if horizon_rendezvous_transport is None else
                        bool(horizon_rendezvous_transport
                             .full_structure_verified)),
                    "horizon_digest_rendezvous_digest_bits": (
                        0 if horizon_rendezvous_transport is None else
                        int(horizon_rendezvous_transport.digest_bits)),
                    "horizon_digest_rendezvous_collision_probability_upper": (
                        None if horizon_rendezvous_transport is None else
                        float(horizon_rendezvous_transport
                              .collision_probability_upper)),
                    "horizon_digest_rendezvous_bits": (
                        0 if horizon_rendezvous_transport is None else
                        int(horizon_rendezvous_transport
                            .total_over_air_bits)),
                    "horizon_digest_rendezvous_latency_s": (
                        0.0 if horizon_rendezvous_transport is None else
                        float(horizon_rendezvous_transport
                              .total_protocol_latency_s)),
                    "horizon_digest_rendezvous_energy_j": (
                        0.0 if horizon_rendezvous_transport is None else
                        float(horizon_rendezvous_transport.total_energy_j)),
                    "horizon_digest_rendezvous_suffix_feasible": (
                        None if horizon_rendezvous_suffix is None else
                        bool(horizon_rendezvous_suffix.feasible)),
                    "horizon_digest_rendezvous_structure_commit_reused": (
                        None if horizon_rendezvous_suffix is None else
                        bool(horizon_rendezvous_suffix
                             .dependency_commit_reused)),
                    "horizon_digest_rendezvous_suffix_reasons": (
                        [] if horizon_rendezvous_suffix is None else
                        list(horizon_rendezvous_suffix.reasons)),
                    "horizon_digest_rendezvous_suffix_bits": (
                        0 if horizon_rendezvous_suffix is None else
                        int(horizon_rendezvous_suffix.total_over_air_bits)),
                    "horizon_digest_rendezvous_suffix_latency_s": (
                        0.0 if horizon_rendezvous_suffix is None else
                        float(horizon_rendezvous_suffix
                              .total_protocol_latency_s)),
                    "horizon_digest_rendezvous_suffix_energy_j": (
                        0.0 if horizon_rendezvous_suffix is None else
                        float(horizon_rendezvous_suffix.total_energy_j)),
                    "horizon_digest_rendezvous_power_balance_error_w": (
                        None if horizon_rendezvous_suffix is None else
                        float(horizon_rendezvous_suffix
                              .max_isac_power_balance_error_w)),
                    "horizon_digest_rendezvous_comm_power_w": (
                        None if horizon_rendezvous_suffix is None else
                        horizon_rendezvous_suffix
                        .projected_comm_power_w.tolist()),
                    "horizon_digest_rendezvous_comm_bound_valid": (
                        horizon_rendezvous_comm_bound_valid),
                    "horizon_digest_rendezvous_compute_s": float(
                        horizon_rendezvous_compute_s),
                    "horizon_digest_rendezvous_noop_rollout_compute_s": (
                        0.0 if horizon_rendezvous_repair is None else
                        float(horizon_rendezvous_repair
                              .noop_rollout_compute_s)),
                    "horizon_digest_rendezvous_candidate_rollout_compute_s": (
                        0.0 if horizon_rendezvous_evaluation is None else
                        float(horizon_rendezvous_evaluation
                              .rollout_compute_s)),
                    "horizon_digest_rendezvous_process_cpu_s": float(
                        horizon_rendezvous_process_cpu_s),
                    "horizon_digest_rendezvous_exact_accept": bool(
                        horizon_rendezvous_evaluation is not None
                        and horizon_rendezvous_evaluation.gate.accept),
                    "horizon_digest_rendezvous_exact_target_safe": (
                        None if horizon_rendezvous_evaluation is None else
                        horizon_rendezvous_evaluation
                        .gate.target_safe.tolist()),
                    "horizon_digest_rendezvous_exact_net_gain_lower": (
                        None if horizon_rendezvous_evaluation is None else
                        float(horizon_rendezvous_evaluation
                              .net_discounted_gain_lower)),
                    "horizon_digest_rendezvous_total_latency_s": (
                        horizon_rendezvous_total_latency_s),
                    "horizon_digest_rendezvous_nominal_total_latency_s": (
                        horizon_rendezvous_nominal_total_latency_s),
                    "horizon_network_repetition_count": int(
                        network_repetitions),
                    "horizon_network_tolerated_erasures_per_packet": int(
                        network_repetitions - 1),
                    "horizon_network_excess_queue_bound_s": float(
                        network_queue_bound),
                    "horizon_network_repetition_feasible": (
                        None
                        if horizon_rendezvous_repetition_certificate is None
                        else bool(
                            horizon_rendezvous_repetition_certificate.feasible)
                    ),
                    "horizon_network_repetition_reasons": (
                        []
                        if horizon_rendezvous_repetition_certificate is None
                        else list(
                            horizon_rendezvous_repetition_certificate.reasons)
                    ),
                    "horizon_network_nominal_protocol_bits": (
                        0
                        if horizon_rendezvous_repetition_certificate is None
                        else int(horizon_rendezvous_repetition_certificate
                                 .nominal_over_air_bits)
                    ),
                    "horizon_network_worst_protocol_bits": (
                        0
                        if horizon_rendezvous_repetition_certificate is None
                        else int(horizon_rendezvous_repetition_certificate
                                 .worst_case_over_air_bits)
                    ),
                    "horizon_network_nominal_rf_energy_j": (
                        0.0
                        if horizon_rendezvous_repetition_certificate is None
                        else float(horizon_rendezvous_repetition_certificate
                                   .nominal_rf_energy_j)
                    ),
                    "horizon_network_worst_rf_energy_j": (
                        0.0
                        if horizon_rendezvous_repetition_certificate is None
                        else float(horizon_rendezvous_repetition_certificate
                                   .worst_case_rf_energy_j)
                    ),
                    "horizon_digest_rendezvous_deadline_pass": (
                        None
                        if horizon_rendezvous_total_latency_s is None
                        else bool(
                            horizon_rendezvous_total_latency_s
                            <= float(cfg.scenario.dt) + 1.0e-12)),
                    "horizon_digest_rendezvous_additional_eligible": bool(
                        horizon_rendezvous_additional_eligible),
                    "horizon_digest_rendezvous_future_valid": bool(
                        horizon_rendezvous_future is not None),
                    "horizon_digest_rendezvous_future_candidate_bound_failure": (
                        None if horizon_rendezvous_future is None else
                        bool(horizon_rendezvous_future
                             .candidate_lower_failure)),
                    "horizon_digest_rendezvous_future_noop_bound_failure": (
                        None if horizon_rendezvous_future is None else
                        bool(horizon_rendezvous_future.noop_upper_failure)),
                    "horizon_digest_rendezvous_future_target_no_harm": (
                        None if horizon_rendezvous_future is None else
                        bool(np.all(
                            horizon_rendezvous_future.target_no_harm))),
                    "horizon_digest_rendezvous_future_candidate_pd": (
                        None if horizon_rendezvous_future is None else
                        horizon_rendezvous_future.candidate_pd.tolist()),
                    "horizon_digest_rendezvous_future_noop_pd": (
                        None if horizon_rendezvous_future is None else
                        horizon_rendezvous_future.noop_pd.tolist()),
                    "horizon_digest_rendezvous_commit_authority": False,
                    "structure_ranking_compute_s": float(
                        structure_ranking_compute_s),
                    "structure_ranking_process_cpu_s": float(
                        structure_ranking_process_cpu_s),
                    "structure_candidate_action_digest": (
                        structure_action_digest),
                    "structure_candidate_discrete_digest": (
                        structure_discrete_digest),
                    "horizon_candidate_action_digest": (
                        horizon_action_digest),
                    "horizon_candidate_discrete_digest": (
                        horizon_discrete_digest),
                    "shadow_router_enabled": bool(horizon_steps > 0),
                    "shadow_router_commit_authority": False,
                    "shadow_router_live_action": (
                        None if shadow_route_decision is None else
                        shadow_route_decision.live_action),
                    "shadow_router_action": (
                        None if shadow_route_decision is None else
                        shadow_route_decision.shadow_action),
                    "shadow_router_reason": (
                        None if shadow_route_decision is None else
                        shadow_route_decision.reason),
                    "shadow_router_candidate_identity_match": (
                        None if shadow_route_decision is None else
                        shadow_route_decision.candidate_identity_agrees),
                    "shadow_router_structure_identity_match": (
                        None if shadow_route_decision is None else
                        shadow_route_decision.structure_identity_agrees),
                    "shadow_router_reconcile_horizon_power": bool(
                        shadow_reconcile_horizon_power),
                    "shadow_router_parallel_latency_s": (
                        None if shadow_route_decision is None else float(
                            shadow_route_decision.parallel_latency_s)),
                    "shadow_router_shared_complete_compute_latency_bound_s": (
                        None if shadow_route_decision is None else float(
                            shadow_route_decision.shared_control_latency_s)),
                    "shadow_router_total_control_energy_j": (
                        None if shadow_route_decision is None else float(
                            shadow_route_decision.total_control_energy_j)),
                    "shadow_router_shared_package_energy_bound_j": (
                        None if shadow_route_decision is None else float(
                            shadow_route_decision.shared_control_energy_j)),
                    "shadow_router_resource_accounting_complete": bool(
                        shadow_resource_accounting_complete),
                    "shadow_router_logical_cpu_power_w": (
                        shadow_cpu_power),
                    "shadow_router_max_control_energy_j": (
                        shadow_energy_limit),
                    "horizon_hold_outcome_valid": bool(
                        horizon_hold_outcome_valid),
                    "horizon_hold_candidate_bound_failure": (
                        horizon_candidate_bound_failure),
                    "horizon_hold_noop_bound_failure": (
                        horizon_noop_bound_failure),
                    "horizon_hold_realized_target_no_harm": (
                        horizon_realized_target_no_harm),
                    "horizon_hold_realized_candidate_pd": (
                        None if horizon_realized_candidate_pd is None else
                        horizon_realized_candidate_pd.tolist()),
                    "horizon_hold_realized_noop_pd": (
                        None if horizon_realized_noop_pd is None else
                        horizon_realized_noop_pd.tolist()),
                    "horizon_future_outcome_valid": bool(
                        horizon_future_outcome_valid),
                    "horizon_future_outcome_reason": str(
                        horizon_future_outcome_reason),
                    "horizon_future_outcome_information_policy": (
                        "post_decision_privileged_frozen_movement_action_tape"
                        if horizon_future_outcome_valid else None),
                    "horizon_future_trace_indices": (
                        None if horizon_future_trace_indices is None else
                        horizon_future_trace_indices.tolist()),
                    "horizon_future_trace_frames": (
                        None if horizon_future_trace_indices is None else
                        frames[horizon_future_trace_indices].tolist()),
                    "horizon_future_candidate_bound_failure": (
                        None if horizon_future is None else bool(
                            horizon_future.candidate_lower_failure)),
                    "horizon_future_noop_bound_failure": (
                        None if horizon_future is None else bool(
                            horizon_future.noop_upper_failure)),
                    "horizon_future_realized_target_no_harm": (
                        None if horizon_future is None else bool(np.all(
                            horizon_future.target_no_harm))),
                    "horizon_future_realized_target_no_harm_by_step": (
                        None if horizon_future is None else
                        horizon_future.target_no_harm.tolist()),
                    "horizon_future_realized_candidate_pd": (
                        None if horizon_future is None else
                        horizon_future.candidate_pd.tolist()),
                    "horizon_future_realized_noop_pd": (
                        None if horizon_future is None else
                        horizon_future.noop_pd.tolist()),
                    "horizon_future_coefficient_audit_count": (
                        0 if horizon_future is None else int(
                            horizon_future.coefficient_score
                            .audited_coefficient_count)),
                    "horizon_future_lower_zero_support_failure_count": (
                        0 if horizon_future is None else int(
                            horizon_future.coefficient_score
                            .lower_zero_support_failure_count)),
                    "horizon_future_upper_zero_support_failure_count": (
                        0 if horizon_future is None else int(
                            horizon_future.coefficient_score
                            .upper_zero_support_failure_count)),
                    "horizon_future_lower_log_score": (
                        None if horizon_future is None or not np.isfinite(
                            horizon_future.coefficient_score.lower) else float(
                                horizon_future.coefficient_score.lower)),
                    "horizon_future_upper_log_score": (
                        None if horizon_future is None or not np.isfinite(
                            horizon_future.coefficient_score.upper) else float(
                                horizon_future.coefficient_score.upper)),
                    "horizon_future_joint_log_score": (
                        None if horizon_future is None or not np.isfinite(
                            horizon_future.coefficient_score.joint) else float(
                                horizon_future.coefficient_score.joint)),
                    "structure_sequence_commit_count": int(
                        structure_transport.commit_count
                        if structure_transport is not None else 0),
                    "structure_sequence_baseline_bits": int(
                        structure_transport.baseline_verification_bits
                        if structure_transport is not None else 0),
                    "structure_sequence_verification_bits": int(
                        structure_transport.verification_bits
                        if structure_transport is not None else 0),
                    "structure_sequence_commit_bits": int(
                        structure_transport.commit_bits
                        if structure_transport is not None else 0),
                    "structure_sequence_total_bits": int(
                        structure_transport.total_over_air_bits
                        if structure_transport is not None else 0),
                    "structure_sequence_max_packet_latency_s": float(
                        structure_transport.max_packet_latency_s
                        if structure_transport is not None else 0.0),
                    "structure_sequence_total_protocol_latency_s": float(
                        structure_transport.total_protocol_latency_s
                        if structure_transport is not None else 0.0),
                    "structure_sequence_total_energy_j": float(
                        structure_transport.total_energy_j
                        if structure_transport is not None else 0.0),
                    "structure_sequence_max_isac_power_balance_error_w": float(
                        structure_transport.max_isac_power_balance_error_w
                        if structure_transport is not None else 0.0),
                    "structure_triggered": structure_triggered,
                    "structure_accepted": structure_accepted,
                    "structure_kind": (
                        structural.accepted_kind
                        if structural is not None else "not_triggered"),
                    "power_only_estimated_worst": float(np.min(
                        estimated_power_pd)),
                    "power_only_worst": float(np.min(realized_power_pd)),
                    "power_only_pd_max_overprediction": float(np.max(
                        estimated_power_pd - realized_power_pd)),
                    "noop_estimated_worst": float(np.min(noop_estimated_pd)),
                    "noop_pd_max_overprediction": float(np.max(
                        noop_estimated_pd - deployed_pd)),
                    "structure_estimated_worst": float(np.min(
                        estimated_structure_pd)),
                    "structure_worst": float(np.min(realized_structure_pd)),
                    "structure_harmed_vs_power_only": bool(
                        np.min(realized_structure_pd)
                        < np.min(realized_power_pd) - 1.0e-12),
                    "structure_incremental_gain": float(
                        np.min(realized_structure_pd)
                        - np.min(realized_power_pd)),
                    "structure_candidate_count": int(
                        structural.candidate_count
                        if structural is not None else 0),
                    "structure_owner_proposal_count": int(
                        structural.owner_proposal_count
                        if structural is not None else 0),
                    "structure_exact_verification_count": int(
                        structural.exact_verification_count
                        if structural is not None else 0),
                    "structure_changed_edge_count": int(
                        structural.changed_edge_count
                        if structural is not None else 0),
                    "structure_accepted_steps": int(
                        structural.accepted_steps
                        if structural is not None else 0),
                    "structure_changed_target_count": int(
                        structural.changed_target_count
                        if structural is not None else 0),
                    "structure_pd_abs_error": float(np.mean(np.abs(
                        estimated_structure_pd - realized_structure_pd))),
                    "structure_pd_max_overprediction": float(np.max(
                        estimated_structure_pd - realized_structure_pd)),
                    "structure_delta_overprediction": float(max(
                        (
                            float(np.min(estimated_structure_pd))
                            - float(np.min(noop_estimated_pd))
                        )
                        - (
                            float(np.min(realized_structure_pd))
                            - float(np.min(deployed_pd))
                        ),
                        0.0,
                    )),
                    "causal_routed_available": routed_action,
                    "causal_routed_estimated_worst": float(np.min(
                        routed_estimated_pd)),
                    "causal_routed_worst": float(np.min(routed_realized_pd)),
                    "causal_routed_estimated_pd": (
                        routed_estimated_pd.tolist()),
                    "causal_routed_realized_pd": (
                        routed_realized_pd.tolist()),
                    "causal_routed_noop_estimated_worst": float(np.min(
                        noop_estimated_pd)),
                    "causal_routed_noop_estimated_pd": (
                        noop_estimated_pd.tolist()),
                    "causal_routed_absolute_uncertainty": (
                        absolute_uncertainty.tolist()),
                    "causal_routed_noop_upper_uncertainty": (
                        noop_upper_uncertainty.tolist()),
                    "target_reserve_enabled": bool(
                        target_reserve_deflection is not None),
                    "target_reserve_possible": bool(
                        target_reserve_possible),
                    "target_reserve_pd": (
                        None if target_reserve_pd is None
                        else target_reserve_pd.tolist()),
                    "target_reserve_deflection": (
                        None if target_reserve_deflection is None
                        else target_reserve_deflection.tolist()),
                    "fixed_target_reserve_feasible": bool(
                        fixed_reserve_feasible),
                    "structure_target_reserve_feasible": bool(
                        structure_reserve_feasible),
                    "causal_routed_delta_uncertainty": (
                        delta_uncertainty.tolist()),
                    "causal_routed_worst_delta_uncertainty": float(
                        worst_delta_uncertainty),
                    "causal_routed_physics_lower_pd": (
                        routed_physics_lower_pd.tolist()),
                    "causal_routed_physics_upper_pd": (
                        routed_physics_upper_pd.tolist()),
                    "causal_routed_noop_physics_lower_pd": (
                        physics_noop_pd.tolist()),
                    "causal_routed_noop_physics_upper_pd": (
                        physics_noop_upper_pd.tolist()),
                    "causal_routed_noop_physics_certificate_upper_pd": (
                        physics_noop_upper_certificate.tolist()),
                    "causal_routed_noop_physics_incomplete_target": (
                        noop_incomplete_target.tolist()),
                    "causal_routed_calibration_score": float(max(
                        float(np.max(
                            routed_estimated_pd - routed_realized_pd)), 0.0)),
                    "causal_routed_noop_calibration_score": float(max(
                        float(np.max(noop_estimated_pd - deployed_pd)), 0.0)),
                    "causal_routed_harmed": bool(
                        np.min(routed_realized_pd)
                        < np.min(deployed_pd) - 1.0e-12),
                    "causal_routed_power_total_variation": float(
                        routed_power_total_variation),
                    "transport_total_bits": int(routed_transport_bits),
                    "transport_max_packet_latency_s": float(
                        routed_transport_max_packet_latency),
                    "transport_total_protocol_latency_s": float(
                        routed_transport_protocol_latency),
                    "transport_total_energy_j": float(
                        routed_transport_energy),
                    "transport_max_isac_power_balance_error_w": float(
                        routed_power_error),
                    "predicted_fixed_upper_worst": float(np.min(
                        fixed_route_upper_pd)),
                    "predicted_relaxed_upper_worst": float(np.min(
                        relaxed_route_upper_pd)),
                    "owner_local_detection_feedback_available": bool(
                        feedback_available),
                    "owner_local_detection_feedback_worst": float(np.min(
                        noop_estimated_pd)),
                    "owner_local_physics_noop_worst": float(np.min(
                        physics_noop_pd)),
                    "true_fixed_upper_worst": _worst_pd(
                        true_fixed_exact.deflection, p_fa),
                    "true_relaxed_upper_worst": _worst_pd(true_relaxed, p_fa),
                    "full_prediction_direct_edge_count": int(
                        prediction_direct_count),
                    "full_prediction_fallback_edge_count": int(
                        prediction_fallback_count),
                    "full_prediction_cached_target_fallback_edge_count": int(
                        prediction_cached_fallback_count),
                    "target_invariant_cache_enabled": bool(
                        cache_max_age > 0),
                    "target_invariant_cache_elapsed_frames": int(
                        cache_elapsed_frames),
                    "target_invariant_cache_available_target_count": int(
                        cache_available_target_count),
                    "target_invariant_cache_refreshed_target_count": int(
                        cache_refreshed_target_count),
                    "target_invariant_cache_retained_target_count": int(
                        cache_retained_target_count),
                    "target_invariant_cache_expired_target_count": int(
                        cache_expired_target_count),
                    "target_invariant_cache_max_available_age_frames": int(
                        cache_max_available_age_frames),
                    "structural_direct_support_edge_count": int(np.sum(
                        structural_support & directly_observed)),
                    "structural_selected_fallback_edge_count": int(np.sum(
                        structural_support & ~directly_observed)),
                    "selected_edge_excluded_by_owner_local_support_count": int(
                        np.sum(selected & ~structural_support)
                        if observation_mode == "owner_local_conformal" else 0),
                    "selected_edge_false_physical_support_count": int(
                        np.sum(selected & ~physical_support)),
                    "physical_support_edge_count": int(np.sum(support)),
                    "provenance_observed_support_edge_count": int(np.sum(
                        support & directly_observed)),
                    "excluded_unobserved_support_edge_count": int(np.sum(
                        support & ~directly_observed)),
                    "unknown_predicted_edge_count": int(np.sum(
                        unknown_prediction_mask)),
                    "unknown_edge_log_overprediction_score": float(
                        unknown_log_overprediction),
                    "unknown_edge_log_underprediction_score": float(
                        unknown_log_underprediction),
                    "all_edge_log_overprediction_score": float(
                        all_log_overprediction),
                    "all_edge_log_underprediction_score": float(
                        all_log_underprediction),
                    "unpredicted_positive_edge_count": int(
                        unpredicted_positive_edge_count),
                    "owner_local_token_candidate_edge_count": int(
                        np.sum(owner_local_candidate[row_id])
                        if observation_mode == "owner_local_conformal" else 0),
                    "owner_local_dd_admitted_edge_count": int(
                        np.sum(dd_admitted)
                        if observation_mode == "owner_local_conformal" else 0),
                    "selected_provenance_renewal_edge_count": int(
                        np.sum(selected_provenance_renewal)
                        if observation_mode == "owner_local_conformal" else 0),
                    "selected_provenance_renewal_new_edge_count": int(
                        np.sum(selected_provenance_renewal & ~dd_admitted)
                        if observation_mode == "owner_local_conformal" else 0),
                    "selected_dd_unsafe_edge_count": int(
                        np.sum(selected & ~dd_safe)
                        if observation_mode == "owner_local_conformal" else 0),
                    "selected_dd_safe_without_provenance_edge_count": int(
                        np.sum(
                            selected & dd_safe
                            & ~np.asarray(
                                owner_local_candidate[row_id], dtype=bool)
                            & ~directly_observed)
                        if observation_mode == "owner_local_conformal" else 0),
                    "owner_local_safe_support_edge_count": int(
                        np.sum(dd_admitted | selected_provenance_renewal)
                        if observation_mode == "owner_local_conformal" else 0),
                    "owner_local_dd_false_admitted_edge_count": int(
                        np.sum(
                            (dd_admitted | selected_provenance_renewal)
                            & ~physical_support)
                        if observation_mode == "owner_local_conformal" else 0),
                    "owner_local_dd_false_support_score": float(
                        dd_false_support_score),
                    "owner_local_dd_covariance_radius": float(
                        dd_covariance_radius),
                    "owner_local_dd_additive_margin": float(
                        dd_support_margin),
                    "owner_local_dd_max_delay_bin_radius": float(
                        dd_max_delay_bin_radius),
                    "owner_local_dd_max_doppler_bin_radius": float(
                        dd_max_doppler_bin_radius),
                    "owner_local_dd_mean_geometry_tightening": float(
                        np.mean(dd_prediction - dd_geometry_lower)),
                })
        applied_selected = selected
        applied_comm_power = comm_power
        applied_sensing_power = sensing_power
        applied_pd = deployed_pd
        controller_accept = False
        if bool(persistent_closed_loop):
            decision = None
            physics_decision = None
            risk_decision = None
            feedback_accept = False
            physics_accept = False
            if bool(report.get("causal_routed_available", False)):
                decision = certified_targetwise_feedback_decision(
                    candidate_estimated=np.asarray(
                        report["causal_routed_estimated_pd"],
                        dtype=np.float64,
                    ),
                    noop_estimated=np.asarray(
                        report["causal_routed_noop_estimated_pd"],
                        dtype=np.float64,
                    ),
                    absolute_uncertainty=np.asarray(
                        report["causal_routed_absolute_uncertainty"],
                        dtype=np.float64,
                    ),
                    noop_upper_uncertainty=np.asarray(
                        report["causal_routed_noop_upper_uncertainty"],
                        dtype=np.float64,
                    ),
                    normalized_margin=float(controller_margin),
                    qos_floor=float(qos_floor),
                )
                physics_decision = certified_physics_interval_decision(
                    candidate_lower=np.asarray(
                        report["causal_routed_physics_lower_pd"],
                        dtype=np.float64,
                    ),
                    noop_lower=np.asarray(
                        report["causal_routed_noop_physics_lower_pd"],
                        dtype=np.float64,
                    ),
                    noop_upper=np.asarray(
                        report[
                            "causal_routed_noop_physics_certificate_upper_pd"
                        ],
                        dtype=np.float64,
                    ),
                    qos_floor=float(qos_floor),
                )
                feedback_accept = bool(decision.accept)
                physics_accept = bool(physics_decision.accept)
                if certificate_mode == "risk_budgeted_union":
                    if controller_risk_budget is None:
                        raise RuntimeError(
                            "controller risk budget was not initialized")
                    risk_decision = risk_budgeted_certificate_union(
                        {
                            "feedback": feedback_accept,
                            "physics": physics_accept,
                        },
                        risk_budget=controller_risk_budget,
                        hard_feasible=bool(report.get(
                            "transport_feasible", False)),
                    )
                    controller_accept = bool(risk_decision.accept)
                else:
                    controller_accept = bool(
                        (
                            feedback_accept
                            or (
                                certificate_mode == "feedback_or_physics"
                                and physics_accept
                            )
                        )
                        and report.get("transport_feasible", False)
                    )
            if controller_accept:
                applied_selected = candidate_selected.copy()
                applied_comm_power = candidate_comm_power.copy()
                applied_sensing_power = candidate_sensing_power.copy()
                applied_pd = candidate_realized_pd.copy()
                deployed_state_version[seed] = (
                    int(deployed_state_version.get(seed, 0)) + 1)
            deployed_selected[seed] = np.asarray(
                applied_selected, dtype=bool).copy()
            deployed_comm_power[seed] = np.asarray(
                applied_comm_power, dtype=np.float64).copy()
            deployed_sensing_power[seed] = np.asarray(
                applied_sensing_power, dtype=np.float64).copy()
            deployed_last_row[seed] = row_id
            report.update({
                "closed_loop_controller_accept": controller_accept,
                "closed_loop_certificate_mode": certificate_mode,
                "closed_loop_feedback_certificate_accept": feedback_accept,
                "closed_loop_physics_certificate_accept": physics_accept,
                "closed_loop_joint_risk_budget_enabled": bool(
                    controller_risk_budget is not None),
                "closed_loop_total_miscoverage": (
                    None if controller_risk_budget is None
                    else float(controller_risk_budget.total)),
                "closed_loop_allocated_miscoverage": (
                    None if risk_decision is None
                    else float(risk_decision.allocated_miscoverage)),
                "closed_loop_risk_accepted_sources": (
                    [] if risk_decision is None
                    else list(risk_decision.accepted_sources)),
                "closed_loop_accept_source": (
                    "both"
                    if feedback_accept and physics_accept
                    else (
                        "feedback" if feedback_accept
                        else ("physics" if physics_accept else "none")
                    )),
                "closed_loop_applied_worst": float(np.min(applied_pd)),
                "closed_loop_applied_pd": np.asarray(
                    applied_pd, dtype=np.float64).tolist(),
                "closed_loop_state_version_after": int(
                    deployed_state_version.get(seed, 0)),
                "closed_loop_candidate_lower": (
                    None if decision is None
                    else decision.candidate_lower.tolist()),
                "closed_loop_noop_lower": (
                    None if decision is None
                    else decision.noop_lower.tolist()),
                "closed_loop_noop_upper": (
                    None if decision is None
                    else decision.noop_upper.tolist()),
                "closed_loop_worst_delta_lower": (
                    None if decision is None
                    else float(decision.worst_delta_lower)),
                "closed_loop_physics_target_safe": (
                    None if physics_decision is None
                    else physics_decision.target_safe.tolist()),
                "closed_loop_physics_worst_delta_lower": (
                    None if physics_decision is None
                    else float(physics_decision.worst_delta_lower)),
            })

        previous_coefficient[seed] = coefficient.copy()
        previous_uav_positions[seed] = np.asarray(
            data["uav_positions"][row_id], dtype=np.float64).copy()
        previous_target_states[seed] = np.asarray(
            data["target_states"][row_id], dtype=np.float64).copy()
        feedback_selected = (
            applied_selected if bool(persistent_closed_loop) else selected)
        feedback_sensing_power = (
            applied_sensing_power
            if bool(persistent_closed_loop) else sensing_power)
        previous_observed_mask[seed] = (
            feedback_selected
            & (feedback_sensing_power[:, None, :] > 1.0e-12)
        )
        if current_owner_local_state is not None:
            previous_owner_local_state[seed] = current_owner_local_state
        previous_event_frame[seed] = int(frames[row_id])
        rows.append(report)

    causal_rows = [row for row in rows if bool(row["causal_available"])]
    feasible_rows = [
        row for row in causal_rows if bool(row.get("transport_feasible", False))
    ]
    triggered = [row for row in feasible_rows if bool(row["structure_triggered"])]
    accepted = [row for row in triggered if bool(row["structure_accepted"])]
    edge_episode_scores: dict[int, float] = {}
    edge_upper_episode_scores: dict[int, float] = {}
    edge_joint_episode_scores: dict[int, float] = {}
    owner_dd_episode_scores: dict[int, float] = {}
    for row in feasible_rows:
        event_seed = int(row["seed"])
        edge_episode_scores[event_seed] = max(
            edge_episode_scores.get(event_seed, 0.0),
            float(row["all_edge_log_overprediction_score"]),
        )
        edge_upper_episode_scores[event_seed] = max(
            edge_upper_episode_scores.get(event_seed, 0.0),
            float(row["all_edge_log_underprediction_score"]),
        )
        edge_joint_episode_scores[event_seed] = max(
            edge_joint_episode_scores.get(event_seed, 0.0),
            float(row["all_edge_log_overprediction_score"]),
            float(row["all_edge_log_underprediction_score"]),
        )
        owner_dd_episode_scores[event_seed] = max(
            owner_dd_episode_scores.get(event_seed, 0.0),
            float(row["owner_local_dd_false_support_score"]),
        )
    edge_margin, edge_coverage, edge_rank = split_conformal_upper(
        edge_episode_scores.values(), alpha=0.05)
    edge_upper_margin, edge_upper_coverage, edge_upper_rank = (
        split_conformal_upper(
            edge_upper_episode_scores.values(), alpha=0.05)
    )
    edge_joint_margin, edge_joint_coverage, edge_joint_rank = (
        split_conformal_upper(
            edge_joint_episode_scores.values(), alpha=0.05)
    )
    if owner_dd_episode_scores:
        owner_dd_margin, owner_dd_coverage, owner_dd_rank = (
            split_conformal_upper(
                owner_dd_episode_scores.values(), alpha=0.05)
        )
    else:
        owner_dd_margin, owner_dd_coverage, owner_dd_rank = 0.0, 0.0, 0
    sequence_rows = [
        row for row in triggered
        if bool(row.get("structure_sequence_transport_enabled", False))
    ]
    horizon_rows = [
        row for row in feasible_rows
        if bool(row.get("horizon_diagnostic_evaluated", False))
    ]
    horizon_accepted_rows = [
        row for row in horizon_rows
        if bool(row.get("horizon_diagnostic_accept", False))
    ]
    horizon_outcome_rows = [
        row for row in horizon_rows
        if bool(row.get("horizon_hold_outcome_valid", False))
    ]
    horizon_accepted_outcome_rows = [
        row for row in horizon_outcome_rows
        if bool(row.get("horizon_diagnostic_accept", False))
    ]
    horizon_future_rows = [
        row for row in horizon_rows
        if bool(row.get("horizon_future_outcome_valid", False))
    ]
    horizon_accepted_future_rows = [
        row for row in horizon_future_rows
        if bool(row.get("horizon_diagnostic_accept", False))
    ]
    horizon_future_episode_scores: dict[int, float] = {}
    horizon_future_infinite_score_seeds: set[int] = set()
    for row in horizon_future_rows:
        event_seed = int(row["seed"])
        score = row.get("horizon_future_joint_log_score")
        support_failure_count = int(row.get(
            "horizon_future_lower_zero_support_failure_count", 0
        )) + int(row.get(
            "horizon_future_upper_zero_support_failure_count", 0
        ))
        if support_failure_count > 0 or score is None:
            horizon_future_infinite_score_seeds.add(event_seed)
            continue
        horizon_future_episode_scores[event_seed] = max(
            horizon_future_episode_scores.get(event_seed, 0.0),
            float(score),
        )
    for event_seed in horizon_future_infinite_score_seeds:
        horizon_future_episode_scores.pop(event_seed, None)
    # The exchangeability unit is the complete episode, not an event selected
    # by the controller.  An episode with an empty eligible error set has the
    # mathematically correct maximum score zero and must remain in calibration
    # (otherwise selection on trigger frequency would bias the sample).
    for event_seed in seed_order:
        if int(event_seed) not in horizon_future_infinite_score_seeds:
            horizon_future_episode_scores.setdefault(int(event_seed), 0.0)
    horizon_ranked_rows = [
        row for row in feasible_rows
        if row.get("horizon_ranking_proxy_lower") is not None
    ]
    horizon_complete_compute_rows = [
        row for row in feasible_rows
        if float(row.get("horizon_complete_branch_compute_s", 0.0)) > 0.0
    ]
    horizon_top1_comparison_rows = [
        row for row in horizon_ranked_rows
        if row.get("horizon_ranked_top1_matches_myopic") is not None
    ]
    horizon_comm_bound_rows = [
        row for row in horizon_ranked_rows
        if row.get("horizon_ranking_comm_bound_valid") is not None
    ]
    horizon_owner_oracle_rows = [
        row for row in feasible_rows
        if int(row.get(
            "horizon_owner_proposal_oracle_candidate_count", 0)) > 0
    ]
    horizon_raw_oracle_rows = [
        row for row in feasible_rows
        if int(row.get(
            "horizon_raw_candidate_oracle_candidate_count", 0)) > 0
    ]
    horizon_rendezvous_rows = [
        row for row in feasible_rows
        if bool(row.get("horizon_digest_rendezvous_attempted", False))
    ]
    horizon_rendezvous_eligible_rows = [
        row for row in horizon_rendezvous_rows
        if bool(row.get(
            "horizon_digest_rendezvous_additional_eligible", False))
    ]
    route_names = (
        "no_op", "fixed_structure_power", "joint_structure_power",
        "slow_geometry", "hold_unverified",
    )
    summary = {
        "causal_event_count": len(causal_rows),
        "transport_feasible_rate": float(np.mean([
            bool(row.get("transport_feasible", False)) for row in causal_rows
        ])),
        "causal_route_counts": {
            name: int(sum(row.get("causal_route") == name for row in causal_rows))
            for name in route_names
        },
        "true_diagnostic_route_counts": {
            name: int(sum(
                row.get("true_diagnostic_route") == name for row in causal_rows))
            for name in route_names
        },
        "route_match_rate": float(np.mean([
            bool(row.get("route_matches_true_diagnostic", False))
            for row in feasible_rows
        ])),
        "structure_trigger_event_count": len(triggered),
        "structure_accept_rate_given_trigger": float(np.mean([
            bool(row["structure_accepted"]) for row in triggered
        ])) if triggered else 0.0,
        "structure_sequence_transport_feasible_rate": float(np.mean([
            bool(row["structure_sequence_transport_feasible"])
            for row in sequence_rows
        ])) if sequence_rows else 0.0,
        "structure_sequence_mean_total_bits": float(np.mean([
            int(row["structure_sequence_total_bits"])
            for row in sequence_rows
        ])) if sequence_rows else 0.0,
        "structure_sequence_mean_owner_proposal_count": float(np.mean([
            int(row["structure_sequence_owner_proposal_count"])
            for row in sequence_rows
        ])) if sequence_rows else 0.0,
        "structure_sequence_mean_owner_proposal_bits": float(np.mean([
            int(row["structure_sequence_owner_proposal_bits"])
            for row in sequence_rows
        ])) if sequence_rows else 0.0,
        "target_invariant_token_transport_enabled": bool(
            target_invariant_token_transport),
        "structure_sequence_mean_target_invariant_packet_count": float(
            np.mean([
                int(row[
                    "structure_sequence_target_invariant_packet_count"])
                for row in sequence_rows
            ])
        ) if sequence_rows else 0.0,
        "structure_sequence_mean_target_invariant_bits": float(np.mean([
            int(row["structure_sequence_target_invariant_bits"])
            for row in sequence_rows
        ])) if sequence_rows else 0.0,
        "structure_sequence_mean_target_invariant_latency_s": float(
            np.mean([
                float(row[
                    "structure_sequence_target_invariant_latency_s"])
                for row in sequence_rows
            ])
        ) if sequence_rows else 0.0,
        "structure_sequence_mean_total_protocol_latency_s": float(np.mean([
            float(row["structure_sequence_total_protocol_latency_s"])
            for row in sequence_rows
        ])) if sequence_rows else 0.0,
        "structure_sequence_max_total_protocol_latency_s": float(max([
            float(row["structure_sequence_total_protocol_latency_s"])
            for row in sequence_rows
        ], default=0.0)),
        "structure_sequence_max_isac_power_balance_error_w": float(max([
            float(row["structure_sequence_max_isac_power_balance_error_w"])
            for row in sequence_rows
        ], default=0.0)),
        "horizon_diagnostic_enabled": bool(horizon_steps > 0),
        "horizon_diagnostic_steps": int(horizon_steps),
        "horizon_diagnostic_evaluated_event_count": len(horizon_rows),
        "horizon_diagnostic_accept_rate": float(np.mean([
            bool(row["horizon_diagnostic_accept"])
            for row in horizon_rows
        ])) if horizon_rows else 0.0,
        "horizon_diagnostic_mean_net_discounted_gain_lower": float(np.mean([
            float(row["horizon_diagnostic_net_discounted_gain_lower"])
            for row in horizon_rows
        ])) if horizon_rows else 0.0,
        "horizon_diagnostic_mean_end_to_end_latency_s": float(np.mean([
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            for row in horizon_rows
        ])) if horizon_rows else 0.0,
        "horizon_diagnostic_max_end_to_end_latency_s": float(max([
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            for row in horizon_rows
        ], default=0.0)),
        "horizon_diagnostic_end_to_end_deadline_rate": float(np.mean([
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            <= float(cfg.scenario.dt) + 1.0e-12
            for row in horizon_rows
        ])) if horizon_rows else 0.0,
        "horizon_accepted_end_to_end_event_count": len(
            horizon_accepted_rows),
        "horizon_accepted_mean_end_to_end_latency_s": float(np.mean([
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            for row in horizon_accepted_rows
        ])) if horizon_accepted_rows else 0.0,
        "horizon_accepted_max_end_to_end_latency_s": float(max([
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            for row in horizon_accepted_rows
        ], default=0.0)),
        "horizon_accepted_end_to_end_deadline_rate": float(np.mean([
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            <= float(cfg.scenario.dt) + 1.0e-12
            for row in horizon_accepted_rows
        ])) if horizon_accepted_rows else 0.0,
        "horizon_end_to_end_deadline_rejection_count": int(sum(
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            > float(cfg.scenario.dt) + 1.0e-12
            and not bool(row.get("horizon_diagnostic_accept", False))
            for row in horizon_rows
        )),
        "horizon_ranked_event_count": len(horizon_ranked_rows),
        "horizon_ranked_mean_candidate_count": float(np.mean([
            int(row["horizon_ranked_candidate_count"])
            for row in horizon_ranked_rows
        ])) if horizon_ranked_rows else 0.0,
        "horizon_ranked_mean_owner_proposal_count": float(np.mean([
            int(row["horizon_ranked_owner_proposal_count"])
            for row in horizon_ranked_rows
        ])) if horizon_ranked_rows else 0.0,
        "horizon_ranking_mean_proxy_lower": float(np.mean([
            float(row["horizon_ranking_proxy_lower"])
            for row in horizon_ranked_rows
        ])) if horizon_ranked_rows else 0.0,
        "horizon_complete_branch_compute_event_count": len(
            horizon_complete_compute_rows),
        "controller_common_preprocessing_mean_s": float(np.mean([
            float(row["controller_common_preprocessing_s"])
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "controller_common_preprocessing_max_s": float(max([
            float(row["controller_common_preprocessing_s"])
            for row in horizon_complete_compute_rows
        ], default=0.0)),
        "controller_common_instrumented_wall_mean_s": float(np.mean([
            float(row["controller_common_instrumented_wall_s"])
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "controller_privileged_audit_mean_s": float(np.mean([
            float(row["controller_privileged_audit_s"])
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "horizon_complete_branch_mean_compute_s": float(np.mean([
            float(row["horizon_complete_branch_compute_s"])
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "horizon_complete_branch_max_compute_s": float(max([
            float(row["horizon_complete_branch_compute_s"])
            for row in horizon_complete_compute_rows
        ], default=0.0)),
        "horizon_complete_branch_compute_deadline_rate": float(np.mean([
            float(row["horizon_complete_branch_compute_s"])
            <= float(cfg.scenario.dt)
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "horizon_complete_branch_mean_process_cpu_s": float(np.mean([
            float(row["horizon_complete_branch_process_cpu_s"])
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "horizon_propagation_mean_compute_s": float(np.mean([
            float(row["horizon_propagation_compute_s"])
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "horizon_envelope_mean_compute_s": float(np.mean([
            float(row["horizon_envelope_compute_s"])
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "horizon_candidate_set_mean_compute_s": float(np.mean([
            float(row["horizon_candidate_set_compute_s"])
            for row in horizon_complete_compute_rows
        ])) if horizon_complete_compute_rows else 0.0,
        "horizon_ranking_mean_compute_s": float(np.mean([
            float(row["horizon_ranking_compute_s"])
            for row in horizon_ranked_rows
        ])) if horizon_ranked_rows else 0.0,
        "horizon_ranking_max_compute_s": float(max([
            float(row["horizon_ranking_compute_s"])
            for row in horizon_ranked_rows
        ], default=0.0)),
        "horizon_ranking_mean_process_cpu_s": float(np.mean([
            float(row["horizon_ranking_process_cpu_s"])
            for row in horizon_ranked_rows
        ])) if horizon_ranked_rows else 0.0,
        "horizon_owner_concurrent_measured_event_count": int(sum(
            bool(row["horizon_owner_concurrent_measured"])
            for row in horizon_ranked_rows
        )),
        "horizon_ranking_compute_deadline_rate": float(np.mean([
            float(row["horizon_ranking_compute_s"])
            <= float(cfg.scenario.dt)
            for row in horizon_ranked_rows
        ])) if horizon_ranked_rows else 0.0,
        "horizon_ranking_mean_parallel_compute_s": float(np.mean([
            float(row["horizon_ranking_parallel_compute_s"])
            for row in horizon_ranked_rows
        ])) if horizon_ranked_rows else 0.0,
        "horizon_ranking_max_parallel_compute_s": float(max([
            float(row["horizon_ranking_parallel_compute_s"])
            for row in horizon_ranked_rows
        ], default=0.0)),
        "horizon_ranking_parallel_compute_deadline_rate": float(np.mean([
            float(row["horizon_ranking_parallel_compute_s"])
            <= float(cfg.scenario.dt)
            for row in horizon_ranked_rows
        ])) if horizon_ranked_rows else 0.0,
        "shadow_router_event_count": int(sum(
            bool(row.get("shadow_router_enabled", False))
            for row in feasible_rows
        )),
        "shadow_router_consensus_candidate_count": int(sum(
            row.get("shadow_router_action") == "consensus_candidate"
            for row in feasible_rows
        )),
        "shadow_router_candidate_identity_match_count": int(sum(
            row.get("shadow_router_candidate_identity_match") is True
            for row in feasible_rows
        )),
        "shadow_router_structure_identity_match_count": int(sum(
            row.get("shadow_router_structure_identity_match") is True
            for row in feasible_rows
        )),
        "shadow_router_structure_reconciliation_eligible_count": int(sum(
            bool(row.get("structure_accepted", False))
            and bool(row.get("horizon_diagnostic_accept", False))
            and row.get("shadow_router_structure_identity_match") is True
            for row in feasible_rows
        )),
        "shadow_router_resource_complete_event_count": int(sum(
            bool(row.get(
                "shadow_router_resource_accounting_complete", False))
            for row in feasible_rows
        )),
        "shadow_router_reason_counts": {
            reason: int(sum(
                row.get("shadow_router_reason") == reason
                for row in feasible_rows
            ))
            for reason in sorted({
                str(row.get("shadow_router_reason"))
                for row in feasible_rows
                if row.get("shadow_router_reason") is not None
            })
        },
        "horizon_top1_comparison_event_count": len(
            horizon_top1_comparison_rows),
        "horizon_top1_disagreement_count": int(sum(
            not bool(row["horizon_ranked_top1_matches_myopic"])
            for row in horizon_top1_comparison_rows
        )),
        "horizon_top1_disagreement_rate": float(np.mean([
            not bool(row["horizon_ranked_top1_matches_myopic"])
            for row in horizon_top1_comparison_rows
        ])) if horizon_top1_comparison_rows else 0.0,
        "horizon_ranking_comm_bound_failure_count": int(sum(
            not bool(row["horizon_ranking_comm_bound_valid"])
            for row in horizon_comm_bound_rows
        )),
        "horizon_ranking_comm_bound_valid_rate": float(np.mean([
            bool(row["horizon_ranking_comm_bound_valid"])
            for row in horizon_comm_bound_rows
        ])) if horizon_comm_bound_rows else 0.0,
        "horizon_candidate_max_comm_power_w": float(max([
            float(row["horizon_candidate_max_comm_power_w"])
            for row in horizon_comm_bound_rows
        ], default=0.0)),
        "horizon_owner_proposal_oracle_event_count": len(
            horizon_owner_oracle_rows),
        "horizon_owner_proposal_oracle_any_accept_rate": float(np.mean([
            bool(row["horizon_owner_proposal_oracle_any_accept"])
            for row in horizon_owner_oracle_rows
        ])) if horizon_owner_oracle_rows else 0.0,
        "horizon_owner_proposal_oracle_top1_regret_count": int(sum(
            bool(row["horizon_owner_proposal_oracle_top1_regret"])
            for row in horizon_owner_oracle_rows
        )),
        "horizon_owner_proposal_oracle_structure_match_event_count": int(sum(
            int(row.get(
                "horizon_owner_proposal_oracle_structure_match_count", 0))
            > 0
            for row in horizon_owner_oracle_rows
        )),
        "horizon_owner_proposal_oracle_structure_match_accept_event_count": int(sum(
            bool(row.get(
                "horizon_owner_proposal_oracle_structure_match_any_accept",
                False))
            for row in horizon_owner_oracle_rows
        )),
        "horizon_owner_proposal_oracle_structure_match_resource_certified": False,
        "horizon_raw_candidate_oracle_event_count": len(
            horizon_raw_oracle_rows),
        "horizon_raw_candidate_oracle_any_accept_rate": float(np.mean([
            bool(row["horizon_raw_candidate_oracle_any_accept"])
            for row in horizon_raw_oracle_rows
        ])) if horizon_raw_oracle_rows else 0.0,
        "horizon_raw_candidate_oracle_owner_compression_regret_count": int(
            sum(bool(row[
                "horizon_raw_candidate_oracle_owner_compression_regret"])
                for row in horizon_raw_oracle_rows)
        ),
        "horizon_raw_candidate_oracle_structure_match_event_count": int(sum(
            int(row.get(
                "horizon_raw_candidate_oracle_structure_match_count", 0))
            > 0
            for row in horizon_raw_oracle_rows
        )),
        "horizon_raw_candidate_oracle_structure_match_accept_event_count": int(sum(
            bool(row.get(
                "horizon_raw_candidate_oracle_structure_match_any_accept",
                False))
            for row in horizon_raw_oracle_rows
        )),
        "horizon_raw_candidate_oracle_structure_match_resource_certified": False,
        "horizon_digest_rendezvous_enabled": bool(
            horizon_digest_rendezvous),
        "horizon_digest_rendezvous_attempt_event_count": len(
            horizon_rendezvous_rows),
        "horizon_digest_rendezvous_attempt_episode_count": len({
            int(row["seed"]) for row in horizon_rendezvous_rows
        }),
        "horizon_digest_rendezvous_transport_feasible_rate": float(np.mean([
            bool(row["horizon_digest_rendezvous_transport_feasible"])
            for row in horizon_rendezvous_rows
        ])) if horizon_rendezvous_rows else 0.0,
        "horizon_digest_rendezvous_deferred_top1_transport": bool(
            horizon_digest_rendezvous_defer_top1_transport),
        "horizon_digest_rendezvous_prefix_feasible_rate": float(np.mean([
            bool(row["horizon_digest_rendezvous_prefix_feasible"])
            for row in horizon_rendezvous_rows
        ])) if (
            horizon_rendezvous_rows
            and bool(horizon_digest_rendezvous_defer_top1_transport)
        ) else 0.0,
        "horizon_digest_rendezvous_suffix_feasible_rate": float(np.mean([
            bool(row["horizon_digest_rendezvous_suffix_feasible"])
            for row in horizon_rendezvous_rows
        ])) if horizon_rendezvous_rows else 0.0,
        "horizon_digest_rendezvous_structure_commit_reuse_enabled": bool(
            horizon_digest_rendezvous_reuse_structure_commit),
        "horizon_digest_rendezvous_structure_commit_reused_count": int(sum(
            bool(row.get(
                "horizon_digest_rendezvous_structure_commit_reused", False))
            for row in horizon_rendezvous_rows
        )),
        "horizon_digest_rendezvous_exact_accept_count": int(sum(
            bool(row["horizon_digest_rendezvous_exact_accept"])
            for row in horizon_rendezvous_rows
        )),
        "horizon_network_repetition_count": int(network_repetitions),
        "horizon_network_tolerated_erasures_per_packet": int(
            network_repetitions - 1),
        "horizon_network_excess_queue_bound_s": float(
            network_queue_bound),
        "horizon_network_repetition_feasible_count": int(sum(
            bool(row.get("horizon_network_repetition_feasible", False))
            for row in horizon_rendezvous_rows
        )),
        "horizon_network_repetition_feasible_rate": float(np.mean([
            bool(row.get("horizon_network_repetition_feasible", False))
            for row in horizon_rendezvous_rows
        ])) if horizon_rendezvous_rows else 0.0,
        "horizon_digest_rendezvous_deadline_pass_count": int(sum(
            bool(row["horizon_digest_rendezvous_deadline_pass"])
            for row in horizon_rendezvous_rows
        )),
        "horizon_digest_rendezvous_additional_eligible_count": len(
            horizon_rendezvous_eligible_rows),
        "horizon_digest_rendezvous_additional_eligible_episode_count": len({
            int(row["seed"])
            for row in horizon_rendezvous_eligible_rows
        }),
        "horizon_digest_rendezvous_mean_total_latency_s": float(np.mean([
            float(row["horizon_digest_rendezvous_total_latency_s"])
            for row in horizon_rendezvous_rows
        ])) if horizon_rendezvous_rows else 0.0,
        "horizon_digest_rendezvous_mean_nominal_total_latency_s": float(
            np.mean([
                float(row[
                    "horizon_digest_rendezvous_nominal_total_latency_s"])
                for row in horizon_rendezvous_rows
            ])) if horizon_rendezvous_rows else 0.0,
        "horizon_digest_rendezvous_max_nominal_total_latency_s": float(max([
            float(row[
                "horizon_digest_rendezvous_nominal_total_latency_s"])
            for row in horizon_rendezvous_rows
        ], default=0.0)),
        "horizon_digest_rendezvous_min_total_latency_s": float(min([
            float(row["horizon_digest_rendezvous_total_latency_s"])
            for row in horizon_rendezvous_rows
        ], default=0.0)),
        "horizon_digest_rendezvous_max_total_latency_s": float(max([
            float(row["horizon_digest_rendezvous_total_latency_s"])
            for row in horizon_rendezvous_rows
        ], default=0.0)),
        "horizon_digest_rendezvous_total_bits": int(sum(
            int(row["horizon_digest_rendezvous_prefix_bits"])
            + int(row["horizon_digest_rendezvous_bits"])
            + int(row["horizon_digest_rendezvous_suffix_bits"])
            for row in horizon_rendezvous_rows
        )),
        "horizon_network_nominal_protocol_bits": int(sum(
            int(row.get("horizon_network_nominal_protocol_bits", 0))
            for row in horizon_rendezvous_rows
        )),
        "horizon_network_worst_protocol_bits": int(sum(
            int(row.get("horizon_network_worst_protocol_bits", 0))
            for row in horizon_rendezvous_rows
        )),
        "horizon_network_nominal_rf_energy_j": float(sum(
            float(row.get("horizon_network_nominal_rf_energy_j", 0.0))
            for row in horizon_rendezvous_rows
        )),
        "horizon_network_worst_rf_energy_j": float(sum(
            float(row.get("horizon_network_worst_rf_energy_j", 0.0))
            for row in horizon_rendezvous_rows
        )),
        "horizon_digest_rendezvous_max_power_balance_error_w": float(max([
            float(row["horizon_digest_rendezvous_power_balance_error_w"])
            for row in horizon_rendezvous_rows
            if row.get(
                "horizon_digest_rendezvous_power_balance_error_w") is not None
        ], default=0.0)),
        "horizon_digest_rendezvous_future_failure_count": int(sum(
            bool(row.get(
                "horizon_digest_rendezvous_future_candidate_bound_failure",
                False))
            or bool(row.get(
                "horizon_digest_rendezvous_future_noop_bound_failure", False))
            or row.get(
                "horizon_digest_rendezvous_future_target_no_harm") is False
            for row in horizon_rendezvous_eligible_rows
        )),
        "horizon_digest_rendezvous_resource_accounting_complete": bool(
            shadow_resource_accounting_complete),
        "horizon_digest_rendezvous_commit_authority": False,
        "horizon_hold_outcome_event_count": len(horizon_outcome_rows),
        "horizon_hold_candidate_bound_failure_rate": float(np.mean([
            bool(row["horizon_hold_candidate_bound_failure"])
            for row in horizon_outcome_rows
        ])) if horizon_outcome_rows else 0.0,
        "horizon_hold_noop_bound_failure_rate": float(np.mean([
            bool(row["horizon_hold_noop_bound_failure"])
            for row in horizon_outcome_rows
        ])) if horizon_outcome_rows else 0.0,
        "horizon_accepted_hold_target_no_harm_rate": float(np.mean([
            bool(row["horizon_hold_realized_target_no_harm"])
            for row in horizon_accepted_outcome_rows
        ])) if horizon_accepted_outcome_rows else 0.0,
        "horizon_accepted_hold_target_no_harm_violation_count": int(sum(
            not bool(row["horizon_hold_realized_target_no_harm"])
            for row in horizon_accepted_outcome_rows
        )),
        "horizon_future_outcome_event_count": len(horizon_future_rows),
        "horizon_future_outcome_episode_count": len({
            int(row["seed"]) for row in horizon_future_rows
        }),
        "horizon_future_coefficient_audit_count": int(sum(
            int(row["horizon_future_coefficient_audit_count"])
            for row in horizon_future_rows
        )),
        "horizon_future_lower_zero_support_failure_count": int(sum(
            int(row["horizon_future_lower_zero_support_failure_count"])
            for row in horizon_future_rows
        )),
        "horizon_future_upper_zero_support_failure_count": int(sum(
            int(row["horizon_future_upper_zero_support_failure_count"])
            for row in horizon_future_rows
        )),
        "horizon_future_infinite_episode_score_count": len(
            horizon_future_infinite_score_seeds),
        "horizon_future_max_finite_episode_log_score": float(max(
            horizon_future_episode_scores.values(), default=0.0)),
        "horizon_future_candidate_bound_failure_rate": float(np.mean([
            bool(row["horizon_future_candidate_bound_failure"])
            for row in horizon_future_rows
        ])) if horizon_future_rows else 0.0,
        "horizon_future_noop_bound_failure_rate": float(np.mean([
            bool(row["horizon_future_noop_bound_failure"])
            for row in horizon_future_rows
        ])) if horizon_future_rows else 0.0,
        "horizon_accepted_future_target_no_harm_rate": float(np.mean([
            bool(row["horizon_future_realized_target_no_harm"])
            for row in horizon_accepted_future_rows
        ])) if horizon_accepted_future_rows else 0.0,
        "horizon_accepted_future_target_no_harm_violation_count": int(sum(
            not bool(row["horizon_future_realized_target_no_harm"])
            for row in horizon_accepted_future_rows
        )),
        "horizon_diagnostic_reason_counts": {
            reason: int(sum(
                row.get("horizon_diagnostic_reason") == reason
                for row in feasible_rows
            ))
            for reason in (
                "evaluated", "not_triggered",
                "incomplete_target_invariant",
                "incomplete_noop_provenance",
                "no_verified_atomic_candidate",
                "ranking_comm_bound_failure",
                "end_to_end_deadline_failure",
            )
        },
        "triggered_power_only_event_mean_worst": float(np.mean([
            float(row["power_only_worst"]) for row in triggered
        ])) if triggered else 0.0,
        "triggered_structure_event_mean_worst": float(np.mean([
            float(row["structure_worst"]) for row in triggered
        ])) if triggered else 0.0,
        "accepted_mean_incremental_gain": float(np.mean([
            float(row["structure_incremental_gain"]) for row in accepted
        ])) if accepted else 0.0,
        "accepted_harm_rate_vs_power_only": float(np.mean([
            bool(row["structure_harmed_vs_power_only"]) for row in accepted
        ])) if accepted else 0.0,
        "triggered_structure_qos_rate": float(np.mean([
            float(row["structure_worst"]) >= float(qos_floor)
            for row in triggered
        ])) if triggered else 0.0,
        "accepted_mean_candidate_count": float(np.mean([
            int(row["structure_candidate_count"]) for row in accepted
        ])) if accepted else 0.0,
        "accepted_mean_exact_verification_count": float(np.mean([
            int(row["structure_exact_verification_count"]) for row in accepted
        ])) if accepted else 0.0,
        "accepted_mean_pd_abs_error": float(np.mean([
            float(row["structure_pd_abs_error"]) for row in accepted
        ])) if accepted else 0.0,
        "accepted_p95_max_overprediction": float(np.quantile([
            float(row["structure_pd_max_overprediction"]) for row in accepted
        ], 0.95)) if accepted else 0.0,
        "raw_routed_event_mean_worst": float(np.mean([
            float(row["causal_routed_worst"])
            for row in feasible_rows
        ])),
        "raw_routed_event_qos_rate": float(np.mean([
            float(row["causal_routed_worst"]) >= float(qos_floor)
            for row in feasible_rows
        ])),
        "raw_routed_episode_mean_worst": _episode_mean(
            feasible_rows, "causal_routed_worst"),
        "raw_routed_harm_rate": float(np.mean([
            bool(row["causal_routed_harmed"]) for row in feasible_rows
        ])),
        "raw_routed_action_rate": float(np.mean([
            bool(row["causal_routed_available"]) for row in feasible_rows
        ])),
        "persistent_closed_loop_enabled": bool(persistent_closed_loop),
        "persistent_rf_recourse_from_trace_enabled": bool(
            persistent_rf_recourse_from_trace),
        "persistent_closed_loop_event_mean_worst": float(np.mean([
            float(row.get(
                "closed_loop_applied_worst", row["deployed_worst"]))
            for row in causal_rows
        ])) if persistent_closed_loop else 0.0,
        "persistent_closed_loop_event_qos_rate": float(np.mean([
            float(row.get(
                "closed_loop_applied_worst", row["deployed_worst"]))
            >= float(qos_floor)
            for row in causal_rows
        ])) if persistent_closed_loop else 0.0,
        "persistent_closed_loop_accept_rate": float(np.mean([
            bool(row.get("closed_loop_controller_accept", False))
            for row in causal_rows
        ])) if persistent_closed_loop else 0.0,
        "persistent_closed_loop_feedback_certificate_rate": float(np.mean([
            bool(row.get(
                "closed_loop_feedback_certificate_accept", False))
            for row in causal_rows
        ])) if persistent_closed_loop else 0.0,
        "persistent_closed_loop_physics_certificate_rate": float(np.mean([
            bool(row.get(
                "closed_loop_physics_certificate_accept", False))
            for row in causal_rows
        ])) if persistent_closed_loop else 0.0,
        "persistent_closed_loop_joint_risk_budget_enabled": bool(
            controller_risk_budget is not None),
        "persistent_closed_loop_total_miscoverage": (
            None if controller_risk_budget is None
            else float(controller_risk_budget.total)),
        "persistent_closed_loop_allocated_miscoverage": (
            None if controller_risk_budget is None
            else float(controller_risk_budget.allocated)),
        "persistent_closed_loop_trace_structure_match_rate": float(np.mean([
            bool(row.get("trace_structure_matches_deployed", True))
            for row in causal_rows
        ])) if persistent_closed_loop else 1.0,
        "persistent_closed_loop_max_state_version": int(max([
            int(row.get("closed_loop_state_version_after", 0))
            for row in causal_rows
        ], default=0)) if persistent_closed_loop else 0,
        "target_reserve_enabled": bool(normalized_margin is not None),
        "target_reserve_possible_rate": float(np.mean([
            bool(row.get("target_reserve_possible", True))
            for row in feasible_rows
        ])) if normalized_margin is not None else 1.0,
        "fixed_target_reserve_feasible_rate": float(np.mean([
            bool(row.get("fixed_target_reserve_feasible", True))
            for row in feasible_rows
        ])) if normalized_margin is not None else 1.0,
        "mean_physical_support_edge_count": float(np.mean([
            int(row["physical_support_edge_count"])
            for row in feasible_rows
        ])),
        "mean_provenance_observed_support_edge_count": float(np.mean([
            int(row["provenance_observed_support_edge_count"])
            for row in feasible_rows
        ])),
        "mean_excluded_unobserved_support_fraction": float(np.mean([
            int(row["excluded_unobserved_support_edge_count"])
            / max(int(row["physical_support_edge_count"]), 1)
            for row in feasible_rows
        ])),
        "unknown_edge_log_conformal_margin": float(edge_margin),
        "unknown_edge_multiplicative_lower_factor": float(np.exp(-edge_margin)),
        "unknown_edge_conformal_coverage_floor": float(edge_coverage),
        "unknown_edge_conformal_rank_one_based": int(edge_rank),
        "unknown_edge_log_upper_conformal_margin": float(edge_upper_margin),
        "unknown_edge_multiplicative_upper_factor": float(np.exp(
            edge_upper_margin)),
        "unknown_edge_upper_conformal_coverage_floor": float(
            edge_upper_coverage),
        "unknown_edge_upper_conformal_rank_one_based": int(edge_upper_rank),
        "all_edge_joint_log_conformal_margin": float(edge_joint_margin),
        "all_edge_joint_multiplicative_lower_factor": float(np.exp(
            -edge_joint_margin)),
        "all_edge_joint_multiplicative_upper_factor": float(np.exp(
            edge_joint_margin)),
        "all_edge_joint_conformal_coverage_floor": float(
            edge_joint_coverage),
        "all_edge_joint_conformal_rank_one_based": int(edge_joint_rank),
        "max_unpredicted_positive_edge_count": int(max([
            int(row["unpredicted_positive_edge_count"])
            for row in feasible_rows
        ], default=0)),
        "unpredicted_positive_edge_event_rate": float(np.mean([
            int(row["unpredicted_positive_edge_count"]) > 0
            for row in feasible_rows
        ])) if feasible_rows else 0.0,
        "target_invariant_cache_enabled": bool(cache_max_age > 0),
        "target_invariant_cache_max_age_frames": int(cache_max_age),
        "target_invariant_cache_mean_available_target_fraction": float(
            np.mean([
                int(row.get(
                    "target_invariant_cache_available_target_count", 0))
                / max(Q, 1)
                for row in feasible_rows
            ])
        ) if cache_max_age > 0 and feasible_rows else 0.0,
        "target_invariant_cache_retention_event_rate": float(np.mean([
            int(row.get(
                "target_invariant_cache_retained_target_count", 0)) > 0
            for row in feasible_rows
        ])) if cache_max_age > 0 and feasible_rows else 0.0,
        "target_invariant_cache_expiry_event_rate": float(np.mean([
            int(row.get(
                "target_invariant_cache_expired_target_count", 0)) > 0
            for row in feasible_rows
        ])) if cache_max_age > 0 and feasible_rows else 0.0,
        "target_invariant_cache_max_used_age_frames": int(max([
            int(row.get(
                "target_invariant_cache_max_available_age_frames", 0))
            for row in feasible_rows
        ], default=0)) if cache_max_age > 0 else 0,
        "mean_cached_target_fallback_edge_count": float(np.mean([
            int(row.get(
                "full_prediction_cached_target_fallback_edge_count", 0))
            for row in feasible_rows
        ])) if cache_max_age > 0 and feasible_rows else 0.0,
        "owner_local_mean_token_candidate_edge_count": float(np.mean([
            int(row["owner_local_token_candidate_edge_count"])
            for row in feasible_rows
        ])) if observation_mode == "owner_local_conformal" else 0.0,
        "owner_local_mean_dd_admitted_edge_count": float(np.mean([
            int(row["owner_local_dd_admitted_edge_count"])
            for row in feasible_rows
        ])) if observation_mode == "owner_local_conformal" else 0.0,
        "owner_local_mean_selected_provenance_renewal_edge_count": float(
            np.mean([
                int(row["selected_provenance_renewal_edge_count"])
                for row in feasible_rows
            ])
        ) if observation_mode == "owner_local_conformal" else 0.0,
        "owner_local_mean_selected_provenance_renewal_new_edge_count": float(
            np.mean([
                int(row["selected_provenance_renewal_new_edge_count"])
                for row in feasible_rows
            ])
        ) if observation_mode == "owner_local_conformal" else 0.0,
        "owner_local_mean_selected_dd_unsafe_edge_count": float(np.mean([
            int(row["selected_dd_unsafe_edge_count"])
            for row in feasible_rows
        ])) if observation_mode == "owner_local_conformal" else 0.0,
        "owner_local_mean_selected_dd_safe_without_provenance_edge_count": (
            float(np.mean([
                int(row["selected_dd_safe_without_provenance_edge_count"])
                for row in feasible_rows
            ]))
            if observation_mode == "owner_local_conformal" else 0.0
        ),
        "owner_local_dd_admitted_physical_precision": (
            1.0 - float(np.sum([
                int(row["owner_local_dd_false_admitted_edge_count"])
                for row in feasible_rows
            ])) / max(float(np.sum([
                int(row["owner_local_safe_support_edge_count"])
                for row in feasible_rows
            ])), 1.0)
            if observation_mode == "owner_local_conformal" else 0.0
        ),
        "owner_local_episode_any_dd_support_violation_rate": float(np.mean([
            score > dd_support_margin
            for score in owner_dd_episode_scores.values()
        ])) if observation_mode == "owner_local_conformal" else 0.0,
        "owner_local_dd_empirical_conformal_margin": float(owner_dd_margin),
        "owner_local_dd_empirical_conformal_coverage_floor": float(
            owner_dd_coverage),
        "owner_local_dd_empirical_conformal_rank_one_based": int(
            owner_dd_rank),
        "owner_local_dd_frozen_additive_margin": float(dd_support_margin),
        "owner_local_dd_covariance_radius": float(dd_covariance_radius),
        "owner_local_dd_mean_geometry_tightening": float(np.mean([
            float(row["owner_local_dd_mean_geometry_tightening"])
            for row in feasible_rows
        ])) if observation_mode == "owner_local_conformal" else 0.0,
    }
    certificate_blockers = [
        "alternate-edge residuals require independent episode calibration",
        (
            "the SNR and latency engineering margins are not statistically "
            "calibrated"
        ),
        "slow-geometry actions are diagnosed but not implemented",
        (
            "the D0.19 horizon candidate remains diagnostic and does not yet "
            "own the live commit path"
            if horizon_steps > 0 else
            "the live replay still certifies one event at a time; the D0.19 "
            "horizon gate requires simultaneous H-by-Q calibrated envelopes"
        ),
    ]
    if network_calibration_metadata is None:
        certificate_blockers.append(
            "network repetition/queue support has no frozen empirical "
            "episode-level calibration epoch")
    elif not bool(network_calibration_metadata["empirical_hardware_source"]):
        certificate_blockers.append(
            "the frozen network epoch is not measured hardware U2U traffic")
    elif not bool(network_calibration_metadata[
        "independent_validation_present"
    ]):
        certificate_blockers.append(
            "the hardware U2U network epoch has no disjoint validation split")
    elif not bool(network_calibration_metadata[
        "validation_supports_declared_risk"
    ]):
        certificate_blockers.append(
            "the hardware U2U validation exact upper exceeds the declared "
            "total link risk")
    elif not bool(network_calibration_metadata["link_certificate_candidate"]):
        certificate_blockers.append(
            "the hardware U2U network epoch is not a link-certificate candidate")
    if compute_energy_calibration_metadata is None:
        certificate_blockers.append(
            "complete controller package energy has no frozen empirical "
            "episode-level calibration epoch")
    elif not bool(compute_energy_calibration_metadata[
        "empirical_hardware_source"
    ]):
        certificate_blockers.append(
            "the frozen compute-energy epoch is not a hardware meter source")
    elif not bool(compute_energy_calibration_metadata[
        "independent_validation_present"
    ]):
        certificate_blockers.append(
            "the hardware compute-energy epoch has no disjoint validation split")
    elif not bool(compute_energy_calibration_metadata[
        "compute_energy_certificate_candidate"
    ]):
        certificate_blockers.append(
            "the hardware compute-energy epoch is not an energy-certificate "
            "candidate")
    if runtime_latency_calibration_metadata is None:
        certificate_blockers.append(
            "complete controller compute latency has no frozen empirical "
            "episode-level calibration epoch")
    elif not bool(runtime_latency_calibration_metadata[
        "deployment_timing_source"
    ]):
        certificate_blockers.append(
            "the frozen runtime epoch is not deployment monotonic timing")
    elif not bool(runtime_latency_calibration_metadata[
        "independent_validation_present"
    ]):
        certificate_blockers.append(
            "the deployment runtime epoch has no disjoint validation split")
    elif not bool(runtime_latency_calibration_metadata[
        "runtime_latency_certificate_candidate"
    ]):
        certificate_blockers.append(
            "the deployment runtime epoch is not a latency-certificate candidate")
    if not resource_risk_budget_complete and any(metadata is not None for metadata in (
        network_calibration_metadata,
        runtime_latency_calibration_metadata,
        compute_energy_calibration_metadata,
    )):
        certificate_blockers.append(
            "link, runtime and energy epoch risks are not jointly allocated "
            "in the system union budget")
    if not resource_finite_sample_plan_complete and any(
        metadata is not None for metadata in (
            network_calibration_metadata,
            runtime_latency_calibration_metadata,
            compute_energy_calibration_metadata,
        )
    ):
        certificate_blockers.append(
            "resource calibration/validation episode counts cannot support "
            "every declared risk even in the zero-failure validation case")
    if certificate_mode == "risk_budgeted_union":
        certificate_blockers.append(
            "feedback and physics bounds require independent episode-level "
            "calibration at their declared risk allocations")
    elif certificate_mode == "feedback_or_physics":
        certificate_blockers.append(
            "the legacy feedback-or-physics union has no explicit joint "
            "episode-level risk allocation")
    else:
        certificate_blockers.append(
            "the feedback certificate requires independent episode-level "
            "risk calibration")
    if cache_max_age > 0:
        certificate_blockers.append(
            "the target-invariant cache is justified only for fixed-target, "
            "constant-RCS, non-Swerling episodes until a target-state "
            "transition uncertainty model is certified")
    if horizon_steps > 0:
        certificate_blockers.append(
            "the simultaneous horizon-by-target coefficient envelope still "
            "requires independent episode calibration at the declared "
            "miscoverage")
        if horizon_ranking_mode == "distributed_screened_margin":
            if (
                horizon_owner_workers > 0
                and (
                    runtime_latency_calibration_metadata is None
                    or not bool(runtime_latency_calibration_metadata[
                        "runtime_latency_certificate_candidate"])
                )
            ):
                certificate_blockers.append(
                    "owner groups use measured concurrent worker threads on "
                    "one workstation, but deployment UAV scheduling and "
                    "network jitter are not yet verified")
            elif horizon_owner_workers <= 0:
                certificate_blockers.append(
                    "owner groups are not executed on concurrent workers")
            if compute_energy_calibration_metadata is not None and bool(
                compute_energy_calibration_metadata[
                    "compute_energy_certificate_candidate"]
            ):
                certificate_blockers.append(
                    "package energy is hardware calibrated, but deployment "
                    "scheduler/thermal timing remains uncertified")
            elif shadow_cpu_power is None:
                certificate_blockers.append(
                    "ranking process CPU seconds are measured, but CPU "
                    "wattage is not hardware calibrated; compute energy "
                    "therefore remains an affine sensitivity model")
            else:
                certificate_blockers.append(
                    "the configured logical CPU wattage is not a "
                    "hardware-calibrated deployment energy measurement")
    if bool(persistent_rf_recourse_from_trace):
        certificate_blockers.append(
            "per-event RF recourse is replayed from a frozen trace rather "
            "than generated by a live deployed controller")
    if bool(certify_structure_sequence):
        if not bool(owner_proposal_transport):
            certificate_blockers.append(
                "owner proposal descriptors and ranking competition are not "
                "transported")
        elif observation_mode == "owner_local_conformal" and not bool(
            target_invariant_token_transport
        ):
            certificate_blockers.append(
                "target-invariant feedback fields and their quantized "
                "transport cost are not encoded")
        elif observation_mode == "selected_lag1":
            certificate_blockers.append(
                "owner-local cache delivery/version provenance remains to be "
                "replayed")
        elif observation_mode == "analytic_full":
            certificate_blockers.append(
                "alternate-edge scores still use the privileged analytic "
                "tensor")
    else:
        certificate_blockers.append(
            "structural dependency-commit wire cost is not included in this "
            "potential audit")
    return {
        "schema_version": 6,
        "scope": (
            "causal geometry prediction, B=2 target-block N5/N6, feasible "
            "proxy ranking and Top-M finite-round quantized verification"
            + (
                "; Top-1 full baseline/verification/commit transport sequence"
                if bool(certify_structure_sequence) else ""
            )
            + (
                "; one quantized bounded proposal per owner and step"
                if bool(owner_proposal_transport) else ""
            )
            + (
                "; owner-authoritative target invariant/version/age packets"
                if bool(target_invariant_token_transport) else ""
            )
            + (
                "; horizon owner pre-ranking mode="
                + horizon_ranking_mode
                + " in a non-committing parallel diagnostic path"
                if horizon_steps > 0 else ""
            )
            + (
                "; bounded 256-bit structure-digest rendezvous with full "
                "identity reconstruction and conservative exact suffix"
                if bool(horizon_digest_rendezvous) else ""
            )
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "seed_order": seed_order,
        "event_count": len(rows),
        "top_m": int(top_m),
        "structure_steps": int(structure_steps),
        "proxy_mode": str(proxy_mode),
        "optimism_weight": float(optimism_weight),
        "observability_mode": observation_mode,
        "frozen_unknown_edge_lower_log_margin": float(lower_log_margin),
        "frozen_unknown_edge_upper_log_margin": float(upper_log_margin),
        "frozen_feedback_normalized_margin": normalized_margin,
        "slow_geometry_fast_fallback_enabled": bool(
            slow_geometry_fast_fallback),
        "persistent_closed_loop_enabled": bool(persistent_closed_loop),
        "frozen_controller_normalized_margin": controller_margin,
        "controller_certificate_mode": certificate_mode,
        "controller_risk_budget": (
            None if controller_risk_budget is None else {
                "total": float(controller_risk_budget.total),
                "feedback": float(controller_risk_budget.feedback),
                "physics": float(controller_risk_budget.physics),
                "link": float(controller_risk_budget.link),
                "runtime": float(controller_risk_budget.runtime),
                "energy": float(controller_risk_budget.energy),
                "allocated": float(controller_risk_budget.allocated),
                "joint_coverage_floor_union_bound": float(
                    1.0 - controller_risk_budget.total),
            }),
        "persistent_rf_recourse_from_trace_enabled": bool(
            persistent_rf_recourse_from_trace),
        "persistent_closed_loop_token_policy": (
            "frozen_exogenous_trace_tokens"
            if persistent_closed_loop else None),
        "persistent_closed_loop_rf_policy": (
            "per_event_frozen_trace_local_proposal"
            if persistent_rf_recourse_from_trace
            else (
                "held_until_certified_action"
                if persistent_closed_loop else None
            )),
        "target_invariant_cache_max_age_frames": int(cache_max_age),
        "target_invariant_cache_policy": (
            "targetwise_versioned_fail_closed_expiry"
            if cache_max_age > 0 else "disabled"),
        "target_invariant_token_transport_enabled": bool(
            target_invariant_token_transport),
        "target_invariant_wire_layout": (
            {
                "invariant_bits": int(invariant_layout.invariant_bits),
                "age_bits": int(invariant_layout.age_bits),
                "version_bits": int(invariant_layout.version_bits),
                "directional_rounding": "log2_lower",
            }
            if bool(target_invariant_token_transport) else None
        ),
        "horizon_diagnostic": {
            "enabled": bool(horizon_steps > 0),
            "steps": int(horizon_steps),
            "proposal_ranking": (
                "causal_set_membership"
                if bool(horizon_set_membership_verifier)
                else horizon_ranking_mode),
            "transition_gate": (
                "common_box_paired_deflection"
                if bool(horizon_paired_deflection_gate)
                else "independent_probability_extrema"),
            "power_plan": (
                "common_robust_weights"
                if bool(horizon_common_power_plan)
                else "stepwise_recourse"),
            "ranking_comm_reserve_w": float(
                horizon_ranking_reserve),
            "weak_target_count": int(horizon_weak_targets),
            "local_shortlist_per_owner": int(
                horizon_local_shortlist),
            "ranking_rounds": int(horizon_rank_rounds),
            "concurrent_owner_workers": int(horizon_owner_workers),
            "deadline_accounting": (
                "complete_common_preprocessing_horizon_envelope_ranking_"
                "wall_plus_transport"),
            "common_preprocessing_timing": (
                "measured_instrumented_wall_minus_explicitly_bracketed_"
                "privileged_audit"),
            "oracle_diagnostics": bool(horizon_oracle_diagnostics),
            "set_membership_verifier": bool(
                horizon_set_membership_verifier),
            "reuse_primal_master_duals": bool(
                horizon_reuse_primal_master_duals),
            "network_repetition_count": int(network_repetitions),
            "network_tolerated_erasures_per_logical_packet": int(
                network_repetitions - 1),
            "network_excess_queue_bound_s": float(network_queue_bound),
            "network_reliability_interpretation": (
                "deterministic_fixed_repetition_support_not_a_packet_error_"
                "probability_or_empirical_loss_calibration"),
            "network_calibration_epoch": network_calibration_metadata,
            "miscoverage": horizon_alpha,
            "discount": float(horizon_gamma),
            "target_acceleration_std_mps2": float(horizon_sigma_a),
            "base_current_log_margin": float(max(
                lower_log_margin, upper_log_margin)),
            "frozen_transition_residual_log_margin": float(
                horizon_residual_margin),
            "combined_horizon_log_margin": float(
                max(lower_log_margin, upper_log_margin)
                + horizon_residual_margin),
            "uav_reachable_position_radius_m": (
                (
                    np.arange(horizon_steps, dtype=np.float64)
                    * float(cfg.uav.v_max)
                    * float(cfg.scenario.dt)
                ).tolist()
                if horizon_steps > 0 else []),
            "uav_reachable_velocity_radius_mps": (
                np.concatenate((
                    np.zeros(1, dtype=np.float64),
                    np.full(
                        max(horizon_steps - 1, 0),
                        float(cfg.uav.v_max),
                        dtype=np.float64,
                    ),
                )).tolist()
                if horizon_steps > 0 else []),
            "future_outcome_policy": (
                "post_decision_privileged_frozen_movement_action_tape"
                if horizon_steps > 0 else None),
            "future_outcome_commit_or_ranking_authority": False,
            "prices": {
                "per_switched_edge": float(
                    horizon_prices.per_switched_edge),
                "per_over_air_bit": float(
                    horizon_prices.per_over_air_bit),
                "per_latency_second": float(
                    horizon_prices.per_latency_second),
                "per_control_joule": float(
                    horizon_prices.per_control_joule),
            },
            "commit_authority": False,
        },
        "shadow_router": {
            "enabled": bool(horizon_steps > 0),
            "policy": "strict_full_action_consensus_else_noop",
            "structure_consensus_horizon_power_enabled": bool(
                shadow_reconcile_horizon_power),
            "action_digest": (
                "sha256(selected,role,owner,sensing_power,comm_power)"),
            "parallel_latency_rule": (
                "complete_compute_bound_once+max(one_step_network,"
                "horizon_network)"),
            "energy_rule": (
                "complete_package_bound_once+one_step_RF+horizon_RF"),
            "logical_cpu_power_w": shadow_cpu_power,
            "shared_package_energy_bound_j": float(
                shared_package_energy_bound_j),
            "compute_energy_calibration_epoch": (
                compute_energy_calibration_metadata),
            "shared_complete_compute_latency_bound_s": float(
                shared_complete_compute_latency_bound_s),
            "runtime_latency_calibration_epoch": (
                runtime_latency_calibration_metadata),
            "current_hardware_id": current_shadow_hardware_id,
            "current_runtime_id": current_shadow_runtime_id,
            "resource_risk_budget_complete": bool(
                resource_risk_budget_complete),
            "resource_finite_sample_plan_complete": bool(
                resource_finite_sample_plan_complete),
            "finite_sample_requirements": [
                {
                    "label": requirement.label,
                    "stage": requirement.stage,
                    "miscoverage": float(requirement.miscoverage),
                    "observed_episode_count": int(
                        requirement.observed_episode_count),
                    "minimum_episode_count": int(
                        requirement.minimum_episode_count),
                    "attainable_in_best_case": bool(
                        requirement.attainable_in_best_case),
                    "rank_one_based": requirement.rank_one_based,
                    "confidence": requirement.confidence,
                }
                for requirement in resource_sample_requirements
            ],
            "resource_risk_ledger": (
                None if resource_risk_ledger is None else {
                    "link_epoch_miscoverage": (
                        resource_risk_ledger.link_epoch_miscoverage),
                    "runtime_epoch_miscoverage": (
                        resource_risk_ledger.runtime_epoch_miscoverage),
                    "energy_epoch_miscoverage": (
                        resource_risk_ledger.energy_epoch_miscoverage),
                    "link_allocation": float(
                        resource_risk_ledger.link_allocation),
                    "runtime_allocation": float(
                        resource_risk_ledger.runtime_allocation),
                    "energy_allocation": float(
                        resource_risk_ledger.energy_allocation),
                    "allocated_resource_miscoverage": float(
                        resource_risk_ledger.allocated_resource_miscoverage),
                    "system_total_miscoverage": float(
                        resource_risk_ledger.system_total_miscoverage),
                    "joint_coverage_floor_union_bound": float(
                        resource_risk_ledger.
                        joint_coverage_floor_union_bound),
                    "complete": bool(resource_risk_ledger.complete),
                }),
            "system_risk_rule": (
                "feedback+physics+link+runtime+energy<=total; "
                "Bonferroni union bound without independence"),
            "max_control_energy_j": shadow_energy_limit,
            "resource_accounting_complete": bool(
                shadow_resource_accounting_complete),
            "incomplete_components": [
                (
                    "hardware-calibrated complete package energy"
                    if compute_energy_calibration_metadata is None or not bool(
                        compute_energy_calibration_metadata[
                            "compute_energy_certificate_candidate"])
                    else "none for energy; package bound is calibrated"
                ),
                (
                    "deployment complete compute critical-path timing"
                    if runtime_latency_calibration_metadata is None or not bool(
                        runtime_latency_calibration_metadata[
                            "runtime_latency_certificate_candidate"])
                    else "none for compute timing; bound is calibrated"
                ),
                "deployment U2U network jitter",
                (
                    "finite independent calibration/validation sample support"
                    if not resource_finite_sample_plan_complete
                    else "none for finite-sample attainability"
                ),
            ],
            "commit_authority": False,
        },
        "horizon_digest_rendezvous": {
            "enabled": bool(horizon_digest_rendezvous),
            "digest_bits": int(rendezvous_layout.digest_bits),
            "collision_probability_upper": float(
                rendezvous_layout.collision_probability_upper),
            "identity_rule": (
                "digest_screens_then_full_selected_role_owner_equality"),
            "traffic_rule": (
                "one broadcast request, fixed reply from every non-"
                "coordinator, at most one bounded proposal"),
            "suffix_rule": (
                "repeat fixed-owner power verification; dependency commit "
                + (
                    "is reused only on exact transition identity"
                    if bool(
                        horizon_digest_rendezvous_reuse_structure_commit)
                    else "is conservatively repeated"
                )),
            "structure_commit_reuse_enabled": bool(
                horizon_digest_rendezvous_reuse_structure_commit),
            "deferred_top1_transport_enabled": bool(
                horizon_digest_rendezvous_defer_top1_transport),
            "set_membership_verifier_enabled": bool(
                horizon_set_membership_verifier),
            "reuse_primal_master_duals": bool(
                horizon_reuse_primal_master_duals),
            "network_repetition_count": int(network_repetitions),
            "network_tolerated_erasures_per_logical_packet": int(
                network_repetitions - 1),
            "network_excess_queue_bound_s": float(network_queue_bound),
            "network_reliability_interpretation": (
                "deterministic_fixed_repetition_support_not_a_packet_error_"
                "probability_or_empirical_loss_calibration"),
            "network_calibration_epoch": network_calibration_metadata,
            "deferred_prefix_rule": (
                "target invariants + No-op baseline power protocol + bounded "
                "owner proposals; no candidate verification/commit"),
            "deadline_rule": (
                "common+max(branch_compute+R*branch_network)+"
                "R*(rendezvous+suffix)+exact_H_compute+queue_bound"),
            "resource_accounting_complete": bool(
                shadow_resource_accounting_complete),
            "commit_authority": False,
        },
        "horizon_future_calibration_diagnostic": {
            "score": (
                "episode max over eligible events, horizon steps, current-"
                "provenance edges and targets of max([log(L/a)]+, "
                "[log(a/U)]+)"),
            "exchangeability_unit": "episode_seed",
            "future_information_used_by_controller": False,
            "finite_episode_scores": {
                str(seed): float(score)
                for seed, score in sorted(
                    horizon_future_episode_scores.items())
            },
            "infinite_score_seeds": sorted(
                horizon_future_infinite_score_seeds),
            "calibration_feasible_with_finite_multiplicative_margin": bool(
                horizon_future_rows
                and not horizon_future_infinite_score_seeds),
        },
        "owner_local_dd_covariance_radius": float(dd_covariance_radius),
        "frozen_owner_local_dd_additive_margin": float(dd_support_margin),
        "action_aligned_owner_state_enabled": bool(
            action_aligned_owner_state),
        "owner_proposal_transport_enabled": bool(owner_proposal_transport),
        "structure_sequence_transport_enabled": bool(
            certify_structure_sequence),
        "weak_target_count": 2,
        "horizon_weak_target_count": int(horizon_weak_targets),
        "summary": summary,
        "certificate_ready": False,
        "certificate_blockers": certificate_blockers,
        "fresh_test_consumed": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--price-bits", type=int, default=6)
    parser.add_argument("--feedback-bits", type=int, default=16)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--structure-steps", type=int, default=3)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument("--snr-margin-db", type=float, default=3.0)
    parser.add_argument("--latency-margin-s", type=float, default=5.0e-4)
    parser.add_argument(
        "--certify-structure-sequence", action="store_true",
        help="charge Top-1 baseline, every verification and every commit")
    parser.add_argument(
        "--proxy-mode", choices=(
            "feasible_mix", "dual_two_column", "dual_interval"),
        default="feasible_mix")
    parser.add_argument("--optimism-weight", type=float, default=0.0)
    parser.add_argument(
        "--owner-proposal-transport", action="store_true",
        help="quantize and charge one bounded proposal per owner and step")
    parser.add_argument(
        "--observability-mode",
        choices=(
            "analytic_full", "selected_lag1", "selected_lag1_conformal",
            "owner_local_conformal"),
        default="analytic_full")
    parser.add_argument("--unknown-edge-lower-log-margin", type=float)
    parser.add_argument("--unknown-edge-upper-log-margin", type=float)
    parser.add_argument(
        "--feedback-normalized-margin", type=float,
        help=(
            "frozen event-level margin used to generate target-reserve-safe "
            "power and structure candidates"))
    parser.add_argument(
        "--slow-geometry-fast-fallback", action="store_true",
        help=(
            "run a certified fast power/structure repair while slow geometry "
            "planning remains pending"))
    parser.add_argument(
        "--persistent-closed-loop", action="store_true",
        help=(
            "persist accepted structure/RF state and recursively replay its "
            "detection feedback; Actor Token visibility remains frozen"))
    parser.add_argument(
        "--controller-normalized-margin", type=float,
        help="frozen two-sided margin used by the persistent safety gate")
    parser.add_argument(
        "--target-invariant-cache-max-age-frames", type=int, default=0,
        help=(
            "retain same-target bistatic invariants for this many frames; "
            "zero disables the cache and expiry always fails closed"))
    parser.add_argument(
        "--target-invariant-token-transport", action="store_true",
        help=(
            "encode owner-authoritative invariant, age and version fields "
            "and charge their broadcast power, latency, bits and energy"))
    parser.add_argument(
        "--horizon-diagnostic-steps", type=int, default=0,
        help=(
            "evaluate each transported Top-1 atomic move over this causal "
            "receding horizon; zero disables diagnostics"))
    parser.add_argument(
        "--horizon-miscoverage", type=float,
        help="declared simultaneous H-by-Q physics-envelope miscoverage")
    parser.add_argument(
        "--horizon-discount", type=float, default=1.0)
    parser.add_argument(
        "--horizon-target-acceleration-std-mps2", type=float, default=0.0)
    parser.add_argument(
        "--horizon-switch-price", type=float, default=0.0,
        help="non-negative switched-edge shadow price in P_D units")
    parser.add_argument(
        "--horizon-bit-price", type=float, default=0.0,
        help="non-negative over-air-bit shadow price in P_D units")
    parser.add_argument(
        "--horizon-latency-price", type=float, default=0.0,
        help="non-negative seconds shadow price in P_D units")
    parser.add_argument(
        "--horizon-energy-price", type=float, default=0.0,
        help="non-negative control-energy shadow price in P_D units")
    parser.add_argument(
        "--horizon-proposal-ranking",
        choices=(
            "myopic", "common_mix_sum", "noharm_slack",
            "finite_horizon_margin", "screened_finite_margin",
            "distributed_screened_margin",
        ),
        default="myopic",
        help=(
            "owner-local Top-1 pre-ranking used only by the horizon "
            "diagnostic path"))
    parser.add_argument(
        "--horizon-ranking-comm-reserve-w", type=float, default=0.25,
        help=(
            "conservative per-UAV communication-power reserve used by "
            "horizon proposal pre-ranking; the transported candidate is "
            "invalidated if its realized protocol reserve exceeds it"))
    parser.add_argument(
        "--horizon-weak-target-count", type=int, default=2,
        help=(
            "number of predicted weak targets exposed to bounded N5/N6 "
            "proposal generation in the diagnostic horizon path"))
    parser.add_argument(
        "--horizon-local-shortlist-per-owner", type=int, default=1,
        help=(
            "cheaply screened candidates per owner receiving finite-horizon "
            "ranking in screened_finite_margin mode"))
    parser.add_argument(
        "--horizon-oracle-diagnostics", action="store_true",
        help=(
            "run optimistic all-owner and all-raw-candidate horizon oracles; "
            "these diagnostics never receive commit authority"))
    parser.add_argument(
        "--horizon-ranking-rounds", type=int, default=0,
        help=(
            "finite protocol rounds used only for horizon pre-ranking; zero "
            "inherits the final verification round count"))
    parser.add_argument(
        "--horizon-residual-log-margin", type=float, default=0.0,
        help=(
            "frozen simultaneous future-transition log margin applied to "
            "every H-by-edge-by-target coefficient interval; future trace "
            "outcomes are audit-only and never estimate this online"))
    parser.add_argument(
        "--horizon-paired-deflection-gate", action="store_true",
        help=(
            "use the exact common-coefficient-box lower bound on candidate "
            "minus No-op Deflection; requires zero resource shadow prices"))
    parser.add_argument(
        "--horizon-concurrent-owner-workers", type=int, default=0,
        help=(
            "execute distributed screened owner groups on this many actual "
            "threads; zero preserves the serial diagnostic implementation"))
    parser.add_argument(
        "--horizon-common-power-plan", action="store_true",
        help=(
            "solve one robust sensing-power weight plan from horizon-minimum "
            "coefficients/budgets and reuse it at every predicted step"))
    parser.add_argument(
        "--shadow-logical-cpu-power-w", type=float,
        help=(
            "frozen logical-CPU power used to charge measured process CPU "
            "seconds; this is a model, not a hardware meter reading"))
    parser.add_argument(
        "--shadow-max-control-energy-j", type=float,
        help="optional hard combined one-step plus horizon energy ceiling")
    parser.add_argument(
        "--shadow-compute-energy-calibration", type=Path,
        help=(
            "frozen output from calibrate_compute_energy_epoch.py; binds a "
            "complete-controller package-energy bound to code/config, "
            "hardware, runtime and disjoint episode splits"))
    parser.add_argument(
        "--shadow-runtime-latency-calibration", type=Path,
        help=(
            "frozen output from calibrate_runtime_latency_epoch.py; binds "
            "the complete compute critical path while excluding network"))
    parser.add_argument(
        "--shadow-current-hardware-id",
        help=(
            "stable identity of the hardware executing this audit; required "
            "and matched exactly when a resource epoch is supplied"))
    parser.add_argument(
        "--shadow-current-runtime-id",
        help=(
            "stable OS/runtime/library identity for this audit; required and "
            "matched exactly when a resource epoch is supplied"))
    parser.add_argument(
        "--shadow-reconcile-horizon-power", action="store_true",
        help=(
            "when both routes accept the same discrete structure, record the "
            "already horizon-certified RF plan as the shadow candidate"))
    parser.add_argument(
        "--horizon-digest-rendezvous", action="store_true",
        help=(
            "after independent branch ranking, screen the raw causal horizon "
            "set with a bounded 256-bit digest exchange, reconstruct one "
            "matching proposal, and charge a conservative exact suffix"))
    parser.add_argument(
        "--horizon-digest-rendezvous-reuse-structure-commit",
        action="store_true",
        help=(
            "reuse the feasible one-step dependency commit only when the "
            "rendezvous candidate has exactly identical structure/role/owner; "
            "the horizon power protocol is still re-run and charged"))
    parser.add_argument(
        "--horizon-digest-rendezvous-defer-top1-transport",
        action="store_true",
        help=(
            "before consensus, transport only target invariants, No-op "
            "baseline and bounded owner proposals; defer candidate-specific "
            "horizon verification/commit until after digest matching"))
    parser.add_argument(
        "--horizon-set-membership-verifier", action="store_true",
        help=(
            "independently enumerate the exact causal atomic candidate set "
            "without ranking unrelated candidates, then H-step verify only "
            "the full-identity digest match supplied by the one-step branch"))
    parser.add_argument(
        "--horizon-reuse-primal-master-duals", action="store_true",
        help=(
            "reuse the restricted primal HiGHS inequality multipliers as "
            "the optimal target-price vector by LP strong duality, avoiding "
            "a redundant paired dual solve without relaxing any RF bound"))
    parser.add_argument(
        "--horizon-network-repetition-count", type=int, default=1,
        help=(
            "fixed copies of every logical horizon-path packet; R copies "
            "deterministically tolerate at most R-1 erasures per packet"))
    parser.add_argument(
        "--horizon-network-excess-queue-bound-s", type=float, default=0.0,
        help=(
            "non-negative upper bound on excess queue/scheduling latency "
            "for the complete repeated horizon protocol path"))
    parser.add_argument(
        "--horizon-network-calibration", type=Path,
        help=(
            "frozen output from calibrate_link_reliability_epoch.py; its "
            "finite R/J override defaults and are checked for split leakage"))
    parser.add_argument(
        "--persistent-rf-recourse-from-trace", action="store_true",
        help=(
            "persist only structure while refreshing the local continuous "
            "RF proposal from each frozen trace event"))
    parser.add_argument(
        "--owner-dd-covariance-radius", type=float, default=0.0,
        help=(
            "target belief ellipsoid radius used by the deterministic DD "
            "geometry lower bound; zero uses the point prediction"))
    parser.add_argument(
        "--owner-dd-additive-margin", type=float,
        help=(
            "frozen additive DD residual margin after the geometry bound; "
            "defaults to the legacy unknown-edge margin"))
    parser.add_argument(
        "--action-aligned-owner-state", action="store_true",
        help=(
            "project action-time local kinematics through the known delta_p "
            "before predicting the post-action physical outcome"))
    parser.add_argument(
        "--controller-certificate-mode",
        choices=(
            "feedback", "feedback_or_physics", "risk_budgeted_union"),
        default="feedback",
        help=(
            "allow a complete calibrated physics interval to certify an "
            "action when the feedback certificate defers"))
    parser.add_argument(
        "--controller-total-miscoverage", type=float,
        help="episode-level total risk budget for risk_budgeted_union")
    parser.add_argument(
        "--controller-feedback-miscoverage", type=float,
        help="risk allocation for the calibrated feedback route")
    parser.add_argument(
        "--controller-physics-miscoverage", type=float,
        help="risk allocation for the calibrated physics route")
    parser.add_argument(
        "--controller-link-miscoverage", type=float,
        help="risk allocation for the U2U link certificate")
    parser.add_argument(
        "--controller-runtime-miscoverage", type=float,
        help="risk allocation for the complete compute latency certificate")
    parser.add_argument(
        "--controller-energy-miscoverage", type=float,
        help="risk allocation for the package energy certificate")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    network_calibration = None
    compute_energy_calibration = None
    runtime_latency_calibration = None
    network_repetition_count = int(args.horizon_network_repetition_count)
    network_queue_bound_s = float(
        args.horizon_network_excess_queue_bound_s)
    if args.horizon_network_calibration is not None:
        calibration_path = Path(args.horizon_network_calibration)
        network_calibration = json.loads(
            calibration_path.read_text(encoding="utf-8"))
        recommended = network_calibration.get(
            "recommended_controller_parameters", {})
        if not bool(recommended.get("usable_for_shadow_controller", False)):
            parser.error("network calibration is not finite/usable")
        calibrated_repetition_count = int(recommended[
            "horizon_network_repetition_count"])
        calibrated_queue_bound_s = float(recommended[
            "horizon_network_excess_queue_bound_s"])
        if (
            network_repetition_count != 1
            and network_repetition_count != calibrated_repetition_count
        ):
            parser.error("explicit repetition count conflicts with calibration")
        if (
            network_queue_bound_s != 0.0
            and not np.isclose(
                network_queue_bound_s,
                calibrated_queue_bound_s,
                rtol=0.0,
                atol=1.0e-15,
            )
        ):
            parser.error("explicit queue bound conflicts with calibration")
        network_repetition_count = calibrated_repetition_count
        network_queue_bound_s = calibrated_queue_bound_s
        provenance = network_calibration.setdefault("provenance", {})
        provenance["calibration_artifact"] = {
            "path": str(calibration_path),
            "sha256": _file_sha256(calibration_path),
        }
    if args.shadow_compute_energy_calibration is not None:
        energy_calibration_path = Path(
            args.shadow_compute_energy_calibration)
        compute_energy_calibration = json.loads(
            energy_calibration_path.read_text(encoding="utf-8"))
        recommended = compute_energy_calibration.get(
            "recommended_controller_parameters", {})
        if not bool(recommended.get("usable_for_shadow_controller", False)):
            parser.error("compute-energy calibration is not finite/usable")
        provenance = compute_energy_calibration.setdefault("provenance", {})
        provenance["calibration_artifact"] = {
            "path": str(energy_calibration_path),
            "sha256": _file_sha256(energy_calibration_path),
        }
    if args.shadow_runtime_latency_calibration is not None:
        runtime_calibration_path = Path(
            args.shadow_runtime_latency_calibration)
        runtime_latency_calibration = json.loads(
            runtime_calibration_path.read_text(encoding="utf-8"))
        recommended = runtime_latency_calibration.get(
            "recommended_controller_parameters", {})
        if not bool(recommended.get("usable_for_shadow_controller", False)):
            parser.error("runtime-latency calibration is not finite/usable")
        provenance = runtime_latency_calibration.setdefault("provenance", {})
        provenance["calibration_artifact"] = {
            "path": str(runtime_calibration_path),
            "sha256": _file_sha256(runtime_calibration_path),
        }
    result = audit(
        args.trace,
        args.config,
        seed_limit=max(1, int(args.seed_limit)),
        rounds=max(1, int(args.rounds)),
        price_bits=int(args.price_bits),
        feedback_bits=int(args.feedback_bits),
        top_m=max(1, int(args.top_m)),
        structure_steps=max(1, int(args.structure_steps)),
        qos_floor=float(args.qos_floor),
        snr_margin_db=float(args.snr_margin_db),
        latency_margin_s=float(args.latency_margin_s),
        certify_structure_sequence=bool(args.certify_structure_sequence),
        proxy_mode=str(args.proxy_mode),
        optimism_weight=float(args.optimism_weight),
        owner_proposal_transport=bool(args.owner_proposal_transport),
        observability_mode=str(args.observability_mode),
        unknown_edge_lower_log_margin=args.unknown_edge_lower_log_margin,
        unknown_edge_upper_log_margin=args.unknown_edge_upper_log_margin,
        feedback_normalized_margin=args.feedback_normalized_margin,
        slow_geometry_fast_fallback=bool(args.slow_geometry_fast_fallback),
        persistent_closed_loop=bool(args.persistent_closed_loop),
        controller_normalized_margin=args.controller_normalized_margin,
        target_invariant_cache_max_age_frames=int(
            args.target_invariant_cache_max_age_frames),
        target_invariant_token_transport=bool(
            args.target_invariant_token_transport),
        horizon_diagnostic_steps=int(args.horizon_diagnostic_steps),
        horizon_miscoverage=args.horizon_miscoverage,
        horizon_discount=float(args.horizon_discount),
        horizon_target_acceleration_std_mps2=float(
            args.horizon_target_acceleration_std_mps2),
        horizon_switch_price=float(args.horizon_switch_price),
        horizon_bit_price=float(args.horizon_bit_price),
        horizon_latency_price=float(args.horizon_latency_price),
        horizon_energy_price=float(args.horizon_energy_price),
        horizon_proposal_ranking=str(args.horizon_proposal_ranking),
        horizon_ranking_comm_reserve_w=float(
            args.horizon_ranking_comm_reserve_w),
        horizon_weak_target_count=int(args.horizon_weak_target_count),
        horizon_local_shortlist_per_owner=int(
            args.horizon_local_shortlist_per_owner),
        horizon_oracle_diagnostics=bool(
            args.horizon_oracle_diagnostics),
        horizon_ranking_rounds=int(args.horizon_ranking_rounds),
        horizon_residual_log_margin=float(
            args.horizon_residual_log_margin),
        horizon_paired_deflection_gate=bool(
            args.horizon_paired_deflection_gate),
        horizon_common_power_plan=bool(args.horizon_common_power_plan),
        horizon_concurrent_owner_workers=int(
            args.horizon_concurrent_owner_workers),
        shadow_logical_cpu_power_w=args.shadow_logical_cpu_power_w,
        shadow_max_control_energy_j=args.shadow_max_control_energy_j,
        shadow_reconcile_horizon_power=bool(
            args.shadow_reconcile_horizon_power),
        horizon_digest_rendezvous=bool(
            args.horizon_digest_rendezvous),
        horizon_digest_rendezvous_reuse_structure_commit=bool(
            args.horizon_digest_rendezvous_reuse_structure_commit),
        horizon_digest_rendezvous_defer_top1_transport=bool(
            args.horizon_digest_rendezvous_defer_top1_transport),
        horizon_set_membership_verifier=bool(
            args.horizon_set_membership_verifier),
        horizon_reuse_primal_master_duals=bool(
            args.horizon_reuse_primal_master_duals),
        horizon_network_repetition_count=int(
            network_repetition_count),
        horizon_network_excess_queue_bound_s=float(
            network_queue_bound_s),
        horizon_network_calibration_epoch=network_calibration,
        shadow_compute_energy_calibration_epoch=(
            compute_energy_calibration),
        shadow_runtime_latency_calibration_epoch=(
            runtime_latency_calibration),
        shadow_current_hardware_id=args.shadow_current_hardware_id,
        shadow_current_runtime_id=args.shadow_current_runtime_id,
        persistent_rf_recourse_from_trace=bool(
            args.persistent_rf_recourse_from_trace),
        owner_dd_covariance_radius=float(args.owner_dd_covariance_radius),
        owner_dd_additive_margin=args.owner_dd_additive_margin,
        action_aligned_owner_state=bool(args.action_aligned_owner_state),
        controller_certificate_mode=str(args.controller_certificate_mode),
        controller_total_miscoverage=args.controller_total_miscoverage,
        controller_feedback_miscoverage=(
            args.controller_feedback_miscoverage),
        controller_physics_miscoverage=(
            args.controller_physics_miscoverage),
        controller_link_miscoverage=args.controller_link_miscoverage,
        controller_runtime_miscoverage=(
            args.controller_runtime_miscoverage),
        controller_energy_miscoverage=args.controller_energy_miscoverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "event_count": result["event_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "summary": result["summary"],
        "certificate_ready": result["certificate_ready"],
    }, indent=2))


if __name__ == "__main__":
    main()
