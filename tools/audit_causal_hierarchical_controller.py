#!/usr/bin/env python
"""Audit D0.47 with only lagged feedback and current local kinematics.

The current physical coefficient is reconstructed only after the controller
returns and is used exclusively as an outcome label.  Structure and role state
persist across events; trace RF weights remain a fast-timescale local recourse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.agents.equivariant_movement_plan import (  # noqa: E402
    FrozenEquivariantMovementPlanner,
)
from tools.audit_local_candidate_upper_bound import (  # noqa: E402
    _infer_observation_slices,
)
from tools.audit_power_repair_transport_trace import (  # noqa: E402
    _communication_model,
)
from tools.audit_qos_threshold_boundary import _coefficient  # noqa: E402
from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _ordered_unique,
    _recorded_power,
)
from uav_isac.coordination.causal_hierarchical_controller import (  # noqa: E402
    FrozenCausalEnvelopeCalibration,
    certified_causal_hierarchical_isac_control,
)
from uav_isac.coordination.causal_joint_plan import (  # noqa: E402
    CausalJointPlanCommitment,
    commit_movement_plan_with_held_rf,
    commit_zero_order_hold_joint_plan,
)
from uav_isac.coordination.certified_geometry_repair import (  # noqa: E402
    CertifiedGeometryRepairConfig,
    certified_trust_region_geometry_repair,
)
from uav_isac.coordination.certified_hierarchical_controller import (  # noqa: E402
    CertifiedHierarchicalControllerConfig,
    certified_hierarchical_isac_control,
)
from uav_isac.coordination.certified_maxmin_power_controller import (  # noqa: E402
    CertifiedMaxMinPowerConfig,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
)
from uav_isac.coordination.owner_local_physics import (  # noqa: E402
    OwnerLocalKinematicState,
    advance_owner_local_kinematics,
    decode_owner_local_kinematics,
    propagate_owner_local_horizon,
    reciprocal_token_candidate_mask,
)
from uav_isac.coordination.owner_proposal_transport import (  # noqa: E402
    quantize_nonnegative_float16_lower,
)
from uav_isac.coordination.persistent_geometry_execution import (  # noqa: E402
    CertifiedGeometryTube,
    FrozenTokenInbox,
    replace_with_geometry_command,
    start_certified_geometry_tube,
)
from uav_isac.coordination.geometry_repair_transport import (  # noqa: E402
    GeometryRepairWireLayout,
    quantize_displacement_toward_zero,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor_from_observables,
)
from uav_isac.evaluation.horizon_future_audit import (  # noqa: E402
    simultaneous_log_envelope_score,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    delivered_target_tokens,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)
from uav_isac.physical.geometry import (  # noqa: E402
    compute_all_bistatic_params,
)
from uav_isac.physical.otfs import compute_dd_effectiveness  # noqa: E402


def _realized_pd(
    coefficient: np.ndarray,
    selected: np.ndarray,
    sensing_power_w: np.ndarray,
    p_fa: float,
) -> np.ndarray:
    gain, _ = fixed_owner_gain_matrix(
        coefficient, [tuple(edge) for edge in np.argwhere(selected)])
    return compute_detection_probabilities(
        np.sum(gain * sensing_power_w, axis=0), float(p_fa))


def _physical_design_fingerprint(cfg: object) -> tuple[object, ...]:
    return (
        int(cfg.scenario.K), int(cfg.scenario.Q),
        tuple(float(value) for value in cfg.scenario.region_size),
        float(cfg.scenario.height), float(cfg.scenario.dt),
        float(cfg.uav.v_max), float(cfg.otfs.fc), float(cfg.otfs.delta_f),
        float(cfg.otfs.T_sym), int(cfg.otfs.M), int(cfg.otfs.N),
        float(cfg.detection.g_min), bool(cfg.channel.use_swerling),
        bool(cfg.marl.tracking_enabled),
    )


def _counterfactual_coefficient(
    state: object,
    target_state: np.ndarray,
    cfg: object,
) -> np.ndarray:
    """Construct a post-decision static-target outcome label."""
    target = np.asarray(target_state, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 4:
        raise ValueError("target state must have shape (Q,4)")
    if np.any(np.abs(target[:, 2:]) > 1.0e-12):
        raise ValueError("geometry shadow audit supports static targets only")
    positions = np.asarray(state.uav_position_m, dtype=np.float64)
    velocities = np.asarray(state.uav_velocity_mps, dtype=np.float64)
    target_positions = np.concatenate((
        target[:, :2], np.zeros((target.shape[0], 1), dtype=np.float64),
    ), axis=1)
    target_velocities = np.zeros_like(target_positions)
    tau, nu, alpha = compute_all_bistatic_params(
        positions,
        velocities,
        target_positions,
        target_velocities,
        np.zeros(positions.shape[0], dtype=np.int8),
        float(cfg.otfs.fc),
        float(cfg.target.rcs),
        role_agnostic=True,
    )
    dd = np.zeros_like(alpha)
    for transmitter, receiver, target_id in np.argwhere(
        ~np.eye(positions.shape[0], dtype=bool)[:, :, None]
        & np.ones_like(alpha, dtype=bool)
    ):
        dd[transmitter, receiver, target_id] = compute_dd_effectiveness(
            float(tau[transmitter, receiver, target_id]),
            float(nu[transmitter, receiver, target_id]),
            float(cfg.otfs.delta_f),
            float(cfg.otfs.T_sym),
            int(cfg.otfs.M),
            int(cfg.otfs.N),
            float(cfg.detection.g_min),
        )
    return per_watt_deflection_tensor_from_observables(
        alpha,
        dd,
        np.ones_like(alpha),
        T_sym=float(cfg.otfs.T_sym),
        M=int(cfg.otfs.M),
        N=int(cfg.otfs.N),
        kT=float(cfg.channel.kT),
        bandwidth_hz=float(cfg.otfs.B),
        noise_figure_db=float(cfg.channel.NF),
        g_tx_dbi=float(cfg.otfs.g_tx_dBi),
        g_rx_dbi=float(cfg.otfs.g_rx_dBi),
        n_cpi=int(cfg.otfs.n_cpi),
        g_min=float(cfg.detection.g_min),
        use_swerling=False,
    )


def _lower_tail_cvar(values: list[float], fraction: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if array.size == 0:
        return float("nan")
    count = max(1, int(np.ceil(float(fraction) * array.size)))
    return float(np.mean(array[:count]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _planner_exact_invariants(
    planner: FrozenEquivariantMovementPlanner | None,
) -> list[str]:
    if planner is None:
        return []
    invariants = [
        "target_permutation_invariance",
        "uav_permutation_equivariance",
        "per_step_speed_disk_projection",
        "known_movement_hold_phase",
        "current_feasible_rf_held_over_horizon",
        "uniform_region_scale_covariance_when_enabled",
    ]
    if planner.metadata.architecture == (
        "shared-uav-peer-target-double-set-equivariant-zoh-residual-v2"
    ):
        invariants.insert(2, "peer_uav_set_aggregation")
    return invariants


def _runtime_state_digest(
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    battery_j: np.ndarray,
    inbox: FrozenTokenInbox,
    selected: np.ndarray,
    role: np.ndarray,
    previous_coefficient: np.ndarray,
    previous_observed: np.ndarray,
    previous_state: OwnerLocalKinematicState,
    target_cache: object | None,
    tube: CertifiedGeometryTube | None,
) -> str:
    """Hash every recurrent variable that can affect a future audit event."""
    digest = hashlib.sha256()
    for value in (
        position_m,
        velocity_mps,
        battery_j,
        inbox.token_mask,
        inbox.age_frames,
        selected,
        role,
        previous_coefficient,
        previous_observed,
        previous_state.uav_position_m,
        previous_state.uav_velocity_mps,
        previous_state.target_mean_by_owner,
        previous_state.target_cov_diag_by_owner,
        previous_state.target_aoi_frames_by_owner,
    ):
        array = np.ascontiguousarray(value)
        if np.issubdtype(array.dtype, np.floating):
            if np.any(~np.isfinite(array)):
                raise ValueError("runtime state digest requires finite values")
            mantissa, exponent = np.frexp(array.astype(np.float64))
            mantissa = np.rint(np.ldexp(mantissa, 30))
            array = np.ldexp(mantissa, exponent - 30)
            array[array == 0.0] = 0.0
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    if target_cache is not None:
        for value in (
            target_cache.target_invariant,
            target_cache.age_frames,
            target_cache.version,
        ):
            array = np.ascontiguousarray(value)
            if np.issubdtype(array.dtype, np.floating):
                if np.any(~np.isfinite(array)):
                    raise ValueError(
                        "runtime cache digest requires finite values")
                mantissa, exponent = np.frexp(array.astype(np.float64))
                mantissa = np.rint(np.ldexp(mantissa, 30))
                array = np.ldexp(mantissa, exponent - 30)
                array[array == 0.0] = 0.0
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
    if tube is None:
        digest.update(b"no-active-tube")
    else:
        digest.update(np.asarray([
            tube.decision_frame,
            tube.start_frame,
            tube.next_step,
            int(tube.proof_available),
        ], dtype="<i8").tobytes())
        digest.update(tube.joint_plan_digest.encode("ascii"))
        digest.update(tube.execution_mode.encode("ascii"))
    return digest.hexdigest()[:24]


def _replace_uav_kinematics(
    state: OwnerLocalKinematicState,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
) -> OwnerLocalKinematicState:
    """Keep the frozen belief path but inject persistent physical UAV state."""
    position = np.asarray(position_m, dtype=np.float64)
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    if (
        position.shape != state.uav_position_m.shape
        or velocity.shape != state.uav_velocity_mps.shape
    ):
        raise ValueError("persistent UAV state dimensions changed")
    return OwnerLocalKinematicState(
        uav_position_m=position.copy(),
        uav_velocity_mps=velocity.copy(),
        target_mean_by_owner=state.target_mean_by_owner.copy(),
        target_cov_diag_by_owner=state.target_cov_diag_by_owner.copy(),
        target_aoi_frames_by_owner=(
            state.target_aoi_frames_by_owner.copy()),
    )


def _flight_energy_by_uav(
    displacement_m: np.ndarray,
    *,
    dt_s: float,
    static_power_w: float,
    quadratic_power_coeff: float,
) -> np.ndarray:
    displacement = np.asarray(displacement_m, dtype=np.float64)
    speed = np.linalg.norm(displacement, axis=1) / float(dt_s)
    return (
        float(static_power_w) + float(quadratic_power_coeff) * speed ** 2
    ) * float(dt_s)


def _quantize_mobility_plan(
    plan_m: np.ndarray,
    *,
    maximum_displacement_m: float,
    bits_per_axis: int,
) -> np.ndarray:
    plan = np.asarray(plan_m, dtype=np.float64)
    if plan.ndim != 3 or plan.shape[2] != 2:
        raise ValueError("mobility plan must have shape (H,K,2)")
    result = np.zeros_like(plan)
    for step in range(plan.shape[0]):
        for agent in range(plan.shape[1]):
            result[step, agent] = quantize_displacement_toward_zero(
                plan[step, agent],
                maximum_component_m=float(maximum_displacement_m),
                bits_per_axis=int(bits_per_axis),
            )
    return result


def _quantize_rf_plan(
    communication_plan_w: np.ndarray,
    sensing_plan_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    communication = np.asarray(communication_plan_w, dtype=np.float64)
    sensing = np.asarray(sensing_plan_w, dtype=np.float64)
    if (
        communication.ndim != 2 or sensing.ndim != 3
        or sensing.shape[:2] != communication.shape
    ):
        raise ValueError("RF plan must have shape (H,K)/(H,K,Q)")
    wire_communication = np.asarray([
        quantize_nonnegative_float16_lower(value)
        for value in communication.reshape(-1)
    ], dtype=np.float64).reshape(communication.shape)
    wire_sensing = np.asarray([
        quantize_nonnegative_float16_lower(value)
        for value in sensing.reshape(-1)
    ], dtype=np.float64).reshape(sensing.shape)
    if np.any(
        wire_communication + np.sum(wire_sensing, axis=2)
        > 1.0 + 1.0e-12
    ):
        raise AssertionError("wire RF plan exceeds the per-UAV power budget")
    return wire_communication, wire_sensing


def _tube_branch_coefficient(
    state: OwnerLocalKinematicState,
    tube: CertifiedGeometryTube,
    step: int,
    target_state: np.ndarray,
    cfg: object,
    *,
    repair: bool,
) -> np.ndarray:
    """Reconstruct one exact tube branch under the common target outcome."""
    movement = (
        tube.repair_movement_plan_m
        if bool(repair) else tube.baseline_movement_plan_m)
    position = (
        tube.repair_position_plan_m
        if bool(repair) else tube.baseline_position_plan_m)
    velocity = np.zeros_like(state.uav_velocity_mps, dtype=np.float64)
    velocity[:, :2] = (
        np.asarray(movement[step], dtype=np.float64)
        / float(cfg.scenario.dt))
    branch_state = _replace_uav_kinematics(
        state,
        np.asarray(position[step], dtype=np.float64),
        velocity,
    )
    return _counterfactual_coefficient(branch_state, target_state, cfg)


def audit(
    trace_path: Path,
    config_path: Path,
    calibration_path: Path,
    *,
    seed_limit: int,
    max_age_frames: int,
    qos_floor: float,
    rounds: int,
    ranking_rounds: int,
    owners_per_target: int,
    snr_margin_db: float,
    latency_margin_s: float,
    persistent_geometry: bool = False,
    enable_geometry: bool = True,
    announced_mobility_horizon: bool = False,
    causal_joint_plan: bool = False,
    causal_plan_checkpoint: Path | None = None,
    causal_plan_role: str = "baseline",
    allow_uniform_plan_region_scaling: bool = False,
    geometry_execution_mode: str = "repair",
    geometry_verification_top_m: int = 4,
    geometry_horizon_steps: int | None = None,
    mobility_reserve_fraction: float = 1.0,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    _, communication_model = _communication_model(
        config_path,
        message_dim=int(np.asarray(data["outgoing_message"]).shape[-1]),
    )
    calibration = FrozenCausalEnvelopeCalibration.from_json(
        calibration_path, source_root=ROOT)
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    pair_limit = int(np.asarray(
        data["target_pair_limit"]).reshape(-1)[0])
    receiver_limit = int(np.asarray(
        data["reports_per_receiver"]).reshape(-1)[0])
    if K != int(cfg.scenario.K) or Q != int(cfg.scenario.Q):
        raise ValueError("trace dimensions do not match config")
    if bool(cfg.marl.tracking_enabled) or bool(cfg.channel.use_swerling):
        raise ValueError("frozen causal calibration domain does not match config")
    reserve_fraction = float(mobility_reserve_fraction)
    if (
        not np.isfinite(reserve_fraction)
        or not 0.0 < reserve_fraction <= 1.0
    ):
        raise ValueError("mobility reserve fraction must lie in (0,1]")
    if bool(announced_mobility_horizon) and bool(causal_joint_plan):
        raise ValueError(
            "recorded-future and causal joint-plan modes are mutually exclusive")
    if causal_plan_checkpoint is not None and not bool(causal_joint_plan):
        raise ValueError(
            "a causal plan checkpoint requires --causal-joint-plan")
    plan_role = str(causal_plan_role)
    if plan_role not in {"baseline", "proposal"}:
        raise ValueError("causal plan role must be baseline or proposal")
    if causal_plan_checkpoint is None and plan_role != "baseline":
        raise ValueError("proposal role requires a causal plan checkpoint")
    execution_mode = str(geometry_execution_mode)
    if execution_mode not in {"repair", "committed_baseline"}:
        raise ValueError("geometry execution mode is invalid")

    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    overlap = set(seed_order) & set(calibration.calibration_episode_ids)
    if overlap:
        raise ValueError(
            f"validation episodes overlap frozen calibration: {sorted(overlap)}")
    certificate_horizon = min(max(
        3,
        int(calibration.horizon_steps)
        if geometry_horizon_steps is None
        else int(geometry_horizon_steps),
    ), 8)
    causal_planner = (
        FrozenEquivariantMovementPlanner.from_checkpoint(
            causal_plan_checkpoint,
            validation_episode_ids=tuple(int(seed) for seed in seed_order),
            horizon_steps=certificate_horizon,
            movement_decision_interval=int(
                cfg.marl.movement_decision_interval),
            region_size_m=tuple(
                float(value) for value in cfg.scenario.region_size),
            maximum_displacement_m=(
                float(cfg.uav.v_max) * float(cfg.scenario.dt)),
            allow_uniform_region_scaling=bool(
                allow_uniform_plan_region_scaling),
            validation_domain_key=f"k{K}q{Q}",
        )
        if causal_plan_checkpoint is not None else None
    )
    runtime_fingerprint = _physical_design_fingerprint(cfg)
    for raw_config in calibration.source_configs:
        source_config = Path(raw_config)
        if not source_config.is_absolute():
            source_config = ROOT / source_config
        source_cfg = load_config(str(source_config))
        if _physical_design_fingerprint(source_cfg) != runtime_fingerprint:
            raise ValueError(
                "runtime physical design differs from calibration sources")
    keep = np.isin(seeds, seed_order)
    indices = np.flatnonzero(keep)
    slices = _infer_observation_slices(
        np.asarray(data["local_obs"]).shape[-1], K, Q)
    token_visible, token_age = delivered_target_tokens(
        np.asarray(data["local_obs"], dtype=np.float32), slices)
    reciprocal_support = reciprocal_token_candidate_mask(token_visible)
    controller_config = CertifiedHierarchicalControllerConfig(
        qos_floor=float(qos_floor),
        target_pair_limit=pair_limit,
        reports_per_receiver=receiver_limit,
        owners_per_target=min(int(owners_per_target), K),
        structure_weak_target_count=Q,
        structure_ranking_rounds=int(ranking_rounds),
        require_pre_reserved_comm_power=True,
        snr_margin_db=float(snr_margin_db),
        latency_margin_s=float(latency_margin_s),
        power=CertifiedMaxMinPowerConfig(
            rounds=int(rounds),
            price_bits=6,
            feedback_bits=16,
            snr_margin_db=float(snr_margin_db),
            latency_margin_s=float(latency_margin_s),
            qos_floor=float(qos_floor),
        ),
    )

    deployed_selected: dict[int, np.ndarray] = {}
    deployed_role: dict[int, np.ndarray] = {}
    previous_coefficient: dict[int, np.ndarray] = {}
    previous_observed: dict[int, np.ndarray] = {}
    previous_state: dict[int, object] = {}
    previous_frame: dict[int, int] = {}
    target_cache: dict[int, object] = {}
    persistent_position: dict[int, np.ndarray] = {}
    persistent_velocity: dict[int, np.ndarray] = {}
    persistent_battery: dict[int, np.ndarray] = {}
    token_inbox: dict[int, FrozenTokenInbox] = {}
    geometry_tube: dict[int, CertifiedGeometryTube] = {}
    last_frame_by_seed = {
        int(seed): int(np.max(frames[seeds == int(seed)]))
        for seed in seed_order
    }
    row_by_seed_frame = {
        (int(seeds[row]), int(frames[row])): int(row)
        for row in indices
    }
    rows: list[dict[str, object]] = []
    episode_scores: dict[int, float] = {seed: 0.0 for seed in seed_order}
    started = perf_counter()
    for row_id in indices:
        row = int(row_id)
        seed = int(seeds[row])
        frame = int(frames[row])
        decoded_state = decode_owner_local_kinematics(
            np.asarray(data["local_obs"][row], dtype=np.float32),
            slices,
            area_size_m=tuple(float(value) for value in cfg.scenario.region_size),
            height_m=float(cfg.scenario.height),
        )
        normalized_self_state = np.asarray(
            slices.extract_self(np.asarray(
                data["local_obs"][row], dtype=np.float32)),
            dtype=np.float64,
        )
        trace_battery_j = normalized_self_state[:, 6] * 50_000.0
        recorded_comm, recorded_sensing, power_error = _recorded_power(
            data, row)
        active_tube = (
            geometry_tube.get(seed) if bool(persistent_geometry) else None)
        if bool(persistent_geometry):
            if seed not in persistent_position:
                persistent_position[seed] = decoded_state.uav_position_m.copy()
                persistent_velocity[seed] = decoded_state.uav_velocity_mps.copy()
                persistent_battery[seed] = trace_battery_j.copy()
                token_inbox[seed] = FrozenTokenInbox.from_observation(
                    token_visible[row], token_age[row],
                    ttl_frames=int(cfg.marl.comm_message_ttl_frames),
                )
            action_state = _replace_uav_kinematics(
                decoded_state,
                persistent_position[seed],
                persistent_velocity[seed],
            )
            mobility_action = replace_with_geometry_command(
                np.asarray(data["delta_p"][row], dtype=np.float64),
                active_tube,
                frame=frame,
            )
            if active_tube is None and (
                bool(announced_mobility_horizon) or bool(causal_joint_plan)
            ):
                mobility_action = reserve_fraction * mobility_action
                mobility_action = _quantize_mobility_plan(
                    mobility_action[None, ...],
                    maximum_displacement_m=(
                        float(cfg.uav.v_max) * float(cfg.scenario.dt)),
                    bits_per_axis=GeometryRepairWireLayout(
                        K, Q).displacement_bits_per_axis,
                )[0]
            observed_battery_j = persistent_battery[seed].copy()
            if active_tube is None:
                if (
                    bool(announced_mobility_horizon)
                    or bool(causal_joint_plan)
                ):
                    comm, sensing = _quantize_rf_plan(
                        recorded_comm[None, :], recorded_sensing[None, ...])
                    comm = comm[0]
                    sensing = sensing[0]
                else:
                    comm = recorded_comm
                    sensing = recorded_sensing
                current_reciprocal_support = reciprocal_token_candidate_mask(
                    token_inbox[seed].visible[None, ...])[0]
            else:
                comm = active_tube.communication_power_w[
                    active_tube.next_step].copy()
                sensing = active_tube.sensing_power_w[
                    active_tube.next_step].copy()
                current_reciprocal_support = reciprocal_token_candidate_mask(
                    token_inbox[seed].visible[None, ...])[0]
        else:
            action_state = decoded_state
            mobility_action = np.asarray(
                data["delta_p"][row], dtype=np.float64)
            observed_battery_j = trace_battery_j
            comm = recorded_comm
            sensing = recorded_sensing
            current_reciprocal_support = np.asarray(
                reciprocal_support[row], dtype=bool)
        current_state = advance_owner_local_kinematics(
            action_state,
            mobility_action,
            dt_s=float(cfg.scenario.dt),
            max_speed_mps=float(cfg.uav.v_max),
            area_size_m=tuple(float(value) for value in cfg.scenario.region_size),
            advance_targets=False,
        )
        target_outcome = np.asarray(
            data["target_states"][row], dtype=np.float64)
        actual_coefficient = (
            _counterfactual_coefficient(current_state, target_outcome, cfg)
            if bool(persistent_geometry)
            else _coefficient(data, row, cfg)
        )
        if seed not in deployed_selected:
            selected = np.asarray(data["teacher_pair"][row], dtype=bool)
            role = 1 - np.asarray(
                data["teacher_role"][row], dtype=np.int8)
            deployed_selected[seed] = selected.copy()
            deployed_role[seed] = role.copy()
            baseline_pd = _realized_pd(
                actual_coefficient, selected, sensing, p_fa)
            decision = None
            causal_available = False
            reason = "first_event_has_no_lagged_feedback"
            candidate_selected = selected
            candidate_role = role
            candidate_sensing = sensing
            candidate_comm = comm
            bits = 0
            latency = 0.0
            energy = 0.0
            route = "unavailable"
            envelope_covered = True
            lower_failure = False
            upper_failure = False
            deployment_certificate = False
            tube_executed = False
            tube_step = -1
            tube_fast_accepted = False
            tube_fast_route = "none"
            tube_controller = None
            support = current_reciprocal_support.copy()
        elif active_tube is not None:
            selected = active_tube.selected.copy()
            role = active_tube.role.copy()
            support = active_tube.certified_support.copy()
            incumbent_sensing = active_tube.sensing_power_w[
                active_tube.next_step].copy()
            incumbent_comm = active_tube.communication_power_w[
                active_tube.next_step].copy()
            baseline_pd = _realized_pd(
                actual_coefficient, selected, incumbent_sensing, p_fa)
            tube_controller = None
            tube_fast_accepted = False
            tube_fast_route = "atomic_transaction_frozen"
            candidate_selected = selected.copy()
            candidate_role = role.copy()
            candidate_sensing = incumbent_sensing.copy()
            candidate_comm = incumbent_comm.copy()
            decision = None
            causal_available = True
            reason = (
                "executed:atomic_certified_joint_plan_geometry_tube")
            bits = 0
            latency = 0.0
            energy = 0.0
            route = (
                "slow_geometry_execution"
                if active_tube.execution_mode == "repair"
                else "slow_geometry_baseline_execution")
            deployment_certificate = False
            tube_executed = True
            tube_step = int(active_tube.next_step)
            if active_tube.proof_available:
                tube_lower = active_tube.coefficient_lower[tube_step]
                tube_upper = active_tube.coefficient_upper[tube_step]
                support = active_tube.certified_support.copy()
                tube_score = simultaneous_log_envelope_score(
                    tube_lower, tube_upper, actual_coefficient, support)
                episode_scores[seed] = max(
                    episode_scores[seed], float(tube_score.joint))
                envelope_covered = bool(tube_score.joint <= 1.0e-12)
                lower_failure = bool(np.any(
                    tube_lower[support]
                    > actual_coefficient[support] + 1.0e-12))
                upper_failure = bool(np.any(
                    tube_upper[support] + 1.0e-12
                    < actual_coefficient[support]))
            else:
                envelope_covered = True
                lower_failure = False
                upper_failure = False
        else:
            selected = deployed_selected[seed].copy()
            role = deployed_role[seed].copy()
            support = (
                current_reciprocal_support
                | (selected & previous_observed[seed])
            )
            decision = certified_causal_hierarchical_isac_control(
                previous_coefficient[seed],
                previous_observed[seed],
                previous_state[seed],
                current_state,
                selected,
                role,
                current_state.uav_position_m,
                comm,
                sensing,
                support,
                current_frame=frame,
                elapsed_frames=frame - previous_frame[seed],
                max_age_frames=int(max_age_frames),
                calibration=calibration,
                communication_model=communication_model,
                control_period_s=float(cfg.scenario.dt),
                p_fa=p_fa,
                controller_config=controller_config,
                carrier_hz=float(cfg.otfs.fc),
                delta_f_hz=float(cfg.otfs.delta_f),
                symbol_period_s=float(cfg.otfs.T_sym),
                delay_bins=int(cfg.otfs.M),
                doppler_bins=int(cfg.otfs.N),
                dd_support_threshold=float(cfg.detection.g_min),
                tracking_enabled=False,
                use_swerling=False,
                prior_cache=target_cache.get(seed),
            )
            target_cache[seed] = decision.envelope.target_invariant_cache
            causal_available = bool(decision.envelope.available)
            reason = decision.reason
            candidate_selected = decision.selected
            candidate_role = decision.role
            candidate_sensing = decision.sensing_power_w
            candidate_comm = decision.communication_power_w
            bits = int(decision.total_over_air_bits)
            latency = float(decision.total_protocol_latency_s)
            energy = float(decision.total_energy_j)
            route = (
                "unavailable" if decision.controller is None
                else decision.controller.route.value)
            deployment_certificate = bool(decision.deployment_certificate)
            baseline_pd = _realized_pd(
                actual_coefficient, selected, sensing, p_fa)
            mask = support
            if causal_available:
                score = simultaneous_log_envelope_score(
                    decision.envelope.lower,
                    decision.envelope.upper,
                    actual_coefficient,
                    mask,
                )
                episode_scores[seed] = max(
                    episode_scores[seed], float(score.joint))
                envelope_covered = bool(score.joint <= 1.0e-12)
                lower_failure = bool(np.any(
                    decision.envelope.lower[mask]
                    > actual_coefficient[mask] + 1.0e-12))
                upper_failure = bool(np.any(
                    decision.envelope.upper[mask] + 1.0e-12
                    < actual_coefficient[mask]))
            else:
                envelope_covered = True
                lower_failure = False
                upper_failure = False
            if decision.accepted:
                deployed_selected[seed] = candidate_selected.copy()
                deployed_role[seed] = candidate_role.copy()
            tube_executed = False
            tube_step = -1
            tube_fast_accepted = False
            tube_fast_route = "none"
            tube_controller = None

        geometry_horizon = certificate_horizon
        geometry_attempted = bool(
            enable_geometry
            and
            active_tube is None
            and
            decision is not None
            and decision.controller is not None
            and route == "slow_geometry"
            and causal_available
            and frame + geometry_horizon <= last_frame_by_seed[seed]
        )
        geometry = None
        geometry_accepted = False
        geometry_reason = "not_routed"
        geometry_candidate_count = 0
        geometry_verified_candidate_count = 0
        geometry_feasible_candidate_count = 0
        geometry_target_no_harm_failure_count = 0
        geometry_window_worst_failure_count = 0
        geometry_fast_opportunity_failure_count = 0
        geometry_strict_improvement_failure_count = 0
        geometry_transport_failure_count = 0
        geometry_energy_failure_count = 0
        geometry_best_verified_first_step_coupled_gain = float("nan")
        geometry_best_verified_minimum_coupled_gain = float("nan")
        geometry_best_verified_window_coupled_gain = float("nan")
        geometry_best_verified_window_worst_pd_gain = float("nan")
        geometry_certified_improvement = 0.0
        geometry_realized_improvement = 0.0
        geometry_target_no_harm = True
        geometry_window_target_no_harm = True
        geometry_window_worst_pd_gain = 0.0
        geometry_window_global_worst_pd_gain = 0.0
        geometry_window_global_worst_no_harm = True
        geometry_terminal_position_error_m = 0.0
        geometry_terminal_velocity_error_mps = 0.0
        geometry_envelope_covered = True
        geometry_coefficient_lower_failure = False
        geometry_coefficient_upper_failure = False
        geometry_displacement_norm_m = 0.0
        geometry_mover = -1
        geometry_mover_count = 0
        geometry_flight_energy_j = 0.0
        geometry_incremental_flight_energy_j = 0.0
        geometry_minimum_separation_m = float("inf")
        geometry_transport_bits = 0
        geometry_transport_latency_s = 0.0
        geometry_transport_energy_j = 0.0
        geometry_transport_feasible = True
        geometry_combined_bits = 0
        geometry_combined_latency_s = 0.0
        geometry_combined_energy_j = 0.0
        geometry_joint_plan_source = None
        geometry_joint_plan_digest = None
        geometry_proposal_source = None
        if geometry_attempted:
            if bool(persistent_geometry):
                current_frame_energy = _flight_energy_by_uav(
                    mobility_action,
                    dt_s=float(cfg.scenario.dt),
                    static_power_w=float(cfg.uav.P_fly_static),
                    quadratic_power_coeff=float(cfg.uav.P_fly_coeff),
                ) + np.sum(sensing, axis=1) * float(cfg.scenario.dt)
            else:
                current_frame_energy = np.full(K, (
                    float(cfg.uav.P_fly_static)
                    + float(cfg.uav.P_fly_coeff) * float(cfg.uav.v_max) ** 2
                    + 1.0
                ) * float(cfg.scenario.dt))
            conservative_battery = np.maximum(
                observed_battery_j - current_frame_energy - float(energy),
                0.0,
            )
            remaining_period = max(
                float(cfg.scenario.dt) - float(latency), 0.0)
            learned_proposal_plan = None
            if bool(causal_joint_plan):
                if causal_planner is None or plan_role == "proposal":
                    commitment = commit_zero_order_hold_joint_plan(
                        mobility_action,
                        comm,
                        sensing,
                        decision_frame=frame,
                        horizon_steps=geometry_horizon,
                    )
                    if causal_planner is not None:
                        learned_proposal_plan = causal_planner.predict(
                            current_state.uav_position_m,
                            target_outcome[:, :2],
                            mobility_action,
                            comm,
                            sensing,
                            decision_frame=frame,
                        )
                        learned_proposal_plan = _quantize_mobility_plan(
                            learned_proposal_plan,
                            maximum_displacement_m=(
                                float(cfg.uav.v_max)
                                * float(cfg.scenario.dt)),
                            bits_per_axis=GeometryRepairWireLayout(
                            K, Q).displacement_bits_per_axis,
                        )
                else:
                    planned_movement = causal_planner.predict(
                        current_state.uav_position_m,
                        target_outcome[:, :2],
                        mobility_action,
                        comm,
                        sensing,
                        decision_frame=frame,
                    )
                    commitment = commit_movement_plan_with_held_rf(
                        planned_movement,
                        comm,
                        sensing,
                        decision_frame=frame,
                        source=(
                            "equivariant_residual_movement_held_current_rf"),
                    )
                baseline_movement_plan = _quantize_mobility_plan(
                    commitment.movement_plan_m,
                    maximum_displacement_m=(
                        float(cfg.uav.v_max) * float(cfg.scenario.dt)),
                    bits_per_axis=GeometryRepairWireLayout(
                        K, Q).displacement_bits_per_axis,
                )
                baseline_communication_plan, baseline_sensing_plan = (
                    _quantize_rf_plan(
                        commitment.communication_power_plan_w,
                        commitment.sensing_power_plan_w,
                    )
                )
                wire_commitment = CausalJointPlanCommitment(
                    decision_frame=frame,
                    movement_plan_m=baseline_movement_plan,
                    communication_power_plan_w=baseline_communication_plan,
                    sensing_power_plan_w=baseline_sensing_plan,
                    source=(
                        "wire_quantized_current_actor_zero_order_hold"
                        if causal_planner is None or plan_role == "proposal"
                        else "wire_quantized_equivariant_residual_movement_"
                        "held_current_rf"
                    ),
                )
                geometry_joint_plan_source = wire_commitment.source
                geometry_joint_plan_digest = wire_commitment.digest
            else:
                next_row = row_by_seed_frame[(seed, frame + 1)]
                next_recorded_displacement = np.asarray(
                    data["delta_p"][next_row], dtype=np.float64)
                baseline_movement_plan = np.zeros(
                    (geometry_horizon, K, 2), dtype=np.float64)
                baseline_movement_plan[0] = next_recorded_displacement
                baseline_communication_plan = np.stack([
                    _recorded_power(
                        data,
                        row_by_seed_frame[(seed, frame + offset)],
                    )[0]
                    for offset in range(1, geometry_horizon + 1)
                ])
                baseline_sensing_plan = np.stack([
                    _recorded_power(
                        data,
                        row_by_seed_frame[(seed, frame + offset)],
                    )[1]
                    for offset in range(1, geometry_horizon + 1)
                ])
            if bool(announced_mobility_horizon):
                baseline_movement_plan = np.stack([
                    np.asarray(
                        data["delta_p"][row_by_seed_frame[
                            (seed, frame + offset)]],
                        dtype=np.float64,
                    )
                    for offset in range(1, geometry_horizon + 1)
                ])
                baseline_movement_plan *= reserve_fraction
                baseline_movement_plan = _quantize_mobility_plan(
                    baseline_movement_plan,
                    maximum_displacement_m=(
                        float(cfg.uav.v_max) * float(cfg.scenario.dt)),
                    bits_per_axis=GeometryRepairWireLayout(
                        K, Q).displacement_bits_per_axis,
                )
                geometry_joint_plan_source = "recorded_future_actor_surrogate"
            baseline_communication_plan, baseline_sensing_plan = (
                _quantize_rf_plan(
                    baseline_communication_plan,
                    baseline_sensing_plan,
                )
            )
            geometry = certified_trust_region_geometry_repair(
                decision.envelope.lower,
                decision.envelope.upper,
                current_state,
                decision.envelope.target_invariant_cache,
                support,
                candidate_selected,
                decision.controller.owner,
                candidate_sensing,
                candidate_comm,
                conservative_battery,
                communication_model=communication_model,
                control_period_s=remaining_period,
                p_fa=p_fa,
                qos_floor=float(qos_floor),
                dt_s=float(cfg.scenario.dt),
                max_speed_mps=float(cfg.uav.v_max),
                area_size_m=tuple(
                    float(value) for value in cfg.scenario.region_size),
                safe_separation_m=float(cfg.uav.d_safe),
                static_flight_power_w=float(cfg.uav.P_fly_static),
                quadratic_flight_power_coeff=float(cfg.uav.P_fly_coeff),
                carrier_hz=float(cfg.otfs.fc),
                delta_f_hz=float(cfg.otfs.delta_f),
                symbol_period_s=float(cfg.otfs.T_sym),
                delay_bins=int(cfg.otfs.M),
                doppler_bins=int(cfg.otfs.N),
                covariance_radius=float(calibration.dd_covariance_radius),
                dd_support_threshold=float(cfg.detection.g_min),
                dd_additive_margin=float(calibration.dd_additive_margin),
                residual_log_margin=float(calibration.current_log_margin),
                ground_communication_enabled=bool(
                    cfg.marl.ground_communication_enabled),
                config=CertifiedGeometryRepairConfig(
                    certificate_horizon_steps=geometry_horizon,
                    verification_top_m=int(geometry_verification_top_m),
                    learned_verification_top_m=(
                        min(4, 8 - int(geometry_verification_top_m))
                        if learned_proposal_plan is not None else 0),
                    weak_target_count=Q,
                    snr_margin_db=float(snr_margin_db),
                    latency_margin_s=float(latency_margin_s),
                ),
                action_origin_state=current_state,
                baseline_displacement_m=baseline_movement_plan[0],
                baseline_movement_plan_m=baseline_movement_plan,
                baseline_sensing_plan_w=baseline_sensing_plan,
                baseline_communication_plan_w=baseline_communication_plan,
                proposal_movement_plan_m=learned_proposal_plan,
                joint_plan_digest=(geometry_joint_plan_digest or ""),
            )
            geometry_accepted = bool(geometry.accepted)
            geometry_reason = geometry.reason
            geometry_candidate_count = int(geometry.candidate_count)
            geometry_verified_candidate_count = int(
                geometry.verified_candidate_count)
            geometry_feasible_candidate_count = int(
                geometry.feasible_candidate_count)
            geometry_target_no_harm_failure_count = int(
                geometry.target_no_harm_failure_count)
            geometry_window_worst_failure_count = int(
                geometry.window_worst_failure_count)
            geometry_fast_opportunity_failure_count = int(
                geometry.fast_opportunity_failure_count)
            geometry_strict_improvement_failure_count = int(
                geometry.strict_improvement_failure_count)
            geometry_transport_failure_count = int(
                geometry.transport_failure_count)
            geometry_energy_failure_count = int(
                geometry.energy_failure_count)
            geometry_best_verified_first_step_coupled_gain = float(
                geometry.best_verified_first_step_coupled_gain)
            geometry_best_verified_minimum_coupled_gain = float(
                geometry.best_verified_minimum_coupled_gain)
            geometry_best_verified_window_coupled_gain = float(
                geometry.best_verified_window_coupled_gain)
            geometry_best_verified_window_worst_pd_gain = float(
                geometry.best_verified_window_worst_pd_gain)
            geometry_certified_improvement = float(
                geometry.certified_worst_pd_improvement
                if geometry.accepted else 0.0)
            if geometry.accepted and geometry.best_candidate is not None:
                best_geometry = geometry.best_candidate
                geometry_proposal_source = best_geometry.proposal_source
                baseline_states = propagate_owner_local_horizon(
                    current_state,
                    baseline_movement_plan,
                    dt_s=float(cfg.scenario.dt),
                    max_speed_mps=float(cfg.uav.v_max),
                    area_size_m=tuple(
                        float(value) for value in cfg.scenario.region_size),
                    advance_targets=False,
                )[1:]
                movement_plan = best_geometry.movement_plan_m.copy()
                moved_states = propagate_owner_local_horizon(
                    current_state,
                    movement_plan,
                    dt_s=float(cfg.scenario.dt),
                    max_speed_mps=float(cfg.uav.v_max),
                    area_size_m=tuple(
                        float(value) for value in cfg.scenario.region_size),
                    advance_targets=False,
                )[1:]
                baseline_coefficient = np.stack([
                    _counterfactual_coefficient(state, target_outcome, cfg)
                    for state in baseline_states
                ])
                moved_coefficient = np.stack([
                    _counterfactual_coefficient(state, target_outcome, cfg)
                    for state in moved_states
                ])
                baseline_pd_horizon = np.stack([
                    _realized_pd(
                        coefficient, candidate_selected,
                        baseline_sensing_plan[step], p_fa)
                    for step, coefficient in enumerate(baseline_coefficient)
                ])
                moved_pd = np.stack([
                    _realized_pd(
                        coefficient, candidate_selected,
                        baseline_sensing_plan[step], p_fa)
                    for step, coefficient in enumerate(moved_coefficient)
                ])
                geometry_realized_improvement = float(
                    np.min(moved_pd[0])
                    - np.min(baseline_pd_horizon[0]))
                geometry_target_no_harm = bool(np.all(
                    moved_pd + 1.0e-9
                    >= np.minimum(
                        baseline_pd_horizon, float(qos_floor))))
                realized_window_gain = np.sum(
                    np.minimum(moved_pd, float(qos_floor))
                    - np.minimum(
                        baseline_pd_horizon, float(qos_floor)),
                    axis=0,
                )
                geometry_window_worst_pd_gain = float(np.min(
                    realized_window_gain))
                geometry_window_target_no_harm = bool(np.all(
                    realized_window_gain + 1.0e-9 >= 0.0))
                geometry_window_global_worst_pd_gain = float(np.sum(
                    np.min(moved_pd, axis=1)
                    - np.min(baseline_pd_horizon, axis=1)))
                geometry_window_global_worst_no_harm = bool(
                    geometry_window_global_worst_pd_gain + 1.0e-9 >= 0.0)
                geometry_terminal_position_error_m = float(np.max(
                    np.linalg.norm(
                        moved_states[-1].uav_position_m
                        - baseline_states[-1].uav_position_m,
                        axis=1,
                    )))
                geometry_terminal_velocity_error_mps = float(np.max(
                    np.linalg.norm(
                        moved_states[-1].uav_velocity_mps
                        - baseline_states[-1].uav_velocity_mps,
                        axis=1,
                    )))
                geometry_mask = support
                geometry_scores = [
                    simultaneous_log_envelope_score(
                        best_geometry.physical_bounds.lower[step],
                        best_geometry.physical_bounds.upper[step],
                        moved_coefficient[step],
                        geometry_mask,
                    )
                    for step in range(geometry_horizon)
                ]
                geometry_envelope_covered = bool(
                    max(score.joint for score in geometry_scores) <= 1.0e-12)
                geometry_coefficient_lower_failure = bool(any(np.any(
                    best_geometry.physical_bounds.lower[step][geometry_mask]
                    > moved_coefficient[step][geometry_mask] + 1.0e-12)
                    for step in range(geometry_horizon)))
                geometry_coefficient_upper_failure = bool(any(np.any(
                    best_geometry.physical_bounds.upper[step][geometry_mask]
                    + 1.0e-12
                    < moved_coefficient[step][geometry_mask])
                    for step in range(geometry_horizon)))
                geometry_displacement_norm_m = float(np.max(np.linalg.norm(
                    best_geometry.displacement_m, axis=1)))
                geometry_mover = int(best_geometry.mover)
                geometry_mover_count = len(best_geometry.movers)
                geometry_flight_energy_j = float(
                    best_geometry.flight_energy_j)
                geometry_incremental_flight_energy_j = float(
                    best_geometry.incremental_flight_energy_j)
                geometry_minimum_separation_m = float(
                    best_geometry.minimum_separation_m)
                geometry_transport_bits = int(
                    best_geometry.transport.total_over_air_bits)
                geometry_transport_latency_s = float(
                    best_geometry.transport.total_protocol_latency_s)
                geometry_transport_energy_j = float(
                    best_geometry.transport.total_energy_j)
                geometry_transport_feasible = bool(
                    best_geometry.transport.feasible)
                geometry_combined_bits = int(
                    bits + geometry_transport_bits)
                geometry_combined_latency_s = float(
                    latency + geometry_transport_latency_s)
                geometry_combined_energy_j = float(
                    energy + geometry_transport_energy_j)
                if bool(persistent_geometry):
                    geometry_tube[seed] = start_certified_geometry_tube(
                        geometry,
                        decision_frame=frame,
                        selected=candidate_selected,
                        role=candidate_role,
                        certified_support=support,
                        origin_position_m=current_state.uav_position_m,
                        execution_mode=execution_mode,
                    )
                    deployed_selected[seed] = candidate_selected.copy()
                    deployed_role[seed] = candidate_role.copy()

        candidate_pd = _realized_pd(
            actual_coefficient,
            candidate_selected,
            candidate_sensing,
            p_fa,
        )
        tube_step_realized_worst_pd = float("nan")
        tube_step_certified_lower_worst_pd = float("nan")
        tube_step_baseline_upper_worst_pd = float("nan")
        tube_step_certified_first_step_worst_gain = float("nan")
        tube_step_certified_first_step_worst_gain_threshold = float("nan")
        tube_step_independent_bound_target_no_harm = True
        tube_step_certified_window_target_no_harm = True
        tube_step_certified_window_worst_no_harm = True
        tube_step_certified_first_step_strict_improvement = True
        tube_step_realized_paired_step_target_no_harm = True
        tube_step_realized_paired_first_step_strict_improvement = True
        tube_step_decision_frame = -1
        tube_step_clipped_pd_gain = np.zeros(Q, dtype=np.float64)
        tube_step_global_worst_pd_gain = 0.0
        tube_step_repair_worst_pd = float("nan")
        tube_step_baseline_worst_pd = float("nan")
        tube_step_active_branch_coefficient_match_error = 0.0
        if active_tube is not None:
            tube_step_decision_frame = int(active_tube.decision_frame)
            tube_step_realized_worst_pd = float(np.min(candidate_pd))
            if active_tube.proof_available:
                tube_step_certified_first_step_worst_gain = float(
                    active_tube.certified_first_step_worst_gain)
                tube_step_certified_first_step_worst_gain_threshold = float(
                    active_tube.certified_first_step_worst_gain_threshold)
                repair_coefficient = _tube_branch_coefficient(
                    current_state,
                    active_tube,
                    tube_step,
                    target_outcome,
                    cfg,
                    repair=True,
                )
                baseline_coefficient = _tube_branch_coefficient(
                    current_state,
                    active_tube,
                    tube_step,
                    target_outcome,
                    cfg,
                    repair=False,
                )
                repair_pd = _realized_pd(
                    repair_coefficient,
                    candidate_selected,
                    candidate_sensing,
                    p_fa,
                )
                paired_baseline_pd = _realized_pd(
                    baseline_coefficient,
                    candidate_selected,
                    candidate_sensing,
                    p_fa,
                )
                active_branch_coefficient = (
                    repair_coefficient
                    if active_tube.execution_mode == "repair"
                    else baseline_coefficient)
                tube_step_active_branch_coefficient_match_error = float(
                    np.max(np.abs(
                        actual_coefficient - active_branch_coefficient)))
                tube_step_repair_worst_pd = float(np.min(repair_pd))
                tube_step_baseline_worst_pd = float(
                    np.min(paired_baseline_pd))
                tube_step_certified_lower_worst_pd = float(np.min(
                    active_tube.candidate_lower_pd[tube_step]))
                tube_step_baseline_upper_worst_pd = float(np.min(
                    active_tube.baseline_upper_pd[tube_step]))
                tube_step_independent_bound_target_no_harm = bool(np.all(
                    repair_pd + 1.0e-9 >= np.minimum(
                        active_tube.baseline_upper_pd[tube_step],
                        float(qos_floor),
                    )
                ))
                tube_step_certified_window_target_no_harm = bool(np.all(
                    active_tube.certified_window_target_gain + 1.0e-9 >= 0.0
                ))
                tube_step_certified_window_worst_no_harm = bool(
                    active_tube.certified_window_worst_gain + 1.0e-9 >= 0.0
                )
                tube_step_certified_first_step_strict_improvement = bool(
                    active_tube.certified_first_step_worst_gain
                    > active_tube.certified_first_step_worst_gain_threshold)
                tube_step_clipped_pd_gain = (
                    np.minimum(repair_pd, float(qos_floor))
                    - np.minimum(paired_baseline_pd, float(qos_floor))
                )
                tube_step_global_worst_pd_gain = float(
                    np.min(repair_pd) - np.min(paired_baseline_pd))
                tube_step_realized_paired_step_target_no_harm = bool(np.all(
                    repair_pd + 1.0e-9
                    >= np.minimum(paired_baseline_pd, float(qos_floor))
                ))
                tube_step_realized_paired_first_step_strict_improvement = bool(
                    tube_step != 0
                    or tube_step_global_worst_pd_gain > 1.0e-9
                )
        accepted = bool(
            (decision is not None and decision.accepted)
            or tube_fast_accepted)
        target_no_harm = bool(
            not accepted or np.all(
                candidate_pd + 1.0e-9
                >= np.minimum(baseline_pd, float(qos_floor))))
        protocol_feasible = bool(
            (
                decision is None
                or decision.controller is None
                or decision.controller.protocol_resources is None
                or decision.controller.protocol_resources.feasible
            )
            and (
                tube_controller is None
                or tube_controller.protocol_resources is None
                or tube_controller.protocol_resources.feasible
            ))
        rows.append({
            "seed": seed,
            "frame": frame,
            "causal_available": causal_available,
            "accepted": accepted,
            "reason": reason,
            "route": route,
            "baseline_worst_pd": float(np.min(baseline_pd)),
            "candidate_worst_pd": float(np.min(candidate_pd)),
            "qos_feasible": bool(
                np.min(candidate_pd) >= float(qos_floor)),
            "target_no_harm": target_no_harm,
            "envelope_covered": envelope_covered,
            "coefficient_lower_failure": lower_failure,
            "coefficient_upper_failure": upper_failure,
            "protocol_feasible": protocol_feasible,
            "total_over_air_bits": bits,
            "total_protocol_latency_s": latency,
            "total_energy_j": energy,
            "communication_power_balance_error_w": float(np.max(np.abs(
                candidate_comm + np.sum(candidate_sensing, axis=1) - 1.0))),
            "recorded_power_balance_error_w": float(power_error),
            "deployment_certificate": deployment_certificate,
            "geometry_tube_executed": tube_executed,
            "geometry_tube_step": tube_step,
            "geometry_tube_decision_frame": tube_step_decision_frame,
            "geometry_tube_joint_plan_digest": (
                active_tube.joint_plan_digest
                if active_tube is not None else None),
            "geometry_tube_proof_available": (
                bool(active_tube.proof_available)
                if active_tube is not None else False),
            "geometry_tube_fast_route": tube_fast_route,
            "geometry_tube_fast_accepted": tube_fast_accepted,
            "geometry_tube_realized_worst_pd": tube_step_realized_worst_pd,
            "geometry_tube_certified_lower_worst_pd": (
                tube_step_certified_lower_worst_pd),
            "geometry_tube_baseline_upper_worst_pd": (
                tube_step_baseline_upper_worst_pd),
            "geometry_tube_certified_first_step_worst_gain": (
                tube_step_certified_first_step_worst_gain),
            "geometry_tube_certified_first_step_worst_gain_threshold": (
                tube_step_certified_first_step_worst_gain_threshold),
            "geometry_tube_independent_bound_step_target_no_harm": (
                tube_step_independent_bound_target_no_harm),
            "geometry_tube_certified_window_target_no_harm": (
                tube_step_certified_window_target_no_harm),
            "geometry_tube_certified_window_worst_no_harm": (
                tube_step_certified_window_worst_no_harm),
            "geometry_tube_certified_first_step_strict_improvement": (
                tube_step_certified_first_step_strict_improvement),
            "geometry_tube_realized_paired_step_target_no_harm": (
                tube_step_realized_paired_step_target_no_harm),
            "geometry_tube_realized_paired_first_step_strict_improvement": (
                tube_step_realized_paired_first_step_strict_improvement),
            "geometry_tube_clipped_pd_gain_by_target": [
                float(value) for value in tube_step_clipped_pd_gain],
            "geometry_tube_global_worst_pd_gain": (
                tube_step_global_worst_pd_gain),
            "geometry_tube_repair_worst_pd": tube_step_repair_worst_pd,
            "geometry_tube_paired_baseline_worst_pd": (
                tube_step_baseline_worst_pd),
            "geometry_tube_active_branch_coefficient_match_error": (
                tube_step_active_branch_coefficient_match_error),
            "geometry_shadow_attempted": geometry_attempted,
            "geometry_shadow_accepted": geometry_accepted,
            "geometry_shadow_reason": geometry_reason,
            "geometry_shadow_candidate_count": geometry_candidate_count,
            "geometry_shadow_verified_candidate_count": (
                geometry_verified_candidate_count),
            "geometry_shadow_feasible_candidate_count": (
                geometry_feasible_candidate_count),
            "geometry_shadow_target_no_harm_failure_count": (
                geometry_target_no_harm_failure_count),
            "geometry_shadow_window_worst_failure_count": (
                geometry_window_worst_failure_count),
            "geometry_shadow_fast_opportunity_failure_count": (
                geometry_fast_opportunity_failure_count),
            "geometry_shadow_strict_improvement_failure_count": (
                geometry_strict_improvement_failure_count),
            "geometry_shadow_transport_failure_count": (
                geometry_transport_failure_count),
            "geometry_shadow_energy_failure_count": (
                geometry_energy_failure_count),
            "geometry_shadow_best_verified_first_step_coupled_gain": (
                geometry_best_verified_first_step_coupled_gain),
            "geometry_shadow_best_verified_minimum_coupled_gain": (
                geometry_best_verified_minimum_coupled_gain),
            "geometry_shadow_best_verified_window_coupled_gain": (
                geometry_best_verified_window_coupled_gain),
            "geometry_shadow_best_verified_window_worst_pd_gain": (
                geometry_best_verified_window_worst_pd_gain),
            "geometry_shadow_certified_worst_pd_improvement": (
                geometry_certified_improvement),
            "geometry_shadow_realized_worst_pd_improvement": (
                geometry_realized_improvement),
            "geometry_shadow_target_no_harm": geometry_target_no_harm,
            "geometry_shadow_window_target_no_harm": (
                geometry_window_target_no_harm),
            "geometry_shadow_window_worst_pd_gain": (
                geometry_window_worst_pd_gain),
            "geometry_shadow_window_global_worst_pd_gain": (
                geometry_window_global_worst_pd_gain),
            "geometry_shadow_window_global_worst_no_harm": (
                geometry_window_global_worst_no_harm),
            "geometry_shadow_terminal_position_error_m": (
                geometry_terminal_position_error_m),
            "geometry_shadow_terminal_velocity_error_mps": (
                geometry_terminal_velocity_error_mps),
            "geometry_shadow_envelope_covered": geometry_envelope_covered,
            "geometry_shadow_coefficient_lower_failure": (
                geometry_coefficient_lower_failure),
            "geometry_shadow_coefficient_upper_failure": (
                geometry_coefficient_upper_failure),
            "geometry_shadow_displacement_norm_m": (
                geometry_displacement_norm_m),
            "geometry_shadow_mover": geometry_mover,
            "geometry_shadow_mover_count": geometry_mover_count,
            "geometry_shadow_flight_energy_j": geometry_flight_energy_j,
            "geometry_shadow_incremental_flight_energy_j": (
                geometry_incremental_flight_energy_j),
            "geometry_shadow_minimum_separation_m": (
                geometry_minimum_separation_m),
            "geometry_shadow_transport_bits": geometry_transport_bits,
            "geometry_shadow_transport_latency_s": (
                geometry_transport_latency_s),
            "geometry_shadow_transport_energy_j": geometry_transport_energy_j,
            "geometry_shadow_transport_feasible": (
                geometry_transport_feasible),
            "geometry_shadow_combined_bits": geometry_combined_bits,
            "geometry_shadow_combined_latency_s": (
                geometry_combined_latency_s),
            "geometry_shadow_combined_energy_j": geometry_combined_energy_j,
            "geometry_joint_plan_source": geometry_joint_plan_source,
            "geometry_joint_plan_digest": geometry_joint_plan_digest,
            "geometry_proposal_source": geometry_proposal_source,
        })
        previous_coefficient[seed] = actual_coefficient.copy()
        previous_observed[seed] = (
            candidate_selected
            & (candidate_sensing[:, None, :] > 1.0e-12)
        )
        previous_state[seed] = current_state
        previous_frame[seed] = frame
        if bool(persistent_geometry):
            persistent_position[seed] = current_state.uav_position_m.copy()
            persistent_velocity[seed] = current_state.uav_velocity_mps.copy()
            frame_flight_energy = (
                active_tube.escrow_flight_energy_plan_j[tube_step].copy()
                if active_tube is not None
                else _flight_energy_by_uav(
                    mobility_action,
                    dt_s=float(cfg.scenario.dt),
                    static_power_w=float(cfg.uav.P_fly_static),
                    quadratic_power_coeff=float(cfg.uav.P_fly_coeff),
                )
            )
            sensing_energy = (
                np.sum(candidate_sensing, axis=1) * float(cfg.scenario.dt))
            protocol_energy_by_uav = np.zeros(K, dtype=np.float64)
            if geometry_accepted and geometry is not None:
                protocol_energy_by_uav += np.asarray(
                    geometry.best_candidate.transport.per_uav_energy_j,
                    dtype=np.float64,
                )
            persistent_battery[seed] = np.maximum(
                persistent_battery[seed]
                - frame_flight_energy
                - sensing_energy
                - float(energy)
                - protocol_energy_by_uav,
                0.0,
            )
            if active_tube is not None:
                token_inbox[seed], common_token_energy = (
                    token_inbox[seed].advance_common_geometry_deliveries(
                        np.asarray(
                            data["outgoing_message"][row], dtype=np.float64),
                        np.asarray(
                            data["outgoing_rate"][row], dtype=np.int64),
                        np.asarray(
                            data["outgoing_token_mask"][row], dtype=np.float64),
                        active_tube.repair_position_plan_m[tube_step],
                        active_tube.baseline_position_plan_m[tube_step],
                        candidate_comm,
                        communication_model,
                    )
                )
                persistent_battery[seed] = np.maximum(
                    persistent_battery[seed] - common_token_energy, 0.0)
            else:
                token_inbox[seed], token_stats = token_inbox[seed].advance(
                    np.asarray(
                        data["outgoing_message"][row], dtype=np.float64),
                    np.asarray(data["outgoing_rate"][row], dtype=np.int64),
                    np.asarray(
                        data["outgoing_token_mask"][row], dtype=np.float64),
                    current_state.uav_position_m,
                    candidate_comm,
                    communication_model,
                )
                persistent_battery[seed] = np.maximum(
                    persistent_battery[seed]
                    - np.asarray([
                        token_stats.per_sender_energy_j.get(agent, 0.0)
                        for agent in range(K)
                    ], dtype=np.float64),
                    0.0,
                )
            if active_tube is not None:
                next_tube = active_tube.advance(frame)
                if next_tube is None:
                    geometry_tube.pop(seed, None)
                else:
                    geometry_tube[seed] = next_tube
            rows[-1]["post_event_runtime_state_digest"] = (
                _runtime_state_digest(
                    position_m=persistent_position[seed],
                    velocity_mps=persistent_velocity[seed],
                    battery_j=persistent_battery[seed],
                    inbox=token_inbox[seed],
                    selected=deployed_selected[seed],
                    role=deployed_role[seed],
                    previous_coefficient=previous_coefficient[seed],
                    previous_observed=previous_observed[seed],
                    previous_state=previous_state[seed],
                    target_cache=target_cache.get(seed),
                    tube=geometry_tube.get(seed),
                )
            )

    eligible = [row for row in rows if bool(row["causal_available"])]
    accepted_rows = [row for row in rows if bool(row["accepted"])]
    scores = np.asarray(list(episode_scores.values()), dtype=np.float64)
    episode_policy_mean = []
    episode_policy_min = []
    episode_baseline_mean = []
    for seed in seed_order:
        episode_rows = [row for row in rows if int(row["seed"]) == seed]
        episode_policy_mean.append(float(np.mean([
            float(row["candidate_worst_pd"]) for row in episode_rows
        ])))
        episode_policy_min.append(float(np.min([
            float(row["candidate_worst_pd"]) for row in episode_rows
        ])))
        episode_baseline_mean.append(float(np.mean([
            float(row["baseline_worst_pd"]) for row in episode_rows
        ])))
    route_counts = {
        route: int(sum(row["route"] == route for row in rows))
        for route in sorted({str(row["route"]) for row in rows})
    }
    route_accept_counts = {
        route: int(sum(
            row["route"] == route and bool(row["accepted"])
            for row in rows
        )) for route in route_counts
    }
    geometry_attempts = [
        row for row in rows if bool(row["geometry_shadow_attempted"])]
    geometry_accepts = [
        row for row in rows if bool(row["geometry_shadow_accepted"])]
    tube_rows = [
        row for row in rows if bool(row["geometry_tube_executed"])]
    proof_tube_rows = [
        row for row in tube_rows
        if bool(row["geometry_tube_proof_available"])]
    proof_tube_start_rows = [
        row for row in proof_tube_rows
        if int(row["geometry_tube_step"]) == 0]
    transaction_keys = {
        (int(row["seed"]), int(row["geometry_tube_decision_frame"]))
        for row in tube_rows
    }
    tube_window_gains: list[np.ndarray] = []
    tube_window_global_worst_gains: list[float] = []
    tube_keys = sorted({
        (int(row["seed"]), int(row["geometry_tube_decision_frame"]))
        for row in proof_tube_rows
    })
    for tube_seed, decision_frame in tube_keys:
        window_rows = [
            row for row in proof_tube_rows
            if int(row["seed"]) == tube_seed
            and int(row["geometry_tube_decision_frame"]) == decision_frame
        ]
        tube_window_gains.append(np.sum([
            np.asarray(
                row["geometry_tube_clipped_pd_gain_by_target"],
                dtype=np.float64,
            )
            for row in window_rows
        ], axis=0))
        tube_window_global_worst_gains.append(float(np.sum([
            float(row["geometry_tube_global_worst_pd_gain"])
            for row in window_rows
        ])))
    summary = {
        "schema_version": 5,
        "scope": (
            "persistent-structure causal D0.47 audit with lagged excited-edge "
            "feedback, current owner-local kinematics and "
            + (
                "persistent delayed certified geometry execution with frozen "
                "learned Token actions"
                if bool(persistent_geometry)
                else "post-decision geometry labels"
            )
        ),
        "causal_certificate": "conditional_on_warm_start_structure",
        "deployment_certificate": bool(
            calibration.system_certificate_ready
            and all(bool(row["deployment_certificate"]) for row in eligible)),
        "trace": str(trace_path),
        "config": str(config_path),
        "calibration": str(calibration_path),
        "calibration_alpha": float(calibration.alpha),
        "calibration_coverage_floor": float(calibration.coverage_floor),
        "calibration_episode_count": int(
            calibration.calibration_episode_count),
        "calibration_validation_seed_overlap_count": 0,
        "physical_design_matches_calibration": True,
        "persistent_geometry_enabled": bool(persistent_geometry),
        "geometry_repair_enabled": bool(enable_geometry),
        "geometry_execution_mode": execution_mode,
        "announced_mobility_horizon_enabled": bool(
            announced_mobility_horizon),
        "causal_joint_plan_enabled": bool(causal_joint_plan),
        "causal_plan_checkpoint": (
            str(causal_plan_checkpoint)
            if causal_plan_checkpoint is not None else None),
        "causal_plan_checkpoint_sha256": (
            _sha256_file(causal_plan_checkpoint)
            if causal_plan_checkpoint is not None else None),
        "causal_plan_architecture": (
            causal_planner.metadata.architecture
            if causal_planner is not None else "zero_order_hold"),
        "causal_plan_role": (
            plan_role if causal_planner is not None else None),
        "causal_plan_uniform_region_scaling_allowed": bool(
            allow_uniform_plan_region_scaling),
        "causal_plan_uniform_region_scaling_applied": (
            bool(causal_planner.uniform_region_scaling_applied)
            if causal_planner is not None else False),
        "causal_plan_source_region_size_m": (
            list(causal_planner.metadata.region_size_m)
            if causal_planner is not None else []),
        "causal_plan_runtime_region_size_m": (
            list(causal_planner.runtime_region_size_m)
            if causal_planner is not None else []),
        "causal_plan_uniform_region_scale": (
            float(causal_planner.uniform_region_scale)
            if causal_planner is not None else None),
        "causal_plan_parameter_count": (
            int(sum(parameter.numel() for parameter in
                    causal_planner.model.parameters()))
            if causal_planner is not None else 0),
        "causal_plan_training_episode_ids": (
            list(causal_planner.metadata.training_episode_ids)
            if causal_planner is not None else []),
        "causal_plan_selection_episode_ids": (
            list(causal_planner.metadata.selection_episode_ids)
            if causal_planner is not None else []),
        "causal_plan_development_validation_overlap_count": (
            0 if causal_planner is not None else None),
        "causal_plan_exact_invariants": _planner_exact_invariants(
            causal_planner),
        "runtime_state_digest_floating_mantissa_bits": 30,
        "mobility_reserve_fraction": reserve_fraction,
        "mobility_control_authority_reserve_fraction": float(
            1.0 - reserve_fraction),
        "announced_mobility_horizon_scope": (
            "recorded future Actor mobility and RF actions surrogate a "
            "required H-step local joint-plan output; diagnostic only until "
            "that output exists at runtime"
            if bool(announced_mobility_horizon) else None
        ),
        "causal_joint_plan_scope": (
            (
                "a frozen holdout-admitted set-equivariant residual head uses "
                "only current post-action own positions, mission-known static "
                "target positions, current movement/RF actions and decision "
                "phase; " + (
                    "it contributes only reserved learned candidates while "
                    "the digest-bound ZOH commitment and original analytic "
                    "Top-M remain unchanged"
                    if plan_role == "proposal"
                    else "RF is held, the complete learned plan is quantized "
                    "and digest-bound before geometry verification"
                ) + "; no future trace row enters the certificate"
                if causal_planner is not None
                else "current local Actor movement and RF actions are "
                "quantized once, repeated by zero-order hold, digest-bound, "
                "and committed before geometry verification; no future trace "
                "row enters the certificate"
            )
            if bool(causal_joint_plan) else None
        ),
        "geometry_decision_to_execution_delay_frames": (
            1 if bool(persistent_geometry) else None),
        "geometry_certificate_horizon_steps": min(
            max(
                3,
                int(calibration.horizon_steps)
                if geometry_horizon_steps is None
                else int(geometry_horizon_steps),
            ),
            8,
        ),
        "geometry_verification_top_m": int(
            geometry_verification_top_m),
        "geometry_execution_policy": (
            (
                "two-phase atomic commit: rejected prepare transactions leave "
                "execution unchanged; accepted transactions execute the "
                "complete repair tube from t+1 and return to the committed "
                "Actor terminal position and velocity"
                if execution_mode == "repair"
                else "execute the exact committed Actor baseline tube at the "
                "same certified acceptance epochs and with the same RF plan"
            )
            if bool(persistent_geometry) else None
        ),
        "persistent_feedback_scope": (
            "UAV position, velocity, battery and receiver-specific Token "
            "visibility are recurrent; target belief values and learned "
            "message/rate/mask actions remain on the frozen validation path"
            if bool(persistent_geometry) else None
        ),
        "warm_start_policy": (
            "first-event teacher structure used only as an external existing "
            "structure; causal claims are conditional on this warm start"
        ),
        "seed_count": len(seed_order),
        "event_count": len(rows),
        "eligible_event_count": len(eligible),
        "causal_availability_rate": float(
            len(eligible) / max(len(rows) - len(seed_order), 1)),
        "acceptance_rate_given_available": float(np.mean([
            bool(row["accepted"]) for row in eligible
        ]) if eligible else 0.0),
        "baseline_mean_worst": float(np.mean([
            float(row["baseline_worst_pd"]) for row in rows])),
        "policy_mean_worst": float(np.mean([
            float(row["candidate_worst_pd"]) for row in rows])),
        "baseline_qos_rate": float(np.mean([
            float(row["baseline_worst_pd"]) >= float(qos_floor)
            for row in rows])),
        "policy_qos_rate": float(np.mean([
            bool(row["qos_feasible"]) for row in rows])),
        "baseline_episode_mean_worst": float(np.mean(
            episode_baseline_mean)),
        "policy_episode_mean_worst": float(np.mean(episode_policy_mean)),
        "policy_worst_episode_mean": float(np.min(episode_policy_mean)),
        "policy_mean_episode_min": float(np.mean(episode_policy_min)),
        "policy_worst_episode_min": float(np.min(episode_policy_min)),
        "policy_event_lower_cvar_05": _lower_tail_cvar([
            float(row["candidate_worst_pd"]) for row in rows
        ], 0.05),
        "policy_episode_mean_lower_cvar_10": _lower_tail_cvar(
            episode_policy_mean, 0.10),
        "accepted_target_no_harm_rate": float(np.mean([
            bool(row["target_no_harm"]) for row in accepted_rows
        ]) if accepted_rows else 1.0),
        "accepted_target_no_harm_violation_count": int(sum(
            not bool(row["target_no_harm"]) for row in accepted_rows)),
        "envelope_event_coverage_rate": float(np.mean([
            bool(row["envelope_covered"]) for row in eligible
        ]) if eligible else 0.0),
        "episode_envelope_coverage_rate": float(np.mean(
            scores <= 1.0e-12)),
        "episode_score_max": float(np.max(scores, initial=0.0)),
        "coefficient_lower_failure_count": int(sum(
            bool(row["coefficient_lower_failure"]) for row in eligible)),
        "coefficient_upper_failure_count": int(sum(
            bool(row["coefficient_upper_failure"]) for row in eligible)),
        "all_accepted_protocol_feasible": bool(all(
            bool(row["protocol_feasible"]) for row in accepted_rows)),
        "mean_available_protocol_bits": float(np.mean([
            float(row["total_over_air_bits"]) for row in eligible
        ]) if eligible else 0.0),
        "max_available_protocol_latency_s": float(np.max([
            float(row["total_protocol_latency_s"]) for row in eligible
        ], initial=0.0)),
        "mean_available_protocol_energy_j": float(np.mean([
            float(row["total_energy_j"]) for row in eligible
        ]) if eligible else 0.0),
        "max_power_balance_error_w": float(np.max([
            float(row["communication_power_balance_error_w"])
            for row in rows
        ], initial=0.0)),
        "route_counts": route_counts,
        "route_accept_counts": route_accept_counts,
        "geometry_shadow_scope": (
            "finite-window static-target expected-detection certificate; "
            "movement is injected into the persistent physical branch"
            if bool(persistent_geometry)
            else "finite-window static-target expected-detection "
            "counterfactual; movement is not injected into the frozen "
            "validation trace"
        ),
        "geometry_sensing_safety_principle": (
            "for every target, the H-frame sum of min(P_D, QoS floor) is no "
            "lower than the Actor baseline; the H-frame sum of global "
            "worst-target P_D is no lower than the Actor baseline; the first "
            "frame global worst-target probability must improve strictly"
        ),
        "geometry_transport_principle": (
            "dynamic-mover coordinator; full-bandwidth TDMA signed state and "
            "owner-gradient gather; local Top-M verification; proof-carrying "
            "prepare broadcast; all-UAV vote gather; decision broadcast"
        ),
        "geometry_shadow_attempt_count": len(geometry_attempts),
        "geometry_shadow_accept_count": len(geometry_accepts),
        "geometry_shadow_acceptance_rate": float(
            len(geometry_accepts) / max(len(geometry_attempts), 1)),
        "geometry_shadow_verified_candidate_count": int(sum(
            int(row["geometry_shadow_verified_candidate_count"])
            for row in geometry_attempts)),
        "geometry_shadow_target_no_harm_failure_count": int(sum(
            int(row["geometry_shadow_target_no_harm_failure_count"])
            for row in geometry_attempts)),
        "geometry_shadow_window_worst_failure_count": int(sum(
            int(row["geometry_shadow_window_worst_failure_count"])
            for row in geometry_attempts)),
        "geometry_shadow_fast_opportunity_failure_count": int(sum(
            int(row["geometry_shadow_fast_opportunity_failure_count"])
            for row in geometry_attempts)),
        "geometry_shadow_strict_improvement_failure_count": int(sum(
            int(row["geometry_shadow_strict_improvement_failure_count"])
            for row in geometry_attempts)),
        "geometry_shadow_transport_failure_count": int(sum(
            int(row["geometry_shadow_transport_failure_count"])
            for row in geometry_attempts)),
        "geometry_shadow_energy_failure_count": int(sum(
            int(row["geometry_shadow_energy_failure_count"])
            for row in geometry_attempts)),
        "geometry_shadow_mean_certified_worst_pd_improvement": float(np.mean([
            float(row["geometry_shadow_certified_worst_pd_improvement"])
            for row in geometry_accepts
        ]) if geometry_accepts else 0.0),
        "geometry_shadow_mean_realized_worst_pd_improvement": float(np.mean([
            float(row["geometry_shadow_realized_worst_pd_improvement"])
            for row in geometry_accepts
        ]) if geometry_accepts else 0.0),
        "geometry_shadow_realized_target_no_harm_violation_count": int(sum(
            not bool(row["geometry_shadow_target_no_harm"])
            for row in geometry_accepts)),
        "geometry_shadow_realized_window_target_no_harm_violation_count": int(
            sum(
                not bool(row["geometry_shadow_window_target_no_harm"])
                for row in geometry_accepts
            )),
        "geometry_shadow_mean_realized_window_worst_pd_gain": float(np.mean([
            float(row["geometry_shadow_window_worst_pd_gain"])
            for row in geometry_accepts
        ]) if geometry_accepts else 0.0),
        "geometry_shadow_realized_window_global_worst_no_harm_violation_count": int(
            sum(
                not bool(row["geometry_shadow_window_global_worst_no_harm"])
                for row in geometry_accepts
            )),
        "geometry_shadow_mean_realized_window_global_worst_pd_gain": float(
            np.mean([
                float(row["geometry_shadow_window_global_worst_pd_gain"])
                for row in geometry_accepts
            ]) if geometry_accepts else 0.0),
        "geometry_shadow_max_terminal_position_error_m": float(np.max([
            float(row["geometry_shadow_terminal_position_error_m"])
            for row in geometry_accepts
        ], initial=0.0)),
        "geometry_shadow_max_terminal_velocity_error_mps": float(np.max([
            float(row["geometry_shadow_terminal_velocity_error_mps"])
            for row in geometry_accepts
        ], initial=0.0)),
        "geometry_shadow_envelope_coverage_rate": float(np.mean([
            bool(row["geometry_shadow_envelope_covered"])
            for row in geometry_accepts
        ]) if geometry_accepts else 0.0),
        "geometry_shadow_coefficient_lower_failure_count": int(sum(
            bool(row["geometry_shadow_coefficient_lower_failure"])
            for row in geometry_accepts)),
        "geometry_shadow_coefficient_upper_failure_count": int(sum(
            bool(row["geometry_shadow_coefficient_upper_failure"])
            for row in geometry_accepts)),
        "geometry_shadow_all_protocol_feasible": bool(all(
            bool(row["geometry_shadow_transport_feasible"])
            for row in geometry_accepts)),
        "geometry_shadow_mean_transport_bits": float(np.mean([
            float(row["geometry_shadow_transport_bits"])
            for row in geometry_accepts
        ]) if geometry_accepts else 0.0),
        "geometry_shadow_max_transport_latency_s": float(np.max([
            float(row["geometry_shadow_transport_latency_s"])
            for row in geometry_accepts
        ], initial=0.0)),
        "geometry_shadow_mean_incremental_flight_energy_j": float(np.mean([
            float(row["geometry_shadow_incremental_flight_energy_j"])
            for row in geometry_accepts
        ]) if geometry_accepts else 0.0),
        "geometry_shadow_mean_combined_bits": float(np.mean([
            float(row["geometry_shadow_combined_bits"])
            for row in geometry_accepts
        ]) if geometry_accepts else 0.0),
        "geometry_shadow_max_combined_latency_s": float(np.max([
            float(row["geometry_shadow_combined_latency_s"])
            for row in geometry_accepts
        ], initial=0.0)),
        "geometry_shadow_mean_combined_energy_j": float(np.mean([
            float(row["geometry_shadow_combined_energy_j"])
            for row in geometry_accepts
        ]) if geometry_accepts else 0.0),
        "geometry_shadow_dual_endpoint_accept_count": int(sum(
            int(row["geometry_shadow_mover_count"]) == 2
            for row in geometry_accepts)),
        "geometry_shadow_minimum_separation_m": float(np.min([
            float(row["geometry_shadow_minimum_separation_m"])
            for row in geometry_accepts
        ], initial=float("inf"))),
        "geometry_tube_started_count": len(geometry_accepts),
        "geometry_atomic_transaction_started_count": len(transaction_keys),
        "geometry_atomic_transaction_executed_step_count": len(tube_rows),
        "geometry_tube_executed_step_count": len(proof_tube_rows),
        "geometry_tube_expected_step_count": int(
            len(geometry_accepts) * geometry_horizon),
        "geometry_tube_complete_execution": bool(
            not bool(persistent_geometry)
            or len(proof_tube_rows)
            == len(geometry_accepts) * geometry_horizon),
        "geometry_atomic_transaction_complete_execution": bool(
            not bool(persistent_geometry)
            or len(tube_rows) == len(transaction_keys) * geometry_horizon),
        "geometry_tube_envelope_coverage_rate": float(np.mean([
            bool(row["envelope_covered"]) for row in proof_tube_rows
        ]) if proof_tube_rows else 0.0),
        "geometry_tube_independent_bound_step_target_no_harm_violation_count": int(sum(
            not bool(
                row["geometry_tube_independent_bound_step_target_no_harm"])
            for row in proof_tube_rows)),
        "geometry_tube_certified_window_target_no_harm_failure_count": int(sum(
            not bool(row["geometry_tube_certified_window_target_no_harm"])
            for row in proof_tube_start_rows)),
        "geometry_tube_certified_window_global_worst_no_harm_failure_count": int(sum(
            not bool(row["geometry_tube_certified_window_worst_no_harm"])
            for row in proof_tube_start_rows)),
        "geometry_tube_certified_first_step_strict_improvement_failure_count": int(sum(
            not bool(
                row[
                    "geometry_tube_certified_first_step_strict_improvement"])
            for row in proof_tube_start_rows)),
        "geometry_tube_minimum_certified_first_step_worst_gain_margin": float(
            np.min([
                float(row[
                    "geometry_tube_certified_first_step_worst_gain"])
                - float(row[
                    "geometry_tube_certified_first_step_worst_gain_threshold"])
                for row in proof_tube_start_rows
            ]) if proof_tube_start_rows else 0.0),
        "geometry_tube_realized_paired_step_target_no_harm_violation_count": int(sum(
            not bool(
                row["geometry_tube_realized_paired_step_target_no_harm"])
            for row in proof_tube_rows)),
        "geometry_tube_realized_paired_window_target_no_harm_violation_count": int(sum(
            np.min(gain) < -1.0e-9 for gain in tube_window_gains)),
        "geometry_tube_mean_realized_paired_window_worst_clipped_pd_gain": float(np.mean([
            float(np.min(gain)) for gain in tube_window_gains
        ]) if tube_window_gains else 0.0),
        "geometry_tube_realized_paired_window_global_worst_no_harm_violation_count": int(sum(
            gain < -1.0e-9 for gain in tube_window_global_worst_gains)),
        "geometry_tube_mean_realized_paired_window_global_worst_pd_gain": float(
            np.mean(tube_window_global_worst_gains)
            if tube_window_global_worst_gains else 0.0),
        "geometry_tube_realized_paired_first_step_strict_improvement_failure_count": int(sum(
            not bool(row[
                "geometry_tube_realized_paired_first_step_strict_improvement"])
            for row in proof_tube_start_rows)),
        "geometry_tube_max_active_branch_coefficient_match_error": float(
            np.max([
                float(row[
                    "geometry_tube_active_branch_coefficient_match_error"])
                for row in proof_tube_rows
            ], initial=0.0)),
        "geometry_tube_mean_realized_worst_pd": float(np.mean([
            float(row["geometry_tube_realized_worst_pd"])
            for row in proof_tube_rows
        ]) if proof_tube_rows else 0.0),
        "elapsed_seconds": float(perf_counter() - started),
        "episode_scores": {
            str(seed): float(episode_scores[seed]) for seed in seed_order},
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--max-age-frames", type=int, default=150)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--ranking-rounds", type=int, default=4)
    parser.add_argument("--owners-per-target", type=int, default=3)
    parser.add_argument("--snr-margin-db", type=float, default=3.0)
    parser.add_argument("--latency-margin-s", type=float, default=5.0e-4)
    parser.add_argument(
        "--persistent-geometry",
        action="store_true",
        help=(
            "execute accepted geometry as a delayed finite-horizon tube and "
            "recompute Token transport on the resulting physical branch"),
    )
    parser.add_argument(
        "--disable-geometry",
        action="store_true",
        help="run a matched persistent branch without geometry interventions",
    )
    parser.add_argument(
        "--announced-mobility-horizon",
        action="store_true",
        help=(
            "use the recorded H-step movement tape as a surrogate for an "
            "Actor-announced mobility plan; this is diagnostic, not a "
            "deployment-causal claim"),
    )
    parser.add_argument(
        "--causal-joint-plan",
        action="store_true",
        help=(
            "commit the current local movement/RF action by zero-order hold "
            "for H steps; unlike the recorded-future surrogate, this mode "
            "does not read future trace rows"),
    )
    parser.add_argument(
        "--causal-plan-checkpoint",
        type=Path,
        default=None,
        help=(
            "holdout-admitted set-equivariant movement-plan checkpoint; RF "
            "remains at the current feasible Actor split"),
    )
    parser.add_argument(
        "--causal-plan-role",
        choices=("baseline", "proposal"),
        default="baseline",
        help=(
            "replace the committed movement baseline, or preserve ZOH and "
            "use the learned intent only in reserved verification slots"),
    )
    parser.add_argument(
        "--allow-uniform-plan-region-scaling",
        action="store_true",
        help=(
            "reparameterize only the planner's dimensionless position and "
            "distance inputs under a uniform region scale; anisotropic "
            "region changes and speed-disk changes remain forbidden"),
    )
    parser.add_argument(
        "--geometry-execution-mode",
        choices=("repair", "committed_baseline"),
        default="repair",
        help=(
            "execute the certified geometry repair or its exact committed "
            "Actor baseline for a matched slow-layer ablation"),
    )
    parser.add_argument(
        "--geometry-verification-top-m",
        type=int,
        default=4,
        choices=range(1, 9),
    )
    parser.add_argument(
        "--geometry-horizon-steps",
        type=int,
        default=None,
        choices=range(3, 9),
    )
    parser.add_argument(
        "--mobility-reserve-fraction",
        type=float,
        default=1.0,
        help=(
            "fraction of v_max dt exposed to the Actor movement plan; the "
            "remainder is reserved for certified geometry recovery"),
    )
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.config,
        args.calibration,
        seed_limit=max(1, int(args.seed_limit)),
        max_age_frames=max(1, int(args.max_age_frames)),
        qos_floor=float(args.qos_floor),
        rounds=max(1, int(args.rounds)),
        ranking_rounds=max(1, int(args.ranking_rounds)),
        owners_per_target=max(1, int(args.owners_per_target)),
        snr_margin_db=float(args.snr_margin_db),
        latency_margin_s=float(args.latency_margin_s),
        persistent_geometry=bool(args.persistent_geometry),
        enable_geometry=not bool(args.disable_geometry),
        announced_mobility_horizon=bool(
            args.announced_mobility_horizon),
        causal_joint_plan=bool(args.causal_joint_plan),
        causal_plan_checkpoint=args.causal_plan_checkpoint,
        causal_plan_role=str(args.causal_plan_role),
        allow_uniform_plan_region_scaling=bool(
            args.allow_uniform_plan_region_scaling),
        geometry_execution_mode=str(args.geometry_execution_mode),
        geometry_verification_top_m=int(
            args.geometry_verification_top_m),
        geometry_horizon_steps=(
            None if args.geometry_horizon_steps is None
            else int(args.geometry_horizon_steps)),
        mobility_reserve_fraction=float(args.mobility_reserve_fraction),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "seed_count", "event_count", "eligible_event_count",
        "causal_availability_rate", "acceptance_rate_given_available",
        "baseline_mean_worst", "policy_mean_worst",
        "baseline_qos_rate", "policy_qos_rate",
        "accepted_target_no_harm_violation_count",
        "episode_envelope_coverage_rate", "route_counts",
        "geometry_shadow_attempt_count", "geometry_shadow_accept_count",
        "geometry_shadow_mean_certified_worst_pd_improvement",
        "geometry_shadow_realized_target_no_harm_violation_count",
        "geometry_shadow_realized_window_target_no_harm_violation_count",
        "geometry_shadow_realized_window_global_worst_no_harm_violation_count",
        "geometry_tube_executed_step_count",
        "geometry_tube_envelope_coverage_rate",
        "geometry_tube_certified_window_target_no_harm_failure_count",
        "geometry_tube_certified_window_global_worst_no_harm_failure_count",
        "geometry_tube_certified_first_step_strict_improvement_failure_count",
        "geometry_tube_realized_paired_window_target_no_harm_violation_count",
        "geometry_tube_realized_paired_window_global_worst_no_harm_violation_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
