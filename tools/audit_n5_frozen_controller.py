#!/usr/bin/env python
"""Replay one N5 intervention with the no-op controller trajectory frozen.

This is a diagnostic path-specific intervention, not a deployable controller.
The baseline and forced-N5 branches start from the same pre-step simulator
state and receive identical movement, Token, rate, mask, power, sensing and
frozen-Student graph submissions recorded on the no-op trajectory.  Therefore
any remaining difference is caused by the structural intervention and its
stateful local-search descendants, rather than by Actor/Student feedback.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from uav_isac.agents.frozen_structure_student import (
    FrozenStructureStudent,
    build_structure_student_features,
)
from uav_isac.coordination.dynamic_local_search import (
    DynamicLocalSearchCoordinator,
)
from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    enumerate_local_moves,
    oracle_best_improvement,
    role_owner_from_structure,
    structure_objective_key,
)
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.observation_slices import ObservationSlices
from uav_isac.evaluation.local_candidate_audit import (
    build_local_candidate_mask,
    delivered_target_tokens,
    select_local_neighbors,
    select_value_guided_neighbors,
)


def _qos(pd: np.ndarray) -> dict[str, float]:
    values = np.asarray(pd, dtype=np.float64).reshape(-1)
    ordered = np.sort(values)
    return {
        "steady": float(np.mean(values)),
        "weak3": float(np.mean(ordered[:min(3, values.size)])),
        "worst": float(ordered[0]),
    }


def _observation_slices(env: UAVISACEnv) -> ObservationSlices:
    builder = env.core.obs_builder
    return ObservationSlices.from_config(
        env.core.K,
        env.core.Q,
        use_p0=bool(builder.use_p0_global_info),
        use_rel_features=bool(builder.use_relative_features),
        use_comm_tokens=bool(builder.use_comm_tokens),
        comm_token_dim=int(builder.comm_token_dim),
        comm_tokens_per_sender=int(builder.comm_tokens_per_sender),
        use_channel_feedback=bool(builder.use_channel_feedback),
        channel_feedback_dim=int(builder.channel_feedback_dim),
    )


def _trace_rows(trace: Any, seed: int) -> dict[int, int]:
    indices = np.flatnonzero(np.asarray(trace["seed"]) == int(seed))
    rows = {
        int(trace["frame"][index]): int(index)
        for index in indices
    }
    if not rows:
        raise ValueError(f"seed {seed} is absent from the trace")
    return rows


def _actions(trace: Any, row: int) -> dict[str, dict[str, Any]]:
    movement = np.asarray(trace["delta_p"][row], dtype=np.float64)
    return {
        str(k): {"delta_p": movement[k].copy(), "role": 0}
        for k in range(movement.shape[0])
    }


def _submit_frozen_controller(
    env: UAVISACEnv,
    trace: Any,
    row: int,
    student: FrozenStructureStudent,
    slices: ObservationSlices,
    *,
    neighbor_topk: int,
    target_topk: int,
    coverage_fraction: float,
    qos_floor: float,
) -> None:
    """Submit every fast/control-plane output from one no-op trace row."""
    messages_array = np.asarray(
        trace["outgoing_message"][row], dtype=np.float64)
    rates_array = np.asarray(trace["outgoing_rate"][row], dtype=np.int64)
    masks_array = np.asarray(
        trace["outgoing_token_mask"][row], dtype=np.float64)
    fractions_array = np.asarray(
        trace["comm_fraction"][row], dtype=np.float64)
    sensing_array = np.asarray(
        trace["sensing_weights"][row], dtype=np.float64)
    K = messages_array.shape[0]
    env.core.submit_learned_communications(
        {k: messages_array[k].copy() for k in range(K)},
        {k: int(rates_array[k]) for k in range(K)},
        {k: float(fractions_array[k]) for k in range(K)},
        {k: sensing_array[k].copy() for k in range(K)},
        token_masks={k: masks_array[k].copy() for k in range(K)},
    )

    # Freeze the complete distributed structural signal too.  It is rebuilt
    # from the recorded no-op observation rather than the counterfactual
    # branch observation, so this audit does not silently re-introduce a
    # Student feedback path while claiming that the controller is frozen.
    local_obs = np.asarray(
        trace["local_obs"][row], dtype=np.float32)[None, ...]
    token_visible, token_age = delivered_target_tokens(local_obs, slices)
    local_pd = slices.extract_pd_hist(local_obs)
    local_neighbors = select_local_neighbors(
        token_visible,
        token_age,
        neighbor_topk=int(neighbor_topk),
    )
    feature_kwargs = dict(
        outgoing_message=messages_array,
        outgoing_token_mask=masks_array,
        outgoing_rate=rates_array,
        comm_fraction=fractions_array,
        sensing_weights=sensing_array,
        rate_scale=student.rate_scale,
    )
    provisional_features = build_structure_student_features(
        local_obs,
        slices,
        neighbor_subset_mask=local_neighbors[0],
        **feature_kwargs,
    )
    provisional_values = student.predict(provisional_features)
    local_neighbors = select_value_guided_neighbors(
        provisional_values,
        local_pd,
        token_visible,
        token_age,
        neighbor_topk=int(neighbor_topk),
        qos_floor=float(qos_floor),
        coverage_fraction=float(coverage_fraction),
    )
    features = build_structure_student_features(
        local_obs,
        slices,
        neighbor_subset_mask=local_neighbors[0],
        **feature_kwargs,
    )
    edge_values = student.predict(features)[0]
    candidate_mask, _ = build_local_candidate_mask(
        edge_values[None, ...],
        local_pd,
        local_neighbors,
        token_visible,
        target_topk=int(target_topk),
        qos_floor=float(qos_floor),
        coverage_fraction=float(coverage_fraction),
        require_reciprocal_link=True,
    )
    env.core.submit_structure_student_candidate_mask(candidate_mask[0])
    env.core.submit_structure_student_edge_values(edge_values)


def _proxy_rebootstrap_move(env: UAVISACEnv, rounds: int) -> LocalMove:
    core = env.core
    coordinator = core._dynamic_local_search_coordinator
    if coordinator is None or coordinator.last_problem() is None:
        raise RuntimeError("dynamic local-search problem is unavailable")
    edge_value, candidate_mask = coordinator.last_problem()
    selected = np.asarray(
        core._cached_p0_solution.z_selected, dtype=bool).copy()
    state = coordinator.get_state()
    if state.get("role") is None:
        raise RuntimeError("dynamic local-search role state is unavailable")
    role, owner = role_owner_from_structure(
        selected,
        fallback_role=np.asarray(state["role"], dtype=np.int8),
    )
    reports_per_receiver = (
        max(1, int(
            core.cfg.p0_solver.capacity_per_rx
            // max(core.cfg.detection.B_q, 1)))
        if core.ground_communication_enabled
        else core.Q * core.cfg.detection.K_q_max
    )
    result = oracle_best_improvement(
        selected,
        edge_value,
        candidate_mask,
        rounds=max(1, int(rounds)),
        neighborhoods=("N5",),
        target_pair_limit=int(core.cfg.detection.K_q_max),
        reports_per_receiver=int(reports_per_receiver),
        initial_role=role,
    )
    if not result.accepted_kinds:
        raise RuntimeError("event has no accepted proxy-weak N5 sequence")
    return LocalMove(
        "N5_sequence",
        result.selected.copy(),
        result.role.copy(),
        result.owner.copy(),
    )


def _all_target_atomic_move(
    env: UAVISACEnv,
    candidate_index: int,
) -> LocalMove:
    """Reconstruct one deterministically ordered all-target audit candidate."""
    core = env.core
    coordinator = core._dynamic_local_search_coordinator
    if coordinator is None or coordinator.last_problem() is None:
        raise RuntimeError("dynamic local-search problem is unavailable")
    edge_value, candidate_mask = coordinator.last_problem()
    selected = np.asarray(
        core._cached_p0_solution.z_selected, dtype=bool).copy()
    state = coordinator.get_state()
    if state.get("role") is None:
        raise RuntimeError("dynamic local-search role state is unavailable")
    role, owner = role_owner_from_structure(
        selected,
        fallback_role=np.asarray(state["role"], dtype=np.int8),
    )
    reports_per_receiver = (
        max(1, int(
            core.cfg.p0_solver.capacity_per_rx
            // max(core.cfg.detection.B_q, 1)))
        if core.ground_communication_enabled
        else core.Q * core.cfg.detection.K_q_max
    )
    moves = enumerate_local_moves(
        selected,
        edge_value,
        candidate_mask,
        role,
        owner,
        neighborhoods=("N5",),
        target_pair_limit=int(core.cfg.detection.K_q_max),
        reports_per_receiver=int(reports_per_receiver),
        n5_target_mode="all",
    )
    ordered = sorted(
        moves,
        key=lambda move: structure_objective_key(
            move.selected, edge_value),
        reverse=True,
    )
    index = int(candidate_index)
    if index < 0 or index >= len(ordered):
        raise IndexError(
            f"candidate index {index} is outside [0, {len(ordered)})")
    move = ordered[index]
    return LocalMove(
        "N5_all_target_atomic",
        move.selected.copy(),
        move.role.copy(),
        move.owner.copy(),
    )


def _rollout(
    pre_step_env: UAVISACEnv,
    controller_trace: Any,
    controller_rows: dict[int, int],
    movement_trace: Any,
    movement_rows: dict[int, int],
    student_trace: Any,
    student_rows: dict[int, int],
    frames: list[int],
    student: FrozenStructureStudent,
    slices: ObservationSlices,
    *,
    forced_move: LocalMove | None,
    neighbor_topk: int,
    target_topk: int,
    coverage_fraction: float,
    qos_floor: float,
) -> list[dict[str, Any]]:
    branch = deepcopy(pre_step_env)
    records: list[dict[str, Any]] = []
    try:
        for offset, frame in enumerate(frames):
            controller_row = controller_rows[frame]
            movement_row = movement_rows[frame]
            student_row = student_rows[frame]
            _submit_frozen_controller(
                branch,
                controller_trace,
                controller_row,
                student,
                slices,
                neighbor_topk=neighbor_topk,
                target_topk=target_topk,
                coverage_fraction=coverage_fraction,
                qos_floor=qos_floor,
            )
            if student_trace is not controller_trace:
                # The physical radio/resource submission comes from the
                # selected controller path, while the public Student graph is
                # replaced by the requested mediation path.
                _submit_frozen_student_graph(
                    branch,
                    student_trace,
                    student_row,
                    student,
                    slices,
                    neighbor_topk=neighbor_topk,
                    target_topk=target_topk,
                    coverage_fraction=coverage_fraction,
                    qos_floor=qos_floor,
                )
            if offset == 0 and forced_move is not None:
                coordinator = branch.core._dynamic_local_search_coordinator
                if coordinator is None:
                    raise RuntimeError("counterfactual branch lost coordinator")
                coordinator.force_next_move_for_audit(forced_move)
            _, _, terminated, truncated, info = branch.step(
                _actions(movement_trace, movement_row))
            pd = np.asarray(info["P_D_q"], dtype=np.float64)
            selected = np.asarray(
                branch.core._cached_p0_solution.z_selected,
                dtype=np.uint8,
            )
            records.append({
                "frame": int(frame),
                "pd": pd.tolist(),
                **_qos(pd),
                "selected": selected,
                "uav_positions": np.stack([
                    uav.get_state().pos for uav in branch.core.uavs]),
                "target_states": np.stack([
                    np.asarray(target.state, dtype=np.float64)
                    for target in branch.core.targets]),
                "power_balance_error_w": float(info.get(
                    "isac_max_power_balance_error_w", 0.0)),
                "done": bool(
                    terminated.get("__all__", False)
                    or truncated.get("__all__", False)),
            })
            if records[-1]["done"] and offset + 1 < len(frames):
                raise RuntimeError("episode ended before the audit horizon")
    finally:
        branch.close()
    return records


def _submit_frozen_student_graph(
    env: UAVISACEnv,
    trace: Any,
    row: int,
    student: FrozenStructureStudent,
    slices: ObservationSlices,
    *,
    neighbor_topk: int,
    target_topk: int,
    coverage_fraction: float,
    qos_floor: float,
) -> None:
    """Replace only the external Student graph from one recorded path."""
    messages_array = np.asarray(
        trace["outgoing_message"][row], dtype=np.float64)
    rates_array = np.asarray(trace["outgoing_rate"][row], dtype=np.int64)
    masks_array = np.asarray(
        trace["outgoing_token_mask"][row], dtype=np.float64)
    fractions_array = np.asarray(
        trace["comm_fraction"][row], dtype=np.float64)
    sensing_array = np.asarray(
        trace["sensing_weights"][row], dtype=np.float64)
    local_obs = np.asarray(
        trace["local_obs"][row], dtype=np.float32)[None, ...]
    token_visible, token_age = delivered_target_tokens(local_obs, slices)
    local_pd = slices.extract_pd_hist(local_obs)
    local_neighbors = select_local_neighbors(
        token_visible, token_age, neighbor_topk=int(neighbor_topk))
    kwargs = dict(
        outgoing_message=messages_array,
        outgoing_token_mask=masks_array,
        outgoing_rate=rates_array,
        comm_fraction=fractions_array,
        sensing_weights=sensing_array,
        rate_scale=student.rate_scale,
    )
    provisional = build_structure_student_features(
        local_obs,
        slices,
        neighbor_subset_mask=local_neighbors[0],
        **kwargs,
    )
    values = student.predict(provisional)
    local_neighbors = select_value_guided_neighbors(
        values,
        local_pd,
        token_visible,
        token_age,
        neighbor_topk=int(neighbor_topk),
        qos_floor=float(qos_floor),
        coverage_fraction=float(coverage_fraction),
    )
    features = build_structure_student_features(
        local_obs,
        slices,
        neighbor_subset_mask=local_neighbors[0],
        **kwargs,
    )
    edge_values = student.predict(features)[0]
    candidate_mask, _ = build_local_candidate_mask(
        edge_values[None, ...],
        local_pd,
        local_neighbors,
        token_visible,
        target_topk=int(target_topk),
        qos_floor=float(qos_floor),
        coverage_fraction=float(coverage_fraction),
        require_reciprocal_link=True,
    )
    env.core.submit_structure_student_candidate_mask(candidate_mask[0])
    env.core.submit_structure_student_edge_values(edge_values)


def _paired_summary(
    baseline: list[dict[str, Any]],
    intervention: list[dict[str, Any]],
    trace: Any,
    rows: dict[int, int],
    *,
    gamma: float,
) -> dict[str, Any]:
    if len(baseline) != len(intervention) or not baseline:
        raise ValueError("paired rollouts must be non-empty and equally long")
    frame_rows = []
    for base, forced in zip(baseline, intervention):
        frame = int(base["frame"])
        delta_pd = np.asarray(forced["pd"]) - np.asarray(base["pd"])
        frame_rows.append({
            "frame": frame,
            "baseline_worst": float(base["worst"]),
            "forced_worst": float(forced["worst"]),
            "delta_worst": float(forced["worst"] - base["worst"]),
            "delta_weak3": float(forced["weak3"] - base["weak3"]),
            "delta_steady": float(forced["steady"] - base["steady"]),
            "delta_pd": delta_pd.tolist(),
            "changed_edges": int(np.count_nonzero(
                np.asarray(forced["selected"])
                != np.asarray(base["selected"]))),
            "uav_position_max_error_m": float(np.max(np.abs(
                np.asarray(forced["uav_positions"])
                - np.asarray(base["uav_positions"])))),
            "target_state_max_error": float(np.max(np.abs(
                np.asarray(forced["target_states"])
                - np.asarray(base["target_states"])))),
        })

    # Index zero is the intervention frame.  The requested future horizon is
    # deliberately reported on indices 1..H, without relabelling the direct
    # same-frame effect as a future return.
    future = frame_rows[1:]
    discounts = np.power(float(gamma), np.arange(len(future)))
    future_delta = np.asarray([
        row["delta_worst"] for row in future], dtype=np.float64)
    baseline_trace_error = max(
        float(np.max(np.abs(
            np.asarray(record["pd"], dtype=np.float64)
            - np.asarray(trace["physical_pd"][rows[int(record["frame"])]],
                         dtype=np.float64))))
        for record in baseline
    )
    return {
        "immediate": frame_rows[0],
        "future": {
            "horizon_frames": int(len(future)),
            "mean_delta_worst": float(np.mean(future_delta))
            if future_delta.size else 0.0,
            "discounted_sum_delta_worst": float(np.sum(
                discounts * future_delta)) if future_delta.size else 0.0,
            "min_frame_delta_worst": float(np.min(future_delta))
            if future_delta.size else 0.0,
            "endpoint_delta_worst": float(future_delta[-1])
            if future_delta.size else 0.0,
            "positive_frame_rate": float(np.mean(future_delta > 0.0))
            if future_delta.size else 0.0,
        },
        "frame_rows": frame_rows,
        "baseline_replay_pd_max_error": float(baseline_trace_error),
        "paired_uav_position_max_error_m": float(max(
            row["uav_position_max_error_m"] for row in frame_rows)),
        "paired_target_state_max_error": float(max(
            row["target_state_max_error"] for row in frame_rows)),
        "max_power_balance_error_w": float(max(
            [record["power_balance_error_w"]
             for record in baseline + intervention] or [0.0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-controller multi-frame N5 causal replay")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument(
        "--feedback-trace",
        default=None,
        help=(
            "optional matched closed-loop intervention trace used for "
            "cross-world movement/radio/Student mediation diagnostics"),
    )
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--ranker-checkpoint", required=True)
    parser.add_argument("--factor-graph-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--event-frame", type=int, required=True)
    parser.add_argument("--future-horizon", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--neighbor-topk", type=int, default=4)
    parser.add_argument("--target-topk", type=int, default=4)
    parser.add_argument("--coverage-fraction", type=float, default=0.30)
    parser.add_argument("--cold-rounds", type=int, default=8)
    parser.add_argument("--warm-rounds", type=int, default=8)
    parser.add_argument("--warm-top-m", type=int, default=5)
    parser.add_argument("--rebootstrap-rounds", type=int, default=8)
    parser.add_argument(
        "--move-mode",
        choices=["proxy_sequence", "all_target_candidate"],
        default="proxy_sequence",
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=-1,
        help=(
            "deterministic all-target atomic candidate index when "
            "--move-mode=all_target_candidate"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    trace_path = Path(args.trace)
    with np.load(trace_path, allow_pickle=False) as trace:
        rows = _trace_rows(trace, args.seed)
        frames = list(range(
            int(args.event_frame),
            int(args.event_frame) + max(0, int(args.future_horizon)) + 1,
        ))
        missing = [frame for frame in frames if frame not in rows]
        if missing:
            raise ValueError(f"trace is missing audit frames: {missing}")

        env = UAVISACEnv(config=cfg, seed=int(args.seed))
        env.core.configure_dynamic_local_search(
            DynamicLocalSearchCoordinator(
                "hybrid",
                ranker_checkpoint=args.ranker_checkpoint,
                cold_initializer="factor_graph",
                factor_graph_checkpoint=args.factor_graph_checkpoint,
                cold_rounds=max(0, int(args.cold_rounds)),
                warm_rounds=max(0, int(args.warm_rounds)),
                warm_top_m=max(1, int(args.warm_top_m)),
                rebootstrap_mode="off",
            )
        )
        student = FrozenStructureStudent(args.student_checkpoint)
        slices = _observation_slices(env)
        observations, _ = env.reset(seed=int(args.seed))
        qos_floor = float(getattr(
            cfg.marl, "comm_qos_worst_min", 0.60))

        pre_step_env = None
        baseline_event_info = None
        obs_max_error = 0.0
        pd_max_error = 0.0
        try:
            for frame in range(1, int(args.event_frame) + 1):
                if frame not in rows:
                    raise ValueError(f"trace is missing prefix frame {frame}")
                row = rows[frame]
                actual_obs = np.stack([
                    observations[str(k)] for k in range(env.core.K)])
                obs_max_error = max(obs_max_error, float(np.max(np.abs(
                    actual_obs - np.asarray(
                        trace["local_obs"][row], dtype=np.float64)))))
                _submit_frozen_controller(
                    env,
                    trace,
                    row,
                    student,
                    slices,
                    neighbor_topk=args.neighbor_topk,
                    target_topk=args.target_topk,
                    coverage_fraction=args.coverage_fraction,
                    qos_floor=qos_floor,
                )
                if frame == int(args.event_frame):
                    pre_step_env = deepcopy(env)
                observations, _, terminated, truncated, info = env.step(
                    _actions(trace, row))
                pd_max_error = max(pd_max_error, float(np.max(np.abs(
                    np.asarray(info["P_D_q"], dtype=np.float64)
                    - np.asarray(trace["physical_pd"][row],
                                 dtype=np.float64)))))
                if frame == int(args.event_frame):
                    baseline_event_info = info
                if (terminated.get("__all__", False)
                        or truncated.get("__all__", False)):
                    raise RuntimeError("baseline ended before the event")

            if pre_step_env is None or baseline_event_info is None:
                raise RuntimeError("failed to capture the event state")
            move = (
                _proxy_rebootstrap_move(
                    env, rounds=max(1, int(args.rebootstrap_rounds)))
                if args.move_mode == "proxy_sequence"
                else _all_target_atomic_move(
                    env, candidate_index=int(args.candidate_index))
            )
            rollout_kwargs = dict(
                frames=frames,
                student=student,
                slices=slices,
                neighbor_topk=args.neighbor_topk,
                target_topk=args.target_topk,
                coverage_fraction=args.coverage_fraction,
                qos_floor=qos_floor,
            )
            baseline = _rollout(
                pre_step_env,
                trace,
                rows,
                trace,
                rows,
                trace,
                rows,
                forced_move=None,
                **rollout_kwargs,
            )
            frozen_intervention = _rollout(
                pre_step_env,
                trace,
                rows,
                trace,
                rows,
                trace,
                rows,
                forced_move=move,
                **rollout_kwargs,
            )
            variants = {
                "all_noop_controller_frozen": _paired_summary(
                    baseline,
                    frozen_intervention,
                    trace,
                    rows,
                    gamma=float(args.gamma),
                )
            }

            feedback_trace_path = (
                Path(args.feedback_trace)
                if args.feedback_trace else None)
            if feedback_trace_path is not None:
                with np.load(
                    feedback_trace_path, allow_pickle=False,
                ) as feedback_trace:
                    feedback_rows = _trace_rows(feedback_trace, args.seed)
                    feedback_missing = [
                        frame for frame in frames
                        if frame not in feedback_rows]
                    if feedback_missing:
                        raise ValueError(
                            "feedback trace is missing audit frames: "
                            f"{feedback_missing}")

                    # These are cross-world path interventions.  They are not
                    # treated as natural indirect effects; the combinations
                    # are diagnostic probes of a nonlinear closed loop.
                    mediation_specs = {
                        "movement_feedback_only": (
                            trace, rows,
                            feedback_trace, feedback_rows,
                            trace, rows,
                        ),
                        "radio_resource_feedback_only": (
                            feedback_trace, feedback_rows,
                            trace, rows,
                            trace, rows,
                        ),
                        "student_feedback_only": (
                            trace, rows,
                            trace, rows,
                            feedback_trace, feedback_rows,
                        ),
                        "radio_resource_student_feedback": (
                            feedback_trace, feedback_rows,
                            trace, rows,
                            feedback_trace, feedback_rows,
                        ),
                        "all_recorded_feedback": (
                            feedback_trace, feedback_rows,
                            feedback_trace, feedback_rows,
                            feedback_trace, feedback_rows,
                        ),
                    }
                    for name, spec in mediation_specs.items():
                        intervention = _rollout(
                            pre_step_env,
                            spec[0], spec[1],
                            spec[2], spec[3],
                            spec[4], spec[5],
                            forced_move=move,
                            **rollout_kwargs,
                        )
                        variant = _paired_summary(
                            baseline,
                            intervention,
                            trace,
                            rows,
                            gamma=float(args.gamma),
                        )
                        variant["feedback_trace_pd_max_error"] = float(max(
                            np.max(np.abs(
                                np.asarray(record["pd"], dtype=np.float64)
                                - np.asarray(feedback_trace["physical_pd"][
                                    feedback_rows[int(record["frame"])]],
                                    dtype=np.float64)))
                            for record in intervention
                        ))
                        variants[name] = variant
            payload = {
                "schema_version": 1,
                "protocol": "n5_frozen_noop_controller_crn",
                "scope": (
                    "development-trace path-specific intervention; movement, "
                    "Token, rate, mask, communication/sensing allocation and "
                    "Student graph submissions are fixed to the no-op path"
                ),
                "seed": int(args.seed),
                "event_frame": int(args.event_frame),
                "future_horizon": int(args.future_horizon),
                "gamma": float(args.gamma),
                "move": {
                    "kind": move.kind,
                    "mode": args.move_mode,
                    "candidate_index": (
                        int(args.candidate_index)
                        if args.move_mode == "all_target_candidate"
                        else None
                    ),
                    "selected_edges": int(np.sum(move.selected)),
                    "changed_edges_vs_event_baseline": int(np.count_nonzero(
                        move.selected != np.asarray(
                            env.core._cached_p0_solution.z_selected,
                            dtype=bool))),
                },
                "prefix_replay": {
                    "observation_max_error": float(obs_max_error),
                    "physical_pd_max_error": float(pd_max_error),
                },
                "paired": variants["all_noop_controller_frozen"],
                "mediation_variants": variants,
                "feedback_trace": (
                    str(feedback_trace_path)
                    if feedback_trace_path is not None else None),
                "limitations": [
                    "This freezes the full no-op controller/Student path; it "
                    "does not estimate a deployable closed-loop policy.",
                    "The result is one event-level causal diagnostic and does "
                    "not estimate a population failure rate.",
                    "Mixed no-op/feedback variants are cross-world path "
                    "interventions; nonlinear interactions prevent treating "
                    "their differences as additive natural indirect effects.",
                ],
                "fresh_test_consumed": False,
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(payload, indent=2))
        finally:
            if pre_step_env is not None:
                pre_step_env.close()
            env.close()


if __name__ == "__main__":
    main()
