#!/usr/bin/env python
"""Replay audited N5 candidates through a physical atomic-commit certificate.

The exact branch uses the communication/sensing allocation that produced the
audited detection result.  A second, explicitly counterfactual branch reserves
control RF power for otherwise-silent participants.  That second branch is a
transport upper bound only: its changed sensing power invalidates the old
detection label and therefore requires a joint physical replay before use.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.audit_n5_frozen_controller import (  # noqa: E402
    _actions,
    _all_target_atomic_move,
    _observation_slices,
    _submit_frozen_controller,
    _trace_rows,
)
from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
)
from uav_isac.coordination.dependency_commit import (  # noqa: E402
    DependencyCommitCertificate,
    DependencyCommitLayout,
    certify_best_dependency_commit,
    minimum_uniform_control_reserve,
)
from uav_isac.coordination.dynamic_local_search import (  # noqa: E402
    DynamicLocalSearchCoordinator,
)
from uav_isac.coordination.local_exchange_oracle import (  # noqa: E402
    role_owner_from_structure,
)
from uav_isac.environment.env_wrapper import UAVISACEnv  # noqa: E402
from uav_isac.evaluation.n5_counterfactual_audit import (  # noqa: E402
    tail_deficit_dominates,
)
from uav_isac.evaluation.certified_feedback import (  # noqa: E402
    OwnerFeedbackLayout,
    OwnerFeedbackTransportCertificate,
    certify_owner_feedback_transport,
    minimum_joint_commit_feedback_reserve,
)


def _certificate_dict(
    certificate: DependencyCommitCertificate,
) -> dict[str, object]:
    return {
        "feasible": bool(certificate.feasible),
        "reasons": list(certificate.reasons),
        "proposer": certificate.proposer,
        "certificate_epoch_id": certificate.certificate_epoch_id,
        "certificate_digest": certificate.certificate_digest,
        "participants": list(certificate.closure.participants),
        "affected_targets": list(certificate.closure.affected_targets),
        "changed_roles": certificate.closure.changed_roles,
        "changed_owners": certificate.closure.changed_owners,
        "toggled_edges": certificate.closure.toggled_edges,
        "prepare_bits": certificate.prepare_bits,
        "vote_bits": certificate.vote_bits,
        "decision_bits": certificate.decision_bits,
        "total_over_air_bits": certificate.total_over_air_bits,
        "total_latency_s": certificate.total_latency_s,
        "total_energy_j": certificate.total_energy_j,
        "per_uav_energy_j": list(certificate.per_uav_energy_j),
        "power_excess_w": list(certificate.power_excess_w),
        "rounds": [
            {
                "name": report.name,
                "active_senders": list(report.active_senders),
                "payload_bits_per_sender": report.payload_bits_per_sender,
                "over_air_bits": report.over_air_bits,
                "duration_s": report.duration_s,
                "energy_j": report.energy_j,
                "min_snr_db": report.min_snr_db,
                "max_link_latency_s": report.max_link_latency_s,
                "feasible": report.feasible,
            }
            for report in certificate.rounds
        ],
    }


def _parse_indices(text: str) -> set[int] | None:
    normalized = str(text).strip()
    if not normalized:
        return None
    return {int(item.strip()) for item in normalized.split(",") if item.strip()}


def _parse_nonnegative_floats(text: str) -> tuple[float, ...]:
    values = tuple(
        float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("sensitivity multipliers must be finite/non-negative")
    if values[0] != 0.0:
        raise ValueError("sensitivity curve must start at multiplier zero")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("sensitivity multipliers must be strictly increasing")
    return values


def _workspace_path(value: str | Path) -> Path:
    """Resolve a recorded or supplied path against the repository root."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def verify_trace_provenance(
    manifest_path: str | Path,
    *,
    trace_path: str | Path,
    config_path: str | Path,
    student_checkpoint: str | Path,
    ranker_checkpoint: str | Path,
    factor_graph_checkpoint: str | Path,
    neighbor_topk: int,
    target_topk: int,
    coverage_fraction: float,
    cold_rounds: int,
    warm_rounds: int,
    warm_top_m: int,
) -> dict[str, object]:
    """Fail closed unless replay inputs match the trace-producing pipeline.

    A numerically different replay is not evidence of model error when the
    controller/checkpoint provenance is different.  Bind every structural
    input that can change the prefix before interpreting a replay residual.
    """
    manifest_file = _workspace_path(manifest_path)
    if not manifest_file.is_file():
        raise RuntimeError(
            f"trace provenance manifest is missing: {manifest_file}")
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    dynamic = payload.get("dynamic_local_search")
    if not isinstance(dynamic, dict):
        raise RuntimeError(
            "trace provenance manifest lacks dynamic_local_search")

    mismatches: list[str] = []

    def require_path(label: str, recorded: object, supplied: str | Path) -> None:
        if not isinstance(recorded, str) or not recorded.strip():
            mismatches.append(f"{label}:missing-in-manifest")
            return
        if _workspace_path(recorded) != _workspace_path(supplied):
            mismatches.append(
                f"{label}:recorded={recorded!r},supplied={str(supplied)!r}")

    def require_exact(label: str, recorded: object, supplied: object) -> None:
        if recorded != supplied:
            mismatches.append(
                f"{label}:recorded={recorded!r},supplied={supplied!r}")

    def require_float(label: str, recorded: object, supplied: float) -> None:
        try:
            value = float(recorded)
        except (TypeError, ValueError):
            mismatches.append(f"{label}:invalid-in-manifest={recorded!r}")
            return
        if not np.isclose(value, float(supplied), rtol=0.0, atol=1.0e-12):
            mismatches.append(
                f"{label}:recorded={value!r},supplied={float(supplied)!r}")

    require_path("trace", payload.get("structure_teacher_trace_output"), trace_path)
    require_path("config", payload.get("config"), config_path)
    require_path(
        "student_checkpoint",
        payload.get("structure_student_checkpoint"),
        student_checkpoint,
    )
    require_path(
        "ranker_checkpoint", dynamic.get("ranker_checkpoint"), ranker_checkpoint)
    require_path(
        "factor_graph_checkpoint",
        dynamic.get("factor_graph_checkpoint"),
        factor_graph_checkpoint,
    )
    require_exact("mode", dynamic.get("mode"), "hybrid")
    require_exact("cold_initializer", dynamic.get("cold_initializer"), "factor_graph")
    require_exact("rebootstrap_mode", dynamic.get("rebootstrap_mode"), "off")
    require_exact("neighbor_topk", dynamic.get("neighbor_topk"), int(neighbor_topk))
    require_exact("target_topk", dynamic.get("target_topk"), int(target_topk))
    require_float(
        "coverage_fraction", dynamic.get("coverage_fraction"), coverage_fraction)
    require_exact("cold_rounds", dynamic.get("cold_rounds"), int(cold_rounds))
    require_exact("warm_rounds", dynamic.get("warm_rounds"), int(warm_rounds))
    require_exact("warm_top_m", dynamic.get("warm_top_m"), int(warm_top_m))

    if mismatches:
        raise RuntimeError(
            "trace provenance mismatch; replay residual is not interpretable: "
            + "; ".join(mismatches))
    return {
        "manifest": str(manifest_file),
        "git_commit": payload.get("git_commit"),
        "verified": True,
        "bound_fields": [
            "trace",
            "config",
            "student_checkpoint",
            "ranker_checkpoint",
            "factor_graph_checkpoint",
            "mode",
            "cold_initializer",
            "rebootstrap_mode",
            "neighbor_topk",
            "target_topk",
            "coverage_fraction",
            "cold_rounds",
            "warm_rounds",
            "warm_top_m",
        ],
    }


def _candidate_digest(move, *, state_version: int, epoch_id: int) -> int:
    """Return the 64-bit wire digest of one frozen proposal identity."""
    digest = hashlib.sha256()
    digest.update(int(state_version).to_bytes(8, "big", signed=False))
    digest.update(int(epoch_id).to_bytes(8, "big", signed=False))
    digest.update(np.asarray(move.selected, dtype=np.uint8).tobytes(order="C"))
    digest.update(np.asarray(move.role, dtype=np.int16).tobytes(order="C"))
    digest.update(np.asarray(move.owner, dtype=np.int64).tobytes(order="C"))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def _feedback_certificate_dict(
    certificate: OwnerFeedbackTransportCertificate,
) -> dict[str, object]:
    return {
        "feasible": certificate.feasible,
        "reasons": list(certificate.reasons),
        "coordinator": certificate.coordinator,
        "certificate_epoch_id": certificate.certificate_epoch_id,
        "certificate_digest": certificate.certificate_digest,
        "senders": list(certificate.senders),
        "total_over_air_bits": certificate.total_over_air_bits,
        "max_latency_s": certificate.max_latency_s,
        "total_energy_j": certificate.total_energy_j,
        "min_snr_db": certificate.min_snr_db,
        "per_sender_bits": list(certificate.per_sender_bits),
        "per_uav_energy_j": list(certificate.per_uav_energy_j),
        "power_excess_w": list(certificate.power_excess_w),
    }


def _same_state_detection_replay(
    *,
    pre_step_env: UAVISACEnv,
    move,
    participants: tuple[int, ...],
    comm_power_w: np.ndarray,
    sensing_power_w: np.ndarray,
    total_power_w: float,
    trace,
    event_row: int,
    baseline_pd: np.ndarray,
    p_d_floor: float,
) -> dict[str, object]:
    """Replay the sensing consequence of one certified RF allocation."""
    branch = deepcopy(pre_step_env)
    try:
        core = branch.core
        coordinator = core._dynamic_local_search_coordinator
        if coordinator is None:
            raise RuntimeError("reserve branch lost coordinator")
        coordinator.force_next_move_for_audit(move)
        for participant in participants:
            core._pending_comm_power_fractions[participant] = (
                float(comm_power_w[participant]) / float(total_power_w))
            if core._pending_comm_rates.get(participant, 0) <= 0:
                core._pending_comm_rates[participant] = 1
        _, _, terminated, truncated, info = branch.step(
            _actions(trace, event_row))
        replay_pd = np.asarray(info["P_D_q"], dtype=np.float64)
        replay_selected = np.asarray(
            core._cached_p0_solution.z_selected, dtype=bool)
        if not np.array_equal(replay_selected, move.selected):
            raise AssertionError("reserve branch executed a different move")
        branch_comm = np.asarray(core._current_comm_power_w, dtype=np.float64)
        branch_sensing = np.asarray(
            core._current_sensing_power_w, dtype=np.float64)
        allocation_error = max(
            float(np.max(np.abs(branch_comm - comm_power_w))),
            float(np.max(np.abs(branch_sensing - sensing_power_w))),
        )
        return {
            "pd": replay_pd.tolist(),
            "delta_pd": (replay_pd - baseline_pd).tolist(),
            "worst": float(np.min(replay_pd)),
            "delta_worst": float(
                np.min(replay_pd) - np.min(baseline_pd)),
            "tail_deficit_dominates": bool(tail_deficit_dominates(
                replay_pd,
                baseline_pd,
                p_d_floor=float(p_d_floor),
            )),
            "allocation_max_error_w": allocation_error,
            "power_balance_error_w": float(info.get(
                "isac_max_power_balance_error_w", 0.0)),
            "done": bool(
                terminated.get("__all__", False)
                or truncated.get("__all__", False)),
            "activation_surrogate": (
                "minimum positive learned rate for silent control participants; "
                "semantic payload is not counted as certificate evidence"
            ),
        }
    finally:
        branch.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument(
        "--trace-manifest",
        default="",
        help="trace run manifest; defaults to run_manifest.json beside --trace",
    )
    parser.add_argument("--event-audit", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--ranker-checkpoint", required=True)
    parser.add_argument("--factor-graph-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--event-frame", type=int, required=True)
    parser.add_argument(
        "--candidate-indices",
        default="",
        help="comma-separated audit indices; default is every positive tail-safe candidate",
    )
    parser.add_argument("--p-d-floor", type=float, default=0.60)
    parser.add_argument(
        "--max-prefix-observation-error", type=float, default=1.0e-6)
    parser.add_argument(
        "--max-prefix-physical-pd-error", type=float, default=1.0e-8)
    parser.add_argument("--feedback-horizon", type=int, default=5)
    parser.add_argument(
        "--channel-sensitivity-multipliers",
        default="0,1,2,3,4,6",
        help=(
            "strictly increasing diagnostic beta values; these do not replace "
            "event-level conformal calibration"
        ),
    )
    parser.add_argument("--snr-resolution-db", type=float, default=1.0)
    parser.add_argument("--latency-resolution-ms", type=float, default=0.1)
    parser.add_argument("--neighbor-topk", type=int, default=4)
    parser.add_argument("--target-topk", type=int, default=4)
    parser.add_argument("--coverage-fraction", type=float, default=0.30)
    parser.add_argument("--cold-rounds", type=int, default=8)
    parser.add_argument("--warm-rounds", type=int, default=8)
    parser.add_argument("--warm-top-m", type=int, default=5)
    parser.add_argument(
        "--reserve-power-w",
        type=float,
        default=None,
        help="prospective per-participant control reserve; defaults to configured U2U TX power",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    trace_manifest = (
        Path(args.trace_manifest)
        if str(args.trace_manifest).strip()
        else trace_path.parent / "run_manifest.json"
    )
    trace_provenance = verify_trace_provenance(
        trace_manifest,
        trace_path=trace_path,
        config_path=args.config,
        student_checkpoint=args.student_checkpoint,
        ranker_checkpoint=args.ranker_checkpoint,
        factor_graph_checkpoint=args.factor_graph_checkpoint,
        neighbor_topk=int(args.neighbor_topk),
        target_topk=int(args.target_topk),
        coverage_fraction=float(args.coverage_fraction),
        cold_rounds=int(args.cold_rounds),
        warm_rounds=int(args.warm_rounds),
        warm_top_m=int(args.warm_top_m),
    )
    cfg = load_config(args.config)
    if int(args.feedback_horizon) < 0:
        raise ValueError("feedback horizon must be non-negative")
    sensitivity_multipliers = _parse_nonnegative_floats(
        args.channel_sensitivity_multipliers)
    snr_resolution_db = float(args.snr_resolution_db)
    latency_resolution_s = float(args.latency_resolution_ms) * 1.0e-3
    if not np.isfinite(snr_resolution_db) or snr_resolution_db <= 0.0:
        raise ValueError("SNR resolution must be finite and positive")
    if not np.isfinite(latency_resolution_s) or latency_resolution_s <= 0.0:
        raise ValueError("latency resolution must be finite and positive")
    audit_payload = json.loads(
        Path(args.event_audit).read_text(encoding="utf-8"))
    event = audit_payload["event"]
    if int(event["episode_seed"]) != int(args.seed):
        raise ValueError("event audit seed does not match --seed")
    if int(event["frame"]) != int(args.event_frame):
        raise ValueError("event audit frame does not match --event-frame")
    baseline_pd = np.asarray(event["baseline_pd"], dtype=np.float64)
    requested_indices = _parse_indices(args.candidate_indices)
    records = {
        int(record["candidate_index"]): record
        for record in event["candidates"]
        if bool(record["is_atomic_candidate"])
        and float(record["delta_worst"]) > 1.0e-4
        and tail_deficit_dominates(
            np.asarray(record["pd"], dtype=np.float64),
            baseline_pd,
            p_d_floor=float(args.p_d_floor),
        )
    }
    if requested_indices is not None:
        missing = sorted(requested_indices - set(records))
        if missing:
            raise ValueError(
                f"requested candidates are absent or not positive tail-safe: {missing}")
        records = {index: records[index] for index in sorted(requested_indices)}
    if not records:
        raise ValueError("event has no selected positive tail-safe candidates")

    student = FrozenStructureStudent(args.student_checkpoint)
    with np.load(trace_path, allow_pickle=False) as trace:
        rows = _trace_rows(trace, int(args.seed))
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
        slices = _observation_slices(env)
        observations, _ = env.reset(seed=int(args.seed))
        qos_floor = float(getattr(cfg.marl, "comm_qos_worst_min", 0.60))
        observation_max_error = 0.0
        physical_pd_max_error = 0.0
        event_row = None
        pre_step_env = None
        try:
            for frame in range(1, int(args.event_frame) + 1):
                if frame not in rows:
                    raise ValueError(f"trace is missing prefix frame {frame}")
                row = rows[frame]
                actual_obs = np.stack([
                    observations[str(k)] for k in range(env.core.K)])
                observation_max_error = max(
                    observation_max_error,
                    float(np.max(np.abs(
                        actual_obs - np.asarray(trace["local_obs"][row])))),
                )
                _submit_frozen_controller(
                    env,
                    trace,
                    row,
                    student,
                    slices,
                    neighbor_topk=int(args.neighbor_topk),
                    target_topk=int(args.target_topk),
                    coverage_fraction=float(args.coverage_fraction),
                    qos_floor=qos_floor,
                )
                if frame == int(args.event_frame):
                    pre_step_env = deepcopy(env)
                observations, _, terminated, truncated, info = env.step(
                    _actions(trace, row))
                physical_pd_max_error = max(
                    physical_pd_max_error,
                    float(np.max(np.abs(
                        np.asarray(info["P_D_q"])
                        - np.asarray(trace["physical_pd"][row])))),
                )
                if frame == int(args.event_frame):
                    event_row = int(row)
                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    raise RuntimeError("baseline ended before the event")

            if event_row is None or pre_step_env is None:
                raise RuntimeError("failed to capture event row")
            if observation_max_error > float(
                args.max_prefix_observation_error
            ):
                raise RuntimeError(
                    "prefix observation replay is incompatible with the trace: "
                    f"max error {observation_max_error:.6g}")
            if physical_pd_max_error > float(args.max_prefix_physical_pd_error):
                raise RuntimeError(
                    "prefix physical detector replay is incompatible with the "
                    f"trace: max P_D error {physical_pd_max_error:.6g}")
            core = env.core
            if core._inter_uav_comm is None:
                raise RuntimeError("event has no physical U2U communication model")
            selected = np.asarray(
                core._cached_p0_solution.z_selected, dtype=bool).copy()
            state = core._dynamic_local_search_coordinator.get_state()
            role, owner = role_owner_from_structure(
                selected,
                fallback_role=np.asarray(state["role"], dtype=np.int8),
            )
            positions = np.stack([
                uav.get_state().pos for uav in core.uavs])
            exact_comm = np.asarray(
                core._current_comm_power_w, dtype=np.float64).copy()
            exact_sensing = np.asarray(
                core._current_sensing_power_w, dtype=np.float64).copy()
            total_power = float(core._isac_total_power_w)
            reserve_power = (
                float(getattr(cfg.marl, "comm_tx_power_w", 0.25))
                if args.reserve_power_w is None
                else float(args.reserve_power_w)
            )
            if not 0.0 <= reserve_power <= total_power:
                raise ValueError("reserve power must lie in [0,total power]")
            sensing_weights = np.asarray(
                trace["sensing_weights"][event_row], dtype=np.float64)
            weight_sum = np.sum(sensing_weights, axis=1, keepdims=True)
            if np.any(weight_sum <= 0.0):
                raise ValueError("event contains an empty sensing allocation")
            # Reapply the same simplex projection used by the controller;
            # float32 trace serialization must not create fictitious watt
            # violations in a prospective allocation.
            sensing_weights = sensing_weights / weight_sum
            layout = DependencyCommitLayout(core.K, core.Q)
            feedback_layout = OwnerFeedbackLayout(
                core.K, core.Q, horizon_frames=int(args.feedback_horizon))
            channel_contract = {
                "bandwidth_hz": float(core._inter_uav_comm.bandwidth_hz),
                "packet_deadline_s": float(core._inter_uav_comm.deadline_s),
                "processing_delay_s": float(
                    core._inter_uav_comm.processing_delay_s),
                "snr_threshold_db": float(
                    core._inter_uav_comm.snr_threshold_db),
                "control_frame_s": float(cfg.scenario.dt),
                "prospective_reserve_power_w": reserve_power,
                "snr_resolution_db": snr_resolution_db,
                "latency_resolution_s": latency_resolution_s,
            }
            candidate_rows = []
            for index, record in sorted(records.items()):
                move = _all_target_atomic_move(env, index)
                changed_edges = int(np.count_nonzero(selected ^ move.selected))
                changed_roles = int(np.count_nonzero(role != move.role))
                changed_owners = int(np.count_nonzero(owner != move.owner))
                expected = (
                    int(record["changed_edges"]),
                    int(record["changed_roles"]),
                    int(record["changed_owners"]),
                )
                if (changed_edges, changed_roles, changed_owners) != expected:
                    raise AssertionError(
                        "candidate reconstruction disagrees with event audit")
                diagnostic_epoch_id = 0
                candidate_digest = _candidate_digest(
                    move,
                    state_version=int(args.event_frame),
                    epoch_id=diagnostic_epoch_id,
                )
                certificate_epoch_ids = np.full(
                    core.K, diagnostic_epoch_id, dtype=np.int64)
                certificate_digests = np.full(
                    core.K, candidate_digest, dtype=np.uint64)

                exact = certify_best_dependency_commit(
                    selected,
                    role,
                    owner,
                    move,
                    positions=positions,
                    comm_power_w=exact_comm,
                    sensing_power_w=exact_sensing,
                    state_versions=np.full(
                        core.K, int(args.event_frame), dtype=np.int64),
                    certificate_epoch_ids=certificate_epoch_ids,
                    certificate_digests=certificate_digests,
                    communication_model=core._inter_uav_comm,
                    total_power_w=total_power,
                    total_deadline_s=float(cfg.scenario.dt),
                    layout=layout,
                )
                prospective_comm = exact_comm.copy()
                prospective_sensing = exact_sensing.copy()
                for participant in exact.closure.participants:
                    prospective_comm[participant] = max(
                        prospective_comm[participant], reserve_power)
                    prospective_sensing[participant] = (
                        total_power - prospective_comm[participant]
                    ) * sensing_weights[participant]
                prospective = certify_best_dependency_commit(
                    selected,
                    role,
                    owner,
                    move,
                    positions=positions,
                    comm_power_w=prospective_comm,
                    sensing_power_w=prospective_sensing,
                    state_versions=np.full(
                        core.K, int(args.event_frame), dtype=np.int64),
                    certificate_epoch_ids=certificate_epoch_ids,
                    certificate_digests=certificate_digests,
                    communication_model=core._inter_uav_comm,
                    total_power_w=total_power,
                    total_deadline_s=float(cfg.scenario.dt),
                    layout=layout,
                )
                minimum_reserve = minimum_uniform_control_reserve(
                    selected,
                    role,
                    owner,
                    move,
                    positions=positions,
                    current_comm_power_w=exact_comm,
                    current_sensing_power_w=exact_sensing,
                    sensing_weights=sensing_weights,
                    state_versions=np.full(
                        core.K, int(args.event_frame), dtype=np.int64),
                    certificate_epoch_ids=certificate_epoch_ids,
                    certificate_digests=certificate_digests,
                    communication_model=core._inter_uav_comm,
                    reserve_upper_w=reserve_power,
                    tolerance_w=1.0e-9,
                    total_power_w=total_power,
                    total_deadline_s=float(cfg.scenario.dt),
                    layout=layout,
                )
                if not minimum_reserve.feasible:
                    raise RuntimeError(
                        "fixed reserve is feasible but minimum-reserve solver failed")
                minimum_comm = np.asarray(
                    minimum_reserve.comm_power_w, dtype=np.float64)
                minimum_sensing = np.asarray(
                    minimum_reserve.sensing_power_w, dtype=np.float64)
                feedback_entries: dict[int, int] = {}
                entries_per_target = int(args.feedback_horizon) + 1
                for target_owner in np.asarray(move.owner, dtype=np.int64):
                    if 0 <= int(target_owner) < core.K:
                        feedback_entries[int(target_owner)] = (
                            feedback_entries.get(int(target_owner), 0)
                            + entries_per_target
                        )
                feedback_coordinator = int(
                    minimum_reserve.certificate.proposer)
                fixed_feedback = certify_owner_feedback_transport(
                    feedback_entries,
                    coordinator=feedback_coordinator,
                    positions=positions,
                    comm_power_w=prospective_comm,
                    sensing_power_w=prospective_sensing,
                    state_versions=np.full(
                        core.K, int(args.event_frame), dtype=np.int64),
                    certificate_epoch_ids=certificate_epoch_ids,
                    certificate_digests=certificate_digests,
                    communication_model=core._inter_uav_comm,
                    layout=feedback_layout,
                    total_power_w=total_power,
                )
                minimum_feedback = certify_owner_feedback_transport(
                    feedback_entries,
                    coordinator=feedback_coordinator,
                    positions=positions,
                    comm_power_w=minimum_comm,
                    sensing_power_w=minimum_sensing,
                    state_versions=np.full(
                        core.K, int(args.event_frame), dtype=np.int64),
                    certificate_epoch_ids=certificate_epoch_ids,
                    certificate_digests=certificate_digests,
                    communication_model=core._inter_uav_comm,
                    layout=feedback_layout,
                    total_power_w=total_power,
                )
                joint_reserve = minimum_joint_commit_feedback_reserve(
                    selected,
                    role,
                    owner,
                    move,
                    feedback_entries_by_owner=feedback_entries,
                    positions=positions,
                    current_comm_power_w=exact_comm,
                    current_sensing_power_w=exact_sensing,
                    sensing_weights=sensing_weights,
                    state_versions=np.full(
                        core.K, int(args.event_frame), dtype=np.int64),
                    certificate_epoch_ids=certificate_epoch_ids,
                    certificate_digests=certificate_digests,
                    communication_model=core._inter_uav_comm,
                    commit_layout=layout,
                    feedback_layout=feedback_layout,
                    reserve_upper_w=reserve_power,
                    total_power_w=total_power,
                    total_deadline_s=float(cfg.scenario.dt),
                    tolerance_w=1.0e-9,
                )
                if not joint_reserve.feasible:
                    raise RuntimeError(
                        "fixed reserve is feasible but joint reserve solver failed")
                joint_comm = np.asarray(
                    joint_reserve.comm_power_w, dtype=np.float64)
                joint_sensing = np.asarray(
                    joint_reserve.sensing_power_w, dtype=np.float64)
                # Joint same-state replay of the sensing-power consequence.
                # The dummy learned rate only activates the environment's
                # exact power projection; it supplies no detection evidence.
                reserve_replay = _same_state_detection_replay(
                    pre_step_env=pre_step_env,
                    move=move,
                    participants=exact.closure.participants,
                    comm_power_w=joint_comm,
                    sensing_power_w=joint_sensing,
                    total_power_w=total_power,
                    trace=trace,
                    event_row=event_row,
                    baseline_pd=baseline_pd,
                    p_d_floor=float(args.p_d_floor),
                )

                robust_margin_curve = []
                curve_results = []
                for multiplier in sensitivity_multipliers:
                    snr_margin = multiplier * snr_resolution_db
                    latency_margin = multiplier * latency_resolution_s
                    curve_result = minimum_joint_commit_feedback_reserve(
                        selected,
                        role,
                        owner,
                        move,
                        feedback_entries_by_owner=feedback_entries,
                        positions=positions,
                        current_comm_power_w=exact_comm,
                        current_sensing_power_w=exact_sensing,
                        sensing_weights=sensing_weights,
                        state_versions=np.full(
                            core.K, int(args.event_frame), dtype=np.int64),
                        certificate_epoch_ids=certificate_epoch_ids,
                        certificate_digests=certificate_digests,
                        communication_model=core._inter_uav_comm,
                        commit_layout=layout,
                        feedback_layout=feedback_layout,
                        reserve_upper_w=reserve_power,
                        total_power_w=total_power,
                        total_deadline_s=float(cfg.scenario.dt),
                        tolerance_w=1.0e-9,
                        snr_margin_db=snr_margin,
                        latency_margin_s=latency_margin,
                    )
                    curve_results.append(curve_result)
                    curve_sensing = np.asarray(
                        curve_result.sensing_power_w, dtype=np.float64)
                    robust_margin_curve.append({
                        "multiplier": multiplier,
                        "snr_margin_db": snr_margin,
                        "latency_margin_s": latency_margin,
                        "feasible_at_reserve_upper": curve_result.feasible,
                        "uniform_comm_floor_w": (
                            curve_result.uniform_comm_floor_w),
                        "coordinator": curve_result.coordinator,
                        "iterations": curve_result.iterations,
                        "commit_latency_s": (
                            curve_result.commit_certificate.total_latency_s),
                        "feedback_latency_s": (
                            curve_result.feedback_certificate.max_latency_s),
                        "commit_energy_j": (
                            curve_result.commit_certificate.total_energy_j),
                        "feedback_energy_j": (
                            curve_result.feedback_certificate.total_energy_j),
                        "feedback_min_snr_db": (
                            curve_result.feedback_certificate.min_snr_db),
                        "maximum_sensing_power_reduction_w": float(np.max(
                            exact_sensing - curve_sensing)),
                    })
                finite_curve_floors = [
                    float(item.uniform_comm_floor_w)
                    for item in curve_results
                    if item.feasible and item.uniform_comm_floor_w is not None
                ]
                if any(
                    right + 1.0e-9 < left
                    for left, right in zip(
                        finite_curve_floors, finite_curve_floors[1:])
                ):
                    raise AssertionError(
                        "robust reserve violates physical margin monotonicity")
                terminal_curve_result = curve_results[-1]
                if terminal_curve_result.feasible:
                    terminal_comm = np.asarray(
                        terminal_curve_result.comm_power_w, dtype=np.float64)
                    terminal_sensing = np.asarray(
                        terminal_curve_result.sensing_power_w, dtype=np.float64)
                    terminal_replay = _same_state_detection_replay(
                        pre_step_env=pre_step_env,
                        move=move,
                        participants=exact.closure.participants,
                        comm_power_w=terminal_comm,
                        sensing_power_w=terminal_sensing,
                        total_power_w=total_power,
                        trace=trace,
                        event_row=event_row,
                        baseline_pd=baseline_pd,
                        p_d_floor=float(args.p_d_floor),
                    )
                else:
                    terminal_replay = None
                candidate_rows.append({
                    "candidate_index": index,
                    "delta_worst": float(record["delta_worst"]),
                    "tail_safe_exact_pd": True,
                    "exact_executed_allocation": _certificate_dict(exact),
                    "prospective_control_reserve": _certificate_dict(prospective),
                    "minimum_control_reserve": {
                        "uniform_comm_floor_w": (
                            minimum_reserve.uniform_comm_floor_w),
                        "iterations": minimum_reserve.iterations,
                        "certificate": _certificate_dict(
                            minimum_reserve.certificate),
                        "comm_power_w": list(minimum_reserve.comm_power_w),
                        "sensing_power_w": [
                            list(row) for row in minimum_reserve.sensing_power_w
                        ],
                    },
                    "owner_feedback": {
                        "horizon_frames": int(args.feedback_horizon),
                        "entries_by_owner": {
                            str(owner_id): entries
                            for owner_id, entries in sorted(
                                feedback_entries.items())
                        },
                        "same_state_nominal_diagnostic": True,
                        "fixed_reserve_transport": (
                            _feedback_certificate_dict(fixed_feedback)),
                        "minimum_commit_reserve_transport": (
                            _feedback_certificate_dict(minimum_feedback)),
                        "minimum_joint_reserve": {
                            "uniform_comm_floor_w": (
                                joint_reserve.uniform_comm_floor_w),
                            "coordinator": joint_reserve.coordinator,
                            "iterations": joint_reserve.iterations,
                            "commit_certificate": _certificate_dict(
                                joint_reserve.commit_certificate),
                            "feedback_certificate": _feedback_certificate_dict(
                                joint_reserve.feedback_certificate),
                            "comm_power_w": list(joint_reserve.comm_power_w),
                            "sensing_power_w": [
                                list(row) for row in joint_reserve.sensing_power_w
                            ],
                        },
                    },
                    "prospective_same_state_detection_replay": reserve_replay,
                    "precalibration_channel_margin_sensitivity": {
                        "resolution_scales_must_be_frozen_before_calibration": True,
                        "calibration_status": (
                            "diagnostic only; beta must be replaced by an "
                            "event-level split-conformal quantile"
                        ),
                        "curve": robust_margin_curve,
                        "maximum_multiplier_detection_replay": terminal_replay,
                    },
                    "original_detection_label_reusable": bool(
                        np.allclose(joint_sensing, exact_sensing)),
                })
        finally:
            if pre_step_env is not None:
                pre_step_env.close()
            env.close()

    quantization_step = (
        1.0 / ((1 << int(layout.lower_bound_bits)) - 1)
        if layout.lower_bound_bits > 0 else 1.0
    )
    output_payload = {
        "schema_version": 4,
        "protocol": "gate_d0_10_fail_closed_bound_transport_v4",
        "scope": "development event; no final test seed",
        "event": {"seed": int(args.seed), "frame": int(args.event_frame)},
        "trace_provenance": trace_provenance,
        "prefix_replay": {
            "observation_max_error": observation_max_error,
            "physical_pd_max_error": physical_pd_max_error,
        },
        "wire_layout": {
            "header_bits": layout.header_bits,
            "epoch_bits": layout.epoch_bits,
            "digest_bits": layout.digest_bits,
            "lower_bound_bits": layout.lower_bound_bits,
            "uncertainty_bits": layout.uncertainty_bits,
            "lower_bound_quantization_step": quantization_step,
            "safe_quantizer": "floor to the lower grid endpoint",
        },
        "channel_contract": channel_contract,
        "feedback_wire_layout": {
            "horizon_frames": feedback_layout.horizon_frames,
            "shared_bits": feedback_layout.shared_bits,
            "entry_bits": feedback_layout.entry_bits,
            "maximum_entries_per_packet": feedback_layout.maximum_entries,
            "timing": (
                "bundled after H; available only to a later calibration epoch"
            ),
        },
        "hard_contract": {
            "per_link": (
                "robust SNR >= threshold and robust latency <= packet deadline; "
                "Shannon rate is recomputed after the SNR margin"
            ),
            "end_to_end": "prepare + vote + decision latency <= one control frame",
            "power": "communication power + target-summed sensing power <= 1 W per UAV",
            "failure": "any violation returns No-op",
            "missing_required_observation": (
                "infinite event score; never removed from conformal calibration"
            ),
            "latency_margin_semantics": (
                "non-Shannon queue/scheduling/processing excess only; observed-SNR "
                "serialization is removed before residual calibration"
            ),
            "identity_binding": (
                "all closure participants must agree on state version, frozen "
                "certificate epoch and 64-bit proposal digest"
            ),
        },
        "risk_accounting": {
            "deployment_event_score": (
                "max(transition_event_score, channel_event_score)"
            ),
            "single_split_conformal_multiplier": True,
            "independence_assumption": False,
            "warning": (
                "separate 5% transition and 5% channel certificates imply "
                "only a direct 10% union-bound guarantee"
            ),
        },
        "candidate_rows": candidate_rows,
        "decision": {
            "exact_executed_allocation": (
                "eligible only when both physical commit and H-step transition certificates pass"
            ),
            "prospective_reserve": (
                "transport is admissible only together with the recomputed same-state detection label; "
                "H-step transition calibration remains mandatory"
            ),
            "feedback_epoch": (
                "not calibrated: this event is mechanism evidence only; "
                "feedback may update a later event-disjoint frozen epoch"
            ),
            "channel_margin": (
                "sensitivity curve is not a deployable fixed margin; collect "
                "independent event-level joint link errors and freeze the "
                "split-conformal multiplier before controller integration"
            ),
            "deployment": "default-off",
        },
        "fresh_test_consumed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "candidate_count": len(candidate_rows),
        "rows": [{
            "candidate_index": row["candidate_index"],
            "delta_worst": row["delta_worst"],
            "exact_feasible": row["exact_executed_allocation"]["feasible"],
            "reserve_feasible": row["prospective_control_reserve"]["feasible"],
            "minimum_reserve_w": row["minimum_control_reserve"][
                "uniform_comm_floor_w"],
            "robust_curve_w": [
                point["uniform_comm_floor_w"]
                for point in row[
                    "precalibration_channel_margin_sensitivity"]["curve"]
            ],
        } for row in candidate_rows],
    }, indent=2))


if __name__ == "__main__":
    main()
