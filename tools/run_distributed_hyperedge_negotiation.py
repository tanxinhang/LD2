#!/usr/bin/env python
"""Replay finite-round distributed negotiation on audited local candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
    build_structure_student_features,
)
from uav_isac.coordination.finite_round_hyperedge import (  # noqa: E402
    gossip_candidate_views,
    initial_finite_round_state,
    negotiate_finite_round_hyperedges,
    solve_replicated_candidate_graph,
)
from uav_isac.environment.observation_slices import (  # noqa: E402
    ObservationSlices,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    episode_detection_summary,
    held_pair_sequence,
    realized_pd_history,
)


def _infer_observation_slices(obs_dim: int, K: int, Q: int) -> ObservationSlices:
    base_without_tokens = (
        8 + 9 * Q + 8 * Q + 3 + 8 * (K - 1) + Q + 16)
    token_count = (K - 1) * Q
    numerator = obs_dim - base_without_tokens - token_count
    if token_count <= 0 or numerator % token_count:
        raise ValueError("cannot infer target-token observation layout")
    return ObservationSlices.from_config(
        K=K,
        Q=Q,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=True,
        comm_token_dim=numerator // token_count,
        comm_tokens_per_sender=Q,
    )


def _subset_trace(
    data: dict[str, np.ndarray],
    seeds: np.ndarray,
) -> dict[str, np.ndarray]:
    keep = np.isin(data["seed"], seeds)
    frame_count = len(data["seed"])
    return {
        key: (value[keep] if value.ndim and value.shape[0] == frame_count
              else value)
        for key, value in data.items()
    }


def _protocol_bits(
    candidate_mask: np.ndarray,
    selected_counts: list[int],
    resolved: np.ndarray,
    *,
    rounds: int,
    price_bits: int,
    header_bits: int,
) -> dict[str, float]:
    F, K, _, Q = candidate_mask.shape
    resolve_indices = np.flatnonzero(resolved)
    if not len(resolve_indices):
        return {"candidate": 0.0, "negotiation": 0.0, "total": 0.0}
    node_bits = max(1, int(np.ceil(np.log2(max(K, 2)))))
    target_bits = max(1, int(np.ceil(np.log2(max(Q, 2)))))
    tuple_bits = 2 * node_bits + target_bits + int(price_bits)
    candidate_per_resolve = float(np.mean([
        int(header_bits) * int(np.any(candidate_mask[frame]))
        + int(np.sum(candidate_mask[frame])) * tuple_bits
        for frame in resolve_indices
    ]))
    fixed_round_bits = (
        Q * (node_bits + int(price_bits))
        + K * (2 * int(price_bits) + 1)
    )
    negotiation_per_resolve = float(np.mean([
        int(rounds) * (fixed_round_bits + count * tuple_bits)
        for count in selected_counts
    ]))
    resolve_rate = len(resolve_indices) / max(F, 1)
    return {
        "candidate_per_resolve": candidate_per_resolve,
        "negotiation_per_resolve": negotiation_per_resolve,
        "candidate_per_frame": candidate_per_resolve * resolve_rate,
        "negotiation_per_frame": negotiation_per_resolve * resolve_rate,
        "total_per_frame": (
            candidate_per_resolve + negotiation_per_resolve) * resolve_rate,
    }


def _gossip_diagnostics(
    candidate_mask: np.ndarray,
    resolved: np.ndarray,
    rounds: list[int],
) -> dict[str, dict[str, float]]:
    frames = [
        candidate_mask[frame]
        for frame in np.flatnonzero(resolved)
        if np.any(candidate_mask[frame])
    ]
    output = {}
    for count in rounds:
        values = [
            gossip_candidate_views(mask, rounds=count)[1:]
            for mask in frames
        ]
        output[f"R{count}"] = {
            "nonempty_resolves": len(values),
            "common_view_rate": float(np.mean([
                item[0] for item in values])) if values else 0.0,
            "mean_edge_agreement": float(np.mean([
                item[1] for item in values])) if values else 0.0,
        }
    return output


def _replay_rounds(
    rounds: int,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, float]]:
    F, K, _, Q = edge_value.shape
    pair_sequence = np.zeros((F, K, K, Q), dtype=bool)
    current = np.zeros((K, K, Q), dtype=bool)
    previous_episode: int | None = None
    state = initial_finite_round_state(K, Q)
    bootstrap_count = 0
    conflict_count = 0
    owner_changes = 0
    convergence = []
    selected_counts = []
    resolve_count = 0
    for frame in range(F):
        episode = int(data["episode"][frame])
        if previous_episode != episode:
            state = initial_finite_round_state(K, Q)
            current.fill(False)
            previous_episode = episode
        if bool(data["p0_resolved"][frame]):
            resolve_count += 1
            result = negotiate_finite_round_hyperedges(
                edge_value[frame],
                candidate_mask[frame],
                state,
                rounds=rounds,
                target_pair_limit=int(data["target_pair_limit"][0]),
                owner_hold_rounds=args.owner_hold,
                target_price_step=args.target_price_step,
                role_price_step=args.role_price_step,
                target_proxy_floor=args.target_proxy_floor,
                price_max=args.price_max,
                communication_cost=args.communication_cost,
                switch_cost=args.switch_cost,
                owner_hold_bonus=args.owner_hold_bonus,
            )
            state = result.state
            current.fill(False)
            for tx, rx, target in result.selected:
                current[tx, rx, target] = True
            bootstrap_count += int(result.bootstrap_required)
            conflict_count += int(result.role_conflicts)
            owner_changes += int(result.owner_changes)
            convergence.append(int(result.convergence_round))
            selected_counts.append(len(result.selected))
        pair_sequence[frame] = current
    diagnostics = {
        "bootstrap_resolve_rate": bootstrap_count / max(resolve_count, 1),
        "role_conflicts_per_resolve": conflict_count / max(resolve_count, 1),
        "owner_changes_per_resolve": owner_changes / max(resolve_count, 1),
        "mean_convergence_round": float(np.mean(convergence))
        if convergence else 0.0,
        "mean_selected_edges": float(np.mean(selected_counts))
        if selected_counts else 0.0,
    }
    diagnostics.update(_protocol_bits(
        candidate_mask,
        selected_counts,
        np.asarray(data["p0_resolved"], dtype=bool),
        rounds=rounds,
        price_bits=args.price_bits,
        header_bits=args.header_bits,
    ))
    return pair_sequence, diagnostics


def _replay_replicated_consensus(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    F, K, _, Q = edge_value.shape
    pair_sequence = np.zeros((F, K, K, Q), dtype=bool)
    current = np.zeros((K, K, Q), dtype=bool)
    previous_episode: int | None = None
    queue = np.zeros(Q, dtype=np.float64)
    owner = np.full(Q, -1, dtype=np.int64)
    qos_floor = float(data["qos_floor"][0])
    for frame in range(F):
        episode = int(data["episode"][frame])
        if previous_episode != episode:
            current.fill(False)
            queue.fill(0.0)
            owner.fill(-1)
            previous_episode = episode
        if bool(data["p0_resolved"][frame]):
            current.fill(False)
            local_pd = np.asarray(data["local_pd"][frame], dtype=np.float64)
            detection = np.max(local_pd, axis=0)
            for target in range(Q):
                if owner[target] >= 0:
                    detection[target] = local_pd[owner[target], target]
            queue = np.clip(
                queue + float(args.replicated_queue_step)
                * (qos_floor - detection),
                0.0,
                float(args.replicated_queue_max),
            )
            if np.any(candidate_mask[frame]):
                selected = solve_replicated_candidate_graph(
                    edge_value[frame],
                    candidate_mask[frame],
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(
                        data["reports_per_receiver"][0]),
                    target_price=1.0 + queue,
                    target_proxy_floor=args.target_proxy_floor,
                )
                selected_owner = np.full(Q, -1, dtype=np.int64)
                for tx, rx, target in selected:
                    current[tx, rx, target] = True
                    selected_owner[target] = rx
                owner = np.where(selected_owner >= 0, selected_owner, owner)
        pair_sequence[frame] = current
    return pair_sequence


def run(args: argparse.Namespace) -> dict[str, object]:
    with np.load(args.trace, allow_pickle=False) as loaded:
        raw = {key: loaded[key] for key in loaded.files}
    with np.load(args.candidate_per_frame, allow_pickle=False) as loaded:
        candidate = {key: loaded[key] for key in loaded.files}
    data = _subset_trace(raw, np.unique(candidate["seed"]))
    for key in ("episode", "seed", "frame"):
        if not np.array_equal(data[key], candidate[key]):
            raise ValueError("candidate artifact does not align with trace")
    K = int(data["num_uavs"][0])
    Q = int(data["num_targets"][0])
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    features = build_structure_student_features(
        data["local_obs"],
        slices,
        outgoing_message=data["outgoing_message"],
        outgoing_token_mask=data["outgoing_token_mask"],
        outgoing_rate=data["outgoing_rate"],
        comm_fraction=data["comm_fraction"],
        sensing_weights=data["sensing_weights"],
        rate_scale=max(float(np.max(data["outgoing_rate"])), 1.0),
        neighbor_subset_mask=candidate["local_neighbors"].astype(bool),
    )
    student = FrozenStructureStudent(args.student_checkpoint)
    edge_value = student.predict(features)
    candidate_mask = candidate["candidate_mask"].astype(bool)
    realized_d = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    p_fa = float(data["p_fa"][0])

    summaries: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    pair_outputs: dict[str, np.ndarray] = {}
    episode_rows = []
    for rounds in args.rounds:
        name = f"R{int(rounds)}"
        pairs, diag = _replay_rounds(
            int(rounds), edge_value, candidate_mask, data, args)
        pd = realized_pd_history(pairs, realized_d, p_fa)
        summary, rows = episode_detection_summary(
            pd,
            data["episode"],
            data["seed"],
            steady_window=args.steady_window,
            qos_thresholds=(args.steady_floor, args.weak3_floor,
                            args.worst_floor),
        )
        summaries[name] = summary
        diagnostics[name] = diag
        pair_outputs[f"pairs_{name}"] = pairs.astype(np.uint8)
        pair_outputs[f"pd_{name}"] = pd.astype(np.float32)
        for row in rows:
            episode_rows.append({"controller": name, **row})

    if args.include_replicated_control:
        replicated_pairs = _replay_replicated_consensus(
            edge_value, candidate_mask, data, args)
        replicated_pd = realized_pd_history(
            replicated_pairs, realized_d, p_fa)
        replicated_summary, rows = episode_detection_summary(
            replicated_pd,
            data["episode"],
            data["seed"],
            steady_window=args.steady_window,
            qos_thresholds=(args.steady_floor, args.weak3_floor,
                            args.worst_floor),
        )
        summaries["replicated_consensus"] = replicated_summary
        pair_outputs["pairs_replicated_consensus"] = (
            replicated_pairs.astype(np.uint8))
        pair_outputs["pd_replicated_consensus"] = (
            replicated_pd.astype(np.float32))
        for row in rows:
            episode_rows.append({"controller": "replicated_consensus", **row})

    controls = {}
    for name in ("full_teacher", "candidate_teacher"):
        key = f"pd_{name}"
        if key in candidate:
            controls[name], _ = episode_detection_summary(
                candidate[key],
                data["episode"],
                data["seed"],
                steady_window=args.steady_window,
                qos_thresholds=(args.steady_floor, args.weak3_floor,
                                args.worst_floor),
            )
    if args.include_central_student_control:
        student_pairs = held_pair_sequence(
            edge_value,
            candidate_mask & np.asarray(
                data["privileged_candidate"], dtype=bool),
            data,
        )
        student_pd = realized_pd_history(student_pairs, realized_d, p_fa)
        controls["candidate_student_projection"], _ = (
            episode_detection_summary(
                student_pd,
                data["episode"],
                data["seed"],
                steady_window=args.steady_window,
                qos_thresholds=(args.steady_floor, args.weak3_floor,
                                args.worst_floor),
            )
        )
        pair_outputs["pairs_candidate_student_projection"] = (
            student_pairs.astype(np.uint8))
        pair_outputs["pd_candidate_student_projection"] = (
            student_pd.astype(np.float32))
    # This controller is a fixed teacher/reference on the candidate graph.  It
    # is not a mathematical upper bound: another scorer/objective can exceed
    # it even on the same candidate set.
    candidate_reference = controls.get("candidate_teacher")
    gates = {}
    for name, summary in summaries.items():
        relative = None
        if candidate_reference is not None:
            relative = {
                key: float(candidate_reference[key]) - float(summary[key])
                for key in ("steady", "weak3", "worst", "cvar")
            }
        checks = {
            "steady": float(summary["steady"]) >= args.steady_floor,
            "weak3": float(summary["weak3"]) >= args.weak3_floor,
            "worst": float(summary["worst"]) >= args.worst_floor,
            "qos_feasible": (
                float(summary["qos_feasible"]) >= args.min_qos_feasible),
        }
        if relative is not None:
            checks["relative_worst_gap"] = (
                relative["worst"] <= args.max_coordination_gap)
            checks["relative_cvar_gap"] = (
                relative["cvar"] <= args.max_coordination_gap)
        gates[name] = {
            "checks": checks,
            "pass": bool(all(checks.values())),
            "candidate_reference_minus_protocol": relative,
        }

    result: dict[str, object] = {
        "protocol": "finite_round_distributed_hyperedge_v2",
        "trace": str(args.trace),
        "candidate_per_frame": str(args.candidate_per_frame),
        "student_checkpoint": str(args.student_checkpoint),
        "seeds": [int(value) for value in np.unique(data["seed"])],
        "config": {
            "rounds": [int(value) for value in args.rounds],
            "owner_hold": args.owner_hold,
            "target_price_step": args.target_price_step,
            "role_price_step": args.role_price_step,
            "target_proxy_floor": args.target_proxy_floor,
            "price_max": args.price_max,
            "communication_cost": args.communication_cost,
            "switch_cost": args.switch_cost,
            "replicated_queue_step": args.replicated_queue_step,
            "replicated_queue_max": args.replicated_queue_max,
        },
        "controls": controls,
        "candidate_reference_label": "candidate-restricted teacher reference",
        "protocol_results": summaries,
        "diagnostics": diagnostics,
        "candidate_gossip": _gossip_diagnostics(
            candidate_mask,
            np.asarray(data["p0_resolved"], dtype=bool),
            [int(value) for value in args.gossip_rounds],
        ),
        "gates": gates,
        "central_repair_used": False,
        "cold_start_policy": "fail_closed_until_candidate_token_arrival",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    with (output_dir / "episode_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fields = ["controller", "episode", "seed", "steady", "weak3",
                  "worst", "qos_feasible", "per_target"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in episode_rows:
            output = dict(row)
            output["per_target"] = json.dumps(output["per_target"])
            writer.writerow(output)
    np.savez_compressed(
        output_dir / "per_frame.npz",
        episode=data["episode"],
        seed=data["seed"],
        frame=data["frame"],
        **pair_outputs,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--candidate-per-frame", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--gossip-rounds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--owner-hold", type=int, default=5)
    parser.add_argument("--target-price-step", type=float, default=0.25)
    parser.add_argument("--role-price-step", type=float, default=0.15)
    parser.add_argument("--target-proxy-floor", type=float, default=0.80)
    parser.add_argument("--price-max", type=float, default=4.0)
    parser.add_argument("--communication-cost", type=float, default=0.02)
    parser.add_argument("--switch-cost", type=float, default=0.05)
    parser.add_argument("--owner-hold-bonus", type=float, default=1.0)
    parser.add_argument("--price-bits", type=int, default=8)
    parser.add_argument("--header-bits", type=int, default=64)
    parser.add_argument("--steady-window", type=int, default=20)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    parser.add_argument("--min-qos-feasible", type=float, default=0.70)
    parser.add_argument("--max-coordination-gap", type=float, default=0.01)
    parser.add_argument(
        "--include-central-student-control", action="store_true")
    parser.add_argument(
        "--include-replicated-control", action="store_true")
    parser.add_argument("--replicated-queue-step", type=float, default=0.25)
    parser.add_argument("--replicated-queue-max", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
