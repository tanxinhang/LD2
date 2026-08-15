#!/usr/bin/env python
"""Export Gate C1.6 move-ranking labels along oracle local-search paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.train_factor_graph_coordinator import (  # noqa: E402
    _aligned_data,
    _edge_values,
    _load_npz,
)
from uav_isac.coordination.local_exchange_oracle import (  # noqa: E402
    enumerate_local_moves,
    objective_strictly_improves,
    rebuild_structure,
    role_first_initial_structure,
    role_owner_from_structure,
    structure_objective_key,
)
from uav_isac.coordination.local_move_ranker import (  # noqa: E402
    FEATURE_NAMES,
    local_move_features,
)


MOVE_KIND = {"N1": 0, "N2": 1, "N3": 2, "N5": 3}


def _rank_targets(keys: list[tuple[float, ...]]) -> np.ndarray:
    order = sorted(
        range(len(keys)), key=lambda index: tuple(keys[index][:4]),
        reverse=True)
    target = np.zeros(len(keys), dtype=np.float32)
    if len(keys) == 1:
        target[0] = 1.0
        return target
    for rank, index in enumerate(order):
        target[index] = 1.0 - float(rank) / float(len(keys) - 1)
    return target


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    data, candidate = _aligned_data(args.trace, args.candidate)
    candidate_mask = candidate["candidate_mask"].astype(bool)
    initial_sequence: np.ndarray | None = None
    if args.initial == "factor_graph":
        artifact = _load_npz(args.initial_artifact)
        initial_sequence = artifact[args.initial_key].astype(bool)
        if initial_sequence.shape != candidate_mask.shape:
            raise ValueError("factor-graph initial sequence does not align")
    edge_value = _edge_values(data, candidate, args.student_checkpoint)
    frames, K, _, _ = candidate_mask.shape
    previous_final = np.zeros_like(candidate_mask[0], dtype=bool)
    previous_role: np.ndarray | None = None
    previous_episode: int | None = None
    feature_rows: list[np.ndarray] = []
    positive_rows: list[np.ndarray] = []
    best_rows: list[np.ndarray] = []
    rank_rows: list[np.ndarray] = []
    delta_rows: list[np.ndarray] = []
    kind_rows: list[np.ndarray] = []
    group_ptr = [0]
    group_seed: list[int] = []
    group_episode: list[int] = []
    group_frame: list[int] = []
    group_round: list[int] = []
    group_has_positive: list[bool] = []
    accepted_kind_counts = {kind: 0 for kind in MOVE_KIND}
    positive_groups = 0
    for frame in range(frames):
        episode = int(data["episode"][frame])
        if episode != previous_episode:
            previous_final.fill(False)
            previous_role = None
            previous_episode = episode
        if not bool(data["p0_resolved"][frame]):
            continue
        if not np.any(candidate_mask[frame]):
            continue
        role_first, fallback_role, _ = role_first_initial_structure(
            edge_value[frame],
            candidate_mask[frame],
            target_pair_limit=int(data["target_pair_limit"][0]),
            reports_per_receiver=int(data["reports_per_receiver"][0]),
        )
        if args.initial == "factor_graph":
            assert initial_sequence is not None
            selected = initial_sequence[frame].copy()
            initial_role = fallback_role
        elif args.initial == "role_first":
            selected = role_first
            initial_role = fallback_role
        elif args.initial == "previous":
            if np.any(previous_final) and previous_role is not None:
                _, previous_owner = role_owner_from_structure(
                    previous_final, fallback_role=previous_role)
                selected = rebuild_structure(
                    edge_value[frame],
                    candidate_mask[frame],
                    previous_role,
                    previous_owner,
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(
                        data["reports_per_receiver"][0]),
                )
                initial_role = previous_role
            else:
                selected = role_first
                initial_role = fallback_role
        else:
            raise ValueError(f"unknown initial: {args.initial}")
        role, owner = role_owner_from_structure(
            selected, fallback_role=initial_role)
        if np.any(owner < 0):
            _, _, fallback_owner = role_first_initial_structure(
                edge_value[frame],
                candidate_mask[frame],
                target_pair_limit=int(data["target_pair_limit"][0]),
                reports_per_receiver=int(data["reports_per_receiver"][0]),
            )
            owner = np.where(owner >= 0, owner, fallback_owner)
            selected = rebuild_structure(
                edge_value[frame],
                candidate_mask[frame],
                role,
                owner,
                target_pair_limit=int(data["target_pair_limit"][0]),
                reports_per_receiver=int(data["reports_per_receiver"][0]),
            )
        for round_index in range(max(0, int(args.rounds))):
            moves = enumerate_local_moves(
                selected,
                edge_value[frame],
                candidate_mask[frame],
                role,
                owner,
                neighborhoods=tuple(args.neighborhoods),
                target_pair_limit=int(data["target_pair_limit"][0]),
                reports_per_receiver=int(data["reports_per_receiver"][0]),
            )
            if not moves:
                break
            current_key = structure_objective_key(
                selected, edge_value[frame])
            exact_keys = [
                structure_objective_key(move.selected, edge_value[frame])
                for move in moves
            ]
            positive = np.asarray([
                objective_strictly_improves(key, current_key)
                for key in exact_keys
            ], dtype=bool)
            best = np.zeros(len(moves), dtype=bool)
            if np.any(positive):
                best_key = max(
                    key for key, is_positive in zip(exact_keys, positive)
                    if is_positive)
                best_prefix = tuple(best_key[:4])
                best = np.asarray([
                    is_positive and tuple(key[:4]) == best_prefix
                    for key, is_positive in zip(exact_keys, positive)
                ], dtype=bool)
                positive_groups += 1
            features = np.stack([
                local_move_features(
                    selected, role, owner, move, edge_value[frame])
                for move in moves
            ])
            physical_delta = np.asarray([
                np.asarray(key[:4], dtype=np.float64)
                - np.asarray(current_key[:4], dtype=np.float64)
                for key in exact_keys
            ], dtype=np.float32)
            feature_rows.append(features)
            positive_rows.append(positive.astype(np.uint8))
            best_rows.append(best.astype(np.uint8))
            rank_rows.append(_rank_targets(exact_keys))
            delta_rows.append(physical_delta)
            kind_rows.append(np.asarray([
                MOVE_KIND[move.kind] for move in moves], dtype=np.int8))
            group_ptr.append(group_ptr[-1] + len(moves))
            group_seed.append(int(data["seed"][frame]))
            group_episode.append(episode)
            group_frame.append(int(data["frame"][frame]))
            group_round.append(round_index)
            group_has_positive.append(bool(np.any(positive)))
            if not np.any(positive):
                break
            best_index = max(
                (index for index in range(len(moves)) if positive[index]),
                key=lambda index: exact_keys[index],
            )
            accepted = moves[best_index]
            selected = accepted.selected
            role = accepted.role
            owner = accepted.owner
            accepted_kind_counts[accepted.kind] += 1
        previous_final[:] = selected
        previous_role = role.copy()
    if not feature_rows:
        raise RuntimeError("no local move groups were exported")
    features = np.concatenate(feature_rows, axis=0).astype(np.float32)
    positive = np.concatenate(positive_rows).astype(np.uint8)
    best = np.concatenate(best_rows).astype(np.uint8)
    rank_target = np.concatenate(rank_rows).astype(np.float32)
    objective_delta = np.concatenate(delta_rows).astype(np.float32)
    move_kind = np.concatenate(kind_rows).astype(np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        positive=positive,
        best=best,
        rank_target=rank_target,
        objective_delta=objective_delta,
        move_kind=move_kind,
        group_ptr=np.asarray(group_ptr, dtype=np.int64),
        group_seed=np.asarray(group_seed, dtype=np.int64),
        group_episode=np.asarray(group_episode, dtype=np.int64),
        group_frame=np.asarray(group_frame, dtype=np.int64),
        group_round=np.asarray(group_round, dtype=np.int16),
        group_has_positive=np.asarray(group_has_positive, dtype=np.uint8),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    result = {
        "protocol": "gate_c1_6_local_move_ranking_dataset_v1",
        "output": str(args.output),
        "initial": args.initial,
        "neighborhoods": list(args.neighborhoods),
        "rounds": args.rounds,
        "seeds": [int(value) for value in np.unique(data["seed"])],
        "groups": len(group_seed),
        "moves": int(len(features)),
        "features": int(features.shape[1]),
        "positive_groups": positive_groups,
        "positive_group_rate": float(np.mean(group_has_positive)),
        "positive_move_rate": float(np.mean(positive)),
        "best_move_rate": float(np.mean(best)),
        "moves_per_group_mean": float(len(features) / len(group_seed)),
        "accepted_kind_counts": accepted_kind_counts,
        "fresh_test_consumed": False,
        "runtime_s": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_teacher_trace_selection20/teacher_trace.npz"))
    parser.add_argument(
        "--candidate", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate10/per_frame.npz"))
    parser.add_argument(
        "--initial-artifact", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_factor_graph_gate_c1_screen/per_frame.npz"))
    parser.add_argument("--initial-key", type=str, default="train_prediction")
    parser.add_argument(
        "--student-checkpoint", type=Path, default=Path(
            "results/architecture_v2_teacher_cleanreset_trace_gate100/frozen_structure_student_endpoint8.pt"))
    parser.add_argument(
        "--output", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_move_rank_train10/previous_n123.npz"))
    parser.add_argument(
        "--initial", choices=("factor_graph", "role_first", "previous"),
        default="previous")
    parser.add_argument(
        "--neighborhoods", nargs="+", default=["N1", "N2", "N3"])
    parser.add_argument("--rounds", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
