#!/usr/bin/env python
"""Audit whether strictly local sparse candidates retain teacher solutions.

This is a mechanism gate, not a deployment evaluation.  The candidate selector
uses only pre-decision local observations and delivered target Tokens.  The
receiver-owner MILP is then restricted to the sparse union and serves only as
an audit upper bound for the future distributed negotiation protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
    build_structure_student_features,
)
from uav_isac.environment.observation_slices import (  # noqa: E402
    ObservationSlices,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    build_local_candidate_mask,
    candidate_equivalence_metrics,
    candidate_recall_metrics,
    delivered_target_tokens,
    episode_detection_summary,
    held_pair_sequence,
    realized_pd_history,
    receiver_owner_from_pairs,
    select_local_neighbors,
    select_value_guided_neighbors,
)


def _infer_observation_slices(obs_dim: int, K: int, Q: int) -> ObservationSlices:
    base_without_tokens = (
        8 + 9 * Q + 8 * Q + 3 + 8 * (K - 1) + Q + 16)
    token_count = (K - 1) * Q
    token_dim_numerator = obs_dim - base_without_tokens - token_count
    if token_count <= 0 or token_dim_numerator % token_count:
        raise ValueError("cannot infer target-token observation layout")
    token_dim = token_dim_numerator // token_count
    if token_dim < 6:
        raise ValueError("inferred target Token is too short")
    return ObservationSlices.from_config(
        K=K,
        Q=Q,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=True,
        comm_token_dim=token_dim,
        comm_tokens_per_sender=Q,
    )


def _subset_seed_bank(
    data: dict[str, np.ndarray],
    max_seeds: int,
    seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    seeds = np.unique(data["seed"])
    seeds = seeds[max(0, int(seed_offset)):]
    if int(max_seeds) > 0:
        seeds = seeds[:int(max_seeds)]
    if seeds.size == 0:
        raise ValueError("seed offset/limit selected no trace episodes")
    keep = np.isin(data["seed"], seeds)
    frame_count = len(data["seed"])
    return {
        key: (value[keep] if value.ndim and value.shape[0] == frame_count
              else value)
        for key, value in data.items()
    }


def _student_features(
    data: dict[str, np.ndarray],
    slices: ObservationSlices,
    *,
    neighbor_subset_mask: np.ndarray | None,
) -> np.ndarray:
    return build_structure_student_features(
        np.asarray(data["local_obs"], dtype=np.float32),
        slices,
        outgoing_message=data["outgoing_message"],
        outgoing_token_mask=data["outgoing_token_mask"],
        outgoing_rate=data["outgoing_rate"],
        comm_fraction=data["comm_fraction"],
        sensing_weights=data["sensing_weights"],
        rate_scale=max(float(np.max(data["outgoing_rate"])), 1.0),
        neighbor_subset_mask=neighbor_subset_mask,
    )


def _write_episode_csv(
    path: Path,
    rows_by_controller: dict[str, list[dict[str, object]]],
) -> None:
    fieldnames = [
        "controller", "episode", "seed", "steady", "weak3", "worst",
        "qos_feasible", "per_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for controller, rows in rows_by_controller.items():
            for row in rows:
                output = dict(row)
                output["controller"] = controller
                output["per_target"] = json.dumps(output["per_target"])
                writer.writerow(output)


def _estimated_candidate_bits(
    candidate_mask: np.ndarray,
    resolved: np.ndarray,
    *,
    score_bits: int,
    header_bits: int,
) -> dict[str, float]:
    candidate = np.asarray(candidate_mask, dtype=bool)
    F, K, _, Q = candidate.shape
    resolve_indices = np.flatnonzero(resolved)
    if resolve_indices.size == 0:
        return {"payload_per_resolve": 0.0, "total_per_resolve": 0.0,
                "amortized_per_frame": 0.0}
    peer_bits = max(1, int(np.ceil(np.log2(max(K, 2)))))
    target_bits = max(1, int(np.ceil(np.log2(max(Q, 2)))))
    tuple_bits = peer_bits + target_bits + max(1, int(score_bits))
    per_resolve = []
    for frame in resolve_indices:
        per_sender = np.sum(candidate[frame], axis=(1, 2))
        active = int(np.sum(per_sender > 0))
        payload = int(np.sum(per_sender)) * tuple_bits
        per_resolve.append((payload, payload + active * int(header_bits)))
    payload_mean = float(np.mean([value[0] for value in per_resolve]))
    total_mean = float(np.mean([value[1] for value in per_resolve]))
    resolve_rate = float(len(resolve_indices) / max(F, 1))
    return {
        "tuple_bits": float(tuple_bits),
        "payload_per_resolve": payload_mean,
        "total_per_resolve": total_mean,
        "amortized_per_frame": total_mean * resolve_rate,
    }


def _gap_recovery(
    teacher: dict[str, object],
    candidate: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in ("steady", "weak3", "worst", "cvar"):
        denominator = float(teacher[key]) - float(baseline[key])
        result[key] = (
            float((float(candidate[key]) - float(baseline[key]))
                  / denominator)
            if abs(denominator) > 1.0e-9 else None
        )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    with np.load(args.trace, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    data = _subset_seed_bank(data, args.max_seeds, args.seed_offset)
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])

    token_visible, token_age = delivered_target_tokens(
        data["local_obs"], slices)
    local_pd = slices.extract_pd_hist(data["local_obs"])
    local_neighbors = select_local_neighbors(
        token_visible,
        token_age,
        neighbor_topk=args.neighbor_topk,
    )
    candidate_student = FrozenStructureStudent(
        args.candidate_student_checkpoint)
    refinement_rounds = (
        max(1, int(args.neighbor_refinement_rounds))
        if args.neighbor_selection == "value_coverage" else 0
    )
    for _ in range(refinement_rounds):
        provisional_features = _student_features(
            data, slices, neighbor_subset_mask=local_neighbors)
        provisional_values = candidate_student.predict(provisional_features)
        local_neighbors = select_value_guided_neighbors(
            provisional_values,
            local_pd,
            token_visible,
            token_age,
            neighbor_topk=args.neighbor_topk,
            qos_floor=floor,
            coverage_fraction=args.coverage_fraction,
        )
    local_features = _student_features(
        data, slices, neighbor_subset_mask=local_neighbors)
    local_edge_values = candidate_student.predict(local_features)
    candidate_mask, target_selected = build_local_candidate_mask(
        local_edge_values,
        local_pd,
        local_neighbors,
        token_visible,
        target_topk=args.target_topk,
        qos_floor=floor,
        coverage_fraction=args.coverage_fraction,
        require_reciprocal_link=not args.allow_unilateral_link,
    )

    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    coord = np.where(
        np.asarray(data["coord_pd_ema_valid"], dtype=bool)[:, None],
        np.asarray(data["coord_pd_ema"], dtype=np.float64),
        np.mean(local_pd, axis=1),
    )
    deficit = np.maximum(floor - coord, 0.0)
    recall = candidate_recall_metrics(
        candidate_mask[resolved],
        data["privileged_candidate"][resolved],
        data["teacher_pair"][resolved],
        data["teacher_receiver_owner"][resolved],
        deficit[resolved],
    )
    warm_resolved = resolved.copy()
    for episode in np.unique(data["episode"]):
        episode_resolves = np.flatnonzero(
            resolved & (data["episode"] == episode))
        if episode_resolves.size:
            warm_resolved[episode_resolves[0]] = False
    warm_recall = (
        candidate_recall_metrics(
            candidate_mask[warm_resolved],
            data["privileged_candidate"][warm_resolved],
            data["teacher_pair"][warm_resolved],
            data["teacher_receiver_owner"][warm_resolved],
            deficit[warm_resolved],
        )
        if np.any(warm_resolved) else dict(recall)
    )
    physical = np.asarray(data["privileged_candidate"], dtype=bool)
    realized_d = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    full_replay_pairs = held_pair_sequence(
        realized_d,
        physical,
        data,
    )
    full_replay_owner = receiver_owner_from_pairs(full_replay_pairs)
    aligned_recall = candidate_recall_metrics(
        candidate_mask[resolved],
        physical[resolved],
        full_replay_pairs[resolved],
        full_replay_owner[resolved],
        deficit[resolved],
    )
    aligned_warm_recall = (
        candidate_recall_metrics(
            candidate_mask[warm_resolved],
            physical[warm_resolved],
            full_replay_pairs[warm_resolved],
            full_replay_owner[warm_resolved],
            deficit[warm_resolved],
        )
        if np.any(warm_resolved) else dict(aligned_recall)
    )
    pair_limit = int(np.asarray(
        data["target_pair_limit"]).reshape(-1)[0])
    equivalence = candidate_equivalence_metrics(
        candidate_mask[resolved],
        physical[resolved],
        full_replay_pairs[resolved],
        full_replay_owner[resolved],
        realized_d[resolved],
        target_pair_limit=pair_limit,
        evidence_ratio=args.evidence_ratio,
    )
    warm_equivalence = (
        candidate_equivalence_metrics(
            candidate_mask[warm_resolved],
            physical[warm_resolved],
            full_replay_pairs[warm_resolved],
            full_replay_owner[warm_resolved],
            realized_d[warm_resolved],
            target_pair_limit=pair_limit,
            evidence_ratio=args.evidence_ratio,
        )
        if np.any(warm_resolved) else dict(equivalence)
    )
    candidate_pairs = held_pair_sequence(
        realized_d,
        candidate_mask & physical,
        data,
    )
    controller_pairs: dict[str, np.ndarray] = {
        "trace_teacher": np.asarray(data["teacher_pair"], dtype=bool),
        "full_teacher": full_replay_pairs,
        "candidate_teacher": candidate_pairs,
    }
    if args.baseline:
        global_features = _student_features(
            data, slices, neighbor_subset_mask=None)
        for item in args.baseline:
            if "=" not in item:
                raise ValueError("--baseline must use NAME=CHECKPOINT")
            name, checkpoint = item.split("=", 1)
            name = name.strip()
            if not name or name in controller_pairs:
                raise ValueError("baseline name is empty or duplicated")
            student = FrozenStructureStudent(checkpoint.strip())
            ranking = student.predict(global_features)
            controller_pairs[name] = held_pair_sequence(
                ranking,
                physical,
                data,
            )

    summaries: dict[str, dict[str, object]] = {}
    episode_rows: dict[str, list[dict[str, object]]] = {}
    pd_histories: dict[str, np.ndarray] = {}
    for name, pairs in controller_pairs.items():
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
        episode_rows[name] = rows
        pd_histories[name] = pd

    teacher = summaries["full_teacher"]
    candidate = summaries["candidate_teacher"]
    pruning_gap = {
        key: float(teacher[key]) - float(candidate[key])
        for key in ("steady", "weak3", "worst", "cvar", "qos_feasible")
    }
    gap_recovery = {
        name: _gap_recovery(teacher, candidate, summary)
        for name, summary in summaries.items()
        if name not in {"trace_teacher", "full_teacher", "candidate_teacher"}
    }
    primary_recovery = None
    if args.baseline:
        primary_name = args.baseline[0].split("=", 1)[0].strip()
        primary_recovery = gap_recovery[primary_name]["worst"]

    gate_checks = {
        "edge_recall": recall["edge_recall"] >= args.min_edge_recall,
        "owner_recall": recall["owner_recall"] >= args.min_owner_recall,
        "target_empty_rate": (
            recall["target_empty_rate"] <= args.max_target_empty_rate),
        "mean_worst_pruning_gap": (
            pruning_gap["worst"] <= args.max_worst_pruning_gap),
        "cvar_pruning_gap": (
            pruning_gap["cvar"] <= args.max_cvar_pruning_gap),
        "oracle_gap_recovery": (
            primary_recovery is None
            or primary_recovery >= args.min_oracle_gap_recovery),
    }
    equivalence_gate_checks = {
        "equivalent_target_recall": (
            warm_equivalence["any_owner_equivalent_recall"]
            >= args.min_equivalent_target_recall),
        "owner_recall": (
            aligned_warm_recall["owner_recall"]
            >= args.min_owner_recall),
        "target_empty_rate": (
            aligned_warm_recall["target_empty_rate"]
            <= args.max_target_empty_rate),
        "mean_worst_pruning_gap": (
            pruning_gap["worst"] <= args.max_worst_pruning_gap),
        "cvar_pruning_gap": (
            pruning_gap["cvar"] <= args.max_cvar_pruning_gap),
        "oracle_gap_recovery": (
            primary_recovery is None
            or primary_recovery >= args.min_oracle_gap_recovery),
    }
    result: dict[str, object] = {
        "protocol": "local_candidate_sufficiency_v1",
        "trace": str(args.trace),
        "candidate_student_checkpoint": str(
            args.candidate_student_checkpoint),
        "num_uavs": K,
        "num_targets": Q,
        "frames": int(len(data["seed"])),
        "resolve_frames": int(np.sum(resolved)),
        "seeds": [int(value) for value in np.unique(data["seed"])],
        "candidate_config": {
            "neighbor_topk": int(args.neighbor_topk),
            "target_topk": int(args.target_topk),
            "coverage_fraction": float(args.coverage_fraction),
            "neighbor_selection": str(args.neighbor_selection),
            "neighbor_refinement_rounds": int(refinement_rounds),
            "require_reciprocal_link": not args.allow_unilateral_link,
            "evidence_ratio": float(args.evidence_ratio),
        },
        "recall": recall,
        "warm_recall_excluding_first_resolve": warm_recall,
        "aligned_full_replay_recall": aligned_recall,
        "aligned_warm_recall_excluding_first_resolve": (
            aligned_warm_recall),
        "equivalence_recall": equivalence,
        "warm_equivalence_recall_excluding_first_resolve": (
            warm_equivalence),
        "controllers": summaries,
        "pruning_gap_full_minus_candidate": pruning_gap,
        "oracle_gap_recovery": gap_recovery,
        "estimated_candidate_bits": _estimated_candidate_bits(
            candidate_mask,
            resolved,
            score_bits=args.score_bits,
            header_bits=args.header_bits,
        ),
        "local_neighbor_fraction": float(np.mean(local_neighbors)),
        "local_target_fraction": float(np.mean(target_selected)),
        "gate": {
            "checks": gate_checks,
            "pass": bool(all(gate_checks.values())),
            "thresholds": {
                "min_edge_recall": args.min_edge_recall,
                "min_owner_recall": args.min_owner_recall,
                "max_target_empty_rate": args.max_target_empty_rate,
                "max_worst_pruning_gap": args.max_worst_pruning_gap,
                "max_cvar_pruning_gap": args.max_cvar_pruning_gap,
                "min_oracle_gap_recovery": args.min_oracle_gap_recovery,
            },
        },
        "equivalence_gate_a2": {
            "scope": "warm candidate composability only",
            "checks": equivalence_gate_checks,
            "pass": bool(all(equivalence_gate_checks.values())),
            "thresholds": {
                "evidence_ratio": args.evidence_ratio,
                "min_equivalent_target_recall": (
                    args.min_equivalent_target_recall),
                "min_owner_recall": args.min_owner_recall,
                "max_target_empty_rate": args.max_target_empty_rate,
                "max_worst_pruning_gap": args.max_worst_pruning_gap,
                "max_cvar_pruning_gap": args.max_cvar_pruning_gap,
                "min_oracle_gap_recovery": args.min_oracle_gap_recovery,
            },
        },
        "end_to_end_protocol_gate": {
            "pass": False,
            "reason": "physical cold-start handshake not yet implemented",
        },
        "runtime_s": float(time.perf_counter() - started),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    _write_episode_csv(output_dir / "episode_metrics.csv", episode_rows)
    arrays = {
        "episode": data["episode"],
        "seed": data["seed"],
        "frame": data["frame"],
        "resolved": resolved.astype(np.uint8),
        "warm_resolved": warm_resolved.astype(np.uint8),
        "local_neighbors": local_neighbors.astype(np.uint8),
        "target_selected": target_selected.astype(np.uint8),
        "candidate_mask": candidate_mask.astype(np.uint8),
        "candidate_pairs": candidate_pairs.astype(np.uint8),
        "full_replay_pairs": full_replay_pairs.astype(np.uint8),
        "teacher_pairs": controller_pairs["trace_teacher"].astype(np.uint8),
    }
    arrays.update({
        f"pd_{name}": values for name, values in pd_histories.items()
    })
    np.savez_compressed(output_dir / "per_frame.npz", **arrays)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--candidate-student-checkpoint", type=Path,
                        required=True)
    parser.add_argument(
        "--baseline", action="append", default=[], metavar="NAME=CHECKPOINT",
        help="optional same-state full-graph Student control; first is the "
             "oracle-gap denominator")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--neighbor-topk", type=int, default=3)
    parser.add_argument("--target-topk", type=int, default=4)
    parser.add_argument("--coverage-fraction", type=float, default=0.30)
    parser.add_argument(
        "--neighbor-selection",
        choices=("availability", "value_coverage"),
        default="availability",
    )
    parser.add_argument("--neighbor-refinement-rounds", type=int, default=1)
    parser.add_argument("--allow-unilateral-link", action="store_true")
    parser.add_argument("--steady-window", type=int, default=20)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    parser.add_argument("--score-bits", type=int, default=8)
    parser.add_argument("--header-bits", type=int, default=64)
    parser.add_argument("--min-edge-recall", type=float, default=0.90)
    parser.add_argument("--evidence-ratio", type=float, default=0.95)
    parser.add_argument(
        "--min-equivalent-target-recall", type=float, default=0.90)
    parser.add_argument("--min-owner-recall", type=float, default=0.90)
    parser.add_argument("--max-target-empty-rate", type=float, default=0.02)
    parser.add_argument("--max-worst-pruning-gap", type=float, default=0.02)
    parser.add_argument("--max-cvar-pruning-gap", type=float, default=0.03)
    parser.add_argument("--min-oracle-gap-recovery", type=float, default=0.90)
    return parser.parse_args()


if __name__ == "__main__":
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
