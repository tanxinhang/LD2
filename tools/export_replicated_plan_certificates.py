#!/usr/bin/env python
"""Export joint near-equivalent certificates from the replicated teacher."""

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
)
from uav_isac.coordination.finite_round_hyperedge import (  # noqa: E402
    solve_replicated_candidate_graph_certificate,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    data, candidate = _aligned_data(args.trace, args.candidate)
    edge_value = _edge_values(data, candidate, args.student_checkpoint)
    candidate_mask = candidate["candidate_mask"].astype(bool)
    seeds = np.unique(data["seed"])
    selected_seeds = seeds[
        int(args.seed_offset):int(args.seed_offset) + int(args.num_seeds)]
    resolve_indices = np.flatnonzero(
        np.isin(data["seed"], selected_seeds)
        & data["p0_resolved"].astype(bool)
        & np.any(candidate_mask, axis=(1, 2, 3)))
    K = candidate_mask.shape[1]
    Q = candidate_mask.shape[3]
    A = max(1, int(args.max_alternatives))
    selected = np.zeros((len(resolve_indices), A, K, K, Q), dtype=np.uint8)
    role = np.full((len(resolve_indices), A, K), -1, dtype=np.int8)
    owner = np.full((len(resolve_indices), A, Q), -1, dtype=np.int16)
    target_value = np.zeros((len(resolve_indices), A, Q), dtype=np.float32)
    weighted_worst = np.full((len(resolve_indices), A), np.nan, dtype=np.float32)
    weighted_sum = np.full((len(resolve_indices), A), np.nan, dtype=np.float32)
    count = np.zeros(len(resolve_indices), dtype=np.int16)
    for row, frame in enumerate(resolve_indices):
        certificate = solve_replicated_candidate_graph_certificate(
            edge_value[frame],
            candidate_mask[frame],
            target_pair_limit=int(data["target_pair_limit"][0]),
            reports_per_receiver=int(data["reports_per_receiver"][0]),
            max_alternatives=A,
            min_worst_ratio=args.min_worst_ratio,
            min_sum_ratio=args.min_sum_ratio,
        )
        count[row] = len(certificate.alternatives)
        for alternative_index, alternative in enumerate(
                certificate.alternatives):
            for edge in alternative.selected:
                selected[row, alternative_index][edge] = 1
            role[row, alternative_index] = alternative.role
            owner[row, alternative_index] = alternative.owner
            target_value[row, alternative_index] = alternative.target_value
            weighted_worst[row, alternative_index] = alternative.weighted_worst
            weighted_sum[row, alternative_index] = alternative.weighted_sum
        if (row + 1) % 10 == 0:
            print(
                f"certificate {row + 1}/{len(resolve_indices)}",
                flush=True,
            )
    valid_rows = np.arange(len(resolve_indices))
    role_diverse = np.array([
        len({tuple(item) for item in role[row, :count[row]]}) > 1
        for row in valid_rows
    ], dtype=bool)
    owner_diverse = np.array([
        len({tuple(item) for item in owner[row, :count[row]]}) > 1
        for row in valid_rows
    ], dtype=bool)
    result = {
        "protocol": "replicated_joint_certificate_v1",
        "seeds": [int(value) for value in selected_seeds],
        "resolve_frames": int(len(resolve_indices)),
        "thresholds": {
            "max_alternatives": A,
            "min_worst_ratio": args.min_worst_ratio,
            "min_sum_ratio": args.min_sum_ratio,
        },
        "alternative_count_mean": float(np.mean(count)) if len(count) else 0.0,
        "multiple_alternative_rate": float(np.mean(count > 1)) if len(count) else 0.0,
        "role_diversity_rate": float(np.mean(role_diverse)) if len(count) else 0.0,
        "owner_diversity_rate": float(np.mean(owner_diverse)) if len(count) else 0.0,
        "runtime_s": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "certificates.npz",
        resolve_index=resolve_indices.astype(np.int32),
        episode=data["episode"][resolve_indices],
        seed=data["seed"][resolve_indices],
        frame=data["frame"][resolve_indices],
        alternative_count=count,
        selected=selected,
        role=role,
        owner=owner,
        target_value=target_value,
        weighted_worst=weighted_worst,
        weighted_sum=weighted_sum,
    )
    (args.output_dir / "summary.json").write_text(
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
        "--student-checkpoint", type=Path, default=Path(
            "results/architecture_v2_teacher_cleanreset_trace_gate100/frozen_structure_student_endpoint8.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_joint_certificate_seed1"))
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--max-alternatives", type=int, default=8)
    parser.add_argument("--min-worst-ratio", type=float, default=0.95)
    parser.add_argument("--min-sum-ratio", type=float, default=0.95)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
