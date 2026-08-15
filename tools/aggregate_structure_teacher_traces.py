#!/usr/bin/env python
"""Aggregate clean teacher and relabelled student-occupancy traces."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1]))

from tools.audit_structure_teacher_trace import (
    _replay_teacher_projection,
)
from uav_isac.evaluation.structure_teacher import (
    structure_teacher_labels,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _relabel_with_frozen_teacher(
    data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    result = {
        key: np.asarray(value).copy()
        for key, value in data.items()
    }
    resolved = np.asarray(result["p0_resolved"], dtype=bool)
    frame_indices = np.flatnonzero(resolved)
    replay = _replay_teacher_projection(
        result,
        resolved,
        frame_indices=frame_indices,
        return_pairs=True,
    )
    replay_pair = np.asarray(
        replay["_pair_matrix"], dtype=np.uint8)
    K = int(np.asarray(result["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(result["num_targets"]).reshape(-1)[0])
    for local_index, frame_index in enumerate(frame_indices):
        selected = [
            tuple(int(value) for value in edge)
            for edge in np.argwhere(replay_pair[local_index] > 0)
        ]
        labels = structure_teacher_labels(selected, K, Q)
        result["teacher_pair"][frame_index] = labels["pair"]
        result["teacher_tx_target"][frame_index] = labels["tx_target"]
        result["teacher_rx_target"][frame_index] = labels["rx_target"]
        result["teacher_endpoint_target"][frame_index] = (
            labels["endpoint_target"])
        result["teacher_receiver_owner"][frame_index] = (
            labels["receiver_owner"])
        result["teacher_role"][frame_index] = labels["role"]
    return result


def aggregate(
    base_path: Path,
    on_policy_path: Path,
    output_path: Path,
    on_policy_repeat: int,
) -> None:
    base = _load(base_path)
    on_policy = _relabel_with_frozen_teacher(
        _load(on_policy_path))
    if set(base) != set(on_policy):
        missing = sorted(set(base).symmetric_difference(on_policy))
        raise ValueError(
            f"trace schemas differ: {missing}")
    base_frames = len(base["seed"])
    on_policy_frames = len(on_policy["seed"])
    repeats = max(1, int(on_policy_repeat))
    output: dict[str, np.ndarray] = {}
    for key in sorted(base):
        base_value = np.asarray(base[key])
        on_policy_value = np.asarray(on_policy[key])
        is_frame_field = (
            base_value.ndim > 0
            and on_policy_value.ndim > 0
            and base_value.shape[0] == base_frames
            and on_policy_value.shape[0] == on_policy_frames
            and base_value.shape[1:] == on_policy_value.shape[1:]
        )
        if is_frame_field:
            output[key] = np.concatenate(
                [base_value]
                + [on_policy_value for _ in range(repeats)],
                axis=0,
            )
        else:
            if not np.array_equal(base_value, on_policy_value):
                raise ValueError(
                    f"static trace field differs: {key}")
            output[key] = base_value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print({
        "output": str(output_path),
        "base_frames": base_frames,
        "on_policy_frames": on_policy_frames,
        "on_policy_repeat": repeats,
        "total_frames": len(output["seed"]),
        "resolve_frames": int(np.sum(output["p0_resolved"])),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--on-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--on-policy-repeat", type=int, default=4)
    args = parser.parse_args()
    aggregate(
        args.base,
        args.on_policy,
        args.output,
        args.on_policy_repeat,
    )


if __name__ == "__main__":
    main()
