#!/usr/bin/env python
"""Audit a frozen structure student's endpoint quantization in same states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_structure_teacher_trace import (
    _edge_value_metrics,
    _infer_observation_slices,
    _per_target_features,
    _replay_teacher_projection,
)
from uav_isac.agents.frozen_structure_student import (
    FrozenStructureStudent,
)
from uav_isac.environment.communication import (
    InterUAVCommunicationModel,
)


def audit(
    trace_path: Path,
    checkpoint: Path,
    bits_per_dim: int,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    unique_seeds = np.unique(data["seed"])
    test_count = max(1, int(np.ceil(0.1 * unique_seeds.size)))
    test_seeds = unique_seeds[-test_count:]
    test_index = np.flatnonzero(
        resolved & np.isin(data["seed"], test_seeds))
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    slices = _infer_observation_slices(
        data["local_obs"].shape[-1], K, Q)
    features = _per_target_features(
        data, "local_obs", test_index, slices)
    student = FrozenStructureStudent(checkpoint)
    raw_protocol = student.encode_endpoint_protocol(features)
    quantized_protocol = (
        InterUAVCommunicationModel.quantize_values_at_bits(
            raw_protocol, bits_per_dim))
    raw_edges = student.predict(features)
    quantized_edges = student.predict_from_endpoint_protocol(
        quantized_protocol)
    target_edges = data["privileged_d_eff"][test_index]

    raw_projection = _replay_teacher_projection(
        data,
        resolved,
        frame_indices=test_index,
        predicted_d_eff=raw_edges,
    )
    quantized_projection = _replay_teacher_projection(
        data,
        resolved,
        frame_indices=test_index,
        predicted_d_eff=quantized_edges,
    )
    return {
        "trace": str(trace_path),
        "checkpoint": str(checkpoint),
        "bits_per_dim": int(bits_per_dim),
        "test_seeds": [int(value) for value in test_seeds],
        "resolve_frames": int(len(test_index)),
        "endpoint_dim": int(student.model.endpoint_dim),
        "protocol_dimensions_per_sender": int(
            Q * 2 * student.model.endpoint_dim),
        "protocol_clip_fraction": float(np.mean(
            np.abs(raw_protocol) >= 1.0)),
        "protocol_quantization_mae": float(np.mean(np.abs(
            quantized_protocol - raw_protocol))),
        "raw_edge_metrics": _edge_value_metrics(
            raw_edges, target_edges),
        "quantized_edge_metrics": _edge_value_metrics(
            quantized_edges, target_edges),
        "raw_projected_structure": raw_projection,
        "quantized_projected_structure": quantized_projection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--bits-per-dim", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(
        args.trace, args.checkpoint, args.bits_per_dim)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
