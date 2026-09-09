"""Benchmark the shadow predictive GNN on an exported teacher trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.prediction.certified_gnn import (
    CertifiedBipartiteGNN,
    audit_predictive_checkpoint,
    append_causal_feature_residual,
    augment_temporal_protocol_features,
    build_endpoint_features,
    decode_candidate_pool,
    require_compatible_predictive_checkpoint,
)
from uav_isac.utils.checkpoint_loading import safe_torch_load


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def _build_features(
    resident: dict[str, np.ndarray],
    feature_dim: int,
) -> np.ndarray:
    features = build_endpoint_features(
        resident["endpoint_position"][0],
        resident["endpoint_velocity"][0],
        resident["target_position"][0],
        resident["target_velocity"][0],
        resident["target_position_uncertainty"][0],
        resident["target_velocity_uncertainty"][0],
        resident["visible"][0],
        distance_scale_m=1000.0,
        velocity_scale_mps=50.0,
    )
    if feature_dim in (18, 37):
        features = augment_temporal_protocol_features(
            features[None],
            resident["edge_index"], resident["edge_mask"], resident["frame"],
            resident["nominal_gain"], resident["lower_gain"],
            resident["upper_gain"], resident["power"], resident["dual_price"],
        )[0]
    if feature_dim == 37:
        previous_base = build_endpoint_features(
            resident["previous_endpoint_position"][0],
            resident["previous_endpoint_velocity"][0],
            resident["previous_target_position"][0],
            resident["previous_target_velocity"][0],
            resident["previous_target_position_uncertainty"][0],
            resident["previous_target_velocity_uncertainty"][0],
            resident["previous_visible"][0],
            distance_scale_m=1000.0,
            velocity_scale_mps=50.0,
        )
        previous_augmented = augment_temporal_protocol_features(
            previous_base[None],
            resident["previous_edge_index"],
            resident["previous_edge_mask"],
            resident["previous_frame"],
            resident["previous_nominal_gain"],
            resident["previous_lower_gain"],
            resident["previous_upper_gain"],
            resident["previous_power"],
            resident["previous_dual_price"],
        )[0]
        features = append_causal_feature_residual(
            features[None], previous_augmented[None],
            resident["history_valid"],
        )[0]
    if features.shape[-1] != int(feature_dim):
        raise ValueError(
            f"checkpoint expects {feature_dim} features, built "
            f"{features.shape[-1]}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="report checkpoint/API compatibility without benchmarking")
    args = parser.parse_args()
    if args.audit_only and not args.checkpoint:
        parser.error("--audit-only requires --checkpoint")

    torch.set_num_threads(1)
    torch.manual_seed(17)
    data = np.load(args.dataset)
    frame = 1 if len(data["frame"]) > 1 else 0
    previous = frame - 1 if frame > 0 else frame
    resident_keys = (
        "endpoint_position", "endpoint_velocity", "target_position",
        "target_velocity", "target_position_uncertainty",
        "target_velocity_uncertainty", "visible", "edge_index",
        "edge_mask", "frame", "nominal_gain", "lower_gain",
        "upper_gain", "power", "dual_price",
    )
    resident = {
        key: data[key][frame:frame + 1]
        for key in resident_keys
    }
    resident.update({
        f"previous_{key}": data[key][previous:previous + 1]
        for key in resident_keys
    })
    resident["history_valid"] = np.asarray([
        float(previous != frame and data["seed"][previous] == data["seed"][frame])
    ], dtype=np.float32)
    checkpoint_payload = None
    if args.checkpoint:
        checkpoint_payload = safe_torch_load(
            args.checkpoint,
            map_location="cpu",
            description="predictive GNN benchmark checkpoint",
            required_keys=("feature_dim", "hidden_dim", "message_rounds"),
            state_dict_keys=("state_dict",),
        )
    feature_dim = int(
        checkpoint_payload["feature_dim"] if checkpoint_payload else 9)
    start = time.perf_counter_ns()
    features_np = _build_features(resident, feature_dim)
    feature_cold_ms = (time.perf_counter_ns() - start) / 1.0e6
    for _ in range(min(args.warmup, 5)):
        _build_features(resident, feature_dim)
    feature_samples: list[float] = []
    for _ in range(min(args.iterations, 50)):
        begin = time.perf_counter_ns()
        _build_features(resident, feature_dim)
        feature_samples.append((time.perf_counter_ns() - begin) / 1.0e6)
    features = torch.from_numpy(features_np)
    visible = torch.from_numpy(resident["visible"][0])
    model = CertifiedBipartiteGNN(
        feature_dim=int(features.shape[-1]),
        hidden_dim=(
            int(checkpoint_payload["hidden_dim"])
            if checkpoint_payload else args.hidden),
        message_rounds=(
            int(checkpoint_payload["message_rounds"])
            if checkpoint_payload else args.rounds),
        boundary_feature_index=(
            checkpoint_payload.get("boundary_feature_index")
            if checkpoint_payload else None),
        boundary_power_residual=(
            checkpoint_payload.get("boundary_power_residual", True)
            if checkpoint_payload else True),
    ).eval()
    checkpoint_audit = None
    if checkpoint_payload:
        checkpoint_audit = audit_predictive_checkpoint(
            checkpoint_payload, model)
        if args.audit_only:
            print(json.dumps(checkpoint_audit, indent=2, allow_nan=False))
            return
        require_compatible_predictive_checkpoint(checkpoint_payload, model)
        model.load_state_dict(checkpoint_payload["state_dict"])

    with torch.inference_mode():
        for _ in range(args.warmup):
            prediction = model(features, visible)
        samples: list[float] = []
        for _ in range(args.iterations):
            begin = time.perf_counter_ns()
            prediction = model(features, visible)
            samples.append((time.perf_counter_ns() - begin) / 1.0e6)
        begin = time.perf_counter_ns()
        candidates = decode_candidate_pool(
            prediction, resident["visible"][0], model=model)
        decode_ms = (time.perf_counter_ns() - begin) / 1.0e6

    print(json.dumps({
        "dataset": args.dataset,
        "input_shape": list(features.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "feature_build_cold_ms": feature_cold_ms,
        "feature_build_mean_ms": float(np.mean(feature_samples)),
        "feature_build_p95_ms": percentile(feature_samples, 95),
        "inference_mean_ms": float(np.mean(samples)),
        "inference_p50_ms": percentile(samples, 50),
        "inference_p95_ms": percentile(samples, 95),
        "conditional_decode_ms": decode_ms,
        "candidate_edges": len(candidates),
        "checkpoint_audit": checkpoint_audit,
        "note": (
            "checkpoint loaded; latency only"
            if checkpoint_payload else "random weights; latency only, not quality"),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
