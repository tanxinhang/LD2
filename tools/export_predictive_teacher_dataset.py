"""Export exact-pipeline traces as a fixed-shape predictive teacher dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "predictive-teacher-dataset/v2-native-objectives"


def export_predictive_teacher_dataset(
    source: str | Path,
    destination: str | Path,
) -> dict[str, object]:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    return export_predictive_teacher_payload(payload, destination)


def export_predictive_teacher_payload(
    payload: dict[str, object],
    destination: str | Path,
) -> dict[str, object]:
    """Export directly from an in-memory strict-run report."""
    records: list[tuple[int, dict[str, object]]] = []
    native_records: list[dict[str, object]] = []
    for episode in payload.get("episodes", []):
        seed = int(episode["seed"])
        trace = episode.get("trace", {})
        snapshots = trace.get("acceleration_golden", [])
        native_keys = (
            "detection",
            "detection_deflection",
            "sensing_power_w",
            "certificate_safe_gain_per_watt",
            "bits",
            "delivery",
        )
        missing_native = [key for key in native_keys if key not in trace]
        if missing_native:
            raise ValueError(
                "trace lacks native objective labels: "
                + ", ".join(missing_native))
        if any(len(trace[key]) != len(snapshots) for key in native_keys):
            raise ValueError("native objective labels do not align with golden frames")
        for index, snapshot in enumerate(snapshots):
            records.append((seed, snapshot))
            native_records.append({
                key: trace[key][index] for key in native_keys
            })
    if not records:
        raise ValueError("source contains no acceleration_golden frames")

    required = (
        "endpoint_position_views",
        "endpoint_velocity_views",
        "target_position_views",
        "target_velocity_views",
        "target_position_uncertainty_views",
        "target_velocity_uncertainty_views",
        "visible_views",
        "public_gain_views",
        "certificate_gain_views",
        "certificate_gain_upper_views",
        "local_power_cache_w",
        "local_prices",
        "local_cache_valid",
        "selected",
    )
    for _, snapshot in records:
        missing = [key for key in required if key not in snapshot]
        if missing:
            raise ValueError(
                "golden trace predates predictive fields: "
                + ", ".join(missing))

    def stacked(key: str, dtype=np.float64) -> np.ndarray:
        arrays = [np.asarray(row[key], dtype=dtype) for _, row in records]
        first_shape = arrays[0].shape
        if any(array.shape != first_shape for array in arrays):
            raise ValueError(f"inconsistent {key} shape across frames")
        return np.stack(arrays, axis=0)

    max_edges = max(len(row["selected"]) for _, row in records)
    edge_index = np.full((len(records), max_edges, 3), -1, dtype=np.int32)
    edge_mask = np.zeros((len(records), max_edges), dtype=bool)
    for frame_index, (_, row) in enumerate(records):
        edges = np.asarray(row["selected"], dtype=np.int32).reshape(-1, 3)
        edge_index[frame_index, :len(edges)] = edges
        edge_mask[frame_index, :len(edges)] = True

    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "seed": np.asarray([seed for seed, _ in records], dtype=np.int64),
        "frame": np.asarray([
            int(row["frame"]) for _, row in records
        ], dtype=np.int64),
        "hold_active": np.asarray([
            bool(row["hold_active"]) for _, row in records
        ], dtype=bool),
        "endpoint_position": stacked("endpoint_position_views"),
        "endpoint_velocity": stacked("endpoint_velocity_views"),
        "target_position": stacked("target_position_views"),
        "target_velocity": stacked("target_velocity_views"),
        "target_position_uncertainty": stacked(
            "target_position_uncertainty_views"),
        "target_velocity_uncertainty": stacked(
            "target_velocity_uncertainty_views"),
        "visible": stacked("visible_views", dtype=bool),
        "nominal_gain": stacked("public_gain_views"),
        "lower_gain": stacked("certificate_gain_views"),
        "upper_gain": stacked("certificate_gain_upper_views"),
        "power": stacked("local_power_cache_w"),
        "dual_price": stacked("local_prices"),
        "power_cache_valid": stacked("local_cache_valid", dtype=bool),
        "edge_index": edge_index,
        "edge_mask": edge_mask,
        # Execution-grounded labels. The legacy source key contains the word
        # certificate, but the exported robust gain is simply the executor's
        # local physical lower envelope and is not transported on air.
        "executed_power": np.asarray([
            row["sensing_power_w"] for row in native_records
        ], dtype=np.float64),
        "executed_robust_gain": np.asarray([
            row["certificate_safe_gain_per_watt"] for row in native_records
        ], dtype=np.float64),
        "executed_nominal_gain": np.stack([
            np.asarray(snapshot["public_gain_views"], dtype=np.float64)[
                np.arange(np.asarray(snapshot["public_gain_views"]).shape[0]),
                np.arange(np.asarray(snapshot["public_gain_views"]).shape[0]),
                :,
            ]
            for _, snapshot in records
        ], axis=0),
        "actual_deflection": np.asarray([
            row["detection_deflection"] for row in native_records
        ], dtype=np.float64),
        "actual_detection": np.asarray([
            row["detection"] for row in native_records
        ], dtype=np.float64),
        "protocol_bits": np.asarray([
            row["bits"] for row in native_records
        ], dtype=np.float64),
        "delivery_rate": np.asarray([
            row["delivery"] for row in native_records
        ], dtype=np.float64),
    }
    if not (
        np.all(arrays["lower_gain"] <= arrays["upper_gain"] + 1.0e-15)
        and np.all(np.isfinite(arrays["endpoint_position"]))
        and np.all(np.isfinite(arrays["nominal_gain"]))
        and np.all(np.isfinite(arrays["actual_deflection"]))
        and np.all(np.isfinite(arrays["executed_nominal_gain"]))
        and np.all(arrays["executed_nominal_gain"] >= 0.0)
        and np.all(arrays["actual_deflection"] >= 0.0)
        and np.all(np.isfinite(arrays["actual_detection"]))
        and np.all((arrays["actual_detection"] >= 0.0)
                   & (arrays["actual_detection"] <= 1.0))
    ):
        raise ValueError("teacher trace violates finite/bound invariants")

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "schema_version": SCHEMA_VERSION,
        "frames": len(records),
        "seeds": len(set(int(seed) for seed, _ in records)),
        "max_edges": max_edges,
        "endpoint_shape": list(arrays["endpoint_position"].shape),
        "output": str(output),
        "bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export predictive teacher NPZ from a strict pilot trace")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--summary")
    args = parser.parse_args()
    summary = export_predictive_teacher_dataset(
        args.source, args.destination)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.summary:
        Path(args.summary).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
