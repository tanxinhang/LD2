#!/usr/bin/env python
"""Audit zero-shot movement-plan transfer on an independent scale trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.train_equivariant_movement_plan import (  # noqa: E402
    _evaluate,
    _metrics,
    _ordered_unique,
    _valid_rows,
)
from uav_isac.agents.equivariant_movement_plan import (  # noqa: E402
    FrozenEquivariantMovementPlanner,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    trace_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    *,
    allow_uniform_region_scaling: bool,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    num_uavs = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    num_targets = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    if (
        num_uavs != int(cfg.scenario.K)
        or num_targets != int(cfg.scenario.Q)
    ):
        raise ValueError("trace dimensions do not match config")
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    seed_order = _ordered_unique(seeds)

    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = payload["metadata"]
    domain_key = f"k{num_uavs}q{num_targets}"
    development_keys = set(
        str(item) for item in metadata.get("training_episode_keys", ())
    ) | set(str(item) for item in metadata.get("selection_episode_keys", ()))
    if development_keys:
        excluded_ids = tuple(
            seed for seed in seed_order
            if f"{domain_key}:{seed}" in development_keys)
    else:
        development_ids = set(
            int(item) for item in metadata["training_episode_ids"]
        ) | set(int(item) for item in metadata["selection_episode_ids"])
        excluded_ids = tuple(sorted(development_ids & set(seed_order)))
    validation_ids = tuple(
        seed for seed in seed_order if seed not in set(excluded_ids))
    if not validation_ids:
        raise ValueError("no episode remains after development-overlap removal")

    horizon = int(metadata["horizon_steps"])
    planner = FrozenEquivariantMovementPlanner.from_checkpoint(
        checkpoint_path,
        validation_episode_ids=validation_ids,
        horizon_steps=horizon,
        movement_decision_interval=int(cfg.marl.movement_decision_interval),
        region_size_m=tuple(
            float(item) for item in cfg.scenario.region_size),
        maximum_displacement_m=(
            float(cfg.uav.v_max) * float(cfg.scenario.dt)),
        allow_uniform_region_scaling=bool(allow_uniform_region_scaling),
        validation_domain_key=domain_key,
    )
    rows, future_rows = _valid_rows(seeds, frames, horizon)
    independent = np.isin(seeds[rows], validation_ids)
    rows = rows[independent]
    future_rows = future_rows[independent]
    if rows.size == 0:
        raise ValueError("independent episodes contain no complete horizon")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    planner.model.to(device)
    _, prediction, target = _evaluate(
        planner.model, data, rows, future_rows, device)
    current = np.asarray(data["delta_p"][rows], dtype=np.float32)
    baseline = np.repeat(current[:, None, :, :], horizon, axis=1)
    learned_metrics = _metrics(prediction, target)
    baseline_metrics = _metrics(baseline, target)
    learned_horizon = np.asarray(
        learned_metrics["per_horizon_trajectory_mae_m"], dtype=np.float64)
    baseline_horizon = np.asarray(
        baseline_metrics["per_horizon_trajectory_mae_m"], dtype=np.float64)
    composite_learned = (
        float(learned_metrics["action_mae_m"])
        + 0.5 * float(learned_metrics["trajectory_mae_m"])
        + 0.25 * float(learned_metrics["endpoint_mae_m"])
    )
    composite_baseline = (
        float(baseline_metrics["action_mae_m"])
        + 0.5 * float(baseline_metrics["trajectory_mae_m"])
        + 0.25 * float(baseline_metrics["endpoint_mae_m"])
    )
    admitted = bool(
        composite_learned <= 0.99 * composite_baseline
        and float(learned_metrics["action_mae_m"])
        <= float(baseline_metrics["action_mae_m"])
        and float(learned_metrics["endpoint_mae_m"])
        <= float(baseline_metrics["endpoint_mae_m"])
        and np.all(learned_horizon <= 1.01 * baseline_horizon)
    )
    maximum_displacement = (
        float(cfg.uav.v_max) * float(cfg.scenario.dt))
    return {
        "schema_version": 1,
        "status": (
            "independent_zero_shot_forecast_transfer_admitted"
            if admitted else "zero_shot_forecast_transfer_not_admitted"),
        "scope": (
            "movement forecast fidelity only; not a detection, QoS, "
            "coefficient-envelope or deployment certificate"),
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "num_uavs": num_uavs,
        "num_targets": num_targets,
        "horizon_steps": horizon,
        "source_region_size_m": list(planner.metadata.region_size_m),
        "runtime_region_size_m": list(planner.runtime_region_size_m),
        "uniform_region_scale": float(planner.uniform_region_scale),
        "uniform_region_scaling_applied": bool(
            planner.uniform_region_scaling_applied),
        "independent_episode_ids": list(validation_ids),
        "independent_episode_count": len(validation_ids),
        "episode_identity": "domain_key:seed",
        "excluded_development_overlap_ids": list(excluded_ids),
        "evaluated_horizon_count": int(rows.size),
        "maximum_predicted_displacement_m": float(np.max(
            np.linalg.norm(prediction, axis=-1), initial=0.0)),
        "maximum_allowed_displacement_m": maximum_displacement,
        "speed_disk_satisfied": bool(np.all(
            np.linalg.norm(prediction, axis=-1)
            <= maximum_displacement + 1.0e-7)),
        "baseline": baseline_metrics,
        "learned": learned_metrics,
        "composite_baseline": composite_baseline,
        "composite_learned": composite_learned,
        "relative_composite_improvement": float(
            (composite_baseline - composite_learned) / composite_baseline),
        "transfer_admitted": admitted,
        "admission_rule": (
            "composite improves >=1%; action and endpoint MAE do not worsen; "
            "each cumulative-horizon MAE worsens by at most 1%"),
        "exact_architecture_invariants": [
            "target_permutation_invariance",
            "uav_permutation_equivariance",
            "uniform_region_scale_covariance",
            "per_step_speed_disk_projection",
            "known_movement_hold_phase",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-uniform-region-scaling", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.config,
        args.checkpoint,
        allow_uniform_region_scaling=bool(
            args.allow_uniform_region_scaling),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "num_uavs", "num_targets",
        "independent_episode_count", "evaluated_horizon_count",
        "relative_composite_improvement", "transfer_admitted",
    )}, indent=2))


if __name__ == "__main__":
    main()
