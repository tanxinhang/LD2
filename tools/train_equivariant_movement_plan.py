#!/usr/bin/env python
"""Train and holdout-select a causal set-equivariant movement-plan head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.agents.equivariant_movement_plan import (  # noqa: E402
    EquivariantResidualMovementPlan,
)


def _ordered_unique(values: np.ndarray) -> list[int]:
    return list(dict.fromkeys(int(item) for item in values.reshape(-1)))


def _valid_rows(
    seeds: np.ndarray,
    frames: np.ndarray,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        (int(seed), int(frame)): index
        for index, (seed, frame) in enumerate(zip(seeds, frames))
    }
    rows: list[int] = []
    future: list[list[int]] = []
    for row, (seed, frame) in enumerate(zip(seeds, frames)):
        key_rows = [
            lookup.get((int(seed), int(frame) + offset))
            for offset in range(1, int(horizon_steps) + 1)
        ]
        if all(item is not None for item in key_rows):
            rows.append(row)
            future.append([int(item) for item in key_rows])
    return np.asarray(rows, dtype=np.int64), np.asarray(future, dtype=np.int64)


def _batch(
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    future_rows: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    weights = np.asarray(data["sensing_weights"][rows], dtype=np.float32)
    weights = weights / np.maximum(
        np.sum(weights, axis=2, keepdims=True), 1.0e-12)
    active = (
        np.asarray(data["outgoing_rate"][rows], dtype=np.int64) > 0
    ) & np.any(
        np.asarray(data["outgoing_token_mask"][rows], dtype=bool), axis=2)
    fraction = np.clip(
        np.asarray(data["comm_fraction"][rows], dtype=np.float32), 0.0, 1.0)
    communication_power = np.where(active, fraction, 0.0)
    sensing_power = (1.0 - communication_power[..., None]) * weights
    return (
        torch.as_tensor(
            data["uav_positions"][rows, :, :2],
            dtype=torch.float32, device=device),
        torch.as_tensor(
            data["target_states"][rows, :, :2],
            dtype=torch.float32, device=device),
        torch.as_tensor(
            data["delta_p"][rows], dtype=torch.float32, device=device),
        torch.as_tensor(
            communication_power,
            dtype=torch.float32, device=device),
        torch.as_tensor(
            sensing_power,
            dtype=torch.float32, device=device),
        torch.as_tensor(
            data["frame"][rows], dtype=torch.long, device=device),
        torch.as_tensor(
            data["delta_p"][future_rows],
            dtype=torch.float32, device=device),
    )


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    action = nn.functional.smooth_l1_loss(prediction, target)
    trajectory = nn.functional.smooth_l1_loss(
        torch.cumsum(prediction, dim=1),
        torch.cumsum(target, dim=1),
    )
    endpoint = nn.functional.smooth_l1_loss(
        torch.sum(prediction, dim=1), torch.sum(target, dim=1))
    return action + 0.5 * trajectory + 0.25 * endpoint


def _metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    action_error = np.linalg.norm(prediction - target, axis=-1)
    trajectory_error = np.linalg.norm(
        np.cumsum(prediction, axis=1) - np.cumsum(target, axis=1), axis=-1)
    endpoint_error = trajectory_error[:, -1]
    return {
        "action_mae_m": float(np.mean(action_error)),
        "trajectory_mae_m": float(np.mean(trajectory_error)),
        "endpoint_mae_m": float(np.mean(endpoint_error)),
        "per_horizon_action_mae_m": np.mean(
            action_error, axis=(0, 2)).tolist(),
        "per_horizon_trajectory_mae_m": np.mean(
            trajectory_error, axis=(0, 2)).tolist(),
    }


def _evaluate(
    model: EquivariantResidualMovementPlan,
    data: dict[str, np.ndarray],
    rows: np.ndarray,
    future_rows: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, rows.size, 256):
            inputs = _batch(
                data, rows[start:start + 256],
                future_rows[start:start + 256], device)
            prediction = model(*inputs[:-1])
            losses.append(float(_loss(prediction, inputs[-1]).cpu()))
            predictions.append(np.asarray(prediction.cpu()))
            targets.append(np.asarray(inputs[-1].cpu()))
    return (
        float(np.mean(losses)),
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-steps", type=int, default=3)
    parser.add_argument("--holdout-seeds", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--residual-limit-fraction", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with np.load(args.trace, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(args.config))
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    seed_order = _ordered_unique(seeds)
    if not 1 <= int(args.holdout_seeds) < len(seed_order):
        raise ValueError("holdout-seeds must leave at least one training seed")
    generator = np.random.default_rng(args.seed)
    shuffled = np.asarray(seed_order, dtype=np.int64)
    generator.shuffle(shuffled)
    selection_seeds = tuple(sorted(
        int(item) for item in shuffled[:args.holdout_seeds]))
    training_seeds = tuple(sorted(
        int(item) for item in shuffled[args.holdout_seeds:]))

    rows, future_rows = _valid_rows(seeds, frames, args.horizon_steps)
    train_mask = np.isin(seeds[rows], training_seeds)
    selection_mask = np.isin(seeds[rows], selection_seeds)
    train_rows, train_future = rows[train_mask], future_rows[train_mask]
    select_rows, select_future = rows[selection_mask], future_rows[selection_mask]
    maximum_displacement = float(cfg.uav.v_max) * float(cfg.scenario.dt)
    model = EquivariantResidualMovementPlan(
        horizon_steps=int(args.horizon_steps),
        movement_decision_interval=int(cfg.marl.movement_decision_interval),
        region_size_m=tuple(float(item) for item in cfg.scenario.region_size),
        maximum_displacement_m=maximum_displacement,
        hidden_dim=int(args.hidden_dim),
        residual_limit_fraction=float(args.residual_limit_fraction),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1.0e-5)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = generator.permutation(train_rows.size)
        for start in range(0, train_rows.size, int(args.batch_size)):
            chosen = order[start:start + int(args.batch_size)]
            inputs = _batch(
                data, train_rows[chosen], train_future[chosen], device)
            prediction = model(*inputs[:-1])
            loss = _loss(prediction, inputs[-1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        selection_loss, _, _ = _evaluate(
            model, data, select_rows, select_future, device)
        if selection_loss < best_loss - 1.0e-8:
            best_loss = selection_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(args.patience):
            break
    if best_state is None:
        raise AssertionError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    _, prediction, target = _evaluate(
        model, data, select_rows, select_future, device)
    current = np.asarray(data["delta_p"][select_rows], dtype=np.float32)
    baseline = np.repeat(
        current[:, None, :, :], int(args.horizon_steps), axis=1)
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
    metadata = {
        "schema_version": 1,
        "architecture": "shared-uav-target-set-invariant-zoh-residual-v1",
        "causal_inputs": [
            "current_post_action_own_position",
            "mission_known_static_target_positions",
            "current_movement_action",
            "current_feasible_communication_power_w",
            "current_feasible_sensing_power_w",
            "current_decision_phase",
        ],
        "horizon_steps": int(args.horizon_steps),
        "movement_decision_interval": int(
            cfg.marl.movement_decision_interval),
        "region_size_m": [float(item) for item in cfg.scenario.region_size],
        "maximum_displacement_m": maximum_displacement,
        "residual_limit_fraction": float(args.residual_limit_fraction),
        "training_episode_ids": list(training_seeds),
        "selection_episode_ids": list(selection_seeds),
        "selection_admitted": admitted,
    }
    report = {
        "metadata": metadata,
        "trace": str(args.trace),
        "config": str(args.config),
        "device": str(device),
        "training_sample_count": int(train_rows.size),
        "selection_sample_count": int(select_rows.size),
        "best_epoch": best_epoch,
        "best_selection_loss": best_loss,
        "baseline": baseline_metrics,
        "learned": learned_metrics,
        "composite_baseline": composite_baseline,
        "composite_learned": composite_learned,
        "relative_composite_improvement": float(
            (composite_baseline - composite_learned) / composite_baseline),
        "admission_rule": (
            "composite improves >=1%; action and endpoint MAE do not worsen; "
            "each cumulative-horizon MAE worsens by at most 1%"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "hidden_dim": int(args.hidden_dim),
        "metadata": metadata,
    }, args.output / "movement_plan.pt")
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
