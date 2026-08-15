#!/usr/bin/env python
"""Train a double-set movement planner on balanced 4/4 and 6/6 domains."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
from tools.train_equivariant_movement_plan import (  # noqa: E402
    _batch,
    _evaluate,
    _metrics,
    _ordered_unique,
    _valid_rows,
)
from uav_isac.agents.equivariant_movement_plan import (  # noqa: E402
    DoubleSetEquivariantResidualMovementPlan,
)


ARCHITECTURE = (
    "shared-uav-peer-target-double-set-equivariant-zoh-residual-v2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class ScaleDataset:
    trace_path: Path
    config_path: Path
    data: dict[str, np.ndarray]
    region_size_m: tuple[float, float]
    num_uavs: int
    num_targets: int
    training_seeds: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    training_rows: np.ndarray
    training_future_rows: np.ndarray
    selection_rows: np.ndarray
    selection_future_rows: np.ndarray

    @property
    def domain_key(self) -> str:
        return f"k{self.num_uavs}q{self.num_targets}"


def _load_dataset(
    trace_path: Path,
    config_path: Path,
    *,
    horizon_steps: int,
    holdout_fraction: float,
    generator: np.random.Generator,
) -> ScaleDataset:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    num_uavs = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    num_targets = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    if (
        num_uavs != int(cfg.scenario.K)
        or num_targets != int(cfg.scenario.Q)
    ):
        raise ValueError(f"{trace_path} dimensions do not match config")
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    seed_order = np.asarray(_ordered_unique(seeds), dtype=np.int64)
    holdout_count = max(1, int(round(seed_order.size * holdout_fraction)))
    if holdout_count >= seed_order.size:
        raise ValueError("holdout fraction leaves no training episode")
    shuffled = seed_order.copy()
    generator.shuffle(shuffled)
    selection_seeds = tuple(sorted(
        int(item) for item in shuffled[:holdout_count]))
    training_seeds = tuple(sorted(
        int(item) for item in shuffled[holdout_count:]))
    rows, future_rows = _valid_rows(seeds, frames, horizon_steps)
    training_mask = np.isin(seeds[rows], training_seeds)
    selection_mask = np.isin(seeds[rows], selection_seeds)
    return ScaleDataset(
        trace_path=trace_path,
        config_path=config_path,
        data=data,
        region_size_m=tuple(
            float(item) for item in cfg.scenario.region_size),
        num_uavs=num_uavs,
        num_targets=num_targets,
        training_seeds=training_seeds,
        selection_seeds=selection_seeds,
        training_rows=rows[training_mask],
        training_future_rows=future_rows[training_mask],
        selection_rows=rows[selection_mask],
        selection_future_rows=future_rows[selection_mask],
    )


def _set_region(
    model: DoubleSetEquivariantResidualMovementPlan,
    region_size_m: tuple[float, float],
) -> None:
    model.region_size_m.copy_(torch.as_tensor(
        region_size_m,
        dtype=model.region_size_m.dtype,
        device=model.region_size_m.device,
    ))


def _domain_metrics(
    model: DoubleSetEquivariantResidualMovementPlan,
    dataset: ScaleDataset,
    device: torch.device,
    horizon_steps: int,
) -> tuple[float, dict[str, object], dict[str, object]]:
    _set_region(model, dataset.region_size_m)
    loss, prediction, target = _evaluate(
        model,
        dataset.data,
        dataset.selection_rows,
        dataset.selection_future_rows,
        device,
    )
    current = np.asarray(
        dataset.data["delta_p"][dataset.selection_rows], dtype=np.float32)
    baseline = np.repeat(current[:, None, :, :], horizon_steps, axis=1)
    return loss, _metrics(prediction, target), _metrics(baseline, target)


def _admission(
    learned: dict[str, object],
    baseline: dict[str, object],
) -> tuple[bool, float, float, float]:
    learned_horizon = np.asarray(
        learned["per_horizon_trajectory_mae_m"], dtype=np.float64)
    baseline_horizon = np.asarray(
        baseline["per_horizon_trajectory_mae_m"], dtype=np.float64)
    learned_composite = (
        float(learned["action_mae_m"])
        + 0.5 * float(learned["trajectory_mae_m"])
        + 0.25 * float(learned["endpoint_mae_m"])
    )
    baseline_composite = (
        float(baseline["action_mae_m"])
        + 0.5 * float(baseline["trajectory_mae_m"])
        + 0.25 * float(baseline["endpoint_mae_m"])
    )
    relative = (
        (baseline_composite - learned_composite) / baseline_composite)
    admitted = bool(
        learned_composite <= 0.99 * baseline_composite
        and float(learned["action_mae_m"])
        <= float(baseline["action_mae_m"])
        and float(learned["endpoint_mae_m"])
        <= float(baseline["endpoint_mae_m"])
        and np.all(learned_horizon <= 1.01 * baseline_horizon)
    )
    return admitted, learned_composite, baseline_composite, float(relative)


def _admission_violation(
    learned: dict[str, object],
    baseline: dict[str, object],
    learned_composite: float,
    baseline_composite: float,
) -> float:
    learned_horizon = np.asarray(
        learned["per_horizon_trajectory_mae_m"], dtype=np.float64)
    baseline_horizon = np.asarray(
        baseline["per_horizon_trajectory_mae_m"], dtype=np.float64)
    ratios = np.concatenate((np.asarray([
        learned_composite / max(0.99 * baseline_composite, 1.0e-12),
        float(learned["action_mae_m"])
        / max(float(baseline["action_mae_m"]), 1.0e-12),
        float(learned["endpoint_mae_m"])
        / max(float(baseline["endpoint_mae_m"]), 1.0e-12),
    ]), learned_horizon / np.maximum(1.01 * baseline_horizon, 1.0e-12)))
    return float(max(np.max(ratios) - 1.0, 0.0))


def _weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    action_weight: float,
    trajectory_weight: float,
    endpoint_weight: float,
) -> torch.Tensor:
    action = nn.functional.smooth_l1_loss(prediction, target)
    trajectory = nn.functional.smooth_l1_loss(
        torch.cumsum(prediction, dim=1),
        torch.cumsum(target, dim=1),
    )
    endpoint = nn.functional.smooth_l1_loss(
        torch.sum(prediction, dim=1), torch.sum(target, dim=1))
    return (
        float(action_weight) * action
        + float(trajectory_weight) * trajectory
        + float(endpoint_weight) * endpoint
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs=2,
        metavar=("TRACE", "CONFIG"),
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-steps", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--residual-limit-fraction", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-loss-weight", type=float, default=0.5)
    parser.add_argument("--endpoint-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if len(args.dataset) < 2:
        raise ValueError("multiscale training requires at least two domains")
    if not 0.0 < float(args.holdout_fraction) < 1.0:
        raise ValueError("holdout fraction must lie in (0,1)")
    if any(float(value) < 0.0 for value in (
        args.action_loss_weight,
        args.trajectory_loss_weight,
        args.endpoint_loss_weight,
    )) or float(
        args.action_loss_weight
        + args.trajectory_loss_weight
        + args.endpoint_loss_weight
    ) <= 0.0:
        raise ValueError("loss weights must be non-negative and non-zero")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = np.random.default_rng(args.seed)
    datasets = [
        _load_dataset(
            Path(trace),
            Path(config),
            horizon_steps=int(args.horizon_steps),
            holdout_fraction=float(args.holdout_fraction),
            generator=generator,
        )
        for trace, config in args.dataset
    ]
    domain_keys = [dataset.domain_key for dataset in datasets]
    if len(set(domain_keys)) != len(domain_keys):
        raise ValueError("each development domain must have distinct K,Q")
    first_cfg = load_config(str(datasets[0].config_path))
    maximum_displacement = (
        float(first_cfg.uav.v_max) * float(first_cfg.scenario.dt))
    movement_interval = int(first_cfg.marl.movement_decision_interval)
    for dataset in datasets[1:]:
        cfg = load_config(str(dataset.config_path))
        if (
            int(cfg.marl.movement_decision_interval) != movement_interval
            or not np.isclose(
                float(cfg.uav.v_max) * float(cfg.scenario.dt),
                maximum_displacement,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError("development domains use different motion laws")

    canonical_region = datasets[0].region_size_m
    model = DoubleSetEquivariantResidualMovementPlan(
        horizon_steps=int(args.horizon_steps),
        movement_decision_interval=movement_interval,
        region_size_m=canonical_region,
        maximum_displacement_m=maximum_displacement,
        hidden_dim=int(args.hidden_dim),
        residual_limit_fraction=float(args.residual_limit_fraction),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=1.0e-5)
    balanced_samples = min(
        dataset.training_rows.size for dataset in datasets)
    steps_per_epoch = int(np.ceil(
        balanced_samples / int(args.batch_size)))
    best_loss = float("inf")
    best_selection_key: tuple[float, ...] | None = None
    best_selection_diagnostics: dict[str, object] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        orders = [
            generator.choice(
                dataset.training_rows.size,
                size=balanced_samples,
                replace=False,
            )
            for dataset in datasets
        ]
        for step in range(steps_per_epoch):
            losses = []
            start = step * int(args.batch_size)
            stop = min(start + int(args.batch_size), balanced_samples)
            for dataset, order in zip(datasets, orders):
                chosen = order[start:stop]
                _set_region(model, dataset.region_size_m)
                inputs = _batch(
                    dataset.data,
                    dataset.training_rows[chosen],
                    dataset.training_future_rows[chosen],
                    device,
                )
                losses.append(_weighted_loss(
                    model(*inputs[:-1]),
                    inputs[-1],
                    action_weight=float(args.action_loss_weight),
                    trajectory_weight=float(args.trajectory_loss_weight),
                    endpoint_weight=float(args.endpoint_loss_weight),
                ))
            loss = torch.mean(torch.stack(losses))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        selection_losses = []
        selection_admitted = []
        selection_relative = []
        selection_violations = []
        for dataset in datasets:
            selection_loss, learned, baseline = _domain_metrics(
                model, dataset, device, int(args.horizon_steps))
            selection_losses.append(selection_loss)
            admitted, learned_composite, baseline_composite, relative = (
                _admission(learned, baseline))
            selection_admitted.append(admitted)
            selection_relative.append(relative)
            selection_violations.append(_admission_violation(
                learned,
                baseline,
                learned_composite,
                baseline_composite,
            ))
        mean_selection_loss = float(np.mean(selection_losses))
        all_selection_admitted = bool(all(selection_admitted))
        selection_key = (
            float(all_selection_admitted),
            -float(max(selection_violations)),
            float(min(selection_relative)),
            -mean_selection_loss,
        )
        if (
            best_selection_key is None
            or selection_key > best_selection_key
        ):
            best_loss = mean_selection_loss
            best_selection_key = selection_key
            best_selection_diagnostics = {
                "all_domains_admitted": all_selection_admitted,
                "maximum_normalized_admission_violation": float(
                    max(selection_violations)),
                "minimum_relative_composite_improvement": float(
                    min(selection_relative)),
                "mean_selection_loss": mean_selection_loss,
            }
            best_epoch = epoch
            _set_region(model, canonical_region)
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

    domain_reports = {}
    all_admitted = True
    for dataset in datasets:
        selection_loss, learned, baseline = _domain_metrics(
            model, dataset, device, int(args.horizon_steps))
        admitted, learned_composite, baseline_composite, relative = (
            _admission(learned, baseline))
        all_admitted = all_admitted and admitted
        domain_reports[dataset.domain_key] = {
            "trace": str(dataset.trace_path),
            "trace_sha256": _sha256(dataset.trace_path),
            "config": str(dataset.config_path),
            "region_size_m": list(dataset.region_size_m),
            "training_episode_ids": list(dataset.training_seeds),
            "selection_episode_ids": list(dataset.selection_seeds),
            "training_sample_count": int(dataset.training_rows.size),
            "selection_sample_count": int(dataset.selection_rows.size),
            "selection_loss": selection_loss,
            "baseline": baseline,
            "learned": learned,
            "composite_baseline": baseline_composite,
            "composite_learned": learned_composite,
            "relative_composite_improvement": relative,
            "selection_admitted": admitted,
        }
    training_ids = sorted(set().union(*(
        set(dataset.training_seeds) for dataset in datasets)))
    selection_ids = sorted(set().union(*(
        set(dataset.selection_seeds) for dataset in datasets)))
    metadata = {
        "schema_version": 2,
        "architecture": ARCHITECTURE,
        "causal_inputs": [
            "current_post_action_own_position",
            "current_post_action_peer_position_set",
            "mission_known_static_target_position_set",
            "current_movement_action_set",
            "current_feasible_communication_power_set",
            "current_feasible_sensing_power_set",
            "current_decision_phase",
        ],
        "horizon_steps": int(args.horizon_steps),
        "movement_decision_interval": movement_interval,
        "region_size_m": list(canonical_region),
        "maximum_displacement_m": maximum_displacement,
        "residual_limit_fraction": float(args.residual_limit_fraction),
        "training_episode_ids": training_ids,
        "selection_episode_ids": selection_ids,
        "training_episode_keys": [
            f"{dataset.domain_key}:{seed}"
            for dataset in datasets for seed in dataset.training_seeds
        ],
        "selection_episode_keys": [
            f"{dataset.domain_key}:{seed}"
            for dataset in datasets for seed in dataset.selection_seeds
        ],
        "selection_admitted": bool(all_admitted),
    }
    report = {
        "metadata": metadata,
        "device": str(device),
        "balanced_samples_per_domain_per_epoch": int(balanced_samples),
        "best_epoch": best_epoch,
        "best_mean_selection_loss": best_loss,
        "best_selection_lexicographic_diagnostics": (
            best_selection_diagnostics),
        "loss_weights": {
            "action": float(args.action_loss_weight),
            "trajectory": float(args.trajectory_loss_weight),
            "endpoint": float(args.endpoint_loss_weight),
        },
        "domains": domain_reports,
        "all_domains_selection_admitted": bool(all_admitted),
        "admission_rule": (
            "every development scale separately: composite improves >=1%; "
            "action and endpoint MAE do not worsen; each cumulative-horizon "
            "MAE worsens by at most 1%"),
        "held_out_scale_policy": (
            "8/8 is absent from training, checkpoint selection and admission"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "hidden_dim": int(args.hidden_dim),
        "metadata": metadata,
    }, args.output / "movement_plan.pt")
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "architecture": ARCHITECTURE,
        "best_epoch": best_epoch,
        "best_mean_selection_loss": best_loss,
        "all_domains_selection_admitted": bool(all_admitted),
        "domain_relative_improvements": {
            key: value["relative_composite_improvement"]
            for key, value in domain_reports.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
