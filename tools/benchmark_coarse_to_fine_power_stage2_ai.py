#!/usr/bin/env python
"""Stage-2 shadow test of an equivariant coarse-to-fine power refiner.

The model consumes only a recorded fixed-structure gain matrix, the local RF
budgets, and the analytic harmonic safe start.  Shared edge/row/target update
weights make it permutation equivariant; a per-UAV softmax is the hard RF
feasibility projection.  A second head predicts simplex target prices, which
provides an independently checkable LP dual upper bound.

This is an offline shadow benchmark.  It does not modify the deployed policy,
does not use test-seed labels for training, and falls back conceptually when
the reported primal-dual certificate is outside tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_coarse_to_fine_power_stage1 import (  # noqa: E402
    _harmonic_safe_start,
    _load_instances,
    _quantiles,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    solve_fixed_structure_maxmin_power_lp,
)


class EquivariantPowerRefiner(nn.Module):
    """Small bipartite message-passing model with no K/Q-specific weights."""

    def __init__(self, hidden_dim: int = 32, blocks: int = 2) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(int(blocks))
        ])
        self.power_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        normalized_gain: torch.Tensor,
        normalized_budget: torch.Tensor,
        harmonic_share: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # gain [B,K,Q], budget [B,K], harmonic_share [B,Q]
        eps = torch.finfo(normalized_gain.dtype).eps
        log_gain = torch.log(normalized_gain.clamp_min(eps))
        log_gain = (
            log_gain - log_gain.mean(dim=(1, 2), keepdim=True)
        ) / log_gain.std(dim=(1, 2), keepdim=True).clamp_min(1.0e-3)
        row_relative = normalized_gain / normalized_gain.amax(
            dim=2, keepdim=True).clamp_min(eps)
        column_relative = normalized_gain / normalized_gain.amax(
            dim=1, keepdim=True).clamp_min(eps)
        target_ceiling = (
            normalized_gain * normalized_budget[:, :, None]
        ).sum(dim=1)
        target_ceiling = target_ceiling / target_ceiling.amax(
            dim=1, keepdim=True).clamp_min(eps)
        features = torch.stack((
            log_gain,
            row_relative,
            column_relative,
            normalized_budget[:, :, None].expand_as(normalized_gain),
            harmonic_share[:, None, :].expand_as(normalized_gain),
            target_ceiling[:, None, :].expand_as(normalized_gain),
        ), dim=-1)
        edge = self.edge_encoder(features)
        for update in self.blocks:
            row = edge.mean(dim=2, keepdim=True).expand_as(edge)
            target = edge.mean(dim=1, keepdim=True).expand_as(edge)
            global_context = edge.mean(
                dim=(1, 2), keepdim=True).expand_as(edge)
            edge = edge + update(torch.cat(
                (edge, row, target, global_context), dim=-1))
        power_share = torch.softmax(
            self.power_head(edge).squeeze(-1), dim=2)
        target_context = edge.mean(dim=1)
        global_target = target_context.mean(
            dim=1, keepdim=True).expand_as(target_context)
        price = torch.softmax(self.price_head(torch.cat(
            (target_context, global_target), dim=-1)).squeeze(-1), dim=1)
        return power_share, price


def _prepare_dataset(
    instances: list[tuple[np.ndarray, np.ndarray, int, int]],
) -> dict[str, np.ndarray]:
    gain_rows: list[np.ndarray] = []
    budget_rows: list[np.ndarray] = []
    harmonic_rows: list[np.ndarray] = []
    exact_share_rows: list[np.ndarray] = []
    exact_worst_rows: list[float] = []
    exact_time_rows: list[float] = []
    seeds: list[int] = []
    frames: list[int] = []
    for gain, budget, seed, frame in instances:
        gain_scale = max(float(np.max(gain)), 1.0e-300)
        budget_scale = max(float(np.max(budget)), 1.0e-12)
        normalized_gain = gain / gain_scale
        normalized_budget = budget / budget_scale
        harmonic_power, _worst, _upper = _harmonic_safe_start(
            normalized_gain, normalized_budget)
        harmonic_share = np.divide(
            harmonic_power.sum(axis=0),
            float(np.sum(harmonic_power)),
        )
        started = perf_counter_ns()
        exact = solve_fixed_structure_maxmin_power_lp(
            normalized_gain, normalized_budget)
        exact_time_rows.append((perf_counter_ns() - started) / 1.0e6)
        exact_share = np.divide(
            exact.power_w,
            normalized_budget[:, None],
            out=np.full_like(exact.power_w, 1.0 / gain.shape[1]),
            where=normalized_budget[:, None] > 0.0,
        )
        gain_rows.append(normalized_gain.astype(np.float32))
        budget_rows.append(normalized_budget.astype(np.float32))
        harmonic_rows.append(harmonic_share.astype(np.float32))
        exact_share_rows.append(exact_share.astype(np.float32))
        exact_worst_rows.append(float(exact.worst_deflection))
        seeds.append(int(seed))
        frames.append(int(frame))
    return {
        "gain": np.stack(gain_rows),
        "budget": np.stack(budget_rows),
        "harmonic": np.stack(harmonic_rows),
        "exact_share": np.stack(exact_share_rows),
        "exact_worst": np.asarray(exact_worst_rows, dtype=np.float32),
        "exact_time_ms": np.asarray(exact_time_rows, dtype=np.float64),
        "seed": np.asarray(seeds, dtype=np.int64),
        "frame": np.asarray(frames, dtype=np.int64),
    }


def _split_by_seed(
    seeds: np.ndarray,
    *,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    unique = np.unique(seeds)
    test_count = max(1, int(np.ceil(float(test_fraction) * unique.size)))
    # Interleaving ordered seeds prevents all large seed identifiers from
    # becoming test-only while preserving a strict scene-level split.
    test_seeds = unique[np.linspace(
        0, unique.size - 1, test_count, dtype=np.int64)]
    test_mask = np.isin(seeds, test_seeds)
    return (
        np.flatnonzero(~test_mask),
        np.flatnonzero(test_mask),
        [int(v) for v in unique[~np.isin(unique, test_seeds)]],
        [int(v) for v in test_seeds],
    )


def _tensor(data: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(data[indices], dtype=np.float32))


def _train(
    model: EquivariantPowerRefiner,
    data: dict[str, np.ndarray],
    train_indices: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(int(seed))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1.0e-5)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, int(epochs) + 1):
        order = train_indices[torch.randperm(
            train_indices.size, generator=generator).numpy()]
        losses: list[float] = []
        regrets: list[float] = []
        for offset in range(0, order.size, int(batch_size)):
            index = order[offset:offset + int(batch_size)]
            gain = _tensor(data["gain"], index)
            budget = _tensor(data["budget"], index)
            harmonic = _tensor(data["harmonic"], index)
            label_share = _tensor(data["exact_share"], index)
            exact_worst = _tensor(data["exact_worst"], index)
            share, prices = model(gain, budget, harmonic)
            power = budget[:, :, None] * share
            deflection = (gain * power).sum(dim=1)
            primal = deflection.amin(dim=1)
            dual = (
                budget * (prices[:, None, :] * gain).amax(dim=2)
            ).sum(dim=1)
            relative_regret = torch.relu(
                exact_worst - primal
            ) / exact_worst.clamp_min(1.0e-8)
            relative_dual_excess = torch.relu(
                dual - exact_worst
            ) / exact_worst.clamp_min(1.0e-8)
            supervised = torch.mean((share - label_share) ** 2)
            loss = (
                torch.mean(relative_regret ** 2)
                + 0.25 * torch.mean(relative_dual_excess ** 2)
                + 0.10 * supervised
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            regrets.append(float(relative_regret.mean().detach()))
        if epoch == 1 or epoch == int(epochs) or epoch % 25 == 0:
            history.append({
                "epoch": int(epoch),
                "loss": float(np.mean(losses)),
                "mean_relative_regret": float(np.mean(regrets)),
            })
    return history


@torch.inference_mode()
def _evaluate(
    model: EquivariantPowerRefiner,
    data: dict[str, np.ndarray],
    test_indices: np.ndarray,
    *,
    certificate_tolerance: float,
) -> dict[str, object]:
    model.eval()
    # Quality is evaluated in one batch; latency is evaluated per scene to
    # match one weak UAV rather than a centralized batch accelerator.
    gain = _tensor(data["gain"], test_indices)
    budget = _tensor(data["budget"], test_indices)
    harmonic = _tensor(data["harmonic"], test_indices)
    share, prices = model(gain, budget, harmonic)
    power = budget[:, :, None] * share
    deflection = (gain * power).sum(dim=1)
    primal = deflection.amin(dim=1).cpu().numpy().astype(np.float64)
    dual = (
        budget * (prices[:, None, :] * gain).amax(dim=2)
    ).sum(dim=1).cpu().numpy().astype(np.float64)
    exact = np.asarray(data["exact_worst"][test_indices], dtype=np.float64)
    ratio = np.divide(primal, exact, out=np.ones_like(primal), where=exact > 0.0)
    relative_gap = np.maximum(dual - primal, 0.0) / np.maximum(dual, 1.0e-300)
    row_error = torch.abs(power.sum(dim=2) - budget).amax(
        dim=1).cpu().numpy().astype(np.float64)

    first = int(test_indices[0])
    for _ in range(100):
        model(
            _tensor(data["gain"], np.asarray([first])),
            _tensor(data["budget"], np.asarray([first])),
            _tensor(data["harmonic"], np.asarray([first])),
        )
    timing_ms: list[float] = []
    repeats = max(4, int(np.ceil(256 / test_indices.size)))
    for _ in range(repeats):
        for index in test_indices:
            singleton = np.asarray([int(index)])
            started = perf_counter_ns()
            model(
                _tensor(data["gain"], singleton),
                _tensor(data["budget"], singleton),
                _tensor(data["harmonic"], singleton),
            )
            timing_ms.append((perf_counter_ns() - started) / 1.0e6)
    fallback = relative_gap > float(certificate_tolerance)
    exact_time = np.asarray(data["exact_time_ms"][test_indices], dtype=np.float64)
    return {
        "utility_ratio_to_exact": _quantiles(ratio),
        "ratio_at_least_0_995_rate": float(np.mean(ratio >= 0.995)),
        "relative_primal_dual_gap": _quantiles(relative_gap),
        "certificate_pass_rate": float(np.mean(~fallback)),
        "exact_fallback_rate": float(np.mean(fallback)),
        "rf_feasible_rate": float(np.mean(row_error <= 1.0e-6)),
        "max_row_budget_error": float(np.max(row_error)),
        "single_node_inference_time_ms": _quantiles(timing_ms),
        "exact_lp_time_ms": _quantiles(exact_time),
        "inference_only_paired_speedup_p50": float(
            np.median(exact_time) / np.median(timing_ms)),
        "end_to_end_with_fallback_expected_time_ms": float(
            np.mean(timing_ms)
            + float(np.mean(fallback)) * float(np.mean(exact_time))
        ),
        "passes_quality_gate": bool(np.quantile(ratio, 0.05) >= 0.995),
        "passes_latency_gate": bool(np.quantile(timing_ms, 0.95) <= 0.5),
        "passes_feasibility_gate": bool(np.all(row_error <= 1.0e-6)),
        "passes_fallback_gate": bool(np.mean(fallback) <= 0.05),
    }


def benchmark(
    trace_path: Path,
    config_path: Path,
    *,
    max_samples: int,
    hidden_dim: int,
    blocks: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    test_fraction: float,
    certificate_tolerance: float,
    seed: int,
) -> tuple[dict[str, object], EquivariantPowerRefiner]:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    instances = _load_instances(
        trace_path, config_path, max_samples=int(max_samples))
    data = _prepare_dataset(instances)
    train_indices, test_indices, train_seeds, test_seeds = _split_by_seed(
        data["seed"], test_fraction=float(test_fraction))
    if train_indices.size < 2 or test_indices.size < 1:
        raise RuntimeError("insufficient distinct trace seeds for strict split")
    model = EquivariantPowerRefiner(
        hidden_dim=int(hidden_dim), blocks=int(blocks))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history = _train(
        model,
        data,
        train_indices,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        seed=int(seed),
    )
    evaluation = _evaluate(
        model,
        data,
        test_indices,
        certificate_tolerance=float(certificate_tolerance),
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": (
            "offline seed-disjoint shadow test of equivariant harmonic-to-AI "
            "power refinement; no deployment or closed-loop claim"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "sample_count": int(len(instances)),
        "train_sample_count": int(train_indices.size),
        "test_sample_count": int(test_indices.size),
        "train_seed_count": len(train_seeds),
        "test_seed_count": len(test_seeds),
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "num_uavs": int(data["gain"].shape[1]),
        "num_targets": int(data["gain"].shape[2]),
        "model": {
            "kind": "bipartite_permutation_equivariant_residual_refiner",
            "hidden_dim": int(hidden_dim),
            "blocks": int(blocks),
            "parameter_count": int(parameter_count),
            "hard_projection": "per-UAV budget times target softmax",
            "dual_certificate": (
                "simplex target prices and separable max-response upper bound"
            ),
        },
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "history": history,
        },
        "certificate_tolerance": float(certificate_tolerance),
        "evaluation": evaluation,
        "deployment_gate_passed": bool(all((
            evaluation["passes_quality_gate"],
            evaluation["passes_latency_gate"],
            evaluation["passes_feasibility_gate"],
            evaluation["passes_fallback_gate"],
        ))),
    }
    return result, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--certificate-tolerance", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()
    result, model = benchmark(
        args.trace,
        args.config,
        max_samples=int(args.max_samples),
        hidden_dim=int(args.hidden_dim),
        blocks=int(args.blocks),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        test_fraction=float(args.test_fraction),
        certificate_tolerance=float(args.certificate_tolerance),
        seed=int(args.seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "hidden_dim": int(args.hidden_dim),
            "blocks": int(args.blocks),
            "metadata": result,
        }, args.model_output)
    print(json.dumps({
        "output": str(args.output),
        "model_output": (
            None if args.model_output is None else str(args.model_output)),
        "deployment_gate_passed": result["deployment_gate_passed"],
        "evaluation": result["evaluation"],
    }, indent=2))


if __name__ == "__main__":
    main()
