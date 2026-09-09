"""Microbenchmark Pareto actor updates against one weighted backward pass."""

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

from uav_isac.optimization import (  # noqa: E402
    assign_constrained_pareto_gradients,
    joint_policy_detection_cvar_residual,
)


class SyntheticMultiObjectiveActor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, objectives: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(objectives)
        ])

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = self.trunk(value)
        return tuple(head(hidden).square().mean() for head in self.heads)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--input", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--objectives", type=int, default=5)
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--solver-iterations", type=int, default=128)
    args = parser.parse_args()
    if min(args.batch, args.input, args.hidden, args.objectives,
           args.agents, args.targets,
           args.iterations, args.solver_iterations) < 1 or args.warmup < 0:
        parser.error("dimensions/iterations must be positive and warmup non-negative")
    if args.solver_tolerance < 0.0:
        parser.error("solver tolerance must be non-negative")

    torch.set_num_threads(1)
    torch.manual_seed(20260908)
    actor = SyntheticMultiObjectiveActor(
        args.input, args.hidden, args.objectives)
    inputs = torch.randn(args.batch, args.input)
    detection_probability = torch.rand(args.batch, args.targets)
    old_agent_log_probability = torch.randn(args.batch, args.agents)
    new_agent_log_probability = old_agent_log_probability.detach().clone()
    new_agent_log_probability.requires_grad_(True)

    def weighted_backward() -> None:
        actor.zero_grad(set_to_none=True)
        losses = actor(inputs)
        torch.stack(losses).mean().backward()

    def pareto_backward():
        actor.zero_grad(set_to_none=True)
        losses = actor(inputs)
        return assign_constrained_pareto_gradients(
            losses, actor.parameters(), tolerance=args.solver_tolerance,
            max_iterations=args.solver_iterations)

    def joint_cvar_backward() -> None:
        new_agent_log_probability.grad = None
        residual = joint_policy_detection_cvar_residual(
            detection_probability,
            0.60,
            new_agent_log_probability,
            old_agent_log_probability,
            tail_fraction=0.20,
        )
        residual.sum().backward()

    for _ in range(args.warmup):
        weighted_backward()
        pareto_backward()
        joint_cvar_backward()
    weighted_ms: list[float] = []
    pareto_ms: list[float] = []
    pareto_iterations: list[int] = []
    pareto_converged: list[float] = []
    joint_cvar_ms: list[float] = []
    for _ in range(args.iterations):
        begin = perf_counter_ns()
        weighted_backward()
        weighted_ms.append((perf_counter_ns() - begin) / 1.0e6)
        begin = perf_counter_ns()
        result = pareto_backward()
        pareto_ms.append((perf_counter_ns() - begin) / 1.0e6)
        pareto_iterations.append(result.pareto.iterations)
        pareto_converged.append(float(result.pareto.converged))
        begin = perf_counter_ns()
        joint_cvar_backward()
        joint_cvar_ms.append((perf_counter_ns() - begin) / 1.0e6)

    print(json.dumps({
        "schema_version": "constrained-pareto-benchmark/v1",
        "batch": args.batch,
        "input_dim": args.input,
        "hidden_dim": args.hidden,
        "objectives": args.objectives,
        "agents": args.agents,
        "targets": args.targets,
        "solver_tolerance": args.solver_tolerance,
        "solver_iteration_limit": args.solver_iterations,
        "parameters": sum(parameter.numel() for parameter in actor.parameters()),
        "weighted_backward_mean_ms": float(np.mean(weighted_ms)),
        "weighted_backward_p95_ms": percentile(weighted_ms, 95),
        "pareto_backward_mean_ms": float(np.mean(pareto_ms)),
        "pareto_backward_p95_ms": percentile(pareto_ms, 95),
        "pareto_to_weighted_ratio": float(
            np.mean(pareto_ms) / max(np.mean(weighted_ms), 1.0e-12)),
        "pareto_solver_iterations_mean": float(np.mean(pareto_iterations)),
        "pareto_solver_converged_rate": float(np.mean(pareto_converged)),
        "joint_cvar_forward_backward_mean_ms": float(np.mean(joint_cvar_ms)),
        "joint_cvar_forward_backward_p95_ms": percentile(joint_cvar_ms, 95),
        "note": "CPU single-thread training microbenchmark; no optimizer step",
    }, indent=2))


if __name__ == "__main__":
    main()
