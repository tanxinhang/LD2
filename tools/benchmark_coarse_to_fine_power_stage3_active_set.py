#!/usr/bin/env python
"""Stage-3 test: use the shadow AI only to screen LP power variables.

The learned refiner never directly controls RF power here.  Its largest row
shares nominate an active set, deterministic target-coverage repair prevents a
structural zero, and a reduced exact LP computes the executed allocation.  LP
target multipliers are lifted to the full gain matrix, yielding a valid global
dual upper bound even when the learned active set missed useful variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
from scipy.optimize import linprog
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_coarse_to_fine_power_stage1 import (  # noqa: E402
    _load_instances,
    _quantiles,
)
from tools.benchmark_coarse_to_fine_power_stage2_ai import (  # noqa: E402
    EquivariantPowerRefiner,
    _prepare_dataset,
    _split_by_seed,
    _tensor,
)
from uav_isac.utils.checkpoint_loading import safe_torch_load  # noqa: E402


def _coverage_repaired_mask(
    gain: np.ndarray,
    predicted_share: np.ndarray,
    top_m: int,
) -> np.ndarray:
    K, Q = gain.shape
    count = min(max(1, int(top_m)), Q)
    mask = np.zeros((K, Q), dtype=bool)
    top = np.argpartition(predicted_share, Q - count, axis=1)[:, -count:]
    mask[np.arange(K)[:, None], top] = True
    # This repair uses the supplied fixed-structure gain matrix, which is
    # already the optimizer input.  It adds at most Q edges and guarantees
    # that screening alone cannot make a reachable target unreachable.
    mask[np.argmax(gain, axis=0), np.arange(Q)] = True
    return mask


def _solve_restricted(
    gain: np.ndarray,
    budget: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    K, Q = gain.shape
    edges = np.argwhere(active)
    E = int(edges.shape[0])
    objective = np.zeros(E + 1, dtype=np.float64)
    objective[-1] = -1.0
    target_rows = np.zeros((Q, E + 1), dtype=np.float64)
    target_rows[:, -1] = 1.0
    budget_rows = np.zeros((K, E + 1), dtype=np.float64)
    for column, (transmitter, target) in enumerate(edges):
        target_rows[int(target), column] = -gain[
            int(transmitter), int(target)]
        budget_rows[int(transmitter), column] = 1.0
    solved = linprog(
        objective,
        A_ub=np.concatenate((target_rows, budget_rows), axis=0),
        b_ub=np.concatenate((np.zeros(Q), budget)),
        bounds=[(0.0, None)] * (E + 1),
        method="highs",
    )
    if not solved.success or solved.x is None:
        raise RuntimeError(f"reduced LP failed: {solved.message}")
    power = np.zeros((K, Q), dtype=np.float64)
    power[edges[:, 0], edges[:, 1]] = np.maximum(solved.x[:E], 0.0)
    deflection = np.sum(gain * power, axis=0)
    lower = float(np.min(deflection))
    # Target lower-bound multipliers form a simplex at a non-degenerate
    # max-min optimum. Normalize small HiGHS residuals, then evaluate the full
    # (not screened) transmitter response for a valid global upper bound.
    marginals = np.asarray(solved.ineqlin.marginals[:Q], dtype=np.float64)
    prices = np.maximum(-marginals, 0.0)
    total = float(np.sum(prices))
    prices = (
        prices / total
        if total > 0.0
        else np.full(Q, 1.0 / Q, dtype=np.float64)
    )
    upper = float(np.sum(
        budget * np.max(prices[None, :] * gain, axis=1)))
    return power, lower, max(upper, lower), E


@torch.inference_mode()
def benchmark(
    trace_path: Path,
    config_path: Path,
    model_path: Path,
    *,
    max_samples: int,
    top_m_values: tuple[int, ...],
    test_fraction: float,
    certificate_tolerance: float,
) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = safe_torch_load(
        model_path,
        map_location="cpu",
        description="coarse-to-fine active-set model",
        required_keys=("hidden_dim", "blocks"),
        state_dict_keys=("state_dict",),
    )
    model = EquivariantPowerRefiner(
        hidden_dim=int(checkpoint["hidden_dim"]),
        blocks=int(checkpoint["blocks"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    instances = _load_instances(
        trace_path, config_path, max_samples=int(max_samples))
    data = _prepare_dataset(instances)
    _train, test, _train_seeds, test_seeds = _split_by_seed(
        data["seed"], test_fraction=float(test_fraction))
    gain_batch = _tensor(data["gain"], test)
    budget_batch = _tensor(data["budget"], test)
    harmonic_batch = _tensor(data["harmonic"], test)
    predicted_share, _price = model(
        gain_batch, budget_batch, harmonic_batch)
    predicted = predicted_share.cpu().numpy().astype(np.float64)
    exact = np.asarray(data["exact_worst"][test], dtype=np.float64)
    exact_time = np.asarray(data["exact_time_ms"][test], dtype=np.float64)
    rows: dict[str, dict[str, object]] = {}
    K, Q = data["gain"].shape[1:]
    for top_m in top_m_values:
        lower_values: list[float] = []
        upper_values: list[float] = []
        timing_ms: list[float] = []
        edge_counts: list[int] = []
        feasible: list[bool] = []
        for position, index in enumerate(test):
            gain = np.asarray(data["gain"][index], dtype=np.float64)
            budget = np.asarray(data["budget"][index], dtype=np.float64)
            singleton = np.asarray([int(index)])
            started = perf_counter_ns()
            share, _prices = model(
                _tensor(data["gain"], singleton),
                _tensor(data["budget"], singleton),
                _tensor(data["harmonic"], singleton),
            )
            mask = _coverage_repaired_mask(
                gain, share[0].cpu().numpy(), int(top_m))
            power, lower, upper, edge_count = _solve_restricted(
                gain, budget, mask)
            timing_ms.append((perf_counter_ns() - started) / 1.0e6)
            lower_values.append(lower)
            upper_values.append(upper)
            edge_counts.append(edge_count)
            feasible.append(bool(
                np.all(power >= -1.0e-12)
                and np.all(np.sum(power, axis=1) <= budget + 1.0e-9)
            ))
        lower_array = np.asarray(lower_values, dtype=np.float64)
        upper_array = np.asarray(upper_values, dtype=np.float64)
        ratio = lower_array / np.maximum(exact, 1.0e-300)
        gap = np.maximum(upper_array - lower_array, 0.0) / np.maximum(
            upper_array, 1.0e-300)
        time_array = np.asarray(timing_ms, dtype=np.float64)
        certified = gap <= float(certificate_tolerance)
        rows[f"top{top_m}"] = {
            "mean_active_edges": float(np.mean(edge_counts)),
            "mean_variable_retention_rate": float(
                np.mean(edge_counts) / float(K * Q)),
            "utility_ratio_to_exact": _quantiles(ratio),
            "ratio_at_least_0_995_rate": float(np.mean(ratio >= 0.995)),
            "full_relative_primal_dual_gap": _quantiles(gap),
            "certificate_pass_rate": float(np.mean(certified)),
            "rf_feasible_rate": float(np.mean(feasible)),
            "ai_plus_reduced_lp_time_ms": _quantiles(time_array),
            "paired_speedup": _quantiles(exact_time / time_array),
            "passes_quality_gate": bool(np.quantile(ratio, 0.05) >= 0.995),
            "passes_latency_improvement_gate": bool(
                np.median(time_array) < np.median(exact_time)),
            "passes_certificate_gate": bool(np.mean(certified) >= 0.95),
        }
    return {
        "schema_version": 1,
        "scope": (
            "seed-disjoint shadow test; AI screens variables only; coverage-"
            "repaired reduced LP executes; full dual certificate checked"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "model": str(model_path),
        "sample_count": int(len(instances)),
        "test_sample_count": int(test.size),
        "test_seeds": test_seeds,
        "num_uavs": int(K),
        "num_targets": int(Q),
        "certificate_tolerance": float(certificate_tolerance),
        "exact_lp_time_ms": _quantiles(exact_time),
        "active_set_results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--top-m", type=int, nargs="+", default=[2, 3, 4, 6])
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--certificate-tolerance", type=float, default=0.005)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.trace,
        args.config,
        args.model,
        max_samples=int(args.max_samples),
        top_m_values=tuple(sorted(set(int(v) for v in args.top_m))),
        test_fraction=float(args.test_fraction),
        certificate_tolerance=float(args.certificate_tolerance),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "exact_lp_time_ms": result["exact_lp_time_ms"],
        "active_set_results": result["active_set_results"],
    }, indent=2))


if __name__ == "__main__":
    main()
