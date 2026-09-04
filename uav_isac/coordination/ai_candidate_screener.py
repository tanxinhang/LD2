"""AI proposal with analytical certification for online candidate screening."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from torch import nn


class CandidateScoreNetwork(nn.Module):
    """A shared scorer; candidate count does not enter the parameter shape."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))

    def set_normalization(
        self,
        mean: np.ndarray | torch.Tensor,
        scale: np.ndarray | torch.Tensor,
    ) -> None:
        mean_t = torch.as_tensor(
            mean, dtype=self.feature_mean.dtype,
            device=self.feature_mean.device)
        scale_t = torch.as_tensor(
            scale, dtype=self.feature_scale.dtype,
            device=self.feature_scale.device)
        if mean_t.shape != self.feature_mean.shape or scale_t.shape != mean_t.shape:
            raise ValueError("normalization shape does not match feature_dim")
        if bool(torch.any(~torch.isfinite(mean_t))
                or torch.any(~torch.isfinite(scale_t))
                or torch.any(scale_t <= 0.0)):
            raise ValueError("normalization must be finite with positive scale")
        self.feature_mean.copy_(mean_t)
        self.feature_scale.copy_(scale_t)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.network(normalized).squeeze(-1)


@dataclass(frozen=True)
class CertifiedScreeningResult:
    chosen_index: int
    exact_gain: float
    certified: bool
    used_incumbent_fallback: bool
    evaluated_indices: tuple[int, ...]
    proposed_indices: tuple[int, ...]
    maximum_pruned_upper_bound: float
    inference_time_ms: float
    total_screening_time_ms: float


def _stable_descending(values: np.ndarray) -> np.ndarray:
    indices = np.arange(values.size, dtype=np.int64)
    return np.lexsort((indices, -values))


def certified_ai_screen(
    model: nn.Module,
    features: np.ndarray,
    upper_bounds: np.ndarray,
    exact_gain_batch: Callable[[np.ndarray], np.ndarray],
    *,
    topk: int,
    incumbent_index: int,
    max_exact_evaluations: int | None = None,
    verification_batch_size: int = 32,
    device: str | torch.device | None = None,
    certificate_tolerance: float = 1.0e-12,
    execute_best_verified_on_budget_exhaustion: bool = False,
) -> CertifiedScreeningResult:
    """Use AI only to propose an order; accept only an analytically proven best.

    ``upper_bounds[i]`` must upper-bound the exact gain of candidate ``i``.
    If the evaluation budget expires before every remaining upper bound is no
    larger than the best verified gain, the default fail-closed mode returns
    the supplied feasible incumbent.  The optional anytime mode instead
    executes the best *exactly evaluated* feasible candidate while retaining
    ``certified=False``.  Thus lack of an optimality proof need not discard a
    useful feasible action, and model error still cannot authorize a candidate
    whose gain was never verified.
    """
    matrix = np.asarray(features, dtype=np.float32)
    bounds = np.asarray(upper_bounds, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != bounds.size or bounds.size == 0:
        raise ValueError("features/upper_bounds must describe N candidates")
    if not np.all(np.isfinite(matrix)) or np.any(np.isnan(bounds)):
        raise ValueError("features must be finite and bounds cannot be NaN")
    count = bounds.size
    if not 0 <= int(incumbent_index) < count:
        raise ValueError("incumbent_index is out of range")
    proposal_count = min(max(int(topk), 1), count)
    evaluation_limit = (
        count if max_exact_evaluations is None
        else min(max(int(max_exact_evaluations), 1), count))
    if evaluation_limit < 1:
        raise ValueError("max_exact_evaluations must be positive")
    if int(verification_batch_size) < 1:
        raise ValueError("verification_batch_size must be positive")
    verification_batch_size = int(verification_batch_size)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    run_device = torch.device(device) if device is not None else model_device
    total_started = perf_counter()
    started = perf_counter()
    with torch.inference_mode():
        tensor = torch.as_tensor(matrix, device=run_device)
        scores = model(tensor).detach().to("cpu", dtype=torch.float64).numpy()
    if scores.shape != (count,) or not np.all(np.isfinite(scores)):
        raise RuntimeError("AI scorer returned invalid candidate scores")
    inference_ms = 1000.0 * (perf_counter() - started)
    ai_order = _stable_descending(scores)
    proposed = list(ai_order[:proposal_count].astype(int))
    if int(incumbent_index) not in proposed:
        proposed.append(int(incumbent_index))
    proposed = proposed[:evaluation_limit]
    if int(incumbent_index) not in proposed:
        proposed[-1] = int(incumbent_index)

    evaluated: dict[int, float] = {}
    evaluated_mask = np.zeros(count, dtype=bool)

    def evaluate(indices: list[int]) -> None:
        fresh = [index for index in indices if index not in evaluated]
        if not fresh:
            return
        values = np.asarray(
            exact_gain_batch(np.asarray(fresh, dtype=np.int64)),
            dtype=np.float64,
        ).reshape(-1)
        if values.size != len(fresh) or np.any(np.isnan(values)):
            raise RuntimeError("exact verifier returned invalid gains")
        evaluated.update(zip(fresh, values.tolist()))
        evaluated_mask[np.asarray(fresh, dtype=np.int64)] = True

    evaluate(proposed)
    bound_order = _stable_descending(bounds)
    bound_cursor = 0
    while len(evaluated) < evaluation_limit:
        best_index = min(
            evaluated,
            key=lambda index: (-evaluated[index], index),
        )
        best_gain = float(evaluated[best_index])
        while (bound_cursor < count
               and evaluated_mask[int(bound_order[bound_cursor])]):
            bound_cursor += 1
        if bound_cursor >= count:
            break
        next_index = int(bound_order[bound_cursor])
        if float(bounds[next_index]) <= best_gain + certificate_tolerance:
            break
        verification_batch: list[int] = []
        remaining_budget = evaluation_limit - len(evaluated)
        batch_limit = min(verification_batch_size, remaining_budget)
        while bound_cursor < count and len(verification_batch) < batch_limit:
            index = int(bound_order[bound_cursor])
            bound_cursor += 1
            if evaluated_mask[index]:
                continue
            if float(bounds[index]) <= best_gain + certificate_tolerance:
                break
            verification_batch.append(index)
        evaluate(verification_batch)

    best_index = min(
        evaluated,
        key=lambda index: (-evaluated[index], index),
    )
    best_gain = float(evaluated[best_index])
    while (bound_cursor < count
           and evaluated_mask[int(bound_order[bound_cursor])]):
        bound_cursor += 1
    maximum_pruned = (
        float(bounds[int(bound_order[bound_cursor])])
        if bound_cursor < count else -np.inf)
    certified = bool(
        maximum_pruned <= best_gain + certificate_tolerance)
    execute_best = bool(
        certified or execute_best_verified_on_budget_exhaustion)
    chosen = int(best_index if execute_best else incumbent_index)
    return CertifiedScreeningResult(
        chosen_index=chosen,
        exact_gain=float(evaluated[chosen]),
        certified=certified,
        used_incumbent_fallback=bool(not certified and not execute_best),
        evaluated_indices=tuple(sorted(evaluated)),
        proposed_indices=tuple(proposed),
        maximum_pruned_upper_bound=maximum_pruned,
        inference_time_ms=float(inference_ms),
        total_screening_time_ms=float(
            1000.0 * (perf_counter() - total_started)),
    )
