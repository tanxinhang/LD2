"""Scientific core for communication-limited correlated soft evidence.

For a delivered evidence set ``S`` with a common covariance under both
hypotheses, the optimal linear detector has

    D(S) = delta_S.T @ Sigma_S^{-1} @ delta_S,
    w_S  = Sigma_S^{-1} @ delta_S.

The marginal value of adding source ``j`` follows from the Schur complement:

    Delta D_j(S) =
      (delta_j - Sigma_jS Sigma_S^{-1} delta_S)^2
      / (Sigma_jj - Sigma_jS Sigma_S^{-1} Sigma_Sj).

This module deliberately contains no epoch, commit, provenance or replay
logic.  Those are assurance-shell concerns.  Selection uses only source-local
evidence descriptors, native communication costs and the covariance model;
fusion evaluates only the evidence that was actually delivered.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


_SPD_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class EvidenceSelectionResult:
    """One deterministic evidence schedule and its model value."""

    selected: tuple[int, ...]
    deflection: float
    communication_bits: int
    rule: str


def _validated_model(
    mean_shift: np.ndarray,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(mean_shift, dtype=np.float64).reshape(-1)
    sigma = np.asarray(covariance, dtype=np.float64)
    if delta.size < 1 or sigma.shape != (delta.size, delta.size):
        raise ValueError("mean_shift and covariance must have shapes (n,), (n,n)")
    if np.any(~np.isfinite(delta)) or np.any(~np.isfinite(sigma)):
        raise ValueError("evidence model must be finite")
    if not np.allclose(sigma, sigma.T, atol=1.0e-12, rtol=1.0e-12):
        raise ValueError("covariance must be symmetric")
    sigma = 0.5 * (sigma + sigma.T)
    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return delta, sigma


def _normalized_indices(
    indices: Iterable[int],
    size: int,
) -> tuple[int, ...]:
    result = tuple(sorted({int(index) for index in indices}))
    if any(index < 0 or index >= int(size) for index in result):
        raise ValueError("evidence index lies outside model support")
    return result


def optimal_linear_soft_fusion(
    mean_shift: np.ndarray,
    covariance: np.ndarray,
    selected: Sequence[int] | None = None,
) -> tuple[float, np.ndarray]:
    """Return exact Deflection and full-length optimal linear weights.

    A Cholesky solve is used instead of explicitly forming ``Sigma^{-1}``.
    Weights use the canonical scale ``w=Sigma^{-1} delta``; multiplying them
    by any non-zero scalar leaves Deflection unchanged.
    """
    delta, sigma = _validated_model(mean_shift, covariance)
    chosen = _normalized_indices(
        range(delta.size) if selected is None else selected,
        delta.size,
    )
    weights = np.zeros(delta.size, dtype=np.float64)
    if not chosen:
        return 0.0, weights
    index = np.asarray(chosen, dtype=np.int64)
    sub_delta = delta[index]
    sub_sigma = sigma[np.ix_(index, index)]
    sub_weights = np.linalg.solve(sub_sigma, sub_delta)
    deflection = float(sub_delta @ sub_weights)
    if deflection < -_SPD_TOLERANCE:
        raise RuntimeError("positive-definite model produced negative Deflection")
    weights[index] = sub_weights
    return max(deflection, 0.0), weights


def conditional_deflection_gain(
    mean_shift: np.ndarray,
    covariance: np.ndarray,
    selected: Sequence[int],
    candidate: int,
) -> float:
    """Return the exact Schur-complement gain of one new source."""
    delta, sigma = _validated_model(mean_shift, covariance)
    chosen = _normalized_indices(selected, delta.size)
    j = int(candidate)
    if j < 0 or j >= delta.size:
        raise ValueError("candidate lies outside model support")
    if j in chosen:
        return 0.0
    if not chosen:
        return float(delta[j] ** 2 / sigma[j, j])
    index = np.asarray(chosen, dtype=np.int64)
    cross = sigma[j, index]
    sub_sigma = sigma[np.ix_(index, index)]
    solved_delta = np.linalg.solve(sub_sigma, delta[index])
    solved_cross = np.linalg.solve(sub_sigma, sigma[index, j])
    innovation = float(delta[j] - cross @ solved_delta)
    conditional_variance = float(sigma[j, j] - cross @ solved_cross)
    scale = max(float(sigma[j, j]), 1.0)
    if conditional_variance <= _SPD_TOLERANCE * scale:
        raise RuntimeError("conditional covariance lost positive definiteness")
    return float(innovation * innovation / conditional_variance)


def _validated_transport(
    size: int,
    bits: np.ndarray,
    success_probability: np.ndarray | None,
    latency_s: np.ndarray | None,
    deadline_s: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_cost = np.asarray(bits).reshape(-1)
    if (
        raw_cost.shape != (size,)
        or not np.issubdtype(raw_cost.dtype, np.number)
        or np.any(~np.isfinite(raw_cost.astype(np.float64)))
        or np.any(raw_cost.astype(np.float64) < 0.0)
        or np.any(raw_cost.astype(np.float64) != np.floor(
            raw_cost.astype(np.float64)))
    ):
        raise ValueError("bits must be a non-negative integer n-vector")
    cost = raw_cost.astype(np.int64)
    success = (
        np.ones(size, dtype=np.float64)
        if success_probability is None
        else np.asarray(success_probability, dtype=np.float64).reshape(-1)
    )
    latency = (
        np.zeros(size, dtype=np.float64)
        if latency_s is None
        else np.asarray(latency_s, dtype=np.float64).reshape(-1)
    )
    if (
        success.shape != (size,) or latency.shape != (size,)
        or np.any(~np.isfinite(success)) or np.any((success < 0.0) | (success > 1.0))
        or np.any(~np.isfinite(latency)) or np.any(latency < 0.0)
    ):
        raise ValueError("success probability/latency must be valid n-vectors")
    if deadline_s is not None and (
        not np.isfinite(deadline_s) or float(deadline_s) < 0.0
    ):
        raise ValueError("deadline_s must be finite and non-negative")
    eligible = success > 0.0
    if deadline_s is not None:
        eligible &= latency <= float(deadline_s)
    return cost, success, latency, eligible


def conditional_information_greedy(
    mean_shift: np.ndarray,
    covariance: np.ndarray,
    bits: np.ndarray,
    budget_bits: int,
    *,
    initial_selected: Sequence[int] = (),
    success_probability: np.ndarray | None = None,
    latency_s: np.ndarray | None = None,
    deadline_s: float | None = None,
    latency_price_bits_per_s: float = 0.0,
) -> EvidenceSelectionResult:
    """Greedily schedule the largest conditional information per cost.

    Reliability multiplies the one-step gain and latency may enter the ranking
    denominator after conversion to bit-equivalent cost.  This ranking is a
    transparent heuristic; only the returned fusion Deflection is exact.
    Native bit budget and deadline remain separate hard constraints.
    """
    delta, sigma = _validated_model(mean_shift, covariance)
    cost, success, latency, eligible = _validated_transport(
        delta.size, bits, success_probability, latency_s, deadline_s)
    budget = int(budget_bits)
    latency_price = float(latency_price_bits_per_s)
    if (
        float(budget_bits) != float(budget) or budget < 0
        or not np.isfinite(latency_price) or latency_price < 0.0
    ):
        raise ValueError("budget and latency price must be non-negative")
    initial = _normalized_indices(initial_selected, delta.size)
    if any(cost[index] != 0 for index in initial):
        raise ValueError("initial local evidence must have zero communication bits")
    selected = list(initial)
    used = 0
    while True:
        best: tuple[float, float, int] | None = None
        for candidate in range(delta.size):
            if candidate in selected or not bool(eligible[candidate]):
                continue
            if cost[candidate] <= 0:
                raise ValueError("non-local candidate evidence must have positive bits")
            if used + int(cost[candidate]) > budget:
                continue
            gain = conditional_deflection_gain(delta, sigma, selected, candidate)
            denominator = float(cost[candidate]) + latency_price * latency[candidate]
            score = float(success[candidate]) * gain / denominator
            key = (score, gain, -candidate)
            if best is None or key > best:
                best = key
        if best is None or best[0] <= 0.0:
            break
        candidate = -best[2]
        selected.append(candidate)
        used += int(cost[candidate])
    chosen = tuple(sorted(selected))
    value, _ = optimal_linear_soft_fusion(delta, sigma, chosen)
    return EvidenceSelectionResult(
        selected=chosen,
        deflection=value,
        communication_bits=used,
        rule="conditional_information_greedy",
    )


def correlation_unaware_greedy(
    mean_shift: np.ndarray,
    covariance: np.ndarray,
    bits: np.ndarray,
    budget_bits: int,
    **kwargs,
) -> EvidenceSelectionResult:
    """Ablation: select under diagonal covariance, fuse under the true model."""
    delta, sigma = _validated_model(mean_shift, covariance)
    diagonal = np.diag(np.diag(sigma))
    scheduled = conditional_information_greedy(
        delta, diagonal, bits, budget_bits, **kwargs)
    value, _ = optimal_linear_soft_fusion(delta, sigma, scheduled.selected)
    return EvidenceSelectionResult(
        selected=scheduled.selected,
        deflection=value,
        communication_bits=scheduled.communication_bits,
        rule="correlation_unaware_greedy",
    )


def exact_budgeted_selection(
    mean_shift: np.ndarray,
    covariance: np.ndarray,
    bits: np.ndarray,
    budget_bits: int,
    *,
    initial_selected: Sequence[int] = (),
    latency_s: np.ndarray | None = None,
    deadline_s: float | None = None,
    maximum_optional_sources: int = 20,
) -> EvidenceSelectionResult:
    """Exhaustive small-scale oracle for the exact delivered-set objective."""
    delta, sigma = _validated_model(mean_shift, covariance)
    cost, _, _, eligible = _validated_transport(
        delta.size, bits, None, latency_s, deadline_s)
    budget = int(budget_bits)
    if float(budget_bits) != float(budget) or budget < 0:
        raise ValueError("budget_bits must be non-negative")
    initial = _normalized_indices(initial_selected, delta.size)
    if any(cost[index] != 0 for index in initial):
        raise ValueError("initial local evidence must have zero communication bits")
    optional = tuple(
        index for index in range(delta.size)
        if index not in initial and bool(eligible[index])
    )
    if any(cost[index] <= 0 for index in optional):
        raise ValueError("non-local candidate evidence must have positive bits")
    if len(optional) > int(maximum_optional_sources):
        raise ValueError("exact oracle optional-source limit exceeded")
    best_selected = initial
    best_value, _ = optimal_linear_soft_fusion(delta, sigma, initial)
    best_bits = 0
    for count in range(1, len(optional) + 1):
        for subset in combinations(optional, count):
            used = int(np.sum(cost[np.asarray(subset, dtype=np.int64)]))
            if used > budget:
                continue
            chosen = tuple(sorted(initial + subset))
            value, _ = optimal_linear_soft_fusion(delta, sigma, chosen)
            if (
                value > best_value + 1.0e-12
                or (
                    abs(value - best_value) <= 1.0e-12
                    and (used, chosen) < (best_bits, best_selected)
                )
            ):
                best_selected = chosen
                best_value = value
                best_bits = used
    return EvidenceSelectionResult(
        selected=best_selected,
        deflection=float(best_value),
        communication_bits=int(best_bits),
        rule="exact_budgeted_oracle",
    )
