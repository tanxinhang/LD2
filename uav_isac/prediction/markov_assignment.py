"""Finite-state dynamic programming for predictive assignment decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarkovAssignmentPlan:
    """Optimal assignment-state path over a finite prediction horizon."""

    state_indices: tuple[int, ...]
    assignments: np.ndarray
    total_cost: float
    stage_cost: float
    switching_cost: float

    @property
    def first_assignment(self) -> np.ndarray:
        return self.assignments[0].copy()


def assignment_switch_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Fraction of endpoint responsibilities changed by a transition."""

    lhs = np.asarray(left, dtype=np.int64).reshape(-1)
    rhs = np.asarray(right, dtype=np.int64).reshape(-1)
    if lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError("assignment vectors must have the same non-empty shape")
    if np.any(lhs < 0) or np.any(rhs < 0):
        raise ValueError("assignment indices must be non-negative")
    return float(np.mean(lhs != rhs))


def solve_markov_assignment_path(
    candidate_assignments: np.ndarray,
    stage_costs: np.ndarray,
    incumbent_assignment: np.ndarray,
    *,
    switch_penalty: float,
    discount: float = 1.0,
) -> MarkovAssignmentPlan:
    """Solve a deterministic finite-horizon assignment Markov decision chain.

    Candidate state ``s`` represents a complete distributed assignment vector.
    ``stage_costs[h,s]`` must already include the desired physical/risk score
    for horizon step ``h``.  Transitions pay the normalized Hamming distance,
    so the optimizer distinguishes genuine predicted benefit from assignment
    churn.  Stable candidate ordering gives deterministic tie-breaking.
    """

    candidates = np.asarray(candidate_assignments, dtype=np.int64)
    costs = np.asarray(stage_costs, dtype=np.float64)
    incumbent = np.asarray(incumbent_assignment, dtype=np.int64).reshape(-1)
    if candidates.ndim != 2 or candidates.shape[0] == 0 or candidates.shape[1] == 0:
        raise ValueError("candidate_assignments must have shape (S,K)")
    states, endpoints = candidates.shape
    if costs.ndim != 2 or costs.shape[1] != states or costs.shape[0] == 0:
        raise ValueError("stage_costs must have shape (H,S)")
    if incumbent.shape != (endpoints,):
        raise ValueError("incumbent_assignment must have shape (K,)")
    if np.any(candidates < 0) or np.any(incumbent < 0):
        raise ValueError("assignment indices must be non-negative")
    if np.any(~np.isfinite(costs)):
        raise ValueError("stage costs must be finite")
    penalty = float(switch_penalty)
    gamma = float(discount)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("switch_penalty must be finite and non-negative")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("discount must lie in (0,1]")

    transitions = np.mean(
        candidates[:, None, :] != candidates[None, :, :], axis=2)
    initial_switch = np.mean(candidates != incumbent[None, :], axis=1)
    horizon = costs.shape[0]
    value = costs[0] + penalty * initial_switch
    parent = np.full((horizon, states), -1, dtype=np.int64)
    for step in range(1, horizon):
        transition_value = value[:, None] + penalty * transitions
        parent[step] = np.argmin(transition_value, axis=0)
        value = transition_value[parent[step], np.arange(states)]
        value += (gamma ** step) * costs[step]

    terminal = int(np.argmin(value))
    path = np.empty(horizon, dtype=np.int64)
    path[-1] = terminal
    for step in range(horizon - 1, 0, -1):
        path[step - 1] = parent[step, path[step]]
    chosen = candidates[path].copy()
    discounted_stage = float(np.sum(
        (gamma ** np.arange(horizon))
        * costs[np.arange(horizon), path]))
    switching = penalty * assignment_switch_distance(incumbent, chosen[0])
    for step in range(1, horizon):
        switching += penalty * assignment_switch_distance(
            chosen[step - 1], chosen[step])
    return MarkovAssignmentPlan(
        state_indices=tuple(int(v) for v in path),
        assignments=chosen,
        total_cost=float(value[terminal]),
        stage_cost=discounted_stage,
        switching_cost=float(switching),
    )
