"""Finite-state dynamic programming for predictive assignment decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


@dataclass(frozen=True)
class MarkovScenarioPlan:
    """Beam-search plan whose continuous state follows the chosen actions."""

    action_indices: tuple[int, ...]
    assignments: np.ndarray
    states: tuple[np.ndarray, ...]
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


def build_target_priority_swap_neighborhood(
    incumbent_assignment: np.ndarray,
    *,
    num_targets: int,
    target_priority: np.ndarray | None = None,
    max_candidates: int = 32,
) -> np.ndarray:
    """Build a bounded, count-preserving neighborhood around an assignment.

    The incumbent is always candidate zero.  For each target in priority
    order, endpoints currently responsible for it are swapped with every other
    endpoint in stable index order.  A swap changes which UAV approaches a
    weak/uncertain target while preserving the exact number of responsibilities
    assigned to every target.  This avoids both exponential enumeration and
    the missing-target artifacts created by single-entry replacement.
    """

    incumbent = np.asarray(incumbent_assignment, dtype=np.int64).reshape(-1)
    targets = int(num_targets)
    limit = int(max_candidates)
    if incumbent.size == 0 or targets != num_targets or targets < 1:
        raise ValueError("incumbent and num_targets must be non-empty and valid")
    if np.any(incumbent < 0) or np.any(incumbent >= targets):
        raise ValueError("incumbent contains an invalid target index")
    if limit != max_candidates or limit < 1:
        raise ValueError("max_candidates must be a positive integer")
    if target_priority is None:
        priority = np.arange(targets, dtype=np.int64)
    else:
        priority = np.asarray(target_priority, dtype=np.int64).reshape(-1)
        if (
            priority.size == 0 or np.any(priority < 0)
            or np.any(priority >= targets)
            or np.unique(priority).size != priority.size
        ):
            raise ValueError("target_priority must contain unique valid indices")
        missing = np.setdiff1d(
            np.arange(targets, dtype=np.int64), priority, assume_unique=True)
        priority = np.concatenate((priority, missing))

    candidates = [incumbent.copy()]
    seen = {tuple(int(value) for value in incumbent)}
    for target in priority:
        owners = np.flatnonzero(incumbent == int(target))
        for owner in owners:
            for partner in range(incumbent.size):
                if partner == owner or incumbent[partner] == incumbent[owner]:
                    continue
                candidate = incumbent.copy()
                candidate[owner], candidate[partner] = (
                    candidate[partner], candidate[owner])
                key = tuple(int(value) for value in candidate)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) >= limit:
                    return np.stack(candidates)
    return np.stack(candidates)


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


def solve_markov_assignment_scenario_tree(
    candidate_assignments: np.ndarray,
    incumbent_assignment: np.ndarray,
    initial_state: np.ndarray,
    *,
    horizon_steps: int,
    transition: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    stage_cost: Callable[[np.ndarray, np.ndarray, int], float],
    switch_penalty: float,
    discount: float = 1.0,
    beam_width: int = 32,
) -> MarkovScenarioPlan:
    """Search an action-conditioned continuous-state Markov scenario tree.

    Unlike :func:`solve_markov_assignment_path`, future costs are evaluated on
    the state reached by the entire preceding assignment path.  This is the
    required contract when UAV geometry changes with earlier assignments.
    Callbacks must be deterministic and causal; physics-specific structure and
    power solves remain outside this generic kernel.
    """

    candidates = np.asarray(candidate_assignments, dtype=np.int64)
    incumbent = np.asarray(incumbent_assignment, dtype=np.int64).reshape(-1)
    state0 = np.asarray(initial_state, dtype=np.float64)
    steps = int(horizon_steps)
    width = int(beam_width)
    penalty = float(switch_penalty)
    gamma = float(discount)
    if candidates.ndim != 2 or candidates.shape[0] == 0 or candidates.shape[1] == 0:
        raise ValueError("candidate_assignments must have shape (S,K)")
    if incumbent.shape != (candidates.shape[1],):
        raise ValueError("incumbent_assignment must have shape (K,)")
    if np.any(candidates < 0) or np.any(incumbent < 0):
        raise ValueError("assignment indices must be non-negative")
    if state0.size == 0 or np.any(~np.isfinite(state0)):
        raise ValueError("initial_state must be non-empty and finite")
    if steps != horizon_steps or steps < 1:
        raise ValueError("horizon_steps must be a positive integer")
    if width != beam_width or width < 1:
        raise ValueError("beam_width must be a positive integer")
    if not callable(transition) or not callable(stage_cost):
        raise ValueError("transition and stage_cost must be callable")
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("switch_penalty must be finite and non-negative")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("discount must lie in (0,1]")

    # (total, stage subtotal, switch subtotal, path, states, state, previous)
    beam = [(0.0, 0.0, 0.0, tuple(), tuple(), state0.copy(), incumbent)]
    for step in range(steps):
        expanded = []
        for total, stage_sum, switch_sum, path, states, state, previous in beam:
            for action_index, assignment in enumerate(candidates):
                next_state = np.asarray(
                    transition(state.copy(), assignment.copy(), step),
                    dtype=np.float64,
                )
                if next_state.shape != state0.shape or np.any(~np.isfinite(next_state)):
                    raise ValueError("transition returned an invalid state")
                physical = float(stage_cost(
                    next_state.copy(), assignment.copy(), step))
                if not np.isfinite(physical):
                    raise ValueError("stage_cost returned a non-finite value")
                stage_increment = (gamma ** step) * physical
                switch_increment = penalty * assignment_switch_distance(
                    previous, assignment)
                expanded.append((
                    total + stage_increment + switch_increment,
                    stage_sum + stage_increment,
                    switch_sum + switch_increment,
                    path + (int(action_index),),
                    states + (next_state.copy(),),
                    next_state,
                    assignment.copy(),
                ))
        expanded.sort(key=lambda item: (item[0], item[3]))
        beam = expanded[:width]

    best = beam[0]
    path = best[3]
    return MarkovScenarioPlan(
        action_indices=path,
        assignments=candidates[np.asarray(path, dtype=np.int64)].copy(),
        states=best[4],
        total_cost=float(best[0]),
        stage_cost=float(best[1]),
        switching_cost=float(best[2]),
    )
