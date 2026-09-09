"""Physics-weighted sparse graph proposals for predictive assignment.

The graph is built without a learned teacher.  KNN controls the edge budget,
counterfactual expected-physics solves provide edge weights, a global bipartite
matching combines them, and a final joint physical solve rejects harmful
linearization error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment


class _Evaluation(Protocol):
    cost: float


class _PhysicalAssignmentModel(Protocol):
    K: int
    Q: int

    def unpack_state(self, state: np.ndarray) -> tuple[np.ndarray, ...]: ...

    def transition(
        self, state: np.ndarray, assignment: np.ndarray, step: int,
    ) -> np.ndarray: ...

    def evaluate(
        self, state: np.ndarray, *, stage_index: int | None = None,
    ) -> _Evaluation: ...


@dataclass(frozen=True)
class PhysicsKNNAssignmentProposal:
    """Auditable output of one physics-weighted graph proposal."""

    assignment: np.ndarray
    raw_assignment: np.ndarray
    adjacency: np.ndarray
    marginal_improvement: np.ndarray
    incumbent_cost: float
    proposal_cost: float
    accepted: bool
    evaluated_edges: int
    neighbors_used: int


@dataclass(frozen=True)
class PhysicsKNNLocalSearchProposal:
    """Budgeted best-improvement path over a sparse physical graph."""

    assignment: np.ndarray
    adjacency: np.ndarray
    incumbent_cost: float
    proposal_cost: float
    accepted_swaps: tuple[tuple[int, int], ...]
    evaluated_assignments: int
    evaluation_budget: int
    neighbors_used: int


def _knn_adjacency(
    uav_xy: np.ndarray,
    target_xy: np.ndarray,
    incumbent: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    """Target-wise KNN graph augmented with every incumbent edge."""

    distance = np.linalg.norm(
        uav_xy[:, None, :] - target_xy[None, :, :], axis=2)
    K, Q = distance.shape
    adjacency = np.zeros((K, Q), dtype=bool)
    count = min(max(int(neighbors), 1), K)
    for target in range(Q):
        nearest = np.argsort(distance[:, target], kind="stable")[:count]
        adjacency[nearest, target] = True
    adjacency[np.arange(K), incumbent] = True
    return adjacency


def propose_physics_knn_assignment(
    model: _PhysicalAssignmentModel,
    state: np.ndarray,
    incumbent_assignment: np.ndarray,
    *,
    stage_index: int,
    k_neighbors: int = 4,
    improvement_tolerance: float = 1.0e-9,
) -> PhysicsKNNAssignmentProposal:
    """Propose a count-preserving graph reassignment with exact rejection.

    The current implementation targets the research K=Q regime, where a
    movement assignment is a permutation.  Each sparse edge ``(uav,target)``
    is valued by a count-preserving two-exchange with the incumbent owner of
    that target.  Hungarian matching composes these exchange values into a
    permutation.  Since the marginal model is not
    additive under bistatic/power coupling, the composed proposal is evaluated
    once more by the full physical model and accepted only on strict gain.
    """

    incumbent = np.asarray(incumbent_assignment, dtype=np.int64).reshape(-1)
    K, Q = int(model.K), int(model.Q)
    stage = int(stage_index)
    neighbors = int(k_neighbors)
    tolerance = float(improvement_tolerance)
    if K != Q:
        raise ValueError("physics KNN matching currently requires K=Q")
    if (
        incumbent.shape != (K,) or np.any(incumbent < 0)
        or np.any(incumbent >= Q)
        or np.unique(incumbent).size != Q
    ):
        raise ValueError("incumbent must be a complete K=Q permutation")
    if stage != stage_index or stage < 0:
        raise ValueError("stage_index must be a non-negative integer")
    if neighbors != k_neighbors or neighbors < 1:
        raise ValueError("k_neighbors must be a positive integer")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("improvement_tolerance must be finite and non-negative")

    uav_pos, _uav_vel, target_pos, target_vel = model.unpack_state(state)
    # Adjacency uses the causal one-step CV mean rather than stale target
    # positions.  A zero-movement surrogate action advances every target with
    # exactly the same transition used by the physical scorer.
    advanced = model.transition(state, incumbent, stage)
    _advanced_uav, _advanced_velocity, advanced_target, _ = (
        model.unpack_state(advanced))
    adjacency = _knn_adjacency(
        uav_pos[:, :2], advanced_target[:, :2], incumbent, neighbors)

    incumbent_cost = float(
        model.evaluate(advanced, stage_index=stage).cost)
    marginal = np.full((K, Q), -np.inf, dtype=np.float64)
    evaluated = 0
    target_owner = np.empty(Q, dtype=np.int64)
    target_owner[incumbent] = np.arange(K, dtype=np.int64)
    for uav, target in np.argwhere(adjacency):
        action = incumbent.copy()
        partner = int(target_owner[int(target)])
        action[int(uav)], action[partner] = action[partner], action[int(uav)]
        next_state = model.transition(state, action, stage)
        edge_cost = float(model.evaluate(next_state, stage_index=stage).cost)
        marginal[int(uav), int(target)] = incumbent_cost - edge_cost
        evaluated += 1

    # A target-wise KNN graph plus incumbent permutation always contains at
    # least one perfect matching. A large finite penalty keeps SciPy versions
    # that reject infinities on the same deterministic path.
    finite = marginal[np.isfinite(marginal)]
    scale = max(1.0, float(np.max(np.abs(finite))) if finite.size else 1.0)
    matching_cost = np.where(np.isfinite(marginal), -marginal, 1.0e9 * scale)
    rows, columns = linear_sum_assignment(matching_cost)
    if rows.size != K or np.any(~adjacency[rows, columns]):
        raise RuntimeError("augmented KNN graph has no valid perfect matching")
    raw = np.empty(K, dtype=np.int64)
    raw[rows] = columns
    proposal_state = model.transition(state, raw, stage)
    proposal_cost = float(model.evaluate(
        proposal_state, stage_index=stage).cost)
    accepted = bool(proposal_cost < incumbent_cost - tolerance)
    assignment = raw.copy() if accepted else incumbent.copy()
    return PhysicsKNNAssignmentProposal(
        assignment=assignment,
        raw_assignment=raw,
        adjacency=adjacency,
        marginal_improvement=marginal,
        incumbent_cost=incumbent_cost,
        proposal_cost=proposal_cost,
        accepted=accepted,
        evaluated_edges=evaluated,
        neighbors_used=min(neighbors, K),
    )


def propose_physics_knn_local_search(
    model: _PhysicalAssignmentModel,
    state: np.ndarray,
    incumbent_assignment: np.ndarray,
    *,
    stage_index: int,
    k_neighbors: int = 4,
    evaluation_budget: int = 32,
    max_rounds: int = 3,
    improvement_tolerance: float = 1.0e-9,
    target_priority: np.ndarray | None = None,
) -> PhysicsKNNLocalSearchProposal:
    """Run exact best-improvement swaps under a fixed evaluation budget.

    KNN determines only which responsibility exchanges may be evaluated.  It
    never supplies an additive objective. At each round all affordable unique
    two-exchanges are evaluated by the complete physical model, the best strict
    improvement is accepted, and subsequent exchanges are generated around the
    new assignment. The incumbent evaluation counts against the budget, making
    direct comparisons with a bounded blind neighborhood transparent.
    """

    incumbent = np.asarray(incumbent_assignment, dtype=np.int64).reshape(-1)
    K, Q = int(model.K), int(model.Q)
    stage = int(stage_index)
    neighbors = int(k_neighbors)
    budget = int(evaluation_budget)
    rounds = int(max_rounds)
    tolerance = float(improvement_tolerance)
    priority = None if target_priority is None else np.asarray(
        target_priority, dtype=np.float64).reshape(-1)
    if K != Q:
        raise ValueError("physics KNN local search currently requires K=Q")
    if (
        incumbent.shape != (K,) or np.any(incumbent < 0)
        or np.any(incumbent >= Q) or np.unique(incumbent).size != Q
    ):
        raise ValueError("incumbent must be a complete K=Q permutation")
    if stage != stage_index or stage < 0:
        raise ValueError("stage_index must be a non-negative integer")
    if neighbors != k_neighbors or neighbors < 1:
        raise ValueError("k_neighbors must be a positive integer")
    if budget != evaluation_budget or budget < 1:
        raise ValueError("evaluation_budget must be a positive integer")
    if rounds != max_rounds or rounds < 1:
        raise ValueError("max_rounds must be a positive integer")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("improvement_tolerance must be finite and non-negative")
    if priority is not None and (
        priority.shape != (Q,) or np.any(~np.isfinite(priority))
        or np.any(priority <= 0.0)
    ):
        raise ValueError("target_priority must be finite, positive, and have shape (Q,)")

    uav_pos, _uav_vel, _target_pos, _target_vel = model.unpack_state(state)
    advanced = model.transition(state, incumbent, stage)
    _next_uav, _next_velocity, advanced_target, _ = model.unpack_state(advanced)
    adjacency = _knn_adjacency(
        uav_pos[:, :2], advanced_target[:, :2], incumbent, neighbors)
    predicted_distance = np.linalg.norm(
        uav_pos[:, None, :2] - advanced_target[None, :, :2], axis=2)
    current = incumbent.copy()
    incumbent_evaluation = model.evaluate(advanced, stage_index=stage)
    incumbent_cost = float(incumbent_evaluation.cost)
    if priority is None:
        candidate_price = np.asarray(
            getattr(incumbent_evaluation, "target_price", np.ones(Q)),
            dtype=np.float64,
        ).reshape(-1)
        if (
            candidate_price.shape == (Q,)
            and np.all(np.isfinite(candidate_price))
            and np.any(candidate_price > 0.0)
        ):
            floor = max(float(np.max(candidate_price)) * 1.0e-6, 1.0e-12)
            priority = np.maximum(candidate_price, floor)
            priority /= float(np.mean(priority))
        else:
            priority = np.ones(Q, dtype=np.float64)
    current_cost = incumbent_cost
    evaluations = 1
    accepted: list[tuple[int, int]] = []

    for _round in range(rounds):
        if evaluations >= budget:
            break
        target_owner = np.empty(Q, dtype=np.int64)
        target_owner[current] = np.arange(K, dtype=np.int64)
        candidates: list[
            tuple[float, tuple[int, ...], int, int, np.ndarray]
        ] = []
        seen: set[tuple[int, ...]] = set()
        for uav, target in np.argwhere(adjacency):
            partner = int(target_owner[int(target)])
            if int(uav) == partner:
                continue
            action = current.copy()
            action[int(uav)], action[partner] = action[partner], action[int(uav)]
            key = tuple(int(value) for value in action)
            if key in seen:
                continue
            seen.add(key)
            current_distance = (
                priority[current[int(uav)]]
                * predicted_distance[int(uav), current[int(uav)]]
                + priority[current[partner]]
                * predicted_distance[partner, current[partner]]
            )
            exchanged_distance = (
                priority[action[int(uav)]]
                * predicted_distance[int(uav), action[int(uav)]]
                + priority[action[partner]]
                * predicted_distance[partner, action[partner]]
            )
            candidates.append((
                float(exchanged_distance - current_distance),
                key, int(uav), partner, action,
            ))
        # Most promising causal geometric exchanges consume the exact-physics
        # budget first. Stable assignment tuples make ties reproducible.
        candidates.sort(key=lambda item: (item[0], item[1]))

        best_cost = current_cost
        best_action = None
        best_swap = None
        for _surrogate, _key, left, right, action in candidates:
            if evaluations >= budget:
                break
            next_state = model.transition(state, action, stage)
            cost = float(model.evaluate(next_state, stage_index=stage).cost)
            evaluations += 1
            if cost < best_cost - tolerance:
                best_cost = cost
                best_action = action
                best_swap = (min(left, right), max(left, right))
        if best_action is None or best_swap is None:
            break
        current = best_action.copy()
        current_cost = best_cost
        accepted.append(best_swap)

    return PhysicsKNNLocalSearchProposal(
        assignment=current,
        adjacency=adjacency,
        incumbent_cost=incumbent_cost,
        proposal_cost=current_cost,
        accepted_swaps=tuple(accepted),
        evaluated_assignments=evaluations,
        evaluation_budget=budget,
        neighbors_used=min(neighbors, K),
    )
