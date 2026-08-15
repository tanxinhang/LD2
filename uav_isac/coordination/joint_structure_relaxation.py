"""Convex upper certificate and sparse candidates for joint ISAC structure.

The LP relaxes binary roles, receiver ownership and admitted hyperedges to
their convex boxes while preserving all linear capacities and perspective RF
flow bounds. Every integer-feasible structure is feasible in this relaxation,
so its max-min value is a valid same-geometry upper certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from uav_isac.coordination.bottleneck_router import (
    BottleneckDecision,
    route_isac_repair,
)
from uav_isac.coordination.maxmin_power import (
    MaxMinPowerResult,
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.physical.detection import compute_detection_probabilities


@dataclass(frozen=True)
class JointStructureRelaxationResult:
    upper_deflection: float
    target_deflection: np.ndarray
    target_prices: np.ndarray
    edge_activity: np.ndarray
    edge_power_w: np.ndarray
    tx_fraction: np.ndarray
    rx_fraction: np.ndarray
    owner_fraction: np.ndarray
    sensing_power_w: np.ndarray
    solve_time_s: float
    solver_status: int
    solver_message: str


@dataclass(frozen=True)
class SparseStructureCandidateResult:
    candidate_mask: np.ndarray
    selected_set: tuple[tuple[int, int, int], ...]
    selected_owner_groups: tuple[tuple[int, int], ...]
    full_edge_count: int
    candidate_edge_count: int
    retained_ceiling_ratio: np.ndarray
    target_prices: np.ndarray


@dataclass(frozen=True)
class HierarchicalQoSRouteResult:
    decision: BottleneckDecision
    fixed_power: MaxMinPowerResult | None
    joint_relaxation: JointStructureRelaxationResult | None


def _validated(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        gain.ndim != 3 or gain.shape[0] != gain.shape[1]
        or gain.shape[0] != budget.size or gain.shape[2] < 1
        or np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError(
            "gain must be finite non-negative (K,K,Q) and budget a K-vector")
    result = gain.copy()
    diagonal = np.arange(result.shape[0])
    result[diagonal, diagonal, :] = 0.0
    return result, budget


def owner_pair_relaxed_target_ceiling(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    target_pair_limit: int,
) -> np.ndarray:
    """Upper-bound each target with one owner and a finite pair count.

    Cross-target role, receiver-capacity and RF sharing constraints are
    relaxed. Unique receiver ownership and the per-target pair limit remain.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, _, Q = gain.shape
    pair_limit = int(target_pair_limit)
    if pair_limit < 1:
        raise ValueError("target_pair_limit must be positive")
    count = min(pair_limit, K - 1)
    ceiling = np.zeros(Q, dtype=np.float64)
    for target in range(Q):
        for receiver in range(K):
            contributions = gain[:, receiver, target] * budget
            ceiling[target] = max(
                ceiling[target],
                float(np.sum(np.sort(contributions)[-count:])),
            )
    return ceiling


def solve_joint_structure_maxmin_lp_relaxation(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    candidate_mask: np.ndarray | None = None,
) -> JointStructureRelaxationResult:
    """Solve the role/owner/capacity-aware max-min LP relaxation."""
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, _, Q = gain.shape
    pair_limit = int(target_pair_limit)
    receiver_limit = int(reports_per_receiver)
    if pair_limit < 1 or receiver_limit < 1:
        raise ValueError("pair and receiver limits must be positive")
    if candidate_mask is None:
        support = gain > 0.0
    else:
        support = np.asarray(candidate_mask, dtype=bool)
        if support.shape != gain.shape:
            raise ValueError("candidate_mask must have shape (K,K,Q)")
        support = support & (gain > 0.0)
    diagonal = np.arange(K)
    support[diagonal, diagonal, :] = False
    edges = [tuple(int(v) for v in edge) for edge in np.argwhere(support)]
    if any(not any(edge[2] == target for edge in edges) for target in range(Q)):
        raise ValueError("every target requires at least one positive candidate edge")
    E = len(edges)

    x_offset = 0
    tx_offset = x_offset + E
    rx_offset = tx_offset + K
    owner_offset = rx_offset + K
    flow_offset = owner_offset + K * Q
    t_index = flow_offset + E
    variable_count = t_index + 1

    def owner_index(receiver: int, target: int) -> int:
        return owner_offset + receiver * Q + target

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[t_index] = -1.0
    lower = np.zeros(variable_count, dtype=np.float64)
    upper = np.ones(variable_count, dtype=np.float64)
    for edge_index, (transmitter, _, _) in enumerate(edges):
        upper[flow_offset + edge_index] = budget[transmitter]
    upper[t_index] = np.inf

    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    row_upper: list[float] = []

    def add(terms: Sequence[tuple[int, float]], bound: float) -> int:
        row = len(row_upper)
        for column, coefficient in terms:
            row_indices.append(row)
            column_indices.append(int(column))
            coefficients.append(float(coefficient))
        row_upper.append(float(bound))
        return row

    for node in range(K):
        add(((tx_offset + node, 1.0), (rx_offset + node, 1.0)), 1.0)
    for target in range(Q):
        add(tuple(
            (owner_index(receiver, target), 1.0)
            for receiver in range(K)
        ), 1.0)
    for transmitter in range(K):
        add(tuple(
            (flow_offset + edge_index, 1.0)
            for edge_index, edge in enumerate(edges)
            if edge[0] == transmitter
        ) + ((tx_offset + transmitter, -budget[transmitter]),), 0.0)
    for target in range(Q):
        add(tuple(
            (x_offset + edge_index, 1.0)
            for edge_index, edge in enumerate(edges)
            if edge[2] == target
        ), float(pair_limit))
    for receiver in range(K):
        receiver_edges = tuple(
            (x_offset + edge_index, 1.0)
            for edge_index, edge in enumerate(edges)
            if edge[1] == receiver
        )
        add(receiver_edges, float(receiver_limit))
        add(receiver_edges + (
            (rx_offset + receiver, -float(receiver_limit)),
        ), 0.0)
    for transmitter in range(K):
        transmitter_edges = tuple(
            (x_offset + edge_index, 1.0)
            for edge_index, edge in enumerate(edges)
            if edge[0] == transmitter
        )
        add(transmitter_edges + (
            (tx_offset + transmitter, -float(Q * pair_limit)),
        ), 0.0)
    for receiver in range(K):
        for target in range(Q):
            owner_edges = tuple(
                (x_offset + edge_index, 1.0)
                for edge_index, edge in enumerate(edges)
                if edge[1] == receiver and edge[2] == target
            )
            add(owner_edges + ((
                owner_index(receiver, target), -float(pair_limit)),
            ), 0.0)
            owner_flows = tuple(
                (flow_offset + edge_index, 1.0)
                for edge_index, edge in enumerate(edges)
                if edge[1] == receiver and edge[2] == target
            )
            add(owner_flows + ((
                owner_index(receiver, target), -float(np.sum(budget))),
            ), 0.0)
    for transmitter in range(K):
        for target in range(Q):
            target_flows = tuple(
                (flow_offset + edge_index, 1.0)
                for edge_index, edge in enumerate(edges)
                if edge[0] == transmitter and edge[2] == target
            )
            owner_mass = tuple(
                (owner_index(receiver, target), -float(budget[transmitter]))
                for receiver in range(K)
            )
            add(target_flows + owner_mass, 0.0)
    for edge_index, (transmitter, receiver, target) in enumerate(edges):
        edge_variable = x_offset + edge_index
        flow_variable = flow_offset + edge_index
        edge_budget = budget[transmitter]
        add(((edge_variable, 1.0), (tx_offset + transmitter, -1.0)), 0.0)
        add(((edge_variable, 1.0), (rx_offset + receiver, -1.0)), 0.0)
        add(((edge_variable, 1.0),
             (owner_index(receiver, target), -1.0)), 0.0)
        add(((flow_variable, 1.0),
             (edge_variable, -edge_budget)), 0.0)
    target_rows = []
    for target in range(Q):
        target_rows.append(add(tuple(
            (flow_offset + edge_index, -gain[edge])
            for edge_index, edge in enumerate(edges)
            if edge[2] == target
        ) + ((t_index, 1.0),), 0.0))

    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(row_upper), variable_count),
        dtype=np.float64,
    ).tocsr()
    started = perf_counter()
    solved = linprog(
        objective,
        A_ub=matrix,
        b_ub=np.asarray(row_upper, dtype=np.float64),
        bounds=list(zip(lower, upper)),
        method="highs",
    )
    elapsed = perf_counter() - started
    if not solved.success or solved.x is None:
        raise RuntimeError(f"joint structure LP relaxation failed: {solved.message}")
    values = np.asarray(solved.x, dtype=np.float64)
    edge_activity = np.zeros((K, K, Q), dtype=np.float64)
    edge_power = np.zeros((K, K, Q), dtype=np.float64)
    for edge_index, edge in enumerate(edges):
        edge_activity[edge] = max(values[x_offset + edge_index], 0.0)
        edge_power[edge] = max(values[flow_offset + edge_index], 0.0)
    target_deflection = np.sum(gain * edge_power, axis=(0, 1))
    prices = np.maximum(
        -np.asarray(solved.ineqlin.marginals)[target_rows], 0.0)
    price_mass = float(np.sum(prices))
    prices = (
        prices / price_mass
        if price_mass > 0.0
        else np.full(Q, 1.0 / Q, dtype=np.float64)
    )
    optimum = max(float(-solved.fun), 0.0)
    numerical_margin = 1.0e-8 * max(1.0, abs(optimum))
    return JointStructureRelaxationResult(
        upper_deflection=float(optimum + numerical_margin),
        target_deflection=target_deflection,
        target_prices=prices,
        edge_activity=edge_activity,
        edge_power_w=edge_power,
        tx_fraction=np.maximum(values[tx_offset:tx_offset + K], 0.0),
        rx_fraction=np.maximum(values[rx_offset:rx_offset + K], 0.0),
        owner_fraction=np.maximum(
            values[owner_offset:owner_offset + K * Q].reshape(K, Q), 0.0),
        sensing_power_w=np.sum(edge_power, axis=1),
        solve_time_s=float(elapsed),
        solver_status=int(solved.status),
        solver_message=str(solved.message),
    )


def dual_guided_sparse_candidate_mask(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    relaxation: JointStructureRelaxationResult,
    *,
    target_pair_limit: int,
    minimum_owner_groups_per_target: int = 1,
    additional_owner_groups: int,
    transmitters_per_owner: int,
    incumbent_selected: Sequence[tuple[int, int, int]] = (),
) -> SparseStructureCandidateResult:
    """Build a sparse graph that preserves every independent target ceiling.

    Each target keeps its strongest owner and that owner's top pair-limited
    transmitters. Additional owner groups are allocated globally by the LP
    target price times owner ceiling, concentrating alternatives on the
    current max-min bottlenecks without fixed target identities.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, _, Q = gain.shape
    pair_limit = min(int(target_pair_limit), K - 1)
    minimum_groups = min(int(minimum_owner_groups_per_target), K)
    transmitter_limit = min(int(transmitters_per_owner), K - 1)
    extra_limit = int(additional_owner_groups)
    if (
        pair_limit < 1 or minimum_groups < 1
        or transmitter_limit < pair_limit or extra_limit < 0
    ):
        raise ValueError(
            "transmitter count must cover the positive pair limit and extras")
    if (
        relaxation.edge_activity.shape != gain.shape
        or relaxation.edge_power_w.shape != gain.shape
        or relaxation.target_prices.shape != (Q,)
    ):
        raise ValueError("relaxation dimensions do not match gain")

    owner_ceiling = np.zeros((K, Q), dtype=np.float64)
    for receiver in range(K):
        for target in range(Q):
            contributions = gain[:, receiver, target] * budget
            owner_ceiling[receiver, target] = float(
                np.sum(np.sort(contributions)[-pair_limit:]))
    groups: set[tuple[int, int]] = set()
    for target in range(Q):
        receivers = sorted(range(K), key=lambda receiver: (
            -float(owner_ceiling[receiver, target]),
            -float(relaxation.owner_fraction[receiver, target]),
            receiver,
        ))
        groups.update(
            (receiver, target) for receiver in receivers[:minimum_groups]
            if owner_ceiling[receiver, target] > 0.0
        )
    alternatives = [
        (receiver, target)
        for target in range(Q)
        for receiver in range(K)
        if owner_ceiling[receiver, target] > 0.0
        and (receiver, target) not in groups
    ]
    alternatives.sort(key=lambda group: (
        -float(relaxation.target_prices[group[1]] * owner_ceiling[group]),
        -float(relaxation.owner_fraction[group]),
        group[1],
        group[0],
    ))
    groups.update(alternatives[:extra_limit])

    candidate = np.zeros_like(gain, dtype=bool)
    for receiver, target in groups:
        transmitters = [
            transmitter for transmitter in range(K)
            if transmitter != receiver and gain[transmitter, receiver, target] > 0.0
        ]
        transmitters.sort(key=lambda transmitter: (
            -float(gain[transmitter, receiver, target] * budget[transmitter]),
            -float(relaxation.edge_power_w[transmitter, receiver, target]),
            -float(relaxation.edge_activity[transmitter, receiver, target]),
            transmitter,
        ))
        for transmitter in transmitters[:transmitter_limit]:
            candidate[transmitter, receiver, target] = True
    for transmitter, receiver, target in incumbent_selected:
        edge = (int(transmitter), int(receiver), int(target))
        if (
            0 <= edge[0] < K and 0 <= edge[1] < K and 0 <= edge[2] < Q
            and edge[0] != edge[1] and gain[edge] > 0.0
        ):
            candidate[edge] = True
            groups.add((edge[1], edge[2]))

    full_ceiling = owner_pair_relaxed_target_ceiling(
        gain, budget, target_pair_limit=pair_limit)
    retained_ceiling = owner_pair_relaxed_target_ceiling(
        np.where(candidate, gain, 0.0),
        budget,
        target_pair_limit=pair_limit,
    )
    ratio = np.divide(
        retained_ceiling,
        full_ceiling,
        out=np.ones_like(full_ceiling),
        where=full_ceiling > 0.0,
    )
    selected = tuple(
        tuple(int(value) for value in edge) for edge in np.argwhere(candidate)
    )
    return SparseStructureCandidateResult(
        candidate_mask=candidate,
        selected_set=selected,
        selected_owner_groups=tuple(sorted(groups, key=lambda x: (x[1], x[0]))),
        full_edge_count=int(np.count_nonzero(gain)),
        candidate_edge_count=int(np.count_nonzero(candidate)),
        retained_ceiling_ratio=ratio,
        target_prices=relaxation.target_prices.copy(),
    )


def certified_hierarchical_qos_route(
    upper_coefficient: np.ndarray,
    selected: Sequence[tuple[int, int, int]],
    sensing_budget_w: np.ndarray,
    *,
    current_lower_pd: float,
    p_fa: float,
    qos_floor: float,
    target_pair_limit: int,
    reports_per_receiver: int,
) -> HierarchicalQoSRouteResult:
    """Route to the cheapest layer using only genuine upper certificates."""
    if float(current_lower_pd) >= float(qos_floor):
        decision = route_isac_repair(
            current_lower_pd,
            fixed_power_upper=None,
            joint_structure_upper=None,
            qos_floor=qos_floor,
        )
        return HierarchicalQoSRouteResult(decision, None, None)
    gain, _ = fixed_owner_gain_matrix(upper_coefficient, selected)
    fixed = solve_fixed_structure_maxmin_power_lp(gain, sensing_budget_w)
    fixed_upper_pd = float(np.min(compute_detection_probabilities(
        fixed.deflection, float(p_fa))))
    if fixed_upper_pd >= float(qos_floor):
        decision = route_isac_repair(
            current_lower_pd,
            fixed_power_upper=fixed_upper_pd,
            joint_structure_upper=None,
            qos_floor=qos_floor,
        )
        return HierarchicalQoSRouteResult(decision, fixed, None)
    relaxation = solve_joint_structure_maxmin_lp_relaxation(
        upper_coefficient,
        sensing_budget_w,
        target_pair_limit=target_pair_limit,
        reports_per_receiver=reports_per_receiver,
    )
    joint_upper_pd = float(compute_detection_probabilities(
        np.asarray([relaxation.upper_deflection]), float(p_fa))[0])
    decision = route_isac_repair(
        current_lower_pd,
        fixed_power_upper=fixed_upper_pd,
        joint_structure_upper=joint_upper_pd,
        qos_floor=qos_floor,
    )
    return HierarchicalQoSRouteResult(decision, fixed, relaxation)
