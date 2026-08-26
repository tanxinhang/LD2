"""Exact shadow oracle for minimum-intervention three-floor L2 repair.

This is a centralized architecture-design oracle, not a live controller.  It
jointly chooses roles, one receiver owner per target, admitted bistatic edges,
and edge power.  Detection constraints use a conservative chord PWL lower
bound.  Three sequential MILPs implement the lexicographic objective:

    dependency-closure cardinality -> prepare payload bits -> sensing power.

The closure contains every affected target and every UAV whose role, incident
edge, or old/new ownership changes.  Thus a nominally small proposal cannot
hide a large structural rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from uav_isac.coordination.dependency_commit import DependencyCommitLayout
from uav_isac.coordination.pwl_pd import saturating_chord_lower_bound
from uav_isac.coordination.scale_capability import task_detection_metrics
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)


@dataclass(frozen=True)
class MinimumInterventionRepairResult:
    feasible: bool
    proven_infeasible: bool
    selected_set: tuple[tuple[int, int, int], ...]
    tx_role: np.ndarray
    rx_role: np.ndarray
    receiver_owner: np.ndarray
    sensing_power_w: np.ndarray
    target_deflection: np.ndarray
    target_pd: np.ndarray
    conservative_pd: np.ndarray
    affected_targets: tuple[int, ...]
    participants: tuple[int, ...]
    changed_roles: int
    changed_owners: int
    toggled_edges: int
    closure_cardinality: int
    prepare_bits: int
    total_sensing_power_w: float
    solve_time_s: float
    solver_status: int
    solver_message: str


def solve_minimum_intervention_task_repair_milp(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    current_selected: np.ndarray,
    current_tx_role: np.ndarray,
    current_rx_role: np.ndarray,
    current_owner: np.ndarray,
    *,
    p_fa: float,
    task_floors: tuple[float, float, float, int] | list[float],
    target_pair_limit: int,
    reports_per_receiver: int,
    pwl_epsilon: float = 1.0e-3,
    time_limit_s: float = 30.0,
    mip_relative_gap: float = 0.0,
    changeable_uavs: tuple[int, ...] | list[int] | None = None,
    changeable_targets: tuple[int, ...] | list[int] | None = None,
    changeable_role_uavs: tuple[int, ...] | list[int] | None = None,
    forbidden_edges: tuple[tuple[int, int, int], ...] | list[tuple[int, int, int]] | None = None,
    feasibility_only: bool = False,
) -> MinimumInterventionRepairResult:
    """Return an exact lexicographic minimum-intervention feasible witness."""
    gain = np.asarray(gain_per_watt, dtype=np.float64).copy()
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    selected0 = np.asarray(current_selected, dtype=bool)
    tx0 = np.asarray(current_tx_role, dtype=bool).reshape(-1)
    rx0 = np.asarray(current_rx_role, dtype=bool).reshape(-1)
    owner0 = np.asarray(current_owner, dtype=np.int64).reshape(-1)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("gain_per_watt must have shape (K,K,Q)")
    K, _, Q = gain.shape
    if budget.shape != (K,) or selected0.shape != (K, K, Q):
        raise ValueError("budget/current_selected shapes do not match gain")
    if tx0.shape != (K,) or rx0.shape != (K,) or owner0.shape != (Q,):
        raise ValueError("current role/owner shapes do not match gain")
    if (
        np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
        or np.any(tx0 & rx0) or np.any((owner0 < 0) | (owner0 >= K))
    ):
        raise ValueError("invalid gain, budget, current role, or owner")
    diagonal = np.arange(K)
    gain[diagonal, diagonal, :] = 0.0
    if np.any(selected0[diagonal, diagonal, :]):
        raise ValueError("monostatic diagonal cannot be selected")
    if len(task_floors) != 4:
        raise ValueError("task_floors must contain three floors and valid k")
    rho_min, rho_tail, rho_avg = (float(v) for v in task_floors[:3])
    tail_k = int(task_floors[3])
    if not 1 <= tail_k <= Q:
        raise ValueError("task_floors must contain three floors and valid k")
    # Audit 2026-08-25: the repair domain stays (0,1) because the floors are
    # inverted to a finite deflection here (unit probability requires unbounded
    # deflection in the Gaussian detector).  scale_capability.task_detection_metrics
    # accepts (0,1] only as a feasibility-check floor, never as an inversion
    # target; the message below explains the split instead of silently looking
    # like a contract bug.
    if not all(0.0 < value < 1.0 for value in (rho_min, rho_tail, rho_avg)):
        raise ValueError(
            "task probability floors must lie in (0,1): perfect detection "
            "(==1.0) cannot be inverted to a finite deflection in this "
            "Gaussian model (scale_capability accepts (0,1] only as a "
            "feasibility-check floor, not as an inversion target)")
    if target_pair_limit < 1 or reports_per_receiver < 1:
        raise ValueError("pair and receiver limits must be positive")
    allowed_uavs = (
        set(range(K)) if changeable_uavs is None
        else {int(value) for value in changeable_uavs}
    )
    allowed_targets = (
        set(range(Q)) if changeable_targets is None
        else {int(value) for value in changeable_targets}
    )
    allowed_roles = (
        set(allowed_uavs) if changeable_role_uavs is None
        else {int(value) for value in changeable_role_uavs}
    )
    if (
        any(value < 0 or value >= K for value in allowed_uavs)
        or any(value < 0 or value >= Q for value in allowed_targets)
        or any(value < 0 or value >= K for value in allowed_roles)
        or not allowed_roles <= allowed_uavs
    ):
        raise ValueError("changeable block contains an out-of-range index")
    forbidden = set() if forbidden_edges is None else {
        tuple(int(component) for component in edge) for edge in forbidden_edges
    }
    if any(
        len(edge) != 3 or edge[0] < 0 or edge[0] >= K
        or edge[1] < 0 or edge[1] >= K or edge[0] == edge[1]
        or edge[2] < 0 or edge[2] >= Q
        for edge in forbidden
    ):
        raise ValueError("forbidden_edges contains an invalid bistatic edge")

    edges = [(i, j, q) for i in range(K) for j in range(K) if i != j
             for q in range(Q)]
    E = len(edges)
    # Variables: x, tx, rx, owner, flow, D, y, tau, z,
    #            edge-change, role-change, owner-change, affected, participant.
    o_x = 0
    o_tx = o_x + E
    o_rx = o_tx + K
    o_owner = o_rx + K
    o_flow = o_owner + K * Q
    o_D = o_flow + E
    o_y = o_D + Q
    o_tau = o_y + Q
    o_z = o_tau + 1
    o_de = o_z + Q
    o_dr = o_de + E
    o_do = o_dr + K
    o_aff = o_do + Q
    o_part = o_aff + Q
    n = o_part + K

    def owner_index(j: int, q: int) -> int:
        return o_owner + j * Q + q

    lower = np.zeros(n, dtype=np.float64)
    upper = np.ones(n, dtype=np.float64)
    ceiling = np.sum(
        budget[:, None] * np.max(gain, axis=1), axis=0)
    d_floor = float(minimum_deflection_for_detection_probability(
        np.asarray([rho_min]), p_fa)[0])
    d_max = max(float(np.max(ceiling)), d_floor + 1.0e-9)
    upper[o_D:o_D + Q] = d_max
    for e, (i, _j, _q) in enumerate(edges):
        upper[o_flow + e] = budget[i]
    integrality = np.zeros(n, dtype=np.int32)
    integrality[o_x:o_flow] = 1
    integrality[o_de:n] = 1

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add(terms, lb: float = -np.inf, ub: float = np.inf) -> None:
        row = np.zeros(n, dtype=np.float64)
        for index, coefficient in terms:
            row[index] += float(coefficient)
        rows.append(row)
        lbs.append(float(lb))
        ubs.append(float(ub))

    # Roles, owners, capacities, and perspective-linked edge flow.
    for i in range(K):
        add(((o_tx + i, 1), (o_rx + i, 1)), ub=1)
    for q in range(Q):
        add(tuple((owner_index(j, q), 1) for j in range(K)), lb=1, ub=1)
    for i in range(K):
        add(tuple((o_flow + e, 1) for e, edge in enumerate(edges)
                  if edge[0] == i), ub=budget[i])
    for q in range(Q):
        add(tuple((o_x + e, 1) for e, edge in enumerate(edges)
                  if edge[2] == q), ub=target_pair_limit)
    for j in range(K):
        add(tuple((o_x + e, 1) for e, edge in enumerate(edges)
                  if edge[1] == j), ub=reports_per_receiver)
    for e, (i, j, q) in enumerate(edges):
        add(((o_x + e, 1), (o_tx + i, -1)), ub=0)
        add(((o_x + e, 1), (o_rx + j, -1)), ub=0)
        add(((o_x + e, 1), (owner_index(j, q), -1)), ub=0)
        add(((o_flow + e, 1), (o_x + e, -budget[i])), ub=0)

    # Deflection and conservative full-task PWL constraints.
    for q in range(Q):
        terms = [(o_D + q, 1.0)]
        terms.extend((o_flow + e, -gain[edge]) for e, edge in enumerate(edges)
                     if edge[2] == q)
        add(tuple(terms), lb=0, ub=0)
        add(((o_D + q, 1),), lb=d_floor)
    slopes, intercepts, _breakpoints = saturating_chord_lower_bound(
        p_fa, d_floor, d_max, float(pwl_epsilon))
    for slope, intercept in zip(slopes, intercepts):
        for q in range(Q):
            add(((o_y + q, 1), (o_D + q, -slope)), ub=intercept)
    add(tuple((o_y + q, 1) for q in range(Q)), lb=Q * rho_avg)
    for q in range(Q):
        add(((o_z + q, -1), (o_tau, 1), (o_y + q, -1)), ub=0)
    add(tuple([(o_tau, -tail_k)] + [(o_z + q, 1) for q in range(Q)]),
        ub=-tail_k * rho_tail)

    # Exact Hamming changes and their dependency closure.
    for e, edge in enumerate(edges):
        current = int(selected0[edge])
        add(((o_de + e, 1), (o_x + e, 1 if current else -1)),
            lb=current, ub=current)
        i, j, q = edge
        add(((o_aff + q, 1), (o_de + e, -1)), lb=0)
        add(((o_part + i, 1), (o_de + e, -1)), lb=0)
        add(((o_part + j, 1), (o_de + e, -1)), lb=0)
    for i in range(K):
        if tx0[i]:
            add(((o_dr + i, 1), (o_tx + i, 1)), lb=1, ub=1)
        elif rx0[i]:
            add(((o_dr + i, 1), (o_rx + i, 1)), lb=1, ub=1)
        else:
            add(((o_dr + i, 1), (o_tx + i, -1), (o_rx + i, -1)),
                lb=0, ub=0)
        add(((o_part + i, 1), (o_dr + i, -1)), lb=0)
    for q in range(Q):
        old = int(owner0[q])
        add(((o_do + q, 1), (owner_index(old, q), 1)), lb=1, ub=1)
        add(((o_aff + q, 1), (o_do + q, -1)), lb=0)
        add(((o_part + old, 1), (o_do + q, -1)), lb=0)
        for j in range(K):
            if j != old:
                add(((o_part + j, 1), (owner_index(j, q), -1)), lb=0)

    # A block is structurally atomic: roles outside its UAV set, owners outside
    # its target set, and every edge not fully contained in UxUxQ are frozen.
    for i in range(K):
        if i not in allowed_roles:
            add(((o_tx + i, 1),), lb=int(tx0[i]), ub=int(tx0[i]))
            add(((o_rx + i, 1),), lb=int(rx0[i]), ub=int(rx0[i]))
    for q in range(Q):
        if q not in allowed_targets:
            for j in range(K):
                current = int(owner0[q] == j)
                add(((owner_index(j, q), 1),), lb=current, ub=current)
        else:
            for j in range(K):
                if j not in allowed_uavs and j != int(owner0[q]):
                    add(((owner_index(j, q), 1),), lb=0, ub=0)
    for e, edge in enumerate(edges):
        i, j, q = edge
        if i not in allowed_uavs or j not in allowed_uavs or q not in allowed_targets:
            current = int(selected0[edge])
            add(((o_x + e, 1),), lb=current, ub=current)
        if edge in forbidden:
            add(((o_x + e, 1),), lb=0, ub=0)

    base_constraint = LinearConstraint(
        np.stack(rows), np.asarray(lbs), np.asarray(ubs))
    bounds = Bounds(lower, upper)
    options = {"time_limit": float(time_limit_s),
               "mip_rel_gap": float(mip_relative_gap)}
    started = perf_counter()

    closure_objective = np.zeros(n)
    closure_objective[o_aff:o_aff + Q] = 1
    closure_objective[o_part:o_part + K] = 1
    first = milp(closure_objective, integrality=integrality, bounds=bounds,
                 constraints=base_constraint, options=options)
    if first.x is None:
        return MinimumInterventionRepairResult(
            False, int(first.status) == 2, tuple(), np.zeros(K, bool),
            np.zeros(K, bool), np.full(Q, -1, dtype=np.int64),
            np.zeros((K, Q)), np.zeros(Q), np.full(Q, p_fa), np.zeros(Q),
            tuple(), tuple(), 0, 0, 0, 0, 0, 0.0,
            perf_counter() - started, int(first.status), str(first.message))
    if feasibility_only:
        value = np.asarray(first.x, dtype=np.float64)
        activity = np.asarray(base_constraint.A @ value, dtype=np.float64)
        primal_tol = 1.0e-7
        valid_incumbent = bool(
            np.all(np.isfinite(value))
            and np.all(value >= np.asarray(bounds.lb) - primal_tol)
            and np.all(value <= np.asarray(bounds.ub) + primal_tol)
            and np.all(activity >= np.asarray(base_constraint.lb) - primal_tol)
            and np.all(activity <= np.asarray(base_constraint.ub) + primal_tol)
            and np.all(np.abs(
                value[np.asarray(integrality, dtype=bool)]
                - np.rint(value[np.asarray(integrality, dtype=bool)]))
                <= primal_tol)
        )
        return MinimumInterventionRepairResult(
            valid_incumbent, False, tuple(), np.zeros(K, bool), np.zeros(K, bool),
            np.full(Q, -1, dtype=np.int64), np.zeros((K, Q)), np.zeros(Q),
            np.full(Q, p_fa), np.zeros(Q), tuple(), tuple(), 0, 0, 0,
            int(round(float(first.fun))), 0, 0.0,
            perf_counter() - started, int(first.status),
            ("feasibility-only primal-validated: " if valid_incumbent else
             "feasibility-only unresolved invalid incumbent: ")
            + str(first.message))
    closure_opt = int(round(float(first.fun)))
    closure_row = np.zeros(n)
    closure_row[o_aff:o_aff + Q] = 1
    closure_row[o_part:o_part + K] = 1
    closure_pin = LinearConstraint(closure_row[None, :], -np.inf, closure_opt)

    layout = DependencyCommitLayout(K, Q)
    bit_objective = np.zeros(n)
    bit_objective[o_dr:o_dr + K] = layout.node_bits + layout.role_bits
    bit_objective[o_do:o_do + Q] = layout.target_bits + layout.node_bits
    bit_objective[o_de:o_de + E] = 2 * layout.node_bits + layout.target_bits + 1
    second = milp(bit_objective, integrality=integrality, bounds=bounds,
                  constraints=(base_constraint, closure_pin), options=options)
    if second.x is None:
        second = first
    bit_opt = int(round(float(bit_objective @ second.x)))
    bit_pin = LinearConstraint(bit_objective[None, :], -np.inf, bit_opt)

    power_objective = np.zeros(n)
    power_objective[o_flow:o_flow + E] = 1
    third = milp(power_objective, integrality=integrality, bounds=bounds,
                 constraints=(base_constraint, closure_pin, bit_pin), options=options)
    solved = third if third.x is not None else second
    value = np.asarray(solved.x)
    selected = tuple(edge for e, edge in enumerate(edges)
                     if value[o_x + e] >= 0.5)
    tx = value[o_tx:o_tx + K] >= 0.5
    rx = value[o_rx:o_rx + K] >= 0.5
    owner = np.asarray([
        int(np.argmax(value[o_owner + q:o_owner + K * Q:Q]))
        for q in range(Q)
    ], dtype=np.int64)
    edge_flow = np.maximum(value[o_flow:o_flow + E], 0.0)
    power = np.zeros((K, Q))
    for e, (i, _j, q) in enumerate(edges):
        power[i, q] += edge_flow[e]
    deflection = np.maximum(value[o_D:o_D + Q], 0.0)
    pd = compute_detection_probabilities(deflection, p_fa)
    conservative = np.clip(value[o_y:o_y + Q], 0.0, 1.0)
    changed_roles = int(np.sum(value[o_dr:o_dr + K] >= 0.5))
    changed_owners = int(np.sum(value[o_do:o_do + Q] >= 0.5))
    toggled_edges = int(np.sum(value[o_de:o_de + E] >= 0.5))
    affected = tuple(np.flatnonzero(value[o_aff:o_aff + Q] >= 0.5).tolist())
    participants = tuple(np.flatnonzero(value[o_part:o_part + K] >= 0.5).tolist())
    prepare_bits = layout.prepare_bits(
        changed_roles=changed_roles, changed_owners=changed_owners,
        toggled_edges=toggled_edges)
    metrics = task_detection_metrics(conservative, task_floors)
    valid = bool(
        metrics["feasibility_ratio"] <= 1.0 + 1.0e-7
        and np.all(np.sum(power, axis=1) <= budget + 1.0e-8)
        and not np.any(tx & rx)
    )
    return MinimumInterventionRepairResult(
        valid, False, selected, tx, rx, owner, power, deflection, pd,
        conservative, affected, participants, changed_roles, changed_owners,
        toggled_edges, len(affected) + len(participants), prepare_bits,
        float(np.sum(power)), perf_counter() - started, int(solved.status),
        str(solved.message),
    )
