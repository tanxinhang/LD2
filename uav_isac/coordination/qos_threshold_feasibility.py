"""Exact QoS-threshold feasibility for single-role local-fusion ISAC.

The max--min sensing problem is quasiconvex in a requested detection floor:
the Gaussian detector is monotone in Deflection, so a probability threshold
can first be mapped to a Deflection threshold.  For a fixed threshold, binary
Tx/Rx roles, receiver ownership and admitted bistatic edges couple to
continuous edge power. Unique receiver ownership means that an integer
solution has at most one receiver edge for each transmitter--target power
stream. A perspective bound ties each edge flow to admission exactly, yielding
a smaller mixed-integer linear feasibility problem without duplicating RF
power across fractional owners.

This module is a full-information reference and architecture-design oracle.
It is not a distributed deployment controller: learned or finite-round local
methods must approximate its decisions while preserving the same hard
communication, ownership, role and RF constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from uav_isac.physical.detection import (
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)


@dataclass(frozen=True)
class QoSThresholdFeasibilityResult:
    feasible: bool
    proven_infeasible: bool
    requested_pd: float
    requested_deflection: float
    selected_set: tuple[tuple[int, int, int], ...]
    tx_role: np.ndarray
    rx_role: np.ndarray
    receiver_owner: np.ndarray
    sensing_power_w: np.ndarray
    target_deflection: np.ndarray
    target_pd: np.ndarray
    solve_time_s: float
    solver_status: int
    solver_message: str


@dataclass(frozen=True)
class MaxMinQoSBoundaryResult:
    lower_pd: float
    upper_pd: float
    lower_deflection: float
    upper_deflection: float
    iterations: int
    exact_infeasible_upper: bool
    best_feasible: QoSThresholdFeasibilityResult
    solve_time_s: float


def _validate_inputs(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        gain.ndim != 3 or gain.shape[0] != gain.shape[1]
        or gain.shape[0] != budget.size or gain.shape[2] < 1
    ):
        raise ValueError(
            "gain_per_watt must have shape (K,K,Q) matching the K-vector budget")
    if np.any(~np.isfinite(gain)) or np.any(gain < 0.0):
        raise ValueError("gain_per_watt must be finite and non-negative")
    if np.any(~np.isfinite(budget)) or np.any(budget < 0.0):
        raise ValueError("sensing_budget_w must be finite and non-negative")
    gain = gain.copy()
    diagonal = np.arange(gain.shape[0])
    gain[diagonal, diagonal, :] = 0.0
    return gain, budget


def _empty_result(
    *,
    K: int,
    Q: int,
    requested_pd: float,
    requested_deflection: float,
    p_fa: float,
    solve_time_s: float,
    solver_status: int,
    solver_message: str,
    proven_infeasible: bool,
) -> QoSThresholdFeasibilityResult:
    return QoSThresholdFeasibilityResult(
        feasible=False,
        proven_infeasible=bool(proven_infeasible),
        requested_pd=float(requested_pd),
        requested_deflection=float(requested_deflection),
        selected_set=tuple(),
        tx_role=np.zeros(K, dtype=bool),
        rx_role=np.zeros(K, dtype=bool),
        receiver_owner=np.full(Q, -1, dtype=np.int64),
        sensing_power_w=np.zeros((K, Q), dtype=np.float64),
        target_deflection=np.zeros(Q, dtype=np.float64),
        target_pd=np.full(Q, float(p_fa), dtype=np.float64),
        solve_time_s=float(solve_time_s),
        solver_status=int(solver_status),
        solver_message=str(solver_message),
    )


def solve_qos_threshold_feasibility_milp(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    requested_pd: float,
    p_fa: float,
    target_pair_limit: int,
    reports_per_receiver: int,
    time_limit_s: float = 10.0,
    mip_relative_gap: float = 0.0,
) -> QoSThresholdFeasibilityResult:
    """Decide a common target-detection floor under exact hard constraints.

    One UAV cannot transmit and receive in the same structural slot.  Every
    target has at most one receiver owner, selected edges must terminate at
    that owner, each receiver has finite reporting capacity, each target has a
    finite pair count, and every transmitter obeys its sensing-power budget.
    """
    gain, budget = _validate_inputs(gain_per_watt, sensing_budget_w)
    K, _, Q = gain.shape
    probability = float(requested_pd)
    false_alarm = float(p_fa)
    if not np.isfinite(probability) or not false_alarm <= probability < 1.0:
        raise ValueError("requested_pd must lie in [p_fa,1)")
    if not np.isfinite(false_alarm) or not 0.0 < false_alarm < 1.0:
        raise ValueError("p_fa must lie in (0,1)")
    pair_limit = int(target_pair_limit)
    receiver_limit = int(reports_per_receiver)
    if pair_limit < 1 or receiver_limit < 1:
        raise ValueError("pair and receiver limits must be positive")
    if not np.isfinite(time_limit_s) or float(time_limit_s) <= 0.0:
        raise ValueError("time_limit_s must be positive")
    if not np.isfinite(mip_relative_gap) or float(mip_relative_gap) < 0.0:
        raise ValueError("mip_relative_gap must be non-negative")

    required = float(minimum_deflection_for_detection_probability(
        np.asarray([probability]), false_alarm)[0])
    edges = [
        (tx, rx, target)
        for tx in range(K)
        for rx in range(K)
        if tx != rx
        for target in range(Q)
        if gain[tx, rx, target] > 0.0 and budget[tx] > 0.0
    ]
    E = len(edges)
    if required > 0.0 and any(
        not any(edge[2] == target for edge in edges)
        for target in range(Q)
    ):
        return _empty_result(
            K=K, Q=Q, requested_pd=probability,
            requested_deflection=required, p_fa=false_alarm,
            solve_time_s=0.0, solver_status=2,
            solver_message="at least one target has no positive-gain edge",
            proven_infeasible=True,
        )

    x_offset = 0
    tx_offset = x_offset + E
    rx_offset = tx_offset + K
    owner_offset = rx_offset + K
    flow_offset = owner_offset + K * Q
    variable_count = flow_offset + E

    def owner_index(receiver: int, target: int) -> int:
        return owner_offset + receiver * Q + target

    objective = np.zeros(variable_count, dtype=np.float64)
    if E:
        objective[x_offset:x_offset + E] = 1.0e-9
    lower = np.zeros(variable_count, dtype=np.float64)
    upper = np.ones(variable_count, dtype=np.float64)
    for edge_index, (transmitter, _, _) in enumerate(edges):
        upper[flow_offset + edge_index] = budget[transmitter]
    integrality = np.zeros(variable_count, dtype=np.int32)
    integrality[:flow_offset] = 1

    rows: list[np.ndarray] = []
    bounds: list[float] = []

    def add(terms: tuple[tuple[int, float], ...], bound: float) -> None:
        row = np.zeros(variable_count, dtype=np.float64)
        for index, coefficient in terms:
            row[index] += float(coefficient)
        rows.append(row)
        bounds.append(float(bound))

    for node in range(K):
        add(((tx_offset + node, 1.0), (rx_offset + node, 1.0)), 1.0)
    for target in range(Q):
        add(tuple(
            (owner_index(receiver, target), 1.0)
            for receiver in range(K)
        ), 1.0)
    for transmitter in range(K):
        terms = tuple(
            (flow_offset + edge_index, 1.0)
            for edge_index, edge in enumerate(edges)
            if edge[0] == transmitter
        ) + ((tx_offset + transmitter, -budget[transmitter]),)
        add(terms, 0.0)
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

    if required > 0.0:
        for target in range(Q):
            add(tuple(
                (flow_offset + edge_index, -gain[edge])
                for edge_index, edge in enumerate(edges)
                if edge[2] == target
            ), -required)

    started = perf_counter()
    solved = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            np.stack(rows) if rows else np.zeros((0, variable_count)),
            lb=np.full(len(rows), -np.inf, dtype=np.float64),
            ub=np.asarray(bounds, dtype=np.float64),
        ),
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": float(mip_relative_gap),
        },
    )
    elapsed = perf_counter() - started
    if solved.x is None:
        return _empty_result(
            K=K, Q=Q, requested_pd=probability,
            requested_deflection=required, p_fa=false_alarm,
            solve_time_s=elapsed, solver_status=int(solved.status),
            solver_message=str(solved.message),
            proven_infeasible=int(solved.status) == 2,
        )

    values = np.asarray(solved.x, dtype=np.float64)
    selected = tuple(sorted(
        edge for edge_index, edge in enumerate(edges)
        if values[x_offset + edge_index] >= 0.5
    ))
    tx_role = values[tx_offset:tx_offset + K] >= 0.5
    rx_role = values[rx_offset:rx_offset + K] >= 0.5
    owner = np.full(Q, -1, dtype=np.int64)
    for target in range(Q):
        candidates = [
            receiver for receiver in range(K)
            if values[owner_index(receiver, target)] >= 0.5
        ]
        if candidates:
            owner[target] = int(candidates[0])
    flows = np.maximum(values[flow_offset:flow_offset + E], 0.0)
    power = np.zeros((K, Q), dtype=np.float64)
    target_deflection = np.zeros(Q, dtype=np.float64)
    for edge_index, edge in enumerate(edges):
        power[edge[0], edge[2]] += flows[edge_index]
        target_deflection[edge[2]] += gain[edge] * flows[edge_index]
    target_pd = compute_detection_probabilities(target_deflection, false_alarm)
    tolerance = max(1.0e-8, required * 1.0e-7)
    feasible = bool(
        np.all(target_deflection + tolerance >= required)
        and np.all(np.sum(power, axis=1) <= budget + 1.0e-8)
        and not np.any(tx_role & rx_role)
    )
    if not feasible:
        return _empty_result(
            K=K, Q=Q, requested_pd=probability,
            requested_deflection=required, p_fa=false_alarm,
            solve_time_s=elapsed, solver_status=int(solved.status),
            solver_message="solver returned a numerically invalid witness",
            proven_infeasible=False,
        )
    return QoSThresholdFeasibilityResult(
        feasible=True,
        proven_infeasible=False,
        requested_pd=probability,
        requested_deflection=required,
        selected_set=selected,
        tx_role=tx_role,
        rx_role=rx_role,
        receiver_owner=owner,
        sensing_power_w=power,
        target_deflection=target_deflection,
        target_pd=target_pd,
        solve_time_s=float(elapsed),
        solver_status=int(solved.status),
        solver_message=str(solved.message),
    )


def solve_maxmin_qos_boundary_bisection(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    p_fa: float,
    target_pair_limit: int,
    reports_per_receiver: int,
    initial_feasible_pd: float | None = None,
    probability_tolerance: float = 1.0e-3,
    max_iterations: int = 16,
    time_limit_s: float = 10.0,
) -> MaxMinQoSBoundaryResult:
    """Bracket the exact common detection floor by monotone MILP bisection."""
    tolerance = float(probability_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("probability_tolerance must be positive")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive")
    false_alarm = float(p_fa)
    lower_probability = (
        false_alarm if initial_feasible_pd is None
        else float(initial_feasible_pd)
    )
    if (
        not np.isfinite(lower_probability)
        or not false_alarm <= lower_probability < 1.0
    ):
        raise ValueError("initial_feasible_pd must lie in [p_fa,1)")
    upper_probability = 1.0 - 1.0e-9
    started = perf_counter()
    best = solve_qos_threshold_feasibility_milp(
        gain_per_watt,
        sensing_budget_w,
        requested_pd=lower_probability,
        p_fa=false_alarm,
        target_pair_limit=target_pair_limit,
        reports_per_receiver=reports_per_receiver,
        time_limit_s=time_limit_s,
    )
    if not best.feasible:
        raise RuntimeError("the supplied initial detection floor is not feasible")
    exact_upper = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        if upper_probability - lower_probability <= tolerance:
            break
        middle = 0.5 * (lower_probability + upper_probability)
        result = solve_qos_threshold_feasibility_milp(
            gain_per_watt,
            sensing_budget_w,
            requested_pd=middle,
            p_fa=false_alarm,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
            time_limit_s=time_limit_s,
        )
        if result.feasible:
            lower_probability = middle
            best = result
        elif result.proven_infeasible:
            upper_probability = middle
            exact_upper = True
        else:
            break
    lower_deflection = float(minimum_deflection_for_detection_probability(
        np.asarray([lower_probability]), false_alarm)[0])
    upper_deflection = float(minimum_deflection_for_detection_probability(
        np.asarray([upper_probability]), false_alarm)[0])
    return MaxMinQoSBoundaryResult(
        lower_pd=float(lower_probability),
        upper_pd=float(upper_probability),
        lower_deflection=lower_deflection,
        upper_deflection=upper_deflection,
        iterations=int(iterations),
        exact_infeasible_upper=bool(exact_upper),
        best_feasible=best,
        solve_time_s=float(perf_counter() - started),
    )
