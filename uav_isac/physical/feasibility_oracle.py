"""Full-information feasibility oracle for joint bistatic pairing and power.

This module is deliberately independent of MARL.  It asks whether a geometry
can satisfy the sensing objective when target state, endpoint roles, reporting
pairs, and the per-UAV sensing split are all known.  The result is an upper
benchmark/teacher, never a deployable controller.

For fixed endpoint roles the remaining problem is mixed discrete/continuous:
reporting pairs are binary while target power is continuous.  We alternate an
exact MILP pair update with an exact LP max-min power update from several
initializations.  This is not a proof of global optimality, but every returned
solution is physically feasible under the supplied role, receiver-capacity,
target-cardinality, and per-UAV RF constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment
from scipy.optimize import linprog, milp

from uav_isac.utils.math_utils import compute_PD
from uav_isac.utils.types import DeflectionEntry


@dataclass(frozen=True)
class OracleSolution:
    mode: str
    tx_indices: Tuple[int, ...]
    rx_indices: Tuple[int, ...]
    selected_set: Tuple[Tuple[int, int, int], ...]
    sensing_power_w: np.ndarray
    D_q: np.ndarray
    P_D_q: np.ndarray

    @property
    def worst(self) -> float:
        return float(np.min(self.P_D_q))

    @property
    def weak3(self) -> float:
        ordered = np.sort(self.P_D_q)
        return float(np.mean(ordered[:min(3, ordered.size)]))

    @property
    def steady(self) -> float:
        return float(np.mean(self.P_D_q))


def reachable_hungarian_geometry(
    uav_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    travel_distance_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move each UAV directly toward its minimum-distance assigned target."""
    uav = np.asarray(uav_positions, dtype=np.float64)
    target = np.asarray(target_positions, dtype=np.float64)
    if uav.ndim != 2 or target.ndim != 2 or uav.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError('UAV and target positions must have shape (N, 3)')
    cost = np.linalg.norm(
        uav[:, None, :2] - target[None, :, :2], axis=-1)
    rows, columns = linear_sum_assignment(cost)
    final = uav.copy()
    velocity_direction = np.zeros_like(uav)
    assignment = np.full(uav.shape[0], -1, dtype=np.int64)
    for k, q in zip(rows, columns):
        delta = target[q, :2] - uav[k, :2]
        distance = float(np.linalg.norm(delta))
        assignment[k] = int(q)
        if distance <= 1e-12:
            continue
        unit = delta / distance
        step = min(max(float(travel_distance_m), 0.0), distance)
        final[k, :2] += step * unit
        velocity_direction[k, :2] = unit
    return final, velocity_direction, assignment


def unit_deflection_tensor(
    entries: Iterable[DeflectionEntry],
    num_uavs: int,
    num_targets: int,
) -> np.ndarray:
    """Collect deflection-per-watt coefficients into a dense K x K x Q tensor."""
    coefficient = np.zeros(
        (int(num_uavs), int(num_uavs), int(num_targets)),
        dtype=np.float64)
    for entry in entries:
        if entry.i != entry.j and entry.d_eff > 0.0:
            coefficient[entry.i, entry.j, entry.q] = float(entry.d_eff)
    return coefficient


def _candidate_edges(
    coefficient: np.ndarray,
    tx_indices: Sequence[int],
    rx_indices: Sequence[int],
) -> List[Tuple[int, int, int]]:
    edges: List[Tuple[int, int, int]] = []
    for i in tx_indices:
        for j in rx_indices:
            if i == j:
                continue
            for q in range(coefficient.shape[2]):
                if coefficient[i, j, q] > 0.0:
                    edges.append((int(i), int(j), int(q)))
    return edges


def _select_pairs_milp(
    coefficient: np.ndarray,
    power: np.ndarray,
    edges: Sequence[Tuple[int, int, int]],
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
) -> Tuple[Tuple[int, int, int], ...]:
    """Maximize the minimum target deflection for a fixed power allocation."""
    q_count = coefficient.shape[2]
    edge_count = len(edges)
    if edge_count == 0:
        return tuple()
    # Binary edge variables followed by a continuous min-deflection variable.
    objective = np.zeros(edge_count + 1, dtype=np.float64)
    objective[-1] = -1.0
    edge_deflection = np.asarray([
        coefficient[i, j, q] * power[i, q] for i, j, q in edges
    ], dtype=np.float64)
    scale = max(float(np.max(edge_deflection)), 1.0)
    objective[:edge_count] = -1e-7 * edge_deflection / scale

    rows = []
    upper = []
    # y <= D_q for every target.
    for q in range(q_count):
        row = np.zeros(edge_count + 1, dtype=np.float64)
        for idx, edge in enumerate(edges):
            if edge[2] == q:
                row[idx] = -edge_deflection[idx]
        row[-1] = 1.0
        rows.append(row)
        upper.append(0.0)
    # Target cardinality.
    for q in range(q_count):
        row = np.zeros(edge_count + 1, dtype=np.float64)
        for idx, edge in enumerate(edges):
            if edge[2] == q:
                row[idx] = 1.0
        rows.append(row)
        upper.append(float(target_pair_limit))
    # Receiver reporting capacity.
    for j in range(coefficient.shape[0]):
        row = np.zeros(edge_count + 1, dtype=np.float64)
        for idx, edge in enumerate(edges):
            if edge[1] == j:
                row[idx] = 1.0
        rows.append(row)
        upper.append(float(reports_per_receiver))

    constraint = LinearConstraint(
        np.stack(rows),
        lb=np.full(len(rows), -np.inf),
        ub=np.asarray(upper, dtype=np.float64))
    lower_bounds = np.zeros(edge_count + 1, dtype=np.float64)
    upper_bounds = np.concatenate([
        np.ones(edge_count, dtype=np.float64), np.array([np.inf])])
    result = milp(
        objective,
        integrality=np.concatenate([
            np.ones(edge_count, dtype=np.int32), np.zeros(1, dtype=np.int32)]),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraint,
        options={'time_limit': 5.0},
    )
    if not result.success or result.x is None:
        return tuple()
    chosen = [edges[idx] for idx in range(edge_count) if result.x[idx] >= 0.5]
    return tuple(sorted(chosen))


def _optimize_power_lp(
    coefficient: np.ndarray,
    selected: Sequence[Tuple[int, int, int]],
    tx_indices: Sequence[int],
    per_uav_sensing_budget_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Max-min target power allocation for a fixed reporting graph."""
    k_count, _, q_count = coefficient.shape
    variable_count = k_count * q_count
    gain = np.zeros((q_count, variable_count), dtype=np.float64)
    for i, j, q in selected:
        gain[q, i * q_count + q] += coefficient[i, j, q]

    objective = np.zeros(variable_count + 1, dtype=np.float64)
    objective[-1] = -1.0
    rows = []
    upper = []
    for q in range(q_count):
        row = np.zeros(variable_count + 1, dtype=np.float64)
        row[:variable_count] = -gain[q]
        row[-1] = 1.0
        rows.append(row)
        upper.append(0.0)
    for k in range(k_count):
        row = np.zeros(variable_count + 1, dtype=np.float64)
        row[k * q_count:(k + 1) * q_count] = 1.0
        rows.append(row)
        upper.append(float(per_uav_sensing_budget_w[k]))

    tx = set(int(i) for i in tx_indices)
    bounds = []
    for k in range(k_count):
        for _ in range(q_count):
            bounds.append((
                0.0,
                float(per_uav_sensing_budget_w[k]) if k in tx else 0.0))
    bounds.append((0.0, None))
    result = linprog(
        objective,
        A_ub=np.stack(rows),
        b_ub=np.asarray(upper, dtype=np.float64),
        bounds=bounds,
        method='highs',
    )
    if not result.success or result.x is None:
        return (
            np.zeros((k_count, q_count), dtype=np.float64),
            np.zeros(q_count, dtype=np.float64),
        )
    power = result.x[:variable_count].reshape(k_count, q_count)
    D_q = gain @ result.x[:variable_count]
    return power, D_q


def _initial_power_allocations(
    tx_indices: Sequence[int],
    budgets: np.ndarray,
    num_targets: int,
    rng: np.random.Generator,
    random_starts: int,
) -> List[np.ndarray]:
    k_count = budgets.size
    starts = []
    uniform = np.zeros((k_count, num_targets), dtype=np.float64)
    for i in tx_indices:
        uniform[i] = budgets[i] / max(num_targets, 1)
    starts.append(uniform)
    for _ in range(max(int(random_starts), 0)):
        allocation = np.zeros_like(uniform)
        for i in tx_indices:
            allocation[i] = budgets[i] * rng.dirichlet(
                np.ones(num_targets, dtype=np.float64))
        starts.append(allocation)
    return starts


def _solve_for_roles(
    coefficient: np.ndarray,
    tx_indices: Sequence[int],
    rx_indices: Sequence[int],
    *,
    P_FA: float,
    per_uav_sensing_budget_w: np.ndarray,
    target_pair_limit: int,
    reports_per_receiver: int,
    alternating_iterations: int,
    random_starts: int,
    rng: np.random.Generator,
    mode: str,
) -> OracleSolution:
    k_count, _, q_count = coefficient.shape
    edges = _candidate_edges(coefficient, tx_indices, rx_indices)
    best: OracleSolution | None = None
    starts = _initial_power_allocations(
        tx_indices, per_uav_sensing_budget_w, q_count, rng, random_starts)
    for initial_power in starts:
        power = initial_power
        previous = None
        for _ in range(max(int(alternating_iterations), 1)):
            selected = _select_pairs_milp(
                coefficient, power, edges,
                target_pair_limit=target_pair_limit,
                reports_per_receiver=reports_per_receiver)
            power, D_q = _optimize_power_lp(
                coefficient, selected, tx_indices,
                per_uav_sensing_budget_w)
            if selected == previous:
                break
            previous = selected
        P_D_q = compute_PD(D_q, P_FA)
        solution = OracleSolution(
            mode=mode,
            tx_indices=tuple(int(x) for x in tx_indices),
            rx_indices=tuple(int(x) for x in rx_indices),
            selected_set=tuple(selected),
            sensing_power_w=power,
            D_q=D_q,
            P_D_q=P_D_q,
        )
        key = (solution.worst, solution.weak3, solution.steady)
        best_key = ((best.worst, best.weak3, best.steady)
                    if best is not None else (-np.inf, -np.inf, -np.inf))
        if key > best_key:
            best = solution
    if best is None:
        zeros = np.zeros(q_count, dtype=np.float64)
        best = OracleSolution(
            mode=mode,
            tx_indices=tuple(int(x) for x in tx_indices),
            rx_indices=tuple(int(x) for x in rx_indices),
            selected_set=tuple(),
            sensing_power_w=np.zeros((k_count, q_count), dtype=np.float64),
            D_q=zeros,
            P_D_q=compute_PD(zeros, P_FA),
        )
    return best


def solve_joint_pair_power_oracle(
    coefficient: np.ndarray,
    *,
    P_FA: float,
    total_power_w: float = 1.0,
    communication_reserve_w: float = 0.0,
    target_pair_limit: int = 3,
    reports_per_receiver: int = 4,
    full_duplex: bool = False,
    alternating_iterations: int = 6,
    random_starts: int = 2,
    seed: int = 0,
) -> OracleSolution:
    """Solve the best role partition, reporting graph, and sensing split.

    ``full_duplex=False`` enforces the current one-role-per-UAV architecture.
    ``full_duplex=True`` is an optimistic simultaneous-endpoint diagnostic in
    which every UAV may transmit to and receive from different peers.
    """
    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.ndim != 3 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError('coefficient must have shape (K, K, Q)')
    k_count = coefficient.shape[0]
    sensing_budget = float(total_power_w) - float(communication_reserve_w)
    if sensing_budget <= 0.0:
        raise ValueError('communication reserve must be below total power')
    budgets = np.full(k_count, sensing_budget, dtype=np.float64)
    rng = np.random.default_rng(int(seed))

    partitions: List[Tuple[Tuple[int, ...], Tuple[int, ...], str]] = []
    if full_duplex:
        all_uavs = tuple(range(k_count))
        partitions.append((all_uavs, all_uavs, 'full_duplex'))
    else:
        for mask in range(1, (1 << k_count) - 1):
            tx = tuple(k for k in range(k_count) if mask & (1 << k))
            rx = tuple(k for k in range(k_count) if not mask & (1 << k))
            partitions.append((tx, rx, 'single_role'))

    best: OracleSolution | None = None
    for tx, rx, mode in partitions:
        solution = _solve_for_roles(
            coefficient, tx, rx,
            P_FA=P_FA,
            per_uav_sensing_budget_w=budgets,
            target_pair_limit=int(target_pair_limit),
            reports_per_receiver=int(reports_per_receiver),
            alternating_iterations=int(alternating_iterations),
            random_starts=int(random_starts),
            rng=rng,
            mode=mode,
        )
        key = (solution.worst, solution.weak3, solution.steady)
        best_key = ((best.worst, best.weak3, best.steady)
                    if best is not None else (-np.inf, -np.inf, -np.inf))
        if key > best_key:
            best = solution
    assert best is not None
    return best
