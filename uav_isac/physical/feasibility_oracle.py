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

from uav_isac.utils.math_utils import Q_inverse, compute_PD
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


def _normalize_oracle_fusion_mode(mode: str) -> str:
    """Map detector labels to the two physical objectives supported here."""
    normalized = str(mode).strip().lower()
    if normalized in {"central_oracle", "legacy_global"}:
        return "central_oracle"
    if normalized == "local_only":
        return "local_only"
    raise ValueError(
        "physical feasibility oracle supports detection_fusion_mode "
        "'local_only', 'central_oracle', or 'legacy_global'; "
        f"received {mode!r}"
    )


def _selected_detection_deflection(
    coefficient: np.ndarray,
    power: np.ndarray,
    selected: Sequence[Tuple[int, int, int]],
    *,
    fusion_mode: str,
) -> np.ndarray:
    """Evaluate a reporting graph using the same receiver boundary as deployment."""
    k_count, _, q_count = coefficient.shape
    receiver_d = np.zeros((k_count, q_count), dtype=np.float64)
    for i, j, q in selected:
        receiver_d[int(j), int(q)] += (
            coefficient[int(i), int(j), int(q)]
            * power[int(i), int(q)]
        )
    if _normalize_oracle_fusion_mode(fusion_mode) == "local_only":
        return np.max(receiver_d, axis=0)
    return np.sum(receiver_d, axis=0)


def _select_pairs_local_only_milp(
    coefficient: np.ndarray,
    power: np.ndarray,
    edges: Sequence[Tuple[int, int, int]],
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
) -> Tuple[Tuple[int, int, int], ...]:
    """Exact fixed-power pair update for receiver-local detection.

    A local detector may accumulate several transmitter echoes at one receiver,
    but it may not add evidence held by different receivers.  An optimum can
    therefore nominate one receiver per target without loss: edges at every
    other receiver can be removed while preserving the target maximum and
    relaxing all capacity constraints.
    """
    q_count = coefficient.shape[2]
    edge_count = len(edges)
    if edge_count == 0:
        return tuple()
    owner_pairs = sorted({(int(j), int(q)) for _, j, q in edges})
    owner_index = {
        pair: edge_count + index for index, pair in enumerate(owner_pairs)
    }
    owner_count = len(owner_pairs)
    min_index = edge_count + owner_count
    variable_count = min_index + 1

    edge_deflection = np.asarray([
        coefficient[i, j, q] * power[i, q] for i, j, q in edges
    ], dtype=np.float64)
    scale = max(float(np.max(edge_deflection)), 1.0)
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:edge_count] = -1.0e-7 * edge_deflection / scale
    objective[min_index] = -1.0

    rows = []
    upper = []
    # With at most one owner, all selected target evidence is receiver-local.
    for q in range(q_count):
        row = np.zeros(variable_count, dtype=np.float64)
        for edge_idx, edge in enumerate(edges):
            if edge[2] == q:
                row[edge_idx] = -edge_deflection[edge_idx]
        row[min_index] = 1.0
        rows.append(row)
        upper.append(0.0)
    for q in range(q_count):
        row = np.zeros(variable_count, dtype=np.float64)
        for edge_idx, edge in enumerate(edges):
            if edge[2] == q:
                row[edge_idx] = 1.0
        rows.append(row)
        upper.append(float(target_pair_limit))
    for j in range(coefficient.shape[0]):
        row = np.zeros(variable_count, dtype=np.float64)
        for edge_idx, edge in enumerate(edges):
            if edge[1] == j:
                row[edge_idx] = 1.0
        rows.append(row)
        upper.append(float(reports_per_receiver))
    # A selected edge activates its receiver as the target owner.
    for edge_idx, (_, j, q) in enumerate(edges):
        row = np.zeros(variable_count, dtype=np.float64)
        row[edge_idx] = 1.0
        row[owner_index[(int(j), int(q))]] = -1.0
        rows.append(row)
        upper.append(0.0)
    # At most one receiver may contribute to each target's detector statistic.
    for q in range(q_count):
        row = np.zeros(variable_count, dtype=np.float64)
        for pair, index in owner_index.items():
            if pair[1] == q:
                row[index] = 1.0
        rows.append(row)
        upper.append(1.0)

    result = milp(
        objective,
        integrality=np.concatenate([
            np.ones(edge_count + owner_count, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        ]),
        bounds=Bounds(
            np.zeros(variable_count, dtype=np.float64),
            np.concatenate([
                np.ones(edge_count + owner_count, dtype=np.float64),
                np.array([np.inf], dtype=np.float64),
            ]),
        ),
        constraints=LinearConstraint(
            np.stack(rows),
            lb=np.full(len(rows), -np.inf),
            ub=np.asarray(upper, dtype=np.float64),
        ),
        options={"time_limit": 5.0},
    )
    if not result.success or result.x is None:
        return tuple()
    return tuple(sorted(
        edges[index] for index in range(edge_count)
        if result.x[index] >= 0.5
    ))


def solve_maxmin_single_role_pairs(
    entries: Iterable[DeflectionEntry],
    *,
    num_uavs: int,
    num_targets: int,
    target_pair_limit: int,
    reports_per_receiver: int,
    p_fa: float,
    p_d_floor: float,
    target_priority: np.ndarray | None = None,
    fusion_mode: str = "central_oracle",
) -> Tuple[Tuple[Tuple[int, int, int], ...], np.ndarray]:
    """QoS-capped lexicographic max-min selection with one role per UAV.

    Pure max-min is degenerate when a filtered graph leaves any target without
    an edge: the strict minimum is identically zero. Capped per-target
    auxiliaries preserve the primary minimum objective and then maximize how
    many remaining targets approach the deployment floor.
    """
    valid = [
        entry for entry in entries
        if entry.i != entry.j and float(entry.d_eff) > 0.0
    ]
    k_count = int(num_uavs)
    q_count = int(num_targets)
    if not valid:
        return tuple(), np.zeros(q_count, dtype=np.float64)
    normalized_fusion = _normalize_oracle_fusion_mode(fusion_mode)
    if normalized_fusion == "local_only":
        # Entries already contain realized, power-weighted deflection.  Treat
        # them as unit-power coefficients so the exact receiver-owner
        # pair-only solver can optimize roles and edges without changing RF
        # allocation.  Strict max-min is the correct monotone objective for
        # worst P_D under a fixed P_FA.
        coefficient = unit_deflection_tensor(
            valid,
            num_uavs=k_count,
            num_targets=q_count,
        )
        fixed_unit_power = np.ones(
            (k_count, q_count), dtype=np.float64)
        solution = solve_pair_only_oracle(
            coefficient,
            fixed_unit_power,
            P_FA=float(p_fa),
            target_pair_limit=int(target_pair_limit),
            reports_per_receiver=int(reports_per_receiver),
            fusion_mode="local_only",
        )
        return solution.selected_set, solution.D_q.copy()

    edge_count = len(valid)
    # Binary edges, binary transmitter roles, capped target values, then min D.
    capped_offset = edge_count + k_count
    variable_count = edge_count + k_count + q_count + 1
    min_index = variable_count - 1
    objective = np.zeros(variable_count, dtype=np.float64)
    deflection = np.asarray(
        [float(entry.d_eff) for entry in valid], dtype=np.float64)
    scale = max(float(np.max(deflection)), 1.0)
    q_fa = float(Q_inverse(np.asarray(float(p_fa))))
    q_pd = float(Q_inverse(np.asarray(float(p_d_floor))))
    deflection_floor = max((q_fa - q_pd) ** 2, 1.0e-9)
    priority = (
        np.ones(q_count, dtype=np.float64)
        if target_priority is None
        else np.asarray(target_priority, dtype=np.float64).reshape(-1)
    )
    if priority.shape != (q_count,):
        raise ValueError("target_priority must have shape (num_targets,)")
    priority = np.maximum(priority, 1.0e-9)
    priority = priority / float(np.sum(priority))
    # Deterministic tertiary total-gain/index objective. Long-term target
    # identity is handled by the deficit-weighted capped target variables.
    for edge_index, entry in enumerate(valid):
        objective[edge_index] = (
            -0.002 / edge_count * deflection[edge_index] / scale
            - 1.0e-6 * (edge_count - edge_index) / edge_count
        )
    objective[capped_offset:capped_offset + q_count] = (
        -0.10 * priority / deflection_floor)
    objective[min_index] = -1.0 / deflection_floor

    rows = []
    upper = []
    # Capped target epigraph z_q <= D_q and strict y <= z_q.
    for target in range(q_count):
        row = np.zeros(variable_count, dtype=np.float64)
        for edge_index, entry in enumerate(valid):
            if int(entry.q) == target:
                row[edge_index] = -float(entry.d_eff)
        row[capped_offset + target] = 1.0
        rows.append(row)
        upper.append(0.0)
        min_row = np.zeros(variable_count, dtype=np.float64)
        min_row[min_index] = 1.0
        min_row[capped_offset + target] = -1.0
        rows.append(min_row)
        upper.append(0.0)
    # Per-target reporting cardinality.
    for target in range(q_count):
        row = np.zeros(variable_count, dtype=np.float64)
        for edge_index, entry in enumerate(valid):
            if int(entry.q) == target:
                row[edge_index] = 1.0
        rows.append(row)
        upper.append(float(target_pair_limit))
    # Per-receiver report capacity.
    for receiver in range(k_count):
        row = np.zeros(variable_count, dtype=np.float64)
        for edge_index, entry in enumerate(valid):
            if int(entry.j) == receiver:
                row[edge_index] = 1.0
        rows.append(row)
        upper.append(float(reports_per_receiver))
    # x_ijq <= role_i and x_ijq <= 1-role_j.
    for edge_index, entry in enumerate(valid):
        tx_role_index = edge_count + int(entry.i)
        rx_role_index = edge_count + int(entry.j)
        tx_row = np.zeros(variable_count, dtype=np.float64)
        tx_row[edge_index] = 1.0
        tx_row[tx_role_index] = -1.0
        rows.append(tx_row)
        upper.append(0.0)
        rx_row = np.zeros(variable_count, dtype=np.float64)
        rx_row[edge_index] = 1.0
        rx_row[rx_role_index] = 1.0
        rows.append(rx_row)
        upper.append(1.0)

    result = milp(
        objective,
        integrality=np.concatenate([
            np.ones(edge_count + k_count, dtype=np.int32),
            np.zeros(q_count + 1, dtype=np.int32),
        ]),
        bounds=Bounds(
            np.zeros(variable_count, dtype=np.float64),
            np.concatenate([
                np.ones(edge_count + k_count, dtype=np.float64),
                np.full(q_count + 1, deflection_floor, dtype=np.float64),
            ]),
        ),
        constraints=LinearConstraint(
            np.stack(rows),
            lb=np.full(len(rows), -np.inf),
            ub=np.asarray(upper, dtype=np.float64),
        ),
        options={"time_limit": 5.0},
    )
    if not result.success or result.x is None:
        return tuple(), np.zeros(q_count, dtype=np.float64)
    selected = tuple(sorted(
        (int(entry.i), int(entry.j), int(entry.q))
        for edge_index, entry in enumerate(valid)
        if result.x[edge_index] >= 0.5
    ))
    lookup = {
        (int(entry.i), int(entry.j), int(entry.q)): float(entry.d_eff)
        for entry in valid
    }
    D_q = np.zeros(q_count, dtype=np.float64)
    for edge in selected:
        D_q[edge[2]] += lookup[edge]
    return selected, D_q


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


def _optimize_power_local_only_milp(
    coefficient: np.ndarray,
    selected: Sequence[Tuple[int, int, int]],
    tx_indices: Sequence[int],
    per_uav_sensing_budget_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact fixed-graph max-min power update for receiver-local detection."""
    k_count, _, q_count = coefficient.shape
    power_count = k_count * q_count
    owner_pairs = sorted({(int(j), int(q)) for i, j, q in selected})
    owners_by_target = {
        q: [pair for pair in owner_pairs if pair[1] == q]
        for q in range(q_count)
    }
    if any(not owners_by_target[q] for q in range(q_count)):
        zeros = np.zeros((k_count, q_count), dtype=np.float64)
        return zeros, np.zeros(q_count, dtype=np.float64)

    owner_count = len(owner_pairs)
    owner_offset = power_count
    min_index = owner_offset + owner_count
    variable_count = min_index + 1
    owner_index = {
        pair: owner_offset + index for index, pair in enumerate(owner_pairs)
    }
    selected_lookup = {
        (int(i), int(j), int(q)) for i, j, q in selected
    }
    maximum_d = 0.0
    for j, q in owner_pairs:
        maximum_d = max(maximum_d, float(sum(
            coefficient[i, j, q] * per_uav_sensing_budget_w[i]
            for i in range(k_count)
            if (i, j, q) in selected_lookup
        )))
    big_m = max(maximum_d, 1.0)

    objective = np.zeros(variable_count, dtype=np.float64)
    objective[min_index] = -1.0
    rows = []
    upper = []
    for k in range(k_count):
        row = np.zeros(variable_count, dtype=np.float64)
        row[k * q_count:(k + 1) * q_count] = 1.0
        rows.append(row)
        upper.append(float(per_uav_sensing_budget_w[k]))
    # If owner w_jq=1, impose y <= D_jq.  Otherwise big-M relaxes it.
    for j, q in owner_pairs:
        row = np.zeros(variable_count, dtype=np.float64)
        for i in range(k_count):
            if (i, j, q) in selected_lookup:
                row[i * q_count + q] = -coefficient[i, j, q]
        row[owner_index[(j, q)]] = big_m
        row[min_index] = 1.0
        rows.append(row)
        upper.append(big_m)
    for q in range(q_count):
        row = np.zeros(variable_count, dtype=np.float64)
        for pair in owners_by_target[q]:
            row[owner_index[pair]] = 1.0
        rows.append(row)
        upper.append(1.0)
        rows.append(-row)
        upper.append(-1.0)

    tx = set(int(i) for i in tx_indices)
    power_upper = np.zeros(power_count, dtype=np.float64)
    for k in range(k_count):
        if k in tx:
            power_upper[
                k * q_count:(k + 1) * q_count
            ] = float(per_uav_sensing_budget_w[k])
    result = milp(
        objective,
        integrality=np.concatenate([
            np.zeros(power_count, dtype=np.int32),
            np.ones(owner_count, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        ]),
        bounds=Bounds(
            np.zeros(variable_count, dtype=np.float64),
            np.concatenate([
                power_upper,
                np.ones(owner_count, dtype=np.float64),
                np.array([big_m], dtype=np.float64),
            ]),
        ),
        constraints=LinearConstraint(
            np.stack(rows),
            lb=np.full(len(rows), -np.inf),
            ub=np.asarray(upper, dtype=np.float64),
        ),
        options={"time_limit": 5.0},
    )
    if not result.success or result.x is None:
        zeros = np.zeros((k_count, q_count), dtype=np.float64)
        return zeros, np.zeros(q_count, dtype=np.float64)
    power = result.x[:power_count].reshape(k_count, q_count)
    D_q = _selected_detection_deflection(
        coefficient,
        power,
        selected,
        fusion_mode="local_only",
    )
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
    fusion_mode: str,
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
            if fusion_mode == "local_only":
                selected = _select_pairs_local_only_milp(
                    coefficient, power, edges,
                    target_pair_limit=target_pair_limit,
                    reports_per_receiver=reports_per_receiver)
                power, D_q = _optimize_power_local_only_milp(
                    coefficient, selected, tx_indices,
                    per_uav_sensing_budget_w)
            else:
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
    fusion_mode: str = "central_oracle",
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
    normalized_fusion = _normalize_oracle_fusion_mode(fusion_mode)

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
            fusion_mode=normalized_fusion,
        )
        key = (solution.worst, solution.weak3, solution.steady)
        best_key = ((best.worst, best.weak3, best.steady)
                    if best is not None else (-np.inf, -np.inf, -np.inf))
        if key > best_key:
            best = solution
    assert best is not None
    return best


def solve_pair_only_oracle(
    coefficient: np.ndarray,
    current_power_w: np.ndarray,
    *,
    P_FA: float,
    target_pair_limit: int = 3,
    reports_per_receiver: int = 4,
    fusion_mode: str = "central_oracle",
) -> OracleSolution:
    """Optimize single-role reporting pairs while holding target power fixed."""
    coefficient = np.asarray(coefficient, dtype=np.float64)
    power = np.asarray(current_power_w, dtype=np.float64)
    if coefficient.ndim != 3 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("coefficient must have shape (K, K, Q)")
    k_count, _, q_count = coefficient.shape
    if power.shape != (k_count, q_count):
        raise ValueError("current power must have shape (K, Q)")
    normalized_fusion = _normalize_oracle_fusion_mode(fusion_mode)
    best: OracleSolution | None = None
    for mask in range(1, (1 << k_count) - 1):
        tx = tuple(k for k in range(k_count) if mask & (1 << k))
        rx = tuple(k for k in range(k_count) if not mask & (1 << k))
        edges = _candidate_edges(coefficient, tx, rx)
        selector = (
            _select_pairs_local_only_milp
            if normalized_fusion == "local_only"
            else _select_pairs_milp
        )
        selected = selector(
            coefficient,
            power,
            edges,
            target_pair_limit=int(target_pair_limit),
            reports_per_receiver=int(reports_per_receiver),
        )
        D_q = _selected_detection_deflection(
            coefficient,
            power,
            selected,
            fusion_mode=normalized_fusion,
        )
        solution = OracleSolution(
            mode="pair_only",
            tx_indices=tx,
            rx_indices=rx,
            selected_set=tuple(selected),
            sensing_power_w=power.copy(),
            D_q=D_q,
            P_D_q=compute_PD(D_q, P_FA),
        )
        key = (solution.worst, solution.weak3, solution.steady)
        best_key = (
            (best.worst, best.weak3, best.steady)
            if best is not None else (-np.inf, -np.inf, -np.inf)
        )
        if key > best_key:
            best = solution
    assert best is not None
    return best


def solve_power_only_oracle(
    coefficient: np.ndarray,
    selected_set: Sequence[Tuple[int, int, int]],
    *,
    P_FA: float,
    total_power_w: float = 1.0,
    communication_reserve_w: float = 0.0,
    fusion_mode: str = "central_oracle",
) -> OracleSolution:
    """Optimize max-min target power while holding reporting pairs fixed."""
    coefficient = np.asarray(coefficient, dtype=np.float64)
    if coefficient.ndim != 3 or coefficient.shape[0] != coefficient.shape[1]:
        raise ValueError("coefficient must have shape (K, K, Q)")
    k_count, _, q_count = coefficient.shape
    sensing_budget = float(total_power_w) - float(communication_reserve_w)
    if sensing_budget <= 0.0:
        raise ValueError("communication reserve must be below total power")
    selected = tuple(
        (int(i), int(j), int(q)) for i, j, q in selected_set)
    tx_indices = tuple(sorted({i for i, _, _ in selected}))
    rx_indices = tuple(sorted({j for _, j, _ in selected}))
    budgets = np.full(k_count, sensing_budget, dtype=np.float64)
    normalized_fusion = _normalize_oracle_fusion_mode(fusion_mode)
    if normalized_fusion == "local_only":
        power, D_q = _optimize_power_local_only_milp(
            coefficient,
            selected,
            tx_indices,
            budgets,
        )
    else:
        power, D_q = _optimize_power_lp(
            coefficient,
            selected,
            tx_indices,
            budgets,
        )
    return OracleSolution(
        mode="power_only",
        tx_indices=tx_indices,
        rx_indices=rx_indices,
        selected_set=selected,
        sensing_power_w=power,
        D_q=D_q,
        P_D_q=compute_PD(D_q, P_FA),
    )
