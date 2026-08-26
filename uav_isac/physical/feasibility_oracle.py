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
from itertools import product
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment
from scipy.optimize import linprog, milp
from scipy.sparse import csr_matrix, vstack as sparse_vstack

from uav_isac.coordination.maxmin_power import (
    solve_fixed_structure_maxmin_power_lp,
)
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


def solve_enumerated_role_ceiling_local_pairs(
    coefficient: np.ndarray,
    per_uav_sensing_budget_w: np.ndarray,
    *,
    P_FA: float,
    p_d_floor: float,
    target_pair_limit: int,
    reports_per_receiver: int,
    target_priority: np.ndarray | None = None,
) -> OracleSolution:
    """Fast feasible role decomposition for small/medium UAV fleets.

    Enumerate every non-trivial Tx/Rx role partition (at most 254 for K=8).
    Within a partition, each target chooses the receiver whose strongest
    ``target_pair_limit`` transmitters have the largest budget-weighted
    coefficient ceiling.  The resulting support is evaluated by the exact
    fixed-structure max-min power LP.  Hence the method is a certified primal
    lower bound: it does not claim global mixed-integer optimality, but its
    returned structure and power are physically executable.
    """
    gain = np.asarray(coefficient, dtype=np.float64).copy()
    budget = np.asarray(
        per_uav_sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = gain.shape
    if budget.shape != (K,):
        raise ValueError("per_uav_sensing_budget_w must have shape (K,)")
    if (
        np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError("coefficient/budget must be finite and non-negative")
    if target_pair_limit < 1 or reports_per_receiver < 1:
        raise ValueError("pair and receiver limits must be positive")
    gain[np.arange(K), np.arange(K), :] = 0.0
    priority = (
        np.ones(Q, dtype=np.float64)
        if target_priority is None
        else np.asarray(target_priority, dtype=np.float64).reshape(-1)
    )
    if priority.shape != (Q,) or np.any(~np.isfinite(priority)):
        raise ValueError("target_priority must be a finite Q-vector")
    priority = np.maximum(priority, 1.0e-9)
    priority /= float(np.sum(priority))
    q_fa = float(Q_inverse(np.asarray(float(P_FA))))
    q_pd = float(Q_inverse(np.asarray(float(p_d_floor))))
    d_floor = max((q_fa - q_pd) ** 2, 1.0e-12)

    best_score: tuple[float, float, float] | None = None
    best_selected: tuple[tuple[int, int, int], ...] = tuple()
    best_power = np.zeros((K, Q), dtype=np.float64)
    best_D = np.zeros(Q, dtype=np.float64)
    # Target priority is only a deterministic capacity-allocation order and a
    # lexicographic tie-break after the primary worst-target LP value.
    target_order = np.argsort(-priority, kind="stable")
    for mask in range(1, (1 << K) - 1):
        tx = tuple(i for i in range(K) if mask & (1 << i))
        rx = tuple(i for i in range(K) if not (mask & (1 << i)))
        tx = tuple(i for i in tx if budget[i] > 0.0)
        if not tx or not rx:
            continue
        receiver_load = {j: 0 for j in rx}
        target_edges: dict[int, tuple[tuple[int, int, int], ...]] = {}
        valid_partition = True
        for q_value in target_order:
            q = int(q_value)
            owner_candidates = []
            for j in rx:
                ranked_tx = sorted(
                    tx,
                    key=lambda i: (
                        budget[i] * gain[i, j, q], -i),
                    reverse=True,
                )
                chosen_tx = tuple(
                    i for i in ranked_tx[:target_pair_limit]
                    if gain[i, j, q] > 0.0
                )
                if not chosen_tx:
                    continue
                if (receiver_load[j] + len(chosen_tx)
                        > reports_per_receiver):
                    continue
                ceiling = float(sum(
                    budget[i] * gain[i, j, q] for i in chosen_tx))
                owner_candidates.append((ceiling, -j, j, chosen_tx))
            if not owner_candidates:
                valid_partition = False
                break
            _ceiling, _tie, owner, chosen_tx = max(owner_candidates)
            edges = tuple((int(i), int(owner), q) for i in chosen_tx)
            target_edges[q] = edges
            receiver_load[owner] += len(edges)
        if not valid_partition:
            continue
        selected = tuple(sorted(
            edge for edges in target_edges.values() for edge in edges))
        fixed_gain = np.zeros((K, Q), dtype=np.float64)
        for i, j, q in selected:
            fixed_gain[i, q] = gain[i, j, q]
        result = solve_fixed_structure_maxmin_power_lp(fixed_gain, budget)
        D_q = np.asarray(result.deflection, dtype=np.float64)
        power_tolerance = max(
            1.0e-12, 1.0e-8 * float(np.max(budget)))
        executable_selected = tuple(
            edge for edge in selected
            if result.power_w[edge[0], edge[2]] > power_tolerance
        )
        score = (
            float(np.min(D_q)),
            float(np.dot(priority, np.minimum(D_q, d_floor))),
            -float(len(executable_selected)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_selected = executable_selected
            best_power = result.power_w.copy()
            best_D = D_q.copy()

    if best_score is None:
        return OracleSolution(
            mode="enumerated_role_ceiling_local_only",
            tx_indices=(), rx_indices=(), selected_set=(),
            sensing_power_w=best_power,
            D_q=best_D,
            P_D_q=compute_PD(best_D, P_FA),
        )

    # Receiver-owner coordinate refinement on the best role partition.  The
    # ceiling owner is only a relaxation because transmitter power is shared
    # across targets.  Re-score every single-target owner replacement with the
    # exact coupled power LP and accept strict lexicographic improvements.
    # Two finite sweeps retain monotonicity while adding at most O(KQ) LPs.
    for _sweep in range(2):
        improved = False
        active_tx = tuple(sorted({i for i, _j, _q in best_selected}))
        candidate_rx = tuple(i for i in range(K) if i not in active_tx)
        if not active_tx or not candidate_rx:
            break
        for q_value in target_order:
            q = int(q_value)
            current_without_q = tuple(
                edge for edge in best_selected if edge[2] != q)
            for owner in candidate_rx:
                ranked_tx = sorted(
                    active_tx,
                    key=lambda i: (
                        budget[i] * gain[i, owner, q], -i),
                    reverse=True,
                )
                chosen_tx = tuple(
                    i for i in ranked_tx[:target_pair_limit]
                    if gain[i, owner, q] > 0.0
                )
                if not chosen_tx:
                    continue
                candidate = tuple(sorted(current_without_q + tuple(
                    (int(i), int(owner), q) for i in chosen_tx)))
                if any(
                    sum(edge[1] == j for edge in candidate)
                    > reports_per_receiver
                    for j in range(K)
                ):
                    continue
                fixed_gain = np.zeros((K, Q), dtype=np.float64)
                for i, j, target in candidate:
                    fixed_gain[i, target] = gain[i, j, target]
                result = solve_fixed_structure_maxmin_power_lp(
                    fixed_gain, budget)
                D_q = np.asarray(result.deflection, dtype=np.float64)
                power_tolerance = max(
                    1.0e-12, 1.0e-8 * float(np.max(budget)))
                executable = tuple(
                    edge for edge in candidate
                    if result.power_w[edge[0], edge[2]] > power_tolerance
                )
                score = (
                    float(np.min(D_q)),
                    float(np.dot(priority, np.minimum(D_q, d_floor))),
                    -float(len(executable)),
                )
                if score > best_score:
                    best_score = score
                    best_selected = executable
                    best_power = result.power_w.copy()
                    best_D = D_q.copy()
                    improved = True
                    break
        if not improved:
            break
    return OracleSolution(
        mode="enumerated_role_ceiling_local_only",
        tx_indices=tuple(sorted({i for i, _j, _q in best_selected})),
        rx_indices=tuple(sorted({j for _i, j, _q in best_selected})),
        selected_set=best_selected,
        sensing_power_w=best_power,
        D_q=best_D,
        P_D_q=compute_PD(best_D, P_FA),
    )

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


def repair_invalid_local_only_edges_min_change(
    coefficient: np.ndarray,
    selected_set: Sequence[Tuple[int, int, int]],
    per_uav_sensing_budget_w: np.ndarray,
    *,
    P_FA: float,
    target_pair_limit: int,
    reports_per_receiver: int,
    max_combinations: int = 256,
) -> OracleSolution | None:
    """Repair stale reporting edges without changing roles or target owners.

    The cached graph may become non-executable between two structure updates
    because a bistatic edge leaves the current geometry/DD support.  A full P0
    replan is safe but needlessly changes every target's coordination context.
    This finite-neighbourhood repair instead replaces each invalid ``(i,j,q)``
    by a currently valid ``(i',j,q)`` where ``i'`` already has the transmitter
    role.  Therefore Tx/Rx roles, target owners, edge cardinality, and receiver
    load are invariant.  Every candidate is certified by the exact
    fixed-structure power LP under the current per-UAV RF budgets.

    ``None`` means the invariants cannot be preserved and the caller must fall
    back to a full structure solve.  The bounded Cartesian enumeration is
    exact over the retained local replacement neighbourhood; candidates are
    only truncated when their product would exceed ``max_combinations``.
    """
    gain = np.asarray(coefficient, dtype=np.float64).copy()
    budget = np.asarray(
        per_uav_sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = gain.shape
    if budget.shape != (K,):
        raise ValueError("per_uav_sensing_budget_w must have shape (K,)")
    if (
        np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError("coefficient/budget must be finite and non-negative")
    if target_pair_limit < 1 or reports_per_receiver < 1:
        raise ValueError("pair and receiver limits must be positive")
    if max_combinations < 1:
        raise ValueError("max_combinations must be positive")
    gain[np.arange(K), np.arange(K), :] = 0.0

    selected = tuple(sorted({
        (int(i), int(j), int(q)) for i, j, q in selected_set
    }))
    if not selected:
        return None
    if any(
        not (0 <= i < K and 0 <= j < K and 0 <= q < Q) or i == j
        for i, j, q in selected
    ):
        return None
    tx_roles = tuple(sorted({i for i, _j, _q in selected}))
    rx_roles = tuple(sorted({j for _i, j, _q in selected}))
    if set(tx_roles) & set(rx_roles):
        return None
    owners: dict[int, set[int]] = {}
    for _i, j, q in selected:
        owners.setdefault(q, set()).add(j)
    if any(len(owner_set) != 1 for owner_set in owners.values()):
        return None
    if any(
        sum(edge[2] == q for edge in selected) > target_pair_limit
        for q in range(Q)
    ) or any(
        sum(edge[1] == j for edge in selected) > reports_per_receiver
        for j in range(K)
    ):
        return None

    invalid = tuple(edge for edge in selected if gain[edge] <= 0.0)
    if not invalid:
        return None
    fixed = tuple(edge for edge in selected if gain[edge] > 0.0)
    replacement_sets: list[list[tuple[int, int, int]]] = []
    selected_lookup = set(selected)
    for old_i, owner, q in invalid:
        candidates = [
            (int(i), int(owner), int(q))
            for i in tx_roles
            if i != owner
            and gain[i, owner, q] > 0.0
            and (i, owner, q) not in selected_lookup
        ]
        # Budget-weighted edge ceiling is only a safe pruning/tie-break score;
        # final selection always uses the exact coupled power LP below.
        candidates.sort(
            key=lambda edge: (
                budget[edge[0]] * gain[edge], -edge[0]),
            reverse=True,
        )
        if not candidates:
            return None
        replacement_sets.append(candidates)

    # Keep the local search bounded deterministically.  Truncate the largest
    # lists one element at a time while preserving at least one alternative.
    while int(np.prod([len(values) for values in replacement_sets])) > max_combinations:
        largest = max(
            range(len(replacement_sets)),
            key=lambda index: len(replacement_sets[index]),
        )
        if len(replacement_sets[largest]) <= 1:
            break
        replacement_sets[largest].pop()

    best_score: tuple[float, float, float] | None = None
    best_selected: tuple[tuple[int, int, int], ...] | None = None
    best_power = np.zeros((K, Q), dtype=np.float64)
    best_D = np.zeros(Q, dtype=np.float64)
    for replacements in product(*replacement_sets):
        if len(set(replacements)) != len(replacements):
            continue
        candidate = tuple(sorted(fixed + tuple(replacements)))
        if len(candidate) != len(selected):
            continue
        fixed_gain = np.zeros((K, Q), dtype=np.float64)
        for i, j, q in candidate:
            # The fixed roles/owner guarantee no cross-receiver fusion and at
            # most one coefficient for a transmitter-target power variable.
            fixed_gain[i, q] = gain[i, j, q]
        try:
            result = solve_fixed_structure_maxmin_power_lp(
                fixed_gain, budget)
        except RuntimeError:
            continue
        D_q = np.asarray(result.deflection, dtype=np.float64)
        score = (
            float(np.min(D_q)),
            float(np.mean(compute_PD(D_q, P_FA))),
            float(np.mean(D_q)),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_selected = candidate
            best_power = np.asarray(result.power_w, dtype=np.float64).copy()
            best_D = D_q.copy()

    if best_selected is None:
        return None
    return OracleSolution(
        mode="topology_min_change_local_only",
        tx_indices=tx_roles,
        rx_indices=rx_roles,
        selected_set=best_selected,
        sensing_power_w=best_power,
        D_q=best_D,
        P_D_q=compute_PD(best_D, P_FA),
    )


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


def solve_budget_coupled_local_only_pairs(
    coefficient: np.ndarray,
    per_uav_sensing_budget_w: np.ndarray,
    *,
    P_FA: float,
    p_d_floor: float,
    target_pair_limit: int,
    reports_per_receiver: int,
    target_priority: np.ndarray | None = None,
    time_limit_s: float = 5.0,
    secondary_objective_enabled: bool = True,
) -> OracleSolution:
    """Jointly select single-role local receivers and feasible edge power.

    The old local-only P0 first optimized structure with ``p[i,q]=1`` and
    only afterwards imposed ``sum_q p[i,q] <= b[i]``.  That relaxation is not
    scale safe: one transmitter can appear to spend one full unit on every
    target.  This MILP keeps the physical per-UAV budget in the structural
    problem itself.

    ``flow[i,j,q]`` is the sensing power sent by transmitter ``i`` for target
    ``q`` and reported to its unique local detector owner ``j``.  Because one
    owner is selected per target, the flow is exactly the physical
    transmitter-target power rather than duplicated receiver evidence.

    The objective is lexicographic when ``secondary_objective_enabled`` is
    true.  Stage 1 maximizes the worst target Deflection.  Stage 2 preserves
    that optimum (within numerical tolerance) and maximizes priority-weighted
    Deflection capped at ``p_d_floor``.  Disabling Stage 2 is a real-time
    variant that retains the primary max-min model and every physical
    constraint.  If the time limit is reached, a candidate is accepted only
    after an independent primal-feasibility/integrality check; in that case it
    is an executable incumbent rather than a global-optimality certificate.
    Every returned solution is feasible for the downstream fixed-structure
    power LP, whose optimum cannot be worse than the returned Stage-1 value.
    """
    gain = np.asarray(coefficient, dtype=np.float64).copy()
    budget = np.asarray(
        per_uav_sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = gain.shape
    if budget.shape != (K,):
        raise ValueError("per_uav_sensing_budget_w must have shape (K,)")
    if (
        np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError("coefficient/budget must be finite and non-negative")
    if target_pair_limit < 1 or reports_per_receiver < 1:
        raise ValueError("pair and receiver limits must be positive")
    if not np.isfinite(time_limit_s) or time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be finite and positive")
    diagonal = np.arange(K)
    gain[diagonal, diagonal, :] = 0.0

    edges = [
        (i, j, q)
        for i in range(K)
        for j in range(K)
        if i != j and budget[i] > 0.0
        for q in range(Q)
        if gain[i, j, q] > 0.0
    ]
    E = len(edges)
    if E == 0 or not np.any(budget > 0.0):
        return OracleSolution(
            mode="budget_coupled_local_only",
            tx_indices=(), rx_indices=(), selected_set=(),
            sensing_power_w=np.zeros((K, Q), dtype=np.float64),
            D_q=np.zeros(Q, dtype=np.float64),
            P_D_q=compute_PD(np.zeros(Q, dtype=np.float64), P_FA),
        )

    # Variables: admitted edge x, edge power f, TX role r, target owner o,
    # floor-capped target value z, and uncapped worst Deflection y.
    o_x = 0
    o_f = o_x + E
    o_r = o_f + E
    o_owner = o_r + K
    o_z = o_owner + K * Q
    o_y = o_z + Q
    n = o_y + 1

    def owner_index(j: int, q: int) -> int:
        return o_owner + j * Q + q

    q_fa = float(Q_inverse(np.asarray(float(P_FA))))
    q_pd = float(Q_inverse(np.asarray(float(p_d_floor))))
    d_floor = max((q_fa - q_pd) ** 2, 1.0e-12)
    relaxed_ceiling = np.asarray([
        sum(
            budget[i] * max(gain[i, j, q] for j in range(K) if j != i)
            for i in range(K)
        )
        for q in range(Q)
    ], dtype=np.float64)
    y_upper = max(float(np.max(relaxed_ceiling)), d_floor, 1.0e-12)

    lower = np.zeros(n, dtype=np.float64)
    upper = np.ones(n, dtype=np.float64)
    for e, (i, _j, _q) in enumerate(edges):
        upper[o_f + e] = budget[i]
    upper[o_z:o_z + Q] = d_floor
    upper[o_y] = y_upper
    integrality = np.zeros(n, dtype=np.int32)
    integrality[o_x:o_f] = 1
    integrality[o_r:o_owner] = 1
    integrality[o_owner:o_z] = 1

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add(row: np.ndarray, lb: float = -np.inf, ub: float = np.inf) -> None:
        rows.append(row)
        lbs.append(float(lb))
        ubs.append(float(ub))

    # Per-UAV physical sensing budget and edge admission/power coupling.
    for i in range(K):
        row = np.zeros(n, dtype=np.float64)
        for e, (tx, _j, _q) in enumerate(edges):
            if tx == i:
                row[o_f + e] = 1.0
        add(row, ub=budget[i])
    for e, (i, j, q) in enumerate(edges):
        row = np.zeros(n, dtype=np.float64)
        row[o_f + e] = 1.0
        row[o_x + e] = -budget[i]
        add(row, ub=0.0)

        # x_ijq <= r_i, x_ijq <= 1-r_j, x_ijq <= owner_jq.
        row = np.zeros(n, dtype=np.float64)
        row[o_x + e] = 1.0
        row[o_r + i] = -1.0
        add(row, ub=0.0)
        row = np.zeros(n, dtype=np.float64)
        row[o_x + e] = 1.0
        row[o_r + j] = 1.0
        add(row, ub=1.0)
        row = np.zeros(n, dtype=np.float64)
        row[o_x + e] = 1.0
        row[owner_index(j, q)] = -1.0
        add(row, ub=0.0)

    # One local detector owner per target, reporting cardinality, and receiver
    # message capacity.  Owners without a positive-power edge disappear when
    # the returned support is extracted.
    for q in range(Q):
        row = np.zeros(n, dtype=np.float64)
        for j in range(K):
            row[owner_index(j, q)] = 1.0
        add(row, lb=1.0, ub=1.0)

        row = np.zeros(n, dtype=np.float64)
        for e, (_i, _j, target) in enumerate(edges):
            if target == q:
                row[o_x + e] = 1.0
        add(row, ub=float(target_pair_limit))
    for j in range(K):
        row = np.zeros(n, dtype=np.float64)
        for e, (_i, rx, _q) in enumerate(edges):
            if rx == j:
                row[o_x + e] = 1.0
        add(row, ub=float(reports_per_receiver))

    # y <= D_q and z_q <= D_q under the unique-owner local detector.
    for q in range(Q):
        row_y = np.zeros(n, dtype=np.float64)
        row_y[o_y] = 1.0
        row_z = np.zeros(n, dtype=np.float64)
        row_z[o_z + q] = 1.0
        for e, (i, j, target) in enumerate(edges):
            if target == q:
                value = gain[i, j, q]
                row_y[o_f + e] = -value
                row_z[o_f + e] = -value
        add(row_y, ub=0.0)
        add(row_z, ub=0.0)

    constraint_matrix = csr_matrix(np.stack(rows))
    constraints = LinearConstraint(
        constraint_matrix,
        lb=np.asarray(lbs, dtype=np.float64),
        ub=np.asarray(ubs, dtype=np.float64),
    )
    bounds = Bounds(lower, upper)

    def verified_incumbent(
        result: object,
        matrix: object,
        constraint_lb: np.ndarray,
        constraint_ub: np.ndarray,
    ) -> np.ndarray | None:
        """Return only a numerically verified MILP primal incumbent."""
        candidate = getattr(result, "x", None)
        if candidate is None:
            return None
        vector = np.asarray(candidate, dtype=np.float64)
        bound_tolerance = 2.0e-7 * (
            1.0 + np.maximum(np.abs(lower), np.abs(upper)))
        if vector.shape != (n,) or np.any(~np.isfinite(vector)):
            return None
        if (
            np.any(vector < lower - bound_tolerance)
            or np.any(vector > upper + bound_tolerance)
        ):
            return None
        integer_mask = integrality != 0
        if np.any(np.abs(vector[integer_mask] - np.rint(
                vector[integer_mask])) > 1.0e-5):
            return None
        activity = np.asarray(matrix @ vector, dtype=np.float64).reshape(-1)
        absolute_activity = np.asarray(
            abs(matrix) @ np.abs(vector), dtype=np.float64).reshape(-1)
        row_scale = 1.0 + absolute_activity
        finite_lb = np.isfinite(constraint_lb)
        finite_ub = np.isfinite(constraint_ub)
        row_scale[finite_lb] += np.abs(constraint_lb[finite_lb])
        row_scale[finite_ub] += np.abs(constraint_ub[finite_ub])
        row_tolerance = 2.0e-7 * row_scale
        if (
            np.any(activity < constraint_lb - row_tolerance)
            or np.any(activity > constraint_ub + row_tolerance)
        ):
            return None
        return vector

    objective = np.zeros(n, dtype=np.float64)
    objective[o_y] = -1.0
    stage1 = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"time_limit": float(time_limit_s)},
    )
    stage1_vector = (
        np.asarray(stage1.x, dtype=np.float64)
        if stage1.success and stage1.x is not None
        else verified_incumbent(
            stage1,
            constraint_matrix,
            np.asarray(lbs, dtype=np.float64),
            np.asarray(ubs, dtype=np.float64),
        )
    )
    if stage1_vector is None:
        raise RuntimeError(
            "budget-coupled local-only Stage-1 MILP failed: "
            f"{stage1.message}")

    vector = stage1_vector
    if secondary_objective_enabled:
        y_star = float(stage1_vector[o_y])
        primary_tolerance = max(1.0e-9, 1.0e-7 * max(1.0, abs(y_star)))
        primary_row = np.zeros(n, dtype=np.float64)
        primary_row[o_y] = 1.0
        stage2_constraints = LinearConstraint(
            sparse_vstack([
                constraint_matrix,
                csr_matrix(primary_row.reshape(1, -1)),
            ], format="csr"),
            lb=np.concatenate([
                np.asarray(lbs, dtype=np.float64),
                np.asarray([max(0.0, y_star - primary_tolerance)]),
            ]),
            ub=np.concatenate([
                np.asarray(ubs, dtype=np.float64),
                np.asarray([np.inf]),
            ]),
        )
        priority = (
            np.ones(Q, dtype=np.float64)
            if target_priority is None
            else np.asarray(target_priority, dtype=np.float64).reshape(-1)
        )
        if priority.shape != (Q,) or np.any(~np.isfinite(priority)):
            raise ValueError("target_priority must be a finite Q-vector")
        priority = np.maximum(priority, 1.0e-9)
        priority /= float(np.sum(priority))
        objective2 = np.zeros(n, dtype=np.float64)
        objective2[o_z:o_z + Q] = -priority
        stage2 = milp(
            objective2,
            integrality=integrality,
            bounds=bounds,
            constraints=stage2_constraints,
            options={"time_limit": float(time_limit_s)},
        )
        stage2_vector = (
            np.asarray(stage2.x, dtype=np.float64)
            if stage2.success and stage2.x is not None
            else verified_incumbent(
                stage2,
                stage2_constraints.A,
                np.asarray(stage2_constraints.lb, dtype=np.float64),
                np.asarray(stage2_constraints.ub, dtype=np.float64),
            )
        )
        if stage2_vector is not None:
            vector = stage2_vector

    admitted = np.asarray(vector[o_x:o_x + E], dtype=np.float64) >= 0.5
    edge_flow = np.where(
        admitted,
        np.maximum(np.asarray(vector[o_f:o_f + E], dtype=np.float64), 0.0),
        0.0,
    )
    # HiGHS may return a continuous incumbent a few feasibility-tolerance
    # units above a small physical budget (here budgets are ~2.5e-2 W).  A
    # radial projection per transmitter only decreases non-negative flows, so
    # every admission/capacity upper constraint remains feasible while the RF
    # budget becomes exact.  Recompute D below from the projected executable
    # flow; never report the unprojected solver objective as achieved physics.
    for i in range(K):
        tx_edge = np.asarray(
            [edge[0] == i for edge in edges], dtype=bool)
        total_flow = float(np.sum(edge_flow[tx_edge]))
        if total_flow > budget[i] and total_flow > 0.0:
            edge_flow[tx_edge] *= budget[i] / total_flow
    flow_tolerance = max(1.0e-12, 1.0e-8 * float(np.max(budget)))
    selected = tuple(sorted(
        edges[e] for e in range(E) if edge_flow[e] > flow_tolerance
    ))
    power = np.zeros((K, Q), dtype=np.float64)
    D_q = np.zeros(Q, dtype=np.float64)
    for e, (i, j, q) in enumerate(edges):
        flow = max(float(edge_flow[e]), 0.0)
        power[i, q] += flow
        D_q[q] += gain[i, j, q] * flow
    tx_indices = tuple(sorted({i for i, _j, _q in selected}))
    rx_indices = tuple(sorted({j for _i, j, _q in selected}))
    if set(tx_indices) & set(rx_indices):
        raise RuntimeError("budget-coupled solution violates single-role rule")
    if np.any(np.sum(power, axis=1) > budget + 1.0e-8):
        raise RuntimeError("budget-coupled solution violates sensing budget")
    for q in range(Q):
        target_edges = [edge for edge in selected if edge[2] == q]
        if len(target_edges) > target_pair_limit:
            raise RuntimeError("budget-coupled solution violates pair limit")
        if len({edge[1] for edge in target_edges}) > 1:
            raise RuntimeError("budget-coupled solution violates unique owner")
    for j in range(K):
        if sum(edge[1] == j for edge in selected) > reports_per_receiver:
            raise RuntimeError(
                "budget-coupled solution violates receiver capacity")
    return OracleSolution(
        mode="budget_coupled_local_only",
        tx_indices=tx_indices,
        rx_indices=rx_indices,
        selected_set=selected,
        sensing_power_w=power,
        D_q=D_q,
        P_D_q=compute_PD(D_q, P_FA),
    )


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
    per_uav_sensing_budget_w: np.ndarray | None = None,
    joint_time_limit_s: float = 5.0,
    joint_secondary_objective_enabled: bool = True,
    joint_solver_mode: str = "milp",
    task_qos_floors: Sequence[float] | None = None,
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
        if per_uav_sensing_budget_w is not None:
            coefficient = unit_deflection_tensor(
                valid,
                num_uavs=k_count,
                num_targets=q_count,
            )
            normalized_solver = str(joint_solver_mode).strip().lower()
            if normalized_solver in {
                    "enumerated_role_ceiling",
                    "satisficing_legacy_then_enumerated"}:
                solution = None
                if normalized_solver == "satisficing_legacy_then_enumerated":
                    # Cheap reconstruction of the observed legacy large-scale
                    # mode: one transmitter serves all targets, while each
                    # target chooses its best capacity-available local owner.
                    # This is only a satisficing retention candidate and is
                    # accepted below solely after the actual-budget LP clears
                    # the complete task gate.
                    priority = (
                        np.ones(q_count, dtype=np.float64)
                        if target_priority is None
                        else np.asarray(
                            target_priority, dtype=np.float64).reshape(-1)
                    )
                    priority = np.maximum(priority, 1.0e-9)
                    priority /= float(np.sum(priority))
                    q_fa = float(Q_inverse(np.asarray(float(p_fa))))
                    q_pd = float(Q_inverse(np.asarray(float(p_d_floor))))
                    d_floor = max((q_fa - q_pd) ** 2, 1.0e-12)
                    legacy_selected = tuple()
                    legacy_unit_score = None
                    for tx in range(k_count):
                        if per_uav_sensing_budget_w[tx] <= 0.0:
                            continue
                        receiver_load = np.zeros(k_count, dtype=np.int64)
                        candidate = []
                        unit_D = np.zeros(q_count, dtype=np.float64)
                        feasible = True
                        for q in range(q_count):
                            owners = sorted(
                                (j for j in range(k_count) if j != tx),
                                key=lambda j: (coefficient[tx, j, q], -j),
                                reverse=True,
                            )
                            owner = next((
                                j for j in owners
                                if receiver_load[j] < reports_per_receiver
                                and coefficient[tx, j, q] > 0.0
                            ), None)
                            if owner is None:
                                feasible = False
                                break
                            receiver_load[owner] += 1
                            candidate.append((tx, int(owner), q))
                            unit_D[q] = coefficient[tx, owner, q]
                        if not feasible:
                            continue
                        score = (
                            float(np.min(unit_D)),
                            float(np.dot(
                                priority, np.minimum(unit_D, d_floor))),
                            -float(tx),
                        )
                        if legacy_unit_score is None or score > legacy_unit_score:
                            legacy_unit_score = score
                            legacy_selected = tuple(sorted(candidate))
                    if legacy_selected:
                        legacy_gain = np.zeros(
                            (k_count, q_count), dtype=np.float64)
                        for i, j, q in legacy_selected:
                            legacy_gain[i, q] = coefficient[i, j, q]
                        legacy_power = solve_fixed_structure_maxmin_power_lp(
                            legacy_gain, per_uav_sensing_budget_w)
                        legacy_pd = compute_PD(
                            legacy_power.deflection, float(p_fa))
                        floors = (
                            (float(p_d_floor), -np.inf, -np.inf, 3)
                            if task_qos_floors is None
                            else tuple(task_qos_floors)
                        )
                        if len(floors) != 4:
                            raise ValueError(
                                "task_qos_floors must contain "
                                "[worst, weak-k, steady, k]")
                        worst_floor = float(floors[0])
                        weak_floor = float(floors[1])
                        steady_floor = float(floors[2])
                        weak_k = int(floors[3])
                        ordered_pd = np.sort(legacy_pd)
                        weak_pd = float(np.mean(
                            ordered_pd[:max(1, min(weak_k, q_count))]))
                        legacy_task_feasible = bool(
                            float(np.min(legacy_pd)) >= worst_floor
                            and weak_pd >= weak_floor
                            and float(np.mean(legacy_pd)) >= steady_floor
                        )
                        if legacy_task_feasible:
                            solution = OracleSolution(
                                mode="satisficing_legacy_local_only",
                                tx_indices=tuple(sorted({
                                    i for i, _j, _q in legacy_selected})),
                                rx_indices=tuple(sorted({
                                    j for _i, j, _q in legacy_selected})),
                                selected_set=legacy_selected,
                                sensing_power_w=legacy_power.power_w.copy(),
                                D_q=legacy_power.deflection.copy(),
                                P_D_q=legacy_pd,
                            )
                if solution is None:
                    solution = solve_enumerated_role_ceiling_local_pairs(
                        coefficient,
                        per_uav_sensing_budget_w,
                        P_FA=float(p_fa),
                        p_d_floor=float(p_d_floor),
                        target_pair_limit=int(target_pair_limit),
                        reports_per_receiver=int(reports_per_receiver),
                        target_priority=target_priority,
                    )
            elif normalized_solver == "milp":
                solution = solve_budget_coupled_local_only_pairs(
                    coefficient,
                    per_uav_sensing_budget_w,
                    P_FA=float(p_fa),
                    p_d_floor=float(p_d_floor),
                    target_pair_limit=int(target_pair_limit),
                    reports_per_receiver=int(reports_per_receiver),
                    target_priority=target_priority,
                    time_limit_s=float(joint_time_limit_s),
                    secondary_objective_enabled=bool(
                        joint_secondary_objective_enabled),
                )
            else:
                raise ValueError(
                    "joint_solver_mode must be 'milp' or "
                    "'enumerated_role_ceiling' or "
                    "'satisficing_legacy_then_enumerated'")
            return solution.selected_set, solution.D_q.copy()
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
    sensing_power_cap_w: float | None = None,
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
    if sensing_power_cap_w is not None:
        cap = float(sensing_power_cap_w)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("sensing power cap must be finite and positive")
        sensing_budget = min(sensing_budget, cap)
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
    # Audit 2026-08-17: for num_uavs == 1 (or an empty role partition) the
    # partition loop is empty and best stays None -> assert crash.  Degrade
    # gracefully to an empty degenerate solution instead.
    if best is None:
        # K==1 (or an empty role partition with no feasible Tx/Rx split).
        # Audit 2026-08-25: `mode` is only bound inside the partition loop, so
        # referencing it here previously raised NameError; and the degenerate
        # P_D must follow the module's own "no deflection => P_D = P_FA"
        # convention (compute_PD(zeros, P_FA)) instead of a hard-coded 0.0.
        q_count = coefficient.shape[2]
        best = OracleSolution(
            mode="degenerate",
            tx_indices=(),
            rx_indices=(),
            selected_set=(),
            sensing_power_w=np.zeros(
                (coefficient.shape[0], q_count), dtype=np.float64),
            D_q=np.zeros(q_count, dtype=np.float64),
            P_D_q=compute_PD(np.zeros(q_count, dtype=np.float64), P_FA),
        )
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
    if best is None:
        # Audit 2026-08-25: same K==1 / empty-partition degrade as the joint
        # variant (previously an unconditional `assert best is not None`).
        best = OracleSolution(
            mode="pair_only_degenerate",
            tx_indices=(),
            rx_indices=(),
            selected_set=(),
            sensing_power_w=np.zeros((k_count, q_count), dtype=np.float64),
            D_q=np.zeros(q_count, dtype=np.float64),
            P_D_q=compute_PD(np.zeros(q_count, dtype=np.float64), P_FA),
        )
    return best


def solve_power_only_oracle(
    coefficient: np.ndarray,
    selected_set: Sequence[Tuple[int, int, int]],
    *,
    P_FA: float,
    total_power_w: float = 1.0,
    communication_reserve_w: float = 0.0,
    sensing_power_cap_w: float | None = None,
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
    if sensing_power_cap_w is not None:
        cap = float(sensing_power_cap_w)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("sensing power cap must be finite and positive")
        sensing_budget = min(sensing_budget, cap)
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
