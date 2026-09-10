"""Finite-round learning-free coordination on sparse directed hyperedges.

The protocol is an unfolded deterministic auction.  Candidate publication is
external and must already respect the physical Token inbox.  Every update can
be implemented by receiver-local bid aggregation, target-owner max consensus,
and node-local role-price updates; the array implementation is only a compact
simulator of those messages and never calls the centralized MILP.
"""

# ----------------------------------------------------------------------
# AUDIT/RESEARCH-ONLY MODULE (2026-08-16 audit remediation)
#
# This module is consumed only by tools/ audit scripts and tests. It is
# NOT part of the deployment execution path (env_core / trainer) and its
# results must not be described as deployed behaviour. It exists to keep
# a specific research question reproducible; see
# docs/EXPERIMENT_LOG.md for the associated gate.
# ----------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Tuple

import numpy as np


Hyperedge = Tuple[int, int, int]


@dataclass(frozen=True)
class FiniteRoundState:
    target_price: np.ndarray
    tx_role_price: np.ndarray
    rx_role_price: np.ndarray
    owner: np.ndarray
    owner_hold: np.ndarray
    selected: Tuple[Hyperedge, ...]
    initialized: bool


@dataclass(frozen=True)
class FiniteRoundResult:
    selected: Tuple[Hyperedge, ...]
    state: FiniteRoundState
    rounds_used: int
    convergence_round: int
    bootstrap_required: bool
    target_value: np.ndarray
    role_conflicts: int
    owner_changes: int


@dataclass(frozen=True)
class ReplicatedPlanAlternative:
    selected: Tuple[Hyperedge, ...]
    role: np.ndarray
    owner: np.ndarray
    target_value: np.ndarray
    weighted_worst: float
    weighted_sum: float


@dataclass(frozen=True)
class ReplicatedPlanCertificate:
    selected: Tuple[Hyperedge, ...]
    role: np.ndarray
    owner: np.ndarray
    target_value: np.ndarray
    alternatives: Tuple[ReplicatedPlanAlternative, ...]


def initial_finite_round_state(
    num_uavs: int,
    num_targets: int,
    *,
    initial_target_price: float = 1.0,
) -> FiniteRoundState:
    K, Q = int(num_uavs), int(num_targets)
    if K < 2 or Q < 1:
        raise ValueError("finite-round protocol requires K>=2 and Q>=1")
    return FiniteRoundState(
        target_price=np.full(Q, float(initial_target_price), dtype=np.float64),
        tx_role_price=np.zeros(K, dtype=np.float64),
        rx_role_price=np.zeros(K, dtype=np.float64),
        owner=np.full(Q, -1, dtype=np.int64),
        owner_hold=np.zeros(Q, dtype=np.int64),
        selected=tuple(),
        initialized=False,
    )


def _normalized_candidate_value(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    value = np.asarray(edge_value, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if value.shape != mask.shape or value.ndim != 3:
        raise ValueError("edge value/mask must have shape (K,K,Q)")
    if value.shape[0] != value.shape[1]:
        raise ValueError("edge value must have equal UAV axes")
    if not np.all(np.isfinite(value)):
        raise ValueError("edge values must be finite")
    normalized = np.zeros_like(value)
    positive = np.where(mask, np.maximum(value, 0.0), 0.0)
    for target in range(value.shape[2]):
        scale = float(np.max(positive[:, :, target]))
        if scale > 1.0e-12:
            normalized[:, :, target] = positive[:, :, target] / scale
    diagonal = np.arange(value.shape[0])
    normalized[diagonal, diagonal, :] = 0.0
    return normalized


def _role_from_selected(
    selected: Tuple[Hyperedge, ...],
    num_uavs: int,
) -> np.ndarray:
    # -1 idle, 0 receiver, 1 transmitter.
    role = np.full(int(num_uavs), -1, dtype=np.int8)
    for tx, rx, _ in selected:
        if role[tx] == 0 or role[rx] == 1:
            raise RuntimeError("selected edges violate the single-role invariant")
        role[tx] = 1
        role[rx] = 0
    return role


def negotiate_finite_round_hyperedges(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    previous: FiniteRoundState,
    *,
    rounds: int,
    target_pair_limit: int,
    owner_hold_rounds: int,
    target_price_step: float,
    role_price_step: float,
    target_proxy_floor: float,
    price_max: float,
    communication_cost: float = 0.0,
    switch_cost: float = 0.0,
    owner_hold_bonus: float = 1.0,
) -> FiniteRoundResult:
    """Negotiate a feasible sparse structure without a central repair step."""
    mask = np.asarray(candidate_mask, dtype=bool)
    normalized = _normalized_candidate_value(edge_value, mask)
    K, _, Q = normalized.shape
    if previous.target_price.shape != (Q,):
        raise ValueError("previous state target cardinality mismatch")
    if previous.tx_role_price.shape != (K,):
        raise ValueError("previous state UAV cardinality mismatch")
    if not np.any(mask & (normalized > 0.0)):
        return FiniteRoundResult(
            selected=tuple(),
            state=FiniteRoundState(
                target_price=previous.target_price.copy(),
                tx_role_price=previous.tx_role_price.copy(),
                rx_role_price=previous.rx_role_price.copy(),
                owner=previous.owner.copy(),
                owner_hold=np.maximum(previous.owner_hold - 1, 0),
                selected=tuple(),
                initialized=previous.initialized,
            ),
            rounds_used=0,
            convergence_round=0,
            bootstrap_required=not previous.initialized,
            target_value=np.zeros(Q, dtype=np.float64),
            role_conflicts=0,
            owner_changes=0,
        )

    target_price = previous.target_price.astype(np.float64, copy=True)
    tx_price = previous.tx_role_price.astype(np.float64, copy=True)
    rx_price = previous.rx_role_price.astype(np.float64, copy=True)
    owner = previous.owner.astype(np.int64, copy=True)
    hold = previous.owner_hold.astype(np.int64, copy=True)
    previous_edges = set(previous.selected)
    previous_role = _role_from_selected(previous.selected, K)
    committed_role = previous_role.copy()
    selected: Tuple[Hyperedge, ...] = tuple()
    last_selected: Tuple[Hyperedge, ...] | None = None
    convergence_round = max(1, int(rounds))
    conflict_count = 0
    owner_changes = 0
    target_value = np.zeros(Q, dtype=np.float64)
    limit = max(1, int(target_pair_limit))

    for round_index in range(1, max(1, int(rounds)) + 1):
        net = np.full_like(normalized, -np.inf)
        for tx, rx, target in np.argwhere(mask & (normalized > 0.0)):
            if committed_role[tx] == 0 or committed_role[rx] == 1:
                continue
            edge = (int(tx), int(rx), int(target))
            switching = 0.0 if edge in previous_edges else float(switch_cost)
            net[edge] = (
                target_price[target] * normalized[edge]
                - tx_price[tx] - rx_price[rx]
                - float(communication_cost) - switching
            )

        tentative: list[Hyperedge] = []
        round_owner = np.full(Q, -1, dtype=np.int64)
        for target in range(Q):
            receiver_options = []
            for rx in range(K):
                incoming = [
                    (float(net[tx, rx, target]), int(tx))
                    for tx in range(K)
                    if tx != rx and np.isfinite(net[tx, rx, target])
                    and net[tx, rx, target] > 0.0
                ]
                incoming.sort(key=lambda item: (-item[0], item[1]))
                retained = incoming[:limit]
                if not retained:
                    continue
                score = float(sum(item[0] for item in retained))
                if owner[target] == rx and hold[target] > 0:
                    score += float(owner_hold_bonus)
                receiver_options.append((score, rx, retained))
            receiver_options.sort(key=lambda item: (-item[0], item[1]))
            if receiver_options:
                _, rx, retained = receiver_options[0]
                round_owner[target] = int(rx)
                tentative.extend(
                    (int(tx), int(rx), int(target)) for _, tx in retained)

        tx_support = np.zeros(K, dtype=np.float64)
        rx_support = np.zeros(K, dtype=np.float64)
        for edge in tentative:
            tx, rx, _ = edge
            score = max(float(net[edge]), 0.0)
            tx_support[tx] += score
            rx_support[rx] += score
        role = np.full(K, -1, dtype=np.int8)
        conflicts = (tx_support > 0.0) & (rx_support > 0.0)
        conflict_count += int(np.sum(conflicts))
        for node in range(K):
            if tx_support[node] <= 0.0 and rx_support[node] <= 0.0:
                continue
            tx_score = tx_support[node]
            rx_score = rx_support[node]
            if previous_role[node] == 1:
                tx_score += float(switch_cost)
            elif previous_role[node] == 0:
                rx_score += float(switch_cost)
            role[node] = 1 if tx_score >= rx_score else 0
            if conflicts[node]:
                if role[node] == 1:
                    rx_price[node] += float(role_price_step)
                    tx_price[node] = max(0.0, tx_price[node]
                                         - 0.5 * float(role_price_step))
                else:
                    tx_price[node] += float(role_price_step)
                    rx_price[node] = max(0.0, rx_price[node]
                                         - 0.5 * float(role_price_step))
            else:
                tx_price[node] *= 0.9
                rx_price[node] *= 0.9
        # The first conflict exchange is a role reservation.  Later rounds
        # re-run target-owner and Tx acceptance inside this hard feasible
        # partition, so rejected conflicts are replaced rather than proposed
        # and deleted repeatedly. Existing active roles persist across solves.
        newly_committed = role >= 0
        committed_role[newly_committed] = role[newly_committed]
        tx_price = np.clip(tx_price, 0.0, float(price_max))
        rx_price = np.clip(rx_price, 0.0, float(price_max))

        selected = tuple(sorted(
            edge for edge in tentative
            if role[edge[0]] == 1 and role[edge[1]] == 0))
        _role_from_selected(selected, K)
        target_value = np.zeros(Q, dtype=np.float64)
        selected_owner = np.full(Q, -1, dtype=np.int64)
        for edge in selected:
            target_value[edge[2]] += normalized[edge]
            selected_owner[edge[2]] = edge[1]
        target_price = np.clip(
            target_price + float(target_price_step)
            * (float(target_proxy_floor) - target_value),
            0.0,
            float(price_max),
        )
        owner_changes += int(np.sum(
            (selected_owner >= 0) & (owner >= 0)
            & (selected_owner != owner)))
        owner = np.where(selected_owner >= 0, selected_owner, owner)

        if last_selected == selected:
            convergence_round = min(convergence_round, round_index)
        last_selected = selected

    new_hold = np.zeros(Q, dtype=np.int64)
    for target in range(Q):
        if owner[target] < 0:
            continue
        if owner[target] == previous.owner[target] and previous.owner_hold[target] > 0:
            new_hold[target] = previous.owner_hold[target] - 1
        else:
            new_hold[target] = max(0, int(owner_hold_rounds) - 1)
    state = FiniteRoundState(
        target_price=target_price,
        tx_role_price=tx_price,
        rx_role_price=rx_price,
        owner=owner,
        owner_hold=new_hold,
        selected=selected,
        initialized=True,
    )
    return FiniteRoundResult(
        selected=selected,
        state=state,
        rounds_used=max(1, int(rounds)),
        convergence_round=convergence_round,
        bootstrap_required=False,
        target_value=target_value,
        role_conflicts=conflict_count,
        owner_changes=owner_changes,
    )


def _solve_replicated_candidate_graph_certificate(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    target_price: np.ndarray | None = None,
    target_proxy_floor: float = 1.0,
    max_alternatives: int = 8,
    min_worst_ratio: float = 0.95,
    min_sum_ratio: float = 0.95,
) -> ReplicatedPlanCertificate:
    """Solve a common graph and retain near-equivalent feasible certificates.

    This is not a centralized repair: once finite-round flooding gives every
    UAV the same candidate Token table, every UAV runs this identical small-K
    dynamic program and obtains the same feasible structure. The returned
    alternatives are an offline teacher artifact, not a deployable search.
    """
    value = np.asarray(edge_value, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if value.shape != mask.shape or value.ndim != 3:
        raise ValueError("edge value/mask must have shape (K,K,Q)")
    K, K2, Q = value.shape
    if K != K2:
        raise ValueError("edge value must have equal UAV axes")
    priority = (
        np.ones(Q, dtype=np.float64)
        if target_price is None
        else np.asarray(target_price, dtype=np.float64)
    )
    if priority.shape != (Q,) or not np.all(np.isfinite(priority)):
        raise ValueError("target_price must have shape (Q,)")
    priority = np.maximum(priority, 1.0e-9)
    positive = np.where(mask, np.maximum(value, 0.0), 0.0)
    diagonal = np.arange(K)
    positive[diagonal, diagonal, :] = 0.0
    limit = max(1, int(target_pair_limit))
    receiver_capacity = max(1, int(reports_per_receiver))
    floor = max(float(target_proxy_floor), 1.0e-12)

    def plan_key(
        target_values: tuple[float, ...],
        edges: tuple[Hyperedge, ...],
    ) -> tuple[object, ...]:
        values_array = np.asarray(target_values)
        weighted = values_array * priority[Q - len(values_array):]
        capped = np.minimum(weighted / floor, 1.0)
        return (
            float(np.min(weighted)) if len(weighted) else 0.0,
            float(np.sum(capped)),
            float(np.sum(weighted)),
            -len(edges),
            tuple(-item for edge in edges for item in edge),
        )

    best_key: tuple[object, ...] | None = None
    best_edges: Tuple[Hyperedge, ...] = tuple()
    plans: list[tuple[
        tuple[object, ...],
        Tuple[Hyperedge, ...],
        tuple[float, ...],
        np.ndarray,
    ]] = []
    for role_tuple in product((False, True), repeat=K):
        role = np.asarray(role_tuple, dtype=bool)  # True=Tx, False=Rx.
        if not np.any(role) or np.all(role):
            continue
        tx_nodes = np.flatnonzero(role)
        rx_nodes = np.flatnonzero(~role)
        target_options: list[list[tuple[int, tuple[tuple[float, Hyperedge], ...]]]] = []
        for target in range(Q):
            options = []
            for rx in rx_nodes:
                incoming = [
                    (float(positive[tx, rx, target]),
                     (int(tx), int(rx), int(target)))
                    for tx in tx_nodes
                    if positive[tx, rx, target] > 0.0
                ]
                incoming.sort(key=lambda item: (-item[0], item[1]))
                if incoming:
                    options.append((int(rx), tuple(incoming[:limit])))
            target_options.append(options)

        @lru_cache(maxsize=None)
        def allocate(
            target: int,
            capacities: tuple[int, ...],
        ) -> tuple[tuple[float, ...], tuple[Hyperedge, ...]]:
            if target >= Q:
                return tuple(), tuple()
            candidates = []
            # An empty target remains an explicit option so infeasible role
            # partitions are scored rather than silently discarded.
            suffix_values, suffix_edges = allocate(target + 1, capacities)
            candidates.append(((0.0,) + suffix_values, suffix_edges))
            for rx, incoming in target_options[target]:
                available = capacities[rx]
                for count in range(1, min(available, len(incoming)) + 1):
                    retained = incoming[:count]
                    updated = list(capacities)
                    updated[rx] -= count
                    suffix_values, suffix_edges = allocate(
                        target + 1, tuple(updated))
                    candidates.append((
                        (float(sum(item[0] for item in retained)),)
                        + suffix_values,
                        tuple(edge for _, edge in retained) + suffix_edges,
                    ))
            return max(
                candidates,
                key=lambda item: plan_key(item[0], tuple(sorted(item[1]))),
            )

        values, edges = allocate(
            0, tuple([receiver_capacity] * K))
        edges = tuple(sorted(edges))
        key = plan_key(values, edges)
        plans.append((key, edges, values, role.copy()))
        if best_key is None or key > best_key:
            best_key = key
            best_edges = edges
    _role_from_selected(best_edges, K)
    if not plans:
        raise RuntimeError("replicated solver found no role partition")
    plans.sort(key=lambda item: item[0], reverse=True)

    def alternative_from_plan(
        plan: tuple[
            tuple[object, ...],
            Tuple[Hyperedge, ...],
            tuple[float, ...],
            np.ndarray,
        ],
    ) -> ReplicatedPlanAlternative:
        _, edges, values, role_mask = plan
        owner = np.full(Q, -1, dtype=np.int64)
        for _, receiver, target in edges:
            owner[target] = receiver
        target_value = np.asarray(values, dtype=np.float64)
        weighted = target_value * priority
        return ReplicatedPlanAlternative(
            selected=edges,
            role=np.where(role_mask, 1, 0).astype(np.int8),
            owner=owner,
            target_value=target_value,
            weighted_worst=float(np.min(weighted)) if Q else 0.0,
            weighted_sum=float(np.sum(weighted)),
        )

    best = alternative_from_plan(plans[0])
    retained = []
    worst_floor = float(min_worst_ratio) * best.weighted_worst
    sum_floor = float(min_sum_ratio) * best.weighted_sum
    for plan in plans:
        candidate = alternative_from_plan(plan)
        if candidate.weighted_worst + 1.0e-12 < worst_floor:
            continue
        if candidate.weighted_sum + 1.0e-12 < sum_floor:
            continue
        retained.append(candidate)
        if len(retained) >= max(1, int(max_alternatives)):
            break
    return ReplicatedPlanCertificate(
        selected=best.selected,
        role=best.role,
        owner=best.owner,
        target_value=best.target_value,
        alternatives=tuple(retained),
    )


def solve_replicated_candidate_graph(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    target_price: np.ndarray | None = None,
    target_proxy_floor: float = 1.0,
) -> Tuple[Hyperedge, ...]:
    """Return the deterministic best plan from the replicated control."""
    return _solve_replicated_candidate_graph_certificate(
        edge_value,
        candidate_mask,
        target_pair_limit=target_pair_limit,
        reports_per_receiver=reports_per_receiver,
        target_price=target_price,
        target_proxy_floor=target_proxy_floor,
        max_alternatives=1,
    ).selected


def solve_replicated_candidate_graph_certificate(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    target_price: np.ndarray | None = None,
    target_proxy_floor: float = 1.0,
    max_alternatives: int = 8,
    min_worst_ratio: float = 0.95,
    min_sum_ratio: float = 0.95,
) -> ReplicatedPlanCertificate:
    """Public offline-teacher API for joint near-equivalent structures."""
    return _solve_replicated_candidate_graph_certificate(
        edge_value,
        candidate_mask,
        target_pair_limit=target_pair_limit,
        reports_per_receiver=reports_per_receiver,
        target_price=target_price,
        target_proxy_floor=target_proxy_floor,
        max_alternatives=max_alternatives,
        min_worst_ratio=min_worst_ratio,
        min_sum_ratio=min_sum_ratio,
    )


def gossip_candidate_views(
    candidate_mask: np.ndarray,
    *,
    rounds: int,
) -> tuple[np.ndarray, bool, float]:
    """Flood endpoint-known candidates for a finite number of U2U rounds.

    Each endpoint initially knows its incident directed hyperedges. During a
    round it sends its complete current table only to peers connected by at
    least one admitted candidate link.  The returned tensor has shape
    ``(viewer,K,K,Q)`` and exposes whether every node reached the same table.
    """
    mask = np.asarray(candidate_mask, dtype=bool)
    if mask.ndim != 3 or mask.shape[0] != mask.shape[1]:
        raise ValueError("candidate_mask must have shape (K,K,Q)")
    K = mask.shape[0]
    adjacency = np.any(mask | np.swapaxes(mask, 0, 1), axis=-1)
    adjacency[np.arange(K), np.arange(K)] = True
    views = np.zeros((K,) + mask.shape, dtype=bool)
    for node in range(K):
        views[node, node, :, :] |= mask[node, :, :]
        views[node, :, node, :] |= mask[:, node, :]
    for _ in range(max(0, int(rounds))):
        previous = views.copy()
        for node in range(K):
            peers = np.flatnonzero(adjacency[node])
            views[node] = np.any(previous[peers], axis=0)
    common = bool(np.all(views == views[0]))
    union = np.any(views, axis=0)
    intersection = np.all(views, axis=0)
    agreement = float(
        np.sum(intersection) / max(int(np.sum(union)), 1))
    return views, common, agreement
