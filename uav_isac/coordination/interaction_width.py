"""Exact small-graph interaction-width diagnostics for task-mode ISAC."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SensingTaskMode:
    target: int
    owner: int
    transmitters: tuple[int, ...]
    isolated_deflection_ceiling: float


@dataclass(frozen=True)
class ExactTreewidthResult:
    width: int
    elimination_order: tuple[str, ...]
    separator_sizes: tuple[int, ...]


def floor_capable_task_modes(
    coefficient: np.ndarray,
    sensing_budget_w: np.ndarray,
    target_pair_limit: int,
    deflection_floor: float,
) -> tuple[tuple[SensingTaskMode, ...], ...]:
    """Enumerate modes not safely pruned by an isolated worst-floor ceiling.

    A discarded mode cannot meet the per-target Deflection floor even when
    every transmitter in the mode spends its full sensing budget on that one
    target.  The test is necessary, not heuristic; retained modes can still be
    globally infeasible through role, capacity, power-sharing, or tail/mean QoS.
    """
    values = np.asarray(coefficient, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = values.shape
    if budget.shape != (K,) or target_pair_limit < 1:
        raise ValueError("invalid budget or target_pair_limit")
    modes: list[tuple[SensingTaskMode, ...]] = []
    for q in range(Q):
        target_modes = []
        for owner in range(K):
            candidates = tuple(i for i in range(K) if i != owner)
            for size in range(1, min(target_pair_limit, len(candidates)) + 1):
                for transmitters in combinations(candidates, size):
                    ceiling = float(sum(
                        budget[i] * values[i, owner, q]
                        for i in transmitters))
                    if ceiling >= float(deflection_floor) - 1.0e-12:
                        target_modes.append(SensingTaskMode(
                            q, owner, tuple(transmitters), ceiling))
        modes.append(tuple(target_modes))
    return tuple(modes)


def interaction_graphs(
    K: int,
    Q: int,
    modes_by_target: tuple[tuple[SensingTaskMode, ...], ...],
) -> dict[str, nx.Graph]:
    """Build raw, resource-primal, incidence, and mode-clique graph views."""
    if len(modes_by_target) != Q:
        raise ValueError("modes_by_target must have one entry per target")
    scopes = {i: set() for i in range(K)}
    mode_primal = nx.Graph()
    mode_primal.add_nodes_from([f"t:{q}" for q in range(Q)])
    mode_primal.add_nodes_from([f"u:{i}" for i in range(K)])
    for q, modes in enumerate(modes_by_target):
        for mode in modes:
            endpoints = {mode.owner, *mode.transmitters}
            bag = [f"t:{q}"] + [f"u:{i}" for i in sorted(endpoints)]
            for left, right in combinations(bag, 2):
                mode_primal.add_edge(left, right)
            for i in endpoints:
                scopes[i].add(q)

    resource = nx.Graph()
    resource.add_nodes_from([f"t:{q}" for q in range(Q)])
    for scope in scopes.values():
        for left, right in combinations(sorted(scope), 2):
            resource.add_edge(f"t:{left}", f"t:{right}")

    raw = resource.copy()
    # The unlifted steady and bottom-k constraints jointly touch all targets.
    for left, right in combinations(range(Q), 2):
        raw.add_edge(f"t:{left}", f"t:{right}")

    incidence = nx.Graph()
    incidence.add_nodes_from([f"t:{q}" for q in range(Q)])
    incidence.add_nodes_from([f"u:{i}" for i in range(K)])
    for i, scope in scopes.items():
        for q in scope:
            incidence.add_edge(f"t:{q}", f"u:{i}")
    return {
        "raw_target_primal": raw,
        "resource_target_primal": resource,
        "summary_lifted_incidence": incidence,
        "task_mode_clique_primal": mode_primal,
    }


def exact_treewidth(graph: nx.Graph) -> ExactTreewidthResult:
    """Exact O(n 2^n) elimination DP, intended for the current n<=12 audit."""
    nodes = tuple(sorted(str(node) for node in graph.nodes))
    n = len(nodes)
    if n > 22:
        raise ValueError("exact_treewidth audit is limited to at most 22 nodes")
    position = {node: index for index, node in enumerate(nodes)}
    adjacency = [0] * n
    for left, right in graph.edges:
        i, j = position[str(left)], position[str(right)]
        adjacency[i] |= 1 << j
        adjacency[j] |= 1 << i

    def marginal_degree(eliminated: int, vertex: int) -> int:
        reachable = 1 << vertex
        frontier = adjacency[vertex] & eliminated
        reachable |= frontier
        while frontier:
            bit = frontier & -frontier
            frontier -= bit
            index = bit.bit_length() - 1
            new = adjacency[index] & eliminated & ~reachable
            reachable |= new
            frontier |= new
        external = 0
        scan = reachable
        while scan:
            bit = scan & -scan
            scan -= bit
            index = bit.bit_length() - 1
            external |= adjacency[index]
        external &= ~eliminated & ~(1 << vertex)
        return external.bit_count()

    total = 1 << n
    infinity = n + 1
    dp = [infinity] * total
    parent = [-1] * total
    dp[0] = 0
    for eliminated in range(total):
        if dp[eliminated] == infinity:
            continue
        for vertex in range(n):
            if eliminated & (1 << vertex):
                continue
            degree = marginal_degree(eliminated, vertex)
            nxt = eliminated | (1 << vertex)
            candidate = max(dp[eliminated], degree)
            if candidate < dp[nxt]:
                dp[nxt] = candidate
                parent[nxt] = vertex
    order_reversed = []
    state = total - 1
    while state:
        vertex = parent[state]
        order_reversed.append(vertex)
        state ^= 1 << vertex
    order = tuple(reversed(order_reversed))
    eliminated = 0
    separators = []
    for vertex in order:
        separators.append(marginal_degree(eliminated, vertex))
        eliminated |= 1 << vertex
    return ExactTreewidthResult(
        int(dp[-1]), tuple(nodes[index] for index in order),
        tuple(int(value) for value in separators))


def mode_overlap_hhi(
    K: int,
    modes_by_target: tuple[tuple[SensingTaskMode, ...], ...],
) -> tuple[float, ...]:
    """Endpoint-use concentration across UAVs for each target's retained modes."""
    values = []
    for modes in modes_by_target:
        counts = np.zeros(K, dtype=np.float64)
        for mode in modes:
            counts[mode.owner] += 1.0
            counts[list(mode.transmitters)] += 1.0
        total = float(np.sum(counts))
        values.append(float(np.sum((counts / total) ** 2)) if total else 0.0)
    return tuple(values)
