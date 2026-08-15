"""Feasible local-exchange oracle for Gate C1.5 headroom audits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Iterable

import numpy as np


Hyperedge = tuple[int, int, int]


@dataclass(frozen=True)
class LocalMove:
    kind: str
    selected: np.ndarray
    role: np.ndarray
    owner: np.ndarray


@dataclass(frozen=True)
class LocalSearchResult:
    selected: np.ndarray
    role: np.ndarray
    owner: np.ndarray
    accepted_kinds: tuple[str, ...]
    objective_history: tuple[tuple[float, ...], ...]
    candidate_counts: tuple[int, ...]


@dataclass(frozen=True)
class RankedLocalSearchResult:
    """Finite-round first-improvement result plus offline ranking diagnostics."""

    selected: np.ndarray
    role: np.ndarray
    owner: np.ndarray
    accepted_kinds: tuple[str, ...]
    objective_history: tuple[tuple[float, ...], ...]
    candidate_counts: tuple[int, ...]
    verification_counts: tuple[int, ...]
    positive_opportunities: tuple[bool, ...]
    top1_positive: tuple[bool, ...]
    top3_positive: tuple[bool, ...]
    top1_best: tuple[bool, ...]
    top3_best: tuple[bool, ...]
    primary_regret: tuple[float, ...]


def structure_objective_key(
    selected: np.ndarray,
    edge_value: np.ndarray,
    *,
    target_price: np.ndarray | None = None,
    target_proxy_floor: float = 1.0,
) -> tuple[float, ...]:
    """Match the replicated teacher's lexicographic structure objective."""
    pair = np.asarray(selected, dtype=bool)
    value = np.asarray(edge_value, dtype=np.float64)
    if pair.shape != value.shape or pair.ndim != 3:
        raise ValueError("selected/value must have shape (K,K,Q)")
    Q = pair.shape[2]
    priority = (
        np.ones(Q, dtype=np.float64)
        if target_price is None
        else np.asarray(target_price, dtype=np.float64)
    )
    if priority.shape != (Q,):
        raise ValueError("target_price must have shape (Q,)")
    target_value = np.sum(np.where(pair, np.maximum(value, 0.0), 0.0),
                          axis=(0, 1))
    weighted = target_value * np.maximum(priority, 1.0e-9)
    floor = max(float(target_proxy_floor), 1.0e-12)
    edges = tuple(tuple(int(item) for item in row)
                  for row in np.argwhere(pair))
    return (
        float(np.min(weighted)) if Q else 0.0,
        float(np.sum(np.minimum(weighted / floor, 1.0))),
        float(np.sum(weighted)),
        -float(len(edges)),
        *(float(-item) for edge in edges for item in edge),
    )


def objective_strictly_improves(
    candidate: tuple[float, ...],
    current: tuple[float, ...],
) -> bool:
    """Ignore deterministic index ties when deciding whether to switch."""
    if len(candidate) < 4 or len(current) < 4:
        raise ValueError("objective keys must contain four physical/cost terms")
    return tuple(candidate[:4]) > tuple(current[:4])


def _target_values(
    selected: np.ndarray,
    edge_value: np.ndarray,
) -> np.ndarray:
    return np.sum(
        np.where(
            np.asarray(selected, dtype=bool),
            np.maximum(np.asarray(edge_value, dtype=np.float64), 0.0),
            0.0,
        ),
        axis=(0, 1),
    )


def local_move_ranking_key(
    selected: np.ndarray,
    move: LocalMove,
    edge_value: np.ndarray,
    *,
    method: str,
    random_value: float = 0.0,
) -> tuple[float, ...]:
    """Return a ranking key without changing feasibility or accepting a move.

    ``scarcity`` is deliberately cheaper and weaker than the exact
    lexicographic verifier: it weights local target-value deltas by current
    scarcity but never evaluates the post-move minimum/capped objective.
    """
    current = np.asarray(selected, dtype=bool)
    proposal = np.asarray(move.selected, dtype=bool)
    if current.shape != proposal.shape:
        raise ValueError("selected and move must have the same shape")
    if method == "oracle":
        return structure_objective_key(proposal, edge_value)
    changed = float(np.count_nonzero(current ^ proposal))
    if method == "random":
        return (float(random_value), -changed)
    current_target = _target_values(current, edge_value)
    proposal_target = _target_values(proposal, edge_value)
    delta = proposal_target - current_target
    if method == "total_gain":
        return (float(np.sum(delta)), -changed)
    if method != "scarcity":
        raise ValueError(f"unknown ranking method: {method}")
    positive = current_target[current_target > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    denominator = np.maximum(current_target, max(0.1 * scale, 1.0e-9))
    scarcity_gain = float(np.sum(delta / denominator))
    positive_targets = float(np.count_nonzero(delta > 0.0))
    return (
        scarcity_gain,
        positive_targets,
        float(np.sum(delta)),
        -changed,
    )


def assert_local_feasible(
    selected: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
) -> None:
    pair = np.asarray(selected, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if pair.shape != candidate.shape or pair.ndim != 3:
        raise ValueError("selected/candidate must have shape (K,K,Q)")
    K = pair.shape[0]
    if np.any(pair & ~candidate):
        raise AssertionError("selected edge is outside the candidate graph")
    if np.any(pair[np.arange(K), np.arange(K), :]):
        raise AssertionError("self edge selected")
    tx = np.any(pair, axis=(1, 2))
    rx = np.any(pair, axis=(0, 2))
    if np.any(tx & rx):
        raise AssertionError("single-role invariant violated")
    receiver_target = np.any(pair, axis=0)
    if np.any(np.sum(receiver_target, axis=0) > 1):
        raise AssertionError("target has multiple owners")
    if np.any(np.sum(pair, axis=(0, 2)) > int(reports_per_receiver)):
        raise AssertionError("receiver capacity exceeded")
    if np.any(np.sum(pair, axis=(0, 1)) > int(target_pair_limit)):
        raise AssertionError("target pair limit exceeded")


def role_owner_from_structure(
    selected: np.ndarray,
    *,
    fallback_role: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pair = np.asarray(selected, dtype=bool)
    K, _, Q = pair.shape
    role = np.full(K, -1, dtype=np.int8)  # 1=Tx, 0=Rx, -1=idle.
    role[np.any(pair, axis=(1, 2))] = 1
    role[np.any(pair, axis=(0, 2))] = 0
    if fallback_role is not None:
        fallback = np.asarray(fallback_role, dtype=np.int8)
        if fallback.shape != (K,):
            raise ValueError("fallback_role must have shape (K,)")
        role = np.where(role >= 0, role, fallback)
    owner = np.full(Q, -1, dtype=np.int64)
    used = np.any(pair, axis=0)
    for target in range(Q):
        receivers = np.flatnonzero(used[:, target])
        if len(receivers) > 1:
            raise ValueError("structure has multiple owners")
        if len(receivers):
            owner[target] = int(receivers[0])
    return role, owner


def role_first_initial_structure(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a deterministic, permutation-equivariant feasible initial plan."""
    value = np.asarray(edge_value, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    K, _, Q = value.shape
    positive = np.where(mask, np.maximum(value, 0.0), 0.0)
    tx_support = np.sum(positive, axis=(1, 2))
    rx_support = np.sum(positive, axis=(0, 2))
    receiver_count = int(np.clip(np.rint(K / 3.0), 1, K - 1))
    margin = rx_support - tx_support
    receivers = np.argsort(-margin, kind="stable")[:receiver_count]
    role = np.ones(K, dtype=np.int8)
    role[receivers] = 0
    owner = np.full(Q, -1, dtype=np.int64)
    for target in range(Q):
        options = []
        for receiver in receivers:
            incoming = positive[role == 1, receiver, target]
            retained = np.sort(incoming)[
                -max(1, int(target_pair_limit)):]
            score = float(np.sum(retained))
            options.append((score, int(receiver)))
        options.sort(key=lambda item: (-item[0], item[1]))
        if options and options[0][0] > 0.0:
            owner[target] = options[0][1]
    selected = rebuild_structure(
        value, mask, role, owner,
        target_pair_limit=target_pair_limit,
        reports_per_receiver=reports_per_receiver,
    )
    return selected, role, owner


def rebuild_structure(
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
) -> np.ndarray:
    """Rebuild target supports for a fixed role partition and owner vector."""
    value = np.asarray(edge_value, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    role = np.asarray(role, dtype=np.int8)
    owner = np.asarray(owner, dtype=np.int64)
    K, _, Q = value.shape
    selected = np.zeros_like(mask)
    capacity = np.full(K, max(1, int(reports_per_receiver)), dtype=int)
    tx_nodes = np.flatnonzero(role == 1)
    # Weakest attainable targets are allocated first when capacity binds.
    target_options: list[tuple[float, int, list[tuple[float, int]]]] = []
    for target in range(Q):
        receiver = int(owner[target])
        if receiver < 0 or receiver >= K or role[receiver] != 0:
            continue
        incoming = [
            (float(value[tx, receiver, target]), int(tx))
            for tx in tx_nodes
            if mask[tx, receiver, target] and value[tx, receiver, target] > 0.0
        ]
        incoming.sort(key=lambda item: (-item[0], item[1]))
        retained = incoming[:max(1, int(target_pair_limit))]
        attainable = float(sum(item[0] for item in retained))
        target_options.append((attainable, target, retained))
    target_options.sort(key=lambda item: (item[0], item[1]))
    for _, target, incoming in target_options:
        receiver = int(owner[target])
        count = min(len(incoming), capacity[receiver])
        for _, tx in incoming[:count]:
            selected[tx, receiver, target] = True
        capacity[receiver] -= count
    return selected


def rebuild_target_block(
    selected: np.ndarray,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    targets: Iterable[int],
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
) -> np.ndarray | None:
    """Rebuild only a declared target block and freeze every outside edge.

    A role reassignment is admissible only when all frozen outside edges and
    owners remain compatible with it.  This gives N5 a genuine dependency-
    closed locality contract: the selected tensor may change only on
    ``targets``.  ``None`` denotes an incompatible role/owner assignment.
    """
    pair = np.asarray(selected, dtype=bool)
    value = np.asarray(edge_value, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    moved_role = np.asarray(role, dtype=np.int8)
    moved_owner = np.asarray(owner, dtype=np.int64)
    if pair.shape != value.shape or pair.shape != mask.shape or pair.ndim != 3:
        raise ValueError("selected/value/mask must have shape (K,K,Q)")
    K, _, Q = pair.shape
    if moved_role.shape != (K,) or moved_owner.shape != (Q,):
        raise ValueError("role/owner shape mismatch")
    target_ids = tuple(sorted(set(int(target) for target in targets)))
    if not target_ids or any(target < 0 or target >= Q for target in target_ids):
        raise ValueError("targets must contain valid target indices")
    target_set = set(target_ids)
    outside_ids = [target for target in range(Q) if target not in target_set]

    if outside_ids:
        outside = pair[:, :, outside_ids]
        outside_tx = np.any(outside, axis=(1, 2))
        outside_rx = np.any(outside, axis=(0, 2))
        if np.any(outside_tx & (moved_role != 1)):
            return None
        if np.any(outside_rx & (moved_role != 0)):
            return None
        if any(
            moved_owner[target] >= 0
            and moved_role[moved_owner[target]] != 0
            for target in outside_ids
        ):
            return None

    proposal = pair.copy()
    proposal[:, :, list(target_ids)] = False
    capacity = (
        np.full(K, max(1, int(reports_per_receiver)), dtype=int)
        - np.sum(proposal, axis=(0, 2), dtype=int)
    )
    if np.any(capacity < 0):
        return None
    tx_nodes = np.flatnonzero(moved_role == 1)
    target_options: list[tuple[float, int, list[tuple[float, int]]]] = []
    for target in target_ids:
        receiver = int(moved_owner[target])
        if receiver < 0 or receiver >= K or moved_role[receiver] != 0:
            continue
        incoming = [
            (float(value[tx, receiver, target]), int(tx))
            for tx in tx_nodes
            if mask[tx, receiver, target]
            and value[tx, receiver, target] > 0.0
        ]
        incoming.sort(key=lambda item: (-item[0], item[1]))
        retained = incoming[:max(1, int(target_pair_limit))]
        target_options.append((
            float(sum(item[0] for item in retained)),
            target,
            retained,
        ))
    target_options.sort(key=lambda item: (item[0], item[1]))
    for _, target, incoming in target_options:
        receiver = int(moved_owner[target])
        count = min(len(incoming), capacity[receiver])
        for _, tx in incoming[:count]:
            proposal[tx, receiver, target] = True
        capacity[receiver] -= count
    return proposal


def _replace_target(
    selected: np.ndarray,
    target: int,
    edges: Iterable[Hyperedge],
) -> np.ndarray:
    proposal = np.asarray(selected, dtype=bool).copy()
    proposal[:, :, int(target)] = False
    for edge in edges:
        proposal[edge] = True
    return proposal


def _target_blocks_for_budget(
    selected: np.ndarray,
    edge_value: np.ndarray,
    *,
    target_mode: str,
    weak_target_count: int,
) -> list[tuple[int, ...]]:
    """Singleton/two-target blocks from a nested public-proxy budget."""
    pair = np.asarray(selected, dtype=bool)
    value = np.asarray(edge_value, dtype=np.float64)
    Q = pair.shape[2]
    normalized = str(target_mode).strip().lower()
    if normalized not in {"proxy_weak", "all"}:
        raise ValueError("n5_target_mode must be proxy_weak or all")
    target_value = np.sum(
        np.where(pair, np.maximum(value, 0.0), 0.0), axis=(0, 1))
    if normalized == "all":
        target_ids = np.arange(Q, dtype=np.int64)
    else:
        count = min(max(1, int(weak_target_count)), Q)
        target_ids = np.argsort(target_value, kind="stable")[:count]
    blocks = [(int(target),) for target in target_ids]
    blocks.extend(
        tuple(int(target) for target in block)
        for block in combinations(target_ids, 2)
    )
    return blocks


def enumerate_local_moves(
    selected: np.ndarray,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    role: np.ndarray,
    owner: np.ndarray,
    *,
    neighborhoods: tuple[str, ...],
    target_pair_limit: int,
    reports_per_receiver: int,
    n5_target_mode: str = "proxy_weak",
    n5_weak_target_count: int = 2,
    n5_rebuild_scope: str = "global",
) -> list[LocalMove]:
    """Enumerate feasible atomic moves.

    ``proxy_weak`` preserves the deployed N5 neighborhood: only singleton and
    paired blocks drawn from the ``n5_weak_target_count`` weakest
    public-proxy targets are considered.  The default count of two is exactly
    the historical deployed neighborhood.  Varying the count creates nested
    candidate-budget audits without changing the atomic block size.
    ``all`` is a diagnostic ceiling that enumerates every singleton and target
    pair.  It must not be described as a deployable low-cost neighborhood.
    ``global`` preserves the historical N5 behavior: a role change rebuilds
    every target support and may therefore have a Q-target dependency closure.
    ``target_block`` freezes every outside-target edge and discards any role
    assignment incompatible with that frozen structure.  Only this latter
    scope justifies describing N5 as a one/two-target atomic reconfiguration.
    N6 is a role-preserving owner/support exchange on the same nested target
    blocks.  It always has a true one/two-target dependency closure and is the
    low-communication counterpart to role-changing N5.
    """
    pair = np.asarray(selected, dtype=bool)
    value = np.asarray(edge_value, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    K, _, Q = pair.shape
    moves: list[LocalMove] = []
    # N5/N6 reach the same role/owner/block state through many different UAV
    # blocks and assignments.  Rebuilding that deterministic state repeatedly
    # changes neither the neighborhood nor its first-occurrence ordering, so
    # cache the pure reconstruction maps within this one enumeration call.
    global_rebuild_cache: dict[tuple[bytes, bytes], np.ndarray] = {}
    block_rebuild_cache: dict[
        tuple[bytes, bytes, tuple[int, ...]], np.ndarray | None
    ] = {}

    def cached_global_rebuild(
        moved_role: np.ndarray,
        moved_owner: np.ndarray,
    ) -> np.ndarray:
        key = (moved_role.tobytes(), moved_owner.tobytes())
        if key not in global_rebuild_cache:
            global_rebuild_cache[key] = rebuild_structure(
                value,
                mask,
                moved_role,
                moved_owner,
                target_pair_limit=target_pair_limit,
                reports_per_receiver=reports_per_receiver,
            )
        return global_rebuild_cache[key]

    def cached_block_rebuild(
        moved_role: np.ndarray,
        moved_owner: np.ndarray,
        target_block: tuple[int, ...],
    ) -> np.ndarray | None:
        normalized_block = tuple(int(target) for target in target_block)
        key = (
            moved_role.tobytes(), moved_owner.tobytes(), normalized_block,
        )
        if key not in block_rebuild_cache:
            block_rebuild_cache[key] = rebuild_target_block(
                pair,
                value,
                mask,
                moved_role,
                moved_owner,
                normalized_block,
                target_pair_limit=target_pair_limit,
                reports_per_receiver=reports_per_receiver,
            )
        return block_rebuild_cache[key]
    tx_nodes = np.flatnonzero(role == 1)
    rx_nodes = np.flatnonzero(role == 0)
    if "N1" in neighborhoods:
        for target in range(Q):
            receiver = int(owner[target])
            if receiver < 0:
                continue
            current = [tuple(int(item) for item in edge)
                       for edge in np.argwhere(pair[:, :, target])]
            alternatives = [
                (float(value[tx, receiver, target]), int(tx))
                for tx in tx_nodes
                if mask[tx, receiver, target]
                and value[tx, receiver, target] > 0.0
            ]
            alternatives.sort(key=lambda item: (-item[0], item[1]))
            for count in range(1, min(
                    len(alternatives), max(1, int(target_pair_limit))) + 1):
                edges = [(tx, receiver, target)
                         for _, tx in alternatives[:count]]
                proposal = _replace_target(pair, target, edges)
                if not np.array_equal(proposal, pair):
                    moves.append(LocalMove(
                        "N1", proposal, role.copy(), owner.copy()))
            for edge in current:
                proposal = pair.copy()
                proposal[edge] = False
                moves.append(LocalMove(
                    "N1", proposal, role.copy(), owner.copy()))
    if "N3" in neighborhoods:
        for target, receiver in product(range(Q), rx_nodes):
            if int(receiver) == int(owner[target]):
                continue
            moved_owner = owner.copy()
            moved_owner[target] = int(receiver)
            proposal = cached_global_rebuild(role, moved_owner)
            moves.append(LocalMove(
                "N3", proposal, role.copy(), moved_owner))
    if "N2" in neighborhoods:
        for transmitter, receiver in product(tx_nodes, rx_nodes):
            moved_role = role.copy()
            moved_role[transmitter] = 0
            moved_role[receiver] = 1
            moved_owner = owner.copy()
            moved_owner[moved_owner == receiver] = int(transmitter)
            proposal = cached_global_rebuild(moved_role, moved_owner)
            moves.append(LocalMove(
                "N2", proposal, moved_role, moved_owner))
    if "N5" in neighborhoods:
        normalized_n5_target_mode = str(n5_target_mode).strip().lower()
        normalized_n5_rebuild_scope = str(n5_rebuild_scope).strip().lower()
        if normalized_n5_rebuild_scope not in {"global", "target_block"}:
            raise ValueError(
                "n5_rebuild_scope must be global or target_block")
        target_blocks = _target_blocks_for_budget(
            pair,
            value,
            target_mode=normalized_n5_target_mode,
            weak_target_count=n5_weak_target_count,
        )
        # For a target-local N5 move, every outside edge is frozen.  Its
        # incident UAV roles and even an owner with zero retained reports are
        # therefore immutable.  Precomputing these necessary role constraints
        # rejects exactly the states that ``rebuild_target_block`` would return
        # as ``None`` before constructing/cache-keying any target support.
        # This is a pure feasibility implication, not candidate screening.
        outside_role_requirement: dict[tuple[int, ...], np.ndarray] = {}
        for target_block in target_blocks:
            target_set = set(target_block)
            outside_ids = [
                target for target in range(Q)
                if target not in target_set
            ]
            required = np.full(K, -1, dtype=np.int8)
            if outside_ids:
                outside = pair[:, :, outside_ids]
                required[np.any(outside, axis=(1, 2))] = 1
                required[np.any(outside, axis=(0, 2))] = 0
                for target in outside_ids:
                    receiver = int(owner[target])
                    if receiver >= 0:
                        required[receiver] = 0
            outside_role_requirement[target_block] = required
        uav_blocks = [
            block
            for size in (2, 3)
            for block in combinations(range(K), min(size, K))
            if size <= K
        ]
        for block in uav_blocks:
            block_array = np.asarray(block, dtype=np.int64)
            for assignment in product((0, 1), repeat=len(block)):
                moved_role = role.copy()
                moved_role[block_array] = np.asarray(
                    assignment, dtype=np.int8)
                if np.array_equal(moved_role, role):
                    continue
                if not np.any(moved_role == 0) or not np.any(moved_role == 1):
                    continue
                for target_block in target_blocks:
                    required = outside_role_requirement[target_block]
                    constrained = required >= 0
                    if np.any(moved_role[constrained] != required[constrained]):
                        continue
                    owner_options = []
                    for target in target_block:
                        options = [
                            int(receiver)
                            for receiver in np.flatnonzero(moved_role == 0)
                            if receiver in block
                            or receiver == int(owner[target])
                        ]
                        if not options:
                            break
                        owner_options.append(tuple(sorted(set(options))))
                    if len(owner_options) != len(target_block):
                        continue
                    for replacement in product(*owner_options):
                        moved_owner = owner.copy()
                        for target, receiver in zip(
                                target_block, replacement):
                            moved_owner[target] = int(receiver)
                        if normalized_n5_rebuild_scope == "target_block":
                            proposal = cached_block_rebuild(
                                moved_role, moved_owner, target_block)
                            if proposal is None:
                                continue
                            if np.array_equal(proposal, pair):
                                # A latent role-only change has switching cost
                                # but no same-frame sensing benefit.  It is not
                                # a target-block structural candidate.
                                continue
                        else:
                            proposal = cached_global_rebuild(
                                moved_role, moved_owner)
                        moves.append(LocalMove(
                            "N5", proposal, moved_role.copy(), moved_owner))
    if "N6" in neighborhoods:
        target_blocks = _target_blocks_for_budget(
            pair,
            value,
            target_mode=n5_target_mode,
            weak_target_count=n5_weak_target_count,
        )
        for target_block in target_blocks:
            owner_options: list[tuple[int, ...]] = []
            for target in target_block:
                options = [
                    int(receiver)
                    for receiver in rx_nodes
                    if any(
                        mask[tx, receiver, target]
                        and value[tx, receiver, target] > 0.0
                        for tx in tx_nodes
                    )
                ]
                if not options:
                    break
                owner_options.append(tuple(sorted(set(options))))
            if len(owner_options) != len(target_block):
                continue
            for replacement in product(*owner_options):
                moved_owner = owner.copy()
                for target, receiver in zip(target_block, replacement):
                    moved_owner[target] = int(receiver)
                proposal = cached_block_rebuild(
                    role, moved_owner, target_block)
                if proposal is None or np.array_equal(proposal, pair):
                    continue
                moves.append(LocalMove(
                    "N6", proposal, role.copy(), moved_owner))
    feasible = []
    seen: set[bytes] = set()
    for move in moves:
        key = move.selected.tobytes()
        if key in seen:
            continue
        try:
            assert_local_feasible(
                move.selected,
                mask,
                target_pair_limit=target_pair_limit,
                reports_per_receiver=reports_per_receiver,
            )
        except AssertionError:
            continue
        seen.add(key)
        feasible.append(move)
    return feasible


def oracle_best_improvement(
    initial: np.ndarray,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    rounds: int,
    neighborhoods: tuple[str, ...],
    target_pair_limit: int,
    reports_per_receiver: int,
    initial_role: np.ndarray | None = None,
    n5_rebuild_scope: str = "global",
) -> LocalSearchResult:
    """Apply strictly improving feasible local moves for a bounded horizon."""
    selected = np.asarray(initial, dtype=bool).copy()
    fallback = initial_role
    if fallback is None or np.any(np.asarray(fallback) < 0):
        _, fallback, _ = role_first_initial_structure(
            edge_value,
            candidate_mask,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
        )
    role, owner = role_owner_from_structure(
        selected, fallback_role=np.asarray(fallback, dtype=np.int8))
    # Missing owners are initialized to their strongest currently feasible Rx.
    if np.any(owner < 0):
        _, _, fallback_owner = role_first_initial_structure(
            edge_value,
            candidate_mask,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
        )
        owner = np.where(owner >= 0, owner, fallback_owner)
        selected = rebuild_structure(
            edge_value, candidate_mask, role, owner,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver)
    assert_local_feasible(
        selected,
        candidate_mask,
        target_pair_limit=target_pair_limit,
        reports_per_receiver=reports_per_receiver,
    )
    objective = structure_objective_key(selected, edge_value)
    history = [objective]
    kinds: list[str] = []
    candidate_counts: list[int] = []
    for _ in range(max(0, int(rounds))):
        moves = enumerate_local_moves(
            selected,
            edge_value,
            candidate_mask,
            role,
            owner,
            neighborhoods=neighborhoods,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
            n5_rebuild_scope=n5_rebuild_scope,
        )
        candidate_counts.append(len(moves))
        best_move = None
        best_key = objective
        for move in moves:
            key = structure_objective_key(move.selected, edge_value)
            if (
                objective_strictly_improves(key, objective)
                and key > best_key
            ):
                best_key = key
                best_move = move
        if best_move is None:
            break
        selected = best_move.selected
        role = best_move.role
        owner = best_move.owner
        objective = best_key
        kinds.append(best_move.kind)
        history.append(objective)
    return LocalSearchResult(
        selected=selected,
        role=role,
        owner=owner,
        accepted_kinds=tuple(kinds),
        objective_history=tuple(history),
        candidate_counts=tuple(candidate_counts),
    )


def ranked_first_improvement(
    initial: np.ndarray,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    rounds: int,
    neighborhoods: tuple[str, ...],
    target_pair_limit: int,
    reports_per_receiver: int,
    ranking_method: str,
    top_m: int,
    initial_role: np.ndarray | None = None,
    random_seed: int = 0,
    ranking_scorer: Callable[[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[LocalMove],
        np.ndarray,
    ], np.ndarray] | None = None,
    collect_full_diagnostics: bool = True,
) -> RankedLocalSearchResult:
    """Rank moves, exactly verify Top-M, and accept the first positive move.

    Exact keys for every candidate are computed only for offline ranking
    diagnostics. ``verification_counts`` records the number that the simulated
    fail-closed executor would inspect.
    """
    if int(top_m) <= 0:
        raise ValueError("top_m must be positive")
    selected = np.asarray(initial, dtype=bool).copy()
    fallback = initial_role
    if fallback is None or np.any(np.asarray(fallback) < 0):
        _, fallback, _ = role_first_initial_structure(
            edge_value,
            candidate_mask,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
        )
    role, owner = role_owner_from_structure(
        selected, fallback_role=np.asarray(fallback, dtype=np.int8))
    if np.any(owner < 0):
        _, _, fallback_owner = role_first_initial_structure(
            edge_value,
            candidate_mask,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
        )
        owner = np.where(owner >= 0, owner, fallback_owner)
        selected = rebuild_structure(
            edge_value,
            candidate_mask,
            role,
            owner,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
        )
    assert_local_feasible(
        selected,
        candidate_mask,
        target_pair_limit=target_pair_limit,
        reports_per_receiver=reports_per_receiver,
    )
    rng = np.random.default_rng(int(random_seed))
    objective = structure_objective_key(selected, edge_value)
    history = [objective]
    kinds: list[str] = []
    candidate_counts: list[int] = []
    verification_counts: list[int] = []
    positive_opportunities: list[bool] = []
    top1_positive: list[bool] = []
    top3_positive: list[bool] = []
    top1_best: list[bool] = []
    top3_best: list[bool] = []
    primary_regret: list[float] = []
    for _ in range(max(0, int(rounds))):
        moves = enumerate_local_moves(
            selected,
            edge_value,
            candidate_mask,
            role,
            owner,
            neighborhoods=neighborhoods,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
        )
        candidate_counts.append(len(moves))
        if not moves:
            verification_counts.append(0)
            break
        random_values = rng.random(len(moves))
        if ranking_method == "learned":
            if ranking_scorer is None:
                raise ValueError("learned ranking requires ranking_scorer")
            learned_scores = np.asarray(
                ranking_scorer(selected, role, owner, moves, edge_value),
                dtype=np.float64,
            )
            if learned_scores.shape != (len(moves),):
                raise ValueError("ranking_scorer returned the wrong shape")
            ranking_keys = [
                (float(learned_scores[index]),)
                for index in range(len(moves))
            ]
        else:
            ranking_keys = [
                local_move_ranking_key(
                    selected,
                    move,
                    edge_value,
                    method=ranking_method,
                    random_value=float(random_values[index]),
                )
                for index, move in enumerate(moves)
            ]
        order = sorted(
            range(len(moves)),
            key=lambda index: ranking_keys[index],
            reverse=True,
        )
        # Full labels are an offline audit only.  The deployment path computes
        # exact objective keys solely for the bounded Top-M verifier; this
        # prevents a simulated verification saving from being mistaken for an
        # actual one.
        exact_keys: dict[int, tuple[float, ...]] = {}
        if collect_full_diagnostics:
            exact_keys = {
                index: structure_objective_key(move.selected, edge_value)
                for index, move in enumerate(moves)
            }
            best_key = max(exact_keys.values())
            positive = objective_strictly_improves(best_key, objective)
            positive_opportunities.append(bool(positive))
            top1_positive.append(bool(objective_strictly_improves(
                exact_keys[order[0]], objective)))
            top3 = order[:min(3, len(order))]
            top3_positive.append(bool(any(
                objective_strictly_improves(exact_keys[index], objective)
                for index in top3)))
            top1_best.append(bool(exact_keys[order[0]] == best_key))
            top3_best.append(bool(any(
                exact_keys[index] == best_key for index in top3)))

        accepted_index: int | None = None
        verified = 0
        for index in order[:min(int(top_m), len(order))]:
            verified += 1
            if index not in exact_keys:
                exact_keys[index] = structure_objective_key(
                    moves[index].selected, edge_value)
            if objective_strictly_improves(exact_keys[index], objective):
                accepted_index = index
                break
        verification_counts.append(verified)
        accepted_key = (
            exact_keys[accepted_index]
            if accepted_index is not None else objective
        )
        if collect_full_diagnostics:
            primary_regret.append(max(
                0.0, float(best_key[0]) - float(accepted_key[0])))
        if accepted_index is None:
            break
        move = moves[accepted_index]
        selected = move.selected
        role = move.role
        owner = move.owner
        objective = accepted_key
        kinds.append(move.kind)
        history.append(objective)

    return RankedLocalSearchResult(
        selected=selected,
        role=role,
        owner=owner,
        accepted_kinds=tuple(kinds),
        objective_history=tuple(history),
        candidate_counts=tuple(candidate_counts),
        verification_counts=tuple(verification_counts),
        positive_opportunities=tuple(positive_opportunities),
        top1_positive=tuple(top1_positive),
        top3_positive=tuple(top3_positive),
        top1_best=tuple(top1_best),
        top3_best=tuple(top3_best),
        primary_regret=tuple(primary_regret),
    )
