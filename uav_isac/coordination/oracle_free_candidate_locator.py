"""Oracle-free nested candidate grammar for M4-D0.

The locator consumes only a frozen L1 state: deployed structure, per-watt
coefficient, residual budgets and canonical capability-gauge duals.  It has no
feasibility-oracle or exact-witness input.  Exact L2 optimization belongs to a
later referee process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations, product
from typing import Iterable

import numpy as np


Edge = tuple[int, int, int]


@dataclass(frozen=True)
class FrozenStructureCandidate:
    edges: tuple[Edge, ...]
    role: tuple[int, ...]
    owner: tuple[int, ...]
    pool: str
    nested_budget: int
    score_mode: str
    support_size: int
    owner_rank: int
    digest: str


def structure_digest(edges: Iterable[Edge], num_uavs: int, num_targets: int) -> str:
    """Content hash independent of enumeration order."""
    canonical = tuple(sorted(tuple(map(int, edge)) for edge in edges))
    payload = (
        f"K={int(num_uavs)};Q={int(num_targets)};" +
        ";".join(f"{i},{j},{q}" for i, j, q in canonical)
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def structure_state(
    edges: Iterable[Edge], num_uavs: int, num_targets: int,
) -> tuple[tuple[Edge, ...], tuple[int, ...], tuple[int, ...]] | None:
    """Derive single-role and unique-owner state, rejecting illegal graphs."""
    K, Q = int(num_uavs), int(num_targets)
    canonical = tuple(sorted(set(tuple(map(int, edge)) for edge in edges)))
    owner = np.full(Q, -1, dtype=np.int64)
    tx = np.zeros(K, dtype=bool)
    rx = np.zeros(K, dtype=bool)
    for i, j, q in canonical:
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q) or i == j:
            return None
        if owner[q] not in (-1, j):
            return None
        owner[q] = j
        tx[i] = True
        rx[j] = True
    if np.any(owner < 0) or np.any(tx & rx):
        return None
    role = np.full(K, -1, dtype=np.int8)
    role[tx] = 0
    role[rx] = 1
    return canonical, tuple(map(int, role)), tuple(map(int, owner))


def _complete_partition(
    coefficient: np.ndarray,
    budget: np.ndarray,
    target_price: np.ndarray,
    scarcity_price: np.ndarray,
    partition: tuple[int, ...],
    *,
    support_size: int,
    owner_rank: int,
    reports_per_receiver: int,
    score_mode: str,
) -> tuple[Edge, ...] | None:
    K, _, Q = coefficient.shape
    tx_nodes = np.flatnonzero(np.asarray(partition) == 0)
    rx_nodes = np.flatnonzero(np.asarray(partition) == 1)
    if tx_nodes.size == 0 or rx_nodes.size == 0:
        return None
    remaining = np.full(K, int(reports_per_receiver), dtype=np.int64)
    selected: list[Edge] = []
    # Binding targets are completed first; the index tie-break is frozen.
    target_order = sorted(
        range(Q), key=lambda q: (-float(target_price[q]), int(q)))
    for q in target_order:
        owner_options = []
        for j in rx_nodes:
            edge_options = []
            for i in tx_nodes:
                a = float(coefficient[i, j, q])
                if a <= 0.0 or i == j:
                    continue
                score = (
                    a if score_mode == "raw_capability"
                    else float(target_price[q]) * a - float(scarcity_price[i])
                )
                edge_options.append((score, a, int(i)))
            edge_options.sort(key=lambda value: (-value[0], -value[1], value[2]))
            count = min(int(support_size), len(edge_options), int(remaining[j]))
            if count < 1:
                continue
            chosen = edge_options[:count]
            # Budget weighting keeps the score in the capability-gauge units.
            owner_score = float(sum(
                max(item[0], 0.0) * budget[item[2]] for item in chosen))
            raw_score = float(sum(item[1] * budget[item[2]] for item in chosen))
            owner_options.append((owner_score, raw_score, int(j), chosen))
        owner_options.sort(key=lambda value: (-value[0], -value[1], value[2]))
        if not owner_options:
            return None
        choice = owner_options[min(int(owner_rank), len(owner_options) - 1)]
        _score, _raw, receiver, chosen = choice
        for _edge_score, _a, transmitter in chosen:
            selected.append((int(transmitter), int(receiver), int(q)))
        remaining[receiver] -= len(chosen)
    return tuple(sorted(selected))


def _target_rewrite_options(
    coefficient: np.ndarray,
    budget: np.ndarray,
    target_price: np.ndarray,
    scarcity_price: np.ndarray,
    target: int,
    *,
    target_pair_limit: int,
    option_limit: int,
    score_mode: str,
) -> tuple[tuple[Edge, ...], ...]:
    """Rank legal one-target owner/support rewrites without feasibility calls."""
    K = coefficient.shape[0]
    q = int(target)
    scored = []
    for receiver in range(K):
        available = [
            transmitter for transmitter in range(K)
            if transmitter != receiver
            and coefficient[transmitter, receiver, q] > 0.0
        ]
        for support_count in range(
            1, min(int(target_pair_limit), len(available)) + 1):
            for support in combinations(available, support_count):
                raw = float(sum(
                    budget[i] * coefficient[i, receiver, q] for i in support))
                reduced = float(sum(
                    budget[i] * max(
                        target_price[q] * coefficient[i, receiver, q]
                        - scarcity_price[i], 0.0)
                    for i in support))
                score = raw if score_mode == "raw_capability" else reduced
                edges = tuple(sorted((int(i), int(receiver), q) for i in support))
                scored.append((score, raw, edges))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    options = []
    seen = set()
    for _score, _raw, edges in scored:
        if edges in seen:
            continue
        seen.add(edges)
        options.append(edges)
        if len(options) >= int(option_limit):
            break
    return tuple(options)


def _structural_rewrite_options(
    coefficient: np.ndarray,
    budget: np.ndarray,
    target_price: np.ndarray,
    scarcity_price: np.ndarray,
    target: int,
    current: tuple[Edge, ...],
    *,
    target_pair_limit: int,
    per_family_limit: int,
) -> tuple[tuple[Edge, ...], ...]:
    """Return a stratified, intervention-aware target grammar.

    A global top-k list systematically hides inexpensive repairs: an added
    second transmitter, for example, competes against every one- and two-edge
    graph even though it toggles only one deployed edge.  We therefore rank
    separately inside physically meaningful edit families.  No task
    feasibility value is evaluated here; ranking uses only per-watt gain,
    residual budgets and the frozen L1 dual prices.
    """
    K = int(coefficient.shape[0])
    q = int(target)
    limit = max(1, int(per_family_limit))
    current = tuple(sorted(current))
    owner = int(current[0][1]) if current else -1
    families: list[list[tuple[Edge, ...]]] = []
    mandatory_minimal: list[tuple[Edge, ...]] = []

    if current:
        if len(current) < int(target_pair_limit):
            additions = [
                tuple(sorted(current + ((i, owner, q),)))
                for i in range(K)
                if i != owner and all(edge[0] != i for edge in current)
            ]
            families.append(additions)
            # Keep every one-edge augmentation.  A transmitter with only the
            # third-best local gain can be globally optimal when the same node
            # is activated for another target, reducing role and participant
            # changes.  Per-target top-k ranking cannot see that economy.
            mandatory_minimal.extend(additions)
        if len(current) > 1:
            removals = [
                tuple(edge for edge in current if edge != removed)
                for removed in current
            ]
            families.append(removals)
            mandatory_minimal.extend(removals)
        replacements = []
        for removed in current:
            for transmitter in range(K):
                if transmitter == owner or any(
                    edge != removed and edge[0] == transmitter
                    for edge in current
                ):
                    continue
                replacement = tuple(sorted(
                    tuple(edge for edge in current if edge != removed)
                    + ((transmitter, owner, q),)))
                replacements.append(replacement)
        families.append(replacements)
        families.append([
            tuple(sorted((i, receiver, q) for i, _j, _q in current))
            for receiver in range(K)
            if receiver != owner and all(i != receiver for i, _j, _q in current)
        ])
        # Smallest simultaneous role flip: old RX becomes TX and ownership
        # moves.  Such a move has no legal one-edge intermediate.
        families.append([
            ((owner, receiver, q),)
            for receiver in range(K) if receiver != owner
        ])
        if len(current) < int(target_pair_limit):
            promoted_owner_support = [
                tuple(sorted(
                    tuple((i, receiver, q) for i, _j, _q in current)
                    + ((owner, receiver, q),)))
                for receiver in range(K)
                if receiver != owner
                and all(i != receiver for i, _j, _q in current)
            ]
            # Coupled owner migration + old-owner promotion is another move
            # with no legal intermediate: after migration the former receiver
            # can immediately provide the second bistatic path.
            families.append(promoted_owner_support)
            mandatory_minimal.extend(promoted_owner_support)

    for score_mode in ("dual_reduced", "raw_capability"):
        families.append(list(_target_rewrite_options(
            coefficient, budget, target_price, scarcity_price, q,
            target_pair_limit=target_pair_limit,
            option_limit=max(2, limit), score_mode=score_mode)))

    def scores(edges: tuple[Edge, ...]) -> tuple[float, float]:
        raw = float(sum(
            budget[i] * coefficient[i, j, q] for i, j, _q in edges))
        reduced = float(sum(
            budget[i] * max(
                target_price[q] * coefficient[i, j, q]
                - scarcity_price[i], 0.0)
            for i, j, _q in edges))
        return reduced, raw

    result: list[tuple[Edge, ...]] = []
    seen: set[tuple[Edge, ...]] = set()
    for item in mandatory_minimal:
        if item != current and item not in seen:
            seen.add(item)
            result.append(item)
    for family in families:
        canonical = tuple(dict.fromkeys(tuple(sorted(item)) for item in family))
        reduced_ranked = sorted(
            canonical,
            key=lambda item: (-scores(item)[0], -scores(item)[1], item))
        raw_ranked = sorted(
            canonical,
            key=lambda item: (-scores(item)[1], -scores(item)[0], item))
        for item in reduced_ranked[:limit] + raw_ranked[:limit]:
            if item == current or item in seen:
                continue
            seen.add(item)
            result.append(item)
    return tuple(result)


def _joint_structural_repair_grammar(
    coefficient: np.ndarray,
    budget: np.ndarray,
    target_price: np.ndarray,
    scarcity_price: np.ndarray,
    deployed_edges: tuple[Edge, ...],
    target_subset: tuple[int, ...],
    *,
    target_pair_limit: int,
    per_family_limit: int,
    beam_width: int,
    required_participant: int | None = None,
    allow_unchanged_targets: bool = False,
    min_changed_targets: int = 1,
    max_changed_targets: int | None = None,
    _precomputed_options: dict[int, tuple[tuple[Edge, ...], ...]] | None = None,
    _precomputed_ceiling: np.ndarray | None = None,
) -> tuple[tuple[Edge, ...], ...]:
    """Compose cross-target edits with a load-aware diverse beam.

    For each target the proxy uses the fraction of an optimistic independent
    capability ceiling.  It intentionally ignores shared-power competition,
    so it is a locator rather than a hidden feasibility oracle.  Closure
    buckets retain minimum-intervention alternatives alongside high-capability
    graphs.
    """
    K, _, Q = coefficient.shape
    # Canonical target order makes the diverse beam invariant to the caller's
    # dual-price enumeration order; prices affect scores, not reachability.
    target_subset = tuple(sorted(int(q) for q in target_subset))
    deployed_by_target = {
        q: tuple(sorted(edge for edge in deployed_edges if edge[2] == q))
        for q in range(Q)
    }
    if _precomputed_options is None:
        option_map = {
            q: _structural_rewrite_options(
                coefficient, budget, target_price, scarcity_price, q,
                deployed_by_target[q], target_pair_limit=target_pair_limit,
                per_family_limit=per_family_limit)
            for q in target_subset
        }
    else:
        option_map = {
            q: tuple(_precomputed_options[q]) for q in target_subset}
    if required_participant is not None:
        hub = int(required_participant)
        if not 0 <= hub < K:
            raise ValueError("required_participant is outside [0,K)")
        for q in target_subset:
            deployed_q = set(deployed_by_target[q])
            filtered = tuple(
                option for option in option_map[q]
                if hub in {
                    int(node)
                    for edge in deployed_q.symmetric_difference(option)
                    for node in edge[:2]
                })
            option_map[q] = (
                (deployed_by_target[q],) + filtered
                if allow_unchanged_targets else filtered)
    elif allow_unchanged_targets:
        for q in target_subset:
            option_map[q] = (deployed_by_target[q],) + option_map[q]
    if any(not option_map[q] for q in target_subset):
        return ()

    if _precomputed_ceiling is None:
        ceiling = np.zeros(Q, dtype=np.float64)
        for q in range(Q):
            best = 0.0
            for receiver in range(K):
                contributions = sorted((
                    float(budget[i] * coefficient[i, receiver, q])
                    for i in range(K) if i != receiver
                ), reverse=True)
                best = max(
                    best, sum(contributions[:int(target_pair_limit)]))
            ceiling[q] = max(best, np.finfo(np.float64).tiny)
    else:
        ceiling = np.asarray(_precomputed_ceiling, dtype=np.float64)
        if ceiling.shape != (Q,) or np.any(~np.isfinite(ceiling)) or np.any(
                ceiling <= 0.0):
            raise ValueError("precomputed ceiling must be finite and positive")
    base = set(deployed_edges)

    def materialize(choices: tuple[tuple[int, tuple[Edge, ...]], ...]):
        changed = {q: edges for q, edges in choices}
        return tuple(sorted(
            edge for q in range(Q)
            for edge in changed.get(q, deployed_by_target[q])))

    def rank_key(choices: tuple[tuple[int, tuple[Edge, ...]], ...]):
        edges = materialize(choices)
        toggled = base.symmetric_difference(edges)
        affected = {int(edge[2]) for edge in toggled}
        participants = {int(node) for edge in toggled for node in edge[:2]}
        capability = np.zeros(Q, dtype=np.float64)
        for i, j, q in edges:
            capability[q] += budget[i] * coefficient[i, j, q]
        ratio = np.clip(capability / ceiling, 0.0, 1.0)
        bottom = np.sort(ratio)[:min(3, Q)]
        dual_value = float(np.dot(target_price, capability))
        return (
            len(affected) + len(participants), len(toggled),
            tuple(sorted(participants)),
            -float(np.min(ratio)), -float(np.mean(bottom)),
            -float(np.mean(ratio)), -dual_value, choices)

    beam: list[tuple[tuple[int, tuple[Edge, ...]], ...]] = [()]
    for q in target_subset:
        expanded = [
            state + ((q, option),)
            for state in beam for option in option_map[q]]
        if max_changed_targets is not None:
            maximum = int(max_changed_targets)
            expanded = [
                state for state in expanded
                if len({
                    int(edge[2])
                    for edge in base.symmetric_difference(
                        materialize(state))
                }) <= maximum
            ]
        buckets: dict[tuple[int, int, tuple[int, ...]], list] = {}
        for state in expanded:
            key = rank_key(state)
            buckets.setdefault((key[0], key[1], key[2]), []).append(
                (key, state))
        diverse = []
        bucket_keep = max(2, int(beam_width) // max(1, len(buckets)))
        for values in buckets.values():
            values.sort(key=lambda item: item[0][3:])
            diverse.extend(values[:bucket_keep])
        diverse.sort(
            key=lambda item: (
                item[0][3:7] + item[0][:2]
                + (item[0][2], item[1])))
        chosen = [state for _key, state in diverse[:int(beam_width)]]
        if required_participant is not None and len(target_subset) >= 4:
            # Radius-4 endpoint-coherent repairs have many physically similar
            # graphs inside one participant signature.  Preserve a shallow
            # Pareto face per signature instead of multiplying the global
            # beam width.  Twelve is sufficient to retain different owner /
            # support factorizations while remaining bounded for K<=8.
            champions = [
                state for values in buckets.values()
                for _key, state in values[:12]
            ]
            chosen = list(dict.fromkeys(chosen + champions))
        beam = chosen
    minimum = max(0, int(min_changed_targets))
    maximum = (
        len(target_subset) if max_changed_targets is None
        else int(max_changed_targets))
    result = []
    for state in beam:
        edges = materialize(state)
        changed = len({
            int(edge[2]) for edge in base.symmetric_difference(edges)})
        if minimum <= changed <= maximum:
            result.append(edges)
    return tuple(result)


def _atomic_local_grammar(
    deployed_edges: tuple[Edge, ...],
    num_uavs: int,
    num_targets: int,
    target_pair_limit: int,
) -> tuple[tuple[Edge, ...], ...]:
    """All one-step owner/support replacements and global role swaps."""
    K, Q = int(num_uavs), int(num_targets)
    base = set(deployed_edges)
    proposals = set()
    by_target = {
        q: tuple(sorted(edge for edge in base if edge[2] == q))
        for q in range(Q)
    }
    for q in range(Q):
        current = by_target[q]
        # Owner replacement retains the target's transmitter support.
        for receiver in range(K):
            proposal = (base - set(current)) | {
                (i, receiver, q) for i, _old_receiver, _q in current
                if i != receiver
            }
            if len([edge for edge in proposal if edge[2] == q]) == len(current):
                proposals.add(tuple(sorted(proposal)))
        # One support replacement, addition, or removal.
        for old in current:
            for transmitter in range(K):
                if transmitter == old[1]:
                    continue
                proposal = (base - {old}) | {(transmitter, old[1], q)}
                proposals.add(tuple(sorted(proposal)))
        if len(current) < int(target_pair_limit) and current:
            receiver = current[0][1]
            for transmitter in range(K):
                if transmitter != receiver:
                    proposals.add(tuple(sorted(
                        base | {(transmitter, receiver, q)})))
        if len(current) > 1:
            for old in current:
                proposals.add(tuple(sorted(base - {old})))
    # A role swap is a global node permutation, so dependency closure is
    # computed from structural consequences rather than supplied externally.
    for first, second in combinations(range(K), 2):
        def swap(node: int) -> int:
            return second if node == first else first if node == second else node
        proposals.add(tuple(sorted(
            (swap(i), swap(j), q) for i, j, q in base)))
    proposals.discard(tuple(sorted(base)))
    return tuple(sorted(proposals))


def generate_oracle_free_candidate_pools(
    coefficient: np.ndarray,
    budget: np.ndarray,
    deployed_edges: Iterable[Edge],
    deployed_role: np.ndarray,
    target_price: np.ndarray,
    scarcity_price: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    nested_budgets: tuple[int, ...] = (1, 2, 3),
) -> tuple[FrozenStructureCandidate, ...]:
    """Generate dual-guided Pool A and role-grammar diagnostic Pool B.

    Pool A changes at most ``B`` role labels from the deployed partition and
    increases support/owner alternatives monotonically with pre-registered
    ``B``.  Pool B enumerates every legal role partition but still completes
    it using only raw/dual local capability, never exact feasibility feedback.
    """
    value = np.asarray(coefficient, dtype=np.float64)
    deployed_edges = tuple(
        sorted(tuple(map(int, edge)) for edge in deployed_edges))
    resource = np.asarray(budget, dtype=np.float64).reshape(-1)
    role0 = np.asarray(deployed_role, dtype=np.int8).reshape(-1)
    pi = np.asarray(target_price, dtype=np.float64).reshape(-1)
    eta = np.asarray(scarcity_price, dtype=np.float64).reshape(-1)
    if value.ndim != 3 or value.shape[0] != value.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = value.shape
    if (
        resource.shape != (K,) or role0.shape != (K,)
        or pi.shape != (Q,) or eta.shape != (K,)
        or np.any(~np.isfinite(value)) or np.any(value < 0.0)
        or np.any(~np.isfinite(resource)) or np.any(resource < 0.0)
        or np.any(~np.isfinite(pi)) or np.any(pi < 0.0)
        or np.any(~np.isfinite(eta)) or np.any(eta < 0.0)
    ):
        raise ValueError("invalid oracle-free locator inputs")
    if target_pair_limit < 1 or reports_per_receiver < 1:
        raise ValueError("structure capacities must be positive")
    budgets = tuple(sorted(set(int(item) for item in nested_budgets)))
    if not budgets or budgets[0] < 1:
        raise ValueError("nested budgets must be positive")

    candidates: dict[str, FrozenStructureCandidate] = {}

    def admit(
        edges: Iterable[Edge], *, pool: str, nested_budget: int,
        score_mode: str, support_size: int, owner_rank: int,
    ) -> None:
        state = structure_state(edges, K, Q)
        if state is None:
            return
        canonical, role, owner = state
        if any(value[edge] <= 0.0 for edge in canonical):
            return
        per_target = np.bincount(
            [edge[2] for edge in canonical], minlength=Q)
        per_receiver = np.bincount(
            [edge[1] for edge in canonical], minlength=K)
        if (
            np.any(per_target > int(target_pair_limit))
            or np.any(per_receiver > int(reports_per_receiver))
        ):
            return
        digest = structure_digest(canonical, K, Q)
        proposal = FrozenStructureCandidate(
            canonical, role, owner, pool, int(nested_budget), score_mode,
            int(support_size), int(owner_rank), digest)
        previous = candidates.get(digest)
        proposal_key = (
            0 if pool == "A_DUAL_NESTED" else 1,
            int(nested_budget), score_mode)
        previous_key = (
            0 if previous is not None and previous.pool == "A_DUAL_NESTED" else 1,
            previous.nested_budget if previous is not None else 10**9,
            previous.score_mode if previous is not None else "")
        if previous is None or proposal_key < previous_key:
            candidates[digest] = proposal

    admit(
        deployed_edges, pool="A_DUAL_NESTED", nested_budget=0,
        score_mode="deployed", support_size=0, owner_rank=0)

    partitions = tuple(product((-1, 0, 1), repeat=K))
    for B in budgets:
        for partition in partitions:
            distance = int(np.sum(np.asarray(partition) != role0))
            if distance > B:
                continue
            for support in range(1, min(B, int(target_pair_limit)) + 1):
                for owner_rank in range(min(B, 2)):
                    edges = _complete_partition(
                        value, resource, pi, eta, partition,
                        support_size=support, owner_rank=owner_rank,
                        reports_per_receiver=reports_per_receiver,
                        score_mode="dual_reduced")
                    if edges is not None:
                        admit(
                            edges, pool="A_DUAL_NESTED", nested_budget=B,
                            score_mode="dual_reduced", support_size=support,
                            owner_rank=owner_rank)

    diagnostic_budget = max(budgets) + 1
    for partition in partitions:
        for score_mode in ("dual_reduced", "raw_capability"):
            for support in range(1, int(target_pair_limit) + 1):
                for owner_rank in range(2):
                    edges = _complete_partition(
                        value, resource, pi, eta, partition,
                        support_size=support, owner_rank=owner_rank,
                        reports_per_receiver=reports_per_receiver,
                        score_mode=score_mode)
                    if edges is not None:
                        admit(
                            edges, pool="B_ROLE_GRAMMAR", nested_budget=diagnostic_budget,
                            score_mode=score_mode, support_size=support,
                            owner_rank=owner_rank)

    # Target-local grammar: unchanged targets retain their exact deployed
    # edges.  This repairs the structural overreach of whole-partition
    # completion while remaining strictly oracle-free.  Pool A uses nested
    # bottleneck-target and option budgets.  Pool B exhausts all target subsets
    # up to the pre-registered radius with the top four raw/dual rewrites.
    deployed_by_target = {
        q: tuple(sorted(edge for edge in deployed_edges if int(edge[2]) == q))
        for q in range(Q)
    }
    target_order = tuple(sorted(
        range(Q), key=lambda q: (-float(pi[q]), int(q))))
    joint_ceiling = np.zeros(Q, dtype=np.float64)
    for q in range(Q):
        for receiver in range(K):
            contributions = sorted((
                float(resource[i] * value[i, receiver, q])
                for i in range(K) if i != receiver
            ), reverse=True)
            joint_ceiling[q] = max(
                joint_ceiling[q],
                sum(contributions[:int(target_pair_limit)]))
    joint_ceiling = np.maximum(joint_ceiling, np.finfo(np.float64).tiny)
    joint_option_cache = {
        limit: {
            q: _structural_rewrite_options(
                value, resource, pi, eta, q, deployed_by_target[q],
                target_pair_limit=int(target_pair_limit),
                per_family_limit=limit)
            for q in range(Q)
        }
        for limit in (1, 2)
    }
    for B in budgets:
        active_targets = target_order[:min(B, Q)]
        option_map = {
            q: _target_rewrite_options(
                value, resource, pi, eta, q,
                target_pair_limit=target_pair_limit,
                option_limit=B, score_mode="dual_reduced")
            for q in active_targets
        }
        for radius in range(1, min(B, len(active_targets)) + 1):
            for targets in combinations(active_targets, radius):
                for choices in product(*(option_map[q] for q in targets)):
                    edges = [
                        edge for q in range(Q) if q not in targets
                        for edge in deployed_by_target[q]
                    ]
                    edges.extend(edge for option in choices for edge in option)
                    admit(
                        edges, pool="A_DUAL_NESTED", nested_budget=B,
                        score_mode="dual_target_local", support_size=max(
                            len(option) for option in choices), owner_rank=0)

        # A target outside the top-B dual list can release a role needed by a
        # binding target.  Search a deterministic 2B superset while changing
        # at most B target blocks.  At the final nested budget admit one extra
        # target block: the formal budget controls role-label changes, whereas
        # reusing one activated endpoint across several targets may change
        # B+1 target blocks without increasing that role budget.
        structural_targets = target_order[:min(2 * B, Q)]
        target_radius = B + (1 if B == max(budgets) else 0)
        family_limit = min(2, B)
        for radius in range(
                1, min(target_radius, len(structural_targets)) + 1):
            for targets in combinations(structural_targets, radius):
                for edges in _joint_structural_repair_grammar(
                    value, resource, pi, eta, deployed_edges, targets,
                    target_pair_limit=int(target_pair_limit),
                    per_family_limit=family_limit,
                    beam_width=128 * max(1, radius - 1),
                    _precomputed_options=joint_option_cache[family_limit],
                    _precomputed_ceiling=joint_ceiling):
                    admit(
                        edges, pool="A_DUAL_NESTED", nested_budget=B,
                        score_mode="joint_load_aware_structural",
                        support_size=0, owner_rank=0)

    # The exact intervention objective contains the set-union term
    # |union_q V(Delta E_q)|.  One endpoint-conditioned dynamic beam per UAV
    # generates all radius-2..4 repairs in one scan, avoiding repeated subset
    # enumeration while exposing this submodular endpoint-reuse economy.
    endpoint_radius = min(max(budgets) + 1, Q)
    for hub in range(K):
        for edges in _joint_structural_repair_grammar(
            value, resource, pi, eta, deployed_edges, tuple(range(Q)),
            target_pair_limit=int(target_pair_limit),
            per_family_limit=2,
            beam_width=512,
            required_participant=hub,
            allow_unchanged_targets=True,
            min_changed_targets=2,
            max_changed_targets=endpoint_radius,
            _precomputed_options=joint_option_cache[2],
            _precomputed_ceiling=joint_ceiling):
            admit(
                edges, pool="A_DUAL_NESTED",
                nested_budget=max(budgets),
                score_mode="endpoint_coherent_joint_dp",
                support_size=0, owner_rank=hub)

    local_radius = min(3, Q)
    local_option_limit = 4
    diagnostic_options = {}
    for q in range(Q):
        combined = []
        for score_mode in ("dual_reduced", "raw_capability"):
            combined.extend(_target_rewrite_options(
                value, resource, pi, eta, q,
                target_pair_limit=target_pair_limit,
                option_limit=local_option_limit,
                score_mode=score_mode))
        diagnostic_options[q] = tuple(dict.fromkeys(combined))
    for radius in range(1, local_radius + 1):
        for targets in combinations(range(Q), radius):
            for choices in product(*(diagnostic_options[q] for q in targets)):
                edges = [
                    edge for q in range(Q) if q not in targets
                    for edge in deployed_by_target[q]
                ]
                edges.extend(edge for option in choices for edge in option)
                admit(
                    edges, pool="B_ROLE_GRAMMAR", nested_budget=diagnostic_budget,
                    score_mode="local_target_rewrite", support_size=max(
                        len(option) for option in choices), owner_rank=0)
    for edges in _atomic_local_grammar(
            deployed_edges, K, Q, int(target_pair_limit)):
        admit(
            edges, pool="B_ROLE_GRAMMAR", nested_budget=diagnostic_budget,
            score_mode="exhaustive_atomic_local", support_size=0,
            owner_rank=0)
    return tuple(sorted(
        candidates.values(), key=lambda item: (item.pool, item.nested_budget,
                                                item.digest)))
