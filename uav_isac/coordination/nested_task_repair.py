"""Dependency-closed nested block construction for shadow task repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RepairBlock:
    uavs: tuple[int, ...]
    targets: tuple[int, ...]
    role_uavs: tuple[int, ...]
    trigger_edge: tuple[int, int, int] | None


def task_violation_subgradient_weights(
    detection_probability: np.ndarray,
    task_floors: tuple[float, float, float, int] | list[float],
    *,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return a non-negative subgradient support of the active task ratio.

    Only the maximally violated normalized branch contributes: worst assigns
    mass to its tied minimizers, bottom-k to its current k-tail, and average to
    all targets.  The magnitude is normalized to one because it is used only
    for candidate ordering, never for admission.
    """
    pd = np.asarray(detection_probability, dtype=np.float64).reshape(-1)
    rho_min, rho_tail, rho_avg = (float(v) for v in task_floors[:3])
    k = int(task_floors[3])
    if pd.size == 0 or np.any(~np.isfinite(pd)) or np.any(pd <= 0.0):
        raise ValueError("detection probabilities must be positive and finite")
    if not 1 <= k <= pd.size:
        raise ValueError("tail k is outside the target count")
    tail = np.argsort(pd, kind="stable")[:k]
    ratios = np.asarray([
        rho_min / float(np.min(pd)),
        rho_tail / float(np.mean(pd[tail])),
        rho_avg / float(np.mean(pd)),
    ])
    active = np.flatnonzero(ratios >= np.max(ratios) - tolerance)
    weight = np.zeros(pd.size, dtype=np.float64)
    if 0 in active:
        worst = np.flatnonzero(pd <= np.min(pd) + tolerance)
        weight[worst] += 1.0 / len(worst)
    if 1 in active:
        weight[tail] += 1.0 / k
    if 2 in active:
        weight += 1.0 / pd.size
    return weight / float(np.sum(weight))


def dependency_closed_block(
    current_selected: np.ndarray,
    current_owner: np.ndarray,
    seed_uavs: set[int],
    seed_targets: set[int],
    role_change_uavs: set[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Close a proposed block under current edge and ownership dependencies."""
    selected = np.asarray(current_selected, dtype=bool)
    owner = np.asarray(current_owner, dtype=np.int64).reshape(-1)
    K, K2, Q = selected.shape
    if K != K2 or owner.shape != (Q,):
        raise ValueError("selected/owner shapes are inconsistent")
    uavs = {int(value) for value in seed_uavs}
    targets = {int(value) for value in seed_targets}
    role_uavs = (
        set(uavs) if role_change_uavs is None
        else {int(value) for value in role_change_uavs}
    )
    changed = True
    while changed:
        old_uavs, old_targets = set(uavs), set(targets)
        # Only a UAV whose role may change invalidates all incident edges.
        # Mere participation in one toggled edge does not propagate globally.
        for i, j, q in np.argwhere(selected):
            if int(i) in role_uavs or int(j) in role_uavs:
                targets.add(int(q))
        # Every affected target brings its old owner and current endpoints.
        for q in tuple(targets):
            uavs.add(int(owner[q]))
            for i, j in np.argwhere(selected[:, :, q]):
                uavs.update((int(i), int(j)))
        changed = uavs != old_uavs or targets != old_targets
    return tuple(sorted(uavs)), tuple(sorted(targets))


def nested_repair_blocks(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    current_selected: np.ndarray,
    current_owner: np.ndarray,
    current_tx_role: np.ndarray,
    current_rx_role: np.ndarray,
    current_pd: np.ndarray,
    task_floors: tuple[float, float, float, int] | list[float],
    *,
    ranking_mode: str = "gain",
) -> tuple[RepairBlock, ...]:
    """Construct monotone blocks from task subgradient-weighted edge scores."""
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    K, K2, Q = gain.shape
    if K != K2 or budget.shape != (K,):
        raise ValueError("gain/budget shapes are inconsistent")
    weights = task_violation_subgradient_weights(current_pd, task_floors)
    if ranking_mode not in {"gain", "closure_efficiency"}:
        raise ValueError("ranking_mode must be 'gain' or 'closure_efficiency'")
    candidates = []
    selected = np.asarray(current_selected, dtype=bool)
    tx_role = np.asarray(current_tx_role, dtype=bool).reshape(-1)
    rx_role = np.asarray(current_rx_role, dtype=bool).reshape(-1)
    if tx_role.shape != (K,) or rx_role.shape != (K,):
        raise ValueError("current role vectors must match K")
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            for q in range(Q):
                if selected[i, j, q]:
                    continue
                score = float(weights[q] * budget[i] * gain[i, j, q])
                candidates.append((score, i, j, q))
    blocks: list[RepairBlock] = []
    seed_uavs: set[int] = set()
    seed_targets: set[int] = set()
    role_uavs: set[int] = set()
    previous = None
    remaining = list(candidates)
    while remaining:
        ranked = []
        for candidate in remaining:
            score, i, j, q = candidate
            trial_uavs = seed_uavs | {i, j}
            trial_targets = seed_targets | {q}
            trial_roles = set(role_uavs)
            if not tx_role[i]:
                trial_roles.add(i)
            if not rx_role[j]:
                trial_roles.add(j)
            closed_uavs, closed_targets = dependency_closed_block(
                selected, current_owner, trial_uavs, trial_targets, trial_roles)
            increment = (
                len(set(closed_uavs) - seed_uavs)
                + len(set(closed_targets) - seed_targets)
                + len(trial_roles - role_uavs)
            )
            efficiency = score / max(increment, 1)
            primary = efficiency if ranking_mode == "closure_efficiency" else score
            ranked.append((
                -primary, increment, len(trial_roles - role_uavs),
                -score, i, j, q, candidate, trial_uavs, trial_targets,
                trial_roles, closed_uavs, closed_targets,
            ))
        chosen = min(ranked)
        (_neg_efficiency, _increment, _new_roles, _neg_score,
         i, j, q, original, trial_uavs, trial_targets, trial_roles,
         uavs, targets) = chosen
        remaining.remove(original)
        seed_uavs = trial_uavs
        seed_targets = trial_targets
        role_uavs = trial_roles
        signature = (uavs, targets, tuple(sorted(role_uavs)))
        if signature != previous:
            blocks.append(RepairBlock(
                uavs, targets, tuple(sorted(role_uavs)), (i, j, q)))
            previous = signature
        if len(uavs) == K and len(targets) == Q:
            break
    if not blocks or len(blocks[-1].uavs) < K or len(blocks[-1].targets) < Q:
        blocks.append(RepairBlock(
            tuple(range(K)), tuple(range(Q)), tuple(range(K)), None))
    return tuple(blocks)
