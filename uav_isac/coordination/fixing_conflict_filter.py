"""Exact deletion filter for an irreducible feasible repair-permission block.

This is not an IIS/Farkas implementation.  Starting from a feasible permission
block, it restores one family of fixing constraints at a time and queries an
exact feasibility oracle.  A retained permission has a reproducible local
necessity certificate: removing it from the final-or-larger block is
infeasible.  Monotonicity of restricted feasible sets makes one pass valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PermissionBlock:
    uavs: tuple[int, ...]
    targets: tuple[int, ...]
    role_uavs: tuple[int, ...]


@dataclass(frozen=True)
class FixingDeletionResult:
    block: PermissionBlock
    feasibility_calls: int
    retained_certificates: tuple[str, ...]
    removed_permissions: tuple[str, ...]


@dataclass(frozen=True)
class RepairBackboneResult:
    backbone: tuple[str, ...]
    feasibility_calls: int


def exact_full_block_repair_backbone(
    full_block: PermissionBlock,
    is_feasible: Callable[[PermissionBlock], bool],
) -> RepairBackboneResult:
    """Return permissions present in every feasible permission block.

    A permission ``g`` is in the backbone iff the full block without ``g`` is
    infeasible.  For UAV permission, its dependent role permission is removed
    simultaneously, matching ``role_i <= uav_i``.
    """
    uavs = set(full_block.uavs)
    targets = set(full_block.targets)
    roles = set(full_block.role_uavs)
    if not roles <= uavs or not is_feasible(full_block):
        raise ValueError("full permission block must be dependency-valid and feasible")
    backbone: list[str] = []
    calls = 1
    for q in sorted(targets):
        calls += 1
        if not is_feasible(PermissionBlock(
            tuple(sorted(uavs)), tuple(sorted(targets - {q})),
            tuple(sorted(roles)))):
            backbone.append(f"target:{q}")
    for i in sorted(uavs):
        calls += 1
        if not is_feasible(PermissionBlock(
            tuple(sorted(uavs - {i})), tuple(sorted(targets)),
            tuple(sorted(roles - {i})))):
            backbone.append(f"uav:{i}")
    for i in sorted(roles):
        calls += 1
        if not is_feasible(PermissionBlock(
            tuple(sorted(uavs)), tuple(sorted(targets)),
            tuple(sorted(roles - {i})))):
            backbone.append(f"role:{i}")
    return RepairBackboneResult(tuple(backbone), calls)


def irreducible_feasible_permission_block(
    full_block: PermissionBlock,
    is_feasible: Callable[[PermissionBlock], bool],
    *,
    strategy: str = "sequential",
) -> FixingDeletionResult:
    """Greedily restore fixings and return a one-deletion-irreducible block.

    The deterministic order reflects the search-space objective, not physical
    admission: target permissions, then UAV edge/owner participation, then
    remaining role permissions.  Different orders may yield different
    irreducible blocks; no global minimum claim is made.
    """
    uavs = set(full_block.uavs)
    targets = set(full_block.targets)
    roles = set(full_block.role_uavs)
    if not roles <= uavs:
        raise ValueError("role_uavs must be a subset of uavs")
    if strategy not in {"sequential", "bisect"}:
        raise ValueError("strategy must be 'sequential' or 'bisect'")
    calls = 1
    if not is_feasible(full_block):
        raise ValueError("full permission block must be feasible")
    retained: list[str] = []
    removed: list[str] = []

    def query(label: str, trial_uavs, trial_targets, trial_roles) -> bool:
        nonlocal calls, uavs, targets, roles
        block = PermissionBlock(
            tuple(sorted(trial_uavs)), tuple(sorted(trial_targets)),
            tuple(sorted(trial_roles)))
        calls += 1
        if is_feasible(block):
            uavs, targets, roles = (
                set(trial_uavs), set(trial_targets), set(trial_roles))
            removed.append(label)
            return True
        retained.append(label)
        return False

    if strategy == "sequential":
        for q in sorted(tuple(targets)):
            query(f"target:{q}", uavs, targets - {q}, roles)
        for i in sorted(tuple(uavs)):
            query(f"uav:{i}", uavs - {i}, targets, roles - {i})
        for i in sorted(tuple(roles)):
            query(f"role:{i}", uavs, targets, roles - {i})
    else:
        def bisect_delete(kind: str, items: tuple[int, ...]) -> None:
            if not items:
                return
            item_set = set(items)
            if kind == "target":
                feasible = query(
                    ",".join(f"target:{v}" for v in items),
                    uavs, targets - item_set, roles)
            elif kind == "uav":
                feasible = query(
                    ",".join(f"uav:{v}" for v in items),
                    uavs - item_set, targets, roles - item_set)
            else:
                feasible = query(
                    ",".join(f"role:{v}" for v in items),
                    uavs, targets, roles - item_set)
            if feasible:
                return
            # The batch label is not an irreducible certificate; replace it by
            # singleton certificates obtained through recursive separation.
            retained.pop()
            if len(items) == 1:
                retained.append(f"{kind}:{items[0]}")
                return
            middle = len(items) // 2
            bisect_delete(kind, items[:middle])
            bisect_delete(kind, items[middle:])

        bisect_delete("target", tuple(sorted(targets)))
        bisect_delete("uav", tuple(sorted(uavs)))
        bisect_delete("role", tuple(sorted(roles)))
    return FixingDeletionResult(
        PermissionBlock(tuple(sorted(uavs)), tuple(sorted(targets)),
                        tuple(sorted(roles))),
        calls, tuple(retained), tuple(removed))
