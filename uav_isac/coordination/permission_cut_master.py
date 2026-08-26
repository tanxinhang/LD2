"""Logic-based conflict-cut master for minimum permission-cardinality repair."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from uav_isac.coordination.fixing_conflict_filter import PermissionBlock


class MonotonePermissionOracle:
    """Exact oracle wrapper with dominance-certified antichain inference.

    Feasibility is upward closed in the selected permission set.  A stored
    feasible subset certifies every superset; a stored infeasible superset
    certifies every subset.  Only incomparable blocks reach ``exact_oracle``.
    """

    def __init__(self, exact_oracle: Callable[[PermissionBlock], bool]):
        self._exact_oracle = exact_oracle
        self._minimal_feasible: list[frozenset[str]] = []
        self._maximal_infeasible: list[frozenset[str]] = []
        self.queries = 0
        self.exact_calls = 0
        self.inferred_feasible = 0
        self.inferred_infeasible = 0

    @staticmethod
    def _key(block: PermissionBlock) -> frozenset[str]:
        if not set(block.role_uavs) <= set(block.uavs):
            raise ValueError("role_uavs must be a subset of uavs")
        return frozenset(_selected_labels(block))

    def __call__(self, block: PermissionBlock) -> bool:
        self.queries += 1
        key = self._key(block)
        if any(certificate <= key for certificate in self._minimal_feasible):
            self.inferred_feasible += 1
            return True
        if any(key <= certificate for certificate in self._maximal_infeasible):
            self.inferred_infeasible += 1
            return False
        self.exact_calls += 1
        answer = bool(self._exact_oracle(block))
        if answer:
            self._minimal_feasible = [
                certificate for certificate in self._minimal_feasible
                if not key <= certificate]
            self._minimal_feasible.append(key)
        else:
            self._maximal_infeasible = [
                certificate for certificate in self._maximal_infeasible
                if not certificate <= key]
            self._maximal_infeasible.append(key)
        return answer


class CertifiedFeasibility(Enum):
    CERT_FEASIBLE = "cert_feasible"
    CERT_INFEASIBLE = "cert_infeasible"
    UNRESOLVED = "unresolved"


class CertifiedMonotonePermissionOracle:
    """Tri-state monotone cache; unresolved results never become certificates."""

    def __init__(
        self,
        exact_oracle: Callable[[PermissionBlock], CertifiedFeasibility],
    ):
        self._exact_oracle = exact_oracle
        self._minimal_feasible: list[frozenset[str]] = []
        self._maximal_infeasible: list[frozenset[str]] = []
        self.queries = 0
        self.exact_calls = 0
        self.inferred_feasible = 0
        self.inferred_infeasible = 0
        self.unresolved = 0

    def __call__(self, block: PermissionBlock) -> CertifiedFeasibility:
        self.queries += 1
        key = MonotonePermissionOracle._key(block)
        if any(certificate <= key for certificate in self._minimal_feasible):
            self.inferred_feasible += 1
            return CertifiedFeasibility.CERT_FEASIBLE
        if any(key <= certificate for certificate in self._maximal_infeasible):
            self.inferred_infeasible += 1
            return CertifiedFeasibility.CERT_INFEASIBLE
        self.exact_calls += 1
        answer = self._exact_oracle(block)
        if answer is CertifiedFeasibility.CERT_FEASIBLE:
            self._minimal_feasible = [
                certificate for certificate in self._minimal_feasible
                if not key <= certificate]
            self._minimal_feasible.append(key)
        elif answer is CertifiedFeasibility.CERT_INFEASIBLE:
            self._maximal_infeasible = [
                certificate for certificate in self._maximal_infeasible
                if not certificate <= key]
            self._maximal_infeasible.append(key)
        else:
            self.unresolved += 1
        return answer


@dataclass(frozen=True)
class PermissionCutResult:
    block: PermissionBlock
    oracle_calls: int
    conflict_cuts: tuple[tuple[str, ...], ...]
    permission_cardinality: int


@dataclass(frozen=True)
class PermissionConflictCoreResult:
    """One-deletion-irreducible closed-permission conflict.

    This is an exact-oracle certificate for the monotone permission system; it
    is deliberately not called an IIS or a Farkas certificate.
    """

    core: tuple[str, ...]
    oracle_calls: int
    removed_permissions: tuple[str, ...]


@dataclass(frozen=True)
class CoreGuidedPermissionResult:
    block: PermissionBlock
    oracle_calls: int
    master_oracle_calls: int
    shrink_oracle_calls: int
    conflict_cores: tuple[tuple[str, ...], ...]
    permission_cardinality: int
    permission_objective: tuple[int, int]


@dataclass(frozen=True)
class ObjectiveLayerResult:
    representative: PermissionBlock
    scalar_cost: int
    permission_objective: tuple[int, int]
    common_zero_permissions: tuple[str, ...]
    common_zero_cost_upper: int
    master_solves: int


@dataclass(frozen=True)
class ObjectiveLayerSeparationResult:
    block: PermissionBlock
    master_iterations: int
    conflict_cuts: tuple[tuple[str, ...], ...]
    cut_kinds: tuple[str, ...]
    lower_bound_trace: tuple[int, ...]
    permission_objective: tuple[int, int]


class ObjectiveLayerSeparationLimit(RuntimeError):
    def __init__(
        self,
        master_iterations: int,
        conflict_cuts: tuple[tuple[str, ...], ...],
        cut_kinds: tuple[str, ...],
        lower_bound_trace: tuple[int, ...],
    ):
        super().__init__("objective-layer separation exceeded max_iterations")
        self.master_iterations = master_iterations
        self.conflict_cuts = conflict_cuts
        self.cut_kinds = cut_kinds
        self.lower_bound_trace = lower_bound_trace


class PermissionCutIterationLimit(RuntimeError):
    def __init__(self, oracle_calls: int, conflict_cuts: tuple[tuple[str, ...], ...]):
        super().__init__("permission cut master exceeded max_iterations")
        self.oracle_calls = int(oracle_calls)
        self.conflict_cuts = conflict_cuts


class CoreGuidedIterationLimit(RuntimeError):
    def __init__(
        self,
        oracle_calls: int,
        master_oracle_calls: int,
        shrink_oracle_calls: int,
        conflict_cores: tuple[tuple[str, ...], ...],
    ):
        super().__init__("core-guided permission master exceeded max_iterations")
        self.oracle_calls = int(oracle_calls)
        self.master_oracle_calls = int(master_oracle_calls)
        self.shrink_oracle_calls = int(shrink_oracle_calls)
        self.conflict_cores = conflict_cores


def permission_labels(K: int, Q: int) -> tuple[str, ...]:
    return tuple(
        [f"target:{q}" for q in range(Q)]
        + [f"uav:{i}" for i in range(K)]
        + [f"role:{i}" for i in range(K)]
    )


def _block(labels: tuple[str, ...], selected: np.ndarray) -> PermissionBlock:
    chosen = {labels[index] for index in np.flatnonzero(selected >= 0.5)}
    return PermissionBlock(
        tuple(i for i in range(sum(label.startswith("uav:") for label in labels))
              if f"uav:{i}" in chosen),
        tuple(q for q in range(sum(label.startswith("target:") for label in labels))
              if f"target:{q}" in chosen),
        tuple(i for i in range(sum(label.startswith("role:") for label in labels))
              if f"role:{i}" in chosen),
    )


def _selected_labels(block: PermissionBlock) -> set[str]:
    return (
        {f"target:{q}" for q in block.targets}
        | {f"uav:{i}" for i in block.uavs}
        | {f"role:{i}" for i in block.role_uavs}
    )


def permission_block_with_closed_set(
    K: int,
    Q: int,
    closed_permissions: tuple[str, ...],
) -> PermissionBlock:
    """Return the largest dependency-valid block avoiding ``closed_permissions``.

    Closing ``uav:i`` also closes ``role:i`` because a role permission without
    its UAV permission violates ``u_role,i <= u_uav,i``.  A core containing an
    UAV therefore yields the valid master cut ``u_uav,i >= 1`` without needing
    a redundant role label.
    """
    labels = permission_labels(int(K), int(Q))
    universe = set(labels)
    closed = set(closed_permissions)
    if not closed <= universe:
        raise ValueError("closed permission is outside the universe")
    for label in tuple(closed):
        if label.startswith("uav:"):
            closed.add(f"role:{int(label.split(':', 1)[1])}")
    selected = np.asarray([label not in closed for label in labels], dtype=float)
    return _block(labels, selected)


def shrink_infeasible_permission_core(
    K: int,
    Q: int,
    closed_permissions: tuple[str, ...],
    is_feasible: Callable[[PermissionBlock], bool],
    *,
    verify_initial: bool = True,
    deletion_order: tuple[str, ...] | None = None,
    strategy: str = "sequential",
) -> PermissionConflictCoreResult:
    """Shrink a closed set to an oracle-certified irreducible conflict.

    For the returned ``C``, the largest legal block avoiding ``C`` is
    infeasible, while deleting any single member of ``C`` makes that largest
    block feasible.  Search order may change which irreducible core is found,
    but never the validity of its hitting-set cut.
    """
    labels = permission_labels(int(K), int(Q))
    universe = set(labels)
    current = set(closed_permissions)
    if not current <= universe:
        raise ValueError("closed permission is outside the universe")
    if strategy not in {"sequential", "quickxplain"}:
        raise ValueError("strategy must be 'sequential' or 'quickxplain'")
    if deletion_order is None:
        order = labels
    else:
        order = tuple(deletion_order)
        if set(order) != universe or len(order) != len(labels):
            raise ValueError("deletion_order must be a permutation of the universe")
    calls = 0
    if verify_initial:
        calls += 1
        if is_feasible(permission_block_with_closed_set(K, Q, tuple(current))):
            raise ValueError("initial closed set must certify infeasibility")
    if strategy == "sequential":
        removed: list[str] = []
        for label in order:
            if label not in current:
                continue
            trial = current - {label}
            calls += 1
            if not is_feasible(permission_block_with_closed_set(K, Q, tuple(trial))):
                current = trial
                removed.append(label)
        core = tuple(label for label in labels if label in current)
    else:
        # QuickXplain for the monotone predicate
        # conflict(C) := not Phi(U\C).  Background and candidates are kept
        # disjoint; every recursive query is still certified by the same exact
        # physical feasibility oracle.
        def has_conflict(closed: tuple[str, ...]) -> bool:
            nonlocal calls
            calls += 1
            return not is_feasible(permission_block_with_closed_set(K, Q, closed))

        if has_conflict(tuple()):
            raise ValueError("full permission block is infeasible; no repair core exists")

        def qx(
            background: tuple[str, ...],
            delta: tuple[str, ...],
            candidates: tuple[str, ...],
        ) -> tuple[str, ...]:
            if delta and has_conflict(background):
                return tuple()
            if len(candidates) == 1:
                return candidates
            middle = len(candidates) // 2
            first = candidates[:middle]
            second = candidates[middle:]
            second_core = qx(
                tuple(dict.fromkeys(background + first)), first, second)
            first_core = qx(
                tuple(dict.fromkeys(background + second_core)),
                second_core, first)
            return first_core + second_core

        ordered_current = tuple(label for label in order if label in current)
        core_set = set(qx(tuple(), tuple(), ordered_current))
        core = tuple(label for label in labels if label in core_set)
        removed = [label for label in labels if label in current - core_set]
    if not core:
        raise ValueError("full permission block is infeasible; no repair core exists")
    return PermissionConflictCoreResult(core, calls, tuple(removed))


def _minimum_hitting_set_block(
    K: int,
    Q: int,
    conflict_cores: tuple[tuple[str, ...], ...],
    objective_mode: str,
) -> PermissionBlock:
    labels = permission_labels(int(K), int(Q))
    index = {label: position for position, label in enumerate(labels)}
    n = len(labels)
    rows = []
    lower = []
    upper = []
    for i in range(K):
        row = np.zeros(n)
        row[index[f"role:{i}"]] = 1
        row[index[f"uav:{i}"]] = -1
        rows.append(row); lower.append(-np.inf); upper.append(0.0)
    for core in conflict_cores:
        if not core or any(label not in index for label in core):
            raise ValueError("conflict core must be nonempty and inside the universe")
        row = np.zeros(n)
        for label in core:
            row[index[label]] = 1
        rows.append(row); lower.append(1.0); upper.append(np.inf)
    constraints = LinearConstraint(
        np.stack(rows) if rows else np.zeros((0, n)),
        np.asarray(lower), np.asarray(upper))
    if objective_mode == "cardinality":
        objective = np.ones(n)
    elif objective_mode == "support_lex":
        # Exact scalarization of lex(primary support, role support): at most K
        # roles can differ, so K+1 makes one primary unit strictly dominant.
        objective = np.asarray([
            1.0 if label.startswith("role:") else float(K + 1)
            for label in labels], dtype=np.float64)
    else:
        raise ValueError("objective_mode must be 'cardinality' or 'support_lex'")
    objective = objective + 1.0e-6 * np.arange(n, dtype=np.float64)
    solved = milp(
        objective, integrality=np.ones(n, dtype=np.int32),
        bounds=Bounds(np.zeros(n), np.ones(n)), constraints=constraints,
        options={"mip_rel_gap": 0.0})
    if solved.x is None:
        raise RuntimeError("core-guided hitting-set master became infeasible")
    return _block(labels, np.asarray(solved.x))


def objective_layer_common_zeros(
    K: int,
    Q: int,
    conflict_cores: tuple[tuple[str, ...], ...],
    *,
    objective_mode: str = "support_lex",
    scalar_cost_upper: int | None = None,
) -> ObjectiveLayerResult:
    """Return permissions fixed to zero over the entire unperturbed optimum face.

    The tiny deterministic tie-break used to choose ``representative`` is not
    part of the layer cost.  Common-zero tests use the exact integer objective
    and therefore describe every minimum-cost hitting set, not one selected
    vertex.
    """
    labels = permission_labels(int(K), int(Q))
    index = {label: position for position, label in enumerate(labels)}
    n = len(labels)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for i in range(K):
        row = np.zeros(n)
        row[index[f"role:{i}"]] = 1
        row[index[f"uav:{i}"]] = -1
        rows.append(row); lower.append(-np.inf); upper.append(0.0)
    for core in conflict_cores:
        if not core or any(label not in index for label in core):
            raise ValueError("conflict core must be nonempty and inside the universe")
        row = np.zeros(n)
        for label in core:
            row[index[label]] = 1
        rows.append(row); lower.append(1.0); upper.append(np.inf)
    if objective_mode == "cardinality":
        cost = np.ones(n)
    elif objective_mode == "support_lex":
        cost = np.asarray([
            1.0 if label.startswith("role:") else float(K + 1)
            for label in labels], dtype=np.float64)
    else:
        raise ValueError("objective_mode must be 'cardinality' or 'support_lex'")

    def solve(
        objective: np.ndarray,
        extra_rows: tuple[tuple[np.ndarray, float, float], ...] = tuple(),
    ):
        all_rows = rows + [item[0] for item in extra_rows]
        all_lower = lower + [item[1] for item in extra_rows]
        all_upper = upper + [item[2] for item in extra_rows]
        constraint = LinearConstraint(
            np.stack(all_rows) if all_rows else np.zeros((0, n)),
            np.asarray(all_lower), np.asarray(all_upper))
        return milp(
            objective, integrality=np.ones(n, dtype=np.int32),
            bounds=Bounds(np.zeros(n), np.ones(n)), constraints=constraint,
            options={"mip_rel_gap": 0.0})

    optimum = solve(cost)
    if optimum.x is None:
        raise RuntimeError("objective-layer master became infeasible")
    scalar_cost = int(round(float(cost @ np.asarray(optimum.x))))
    zero_cost_upper = (
        scalar_cost if scalar_cost_upper is None else int(scalar_cost_upper))
    if zero_cost_upper < scalar_cost:
        raise ValueError("scalar_cost_upper cannot be below the master optimum")
    optimum_layer = (cost.copy(), -np.inf, float(scalar_cost))
    zero_sublevel = (cost.copy(), -np.inf, float(zero_cost_upper))
    tie = 1.0e-6 * np.arange(n, dtype=np.float64)
    representative_solve = solve(tie, (optimum_layer,))
    if representative_solve.x is None:
        raise RuntimeError("objective-layer representative became infeasible")
    representative = _block(labels, np.asarray(representative_solve.x))
    common_zero: list[str] = []
    solves = 2
    for label in labels:
        force = np.zeros(n); force[index[label]] = 1.0
        probe = solve(np.zeros(n), (zero_sublevel, (force, 1.0, 1.0)))
        solves += 1
        if probe.x is None:
            if int(probe.status) != 2:
                raise RuntimeError("objective-layer zero test was unresolved")
            common_zero.append(label)
    primary = len(representative.uavs) + len(representative.targets)
    secondary = len(representative.role_uavs)
    return ObjectiveLayerResult(
        representative, scalar_cost, (primary, secondary),
        tuple(common_zero), zero_cost_upper, solves)


def _enumerate_objective_layer_blocks(
    K: int,
    Q: int,
    conflict_cores: tuple[tuple[str, ...], ...],
    scalar_cost: int,
    *,
    limit: int,
) -> tuple[PermissionBlock, ...]:
    labels = permission_labels(K, Q)
    index = {label: position for position, label in enumerate(labels)}
    n = len(labels)
    cost = np.asarray([
        1.0 if label.startswith("role:") else float(K + 1)
        for label in labels], dtype=np.float64)
    rows = []
    lower = []
    upper = []
    for i in range(K):
        row = np.zeros(n); row[index[f"role:{i}"]] = 1
        row[index[f"uav:{i}"]] = -1
        rows.append(row); lower.append(-np.inf); upper.append(0.0)
    for core in conflict_cores:
        row = np.zeros(n)
        for label in core:
            row[index[label]] = 1
        rows.append(row); lower.append(1.0); upper.append(np.inf)
    rows.append(cost); lower.append(-np.inf); upper.append(float(scalar_cost))
    found = []
    tie = 1.0e-6 * np.arange(n, dtype=np.float64)
    for _ in range(int(limit)):
        constraint = LinearConstraint(
            np.stack(rows), np.asarray(lower), np.asarray(upper))
        solved = milp(
            tie, integrality=np.ones(n, dtype=np.int32),
            bounds=Bounds(np.zeros(n), np.ones(n)), constraints=constraint,
            options={"mip_rel_gap": 0.0})
        if solved.x is None:
            if int(solved.status) != 2:
                raise RuntimeError("objective-layer enumeration was unresolved")
            break
        selected = np.asarray(solved.x) >= 0.5
        found.append(_block(labels, selected.astype(float)))
        # Exclude exactly this binary support.
        no_good = np.where(selected, -1.0, 1.0)
        rows.append(no_good)
        lower.append(1.0 - float(np.sum(selected)))
        upper.append(np.inf)
    return tuple(found)


def objective_layer_separating_permission_block(
    K: int,
    Q: int,
    oracle: Callable[[PermissionBlock], CertifiedFeasibility],
    *,
    max_iterations: int = 16,
    max_bundle_supports: int = 8,
) -> ObjectiveLayerSeparationResult:
    """Core-guided support search using objective-sublevel separation.

    A sublevel cut is added only when the union of every master support with
    cost at most ``tau`` is certified infeasible.  It therefore raises the
    integer support-lex lower bound above ``tau``.  If the optimum-face union
    is feasible, a bounded bundle combines alternative optimum supports while
    their union remains certified infeasible.  Unresolved oracle states never
    generate cuts.
    """
    labels = permission_labels(K, Q)
    cores: list[tuple[str, ...]] = []
    kinds: list[str] = []
    bounds_trace: list[int] = []
    full_cost = (K + Q) * (K + 1) + K

    def require(status: CertifiedFeasibility) -> CertifiedFeasibility:
        if status is CertifiedFeasibility.UNRESOLVED:
            raise RuntimeError("certified permission oracle was unresolved")
        return status

    for iteration in range(1, int(max_iterations) + 1):
        layer = objective_layer_common_zeros(
            K, Q, tuple(cores), objective_mode="support_lex")
        bounds_trace.append(layer.scalar_cost)
        representative_status = require(oracle(layer.representative))
        if representative_status is CertifiedFeasibility.CERT_FEASIBLE:
            return ObjectiveLayerSeparationResult(
                layer.representative, iteration, tuple(cores), tuple(kinds),
                tuple(bounds_trace), layer.permission_objective)

        face_block = permission_block_with_closed_set(
            K, Q, layer.common_zero_permissions)
        face_status = (
            representative_status if face_block == layer.representative
            else require(oracle(face_block)))
        if face_status is CertifiedFeasibility.CERT_INFEASIBLE:
            low = layer.scalar_cost
            low_zero = layer.common_zero_permissions
            high = full_cost
            high_layer = objective_layer_common_zeros(
                K, Q, tuple(cores), objective_mode="support_lex",
                scalar_cost_upper=high)
            high_block = permission_block_with_closed_set(
                K, Q, high_layer.common_zero_permissions)
            if require(oracle(high_block)) is not CertifiedFeasibility.CERT_FEASIBLE:
                raise RuntimeError("full permission block is certified infeasible")
            while high - low > 1:
                middle = (low + high) // 2
                middle_layer = objective_layer_common_zeros(
                    K, Q, tuple(cores), objective_mode="support_lex",
                    scalar_cost_upper=middle)
                middle_block = permission_block_with_closed_set(
                    K, Q, middle_layer.common_zero_permissions)
                middle_status = require(oracle(middle_block))
                if middle_status is CertifiedFeasibility.CERT_INFEASIBLE:
                    low = middle
                    low_zero = middle_layer.common_zero_permissions
                else:
                    high = middle
            if not low_zero:
                raise RuntimeError("infeasible sublevel produced an empty cut")
            cores.append(low_zero)
            kinds.append("sublevel")
            continue

        supports = _enumerate_objective_layer_blocks(
            K, Q, tuple(cores), layer.scalar_cost,
            limit=max_bundle_supports)
        union = _selected_labels(layer.representative)
        accepted = 1
        for alternative in supports:
            alternative_labels = _selected_labels(alternative)
            if alternative_labels <= union:
                continue
            trial_labels = union | alternative_labels
            trial = _block(
                labels, np.asarray([
                    label in trial_labels for label in labels], dtype=float))
            trial_status = require(oracle(trial))
            if trial_status is CertifiedFeasibility.CERT_INFEASIBLE:
                union = trial_labels
                accepted += 1
        closed = tuple(label for label in labels if label not in union)
        if not closed:
            raise RuntimeError("bundle separation produced an empty cut")
        cores.append(closed)
        kinds.append(f"bundle:{accepted}")
    raise ObjectiveLayerSeparationLimit(
        int(max_iterations), tuple(cores), tuple(kinds), tuple(bounds_trace))


def core_guided_minimum_permission_block(
    K: int,
    Q: int,
    is_feasible: Callable[[PermissionBlock], bool],
    *,
    max_iterations: int = 16,
    deletion_order: tuple[str, ...] | None = None,
    shrink_strategy: str = "sequential",
    objective_mode: str = "cardinality",
) -> CoreGuidedPermissionResult:
    """Return a globally minimum-cardinality permission block via core cuts.

    Every learned core ``C`` satisfies that the largest dependency-valid block
    avoiding ``C`` is infeasible, so every feasible repair must hit ``C``.
    Consequently the first feasible minimum hitting set is globally optimal for
    permission cardinality.  This does not imply minimum realized G4-A closure.
    ``max_iterations`` bounds master candidates; shrink calls are reported
    separately rather than hidden in that budget.
    """
    labels = permission_labels(int(K), int(Q))
    cores: list[tuple[str, ...]] = []
    master_calls = 0
    shrink_calls = 0
    for _iteration in range(int(max_iterations)):
        candidate = _minimum_hitting_set_block(
            K, Q, tuple(cores), objective_mode)
        master_calls += 1
        if is_feasible(candidate):
            cardinality = len(_selected_labels(candidate))
            return CoreGuidedPermissionResult(
                candidate, master_calls + shrink_calls, master_calls,
                shrink_calls, tuple(cores), cardinality,
                (len(candidate.uavs) + len(candidate.targets),
                 len(candidate.role_uavs)))
        chosen = _selected_labels(candidate)
        closed = tuple(label for label in labels if label not in chosen)
        shrunk = shrink_infeasible_permission_core(
            K, Q, closed, is_feasible, verify_initial=False,
            deletion_order=deletion_order, strategy=shrink_strategy)
        shrink_calls += shrunk.oracle_calls
        if shrunk.core in cores:
            raise RuntimeError("oracle produced a duplicate conflict core")
        cores.append(shrunk.core)
    raise CoreGuidedIterationLimit(
        master_calls + shrink_calls, master_calls, shrink_calls, tuple(cores))


def conflict_cut_minimum_permission_block(
    K: int,
    Q: int,
    is_feasible: Callable[[PermissionBlock], bool],
    *,
    forced_permissions: tuple[str, ...] = tuple(),
    max_iterations: int = 10000,
) -> PermissionCutResult:
    """Solve a finite logic-based master using monotone infeasible-set cuts.

    If block ``B`` is infeasible, every feasible block must select at least one
    permission in ``U\\B``.  The resulting cut is exact and requires neither an
    IIS nor a Farkas ray.  The first oracle-feasible master solution is global
    minimum *permission cardinality*; this is not automatically the minimum
    realized structural-intervention cost.
    """
    labels = permission_labels(int(K), int(Q))
    index = {label: position for position, label in enumerate(labels)}
    forced = tuple(sorted(set(forced_permissions)))
    if any(label not in index for label in forced):
        raise ValueError("forced permission is outside the universe")
    n = len(labels)
    # Tiny deterministic perturbation cannot offset a one-unit cardinality gap.
    objective = np.ones(n) + 1.0e-6 * np.arange(n, dtype=np.float64)
    cuts: list[tuple[str, ...]] = []
    calls = 0
    for _iteration in range(int(max_iterations)):
        rows = []
        lower = []
        upper = []
        # role_i <= uav_i.
        for i in range(K):
            row = np.zeros(n)
            row[index[f"role:{i}"]] = 1
            row[index[f"uav:{i}"]] = -1
            rows.append(row); lower.append(-np.inf); upper.append(0.0)
        for label in forced:
            row = np.zeros(n); row[index[label]] = 1
            rows.append(row); lower.append(1.0); upper.append(1.0)
        for conflict in cuts:
            row = np.zeros(n)
            for label in conflict:
                row[index[label]] = 1
            rows.append(row); lower.append(1.0); upper.append(np.inf)
        constraints = LinearConstraint(
            np.stack(rows) if rows else np.zeros((0, n)),
            np.asarray(lower), np.asarray(upper))
        solved = milp(
            objective, integrality=np.ones(n, dtype=np.int32),
            bounds=Bounds(np.zeros(n), np.ones(n)), constraints=constraints,
            options={"mip_rel_gap": 0.0})
        if solved.x is None:
            raise RuntimeError("permission master became infeasible")
        candidate = _block(labels, np.asarray(solved.x))
        calls += 1
        if is_feasible(candidate):
            cardinality = (
                len(candidate.uavs) + len(candidate.targets)
                + len(candidate.role_uavs))
            return PermissionCutResult(
                candidate, calls, tuple(cuts), cardinality)
        chosen = {
            f"target:{q}" for q in candidate.targets
        } | {f"uav:{i}" for i in candidate.uavs} | {
            f"role:{i}" for i in candidate.role_uavs
        }
        conflict = tuple(label for label in labels if label not in chosen)
        if not conflict or conflict in cuts:
            raise RuntimeError("infeasible oracle produced no new valid conflict cut")
        cuts.append(conflict)
    raise PermissionCutIterationLimit(calls, tuple(cuts))
