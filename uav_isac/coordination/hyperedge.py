"""Learning-free local-view negotiation for directed bistatic hyperedges.

One sensing decision is a directed triple ``(tx, rx, target)`` rather than
two independent node-target weights.  Every UAV can run the same deterministic
planner from its own offer plus offers that arrived through its local inbox.
The environment integration activates only endpoint-mutual plans; these pure
functions never inspect global simulator state.
"""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, Tuple, Union

import numpy as np

from uav_isac.physical.geometry import C_LIGHT


Hyperedge = Tuple[int, int, int]


def _sinc_alignment_lower_array(
    center: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """Minimum ``abs(sinc(x-round(x)))`` on interval ``center +/- radius``."""
    x = np.asarray(center, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    left = x - r
    right = x + r
    crosses_half_integer = (
        np.ceil(left - 0.5) <= np.floor(right - 0.5))
    endpoint_distance = np.minimum(0.5, np.maximum(
        np.abs(left - np.round(left)),
        np.abs(right - np.round(right)),
    ))
    distance = np.where(
        (r >= 0.5) | crosses_half_integer,
        0.5,
        endpoint_distance,
    )
    return np.abs(np.sinc(distance))


def _sinc_alignment_upper_array(
    center: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """Maximum ``abs(sinc(x-round(x)))`` on ``center +/- radius``."""
    x = np.asarray(center, dtype=np.float64)
    r = np.asarray(radius, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("sinc interval center/radius must be finite/non-negative")
    left = x - r
    right = x + r
    contains_integer = np.ceil(left) <= np.floor(right)
    endpoint_distance = np.minimum(
        np.abs(left - np.round(left)),
        np.abs(right - np.round(right)),
    )
    distance = np.where(contains_integer, 0.0, endpoint_distance)
    return np.abs(np.sinc(np.minimum(distance, 0.5)))


def deterministic_bottleneck_cost_assignment(
    cost_matrix: np.ndarray,
) -> np.ndarray:
    """Return a deterministic assignment minimizing the largest finite cost."""
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must have shape (K,Q)")
    if np.any(np.isnan(cost)) or np.any(cost < 0.0):
        raise ValueError("assignment costs must be non-negative and not NaN")
    K, Q = cost.shape
    assignment = np.full(K, -1, dtype=np.int64)
    if K == 0 or Q == 0:
        return assignment
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return assignment
    thresholds = np.unique(finite)
    required = min(K, Q)
    lo, hi = 0, len(thresholds) - 1
    best = float(thresholds[-1])
    found = False
    while lo <= hi:
        mid = (lo + hi) // 2
        threshold = float(thresholds[mid])
        blocked = (~np.isfinite(cost) | (cost > threshold)).astype(np.float64)
        rows, cols = linear_sum_assignment(blocked)
        feasible = bool(
            len(rows) == required
            and np.all(np.isfinite(cost[rows, cols]))
            and np.all(cost[rows, cols] <= threshold + 1.0e-12)
        )
        if feasible:
            best = threshold
            found = True
            hi = mid - 1
        else:
            lo = mid + 1
    if not found:
        return assignment
    scale = max(float(np.max(finite)), 1.0)
    penalty = scale * (required + 1)
    restricted = np.where(
        np.isfinite(cost) & (cost <= best + 1.0e-12), cost, penalty)
    tie = 1.0e-12 * (
        np.arange(K, dtype=np.float64)[:, None] * max(Q, 1)
        + np.arange(Q, dtype=np.float64)[None, :]
    )
    rows, cols = linear_sum_assignment(restricted + tie)
    assignment[rows] = cols
    return assignment


def deterministic_bottleneck_matching(
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
) -> np.ndarray:
    """Return a deterministic minimum-bottleneck node-target assignment.

    The primary objective minimizes the maximum travel distance.  A binary
    search over the finite distance thresholds uses bipartite feasibility;
    within the smallest feasible threshold a second assignment minimizes total
    distance.  This matches worst-target sensing geometry better than a greedy
    minimum-edge rule while remaining polynomial-time.
    """
    nodes = np.asarray(node_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_xy, dtype=np.float64)
    if (nodes.ndim != 2 or targets.ndim != 2
            or nodes.shape[1:] != (2,) or targets.shape[1:] != (2,)):
        raise ValueError("node/target positions must have shape (N,2)")
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(targets))):
        raise ValueError("matching positions must be finite")
    distance = np.linalg.norm(
        nodes[:, None, :] - targets[None, :, :], axis=-1)
    return deterministic_bottleneck_cost_assignment(distance)


def role_capacity_bottleneck_assignment(
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
    role_mask: np.ndarray,
    *,
    height_m: float,
    capacity: Union[int, Tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Capacitated Tx/Rx geometric responsibility assignment.

    For each role (the public Tx partition and its Rx complement), solve

        min_{pi_r}  max_q (H^2 + ||x_{pi_r(q)} - z_q||^2)
        s.t.        sum_q 1{pi_r(q) = k} <= c_r

    where ``c_r`` is the per-node capacity of role ``r`` (a single int applies
    to both roles; a ``(tx_capacity, rx_capacity)`` tuple gives per-role
    bounds).  Every target receives exactly one Tx responsibility and one Rx
    responsibility; a node may serve up to its role capacity targets within its
    own role.  Each role subproblem is a bipartite b-matching: rows are
    node-duplicated ``c_r`` times, and a binary search over the finite
    squared-range thresholds tests feasibility with ``linear_sum_assignment``,
    followed by a total-range refinement at the smallest feasible threshold.  A
    deterministic row-major index tie-break makes identical public views
    reproduce identical responsibilities without any global optimizer
    certificate.

    Returns ``(tx_responsibility, rx_responsibility, bottleneck_cost,
    feasible)`` where
    each responsibility array has shape ``(K, Q)`` with 1 marking that node
    ``k`` holds the geometric responsibility for target ``q`` in that role, and
    ``bottleneck_cost`` is the worst-case squared range ``H^2 + ||x - z||^2``
    over both roles after the assignment.  If either role lacks enough total
    capacity, both responsibility arrays are zero, the cost is ``inf``, and
    ``feasible`` is false; an infeasible assignment is never represented by a
    misleading zero bottleneck.
    """
    nodes = np.asarray(node_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_xy, dtype=np.float64)
    roles = np.asarray(role_mask, dtype=bool).reshape(-1)
    from scipy.optimize import linear_sum_assignment
    if (
        nodes.ndim != 2 or targets.ndim != 2
        or nodes.shape[1:] != (2,) or targets.shape[1:] != (2,)
        or roles.shape != (nodes.shape[0],)
        or np.any(~np.isfinite(nodes)) or np.any(~np.isfinite(targets))
    ):
        raise ValueError("invalid role-capacity assignment inputs")
    height = max(float(height_m), 0.0)
    if isinstance(capacity, (tuple, list)):
        tx_cap = max(1, int(capacity[0]))
        rx_cap = max(1, int(capacity[1]))
    else:
        tx_cap = max(1, int(capacity))
        rx_cap = tx_cap
    K = int(nodes.shape[0])
    Q = int(targets.shape[0])
    if K == 0 or Q == 0:
        empty = np.zeros((K, Q), dtype=np.int8)
        return empty, empty, 0.0, True
    if np.all(roles) or not np.any(roles):
        raise ValueError("role-capacity assignment requires both Tx and Rx")

    horizontal_sq = np.sum(
        (nodes[:, None, :] - targets[None, :, :]) ** 2, axis=-1)
    range_sq = height * height + horizontal_sq

    tx_resp = np.zeros((K, Q), dtype=np.int8)
    rx_resp = np.zeros((K, Q), dtype=np.int8)
    worst = 0.0

    def solve_role(role_nodes: np.ndarray, is_tx: bool) -> bool:
        nonlocal worst
        count = int(role_nodes.size)
        if count == 0:
            return False
        cap = tx_cap if is_tx else rx_cap
        rows = np.repeat(role_nodes, cap)
        if rows.size < Q:
            # Not enough duplicated rows to cover every target: the requested
            # capacity is below ceil(Q/|K_r|).  Fail closed (no responsibility
            # for this role) rather than silently dropping targets.
            return False
        role_cost = range_sq[rows]  # (rows, Q)
        finite = role_cost[np.isfinite(role_cost)]
        if finite.size == 0:
            return False
        thresholds = np.unique(finite)
        lo, hi = 0, len(thresholds) - 1
        best = float(thresholds[-1])
        found = False
        while lo <= hi:
            mid = (lo + hi) // 2
            threshold = float(thresholds[mid])
            blocked = (~np.isfinite(role_cost)
                       | (role_cost > threshold)).astype(np.float64)
            assigned_rows, assigned_cols = linear_sum_assignment(blocked)
            feasible = bool(
                assigned_cols.size == Q
                and np.all(role_cost[assigned_rows, assigned_cols]
                           <= threshold + 1.0e-12)
            )
            if feasible:
                best = threshold
                found = True
                hi = mid - 1
            else:
                lo = mid + 1
        if not found:
            return False
        scale = max(float(np.max(finite)), 1.0)
        penalty = scale * (rows.size + 1)
        restricted = np.where(
            np.isfinite(role_cost) & (role_cost <= best + 1.0e-12),
            role_cost,
            penalty,
        )
        tie = 1.0e-12 * (
            np.arange(rows.size, dtype=np.float64)[:, None] * max(Q, 1)
            + np.arange(Q, dtype=np.float64)[None, :]
        )
        assigned_rows, assigned_cols = linear_sum_assignment(
            restricted + tie)
        if (
            assigned_cols.size != Q
            or np.any(restricted[assigned_rows, assigned_cols] >= penalty)
        ):
            return False
        for row, col in zip(assigned_rows, assigned_cols):
            node = int(rows[row])
            if is_tx:
                tx_resp[node, col] = 1
            else:
                rx_resp[node, col] = 1
            worst = max(worst, float(role_cost[row, col]))
        return True

    tx_nodes = np.flatnonzero(roles)
    rx_nodes = np.flatnonzero(~roles)
    tx_feasible = solve_role(tx_nodes, is_tx=True)
    rx_feasible = solve_role(rx_nodes, is_tx=False)
    feasible = bool(
        tx_feasible
        and rx_feasible
        and np.all(tx_resp.sum(axis=0) == 1)
        and np.all(rx_resp.sum(axis=0) == 1)
    )
    if not feasible:
        tx_resp.fill(0)
        rx_resp.fill(0)
        return tx_resp, rx_resp, float("inf"), False
    return tx_resp, rx_resp, worst, True


def gap_coverage_bottleneck_assignment(
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
    tx_role_mask: np.ndarray,
    *,
    critical_radius_m: float,
    capacity: int,
    target_deficit: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic gap-coverage Tx/Rx responsibility assignment.

    The geometric coverage objective asks for every target to have at least
    one Tx endpoint and one Rx endpoint within ``critical_radius_m``.  This is
    a design radius, not a universal P_D threshold: detection also depends on
    power, DD support, fusion and both endpoint ranges.  Targets violating the
    objective form the uncovered set U; each
    uncovered target receives a responsibility from the closest endpoint of
    the missing role (Tx first when both are missing), subject to a per-node
    ``capacity``.  Greedy order is fully deterministic (target id ascending;
    endpoint distance ascending; endpoint id ascending), and a previously
    assigned endpoint is retained when it is still available and still the
    closest -- so identical public views reproduce identical responsibilities
    without any optimizer certificate and without per-frame churn.

    ``target_deficit`` (optional, length Q, larger = weaker) orders the
    assignment so weaker targets receive their Tx and Rx responsibilities
    first (Phase B reinforcement: weak targets are reinforced, saturated
    targets release endpoints when the node capacity is exhausted).

    Returns ``(tx_responsibility, rx_responsibility, uncovered)`` where each
    responsibility array has shape ``(K, Q)`` (1 = node k is responsible for
    target q in that role) and ``uncovered`` is the boolean (Q,) set of
    targets violating C1 under the current public geometry.
    """
    nodes = np.asarray(node_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_xy, dtype=np.float64)
    roles = np.asarray(tx_role_mask, dtype=bool).reshape(-1)
    if (
        nodes.ndim != 2 or targets.ndim != 2
        or nodes.shape[1:] != (2,) or targets.shape[1:] != (2,)
        or roles.shape != (nodes.shape[0],)
        or np.any(~np.isfinite(nodes)) or np.any(~np.isfinite(targets))
    ):
        raise ValueError("invalid gap-coverage assignment inputs")
    r_crit = max(float(critical_radius_m), 1.0e-9)
    cap = max(1, int(capacity))
    K = int(nodes.shape[0])
    Q = int(targets.shape[0])
    if K == 0 or Q == 0:
        empty = np.zeros((K, Q), dtype=np.int8)
        return empty, empty, np.zeros(Q, dtype=bool)
    if np.all(roles) or not np.any(roles):
        raise ValueError("gap-coverage assignment requires both Tx and Rx")

    horizontal = np.linalg.norm(
        nodes[:, None, :] - targets[None, :, :], axis=-1)  # (K, Q)
    tx_nodes = np.flatnonzero(roles)
    rx_nodes = np.flatnonzero(~roles)
    nearest_tx = np.full(Q, np.inf)
    nearest_rx = np.full(Q, np.inf)
    if tx_nodes.size:
        nearest_tx = np.min(horizontal[tx_nodes], axis=0)
    if rx_nodes.size:
        nearest_rx = np.min(horizontal[rx_nodes], axis=0)
    uncovered = (nearest_tx > r_crit) | (nearest_rx > r_crit)

    if target_deficit is not None:
        deficit = np.asarray(target_deficit, dtype=np.float64).reshape(-1)
        if deficit.shape != (Q,):
            raise ValueError("target_deficit must have shape (Q,)")
        order = np.argsort(-deficit, kind="stable")
    else:
        order = np.arange(Q)

    tx_resp = np.zeros((K, Q), dtype=np.int8)
    rx_resp = np.zeros((K, Q), dtype=np.int8)
    node_load = np.zeros(K, dtype=np.int64)

    def assign(target: int, want_tx: bool) -> bool:
        nonlocal node_load
        pool = tx_nodes if want_tx else rx_nodes
        if pool.size == 0:
            return False
        distance = horizontal[pool, target]
        order_pool = pool[np.argsort(distance, kind="stable")]
        for node in order_pool:
            if node_load[int(node)] >= cap:
                continue
            if want_tx:
                tx_resp[int(node), target] = 1
            else:
                rx_resp[int(node), target] = 1
            node_load[int(node)] += 1
            return True
        return False

    # 1) Recovery responsibilities first (weakest targets first): every
    #    uncovered target receives the missing role(s).  This does not make
    #    the geometry covered instantaneously; it assigns movers that reduce
    #    the deficit over subsequent safe steps.
    for target in order:
        target = int(target)
        if not uncovered[target]:
            continue
        missing_tx = nearest_tx[target] > r_crit
        missing_rx = nearest_rx[target] > r_crit
        if missing_tx:
            assign(target, want_tx=True)
        if missing_rx:
            assign(target, want_tx=False)
    # 2) Then complete the assignment (weakest first) so no UAV is idle:
    #    each target keeps its nearest Tx and Rx endpoint within the
    #    capacity limits.
    for target in order:
        target = int(target)
        if tx_resp[:, target].sum() == 0:
            assign(target, want_tx=True)
        if rx_resp[:, target].sum() == 0:
            assign(target, want_tx=False)
    return tx_resp, rx_resp, uncovered


def role_aware_bistatic_movement_cost(
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
    tx_role_mask: np.ndarray,
    *,
    height_m: float,
    movement_step_m: float,
    standoff_m: float = 0.0,
    complement_exponent: float = 1.0,
) -> np.ndarray:
    """One-step bistatic range-product cost using the nearest opposite role.

    For mover ``k`` and target ``q``, the cost is
    ``R_kq,next^2 * min_{l: role_l != role_k} R_lq^2``. Since free-space
    bistatic gain is proportional to the inverse of this product, minimizing
    the maximum assigned cost aligns L3 responsibility with worst-target
    sensing geometry. Every row uses only public positions, fixed public roles
    and the common target map; no optimizer certificate is required.
    """
    nodes = np.asarray(node_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_xy, dtype=np.float64)
    roles = np.asarray(tx_role_mask, dtype=bool).reshape(-1)
    if (
        nodes.ndim != 2 or targets.ndim != 2
        or nodes.shape[1:] != (2,) or targets.shape[1:] != (2,)
        or roles.shape != (nodes.shape[0],)
        or np.any(~np.isfinite(nodes)) or np.any(~np.isfinite(targets))
    ):
        raise ValueError("invalid role-aware movement inputs")
    if nodes.shape[0] > 0 and (np.all(roles) or not np.any(roles)):
        raise ValueError("role-aware bistatic movement requires both Tx and Rx")
    height = max(float(height_m), 0.0)
    step = max(float(movement_step_m), 0.0)
    standoff = max(float(standoff_m), 0.0)
    exponent = float(complement_exponent)
    if not np.isfinite(exponent) or not 0.0 <= exponent <= 1.0:
        raise ValueError("complement_exponent must lie in [0,1]")
    horizontal = np.linalg.norm(
        nodes[:, None, :] - targets[None, :, :], axis=-1)
    next_horizontal = np.maximum(horizontal - step, standoff)
    own_range_sq = height * height + next_horizontal * next_horizontal
    current_range_sq = height * height + horizontal * horizontal
    complement_range_sq = np.zeros_like(current_range_sq)
    for node in range(nodes.shape[0]):
        complement = np.flatnonzero(roles != roles[node])
        complement_range_sq[node] = np.min(
            current_range_sq[complement], axis=0)
    return own_range_sq * np.power(complement_range_sq, exponent)


def gauss_southwell_bistatic_geometry_step(
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
    tx_role_mask: np.ndarray,
    *,
    height_m: float,
    maximum_step_m: float,
    standoff_m: float = 0.0,
    minimum_log_improvement: float = 0.0,
    eligible_node_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int] | None, float, float, float]:
    """Choose the single UAV-target radial step with largest tail gain.

    The public range-product potential is

    ``Phi(X)=max_q min_Tx R_iq^2 * min_Rx R_jq^2``.

    Every admissible coordinate ``(node,target)`` moves that node at most one
    step toward the target.  The Gauss--Southwell rule selects the coordinate
    with the largest positive ``log(Phi_before/Phi_after)``; deterministic
    node/target iteration resolves ties.  The function does not inspect target
    truth, power certificates, or optimizer state.  Collision safety is a
    lexicographically higher layer and must project the returned action before
    execution.
    """
    nodes = np.asarray(node_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_xy, dtype=np.float64)
    roles = np.asarray(tx_role_mask, dtype=bool).reshape(-1)
    eligible = (
        np.ones(nodes.shape[0], dtype=bool)
        if eligible_node_mask is None
        else np.asarray(eligible_node_mask, dtype=bool).reshape(-1)
    )
    if (
        nodes.ndim != 2 or targets.ndim != 2
        or nodes.shape[1:] != (2,) or targets.shape[1:] != (2,)
        or roles.shape != (nodes.shape[0],)
        or eligible.shape != (nodes.shape[0],)
        or np.any(~np.isfinite(nodes)) or np.any(~np.isfinite(targets))
    ):
        raise ValueError('invalid Gauss-Southwell geometry inputs')
    if nodes.shape[0] == 0 or targets.shape[0] == 0:
        return np.zeros_like(nodes), None, 0.0, 0.0, 0.0
    if np.all(roles) or not np.any(roles):
        raise ValueError('Gauss-Southwell geometry requires Tx and Rx roles')
    height = float(height_m)
    step = float(maximum_step_m)
    standoff = float(standoff_m)
    threshold = float(minimum_log_improvement)
    if not (
        np.isfinite(height) and height >= 0.0
        and np.isfinite(step) and step >= 0.0
        and np.isfinite(standoff) and standoff >= 0.0
        and np.isfinite(threshold) and threshold >= 0.0
    ):
        raise ValueError('invalid Gauss-Southwell scalar parameter')

    def potential(positions: np.ndarray) -> float:
        horizontal_sq = np.sum(
            (positions[:, None, :] - targets[None, :, :]) ** 2,
            axis=-1,
        )
        range_sq = height * height + horizontal_sq
        nearest_tx = np.min(range_sq[roles], axis=0)
        nearest_rx = np.min(range_sq[~roles], axis=0)
        return float(np.max(nearest_tx * nearest_rx))

    base = potential(nodes)
    desired = np.zeros_like(nodes)
    best_coordinate = None
    best_improvement = threshold
    best_potential = base
    for node in range(nodes.shape[0]):
        if not eligible[node]:
            continue
        for target in range(targets.shape[0]):
            radial = targets[target] - nodes[node]
            distance = float(np.linalg.norm(radial))
            travel = min(step, max(distance - standoff, 0.0))
            if travel <= 1.0e-12:
                continue
            delta = travel * radial / max(distance, 1.0e-12)
            candidate = nodes.copy()
            candidate[node] += delta
            candidate_potential = potential(candidate)
            improvement = float(np.log(
                max(base, 1.0e-300)
                / max(candidate_potential, 1.0e-300)))
            if improvement > best_improvement + 1.0e-12:
                best_coordinate = (int(node), int(target))
                best_improvement = improvement
                best_potential = candidate_potential
                desired[:] = 0.0
                desired[node] = delta
    if best_coordinate is None:
        return desired, None, 0.0, base, base
    return desired, best_coordinate, best_improvement, base, best_potential


def gauss_southwell_bistatic_geometry_sweep(
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
    tx_role_mask: np.ndarray,
    *,
    height_m: float,
    maximum_step_m: float,
    standoff_m: float = 0.0,
    minimum_log_improvement: float = 0.0,
    maximum_selected_nodes: int | None = None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], float, float, float]:
    """Greedily sweep distinct UAV coordinates in one geometry block.

    Each inner iteration applies the one-coordinate Gauss--Southwell rule to
    the updated public geometry while excluding UAVs already selected in this
    frame. Thus every accepted inner step strictly decreases the same public
    tail potential, every UAV moves at most once, and a K-UAV fleet does not
    suffer the 1/K movement-bandwidth collapse of a single-coordinate frame.
    The returned whole-view action still requires the external d_safe shield.
    """
    nodes = np.asarray(node_positions_xy, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1:] != (2,):
        raise ValueError('invalid Gauss-Southwell sweep node positions')
    limit = (
        int(nodes.shape[0])
        if maximum_selected_nodes is None
        else int(maximum_selected_nodes)
    )
    if limit < 0:
        raise ValueError('maximum_selected_nodes must be nonnegative')
    limit = min(limit, int(nodes.shape[0]))
    working = nodes.copy()
    desired = np.zeros_like(nodes)
    eligible = np.ones(nodes.shape[0], dtype=bool)
    coordinates = []
    total_gain = 0.0
    initial_potential = None
    final_potential = None
    for _ in range(limit):
        delta, coordinate, gain, before, after = (
            gauss_southwell_bistatic_geometry_step(
                working,
                target_positions_xy,
                tx_role_mask,
                height_m=height_m,
                maximum_step_m=maximum_step_m,
                standoff_m=standoff_m,
                minimum_log_improvement=minimum_log_improvement,
                eligible_node_mask=eligible,
            ))
        if initial_potential is None:
            initial_potential = float(before)
            final_potential = float(before)
        if coordinate is None:
            break
        node, _target = coordinate
        desired[node] = delta[node]
        working[node] += delta[node]
        eligible[node] = False
        coordinates.append(coordinate)
        total_gain += float(gain)
        final_potential = float(after)
    if initial_potential is None:
        # limit=0 still validates the physical inputs and exposes its potential.
        _, _, _, initial_potential, final_potential = (
            gauss_southwell_bistatic_geometry_step(
                working,
                target_positions_xy,
                tx_role_mask,
                height_m=height_m,
                maximum_step_m=maximum_step_m,
                standoff_m=standoff_m,
                minimum_log_improvement=minimum_log_improvement,
                eligible_node_mask=np.zeros(nodes.shape[0], dtype=bool),
            ))
    return (
        desired,
        tuple(coordinates),
        float(total_gain),
        float(initial_potential),
        float(final_potential),
    )


def annular_alternating_movement_delta(
    node_position_xy: np.ndarray,
    target_position_xy: np.ndarray,
    complement_position_xy: np.ndarray,
    *,
    phase: str,
    inner_radius_m: float,
    outer_radius_m: float,
    maximum_step_m: float,
    desired_bistatic_angle_deg: float = 90.0,
    orientation_sign: int = 1,
) -> tuple[np.ndarray, str]:
    """One local block-coordinate step on an annular bistatic geometry.

    Range infeasibility has lexicographic priority in both phases: a node
    outside the annulus moves toward its nearest boundary.  Inside the
    annulus, the range phase holds radius while the strategy phase rotates the
    node about the target toward a desired signed bistatic angle relative to
    its nearest opposite-role complement.  The rotation preserves range
    exactly, so strategy improvement cannot silently become further radial
    approach.

    The function consumes one node, one target and one public complement only;
    it never reads a fleet-wide target-quality vector.
    """
    node = np.asarray(node_position_xy, dtype=np.float64).reshape(-1)
    target = np.asarray(target_position_xy, dtype=np.float64).reshape(-1)
    complement = np.asarray(
        complement_position_xy, dtype=np.float64).reshape(-1)
    if node.shape != (2,) or target.shape != (2,) or complement.shape != (2,):
        raise ValueError("annular movement inputs must be planar vectors")
    if not (
        np.all(np.isfinite(node))
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(complement))
    ):
        raise ValueError("annular movement inputs must be finite")
    inner = float(inner_radius_m)
    outer = float(outer_radius_m)
    step = float(maximum_step_m)
    angle = float(desired_bistatic_angle_deg)
    normalized_phase = str(phase).strip().lower()
    if not (0.0 <= inner < outer and step >= 0.0):
        raise ValueError("annular radii require 0 <= inner < outer")
    if not 0.0 < angle < 180.0:
        raise ValueError("desired bistatic angle must lie in (0,180) degrees")
    if normalized_phase not in {"range", "strategy"}:
        raise ValueError("phase must be range or strategy")

    radial = node - target
    radius = float(np.linalg.norm(radial))
    if radius <= 1.0e-12 or step <= 0.0:
        return np.zeros(2, dtype=np.float64), "degenerate_hold"
    radial_unit = radial / radius
    if radius > outer + 1.0e-12:
        travel = min(step, radius - outer)
        return -travel * radial_unit, "far_recovery"
    if radius < inner - 1.0e-12:
        travel = min(step, inner - radius)
        return travel * radial_unit, "near_recovery"
    if normalized_phase == "range":
        return np.zeros(2, dtype=np.float64), "range_hold"

    complement_radial = complement - target
    complement_radius = float(np.linalg.norm(complement_radial))
    if complement_radius <= 1.0e-12:
        return np.zeros(2, dtype=np.float64), "strategy_hold"
    cross = float(
        complement_radial[0] * radial[1]
        - complement_radial[1] * radial[0])
    dot = float(np.dot(complement_radial, radial))
    signed_angle = float(np.arctan2(cross, dot))
    desired_abs = float(np.deg2rad(angle))
    if abs(signed_angle) <= 1.0e-12:
        desired_signed = desired_abs * (1.0 if int(orientation_sign) >= 0 else -1.0)
    else:
        desired_signed = desired_abs * np.sign(signed_angle)
    angle_error = float(np.arctan2(
        np.sin(desired_signed - signed_angle),
        np.cos(desired_signed - signed_angle),
    ))
    maximum_angle_step = step / max(radius, 1.0e-12)
    rotation = float(np.clip(
        angle_error, -maximum_angle_step, maximum_angle_step))
    if abs(rotation) <= 1.0e-12:
        return np.zeros(2, dtype=np.float64), "strategy_hold"
    cosine, sine = float(np.cos(rotation)), float(np.sin(rotation))
    rotated = np.asarray([
        cosine * radial[0] - sine * radial[1],
        sine * radial[0] + cosine * radial[1],
    ])
    return rotated - radial, "strategy_tangent"


def bistatic_geometry_tail_ratio(
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
    tx_role_mask: np.ndarray,
    *,
    height_m: float,
) -> float:
    """Return a scale-free worst/median bistatic range-product ratio.

    For each target, the public-geometry proxy is the product of the squared
    ranges to its nearest Tx and nearest Rx.  The maximum divided by the
    median is invariant to a common spatial scaling (apart from the physical
    height term), so it detects a genuine isolated bistatic tail rather than
    merely reacting to a larger deployment region.
    """
    nodes = np.asarray(node_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_xy, dtype=np.float64)
    roles = np.asarray(tx_role_mask, dtype=bool).reshape(-1)
    if (
        nodes.ndim != 2 or targets.ndim != 2
        or nodes.shape[1:] != (2,) or targets.shape[1:] != (2,)
        or roles.shape != (nodes.shape[0],)
        or np.any(~np.isfinite(nodes)) or np.any(~np.isfinite(targets))
    ):
        raise ValueError("invalid bistatic tail-ratio inputs")
    if targets.shape[0] == 0:
        return 1.0
    if nodes.shape[0] == 0 or np.all(roles) or not np.any(roles):
        raise ValueError("bistatic tail ratio requires both Tx and Rx")
    height = max(float(height_m), 0.0)
    range_sq = (
        height * height
        + np.sum((nodes[:, None, :] - targets[None, :, :]) ** 2, axis=-1)
    )
    target_product = (
        np.min(range_sq[roles], axis=0)
        * np.min(range_sq[~roles], axis=0)
    )
    median = float(np.median(target_product))
    return float(np.max(target_product) / max(median, np.finfo(float).tiny))


def project_pairwise_safe_movement(
    node_positions_xy: np.ndarray,
    desired_delta_xy: np.ndarray,
    *,
    minimum_distance_m: float,
    maximum_step_m: float,
    area_size_xy: Tuple[float, float],
    return_diagnostics: bool = False,
    outside_invariant_recovery: bool = False,
    recovery_gain: float = 1.0,
    independently_composable: bool = False,
    analytic_composable_projection: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, object]]]:
    """Project fleet movement onto a conservative one-step safety set.

    For a nearby pair with relative position ``r_ij`` and relative movement
    ``du_ij``, the affine constraint

    ``r_ij.T @ du_ij >= (d_safe**2 - ||r_ij||**2) / 2``

    is a sufficient affine condition for safe next-frame endpoints because
    ``||r_ij + du_ij||^2`` contains the additional non-negative
    ``||du_ij||^2`` term. Pairs farther than ``d_safe + 2*max_step`` need
    no constraint because bounded movements cannot reach the unsafe set in one
    frame.  The strictly convex least-change objective preserves the sensing
    movement whenever it is already safe.  Per-node Euclidean speed balls and
    rectangular flight bounds are enforced exactly.  With endpoint-split
    constraints, ``analytic_composable_projection`` exploits their Cartesian
    product structure and solves one exact two-dimensional projection per
    node.  Its finite candidate set contains affine-boundary projections,
    affine intersections and affine/speed-circle intersections.  When zero is
    feasible, the projection variational inequality additionally proves the
    speed ball redundant.  An empty candidate set certifies infeasibility and
    returns the same fail-closed hold without an iterative SLSQP retry.
    """
    from time import perf_counter

    started = perf_counter()
    initially_safe = True
    minimum_initial_distance = float('inf')
    recovery_pair_count = 0
    projection_solver = 'certified_noop'
    analytic_fallback = False

    def finish(
        value: np.ndarray,
        *,
        intervened: bool,
        fail_closed: bool,
        constraint_count: int,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, object]]]:
        if not return_diagnostics:
            return value
        return value, {
            'intervened': bool(intervened),
            'fail_closed': bool(fail_closed),
            'solve_time_s': float(perf_counter() - started),
            'pairwise_constraint_count': int(constraint_count),
            'initially_safe': bool(initially_safe),
            'minimum_initial_distance_m': float(minimum_initial_distance),
            'recovery_pair_count': int(recovery_pair_count),
            'projection_solver': str(projection_solver),
            'analytic_fallback': bool(analytic_fallback),
        }

    positions = np.asarray(node_positions_xy, dtype=np.float64)
    desired = np.asarray(desired_delta_xy, dtype=np.float64)
    if (
        positions.ndim != 2 or positions.shape[1:] != (2,)
        or desired.shape != positions.shape
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(desired))
    ):
        raise ValueError("invalid pairwise movement projection inputs")
    distance = max(float(minimum_distance_m), 0.0)
    step = max(float(maximum_step_m), 0.0)
    area = np.asarray(area_size_xy, dtype=np.float64).reshape(-1)
    if area.shape != (2,) or np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("area_size_xy must contain two positive finite values")
    if positions.shape[0] == 0 or step <= 0.0:
        stopped = np.zeros_like(desired)
        return finish(
            stopped,
            intervened=bool(np.any(np.abs(stopped - desired) > 1.0e-12)),
            fail_closed=False,
            constraint_count=0,
        )

    desired = desired.copy()
    desired_norm = np.linalg.norm(desired, axis=1)
    scale = np.minimum(1.0, step / np.maximum(desired_norm, 1.0e-15))
    desired *= scale[:, None]
    original_desired = desired.copy()
    K = positions.shape[0]
    if K >= 2:
        initial_pair_distance = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=-1,
        )[np.triu_indices(K, k=1)]
        minimum_initial_distance = float(np.min(initial_pair_distance))
        initially_safe = bool(
            minimum_initial_distance >= distance - 1.0e-9)
    rows = []
    lower = []
    recovery_bias = np.zeros_like(desired)
    influence = distance + 2.0 * step
    for i in range(K):
        for j in range(i + 1, K):
            relative = positions[i] - positions[j]
            current = float(np.linalg.norm(relative))
            if current > influence + 1.0e-12:
                continue
            outside = bool(current < distance - 1.0e-9)
            if outside and outside_invariant_recovery:
                # Zero is feasible and keeps the public squared distance from
                # decreasing. A truncated gradient of the pairwise shortfall
                # potential biases the least-change objective toward gradual
                # recovery without demanding an unreachable one-frame jump.
                pair_lower = 0.0
                recovery_pair_count += 1
                if current > 1.0e-12:
                    unit = relative / current
                    # First cancel the desired relative velocity that closes
                    # this pair, then add a bounded shortfall-gradient step.
                    # Without the cancellation term, a large sensing command
                    # can dominate the recovery bias and the QP merely lands
                    # on the non-decreasing-distance boundary.
                    desired_relative = desired[i] - desired[j]
                    closing_speed = max(
                        0.0, -float(unit @ desired_relative))
                    recovery_speed = (
                        max(float(recovery_gain), 0.0)
                        * step
                        * (distance - current) / max(distance, 1.0e-12)
                    )
                    magnitude = closing_speed + recovery_speed
                    recovery_bias[i] += 0.5 * magnitude * unit
                    recovery_bias[j] -= 0.5 * magnitude * unit
            else:
                pair_lower = 0.5 * (
                    distance * distance - current * current)
            if independently_composable:
                # Split the affine barrier budget equally between endpoints:
                #   r^T u_i >= b/2,  (-r)^T u_j >= b/2.
                # Summing locally verified endpoint actions recovers the joint
                # barrier r^T(u_i-u_j)>=b. A stale endpoint's zero action is
                # also feasible because b<=0 inside the invariant set and the
                # recovery rule sets b=0 outside it.
                row_i = np.zeros((K, 2), dtype=np.float64)
                row_j = np.zeros((K, 2), dtype=np.float64)
                row_i[i] = relative
                row_j[j] = -relative
                rows.extend((row_i.reshape(-1), row_j.reshape(-1)))
                lower.extend((0.5 * pair_lower, 0.5 * pair_lower))
            else:
                row = np.zeros((K, 2), dtype=np.float64)
                row[i] = relative
                row[j] = -relative
                rows.append(row.reshape(-1))
                lower.append(pair_lower)

    recovery_reference = desired + recovery_bias
    recovery_norm = np.linalg.norm(recovery_reference, axis=1)
    recovery_scale = np.minimum(
        1.0, step / np.maximum(recovery_norm, 1.0e-15))
    recovery_reference *= recovery_scale[:, None]

    def satisfies_affine(value: np.ndarray, tol: float = 1.0e-8) -> bool:
        if not rows:
            return True
        return bool(np.all(
            np.asarray(rows) @ value.reshape(-1)
            >= np.asarray(lower) - tol))

    next_position = positions + desired
    if K >= 2:
        desired_pair_distance = np.linalg.norm(
            next_position[:, None, :] - next_position[None, :, :],
            axis=-1,
        )[np.triu_indices(K, k=1)]
    else:
        desired_pair_distance = np.asarray([np.inf])
    desired_feasible = bool(
        np.all(desired_pair_distance >= distance - 1.0e-9)
        and satisfies_affine(desired)
        and np.all(next_position >= -1.0e-9)
        and np.all(next_position <= area[None, :] + 1.0e-9)
    )
    if desired_feasible and recovery_pair_count == 0:
        return finish(
            desired,
            intervened=False,
            fail_closed=False,
            constraint_count=len(rows),
        )

    x0 = np.zeros_like(desired).reshape(-1)
    lower_bound = np.maximum(-step, -positions).reshape(-1)
    upper_bound = np.minimum(step, area[None, :] - positions).reshape(-1)
    matrix = np.asarray(rows, dtype=np.float64).reshape(-1, 2 * K)
    rhs = np.asarray(lower, dtype=np.float64)

    # When every affine/bound constraint contains zero and the reference is
    # inside the speed balls, those balls are mathematically redundant.  For
    # C closed, convex and 0 in C, the projection variational inequality gives
    #   <x-P_C(x), -P_C(x)> <= 0,
    # hence ||P_C(x)||^2 <= <x,P_C(x)> and ||P_C(x)|| <= ||x||.
    # The guarded reduction removes K nonlinear constraints without changing
    # the optimizer. Defensive states that violate any premise keep them.
    reduced_linear_qp = bool(
        analytic_composable_projection
        and independently_composable
        and np.all(rhs <= 1.0e-12)
        and np.all(lower_bound <= 1.0e-12)
        and np.all(upper_bound >= -1.0e-12)
        and np.all(np.linalg.norm(recovery_reference, axis=1) <= step + 1.0e-12)
    )

    projected: np.ndarray | None = None
    separable_2d_qp = bool(
        analytic_composable_projection and independently_composable)

    if separable_2d_qp:
        projection_solver = 'separable_2d_qp'
        # Endpoint-split barrier rows contain variables from exactly one node,
        # so the fleet projection is the Cartesian product of K two-dimensional
        # convex sets.  Their boundaries are affine segments and the speed
        # circle.  The closest point is therefore the reference itself, a
        # projection onto one boundary, or an intersection of two boundaries.
        # Enumerating those candidates is an exact replacement for the
        # separable 32-D SLSQP, including positive AoI safety margins where
        # the speed ball is not redundant.
        projected_rows = np.zeros((K, 2), dtype=np.float64)
        analytic_ok = True
        feasibility_tolerance = 1.0e-9
        for node in range(K):
            local_matrix = matrix[:, 2 * node:2 * node + 2]
            active_rows = np.linalg.norm(local_matrix, axis=1) > 1.0e-15
            local_matrix = local_matrix[active_rows]
            local_rhs = rhs[active_rows]
            local_matrix = np.vstack((
                local_matrix,
                np.asarray([
                    [1.0, 0.0], [-1.0, 0.0],
                    [0.0, 1.0], [0.0, -1.0],
                ], dtype=np.float64),
            ))
            local_rhs = np.concatenate((
                local_rhs,
                np.asarray([
                    lower_bound[2 * node], -upper_bound[2 * node],
                    lower_bound[2 * node + 1], -upper_bound[2 * node + 1],
                ], dtype=np.float64),
            ))
            reference = recovery_reference[node]

            def local_feasible(value: np.ndarray) -> bool:
                return bool(
                    np.all(
                        local_matrix @ value
                        >= local_rhs - feasibility_tolerance)
                    and np.linalg.norm(value) <= step + feasibility_tolerance
                )

            candidates = []
            if local_feasible(reference):
                candidates.append(reference.copy())
            for row, bound in zip(local_matrix, local_rhs):
                norm_sq = float(row @ row)
                if norm_sq <= 1.0e-24:
                    continue
                candidate = reference + (
                    (float(bound) - float(row @ reference)) / norm_sq
                ) * row
                if local_feasible(candidate):
                    candidates.append(candidate)
                # Intersections between this affine boundary and the speed
                # circle cover optima where both constraints are active.
                closest_to_origin = float(bound) * row / norm_sq
                radius_sq = float(step * step - (
                    closest_to_origin @ closest_to_origin))
                if radius_sq >= -1.0e-10:
                    tangent = np.asarray([-row[1], row[0]]) / np.sqrt(norm_sq)
                    offset = np.sqrt(max(radius_sq, 0.0)) * tangent
                    for circle_candidate in (
                        closest_to_origin + offset,
                        closest_to_origin - offset,
                    ):
                        if local_feasible(circle_candidate):
                            candidates.append(circle_candidate)
            for first in range(local_matrix.shape[0]):
                for second in range(first + 1, local_matrix.shape[0]):
                    pair = local_matrix[[first, second]]
                    determinant = float(np.linalg.det(pair))
                    if abs(determinant) <= 1.0e-12:
                        continue
                    candidate = np.linalg.solve(
                        pair, local_rhs[[first, second]])
                    if local_feasible(candidate):
                        candidates.append(candidate)
            if not candidates:
                analytic_ok = False
                break
            projected_rows[node] = min(
                candidates,
                key=lambda value: float(np.sum((value - reference) ** 2)),
            )
        if analytic_ok:
            projected = projected_rows
        else:
            # The two-dimensional candidate set is exhaustive for an
            # intersection of affine half-planes, a box and a disk.  No
            # candidate therefore certifies that the endpoint-split problem
            # is infeasible within this frame's speed bound.  Preserve the
            # existing fail-closed zero action without spending a full SLSQP
            # iteration budget to rediscover the same infeasibility.
            projection_solver = 'separable_2d_infeasible'
            projected = np.zeros((K, 2), dtype=np.float64)

    if projected is None:
        from scipy.optimize import minimize

        projection_solver = (
            'reduced_linear_qp' if reduced_linear_qp
            else ('slsqp_fallback' if analytic_fallback else 'slsqp'))
        constraints = []
        if rows:
            constraints.append({
                'type': 'ineq',
                'fun': lambda value, A=matrix, b=rhs: A @ value - b,
                'jac': lambda value, A=matrix, b=rhs: A,
            })
        if not reduced_linear_qp:
            constraints.append({
                'type': 'ineq',
                'fun': lambda value: (
                    step * step
                    - np.sum(value.reshape(K, 2) ** 2, axis=1)),
                'jac': lambda value: np.asarray([
                    np.concatenate([
                        np.zeros(2 * node),
                        -2.0 * value.reshape(K, 2)[node],
                        np.zeros(2 * (K - node - 1)),
                    ])
                    for node in range(K)
                ]),
            })
        result = minimize(
            lambda value: 0.5 * float(np.sum(
                (value - recovery_reference.reshape(-1)) ** 2)),
            x0,
            jac=lambda value: value - recovery_reference.reshape(-1),
            bounds=list(zip(lower_bound, upper_bound)),
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1.0e-10, 'maxiter': 200, 'disp': False},
        )
        projected = np.asarray(result.x, dtype=np.float64).reshape(K, 2)
    projected_next = positions + projected
    if K >= 2:
        projected_pair_distance = np.linalg.norm(
            projected_next[:, None, :] - projected_next[None, :, :],
            axis=-1,
        )[np.triu_indices(K, k=1)]
    else:
        projected_pair_distance = np.asarray([np.inf])
    if K >= 2:
        required_pair_distance = np.where(
            (initial_pair_distance < distance)
            & bool(outside_invariant_recovery),
            initial_pair_distance,
            distance,
        )
    else:
        required_pair_distance = np.asarray([np.inf])
    feasible = bool(
        satisfies_affine(projected, tol=1.0e-6)
        and np.all(np.linalg.norm(projected, axis=1) <= step + 1.0e-6)
        and np.all(
            projected_pair_distance >= required_pair_distance - 1.0e-6)
        and np.all(projected_next >= -1.0e-6)
        and np.all(projected_next <= area[None, :] + 1.0e-6)
    )
    if not feasible:
        # Numerical optimization is not allowed to crash the flight loop or
        # release the unsafe desired action.  Holding position is the
        # deterministic fail-closed action: it preserves every currently safe
        # physical separation and never increases the violation of an already
        # unsafe pair.  A later frame may retry the least-change projection.
        return finish(
            np.zeros_like(desired),
            intervened=True,
            fail_closed=True,
            constraint_count=len(rows),
        )
    return finish(
        projected,
        intervened=bool(np.any(
            np.abs(projected - original_desired) > 1.0e-9)),
        fail_closed=False,
        constraint_count=len(rows),
    )


def is_pairwise_safe_movement(
    node_positions_xy: np.ndarray,
    desired_delta_xy: np.ndarray,
    *,
    minimum_distance_m: float,
    maximum_step_m: float,
    area_size_xy: Tuple[float, float],
    outside_invariant_recovery: bool = False,
    independently_composable: bool = False,
    analytic_composable_projection: bool = False,
) -> bool:
    """Return whether the safety projector is provably an exact no-op.

    This is the analytic feasibility predicate used by
    :func:`project_pairwise_safe_movement`: endpoint separation, affine
    barrier, per-node speed and flight-area constraints must all hold.  When
    recovery is requested from outside the invariant set, the answer is
    deliberately false because the projector also applies a recovery bias.
    ``analytic_composable_projection`` does not alter feasibility; it is
    accepted so callers can share one safety-configuration mapping with the
    projector.

    The function is useful for candidate envelopes: if every candidate is a
    certified no-op, they may be scored before projection and only the winner
    needs to pass through the projector.  This removes redundant optimization
    without weakening the post-projection ordering guarantee.
    """
    positions = np.asarray(node_positions_xy, dtype=np.float64)
    desired = np.asarray(desired_delta_xy, dtype=np.float64)
    if (
        positions.ndim != 2 or positions.shape[1:] != (2,)
        or desired.shape != positions.shape
        or np.any(~np.isfinite(positions))
        or np.any(~np.isfinite(desired))
    ):
        raise ValueError("invalid pairwise movement feasibility inputs")
    distance = max(float(minimum_distance_m), 0.0)
    step = max(float(maximum_step_m), 0.0)
    area = np.asarray(area_size_xy, dtype=np.float64).reshape(-1)
    if area.shape != (2,) or np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("area_size_xy must contain two positive finite values")
    # The projector scales every norm strictly greater than ``step``.  Do not
    # admit an epsilon band here: this predicate certifies an *exact* no-op,
    # not merely a numerically close action.
    if np.any(np.linalg.norm(desired, axis=1) > step):
        return False
    next_position = positions + desired
    if (
        np.any(next_position < -1.0e-9)
        or np.any(next_position > area[None, :] + 1.0e-9)
    ):
        return False

    K = positions.shape[0]
    influence = distance + 2.0 * step
    for i in range(K):
        for j in range(i + 1, K):
            relative = positions[i] - positions[j]
            current = float(np.linalg.norm(relative))
            if (
                outside_invariant_recovery
                and current < distance - 1.0e-9
            ):
                return False
            if np.linalg.norm(
                next_position[i] - next_position[j]
            ) < distance - 1.0e-9:
                return False
            if current > influence + 1.0e-12:
                continue
            lower = 0.5 * (distance * distance - current * current)
            if independently_composable:
                if (
                    float(relative @ desired[i]) < 0.5 * lower - 1.0e-8
                    or float((-relative) @ desired[j])
                    < 0.5 * lower - 1.0e-8
                ):
                    return False
            elif (
                float(relative @ (desired[i] - desired[j]))
                < lower - 1.0e-8
            ):
                return False
    return True


@dataclass(frozen=True)
class LocalHyperedgePlan:
    """One UAV's deterministic plan reconstructed from its local view."""

    selected: Tuple[Hyperedge, ...]
    proxy_target_value: np.ndarray
    proxy_scores: Dict[Hyperedge, float]
    role_mask: np.ndarray


def factorized_endpoint_capabilities(
    distance_m: np.ndarray,
    sensing_fraction: np.ndarray,
    *,
    distance_scale_m: float,
    mode: str = "exponential",
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode factorized bistatic Tx/Rx capability in bounded coordinates."""
    distance = np.asarray(distance_m, dtype=np.float64)
    sensing = np.asarray(sensing_fraction, dtype=np.float64)
    if distance.shape != sensing.shape:
        raise ValueError("distance and sensing_fraction must have equal shape")
    if not (np.all(np.isfinite(distance)) and np.all(np.isfinite(sensing))):
        raise ValueError("capability inputs must be finite")
    scale = max(float(distance_scale_m), 1.0e-9)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "exponential":
        receive = np.exp(-np.maximum(distance, 0.0) / scale)
    elif normalized_mode == "inverse_square":
        ratio = np.maximum(distance, 0.0) / scale
        receive = 1.0 / (1.0 + ratio ** 2)
    else:
        raise ValueError(
            "capability mode must be exponential or inverse_square")
    receive = np.clip(receive, 0.0, 1.0)
    transmit = np.clip(
        receive * np.sqrt(np.maximum(sensing, 0.0)),
        0.0,
        1.0,
    )
    return transmit, receive


def encode_offer_stream(
    tx_capability: np.ndarray,
    rx_capability: np.ndarray,
    deficit: np.ndarray,
) -> np.ndarray:
    """Encode three normalized protocol fields into the radio range [-1, 1]."""
    tx = np.asarray(tx_capability, dtype=np.float64)
    rx = np.asarray(rx_capability, dtype=np.float64)
    gap = np.asarray(deficit, dtype=np.float64)
    if not (tx.shape == rx.shape == gap.shape) or tx.ndim != 1:
        raise ValueError("offer fields must be same-shape one-dimensional arrays")
    fields = np.stack([tx, rx, gap], axis=-1)
    if not np.all(np.isfinite(fields)):
        raise ValueError("offer fields must be finite")
    return 2.0 * np.clip(fields, 0.0, 1.0) - 1.0


def decode_offer_stream(stream: np.ndarray) -> np.ndarray:
    """Decode a ``(..., 3)`` quantized offer stream into [0, 1]."""
    encoded = np.asarray(stream, dtype=np.float64)
    if encoded.shape[-1:] != (3,):
        raise ValueError("offer stream must have final dimension 3")
    if not np.all(np.isfinite(encoded)):
        raise ValueError("offer stream must be finite")
    return np.clip(0.5 * (encoded + 1.0), 0.0, 1.0)


def reconstruct_bistatic_pair_value(
    tx_capability: np.ndarray,
    node_positions_xy: np.ndarray,
    target_positions_xy: np.ndarray,
    visible: np.ndarray,
    *,
    distance_scale_m: float,
) -> np.ndarray:
    """Reconstruct normalized pair value from public, quantized local state.

    No realized global deflection entry is used.  Mission target positions are
    common knowledge; node positions and Tx capability must come from the
    viewer's own public offer or an actually delivered neighbor offer.
    """
    tx_value = np.asarray(tx_capability, dtype=np.float64)
    positions = np.asarray(node_positions_xy, dtype=np.float64)
    targets = np.asarray(target_positions_xy, dtype=np.float64)
    seen = np.asarray(visible, dtype=bool)
    if tx_value.ndim != 2:
        raise ValueError("tx_capability must have shape (K, Q)")
    k_count, q_count = tx_value.shape
    if positions.shape != (k_count, q_count, 2):
        raise ValueError("node_positions_xy must have shape (K, Q, 2)")
    if targets.shape != (q_count, 2):
        raise ValueError("target_positions_xy must have shape (Q, 2)")
    if seen.shape != (k_count, q_count):
        raise ValueError("visible must have shape (K, Q)")
    if not (np.all(np.isfinite(tx_value))
            and np.all(np.isfinite(positions))
            and np.all(np.isfinite(targets))):
        raise ValueError("public reconstruction inputs must be finite")

    scale = max(float(distance_scale_m), 1.0e-9)
    pair_value = np.zeros((k_count, k_count, q_count), dtype=np.float64)
    for target in range(q_count):
        ranges = np.linalg.norm(
            positions[:, target] - targets[target], axis=-1)
        for tx in range(k_count):
            if not seen[tx, target]:
                continue
            tx_range = max(float(ranges[tx]), 1.0)
            geometry_proxy = np.exp(-tx_range / scale)
            estimated_power = np.clip(
                float(tx_value[tx, target])
                / max(float(geometry_proxy), 1.0e-9),
                0.0,
                1.0,
            ) ** 2
            for rx in range(k_count):
                if tx == rx or not seen[rx, target]:
                    continue
                rx_range = max(float(ranges[rx]), 1.0)
                pair_value[tx, rx, target] = (
                    estimated_power
                    / (tx_range ** 2 * rx_range ** 2)
                )
        target_max = float(np.max(pair_value[:, :, target]))
        if target_max > 0.0:
            pair_value[:, :, target] /= target_max
    return pair_value


def reconstruct_bistatic_coefficient_from_public_state(
    node_positions_xy: np.ndarray,
    node_velocities_xy: np.ndarray,
    target_positions: np.ndarray,
    target_velocities: np.ndarray,
    visible: np.ndarray,
    *,
    uav_height_m: float,
    fc_hz: float,
    rcs_m2: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    dd_gate_min: float,
    coefficient_scale: float,
    position_uncertainty_m: float | np.ndarray = 0.0,
    target_position_uncertainty_m: float | np.ndarray = 0.0,
    velocity_uncertainty_mps: float | np.ndarray = 0.0,
    target_velocity_uncertainty_mps: float | np.ndarray = 0.0,
    dd_gain_mode: str = "binary",
    robust_dd_uncertainty: bool = True,
) -> np.ndarray:
    """Reconstruct the executable per-watt graph from delivered endpoint state.

    The bistatic radar equation factorizes through the two endpoint ranges,
    while delay/Doppler support is deterministic given endpoint and target
    state.  Consequently broadcasting one position/velocity tuple per UAV is
    a sufficient statistic for all ``K(K-1)Q`` coefficients: a receiver need
    not transmit a dense edge matrix.  Missing endpoint messages fail closed
    to zero coefficient.

    This reconstruction is exact for the deployed deterministic analytical
    sensing model when reporting loss and Swerling fading are disabled.  With
    quantized state it is a public, reproducible approximation rather than
    privileged access to the simulator's realized deflection entries.
    """
    positions_xy = np.asarray(node_positions_xy, dtype=np.float64)
    velocities_xy = np.asarray(node_velocities_xy, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    target_velocity = np.asarray(target_velocities, dtype=np.float64)
    seen = np.asarray(visible, dtype=bool)
    if positions_xy.ndim != 3 or positions_xy.shape[-1] != 2:
        raise ValueError("node_positions_xy must have shape (K,Q,2)")
    if velocities_xy.shape != positions_xy.shape:
        raise ValueError("node_velocities_xy must match node_positions_xy")
    K, Q, _ = positions_xy.shape
    if targets.shape != (Q, 3) or target_velocity.shape != (Q, 3):
        raise ValueError("target position/velocity must have shape (Q,3)")
    if seen.shape != (K, Q):
        raise ValueError("visible must have shape (K,Q)")
    uncertainty = np.asarray(
        position_uncertainty_m, dtype=np.float64)
    if uncertainty.ndim == 0:
        uncertainty = np.full(K, float(uncertainty), dtype=np.float64)
    elif uncertainty.shape != (K,):
        raise ValueError(
            "position_uncertainty_m must be a scalar or K-vector")
    target_uncertainty = np.asarray(
        target_position_uncertainty_m, dtype=np.float64)
    if target_uncertainty.ndim == 0:
        target_uncertainty = np.full(
            Q, float(target_uncertainty), dtype=np.float64)
    elif target_uncertainty.shape != (Q,):
        raise ValueError(
            "target_position_uncertainty_m must be a scalar or Q-vector")
    scalar_values = (
        float(uav_height_m), float(fc_hz), float(rcs_m2),
        float(delta_f_hz), float(symbol_period_s),
        float(dd_gate_min), float(coefficient_scale),
    )
    if (
        not np.all(np.isfinite(positions_xy))
        or not np.all(np.isfinite(velocities_xy))
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(target_velocity))
        or not np.all(np.isfinite(uncertainty))
        or not np.all(np.isfinite(target_uncertainty))
        or not all(np.isfinite(value) for value in scalar_values)
    ):
        raise ValueError("public physical reconstruction inputs must be finite")
    if (delay_bins < 1 or doppler_bins < 1 or coefficient_scale <= 0.0
            or np.any(uncertainty < 0.0)):
        raise ValueError(
            "grid sizes/scale must be positive and uncertainty non-negative")
    if np.any(target_uncertainty < 0.0):
        raise ValueError("target uncertainty must be non-negative")
    velocity_uncertainty = np.asarray(
        velocity_uncertainty_mps, dtype=np.float64)
    if velocity_uncertainty.ndim == 0:
        velocity_uncertainty = np.full(K, float(velocity_uncertainty))
    else:
        velocity_uncertainty = velocity_uncertainty.reshape(-1)
    target_velocity_uncertainty = np.asarray(
        target_velocity_uncertainty_mps, dtype=np.float64)
    if target_velocity_uncertainty.ndim == 0:
        target_velocity_uncertainty = np.full(
            Q, float(target_velocity_uncertainty))
    else:
        target_velocity_uncertainty = target_velocity_uncertainty.reshape(-1)
    if (
        velocity_uncertainty.shape != (K,)
        or target_velocity_uncertainty.shape != (Q,)
        or np.any(~np.isfinite(velocity_uncertainty))
        or np.any(~np.isfinite(target_velocity_uncertainty))
        or np.any(velocity_uncertainty < 0.0)
        or np.any(target_velocity_uncertainty < 0.0)
    ):
        raise ValueError("velocity uncertainty must be non-negative K/Q vectors")

    # Vectorized K x K x Q reconstruction.  The previous implementation called
    # the full geometry routine once per target and then evaluated every edge
    # in Python.  In a K-view distributed simulation that created K redundant
    # scalar triple loops per frame.  These expressions are the same bistatic
    # range, signed-Doppler, radar-path and DD equations, evaluated as arrays.
    positions = np.concatenate((
        positions_xy,
        np.full((K, Q, 1), float(uav_height_m), dtype=np.float64),
    ), axis=-1)
    velocities = np.concatenate((
        velocities_xy,
        np.zeros((K, Q, 1), dtype=np.float64),
    ), axis=-1)
    endpoint_vector = targets[None, :, :] - positions
    endpoint_range = np.linalg.norm(endpoint_vector, axis=-1)
    endpoint_unit = endpoint_vector / (endpoint_range[..., None] + 1.0e-10)

    bistatic_range = (
        endpoint_range[:, None, :] + endpoint_range[None, :, :])
    tau = bistatic_range / C_LIGHT
    node_radial = np.sum(velocities * endpoint_unit, axis=-1)
    target_radial = np.sum(
        target_velocity[None, :, :] * endpoint_unit, axis=-1)
    nu = (float(fc_hz) / C_LIGHT) * (
        node_radial[:, None, :] + node_radial[None, :, :]
        - target_radial[:, None, :] - target_radial[None, :, :]
    )

    safe_range = np.maximum(endpoint_range, 1.0e-6)
    wavelength = C_LIGHT / float(fc_hz)
    path_constant = (
        wavelength * wavelength * float(rcs_m2) / (4.0 * np.pi) ** 3)
    alpha_sq = path_constant / (
        safe_range[:, None, :] ** 2 * safe_range[None, :, :] ** 2)

    delay_fraction = tau * int(delay_bins) * float(delta_f_hz)
    doppler_fraction = (
        nu * int(doppler_bins) * float(symbol_period_s))
    delay_offset = delay_fraction - np.round(delay_fraction)
    doppler_offset = doppler_fraction - np.round(doppler_fraction)
    ambiguity_amplitude = np.abs(
        np.sinc(delay_offset) * np.sinc(doppler_offset))
    if not bool(robust_dd_uncertainty):
        # The nominal ranking/power view does not consume uncertainty-set DD
        # bounds.  Previously the complete radius propagation and two sinc
        # lower-bound kernels ran first and were discarded here.  Keep the
        # exact historical nominal result while avoiding that dead work.
        delay_radius = np.zeros_like(delay_fraction)
        doppler_radius = np.zeros_like(doppler_fraction)
        ambiguity_lower = ambiguity_amplitude
    else:
        # Deterministic uncertainty-set lower bound.  The bistatic delay is
        # 1-Lipschitz in each endpoint/target range.  Doppler uses the standard
        # unit-vector perturbation bound min(2, 2r/(R-r)); velocity and
        # direction errors are combined by the triangle inequality.
        endpoint_position_radius = (
            uncertainty[:, None] + target_uncertainty[None, :])
        delay_radius = (
            uncertainty[:, None, None]
            + uncertainty[None, :, None]
            + 2.0 * target_uncertainty[None, None, :]
        ) / C_LIGHT * int(delay_bins) * float(delta_f_hz)
        unit_radius = np.minimum(
            2.0,
            2.0 * endpoint_position_radius
            / np.maximum(
                endpoint_range - endpoint_position_radius, 1.0e-9),
        )
        relative_velocity = velocities - target_velocity[None, :, :]
        relative_speed = np.linalg.norm(relative_velocity, axis=-1)
        relative_velocity_radius = (
            velocity_uncertainty[:, None]
            + target_velocity_uncertainty[None, :])
        radial_radius = (
            relative_velocity_radius
            + (relative_speed + relative_velocity_radius) * unit_radius
        )
        doppler_radius = (
            float(fc_hz) / C_LIGHT
            * (radial_radius[:, None, :] + radial_radius[None, :, :])
            * int(doppler_bins) * float(symbol_period_s)
        )
        ambiguity_lower = (
            _sinc_alignment_lower_array(delay_fraction, delay_radius)
            * _sinc_alignment_lower_array(doppler_fraction, doppler_radius)
        )
    if str(dd_gain_mode) == "continuous":
        support = (
            (tau - delay_radius / (
                int(delay_bins) * float(delta_f_hz)) >= 0.0)
            & (tau + delay_radius / (
                int(delay_bins) * float(delta_f_hz))
               < 1.0 / float(delta_f_hz))
            & ((np.abs(nu) + doppler_radius / (
                int(doppler_bins) * float(symbol_period_s)))
               <= 1.0 / (2.0 * float(symbol_period_s)))
        )
        dd_factor = support.astype(np.float64) * ambiguity_lower ** 2
    else:
        dd_factor = (
            ambiguity_lower >= float(dd_gate_min)).astype(np.float64)

    robust_endpoint = endpoint_range / (
        endpoint_range
        + uncertainty[:, None]
        + target_uncertainty[None, :]
    )
    robust_pathloss = (
        robust_endpoint[:, None, :] ** 2
        * robust_endpoint[None, :, :] ** 2)
    executable = seen[:, None, :] & seen[None, :, :]
    executable &= ~np.eye(K, dtype=bool)[:, :, None]
    return (
        float(coefficient_scale)
        * alpha_sq
        * robust_pathloss
        * dd_factor
        * executable.astype(np.float64)
    )


def reconstruct_bistatic_coefficient_upper_from_public_state(
    positions_xy_by_target: np.ndarray,
    target_positions_m: np.ndarray,
    seen_mask: np.ndarray,
    *,
    uav_height_m: float,
    fc_hz: float,
    rcs_m2: float,
    coefficient_scale: float,
    position_uncertainty_m: float | np.ndarray = 0.0,
    target_position_uncertainty_m: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Upper-bound every fixed-edge coefficient over position uncertainty.

    The continuous DD power gain is at most one.  Visible endpoints use their
    minimum possible slant range; missing endpoints use the physical altitude
    floor, so an incomplete private view remains valid (although loose).
    """
    positions_xy = np.asarray(positions_xy_by_target, dtype=np.float64)
    targets = np.asarray(target_positions_m, dtype=np.float64)
    seen = np.asarray(seen_mask, dtype=bool)
    if (
        positions_xy.ndim != 3 or positions_xy.shape[2] != 2
    ):
        raise ValueError("positions must have shape (K,Q,2)")
    K, Q, _ = positions_xy.shape
    if targets.shape != (Q, 3) or seen.shape != (K, Q):
        raise ValueError("target/visibility shapes do not match (K,Q)")
    uncertainty = np.asarray(position_uncertainty_m, dtype=np.float64)
    if uncertainty.ndim == 0:
        uncertainty = np.full(K, float(uncertainty))
    else:
        uncertainty = uncertainty.reshape(-1)
    target_uncertainty = np.asarray(
        target_position_uncertainty_m, dtype=np.float64)
    if target_uncertainty.ndim == 0:
        target_uncertainty = np.full(Q, float(target_uncertainty))
    else:
        target_uncertainty = target_uncertainty.reshape(-1)
    values = (positions_xy, targets, uncertainty, target_uncertainty)
    if (
        uncertainty.shape != (K,) or target_uncertainty.shape != (Q,)
        or any(np.any(~np.isfinite(value)) for value in values)
        or np.any(uncertainty < 0.0)
        or np.any(target_uncertainty < 0.0)
        or not np.isfinite(uav_height_m) or float(uav_height_m) <= 0.0
        or not np.isfinite(fc_hz) or float(fc_hz) <= 0.0
        or not np.isfinite(rcs_m2) or float(rcs_m2) < 0.0
        or not np.isfinite(coefficient_scale) or float(coefficient_scale) < 0.0
    ):
        raise ValueError("coefficient upper-bound inputs are invalid")
    positions = np.concatenate((
        positions_xy,
        np.full((K, Q, 1), float(uav_height_m), dtype=np.float64),
    ), axis=-1)
    nominal_range = np.linalg.norm(
        targets[None, :, :] - positions, axis=-1)
    radius = uncertainty[:, None] + target_uncertainty[None, :]
    minimum_range = np.maximum(
        nominal_range - radius, float(uav_height_m))
    minimum_range = np.where(seen, minimum_range, float(uav_height_m))
    wavelength = C_LIGHT / float(fc_hz)
    path_constant = (
        wavelength * wavelength * float(rcs_m2) / (4.0 * np.pi) ** 3)
    upper = (
        float(coefficient_scale) * path_constant
        / (
            minimum_range[:, None, :] ** 2
            * minimum_range[None, :, :] ** 2
        )
    )
    upper *= (~np.eye(K, dtype=bool))[:, :, None]
    return upper


def reconstruct_bistatic_coefficient_dd_upper_from_public_state(
    node_positions_xy: np.ndarray,
    node_velocities_xy: np.ndarray,
    target_positions: np.ndarray,
    target_velocities: np.ndarray,
    visible: np.ndarray,
    *,
    uav_height_m: float,
    fc_hz: float,
    rcs_m2: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    dd_gate_min: float,
    coefficient_scale: float,
    position_uncertainty_m: float | np.ndarray = 0.0,
    target_position_uncertainty_m: float | np.ndarray = 0.0,
    velocity_uncertainty_mps: float | np.ndarray = 0.0,
    target_velocity_uncertainty_mps: float | np.ndarray = 0.0,
    dd_gain_mode: str = "binary",
) -> np.ndarray:
    """Tighten the range upper with an exact interval DD-gain maximum."""
    range_upper = reconstruct_bistatic_coefficient_upper_from_public_state(
        node_positions_xy,
        target_positions,
        visible,
        uav_height_m=uav_height_m,
        fc_hz=fc_hz,
        rcs_m2=rcs_m2,
        coefficient_scale=coefficient_scale,
        position_uncertainty_m=position_uncertainty_m,
        target_position_uncertainty_m=target_position_uncertainty_m,
    )
    positions_xy = np.asarray(node_positions_xy, dtype=np.float64)
    velocities_xy = np.asarray(node_velocities_xy, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    target_velocity = np.asarray(target_velocities, dtype=np.float64)
    K, Q, _ = positions_xy.shape
    if (
        velocities_xy.shape != positions_xy.shape
        or targets.shape != (Q, 3)
        or target_velocity.shape != (Q, 3)
        or delay_bins < 1
        or doppler_bins < 1
        or not np.isfinite(delta_f_hz)
        or float(delta_f_hz) <= 0.0
        or not np.isfinite(symbol_period_s)
        or float(symbol_period_s) <= 0.0
    ):
        raise ValueError("DD upper-bound inputs are invalid")

    def vector(value: float | np.ndarray, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        array = (
            np.full(size, float(array), dtype=np.float64)
            if array.ndim == 0 else array.reshape(-1))
        if (
            array.shape != (size,)
            or np.any(~np.isfinite(array))
            or np.any(array < 0.0)
        ):
            raise ValueError(f"{name} must be a non-negative vector")
        return array

    node_position_radius = vector(
        position_uncertainty_m, K, "position uncertainty")
    target_position_radius = vector(
        target_position_uncertainty_m, Q, "target position uncertainty")
    node_velocity_radius = vector(
        velocity_uncertainty_mps, K, "velocity uncertainty")
    target_velocity_radius = vector(
        target_velocity_uncertainty_mps, Q,
        "target velocity uncertainty")
    positions = np.concatenate((
        positions_xy,
        np.full((K, Q, 1), float(uav_height_m), dtype=np.float64),
    ), axis=-1)
    velocities = np.concatenate((
        velocities_xy,
        np.zeros((K, Q, 1), dtype=np.float64),
    ), axis=-1)
    endpoint_vector = targets[None, :, :] - positions
    endpoint_range = np.linalg.norm(endpoint_vector, axis=-1)
    endpoint_unit = endpoint_vector / (endpoint_range[..., None] + 1.0e-10)
    tau = (
        endpoint_range[:, None, :] + endpoint_range[None, :, :]
    ) / C_LIGHT
    node_radial = np.sum(velocities * endpoint_unit, axis=-1)
    target_radial = np.sum(
        target_velocity[None, :, :] * endpoint_unit, axis=-1)
    nu = (float(fc_hz) / C_LIGHT) * (
        node_radial[:, None, :] + node_radial[None, :, :]
        - target_radial[:, None, :] - target_radial[None, :, :])
    endpoint_radius = (
        node_position_radius[:, None] + target_position_radius[None, :])
    delay_radius = (
        node_position_radius[:, None, None]
        + node_position_radius[None, :, None]
        + 2.0 * target_position_radius[None, None, :]
    ) / C_LIGHT * int(delay_bins) * float(delta_f_hz)
    unit_radius = np.minimum(
        2.0,
        2.0 * endpoint_radius
        / np.maximum(endpoint_range - endpoint_radius, 1.0e-9),
    )
    relative_velocity = velocities - target_velocity[None, :, :]
    relative_speed = np.linalg.norm(relative_velocity, axis=-1)
    relative_velocity_radius = (
        node_velocity_radius[:, None] + target_velocity_radius[None, :])
    radial_radius = (
        relative_velocity_radius
        + (relative_speed + relative_velocity_radius) * unit_radius)
    doppler_radius = (
        float(fc_hz) / C_LIGHT
        * (radial_radius[:, None, :] + radial_radius[None, :, :])
        * int(doppler_bins) * float(symbol_period_s))
    delay_fraction = tau * int(delay_bins) * float(delta_f_hz)
    doppler_fraction = (
        nu * int(doppler_bins) * float(symbol_period_s))
    ambiguity_upper = (
        _sinc_alignment_upper_array(delay_fraction, delay_radius)
        * _sinc_alignment_upper_array(doppler_fraction, doppler_radius))
    if str(dd_gain_mode) == "continuous":
        delay_scale = int(delay_bins) * float(delta_f_hz)
        doppler_scale = int(doppler_bins) * float(symbol_period_s)
        support_possible = (
            (tau + delay_radius / delay_scale >= 0.0)
            & (tau - delay_radius / delay_scale < 1.0 / float(delta_f_hz))
            & (np.maximum(np.abs(nu) - doppler_radius / doppler_scale, 0.0)
               <= 1.0 / (2.0 * float(symbol_period_s)))
        )
        dd_upper = support_possible.astype(np.float64) * ambiguity_upper ** 2
    elif str(dd_gain_mode) == "binary":
        dd_upper = (ambiguity_upper >= float(dd_gate_min)).astype(np.float64)
    else:
        raise ValueError("dd_gain_mode must be binary or continuous")
    return range_upper * dd_upper


@dataclass(frozen=True)
class SelectedBistaticCoefficients:
    """Nominal and certificate coefficients for one sparse edge set."""

    edges: Tuple[Hyperedge, ...]
    nominal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def reconstruct_selected_bistatic_coefficients_from_public_state(
    positions_xy_by_target: np.ndarray,
    velocities_xy_by_target: np.ndarray,
    target_positions_m: np.ndarray,
    target_velocities_mps: np.ndarray,
    seen_mask: np.ndarray,
    edges: Iterable[Hyperedge],
    *,
    uav_height_m: float,
    fc_hz: float,
    rcs_m2: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    dd_gate_min: float,
    coefficient_scale: float,
    nominal_position_uncertainty_m: float | np.ndarray = 0.0,
    certificate_position_uncertainty_m: float | np.ndarray = 0.0,
    target_position_uncertainty_m: float | np.ndarray = 0.0,
    velocity_uncertainty_mps: float | np.ndarray = 0.0,
    target_velocity_uncertainty_mps: float | np.ndarray = 0.0,
    dd_gain_mode: str = "binary",
) -> SelectedBistaticCoefficients:
    """Reconstruct only selected edges and share their endpoint geometry.

    This is the hold-path counterpart of the two dense reconstruction
    functions above.  It preserves their semantics: nominal coefficients use
    point DD alignment plus endpoint path-loss uncertainty, lower coefficients
    use the complete position/velocity uncertainty set, and upper coefficients
    use minimum possible range with DD gain bounded by one.
    """
    positions_xy = np.asarray(positions_xy_by_target, dtype=np.float64)
    velocities_xy = np.asarray(velocities_xy_by_target, dtype=np.float64)
    targets = np.asarray(target_positions_m, dtype=np.float64)
    target_velocity = np.asarray(target_velocities_mps, dtype=np.float64)
    seen = np.asarray(seen_mask, dtype=bool)
    selected = tuple(tuple(int(value) for value in edge) for edge in edges)
    if positions_xy.ndim != 3 or positions_xy.shape[-1] != 2:
        raise ValueError("positions must have shape (K,Q,2)")
    K, Q, _ = positions_xy.shape
    if (
        velocities_xy.shape != positions_xy.shape
        or targets.shape != (Q, 3)
        or target_velocity.shape != (Q, 3)
        or seen.shape != (K, Q)
    ):
        raise ValueError("selected coefficient input shapes are inconsistent")
    edge_array = np.asarray(selected, dtype=np.int64).reshape(-1, 3)
    if edge_array.size and (
        np.any(edge_array[:, 0] < 0)
        or np.any(edge_array[:, 0] >= K)
        or np.any(edge_array[:, 1] < 0)
        or np.any(edge_array[:, 1] >= K)
        or np.any(edge_array[:, 2] < 0)
        or np.any(edge_array[:, 2] >= Q)
        or np.any(edge_array[:, 0] == edge_array[:, 1])
    ):
        raise ValueError("selected edge is outside the physical graph")
    if len(set(selected)) != len(selected):
        raise ValueError("selected edges must be unique")

    def uncertainty_vector(value, size, name):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(size, float(array), dtype=np.float64)
        else:
            array = array.reshape(-1)
        if array.shape != (size,) or np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be a non-negative {size}-vector")
        return array

    nominal_uncertainty = uncertainty_vector(
        nominal_position_uncertainty_m, K,
        "nominal_position_uncertainty_m")
    certificate_uncertainty = uncertainty_vector(
        certificate_position_uncertainty_m, K,
        "certificate_position_uncertainty_m")
    target_uncertainty = uncertainty_vector(
        target_position_uncertainty_m, Q,
        "target_position_uncertainty_m")
    velocity_uncertainty = uncertainty_vector(
        velocity_uncertainty_mps, K, "velocity_uncertainty_mps")
    target_velocity_uncertainty = uncertainty_vector(
        target_velocity_uncertainty_mps, Q,
        "target_velocity_uncertainty_mps")
    scalar_values = (
        float(uav_height_m), float(fc_hz), float(rcs_m2),
        float(delta_f_hz), float(symbol_period_s), float(dd_gate_min),
        float(coefficient_scale),
    )
    if (
        any(np.any(~np.isfinite(value)) for value in (
            positions_xy, velocities_xy, targets, target_velocity))
        or not all(np.isfinite(value) for value in scalar_values)
        or delay_bins < 1 or doppler_bins < 1
        or float(uav_height_m) <= 0.0 or float(fc_hz) <= 0.0
        or float(rcs_m2) < 0.0 or float(coefficient_scale) < 0.0
    ):
        raise ValueError("selected coefficient physical inputs are invalid")
    if not len(selected):
        empty = np.zeros(0, dtype=np.float64)
        return SelectedBistaticCoefficients(selected, empty, empty, empty)

    tx = edge_array[:, 0]
    rx = edge_array[:, 1]
    target = edge_array[:, 2]
    altitude = np.full((len(selected), 1), float(uav_height_m))
    tx_position = np.concatenate(
        (positions_xy[tx, target], altitude), axis=1)
    rx_position = np.concatenate(
        (positions_xy[rx, target], altitude), axis=1)
    tx_velocity = np.concatenate(
        (velocities_xy[tx, target], np.zeros_like(altitude)), axis=1)
    rx_velocity = np.concatenate(
        (velocities_xy[rx, target], np.zeros_like(altitude)), axis=1)
    target_position = targets[target]
    target_velocity_edge = target_velocity[target]
    tx_vector = target_position - tx_position
    rx_vector = target_position - rx_position
    tx_range = np.linalg.norm(tx_vector, axis=1)
    rx_range = np.linalg.norm(rx_vector, axis=1)
    tx_unit = tx_vector / (tx_range[:, None] + 1.0e-10)
    rx_unit = rx_vector / (rx_range[:, None] + 1.0e-10)
    tau = (tx_range + rx_range) / C_LIGHT
    node_radial = (
        np.sum(tx_velocity * tx_unit, axis=1)
        + np.sum(rx_velocity * rx_unit, axis=1))
    target_radial = (
        np.sum(target_velocity_edge * tx_unit, axis=1)
        + np.sum(target_velocity_edge * rx_unit, axis=1))
    nu = float(fc_hz) / C_LIGHT * (node_radial - target_radial)

    wavelength = C_LIGHT / float(fc_hz)
    path_constant = (
        wavelength * wavelength * float(rcs_m2) / (4.0 * np.pi) ** 3)
    alpha_sq = path_constant / (
        np.maximum(tx_range, 1.0e-6) ** 2
        * np.maximum(rx_range, 1.0e-6) ** 2)
    delay_fraction = tau * int(delay_bins) * float(delta_f_hz)
    doppler_fraction = nu * int(doppler_bins) * float(symbol_period_s)
    ambiguity_amplitude = np.abs(
        np.sinc(delay_fraction - np.round(delay_fraction))
        * np.sinc(doppler_fraction - np.round(doppler_fraction)))
    nominal_support = (
        (tau >= 0.0)
        & (tau < 1.0 / float(delta_f_hz))
        & (np.abs(nu) <= 1.0 / (2.0 * float(symbol_period_s))))
    nominal_dd = (
        nominal_support.astype(np.float64) * ambiguity_amplitude ** 2
        if str(dd_gain_mode) == "continuous"
        else (ambiguity_amplitude >= float(dd_gate_min)).astype(np.float64)
    )
    nominal_tx_path = tx_range / (
        tx_range + nominal_uncertainty[tx])
    nominal_rx_path = rx_range / (
        rx_range + nominal_uncertainty[rx])
    executable = seen[tx, target] & seen[rx, target]
    nominal = (
        float(coefficient_scale) * alpha_sq
        * nominal_tx_path ** 2 * nominal_rx_path ** 2
        * nominal_dd * executable.astype(np.float64))

    tx_position_radius = (
        certificate_uncertainty[tx] + target_uncertainty[target])
    rx_position_radius = (
        certificate_uncertainty[rx] + target_uncertainty[target])
    delay_radius = (
        certificate_uncertainty[tx]
        + certificate_uncertainty[rx]
        + 2.0 * target_uncertainty[target]
    ) / C_LIGHT * int(delay_bins) * float(delta_f_hz)
    tx_unit_radius = np.minimum(
        2.0,
        2.0 * tx_position_radius
        / np.maximum(tx_range - tx_position_radius, 1.0e-9))
    rx_unit_radius = np.minimum(
        2.0,
        2.0 * rx_position_radius
        / np.maximum(rx_range - rx_position_radius, 1.0e-9))
    tx_relative_speed = np.linalg.norm(
        tx_velocity - target_velocity_edge, axis=1)
    rx_relative_speed = np.linalg.norm(
        rx_velocity - target_velocity_edge, axis=1)
    tx_velocity_radius = (
        velocity_uncertainty[tx] + target_velocity_uncertainty[target])
    rx_velocity_radius = (
        velocity_uncertainty[rx] + target_velocity_uncertainty[target])
    tx_radial_radius = (
        tx_velocity_radius
        + (tx_relative_speed + tx_velocity_radius) * tx_unit_radius)
    rx_radial_radius = (
        rx_velocity_radius
        + (rx_relative_speed + rx_velocity_radius) * rx_unit_radius)
    doppler_radius = (
        float(fc_hz) / C_LIGHT
        * (tx_radial_radius + rx_radial_radius)
        * int(doppler_bins) * float(symbol_period_s))
    ambiguity_lower = (
        _sinc_alignment_lower_array(delay_fraction, delay_radius)
        * _sinc_alignment_lower_array(doppler_fraction, doppler_radius))
    if str(dd_gain_mode) == "continuous":
        lower_support = (
            (tau - delay_radius / (
                int(delay_bins) * float(delta_f_hz)) >= 0.0)
            & (tau + delay_radius / (
                int(delay_bins) * float(delta_f_hz))
               < 1.0 / float(delta_f_hz))
            & ((np.abs(nu) + doppler_radius / (
                int(doppler_bins) * float(symbol_period_s)))
               <= 1.0 / (2.0 * float(symbol_period_s))))
        lower_dd = lower_support.astype(np.float64) * ambiguity_lower ** 2
    else:
        lower_dd = (
            ambiguity_lower >= float(dd_gate_min)).astype(np.float64)
    lower_tx_path = tx_range / (
        tx_range + certificate_uncertainty[tx]
        + target_uncertainty[target])
    lower_rx_path = rx_range / (
        rx_range + certificate_uncertainty[rx]
        + target_uncertainty[target])
    lower = (
        float(coefficient_scale) * alpha_sq
        * lower_tx_path ** 2 * lower_rx_path ** 2
        * lower_dd * executable.astype(np.float64))

    upper_tx_range = np.maximum(
        tx_range - certificate_uncertainty[tx]
        - target_uncertainty[target], float(uav_height_m))
    upper_rx_range = np.maximum(
        rx_range - certificate_uncertainty[rx]
        - target_uncertainty[target], float(uav_height_m))
    upper_tx_range = np.where(
        seen[tx, target], upper_tx_range, float(uav_height_m))
    upper_rx_range = np.where(
        seen[rx, target], upper_rx_range, float(uav_height_m))
    upper = (
        float(coefficient_scale) * path_constant
        / (upper_tx_range ** 2 * upper_rx_range ** 2))
    return SelectedBistaticCoefficients(
        selected, nominal, lower, upper)


def reconstruct_selected_bistatic_coefficients_batch_from_public_state(
    positions_xy_by_view: np.ndarray,
    velocities_xy_by_view: np.ndarray,
    target_positions_by_view: np.ndarray,
    target_velocities_by_view: np.ndarray,
    seen_mask_by_view: np.ndarray,
    edges: Iterable[Hyperedge],
    *,
    uav_height_m: float,
    fc_hz: float,
    rcs_m2: float,
    delta_f_hz: float,
    symbol_period_s: float,
    delay_bins: int,
    doppler_bins: int,
    dd_gate_min: float,
    coefficient_scale: float,
    nominal_position_uncertainty_m: float | np.ndarray = 0.0,
    certificate_position_uncertainty_m: float | np.ndarray = 0.0,
    target_position_uncertainty_m: float | np.ndarray = 0.0,
    velocity_uncertainty_mps: float | np.ndarray = 0.0,
    target_velocity_uncertainty_mps: float | np.ndarray = 0.0,
    dd_gain_mode: str = "binary",
) -> SelectedBistaticCoefficients:
    """Jointly reconstruct the same selected COO edges for all viewers.

    Arrays carry a leading viewer dimension ``V``.  The returned nominal,
    lower and upper arrays have shape ``(V,E)`` and are exactly the per-view
    selected kernel evaluated without its repeated Python validation/setup.
    """
    positions = np.asarray(positions_xy_by_view, dtype=np.float64)
    velocities = np.asarray(velocities_xy_by_view, dtype=np.float64)
    targets = np.asarray(target_positions_by_view, dtype=np.float64)
    target_velocities = np.asarray(
        target_velocities_by_view, dtype=np.float64)
    seen = np.asarray(seen_mask_by_view, dtype=bool)
    selected = tuple(tuple(int(value) for value in edge) for edge in edges)
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (V,K,Q,2)")
    V, K, Q, _ = positions.shape
    if (
        velocities.shape != positions.shape
        or targets.shape != (V, Q, 3)
        or target_velocities.shape != (V, Q, 3)
        or seen.shape != (V, K, Q)
    ):
        raise ValueError("batched selected coefficient shapes are inconsistent")
    edge_array = np.asarray(selected, dtype=np.int64).reshape(-1, 3)
    if edge_array.size and (
        np.any(edge_array[:, 0] < 0)
        or np.any(edge_array[:, 0] >= K)
        or np.any(edge_array[:, 1] < 0)
        or np.any(edge_array[:, 1] >= K)
        or np.any(edge_array[:, 2] < 0)
        or np.any(edge_array[:, 2] >= Q)
        or np.any(edge_array[:, 0] == edge_array[:, 1])
    ):
        raise ValueError("selected edge is outside the batched physical graph")
    if len(set(selected)) != len(selected):
        raise ValueError("selected edges must be unique")

    def uncertainty_table(value, rows, columns, name):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full((rows, columns), float(array), dtype=np.float64)
        elif array.shape == (columns,):
            array = np.broadcast_to(array[None, :], (rows, columns))
        elif array.shape != (rows, columns):
            raise ValueError(
                f"{name} must be scalar, ({columns},), or "
                f"({rows},{columns})")
        if np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
        return array

    nominal_uncertainty = uncertainty_table(
        nominal_position_uncertainty_m, V, K,
        "nominal_position_uncertainty_m")
    certificate_uncertainty = uncertainty_table(
        certificate_position_uncertainty_m, V, K,
        "certificate_position_uncertainty_m")
    target_uncertainty = uncertainty_table(
        target_position_uncertainty_m, V, Q,
        "target_position_uncertainty_m")
    velocity_uncertainty = uncertainty_table(
        velocity_uncertainty_mps, V, K, "velocity_uncertainty_mps")
    target_velocity_uncertainty = uncertainty_table(
        target_velocity_uncertainty_mps, V, Q,
        "target_velocity_uncertainty_mps")
    scalar_values = (
        float(uav_height_m), float(fc_hz), float(rcs_m2),
        float(delta_f_hz), float(symbol_period_s), float(dd_gate_min),
        float(coefficient_scale),
    )
    if (
        any(np.any(~np.isfinite(value)) for value in (
            positions, velocities, targets, target_velocities))
        or not all(np.isfinite(value) for value in scalar_values)
        or delay_bins < 1 or doppler_bins < 1
        or float(uav_height_m) <= 0.0 or float(fc_hz) <= 0.0
        or float(rcs_m2) < 0.0 or float(coefficient_scale) < 0.0
    ):
        raise ValueError("batched selected physical inputs are invalid")
    if not selected:
        empty = np.zeros((V, 0), dtype=np.float64)
        return SelectedBistaticCoefficients(selected, empty, empty, empty)

    tx = edge_array[:, 0]
    rx = edge_array[:, 1]
    target = edge_array[:, 2]
    altitude = np.full((V, len(selected), 1), float(uav_height_m))
    zeros = np.zeros_like(altitude)
    tx_position = np.concatenate((positions[:, tx, target], altitude), axis=2)
    rx_position = np.concatenate((positions[:, rx, target], altitude), axis=2)
    tx_velocity = np.concatenate((velocities[:, tx, target], zeros), axis=2)
    rx_velocity = np.concatenate((velocities[:, rx, target], zeros), axis=2)
    target_position = targets[:, target]
    target_velocity = target_velocities[:, target]
    tx_vector = target_position - tx_position
    rx_vector = target_position - rx_position
    tx_range = np.linalg.norm(tx_vector, axis=2)
    rx_range = np.linalg.norm(rx_vector, axis=2)
    tx_unit = tx_vector / (tx_range[:, :, None] + 1.0e-10)
    rx_unit = rx_vector / (rx_range[:, :, None] + 1.0e-10)
    tau = (tx_range + rx_range) / C_LIGHT
    node_radial = (
        np.sum(tx_velocity * tx_unit, axis=2)
        + np.sum(rx_velocity * rx_unit, axis=2))
    target_radial = (
        np.sum(target_velocity * tx_unit, axis=2)
        + np.sum(target_velocity * rx_unit, axis=2))
    nu = float(fc_hz) / C_LIGHT * (node_radial - target_radial)

    wavelength = C_LIGHT / float(fc_hz)
    path_constant = (
        wavelength * wavelength * float(rcs_m2) / (4.0 * np.pi) ** 3)
    alpha_sq = path_constant / (
        np.maximum(tx_range, 1.0e-6) ** 2
        * np.maximum(rx_range, 1.0e-6) ** 2)
    delay_fraction = tau * int(delay_bins) * float(delta_f_hz)
    doppler_fraction = nu * int(doppler_bins) * float(symbol_period_s)
    ambiguity_amplitude = np.abs(
        np.sinc(delay_fraction - np.round(delay_fraction))
        * np.sinc(doppler_fraction - np.round(doppler_fraction)))
    nominal_support = (
        (tau >= 0.0)
        & (tau < 1.0 / float(delta_f_hz))
        & (np.abs(nu) <= 1.0 / (2.0 * float(symbol_period_s))))
    nominal_dd = (
        nominal_support.astype(np.float64) * ambiguity_amplitude ** 2
        if str(dd_gain_mode) == "continuous"
        else (ambiguity_amplitude >= float(dd_gate_min)).astype(np.float64)
    )
    nominal_tx_uncertainty = nominal_uncertainty[:, tx]
    nominal_rx_uncertainty = nominal_uncertainty[:, rx]
    executable = seen[:, tx, target] & seen[:, rx, target]
    nominal = (
        float(coefficient_scale) * alpha_sq
        * (tx_range / (tx_range + nominal_tx_uncertainty)) ** 2
        * (rx_range / (rx_range + nominal_rx_uncertainty)) ** 2
        * nominal_dd * executable.astype(np.float64))

    certificate_tx = certificate_uncertainty[:, tx]
    certificate_rx = certificate_uncertainty[:, rx]
    target_radius = target_uncertainty[:, target]
    tx_position_radius = certificate_tx + target_radius
    rx_position_radius = certificate_rx + target_radius
    delay_radius = (
        certificate_tx + certificate_rx + 2.0 * target_radius
    ) / C_LIGHT * int(delay_bins) * float(delta_f_hz)
    tx_unit_radius = np.minimum(
        2.0, 2.0 * tx_position_radius
        / np.maximum(tx_range - tx_position_radius, 1.0e-9))
    rx_unit_radius = np.minimum(
        2.0, 2.0 * rx_position_radius
        / np.maximum(rx_range - rx_position_radius, 1.0e-9))
    tx_relative_speed = np.linalg.norm(tx_velocity - target_velocity, axis=2)
    rx_relative_speed = np.linalg.norm(rx_velocity - target_velocity, axis=2)
    target_velocity_radius = target_velocity_uncertainty[:, target]
    tx_velocity_radius = velocity_uncertainty[:, tx] + target_velocity_radius
    rx_velocity_radius = velocity_uncertainty[:, rx] + target_velocity_radius
    tx_radial_radius = tx_velocity_radius + (
        tx_relative_speed + tx_velocity_radius) * tx_unit_radius
    rx_radial_radius = rx_velocity_radius + (
        rx_relative_speed + rx_velocity_radius) * rx_unit_radius
    doppler_radius = (
        float(fc_hz) / C_LIGHT
        * (tx_radial_radius + rx_radial_radius)
        * int(doppler_bins) * float(symbol_period_s))
    ambiguity_lower = (
        _sinc_alignment_lower_array(delay_fraction, delay_radius)
        * _sinc_alignment_lower_array(doppler_fraction, doppler_radius))
    if str(dd_gain_mode) == "continuous":
        lower_support = (
            (tau - delay_radius / (
                int(delay_bins) * float(delta_f_hz)) >= 0.0)
            & (tau + delay_radius / (
                int(delay_bins) * float(delta_f_hz))
               < 1.0 / float(delta_f_hz))
            & ((np.abs(nu) + doppler_radius / (
                int(doppler_bins) * float(symbol_period_s)))
               <= 1.0 / (2.0 * float(symbol_period_s))))
        lower_dd = lower_support.astype(np.float64) * ambiguity_lower ** 2
    else:
        lower_dd = (
            ambiguity_lower >= float(dd_gate_min)).astype(np.float64)
    lower = (
        float(coefficient_scale) * alpha_sq
        * (tx_range / (tx_range + certificate_tx + target_radius)) ** 2
        * (rx_range / (rx_range + certificate_rx + target_radius)) ** 2
        * lower_dd * executable.astype(np.float64))

    upper_tx_range = np.maximum(
        tx_range - certificate_tx - target_radius, float(uav_height_m))
    upper_rx_range = np.maximum(
        rx_range - certificate_rx - target_radius, float(uav_height_m))
    upper_tx_range = np.where(
        seen[:, tx, target], upper_tx_range, float(uav_height_m))
    upper_rx_range = np.where(
        seen[:, rx, target], upper_rx_range, float(uav_height_m))
    upper = float(coefficient_scale) * path_constant / (
        upper_tx_range ** 2 * upper_rx_range ** 2)
    return SelectedBistaticCoefficients(selected, nominal, lower, upper)


def plan_budget_certified_hyperedges(
    coefficient: np.ndarray,
    sensing_budget_w: np.ndarray,
    target_deficit: np.ndarray,
    *,
    p_fa: float,
    p_d_floor: float,
    target_pair_limit: int,
    reports_per_receiver: int,
) -> LocalHyperedgePlan:
    """Replicate a budget-feasible structure solve from one public local view.

    The solver receives only a graph reconstructed from delivered endpoint
    messages.  Its returned support is a feasible primal certificate under
    single-role, unique-owner, target-cardinality, receiver-capacity, and
    per-UAV sensing-power constraints.  Global optimality is not claimed.
    """
    from uav_isac.physical.feasibility_oracle import (
        solve_enumerated_role_ceiling_local_pairs,
    )

    gain = np.asarray(coefficient, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    deficit = np.asarray(target_deficit, dtype=np.float64).reshape(-1)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = gain.shape
    if budget.shape != (K,) or deficit.shape != (Q,):
        raise ValueError("budget/deficit shapes do not match coefficient")
    priority = 1.0 + np.clip(deficit, 0.0, 1.0)
    solution = solve_enumerated_role_ceiling_local_pairs(
        gain,
        budget,
        P_FA=float(p_fa),
        p_d_floor=float(p_d_floor),
        target_pair_limit=int(target_pair_limit),
        reports_per_receiver=int(reports_per_receiver),
        target_priority=priority,
    )
    scores = {
        edge: float(
            gain[edge] * solution.sensing_power_w[edge[0], edge[2]])
        for edge in solution.selected_set
    }
    role = np.zeros(K, dtype=bool)
    if solution.tx_indices:
        role[np.asarray(solution.tx_indices, dtype=np.int64)] = True
    return LocalHyperedgePlan(
        selected=tuple(solution.selected_set),
        proxy_target_value=np.asarray(solution.D_q, dtype=np.float64),
        proxy_scores=scores,
        role_mask=role,
    )


def plan_reserved_endpoint_hyperedges(
    coefficient: np.ndarray,
    visible: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    viewer: int,
    tx_role_mask: np.ndarray,
    target_pair_limit: int,
    normalized_edge_information: np.ndarray | None = None,
    information_weight: float = 0.0,
) -> LocalHyperedgePlan:
    """Create one endpoint-local plan under stable role/owner reservations.

    Every Rx ranks only Tx packets present in its inbox; a Tx independently
    endorses each visible Rx edge. Hence an endorsement depends only on the two
    endpoints, not on identical global snapshots. ``mutual_endpoint_consensus``
    subsequently elects one owner from mutually endorsed receiver bids.
    """
    gain = np.asarray(coefficient, dtype=np.float64)
    seen = np.asarray(visible, dtype=bool)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    role = np.asarray(tx_role_mask, dtype=bool).reshape(-1)
    edge_information = None
    if normalized_edge_information is not None:
        edge_information = np.asarray(
            normalized_edge_information, dtype=np.float64)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = gain.shape
    if seen.shape != (K, Q) or budget.shape != (K,) or role.shape != (K,):
        raise ValueError("reserved endpoint input shapes are inconsistent")
    weight = float(information_weight)
    if (
        not np.isfinite(weight) or weight < 0.0
        or (
            edge_information is not None
            and (
                edge_information.shape != (K, K, Q, 4, 4)
                or np.any(~np.isfinite(edge_information))
            )
        )
    ):
        raise ValueError('reserved endpoint information inputs are invalid')
    if not 0 <= int(viewer) < K:
        raise ValueError("viewer index is out of range")
    tx_nodes = tuple(int(i) for i in np.flatnonzero(role))
    rx_nodes = tuple(int(i) for i in np.flatnonzero(~role))
    if not tx_nodes or not rx_nodes:
        return LocalHyperedgePlan(
            selected=(), proxy_target_value=np.zeros(Q),
            proxy_scores={}, role_mask=role.copy())
    limit = max(1, int(target_pair_limit))
    selected: list[Hyperedge] = []
    scores: Dict[Hyperedge, float] = {}
    target_value = np.zeros(Q, dtype=np.float64)
    for q in range(Q):
        if int(viewer) in rx_nodes:
            owner = int(viewer)
            incoming = []
            for tx in tx_nodes:
                if not (seen[tx, q] and seen[owner, q]):
                    continue
                score = float(budget[tx] * gain[tx, owner, q])
                if score > 0.0:
                    incoming.append((score, (tx, owner, q)))
            if edge_information is None or weight <= 0.0:
                incoming.sort(key=lambda item: (-item[0], item[1]))
                chosen = incoming[:limit]
            else:
                # F(S)=log(1+sum D_e)+w/2 logdet(I+sum A_e), where
                # A_e=P^(1/2) H_e^T R_e^-1 H_e P^(1/2) is PSD.  Both terms
                # are monotone submodular in S, hence cardinality-greedy has
                # the standard (1-1/e) approximation guarantee.
                remaining = list(incoming)
                chosen = []
                total_deflection = 0.0
                total_information = np.zeros((4, 4), dtype=np.float64)

                def objective(deflection, information):
                    sign, logdet = np.linalg.slogdet(
                        np.eye(4, dtype=np.float64) + information)
                    if sign <= 0.0:
                        raise ValueError('normalized information must be PSD')
                    return (
                        float(np.log1p(deflection))
                        + 0.5 * weight * float(logdet)
                    )

                current_objective = objective(
                    total_deflection, total_information)
                while remaining and len(chosen) < limit:
                    candidates = []
                    for nominal, edge in remaining:
                        increment = edge_information[edge]
                        candidate_objective = objective(
                            total_deflection + nominal,
                            total_information + increment,
                        )
                        candidates.append((
                            candidate_objective - current_objective,
                            nominal,
                            edge,
                            increment,
                            candidate_objective,
                        ))
                    candidates.sort(key=lambda item: (-item[0], item[2]))
                    marginal, nominal, edge, increment, updated = candidates[0]
                    chosen.append((marginal, edge))
                    total_deflection += nominal
                    total_information += increment
                    current_objective = updated
                    remaining = [
                        item for item in remaining if item[1] != edge]
            for score, edge in chosen:
                selected.append(edge)
                scores[edge] = score
                target_value[q] += score
        elif role[int(viewer)]:
            tx = int(viewer)
            for owner in rx_nodes:
                if seen[tx, q] and seen[owner, q]:
                    score = float(budget[tx] * gain[tx, owner, q])
                    if score > 0.0:
                        edge = (tx, owner, q)
                        selected.append(edge)
                        scores[edge] = score
                        target_value[q] += score
    return LocalHyperedgePlan(
        selected=tuple(sorted(selected)),
        proxy_target_value=target_value,
        proxy_scores=scores,
        role_mask=role.copy(),
    )


def refine_role_mask_local_search(
    coefficient: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    target_pair_limit: int,
    initial_role_mask: np.ndarray | None = None,
    max_rounds: int | None = None,
) -> np.ndarray:
    """Find a polynomial-time max-min Tx/Rx cut from public coefficients.

    For a role cut, each target's ceiling is the strongest receiver-local sum
    of at most ``target_pair_limit`` incoming Tx contributions.  Candidate
    one-node flips and Tx/Rx swaps are compared by the sorted vector of target
    ceilings, which is the lexicographic max-min order.  Each accepted move
    strictly improves that finite objective, so the procedure terminates at a
    1-flip/1-swap local optimum.  It avoids the ``2**K`` role enumeration used
    by the diagnostic replicated-global solver.
    """
    gain = np.asarray(coefficient, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = gain.shape
    if budget.shape != (K,):
        raise ValueError("sensing_budget_w must have shape (K,)")
    if not (np.all(np.isfinite(gain)) and np.all(np.isfinite(budget))):
        raise ValueError("role-refinement inputs must be finite")
    if initial_role_mask is None:
        role = np.arange(K, dtype=np.int64) % 2 == 0
    else:
        role = np.asarray(initial_role_mask, dtype=bool).reshape(-1).copy()
        if role.shape != (K,):
            raise ValueError("initial_role_mask must have shape (K,)")
    if K < 2:
        return role
    if not np.any(role) or np.all(role):
        role = np.arange(K, dtype=np.int64) % 2 == 0
    limit = max(1, int(target_pair_limit))

    def target_ceiling(candidate: np.ndarray) -> np.ndarray:
        tx_nodes = np.flatnonzero(candidate)
        rx_nodes = np.flatnonzero(~candidate)
        values = np.zeros(Q, dtype=np.float64)
        for q in range(Q):
            for rx in rx_nodes:
                incoming = np.sort(
                    budget[tx_nodes] * gain[tx_nodes, int(rx), q])
                if incoming.size:
                    values[q] = max(
                        values[q], float(np.sum(incoming[-limit:])))
        return values

    def key(candidate: np.ndarray) -> tuple:
        values = target_ceiling(candidate)
        # Rounded physics values prevent machine-epsilon changes from causing
        # different discrete roles on otherwise identical quantized caches.
        fairness = tuple(np.round(np.sort(values), 15).tolist())
        return fairness + (
            round(float(np.sum(values)), 15),
            tuple(bool(value) for value in candidate),
        )

    current_key = key(role)
    rounds = max(1, int(max_rounds if max_rounds is not None else 2 * K))
    for _ in range(rounds):
        best_role = role
        best_key = current_key
        candidates = []
        for node in range(K):
            candidate = role.copy()
            candidate[node] = ~candidate[node]
            if np.any(candidate) and not np.all(candidate):
                candidates.append(candidate)
        for tx in np.flatnonzero(role):
            for rx in np.flatnonzero(~role):
                candidate = role.copy()
                candidate[int(tx)] = False
                candidate[int(rx)] = True
                candidates.append(candidate)
        for candidate in candidates:
            candidate_key = key(candidate)
            if candidate_key > best_key:
                best_key = candidate_key
                best_role = candidate
        if best_key <= current_key:
            break
        role = best_role
        current_key = best_key
    return role


def plan_refined_endpoint_hyperedges(
    coefficient: np.ndarray,
    visible: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    viewer: int,
    target_pair_limit: int,
) -> LocalHyperedgePlan:
    """Endpoint proposal using a locally reconstructed max-min role cut."""
    role = refine_role_mask_local_search(
        coefficient,
        sensing_budget_w,
        target_pair_limit=target_pair_limit,
    )
    return plan_reserved_endpoint_hyperedges(
        coefficient,
        visible,
        sensing_budget_w,
        viewer=viewer,
        tx_role_mask=role,
        target_pair_limit=target_pair_limit,
    )


def select_sparse_tx_coalition_role(
    coefficient: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    target_pair_limit: int,
    max_transmitters: int,
) -> np.ndarray:
    """Select a budget-coupled sparse Tx coalition in polynomial time.

    For each coalition of at most ``max_transmitters`` nodes, every target
    nominates the receiver with the largest coalition ceiling.  The resulting
    fixed structure is then scored by the exact per-UAV max-min power LP, so a
    transmitter's budget cannot be reused independently by every target.  For
    fixed coalition size ``r`` the search is ``O(K**r)`` rather than ``2**K``.
    """
    from uav_isac.coordination.maxmin_power import (
        solve_fixed_structure_maxmin_power_lp,
    )

    gain = np.asarray(coefficient, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    K, _, Q = gain.shape
    if budget.shape != (K,):
        raise ValueError("sensing_budget_w must have shape (K,)")
    if not (np.all(np.isfinite(gain)) and np.all(np.isfinite(budget))):
        raise ValueError("coalition inputs must be finite")
    if K < 2:
        return np.zeros(K, dtype=bool)
    limit = max(1, int(target_pair_limit))
    coalition_limit = max(1, min(int(max_transmitters), K - 1))
    eligible = tuple(int(i) for i in np.flatnonzero(budget > 0.0))
    best_key = None
    best_role = np.arange(K, dtype=np.int64) % 2 == 0
    for size in range(1, min(coalition_limit, len(eligible)) + 1):
        for coalition in combinations(eligible, size):
            receivers = tuple(i for i in range(K) if i not in coalition)
            if not receivers:
                continue
            fixed_gain = np.zeros((K, Q), dtype=np.float64)
            feasible = True
            for q in range(Q):
                owners = []
                for owner in receivers:
                    ranked = sorted(
                        coalition,
                        key=lambda tx: (
                            budget[tx] * gain[tx, owner, q], -tx),
                        reverse=True,
                    )
                    chosen = tuple(
                        tx for tx in ranked[:limit]
                        if gain[tx, owner, q] > 0.0
                    )
                    if chosen:
                        owners.append((
                            float(sum(
                                budget[tx] * gain[tx, owner, q]
                                for tx in chosen)),
                            -int(owner), int(owner), chosen,
                        ))
                if not owners:
                    feasible = False
                    break
                _ceiling, _tie, owner, chosen = max(owners)
                for tx in chosen:
                    fixed_gain[int(tx), q] = gain[int(tx), int(owner), q]
            if not feasible:
                continue
            result = solve_fixed_structure_maxmin_power_lp(
                fixed_gain, budget)
            target_value = np.asarray(result.deflection, dtype=np.float64)
            score = tuple(np.round(np.sort(target_value), 15).tolist()) + (
                round(float(np.sum(target_value)), 15),
                -len(coalition),
                tuple(-int(node) for node in coalition),
            )
            if best_key is None or score > best_key:
                best_key = score
                best_role = np.zeros(K, dtype=bool)
                best_role[np.asarray(coalition, dtype=np.int64)] = True
    return best_role


def plan_sparse_coalition_endpoint_hyperedges(
    coefficient: np.ndarray,
    visible: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    viewer: int,
    target_pair_limit: int,
    max_transmitters: int,
) -> LocalHyperedgePlan:
    """Endpoint proposals under a public, budget-coupled sparse Tx cut."""
    role = select_sparse_tx_coalition_role(
        coefficient,
        sensing_budget_w,
        target_pair_limit=target_pair_limit,
        max_transmitters=max_transmitters,
    )
    return plan_reserved_endpoint_hyperedges(
        coefficient,
        visible,
        sensing_budget_w,
        viewer=viewer,
        tx_role_mask=role,
        target_pair_limit=target_pair_limit,
    )


def _edge_score(
    tx_capability: np.ndarray,
    rx_capability: np.ndarray,
    target_deficit: np.ndarray,
    tx: int,
    rx: int,
    target: int,
    deficit_gain: float,
    pair_value: np.ndarray | None = None,
) -> float:
    # The geometric mean avoids collapsing a useful edge merely because one
    # role proxy is numerically sharper than the other, while still requiring
    # both directed endpoints to be capable.
    role_value = (
        float(pair_value[tx, rx, target])
        if pair_value is not None
        else np.sqrt(max(
            float(tx_capability[tx, target])
            * float(rx_capability[rx, target]),
            0.0,
        ))
    )
    return role_value * (
        1.0 + max(0.0, float(deficit_gain))
        * float(np.clip(target_deficit[target], 0.0, 1.0))
    )


def plan_local_hyperedges(
    tx_capability: np.ndarray,
    rx_capability: np.ndarray,
    visible: np.ndarray,
    target_deficit: np.ndarray,
    *,
    target_pair_limit: int,
    deficit_gain: float,
    proxy_floor: float,
    pair_value: np.ndarray | None = None,
) -> LocalHyperedgePlan:
    """Choose a lexicographic max-min directed plan from one local view.

    A role mask is enumerated exactly.  Once roles are fixed, targets are
    separable because an endpoint may sense multiple targets while retaining
    one Tx/Rx role.  This is inexpensive for the intended 4--8 UAV audit and
    avoids importing a centralized environment solver.
    """
    tx_value = np.asarray(tx_capability, dtype=np.float64)
    rx_value = np.asarray(rx_capability, dtype=np.float64)
    seen = np.asarray(visible, dtype=bool)
    deficit = np.asarray(target_deficit, dtype=np.float64)
    if tx_value.shape != rx_value.shape or tx_value.ndim != 2:
        raise ValueError("capability matrices must have shape (K, Q)")
    k_count, q_count = tx_value.shape
    if seen.shape != (k_count, q_count):
        raise ValueError("visible must have shape (K, Q)")
    if deficit.shape != (q_count,):
        raise ValueError("target_deficit must have shape (Q,)")
    pair_score = None
    if pair_value is not None:
        pair_score = np.asarray(pair_value, dtype=np.float64)
        if pair_score.shape != (k_count, k_count, q_count):
            raise ValueError("pair_value must have shape (K, K, Q)")
    if not (np.all(np.isfinite(tx_value))
            and np.all(np.isfinite(rx_value))
            and np.all(np.isfinite(deficit))
            and (pair_score is None or np.all(np.isfinite(pair_score)))):
        raise ValueError("planner inputs must be finite")

    limit = max(1, int(target_pair_limit))
    floor = max(float(proxy_floor), 1e-9)
    best_key = None
    best_edges: Tuple[Hyperedge, ...] = tuple()
    best_target_value = np.zeros(q_count, dtype=np.float64)
    best_scores: Dict[Hyperedge, float] = {}
    best_roles = np.zeros(k_count, dtype=bool)

    # False=Rx, True=Tx. Exclude the all-same partitions.
    for role_tuple in product((False, True), repeat=k_count):
        role = np.asarray(role_tuple, dtype=bool)
        if not np.any(role) or np.all(role):
            continue
        scores: Dict[Hyperedge, float] = {}
        selected = []
        target_value = np.zeros(q_count, dtype=np.float64)
        tx_nodes = np.flatnonzero(role)
        rx_nodes = np.flatnonzero(~role)
        for target in range(q_count):
            # The deployed local-only detector may coherently accumulate
            # several Tx echoes at one receiver, but cannot add statistics
            # held by different receivers.  Nominate exactly one receiver
            # owner for each target before choosing its incoming Tx edges.
            owner_candidates = []
            for rx in rx_nodes:
                if not seen[rx, target]:
                    continue
                incoming = []
                for tx in tx_nodes:
                    if tx == rx or not seen[tx, target]:
                        continue
                    edge = (int(tx), int(rx), int(target))
                    score = _edge_score(
                        tx_value, rx_value, deficit,
                        int(tx), int(rx), int(target), deficit_gain,
                        pair_score,
                    )
                    if score > 0.0:
                        incoming.append((score, edge))
                incoming.sort(key=lambda item: (-item[0], item[1]))
                retained = tuple(incoming[:limit])
                if retained:
                    owner_candidates.append((
                        float(sum(score for score, _ in retained)),
                        int(rx),
                        retained,
                    ))
            owner_candidates.sort(
                key=lambda item: (-item[0], item[1]))
            retained = (
                owner_candidates[0][2] if owner_candidates else tuple())
            for score, edge in retained:
                selected.append(edge)
                scores[edge] = float(score)
                target_value[target] += float(score)

        # Primary: max-min proxy. Secondary: deficit-weighted floor coverage.
        # The remaining fields are deterministic stabilizers.
        capped = np.minimum(target_value / floor, 1.0)
        weights = 1.0 + np.clip(deficit, 0.0, 1.0)
        coverage_value = float(np.dot(weights, capped))
        selected_tuple = tuple(sorted(selected))
        key = (
            float(np.min(target_value)) if q_count else 0.0,
            coverage_value,
            float(np.sum(target_value)),
            -len(selected_tuple),
            tuple(-value for edge in selected_tuple for value in edge),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_edges = selected_tuple
            best_target_value = target_value
            best_scores = scores
            best_roles = role

    return LocalHyperedgePlan(
        selected=best_edges,
        proxy_target_value=best_target_value,
        proxy_scores=best_scores,
        role_mask=best_roles,
    )


def mutual_endpoint_consensus(
    plans: Iterable[LocalHyperedgePlan],
    *,
    num_uavs: int,
    num_targets: int,
    target_pair_limit: int,
) -> Tuple[Hyperedge, ...]:
    """Keep only directed edges endorsed by both of their endpoints."""
    local = tuple(plans)
    if len(local) != int(num_uavs):
        raise ValueError("one local plan is required per UAV")
    selected_sets = [set(plan.selected) for plan in local]
    candidates = set()
    for tx, rx, target in set().union(*selected_sets):
        edge = (int(tx), int(rx), int(target))
        if not (0 <= tx < num_uavs and 0 <= rx < num_uavs
                and 0 <= target < num_targets and tx != rx):
            continue
        if edge in selected_sets[tx] and edge in selected_sets[rx]:
            candidates.add(edge)

    # Mutual endorsement preserves the stable endpoint roles. Receiver-local
    # detection additionally requires one owner per target: aggregate the
    # mutually endorsed support by receiver, elect one owner, then retain at
    # most the target edge limit. This is the deterministic reduction every
    # node can reproduce from the same endorsement set in a commit round.
    by_target_owner: dict[int, dict[int, list[tuple[float, Hyperedge]]]] = {
        q: {} for q in range(int(num_targets))}
    for edge in sorted(candidates):
        tx, rx, target = edge
        support = 0.5 * (
            float(local[tx].proxy_scores.get(edge, 0.0))
            + float(local[rx].proxy_scores.get(edge, 0.0))
        )
        by_target_owner[target].setdefault(rx, []).append((support, edge))

    chosen = []
    limit = max(1, int(target_pair_limit))
    for target in range(int(num_targets)):
        owner_groups = by_target_owner[target]
        if not owner_groups:
            continue
        owner = min(
            owner_groups,
            key=lambda rx: (
                -sum(score for score, _edge in owner_groups[rx]), rx),
        )
        ranked = sorted(
            owner_groups[owner], key=lambda item: (-item[0], item[1]))
        chosen.extend(edge for _, edge in ranked[:limit])

    # A final explicit invariant check makes integration errors fail closed.
    tx_nodes = {tx for tx, _, _ in chosen}
    rx_nodes = {rx for _, rx, _ in chosen}
    if tx_nodes & rx_nodes:
        raise RuntimeError("mutual plans violated the single-role invariant")
    return tuple(sorted(chosen))


def update_consensus_streak(
    previous: np.ndarray,
    mutual_edges: Iterable[Hyperedge],
    *,
    consensus_rounds: int,
) -> Tuple[np.ndarray, Tuple[Hyperedge, ...]]:
    """Require the same reciprocal edge for consecutive communication rounds."""
    streak = np.asarray(previous, dtype=np.int64).copy()
    if streak.ndim != 3 or streak.shape[0] != streak.shape[1]:
        raise ValueError("previous streak must have shape (K, K, Q)")
    mutual_mask = np.zeros_like(streak, dtype=bool)
    for tx, rx, target in mutual_edges:
        mutual_mask[int(tx), int(rx), int(target)] = True
    streak = np.where(mutual_mask, streak + 1, 0)
    required = max(1, int(consensus_rounds))
    active = tuple(
        (int(tx), int(rx), int(target))
        for tx, rx, target in np.argwhere(streak >= required)
    )
    return streak, tuple(sorted(active))
