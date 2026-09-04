"""Bistatic movement potential (O5 / T7, roadmap 2026-08-29).

Under the continuous-DD physical model a target's per-watt gain scales as
``a_q ~ 1 / (R_tx(q)^2 * R_rx(q)^2)`` (bistatic distance product, manifest
T1).  Define the potential of target q as

    Phi_q = R_tx(q) * R_rx(q)

with ``R_tx(q)`` the distance from q to its closest TX UAV and ``R_rx(q)``
the distance to its closest RX UAV.  A smaller ``Phi_q`` strictly raises the
gain ceiling of q: for fixed powers the per-watt deflection scales as
``1 / Phi_q^2``, and the all-budget ceiling is ``D_q^max = sum_i A[i,q] b_i``
(T6) whose geometry-sensitive part is monotone in ``1/Phi_q^2``.

T7 (potential -- reachability monotone agreement):
    Phi_bottleneck non-increasing along accepted moves
    ==> q's gain ceiling non-decreasing ==> gate deficit
    ``D_c(q) - D_q^max`` non-increasing.
Accepted moves are exactly those that strictly decrease the bottleneck
potential, so the acceptance rule is a Lyapunov-style descent: Phi has a
positive lower bound (minimum physical stand-off), the accepted sequence is
monotone non-increasing, and over a finite candidate grid it terminates at a
local minimum in finitely many steps.

C3 boundary: the module consumes only PUBLIC geometry -- UAV positions, the
locally-believed target positions, and role masks.  No simulator truth, no
RNG (deterministic).
"""

from __future__ import annotations

import numpy as np


def coverage_deficit_lexicographic_score(
    target_xy: np.ndarray,
    uav_xy: np.ndarray,
    tx_mask: np.ndarray,
    critical_radius_m: float,
    *,
    height_m: float = 0.0,
) -> tuple[int, float, float, float, float]:
    """Public-belief score for baseline-enveloped movement.

    The tuple is minimized lexicographically:

    1. number of targets missing a Tx or Rx endpoint inside ``R_crit``;
    2. largest normalized endpoint-radius deficit;
    3. sum of normalized endpoint-radius deficits;
    4. largest log bistatic range product;
    5. sum of log bistatic range products.

    The first three terms implement the operational coverage invariant.  Once
    two actions have the same coverage deficit, the last two minimize
    ``log((H^2+r_tx^2)(H^2+r_rx^2))``.  Since fixed-geometry bistatic gain is
    proportional to the reciprocal product, this ordering is monotone with
    the physical gain ceiling while avoiding overflow and arbitrary weights.
    """
    target = np.asarray(target_xy, dtype=np.float64)
    uav = np.asarray(uav_xy, dtype=np.float64)
    tx = np.asarray(tx_mask, dtype=bool).reshape(-1)
    radius = float(critical_radius_m)
    height = float(height_m)
    if (
        target.ndim != 2 or target.shape[1] != 2
        or uav.ndim != 2 or uav.shape[1] != 2
        or tx.shape != (uav.shape[0],)
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(uav))
        or not np.isfinite(radius) or radius <= 0.0
        or not np.isfinite(height) or height < 0.0
        or not np.any(tx) or not np.any(~tx)
    ):
        raise ValueError(
            "finite target/uav (N,2), nonempty Tx/Rx masks, positive radius "
            "and non-negative height required")

    horizontal_sq = np.sum(
        (uav[:, None, :] - target[None, :, :]) ** 2, axis=-1)
    tx_horizontal = np.sqrt(np.min(horizontal_sq[tx], axis=0))
    rx_horizontal = np.sqrt(np.min(horizontal_sq[~tx], axis=0))
    tx_deficit = np.maximum(tx_horizontal / radius - 1.0, 0.0)
    rx_deficit = np.maximum(rx_horizontal / radius - 1.0, 0.0)
    endpoint_deficit = np.maximum(tx_deficit, rx_deficit)
    uncovered_count = int(np.count_nonzero(endpoint_deficit > 0.0))

    tx_range_sq = height * height + tx_horizontal * tx_horizontal
    rx_range_sq = height * height + rx_horizontal * rx_horizontal
    # The 1e-24 floor is below any physical stand-off squared product; it only
    # defines the degenerate zero-height/coincident-point logarithm.
    log_product = np.log(np.maximum(tx_range_sq * rx_range_sq, 1.0e-24))
    return (
        uncovered_count,
        float(np.max(endpoint_deficit)),
        float(np.sum(endpoint_deficit)),
        float(np.max(log_product)),
        float(np.sum(log_product)),
    )


def select_baseline_enveloped_movement(
    target_xy: np.ndarray,
    uav_xy: np.ndarray,
    tx_mask: np.ndarray,
    gap_delta_xy: np.ndarray,
    baseline_delta_xy: np.ndarray,
    critical_radius_m: float,
    *,
    height_m: float = 0.0,
) -> tuple[np.ndarray, bool, tuple[float, ...], tuple[float, ...]]:
    """Select gap movement only when its public score beats baseline.

    The baseline action is explicitly in the candidate set.  Consequently the
    selected next-state score is never lexicographically worse than baseline
    under the same delivered public belief and safety projection.  Exact ties
    choose baseline, preventing gratuitous trajectory drift.
    """
    uav = np.asarray(uav_xy, dtype=np.float64)
    gap = np.asarray(gap_delta_xy, dtype=np.float64)
    baseline = np.asarray(baseline_delta_xy, dtype=np.float64)
    if gap.shape != uav.shape or baseline.shape != uav.shape:
        raise ValueError("movement deltas must match uav_xy shape")
    if not (np.all(np.isfinite(gap)) and np.all(np.isfinite(baseline))):
        raise ValueError("movement deltas must be finite")
    gap_score = coverage_deficit_lexicographic_score(
        target_xy, uav + gap, tx_mask, critical_radius_m,
        height_m=height_m)
    baseline_score = coverage_deficit_lexicographic_score(
        target_xy, uav + baseline, tx_mask, critical_radius_m,
        height_m=height_m)
    use_gap = bool(gap_score < baseline_score)
    selected = gap if use_gap else baseline
    return selected.copy(), use_gap, gap_score, baseline_score


def role_aware_bistatic_products(
    endpoint_range_sq: np.ndarray,
    tx_mask: np.ndarray,
) -> np.ndarray:
    """Return each endpoint's product with its nearest opposite role.

    ``product[k,q]`` is ``R_kq^2 min_{j:role(j)!=role(k)} R_jq^2``.
    The role condition is essential: pairing a Tx with the nearest Tx (or an
    Rx with the nearest Rx) is not a physically valid bistatic path.
    """
    range_sq = np.asarray(endpoint_range_sq, dtype=np.float64)
    tx = np.asarray(tx_mask, dtype=bool).reshape(-1)
    if (
        range_sq.ndim != 2
        or tx.shape != (range_sq.shape[0],)
        or np.any(~np.isfinite(range_sq))
        or np.any(range_sq < 0.0)
        or not np.any(tx)
        or not np.any(~tx)
    ):
        raise ValueError(
            "finite non-negative (K,Q) ranges and nonempty Tx/Rx required")
    nearest_tx = np.min(range_sq[tx], axis=0)
    nearest_rx = np.min(range_sq[~tx], axis=0)
    complement = np.where(tx[:, None], nearest_rx, nearest_tx)
    return range_sq * complement


def _closest_role_distance(
    target_xy: np.ndarray,   # (Q,2)
    uav_xy: np.ndarray,      # (K,2)
    role_mask: np.ndarray,   # (K,) bool
) -> np.ndarray:
    """Closest distance from each target to any UAV in the role."""
    dist = np.linalg.norm(
        uav_xy[role_mask][None, :, :] - target_xy[:, None, :], axis=-1)
    if dist.size == 0:
        return np.full(target_xy.shape[0], np.inf, dtype=np.float64)
    return np.min(dist, axis=1)


def bistatic_potential(
    target_xy: np.ndarray,   # (Q,2) locally-believed target positions
    uav_xy: np.ndarray,      # (K,2)
    tx_mask: np.ndarray,     # (K,) bool
    rx_mask: np.ndarray,     # (K,) bool
) -> tuple[np.ndarray, float, int]:
    """Return ``(phi_all, phi_worst, worst_q)``.

    ``phi_all[q] = R_tx(q) * R_rx(q)``; ``phi_worst`` and ``worst_q`` are the
    maximum over targets (the bottleneck).  No truth/RNG.
    """
    target = np.asarray(target_xy, dtype=np.float64)
    uav = np.asarray(uav_xy, dtype=np.float64)
    tx = np.asarray(tx_mask, dtype=bool).reshape(-1)
    rx = np.asarray(rx_mask, dtype=bool).reshape(-1)
    if (
        target.ndim != 2 or target.shape[1] != 2
        or uav.ndim != 2 or uav.shape[1] != 2
        or tx.shape != (uav.shape[0],) or rx.shape != (uav.shape[0],)
    ):
        raise ValueError("target (Q,2), uav (K,2), masks (K,) required")
    if not (np.all(np.isfinite(target)) and np.all(np.isfinite(uav))):
        raise ValueError("positions must be finite")
    r_tx = _closest_role_distance(target, uav, tx)
    r_rx = _closest_role_distance(target, uav, rx)
    phi = r_tx * r_rx
    worst_q = int(np.argmax(phi))
    return phi, float(phi[worst_q]), worst_q


def accept_move_candidate(
    target_xy: np.ndarray,
    uav_xy: np.ndarray,
    tx_mask: np.ndarray,
    rx_mask: np.ndarray,
    uav_index: int,
    candidate_xy: np.ndarray,
) -> tuple[bool, float, int]:
    """Accept a single-UAV move iff it strictly lowers the bottleneck potential.

    Returns ``(accepted, phi_worst_after, worst_q_after)``; a move that does
    not strictly decrease the bottleneck (including ties) is rejected.
    """
    uav = np.asarray(uav_xy, dtype=np.float64)
    cand = np.asarray(candidate_xy, dtype=np.float64)
    if cand.shape != (2,):
        raise ValueError("candidate position must be (2,)")
    if not (int(uav_index) >= 0 and int(uav_index) < uav.shape[0]):
        raise ValueError("uav_index out of range")
    _, phi_before, _ = bistatic_potential(
        target_xy, uav_xy, tx_mask, rx_mask)
    next_uav = uav.copy()
    next_uav[int(uav_index)] = cand
    _, phi_after, worst_after = bistatic_potential(
        target_xy, next_uav, tx_mask, rx_mask)
    return (phi_after < phi_before), phi_after, worst_after


def potential_descent_sequence(
    target_xy: np.ndarray,
    uav_xy: np.ndarray,
    tx_mask: np.ndarray,
    rx_mask: np.ndarray,
    candidate_grid: np.ndarray,   # (M, 2) move positions offered per step
    max_steps: int,
) -> tuple[np.ndarray, float, int]:
    """Greedy bottleneck descent with the strict-accept rule.

    At each step try every UAV against every grid candidate; apply the first
    strictly-decreasing move; stop when no candidate lowers the bottleneck.
    Returns ``(phi_history, final_phi, steps_taken)``.  Deterministic, no RNG.
    """
    uav = np.asarray(uav_xy, dtype=np.float64).copy()
    phi_hist = []
    for _step in range(int(max_steps)):
        _, phi_current, worst_q = bistatic_potential(
            target_xy, uav, tx_mask, rx_mask)
        phi_hist.append(phi_current)
        moved = False
        for k in range(uav.shape[0]):
            for cand in np.asarray(candidate_grid, dtype=np.float64):
                ok, phi_new, _ = accept_move_candidate(
                    target_xy, uav, tx_mask, rx_mask, k, cand)
                if ok:
                    uav[k] = cand
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break
    _, final_phi, _ = bistatic_potential(target_xy, uav, tx_mask, rx_mask)
    return np.asarray(phi_hist, dtype=np.float64), final_phi, len(phi_hist)
