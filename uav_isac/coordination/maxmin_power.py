"""Fixed-structure max-min sensing-power repair.

Once reporting roles and one receiver owner per target are frozen, target
Deflection is linear in transmitter sensing power.  The resulting max-min
allocation is an LP.  Its simplex dual admits a distributed exponentiated-
gradient update using public target prices and transmitter-local gains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class MaxMinPowerResult:
    power_w: np.ndarray
    deflection: np.ndarray
    worst_deflection: float
    prices: np.ndarray
    dual_upper_bound: float
    primal_dual_gap: float
    rounds: int
    worst_history: tuple[float, ...]
    reserve_feasible: bool = True
    reserve_shortfall: float = 0.0
    minimum_deflection: tuple[float, ...] = ()
    communication_bits: int = 0


@dataclass(frozen=True)
class ReplicatedLocalPowerResult:
    """Row-wise execution assembled from independent node-local solves."""

    power_w: np.ndarray
    local_worst_deflection: np.ndarray
    local_full_coverage: np.ndarray
    common_view: bool


def blend_row_feasible_power_with_inertia(
    candidate_power_w: np.ndarray,
    previous_power_w: np.ndarray,
    sensing_budget_w: np.ndarray,
    inertia: float,
) -> np.ndarray:
    """Blend two row allocations without leaving the local RF simplex.

    Both allocations are first projected onto each transmitter's current
    budget simplex.  Their convex combination is therefore non-negative and
    exactly budget feasible.  The operation is entirely local: it reuses the
    transmitter's last executed row and exchanges no optimizer certificate.
    """
    candidate = np.asarray(candidate_power_w, dtype=np.float64)
    previous = np.asarray(previous_power_w, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        candidate.ndim != 2 or previous.shape != candidate.shape
        or candidate.shape[0] != budget.size or candidate.shape[1] < 1
    ):
        raise ValueError(
            "candidate_power_w and previous_power_w must have shape (K,Q) "
            "matching sensing_budget_w")
    if (
        np.any(~np.isfinite(candidate)) or np.any(candidate < 0.0)
        or np.any(~np.isfinite(previous)) or np.any(previous < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError("power and budget inputs must be finite and non-negative")
    rho = float(inertia)
    if not np.isfinite(rho) or not 0.0 <= rho < 1.0:
        raise ValueError("inertia must be finite and lie in [0,1)")

    def project_rows(power: np.ndarray) -> np.ndarray:
        projected = power.copy()
        for transmitter in range(projected.shape[0]):
            row_sum = float(np.sum(projected[transmitter]))
            row_budget = float(budget[transmitter])
            if row_budget <= 0.0:
                projected[transmitter] = 0.0
            elif row_sum > 0.0:
                projected[transmitter] *= row_budget / row_sum
            else:
                projected[transmitter] = row_budget / projected.shape[1]
        return projected

    current = project_rows(candidate)
    history = project_rows(previous)
    blended = (1.0 - rho) * current + rho * history
    # Remove only floating-point simplex residuals, not physical power.
    return project_rows(blended)


def _validated(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 2 or gain.shape[0] != budget.size or gain.shape[1] < 1:
        raise ValueError("gain_per_watt must have shape (K,Q) matching budget")
    if np.any(~np.isfinite(gain)) or np.any(gain < 0.0):
        raise ValueError("gain_per_watt must be finite and non-negative")
    if np.any(~np.isfinite(budget)) or np.any(budget < 0.0):
        raise ValueError("sensing_budget_w must be finite and non-negative")
    return gain, budget


def _fill_budget(
    power: np.ndarray,
    gain: np.ndarray,
    budget: np.ndarray,
) -> np.ndarray:
    """Select a budget-saturated optimum in the ordinary monotone regime.

    With non-negative gains, no exposure upper bound, and no power penalty,
    unused power can be placed on a best-gain target without reducing the
    max-min objective.  This is an optimizer tie-break, not a physical RF
    equality.  Constrained/covert solvers must use their own headroom-aware
    logic and may legitimately leave power unused.
    """
    result = np.maximum(np.asarray(power, dtype=np.float64), 0.0).copy()
    for transmitter in range(result.shape[0]):
        slack = float(budget[transmitter] - np.sum(result[transmitter]))
        if slack < 0.0:
            if slack < -1.0e-9:
                raise AssertionError("power allocation exceeds sensing budget")
            target = int(np.argmax(result[transmitter]))
            result[transmitter, target] += slack
            slack = float(
                budget[transmitter] - np.sum(result[transmitter]))
        if slack > 1.0e-10:
            target = int(np.argmax(gain[transmitter]))
            result[transmitter, target] += slack
    return result


def solve_fixed_structure_maxmin_power_lp(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    minimum_deflection: np.ndarray | None = None,
) -> MaxMinPowerResult:
    """Solve fixed-owner max-min and select a saturated ordinary optimum.

    When ``minimum_deflection`` is supplied, the LP additionally enforces the
    per-target reserve ``sum_i a_iq p_iq >= r_q`` before maximising the worst
    target (reserve-first).  This separates "keep every target above its
    QoS-derived floor" from "then push the weakest as high as possible", so the
    steady mean is not silently sacrificed to the max-min objective.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    reserve = None
    if minimum_deflection is not None:
        reserve = np.asarray(minimum_deflection, dtype=np.float64).reshape(-1)
        if (
            reserve.shape != (Q,) or np.any(~np.isfinite(reserve))
            or np.any(reserve < 0.0)
        ):
            raise ValueError(
                "minimum_deflection must be a finite non-negative Q-vector")
    variables = K * Q
    objective = np.zeros(variables + 1, dtype=np.float64)
    objective[-1] = -1.0
    rows = []
    upper = []
    for target in range(Q):
        row = np.zeros(variables + 1, dtype=np.float64)
        for transmitter in range(K):
            row[transmitter * Q + target] = -gain[transmitter, target]
        row[-1] = 1.0
        rows.append(row)
        upper.append(0.0)
    if reserve is not None:
        for target in range(Q):
            row = np.zeros(variables + 1, dtype=np.float64)
            for transmitter in range(K):
                row[transmitter * Q + target] = -gain[transmitter, target]
            rows.append(row)
            upper.append(float(-reserve[target]))
    for transmitter in range(K):
        row = np.zeros(variables + 1, dtype=np.float64)
        row[transmitter * Q:(transmitter + 1) * Q] = 1.0
        rows.append(row)
        upper.append(float(budget[transmitter]))
    bounds = [
        (0.0, float(budget[transmitter]))
        for transmitter in range(K)
        for _ in range(Q)
    ] + [(0.0, None)]
    solved = linprog(
        objective,
        A_ub=np.stack(rows),
        b_ub=np.asarray(upper, dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if not solved.success or solved.x is None:
        if reserve is not None:
            # A reserve that exceeds the reachable ceiling makes the LP
            # infeasible; report it explicitly rather than masking the cause.
            ceiling = np.sum(gain * budget[:, None], axis=0)
            raise RuntimeError(
                "fixed-structure max-min LP with reserve is infeasible; "
                f"reachable per-target ceiling={ceiling.tolist()}, "
                f"reserve={reserve.tolist()}")
        raise RuntimeError(f"fixed-structure max-min LP failed: {solved.message}")
    power = _fill_budget(
        solved.x[:variables].reshape(K, Q), gain, budget)
    deflection = np.sum(gain * power, axis=0)
    worst = float(np.min(deflection))
    reserve_feasible = bool(
        reserve is None or np.all(deflection + 1.0e-12 >= reserve))
    # Strong duality holds for this feasible bounded LP.  The optimum itself
    # is the tightest reference upper bound available from linprog.
    return MaxMinPowerResult(
        power_w=power,
        deflection=deflection,
        worst_deflection=worst,
        prices=np.full(Q, 1.0 / Q, dtype=np.float64),
        dual_upper_bound=worst,
        primal_dual_gap=0.0,
        rounds=0,
        worst_history=(worst,),
        reserve_feasible=reserve_feasible,
        reserve_shortfall=float(
            0.0 if reserve is None else np.max(np.maximum(reserve - deflection, 0.0))),
        minimum_deflection=tuple(
            () if reserve is None else reserve.tolist()),
    )


def replicated_local_row_maxmin_power(
    gain_views_per_watt: np.ndarray,
    public_sensing_budget_w: np.ndarray,
    executed_sensing_budget_w: np.ndarray | None = None,
    unknown_target_reserve_fraction: float = 1.0,
    incomplete_view_prior_share: np.ndarray | None = None,
) -> ReplicatedLocalPowerResult:
    """Execute only each transmitter's row from its private public-cache LP.

    ``gain_views_per_watt[k]`` is UAV ``k``'s locally reconstructed ``(K,Q)``
    gain matrix. Every UAV solves the same fixed-structure LP using its own
    cache, but transmitter ``k`` executes only row ``k``. No primal/dual
    certificate or optimizer state is exchanged.

    When all public views and budgets agree, deterministic local solves are
    identical and the assembled allocation equals the ordinary centralized LP
    optimum. Under packet loss the views may disagree; global max-min
    optimality is then deliberately not claimed, while each executed row is
    rescaled to its own physical budget and remains feasible.
    """
    views = np.asarray(gain_views_per_watt, dtype=np.float64)
    public_budget = np.asarray(
        public_sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        views.ndim != 3 or views.shape[0] != views.shape[1]
        or views.shape[0] != public_budget.size or views.shape[2] < 1
    ):
        raise ValueError(
            "gain_views_per_watt must have shape (K,K,Q) matching budget")
    if np.any(~np.isfinite(views)) or np.any(views < 0.0):
        raise ValueError(
            "gain_views_per_watt must be finite and non-negative")
    if np.any(~np.isfinite(public_budget)) or np.any(public_budget < 0.0):
        raise ValueError(
            "public_sensing_budget_w must be finite and non-negative")
    executed_budget = (
        public_budget.copy()
        if executed_sensing_budget_w is None
        else np.asarray(
            executed_sensing_budget_w, dtype=np.float64).reshape(-1)
    )
    if (
        executed_budget.shape != public_budget.shape
        or np.any(~np.isfinite(executed_budget))
        or np.any(executed_budget < 0.0)
    ):
        raise ValueError(
            "executed_sensing_budget_w must be a finite non-negative K-vector")
    reserve_fraction = float(unknown_target_reserve_fraction)
    if (
        not np.isfinite(reserve_fraction)
        or not 0.0 <= reserve_fraction <= 1.0
    ):
        raise ValueError(
            "unknown_target_reserve_fraction must lie in [0,1]")
    prior_share = None
    if incomplete_view_prior_share is not None:
        prior_share = np.asarray(
            incomplete_view_prior_share, dtype=np.float64)
        if (
            prior_share.shape != (views.shape[0], views.shape[2])
            or np.any(~np.isfinite(prior_share))
            or np.any(prior_share < 0.0)
            or np.any(np.sum(prior_share, axis=1) <= 0.0)
        ):
            raise ValueError(
                "incomplete_view_prior_share must be a finite non-negative "
                "(K,Q) array with positive rows")
        prior_share = prior_share / np.sum(
            prior_share, axis=1, keepdims=True)

    K, _, Q = views.shape
    power = np.zeros((K, Q), dtype=np.float64)
    local_worst = np.zeros(K, dtype=np.float64)
    local_coverage = np.zeros(K, dtype=bool)
    for viewer in range(K):
        local_gain = views[viewer]
        ceiling = np.sum(
            local_gain * public_budget[:, None], axis=0)
        local_coverage[viewer] = bool(np.all(ceiling > 0.0))
        if not local_coverage[viewer]:
            # A zero in an incomplete/quantized public graph is not evidence
            # that the already selected physical edge is truly unreachable.
            # Keep an explicit uniform floor for unknown targets, then use the
            # remainder on targets that are reachable in this local cache.
            # No reachability certificate is exchanged: the mask is private.
            uniform = np.full(Q, 1.0 / Q, dtype=np.float64)
            observed = ceiling > 0.0
            exploit = np.zeros(Q, dtype=np.float64)
            if reserve_fraction < 1.0 and prior_share is not None:
                exploit = prior_share[viewer].copy()
            elif reserve_fraction < 1.0 and np.any(observed):
                local = solve_fixed_structure_maxmin_power_lp(
                    local_gain[:, observed], public_budget)
                local_worst[viewer] = float(local.worst_deflection)
                row = np.maximum(local.power_w[viewer], 0.0)
                row_mass = float(np.sum(row))
                if row_mass > 0.0:
                    exploit[observed] = row / row_mass
                else:
                    own = np.maximum(local_gain[viewer, observed], 0.0)
                    if float(np.sum(own)) > 0.0:
                        observed_indices = np.flatnonzero(observed)
                        exploit[observed_indices[int(np.argmax(own))]] = 1.0
                    else:
                        exploit = uniform.copy()
            else:
                exploit = uniform.copy()
            share = (
                reserve_fraction * uniform
                + (1.0 - reserve_fraction) * exploit
            )
            power[viewer] = float(executed_budget[viewer]) * share
            continue
        local = solve_fixed_structure_maxmin_power_lp(
            local_gain, public_budget)
        local_worst[viewer] = float(local.worst_deflection)
        row = np.maximum(local.power_w[viewer], 0.0)
        row_mass = float(np.sum(row))
        if row_mass > 0.0:
            power[viewer] = row * (
                float(executed_budget[viewer]) / row_mass)
        elif executed_budget[viewer] > 0.0:
            # Stable local tie-break; this does not create a coordination
            # channel and still respects the transmitter's own RF cap.
            target = int(np.argmax(local_gain[viewer]))
            power[viewer, target] = float(executed_budget[viewer])

    common_view = bool(np.allclose(
        views, views[0][None, :, :], rtol=0.0, atol=1.0e-15))
    return ReplicatedLocalPowerResult(
        power_w=power,
        local_worst_deflection=local_worst,
        local_full_coverage=local_coverage,
        common_view=common_view,
    )


def local_transmitter_range_minimax_share(
    transmitter_positions_m: np.ndarray,
    target_positions_m: np.ndarray,
) -> np.ndarray:
    """Return the row-local range-only minimax power prior.

    If an unknown receiver range is bounded by the common mission-domain
    radius ``R_max``, bistatic free-space gain obeys
    ``a_ijq >= C/(R_iq^2 R_max^2)`` (conditional on the selected DD gate).
    Choosing ``p_iq proportional to R_iq^2`` equalizes this conservative
    transmitter-side contribution across targets. Each row depends only on
    transmitter ``i``'s own position and the common target map.
    """
    transmitters = np.asarray(transmitter_positions_m, dtype=np.float64)
    targets = np.asarray(target_positions_m, dtype=np.float64)
    if (
        transmitters.ndim != 2 or targets.ndim != 2
        or transmitters.shape[1] != targets.shape[1]
        or transmitters.shape[0] < 1 or targets.shape[0] < 1
        or np.any(~np.isfinite(transmitters))
        or np.any(~np.isfinite(targets))
    ):
        raise ValueError(
            "transmitter_positions_m and target_positions_m must be finite "
            "non-empty arrays with a common coordinate dimension")
    range_squared = np.sum(
        (transmitters[:, None, :] - targets[None, :, :]) ** 2,
        axis=-1,
    )
    # Coincident positions need no infinite priority; the smallest positive
    # float keeps every row normalizable and its corresponding share minimal.
    range_squared = np.maximum(range_squared, np.finfo(np.float64).tiny)
    return range_squared / np.sum(range_squared, axis=1, keepdims=True)


def _quantize_simplex(prices: np.ndarray, bits: int) -> np.ndarray:
    if int(bits) <= 0:
        return prices
    levels = (1 << int(bits)) - 1
    rounded = np.rint(prices * levels) / levels
    total = float(np.sum(rounded))
    return (
        rounded / total
        if total > 0.0
        else np.full_like(prices, 1.0 / prices.size)
    )


def _quantize_deflection_feedback(
    deflection: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Apply the actual scalar codec used by owner feedback packets.

    Zero bits keeps the historical exact diagnostic.  Deployment-supported
    packets use standard IEEE binary16/binary32 scalars, so no uncounted
    dynamic-range side channel is required.  Values outside binary16 support
    saturate explicitly and their resulting optimizer error remains visible
    to the outer feedback calibration layer.
    """
    values = np.asarray(deflection, dtype=np.float64)
    count = int(bits)
    if count == 0 or count == 64:
        return values.copy()
    if count == 16:
        limit = float(np.finfo(np.float16).max)
        return np.asarray(np.minimum(values, limit), dtype=np.float16).astype(
            np.float64)
    if count == 32:
        limit = float(np.finfo(np.float32).max)
        return np.asarray(np.minimum(values, limit), dtype=np.float32).astype(
            np.float64)
    raise ValueError("feedback_bits must be one of {0,16,32,64}")


def distributed_dual_maxmin_power(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    rounds: int,
    price_bits: int = 0,
    feedback_bits: int = 0,
    initial_prices: np.ndarray | None = None,
    harmonic_safe_start: bool = True,
) -> MaxMinPowerResult:
    """Finite-round distributed dual mirror descent with safe primal recovery.

    A target-price vector is public.  In each round every transmitter spends
    its sensing budget on the target maximizing ``lambda_q * a_iq``.  Owners
    aggregate achieved target Deflection, and exponentiated-gradient descent
    updates the simplex price.  The best feasible ergodic primal average is
    returned; no monotonicity is assumed for the last iterate.

    The optional safe start removes the finite-round target-starvation failure
    of winner-take-all responses.  With ``c_q = sum_i b_i a_iq`` and
    ``w_q = c_q^-1 / sum_r c_r^-1``, the allocation ``p_iq=b_i w_q`` is
    feasible and gives every target the same Deflection
    ``t_H=(sum_q c_q^-1)^-1``.  It is therefore a constructive distributed
    lower bound, not an oracle fallback.  Subsequent ergodic checkpoints are
    accepted only when they improve this incumbent.

    ``price_bits=feedback_bits=0`` retains the exact floating-point diagnostic
    and reports zero wire bits.  A deployable run must select explicit codecs;
    feedback uses the binary16/binary32 codec accepted by
    :func:`_quantize_deflection_feedback`.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    count = int(rounds)
    if count < 1:
        raise ValueError("rounds must be positive")
    if int(price_bits) < 0 or int(price_bits) > 24:
        raise ValueError("price_bits must lie in [0,24]")
    if int(feedback_bits) not in (0, 16, 32, 64):
        raise ValueError("feedback_bits must be one of {0,16,32,64}")
    if initial_prices is None:
        prices = np.full(Q, 1.0 / Q, dtype=np.float64)
    else:
        prices = np.asarray(initial_prices, dtype=np.float64).reshape(-1)
        if prices.shape != (Q,) or np.any(~np.isfinite(prices)) or np.any(prices < 0.0):
            raise ValueError("initial_prices must be a non-negative Q-vector")
        total = float(np.sum(prices))
        if total <= 0.0:
            raise ValueError("initial_prices must have positive mass")
        prices = prices / total

    scale = max(float(np.max(gain)), 1.0e-12)
    normalized_gain = gain / scale
    subgradient_bound = max(float(np.sum(budget)), 1.0e-12)
    eta = np.sqrt(2.0 * np.log(max(Q, 2)) / count) / subgradient_bound
    average_power = np.zeros((K, Q), dtype=np.float64)
    averaging_mass = 0.0
    if bool(harmonic_safe_start):
        target_ceiling = np.sum(gain * budget[:, None], axis=0)
        if np.any(target_ceiling <= 0.0):
            # A structurally unreachable target makes the max-min optimum
            # exactly zero.  Return a feasible allocation and the matching
            # one-hot dual certificate rather than iterating indefinitely.
            unreachable = int(np.flatnonzero(target_ceiling <= 0.0)[0])
            for transmitter in range(K):
                average_power[
                    transmitter, int(np.argmax(gain[transmitter]))
                ] = budget[transmitter]
            deflection = np.sum(gain * average_power, axis=0)
            prices = np.zeros(Q, dtype=np.float64)
            prices[unreachable] = 1.0
            return MaxMinPowerResult(
                power_w=average_power,
                deflection=deflection,
                worst_deflection=0.0,
                prices=prices,
                dual_upper_bound=0.0,
                primal_dual_gap=0.0,
                rounds=0,
                worst_history=(0.0,),
                communication_bits=0,
            )
        public_ceiling = _quantize_deflection_feedback(
            target_ceiling, int(feedback_bits))
        inverse_ceiling = 1.0 / np.maximum(public_ceiling, 1.0e-300)
        coverage_share = inverse_ceiling / float(np.sum(inverse_ceiling))
        average_power = budget[:, None] * coverage_share[None, :]
        averaging_mass = 1.0
    best_power = average_power.copy()
    best_worst = float(np.min(np.sum(gain * best_power, axis=0)))
    best_dual_upper = np.inf
    best_dual_prices = prices.copy()
    history = []
    for _iteration in range(1, count + 1):
        public_prices = _quantize_simplex(prices, int(price_bits))
        dual_upper = float(scale * np.sum(
            budget * np.max(
                public_prices[None, :] * normalized_gain, axis=1)))
        if dual_upper < best_dual_upper:
            best_dual_upper = dual_upper
            best_dual_prices = public_prices.copy()
        allocation = np.zeros((K, Q), dtype=np.float64)
        for transmitter in range(K):
            scores = public_prices * normalized_gain[transmitter]
            target = int(np.argmax(scores))
            allocation[transmitter, target] = budget[transmitter]
        averaging_mass += 1.0
        average_power += (
            allocation - average_power
        ) / averaging_mass
        current_deflection = np.sum(gain * average_power, axis=0)
        current_worst = float(np.min(current_deflection))
        history.append(current_worst)
        if current_worst > best_worst:
            best_worst = current_worst
            best_power = average_power.copy()

        normalized_deflection = _quantize_deflection_feedback(
            np.sum(normalized_gain * allocation, axis=0),
            int(feedback_bits),
        )
        log_prices = np.log(np.maximum(prices, 1.0e-300))
        log_prices -= eta * normalized_deflection
        log_prices -= float(np.max(log_prices))
        prices = np.exp(log_prices)
        prices /= float(np.sum(prices))

    best_power = _fill_budget(best_power, gain, budget)
    deflection = np.sum(gain * best_power, axis=0)
    worst = float(np.min(deflection))
    public_prices = _quantize_simplex(prices, int(price_bits))
    final_dual_upper = float(scale * np.sum(
        budget * np.max(
            public_prices[None, :] * normalized_gain, axis=1)))
    if final_dual_upper < best_dual_upper:
        best_dual_upper = final_dual_upper
        best_dual_prices = public_prices.copy()
    # One target-price broadcast and one owner-feedback scalar per target and
    # round.  The initial harmonic ceiling reuses one additional feedback
    # vector.  Zero denotes the historical exact/unpriced diagnostic.
    communication_bits = 0
    if int(price_bits) > 0 or int(feedback_bits) > 0:
        communication_bits = int(
            count * Q * (int(price_bits) + int(feedback_bits))
            + (Q * int(feedback_bits) if harmonic_safe_start else 0)
        )
    return MaxMinPowerResult(
        power_w=best_power,
        deflection=deflection,
        worst_deflection=worst,
        prices=best_dual_prices,
        dual_upper_bound=max(best_dual_upper, worst),
        primal_dual_gap=max(best_dual_upper - worst, 0.0),
        rounds=count,
        worst_history=tuple(history),
        communication_bits=communication_bits,
    )


def _restricted_column_master(
    deflection_columns: np.ndarray,
    *,
    reuse_primal_duals: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the paired restricted primal/master over collected RF columns."""
    columns = np.asarray(deflection_columns, dtype=np.float64)
    if columns.ndim != 2 or columns.shape[0] < 1 or columns.shape[1] < 1:
        raise ValueError("deflection_columns must have shape (R,Q)")
    R, Q = columns.shape

    # Restricted primal: time-share complete per-UAV allocations (columns).
    primal_objective = np.zeros(R + 1, dtype=np.float64)
    primal_objective[-1] = -1.0
    primal_ub = np.zeros((Q, R + 1), dtype=np.float64)
    primal_ub[:, :R] = -columns.T
    primal_ub[:, -1] = 1.0
    primal = linprog(
        primal_objective,
        A_ub=primal_ub,
        b_ub=np.zeros(Q, dtype=np.float64),
        A_eq=np.concatenate((np.ones(R), np.zeros(1)))[None, :],
        b_eq=np.ones(1, dtype=np.float64),
        bounds=[(0.0, 1.0)] * R + [(0.0, None)],
        method="highs",
    )
    if not primal.success or primal.x is None:
        raise RuntimeError(f"restricted power master failed: {primal.message}")

    if bool(reuse_primal_duals):
        # HiGHS returns the optimal multipliers of the target lower-bound
        # inequalities.  By LP strong duality these are a valid optimal price
        # vector for the paired dual master; solving a second LP is redundant.
        # The sign is reversed because SciPy reports <=-row marginals in the
        # minimization convention.  Normalization closes small solver KKT
        # tolerances without changing the simplex price semantics.
        marginals = np.asarray(primal.ineqlin.marginals, dtype=np.float64)
        prices = np.maximum(-marginals, 0.0)
        if float(np.sum(prices)) <= 0.0:
            prices = np.full(Q, 1.0 / Q, dtype=np.float64)
        else:
            prices /= float(np.sum(prices))
        return (
            np.maximum(primal.x[:R], 0.0),
            prices,
            float(-primal.fun),
        )

    # Dual master: find target prices minimizing the largest collected column.
    dual_objective = np.zeros(Q + 1, dtype=np.float64)
    dual_objective[-1] = 1.0
    dual_ub = np.zeros((R, Q + 1), dtype=np.float64)
    dual_ub[:, :Q] = columns
    dual_ub[:, -1] = -1.0
    dual = linprog(
        dual_objective,
        A_ub=dual_ub,
        b_ub=np.zeros(R, dtype=np.float64),
        A_eq=np.concatenate((np.ones(Q), np.zeros(1)))[None, :],
        b_eq=np.ones(1, dtype=np.float64),
        bounds=[(0.0, 1.0)] * Q + [(0.0, None)],
        method="highs",
    )
    if not dual.success or dual.x is None:
        raise RuntimeError(f"restricted price master failed: {dual.message}")
    return (
        np.maximum(primal.x[:R], 0.0),
        np.maximum(dual.x[:Q], 0.0),
        float(-primal.fun),
    )


def _restricted_reserve_master(
    deflection_columns: np.ndarray,
    minimum_deflection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Max-min master with simultaneous target-wise Deflection reserves.

    The reserve rows are linear because a reporting structure is frozen and
    every collected column is a complete per-UAV RF allocation.  The returned
    target prices combine the max-min and reserve-row dual multipliers; each
    transmitter can therefore price the next column using only local gains.
    ``None`` means that the current restricted column set cannot meet all
    reserves, not that the full RF polytope is infeasible.
    """
    columns = np.asarray(deflection_columns, dtype=np.float64)
    reserve = np.asarray(minimum_deflection, dtype=np.float64).reshape(-1)
    if (
        columns.ndim != 2 or columns.shape[0] < 1
        or reserve.shape != (columns.shape[1],)
        or np.any(~np.isfinite(reserve)) or np.any(reserve < 0.0)
    ):
        raise ValueError("reserve master dimensions are inconsistent")
    R, Q = columns.shape
    objective = np.zeros(R + 1, dtype=np.float64)
    objective[-1] = -1.0
    upper_matrix = np.zeros((2 * Q, R + 1), dtype=np.float64)
    upper_matrix[:Q, :R] = -columns.T
    upper_matrix[:Q, -1] = 1.0
    upper_matrix[Q:, :R] = -columns.T
    upper = np.concatenate((np.zeros(Q), -reserve))
    solved = linprog(
        objective,
        A_ub=upper_matrix,
        b_ub=upper,
        A_eq=np.concatenate((np.ones(R), np.zeros(1)))[None, :],
        b_eq=np.ones(1, dtype=np.float64),
        bounds=[(0.0, 1.0)] * R + [(0.0, None)],
        method="highs",
    )
    if not solved.success or solved.x is None:
        return None
    marginals = np.asarray(solved.ineqlin.marginals, dtype=np.float64)
    prices = np.maximum(-marginals[:Q] - marginals[Q:], 0.0)
    if float(np.sum(prices)) <= 0.0:
        prices = np.full(Q, 1.0 / Q, dtype=np.float64)
    else:
        prices /= float(np.sum(prices))
    return np.maximum(solved.x[:R], 0.0), prices, float(-solved.fun)


def distributed_column_generation_maxmin_power(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    rounds: int,
    price_bits: int = 0,
    feedback_bits: int = 0,
    relative_tolerance: float = 1.0e-6,
    minimum_deflection: np.ndarray | None = None,
    incumbent_power_w: np.ndarray | None = None,
    reuse_primal_master_duals: bool = False,
) -> MaxMinPowerResult:
    """Finite-round distributed Dantzig--Wolfe power repair.

    Each column is a complete feasible RF allocation.  The initialization
    focuses all transmitters on each target once, so the restricted primal is
    feasible for every reachable target from round zero.  A Q-dimensional
    owner master broadcasts target prices; each transmitter then creates its
    part of the maximizing column using only local gains.  Time-sharing the
    collected columns preserves every per-UAV sensing budget exactly, while
    the full dual response supplies a valid upper certificate.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    count = int(rounds)
    if count < 1:
        raise ValueError("rounds must be positive")
    if int(price_bits) < 0 or int(price_bits) > 24:
        raise ValueError("price_bits must lie in [0,24]")
    if int(feedback_bits) not in (0, 16, 32, 64):
        raise ValueError("feedback_bits must be one of {0,16,32,64}")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    reserve = None
    if minimum_deflection is not None:
        reserve = np.asarray(
            minimum_deflection, dtype=np.float64).reshape(-1)
        if (
            reserve.shape != (Q,) or np.any(~np.isfinite(reserve))
            or np.any(reserve < 0.0)
        ):
            raise ValueError(
                "minimum_deflection must be a finite non-negative Q-vector")
    incumbent = None
    if incumbent_power_w is not None:
        incumbent = np.asarray(incumbent_power_w, dtype=np.float64)
        if (
            incumbent.shape != (K, Q) or np.any(~np.isfinite(incumbent))
            or np.any(incumbent < -1.0e-9)
            or np.any(np.sum(incumbent, axis=1) > budget + 1.0e-9)
        ):
            minimum = float(np.min(incumbent)) if incumbent.size else 0.0
            maximum_excess = (
                float(np.max(np.sum(incumbent, axis=1) - budget))
                if incumbent.shape == (K, Q) else float("nan")
            )
            raise ValueError(
                "incumbent_power_w must be a feasible non-negative KxQ "
                f"array; minimum={minimum:.17g}, "
                f"maximum_row_budget_excess={maximum_excess:.17g}")
        incumbent = _fill_budget(incumbent, gain, budget)
    target_ceiling = np.sum(gain * budget[:, None], axis=0)
    if np.any(target_ceiling <= 0.0):
        # The structural layer failed to connect at least one target.  Return
        # a feasible RF allocation and a zero certificate; power cannot fix it.
        power = np.zeros((K, Q), dtype=np.float64)
        for transmitter in range(K):
            power[transmitter, int(np.argmax(gain[transmitter]))] = budget[transmitter]
        deflection = np.sum(gain * power, axis=0)
        return MaxMinPowerResult(
            power_w=power,
            deflection=deflection,
            worst_deflection=float(np.min(deflection)),
            prices=np.full(Q, 1.0 / Q, dtype=np.float64),
            dual_upper_bound=0.0,
            primal_dual_gap=0.0,
            rounds=0,
            worst_history=(0.0,),
            reserve_feasible=bool(
                reserve is None or np.all(deflection + 1.0e-12 >= reserve)),
            reserve_shortfall=float(
                0.0 if reserve is None else np.max(
                    np.maximum(reserve - deflection, 0.0))),
            minimum_deflection=tuple(
                () if reserve is None else reserve.tolist()),
        )

    power_columns: list[np.ndarray] = []
    deflection_columns: list[np.ndarray] = []
    # Deterministic coverage columns prevent the unsafe zero-target transient
    # of winner-take-all subgradient descent.
    for target in range(Q):
        allocation = np.zeros((K, Q), dtype=np.float64)
        allocation[:, target] = budget
        power_columns.append(allocation)
        deflection_columns.append(_quantize_deflection_feedback(
            np.sum(gain * allocation, axis=0), int(feedback_bits)))
    if incumbent is not None and not any(
        np.allclose(incumbent, old, rtol=0.0, atol=1.0e-15)
        for old in power_columns
    ):
        power_columns.append(incumbent)
        deflection_columns.append(_quantize_deflection_feedback(
            np.sum(gain * incumbent, axis=0), int(feedback_bits)))

    best_upper = np.inf
    best_prices = np.full(Q, 1.0 / Q, dtype=np.float64)
    best_primal_power: np.ndarray | None = None
    best_primal_deflection: np.ndarray | None = None
    best_primal_worst = -np.inf
    if incumbent is not None:
        incumbent_deflection = np.sum(gain * incumbent, axis=0)
        if reserve is None or np.all(
            incumbent_deflection + 1.0e-12 >= reserve
        ):
            best_primal_power = incumbent.copy()
            best_primal_deflection = incumbent_deflection.copy()
            best_primal_worst = float(np.min(incumbent_deflection))
    history: list[float] = []
    iterations = 0
    weights = np.full(Q, 1.0 / Q, dtype=np.float64)
    for _ in range(count):
        columns = np.stack(deflection_columns)
        reserve_master = (
            None if reserve is None
            else _restricted_reserve_master(columns, reserve)
        )
        if reserve_master is None:
            weights, master_prices, lower = _restricted_column_master(
                columns,
                reuse_primal_duals=bool(reuse_primal_master_duals),
            )
        else:
            weights, master_prices, lower = reserve_master
        checkpoint_power = np.sum(
            weights[:, None, None] * np.stack(power_columns), axis=0)
        checkpoint_power = _fill_budget(checkpoint_power, gain, budget)
        checkpoint_deflection = np.sum(gain * checkpoint_power, axis=0)
        checkpoint_feasible = bool(
            reserve is None
            or np.all(checkpoint_deflection + 1.0e-12 >= reserve)
        )
        checkpoint_worst = float(np.min(checkpoint_deflection))
        if checkpoint_feasible and checkpoint_worst > best_primal_worst:
            best_primal_power = checkpoint_power.copy()
            best_primal_deflection = checkpoint_deflection.copy()
            best_primal_worst = checkpoint_worst
        public_prices = _quantize_simplex(master_prices, int(price_bits))
        allocation = np.zeros((K, Q), dtype=np.float64)
        for transmitter in range(K):
            target = int(np.argmax(public_prices * gain[transmitter]))
            allocation[transmitter, target] = budget[transmitter]
        oracle_deflection = np.sum(gain * allocation, axis=0)
        upper = float(np.dot(public_prices, oracle_deflection))
        history.append(float(lower))
        iterations += 1
        if upper < best_upper:
            best_upper = upper
            best_prices = public_prices.copy()
        gap = max(upper - lower, 0.0)
        if gap <= tolerance * max(1.0, abs(lower)):
            break
        duplicate = any(np.array_equal(allocation, old) for old in power_columns)
        if duplicate:
            # Price quantization can map a new master point to an old response.
            # Continuing cannot enrich the restricted feasible set.
            break
        power_columns.append(allocation)
        deflection_columns.append(_quantize_deflection_feedback(
            oracle_deflection, int(feedback_bits)))

    # Re-solve after the last admitted column.  The returned primal is always
    # feasible and its lower value is monotone with the collected column set.
    columns = np.stack(deflection_columns)
    reserve_master = (
        None if reserve is None
        else _restricted_reserve_master(columns, reserve)
    )
    if reserve_master is None:
        weights, master_prices, lower = _restricted_column_master(
            columns,
            reuse_primal_duals=bool(reuse_primal_master_duals),
        )
    else:
        weights, master_prices, lower = reserve_master
    power = np.sum(
        weights[:, None, None] * np.stack(power_columns), axis=0)
    power = _fill_budget(power, gain, budget)
    deflection = np.sum(gain * power, axis=0)
    reserve_feasible = bool(
        reserve is None or np.all(deflection + 1.0e-12 >= reserve))
    if best_primal_power is not None and (
        not reserve_feasible
        or best_primal_worst > float(np.min(deflection)) + 1.0e-12
    ):
        power = best_primal_power.copy()
        deflection = best_primal_deflection.copy()
        reserve_feasible = True
    worst = float(np.min(deflection))
    public_prices = _quantize_simplex(master_prices, int(price_bits))
    final_upper = float(np.sum(
        budget * np.max(public_prices[None, :] * gain, axis=1)))
    if final_upper < best_upper:
        best_upper = final_upper
        best_prices = public_prices.copy()
    certified_upper = max(float(best_upper), worst)
    return MaxMinPowerResult(
        power_w=power,
        deflection=deflection,
        worst_deflection=worst,
        prices=best_prices,
        dual_upper_bound=certified_upper,
        primal_dual_gap=max(certified_upper - worst, 0.0),
        rounds=iterations,
        worst_history=tuple(history + [worst]),
        reserve_feasible=reserve_feasible,
        reserve_shortfall=float(
            0.0 if reserve is None else np.max(
                np.maximum(reserve - deflection, 0.0))),
        minimum_deflection=tuple(
            () if reserve is None else reserve.tolist()),
    )


def optimal_maxmin_dual_prices(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Optimal simplex dual prices of the fixed-owner max-min power LP.

    The max-min power LP (``max_p min_q sum_i a_iq p_iq`` over per-UAV RF
    budgets) has the equivalent dual

        min_{lambda in simplex(Q)}  sum_i b_i max_q lambda_q a_iq,

    where ``a_iq = gain_per_watt[i, q]`` and ``b_i = sensing_budget_w[i]``.
    The optimal dual ``lambda*`` concentrates mass on the bottleneck targets
    (those attaining the max-min value by complementary slackness) and is the
    exact shadow price of the per-target lower-bound constraints.

    The dual is solved as the LP

        min_{u, lambda}  sum_i b_i u_i
        s.t.  a_iq lambda_q - u_i <= 0  (all i, q)
              sum_q lambda_q = 1,  lambda >= 0,  u >= 0,

    which is finite and bounded whenever ``gain`` is finite and non-negative.
    Returns ``(lambda*, value)`` with ``value`` equal to the max-min deflection
    by LP strong duality.  A zero-gain degenerate case returns the uniform
    price and value ``0.0``.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    if not np.any(gain > 0.0) or not np.any(budget > 0.0):
        return np.full(Q, 1.0 / Q, dtype=np.float64), 0.0

    # Variables: u (K) then lambda (Q).
    variables = K + Q
    objective = np.zeros(variables, dtype=np.float64)
    objective[:K] = budget

    rows = []
    upper = []
    for i in range(K):
        for q in range(Q):
            row = np.zeros(variables, dtype=np.float64)
            row[i] = -1.0
            row[K + q] = float(gain[i, q])
            rows.append(row)
            upper.append(0.0)
    equality = np.zeros((1, variables), dtype=np.float64)
    equality[0, K:] = 1.0
    bounds = [(0.0, None)] * variables
    solved = linprog(
        objective,
        A_ub=np.stack(rows),
        b_ub=np.asarray(upper, dtype=np.float64),
        A_eq=equality,
        b_eq=np.ones(1, dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if not solved.success or solved.x is None:
        # Dual is a bounded convex LP; a solver failure is a numerical
        # degenerate case, not a modelling gap.  Fall back to a uniform
        # price so the caller can still produce a conservative reward.
        return np.full(Q, 1.0 / Q, dtype=np.float64), 0.0
    prices = np.maximum(solved.x[K:], 0.0)
    total = float(np.sum(prices))
    if total <= 0.0:
        prices = np.full(Q, 1.0 / Q, dtype=np.float64)
    else:
        prices /= total
    value = float(np.dot(objective, solved.x))
    return prices, value


def canonical_maxmin_dual_prices(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Unique minimum-L2 price on the optimal dual face.

    The LP-optimal target price need not be unique, which makes learning
    labels solver- and ordering-dependent.  This secondary convex program
    minimizes ``||lambda||_2^2`` over the complete optimal dual set while
    preserving the max-min value.  Strict convexity in ``lambda`` makes the
    returned price unique; no perturbation of the primary task objective is
    introduced.
    """
    from scipy.optimize import Bounds, LinearConstraint, minimize

    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    if not np.any(gain > 0.0) or not np.any(budget > 0.0):
        return np.full(Q, 1.0 / Q, dtype=np.float64), 0.0
    initial_lambda, value = optimal_maxmin_dual_prices(gain, budget)
    initial_u = np.max(gain * initial_lambda[None, :], axis=1)
    x0 = np.concatenate((initial_u, initial_lambda))

    rows = []
    lower = []
    upper = []
    for i in range(K):
        for q in range(Q):
            row = np.zeros(K + Q, dtype=np.float64)
            row[i] = 1.0
            row[K + q] = -gain[i, q]
            rows.append(row)
            lower.append(0.0)
            upper.append(np.inf)
    face = np.zeros(K + Q, dtype=np.float64)
    face[:K] = budget
    face_tol = 1.0e-8 * max(1.0, abs(value))
    rows.append(face)
    lower.append(value - face_tol)
    upper.append(value + face_tol)
    inequality_constraint = LinearConstraint(
        np.stack(rows), np.asarray(lower), np.asarray(upper))
    simplex = np.zeros((1, K + Q), dtype=np.float64)
    simplex[0, K:] = 1.0
    simplex_constraint = LinearConstraint(simplex, [1.0], [1.0])

    def objective(x: np.ndarray) -> float:
        return 0.5 * float(np.dot(x[K:], x[K:]))

    def gradient(x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        grad[K:] = x[K:]
        return grad

    solved = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(np.zeros(K + Q), np.full(K + Q, np.inf)),
        constraints=[inequality_constraint, simplex_constraint],
        options={"ftol": 1.0e-12, "maxiter": 2000, "disp": False},
    )
    if not solved.success or solved.x is None:
        raise RuntimeError(
            f"canonical max-min dual solve failed: {solved.message}")
    prices = np.maximum(np.asarray(solved.x[K:], dtype=np.float64), 0.0)
    prices /= float(np.sum(prices))
    dual_value = float(np.sum(
        budget * np.max(prices[None, :] * gain, axis=1)))
    if abs(dual_value - value) > 5.0 * face_tol:
        raise RuntimeError("canonical dual left the optimal dual face")
    return prices, value


def fixed_owner_gain_matrix(
    coefficient: np.ndarray,
    selected: Sequence[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a fixed unique-owner reporting graph to transmitter gains.

    The downstream LP is valid under *conditionally separable evidence*:
    structure, DD support and CSI are frozen during the inner solve, and each
    supplied per-watt coefficient is independent of the optimized power.  If
    sensing interference, power-dependent CSI/DD support, or nonlinear RF
    coupling is present, callers must rebuild the physical model instead of
    treating this matrix as an LP coefficient.
    """
    values = np.asarray(coefficient, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError("coefficient must have shape (K,K,Q)")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("coefficient must be finite and non-negative")
    K, _, Q = values.shape
    owners = np.full(Q, -1, dtype=np.int64)
    gain = np.zeros((K, Q), dtype=np.float64)
    for transmitter, receiver, target in selected:
        i, j, q = int(transmitter), int(receiver), int(target)
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q) or i == j:
            raise ValueError("selected edge is outside coefficient support")
        if owners[q] not in (-1, j):
            raise ValueError("fixed-owner LP requires one receiver per target")
        owners[q] = j
        gain[i, q] += values[i, j, q]
    if np.any(owners < 0):
        raise ValueError("fixed-owner LP requires every target to have an owner")
    return gain, owners


def relaxed_same_geometry_target_ceiling(
    coefficient: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> np.ndarray:
    """A genuine per-target upper ceiling after relaxing all RF coupling.

    Each target is allowed to use every UAV's full sensing budget
    independently, and each transmitter may choose its best receiver for that
    target.  This simultaneously violates cross-target power sharing, common
    owner, role and capacity constraints, so every feasible same-geometry
    allocation is bounded above componentwise by the returned vector.
    """
    values = np.asarray(coefficient, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        values.ndim != 3
        or values.shape[0] != values.shape[1]
        or values.shape[0] != budget.size
        or values.shape[2] < 1
    ):
        raise ValueError("coefficient must have shape (K,K,Q) matching budget")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("coefficient must be finite and non-negative")
    if np.any(~np.isfinite(budget)) or np.any(budget < 0.0):
        raise ValueError("sensing_budget_w must be finite and non-negative")
    best_receiver_gain = np.max(values, axis=1)
    return np.sum(budget[:, None] * best_receiver_gain, axis=0)
