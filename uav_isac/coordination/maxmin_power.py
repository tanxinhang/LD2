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
    """Turn LP inequalities into the exact per-UAV RF equality harmlessly."""
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
    """Solve the fixed-owner max-min power LP and preserve exact RF budgets.

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
    initial_prices: np.ndarray | None = None,
) -> MaxMinPowerResult:
    """Finite-round distributed dual mirror descent with primal averaging.

    A target-price vector is public.  In each round every transmitter spends
    its sensing budget on the target maximizing ``lambda_q * a_iq``.  Owners
    aggregate achieved target Deflection, and exponentiated-gradient descent
    updates the simplex price.  The best feasible ergodic primal average is
    returned; no monotonicity is assumed for the last iterate.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    count = int(rounds)
    if count < 1:
        raise ValueError("rounds must be positive")
    if int(price_bits) < 0 or int(price_bits) > 24:
        raise ValueError("price_bits must lie in [0,24]")
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
    best_power = average_power.copy()
    best_worst = -np.inf
    history = []
    for iteration in range(1, count + 1):
        public_prices = _quantize_simplex(prices, int(price_bits))
        allocation = np.zeros((K, Q), dtype=np.float64)
        for transmitter in range(K):
            scores = public_prices * normalized_gain[transmitter]
            target = int(np.argmax(scores))
            allocation[transmitter, target] = budget[transmitter]
        average_power += (allocation - average_power) / float(iteration)
        current_deflection = np.sum(gain * average_power, axis=0)
        current_worst = float(np.min(current_deflection))
        history.append(current_worst)
        if current_worst > best_worst:
            best_worst = current_worst
            best_power = average_power.copy()

        normalized_deflection = np.sum(
            normalized_gain * allocation, axis=0)
        log_prices = np.log(np.maximum(prices, 1.0e-300))
        log_prices -= eta * normalized_deflection
        log_prices -= float(np.max(log_prices))
        prices = np.exp(log_prices)
        prices /= float(np.sum(prices))

    best_power = _fill_budget(best_power, gain, budget)
    deflection = np.sum(gain * best_power, axis=0)
    worst = float(np.min(deflection))
    public_prices = _quantize_simplex(prices, int(price_bits))
    dual_upper = float(scale * np.sum(
        budget * np.max(
            public_prices[None, :] * normalized_gain, axis=1)))
    return MaxMinPowerResult(
        power_w=best_power,
        deflection=deflection,
        worst_deflection=worst,
        prices=public_prices,
        dual_upper_bound=max(dual_upper, worst),
        primal_dual_gap=max(dual_upper - worst, 0.0),
        rounds=count,
        worst_history=tuple(history),
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


def fixed_owner_gain_matrix(
    coefficient: np.ndarray,
    selected: Sequence[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a fixed unique-owner reporting graph to transmitter gains."""
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
