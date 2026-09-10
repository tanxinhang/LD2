"""Fixed-structure max-min sensing-power repair.

Once reporting roles and one receiver owner per target are frozen, target
Deflection is linear in transmitter sensing power.  The resulting max-min
allocation is an LP.  Its simplex dual admits a distributed exponentiated-
gradient update using public target prices and transmitter-local gains.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.optimize import linprog, minimize


# HiGHS certifies feasibility in scaled coordinates.  Reconstructing the
# physical objective from its basis marginals can therefore undershoot by a
# small solver tolerance.  A looser upper bound remains mathematically sound;
# its actual gap must be propagated instead of rejected or reported as zero.
_PRIMAL_DUAL_RELATIVE_TOLERANCE = 1.0e-7
_LP_FEASIBILITY_RELATIVE_TOLERANCE = 1.0e-8
_LP_FEASIBILITY_ABSOLUTE_TOLERANCE_W = 1.0e-7


def _primal_dual_tolerance(primal: float, dual: float) -> float:
    scale = max(abs(float(primal)), abs(float(dual)), np.finfo(np.float64).tiny)
    return _PRIMAL_DUAL_RELATIVE_TOLERANCE * scale


def _dual_upper_is_sound(primal: float, dual: float) -> bool:
    return bool(
        np.isfinite(dual)
        and float(dual) >= float(primal) - _primal_dual_tolerance(primal, dual)
    )


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
    common_model_certificate: bool
    local_full_power_w: np.ndarray
    local_prices: np.ndarray
    local_cache_valid: np.ndarray
    local_resolved: np.ndarray
    local_relative_gap: np.ndarray
    local_compute_time_s: np.ndarray
    unique_local_problem_count: int
    parallel_execution_used: bool = False
    parallel_worker_count: int = 1
    parallel_batch_wall_time_s: float = 0.0
    parallel_fallback_to_serial: bool = False
    deadline_incumbent_used_fraction: float = 0.0
    deadline_incumbent_certificate_fraction: float = 0.0
    deadline_uniform_fallback_fraction: float = 0.0
    deadline_harmonic_fallback_fraction: float = 0.0
    deadline_composable_deflection_floor: float = 0.0
    history_reserve_used_fraction: float = 0.0
    common_model_fallback_fraction: float = 0.0


class IncompleteFixedOwnerStructureError(ValueError):
    """The reporting graph is valid so far but does not own every target."""


class NonUniqueFixedOwnerStructureError(ValueError):
    """A target has valid reporting edges assigned to multiple receivers."""


def sparse_harmonic_row_power(
    conservative_gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    zero_support_fallback_power_w: np.ndarray | None = None,
) -> np.ndarray:
    """Equalize each row over its positive conservative-gain support.

    For ``S_k={q:a[k,q]>0}``, the returned positive-support row is the exact
    solution of ``max min_{q in S_k} a[k,q]p[k,q]`` on node ``k``'s RF
    simplex. Rows with empty support cannot certify any target; they retain a
    supplied feasible fallback share, or use a uniform share if none is given.
    The construction is separable across nodes and costs ``O(KQ)``.
    """
    gain = np.asarray(conservative_gain_per_watt, dtype=np.float64)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if gain.ndim != 2 or gain.shape[0] != budget.size or gain.shape[1] < 1:
        raise ValueError("conservative gain must be (K,Q) matching budget")
    if (
        np.any(~np.isfinite(gain)) or np.any(gain < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError("gain and budget must be finite and non-negative")
    fallback = None
    if zero_support_fallback_power_w is not None:
        fallback = np.asarray(
            zero_support_fallback_power_w, dtype=np.float64)
        if (
            fallback.shape != gain.shape
            or np.any(~np.isfinite(fallback))
            or np.any(fallback < 0.0)
        ):
            raise ValueError(
                "zero-support fallback must be finite non-negative (K,Q)")
    K, Q = gain.shape
    power = np.zeros_like(gain)
    for node in range(K):
        support = gain[node] > 0.0
        if np.any(support):
            inverse = 1.0 / gain[node, support]
            power[node, support] = (
                float(budget[node]) * inverse / float(np.sum(inverse)))
            continue
        if fallback is not None and float(np.sum(fallback[node])) > 0.0:
            share = fallback[node] / float(np.sum(fallback[node]))
        else:
            share = np.full(Q, 1.0 / Q, dtype=np.float64)
        power[node] = float(budget[node]) * share
    return power


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
    raw = np.asarray(power, dtype=np.float64)
    if np.any(~np.isfinite(raw)):
        raise RuntimeError(
            "fixed-structure max-min LP returned non-finite power")
    for transmitter in range(raw.shape[0]):
        negative_tolerance = max(
            _LP_FEASIBILITY_ABSOLUTE_TOLERANCE_W,
            _LP_FEASIBILITY_RELATIVE_TOLERANCE * float(budget[transmitter]),
            64.0 * float(np.spacing(budget[transmitter])),
        )
        if float(np.min(raw[transmitter])) < -negative_tolerance:
            raise RuntimeError(
                "fixed-structure max-min LP returned materially negative "
                f"power for transmitter={transmitter}, "
                f"minimum={float(np.min(raw[transmitter]))}")
    result = np.maximum(raw, 0.0).copy()
    for transmitter in range(result.shape[0]):
        slack = float(budget[transmitter] - np.sum(result[transmitter]))
        if slack < 0.0:
            tolerance = max(
                _LP_FEASIBILITY_ABSOLUTE_TOLERANCE_W,
                _LP_FEASIBILITY_RELATIVE_TOLERANCE * float(budget[transmitter]),
                64.0 * float(np.spacing(budget[transmitter])),
            )
            if slack < -tolerance:
                raise RuntimeError(
                    "fixed-structure max-min LP returned a materially "
                    "budget-infeasible allocation; "
                    f"transmitter={transmitter}, "
                    f"allocated={float(np.sum(result[transmitter]))}, "
                    f"budget={float(budget[transmitter])}")
            # Project a solver-scale residual radially instead of subtracting
            # it from one component, which could create a negative entry.
            row_sum = float(np.sum(result[transmitter]))
            if row_sum > 0.0:
                result[transmitter] *= float(budget[transmitter]) / row_sum
            slack = float(
                budget[transmitter] - np.sum(result[transmitter]))
        if slack > 0.0:
            target = int(np.argmax(gain[transmitter]))
            result[transmitter, target] += slack
    return result


@lru_cache(maxsize=24)
def _fixed_structure_lp_layout(
    num_transmitters: int,
    num_targets: int,
    has_reserve: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cache the dimension-only dense LP layout used by repeated local solves.

    Strict distributed execution solves one private-view LP per UAV, often at
    the same K/Q dimensions for an entire mission.  Rebuilding its objective,
    budget rows, and scalar indices in Python is pure setup overhead.  Cached
    arrays are read-only; callers copy the matrix/right-hand side before
    inserting the current gains and budgets.
    """
    K = int(num_transmitters)
    Q = int(num_targets)
    variables = K * Q
    target_row_count = Q * (2 if bool(has_reserve) else 1)
    objective = np.zeros(variables + 1, dtype=np.float64)
    objective[-1] = -1.0
    matrix = np.zeros(
        (target_row_count + K, variables + 1), dtype=np.float64)
    matrix[:Q, -1] = 1.0
    for transmitter in range(K):
        begin = transmitter * Q
        matrix[target_row_count + transmitter, begin:begin + Q] = 1.0
    base_target_rows = np.repeat(np.arange(Q), K)
    target_columns = (
        np.arange(K)[:, None] * Q + np.arange(Q)[None, :]
    ).T.reshape(-1)
    target_rows = (
        np.concatenate((base_target_rows, base_target_rows + Q))
        if bool(has_reserve) else base_target_rows
    )
    gain_columns = (
        np.concatenate((target_columns, target_columns))
        if bool(has_reserve) else target_columns
    )
    for array in (objective, matrix, target_rows, gain_columns):
        array.setflags(write=False)
    return objective, matrix, target_rows, gain_columns


def _effective_gain_and_ceiling(
    gain: np.ndarray,
    budget: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return budget-normalized gain, rejecting unrepresentable products."""
    with np.errstate(over="ignore", invalid="ignore"):
        effective_gain = gain * budget[:, None]
        ceiling = np.sum(effective_gain, axis=0)
    if np.any(~np.isfinite(effective_gain)) or np.any(~np.isfinite(ceiling)):
        raise ValueError(
            "gain_per_watt * sensing_budget_w exceeds float64 range")
    return effective_gain, ceiling


def _requires_relative_primal_scaling(
    gain: np.ndarray,
    effective_gain: np.ndarray,
) -> bool:
    """Whether one global gain scale would erase a physically relevant link."""
    for values in (gain, effective_gain):
        positive = values[values > 0.0]
        if (
            positive.size >= 2
            and float(np.max(positive)) / float(np.min(positive)) >= 1.0e6
        ):
            return True
    return False


def _solve_relative_scaled_fixed_structure_lp(
    gain: np.ndarray,
    budget: np.ndarray,
    reserve: np.ndarray | None,
    effective_gain: np.ndarray,
    ceiling: np.ndarray,
):
    """Solve the same LP in row-simplex shares and target-relative units.

    ``y_iq = p_iq / b_i`` makes every transmitter budget row unit scale.
    Scaling ``t`` by the smallest reachable target ceiling ensures that a
    weak-but-reachable target is not rounded away by HiGHS' absolute
    feasibility tolerance.  This path is reserved for large within-problem
    dynamic ranges so the historical ordinary-scale basis is unchanged.
    """
    K, Q = gain.shape
    variables = K * Q
    positive_ceiling = ceiling[ceiling > 0.0]
    target_scale = (
        float(np.min(positive_ceiling))
        if positive_ceiling.size else 1.0
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized_target_gain = effective_gain / target_scale
    if np.any(~np.isfinite(normalized_target_gain)):
        raise RuntimeError(
            "fixed-structure max-min LP dynamic range is not representable "
            "after target-relative scaling")

    objective, template, target_rows, gain_columns = (
        _fixed_structure_lp_layout(K, Q, reserve is not None)
    )
    matrix = template.copy()
    target_entries = K * Q
    matrix[
        target_rows[:target_entries], gain_columns[:target_entries]
    ] = -normalized_target_gain.T.reshape(-1)
    upper = np.zeros(matrix.shape[0], dtype=np.float64)
    target_row_count = Q * (2 if reserve is not None else 1)
    if reserve is not None:
        if np.any(reserve > ceiling):
            raise RuntimeError(
                "fixed-structure max-min LP with reserve is infeasible; "
                f"reachable per-target ceiling={ceiling.tolist()}, "
                f"reserve={reserve.tolist()}")
        # Scale each feasibility-only reserve row by its own reachable
        # ceiling.  Unlike the first Q rows these rows contain no objective
        # variable, so independent positive row scaling is exact.
        reserve_scale = np.where(ceiling > 0.0, ceiling, 1.0)
        normalized_reserve_gain = effective_gain / reserve_scale[None, :]
        matrix[
            target_rows[target_entries:], gain_columns[target_entries:]
        ] = -normalized_reserve_gain.T.reshape(-1)
        upper[Q:2 * Q] = -reserve / reserve_scale
    upper[target_row_count:] = 1.0
    solved = linprog(
        objective,
        A_ub=matrix,
        b_ub=upper,
        bounds=(0.0, None),
        method="highs-ds",
        options={"presolve": False},
    )
    if solved.success and solved.x is not None:
        power = (
            solved.x[:variables].reshape(K, Q) * budget[:, None])
    else:
        power = None
    return solved, power


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
    effective_gain, reachable_ceiling = _effective_gain_and_ceiling(
        gain, budget)
    reserve = None
    if minimum_deflection is not None:
        reserve = np.asarray(minimum_deflection, dtype=np.float64).reshape(-1)
        if (
            reserve.shape != (Q,) or np.any(~np.isfinite(reserve))
            or np.any(reserve < 0.0)
        ):
            raise ValueError(
                "minimum_deflection must be a finite non-negative Q-vector")
    use_relative_scaling = _requires_relative_primal_scaling(
        gain, effective_gain)
    # HiGHS feasibility/optimality tolerances are absolute.  Physical
    # per-watt coefficients can be far below those tolerances (for example
    # 1e-12 in a weak/long-range link), in which case the unscaled LP may
    # legally return t=0 and an arbitrary saturated power row.  A single
    # positive scalar leaves the primal optimizer and simplex dual prices
    # invariant; the guarded path below places the largest coefficient at one.
    gain_scale = float(np.max(gain))
    # Preserve the historical HiGHS basis in the already-certified ordinary
    # operating range.  Although global scaling leaves the mathematical LP
    # unchanged, a different basis on a degenerate optimal face changes the
    # full local plan and therefore the independently stitched row execution.
    # Rescale only where absolute solver tolerances or extreme coefficients
    # threaten correctness; this is a numerical guard, not a new tie-break.
    lp_scale = (
        gain_scale
        if gain_scale > 0.0
        and (gain_scale < 1.0e-6 or gain_scale > 1.0e6)
        else 1.0
    )
    normalized_gain = (
        gain / lp_scale
        if lp_scale != 1.0
        else gain
    )
    variables = K * Q
    if use_relative_scaling:
        solved, raw_power = _solve_relative_scaled_fixed_structure_lp(
            gain, budget, reserve, effective_gain, reachable_ceiling)
    else:
        normalized_reserve = None
        if reserve is not None:
            if gain_scale <= 0.0 and np.any(reserve > 0.0):
                raise RuntimeError(
                    "fixed-structure max-min LP with reserve is infeasible; "
                    "all gains are zero")
            normalized_reserve = (
                reserve / lp_scale
                if lp_scale != 1.0 else reserve
            )
        objective, template, target_rows, gain_columns = (
            _fixed_structure_lp_layout(K, Q, reserve is not None)
        )
        matrix = template.copy()
        gain_values = -normalized_gain.T.reshape(-1)
        if reserve is not None:
            gain_values = np.concatenate((gain_values, gain_values))
        matrix[target_rows, gain_columns] = gain_values
        upper = np.zeros(matrix.shape[0], dtype=np.float64)
        target_row_count = Q * (2 if reserve is not None else 1)
        if normalized_reserve is not None:
            upper[Q:2 * Q] = -normalized_reserve
        upper[target_row_count:] = budget
        solved = linprog(
            objective,
            A_ub=matrix,
            b_ub=upper,
            # Per-variable p_iq <= b_i is implied by p >= 0 and the row
            # budget; omitting redundant bounds reduces wrapper work.
            bounds=(0.0, None),
            method="highs-ds",
            options={"presolve": False},
        )
        raw_power = (
            None if solved.x is None
            else solved.x[:variables].reshape(K, Q)
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
    if raw_power is None:
        raise RuntimeError(
            "fixed-structure max-min LP reported success without a solution")
    power = _fill_budget(raw_power, gain, budget)
    deflection = np.sum(gain * power, axis=0)
    worst = float(np.min(deflection))
    # Derive tolerances from reachable physical target ceilings.  The looser
    # ``max(gain) * sum(budget)`` cross-product can overflow even when every
    # realizable gain-budget product is finite, disabling validation exactly
    # on the sparse extreme-scale cases this guard protects.
    reserve_scale = np.maximum(
        reachable_ceiling,
        np.zeros(Q, dtype=np.float64) if reserve is None else reserve,
    )
    reserve_tolerance = 1.0e-8 * np.maximum(
        reserve_scale, np.finfo(np.float64).tiny)
    reserve_feasible = bool(
        reserve is None
        or np.all(deflection + reserve_tolerance >= reserve))
    # The first Q inequality marginals are the max-min target multipliers.
    # Their sign is reversed because SciPy reports <=-row marginals for a
    # minimization problem.  Keeping the genuine simplex prices enables an
    # O(KQ) cross-frame upper certificate instead of returning a diagnostic
    # uniform vector that cannot safely drive event-triggered reuse.
    marginals = np.asarray(solved.ineqlin.marginals[:Q], dtype=np.float64)
    prices = np.maximum(-marginals, 0.0)
    price_mass = float(np.sum(prices))
    zero_ceiling = reachable_ceiling == 0.0
    if np.any(zero_ceiling):
        # A zero-ceiling target proves the primal optimum is exactly zero.
        # Concentrating price on those targets gives the matching exact dual
        # certificate even on a fully degenerate HiGHS basis.
        prices = zero_ceiling.astype(np.float64)
        prices /= float(np.sum(prices))
    else:
        prices = (
            prices / price_mass
            if price_mass > 0.0
            else np.full(Q, 1.0 / Q, dtype=np.float64)
        )
    raw_dual_upper = float(np.sum(
        budget * np.max(prices[None, :] * gain, axis=1)))
    # Any simplex target price is a valid upper bound for the unconstrained
    # max-min problem and therefore also for its reserve-constrained subset.
    if not _dual_upper_is_sound(worst, raw_dual_upper):
        raise RuntimeError(
            "fixed-structure max-min LP failed its primal-dual "
            "certificate; refusing a possible numerical false optimum: "
            f"primal={worst}, dual={raw_dual_upper}, "
            f"relative_scaled={use_relative_scaling}")
    # Clamp only a certified sub-tolerance roundoff under-shoot.  Performing
    # this before the check would hide a broken upper certificate.
    dual_upper = max(raw_dual_upper, worst)
    return MaxMinPowerResult(
        power_w=power,
        deflection=deflection,
        worst_deflection=worst,
        prices=prices,
        dual_upper_bound=dual_upper,
        primal_dual_gap=max(dual_upper - worst, 0.0),
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
    *,
    previous_local_power_w: np.ndarray | None = None,
    previous_local_prices: np.ndarray | None = None,
    previous_local_cache_valid: np.ndarray | None = None,
    reuse_relative_tolerance: float = 0.0,
    parallel_executor=None,
    parallel_fallback_to_serial: bool = True,
    parallel_failure_mode: str = "serial",
    deadline_incumbent_relative_tolerance: float = 0.05,
    force_deadline_fallback: bool = False,
    deadline_safe_row_gain_per_watt: np.ndarray | None = None,
    history_reserve_deflection_cap: float | None = None,
    require_common_model_certificate: bool = False,
) -> ReplicatedLocalPowerResult:
    """Execute only each transmitter's row from its private public-cache LP.

    ``gain_views_per_watt[k]`` is UAV ``k``'s locally reconstructed ``(K,Q)``
    gain matrix. Every UAV solves the same fixed-structure LP using its own
    cache, but transmitter ``k`` executes only row ``k``. No primal/dual
    certificate or optimizer state is exchanged.

    When all public views and budgets agree byte-for-byte, deterministic local solves are
    identical and the assembled allocation equals the ordinary centralized LP
    optimum.  If ``require_common_model_certificate`` is true, unequal views
    are never stitched: every transmitter instead executes a row-local safe
    allocation, derived from ``deadline_safe_row_gain_per_watt`` when
    available and uniform otherwise.  This keeps feasibility and the
    composable row certificate without pretending that unrelated local LP
    optima form a global solution.
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
    failure_mode = str(parallel_failure_mode).strip().lower()
    if failure_mode not in {
        "serial", "cached_or_uniform", "cached_or_harmonic"
    }:
        raise ValueError(
            "parallel_failure_mode must be serial, cached_or_uniform, or "
            "cached_or_harmonic")
    incumbent_tolerance = float(deadline_incumbent_relative_tolerance)
    if (
        not np.isfinite(incumbent_tolerance)
        or not 0.0 <= incumbent_tolerance < 1.0
    ):
        raise ValueError(
            "deadline incumbent tolerance must be finite and lie in [0,1)")
    safe_row_gain = None
    if deadline_safe_row_gain_per_watt is not None:
        safe_row_gain = np.asarray(
            deadline_safe_row_gain_per_watt, dtype=np.float64)
        if (
            safe_row_gain.shape != (views.shape[0], views.shape[2])
            or np.any(~np.isfinite(safe_row_gain))
            or np.any(safe_row_gain < 0.0)
        ):
            raise ValueError(
                "deadline safe row gain must be finite non-negative (K,Q)")
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
    common_model_certificate = bool(np.array_equal(
        views, np.broadcast_to(views[0], views.shape)))
    history_cap = None
    if history_reserve_deflection_cap is not None:
        history_cap = float(history_reserve_deflection_cap)
        if not np.isfinite(history_cap) or history_cap <= 0.0:
            raise ValueError(
                "history reserve deflection cap must be finite and positive")
    reuse_tolerance = float(reuse_relative_tolerance)
    if (
        not np.isfinite(reuse_tolerance)
        or not 0.0 <= reuse_tolerance < 1.0
    ):
        raise ValueError(
            "reuse_relative_tolerance must be finite and lie in [0,1)")
    previous_power = None
    previous_prices = None
    previous_valid = np.zeros(K, dtype=bool)
    cache_supplied = any(value is not None for value in (
        previous_local_power_w,
        previous_local_prices,
        previous_local_cache_valid,
    ))
    if cache_supplied:
        if any(value is None for value in (
            previous_local_power_w,
            previous_local_prices,
            previous_local_cache_valid,
        )):
            raise ValueError(
                "all previous local power, price, and validity caches are "
                "required together")
        previous_power = np.asarray(
            previous_local_power_w, dtype=np.float64)
        previous_prices = np.asarray(
            previous_local_prices, dtype=np.float64)
        previous_valid = np.asarray(
            previous_local_cache_valid, dtype=bool).reshape(-1)
        if (
            previous_power.shape != (K, K, Q)
            or previous_prices.shape != (K, Q)
            or previous_valid.shape != (K,)
            or np.any(~np.isfinite(previous_power))
            or np.any(previous_power < 0.0)
            or np.any(~np.isfinite(previous_prices))
            or np.any(previous_prices < 0.0)
        ):
            raise ValueError(
                "previous local caches must have shapes (K,K,Q), (K,Q), "
                "and (K,) with finite non-negative values")

    power = np.zeros((K, Q), dtype=np.float64)
    local_worst = np.zeros(K, dtype=np.float64)
    local_coverage = np.zeros(K, dtype=bool)
    local_full_power = np.zeros((K, K, Q), dtype=np.float64)
    local_prices = np.full((K, Q), 1.0 / Q, dtype=np.float64)
    local_cache_valid = np.zeros(K, dtype=bool)
    local_resolved = np.zeros(K, dtype=bool)
    local_relative_gap = np.full(K, np.inf, dtype=np.float64)
    local_compute_time_s = np.zeros(K, dtype=np.float64)
    # First perform every private cache/coverage decision in deterministic
    # viewer order. Only complete, unresolved, immutable LP inputs may leave
    # this process. This preserves the information boundary and keeps unknown-
    # target fail-safe allocation local.
    pending_problem_viewers: dict[tuple[bytes, bytes], list[int]] = {}
    pending_problem_gains: dict[tuple[bytes, bytes], np.ndarray] = {}
    pending_problem_reserves: dict[
        tuple[bytes, bytes], np.ndarray | None
    ] = {}
    history_reserve_count = 0
    common_model_fallback_count = 0
    certified_fallback = None
    if require_common_model_certificate and not common_model_certificate:
        certified_fallback = (
            sparse_harmonic_row_power(safe_row_gain, executed_budget)
            if safe_row_gain is not None
            else np.repeat((executed_budget / Q)[:, None], Q, axis=1)
        )

    def commit_complete_solution(
        viewer: int,
        local_gain: np.ndarray,
        local_power: np.ndarray,
        local_price: np.ndarray,
        compute_time_s: float,
    ) -> None:
        local_cache_valid[viewer] = True
        local_full_power[viewer] = local_power
        local_prices[viewer] = local_price
        local_worst[viewer] = float(np.min(np.sum(
            local_gain * local_power, axis=0)))
        row = np.maximum(local_power[viewer], 0.0)
        row_mass = float(np.sum(row))
        if row_mass > 0.0:
            power[viewer] = row * (
                float(executed_budget[viewer]) / row_mass)
        elif executed_budget[viewer] > 0.0:
            target = int(np.argmax(local_gain[viewer]))
            power[viewer, target] = float(executed_budget[viewer])
        local_compute_time_s[viewer] = float(compute_time_s)

    for viewer in range(K):
        local_compute_started = perf_counter()
        local_gain = views[viewer]
        if certified_fallback is not None:
            power[viewer] = certified_fallback[viewer]
            local_compute_time_s[viewer] = float(
                perf_counter() - local_compute_started)
            common_model_fallback_count += 1
            continue
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
            local_compute_time_s[viewer] = float(
                perf_counter() - local_compute_started)
            continue
        local_power = None
        local_price = None
        held = None
        history_reserve = None
        if (
            previous_power is not None
            and bool(previous_valid[viewer])
            and (reuse_tolerance > 0.0 or history_cap is not None)
        ):
            held = previous_power[viewer].copy()
            held_row_sum = np.sum(held, axis=1, keepdims=True)
            held_share = np.divide(
                held,
                held_row_sum,
                out=np.full_like(held, 1.0 / Q),
                where=held_row_sum > 0.0,
            )
            held = public_budget[:, None] * held_share
            if history_cap is not None:
                # ``held`` is a feasible full plan under the current public
                # budget. Its current-view deflection is therefore a primal
                # witness for the clipped historical reserve. In a common
                # private view, the reserve LP cannot reduce any target below
                # min{D_q(held), cap}, and its max-min value cannot fall below
                # the historical worst value.
                history_reserve = np.minimum(
                    np.sum(local_gain * held, axis=0),
                    history_cap,
                )
                history_reserve_count += 1
        if (
            reuse_tolerance > 0.0
            and held is not None
        ):
            lower = float(np.min(np.sum(local_gain * held, axis=0)))
            cached_price = previous_prices[viewer]
            price_mass = float(np.sum(cached_price))
            if price_mass > 0.0:
                cached_price = cached_price / price_mass
                upper_bound = float(np.sum(
                    public_budget * np.max(
                        cached_price[None, :] * local_gain,
                        axis=1,
                    )
                ))
                upper_bound = max(upper_bound, lower)
                relative_gap = max(
                    upper_bound - lower, 0.0
                ) / max(upper_bound, 1.0e-300)
                if relative_gap <= reuse_tolerance:
                    local_power = held
                    local_price = cached_price.copy()
                    local_relative_gap[viewer] = relative_gap
        if local_power is not None:
            commit_complete_solution(
                viewer,
                local_gain,
                local_power,
                local_price,
                perf_counter() - local_compute_started,
            )
            continue
        reserve_key = (
            b"" if history_reserve is None
            else history_reserve.tobytes(order="C")
        )
        problem_key = (local_gain.tobytes(order="C"), reserve_key)
        pending_problem_viewers.setdefault(problem_key, []).append(viewer)
        pending_problem_gains.setdefault(problem_key, local_gain.copy())
        pending_problem_reserves.setdefault(
            problem_key,
            None if history_reserve is None else history_reserve.copy(),
        )
        local_resolved[viewer] = True
        local_relative_gap[viewer] = 0.0

    # Exact within-frame common-subexpression elimination precedes parallel
    # dispatch. Bit-identical private inputs generate one deterministic job;
    # its result is copied back only to viewers having that exact byte key.
    problem_keys = tuple(pending_problem_viewers)
    problem_gains = tuple(pending_problem_gains[key] for key in problem_keys)
    problem_reserves = tuple(
        pending_problem_reserves[key] for key in problem_keys)
    solved_results = ()
    solve_times = ()
    parallel_used = False
    parallel_workers = 1
    parallel_batch_wall_time_s = 0.0
    parallel_fallback = False
    deadline_fallback = False
    deadline_incumbent_count = 0
    deadline_incumbent_certified_count = 0
    deadline_uniform_count = 0
    deadline_harmonic_count = 0
    if (
        problem_gains
        and bool(force_deadline_fallback)
        and failure_mode in {"cached_or_uniform", "cached_or_harmonic"}
    ):
        parallel_fallback = True
        deadline_fallback = True
    if (
        problem_gains
        and parallel_executor is not None
        and not deadline_fallback
    ):
        parallel_attempt_started = perf_counter()
        try:
            if any(value is not None for value in problem_reserves):
                batch = parallel_executor.solve_many(
                    problem_gains,
                    public_budget,
                    minimum_deflections=problem_reserves,
                )
            else:
                batch = parallel_executor.solve_many(
                    problem_gains, public_budget)
            solved_results = tuple(batch.results)
            solve_times = tuple(batch.node_compute_time_s)
            parallel_batch_wall_time_s = float(batch.batch_wall_time_s)
            parallel_workers = int(batch.worker_count)
            if (
                len(solved_results) != len(problem_gains)
                or len(solve_times) != len(problem_gains)
                or parallel_workers < 1
            ):
                raise RuntimeError(
                    "parallel power executor returned an invalid batch")
            parallel_used = True
        except Exception as error:
            parallel_batch_wall_time_s = float(
                perf_counter() - parallel_attempt_started)
            if not bool(parallel_fallback_to_serial):
                raise RuntimeError(
                    "parallel private power execution failed") from error
            parallel_fallback = True
            solved_results = ()
            solve_times = ()
            deadline_fallback = failure_mode in {
                "cached_or_uniform", "cached_or_harmonic"}
    if problem_gains and not solved_results and not deadline_fallback:
        serial_batch_started = perf_counter()
        serial_results = []
        serial_times = []
        for local_gain, local_reserve in zip(
            problem_gains, problem_reserves
        ):
            solve_started = perf_counter()
            serial_results.append(solve_fixed_structure_maxmin_power_lp(
                local_gain,
                public_budget,
                minimum_deflection=local_reserve,
            ))
            serial_times.append(float(perf_counter() - solve_started))
        solved_results = tuple(serial_results)
        solve_times = tuple(serial_times)
        parallel_batch_wall_time_s += float(
            perf_counter() - serial_batch_started)

    if deadline_fallback:
        fallback_started = perf_counter()
        for problem_key in problem_keys:
            for viewer in pending_problem_viewers[problem_key]:
                local_gain = views[viewer]
                if previous_power is not None and bool(previous_valid[viewer]):
                    held = previous_power[viewer].copy()
                    held_sum = np.sum(held, axis=1, keepdims=True)
                    held_share = np.divide(
                        held,
                        held_sum,
                        out=np.full_like(held, 1.0 / Q),
                        where=held_sum > 0.0,
                    )
                    local_power = public_budget[:, None] * held_share
                    local_price = previous_prices[viewer].copy()
                    price_mass = float(np.sum(local_price))
                    local_price = (
                        local_price / price_mass
                        if price_mass > 0.0
                        else np.full(Q, 1.0 / Q, dtype=np.float64)
                    )
                    deadline_incumbent_count += 1
                else:
                    local_power = np.repeat(
                        (public_budget / Q)[:, None], Q, axis=1)
                    local_price = np.full(
                        Q, 1.0 / Q, dtype=np.float64)
                    deadline_uniform_count += 1
                lower = float(np.min(np.sum(
                    local_gain * local_power, axis=0)))
                upper = float(np.sum(
                    public_budget * np.max(
                        local_price[None, :] * local_gain, axis=1)))
                upper = max(upper, lower)
                relative_gap = max(
                    upper - lower, 0.0) / max(upper, 1.0e-300)
                incumbent_certified = bool(
                    previous_power is not None
                    and bool(previous_valid[viewer])
                    and relative_gap <= incumbent_tolerance
                )
                if incumbent_certified:
                    deadline_incumbent_certified_count += 1
                elif (
                    failure_mode == "cached_or_harmonic"
                    and safe_row_gain is not None
                    and np.any(safe_row_gain[viewer] > 0.0)
                ):
                    harmonic = sparse_harmonic_row_power(
                        safe_row_gain[viewer:viewer + 1],
                        public_budget[viewer:viewer + 1],
                    )
                    local_power[viewer] = harmonic[0]
                    deadline_harmonic_count += 1
                    # The cached incumbent was considered but not executed.
                    if previous_power is not None and bool(previous_valid[viewer]):
                        deadline_incumbent_count -= 1
                    else:
                        deadline_uniform_count -= 1
                    # This closed-form row allocation exactly maximizes the
                    # minimum conservative contribution over the targets
                    # reachable by node ``viewer``. Sparse zero-gain targets
                    # receive no power from this row and are covered by other
                    # rows in the composable target-wise certificate.
                    relative_gap = 0.0
                local_relative_gap[viewer] = relative_gap
                local_resolved[viewer] = False
                commit_complete_solution(
                    viewer,
                    local_gain,
                    local_power,
                    local_price,
                    perf_counter() - fallback_started,
                )
        parallel_batch_wall_time_s += float(
            perf_counter() - fallback_started)
    else:
        for problem_index, problem_key in enumerate(problem_keys):
            local = solved_results[problem_index]
            solve_elapsed = float(solve_times[problem_index])
            for viewer in pending_problem_viewers[problem_key]:
                local_relative_gap[viewer] = (
                    local.primal_dual_gap
                    / max(local.dual_upper_bound, np.finfo(np.float64).tiny)
                )
                commit_complete_solution(
                    viewer,
                    views[viewer],
                    local.power_w,
                    local.prices,
                    solve_elapsed,
                )

    common_view = bool(np.allclose(
        views, views[0][None, :, :], rtol=0.0, atol=1.0e-15))
    return ReplicatedLocalPowerResult(
        power_w=power,
        local_worst_deflection=local_worst,
        local_full_coverage=local_coverage,
        common_view=common_view,
        common_model_certificate=common_model_certificate,
        local_full_power_w=local_full_power,
        local_prices=local_prices,
        local_cache_valid=local_cache_valid,
        local_resolved=local_resolved,
        local_relative_gap=local_relative_gap,
        local_compute_time_s=local_compute_time_s,
        unique_local_problem_count=len(problem_keys),
        parallel_execution_used=parallel_used,
        parallel_worker_count=parallel_workers,
        parallel_batch_wall_time_s=parallel_batch_wall_time_s,
        parallel_fallback_to_serial=parallel_fallback,
        deadline_incumbent_used_fraction=float(
            deadline_incumbent_count / max(K, 1)),
        deadline_incumbent_certificate_fraction=float(
            deadline_incumbent_certified_count / max(K, 1)),
        deadline_uniform_fallback_fraction=float(
            deadline_uniform_count / max(K, 1)),
        deadline_harmonic_fallback_fraction=float(
            deadline_harmonic_count / max(K, 1)),
        # Each c[k,q]=a^-[k,q]p[k,q] is independently lower-bounded;
        # target-responsibility aggregation therefore certifies
        # min_q sum_k c[k,q], including sparse hyperedge rows.
        deadline_composable_deflection_floor=(
            float(np.min(np.sum(safe_row_gain * power, axis=0)))
            if safe_row_gain is not None else 0.0),
        history_reserve_used_fraction=float(
            history_reserve_count / max(K, 1)),
        common_model_fallback_fraction=float(
            common_model_fallback_count / max(K, 1)),
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

    effective_gain, _ceiling = _effective_gain_and_ceiling(gain, budget)
    if _requires_relative_primal_scaling(gain, effective_gain):
        # The compact epigraph dual has the same within-matrix conditioning
        # problem as the ordinary primal (for example [1e9, 1]).  On this
        # rare path, obtain the exact simplex multipliers from the robust
        # share-scaled primal instead of accepting an apparent zero optimum.
        primal = solve_fixed_structure_maxmin_power_lp(gain, budget)
        return primal.prices.copy(), float(primal.dual_upper_bound)

    # Normalize the gain coefficients before HiGHS.  Its feasibility tolerances
    # are absolute; solving physical radar coefficients near 1e-10 directly
    # can otherwise return a price whose evaluated dual objective differs from
    # the reported LP value by orders of magnitude.  Positive scalar scaling
    # leaves the simplex optimizer unchanged.
    gain_scale = max(float(np.max(gain)), np.finfo(np.float64).tiny)
    normalized_gain = gain / gain_scale

    # Variables: normalized u (K) then lambda (Q).
    variables = K + Q
    objective = np.zeros(variables, dtype=np.float64)
    objective[:K] = budget

    rows = []
    upper = []
    for i in range(K):
        for q in range(Q):
            row = np.zeros(variables, dtype=np.float64)
            row[i] = -1.0
            row[K + q] = float(normalized_gain[i, q])
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
        # This LP is always feasible and bounded for validated inputs.  A
        # numerical solver failure must not masquerade as a zero optimum.
        raise RuntimeError(f"optimal max-min dual LP failed: {solved.message}")
    prices = np.maximum(solved.x[K:], 0.0)
    total = float(np.sum(prices))
    if total <= 0.0:
        prices = np.full(Q, 1.0 / Q, dtype=np.float64)
    else:
        prices /= total
    reported_value = float(gain_scale * np.dot(objective, solved.x))
    value = float(np.sum(
        budget * np.max(prices[None, :] * gain, axis=1)))
    comparison_scale = max(
        abs(value), abs(reported_value), np.finfo(np.float64).tiny)
    if (
        not np.isfinite(value)
        or abs(value - reported_value) > 1.0e-7 * comparison_scale
    ):
        raise RuntimeError(
            "optimal max-min dual LP failed its physical-unit certificate; "
            f"reported={reported_value}, evaluated={value}")
    return prices, value


def _maxmin_dual_value(
    gain: np.ndarray,
    budget: np.ndarray,
    prices: np.ndarray,
) -> float:
    """Evaluate ``sum_i b_i max_q lambda_q a_iq`` without changing units."""
    return float(np.sum(
        budget * np.max(prices[None, :] * gain, axis=1)))


def _central_dual_price(
    gain: np.ndarray,
    budget: np.ndarray,
    lp_price: np.ndarray,
    f_star: float,
    *,
    value_slack: float,
    objective: str,
) -> tuple[np.ndarray, float, bool]:
    """Solve a strictly-convex secondary objective on a dual sublevel set.

    Variables are ``(u_1,...,u_K, lambda_1,...,lambda_Q)``.  Gain
    normalization changes neither the feasible prices nor the secondary
    optimum, while avoiding SLSQP tolerances being interpreted in arbitrary
    radar-coefficient units.  The returned boolean records whether the
    secondary optimum was obtained; callers retain an exact LP fallback.
    """
    K, Q = gain.shape
    gain_scale = max(float(np.max(gain)), np.finfo(np.float64).tiny)
    normalized_gain = gain / gain_scale
    lp = np.maximum(np.asarray(lp_price, dtype=np.float64), 0.0)
    lp_total = float(np.sum(lp))
    if lp_total <= 0.0:
        lp = np.full(Q, 1.0 / Q, dtype=np.float64)
    else:
        lp /= lp_total

    lp_value = _maxmin_dual_value(gain, budget, lp)
    maximum_value = float(np.sum(
        budget * np.max(gain, axis=1)))
    normalized_reference = max(
        1.0,
        abs(float(f_star)) / gain_scale,
        abs(lp_value) / gain_scale,
        abs(maximum_value) / gain_scale,
    )
    numerical_face_tolerance = float(
        128.0 * np.finfo(np.float64).eps
        * normalized_reference * gain_scale)
    certified_cap = max(
        float(f_star) + float(value_slack), lp_value) + numerical_face_tolerance
    normalized_cap = certified_cap / gain_scale

    # Start as close to the entropy/L2 unconstrained minimizer (uniform) as
    # the certified sublevel set permits.  Convexity makes every point on the
    # segment from the LP optimum to this start feasible.
    uniform = np.full(Q, 1.0 / Q, dtype=np.float64)
    if _maxmin_dual_value(gain, budget, uniform) <= certified_cap:
        initial_price = uniform
    else:
        low, high = 0.0, 1.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            mixed = (1.0 - middle) * lp + middle * uniform
            if _maxmin_dual_value(gain, budget, mixed) <= certified_cap:
                low = middle
            else:
                high = middle
        initial_price = (1.0 - low) * lp + low * uniform

    variable_count = K + Q
    upper_matrix = np.zeros((K * Q + 1, variable_count), dtype=np.float64)
    row = 0
    for i in range(K):
        for q in range(Q):
            upper_matrix[row, i] = -1.0
            upper_matrix[row, K + q] = normalized_gain[i, q]
            row += 1
    upper_matrix[-1, :K] = budget
    upper_bound = np.zeros(K * Q + 1, dtype=np.float64)
    upper_bound[-1] = normalized_cap
    equality = np.zeros((1, variable_count), dtype=np.float64)
    equality[0, K:] = 1.0

    initial_u = np.max(
        normalized_gain * initial_price[None, :], axis=1)
    initial = np.concatenate((initial_u, initial_price))
    if objective == "l2":
        def secondary(value: np.ndarray) -> float:
            prices = value[K:]
            return 0.5 * float(np.dot(prices, prices))

        def secondary_jacobian(value: np.ndarray) -> np.ndarray:
            gradient = np.zeros(variable_count, dtype=np.float64)
            gradient[K:] = value[K:]
            return gradient
    elif objective == "entropy":
        def secondary(value: np.ndarray) -> float:
            prices = value[K:]
            positive = prices > 0.0
            return float(np.sum(
                prices[positive] * np.log(prices[positive])))

        def secondary_jacobian(value: np.ndarray) -> np.ndarray:
            gradient = np.zeros(variable_count, dtype=np.float64)
            # The exact right derivative is -infinity at zero.  This finite
            # machine surrogate points into the simplex interior without
            # changing the evaluated entropy or the certified constraints.
            safe = np.maximum(value[K:], np.finfo(np.float64).tiny)
            gradient[K:] = np.log(safe) + 1.0
            return gradient
    else:
        raise ValueError("unknown secondary dual objective")

    constraints = (
        {
            "type": "ineq",
            "fun": lambda value: upper_bound - upper_matrix @ value,
            "jac": lambda _value: -upper_matrix,
        },
        {
            "type": "eq",
            "fun": lambda value: equality @ value - np.ones(1),
            "jac": lambda _value: equality,
        },
    )
    solved = minimize(
        secondary,
        initial,
        jac=secondary_jacobian,
        method="SLSQP",
        bounds=[(0.0, None)] * K + [(0.0, 1.0)] * Q,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1.0e-12, "disp": False},
    )
    if not solved.success or solved.x is None:
        return lp, lp_value, False
    prices = np.maximum(np.asarray(solved.x[K:], dtype=np.float64), 0.0)
    total = float(np.sum(prices))
    if not np.all(np.isfinite(prices)) or total <= 0.0:
        return lp, lp_value, False
    prices /= total

    # Normalization may move a solution by a few ulps.  Project it back along
    # the segment to the exact LP certificate if that tiny change crosses the
    # explicit cap; this preserves the advertised bound by construction.
    f_value = _maxmin_dual_value(gain, budget, prices)
    if f_value > certified_cap:
        low, high = 0.0, 1.0
        for _ in range(64):
            middle = 0.5 * (low + high)
            mixed = (1.0 - middle) * lp + middle * prices
            if _maxmin_dual_value(gain, budget, mixed) <= certified_cap:
                low = middle
            else:
                high = middle
        prices = (1.0 - low) * lp + low * prices
        prices /= float(np.sum(prices))
        f_value = _maxmin_dual_value(gain, budget, prices)
    feasible = bool(
        np.all(prices >= 0.0)
        and abs(float(np.sum(prices)) - 1.0) <= 1.0e-9
        and f_value <= certified_cap
    )
    return prices, f_value, feasible


def canonical_maxmin_dual_prices(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the unique minimum-L2 price on the optimal dual face.

    After obtaining the exact LP value ``f_star``, solve the secondary convex
    problem

        min  0.5 ||lambda||_2^2
        s.t. lambda in Delta_Q,  f(lambda) <= f_star + eps_num.

    The epigraph variables ``u_i >= a_iq lambda_q`` keep every constraint
    linear.  Strict convexity in ``lambda`` makes the returned price unique
    (up to the declared numerical face tolerance), permutation equivariant,
    and independent of which optimal LP basis HiGHS happened to return.
    """
    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    if not np.any(gain > 0.0) or not np.any(budget > 0.0):
        return np.full(Q, 1.0 / Q, dtype=np.float64), 0.0

    lam_lp, f_star = optimal_maxmin_dual_prices(gain, budget)
    prices, f_value, solved = _central_dual_price(
        gain,
        budget,
        lam_lp,
        f_star,
        value_slack=0.0,
        objective="l2",
    )
    if not solved:
        # The LP point remains a valid exact dual certificate.  Falling back
        # to it preserves control safety, although the rare numerical failure
        # forfeits only the secondary uniqueness property.
        return lam_lp, _maxmin_dual_value(gain, budget, lam_lp)
    return prices, f_value


def entropic_maxmin_dual_prices(
    gain_per_watt: np.ndarray,
    sensing_budget_w: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, float]:
    """Maximum-entropy dual price under a certified optimality cap.

    The fixed-owner max-min sensing-power LP has the simplex dual

        min_{lambda in Delta_Q}  f(lambda) = sum_i b_i max_q lambda_q a_iq.

    ``f`` is convex and piecewise-linear, so ``lambda*`` is a convex face in
    general and any LP solver returns a basis-dependent vertex of that face.
    This is exactly why the geometry layer's price-weighted deficit direction
    (``lambda`` times the per-target reconstruction) can jump from frame to
    frame under a gain perturbation (see the C3 dual-uniqueness note).

    We minimize negative entropy over the convex dual sublevel set

        f(lambda) <= f_star + tau log Q.

    This lexicographic/capped construction makes the optimality guarantee a
    feasibility invariant instead of assuming that an arbitrary mixture with
    the uniform price has the right physical units.  Strict convexity gives a
    unique price on the feasible support and, for the exact optimum
    ``lambda*``, directly guarantees

        (i)   0 <= f(lambda*_tau) - f(lambda*) <= tau log Q,
        (ii)  lambda*_tau -> lambda*_ent (max-entropy optimum) as tau -> 0.

    Epigraph variables ``u_i >= max_q lambda_q a_iq`` turn the sublevel set
    into KQ linear inequalities.  At ``tau=0`` this is the maximum-entropy
    point of the optimal face.  A face can force some prices to zero, so
    simplex-interiority is intentionally not claimed in that degenerate case.

    Args:
        gain_per_watt: (K, Q) per-watt deflection gain.
        sensing_budget_w: (K,) sensing power budget per UAV.
        tau: non-negative dual-objective scale.  The permitted physical
            objective slack is exactly ``tau * log(Q)``; ``0`` selects the
            maximum-entropy point on the numerically exact optimal face.

    Returns:
        ``(lambda*_tau, f_value)`` where ``f_value = f(lambda*_tau)`` is a
        valid dual upper bound satisfying the cap up to solver tolerance.
    """

    gain, budget = _validated(gain_per_watt, sensing_budget_w)
    K, Q = gain.shape
    if not np.any(gain > 0.0) or not np.any(budget > 0.0):
        return np.full(Q, 1.0 / Q, dtype=np.float64), 0.0
    if not np.isfinite(float(tau)) or float(tau) < 0.0:
        raise ValueError("tau must be finite and non-negative")

    lam_lp, f_star = optimal_maxmin_dual_prices(gain, budget)
    value_slack = float(tau) * float(np.log(Q))
    prices, f_value, solved = _central_dual_price(
        gain,
        budget,
        lam_lp,
        f_star,
        value_slack=value_slack,
        objective="entropy",
    )
    if not solved:
        # The exact LP price satisfies every requested cap.  This fail-safe is
        # preferable to returning an attractive but uncertified smooth price.
        return lam_lp, _maxmin_dual_value(gain, budget, lam_lp)
    return prices, f_value


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
            raise NonUniqueFixedOwnerStructureError(
                "fixed-owner LP requires one receiver per target")
        owners[q] = j
        gain[i, q] += values[i, j, q]
    if np.any(owners < 0):
        raise IncompleteFixedOwnerStructureError(
            "fixed-owner LP requires every target to have an owner")
    return gain, owners


def fixed_owner_gain_matrix_from_selected_values(
    selected_values: np.ndarray,
    selected: Sequence[tuple[int, int, int]],
    *,
    num_uavs: int,
    num_targets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse COO edge values without materializing ``(...,K,K,Q)``.

    Leading dimensions are preserved, so a viewer batch ``(V,E)`` becomes
    ``(V,K,Q)``.  Edge iteration deliberately matches
    :func:`fixed_owner_gain_matrix`, including its accumulation order.
    """
    values = np.asarray(selected_values, dtype=np.float64)
    edges = tuple(tuple(int(value) for value in edge) for edge in selected)
    K = int(num_uavs)
    Q = int(num_targets)
    if values.ndim < 1 or values.shape[-1] != len(edges):
        raise ValueError("selected_values must have trailing edge dimension E")
    if K < 1 or Q < 1:
        raise ValueError("num_uavs and num_targets must be positive")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("selected_values must be finite and non-negative")
    owners = np.full(Q, -1, dtype=np.int64)
    gain = np.zeros(values.shape[:-1] + (K, Q), dtype=np.float64)
    for edge_index, (i, j, q) in enumerate(edges):
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q) or i == j:
            raise ValueError("selected edge is outside coefficient support")
        if owners[q] not in (-1, j):
            raise NonUniqueFixedOwnerStructureError(
                "fixed-owner LP requires one receiver per target")
        owners[q] = j
        gain[..., i, q] += values[..., edge_index]
    if np.any(owners < 0):
        raise IncompleteFixedOwnerStructureError(
            "fixed-owner LP requires every target to have an owner")
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
