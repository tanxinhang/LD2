"""Correlation-aware, dual-pruned, exact local structural exchange.

This is a research-gate kernel rather than an online distributed protocol.
It reuses the audited N5/N6 move neighborhood, ranks candidates with the
incumbent LP dual prices, safely prunes candidates whose weak-duality upper
bound cannot improve the incumbent, and accepts a move only after solving the
correlation-calibrated fixed-structure LP exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    enumerate_local_moves,
    role_owner_from_structure,
)
from uav_isac.coordination.maxmin_power import (
    MaxMinPowerResult,
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.physical.correlation_calibration import (
    apply_target_correlation_factors,
    otfs_target_correlation_factor_from_arrays,
    selected_otfs_correlation_factors_from_arrays,
)


@dataclass(frozen=True)
class CorrelationAwareExchangeResult:
    selected: np.ndarray
    role: np.ndarray
    owner: np.ndarray
    baseline_worst_deflection: float
    power_result: MaxMinPowerResult
    accepted: bool
    accepted_kind: str
    candidate_count: int
    dual_upper_pruned_count: int
    raw_gain_upper_pruned_count: int
    exact_verification_count: int
    physical_prefilter_rejected_count: int
    exact_gate_rejected_count: int
    baseline_correlation_factors: np.ndarray
    final_correlation_factors: np.ndarray
    verified_moves: tuple[LocalMove, ...]
    verified_dual_scores: tuple[float, ...]
    gram_compute_time_s: float
    exact_lp_time_s: float
    correlation_factor_cache_hits: int
    correlation_factor_cache_misses: int
    dual_bound_early_stopped: bool


def _dual_upper(
    gain: np.ndarray,
    budget: np.ndarray,
    prices: np.ndarray,
) -> float:
    """Weak-duality upper bound at a feasible target-simplex price."""
    lam = np.asarray(prices, dtype=np.float64).reshape(-1)
    if np.any(lam < 0.0) or not np.isfinite(lam).all() or np.sum(lam) <= 0.0:
        raise ValueError("dual prices must be a finite non-negative simplex")
    lam = lam / float(np.sum(lam))
    return float(np.sum(
        np.asarray(budget, dtype=np.float64)
        * np.max(np.asarray(gain, dtype=np.float64) * lam[None, :], axis=1)
    ))


def correlation_calibrated_structure_gain(
    coefficient_per_watt: np.ndarray,
    selected: np.ndarray,
    delay_s: np.ndarray,
    doppler_hz: np.ndarray,
    *,
    delay_size: int,
    doppler_size: int,
    delta_f_hz: float,
    symbol_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return calibrated ``(K,Q)`` gain, owners and target factors."""
    coefficient = np.asarray(coefficient_per_watt, dtype=np.float64)
    mask = np.asarray(selected, dtype=bool)
    gain, owner = fixed_owner_gain_matrix(
        coefficient,
        [tuple(int(value) for value in edge) for edge in np.argwhere(mask)],
    )
    factors = selected_otfs_correlation_factors_from_arrays(
        mask,
        coefficient,
        delay_s,
        doppler_hz,
        delay_size=int(delay_size),
        doppler_size=int(doppler_size),
        delta_f_hz=float(delta_f_hz),
        symbol_time_s=float(symbol_time_s),
    )
    return apply_target_correlation_factors(gain, factors), owner, factors


def correlation_aware_dual_pruned_exact_exchange(
    initial_selected: np.ndarray,
    coefficient_per_watt: np.ndarray,
    candidate_mask: np.ndarray,
    initial_role: np.ndarray,
    sensing_budget_w: np.ndarray,
    delay_s: np.ndarray,
    doppler_hz: np.ndarray,
    *,
    delay_size: int,
    doppler_size: int,
    delta_f_hz: float,
    symbol_time_s: float,
    target_pair_limit: int,
    reports_per_receiver: int,
    top_m: int = 8,
    weak_target_count: int = 2,
    improvement_tolerance: float = 1.0e-10,
    minimum_deflection: np.ndarray | None = None,
    neighborhoods: tuple[str, ...] = ("N5", "N6"),
    pre_candidate_gate: Callable[[LocalMove], bool] | None = None,
    exact_candidate_gate: Callable[
        [LocalMove, np.ndarray, np.ndarray, MaxMinPowerResult], bool
    ] | None = None,
    dual_bound_early_stop: bool = True,
    raw_gain_upper_prune: bool = True,
) -> CorrelationAwareExchangeResult:
    """Evaluate one bounded structural exchange around an exact incumbent.

    The primary screening score is the candidate headroom under the incumbent
    dual price, ``U_lambda(E')-eta*``.  The incumbent-power replay sensitivity
    is retained as a deterministic secondary key:

        U_lambda(E') = sum_i b_i max_q lambda_q g'_iq,
        r(E') = lambda^T [D(E', p*) - D(E, p*)].

    This is a first-order ranker only.  The weak-duality upper bound is used
    solely for sound rejection, and the best Top-M candidate is selected by
    an exact LP.  Retaining the incumbent as an implicit candidate proves the
    returned objective never decreases within numerical tolerance.
    """
    selected = np.asarray(initial_selected, dtype=bool)
    coefficient = np.asarray(coefficient_per_watt, dtype=np.float64)
    support = np.asarray(candidate_mask, dtype=bool)
    role = np.asarray(initial_role, dtype=np.int8).reshape(-1)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    tau = np.asarray(delay_s, dtype=np.float64)
    nu = np.asarray(doppler_hz, dtype=np.float64)
    if (
        selected.shape != coefficient.shape or support.shape != selected.shape
        or tau.shape != selected.shape or nu.shape != selected.shape
        or selected.ndim != 3 or selected.shape[0] != selected.shape[1]
    ):
        raise ValueError("structure tensors must share shape (K,K,Q)")
    K, _, Q = selected.shape
    if role.shape != (K,) or budget.shape != (K,):
        raise ValueError("role and sensing budget must match K")
    if (
        np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
        or int(top_m) < 1 or int(weak_target_count) < 1
        or int(target_pair_limit) < 1 or int(reports_per_receiver) < 1
        or not np.isfinite(improvement_tolerance)
        or float(improvement_tolerance) < 0.0
    ):
        raise ValueError("exchange inputs are outside finite support")
    reserve = None
    if minimum_deflection is not None:
        reserve = np.asarray(minimum_deflection, dtype=np.float64).reshape(-1)
        if (
            reserve.shape != (Q,) or np.any(~np.isfinite(reserve))
            or np.any(reserve < 0.0)
        ):
            raise ValueError("minimum_deflection must be non-negative Q-vector")

    gram_compute_time_s = 0.0
    exact_lp_time_s = 0.0
    factor_cache: dict[tuple[int, bytes], float] = {}
    factor_cache_hits = 0
    factor_cache_misses = 0

    def cached_calibrated_gain(
        candidate_selected: np.ndarray,
        *,
        inherited_factors: np.ndarray | None = None,
        changed_targets: np.ndarray | None = None,
        supplied_raw_gain: np.ndarray | None = None,
        supplied_owner: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal factor_cache_hits, factor_cache_misses
        mask = np.asarray(candidate_selected, dtype=bool)
        if supplied_raw_gain is None or supplied_owner is None:
            raw_gain, candidate_owner = fixed_owner_gain_matrix(
                coefficient,
                [tuple(int(value) for value in edge)
                 for edge in np.argwhere(mask)],
            )
        else:
            raw_gain = np.asarray(supplied_raw_gain, dtype=np.float64)
            candidate_owner = np.asarray(supplied_owner, dtype=np.int64)
        active = mask & (coefficient > 0.0)
        factors = (
            np.ones(Q, dtype=np.float64)
            if inherited_factors is None
            else np.asarray(inherited_factors, dtype=np.float64).copy()
        )
        targets = (
            range(Q)
            if changed_targets is None
            else (int(value) for value in np.asarray(
                changed_targets, dtype=np.int64).reshape(-1))
        )
        for target in targets:
            target_mask = np.ascontiguousarray(active[:, :, target])
            key = (
                target,
                np.packbits(
                    target_mask.reshape(-1), bitorder="big").tobytes(),
            )
            cached = factor_cache.get(key)
            if cached is None:
                cached = otfs_target_correlation_factor_from_arrays(
                    target_mask,
                    tau[:, :, target],
                    nu[:, :, target],
                    delay_size=int(delay_size),
                    doppler_size=int(doppler_size),
                    delta_f_hz=float(delta_f_hz),
                    symbol_time_s=float(symbol_time_s),
                )
                factor_cache[key] = float(cached)
                factor_cache_misses += 1
            else:
                factor_cache_hits += 1
            factors[target] = float(cached)
        return (
            apply_target_correlation_factors(raw_gain, factors),
            candidate_owner,
            factors,
        )

    gram_started = perf_counter()
    baseline_gain, baseline_owner, baseline_factors = cached_calibrated_gain(
        selected)
    gram_compute_time_s += perf_counter() - gram_started
    lp_started = perf_counter()
    baseline = solve_fixed_structure_maxmin_power_lp(
        baseline_gain, budget, minimum_deflection=reserve)
    exact_lp_time_s += perf_counter() - lp_started
    inferred_role, _ = role_owner_from_structure(
        selected, fallback_role=role)
    if not np.array_equal(inferred_role, role):
        raise ValueError("initial_role is inconsistent with selected structure")

    moves = enumerate_local_moves(
        selected,
        coefficient * budget[:, None, None],
        support,
        role,
        baseline_owner,
        neighborhoods=tuple(str(value) for value in neighborhoods),
        target_pair_limit=int(target_pair_limit),
        reports_per_receiver=int(reports_per_receiver),
        n5_target_mode="proxy_weak",
        n5_weak_target_count=int(weak_target_count),
        n5_rebuild_scope="target_block",
    )
    baseline_replay = np.sum(
        baseline_gain * baseline.power_w, axis=0)
    ranked: list[tuple[
        float, float, int, int, LocalMove, np.ndarray, np.ndarray,
    ]] = []
    pruned = 0
    raw_upper_pruned = 0
    physical_prefilter_rejected = 0
    for index, move in enumerate(moves):
        if pre_candidate_gate is not None and not bool(pre_candidate_gate(move)):
            physical_prefilter_rejected += 1
            continue
        try:
            raw_gain, raw_owner = fixed_owner_gain_matrix(
                coefficient,
                [tuple(int(value) for value in edge)
                 for edge in np.argwhere(move.selected)],
            )
            # Since every correlation factor is >=1, calibrated_gain<=raw_gain
            # componentwise. The raw-gain dual is therefore a cheap valid
            # upper bound; rejecting here cannot discard an improving move.
            raw_upper = _dual_upper(raw_gain, budget, baseline.prices)
            if bool(raw_gain_upper_prune) and raw_upper <= (
                baseline.worst_deflection + float(improvement_tolerance)
            ):
                pruned += 1
                raw_upper_pruned += 1
                continue
            gram_started = perf_counter()
            changed_targets = np.flatnonzero(np.any(
                np.asarray(move.selected, dtype=bool) != selected,
                axis=(0, 1),
            ))
            gain, _, factors = cached_calibrated_gain(
                move.selected,
                inherited_factors=baseline_factors,
                changed_targets=changed_targets,
                supplied_raw_gain=raw_gain,
                supplied_owner=raw_owner,
            )
            gram_compute_time_s += perf_counter() - gram_started
        except ValueError:
            continue
        if reserve is not None and np.any(
            np.sum(gain * budget[:, None], axis=0) + 1.0e-12 < reserve
        ):
            pruned += 1
            continue
        upper = _dual_upper(gain, budget, baseline.prices)
        if upper <= baseline.worst_deflection + float(improvement_tolerance):
            pruned += 1
            continue
        candidate_replay = np.sum(gain * baseline.power_w, axis=0)
        replay_sensitivity = float(np.dot(
            baseline.prices, candidate_replay - baseline_replay))
        opportunity = float(upper - baseline.worst_deflection)
        edge_changes = int(np.sum(move.selected != selected))
        ranked.append((
            opportunity, replay_sensitivity, -edge_changes, -index,
            move, gain, factors,
        ))
    ranked.sort(key=lambda item: item[:4], reverse=True)

    best = baseline
    best_move: LocalMove | None = None
    best_factors = baseline_factors
    verified_moves: list[LocalMove] = []
    verified_scores: list[float] = []
    gate_rejected = 0
    early_stopped = False
    ranked_limit = ranked[:int(top_m)]
    for rank_index, (
        opportunity, _, _, _, move, gain, factors,
    ) in enumerate(ranked_limit):
        verified_moves.append(move)
        verified_scores.append(float(opportunity))
        try:
            lp_started = perf_counter()
            result = solve_fixed_structure_maxmin_power_lp(
                gain, budget, minimum_deflection=reserve)
            exact_lp_time_s += perf_counter() - lp_started
        except RuntimeError:
            exact_lp_time_s += perf_counter() - lp_started
            continue
        if (
            exact_candidate_gate is not None
            and not bool(exact_candidate_gate(move, gain, factors, result))
        ):
            gate_rejected += 1
            continue
        if result.worst_deflection > (
            best.worst_deflection + float(improvement_tolerance)
        ):
            best = result
            best_move = move
            best_factors = factors
        if bool(dual_bound_early_stop):
            next_index = rank_index + 1
            next_upper = (
                baseline.worst_deflection + ranked[next_index][0]
                if next_index < len(ranked)
                else float("-inf")
            )
            if best.worst_deflection + float(improvement_tolerance) >= next_upper:
                early_stopped = True
                break

    if best_move is None:
        return CorrelationAwareExchangeResult(
            selected=selected.copy(), role=role.copy(),
            owner=baseline_owner.copy(), power_result=baseline,
            baseline_worst_deflection=float(baseline.worst_deflection),
            accepted=False, accepted_kind="no_op",
            candidate_count=(
                len(ranked) + pruned + physical_prefilter_rejected),
            dual_upper_pruned_count=pruned,
            raw_gain_upper_pruned_count=raw_upper_pruned,
            exact_verification_count=len(verified_moves),
            physical_prefilter_rejected_count=physical_prefilter_rejected,
            exact_gate_rejected_count=gate_rejected,
            baseline_correlation_factors=baseline_factors.copy(),
            final_correlation_factors=baseline_factors.copy(),
            verified_moves=tuple(verified_moves),
            verified_dual_scores=tuple(verified_scores),
            gram_compute_time_s=float(gram_compute_time_s),
            exact_lp_time_s=float(exact_lp_time_s),
            correlation_factor_cache_hits=int(factor_cache_hits),
            correlation_factor_cache_misses=int(factor_cache_misses),
            dual_bound_early_stopped=early_stopped,
        )
    return CorrelationAwareExchangeResult(
        selected=best_move.selected.copy(), role=best_move.role.copy(),
        owner=best_move.owner.copy(), power_result=best,
        baseline_worst_deflection=float(baseline.worst_deflection),
        accepted=True, accepted_kind=str(best_move.kind),
        candidate_count=(len(ranked) + pruned + physical_prefilter_rejected),
        dual_upper_pruned_count=pruned,
        raw_gain_upper_pruned_count=raw_upper_pruned,
        exact_verification_count=len(verified_moves),
        physical_prefilter_rejected_count=physical_prefilter_rejected,
        exact_gate_rejected_count=gate_rejected,
        baseline_correlation_factors=baseline_factors.copy(),
        final_correlation_factors=np.asarray(best_factors).copy(),
        verified_moves=tuple(verified_moves),
        verified_dual_scores=tuple(verified_scores),
        gram_compute_time_s=float(gram_compute_time_s),
        exact_lp_time_s=float(exact_lp_time_s),
        correlation_factor_cache_hits=int(factor_cache_hits),
        correlation_factor_cache_misses=int(factor_cache_misses),
        dual_bound_early_stopped=early_stopped,
    )
