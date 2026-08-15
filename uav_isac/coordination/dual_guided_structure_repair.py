"""Dual-guided atomic structural repair above the fixed-power layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    enumerate_local_moves,
    role_owner_from_structure,
)
from uav_isac.coordination.maxmin_power import (
    MaxMinPowerResult,
    distributed_column_generation_maxmin_power,
    fixed_owner_gain_matrix,
)
from uav_isac.coordination.owner_proposal_transport import (
    RankedOwnerProposal,
    quantize_nonnegative_float16_lower,
    quantize_nonnegative_float16_upper,
)


@dataclass(frozen=True)
class DualGuidedStructureRepairResult:
    selected: np.ndarray
    role: np.ndarray
    owner: np.ndarray
    power_result: MaxMinPowerResult
    accepted: bool
    accepted_kind: str
    accepted_steps: int
    candidate_count: int
    exact_verification_count: int
    owner_proposal_count: int
    proxy_best_lower: float
    changed_edge_count: int
    changed_target_count: int
    ranked_candidate_moves: tuple[LocalMove, ...]
    verified_moves: tuple[LocalMove, ...]
    accepted_moves: tuple[LocalMove, ...]
    owner_proposal_rounds: tuple[tuple[RankedOwnerProposal, ...], ...]


@dataclass(frozen=True)
class DualGuidedAtomicCandidateSet:
    """Unranked, independently generated first-step atomic candidate set."""

    moves: tuple[LocalMove, ...]
    proposers: tuple[int, ...]
    initial_owner: np.ndarray


def enumerate_dual_guided_atomic_candidates(
    initial_selected: np.ndarray,
    predicted_coefficient: np.ndarray,
    candidate_mask: np.ndarray,
    initial_role: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    weak_target_count: int = 2,
    minimum_deflection: np.ndarray | None = None,
) -> DualGuidedAtomicCandidateSet:
    """Generate the exact first-step set before any proxy/H-step ranking.

    This function intentionally performs no max-min power solve and consumes no
    preferred action.  It exposes the same valid/reserve-possible moves, in the
    same first-occurrence order, that ``dual_guided_atomic_structure_repair``
    passes to its ranker when ``max_steps=1``.  A later digest match can thus
    select one independently generated member without ranking unrelated moves.
    """
    selected = np.asarray(initial_selected, dtype=bool)
    coefficient = np.asarray(predicted_coefficient, dtype=np.float64)
    support = np.asarray(candidate_mask, dtype=bool)
    role = np.asarray(initial_role, dtype=np.int8).reshape(-1)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        selected.shape != coefficient.shape
        or selected.shape != support.shape
        or coefficient.ndim != 3
        or coefficient.shape[0] != coefficient.shape[1]
    ):
        raise ValueError("selected/coefficient/support must share shape (K,K,Q)")
    K, _, Q = coefficient.shape
    if role.shape != (K,) or budget.shape != (K,):
        raise ValueError("role and sensing budget must match K")
    if (
        np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
        or int(target_pair_limit) < 1 or int(reports_per_receiver) < 1
        or int(weak_target_count) < 1
    ):
        raise ValueError("candidate-set inputs are outside finite support")
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
    _, initial_owner = fixed_owner_gain_matrix(
        coefficient, [tuple(edge) for edge in np.argwhere(selected)])
    inferred_role, _ = role_owner_from_structure(
        selected, fallback_role=role)
    if not np.array_equal(inferred_role, role):
        raise ValueError("initial_role is inconsistent with selected structure")
    moves = enumerate_local_moves(
        selected,
        coefficient * budget[:, None, None],
        support,
        role,
        initial_owner,
        neighborhoods=("N5", "N6"),
        target_pair_limit=int(target_pair_limit),
        reports_per_receiver=int(reports_per_receiver),
        n5_target_mode="proxy_weak",
        n5_weak_target_count=int(weak_target_count),
        n5_rebuild_scope="target_block",
    )
    admitted: list[LocalMove] = []
    proposers: list[int] = []
    for move in moves:
        try:
            gain, _ = fixed_owner_gain_matrix(
                coefficient,
                [tuple(edge) for edge in np.argwhere(move.selected)],
            )
        except ValueError:
            continue
        if reserve is not None and np.any(
            np.sum(gain * budget[:, None], axis=0)
            + 1.0e-12 < reserve
        ):
            continue
        affected_targets = np.flatnonzero(
            np.any(move.selected != selected, axis=(0, 1))
            | (move.owner != initial_owner)
        )
        if affected_targets.size == 0:
            continue
        proposer = int(np.min(initial_owner[affected_targets]))
        if proposer < 0 or proposer >= K:
            continue
        admitted.append(move)
        proposers.append(proposer)
    return DualGuidedAtomicCandidateSet(
        moves=tuple(admitted),
        proposers=tuple(proposers),
        initial_owner=initial_owner.copy(),
    )


def _common_mix_lower(gain: np.ndarray, budget: np.ndarray) -> float:
    """Feasible common-column time-sharing lower bound.

    If all UAVs focus on target q, its Deflection is c_q.  Time-sharing these
    Q complete RF columns with weights proportional to 1/c_q achieves the
    common value 1/sum_q(1/c_q), preserving every transmitter budget.
    """
    capacity = np.sum(gain * budget[:, None], axis=0)
    if np.any(capacity <= 0.0):
        return 0.0
    return float(1.0 / np.sum(1.0 / capacity))


def _common_mix_reserve_lower(
    gain: np.ndarray,
    budget: np.ndarray,
    minimum_deflection: np.ndarray,
) -> float:
    """Feasible reserve-first water filling over public target columns.

    The all-UAV-on-target-q column has capacity ``c_q``.  A convex mixture
    with weights ``w_q=max(r_q,z)/c_q`` satisfies every reserve ``r_q`` and
    achieves common Deflection ``z`` whenever the weights sum to at most one.
    Bisection returns the largest such feasible ``z``.  Zero denotes that this
    deliberately restricted public-column proxy cannot certify feasibility.
    """
    values = np.asarray(gain, dtype=np.float64)
    reserve = np.asarray(minimum_deflection, dtype=np.float64).reshape(-1)
    capacity = np.sum(values * np.asarray(budget)[:, None], axis=0)
    if (
        reserve.shape != capacity.shape or np.any(reserve < 0.0)
        or np.any(capacity <= 0.0)
        or float(np.sum(reserve / capacity)) > 1.0 + 1.0e-12
    ):
        return 0.0
    lower = 0.0
    upper = float(np.max(capacity))
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if float(np.sum(np.maximum(reserve, middle) / capacity)) <= 1.0:
            lower = middle
        else:
            upper = middle
    return float(lower)


def _common_mix_power_witness(
    gain: np.ndarray,
    budget: np.ndarray,
    minimum_deflection: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the public-column lower bound and its feasible RF witness."""
    values = np.asarray(gain, dtype=np.float64)
    reserve = np.asarray(budget, dtype=np.float64).reshape(-1)
    K, Q = values.shape
    if reserve.shape != (K,):
        raise ValueError("budget must match the gain transmitters")
    capacity = np.sum(values * reserve[:, None], axis=0)
    if np.any(capacity <= 0.0):
        return 0.0, np.zeros((K, Q), dtype=np.float64)
    if minimum_deflection is None:
        weights = (1.0 / capacity) / float(np.sum(1.0 / capacity))
        lower = float(1.0 / np.sum(1.0 / capacity))
    else:
        target_reserve = np.asarray(
            minimum_deflection, dtype=np.float64).reshape(-1)
        if (
            target_reserve.shape != (Q,) or np.any(target_reserve < 0.0)
            or float(np.sum(target_reserve / capacity)) > 1.0 + 1.0e-12
        ):
            return 0.0, np.zeros((K, Q), dtype=np.float64)
        lower = _common_mix_reserve_lower(
            values, reserve, target_reserve)
        weights = np.maximum(target_reserve, lower) / capacity
        slack = max(1.0 - float(np.sum(weights)), 0.0)
        weights[int(np.argmax(capacity))] += slack
    power = reserve[:, None] * weights[None, :]
    deflection = np.sum(values * power, axis=0)
    if minimum_deflection is not None and np.any(
        deflection + 1.0e-10 < np.asarray(minimum_deflection)
    ):
        raise AssertionError("common-column RF witness violates its reserve")
    return float(np.min(deflection)), power


def _line_segment_maxmin_lower(
    first_deflection: np.ndarray,
    second_deflection: np.ndarray,
) -> float:
    """Maximise the worst target on a feasible two-column line segment."""
    first = np.asarray(first_deflection, dtype=np.float64).reshape(-1)
    second = np.asarray(second_deflection, dtype=np.float64).reshape(-1)
    if (
        first.shape != second.shape or first.size < 1
        or np.any(~np.isfinite(first)) or np.any(~np.isfinite(second))
        or np.any(first < 0.0) or np.any(second < 0.0)
    ):
        raise ValueError("deflection columns must be finite non-negative peers")
    slope = second - first
    candidates = [0.0, 1.0]
    for left in range(first.size):
        for right in range(left + 1, first.size):
            denominator = slope[left] - slope[right]
            if abs(float(denominator)) <= 1.0e-15:
                continue
            theta = float((first[right] - first[left]) / denominator)
            if 0.0 < theta < 1.0:
                candidates.append(theta)
    return float(max(
        np.min(first + theta * slope) for theta in candidates))


def _line_segment_power_witness(
    first_power: np.ndarray,
    second_power: np.ndarray,
    gain: np.ndarray,
    minimum_deflection: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the best feasible point and RF witness on a power segment."""
    first = np.asarray(first_power, dtype=np.float64)
    second = np.asarray(second_power, dtype=np.float64)
    values = np.asarray(gain, dtype=np.float64)
    if first.shape != second.shape or first.shape != values.shape:
        raise ValueError("power witnesses and gain must share shape (K,Q)")
    first_deflection = np.sum(values * first, axis=0)
    second_deflection = np.sum(values * second, axis=0)
    slope = second_deflection - first_deflection
    low, high = 0.0, 1.0
    reserve = None
    if minimum_deflection is not None:
        reserve = np.asarray(
            minimum_deflection, dtype=np.float64).reshape(-1)
        if reserve.shape != first_deflection.shape:
            raise ValueError("minimum_deflection must match target count")
        for target in range(reserve.size):
            if abs(float(slope[target])) <= 1.0e-15:
                if first_deflection[target] + 1.0e-12 < reserve[target]:
                    return 0.0, first.copy()
                continue
            crossing = float(
                (reserve[target] - first_deflection[target]) / slope[target])
            if slope[target] > 0.0:
                low = max(low, crossing)
            else:
                high = min(high, crossing)
        low = max(low, 0.0)
        high = min(high, 1.0)
        if low > high + 1.0e-12:
            return 0.0, first.copy()
    candidates = [low, high]
    for left in range(first_deflection.size):
        for right in range(left + 1, first_deflection.size):
            denominator = slope[left] - slope[right]
            if abs(float(denominator)) <= 1.0e-15:
                continue
            theta = float(
                (first_deflection[right] - first_deflection[left])
                / denominator)
            if low < theta < high:
                candidates.append(theta)
    theta = max(candidates, key=lambda value: (
        float(np.min(first_deflection + value * slope)), -float(value)))
    power = first + theta * (second - first)
    deflection = np.sum(values * power, axis=0)
    if reserve is not None and np.any(deflection + 1.0e-10 < reserve):
        raise AssertionError("line-segment RF witness violates its reserve")
    return float(np.min(deflection)), power


def _line_segment_reserve_lower(
    first_deflection: np.ndarray,
    second_deflection: np.ndarray,
    minimum_deflection: np.ndarray,
) -> float:
    """Best reserve-feasible worst value on a two-column segment."""
    first = np.asarray(first_deflection, dtype=np.float64).reshape(-1)
    second = np.asarray(second_deflection, dtype=np.float64).reshape(-1)
    reserve = np.asarray(minimum_deflection, dtype=np.float64).reshape(-1)
    if reserve.shape != first.shape:
        raise ValueError("minimum_deflection must match the segment targets")
    slope = second - first
    low, high = 0.0, 1.0
    for q in range(first.size):
        if abs(float(slope[q])) <= 1.0e-15:
            if first[q] + 1.0e-12 < reserve[q]:
                return 0.0
            continue
        crossing = float((reserve[q] - first[q]) / slope[q])
        if slope[q] > 0.0:
            low = max(low, crossing)
        else:
            high = min(high, crossing)
    low = max(low, 0.0)
    high = min(high, 1.0)
    if low > high + 1.0e-12:
        return 0.0
    candidates = [low, high]
    for left in range(first.size):
        for right in range(left + 1, first.size):
            denominator = slope[left] - slope[right]
            if abs(float(denominator)) <= 1.0e-15:
                continue
            theta = float((first[right] - first[left]) / denominator)
            if low < theta < high:
                candidates.append(theta)
    return float(max(
        np.min(first + theta * slope) for theta in candidates))


def _dual_seeded_two_column_lower(
    gain: np.ndarray,
    budget: np.ndarray,
    incumbent_power: np.ndarray,
    public_prices: np.ndarray,
    minimum_deflection: np.ndarray | None = None,
) -> float:
    """Feasible two-column lower bound seeded by public dual prices."""
    values = np.asarray(gain, dtype=np.float64)
    reserve = np.asarray(budget, dtype=np.float64).reshape(-1)
    incumbent = np.asarray(incumbent_power, dtype=np.float64)
    prices = np.asarray(public_prices, dtype=np.float64).reshape(-1)
    K, Q = values.shape
    if (
        reserve.shape != (K,) or incumbent.shape != (K, Q)
        or prices.shape != (Q,) or np.any(~np.isfinite(prices))
        or np.any(prices < 0.0) or float(np.sum(prices)) <= 0.0
    ):
        raise ValueError("dual-seeded lower-bound dimensions are inconsistent")
    prices = prices / float(np.sum(prices))
    response = np.zeros_like(incumbent)
    for transmitter in range(K):
        target = int(np.argmax(prices * values[transmitter]))
        response[transmitter, target] = reserve[transmitter]
    replay_deflection = np.sum(values * incumbent, axis=0)
    response_deflection = np.sum(values * response, axis=0)
    return (
        _line_segment_maxmin_lower(replay_deflection, response_deflection)
        if minimum_deflection is None
        else _line_segment_reserve_lower(
            replay_deflection, response_deflection, minimum_deflection)
    )


def _dual_seeded_two_column_witness(
    gain: np.ndarray,
    budget: np.ndarray,
    incumbent_power: np.ndarray,
    public_prices: np.ndarray,
    minimum_deflection: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the dual-seeded lower bound with its convex RF witness."""
    values = np.asarray(gain, dtype=np.float64)
    reserve = np.asarray(budget, dtype=np.float64).reshape(-1)
    incumbent = np.asarray(incumbent_power, dtype=np.float64)
    prices = np.asarray(public_prices, dtype=np.float64).reshape(-1)
    K, Q = values.shape
    if (
        reserve.shape != (K,) or incumbent.shape != (K, Q)
        or prices.shape != (Q,) or np.any(~np.isfinite(prices))
        or np.any(prices < 0.0) or float(np.sum(prices)) <= 0.0
    ):
        raise ValueError("dual-seeded witness dimensions are inconsistent")
    prices = prices / float(np.sum(prices))
    response = np.zeros_like(incumbent)
    for transmitter in range(K):
        target = int(np.argmax(prices * values[transmitter]))
        response[transmitter, target] = reserve[transmitter]
    return _line_segment_power_witness(
        incumbent,
        response,
        values,
        minimum_deflection=minimum_deflection,
    )


def _public_price_dual_upper(
    gain: np.ndarray,
    budget: np.ndarray,
    public_prices: np.ndarray,
) -> float:
    """Valid max-min upper bound at any public simplex price vector."""
    values = np.asarray(gain, dtype=np.float64)
    reserve = np.asarray(budget, dtype=np.float64).reshape(-1)
    prices = np.asarray(public_prices, dtype=np.float64).reshape(-1)
    K, Q = values.shape
    if (
        reserve.shape != (K,) or prices.shape != (Q,)
        or np.any(~np.isfinite(prices)) or np.any(prices < 0.0)
        or float(np.sum(prices)) <= 0.0
    ):
        raise ValueError("dual upper-bound dimensions are inconsistent")
    prices = prices / float(np.sum(prices))
    return float(np.sum(
        reserve * np.max(prices[None, :] * values, axis=1)))


def _candidate_proxy_lower(
    gain: np.ndarray,
    budget: np.ndarray,
    incumbent_power: np.ndarray,
    public_prices: np.ndarray | None = None,
    minimum_deflection: np.ndarray | None = None,
) -> float:
    replay_deflection = np.sum(gain * incumbent_power, axis=0)
    if minimum_deflection is None:
        replay = float(np.min(replay_deflection))
        common = _common_mix_lower(gain, budget)
    else:
        reserve = np.asarray(minimum_deflection, dtype=np.float64)
        replay = (
            float(np.min(replay_deflection))
            if np.all(replay_deflection + 1.0e-12 >= reserve) else 0.0
        )
        common = _common_mix_reserve_lower(gain, budget, reserve)
    lower = max(replay, common)
    if public_prices is not None:
        lower = max(lower, _dual_seeded_two_column_lower(
            gain,
            budget,
            incumbent_power,
            public_prices,
            minimum_deflection=minimum_deflection,
        ))
    return lower


def _candidate_proxy_witness(
    gain: np.ndarray,
    budget: np.ndarray,
    incumbent_power: np.ndarray,
    public_prices: np.ndarray | None = None,
    minimum_deflection: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the strongest proxy lower bound and its executable RF plan."""
    values = np.asarray(gain, dtype=np.float64)
    incumbent = np.asarray(incumbent_power, dtype=np.float64)
    incumbent_deflection = np.sum(values * incumbent, axis=0)
    reserve = (
        None if minimum_deflection is None
        else np.asarray(minimum_deflection, dtype=np.float64).reshape(-1)
    )
    plans: list[tuple[float, np.ndarray]] = []
    if reserve is None or np.all(incumbent_deflection + 1.0e-12 >= reserve):
        plans.append((float(np.min(incumbent_deflection)), incumbent.copy()))
    plans.append(_common_mix_power_witness(
        values, budget, minimum_deflection=reserve))
    if public_prices is not None:
        plans.append(_dual_seeded_two_column_witness(
            values,
            budget,
            incumbent,
            public_prices,
            minimum_deflection=reserve,
        ))
    return max(plans, key=lambda item: float(item[0]))


def dual_guided_atomic_structure_repair(
    initial_selected: np.ndarray,
    predicted_coefficient: np.ndarray,
    candidate_mask: np.ndarray,
    initial_role: np.ndarray,
    sensing_budget_w: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    rounds: int = 4,
    ranking_rounds: int | None = None,
    price_bits: int = 6,
    feedback_bits: int = 16,
    top_m: int = 8,
    max_steps: int = 1,
    weak_target_count: int = 2,
    proxy_mode: str = "feasible_mix",
    optimism_weight: float = 0.0,
    interval_policy: str = "convex",
    owner_proposal_mode: bool = False,
    improvement_tolerance: float = 1.0e-10,
    minimum_deflection: np.ndarray | None = None,
    incumbent_power_w: np.ndarray | None = None,
    candidate_interval_ranker: Callable[
        [LocalMove], tuple[float, float, float]
    ] | None = None,
    candidate_batch_interval_ranker: Callable[
        [tuple[LocalMove, ...]],
        Sequence[tuple[float, float, float]],
    ] | None = None,
) -> DualGuidedStructureRepairResult:
    """Rank bounded atomic N5/N6 moves, then physically verify only Top-M.

    The proxy is not treated as a certificate: it is the maximum of two
    explicitly feasible lower bounds (replaying the incumbent RF allocation
    and time-sharing Q all-target columns).  ``dual_interval`` may either use a
    convex interval score or ``certificate_then_upper``: a proposal whose
    quantized feasible lower bound strictly improves the broadcast incumbent
    outranks every exploratory proposal; otherwise its valid dual upper bound
    controls exploration.  Only finite-round, quantized column-generation
    results may replace No-op.  N5 uses target-block rebuilds so every move has
    a one/two-target dependency closure; the candidate budget remains B=2 and
    is not enlarged when the fixed-power layer fails.
    """
    selected = np.asarray(initial_selected, dtype=bool)
    coefficient = np.asarray(predicted_coefficient, dtype=np.float64)
    support = np.asarray(candidate_mask, dtype=bool)
    role = np.asarray(initial_role, dtype=np.int8).reshape(-1)
    budget = np.asarray(sensing_budget_w, dtype=np.float64).reshape(-1)
    if (
        selected.shape != coefficient.shape
        or selected.shape != support.shape
        or coefficient.ndim != 3
        or coefficient.shape[0] != coefficient.shape[1]
    ):
        raise ValueError("selected/coefficient/support must share shape (K,K,Q)")
    K, _, Q = coefficient.shape
    if role.shape != (K,) or budget.shape != (K,):
        raise ValueError("role and sensing budget must match K")
    if (
        np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0)
        or np.any(~np.isfinite(budget)) or np.any(budget < 0.0)
    ):
        raise ValueError("gain and budget must be finite and non-negative")
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
    incumbent_power = None
    if incumbent_power_w is not None:
        incumbent_power = np.asarray(incumbent_power_w, dtype=np.float64)
        if incumbent_power.shape != (K, Q):
            raise ValueError("incumbent_power_w must have shape (K,Q)")
    if int(top_m) < 1 or int(max_steps) < 1 or int(weak_target_count) < 1:
        raise ValueError(
            "top_m, max_steps and weak_target_count must be positive")
    price_rounds = int(rounds) if ranking_rounds is None else int(ranking_rounds)
    if int(rounds) < 1 or price_rounds < 1 or price_rounds > int(rounds):
        raise ValueError(
            "ranking_rounds must lie between one and verification rounds")
    mode = str(proxy_mode)
    if mode not in ("feasible_mix", "dual_two_column", "dual_interval"):
        raise ValueError(
            "proxy_mode must be feasible_mix, dual_two_column or dual_interval")
    optimism = float(optimism_weight)
    if not np.isfinite(optimism) or not 0.0 <= optimism <= 1.0:
        raise ValueError("optimism_weight must lie in [0,1]")
    ranking_policy = str(interval_policy)
    if ranking_policy not in ("convex", "certificate_then_upper"):
        raise ValueError(
            "interval_policy must be convex or certificate_then_upper")
    if ranking_policy == "certificate_then_upper" and mode != "dual_interval":
        raise ValueError(
            "certificate_then_upper requires proxy_mode='dual_interval'")
    if ranking_policy == "certificate_then_upper" and (
        candidate_interval_ranker is not None
        or candidate_batch_interval_ranker is not None
    ):
        raise ValueError(
            "certificate_then_upper requires internally certified intervals")
    if bool(owner_proposal_mode) and int(top_m) != 1:
        raise ValueError("owner-proposal arbitration requires Top-1")
    if (
        candidate_interval_ranker is not None
        and candidate_batch_interval_ranker is not None
    ):
        raise ValueError("scalar and batch interval rankers are mutually exclusive")

    baseline_gain, baseline_owner = fixed_owner_gain_matrix(
        coefficient, [tuple(edge) for edge in np.argwhere(selected)])
    baseline = distributed_column_generation_maxmin_power(
        baseline_gain,
        budget,
        rounds=price_rounds,
        price_bits=int(price_bits),
        feedback_bits=int(feedback_bits),
        minimum_deflection=reserve,
        incumbent_power_w=incumbent_power,
    )
    inferred_role, _ = role_owner_from_structure(
        selected, fallback_role=role)
    if not np.array_equal(inferred_role, role):
        raise ValueError("initial_role is inconsistent with selected structure")
    edge_value = coefficient * budget[:, None, None]
    current_selected = selected.copy()
    current_role = role.copy()
    current_owner = baseline_owner.copy()
    current_result = baseline
    accepted_kinds: list[str] = []
    ranked_candidate_moves: list[LocalMove] = []
    verified_moves: list[LocalMove] = []
    accepted_moves: list[LocalMove] = []
    proposal_rounds: list[tuple[RankedOwnerProposal, ...]] = []
    candidate_count = 0
    proposal_count = 0
    verification_count = 0
    proxy_best = 0.0
    for _ in range(int(max_steps)):
        moves = enumerate_local_moves(
            current_selected,
            edge_value,
            support,
            current_role,
            current_owner,
            neighborhoods=("N5", "N6"),
            target_pair_limit=int(target_pair_limit),
            reports_per_receiver=int(reports_per_receiver),
            n5_target_mode="proxy_weak",
            n5_weak_target_count=int(weak_target_count),
            n5_rebuild_scope="target_block",
        )
        ranked: list[
            tuple[
                float, float, int, int, LocalMove, np.ndarray, float, int,
                np.ndarray,
            ]
        ] = []
        pending_batch: list[
            tuple[int, LocalMove, np.ndarray, int, int]
        ] = []
        for index, move in enumerate(moves):
            try:
                gain, _ = fixed_owner_gain_matrix(
                    coefficient,
                    [tuple(edge) for edge in np.argwhere(move.selected)],
                )
            except ValueError:
                continue
            if reserve is not None and np.any(
                np.sum(gain * budget[:, None], axis=0)
                + 1.0e-12 < reserve
            ):
                # This component-wise ceiling relaxes all cross-target RF
                # coupling.  Failing it proves that the local move cannot
                # satisfy the certified target reserve, so it is safe to
                # suppress before proposal transport.
                continue
            edge_changes = int(np.sum(move.selected != current_selected))
            affected_targets = np.flatnonzero(
                np.any(move.selected != current_selected, axis=(0, 1))
                | (move.owner != current_owner)
            )
            if affected_targets.size == 0:
                continue
            proposer = int(np.min(current_owner[affected_targets]))
            if proposer < 0 or proposer >= K:
                continue
            if candidate_batch_interval_ranker is not None:
                pending_batch.append((
                    index, move, gain, edge_changes, proposer))
                continue
            if candidate_interval_ranker is None:
                lower, witness = _candidate_proxy_witness(
                    gain,
                    budget,
                    current_result.power_w,
                    (
                        current_result.prices
                        if mode in (
                            "dual_two_column", "dual_interval") else None
                    ),
                    minimum_deflection=reserve,
                )
                score = lower
                upper = lower
                if mode == "dual_interval":
                    upper = _public_price_dual_upper(
                        gain, budget, current_result.prices)
                    upper = max(upper, lower)
                    if ranking_policy == "certificate_then_upper":
                        score = (
                            lower
                            if lower > current_result.worst_deflection
                            + float(improvement_tolerance)
                            else upper
                        )
                    else:
                        score = (
                            (1.0 - optimism) * lower + optimism * upper
                            if reserve is None or lower > 0.0 else 0.0
                        )
            else:
                witness = current_result.power_w.copy()
                score, lower, upper = (
                    float(value)
                    for value in candidate_interval_ranker(move)
                )
                if (
                    not np.isfinite(score) or not np.isfinite(lower)
                    or not np.isfinite(upper) or lower < 0.0
                    or upper < lower or score < lower - 1.0e-12
                    or score > upper + 1.0e-12
                ):
                    raise ValueError(
                        "external candidate interval must satisfy "
                        "0 <= lower <= score <= upper")
            ranked.append((
                score, lower, -edge_changes, -index, move, gain, upper,
                proposer, witness))
        if candidate_batch_interval_ranker is not None and pending_batch:
            intervals = tuple(candidate_batch_interval_ranker(tuple(
                item[1] for item in pending_batch
            )))
            if len(intervals) != len(pending_batch):
                raise ValueError(
                    "batch interval ranker must return one interval per move")
            for item, interval in zip(pending_batch, intervals):
                index, move, gain, edge_changes, proposer = item
                score, lower, upper = (
                    float(value) for value in interval)
                if (
                    not np.isfinite(score) or not np.isfinite(lower)
                    or not np.isfinite(upper) or lower < 0.0
                    or upper < lower or score < lower - 1.0e-12
                    or score > upper + 1.0e-12
                ):
                    raise ValueError(
                        "external candidate interval must satisfy "
                        "0 <= lower <= score <= upper")
                ranked.append((
                    score, lower, -edge_changes, -index, move, gain, upper,
                    proposer, current_result.power_w.copy()))
        candidate_count += len(ranked)
        ranked_candidate_moves.extend(item[4] for item in ranked)
        def rank_key(item):
            if ranking_policy == "certificate_then_upper":
                return (
                    int(item[1] > current_result.worst_deflection
                        + float(improvement_tolerance)),
                    item[0], item[1], item[2], item[3],
                )
            return item[:4]

        if bool(owner_proposal_mode):
            owner_best: dict[int, tuple[
                float, float, int, int, LocalMove, np.ndarray, float, int,
                np.ndarray,
            ]] = {}
            for item in ranked:
                quantized_lower = quantize_nonnegative_float16_lower(item[1])
                quantized_upper = quantize_nonnegative_float16_upper(item[6])
                if ranking_policy == "certificate_then_upper":
                    quantized_score = (
                        quantized_lower
                        if quantized_lower > current_result.worst_deflection
                        + float(improvement_tolerance)
                        else quantized_upper
                    )
                else:
                    quantized_score = (
                        (1.0 - optimism) * quantized_lower
                        + optimism * quantized_upper
                        if mode == "dual_interval"
                        else quantized_lower
                    )
                quantized = (
                    quantized_score, quantized_lower, item[2], item[3],
                    item[4], item[5], quantized_upper, item[7], item[8],
                )
                previous = owner_best.get(item[7])
                if previous is None or rank_key(quantized) > rank_key(previous):
                    owner_best[item[7]] = quantized
            ranked = list(owner_best.values())
            if len(ranked) > int(weak_target_count):
                raise AssertionError(
                    "owner proposals exceeded the weak-target budget")
            proposal_count += len(ranked)
        ranked.sort(key=rank_key, reverse=True)
        if ranked:
            proxy_best = max(proxy_best, max(float(item[1]) for item in ranked))
        step_verifications = min(len(ranked), int(top_m))
        if bool(owner_proposal_mode) and step_verifications:
            proposal_rounds.append(tuple(
                RankedOwnerProposal(
                    proposer=int(item[7]),
                    move=item[4],
                    lower=float(item[1]),
                    upper=float(item[6]),
                    score=float(item[0]),
                    certified_improvement=bool(
                        ranking_policy == "certificate_then_upper"
                        and item[1] > current_result.worst_deflection
                        + float(improvement_tolerance)
                    ),
                )
                for item in sorted(ranked, key=lambda value: value[7])
            ))
        verification_count += step_verifications
        best_result = current_result
        best_move: LocalMove | None = None
        for _, _, _, _, move, gain, _, _, witness in ranked[:step_verifications]:
            verified_moves.append(move)
            repaired = distributed_column_generation_maxmin_power(
                gain,
                budget,
                rounds=int(rounds),
                price_bits=int(price_bits),
                feedback_bits=int(feedback_bits),
                minimum_deflection=reserve,
                incumbent_power_w=witness,
            )
            if repaired.reserve_feasible and repaired.worst_deflection > (
                best_result.worst_deflection + float(improvement_tolerance)
            ):
                best_result = repaired
                best_move = move
        if best_move is None:
            break
        current_selected = best_move.selected.copy()
        current_role = best_move.role.copy()
        current_owner = best_move.owner.copy()
        current_result = best_result
        accepted_kinds.append(str(best_move.kind))
        accepted_moves.append(best_move)

    final_selected = current_selected
    final_role = current_role
    final_owner = current_owner
    accepted_kind = "+".join(accepted_kinds) if accepted_kinds else "no_op"
    changed = final_selected != selected
    return DualGuidedStructureRepairResult(
        selected=final_selected,
        role=final_role,
        owner=final_owner,
        power_result=current_result,
        accepted=bool(accepted_kinds),
        accepted_kind=accepted_kind,
        accepted_steps=len(accepted_kinds),
        candidate_count=candidate_count,
        exact_verification_count=verification_count,
        owner_proposal_count=proposal_count,
        proxy_best_lower=proxy_best,
        changed_edge_count=int(np.sum(changed)),
        changed_target_count=int(np.sum(np.any(changed, axis=(0, 1)))),
        ranked_candidate_moves=tuple(ranked_candidate_moves),
        verified_moves=tuple(verified_moves),
        accepted_moves=tuple(accepted_moves),
        owner_proposal_rounds=tuple(proposal_rounds),
    )
