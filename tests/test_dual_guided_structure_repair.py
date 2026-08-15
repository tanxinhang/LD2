import numpy as np

from uav_isac.coordination.dual_guided_structure_repair import (
    _common_mix_lower,
    _common_mix_power_witness,
    _common_mix_reserve_lower,
    _dual_seeded_two_column_lower,
    _dual_seeded_two_column_witness,
    _line_segment_maxmin_lower,
    _line_segment_reserve_lower,
    _public_price_dual_upper,
    dual_guided_atomic_structure_repair,
    enumerate_dual_guided_atomic_candidates,
)


def test_two_column_lower_solves_piecewise_linear_maxmin_exactly():
    first = np.asarray([0.2, 0.8])
    second = np.asarray([0.8, 0.2])
    assert np.isclose(_line_segment_maxmin_lower(first, second), 0.5)


def test_dual_seeded_proxy_is_a_feasible_convex_mixture():
    gain = np.asarray([[2.0, 0.1], [0.1, 2.0]])
    budget = np.ones(2)
    incumbent = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    lower = _dual_seeded_two_column_lower(
        gain, budget, incumbent, np.asarray([0.5, 0.5]))
    assert np.isclose(lower, 2.0)


def test_public_price_value_upper_bounds_a_feasible_maxmin_point():
    gain = np.asarray([[4.0, 1.0], [2.0, 3.0]])
    budget = np.asarray([0.8, 0.6])
    prices = np.asarray([0.4, 0.6])
    upper = _public_price_dual_upper(gain, budget, prices)
    feasible_power = budget[:, None] * np.asarray([[0.5, 0.5]])
    feasible_worst = float(np.min(np.sum(gain * feasible_power, axis=0)))
    assert feasible_worst <= upper + 1.0e-12


def test_common_mix_value_is_achievable_by_shared_time_weights():
    gain = np.asarray([[4.0, 1.0], [2.0, 3.0]])
    budget = np.asarray([0.8, 0.6])
    capacity = np.sum(gain * budget[:, None], axis=0)
    weights = (1.0 / capacity) / np.sum(1.0 / capacity)
    power = budget[:, None] * weights[None, :]
    achieved = np.sum(gain * power, axis=0)
    np.testing.assert_allclose(
        achieved, np.full(2, _common_mix_lower(gain, budget)))


def test_common_mix_reserve_waterfill_is_feasible_and_tight():
    gain = np.asarray([[2.0, 1.0]])
    reserve = np.asarray([0.9, 0.4])
    lower = _common_mix_reserve_lower(gain, np.ones(1), reserve)
    assert np.isclose(lower, 0.55)
    weights = np.maximum(reserve, lower) / np.asarray([2.0, 1.0])
    assert np.sum(weights) <= 1.0 + 1.0e-12


def test_common_mix_witness_attains_reported_reserve_lower():
    gain = np.asarray([[2.0, 1.0], [1.0, 3.0]])
    budget = np.asarray([0.8, 0.6])
    reserve = np.asarray([0.7, 0.5])
    lower, power = _common_mix_power_witness(
        gain, budget, minimum_deflection=reserve)
    deflection = np.sum(gain * power, axis=0)
    assert np.all(deflection + 1e-10 >= reserve)
    assert np.isclose(lower, np.min(deflection))
    np.testing.assert_allclose(np.sum(power, axis=1), budget)


def test_dual_seeded_witness_attains_reported_lower():
    gain = np.asarray([[2.0, 0.1], [0.1, 2.0]])
    budget = np.ones(2)
    incumbent = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    lower, power = _dual_seeded_two_column_witness(
        gain, budget, incumbent, np.asarray([0.5, 0.5]))
    assert np.isclose(lower, np.min(np.sum(gain * power, axis=0)))
    np.testing.assert_allclose(np.sum(power, axis=1), budget)


def test_two_column_reserve_proxy_rejects_infeasible_segment():
    assert _line_segment_reserve_lower(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.8, 0.8]),
    ) == 0.0


def test_atomic_owner_exchange_is_verified_before_acceptance():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    support = np.zeros_like(coefficient, dtype=bool)
    coefficient[0, 2, 0] = 10.0
    coefficient[1, 2, 1] = 1.0
    coefficient[1, 3, 1] = 10.0
    support[0, 2, 0] = True
    support[1, 2, 1] = True
    support[1, 3, 1] = True
    selected = np.zeros_like(support)
    selected[0, 2, 0] = True
    selected[1, 2, 1] = True
    result = dual_guided_atomic_structure_repair(
        selected,
        coefficient,
        support,
        np.asarray([1, 1, 0, 0], dtype=np.int8),
        np.ones(4),
        target_pair_limit=1,
        reports_per_receiver=4,
        top_m=8,
    )
    assert result.accepted
    assert result.accepted_steps == 1
    assert result.owner[1] == 3
    assert result.changed_target_count <= 2
    assert result.power_result.worst_deflection > 1.0


def test_no_op_remains_explicit_when_atomic_moves_do_not_help():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    support = np.zeros_like(coefficient, dtype=bool)
    coefficient[0, 2, 0] = 4.0
    coefficient[1, 2, 1] = 4.0
    coefficient[1, 3, 1] = 1.0
    support[0, 2, 0] = True
    support[1, 2, 1] = True
    support[1, 3, 1] = True
    selected = np.zeros_like(support)
    selected[0, 2, 0] = True
    selected[1, 2, 1] = True
    result = dual_guided_atomic_structure_repair(
        selected,
        coefficient,
        support,
        np.asarray([1, 1, 0, 0], dtype=np.int8),
        np.ones(4),
        target_pair_limit=1,
        reports_per_receiver=4,
        top_m=8,
    )
    assert not result.accepted
    assert result.accepted_kind == "no_op"
    assert result.accepted_steps == 0
    np.testing.assert_array_equal(result.selected, selected)


def test_external_interval_ranker_controls_only_bounded_preselection():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    support = np.zeros_like(coefficient, dtype=bool)
    for edge, value in {
        (0, 2, 0): 10.0,
        (0, 3, 0): 8.0,
        (1, 2, 1): 1.0,
        (1, 3, 1): 10.0,
    }.items():
        coefficient[edge] = value
        support[edge] = True
    selected = np.zeros_like(support)
    selected[0, 2, 0] = True
    selected[1, 2, 1] = True
    scored = []

    def ranker(move):
        preferred = bool(move.owner[0] == 3)
        score = 2.0 if preferred else 1.0
        scored.append((preferred, move))
        return score, score, score

    result = dual_guided_atomic_structure_repair(
        selected,
        coefficient,
        support,
        np.asarray([1, 1, 0, 0], dtype=np.int8),
        np.ones(4),
        target_pair_limit=1,
        reports_per_receiver=4,
        top_m=1,
        max_steps=1,
        candidate_interval_ranker=ranker,
    )

    assert any(preferred for preferred, _ in scored)
    assert len(result.verified_moves) == 1
    assert result.verified_moves[0].owner[0] == 3


def test_batch_interval_ranker_scores_all_eligible_moves_once():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    support = np.zeros_like(coefficient, dtype=bool)
    for edge, value in {
        (0, 2, 0): 10.0,
        (0, 3, 0): 8.0,
        (1, 2, 1): 1.0,
        (1, 3, 1): 10.0,
    }.items():
        coefficient[edge] = value
        support[edge] = True
    selected = np.zeros_like(support)
    selected[0, 2, 0] = True
    selected[1, 2, 1] = True
    calls = []

    def ranker(moves):
        calls.append(moves)
        return tuple(
            (2.0, 2.0, 2.0)
            if move.owner[0] == 3 else (1.0, 1.0, 1.0)
            for move in moves
        )

    result = dual_guided_atomic_structure_repair(
        selected,
        coefficient,
        support,
        np.asarray([1, 1, 0, 0], dtype=np.int8),
        np.ones(4),
        target_pair_limit=1,
        reports_per_receiver=4,
        top_m=1,
        max_steps=1,
        candidate_batch_interval_ranker=ranker,
    )

    assert len(calls) == 1
    assert len(calls[0]) == result.candidate_count
    assert len(result.ranked_candidate_moves) == result.candidate_count
    assert result.verified_moves[0].owner[0] == 3

    candidate_set = enumerate_dual_guided_atomic_candidates(
        selected,
        coefficient,
        support,
        np.asarray([1, 1, 0, 0], dtype=np.int8),
        np.ones(4),
        target_pair_limit=1,
        reports_per_receiver=4,
    )
    assert len(candidate_set.moves) == result.candidate_count
    for generated, ranked in zip(
        candidate_set.moves, result.ranked_candidate_moves
    ):
        np.testing.assert_array_equal(generated.selected, ranked.selected)
        np.testing.assert_array_equal(generated.role, ranked.role)
        np.testing.assert_array_equal(generated.owner, ranked.owner)


def test_certificate_then_upper_policy_requires_internal_dual_intervals():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    support = np.zeros_like(coefficient, dtype=bool)
    coefficient[0, 2, 0] = 4.0
    coefficient[1, 3, 1] = 4.0
    support[0, 2, 0] = True
    support[1, 3, 1] = True
    selected = support.copy()

    with np.testing.assert_raises_regex(
        ValueError, "requires proxy_mode='dual_interval'"
    ):
        dual_guided_atomic_structure_repair(
            selected,
            coefficient,
            support,
            np.asarray([1, 1, 0, 0], dtype=np.int8),
            np.ones(4),
            target_pair_limit=1,
            reports_per_receiver=4,
            interval_policy="certificate_then_upper",
        )


def test_ranking_rounds_cannot_exceed_verification_rounds():
    coefficient = np.zeros((4, 4, 2), dtype=np.float64)
    coefficient[0, 2, 0] = 4.0
    coefficient[1, 3, 1] = 4.0
    selected = coefficient > 0.0
    with np.testing.assert_raises_regex(ValueError, "ranking_rounds"):
        dual_guided_atomic_structure_repair(
            selected,
            coefficient,
            selected,
            np.asarray([1, 1, 0, 0], dtype=np.int8),
            np.ones(4),
            target_pair_limit=1,
            reports_per_receiver=4,
            rounds=4,
            ranking_rounds=5,
        )
