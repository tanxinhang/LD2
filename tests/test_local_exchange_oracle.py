import numpy as np

from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    assert_local_feasible,
    enumerate_local_moves,
    oracle_best_improvement,
    objective_strictly_improves,
    ranked_first_improvement,
    role_first_initial_structure,
    structure_objective_key,
)
from uav_isac.coordination.local_move_ranker import local_move_features
from uav_isac.coordination.dynamic_local_search import (
    DynamicLocalSearchCoordinator,
)


def _graph(K=4, Q=3):
    value = np.ones((K, K, Q), dtype=float)
    mask = np.ones_like(value, dtype=bool)
    mask[np.arange(K), np.arange(K), :] = False
    for tx in range(K):
        for rx in range(K):
            value[tx, rx] += 0.2 * tx + 0.1 * rx
    return value, mask


def test_role_first_initial_structure_is_feasible():
    value, mask = _graph()
    selected, role, owner = role_first_initial_structure(
        value, mask, target_pair_limit=2, reports_per_receiver=6)
    assert role.shape == (4,)
    assert owner.shape == (3,)
    assert_local_feasible(
        selected, mask, target_pair_limit=2, reports_per_receiver=6)


def test_oracle_local_search_is_monotonic_and_feasible():
    value, mask = _graph()
    selected, role, _ = role_first_initial_structure(
        value, mask, target_pair_limit=1, reports_per_receiver=6)
    # Deliberately remove support so N1 has a positive add/replace action.
    selected[:, :, 0] = False
    result = oracle_best_improvement(
        selected,
        value,
        mask,
        rounds=5,
        neighborhoods=("N1", "N2", "N3"),
        target_pair_limit=2,
        reports_per_receiver=6,
        initial_role=role,
    )
    keys = result.objective_history
    assert all(right > left for left, right in zip(keys, keys[1:]))
    assert structure_objective_key(result.selected, value) == keys[-1]
    assert_local_feasible(
        result.selected, mask, target_pair_limit=2, reports_per_receiver=6)


def test_small_lns_moves_remain_atomic_and_feasible():
    value, mask = _graph(K=5, Q=4)
    selected, role, _ = role_first_initial_structure(
        value, mask, target_pair_limit=2, reports_per_receiver=8)
    result = oracle_best_improvement(
        selected,
        value,
        mask,
        rounds=2,
        neighborhoods=("N5",),
        target_pair_limit=2,
        reports_per_receiver=8,
        initial_role=role,
    )
    assert_local_feasible(
        result.selected, mask, target_pair_limit=2, reports_per_receiver=8)
    assert all(kind == "N5" for kind in result.accepted_kinds)


def test_all_target_n5_pool_is_a_strict_diagnostic_superset():
    value, mask = _graph(K=5, Q=4)
    selected, role, owner = role_first_initial_structure(
        value, mask, target_pair_limit=2, reports_per_receiver=8)
    proxy_moves = enumerate_local_moves(
        selected,
        value,
        mask,
        role,
        owner,
        neighborhoods=("N5",),
        target_pair_limit=2,
        reports_per_receiver=8,
        n5_target_mode="proxy_weak",
    )
    all_target_moves = enumerate_local_moves(
        selected,
        value,
        mask,
        role,
        owner,
        neighborhoods=("N5",),
        target_pair_limit=2,
        reports_per_receiver=8,
        n5_target_mode="all",
    )
    proxy_structures = {move.selected.tobytes() for move in proxy_moves}
    all_structures = {move.selected.tobytes() for move in all_target_moves}
    assert proxy_structures < all_structures


def test_n5_weak_target_budgets_form_nested_candidate_pools():
    value, mask = _graph(K=5, Q=4)
    selected, role, owner = role_first_initial_structure(
        value, mask, target_pair_limit=2, reports_per_receiver=8)

    pools = []
    for budget in (1, 2, 3, 4):
        moves = enumerate_local_moves(
            selected,
            value,
            mask,
            role,
            owner,
            neighborhoods=("N5",),
            target_pair_limit=2,
            reports_per_receiver=8,
            n5_target_mode="proxy_weak",
            n5_weak_target_count=budget,
        )
        pools.append({move.selected.tobytes() for move in moves})

    assert all(
        left <= right for left, right in zip(pools, pools[1:]))
    assert pools[0] < pools[-1]

    all_target = enumerate_local_moves(
        selected,
        value,
        mask,
        role,
        owner,
        neighborhoods=("N5",),
        target_pair_limit=2,
        reports_per_receiver=8,
        n5_target_mode="all",
    )
    assert pools[-1] == {
        move.selected.tobytes() for move in all_target}


def test_target_block_n5_has_a_true_two_target_dependency_closure():
    K, Q = 4, 3
    value = np.ones((K, K, Q), dtype=float)
    mask = np.ones_like(value, dtype=bool)
    mask[np.arange(K), np.arange(K), :] = False
    role = np.asarray([1, 1, 0, 0], dtype=np.int8)
    owner = np.asarray([2, 2, 2], dtype=np.int64)
    selected = np.zeros_like(mask)
    selected[0, 2, :] = True

    moves = enumerate_local_moves(
        selected,
        value,
        mask,
        role,
        owner,
        neighborhoods=("N5",),
        target_pair_limit=2,
        reports_per_receiver=8,
        n5_target_mode="all",
        n5_rebuild_scope="target_block",
    )

    assert moves
    for move in moves:
        changed_targets = np.flatnonzero(np.any(
            selected ^ move.selected, axis=(0, 1)))
        assert 1 <= len(changed_targets) <= 2
        assert_local_feasible(
            move.selected,
            mask,
            target_pair_limit=2,
            reports_per_receiver=8,
        )


def test_n6_preserves_roles_and_has_a_two_target_dependency_closure():
    K, Q = 4, 3
    value = np.ones((K, K, Q), dtype=float)
    mask = np.ones_like(value, dtype=bool)
    mask[np.arange(K), np.arange(K), :] = False
    role = np.asarray([1, 1, 0, 0], dtype=np.int8)
    owner = np.asarray([2, 2, 2], dtype=np.int64)
    selected = np.zeros_like(mask)
    selected[0, 2, :] = True

    moves = enumerate_local_moves(
        selected,
        value,
        mask,
        role,
        owner,
        neighborhoods=("N6",),
        target_pair_limit=2,
        reports_per_receiver=8,
        n5_target_mode="all",
    )

    assert moves
    for move in moves:
        np.testing.assert_array_equal(move.role, role)
        changed_targets = np.flatnonzero(np.any(
            selected ^ move.selected, axis=(0, 1)))
        assert 1 <= len(changed_targets) <= 2
        assert_local_feasible(
            move.selected,
            mask,
            target_pair_limit=2,
            reports_per_receiver=8,
        )


def test_oracle_ranked_top1_matches_best_improvement():
    value, mask = _graph()
    selected, role, _ = role_first_initial_structure(
        value, mask, target_pair_limit=1, reports_per_receiver=6)
    selected[:, :, 0] = False
    expected = oracle_best_improvement(
        selected,
        value,
        mask,
        rounds=4,
        neighborhoods=("N1", "N2", "N3"),
        target_pair_limit=2,
        reports_per_receiver=6,
        initial_role=role,
    )
    ranked = ranked_first_improvement(
        selected,
        value,
        mask,
        rounds=4,
        neighborhoods=("N1", "N2", "N3"),
        target_pair_limit=2,
        reports_per_receiver=6,
        ranking_method="oracle",
        top_m=1,
        initial_role=role,
    )
    assert np.array_equal(ranked.selected, expected.selected)
    assert ranked.objective_history == expected.objective_history
    assert all(count <= 1 for count in ranked.verification_counts)


def test_scarcity_ranker_is_fail_closed_and_monotonic():
    value, mask = _graph(K=5, Q=4)
    selected, role, _ = role_first_initial_structure(
        value, mask, target_pair_limit=1, reports_per_receiver=8)
    ranked = ranked_first_improvement(
        selected,
        value,
        mask,
        rounds=5,
        neighborhoods=("N1", "N2", "N3"),
        target_pair_limit=2,
        reports_per_receiver=8,
        ranking_method="scarcity",
        top_m=3,
        initial_role=role,
    )
    assert all(
        right > left
        for left, right in zip(
            ranked.objective_history, ranked.objective_history[1:]))
    assert all(count <= 3 for count in ranked.verification_counts)
    assert_local_feasible(
        ranked.selected, mask, target_pair_limit=2, reports_per_receiver=8)


def test_local_move_features_are_permutation_invariant():
    value, mask = _graph(K=5, Q=4)
    selected, role, owner = role_first_initial_structure(
        value, mask, target_pair_limit=2, reports_per_receiver=8)
    move = enumerate_local_moves(
        selected,
        value,
        mask,
        role,
        owner,
        neighborhoods=("N2", "N3"),
        target_pair_limit=2,
        reports_per_receiver=8,
    )[0]
    expected = local_move_features(selected, role, owner, move, value)
    uav_order = np.asarray([3, 0, 4, 1, 2])
    target_order = np.asarray([2, 0, 3, 1])
    inverse_uav = np.argsort(uav_order)

    def permute_pair(pair):
        return pair[uav_order][:, uav_order][:, :, target_order]

    moved_owner = inverse_uav[move.owner[target_order]]
    permuted_move = LocalMove(
        move.kind,
        permute_pair(move.selected),
        move.role[uav_order],
        moved_owner,
    )
    actual = local_move_features(
        permute_pair(selected),
        role[uav_order],
        inverse_uav[owner[target_order]],
        permuted_move,
        permute_pair(value),
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_deterministic_index_tie_is_not_a_positive_gain():
    current = (1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0)
    tie_better = (1.0, 2.0, 3.0, -4.0, -2.0, -2.0, -1.0)
    fewer_edges = (1.0, 2.0, 3.0, -3.0, -9.0, -9.0, -9.0)
    assert not objective_strictly_improves(tie_better, current)
    assert objective_strictly_improves(fewer_edges, current)


def test_learned_ranking_callback_remains_fail_closed():
    value, mask = _graph(K=5, Q=4)
    selected, role, _ = role_first_initial_structure(
        value, mask, target_pair_limit=1, reports_per_receiver=8)

    def scorer(_selected, _role, _owner, moves, _value):
        return np.linspace(0.0, 1.0, num=len(moves), dtype=np.float32)

    ranked = ranked_first_improvement(
        selected,
        value,
        mask,
        rounds=3,
        neighborhoods=("N1", "N2", "N3"),
        target_pair_limit=2,
        reports_per_receiver=8,
        ranking_method="learned",
        ranking_scorer=scorer,
        top_m=3,
        initial_role=role,
    )
    assert all(
        objective_strictly_improves(right, left)
        for left, right in zip(
            ranked.objective_history, ranked.objective_history[1:]))
    assert_local_feasible(
        ranked.selected, mask, target_pair_limit=2, reports_per_receiver=8)


def test_deployment_ranker_only_verifies_bounded_top_m():
    value, mask = _graph(K=5, Q=4)
    selected, role, _ = role_first_initial_structure(
        value, mask, target_pair_limit=1, reports_per_receiver=8)

    def scorer(_selected, _role, _owner, moves, _value):
        return np.arange(len(moves), dtype=np.float32)

    ranked = ranked_first_improvement(
        selected,
        value,
        mask,
        rounds=4,
        neighborhoods=("N1", "N2", "N3"),
        target_pair_limit=2,
        reports_per_receiver=8,
        ranking_method="learned",
        ranking_scorer=scorer,
        top_m=3,
        initial_role=role,
        collect_full_diagnostics=False,
    )
    assert all(count <= 3 for count in ranked.verification_counts)
    assert ranked.primary_regret == ()
    assert ranked.positive_opportunities == ()


def test_dynamic_local_search_preserves_state_and_feasibility():
    value, mask = _graph(K=5, Q=4)
    coordinator = DynamicLocalSearchCoordinator(
        "oracle", cold_rounds=2, warm_rounds=2)
    cold = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
    )
    assert cold.diagnostics["local_search_cold_start"] == 1.0
    warm = coordinator.resolve(
        value * 1.01,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
    )
    assert warm.diagnostics["local_search_cold_start"] == 0.0
    assert_local_feasible(
        warm.selected, mask, target_pair_limit=2, reports_per_receiver=8)


def test_dynamic_local_search_snapshots_exact_last_public_problem():
    value, mask = _graph(K=5, Q=4)
    mask[0, 1, 0] = False
    coordinator = DynamicLocalSearchCoordinator(
        "oracle", cold_rounds=1, warm_rounds=1)
    coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
    )
    problem = coordinator.last_problem()
    assert problem is not None
    saved_value, saved_mask = problem
    np.testing.assert_array_equal(saved_value, value)
    np.testing.assert_array_equal(saved_mask, mask & (value > 0.0))

    restored = DynamicLocalSearchCoordinator(
        "oracle", cold_rounds=1, warm_rounds=1)
    restored.set_state(coordinator.get_state())
    restored_problem = restored.last_problem()
    assert restored_problem is not None
    np.testing.assert_array_equal(restored_problem[0], saved_value)
    np.testing.assert_array_equal(restored_problem[1], saved_mask)


def test_dynamic_local_search_forced_audit_move_is_one_shot():
    value, mask = _graph(K=5, Q=4)
    coordinator = DynamicLocalSearchCoordinator(
        "previous", cold_rounds=1, warm_rounds=0)
    initial = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
    )
    state = coordinator.get_state()
    role = state["role"]
    assert role is not None
    from uav_isac.coordination.local_exchange_oracle import (
        role_owner_from_structure,
    )
    _, owner = role_owner_from_structure(
        initial.selected, fallback_role=role)
    moves = enumerate_local_moves(
        initial.selected,
        value,
        mask,
        role,
        owner,
        neighborhoods=("N5",),
        target_pair_limit=2,
        reports_per_receiver=8,
    )
    assert moves
    coordinator.force_next_move_for_audit(moves[0])
    forced = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
    )
    np.testing.assert_array_equal(forced.selected, moves[0].selected)
    assert forced.diagnostics["local_search_audit_forced_move"] == 1.0
    ordinary = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
    )
    assert "local_search_audit_forced_move" not in ordinary.diagnostics


def test_replicated_candidate_control_is_feasible():
    value, mask = _graph(K=5, Q=4)
    coordinator = DynamicLocalSearchCoordinator("replicated")
    result = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
    )
    assert_local_feasible(
        result.selected, mask, target_pair_limit=2, reports_per_receiver=8)


def test_periodic_rebootstrap_runs_only_on_scheduled_warm_frame():
    value, mask = _graph(K=5, Q=4)
    coordinator = DynamicLocalSearchCoordinator(
        "previous",
        cold_rounds=1,
        rebootstrap_mode="periodic",
        rebootstrap_interval_frames=5,
        rebootstrap_rounds=1,
    )
    cold = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
        frame_index=1,
    )
    assert cold.diagnostics["local_search_rebootstrap_attempted"] == 0.0
    unscheduled = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
        frame_index=3,
    )
    assert unscheduled.diagnostics[
        "local_search_rebootstrap_attempted"] == 0.0
    scheduled = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
        frame_index=5,
    )
    assert scheduled.diagnostics[
        "local_search_rebootstrap_attempted"] == 1.0
    assert_local_feasible(
        scheduled.selected, mask,
        target_pair_limit=2, reports_per_receiver=8)


def test_deficit_gate_blocks_safe_rebootstrap_but_allows_qos_deficit():
    value, mask = _graph(K=5, Q=4)
    coordinator = DynamicLocalSearchCoordinator(
        "previous",
        cold_rounds=1,
        rebootstrap_mode="periodic",
        rebootstrap_interval_frames=5,
        rebootstrap_rounds=1,
        rebootstrap_require_deficit=True,
    )
    coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
        frame_index=1,
    )
    safe = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
        target_deficit=np.full(4, -0.01),
        frame_index=5,
    )
    assert safe.diagnostics[
        "local_search_rebootstrap_attempted"] == 0.0
    assert safe.diagnostics[
        "local_search_rebootstrap_blocked_no_deficit"] == 1.0

    deficit = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
        target_deficit=np.asarray([-0.01, 0.02, -0.03, -0.04]),
        frame_index=10,
    )
    assert deficit.diagnostics[
        "local_search_rebootstrap_attempted"] == 1.0
    assert deficit.diagnostics[
        "local_search_rebootstrap_deficit_present"] == 1.0


def test_oracle_rebootstrap_checks_every_warm_resolve():
    value, mask = _graph(K=5, Q=4)
    coordinator = DynamicLocalSearchCoordinator(
        "previous",
        cold_rounds=1,
        rebootstrap_mode="oracle",
        rebootstrap_rounds=1,
    )
    coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
        frame_index=1,
    )
    warm = coordinator.resolve(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=8,
        frame_index=2,
    )
    assert warm.diagnostics[
        "local_search_rebootstrap_attempted"] == 1.0
