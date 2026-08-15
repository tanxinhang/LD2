import numpy as np

from uav_isac.coordination.finite_round_hyperedge import (
    gossip_candidate_views,
    initial_finite_round_state,
    negotiate_finite_round_hyperedges,
    solve_replicated_candidate_graph,
    solve_replicated_candidate_graph_certificate,
)


def _kwargs(rounds=3):
    return dict(
        rounds=rounds,
        target_pair_limit=2,
        owner_hold_rounds=2,
        target_price_step=0.2,
        role_price_step=0.1,
        target_proxy_floor=0.8,
        price_max=4.0,
        communication_cost=0.0,
        switch_cost=0.05,
    )


def test_finite_round_protocol_fails_closed_during_cold_start():
    state = initial_finite_round_state(4, 3)
    result = negotiate_finite_round_hyperedges(
        np.ones((4, 4, 3)),
        np.zeros((4, 4, 3), dtype=bool),
        state,
        **_kwargs(),
    )
    assert result.selected == tuple()
    assert result.bootstrap_required
    assert not result.state.initialized


def test_finite_round_protocol_enforces_roles_owner_and_capacity():
    K, Q = 4, 2
    value = np.zeros((K, K, Q), dtype=np.float64)
    mask = np.zeros_like(value, dtype=bool)
    # Both targets favor receiver 3, with transmitters 0 and 1.
    for target in range(Q):
        value[0, 3, target] = 1.0
        value[1, 3, target] = 0.8
        value[3, 0, target] = 0.4  # creates a role-conflict alternative
        mask[:, :, target] = value[:, :, target] > 0.0
    result = negotiate_finite_round_hyperedges(
        value,
        mask,
        initial_finite_round_state(K, Q),
        **_kwargs(),
    )
    tx_nodes = {edge[0] for edge in result.selected}
    rx_nodes = {edge[1] for edge in result.selected}
    assert not (tx_nodes & rx_nodes)
    for target in range(Q):
        edges = [edge for edge in result.selected if edge[2] == target]
        assert len(edges) <= 2
        assert len({edge[1] for edge in edges}) <= 1
    assert result.state.initialized


def test_starved_target_price_rises_while_served_target_price_falls():
    K, Q = 3, 2
    value = np.zeros((K, K, Q), dtype=np.float64)
    mask = np.zeros_like(value, dtype=bool)
    value[0, 2, 0] = 1.0
    mask[0, 2, 0] = True
    state = initial_finite_round_state(K, Q)
    result = negotiate_finite_round_hyperedges(
        value,
        mask,
        state,
        **_kwargs(rounds=1),
    )
    assert result.state.target_price[0] < state.target_price[0]
    assert result.state.target_price[1] > state.target_price[1]


def test_round_protocol_is_deterministic():
    rng = np.random.default_rng(17)
    value = rng.uniform(0.1, 1.0, size=(5, 5, 3))
    mask = np.ones_like(value, dtype=bool)
    diagonal = np.arange(5)
    mask[diagonal, diagonal] = False
    state = initial_finite_round_state(5, 3)
    first = negotiate_finite_round_hyperedges(
        value, mask, state, **_kwargs(rounds=4))
    second = negotiate_finite_round_hyperedges(
        value, mask, state, **_kwargs(rounds=4))
    assert first.selected == second.selected
    np.testing.assert_array_equal(
        first.state.target_price, second.state.target_price)


def test_replicated_graph_solver_enforces_all_hard_invariants():
    rng = np.random.default_rng(23)
    K, Q = 5, 4
    value = rng.uniform(0.1, 1.0, size=(K, K, Q))
    mask = np.ones_like(value, dtype=bool)
    diagonal = np.arange(K)
    mask[diagonal, diagonal] = False
    selected = solve_replicated_candidate_graph(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=3,
        target_price=np.ones(Q),
    )
    tx_nodes = {edge[0] for edge in selected}
    rx_nodes = {edge[1] for edge in selected}
    assert not (tx_nodes & rx_nodes)
    for target in range(Q):
        edges = [edge for edge in selected if edge[2] == target]
        assert len(edges) <= 2
        assert len({edge[1] for edge in edges}) <= 1
    for receiver in range(K):
        assert sum(edge[1] == receiver for edge in selected) <= 3


def test_candidate_gossip_reaches_common_view_on_line_graph():
    mask = np.zeros((4, 4, 1), dtype=bool)
    mask[0, 1, 0] = True
    mask[1, 2, 0] = True
    mask[2, 3, 0] = True
    one_round, common_one, _ = gossip_candidate_views(mask, rounds=1)
    assert not common_one
    assert not one_round[0, 2, 3, 0]
    two_round, common_two, agreement = gossip_candidate_views(mask, rounds=2)
    assert common_two
    assert agreement == 1.0
    assert np.all(two_round == mask)


def test_replicated_certificate_exposes_joint_feasible_alternatives():
    value = np.zeros((4, 4, 3), dtype=float)
    mask = np.zeros_like(value, dtype=bool)
    for tx in range(4):
        for rx in range(4):
            if tx == rx:
                continue
            mask[tx, rx, :] = True
            value[tx, rx, :] = 1.0 + 0.1 * tx + 0.01 * rx
    certificate = solve_replicated_candidate_graph_certificate(
        value,
        mask,
        target_pair_limit=2,
        reports_per_receiver=6,
        max_alternatives=4,
        min_worst_ratio=0.8,
        min_sum_ratio=0.8,
    )
    assert certificate.selected == certificate.alternatives[0].selected
    assert certificate.role.shape == (4,)
    assert certificate.owner.shape == (3,)
    assert certificate.target_value.shape == (3,)
    assert 1 <= len(certificate.alternatives) <= 4
    previous_worst = float("inf")
    for alternative in certificate.alternatives:
        pairs = np.zeros_like(mask)
        for edge in alternative.selected:
            pairs[edge] = True
            assert alternative.role[edge[0]] == 1
            assert alternative.role[edge[1]] == 0
            assert alternative.owner[edge[2]] == edge[1]
        tx = np.any(pairs, axis=(1, 2))
        rx = np.any(pairs, axis=(0, 2))
        assert not np.any(tx & rx)
        assert alternative.weighted_worst <= previous_worst + 1.0e-12
        previous_worst = alternative.weighted_worst
