import numpy as np

from config.params import load_config
from uav_isac.coordination.hyperedge import (
    decode_offer_stream,
    encode_offer_stream,
    mutual_endpoint_consensus,
    plan_local_hyperedges,
    update_consensus_streak,
)
from uav_isac.environment.env_wrapper import UAVISACEnv


def _plan(tx, rx, visible, deficit):
    return plan_local_hyperedges(
        tx, rx, visible, deficit,
        target_pair_limit=1,
        deficit_gain=2.0,
        proxy_floor=0.25,
    )


def test_offer_stream_round_trip_is_bounded():
    tx = np.array([0.1, 0.7, 1.2])
    rx = np.array([0.9, 0.2, -0.1])
    gap = np.array([0.0, 0.4, 1.0])
    decoded = decode_offer_stream(encode_offer_stream(tx, rx, gap))
    np.testing.assert_allclose(decoded[:, 0], np.clip(tx, 0.0, 1.0))
    np.testing.assert_allclose(decoded[:, 1], np.clip(rx, 0.0, 1.0))
    np.testing.assert_allclose(decoded[:, 2], np.clip(gap, 0.0, 1.0))


def test_local_plan_enforces_directed_single_roles_and_target_capacity():
    tx = np.array([
        [0.95, 0.90],
        [0.85, 0.80],
        [0.20, 0.15],
        [0.10, 0.25],
    ])
    rx = np.array([
        [0.10, 0.20],
        [0.15, 0.10],
        [0.90, 0.85],
        [0.80, 0.95],
    ])
    plan = _plan(tx, rx, np.ones_like(tx, dtype=bool), np.array([0.6, 0.4]))
    assert len(plan.selected) == 2
    tx_nodes = {edge[0] for edge in plan.selected}
    rx_nodes = {edge[1] for edge in plan.selected}
    assert not (tx_nodes & rx_nodes)
    assert {edge[2] for edge in plan.selected} == {0, 1}


def test_local_plan_is_equivariant_without_score_ties():
    tx = np.array([
        [0.91, 0.62],
        [0.73, 0.81],
        [0.22, 0.31],
        [0.16, 0.27],
    ])
    rx = np.array([
        [0.18, 0.21],
        [0.25, 0.17],
        [0.88, 0.74],
        [0.69, 0.93],
    ])
    visible = np.ones_like(tx, dtype=bool)
    deficit = np.array([0.7, 0.3])
    base = _plan(tx, rx, visible, deficit)

    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    permuted = _plan(
        tx[permutation], rx[permutation], visible[permutation], deficit)
    mapped = tuple(sorted(
        (int(permutation[edge[0]]), int(permutation[edge[1]]), edge[2])
        for edge in permuted.selected
    ))
    # ``permutation[new] = old`` maps the permuted node index back to the
    # original index. The inverse is deliberately kept to catch map reversal.
    assert inverse.shape == permutation.shape
    assert mapped == base.selected


def test_reconstructable_pair_value_overrides_scalar_endpoint_proxy():
    tx = np.array([[0.9], [0.8], [0.1]])
    rx = np.array([[0.1], [0.2], [0.9]])
    visible = np.ones_like(tx, dtype=bool)
    pair_value = np.zeros((3, 3, 1))
    pair_value[1, 2, 0] = 9.0
    pair_value[0, 2, 0] = 1.0
    plan = plan_local_hyperedges(
        tx, rx, visible, np.array([0.8]),
        target_pair_limit=1,
        deficit_gain=2.0,
        proxy_floor=0.25,
        pair_value=pair_value,
    )
    assert plan.selected == ((1, 2, 0),)


def test_local_plan_uses_one_receiver_owner_per_target():
    tx = np.array([
        [0.95],
        [0.90],
        [0.20],
        [0.15],
    ])
    rx = np.array([
        [0.10],
        [0.15],
        [0.92],
        [0.88],
    ])
    plan = plan_local_hyperedges(
        tx,
        rx,
        np.ones_like(tx, dtype=bool),
        np.array([0.8]),
        target_pair_limit=2,
        deficit_gain=2.0,
        proxy_floor=0.25,
    )
    assert len(plan.selected) == 2
    assert len({receiver for _, receiver, _ in plan.selected}) == 1


def test_mutual_consensus_requires_both_endpoints():
    tx = np.array([
        [0.95],
        [0.10],
        [0.20],
    ])
    rx = np.array([
        [0.10],
        [0.90],
        [0.50],
    ])
    full = np.ones_like(tx, dtype=bool)
    plans = [_plan(tx, rx, full, np.array([0.8])) for _ in range(3)]
    mutual = mutual_endpoint_consensus(
        plans, num_uavs=3, num_targets=1, target_pair_limit=1)
    assert mutual == plans[0].selected

    # Receiver 1 loses sender 0's offer. Its local plan no longer endorses the
    # same directed edge, so the edge must fail closed.
    lost = full.copy()
    lost[0, 0] = False
    plans[1] = _plan(tx, rx, lost, np.array([0.8]))
    mutual = mutual_endpoint_consensus(
        plans, num_uavs=3, num_targets=1, target_pair_limit=1)
    assert mutual == tuple()


def test_consensus_streak_requires_configured_rounds():
    previous = np.zeros((3, 3, 2), dtype=np.int64)
    edge = (0, 1, 0)
    first, active_first = update_consensus_streak(
        previous, [edge], consensus_rounds=2)
    assert active_first == tuple()
    second, active_second = update_consensus_streak(
        first, [edge], consensus_rounds=2)
    assert active_second == (edge,)
    reset, active_reset = update_consensus_streak(
        second, [], consensus_rounds=2)
    assert active_reset == tuple()
    assert np.max(reset) == 0


def test_physical_hyperedge_stream_is_charged_and_uses_two_rounds():
    cfg = load_config(
        'config/exp_800_q4_architecture_v2_hyperedge_gate.yaml')
    cfg.scenario.T = 3
    env = UAVISACEnv(cfg, seed=37)
    env.reset(seed=37)
    core = env.core
    K, Q = core.K, core.Q
    token_dim = core._comm_target_token_dim
    messages = {k: np.zeros(Q * token_dim) for k in range(K)}
    rates = {k: 1 for k in range(K)}
    fractions = {k: 0.2 for k in range(K)}
    weights = {k: np.full(Q, 1.0 / Q) for k in range(K)}
    actions = {
        str(k): {'delta_p': np.zeros(2), 'role': 2}
        for k in range(K)
    }

    protocol_used = []
    for _ in range(2):
        core.submit_learned_communications(
            messages, rates, fractions, weights,
            token_masks={k: np.ones(Q) for k in range(K)},
        )
        assert all(
            protocol.shape == (Q, 3)
            for protocol in core._pending_hyperedge_protocol.values())
        _, _, _, _, info = env.step(actions)
        protocol_used.append(info['hyperedge_protocol_used'])
        expected_per_sender = 64 + 4 * Q * (token_dim + 3)
        assert info['learned_comm_bits'] == K * expected_per_sender
        combined = (
            info['isac_per_uav_comm_power_w']
            + info['isac_per_uav_sensing_power_w']
        )
        np.testing.assert_allclose(
            combined, env.cfg.uav.P_isac_total, atol=1e-12)
        assert info['isac_max_power_balance_error_w'] < 1e-12

    assert protocol_used == [0.0, 1.0]


def test_hyperedge_snapshot_restore_is_exact():
    cfg = load_config(
        'config/exp_800_q4_architecture_v2_hyperedge_gate.yaml')
    cfg.scenario.T = 2
    env = UAVISACEnv(cfg, seed=41)
    env.reset(seed=41)
    core = env.core
    core._hyperedge_local_offer[:] = 0.37
    core._hyperedge_received_offer[:] = 0.21
    core._hyperedge_received_last_seen[:] = 3
    core._hyperedge_consensus_streak[0, 1, 0] = 2
    core._hyperedge_selected_set = ((0, 1, 0),)
    snapshot = core.get_state()

    core._hyperedge_local_offer[:] = 0.0
    core._hyperedge_received_offer[:] = 0.0
    core._hyperedge_received_last_seen[:] = -99
    core._hyperedge_consensus_streak[:] = 0
    core._hyperedge_selected_set = tuple()
    core.set_state(snapshot)

    np.testing.assert_allclose(
        core._hyperedge_local_offer,
        snapshot['hyperedge_local_offer'])
    np.testing.assert_allclose(
        core._hyperedge_received_offer,
        snapshot['hyperedge_received_offer'])
    np.testing.assert_array_equal(
        core._hyperedge_received_last_seen,
        snapshot['hyperedge_received_last_seen'])
    np.testing.assert_array_equal(
        core._hyperedge_consensus_streak,
        snapshot['hyperedge_consensus_streak'])
    assert core._hyperedge_selected_set == ((0, 1, 0),)
