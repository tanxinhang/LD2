import numpy as np

from config.params import load_config
from uav_isac.coordination.qpd import (
    local_primal_dual_update,
    project_capped_simplex,
    update_virtual_queue,
)
from uav_isac.environment.env_wrapper import UAVISACEnv


def test_qpd_queue_has_correct_deficit_sign():
    queue = np.array([0.2, 0.2])
    updated = update_virtual_queue(
        queue,
        np.array([0.1, 0.9]),
        qos_floor=0.6,
        step_size=0.25,
        queue_max=4.0,
    )
    assert updated[0] > queue[0]
    assert updated[1] < queue[1]


def test_capped_simplex_projection_enforces_inequality_budget():
    below = project_capped_simplex(np.array([0.1, 0.2]), capacity=1.0)
    np.testing.assert_allclose(below, np.array([0.1, 0.2]))

    projected = project_capped_simplex(
        np.array([2.0, 1.0, 0.5]), capacity=1.25)
    assert np.all(projected >= 0.0)
    assert np.all(projected <= 1.0)
    assert np.isclose(np.sum(projected), 1.25)


def test_local_qpd_update_is_target_permutation_equivariant():
    bid = np.array([0.4, -0.2, 0.1])
    peer = np.array([0.2, 1.3, 0.4])
    primal = np.array([0.5, 0.3, 0.2])
    price = np.array([0.1, -0.1, 0.0])
    kwargs = dict(
        row_capacity=1.5,
        target_capacity=1.0,
        primal_step=0.8,
        dual_step=0.2,
        rounds=2,
        price_max=4.0,
    )
    result = local_primal_dual_update(
        bid, peer, primal, price, **kwargs)
    permutation = np.array([2, 0, 1])
    permuted = local_primal_dual_update(
        bid[permutation],
        peer[permutation],
        primal[permutation],
        price[permutation],
        **kwargs,
    )
    np.testing.assert_allclose(
        permuted.primal, result.primal[permutation])
    np.testing.assert_allclose(
        permuted.target_price, result.target_price[permutation])


def _qpd_env(seed=31):
    cfg = load_config(
        'config/exp_800_q4_architecture_v2_qpd_gate.yaml')
    cfg.scenario.T = 2
    env = UAVISACEnv(cfg, seed=seed)
    env.reset(seed=seed)
    return env


def test_qpd_received_packet_changes_only_receivers_local_row():
    control = _qpd_env()
    treatment = _qpd_env()
    for core in (control.core, treatment.core):
        core._qpd_bid[:] = np.array([0.3, 0.1, -0.1, -0.2])
        core._qpd_target_price[:] = 0.0

    token_dim = treatment.core._comm_target_token_dim
    payload = np.zeros((treatment.core.Q, token_dim))
    payload[:, 3] = -1.0  # decoded peer primal = zero
    payload[0, 3] = 1.0   # decoded peer primal for target 0 = one
    treatment.core._merge_received_qpd_protocol(
        0, 1, payload.reshape(-1), np.ones(treatment.core.Q))

    control_mask = control.core._resolve_qpd_commitments()
    treatment_mask = treatment.core._resolve_qpd_commitments()

    assert not np.allclose(
        treatment.core._qpd_primal[0],
        control.core._qpd_primal[0],
    )
    np.testing.assert_allclose(
        treatment.core._qpd_primal[1:],
        control.core._qpd_primal[1:],
    )
    np.testing.assert_array_equal(
        treatment_mask[1:], control_mask[1:])


def test_qpd_protocol_uses_physical_tokens_and_keeps_exact_power_budget():
    env = _qpd_env(seed=37)
    core = env.core
    K, Q = core.K, core.Q
    token_dim = core._comm_target_token_dim
    messages = {k: np.zeros(Q * token_dim) for k in range(K)}
    rates = {k: 1 for k in range(K)}
    fractions = {k: 0.2 for k in range(K)}
    weights = {k: np.full(Q, 1.0 / Q) for k in range(K)}

    core.submit_learned_communications(
        messages, rates, fractions, weights,
        token_masks={k: np.ones(Q) for k in range(K)},
    )
    assert any(np.any(mask > 0.5)
               for mask in core._pending_comm_token_masks.values())
    sent_masks = {
        k: mask.copy()
        for k, mask in core._pending_comm_token_masks.items()}
    protocol = core._pending_qpd_protocol[0]
    assert protocol.shape == (Q, 5)
    assert np.any(np.abs(protocol) > 0.0)

    actions = {
        str(k): {'delta_p': np.zeros(2), 'role': 2}
        for k in range(K)
    }
    _, _, _, _, info = env.step(actions)
    expected_bits = sum(
        64 + 4 * int(np.sum(mask > 0.5)) * (token_dim + 5)
        for mask in sent_masks.values()
        if np.any(mask > 0.5)
    )
    assert info['learned_comm_bits'] == expected_bits
    combined = (
        info['isac_per_uav_comm_power_w']
        + info['isac_per_uav_sensing_power_w']
    )
    assert np.all(combined <= env.cfg.uav.P_isac_total + 1e-12)
    assert np.all(
        info['isac_per_uav_sensing_power_w']
        <= env.cfg.uav.P_sense_max + 1e-12)
    assert info['isac_max_power_budget_violation_w'] < 1e-12
    assert info['qpd_enabled'] == 1.0


def test_qpd_snapshot_restore_is_exact():
    env = _qpd_env(seed=41)
    core = env.core
    core._qpd_queue[:] = np.arange(core.K * core.Q).reshape(
        core.K, core.Q) / 10.0
    core._qpd_primal[:] = 0.37
    core._qpd_target_price[:] = -0.2
    snapshot = core.get_state()

    core._qpd_queue[:] = 99.0
    core._qpd_primal[:] = 0.0
    core._qpd_target_price[:] = 3.0
    core.set_state(snapshot)

    np.testing.assert_allclose(
        core._qpd_queue, snapshot['qpd_queue'])
    np.testing.assert_allclose(
        core._qpd_primal, snapshot['qpd_primal'])
    np.testing.assert_allclose(
        core._qpd_target_price, snapshot['qpd_target_price'])
