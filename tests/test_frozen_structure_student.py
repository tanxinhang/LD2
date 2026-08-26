import numpy as np
import torch

from uav_isac.agents.frozen_structure_student import (
    CardinalityResidualStructureStudent,
    FactorizedStructureStudent,
    FrozenStructureStudent,
    build_structure_student_features,
    save_cardinality_residual_structure_student,
    save_frozen_structure_student,
)
from uav_isac.environment.observation_slices import ObservationSlices
from config.params import get_default_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def _slices() -> ObservationSlices:
    return ObservationSlices.from_config(
        K=4,
        Q=4,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=True,
        comm_token_dim=21,
        comm_tokens_per_sender=4,
    )


def test_structure_student_feature_builder_accepts_single_frame():
    slices = _slices()
    rng = np.random.default_rng(7)
    features = build_structure_student_features(
        rng.normal(size=(4, slices.total_dim)),
        slices,
        outgoing_message=rng.normal(size=(4, 64)),
        outgoing_token_mask=np.ones((4, 4)),
        outgoing_rate=np.array([0, 1, 2, 1]),
        comm_fraction=np.full(4, 0.25),
        sensing_weights=np.full((4, 4), 0.25),
        rate_scale=2.0,
    )
    assert features.shape[:3] == (1, 4, 4)
    assert np.all(np.isfinite(features))


def test_structure_student_local_neighbor_subset_preserves_default_all_peers():
    slices = _slices()
    K, Q = slices.K, slices.Q
    obs = np.zeros((1, K, slices.total_dim), dtype=np.float32)
    obs[..., slices.comm_mask_start:slices.comm_mask_start
        + slices.comm_mask_len] = 1.0
    rng = np.random.default_rng(31)
    token_length = slices.comm_mask_start - slices.comm_token_start
    obs[..., slices.comm_token_start:slices.comm_mask_start] = rng.normal(
        size=(1, K, token_length))
    kwargs = {
        "outgoing_message": np.zeros((1, K, Q), dtype=np.float32),
        "outgoing_token_mask": np.ones((1, K, Q), dtype=np.float32),
        "outgoing_rate": np.zeros((1, K), dtype=np.float32),
        "comm_fraction": np.zeros((1, K), dtype=np.float32),
        "sensing_weights": np.full((1, K, Q), 1.0 / Q, dtype=np.float32),
        "rate_scale": 1.0,
    }
    default = build_structure_student_features(obs, slices, **kwargs)
    all_peers = np.ones((1, K, K), dtype=bool)
    for k in range(K):
        all_peers[:, k, k] = False
    explicit = build_structure_student_features(
        obs, slices, neighbor_subset_mask=all_peers, **kwargs)
    np.testing.assert_array_equal(default, explicit)

    one_peer = np.zeros((1, K, K), dtype=bool)
    for k in range(K):
        one_peer[:, k, (k + 1) % K] = True
    local = build_structure_student_features(
        obs, slices, neighbor_subset_mask=one_peer, **kwargs)
    assert not np.array_equal(default, local)


def test_factorized_student_is_agent_and_target_permutation_equivariant():
    torch.manual_seed(11)
    model = FactorizedStructureStudent(input_dim=9)
    features = torch.randn(2, 4, 4, 9)
    base = model(features)
    agent_order = torch.tensor([2, 0, 3, 1])
    target_order = torch.tensor([3, 1, 0, 2])
    permuted = model(
        features[:, agent_order][:, :, target_order])
    expected = base[
        :, agent_order][:, :, agent_order][:, :, :, target_order]
    assert torch.allclose(permuted, expected, atol=1.0e-6)


def test_frozen_structure_student_artifact_roundtrip(tmp_path):
    torch.manual_seed(19)
    model = FactorizedStructureStudent(input_dim=5)
    features = np.random.default_rng(19).normal(
        size=(3, 4, 4, 5)).astype(np.float32)
    mean = features.reshape(-1, 5).mean(axis=0, keepdims=True)
    scale = np.maximum(
        features.reshape(-1, 5).std(axis=0, keepdims=True), 1.0e-5)
    checkpoint = tmp_path / "student.pt"
    save_frozen_structure_student(
        checkpoint,
        model,
        mean,
        scale,
        num_uavs=4,
        num_targets=4,
        rate_scale=2.0,
        metadata={"test": True},
    )
    loaded = FrozenStructureStudent(checkpoint)
    expected_input = torch.as_tensor(
        (features - mean.reshape(1, 1, 1, -1))
        / scale.reshape(1, 1, 1, -1),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        expected = torch.expm1(model(expected_input)).clamp_min(0.0).numpy()
    for k in range(4):
        expected[:, k, k, :] = 0.0
    actual = loaded.predict(features)
    assert np.allclose(actual, expected, atol=1.0e-6)
    assert loaded.metadata == {"test": True}


def test_endpoint_protocol_reconstructs_direct_prediction_without_clipping(
    tmp_path,
):
    torch.manual_seed(23)
    model = FactorizedStructureStudent(
        input_dim=5, hidden_dim=12, endpoint_dim=3)
    features = np.random.default_rng(23).normal(
        size=(2, 4, 4, 5)).astype(np.float32)
    mean = features.reshape(-1, 5).mean(axis=0, keepdims=True)
    scale = np.maximum(
        features.reshape(-1, 5).std(axis=0, keepdims=True), 1.0e-5)
    normalized = torch.as_tensor(
        (features - mean.reshape(1, 1, 1, -1))
        / scale.reshape(1, 1, 1, -1),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        tx, rx = model.encode_endpoints(normalized)
        endpoints = torch.cat([tx, rx], dim=-1).numpy()
    endpoint_scale = (
        np.max(np.abs(endpoints), axis=(0, 1, 2)) + 1.0e-3)
    checkpoint = tmp_path / "student_protocol.pt"
    save_frozen_structure_student(
        checkpoint,
        model,
        mean,
        scale,
        num_uavs=4,
        num_targets=4,
        rate_scale=2.0,
        endpoint_scale=endpoint_scale,
    )
    loaded = FrozenStructureStudent(checkpoint)
    protocol = loaded.encode_endpoint_protocol(features)
    reconstructed = loaded.predict_from_endpoint_protocol(protocol)
    direct = loaded.predict(features)
    assert np.allclose(reconstructed, direct, atol=1.0e-6)


def test_structure_student_endpoint_stream_uses_physical_u2u_packet(
    tmp_path,
):
    config = get_default_config()
    config.marl.joint_isac_power_enabled = True
    config.marl.comm_bandwidth_hz = 1.0e9
    config.marl.comm_deadline_s = 1.0
    config.marl.comm_snr_threshold_db = -300.0
    K, Q = config.scenario.K, config.scenario.Q
    model = FactorizedStructureStudent(
        input_dim=3, hidden_dim=8, endpoint_dim=2)
    checkpoint = tmp_path / "student_channel.pt"
    save_frozen_structure_student(
        checkpoint,
        model,
        np.zeros((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        num_uavs=K,
        num_targets=Q,
        rate_scale=1.0,
        endpoint_scale=np.ones(4, dtype=np.float32),
    )
    student = FrozenStructureStudent(checkpoint)
    env = UAVISACEnv(config=config, seed=31)
    env.reset(seed=31)
    env.core.configure_structure_student_channel(
        student, bits_per_dim=4, min_comm_fraction=0.25)
    message_dim = env.core._comm_payload_dim
    env.core.submit_learned_communications(
        {
            k: np.zeros(message_dim, dtype=np.float64)
            for k in range(K)
        },
        {k: 0 for k in range(K)},
        comm_power_fractions={k: 0.25 for k in range(K)},
        sensing_target_weights={
            k: np.full(Q, 1.0 / Q) for k in range(K)
        },
        token_masks={k: np.zeros(Q) for k in range(K)},
    )
    protocol = np.linspace(
        -0.8, 0.8, K * Q * 4).reshape(K, Q, 4)
    env.core.submit_structure_student_endpoint_protocol(protocol)
    # Structural precision is independent: a mandatory endpoint packet must
    # not turn the actor's silent latent Token into an unintended message.
    assert all(
        rate == 0 for rate in env.core._pending_comm_rates.values())
    positions = np.stack([uav.pos for uav in env.core.uavs])
    stats = env.core._process_learned_communications(positions)

    expected_bits_per_sender = (
        config.marl.comm_header_bits
        + Q * 4 * config.marl.comm_rate_bits_per_dim[1]
    )
    assert stats.active_senders == K
    assert stats.total_bits == K * expected_bits_per_sender
    assert env.core._structure_student_metrics[
        "structure_student_atomic_delivery_rate"] == 1.0
    assert np.all(env.core._structure_student_public_valid)
    assert env.core._external_structure_edge_values is not None
    assert np.allclose(
        env.core._current_comm_power_w, 0.25 * config.uav.P_isac_total)
    combined = (
        env.core._current_comm_power_w
        + env.core._current_sensing_power_w.sum(axis=1))
    assert np.all(combined <= config.uav.P_isac_total + 1.0e-12)
    assert np.all(
        env.core._current_sensing_power_w.sum(axis=1)
        <= config.uav.P_sense_max + 1.0e-12)


def test_structure_student_adaptive_precision_selects_highest_feasible_rate(
    tmp_path,
):
    config = get_default_config()
    config.marl.joint_isac_power_enabled = True
    config.marl.comm_bandwidth_hz = 5.0e4
    config.marl.comm_snr_threshold_db = -300.0
    K, Q = config.scenario.K, config.scenario.Q
    model = FactorizedStructureStudent(
        input_dim=3, hidden_dim=8, endpoint_dim=2)
    checkpoint = tmp_path / "student_adaptive_rate.pt"
    save_frozen_structure_student(
        checkpoint,
        model,
        np.zeros((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        num_uavs=K,
        num_targets=Q,
        rate_scale=1.0,
        endpoint_scale=np.ones(4, dtype=np.float32),
    )
    env = UAVISACEnv(config=config, seed=33)
    env.reset(seed=33)
    env.core.configure_structure_student_channel(
        FrozenStructureStudent(checkpoint),
        bits_per_dim=8,
        adaptive_min_bits_per_dim=4,
        min_comm_fraction=0.25,
    )
    message_dim = env.core._comm_payload_dim
    env.core.submit_learned_communications(
        {k: np.zeros(message_dim, dtype=np.float64) for k in range(K)},
        {k: 0 for k in range(K)},
        comm_power_fractions={k: 0.25 for k in range(K)},
        sensing_target_weights={
            k: np.full(Q, 1.0 / Q) for k in range(K)
        },
        token_masks={k: np.zeros(Q) for k in range(K)},
    )
    endpoint_width = 4
    endpoint_dimensions = Q * endpoint_width
    env.core.submit_structure_student_endpoint_protocol(np.linspace(
        -0.8, 0.8, K * endpoint_dimensions,
    ).reshape(K, Q, endpoint_width))
    positions = np.asarray([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [100.0, 100.0, 0.0],
    ])
    rate_header_bits = 3  # ceil(log2({4,5,6,7,8}))
    effective_bandwidth = config.marl.comm_bandwidth_hz / K

    def farthest_latency(bits_per_dim):
        packet_bits = (
            config.marl.comm_header_bits
            + endpoint_dimensions * bits_per_dim
            + rate_header_bits
        )
        return env.core._inter_uav_comm._link(
            positions[0], positions[3], packet_bits,
            effective_bandwidth, 0.25 * config.uav.P_isac_total,
        )[-1]

    latency_6 = farthest_latency(6)
    latency_7 = farthest_latency(7)
    assert latency_6 < latency_7
    env.core._inter_uav_comm.deadline_s = 0.5 * (
        latency_6 + latency_7)
    stats = env.core._process_learned_communications(positions)

    expected_bits_per_sender = (
        config.marl.comm_header_bits
        + endpoint_dimensions * 6
        + rate_header_bits
    )
    assert stats.total_bits == K * expected_bits_per_sender
    assert env.core._structure_student_metrics[
        "structure_student_selected_bits_per_dim"] == 6.0
    assert env.core._structure_student_metrics[
        "structure_student_atomic_delivery_rate"] == 1.0


def test_structure_student_resolve_aligned_send_schedule(tmp_path):
    config = get_default_config()
    config.marl.p0_maxmin_pairing_enabled = True
    config.marl.p0_maxmin_pairing_hold_frames = 5
    K, Q = config.scenario.K, config.scenario.Q
    model = FactorizedStructureStudent(
        input_dim=3, hidden_dim=8, endpoint_dim=2)
    checkpoint = tmp_path / "student_schedule.pt"
    save_frozen_structure_student(
        checkpoint,
        model,
        np.zeros((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        num_uavs=K,
        num_targets=Q,
        rate_scale=1.0,
    )
    env = UAVISACEnv(config=config, seed=37)
    env.reset(seed=37)
    env.core.configure_structure_student_channel(
        FrozenStructureStudent(checkpoint))
    assert env.core.structure_student_protocol_due_next_step()
    env.core._cached_p0_solution = object()
    env.core.t = 1
    assert not env.core.structure_student_protocol_due_next_step()
    env.core.t = 4
    assert env.core.structure_student_protocol_due_next_step()
    env.core.t = 5
    assert not env.core.structure_student_protocol_due_next_step()


def test_structure_student_channel_fails_closed_without_public_cache(
    tmp_path,
):
    config = get_default_config()
    K, Q = config.scenario.K, config.scenario.Q
    model = FactorizedStructureStudent(
        input_dim=3, hidden_dim=8, endpoint_dim=2)
    checkpoint = tmp_path / "student_fail_closed.pt"
    save_frozen_structure_student(
        checkpoint,
        model,
        np.zeros((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        num_uavs=K,
        num_targets=Q,
        rate_scale=1.0,
    )
    env = UAVISACEnv(config=config, seed=41)
    env.reset(seed=41)
    env.core.configure_structure_student_channel(
        FrozenStructureStudent(checkpoint))
    env.core._refresh_structure_student_edges()
    graph = env.core._external_structure_edge_values
    assert graph is not None
    assert graph.shape == (K, K, Q)
    assert np.count_nonzero(graph) == 0


def test_structure_student_scale_migration_requires_explicit_opt_in(
    tmp_path,
):
    config = get_default_config()
    config.scenario.K = 6
    config.scenario.Q = 6
    config.target.omega_q = [1.0 / 6.0] * 6
    model = FactorizedStructureStudent(
        input_dim=3, hidden_dim=8, endpoint_dim=2)
    checkpoint = tmp_path / "student_k4_metadata.pt"
    save_frozen_structure_student(
        checkpoint,
        model,
        np.zeros((1, 3), dtype=np.float32),
        np.ones((1, 3), dtype=np.float32),
        num_uavs=4,
        num_targets=4,
        rate_scale=1.0,
    )
    student = FrozenStructureStudent(checkpoint)
    env = UAVISACEnv(config=config, seed=43)
    env.reset(seed=43)
    with np.testing.assert_raises(ValueError):
        env.core.configure_structure_student_channel(student)
    env.core.configure_structure_student_channel(
        student, allow_cardinality_mismatch=True)
    assert env.core._structure_student_endpoint_width == 4


def test_cardinality_residual_is_exactly_zero_at_anchor():
    torch.manual_seed(12)
    base = FactorizedStructureStudent(
        input_dim=5, hidden_dim=7, endpoint_dim=3)
    model = CardinalityResidualStructureStudent(
        input_dim=5,
        hidden_dim=7,
        endpoint_dim=3,
        anchor_num_uavs=4,
        anchor_num_targets=4,
        reference_num_uavs=6,
        reference_num_targets=6,
    )
    model.base.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        model.tx_residual[-1].bias.fill_(0.7)
        model.rx_residual[-1].bias.fill_(-0.4)
        model.decoder_residual[-1].bias.fill_(0.2)
    anchor = torch.randn(2, 4, 4, 5)
    larger = torch.randn(2, 6, 6, 5)
    with torch.inference_mode():
        anchor_base_endpoints = base.encode_endpoints(anchor)
        anchor_residual_endpoints = model.encode_endpoints(anchor)
        anchor_base = base(anchor)
        anchor_residual = model(anchor)
        larger_base = base(larger)
        larger_residual = model(larger)
    torch.testing.assert_close(
        anchor_residual_endpoints[0], anchor_base_endpoints[0],
        atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        anchor_residual_endpoints[1], anchor_base_endpoints[1],
        atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        anchor_residual, anchor_base, atol=0.0, rtol=0.0)
    assert not torch.equal(larger_residual, larger_base)
    assert model.cardinality_gate(4, 4) == 0.0
    assert model.cardinality_gate(6, 6) == 1.0
    assert model.cardinality_gate(8, 8) == 1.0


def test_cardinality_residual_artifact_roundtrip(tmp_path):
    torch.manual_seed(13)
    model = CardinalityResidualStructureStudent(
        input_dim=5,
        hidden_dim=7,
        endpoint_dim=3,
        anchor_num_uavs=4,
        anchor_num_targets=4,
        reference_num_uavs=6,
        reference_num_targets=6,
    )
    checkpoint = tmp_path / "cardinality_residual.pt"
    save_cardinality_residual_structure_student(
        checkpoint,
        model,
        np.zeros(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        num_uavs=4,
        num_targets=4,
        rate_scale=8.0,
        endpoint_scale=np.ones(6, dtype=np.float32),
        metadata={"test": True},
    )
    loaded = FrozenStructureStudent(checkpoint)
    features = np.random.default_rng(2).normal(
        size=(2, 6, 6, 5)).astype(np.float32)
    with torch.inference_mode():
        expected_log = model(torch.as_tensor(features))
        expected = torch.expm1(expected_log).clamp_min(0).cpu().numpy()
    for k in range(6):
        expected[:, k, k, :] = 0.0
    np.testing.assert_allclose(loaded.predict(features), expected)
    assert loaded.schema_version == 2
    assert loaded.metadata["test"] is True


def test_environment_accepts_validated_external_student_graph():
    config = get_default_config()
    env = UAVISACEnv(config=config, seed=5)
    values = np.ones(
        (env.K, env.K, env.Q), dtype=np.float64)
    env.core.submit_structure_student_edge_values(values)
    stored = env.core._external_structure_edge_values
    assert stored is not values
    assert np.all(stored[np.eye(env.K, dtype=bool)] == 0.0)
    with np.testing.assert_raises(ValueError):
        env.core.submit_structure_student_edge_values(
            np.ones((env.K, env.K, env.Q + 1), dtype=np.float64))


def test_environment_reset_clears_local_detection_history():
    config = get_default_config()
    env = UAVISACEnv(config=config, seed=9)
    clean_observations, _ = env.reset(seed=17)
    env.core.prev_P_D_local = {
        k: np.ones(env.Q, dtype=np.float64)
        for k in range(env.K)
    }
    repeated_observations, _ = env.reset(seed=17)
    assert env.core.prev_P_D_local == {}
    for agent in clean_observations:
        assert np.array_equal(
            clean_observations[agent], repeated_observations[agent])
