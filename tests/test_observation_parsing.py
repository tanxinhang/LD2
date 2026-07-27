"""P0 sentinel: verify observation parsing correctly assigns belief/geometry per target."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from uav_isac.environment.observation_slices import ObservationSlices


@pytest.mark.parametrize("k,q,use_p0", [(4, 4, False), (4, 4, True), (8, 8, False)])
def test_observation_slices_correctly_parse_beliefs_and_geometry(k, q, use_p0):
    """Each target's belief and geometry must come from the SAME target index."""
    slices = ObservationSlices.from_config(K=k, Q=q, use_p0=use_p0)

    # Build a sentinel observation where each target gets unique IDs
    obs = np.zeros(slices.total_dim, dtype=np.float64)

    for target_q in range(q):
        # Belief: fill with target_q * 10 + 1
        b_start = slices.belief_start + target_q * slices.belief_per_target
        obs[b_start:b_start + slices.belief_per_target] = target_q * 10 + 1

        # Geometry: fill with target_q * 10 + 2
        if slices.has_rel_features:
            g_start = slices.geom_start + target_q * slices.geom_per_target
            obs[g_start:g_start + slices.geom_per_target] = target_q * 10 + 2

    # Extract and verify
    beliefs = slices.extract_beliefs(obs)  # (Q, 9)
    geometry = slices.extract_geometry(obs)  # (Q, 8) or None

    for q_idx in range(q):
        b = beliefs[q_idx]
        assert np.allclose(b, q_idx * 10 + 1, atol=1e-6), \
            f"Q={q_idx}: belief has wrong values. Expected {q_idx*10+1}, got {b[:3]}"

        if geometry is not None:
            g = geometry[q_idx]
            assert np.allclose(g, q_idx * 10 + 2, atol=1e-6), \
                f"Q={q_idx}: geometry has wrong values. Expected {q_idx*10+2}, got {g[:3]}"


def test_observation_slices_self_physics_pd_comm():
    """Verify self, physics, PD_hist, comm extraction."""
    slices = ObservationSlices.from_config(K=4, Q=4)

    obs = np.zeros(slices.total_dim, dtype=np.float64)
    obs[slices.self_start:slices.self_start + 8] = 1.0
    obs[slices.physics_start:slices.physics_start + 3] = 2.0
    obs[slices.pd_hist_start:slices.pd_hist_start + 4] = np.arange(4) * 10
    obs[slices.comm_start:slices.comm_start + 16] = 3.0

    assert np.allclose(slices.extract_self(obs), 1.0)
    assert np.allclose(slices.extract_physics(obs), 2.0)
    assert np.allclose(slices.extract_pd_hist(obs), np.arange(4) * 10)
    assert np.allclose(slices.extract_comm(obs), 3.0)


def test_local_channel_feedback_slice_and_rate_head_are_isolated():
    """Local channel state must affect only the categorical rate decision."""
    from uav_isac.agents.networks import StructuredActorNetwork

    k = q = 4
    token_dim = 22
    slices = ObservationSlices.from_config(
        K=k, Q=q, use_p0=False, use_rel_features=True,
        use_comm_tokens=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, use_channel_feedback=True,
        channel_feedback_dim=6)
    actor = StructuredActorNetwork(
        obs_dim=slices.total_dim, K=k, Q=q, entity_dim=64,
        use_comm_cross_attention=True, comm_token_dim=token_dim,
        comm_tokens_per_sender=q, comm_payload_dim=q * 16,
        comm_target_token_enabled=True, comm_target_token_dim=16,
        comm_channel_feedback_rate_enabled=True,
        comm_channel_feedback_dim=6).cpu().eval()

    nominal = torch.zeros(2, slices.total_dim)
    stressed = nominal.clone()
    stressed[:, slices.channel_feedback_start + 4] = -0.8
    with torch.no_grad():
        dp_nom, _, _, msg_nom, _, _ = actor(nominal)
        _, logits_nom = actor.communication_parameters(msg_nom)
        dp_stress, _, _, msg_stress, _, _ = actor(stressed)
        _, logits_stress_zero = actor.communication_parameters(msg_stress)

    # The zero-initialized feedback adapter is checkpoint-compatible.
    torch.testing.assert_close(dp_nom, dp_stress, rtol=0.0, atol=0.0)
    torch.testing.assert_close(msg_nom, msg_stress, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        logits_nom, logits_stress_zero, rtol=0.0, atol=0.0)

    with torch.no_grad():
        actor.comm_rate_feedback_head.weight[1, 4] = -1.0
        actor(stressed)
        _, logits_stress_active = actor.communication_parameters(msg_stress)
    assert torch.all(logits_stress_active[:, 1] > logits_stress_zero[:, 1])
    np.testing.assert_allclose(
        slices.extract_channel_feedback(stressed.numpy()),
        stressed[:, slices.channel_feedback_start:
                 slices.channel_feedback_start + 6].numpy())


def test_environment_exposes_only_local_channel_feedback_summary():
    from config.params import load_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    cfg = load_config(
        'config/exp_800_q4_u2u_hierarchical_multistatic_'
        'distributed_matching_hybrid50_paper_top1_u2u_evidence_'
        'selection_calibrated_eval.yaml')
    cfg.marl.comm_channel_feedback_rate_enabled = True
    cfg.marl.comm_channel_feedback_dim = 6
    env = UAVISACEnv(config=cfg, seed=42)
    try:
        obs, _ = env.reset(seed=42)
        core = env.core
        slices = ObservationSlices.from_config(
            K=cfg.scenario.K, Q=cfg.scenario.Q, use_p0=False,
            use_rel_features=True, use_comm_tokens=True,
            comm_token_dim=cfg.marl.comm_target_token_dim + 6,
            comm_tokens_per_sender=cfg.scenario.Q,
            use_channel_feedback=True, channel_feedback_dim=6)
        assert obs['0'].shape == (slices.total_dim,)
        initial = slices.extract_channel_feedback(obs['0'])
        assert initial[0] == 0.0  # no fabricated acknowledgement at reset
        assert initial[3] == 1.0  # no received packet -> maximally stale
        assert initial[4] == 0.0  # nominal configured deadline
        assert initial[5] == 0.0  # nominal configured SNR threshold

        core._active_comm_deadline_s = 8.0e-4
        stressed = core._build_local_channel_feedback({
            1: {
                'age_frames': 0,
                'snr_db': core._active_comm_snr_threshold_db + 10.0,
                'latency_s': 4.0e-4,
            },
        })
        assert stressed[0] == pytest.approx(1.0 / (cfg.scenario.K - 1))
        assert stressed[1] > 0.0
        assert stressed[2] == pytest.approx(0.5)
        assert stressed[4] < 0.0
    finally:
        env.close()


@pytest.mark.parametrize("k,q,use_p0", [(4, 4, False), (4, 4, True), (8, 8, False), (8, 8, True)])
def test_frame_encoder_target_isolation(k, q, use_p0):
    """Changing only target q must NOT affect other targets' tokens."""
    import torch
    from uav_isac.agents.tica_actor import FrameEncoder
    from uav_isac.environment.observation_slices import ObservationSlices

    sl = ObservationSlices.from_config(K=k, Q=q, use_p0=use_p0)
    encoder = FrameEncoder(sl.total_dim, K=k, Q=q, D=64, use_p0=use_p0).cpu()
    encoder.eval()

    obs = torch.randn(2, sl.total_dim)
    with torch.no_grad():
        _, tokens_before, _ = encoder(obs)

    # Modify target 0's belief block
    obs_modified = obs.clone()
    b_start = sl.belief_start
    obs_modified[:, b_start:b_start + sl.belief_per_target] += 10.0
    # Also modify geometry if present
    if sl.has_rel_features and sl.geom_per_target > 0:
        g_start = sl.geom_start
        obs_modified[:, g_start:g_start + sl.geom_per_target] += 10.0

    with torch.no_grad():
        _, tokens_after, _ = encoder(obs_modified)

    # Target 0 MUST change
    assert not torch.allclose(tokens_before[:, 0], tokens_after[:, 0], atol=1e-4), \
        "Target 0 token unchanged after modifying its input"

    # Other targets must NOT change
    for r in range(1, q):
        assert torch.allclose(tokens_before[:, r], tokens_after[:, r], atol=1e-4), \
            f"Target {r} token changed when only target 0 was modified"


def test_structured_actor_corrected_parser():
    """StructuredActor with use_corrected_parser=True must produce valid output."""
    import torch
    from uav_isac.agents.networks import StructuredActorNetwork
    from uav_isac.environment.observation_slices import ObservationSlices

    for k, q in [(4, 4), (8, 8)]:
        sl = ObservationSlices.from_config(K=k, Q=q)
        actor = StructuredActorNetwork(
            obs_dim=sl.total_dim, K=k, Q=q, entity_dim=64,
            use_corrected_parser=True).cpu()
        actor.eval()
        obs = torch.randn(2, sl.total_dim)
        with torch.no_grad():
            dp, ls, rl, cm, pd, hn = actor(obs)
        assert dp.shape == (2, 2), f"K={k},Q={q}: dp shape {dp.shape}"
        assert rl.shape == (2, 3)
        assert cm.shape == (2, 16)


def test_observation_slices_total_dim_matches_env():
    """Slices total dim must match actual env observation dim."""
    from config.params import load_config
    from uav_isac.environment.env_wrapper import UAVISACEnv

    for config_path in ['config/exp_800_q4.yaml', 'config/exp_800_k8_q8.yaml']:
        cfg = load_config(config_path)
        cfg.marl.num_envs = 1
        env = UAVISACEnv(config=cfg, seed=42)
        K, Q = cfg.scenario.K, cfg.scenario.Q
        use_p0 = cfg.marl.use_p0_sinr_gated
        use_rel = cfg.marl.rel_features

        slices = ObservationSlices.from_config(K=K, Q=Q,
                                                use_p0=use_p0,
                                                use_rel_features=use_rel)
        actual_dim = env.core.obs_builder.get_obs_dim()
        assert slices.total_dim == actual_dim, \
            f"{config_path}: slices.total_dim={slices.total_dim}, env obs_dim={actual_dim}"
        env.close()
