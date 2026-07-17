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
