"""P0 regression: smooth-disk movement parameterization (B5 / advice 014).

The legacy ``radial_clip`` path maps the Gaussian pre-image through
tanh -> scale -> radial projection onto the disk.  The projection is a
many-to-one map (~21% of the tanh box lies outside the inscribed circle and
triggers it), so ``log pi(executed action)`` is NOT the true density of the
executed action and the PPO importance ratio is mathematically inexact.

``smooth_disk`` replaces the chain with the bijection

    dp = d_max * z / sqrt(1 + |z|^2),     |det d(dp)/dz| = d_max^2/(1+|z|^2)^2,

whose exact change-of-variables correction

    log pi_dp(dp) = log N(z; mu, sigma) + 2*log(1+|z|^2) - 2*log(d_max)

makes the PPO ratio exact for every executed action.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from config.params import load_config
from uav_isac.environment.action import (
    ActionSpace, DP_PARAM_SMOOTH_DISK, DP_PARAM_RADIAL_CLIP)
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.networks import StructuredActorNetwork
from uav_isac.utils.types import Action


def _aspace(**kw):
    kw.setdefault('dp_parameterization', DP_PARAM_SMOOTH_DISK)
    return ActionSpace(v_max=25.0, dt=0.1, learn_roles=False, **kw)


# ---------------------------------------------------------------------------
# 1. Geometry of the bijection
# ---------------------------------------------------------------------------

def test_map_is_bounded_and_bijective():
    rng = np.random.default_rng(0)
    aspace = _aspace()
    zs = rng.normal(size=(500, 2)) * 8.0  # includes very large |z| (near rim)
    for z in zs:
        dp = aspace._map_smooth_disk(z)
        assert float(np.linalg.norm(dp)) < aspace.max_dp, "map must stay inside the disk"
        z_back = aspace._inverse_smooth_disk(dp)
        assert np.allclose(z_back, z, atol=1e-9), "inverse must recover z exactly"
    # injectivity: distinct z -> distinct dp
    for i in range(40):
        zi = zs[i]
        for j in range(i + 1, 40):
            if not np.allclose(zs[j], zi):
                assert not np.allclose(
                    aspace._map_smooth_disk(zs[j]),
                    aspace._map_smooth_disk(zi),
                ), "map must be one-to-one"


def test_deterministic_mean_bounded():
    aspace = _aspace()
    rng = np.random.default_rng(4)
    for mu in rng.normal(size=(200, 2)) * 20.0:  # huge means too
        dp = aspace._map_smooth_disk(mu)
        assert float(np.linalg.norm(dp)) < aspace.max_dp
        act = aspace.decode_deterministic(mu, np.zeros(3))
        assert float(np.linalg.norm(act.delta_p)) <= aspace.max_dp


def test_seeded_action_space_replays_stochastic_actions():
    """The experiment seed owns the NumPy action-sampling stream."""
    first = ActionSpace(seed=20260820, learn_roles=True)
    second = ActionSpace(seed=20260820, learn_roles=True)
    mean = np.asarray([0.2, -0.3], dtype=np.float64)
    log_std = np.asarray([-0.7, -0.4], dtype=np.float64)
    logits = np.asarray([0.1, 0.2, -0.4], dtype=np.float64)

    for _ in range(8):
        action_a, log_prob_a = first.decode(
            mean, log_std, logits, deterministic=False)
        action_b, log_prob_b = second.decode(
            mean, log_std, logits, deterministic=False)
        np.testing.assert_array_equal(action_a.delta_p, action_b.delta_p)
        assert action_a.role == action_b.role
        assert log_prob_a == log_prob_b


def test_action_space_rejects_ambiguous_rng_ownership():
    with pytest.raises(ValueError, match="either rng or seed"):
        ActionSpace(rng=np.random.default_rng(1), seed=1)


# ---------------------------------------------------------------------------
# 2. Jacobian
# ---------------------------------------------------------------------------

def test_jacobian_matches_finite_difference():
    aspace = _aspace()
    rng = np.random.default_rng(1)
    eps = 1e-6
    for z in rng.normal(size=(20, 2)) * 3.0:
        f = lambda v: aspace._map_smooth_disk(v)
        J = np.zeros((2, 2))
        for c in range(2):
            e = np.zeros(2)
            e[c] = eps
            J[:, c] = (f(z + e) - f(z - e)) / (2 * eps)
        det_num = float(np.abs(np.linalg.det(J)))
        det_exact = aspace.max_dp ** 2 / (1.0 + float(z @ z)) ** 2
        assert np.isclose(det_num, det_exact, rtol=1e-4), (
            f"|det J| numeric {det_num:.6e} != exact {det_exact:.6e} at z={z}")


def test_log_prob_matches_density_formula():
    """compute_log_prob must equal log N(z) + log|det dz/ddp| for map outputs."""
    aspace = _aspace()
    mu = np.array([0.3, -0.2])
    log_std = np.array([-0.5, -0.5])
    std = np.exp(log_std)
    rng = np.random.default_rng(2)
    for z in rng.normal(size=(30, 2)) * 4.0:
        dp = aspace._map_smooth_disk(z)
        lp = aspace.compute_log_prob(
            Action(delta_p=dp, role=0), mu, log_std, np.zeros(3))
        log_nz = float(np.sum(
            -0.5 * ((z - mu) / std) ** 2
            - np.log(std) - 0.5 * np.log(2 * np.pi)))
        corr = aspace._smooth_disk_log_correction(z)
        assert np.isclose(lp, log_nz + corr, atol=1e-9), (
            f"log-prob {lp:.9f} != log N(z) {log_nz:.9f} + corr {corr:.9f}")


# ---------------------------------------------------------------------------
# 3. PPO-ratio exactness (the point of the fix)
# ---------------------------------------------------------------------------

def test_ppo_ratio_exact_under_smooth_disk():
    """rollout log-prob == recomputed evaluate_actions log-prob, near the rim."""
    cfg = load_config('config/exp_800_k8_q8.yaml')
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    k, q = 8, 8
    aspace = _aspace()
    aspace.num_targets = q
    aspace.structured_actor = True
    aspace.structured_entity_dim = 64
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(
        obs_dim=obs_dim, K=k, Q=q, entity_dim=64, max_dp=max_dp)
    actor.eval()
    agent = MAPPOAgent(0, obs_dim, 50, aspace, num_agents=k, num_targets=q)
    agent.actor = actor
    agent.action_space = aspace

    rng = np.random.default_rng(3)
    B = 12
    obs = torch.randn(B, obs_dim)
    with torch.no_grad():
        dp_mean, dp_log_std, role_logits, _, _, _ = actor(obs)
    dp_mean_np = dp_mean.numpy()
    dp_std_np = dp_log_std.numpy()
    rl_np = role_logits.numpy()

    actions_dp = np.zeros((B, 2))
    actions_role = np.zeros(B, dtype=np.int64)
    old_lps = np.zeros(B)
    rim_norms = []
    for i in range(B):
        # Large |z| forces |dp| close to the rim: the regime that was many-to-one
        # under radial_clip and produced the wrong density there.  Mirror the
        # decode() float32 quantization so the scored action equals the value the
        # buffer/update path will see (the smooth inverse is ill-conditioned near
        # the rim, so even a 1-ULP mismatch would leak into the ratio).
        z = rng.normal(size=2) * 6.0 + dp_mean_np[i]
        dp = aspace._map_smooth_disk(z)
        dp = dp.astype(np.float32).astype(np.float64)
        actions_dp[i] = dp
        actions_role[i] = 0
        old_lps[i] = aspace.compute_log_prob(
            Action(delta_p=dp, role=0), dp_mean_np[i], dp_std_np, rl_np[i])
        rim_norms.append(float(np.linalg.norm(dp)) / aspace.max_dp)
    assert min(rim_norms) > 0.6, "test must include near-rim actions"

    with torch.no_grad():
        new_lp, _, _, _, _, _ = agent.evaluate_actions(
            obs, torch.zeros(B, 50),
            torch.as_tensor(actions_dp, dtype=torch.float32),
            torch.as_tensor(actions_role),
        )
    max_diff = float((old_lps - new_lp.numpy()).__abs__().max())
    assert max_diff < 1e-5, (
        f"max|rollout_lp - recomputed_lp| = {max_diff:.2e} >= 1e-5; "
        f"PPO ratio is NOT exact under smooth_disk")


def test_buffer_pipeline_consistent_under_smooth_disk():
    """Full buffer -> evaluate_actions path keeps the ratio exact (smooth_disk)."""
    from uav_isac.agents.buffer import RolloutBuffer

    k, q = 4, 4
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    gs_dim = 50
    aspace = _aspace()
    aspace.num_targets = q
    aspace.structured_actor = True
    aspace.structured_entity_dim = 64

    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64)
    actor.eval()
    agent = MAPPOAgent(0, obs_dim, gs_dim, aspace, num_agents=k, num_targets=q)
    agent.actor = actor
    agent.action_space = aspace

    rng = np.random.default_rng(7)
    buf = RolloutBuffer(buffer_size=4, num_agents=k, obs_dim=obs_dim,
                        global_state_dim=gs_dim, gamma=0.99, gae_lambda=0.95,
                        num_targets=q, gru_hidden_dim=64)
    stored_lps = []
    for t in range(4):
        ob = np.random.randn(k, obs_dim).astype(np.float32)
        with torch.no_grad():
            dp_mean, dp_log_std, role_logits, _, _, _ = actor(
                torch.as_tensor(ob, dtype=torch.float32))
        dp_mean_np = dp_mean.numpy()
        dp_std_np = dp_log_std.numpy()
        rl_np = role_logits.numpy()
        actions_dp = np.zeros((k, 2))
        lps = np.zeros(k)
        for i in range(k):
            z = rng.normal(size=2) * 5.0 + dp_mean_np[i]  # near-rim
            dp = aspace._map_smooth_disk(z)
            dp = dp.astype(np.float32).astype(np.float64)  # mirror decode()
            actions_dp[i] = dp
            lps[i] = aspace.compute_log_prob(
                Action(delta_p=dp, role=0), dp_mean_np[i], dp_std_np, rl_np[i])
        stored_lps.append(lps.copy())
        buf.store({i: ob[i] for i in range(k)}, np.zeros(gs_dim),
                  actions_dp, np.zeros(k, dtype=np.int32), lps, np.zeros(k),
                  {i: 0.0 for i in range(k)},
                  {i: False for i in range(k)},
                  h_prev=np.zeros((k, k - 1, 64), dtype=np.float32))
    buf.compute_gae(np.zeros(k))
    data = buf.get_training_data()
    with torch.no_grad():
        new_lp, _, _, _, _, _ = agent.evaluate_actions(
            data['obs'], torch.zeros(len(data['obs']), gs_dim),
            data['actions_dp'], data['actions_role'])
    max_diff = float((data['old_log_probs'] - new_lp).abs().max())
    assert max_diff < 1e-5, (
        f"buffer pipeline max|old - new| = {max_diff:.2e} under smooth_disk")


# ---------------------------------------------------------------------------
# 4. Backward compatibility of the legacy path
# ---------------------------------------------------------------------------

def test_radial_clip_default_and_unchanged():
    """Default parameterization stays radial_clip; legacy behavior preserved."""
    aspace = ActionSpace(v_max=25.0, dt=0.1)
    assert aspace.dp_parameterization == DP_PARAM_RADIAL_CLIP
    rng = np.random.default_rng(5)
    aspace.rng = rng
    dp_mean = np.array([0.5, 0.5])
    dp_std = np.array([0.0, 0.0])
    action, lp = aspace.decode(dp_mean, dp_std, np.zeros(3),
                               dp_deterministic=False)
    # tanh(0.5)*2.5 = 1.096... < max_dp -> no projection in the mean direction;
    # a large mean must still be projected (legacy radial clamp).
    big = np.array([10.0, 10.0])
    action_big = aspace.decode_deterministic(big, np.zeros(3))
    assert float(np.linalg.norm(action_big.delta_p)) <= aspace.max_dp + 1e-9
    # decode log-prob still equals compute_log_prob under radial_clip
    lp2 = aspace.compute_log_prob(action, dp_mean, dp_std, np.zeros(3))
    assert np.isclose(lp, lp2, atol=1e-9)


def test_smooth_disk_differs_from_radial_clip():
    """The two parameterizations are genuinely different maps."""
    aspace_sd = _aspace()
    aspace_rc = _aspace(dp_parameterization=DP_PARAM_RADIAL_CLIP)
    mu = np.array([3.0, -3.0])  # large mean: tanh saturates, smooth map does not
    dp_sd = aspace_sd._map_smooth_disk(mu)
    dp_rc = np.tanh(mu) * aspace_rc.dp_scale
    assert not np.allclose(dp_sd, dp_rc), "parameterizations must differ"


def test_env_core_threads_parameterization():
    """UAVISACEnv action space must inherit marl.dp_parameterization."""
    from uav_isac.environment.env_wrapper import UAVISACEnv
    cfg = load_config('config/exp_800_q4.yaml')
    setattr(cfg.marl, 'dp_parameterization', DP_PARAM_SMOOTH_DISK)
    env = UAVISACEnv(config=cfg, seed=0)
    try:
        assert env.core.action_space.dp_parameterization == DP_PARAM_SMOOTH_DISK
    finally:
        env.close()


def test_env_reseed_rebinds_all_persistent_rng_clients():
    """Gym reset(seed=...) must not leave stale generators in subcomponents."""
    from uav_isac.environment.env_wrapper import UAVISACEnv
    cfg = load_config('config/exp_800_q4.yaml')
    env = UAVISACEnv(config=cfg, seed=1)
    try:
        env.reset(seed=91)
        assert env.rng is env.core.rng
        assert env.core.action_space.rng is env.core.rng
        assert env.core.deflection_computer.rng is env.core.rng
    finally:
        env.close()


def test_repeated_seed_reset_replays_initial_state_on_same_instance():
    from uav_isac.environment.env_wrapper import UAVISACEnv
    cfg = load_config('config/exp_800_q4.yaml')
    env = UAVISACEnv(config=cfg, seed=2)
    try:
        first, _ = env.reset(seed=37)
        second, _ = env.reset(seed=37)
        for agent_id in first:
            np.testing.assert_array_equal(first[agent_id], second[agent_id])
    finally:
        env.close()
