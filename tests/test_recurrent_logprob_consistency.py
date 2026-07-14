"""P0 regression: verify that old_log_prob can be recomputed exactly.

This test must pass before any PPO training. If it fails, the PPO ratio
is invalid and all training results are contaminated.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from config.params import load_config
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.networks import StructuredActorNetwork


@pytest.mark.parametrize("k,q", [(4, 4), (8, 8)])
def test_logprob_recomputation_matches_rollout(k, q):
    """Recomputing log-prob with same h_prev must match rollout log-prob."""
    cfg = load_config('config/exp_800_k8_q8.yaml' if k == 8 else 'config/exp_800_q4.yaml')
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    device = 'cpu'

    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
    aspace.num_targets = q
    aspace.structured_actor = True
    aspace.structured_entity_dim = 64

    # Single-frame obs dim: self(8) + physics(3) + targets(q*17) + neighbors((k-1)*8) + global(2) + P_D(q) + comm(16)
    # Without P0: 8+3+17q+8(k-1)+2+q+16 = 29 + 18q + 8(k-1)
    obs_dim = 29 + 18*q + 8*(k-1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64, max_dp=max_dp).to(device)

    # Create a rollout-like scenario with non-zero h_prev
    B = 4
    obs = torch.randn(B, obs_dim)
    h_prev = torch.randn(1, B*(k-1), 64) * 0.1  # small random GRU state

    # First pass (simulating rollout): get actions + log_probs
    with torch.no_grad():
        dp_mean, dp_log_std, role_logits, _, _, _ = actor(obs, h_prev)

    # Simulate an action
    dp_norm = torch.tanh(dp_mean) * 0.5  # some action within range
    dp_raw = torch.atanh(torch.clamp(dp_norm, -0.999, 0.999))
    dp_std_pos = torch.exp(torch.clamp(dp_log_std, -20, 2))
    var = dp_std_pos ** 2
    rollout_log_prob = -0.5 * (
        ((dp_raw - dp_mean) ** 2) / (var + 1e-6) + torch.log(2*np.pi*var + 1e-6)
    ).sum(dim=-1)
    rollout_log_prob -= torch.log(1.0 - dp_norm**2 + 1e-6).sum(dim=-1)

    # Second pass (simulating update recomputation): same obs, same h_prev
    dp_mean2, dp_log_std2, _, _, _, _ = actor(obs, h_prev)

    dp_std_pos2 = torch.exp(torch.clamp(dp_log_std2, -20, 2))
    var2 = dp_std_pos2 ** 2
    recomputed_log_prob = -0.5 * (
        ((dp_raw - dp_mean2) ** 2) / (var2 + 1e-6) + torch.log(2*np.pi*var2 + 1e-6)
    ).sum(dim=-1)
    recomputed_log_prob -= torch.log(1.0 - dp_norm**2 + 1e-6).sum(dim=-1)

    max_diff = (rollout_log_prob - recomputed_log_prob).abs().max().item()
    assert max_diff < 1e-4, (
        f"K={k},Q={q}: max|rollout_lp - recomputed_lp| = {max_diff:.2e} >= 1e-4. "
        f"PPO ratio is INVALID."
    )


def test_h_prev_none_vs_zero_are_equivalent():
    """h_prev=None (zero-init) and h_prev=zeros should give same output."""
    k, q = 4, 4
    obs_dim = 29 + 18*q + 8*(k-1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    obs = torch.randn(2, obs_dim)

    with torch.no_grad():
        out_none = actor(obs, None)
        h_zero = torch.zeros(1, 2*(k-1), 64)
        out_zero = actor(obs, h_zero)

    # dp_mean should be identical
    max_diff = (out_none[0] - out_zero[0]).abs().max().item()
    assert max_diff < 1e-5, f"h_prev=None vs zeros: dp_mean diff={max_diff:.2e}"


def test_buffer_to_evaluate_actions_logprob_pipeline():
    """Integration: real buffer store → get_training_data → evaluate_actions.

    This is the critical pipeline that the PPO ratio depends on.
    Must verify that old_log_probs from rollout match recomputed log_probs
    through the FULL buffer→Agent→evaluate_actions path.
    """
    import numpy as np
    import torch
    from uav_isac.agents.buffer import RolloutBuffer
    from uav_isac.environment.action import ActionSpace

    k, q = 4, 4
    obs_dim = 29 + 18*q + 8*(k-1)
    gs_dim = 50
    device = 'cpu'

    aspace = ActionSpace(v_max=25.0, dt=0.1, learn_roles=False)
    aspace.num_targets = q
    aspace.structured_actor = True
    aspace.structured_entity_dim = 64

    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).to(device)
    actor.eval()

    # Simulate a short rollout: 4 steps, K=4 agents
    buf = RolloutBuffer(buffer_size=4, num_agents=k, obs_dim=obs_dim,
                        global_state_dim=gs_dim, gamma=0.99, gae_lambda=0.95,
                        num_targets=q, gru_hidden_dim=64)

    stored_log_probs = []
    for t in range(4):
        obs_dict = {}
        h_prev_arr = np.zeros((k, k-1, 64), dtype=np.float32)
        # Vary h_prev per timestep to test GRU consistency
        if t > 0:
            h_prev_arr = h_prev_arr + 0.01 * t * np.random.randn(k, k-1, 64).astype(np.float32)

        for agent_id in range(k):
            obs_dict[agent_id] = np.random.randn(obs_dim).astype(np.float64)

        ob = np.stack([obs_dict[i] for i in range(k)])
        h_batch = torch.as_tensor(h_prev_arr.reshape(1, k*(k-1), 64), dtype=torch.float32)

        with torch.no_grad():
            dp_mean, dp_log_std, role_logits, _, _, _ = actor(
                torch.as_tensor(ob, dtype=torch.float32), h_batch)

        # Decode actions and compute log_probs (simulating rollout)
        dp_mean_np = dp_mean.numpy()
        dp_std_np = dp_log_std.numpy()
        actions_dp = np.zeros((k, 2))
        log_probs_arr = np.zeros(k)
        for i in range(k):
            action, lp = aspace.decode(dp_mean_np[i], dp_std_np, np.zeros(3),
                                       dp_deterministic=False)
            actions_dp[i] = action.delta_p
            log_probs_arr[i] = lp

        stored_log_probs.append(log_probs_arr.copy())
        buf.store(obs_dict, np.zeros(gs_dim), actions_dp,
                  np.zeros(k, dtype=np.int32), log_probs_arr, np.zeros(k),
                  {i: 0.0 for i in range(k)}, {i: False for i in range(k)},
                  h_prev=h_prev_arr)

    buf.compute_gae(np.zeros(k))
    data = buf.get_training_data()
    assert 'h_prev' in data, "h_prev missing from training data"

    # Now recompute through evaluate_actions (the PPO update path)
    from uav_isac.agents.mappo_agent import MAPPOAgent
    agent = MAPPOAgent(0, obs_dim, gs_dim, aspace, num_agents=k,
                       num_targets=q, device=device)
    agent.actor = actor
    agent.action_space = aspace

    obs_t = data['obs']
    dp_t = data['actions_dp']
    role_t = data['actions_role']
    old_lp = data['old_log_probs']
    h_t = data['h_prev']  # (T*K, K-1, D)
    h_for_eval = h_t.reshape(1, -1, 64)  # (1, T*K*(K-1), D)

    with torch.no_grad():
        new_lp, _, _, _, _, _ = agent.evaluate_actions(
            obs_t, torch.zeros(len(obs_t), gs_dim),
            dp_t, role_t, h_prev=h_for_eval,
        )

    max_diff = (old_lp - new_lp).abs().max().item()
    assert max_diff < 1e-4, (
        f"Buffer→evaluate_actions pipeline: max|old_lp - new_lp| = {max_diff:.2e} >= 1e-4. "
        f"PPO ratio is INVALID in the full pipeline."
    )
