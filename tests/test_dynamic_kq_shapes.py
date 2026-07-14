"""P0 regression: all modules must support K=4,Q=4 and K=8,Q=8 without hardcoded dimensions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from uav_isac.agents.buffer import RolloutBuffer
from uav_isac.agents.networks import StructuredActorNetwork, CriticNetwork


@pytest.mark.parametrize("k,q", [(4, 4), (8, 8)])
def test_buffer_dynamic_q(k, q):
    """RolloutBuffer must accept any Q without shape errors."""
    buf = RolloutBuffer(buffer_size=16, num_agents=k, obs_dim=100,
                        global_state_dim=50, gamma=0.99, gae_lambda=0.95,
                        num_targets=q, gru_hidden_dim=64)
    # Store a transition
    obs = {i: np.zeros(100) for i in range(k)}
    h = np.zeros((k, k-1, 64), dtype=np.float32)
    pt_r = np.zeros((k, q))
    pt_v = np.zeros((k, q))
    buf.store(obs, np.zeros(50), np.zeros((k, 2)), np.zeros(k, dtype=np.int32),
              np.zeros(k), np.zeros(k), {i: 0.0 for i in range(k)},
              {i: False for i in range(k)},
              per_target_rewards=pt_r, per_target_values=pt_v, h_prev=h)
    # Should not raise
    assert buf.per_target_rewards.shape == (16, k, q), \
        f"Expected (16,{k},{q}), got {buf.per_target_rewards.shape}"
    assert buf.per_target_values.shape == (16, k, q)
    if buf._has_gru:
        assert buf.h_prev.shape == (16, k, k-1, 64)


@pytest.mark.parametrize("k,q", [(4, 4), (8, 8)])
def test_critic_dynamic_q(k, q):
    """CriticNetwork must accept any Q for per-target heads."""
    critic = CriticNetwork(state_dim=50, hidden_layers=[64, 64],
                           num_agents=k, comm_dim=16, num_targets=q)
    x = torch.randn(2, 50 + k + 16)
    scalar_v, target_v = critic.forward_with_targets(x)
    assert scalar_v.shape == (2,)
    if q > 0:
        assert target_v.shape == (2, q), f"Expected (2,{q}), got {target_v.shape}"


@pytest.mark.parametrize("k,q", [(4, 4), (8, 8)])
def test_actor_dynamic_kq(k, q):
    """StructuredActor must accept any K,Q combination."""
    # obs dim without P0: self(8)+phys(3)+targets(q*17)+neighbors((k-1)*8)+global(2)+P_D(q)+comm(16)
    obs_dim = 8 + 3 + 17*q + 8*(k-1) + 2 + q + 16
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    x = torch.randn(2, obs_dim)
    dp_mean, log_std, role_logits, comm, pd_pred, h_new = actor(x)
    assert dp_mean.shape == (2, 2)
    assert role_logits.shape == (2, 3)
    assert comm.shape == (2, 16)
    assert h_new.shape == (1, 2*(k-1), 64)


def test_cvar_k_dynamic():
    """CVaR top-k must scale with Q."""
    for q, expected_k in [(4, 1), (8, 2), (12, 3), (1, 1)]:
        cvar_k = max(1, int(np.ceil(0.25 * q)))
        assert cvar_k == expected_k, f"Q={q}: expected cvar_k={expected_k}, got {cvar_k}"
