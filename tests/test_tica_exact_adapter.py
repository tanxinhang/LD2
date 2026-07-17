"""P0: D1TICAResidualActor exact-preserving initialization tests.

Six invariants at initialization:
  1. dp_mean identical to D1 base
  2. role_logits identical to D1 base
  3. log_std and action log-prob identical
  4. Non-zero h_prev still produces identical output
  5. Early window frames do not affect initial output
  6. Backward: D1 base has no grad, residual heads have grad
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from uav_isac.agents.tica_actor import TICAActor, D1TICAResidualActor
from uav_isac.agents.networks import StructuredActorNetwork
from uav_isac.environment.observation_slices import ObservationSlices


def build_adapter(K=4, Q=4, D=64, L=8):
    """Build a D1TICAResidualActor with minimal dimensions for fast testing."""
    sl = ObservationSlices.from_config(K=K, Q=Q)
    obs_dim = sl.total_dim

    base = StructuredActorNetwork(
        obs_dim=obs_dim, K=K, Q=Q, entity_dim=D,
        use_corrected_parser=True).cpu()
    base.eval()

    tica = TICAActor(obs_dim=obs_dim, K=K, Q=Q, D=D, L=L).cpu()
    tica.eval()

    adapter = D1TICAResidualActor(base, tica).cpu()
    adapter.eval()
    return adapter, base, sl


def test_dp_mean_identical():
    """dp_mean must match D1 base exactly at init."""
    adapter, base, sl = build_adapter()
    obs = torch.randn(4, sl.total_dim)

    with torch.no_grad():
        base_out = base(obs)
        adapter_out = adapter(obs)

    assert torch.allclose(adapter_out[0], base_out[0], atol=1e-7), \
        f"dp_mean mismatch: max|diff|={(adapter_out[0]-base_out[0]).abs().max():.2e}"


def test_role_logits_identical():
    """role_logits must match D1 base exactly at init."""
    adapter, base, sl = build_adapter()
    obs = torch.randn(4, sl.total_dim)

    with torch.no_grad():
        base_out = base(obs)
        adapter_out = adapter(obs)

    assert torch.allclose(adapter_out[2], base_out[2], atol=1e-7), \
        f"role_logits mismatch: max|diff|={(adapter_out[2]-base_out[2]).abs().max():.2e}"


def test_log_std_and_action_prob_identical():
    """log_std, comm, P_D must match, and resulting action log-prob must match."""
    adapter, base, sl = build_adapter()
    obs = torch.randn(4, sl.total_dim)

    with torch.no_grad():
        base_out = base(obs)
        adapter_out = adapter(obs)

    # log_std
    assert torch.allclose(adapter_out[1], base_out[1], atol=1e-7)
    # comm
    assert torch.allclose(adapter_out[3], base_out[3], atol=1e-7)
    # P_D
    assert torch.allclose(adapter_out[4], base_out[4], atol=1e-7)

    # Verify action log-prob: simulate a tanh-squashed action
    dp_scale = 2.5
    dp_norm_base = torch.tanh(base_out[0]) * 0.5
    dp_norm_ada = torch.tanh(adapter_out[0]) * 0.5
    assert torch.allclose(dp_norm_base, dp_norm_ada, atol=1e-7), \
        "Action log-prob would differ — dp_mean not identical enough"


def test_nonzero_h_prev_still_identical():
    """Even with non-zero h_prev (streaming GRU state), outputs must match."""
    adapter, base, sl = build_adapter(K=4, Q=4)
    obs = torch.randn(4, sl.total_dim)
    h_prev = torch.randn(1, 4 * 3, 64) * 0.1  # (1, K*(K-1), D)

    with torch.no_grad():
        base_out = base(obs, h_prev=h_prev)
        adapter_out = adapter(obs, h_prev=h_prev)

    assert torch.allclose(adapter_out[0], base_out[0], atol=1e-7), \
        f"h_prev≠0 dp_mean mismatch: max|diff|={(adapter_out[0]-base_out[0]).abs().max():.2e}"
    assert torch.allclose(adapter_out[2], base_out[2], atol=1e-7)
    # h_new should also match
    if base_out[5] is not None and adapter_out[5] is not None:
        assert torch.allclose(adapter_out[5], base_out[5], atol=1e-7)


def test_early_window_frames_dont_change_initial_output():
    """At init, changing history frames must NOT change output (residual is zero)."""
    adapter, base, sl = build_adapter(K=4, Q=4, L=8)
    last_obs = torch.randn(4, sl.total_dim)

    # Build two windows with same last frame, different history
    obs_a = torch.randn(4, 8, sl.total_dim)
    obs_b = torch.randn(4, 8, sl.total_dim)
    obs_a[:, -1, :] = last_obs
    obs_b[:, -1, :] = last_obs

    with torch.no_grad():
        out_a = adapter(obs_a)
        out_b = adapter(obs_b)

    for i, name in enumerate(['dp_mean', 'log_std', 'role_logits', 'comm', 'pd_pred']):
        assert torch.allclose(out_a[i], out_b[i], atol=1e-7), \
            f"{name} changed with different history at init"


def test_backward_d1_frozen_residual_grad():
    """D1 base must have no grad; residual heads must receive grad."""
    adapter, base, sl = build_adapter(K=4, Q=4, L=8)
    adapter.train()

    obs = torch.randn(2, sl.total_dim)
    dp, _, role, _, _, _ = adapter(obs)
    loss = dp.sum() + role.sum()  # both heads must receive grad
    loss.backward()

    # D1 base: all params must have None grad
    d1_no_grad = 0
    for p in base.parameters():
        if p.grad is not None:
            d1_no_grad += 1
    assert d1_no_grad == 0, f"{d1_no_grad} D1 base params received gradient"

    # Residual heads: must have grad
    assert adapter.delta_dp.weight.grad is not None, "delta_dp has no grad"
    assert adapter.delta_role.weight.grad is not None, "delta_role has no grad"
    assert adapter.delta_dp.weight.grad.abs().sum() > 0, "delta_dp grad is all-zero"
    assert adapter.delta_role.weight.grad.abs().sum() > 0, "delta_role grad is all-zero"


def test_real_d1_checkpoint_exact_match():
    """Load actual D1-corrected checkpoint and verify exact match."""
    d1_path = 'results/dagger_corrected/dagger_D1.pt'
    if not os.path.exists(d1_path):
        pytest.skip(f"{d1_path} not found")

    d1_ckpt = torch.load(d1_path, map_location='cpu', weights_only=False)
    K, Q = 4, 4
    sl = ObservationSlices.from_config(K=K, Q=Q)
    obs_dim = sl.total_dim

    base = StructuredActorNetwork(
        obs_dim=obs_dim, K=K, Q=Q, entity_dim=64,
        use_corrected_parser=True).cpu()
    base.load_state_dict(d1_ckpt, strict=False)
    base.zero_init_new_layers(set(d1_ckpt.keys()))
    base.eval()

    tica = TICAActor(obs_dim=obs_dim, K=K, Q=Q, D=64, L=8).cpu()
    adapter = D1TICAResidualActor(base, tica).cpu()
    adapter.eval()

    obs = torch.randn(4, obs_dim)
    with torch.no_grad():
        base_out = base(obs)
        adapter_out = adapter(obs)

    for i, name in enumerate(['dp_mean', 'log_std', 'role_logits', 'comm', 'pd_pred']):
        diff = (adapter_out[i] - base_out[i]).abs().max().item()
        assert diff < 1e-6, f"{name}: max|diff|={diff:.2e} (real D1 checkpoint)"
        print(f"  {name}: max|diff|={diff:.2e} ✓")

    print("D1-corrected exact match: PASSED")
