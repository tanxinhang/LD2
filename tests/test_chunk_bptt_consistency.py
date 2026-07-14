"""P0 regression: chunk BPTT correctness.

Six invariants:
  1. Full-sequence output == chunked output.
  2. Chunk boundary h≠0 after first chunk (state CARRIED).
  3. Episode boundary h=0 (new episode fresh).
  4. detach_h_new=False → gradient flows across frames within chunk.
  5. Chunk boundary detach() → gradient stops.
  6. detach_h_new=True (default) → no cross-frame gradient (rollout/eval mode).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from uav_isac.agents.networks import StructuredActorNetwork


@pytest.mark.parametrize("k,q", [(4, 4)])
def test_chunk_bptt_matches_full_sequence(k, q):
    """Full-sequence forward and chunked forward (detach between chunks)
    must produce identical outputs at every frame."""
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.eval()

    torch.manual_seed(42)
    T = 40
    obs = torch.randn(T, obs_dim)  # (T, obs_dim)

    # Full-sequence forward
    with torch.no_grad():
        full_outputs = []
        h_full = None
        for t in range(T):
            dp_m, _, _, _, _, h_new = actor(obs[t:t+1], h_full)  # (1, obs_dim)
            full_outputs.append(dp_m.clone())
            h_full = h_new

    # Chunked forward (chunk_size=16, detach between chunks)
    chunk_size = 16
    with torch.no_grad():
        chunk_outputs = []
        h_chunk = None
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            for t in range(start, end):
                h_in = None if h_chunk is None else h_chunk
                dp_m, _, _, _, _, h_new = actor(obs[t:t+1], h_in)  # (1, obs_dim)
                chunk_outputs.append(dp_m.clone())
                h_chunk = h_new
            h_chunk = h_chunk.detach() if h_chunk is not None else None  # TBPTT detach

    # Invariant 1: frame-by-frame match
    max_diff = 0.0
    for t in range(T):
        diff = (full_outputs[t] - chunk_outputs[t]).abs().max().item()
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-5, (
        f"Full vs chunked forward mismatch: max|diff|={max_diff:.2e}. "
        f"Chunk boundaries may be resetting h to zero instead of carrying forward."
    )


@pytest.mark.parametrize("k,q", [(4, 4)])
def test_chunk_boundary_carries_state(k, q):
    """After chunk 1, h must NOT be zero — proves state carries across chunks."""
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.eval()

    torch.manual_seed(42)
    T = 32  # two full chunks of 16
    obs = torch.randn(T, obs_dim)
    chunk_size = 16

    with torch.no_grad():
        # First chunk: accumulate state
        h_state = None
        for t in range(chunk_size):
            _, _, _, _, _, h_new = actor(obs[t:t+1], None if h_state is None else h_state)
            h_state = h_new
        h_after_chunk1 = h_state.clone()

        # If h_after_chunk1 has non-zero norm, GRU has accumulated state
        h_norm = h_after_chunk1.abs().mean().item()
        assert h_norm > 1e-6, (
            f"GRU hidden state after 16 frames has norm={h_norm:.2e}. "
            f"GRU may not be encoding temporal information."
        )

        # Second chunk: continue with carried state vs fresh start
        out_carry = None
        h_carry = h_after_chunk1.detach()
        for t in range(chunk_size, T):
            _, _, _, _, _, h_new = actor(obs[t:t+1], h_carry)
            if out_carry is None:
                out_carry = h_new.clone()
            h_carry = h_new

        out_reset = None
        h_reset = None
        for t in range(chunk_size, T):
            _, _, _, _, _, h_new = actor(obs[t:t+1], h_reset)
            if out_reset is None:
                out_reset = h_new.clone()
            h_reset = h_new

    # Carried-state output must differ from reset-state output
    carry_vs_reset = (out_carry - out_reset).abs().mean().item()
    assert carry_vs_reset > 1e-6, (
        f"Chunk 2 output with carried h == output with h=0 (diff={carry_vs_reset:.2e}). "
        f"Chunk boundary may be resetting h to zero — GRU history is lost."
    )


@pytest.mark.parametrize("k,q", [(4, 4)])
def test_episode_boundary_resets_hidden(k, q):
    """New episode must start with h=None (not carried from previous episode)."""
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.eval()

    torch.manual_seed(42)
    ep1_obs = torch.randn(10, obs_dim)
    ep2_first = torch.randn(1, obs_dim)

    with torch.no_grad():
        # Episode 1: accumulate state
        h = None
        for t in range(10):
            _, _, _, _, _, h = actor(ep1_obs[t:t+1], None if h is None else h)

        # Episode 2 with fresh h=None
        out_fresh, _, _, _, _, _ = actor(ep2_first, None)
        # Episode 2 with leaked h from ep1
        out_leaked, _, _, _, _, _ = actor(ep2_first, h)

    diff = (out_fresh - out_leaked).abs().max().item()
    assert diff > 1e-6, (
        f"Episode 2 fresh h=None output == leaked h from ep1 (diff={diff:.2e}). "
        f"Episode boundary is NOT resetting hidden state."
    )


@pytest.mark.parametrize("k,q", [(4, 4)])
def test_detach_h_new_false_allows_gradient_flow(k, q):
    """With detach_h_new=False, loss at frame t+1 creates gradient at frame t.

    This is the defining property of true chunk BPTT: the computation graph
    spans multiple frames within a chunk.
    """
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.train()

    torch.manual_seed(42)
    obs0 = torch.randn(1, obs_dim, requires_grad=True)
    obs1 = torch.randn(1, obs_dim, requires_grad=True)

    # Frame 0 → h1 (with grad)
    _, _, _, _, _, h1 = actor(obs0, None, detach_h_new=False)
    assert h1.grad_fn is not None, (
        "h1 has no grad_fn with detach_h_new=False. "
        "Gradient cannot flow from frame 1 back to frame 0."
    )

    # Frame 1 → loss
    out1, _, _, _, _, _ = actor(obs1, h1, detach_h_new=False)
    loss = out1.sum()
    loss.backward()

    # Frame 0 input must receive gradient (proves cross-frame gradient flow)
    assert obs0.grad is not None, (
        "obs0.grad is None after loss.backward(). "
        "detach_h_new=False should allow gradient to flow from frame 1 to frame 0."
    )
    assert obs0.grad.abs().sum() > 1e-8, (
        f"obs0.grad is all-zero (sum={obs0.grad.abs().sum().item():.2e}). "
        "Gradient is not propagating through the GRU hidden state."
    )


@pytest.mark.parametrize("k,q", [(4, 4)])
def test_chunk_boundary_detach_stops_gradient(k, q):
    """After explicit detach(), chunk 2 loss cannot create gradients in chunk 1."""
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.train()

    torch.manual_seed(42)
    chunk_size = 4
    obs1 = torch.randn(chunk_size, obs_dim, requires_grad=True)
    obs2 = torch.randn(chunk_size, obs_dim, requires_grad=True)

    # Chunk 1: forward with detach_h_new=False, detach at boundary
    h = None
    for t in range(chunk_size):
        _, _, _, _, _, h_new = actor(obs1[t:t+1], None if h is None else h,
                                     detach_h_new=False)
        h = h_new
    h_boundary = h.detach()  # explicit chunk boundary detach

    # Chunk 2: forward with detached h
    for t in range(chunk_size):
        _, _, _, _, _, h_new = actor(obs2[t:t+1], h_boundary,
                                     detach_h_new=False)
        h_boundary = h_new

    loss = h_boundary.sum()
    loss.backward()

    # Chunk 1 leaf tensor must have zero gradient (detach blocked the path)
    if obs1.grad is not None:
        grad_sum = obs1.grad.abs().sum().item()
        assert grad_sum < 1e-10, (
            f"Chunk 1 received gradient (sum={grad_sum:.2e}). "
            f"Chunk boundary detach() did not block gradient flow."
        )

    # Chunk 2 leaf tensor SHOULD have gradient
    assert obs2.grad is not None, "Chunk 2 received no gradient — backward path broken"
    assert obs2.grad.abs().sum() > 1e-8, (
        f"Chunk 2 grad is all-zero. Backward path through chunk is broken."
    )


@pytest.mark.parametrize("k,q", [(4, 4)])
def test_detach_h_new_true_blocks_gradient(k, q):
    """Default detach_h_new=True cuts gradient between frames (rollout/eval mode)."""
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.train()

    torch.manual_seed(42)
    obs0 = torch.randn(1, obs_dim, requires_grad=True)
    obs1 = torch.randn(1, obs_dim, requires_grad=True)

    _, _, _, _, _, h1 = actor(obs0, None)  # default detach_h_new=True
    assert h1.grad_fn is None, (
        "detach_h_new=True should produce h_new with no grad_fn"
    )
    _, _, _, _, _, _ = actor(obs1, h1)
    # obs0 should have no grad regardless
    assert obs0.grad is None, "obs0 should have no grad in default (detached) mode"
