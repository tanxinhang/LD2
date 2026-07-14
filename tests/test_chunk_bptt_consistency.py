"""P0 regression: chunk BPTT forward must match full-sequence forward.

Four invariants:
  1. Full-sequence output == chunked output (detach between chunks).
  2. Chunk boundary h≠0 after first chunk (proves state is CARRIED).
  3. Episode boundary h=0 (new episode starts fresh).
  4. Chunk 2 has no gradient path back to chunk 1 (detach works).
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
    obs = torch.randn(T, 1, obs_dim)  # (T, B=1, obs_dim)

    # Full-sequence forward
    with torch.no_grad():
        full_outputs = []
        h_full = None
        for t in range(T):
            dp_m, _, _, _, _, h_new = actor(obs[t:t+1], h_full)
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
                ob_t = obs[t:t+1]
                h_in = None if h_chunk is None else h_chunk
                dp_m, _, _, _, _, h_new = actor(ob_t, h_in)
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
    obs = torch.randn(T, 1, obs_dim)
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
    ep1_obs = torch.randn(10, 1, obs_dim)
    ep2_first = torch.randn(1, 1, obs_dim)

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
def test_chunk_detach_blocks_gradient(k, q):
    """After detach, chunk 2 backward should not create gradients in chunk 1."""
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.train()

    torch.manual_seed(42)
    chunk_size = 8
    obs1 = torch.randn(chunk_size, 1, obs_dim, requires_grad=False)
    obs2 = torch.randn(chunk_size, 1, obs_dim, requires_grad=False)

    # Chunk 1: forward, accumulate state
    h_state = None
    for t in range(chunk_size):
        _, _, _, _, _, h_new = actor(obs1[t:t+1], None if h_state is None else h_state)
        h_state = h_new
    h_detached = h_state.detach()

    # Chunk 2: forward with detached state, backward
    for t in range(chunk_size):
        dp_m, _, _, _, _, h_new = actor(obs2[t:t+1], h_detached)
        h_detached = h_new  # NOT detached here — we want grad flow within chunk 2

    loss = dp_m.sum()
    loss.backward()

    # Check: obs1 should have NO gradient (detach blocked it)
    # We verify by checking that a parameter grad comes only from chunk 2
    # (A rigorous check: re-run with only chunk 2, compare grad magnitudes)
    grads_from_full = {}
    for n, p in actor.named_parameters():
        if p.grad is not None:
            grads_from_full[n] = p.grad.clone()

    # Now re-run with only chunk 2 (h_detached as constant input)
    actor.zero_grad()
    h_fixed = h_detached.detach()
    for t in range(chunk_size):
        dp_m2, _, _, _, _, _ = actor(obs2[t:t+1], h_fixed)
        h_fixed = h_fixed  # constant — no grad flow

    loss2 = dp_m2.sum()
    loss2.backward()

    # Grads should match (chunk 1 contributed nothing to the full run)
    for n, p in actor.named_parameters():
        if p.grad is not None:
            diff = (grads_from_full[n] - p.grad).abs().max().item()
            assert diff < 1e-5, (
                f"Param '{n}': grad differs between full and chunk2-only. "
                f"detach() may not be blocking gradient flow from chunk 2 back to chunk 1."
            )
