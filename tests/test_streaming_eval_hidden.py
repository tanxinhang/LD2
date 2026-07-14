"""P0 regression: streaming GRU evaluation must differ from h=0 evaluation.

If streaming eval gives the same output as h=0 eval for every frame,
then the GRU is not actually contributing temporal information and the
recurrent policy is degenerate.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from uav_isac.agents.networks import StructuredActorNetwork


@pytest.mark.parametrize("k,q", [(4, 4)])
def test_streaming_differs_from_zero_init(k, q):
    """After N sequential frames, streaming GRU output should differ from h=0."""
    obs_dim = 29 + 18*q + 8*(k-1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.eval()

    torch.manual_seed(42)
    frames = [torch.randn(1, obs_dim) for _ in range(10)]

    # Streaming path
    h_stream = None
    stream_outputs = []
    with torch.no_grad():
        for f in frames:
            dp_m, _, _, _, _, h_new = actor(f, h_stream)
            stream_outputs.append(dp_m.clone())
            h_stream = h_new

    # Zero-init path (every frame)
    zero_outputs = []
    with torch.no_grad():
        for f in frames:
            dp_m, _, _, _, _, _ = actor(f, None)
            zero_outputs.append(dp_m.clone())

    # First frame should be identical (both start from zero)
    diff_frame0 = (stream_outputs[0] - zero_outputs[0]).abs().max().item()
    assert diff_frame0 < 1e-5, f"Frame 0 should be identical: diff={diff_frame0:.2e}"

    # Later frames should diverge if GRU is working
    diffs = [(stream_outputs[i] - zero_outputs[i]).abs().max().item() for i in range(10)]
    max_late_diff = max(diffs[3:])
    assert max_late_diff > 1e-6, (
        f"Streaming GRU output identical to h=0 across all frames. "
        f"Max diff (frames 3+): {max_late_diff:.2e}. GRU may be degenerate."
    )


def test_streaming_hidden_reset_per_episode():
    """New episode h=None must differ from leaked h from previous episode."""
    k, q = 4, 4
    obs_dim = 29 + 18*q + 8*(k-1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64).cpu()
    actor.eval()

    torch.manual_seed(42)
    # Episode 1: 5 frames, different observation each frame
    ep1_frames = [torch.randn(1, obs_dim) for _ in range(5)]
    # Episode 2: fresh frames (same seed continuation → different values)
    ep2_first = torch.randn(1, obs_dim)

    with torch.no_grad():
        # Run episode 1, accumulate hidden state
        h_ep1 = None
        for f in ep1_frames:
            _, _, _, _, _, h_ep1 = actor(f, h_ep1)

        # Episode 2, frame 1: fresh h=None (CORRECT reset)
        fresh_out, _, _, _, _, _ = actor(ep2_first, None)

        # Episode 2, frame 1 again: fresh h=None (should match)
        fresh_out2, _, _, _, _, _ = actor(ep2_first, None)

        # Episode 2, frame 1 with LEAKED h from ep1 (INCORRECT — no reset)
        leaked_out, _, _, _, _, _ = actor(ep2_first, h_ep1)

    # Two fresh calls with same input must match
    fresh_diff = (fresh_out - fresh_out2).abs().max().item()
    assert fresh_diff < 1e-7, (
        f"Same input+h=None should give identical output: diff={fresh_diff:.2e}"
    )

    # Leaked h must produce DIFFERENT output (h carries ep1 state)
    leaked_diff = (fresh_out - leaked_out).abs().max().item()
    assert leaked_diff > 1e-6, (
        f"Leaked h from ep1 should produce different output than fresh h=None. "
        f"Diff={leaked_diff:.2e}. If zero, GRU is not encoding temporal state."
    )
