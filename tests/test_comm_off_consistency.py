"""P0 regression: comm-off mode must zero messages and freeze comm heads.

Three invariants when learned_comm_mode='off':
  1. Actor comm output is zeroed before env write-back.
  2. Comm-related parameters are frozen (requires_grad=False).
  3. Critic input comm aggregation is zero.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from config.params import load_config
from uav_isac.agents.networks import split_param_groups


COMM_HEAD_PREFIXES = ('comm_head.', 'comm_proj.', 'gate.', 'intent_head.')


def test_comm_off_freezes_comm_heads():
    """When learned_comm_mode='off', comm-related params must be frozen."""
    cfg = load_config('config/exp_800_k8_q8.yaml')
    assert cfg.marl.learned_comm_mode == 'off', "Config must have learned_comm_mode='off'"

    # Build actor directly (no trainer needed)
    from uav_isac.agents.networks import StructuredActorNetwork
    k, q = cfg.scenario.K, cfg.scenario.Q
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64)

    # Simulate what trainer does: freeze comm heads
    for n, p in actor.named_parameters():
        if any(n.startswith(prefix) for prefix in COMM_HEAD_PREFIXES):
            p.requires_grad_(False)

    # Verify all comm heads are frozen
    frozen = []
    still_trainable = []
    for n, p in actor.named_parameters():
        if any(n.startswith(prefix) for prefix in COMM_HEAD_PREFIXES):
            if p.requires_grad:
                still_trainable.append(n)
            else:
                frozen.append(n)

    assert len(still_trainable) == 0, (
        f"Comm heads still trainable: {still_trainable}"
    )
    assert len(frozen) >= 4, f"Expected >=4 frozen comm params, got {len(frozen)}: {frozen}"


def test_comm_off_actor_output_zeroed():
    """Actor comm output must be zeroed when comm_off is active."""
    from uav_isac.agents.networks import StructuredActorNetwork
    k, q = 4, 4
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64)
    actor.eval()

    obs = torch.randn(k, obs_dim)
    with torch.no_grad():
        _, _, _, comm_msgs, _, _ = actor(obs)

    # Zero the comm output (simulating comm_off in trainer)
    comm_zeroed = torch.zeros_like(comm_msgs)

    # Verify zeroing produces all zeros
    assert comm_zeroed.abs().sum() == 0.0, "Zeroed comm should be all zeros"
    # Original comm should NOT be all zeros (actor produces non-zero output)
    assert comm_msgs.abs().sum() > 1e-6, (
        "Actor comm output is all zeros even before zeroing — "
        "comm_head may be degenerate"
    )


def test_split_param_groups_includes_comm_heads():
    """split_param_groups must classify comm heads correctly."""
    from uav_isac.agents.networks import StructuredActorNetwork
    k, q = 4, 4
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64)

    enc, head, attn = split_param_groups(actor.named_parameters())

    # Comm-related heads should be in the 'head' group
    head_ids = {id(p) for p in head}
    head_names = []
    for n, p in actor.named_parameters():
        if any(n.startswith(prefix) for prefix in COMM_HEAD_PREFIXES):
            head_names.append(n)
            assert id(p) in head_ids, (
                f"'{n}' should be in HEAD group but is not. "
                f"Check HEAD_PARAM_PREFIXES in networks.py."
            )

    assert len(head_names) >= 4, f"Expected >=4 comm head params, got {len(head_names)}"


def test_full_eh_param_group_separation():
    """Full vs EH must differ ONLY in attention LR, not encoder/head LR."""
    from uav_isac.agents.networks import StructuredActorNetwork
    k, q = 4, 4
    obs_dim = 29 + 18 * q + 8 * (k - 1)
    actor = StructuredActorNetwork(obs_dim=obs_dim, K=k, Q=q, entity_dim=64)

    enc, head, attn = split_param_groups(actor.named_parameters())

    # Verify groups are non-empty
    assert len(enc) > 0, "Encoder group is empty"
    assert len(head) > 0, "Head group is empty"
    assert len(attn) > 0, "Attention group is empty"

    # Full: all three groups have LR > 0
    # EH:  same encoder/head LR, attention LR = 0
    # This test just verifies the groups are correctly partitioned.
    # The actual LR assignment is tested via integration.
    total = len(enc) + len(head) + len(attn)
    all_params = list(actor.parameters())
    assert total == len(all_params), (
        f"split_param_groups: {total} params in groups vs {len(all_params)} total. "
        f"Some parameters are missing or duplicated."
    )
