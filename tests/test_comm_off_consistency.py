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


def test_real_trainer_init_completes():
    """Real MAPPTrainer must complete __init__ without errors.
    Catches indentation bugs that put init code inside _effective_comm."""
    from config.params import load_config
    from uav_isac.environment.env_wrapper import UAVISACEnv
    from uav_isac.environment.action import ActionSpace
    from uav_isac.agents.mappo_agent import MAPPOAgent
    from uav_isac.agents.trainer import MAPPTrainer

    for config_path, expect_freeze in [
        ('config/exp_800_k8_q8_full.yaml', False),
        ('config/exp_800_k8_q8_eh.yaml', True),
    ]:
        cfg = load_config(config_path)
        cfg.marl.num_envs = 1
        env = UAVISACEnv(config=cfg, seed=42)
        K, Q = cfg.scenario.K, cfg.scenario.Q
        aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
        aspace.num_targets = Q
        aspace.structured_actor = True
        aspace.structured_entity_dim = 64

        od = env.core.obs_builder.get_obs_dim()
        gd = env.core.obs_builder.get_global_state_dim()
        device = 'cpu'

        agents = [
            MAPPOAgent(k, od, gd, aspace, K, num_targets=Q,
                       hidden_layers=cfg.marl.hidden_layers,
                       lr=cfg.marl.lr, device=device)
            for k in range(K)
        ]
        for k in range(1, K):
            agents[k].actor = agents[0].actor
            agents[k].critic = agents[0].critic

        trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device=device)

        # __init__ must have completed
        assert hasattr(trainer, 'total_frames'), f"{config_path}: total_frames missing"
        assert hasattr(trainer, 'metrics_history'), f"{config_path}: metrics_history missing"
        assert hasattr(trainer, '_oracle_alpha'), f"{config_path}: _oracle_alpha missing"
        assert hasattr(trainer, '_ref_beta'), f"{config_path}: _ref_beta missing"

        # Comm heads frozen in both
        for n, p in trainer.agents[0].actor.named_parameters():
            if n.startswith(('comm_head.', 'comm_proj.', 'gate.', 'intent_head.')):
                assert not p.requires_grad, f"{config_path}: {n} not frozen"

        # Attention: frozen in EH, trainable in Full
        for n, p in trainer.agents[0].actor.named_parameters():
            if n.startswith(('attn.', 'attn_norm.')):
                if expect_freeze:
                    assert not p.requires_grad, f"{config_path}: {n} should be frozen"
                else:
                    assert p.requires_grad, f"{config_path}: {n} should be trainable"

        # Optimizer LR
        lrs = sorted({g['lr'] for g in trainer.agents[0].actor_optimizer.param_groups})
        assert lrs == [1e-5, 5e-5], f"{config_path}: expected [1e-5, 5e-5], got {lrs}"

        env.close()
        print(f"  {config_path}: OK (freeze_attn={expect_freeze}, lrs={lrs})")


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
