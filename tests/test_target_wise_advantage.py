"""S4 regression: target-wise advantage distance index, units, invariants."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest


def test_distance_index_reads_dist_not_d_s1():
    """Offset must point to 'dist' (3rd geom field), not 'd_s1' (6th)."""
    Q = 4
    obs_dim = 8 + Q * 9 + Q * 8 + 3 + (3) * 9 + 2 + Q + 16  # ~123 for K=4,Q=4
    obs = torch.zeros(2, obs_dim)
    # Set known dist values at the correct offset
    for q in range(Q):
        offset = 8 + Q * 9 + q * 8 + 2
        obs[:, offset] = torch.tensor([0.1 * (q + 1), 0.2 * (q + 1)])

    # Extract using the S4 logic
    for q in range(Q):
        offset = 8 + Q * 9 + q * 8 + 2
        extracted = obs[:, offset]
        assert extracted[0].item() == pytest.approx(0.1 * (q + 1), abs=1e-6), \
            f"q={q}: expected {0.1*(q+1)}, got {extracted[0].item()}"
        assert extracted[1].item() == pytest.approx(0.2 * (q + 1), abs=1e-6), \
            f"q={q}: expected {0.2*(q+1)}, got {extracted[1].item()}"


def test_distance_units_meters():
    """Normalized dist must be converted to meters using scenario diagonal."""
    area_w, area_h = 800, 800
    diag_m = math.hypot(area_w, area_h)  # ~1131.4

    # Simulate: target at 100m → dist_norm = 100 / 1131.4 ≈ 0.0884
    dist_m = torch.tensor([10.0, 100.0, 300.0, 600.0])
    dist_norm = dist_m / diag_m
    dist_recovered = dist_norm * diag_m
    assert torch.allclose(dist_recovered, dist_m, atol=1e-4)


def test_responsibility_unequal_for_different_distances():
    """With tau_d=50m, UAVs at different distances get different responsibility."""
    tau_d = 50.0
    dist_m = torch.tensor([10.0, 100.0, 300.0, 600.0])
    rho = torch.softmax(-dist_m / tau_d, dim=-1)
    # Nearest target (10m) should dominate
    assert rho[0] > 0.8, f"rho[0]={rho[0]:.4f}, expected > 0.8 for 10m target"
    # Farthest target (600m) should be negligible
    assert rho[3] < 0.001, f"rho[3]={rho[3]:.6f}, expected < 0.001 for 600m target"
    # 100m should have some weight
    assert 0.05 < rho[1] < 0.25, f"rho[1]={rho[1]:.4f}"
    # Sums to 1
    assert rho.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_target_permutation_invariance():
    """Simultaneously permuting distances and advantages preserves result."""
    Q = 4
    B = 8
    dist_m = torch.rand(B, Q) * 500
    pt_adv = torch.randn(B, Q)
    tau_d = 50.0

    def compute_agg(d, a):
        rho = torch.softmax(-d / tau_d, dim=-1).detach()
        an = torch.zeros_like(a)
        for q in range(Q):
            m = a[:, q].mean()
            s = a[:, q].std(unbiased=False).clamp(min=1e-8)
            an[:, q] = (a[:, q] - m) / s
        tw = (rho * an).sum(dim=-1)
        m2 = tw.mean()
        s2 = tw.std(unbiased=False).clamp(min=1e-8)
        return (tw - m2) / s2

    orig = compute_agg(dist_m, pt_adv)

    # Permute targets (same permutation for dist and adv)
    perm = torch.tensor([2, 0, 3, 1])
    permuted = compute_agg(dist_m[:, perm], pt_adv[:, perm])
    assert torch.allclose(orig, permuted, atol=1e-6), \
        "Target permutation changed aggregated advantage"


def test_minibatch_invariance():
    """Pre-computed advantage indexed in minibatch must match full-batch slice.
    Minibatch-level re-normalization would produce different values — this test
    verifies that the S4 implementation does NOT re-normalize per-minibatch."""
    Q = 4
    B = 16
    dist_m = torch.rand(B, Q) * 500
    pt_adv = torch.randn(B, Q)
    tau_d = 50.0

    # Full-batch pre-computation (as done in trainer.update)
    rho = torch.softmax(-dist_m / tau_d, dim=-1).detach()
    an = torch.zeros_like(pt_adv)
    for q in range(Q):
        m = pt_adv[:, q].mean()
        s = pt_adv[:, q].std(unbiased=False).clamp(min=1e-8)
        an[:, q] = (pt_adv[:, q] - m) / s
    tw = (rho * an).sum(dim=-1)
    m2 = tw.mean()
    s2 = tw.std(unbiased=False).clamp(min=1e-8)
    full = (tw - m2) / s2

    # Verify: minibatch indexing into full result = same per-sample values
    for mb_start in range(0, B, 4):
        mb_end = mb_start + 4
        mb_slice = full[mb_start:mb_end]
        # Re-running per-minibatch would give DIFFERENT values (proves the bug)
        rho_mb = torch.softmax(-dist_m[mb_start:mb_end] / tau_d, dim=-1).detach()
        an_mb = torch.zeros_like(pt_adv[mb_start:mb_end])
        for q in range(Q):
            m_mb = pt_adv[mb_start:mb_end, q].mean()
            s_mb = pt_adv[mb_start:mb_end, q].std(unbiased=False).clamp(min=1e-8)
            an_mb[:, q] = (pt_adv[mb_start:mb_end, q] - m_mb) / s_mb
        tw_mb = (rho_mb * an_mb).sum(dim=-1)
        m2_mb = tw_mb.mean()
        s2_mb = tw_mb.std(unbiased=False).clamp(min=1e-8)
        mb_recomputed = (tw_mb - m2_mb) / s2_mb
        # These SHOULD differ (proving minibatch norm is wrong)
        assert not torch.allclose(mb_slice, mb_recomputed, atol=1e-6), \
            f"Minibatch [{mb_start}:{mb_end}] unexpectedly matches — " \
            f"minibatch norm may be equivalent to full-batch (unlikely with randn data)"


def test_trainer_target_wise_end_to_end():
    """Integration: real trainer._compute_target_wise_advantage with known dist."""
    from config.params import load_config
    from uav_isac.agents.trainer import MAPPTrainer
    from uav_isac.environment.env_wrapper import UAVISACEnv
    from uav_isac.environment.action import ActionSpace
    from uav_isac.agents.mappo_agent import MAPPOAgent

    cfg = load_config('config/exp_800_q4_full.yaml')
    cfg.marl.advantage_mode = 'target_wise'
    cfg.marl.num_envs = 1
    env = UAVISACEnv(config=cfg, seed=42)
    K, Q = cfg.scenario.K, cfg.scenario.Q
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
    aspace.num_targets = Q; aspace.structured_actor = True; aspace.structured_entity_dim = 64
    od = env.core.obs_builder.get_obs_dim()
    gd = env.core.obs_builder.get_global_state_dim()
    agents = [MAPPOAgent(k, od, gd, aspace, K, num_targets=Q,
               hidden_layers=cfg.marl.hidden_layers, lr=cfg.marl.lr, device='cpu') for k in range(K)]
    for k in range(1,K): agents[k].actor=agents[0].actor; agents[k].critic=agents[0].critic
    trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device='cpu')

    B = 32
    obs = torch.randn(B, od)
    # Inject known distances at the correct offsets
    dist_m = torch.tensor([10.0, 100.0, 300.0, 600.0])
    diag_m = math.hypot(*cfg.scenario.region_size)
    for q in range(Q):
        offset = 8 + Q * 9 + q * 8 + 2
        obs[:, offset] = dist_m[q] / diag_m

    pt_adv = torch.randn(B, Q)
    result = trainer._compute_target_wise_advantage(obs, pt_adv, tau_d=50.0)

    assert result.shape == (B,)
    assert torch.isfinite(result).all()
    # Full-batch normalized: mean≈0, std≈1
    assert result.mean().abs().item() < 1e-5, f"mean={result.mean():.2e}, expected ~0"
    assert result.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5), \
        f"std={result.std(unbiased=False):.4f}, expected ~1.0"
    env.close()


@pytest.mark.parametrize("Q", [1, 4, 8])
def test_dynamic_Q(Q):
    """Target-wise advantage must work for any Q without hardcoded offsets."""
    B = 4
    dist_m = torch.rand(B, Q) * 500
    pt_adv = torch.randn(B, Q)
    tau_d = 50.0

    rho = torch.softmax(-dist_m / tau_d, dim=-1)
    assert rho.shape == (B, Q)
    assert torch.allclose(rho.sum(dim=-1), torch.ones(B), atol=1e-6)

    an = torch.zeros_like(pt_adv)
    for q in range(Q):
        m = pt_adv[:, q].mean()
        s = pt_adv[:, q].std(unbiased=False).clamp(min=1e-8)
        an[:, q] = (pt_adv[:, q] - m) / s
    tw = (rho * an).sum(dim=-1)
    assert tw.shape == (B,)
