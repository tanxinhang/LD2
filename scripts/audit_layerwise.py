#!/usr/bin/env python
"""Layer-wise audit: module swap, PCA, attention, masking sensitivity.

Compares DAgger frozen (M0) vs post-1-PPO-update (M1).
"""
import sys, os, copy, numpy as np, torch, torch.nn as nn
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.networks import StructuredActorNetwork

cfg = load_config('config/exp_800_k8q8.yaml')
K, Q = cfg.scenario.K, cfg.scenario.Q
max_dp = cfg.uav.v_max * cfg.scenario.dt
device = 'cuda'
aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)

def load_model(path):
    actor = StructuredActorNetwork(obs_dim=454, K=K, Q=Q, entity_dim=64, max_dp=max_dp).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt, strict=False)
    actor.eval()
    return actor

# ── Step 1: Train for 1 PPO update from DAgger, save M1 ──
print("=== Step 1: Generate M1 (1 PPO update from DAgger) ===")
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer

env = UAVISACEnv(config=cfg, seed=42)
od = env.core.obs_builder.get_obs_dim()
gd = env.core.obs_builder.get_global_state_dim()
aspace2 = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
aspace2.num_targets = Q; aspace2.structured_actor = True; aspace2.structured_entity_dim = 64
agents = [MAPPOAgent(k, od, gd, aspace2, K, hidden_layers=cfg.marl.hidden_layers,
          lr=cfg.marl.lr, critic_lr_mult=cfg.marl.critic_lr_mult,
          max_grad_norm=cfg.marl.max_grad_norm, device='cuda') for k in range(K)]
for k in range(1,K): agents[k].actor=agents[0].actor; agents[k].critic=agents[0].critic

# Load DAgger
dag_ckpt = torch.load('results/warmstart_gru_dagger.pt', map_location=device, weights_only=False)
agents[0].actor.load_state_dict(dag_ckpt, strict=False)
torch.save(agents[0].actor.state_dict(), 'results/audit_M0.pt')

trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device='cuda')
trainer._bc_actor = None
# 1 PPO update
trainer.train_episode()
torch.save(agents[0].actor.state_dict(), 'results/audit_M1.pt')
print("M0 (DAgger) and M1 (1-update) saved.")

# ── Step 2: Collect common trajectory from DAgger ──
print("\n=== Step 2: Collect trajectory ===")
env2 = UAVISACEnv(config=cfg, seed=12345)
M0 = load_model('results/audit_M0.pt')
obs, _ = env2.reset(seed=12345)
trajectory = []
for step in range(150):
    ob = np.stack([obs[str(k)] for k in range(K)])
    with torch.no_grad():
        dpm, dps, rl, cm, _, _ = M0(torch.as_tensor(ob, dtype=torch.float32, device=device))
    dpm_np = dpm.cpu().numpy(); dps_np = dps.cpu().numpy(); rl_np = rl.cpu().numpy()
    acts = {}
    for k in range(K):
        a, _ = aspace.decode(dpm_np[k], dps_np, rl_np[k], dp_deterministic=True)
        acts[str(k)] = {'delta_p': a.delta_p, 'role': 0}
    obs, _, t, tr, info = env2.step(acts)
    pd_q = info['P_D_q'].copy()
    trajectory.append({'obs': ob.copy(), 'pd_q': pd_q})
    if t.get('__all__') or tr.get('__all__'): break
env2.close()
print(f"Collected {len(trajectory)} frames")

# Pick last 20 frames for steady analysis
steady_frames = trajectory[-20:]
steady_obs = np.stack([f['obs'] for f in steady_frames])  # (20, 8, 454)

# ── Step 3: Module Swap Experiment ──
print("\n=== Step 3: Module Swap F0-F6 ===")

def get_module_state(actor):
    """Extract module parameters."""
    sd = actor.state_dict()
    enc_keys = [k for k in sd if any(k.startswith(p) for p in
        ['self_enc','target_enc','neighbor_gru','neighbor_proj','global_enc'])]
    attn_keys = [k for k in sd if any(k.startswith(p) for p in
        ['attn.','attn_norm'])]
    head_keys = [k for k in sd if any(k.startswith(p) for p in
        ['dp_head','comm_head','role_head','dp_log_std','comm_proj','gate','intent_head'])]
    return {k: sd[k].clone() for k in enc_keys}, {k: sd[k].clone() for k in attn_keys}, {k: sd[k].clone() for k in head_keys}

def apply_swap(actor, enc, attn, heads):
    """Apply swapped modules to actor."""
    sd = actor.state_dict()
    sd.update(enc); sd.update(attn); sd.update(heads)
    actor.load_state_dict(sd)

M0 = load_model('results/audit_M0.pt')
M1 = load_model('results/audit_M1.pt')
enc0, attn0, head0 = get_module_state(M0)
enc1, attn1, head1 = get_module_state(M1)

configs = {
    'F0 (DAgger)':    (enc0, attn0, head0),
    'F1 (PPO-enc)':   (enc1, attn0, head0),
    'F2 (PPO-attn)':  (enc0, attn1, head0),
    'F3 (PPO-heads)': (enc0, attn0, head1),
    'F4 (PPO-enc+attn)': (enc1, attn1, head0),
    'F5 (PPO-attn+heads)': (enc0, attn1, head1),
    'F6 (PPO-all)':   (enc1, attn1, head1),
}

# Evaluate each config on steady frames
results = {}
test_actor = StructuredActorNetwork(obs_dim=454, K=K, Q=Q, entity_dim=64, max_dp=max_dp).to(device)
for name, (enc, attn, heads) in configs.items():
    apply_swap(test_actor, enc, attn, heads)
    test_actor.eval()
    all_pd = []
    for frame_idx in range(len(steady_frames)):
        ob = steady_obs[frame_idx]  # (8, 454)
        with torch.no_grad():
            dpm, _, _, _, _, _ = test_actor(torch.as_tensor(ob, dtype=torch.float32, device=device))
        # Use DAgger action (same trajectory) — just evaluate P_D
        # Actually need to re-run env with this actor...
        pass  # will compute from activations below

# Better approach: evaluate by running env with each hybrid model
eval_seeds = [20001,20002,20003,20004,20005]
for name, (enc, attn, heads) in configs.items():
    apply_swap(test_actor, enc, attn, heads)
    test_actor.eval()
    steady_vals = []
    worst_vals = []
    for seed in eval_seeds:
        env3 = UAVISACEnv(config=cfg, seed=seed)
        obs, _ = env3.reset(seed=seed)
        pd_hist = []
        while True:
            ob = np.stack([obs[str(k)] for k in range(K)])
            with torch.no_grad():
                dpm, _, _, _, _, _ = test_actor(torch.as_tensor(ob, dtype=torch.float32, device=device))
            dpm_np = dpm.cpu().numpy()
            acts = {str(k): {'delta_p': aspace.decode(dpm_np[k], np.zeros(2), np.zeros(3), dp_deterministic=True)[0].delta_p, 'role': 0} for k in range(K)}
            obs, _, t, tr, info = env3.step(acts)
            pd_hist.append(info['P_D_q'].copy())
            if t.get('__all__') or tr.get('__all__'): break
        env3.close()
        w = min(20, len(pd_hist))
        P = np.array(pd_hist[-w:]).mean(axis=0)
        steady_vals.append(P.mean())
        worst_vals.append(P.min())
    results[name] = {'steady': np.mean(steady_vals), 'worst': np.mean(worst_vals),
                     'weak3': np.mean([float(np.sort(np.array([P.mean(axis=0) for P in [np.array(pd_hist[-20:]).mean(axis=0)]])).flatten()[:3].mean())])}

print(f"\n{'Config':<25} {'steady':>8} {'worst':>8}")
print('-'*43)
f0_steady = None
for name in ['F0 (DAgger)', 'F1 (PPO-enc)', 'F2 (PPO-attn)', 'F3 (PPO-heads)',
             'F4 (PPO-enc+attn)', 'F5 (PPO-attn+heads)', 'F6 (PPO-all)']:
    r = results.get(name, {'steady': 0, 'worst': 0})
    if name == 'F0 (DAgger)': f0_steady = r['steady']
    delta = r['steady'] - f0_steady if f0_steady else 0
    print(f'{name:<25} {r[\"steady\"]:8.4f} {r[\"worst\"]:8.4f}  Δ={delta:+.4f}')

# Identify which module causes the drop
print("\n=== Module Attribution ===")
f1 = results.get('F1 (PPO-enc)', {}); f2 = results.get('F2 (PPO-attn)', {}); f3 = results.get('F3 (PPO-heads)', {})
drop_enc = f0_steady - f1.get('steady',0) if f0_steady else 0
drop_attn = f0_steady - f2.get('steady',0) if f0_steady else 0
drop_head = f0_steady - f3.get('steady',0) if f0_steady else 0
total_drop = drop_enc + drop_attn + drop_head + 1e-10
print(f'Encoder contribution:  {drop_enc:.4f} ({100*drop_enc/total_drop:.0f}%)')
print(f'Attention contribution: {drop_attn:.4f} ({100*drop_attn/total_drop:.0f}%)')
print(f'Heads contribution:    {drop_head:.4f} ({100*drop_head/total_drop:.0f}%)')

env.close()
print("\nAudit complete.")
