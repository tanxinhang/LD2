#!/usr/bin/env python
"""Validate P0 fixes: PPO ratio consistency + DAgger→1-PPO-update weak3 test.

Key questions:
  Q1: Does the old-log-prob consistency assertion pass? (ratio problem fixed?)
  Q2: Does Full PPO 1 update still destroy weak3? (if YES → ratio was the cause)
  Q3: How do EH, Encoder-only, Heads-only compare?
"""
import sys, os, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.networks import StructuredActorNetwork, split_param_groups
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer

# ── Config ──
cfg = load_config('config/exp_800_k8_q8.yaml')
K, Q = cfg.scenario.K, cfg.scenario.Q
max_dp = cfg.uav.v_max * cfg.scenario.dt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}, K={K}, Q={Q}")
print(f"Config: {cfg.marl.ppo_epochs} ppo_epochs, lr={cfg.marl.lr}, clip={cfg.marl.ppo_clip}")

# ── Eval helper ──
def evaluate(actor, aspace, env_seeds=[20001,20002,20003,20004,20005], steady_window=20):
    """Run deterministic eval on fixed seeds. Returns {steady, worst, weak3, per_target}."""
    results = {'steady': [], 'worst': [], 'weak3': [], 'per_target': []}
    for seed in env_seeds:
        env = UAVISACEnv(config=cfg, seed=seed)
        obs, _ = env.reset(seed=seed)
        pd_per_target = []
        while True:
            ob = np.stack([obs[str(k)] for k in range(K)])
            with torch.no_grad():
                dpm, _, rl, _, _, _ = actor(torch.as_tensor(ob, dtype=torch.float32, device=device))
            dpm_np = dpm.cpu().numpy()
            rl_np = rl.cpu().numpy()
            acts = {}
            for k in range(K):
                a, _ = aspace.decode(dpm_np[k], np.zeros(2), rl_np[k], dp_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': 0}
            obs, _, t, tr, info = env.step(acts)
            pd_per_target.append(info['P_D_q'].copy())
            if t.get('__all__') or tr.get('__all__'):
                break
        env.close()
        w = min(steady_window, len(pd_per_target))
        steady = np.array(pd_per_target[-w:])  # (w, Q)
        pt_mean = steady.mean(axis=0)          # (Q,)
        results['steady'].append(float(pt_mean.mean()))
        results['worst'].append(float(pt_mean.min()))
        sorted_q = np.sort(pt_mean)
        results['weak3'].append(float(np.mean(sorted_q[:3])))
        results['per_target'].append(pt_mean)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Load DAgger warmstart, eval baseline
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: DAgger baseline evaluation")
print("=" * 70)

aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
aspace.num_targets = Q
aspace.structured_actor = True
aspace.structured_entity_dim = 64

dag_path = 'results/warmstart_gru_dagger.pt'
dag_ckpt = torch.load(dag_path, map_location=device, weights_only=False)

# Build actor with NEW architecture (has pd_hist_proj, which old ckpt lacks)
obs_dim = 454  # K=8, Q=8 with 2-frame stacking
actor = StructuredActorNetwork(obs_dim=obs_dim, K=K, Q=Q, entity_dim=64, max_dp=max_dp).to(device)
missing, unexpected = actor.load_state_dict(dag_ckpt, strict=False)
print(f"DAgger checkpoint loaded: {len(missing)} new keys (pd_hist_proj etc), {len(unexpected)} unexpected")
actor.eval()

dag_baseline = evaluate(actor, aspace)
print(f"DAgger baseline: steady={dag_baseline['steady'][0]:.4f}±{dag_baseline['steady'][1]:.4f}, "
      f"weak3={dag_baseline['weak3'][0]:.4f}±{dag_baseline['weak3'][1]:.4f}, "
      f"worst={dag_baseline['worst'][0]:.4f}±{dag_baseline['worst'][1]:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Set up trainer with frozen DAgger actor, run 1 PPO update
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: 1 PPO update (Full) — testing PPO ratio consistency")
print("=" * 70)

env = UAVISACEnv(config=cfg, seed=42)
od = env.core.obs_builder.get_obs_dim()
gd = env.core.obs_builder.get_global_state_dim()
print(f"Actual obs_dim={od}, global_dim={gd}")

agents = [
    MAPPOAgent(
        agent_id=k, obs_dim=od, global_state_dim=gd,
        action_space=aspace, num_agents=K,
        num_targets=Q,
        hidden_layers=cfg.marl.hidden_layers,
        lr=cfg.marl.lr,
        critic_lr_mult=cfg.marl.critic_lr_mult,
        max_grad_norm=cfg.marl.max_grad_norm,
        device=device,
    )
    for k in range(K)
]
for k in range(1, K):
    agents[k].actor = agents[0].actor
    agents[k].critic = agents[0].critic

# Load DAgger weights
agents[0].actor.load_state_dict(dag_ckpt, strict=False)

# Create fresh trainer (this triggers the consistency assertion on first update)
trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device=device)
trainer._bc_actor = None  # disable BC anchor for clean test

# Run 1 PPO update — the consistency assertion fires here
print("\n--- Running 1 PPO update (Full) ---")
trainer.train_episode()

# Save post-PPO actor
post_ppo_state = {k: v.clone() for k, v in agents[0].actor.state_dict().items()}

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Eval post-1-update
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Post-1-PPO-update evaluation (Full)")
print("=" * 70)

agents[0].actor.eval()
post_ppo = evaluate(agents[0].actor, aspace)
d_steady = post_ppo['steady'][0] - dag_baseline['steady'][0]
d_weak3 = post_ppo['weak3'][0] - dag_baseline['weak3'][0]
d_worst = post_ppo['worst'][0] - dag_baseline['worst'][0]
print(f"Post-PPO (Full):  steady={post_ppo['steady'][0]:.4f}±{post_ppo['steady'][1]:.4f}  Δ={d_steady:+.4f}")
print(f"                   weak3={post_ppo['weak3'][0]:.4f}±{post_ppo['weak3'][1]:.4f}  Δ={d_weak3:+.4f}")
print(f"                   worst={post_ppo['worst'][0]:.4f}±{post_ppo['worst'][1]:.4f}  Δ={d_worst:+.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: EH-only update (freeze attention, test selective plasticity)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: EH-only 1 PPO update (Attention frozen)")
print("=" * 70)

# Reload DAgger
agents[0].actor.load_state_dict(dag_ckpt, strict=False)
agents[0].actor.train()  # must be in train mode for GRU backward

# Freeze attention (with the FIXED condition that includes attn_norm)
enc_params, head_params, attn_params = split_param_groups(agents[0].actor.named_parameters())
for p in attn_params:
    p.requires_grad_(False)
print(f"Frozen {len(attn_params)} attention params (incl. attn_norm): "
      f"{[n for n, p in agents[0].actor.named_parameters() if not p.requires_grad]}")

# Rebuild optimizer with only trainable params
trainable = [p for p in agents[0].actor.parameters() if p.requires_grad]
agents[0].actor_optimizer = torch.optim.Adam(trainable, lr=cfg.marl.lr)

# Run 1 PPO update
print("\n--- Running 1 PPO update (EH: Attention frozen) ---")
trainer.train_episode()

# Eval
agents[0].actor.eval()
post_eh = evaluate(agents[0].actor, aspace)
d_steady_eh = post_eh['steady'][0] - dag_baseline['steady'][0]
d_weak3_eh = post_eh['weak3'][0] - dag_baseline['weak3'][0]
print(f"Post-PPO (EH):    steady={post_eh['steady'][0]:.4f}±{post_eh['steady'][1]:.4f}  Δ={d_steady_eh:+.4f}")
print(f"                   weak3={post_eh['weak3'][0]:.4f}±{post_eh['weak3'][1]:.4f}  Δ={d_weak3_eh:+.4f}")
print(f"                   worst={post_eh['worst'][0]:.4f}±{post_eh['worst'][1]:.4f}  Δ={post_eh['worst'][0]-dag_baseline['worst'][0]:+.4f}")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'':<20} {'steady_P_D':>12} {'weak3':>12} {'worst':>12}")
print(f"{'DAgger baseline':<20} {dag_baseline['steady'][0]:12.4f} {dag_baseline['weak3'][0]:12.4f} {dag_baseline['worst'][0]:12.4f}")
print(f"{'Full PPO 1 upd':<20} {post_ppo['steady'][0]:12.4f} {post_ppo['weak3'][0]:12.4f} {post_ppo['worst'][0]:12.4f}")
print(f"{'EH PPO 1 upd':<20} {post_eh['steady'][0]:12.4f} {post_eh['weak3'][0]:12.4f} {post_eh['worst'][0]:12.4f}")
print(f"\n{'Δ Full - DAgger':<20} {d_steady:+12.4f} {d_weak3:+12.4f} {d_worst:+12.4f}")
print(f"{'Δ EH - DAgger':<20} {d_steady_eh:+12.4f} {d_weak3_eh:+12.4f} {d_worst_eh:+12.4f}")

if abs(d_weak3) < 0.03:
    print("\n✓ Full PPO weak3 change < 0.03 — PPO ratio fix RESOLVED the instant-destruction problem.")
    print("  Previous 'PPO systematically fails' conclusion should be downgraded to:")
    print("  'Old recurrent-state handling caused PPO update invalidity.'")
elif d_weak3 < -0.05:
    print(f"\n✗ Full PPO still destroys weak3 (Δ={d_weak3:+.4f}) despite ratio fix.")
    print("  → Problem is NOT only GRU/PPO state consistency. Investigate further.")
else:
    print(f"\n~ Full PPO weak3 change Δ={d_weak3:+.4f} — moderate. Run more seeds for significance.")

print("\nDone.")
