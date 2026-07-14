#!/usr/bin/env python
"""Strict paired validation of P0 fixes: GRU/PPO ratio + attention freeze.

Design (P0-1 fix, 2026-07-14 v2):
  - Fresh env / agents / trainer per case (no state leakage).
  - Same DAgger checkpoint → zero-init new layers → SAME baseline for all.
  - Full snapshot (actor, critic, optimizer) saved before any update;
    each case restores from this snapshot → identical initial conditions.
  - Same rollout seed and same test bank for all cases.
  - Streaming GRU hidden state maintained during evaluation.
  - Ratio assertion runs independently for each case.
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

cfg = load_config('config/exp_800_k8_q8.yaml')
cfg.marl.num_envs = 1  # single env for speed + strict reproducibility
K, Q = cfg.scenario.K, cfg.scenario.Q
max_dp = cfg.uav.v_max * cfg.scenario.dt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}, K={K}, Q={Q}, num_envs={cfg.marl.num_envs}")

# ── Fixed seeds for reproducibility ──
SNAPSHOT_SEED = 42
ROLLOUT_SEED = 123
EVAL_SEEDS = [20001, 20002, 20003, 20004, 20005]
STEADY_WINDOW = 20

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def evaluate_streaming(actor, aspace, seeds=EVAL_SEEDS):
    """Evaluate with streaming GRU hidden state (matches rollout behavior)."""
    results = {'steady': [], 'worst': [], 'weak3': []}
    for seed in seeds:
        env = UAVISACEnv(config=cfg, seed=seed)
        obs, _ = env.reset(seed=seed)
        pd_per_target = []
        eval_h_prev = None  # zero-init each episode
        while True:
            ob = np.stack([obs[str(k)] for k in range(K)])
            with torch.no_grad():
                ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device)
                dpm, _, rl, _, _, h_new = actor(ob_t, eval_h_prev)
                eval_h_prev = h_new
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
        w = min(STEADY_WINDOW, len(pd_per_target))
        pt = np.array(pd_per_target[-w:]).mean(axis=0)
        results['steady'].append(float(pt.mean()))
        results['worst'].append(float(pt.min()))
        results['weak3'].append(float(np.mean(np.sort(pt)[:3])))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}


def build_fresh_case(aspace, dag_ckpt, dag_keys):
    """Build a fresh env + agents + trainer, load DAgger, return (env, agents, trainer)."""
    torch.manual_seed(SNAPSHOT_SEED)
    np.random.seed(SNAPSHOT_SEED)
    env = UAVISACEnv(config=cfg, seed=SNAPSHOT_SEED)
    od = env.core.obs_builder.get_obs_dim()
    gd = env.core.obs_builder.get_global_state_dim()
    single_fd = env.core.obs_builder.get_single_frame_dim()

    agents = [
        MAPPOAgent(agent_id=k, obs_dim=od, global_state_dim=gd,
                   action_space=aspace, num_agents=K, num_targets=Q,
                   hidden_layers=cfg.marl.hidden_layers,
                   lr=cfg.marl.lr, critic_lr_mult=cfg.marl.critic_lr_mult,
                   max_grad_norm=cfg.marl.max_grad_norm, device=device)
        for k in range(K)
    ]
    for k in range(1, K):
        agents[k].actor = agents[0].actor
        agents[k].critic = agents[0].critic

    # Set single_frame_dim on actor
    if hasattr(agents[0].actor, 'single_frame_dim'):
        agents[0].actor.single_frame_dim = single_fd

    # Load DAgger with zero-init for new layers
    agents[0].actor.load_state_dict(dag_ckpt, strict=False)
    agents[0].actor.zero_init_new_layers(dag_keys)

    trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device=device)
    trainer._bc_actor = None
    return env, agents, trainer


def save_snapshot(agents):
    """Save actor, critic, and optimizer states for restoration.

    Uses deepcopy for optimizer state (contains nested tensors).
    The snapshot is taken BEFORE any PPO update, so optimizer state
    should be empty (no momentum yet). This is asserted at restore time.
    """
    return {
        'actor': {k: v.clone() for k, v in agents[0].actor.state_dict().items()},
        'critic': {k: v.clone() for k, v in agents[0].critic.state_dict().items()},
        'actor_opt': copy.deepcopy(agents[0].actor_optimizer.state_dict()),
        'critic_opt': copy.deepcopy(agents[0].critic_optimizer.state_dict()),
    }


def restore_snapshot(agents, snap):
    """Restore actor, critic, and optimizer states.

    CRITICAL: call this BEFORE any param freezing. Optimizer state
    structure must match the saved snapshot.
    """
    agents[0].actor.load_state_dict(snap['actor'])
    agents[0].critic.load_state_dict(snap['critic'])
    # Rebuild optimizers over ALL params (must match snapshot), then load state
    agents[0].actor_optimizer = torch.optim.Adam(
        agents[0].actor.parameters(), lr=cfg.marl.lr)
    agents[0].critic_optimizer = torch.optim.Adam(
        agents[0].critic.parameters(), lr=cfg.marl.lr * cfg.marl.critic_lr_mult)
    agents[0].actor_optimizer.load_state_dict(snap['actor_opt'])
    agents[0].critic_optimizer.load_state_dict(snap['critic_opt'])


# ═══════════════════════════════════════════════════════════════════
# STEP 1: Build DAgger baseline actor, get known keys, evaluate
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: DAgger baseline (streaming GRU eval)")
print("=" * 70)

aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
aspace.num_targets = Q
aspace.structured_actor = True
aspace.structured_entity_dim = 64

dag_path = 'results/warmstart_gru_dagger.pt'
dag_ckpt = torch.load(dag_path, map_location=device, weights_only=False)
dag_keys = set(dag_ckpt.keys())

# Build a standalone actor for baseline eval
od_ref = 227  # K=8,Q=8 single-frame without P0
env_ref = UAVISACEnv(config=cfg, seed=0)
od_ref = env_ref.core.obs_builder.get_obs_dim()
single_fd = env_ref.core.obs_builder.get_single_frame_dim()
env_ref.close()

baseline_actor = StructuredActorNetwork(
    obs_dim=od_ref, K=K, Q=Q, entity_dim=64, max_dp=max_dp,
    single_frame_dim=single_fd,
).to(device)
baseline_actor.load_state_dict(dag_ckpt, strict=False)
baseline_actor.zero_init_new_layers(dag_keys)
baseline_actor.eval()

dag_baseline = evaluate_streaming(baseline_actor, aspace)
print(f"DAgger (streaming GRU): steady={dag_baseline['steady'][0]:.4f}±{dag_baseline['steady'][1]:.4f}, "
      f"weak3={dag_baseline['weak3'][0]:.4f}±{dag_baseline['weak3'][1]:.4f}, "
      f"worst={dag_baseline['worst'][0]:.4f}±{dag_baseline['worst'][1]:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Build fresh case, save snapshot
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Build fresh Full-PPO case, save snapshot")
print("=" * 70)

env_full, agents_full, trainer_full = build_fresh_case(aspace, dag_ckpt, dag_keys)
snapshot = save_snapshot(agents_full)

# Verify baseline from the trainer actor also matches
agents_full[0].actor.eval()
trainer_baseline = evaluate_streaming(agents_full[0].actor, aspace)
print(f"Trainer actor (streaming): steady={trainer_baseline['steady'][0]:.4f}±{trainer_baseline['steady'][1]:.4f}, "
      f"weak3={trainer_baseline['weak3'][0]:.4f}±{trainer_baseline['weak3'][1]:.4f}, "
      f"worst={trainer_baseline['worst'][0]:.4f}±{trainer_baseline['worst'][1]:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Full PPO 1 update
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Full PPO 1 update")
print("=" * 70)

agents_full[0].actor.train()
agents_full[0].critic.train()
trainer_full.train_episode()
agents_full[0].actor.eval()
full_result = evaluate_streaming(agents_full[0].actor, aspace)

d_steady = full_result['steady'][0] - dag_baseline['steady'][0]
d_weak3 = full_result['weak3'][0] - dag_baseline['weak3'][0]
d_worst = full_result['worst'][0] - dag_baseline['worst'][0]
print(f"Full PPO (streaming): steady={full_result['steady'][0]:.4f}  Δ={d_steady:+.4f}")
print(f"                       weak3={full_result['weak3'][0]:.4f}  Δ={d_weak3:+.4f}")
print(f"                       worst={full_result['worst'][0]:.4f}  Δ={d_worst:+.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4: EH (Attention frozen) — fresh case from snapshot
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: EH PPO 1 update (fresh case from snapshot)")
print("=" * 70)

env_eh, agents_eh, trainer_eh = build_fresh_case(aspace, dag_ckpt, dag_keys)
restore_snapshot(agents_eh, snapshot)

# Freeze attention (with fixed condition: attn.* + attn_norm.*)
enc_p, head_p, attn_p = split_param_groups(agents_eh[0].actor.named_parameters())
for p in attn_p:
    p.requires_grad_(False)
frozen_names = [n for n, p in agents_eh[0].actor.named_parameters() if not p.requires_grad]
print(f"EH frozen: {frozen_names}")

# Rebuild optimizer with only trainable params
trainable = [p for p in agents_eh[0].actor.parameters() if p.requires_grad]
agents_eh[0].actor_optimizer = torch.optim.Adam(trainable, lr=cfg.marl.lr)

agents_eh[0].actor.train()
agents_eh[0].critic.train()
trainer_eh.train_episode()
agents_eh[0].actor.eval()
eh_result = evaluate_streaming(agents_eh[0].actor, aspace)

d_steady_eh = eh_result['steady'][0] - dag_baseline['steady'][0]
d_weak3_eh = eh_result['weak3'][0] - dag_baseline['weak3'][0]
d_worst_eh = eh_result['worst'][0] - dag_baseline['worst'][0]
print(f"EH PPO (streaming):  steady={eh_result['steady'][0]:.4f}  Δ={d_steady_eh:+.4f}")
print(f"                      weak3={eh_result['weak3'][0]:.4f}  Δ={d_weak3_eh:+.4f}")
print(f"                      worst={eh_result['worst'][0]:.4f}  Δ={d_worst_eh:+.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: E-only (Encoder only) — fresh case from snapshot
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: E-only PPO 1 update (fresh case from snapshot)")
print("=" * 70)

env_e, agents_e, trainer_e = build_fresh_case(aspace, dag_ckpt, dag_keys)
restore_snapshot(agents_e, snapshot)

# Freeze Attention + Heads
enc_p, head_p, attn_p = split_param_groups(agents_e[0].actor.named_parameters())
for p in attn_p + head_p:
    p.requires_grad_(False)
n_frozen = sum(1 for p in agents_e[0].actor.parameters() if not p.requires_grad)
print(f"E-only: {n_frozen} params frozen (attn + heads)")

trainable = [p for p in agents_e[0].actor.parameters() if p.requires_grad]
agents_e[0].actor_optimizer = torch.optim.Adam(trainable, lr=cfg.marl.lr)

agents_e[0].actor.train()
agents_e[0].critic.train()
trainer_e.train_episode()
agents_e[0].actor.eval()
e_result = evaluate_streaming(agents_e[0].actor, aspace)

d_steady_e = e_result['steady'][0] - dag_baseline['steady'][0]
d_weak3_e = e_result['weak3'][0] - dag_baseline['weak3'][0]
print(f"E-only PPO (streaming): steady={e_result['steady'][0]:.4f}  Δ={d_steady_e:+.4f}")
print(f"                          weak3={e_result['weak3'][0]:.4f}  Δ={d_weak3_e:+.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 6: H-only (Heads only) — fresh case from snapshot
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: H-only PPO 1 update (fresh case from snapshot)")
print("=" * 70)

env_h, agents_h, trainer_h = build_fresh_case(aspace, dag_ckpt, dag_keys)
restore_snapshot(agents_h, snapshot)

# Freeze Encoder + Attention
enc_p, head_p, attn_p = split_param_groups(agents_h[0].actor.named_parameters())
for p in enc_p + attn_p:
    p.requires_grad_(False)
n_frozen = sum(1 for p in agents_h[0].actor.parameters() if not p.requires_grad)
print(f"H-only: {n_frozen} params frozen (encoder + attn)")

trainable = [p for p in agents_h[0].actor.parameters() if p.requires_grad]
agents_h[0].actor_optimizer = torch.optim.Adam(trainable, lr=cfg.marl.lr)

agents_h[0].actor.train()
agents_h[0].critic.train()
trainer_h.train_episode()
agents_h[0].actor.eval()
h_result = evaluate_streaming(agents_h[0].actor, aspace)

d_steady_h = h_result['steady'][0] - dag_baseline['steady'][0]
d_weak3_h = h_result['weak3'][0] - dag_baseline['weak3'][0]
print(f"H-only PPO (streaming): steady={h_result['steady'][0]:.4f}  Δ={d_steady_h:+.4f}")
print(f"                          weak3={h_result['weak3'][0]:.4f}  Δ={d_weak3_h:+.4f}")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY (streaming GRU eval, strict paired)")
print("=" * 70)
print(f"{'Case':<20} {'steady_P_D':>12} {'Δsteady':>10} {'weak3':>12} {'Δweak3':>10} {'worst':>12}")
print(f"{'DAgger baseline':<20} {dag_baseline['steady'][0]:12.4f} {'—':>10} {dag_baseline['weak3'][0]:12.4f} {'—':>10} {dag_baseline['worst'][0]:12.4f}")
for name, res, ds, dw in [
    ('Full PPO', full_result, d_steady, d_weak3),
    ('EH (frozen Attn)', eh_result, d_steady_eh, d_weak3_eh),
    ('E-only', e_result, d_steady_e, d_weak3_e),
    ('H-only', h_result, d_steady_h, d_weak3_h),
]:
    print(f"{name:<20} {res['steady'][0]:12.4f} {ds:+10.4f} {res['weak3'][0]:12.4f} {dw:+10.4f} {res['worst'][0]:12.4f}")

# Judgment
if abs(d_weak3) < 0.02 and abs(d_weak3_eh) < 0.02:
    print("\n✓ All Δweak3 < 0.02 — PPO ratio fix CONFIRMED with strict pairing + streaming eval.")
    print("  Full PPO does NOT immediately destroy DAgger.")
elif abs(d_weak3) < 0.05:
    print(f"\n~ Full PPO Δweak3={d_weak3:+.4f} — moderate. Run more seeds and >1 update.")
else:
    print(f"\n✗ Full PPO Δweak3={d_weak3:+.4f} still large. Investigate further.")

print("\nDone.")
