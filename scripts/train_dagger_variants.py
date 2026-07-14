#!/usr/bin/env python
"""Train DAgger variants for local-PD ablation: D0 (none), D1 (local), D2 (local+comm).

D0: PD_hist=zeros, comm=zeros  →  baseline: no detection history, no communication
D1: PD_hist=RX-only local, comm=zeros  →  local confidence alone
D2: PD_hist=RX-only local, comm=enabled  →  local confidence + neighbor propagation

All variants use the SAME StructuredActorNetwork architecture (with pd_hist_proj).
The only difference is what the actor sees in its observation during training AND eval.

Usage:
  python scripts/train_dagger_variants.py --mode all
  python scripts/train_dagger_variants.py --mode D0
  python scripts/train_dagger_variants.py --mode D1 --config config/exp_800_k8_q8.yaml
"""
import sys, os, argparse, json, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config.params import load_config, get_default_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.networks import StructuredActorNetwork


# ═══════════════════════════════════════════════════════════════════
# Observation modification
# ═══════════════════════════════════════════════════════════════════

def modify_obs(obs_vector: np.ndarray, Q: int, mode: str) -> np.ndarray:
    """Apply PD/comm masking to a single observation vector.

    obs layout: ... + PD_hist(Q) + comm_agg(16)
    The last (Q+16) dims are: [pd_0, ..., pd_{Q-1}, comm_0, ..., comm_15]

    Args:
        obs_vector: raw observation from env
        Q: number of targets
        mode: 'none' (D0), 'local' (D1), 'local_comm' (D2)

    Returns:
        modified observation (copy)
    """
    out = obs_vector.copy()
    if mode == 'none':
        # Zero both PD_hist and comm
        out[-(Q + 16):] = 0.0
    elif mode == 'local':
        # Keep PD_hist, zero comm only
        out[-16:] = 0.0
    elif mode == 'local_comm':
        # Keep everything
        pass
    return out


# ═══════════════════════════════════════════════════════════════════
# Teacher
# ═══════════════════════════════════════════════════════════════════

def nearest_teacher_dp(env, K, Q, max_dp):
    """Expert: each UAV homes toward its nearest true target."""
    tgt = np.array([t.get_position_3d() for t in env.core.targets])
    dp = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        pos = env.core.uavs[k].pos[:2]
        q = int(np.argmin([np.linalg.norm(tgt[qq][:2] - pos) for qq in range(Q)]))
        d = tgt[q][:2] - pos
        n = np.linalg.norm(d)
        dp[k] = d / n * max_dp if n > 1e-6 else np.zeros(2)
    return dp


# ═══════════════════════════════════════════════════════════════════
# Data collection
# ═══════════════════════════════════════════════════════════════════

def collect_teacher_rollout(cfg, n_eps, seed, Q, pd_mode):
    """Roll out teacher; record (modified_obs, teacher_dp)."""
    K = cfg.scenario.K
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    O, A = [], []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_eps):
        o, _ = env.reset(seed=seed + ep)
        while True:
            dp = nearest_teacher_dp(env, K, Q, max_dp)
            for k in range(K):
                O.append(modify_obs(o[str(k)].copy(), Q, pd_mode))
                A.append(dp[k].copy())
            o, _, t, tr, _ = env.step(
                {str(k): {'delta_p': dp[k], 'role': 0} for k in range(K)})
            if t.get('__all__') or tr.get('__all__'):
                break
    env.close()
    return np.asarray(O, dtype=np.float64), np.asarray(A, dtype=np.float64)


def collect_student_rollout(cfg, actor, device, n_eps, seed, Q, pd_mode):
    """Roll out student deterministically; label with teacher action at visited states."""
    K = cfg.scenario.K
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    O, A = [], []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_eps):
        o, _ = env.reset(seed=seed + ep)
        while True:
            label = nearest_teacher_dp(env, K, Q, max_dp)
            ob_raw = np.stack([o[str(k)] for k in range(K)])
            ob_mod = np.stack([modify_obs(o[str(k)].copy(), Q, pd_mode) for k in range(K)])
            with torch.no_grad():
                dpm, dps, rl, _, _, _ = actor(
                    torch.as_tensor(ob_mod, dtype=torch.float32, device=device))
            dpm = dpm.cpu().numpy()
            dps = dps.cpu().numpy()
            rl = rl.cpu().numpy()
            acts = {}
            for k in range(K):
                O.append(ob_mod[k].copy())
                A.append(label[k].copy())
                a, _ = aspace.decode(dpm[k], dps, rl[k],
                                     dp_deterministic=True, role_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
            o, _, t, tr, _ = env.step(acts)
            if t.get('__all__') or tr.get('__all__'):
                break
    env.close()
    return np.asarray(O, dtype=np.float64), np.asarray(A, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_supervised(actor, O, A, epochs, lr, max_dp, device):
    """MSE supervised learning on aggregated DAgger dataset."""
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    Ot = torch.as_tensor(O, dtype=torch.float32, device=device)
    At = torch.as_tensor(A, dtype=torch.float32, device=device)
    n, bs, dp_scale = Ot.shape[0], 256, float(max_dp)
    last = 0.0
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, bs):
            mb = idx[s:s + bs]
            dp_mean, _, _, _, _, _ = actor(Ot[mb])
            pred = torch.tanh(dp_mean) * dp_scale
            loss = ((pred - At[mb]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(mb)
        last = tot / n
    return last


# ═══════════════════════════════════════════════════════════════════
# Evaluation (streaming GRU)
# ═══════════════════════════════════════════════════════════════════

def evaluate_streaming(cfg, actor, device, Q, pd_mode, seeds):
    """Evaluate with streaming GRU hidden state and same obs modification as training."""
    K = cfg.scenario.K
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    env = UAVISACEnv(config=cfg, seed=12345)
    all_steady, all_worst, all_weak3, all_ep_fail = [], [], [], []
    all_per_target = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        pd_per_target = []
        eval_h = None  # streaming GRU
        while True:
            ob_raw = np.stack([obs[str(k)] for k in range(K)])
            ob_mod = np.stack([modify_obs(obs[str(k)].copy(), Q, pd_mode) for k in range(K)])
            with torch.no_grad():
                dpm, _, rl, _, _, h_new = actor(
                    torch.as_tensor(ob_mod, dtype=torch.float32, device=device), eval_h)
                eval_h = h_new
            dpm_np = dpm.cpu().numpy()
            rl_np = rl.cpu().numpy()
            acts = {}
            for k in range(K):
                a, _ = aspace.decode(dpm_np[k], np.zeros(2), rl_np[k],
                                     dp_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': 0}
            obs, _, t, tr, info = env.step(acts)
            pd_per_target.append(info['P_D_q'].copy())
            if t.get('__all__') or tr.get('__all__'):
                break

        w = min(20, len(pd_per_target))
        pt = np.array(pd_per_target[-w:]).mean(axis=0)  # (Q,)
        all_steady.append(float(pt.mean()))
        all_worst.append(float(pt.min()))
        all_weak3.append(float(np.mean(np.sort(pt)[:3])))
        all_ep_fail.append(1.0 if pt.min() < 0.05 else 0.0)
        all_per_target.append(pt)

    env.close()
    per_target_mat = np.array(all_per_target)
    return {
        'steady_mean': float(np.mean(all_steady)),
        'steady_std': float(np.std(all_steady)),
        'worst_mean': float(np.mean(all_worst)),
        'worst_std': float(np.std(all_worst)),
        'weak3_mean': float(np.mean(all_weak3)),
        'weak3_std': float(np.std(all_weak3)),
        'ep_fail_rate': float(np.mean(all_ep_fail)),
        'per_target_mean': per_target_mat.mean(axis=0).tolist(),
        'per_target_std': per_target_mat.std(axis=0).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def train_one(cfg, pd_mode, device, args):
    """Train a single DAgger variant. Returns (actor_state_dict, metrics)."""
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt

    # Get obs dim and single-frame dim from a fresh env
    tmp_env = UAVISACEnv(config=cfg, seed=0)
    obs_dim = tmp_env.core.obs_builder.get_obs_dim()
    single_fd = tmp_env.core.obs_builder.get_single_frame_dim()
    tmp_env.close()

    print(f"\n{'='*60}")
    print(f"Mode: {pd_mode}  K={K} Q={Q} obs_dim={obs_dim} single_frame_dim={single_fd}")
    print(f"{'='*60}")

    # Build actor with pd_hist_proj (trained from scratch — no old checkpoint)
    torch.manual_seed(args.seed)
    actor = StructuredActorNetwork(
        obs_dim=obs_dim, K=K, Q=Q, entity_dim=64, max_dp=max_dp,
        single_frame_dim=single_fd,
    ).to(device)
    print(f"Actor params: {sum(p.numel() for p in actor.parameters())}")

    eval_seeds = list(range(20001, 20001 + args.test_episodes))

    # Iter 0: teacher data
    t0 = time.time()
    O, A = collect_teacher_rollout(cfg, args.teacher_eps, args.seed, Q, pd_mode)
    print(f"Teacher dataset: {O.shape[0]} pairs ({time.time()-t0:.1f}s)")

    best_pd = -1.0
    best_state = None
    history = []

    for it in range(args.dagger_iters):
        mse = train_supervised(actor, O, A, args.sup_epochs, args.lr, max_dp, device)
        actor.eval()
        metrics = evaluate_streaming(cfg, actor, device, Q, pd_mode, eval_seeds)
        actor.train()
        metrics['iteration'] = it
        metrics['dataset_size'] = O.shape[0]
        metrics['sup_mse'] = float(mse)
        history.append(metrics)

        print(f"  iter {it}: mse={mse:.4f}  steady={metrics['steady_mean']:.4f}±{metrics['steady_std']:.4f}  "
              f"weak3={metrics['weak3_mean']:.4f}±{metrics['weak3_std']:.4f}  "
              f"worst={metrics['worst_mean']:.4f}  ep_fail={metrics['ep_fail_rate']:.3f}")

        if metrics['steady_mean'] > best_pd:
            best_pd = metrics['steady_mean']
            best_state = {k: v.cpu().clone() for k, v in actor.state_dict().items()}

        # Student aggregation
        sO, sA = collect_student_rollout(cfg, actor, device, args.student_eps,
                                          args.seed + 1000 + it, Q, pd_mode)
        O = np.concatenate([O, sO])
        A = np.concatenate([A, sA])
        if O.shape[0] > args.max_pairs:
            O = O[-args.max_pairs:]
            A = A[-args.max_pairs:]

    # Restore best
    if best_state is not None:
        actor.load_state_dict(best_state)
    actor.eval()

    # Final evaluation
    final = evaluate_streaming(cfg, actor, device, Q, pd_mode, eval_seeds)
    print(f"\n  FINAL {pd_mode}: steady={final['steady_mean']:.4f}±{final['steady_std']:.4f}  "
          f"weak3={final['weak3_mean']:.4f}±{final['weak3_std']:.4f}  "
          f"worst={final['worst_mean']:.4f}  ep_fail={final['ep_fail_rate']:.3f}")

    return {
        'state_dict': {k: v.cpu().clone() for k, v in actor.state_dict().items()},
        'final_metrics': final,
        'history': history,
    }


def main():
    ap = argparse.ArgumentParser(description="Train DAgger variants D0/D1/D2")
    ap.add_argument("--mode", default="all",
                    choices=["D0", "D1", "D2", "all"],
                    help="Which variant(s) to train")
    ap.add_argument("--config", default="config/exp_800_q4.yaml")
    ap.add_argument("--dagger-iters", type=int, default=5)
    ap.add_argument("--teacher-eps", type=int, default=60)
    ap.add_argument("--student-eps", type=int, default=40)
    ap.add_argument("--sup-epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-pairs", type=int, default=400000)
    ap.add_argument("--test-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/dagger_variants")
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else get_default_config()
    cfg.marl.num_envs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Config: {args.config}  Device: {device}  Seed: {args.seed}")

    MODES = {
        'D0': 'none',
        'D1': 'local',
        'D2': 'local_comm',
    }
    if args.mode == 'all':
        to_run = list(MODES.items())
    else:
        to_run = [(args.mode, MODES[args.mode])]

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}

    for mode_name, pd_mode in to_run:
        result = train_one(cfg, pd_mode, device, args)
        all_results[mode_name] = {
            'pd_mode': pd_mode,
            'final_metrics': result['final_metrics'],
            'history': result['history'],
        }
        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"dagger_{mode_name}.pt")
        torch.save(result['state_dict'], ckpt_path)
        print(f"  saved → {ckpt_path}")

    # Save summary
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'config': args.config,
            'git_commit': os.popen('git rev-parse HEAD').read().strip() if os.path.exists('.git') else 'unknown',
            'seed': args.seed,
            'K': cfg.scenario.K,
            'Q': cfg.scenario.Q,
            'pd_modes': {k: v['pd_mode'] for k, v in all_results.items()},
            'results': {
                k: {
                    'steady': v['final_metrics']['steady_mean'],
                    'weak3': v['final_metrics']['weak3_mean'],
                    'worst': v['final_metrics']['worst_mean'],
                    'ep_fail': v['final_metrics']['ep_fail_rate'],
                    'per_target': v['final_metrics']['per_target_mean'],
                }
                for k, v in all_results.items()
            },
        }, f, indent=2)
    print(f"\nSummary → {summary_path}")

    # Comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"{'Variant':<8} {'steady':>10} {'weak3':>10} {'worst':>10} {'ep_fail':>10}")
        print(f"{'-'*48}")
        for name in ['D0', 'D1', 'D2']:
            if name in all_results:
                m = all_results[name]['final_metrics']
                print(f"{name:<8} {m['steady_mean']:10.4f} {m['weak3_mean']:10.4f} "
                      f"{m['worst_mean']:10.4f} {m['ep_fail_rate']:10.3f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
