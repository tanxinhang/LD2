#!/usr/bin/env python
"""Train recurrent DAgger variants for local-PD ablation: D0 (none), D1 (local).

D0: PD_hist=zeros, comm=zeros  →  baseline: no detection history
D1: PD_hist=RX-only local, comm=zeros  →  local confidence alone

Both use the SAME StructuredActorNetwork (with pd_hist_proj, GRU).
The only difference is PD_hist/comm masking in the observation.

Key fixes over v1 (2026-07-14):
  - Student rollout uses streaming GRU (h_prev passed across frames).
  - Supervised training stores h_prev per frame, passes it during forward.
  - Validation (20 eps) and test (100 eps) are separate banks.
  - Episode-level metrics saved for paired bootstrap.
  - ep_fail reported at τ=0.3 (primary) and τ=0.05 (secondary).
  - D2 removed: communication training deferred to PPO stage.

Limitations (documented):
  - Single training seed (seed=42). Multi-seed requires separate runs.
  - Chunk-based BPTT not used; per-frame h_prev stored instead.
    This is correct for DAgger (independent teacher labels per frame).

Usage:
  python scripts/train_dagger_variants.py --mode all
  python scripts/train_dagger_variants.py --mode D0
"""
import sys, os, argparse, json, csv, time, copy
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
    """Apply PD/comm masking. PD_hist at [-16-Q:-16], comm at [-16:]."""
    out = obs_vector.copy()
    if mode == 'none':
        out[-(Q + 16):] = 0.0
    elif mode == 'local':
        out[-16:] = 0.0
    # 'local_comm' keeps everything (not used in DAgger; deferred to PPO)
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
# Data collection (with streaming GRU)
# ═══════════════════════════════════════════════════════════════════

def collect_teacher_rollout(cfg, n_eps, seed, Q, pd_mode):
    """Teacher rollout: record (modified_obs, teacher_dp, h_prev).

    Teacher doesn't use the actor, so h_prev is set to zeros for all frames.
    Supervision will also use h_prev=zeros, maintaining consistency.
    """
    K = cfg.scenario.K
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    obs_dim = cfg.marl.hidden_layers[0]  # placeholder; we read actual dim from env
    O, A, H = [], [], []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_eps):
        o, _ = env.reset(seed=seed + ep)
        # Per-episode GRU state: (K, K-1, D) — zero init each episode
        ep_h = np.zeros((K, K - 1, 64), dtype=np.float32)
        while True:
            dp = nearest_teacher_dp(env, K, Q, max_dp)
            for k in range(K):
                O.append(modify_obs(o[str(k)].copy(), Q, pd_mode))
                A.append(dp[k].copy())
                H.append(ep_h[k].copy())  # (K-1, D) per agent
            o, _, t, tr, _ = env.step(
                {str(k): {'delta_p': dp[k], 'role': 0} for k in range(K)})
            # Teacher GRU state stays at zero — no actor forward needed
            if t.get('__all__') or tr.get('__all__'):
                break
    env.close()
    Oa = np.asarray(O, dtype=np.float64)
    Aa = np.asarray(A, dtype=np.float64)
    Ha = np.asarray(H, dtype=np.float32)  # (N, K-1, D)
    return Oa, Aa, Ha


def collect_student_rollout(cfg, actor, device, n_eps, seed, Q, pd_mode):
    """Student rollout with streaming GRU.

    Actor sees modified obs + maintained h_prev. Records (obs, h_prev, teacher_label).
    """
    K = cfg.scenario.K
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    O, A, H = [], [], []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_eps):
        o, _ = env.reset(seed=seed + ep)
        # Per-episode streaming GRU hidden state
        ep_h = None  # None → zero-init on first frame
        while True:
            label = nearest_teacher_dp(env, K, Q, max_dp)
            ob_raw = np.stack([o[str(k)] for k in range(K)])
            ob_mod = np.stack([modify_obs(o[str(k)].copy(), Q, pd_mode) for k in range(K)])

            # Prepare h_prev for batched forward
            if ep_h is None:
                h_batch = None  # zero-init
            else:
                # ep_h: (K, K-1, D) → (1, K*(K-1), D)
                h_batch = torch.as_tensor(
                    ep_h.reshape(1, K * (K - 1), 64),
                    dtype=torch.float32, device=device)

            with torch.no_grad():
                dpm, dps, rl, _, _, h_new = actor(
                    torch.as_tensor(ob_mod, dtype=torch.float32, device=device), h_batch)

            # Save per-agent: obs, h_prev used, teacher action
            for k in range(K):
                # Per-agent h_prev: (K-1, D) from the ep_h array
                agent_h_prev = (ep_h[k].copy() if ep_h is not None
                                else np.zeros((K - 1, 64), dtype=np.float32))
                O.append(ob_mod[k].copy())
                A.append(label[k].copy())
                H.append(agent_h_prev)

            # Update streaming state for next frame
            h_new_np = h_new.cpu().numpy().reshape(K, K - 1, 64) if h_new is not None else None
            ep_h = h_new_np

            # Decode actions and step env
            dpm_np = dpm.cpu().numpy()
            dps_np = dps.cpu().numpy()
            rl_np = rl.cpu().numpy()
            acts = {}
            for k in range(K):
                a, _ = aspace.decode(dpm_np[k], dps_np, rl_np[k],
                                     dp_deterministic=True, role_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
            o, _, t, tr, _ = env.step(acts)
            if t.get('__all__') or tr.get('__all__'):
                break
    env.close()
    return (np.asarray(O, dtype=np.float64),
            np.asarray(A, dtype=np.float64),
            np.asarray(H, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════
# Recurrent supervised training
# ═══════════════════════════════════════════════════════════════════

def train_supervised_recurrent(actor, O, A, H, epochs, lr, max_dp, device):
    """MSE training with per-frame stored h_prev.

    Each sample = (obs, h_prev, teacher_action). During forward, stored
    h_prev is passed to actor, ensuring the same GRU condition as rollout.
    """
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    Ot = torch.as_tensor(O, dtype=torch.float32, device=device)
    At = torch.as_tensor(A, dtype=torch.float32, device=device)
    Ht = torch.as_tensor(H, dtype=torch.float32, device=device)  # (N, K-1, D)
    n, bs, dp_scale = Ot.shape[0], 256, float(max_dp)
    K_minus_1 = Ht.shape[1]
    D_gru = Ht.shape[2]
    last = 0.0
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, bs):
            mb = idx[s:s + bs]
            # Reshape h_prev: (mb, K-1, D) → (1, mb*(K-1), D)
            mb_h = Ht[mb].reshape(1, -1, D_gru)  # (1, mb*(K-1), D)
            dp_mean, _, _, _, _, _ = actor(Ot[mb], mb_h)
            pred = torch.tanh(dp_mean) * dp_scale
            loss = ((pred - At[mb]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(mb)
        last = tot / n
    return last


# ═══════════════════════════════════════════════════════════════════
# Evaluation (streaming GRU, episode-level metrics)
# ═══════════════════════════════════════════════════════════════════

def evaluate_streaming(cfg, actor, device, Q, pd_mode, seeds):
    """Evaluate with streaming GRU. Returns per-episode metrics list."""
    K = cfg.scenario.K
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    env = UAVISACEnv(config=cfg, seed=12345)
    episodes = []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        pd_per_target = []
        eval_h = None
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
        episodes.append({
            'seed': seed,
            'steady': float(pt.mean()),
            'worst': float(pt.min()),
            'weak3': float(np.mean(np.sort(pt)[:3])),
            'tstd': float(pt.std()),
            'per_target': pt.tolist(),
            'ep_fail_030': 1.0 if pt.min() < 0.3 else 0.0,
            'ep_fail_005': 1.0 if pt.min() < 0.05 else 0.0,
        })

    env.close()
    return episodes


def summarize_episodes(episodes):
    """Aggregate episode-level metrics into summary dict."""
    def _stat(key):
        vals = [e[key] for e in episodes]
        return float(np.mean(vals)), float(np.std(vals))
    return {
        'steady_mean': _stat('steady')[0], 'steady_std': _stat('steady')[1],
        'weak3_mean': _stat('weak3')[0], 'weak3_std': _stat('weak3')[1],
        'worst_mean': _stat('worst')[0], 'worst_std': _stat('worst')[1],
        'tstd_mean': _stat('tstd')[0],
        'ep_fail_030': float(np.mean([e['ep_fail_030'] for e in episodes])),
        'ep_fail_005': float(np.mean([e['ep_fail_005'] for e in episodes])),
        'n_episodes': len(episodes),
    }


def save_episode_csv(episodes, path):
    """Save per-episode metrics to CSV for paired bootstrap."""
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'seed', 'steady', 'worst', 'weak3', 'tstd', 'ep_fail_030', 'ep_fail_005'])
        writer.writeheader()
        for e in episodes:
            writer.writerow({
                'seed': e['seed'], 'steady': e['steady'], 'worst': e['worst'],
                'weak3': e['weak3'], 'tstd': e['tstd'],
                'ep_fail_030': e['ep_fail_030'], 'ep_fail_005': e['ep_fail_005'],
            })


# ═══════════════════════════════════════════════════════════════════
# Train one variant
# ═══════════════════════════════════════════════════════════════════

def train_one(cfg, pd_mode, device, args, val_seeds, test_seeds):
    """Train a single recurrent DAgger variant."""
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt

    tmp_env = UAVISACEnv(config=cfg, seed=0)
    obs_dim = tmp_env.core.obs_builder.get_obs_dim()
    single_fd = tmp_env.core.obs_builder.get_single_frame_dim()
    tmp_env.close()

    print(f"\n{'='*60}")
    print(f"Mode: {pd_mode}  K={K} Q={Q} obs_dim={obs_dim} single_frame_dim={single_fd}")
    print(f"Val seeds: {len(val_seeds)}  Test seeds: {len(test_seeds)}")
    print(f"{'='*60}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor = StructuredActorNetwork(
        obs_dim=obs_dim, K=K, Q=Q, entity_dim=64, max_dp=max_dp,
        single_frame_dim=single_fd,
    ).to(device)
    print(f"Actor params: {sum(p.numel() for p in actor.parameters())}")

    # Iter 0: teacher data
    t0 = time.time()
    O, A, H = collect_teacher_rollout(cfg, args.teacher_eps, args.seed, Q, pd_mode)
    print(f"Teacher dataset: {O.shape[0]} pairs ({time.time()-t0:.1f}s)")

    best_val_weak3 = -1.0
    best_state = None
    history = []

    for it in range(args.dagger_iters):
        # Supervised training (recurrent: uses stored h_prev)
        mse = train_supervised_recurrent(
            actor, O, A, H, args.sup_epochs, args.lr, max_dp, device)
        actor.eval()

        # Validation
        val_eps = evaluate_streaming(cfg, actor, device, Q, pd_mode, val_seeds)
        val_summary = summarize_episodes(val_eps)
        val_summary['iteration'] = it
        val_summary['dataset_size'] = O.shape[0]
        val_summary['sup_mse'] = float(mse)
        history.append(val_summary)

        print(f"  iter {it}: mse={mse:.4f}  "
              f"val_steady={val_summary['steady_mean']:.4f}±{val_summary['steady_std']:.4f}  "
              f"val_weak3={val_summary['weak3_mean']:.4f}±{val_summary['weak3_std']:.4f}  "
              f"val_worst={val_summary['worst_mean']:.4f}  "
              f"val_ep_fail_030={val_summary['ep_fail_030']:.3f}")

        # Checkpoint selection: max weak3, subject to steady not dropping
        if it == 0:
            base_steady = val_summary['steady_mean']
        if (val_summary['weak3_mean'] > best_val_weak3 and
                val_summary['steady_mean'] >= base_steady - 0.01):
            best_val_weak3 = val_summary['weak3_mean']
            best_state = {k: v.cpu().clone() for k, v in actor.state_dict().items()}
            best_iter = it

        actor.train()

        # Student aggregation (streaming GRU)
        sO, sA, sH = collect_student_rollout(
            cfg, actor, device, args.student_eps, args.seed + 1000 + it, Q, pd_mode)
        O = np.concatenate([O, sO])
        A = np.concatenate([A, sA])
        H = np.concatenate([H, sH])
        if O.shape[0] > args.max_pairs:
            O = O[-args.max_pairs:]
            A = A[-args.max_pairs:]
            H = H[-args.max_pairs:]

    # Restore best checkpoint (by val weak3)
    if best_state is not None:
        actor.load_state_dict(best_state)
        print(f"  restored best: iter={best_iter} val_weak3={best_val_weak3:.4f}")
    actor.eval()

    # Final TEST evaluation (separate bank, only after checkpoint frozen)
    test_eps = evaluate_streaming(cfg, actor, device, Q, pd_mode, test_seeds)
    test_summary = summarize_episodes(test_eps)

    print(f"\n  FINAL {pd_mode} (test, {len(test_seeds)} eps): "
          f"steady={test_summary['steady_mean']:.4f}±{test_summary['steady_std']:.4f}  "
          f"weak3={test_summary['weak3_mean']:.4f}±{test_summary['weak3_std']:.4f}  "
          f"worst={test_summary['worst_mean']:.4f}  "
          f"ep_fail_030={test_summary['ep_fail_030']:.3f}  "
          f"ep_fail_005={test_summary['ep_fail_005']:.3f}")

    return {
        'state_dict': {k: v.cpu().clone() for k, v in actor.state_dict().items()},
        'test_summary': test_summary,
        'test_episodes': test_eps,
        'val_history': history,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

MODES = {
    'D0': 'none',
    'D1': 'local',
}


def main():
    ap = argparse.ArgumentParser(description="Train recurrent DAgger variants D0/D1")
    ap.add_argument("--mode", default="all",
                    choices=["D0", "D1", "all"],
                    help="Which variant(s) to train")
    ap.add_argument("--config", default="config/exp_800_q4.yaml")
    ap.add_argument("--dagger-iters", type=int, default=5)
    ap.add_argument("--teacher-eps", type=int, default=60)
    ap.add_argument("--student-eps", type=int, default=40)
    ap.add_argument("--sup-epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-pairs", type=int, default=400000)
    ap.add_argument("--val-episodes", type=int, default=20)
    ap.add_argument("--test-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/dagger_variants")
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else get_default_config()
    cfg.marl.num_envs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Separate validation and test banks
    val_seeds = list(range(20001, 20001 + args.val_episodes))
    test_seeds = list(range(30001, 30001 + args.test_episodes))

    commit_hash = os.popen('git rev-parse HEAD').read().strip() if os.path.exists('.git') else 'unknown'

    print(f"Config: {args.config}  Device: {device}  Seed: {args.seed}")
    print(f"Commit: {commit_hash}")
    print(f"Validation: seeds {val_seeds[0]}-{val_seeds[-1]} ({len(val_seeds)} eps)")
    print(f"Test:       seeds {test_seeds[0]}-{test_seeds[-1]} ({len(test_seeds)} eps)")

    if args.mode == 'all':
        to_run = list(MODES.items())
    else:
        to_run = [(args.mode, MODES[args.mode])]

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}

    for mode_name, pd_mode in to_run:
        result = train_one(cfg, pd_mode, device, args, val_seeds, test_seeds)

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"dagger_{mode_name}.pt")
        torch.save(result['state_dict'], ckpt_path)
        print(f"  saved → {ckpt_path}")

        # Save episode-level test metrics
        csv_path = os.path.join(args.out_dir, f"test_episodes_{mode_name}.csv")
        save_episode_csv(result['test_episodes'], csv_path)
        print(f"  saved → {csv_path}")

        # Save validation history
        hist_path = os.path.join(args.out_dir, f"val_history_{mode_name}.csv")
        with open(hist_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'iteration', 'dataset_size', 'sup_mse',
                'steady_mean', 'steady_std', 'weak3_mean', 'weak3_std',
                'worst_mean', 'worst_std', 'ep_fail_030', 'ep_fail_005'])
            writer.writeheader()
            for h in result['val_history']:
                writer.writerow({k: h.get(k, '') for k in writer.fieldnames})
        print(f"  saved → {hist_path}")

        all_results[mode_name] = result

    # Save manifest
    manifest = {
        'git_commit': commit_hash,
        'config': args.config,
        'training_seed': args.seed,
        'K': cfg.scenario.K, 'Q': cfg.scenario.Q,
        'dagger_iters': args.dagger_iters,
        'teacher_eps': args.teacher_eps,
        'student_eps': args.student_eps,
        'val_seeds': f"{val_seeds[0]}-{val_seeds[-1]}",
        'test_seeds': f"{test_seeds[0]}-{test_seeds[-1]}",
        'protocol': 'recurrent DAgger (streaming GRU rollout + stored h_prev training)',
        'checkpoint_selection': 'max val weak3, steady >= base_steady - 0.01',
        'limitations': [
            'Single training seed (seed=42). Multi-seed requires separate runs.',
            'Per-frame stored h_prev, not chunk-based BPTT.',
            'Communication not trained; deferred to PPO stage.',
        ],
        'results': {
            k: {
                'pd_mode': v['test_summary'].get('pd_mode', MODES[k]),
                'steady_mean': v['test_summary']['steady_mean'],
                'steady_std': v['test_summary']['steady_std'],
                'weak3_mean': v['test_summary']['weak3_mean'],
                'weak3_std': v['test_summary']['weak3_std'],
                'worst_mean': v['test_summary']['worst_mean'],
                'ep_fail_030': v['test_summary']['ep_fail_030'],
                'ep_fail_005': v['test_summary']['ep_fail_005'],
            }
            for k, v in all_results.items()
        },
    }
    manifest_path = os.path.join(args.out_dir, "run_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}")

    # Comparison table
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"{'Variant':<8} {'steady':>10} {'weak3':>10} {'worst':>10} "
              f"{'ep_fail_030':>12} {'ep_fail_005':>12}")
        print(f"{'-'*60}")
        for name in ['D0', 'D1']:
            if name in all_results:
                m = all_results[name]['test_summary']
                print(f"{name:<8} {m['steady_mean']:10.4f} {m['weak3_mean']:10.4f} "
                      f"{m['worst_mean']:10.4f} {m['ep_fail_030']:12.3f} "
                      f"{m['ep_fail_005']:12.3f}")
        print(f"{'='*80}")
        print("Note: single training seed. Δ < per-ep std → not a significant difference.")
        print("D2 removed: communication training deferred to PPO stage.")
        print("ep_fail_030 = fraction of episodes with min_q P_D < 0.3 in steady window.")
        print("ep_fail_005 = fraction with min_q P_D < 0.05 (legacy threshold).")


if __name__ == "__main__":
    main()
