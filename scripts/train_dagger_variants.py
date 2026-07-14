#!/usr/bin/env python
"""Train recurrent DAgger variants for local-PD ablation: D0 (none), D1 (local).

D0: PD_hist=zeros, comm=zeros  →  baseline: no detection history
D1: PD_hist=RX-only local, comm=zeros  →  local confidence alone

Both use the SAME StructuredActorNetwork (with pd_hist_proj, GRU).

Protocol (v3, 2026-07-14):
  - Student rollout uses streaming GRU (h_prev across frames, reset per ep).
  - Data saved as episode sequences: [(o_1,a_1), (o_2,a_2), ...] per episode.
  - Training: chunk-based truncated BPTT (chunk_size=16).
    h=0 ONLY at episode boundary. Between chunks within an episode:
    h_next = detach(h_prev_chunk_end) — preserves state value, cuts gradient.
    Optimizer steps once per episode (all chunks within an episode see the
    same parameters). This eliminates stored-h_prev staleness from old policy.
  - Hidden drift diagnostic measured after each DAgger iteration.
  - D2 removed: communication training deferred to PPO stage.

Usage:
  python scripts/train_dagger_variants.py --mode all
  python scripts/train_dagger_variants.py --mode D0 --chunk-size 16
"""
import sys, os, argparse, json, csv, time
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
    out = obs_vector.copy()
    if mode == 'none':
        out[-(Q + 16):] = 0.0
    elif mode == 'local':
        out[-16:] = 0.0
    return out


# ═══════════════════════════════════════════════════════════════════
# Teacher
# ═══════════════════════════════════════════════════════════════════

def nearest_teacher_dp(env, K, Q, max_dp):
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
# Episode-based data collection
# ═══════════════════════════════════════════════════════════════════

def collect_teacher_episodes(cfg, n_eps, seed, Q, pd_mode):
    """Teacher rollout: return list of episodes, each = [(obs_k, dp_k), ...].

    Teacher uses true target positions; GRU state not needed (labels are
    independent of actor). During training, chunks start with h=0.
    """
    K = cfg.scenario.K
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    episodes = []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_eps):
        o, _ = env.reset(seed=seed + ep)
        frames = []  # list of (obs_array (K, obs_dim), dp_array (K, 2))
        while True:
            dp = nearest_teacher_dp(env, K, Q, max_dp)
            obs_arr = np.stack([modify_obs(o[str(k)].copy(), Q, pd_mode) for k in range(K)])
            frames.append((obs_arr.astype(np.float64), dp.copy()))
            o, _, t, tr, _ = env.step(
                {str(k): {'delta_p': dp[k], 'role': 0} for k in range(K)})
            if t.get('__all__') or tr.get('__all__'):
                break
        if frames:
            episodes.append(frames)
    env.close()
    return episodes


def collect_student_episodes(cfg, actor, device, n_eps, seed, Q, pd_mode):
    """Student rollout with streaming GRU.

    Returns list of episodes. Each episode = [(obs_arr (K,obs_dim), teacher_dp (K,2)), ...].
    The actor drives the trajectory using its own streaming GRU state.
    """
    K = cfg.scenario.K
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    episodes = []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_eps):
        o, _ = env.reset(seed=seed + ep)
        frames = []
        ep_h = None  # streaming GRU: None → zero-init first frame
        while True:
            label = nearest_teacher_dp(env, K, Q, max_dp)
            ob_mod = np.stack([modify_obs(o[str(k)].copy(), Q, pd_mode) for k in range(K)])

            if ep_h is None:
                h_batch = None
            else:
                h_batch = torch.as_tensor(
                    ep_h.reshape(1, K * (K - 1), 64), dtype=torch.float32, device=device)

            with torch.no_grad():
                dpm, dps, rl, _, _, h_new = actor(
                    torch.as_tensor(ob_mod, dtype=torch.float32, device=device), h_batch)

            frames.append((ob_mod.astype(np.float64), label.copy()))

            h_new_np = h_new.cpu().numpy().reshape(K, K - 1, 64) if h_new is not None else None
            ep_h = h_new_np

            dpm_np = dpm.cpu().numpy(); dps_np = dps.cpu().numpy(); rl_np = rl.cpu().numpy()
            acts = {}
            for k in range(K):
                a, _ = aspace.decode(dpm_np[k], dps_np, rl_np[k],
                                     dp_deterministic=True, role_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
            o, _, t, tr, _ = env.step(acts)
            if t.get('__all__') or tr.get('__all__'):
                break
        if frames:
            episodes.append(frames)
    env.close()
    return episodes


# ═══════════════════════════════════════════════════════════════════
# Chunk-based recurrent training (truncated BPTT)
# ═══════════════════════════════════════════════════════════════════

def train_chunk_bptt(actor, episodes, epochs, lr, max_dp, device, chunk_size=16):
    """Chunk-based truncated BPTT training.

    Each episode: h_0 = 0 at episode boundary.
    Chunk i: forward frames [i*L, (i+1)*L) with h_in from previous chunk end.
    Actor internally detaches h_new (networks.py:329) — no explicit detach needed.
    Optimizer steps once per episode (all chunks see same parameters).
    """
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    dp_scale = float(max_dp)

    all_obs, all_act = [], []
    for ep in episodes:
        all_obs.append(np.stack([f[0] for f in ep]))
        all_act.append(np.stack([f[1] for f in ep]))

    n_episodes = len(all_obs)
    K = all_obs[0].shape[1]

    for epoch in range(epochs):
        ep_order = np.random.permutation(n_episodes)
        total_loss = 0.0
        total_frames = 0

        for ep_idx in ep_order:
            ep_obs = all_obs[ep_idx]
            ep_act = all_act[ep_idx]
            T = ep_obs.shape[0]

            opt.zero_grad()
            h_state = None  # h=0 ONLY at episode boundary
            ep_loss = 0.0
            ep_frames = 0

            for chunk_start in range(0, T, chunk_size):
                chunk_end = min(chunk_start + chunk_size, T)
                chunk_obs = torch.as_tensor(
                    ep_obs[chunk_start:chunk_end], dtype=torch.float32, device=device)
                chunk_act = torch.as_tensor(
                    ep_act[chunk_start:chunk_end], dtype=torch.float32, device=device)
                L = chunk_obs.shape[0]

                preds = []
                for t in range(L):
                    ob_t = chunk_obs[t:t+1]
                    h_in = None if h_state is None else h_state
                    dp_mean, _, _, _, _, h_new = actor(ob_t, h_in)
                    preds.append(torch.tanh(dp_mean) * dp_scale)
                    h_state = h_new  # actor already detaches internally (networks.py:329)

                pred = torch.cat(preds, dim=0)
                # Accumulate weighted loss: sum of per-frame squared errors
                chunk_loss = ((pred - chunk_act) ** 2).sum()  # sum, not mean
                ep_loss += chunk_loss
                ep_frames += L

            # One optimizer step per episode — hidden state was produced by
            # the SAME parameters throughout the episode.
            (ep_loss / ep_frames).backward()
            opt.step()

            total_loss += ep_loss.item()
            total_frames += ep_frames

    return total_loss / max(total_frames, 1)


def measure_hidden_drift(actor, episodes, device, chunk_size=16):
    """Diagnostic: compare stored student-rollout h (from old policy) with
    recomputed h (from current actor) on the same observation sequences.

    Returns mean relative drift: |h_stored - h_recomputed| / (|h_recomputed| + ε).
    Large values indicate stored h_prev staleness.
    """
    if not episodes:
        return 0.0

    K = episodes[0][0][0].shape[0]
    drifts = []
    n_samples = 0

    with torch.no_grad():
        for ep in episodes[:min(5, len(episodes))]:  # sample 5 episodes
            obs_arr = np.stack([f[0] for f in ep])
            T = obs_arr.shape[0]
            h_state = None
            for t in range(T):
                ob_t = torch.as_tensor(obs_arr[t:t+1], dtype=torch.float32, device=device)
                _, _, _, _, _, h_new = actor(ob_t, h_state if h_state is not None else None)
                if h_state is not None and h_new is not None:
                    drift = (h_state - h_new).abs().mean().item()
                    norm = h_new.abs().mean().item() + 1e-8
                    drifts.append(drift / norm)
                    n_samples += 1
                h_state = h_new.detach() if h_new is not None else None

    return float(np.mean(drifts)) if drifts else 0.0


# ═══════════════════════════════════════════════════════════════════
# Evaluation (streaming GRU, episode-level metrics)
# ═══════════════════════════════════════════════════════════════════

def evaluate_streaming(cfg, actor, device, Q, pd_mode, seeds):
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
            ob_mod = np.stack([modify_obs(obs[str(k)].copy(), Q, pd_mode) for k in range(K)])
            with torch.no_grad():
                dpm, _, rl, _, _, h_new = actor(
                    torch.as_tensor(ob_mod, dtype=torch.float32, device=device), eval_h)
            eval_h = h_new
            dpm_np = dpm.cpu().numpy(); rl_np = rl.cpu().numpy()
            acts = {}
            for k in range(K):
                a, _ = aspace.decode(dpm_np[k], np.zeros(2), rl_np[k], dp_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': 0}
            obs, _, t, tr, info = env.step(acts)
            pd_per_target.append(info['P_D_q'].copy())
            if t.get('__all__') or tr.get('__all__'):
                break
        w = min(20, len(pd_per_target))
        pt = np.array(pd_per_target[-w:]).mean(axis=0)
        episodes.append({
            'seed': seed, 'steady': float(pt.mean()), 'worst': float(pt.min()),
            'weak3': float(np.mean(np.sort(pt)[:3])), 'tstd': float(pt.std()),
            'per_target': pt.tolist(),
            'ep_fail_030': 1.0 if pt.min() < 0.3 else 0.0,
            'ep_fail_005': 1.0 if pt.min() < 0.05 else 0.0,
        })
    env.close()
    return episodes


def summarize_episodes(episodes):
    def _stat(key):
        vals = [e[key] for e in episodes]
        return float(np.mean(vals)), float(np.std(vals))
    return {
        'steady_mean': _stat('steady')[0], 'steady_std': _stat('steady')[1],
        'weak3_mean': _stat('weak3')[0], 'weak3_std': _stat('weak3')[1],
        'worst_mean': _stat('worst')[0], 'worst_std': _stat('worst')[1],
        'ep_fail_030': float(np.mean([e['ep_fail_030'] for e in episodes])),
        'ep_fail_005': float(np.mean([e['ep_fail_005'] for e in episodes])),
    }


def save_episode_csv(episodes, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'seed', 'steady', 'worst', 'weak3', 'tstd', 'ep_fail_030', 'ep_fail_005'])
        w.writeheader()
        for e in episodes:
            w.writerow({k: e[k] for k in w.fieldnames})


# ═══════════════════════════════════════════════════════════════════
# Train one variant
# ═══════════════════════════════════════════════════════════════════

def train_one(cfg, pd_mode, device, args, val_seeds, test_seeds):
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt

    tmp_env = UAVISACEnv(config=cfg, seed=0)
    obs_dim = tmp_env.core.obs_builder.get_obs_dim()
    single_fd = tmp_env.core.obs_builder.get_single_frame_dim()
    tmp_env.close()

    print(f"\n{'='*60}")
    print(f"Mode: {pd_mode}  K={K} Q={Q} obs_dim={obs_dim}")
    print(f"Chunk size: {args.chunk_size}  Val: {len(val_seeds)} eps  Test: {len(test_seeds)} eps")
    print(f"{'='*60}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    actor = StructuredActorNetwork(
        obs_dim=obs_dim, K=K, Q=Q, entity_dim=64, max_dp=max_dp,
        single_frame_dim=single_fd).to(device)
    print(f"Actor params: {sum(p.numel() for p in actor.parameters())}")

    # Iter 0: teacher data
    t0 = time.time()
    episodes = collect_teacher_episodes(cfg, args.teacher_eps, args.seed, Q, pd_mode)
    total_frames = sum(len(ep) for ep in episodes)
    print(f"Teacher: {len(episodes)} episodes, {total_frames} frames ({time.time()-t0:.1f}s)")

    base_steady = None
    best_val_weak3 = -1.0
    best_state = None
    history = []

    for it in range(args.dagger_iters):
        # Chunk-based recurrent training
        actor.train()
        mse = train_chunk_bptt(actor, episodes, args.sup_epochs, args.lr,
                               max_dp, device, args.chunk_size)
        actor.eval()

        # Hidden drift diagnostic
        drift = measure_hidden_drift(actor, episodes, device, args.chunk_size)

        # Validation
        val_eps = evaluate_streaming(cfg, actor, device, Q, pd_mode, val_seeds)
        val_summary = summarize_episodes(val_eps)
        val_summary['iteration'] = it
        val_summary['n_episodes'] = len(episodes)
        val_summary['total_frames'] = total_frames
        val_summary['sup_mse'] = float(mse)
        val_summary['hidden_drift'] = float(drift)
        history.append(val_summary)

        print(f"  iter {it}: mse={mse:.4f}  h_drift={drift:.4f}  "
              f"val_steady={val_summary['steady_mean']:.4f}±{val_summary['steady_std']:.4f}  "
              f"val_weak3={val_summary['weak3_mean']:.4f}±{val_summary['weak3_std']:.4f}  "
              f"val_worst={val_summary['worst_mean']:.4f}  "
              f"val_ep_fail_030={val_summary['ep_fail_030']:.3f}")

        # Checkpoint selection
        if it == 0:
            base_steady = val_summary['steady_mean']
        if (val_summary['weak3_mean'] > best_val_weak3 and
                val_summary['steady_mean'] >= base_steady - 0.01):
            best_val_weak3 = val_summary['weak3_mean']
            best_state = {k: v.cpu().clone() for k, v in actor.state_dict().items()}
            best_iter = it

        # Student aggregation
        student_eps = collect_student_episodes(
            cfg, actor, device, args.student_eps, args.seed + 1000 + it, Q, pd_mode)
        student_frames = sum(len(ep) for ep in student_eps)
        # Merge: keep most recent data if over capacity
        episodes = episodes + student_eps
        total_frames += student_frames
        while total_frames > args.max_pairs:
            removed = episodes.pop(0)
            total_frames -= len(removed)
        print(f"    student: +{len(student_eps)} eps, {student_frames} frames → "
              f"dataset {len(episodes)} eps, {total_frames} frames")

    # Restore best
    if best_state is not None:
        actor.load_state_dict(best_state)
        print(f"  restored best: iter={best_iter} val_weak3={best_val_weak3:.4f}")
    actor.eval()

    # Final TEST evaluation
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
        'test_summary': test_summary, 'test_episodes': test_eps,
        'val_history': history,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

MODES = {'D0': 'none', 'D1': 'local'}


def main():
    ap = argparse.ArgumentParser(description="Recurrent DAgger D0/D1 (chunk BPTT)")
    ap.add_argument("--mode", default="all", choices=["D0", "D1", "all"])
    ap.add_argument("--config", default="config/exp_800_q4.yaml")
    ap.add_argument("--dagger-iters", type=int, default=5)
    ap.add_argument("--teacher-eps", type=int, default=60)
    ap.add_argument("--student-eps", type=int, default=40)
    ap.add_argument("--sup-epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-pairs", type=int, default=400000)
    ap.add_argument("--chunk-size", type=int, default=16,
                    help="Truncated BPTT chunk length (frames)")
    ap.add_argument("--val-episodes", type=int, default=20)
    ap.add_argument("--test-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/dagger_variants")
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else get_default_config()
    cfg.marl.num_envs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_seeds = list(range(20001, 20001 + args.val_episodes))
    test_seeds = list(range(30001, 30001 + args.test_episodes))
    commit_hash = os.popen('git rev-parse HEAD').read().strip() if os.path.exists('.git') else 'unknown'

    print(f"Config: {args.config}  Device: {device}  Seed: {args.seed}  Commit: {commit_hash}")
    print(f"Val: {val_seeds[0]}-{val_seeds[-1]}  Test: {test_seeds[0]}-{test_seeds[-1]}")

    to_run = list(MODES.items()) if args.mode == 'all' else [(args.mode, MODES[args.mode])]
    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}

    for mode_name, pd_mode in to_run:
        result = train_one(cfg, pd_mode, device, args, val_seeds, test_seeds)

        ckpt_path = os.path.join(args.out_dir, f"dagger_{mode_name}.pt")
        torch.save(result['state_dict'], ckpt_path)
        print(f"  saved → {ckpt_path}")

        csv_path = os.path.join(args.out_dir, f"test_episodes_{mode_name}.csv")
        save_episode_csv(result['test_episodes'], csv_path)

        hist_path = os.path.join(args.out_dir, f"val_history_{mode_name}.csv")
        with open(hist_path, 'w', newline='') as f:
            fields = ['iteration', 'n_episodes', 'total_frames', 'sup_mse', 'hidden_drift',
                      'steady_mean', 'steady_std', 'weak3_mean', 'weak3_std',
                      'worst_mean', 'worst_std', 'ep_fail_030', 'ep_fail_005']
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for h in result['val_history']:
                w.writerow({k: h.get(k, '') for k in fields})

        all_results[mode_name] = result

    manifest = {
        'git_commit': commit_hash, 'config': args.config,
        'training_seed': args.seed, 'K': cfg.scenario.K, 'Q': cfg.scenario.Q,
        'chunk_size': args.chunk_size,
        'dagger_iters': args.dagger_iters,
        'val_seeds': f"{val_seeds[0]}-{val_seeds[-1]}",
        'test_seeds': f"{test_seeds[0]}-{test_seeds[-1]}",
        'protocol': 'chunk-based truncated BPTT (h=0 at chunk start, detach between chunks)',
        'checkpoint_selection': 'max val weak3, steady >= base_steady - 0.01',
        'limitations': [
            'Single training seed.',
            'Chunk size=16; no full-episode BPTT.',
            'Communication not trained; deferred to PPO.',
        ],
        'results': {
            k: {
                'pd_mode': MODES[k],
                'steady_mean': v['test_summary']['steady_mean'],
                'weak3_mean': v['test_summary']['weak3_mean'],
                'worst_mean': v['test_summary']['worst_mean'],
                'ep_fail_030': v['test_summary']['ep_fail_030'],
                'ep_fail_005': v['test_summary']['ep_fail_005'],
            } for k, v in all_results.items()
        },
    }
    with open(os.path.join(args.out_dir, "run_manifest.json"), 'w') as f:
        json.dump(manifest, f, indent=2)

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
        print("Note: single training seed; D1 chosen as PPO init for interface consistency.")
        print("ep_fail_030 = fraction of episodes with min_q P_D < 0.3 in steady window.")


if __name__ == "__main__":
    main()
