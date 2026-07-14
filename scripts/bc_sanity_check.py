#!/usr/bin/env python
"""Behavior-cloning sanity check + confound isolation.

MAPPO collapses to BELOW random. BC of the Greedy-Oracle teacher (index teacher,
belief obs) also reached only ~random (0.163 vs random 0.158, greedy 0.713),
which LOOKS like a representation/observation bottleneck. But two confounds can
produce ~random even if the obs were fine:

  C1 (agent-id): the default 'index' teacher assigns UAV k -> target k%Q, yet the
     ACTOR obs has NO agent identity (the agent one-hot is added only to the CRITIC
     input, never the actor). A shared net cannot map near-identical obs to
     different targets -> regression averages -> ~random. This would ALSO explain
     the RL collapse (a shared actor can't differentiate UAVs to distinct targets).
  C2 (belief quality): the teacher decides from TRUE targets; the student sees
     BELIEF. If belief drifts far from truth (sparse detection), the true-based
     action is unrecoverable from the obs -> partial observability, not obs format.

This script isolates them with two knobs:
  --oracle-obs        feed TRUE target state into the obs slots (cfg.marl.oracle_obs).
                      BC(index, oracle) high  => obs/network fine, belief QUALITY was
                      the issue (C2). BC(index, oracle) still ~random => agent-id (C1).
  --teacher nearest   symmetric, obs-derivable teacher (each UAV -> nearest target).
                      No agent-id needed; BC(nearest, belief) high => original failure
                      was the index teacher (C1), not the obs.

Decision matrix (run all 3-4 cells, compare to per-run Random / teacher refs):
  index   + belief   (baseline; ~0.16 observed)
  index   + oracle   --oracle-obs
  nearest + belief   --teacher nearest
  nearest + oracle   --teacher nearest --oracle-obs

Needs torch (run on GPU machine). Teacher data is generated via the env.

Usage:
    python scripts/bc_sanity_check.py --config config/exp_800_q4.yaml [--teacher nearest] [--oracle-obs]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config.params import load_config, get_default_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.networks import ActorNetwork


def teacher_dp(env, K, Q, max_dp, mode):
    """Teacher action (toward TRUE targets, for action quality).

    mode='index'   : UAV k -> target (k % Q)         [depends on agent index]
    mode='nearest' : UAV k -> its nearest target     [symmetric, no agent-id needed]
    """
    tgt = np.array([t.get_position_3d() for t in env.core.targets])
    dp = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        pos = env.core.uavs[k].pos[:2]
        if mode == 'nearest':
            q = int(np.argmin([np.linalg.norm(tgt[qq][:2] - pos) for qq in range(Q)]))
        else:
            q = k % Q
        d = tgt[q][:2] - pos
        n = np.linalg.norm(d)
        dp[k] = d / n * max_dp if n > 1e-6 else np.zeros(2)
    return dp


def _aug(o_k, k, K, use_id):
    """Optionally append a K-dim agent one-hot to a single agent's obs."""
    if not use_id:
        return np.asarray(o_k, dtype=np.float64)
    oh = np.zeros(K, dtype=np.float64); oh[k] = 1.0
    return np.concatenate([np.asarray(o_k, dtype=np.float64), oh])


def collect_teacher(cfg, n_episodes, seed, mode, agent_id):
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    obs_buf, act_buf = [], []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_episodes):
        o, _ = env.reset(seed=seed + ep)
        while True:
            dp = teacher_dp(env, K, Q, max_dp, mode)
            for k in range(K):
                obs_buf.append(_aug(o[str(k)], k, K, agent_id))
                act_buf.append(dp[k].copy())
            acts = {str(k): {'delta_p': dp[k], 'role': 0} for k in range(K)}
            o, _, t, tr, _ = env.step(acts)
            if t.get('__all__') or tr.get('__all__'):
                break
    return np.asarray(obs_buf), np.asarray(act_buf)


def train_bc(obs, act, obs_dim, max_dp, epochs, lr, device):
    actor = ActorNetwork(obs_dim=obs_dim, max_dp=max_dp).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    O = torch.as_tensor(obs, dtype=torch.float32, device=device)
    A = torch.as_tensor(act, dtype=torch.float32, device=device)
    n, bs, dp_scale = O.shape[0], 256, float(max_dp)
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, bs):
            mb = idx[s:s + bs]
            dp_mean, _, _ = actor(O[mb])
            pred = torch.tanh(dp_mean) * dp_scale
            loss = ((pred - A[mb]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(mb)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  BC epoch {ep:3d}: mse={tot/n:.5f}")
    return actor


def eval_policy_fn(cfg, policy_fn):
    K = cfg.scenario.K
    env = UAVISACEnv(config=cfg, seed=12345)
    steady = []
    for s in cfg.marl.eval_seeds:
        o, _ = env.reset(seed=s); rng = np.random.default_rng(s); pdh = []
        while True:
            o, _, t, tr, info = env.step(policy_fn(env, rng))
            pdh.append(np.mean(info['P_D_q']))
            if t.get('__all__') or tr.get('__all__'):
                break
        pdh = np.array(pdh); steady.append(pdh[-20:].mean())
    return float(np.mean(steady))


def eval_actor(cfg, actor, device, agent_id):
    K = cfg.scenario.K
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    env = UAVISACEnv(config=cfg, seed=12345)
    steady = []
    for s in cfg.marl.eval_seeds:
        o, _ = env.reset(seed=s); pdh = []
        while True:
            ob = np.stack([_aug(o[str(k)], k, K, agent_id) for k in range(K)])
            with torch.no_grad():
                dpm, dps, rl = actor(torch.as_tensor(ob, dtype=torch.float32, device=device))
            dpm = dpm.cpu().numpy(); dps = dps.cpu().numpy(); rl = rl.cpu().numpy()
            acts = {}
            for k in range(K):
                a, _ = aspace.decode(dpm[k], dps, rl[k],
                                     dp_deterministic=True, role_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
            o, _, t, tr, info = env.step(acts); pdh.append(np.mean(info['P_D_q']))
            if t.get('__all__') or tr.get('__all__'):
                break
        pdh = np.array(pdh); steady.append(pdh[-20:].mean())
    return float(np.mean(steady))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/exp_800_q4.yaml")
    ap.add_argument("--teacher", choices=["index", "nearest"], default="index")
    ap.add_argument("--oracle-obs", action="store_true",
                    help="feed TRUE target state into the obs (isolates belief quality).")
    ap.add_argument("--agent-id", action="store_true",
                    help="append a K-dim agent one-hot to the actor obs (tests C1: differentiation).")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else get_default_config()
    if args.oracle_obs:
        cfg.marl.oracle_obs = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    base_obs_dim = UAVISACEnv(config=cfg, seed=0).core.obs_builder.get_obs_dim()
    obs_dim = base_obs_dim + (K if args.agent_id else 0)   # actor input dim (incl. agent-id)
    print(f"config={args.config} teacher={args.teacher} oracle_obs={cfg.marl.oracle_obs} "
          f"agent_id={args.agent_id} K={K} Q={Q} obs_dim={obs_dim} device={device}")

    def f_random(env, rng):
        return {str(k): {'delta_p': rng.normal(0, 1.5, 2), 'role': 0} for k in range(K)}
    def f_teacher(env, rng):
        dp = teacher_dp(env, K, Q, max_dp, args.teacher)
        return {str(k): {'delta_p': dp[k], 'role': 0} for k in range(K)}
    ref_random = eval_policy_fn(cfg, f_random)
    ref_teacher = eval_policy_fn(cfg, f_teacher)
    print(f"reference: Random={ref_random:.3f}  teacher({args.teacher})={ref_teacher:.3f}")

    print(f"collecting teacher data ({args.episodes} eps)...")
    obs, act = collect_teacher(cfg, args.episodes, args.seed, args.teacher, args.agent_id)
    print(f"dataset: {obs.shape[0]} (obs, dp) pairs, obs_dim={obs.shape[1]}; "
          f"training BC {args.bc_epochs} epochs...")
    actor = train_bc(obs, act, obs_dim, max_dp, args.bc_epochs, args.lr, device)

    bc_pd = eval_actor(cfg, actor, device, args.agent_id)
    gap = (bc_pd - ref_random) / max(ref_teacher - ref_random, 1e-6)
    print("\n" + "=" * 62)
    print(f"[teacher={args.teacher}, oracle_obs={cfg.marl.oracle_obs}]")
    print(f"BC eval steady_P_D = {bc_pd:.3f}   (Random={ref_random:.3f}, teacher={ref_teacher:.3f})")
    print(f"BC recovers {100*gap:.0f}% of the (teacher - random) gap")
    print("-" * 62)
    if bc_pd > ref_random + 0.5 * (ref_teacher - ref_random):
        print("This cell: MLP CAN clone the teacher from these obs.")
    else:
        print("This cell: BC ~ random — teacher not learnable from these obs.")
    print("Compare cells to localize: see file header decision matrix.")
    print("=" * 62)


if __name__ == "__main__":
    main()
