#!/usr/bin/env python
"""DAgger warm-start: clone a good obs-derivable policy into the actor, robust to
the distribution shift that made plain BC fail.

Why DAgger: plain behavior cloning collapsed to ~random not because the obs lacks
features (an obs-reading homing policy scores ~0.5-0.72), but because open-loop BC
drifts off the teacher's state distribution. DAgger fixes this by iteratively
rolling out the STUDENT, labelling the states IT visits with the teacher action,
and aggregating — so the clone learns to recover from its own drift.

Teacher: 'nearest' homing (each UAV flies toward its NEAREST true target). This is
a FUNCTION OF THE OBS (no agent identity needed), so it is clonable into the RL
actor, whose local obs has no agent-id. Ceiling ~0.50 steady_P_D (>> random 0.16,
>> the frozen RL policy's 0.12). Reaching the index teacher's 0.72 needs agent-id
in the obs (a separate change); this script targets the no-agent-id actor used by
the current RL pipeline so the checkpoint can warm-start it directly.

Output: results/warmstart_actor.pt (actor state_dict), loadable via
    python scripts/run_mappo.py --config config/exp_800_q4.yaml --warm-start results/warmstart_actor.pt

Decisive test after warm-start: does PPO PRESERVE the ~0.5 init or DESTROY it?
  preserve/improve -> exploration was the only problem; warm-start is the fix.
  destroy (-> random) -> the PPO update itself is broken; fix advantages/critic/LR.

Needs torch -> run on GPU machine.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config.params import load_config, get_default_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.networks import ActorNetwork


def nearest_teacher_dp(env, K, Q, max_dp):
    """Expert label: each UAV homes toward its NEAREST true target (obs-derivable)."""
    tgt = np.array([t.get_position_3d() for t in env.core.targets])
    dp = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        pos = env.core.uavs[k].pos[:2]
        q = int(np.argmin([np.linalg.norm(tgt[qq][:2] - pos) for qq in range(Q)]))
        d = tgt[q][:2] - pos
        n = np.linalg.norm(d)
        dp[k] = d / n * max_dp if n > 1e-6 else np.zeros(2)
    return dp


def collect_teacher_rollout(cfg, n_eps, seed):
    """Roll out the teacher; record (obs, teacher_dp)."""
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    O, A = [], []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_eps):
        o, _ = env.reset(seed=seed + ep)
        while True:
            dp = nearest_teacher_dp(env, K, Q, max_dp)
            for k in range(K):
                O.append(o[str(k)].copy()); A.append(dp[k].copy())
            o, _, t, tr, _ = env.step({str(k): {'delta_p': dp[k], 'role': 0} for k in range(K)})
            if t.get('__all__') or tr.get('__all__'):
                break
    return np.asarray(O), np.asarray(A)


def collect_student_rollout(cfg, actor, device, n_eps, seed):
    """Roll out the STUDENT (actor) deterministically; label visited states with the
    teacher action. This is the DAgger aggregation step that fixes distribution shift."""
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    O, A = [], []
    env = UAVISACEnv(config=cfg, seed=seed)
    for ep in range(n_eps):
        o, _ = env.reset(seed=seed + ep)
        while True:
            label = nearest_teacher_dp(env, K, Q, max_dp)     # expert label at student's state
            ob = np.stack([o[str(k)] for k in range(K)])
            with torch.no_grad():
                dpm, dps, rl, _, _, _ = actor(torch.as_tensor(ob, dtype=torch.float32, device=device))
            dpm = dpm.cpu().numpy(); dps = dps.cpu().numpy(); rl = rl.cpu().numpy()
            acts = {}
            for k in range(K):
                O.append(o[str(k)].copy()); A.append(label[k].copy())
                a, _ = aspace.decode(dpm[k], dps, rl[k],
                                     dp_deterministic=True, role_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
            o, _, t, tr, _ = env.step(acts)                   # student drives the trajectory
            if t.get('__all__') or tr.get('__all__'):
                break
    return np.asarray(O), np.asarray(A)


def train_supervised(actor, O, A, epochs, lr, max_dp, device):
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    Ot = torch.as_tensor(O, dtype=torch.float32, device=device)
    At = torch.as_tensor(A, dtype=torch.float32, device=device)
    n, bs, dp_scale = Ot.shape[0], 256, float(max_dp)
    last = 0.0
    for ep in range(epochs):
        idx = torch.randperm(n, device=device); tot = 0.0
        for s in range(0, n, bs):
            mb = idx[s:s + bs]
            dp_mean, _, _, _, _ = actor(Ot[mb])
            pred = torch.tanh(dp_mean) * dp_scale
            loss = ((pred - At[mb]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(mb)
        last = tot / n
    return last


def eval_actor(cfg, actor, device):
    K = cfg.scenario.K
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    env = UAVISACEnv(config=cfg, seed=12345)
    steady = []
    for s in cfg.marl.eval_seeds:
        o, _ = env.reset(seed=s); pdh = []
        while True:
            ob = np.stack([o[str(k)] for k in range(K)])
            with torch.no_grad():
                dpm, dps, rl, _, _, _ = actor(torch.as_tensor(ob, dtype=torch.float32, device=device))
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
    ap.add_argument("--dagger-iters", type=int, default=5)
    ap.add_argument("--teacher-eps", type=int, default=60, help="teacher rollout eps (iter 0)")
    ap.add_argument("--student-eps", type=int, default=40, help="student rollout eps per DAgger iter")
    ap.add_argument("--sup-epochs", type=int, default=60, help="supervised epochs per iter")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-pairs", type=int, default=400000, help="cap aggregated dataset size")
    ap.add_argument("--out", default="results/warmstart_actor.pt")
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else get_default_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    obs_dim = UAVISACEnv(config=cfg, seed=0).core.obs_builder.get_obs_dim()
    print(f"config={args.config} K={K} Q={Q} obs_dim={obs_dim} device={device}")

    actor = ActorNetwork(obs_dim=obs_dim, max_dp=max_dp).to(device)

    # Iter 0: pure teacher data (BC start).
    O, A = collect_teacher_rollout(cfg, args.teacher_eps, seed=0)
    print(f"teacher dataset: {O.shape[0]} pairs")
    for it in range(args.dagger_iters):
        mse = train_supervised(actor, O, A, args.sup_epochs, args.lr, max_dp, device)
        pd = eval_actor(cfg, actor, device)
        print(f"DAgger iter {it}: dataset={O.shape[0]} sup_mse={mse:.4f} eval_steady_P_D={pd:.3f}")
        # Aggregate student-visited states labelled by the teacher.
        sO, sA = collect_student_rollout(cfg, actor, device, args.student_eps, seed=1000 + it)
        O = np.concatenate([O, sO]); A = np.concatenate([A, sA])
        if O.shape[0] > args.max_pairs:                    # keep most-recent (student) data
            O = O[-args.max_pairs:]; A = A[-args.max_pairs:]

    final_pd = eval_actor(cfg, actor, device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(actor.state_dict(), args.out)
    print("\n" + "=" * 60)
    print(f"final DAgger clone eval steady_P_D = {final_pd:.3f}  (random≈0.16, nearest-teacher≈0.50)")
    print(f"saved actor -> {args.out}")
    print("warm-start PPO with:")
    print(f"  python scripts/run_mappo.py --config {args.config} --warm-start {args.out}")
    print("then watch: PPO PRESERVES ~0.5 => exploration was the issue (warm-start fixes it);")
    print("            PPO DESTROYS -> random => the PPO update is broken (fix advantages/critic/LR).")
    print("=" * 60)


if __name__ == "__main__":
    main()
