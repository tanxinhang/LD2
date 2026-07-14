#!/usr/bin/env python
"""Head-to-head baseline comparison (torch-free).

Compares non-learning policies to confirm the learnable gap exists:
  - Random        : random outer actions (floor)
  - P0-Fixed      : fixed circular trajectory + inner P0 (no traj. optimization)
  - Greedy-Approach: each tx/rx pair flies toward its assigned target + inner P0

Reports avg_P_D / worst_P_D / jain over >=5 seeds (mean +/- std) and a paired
t-test (Greedy vs Random) on per-seed avg_P_D. No torch required.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from config.params import get_default_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.p0_fixed_agent import P0FixedAgent
from uav_isac.evaluation.metrics import compute_episode_metrics


def run_episode(env, action_fn):
    """Run one episode driven by action_fn(obs, env) -> actions dict."""
    obs, info = env.reset()
    P_D_history, p0_solutions, constraint_infos, team_rewards = [], [], [], []
    init_bat = np.array([u.battery for u in env.core.uavs])
    while True:
        actions = action_fn(obs, env)
        obs, rewards, terminated, truncated, info = env.step(actions)
        P_D_history.append(info['P_D_q'].copy())
        if env.current_step_info is not None:
            p0_solutions.append(env.current_step_info.p0_solution)
            constraint_infos.append(env.current_step_info.constraint_info)
            team_rewards.append(env.current_step_info.team_reward)
        if terminated.get('__all__', False):
            break
    final_bat = np.array([u.battery for u in env.core.uavs])
    m = compute_episode_metrics(
        P_D_history=P_D_history, p0_solutions=p0_solutions,
        constraint_infos=constraint_infos,
        initial_batteries=init_bat, final_batteries=final_bat,
        team_rewards=team_rewards,
    )
    m['episode_length'] = len(P_D_history)
    return m


# ---------------- policies ----------------
def make_random_fn(aspace):
    def fn(obs, env):
        acts = {}
        for k in range(env.core.K):
            a = aspace.sample()
            acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
        return acts
    return fn


def make_p0fixed_fn(agents):
    def fn(obs, env):
        acts = {}
        for k in range(env.core.K):
            a, _, _ = agents[k].act(obs[str(k)])
            acts[str(k)] = {'delta_p': a.delta_p, 'role': a.role}
        return acts
    return fn


def make_greedy_fn(aspace):
    """Each UAV is paired to a target; pair = (tx, rx); both fly toward target."""
    def fn(obs, env):
        K, Q = env.core.K, env.core.Q
        tgt_pos = np.array([t.get_position_3d() for t in env.core.targets])
        acts = {}
        for k in range(K):
            q = (k // 2) % Q                  # pair UAVs (0,1)->T0, (2,3)->T1, ...
            role = 0 if (k % 2 == 0) else 1   # even=tx, odd=rx  -> each target gets a tx+rx
            d = tgt_pos[q][:2] - env.core.uavs[k].pos[:2]
            n = np.linalg.norm(d)
            dp = d / n * aspace.max_dp if n > 1e-6 else np.zeros(2)
            dp = aspace.clip_dp(dp) if hasattr(aspace, 'clip_dp') else dp
            acts[str(k)] = {'delta_p': dp, 'role': role}
        return acts
    return fn


def main():
    cfg = get_default_config()
    seeds = cfg.seeds[:5]
    methods = ['Random', 'P0-Fixed', 'Greedy-Approach']
    results = {m: {'avg_P_D': [], 'worst_P_D': [], 'steady_avg_P_D': [],
                   'steady_worst_P_D': [], 'jain_fairness': [],
                   'cumulative_energy_J': []} for m in methods}

    for seed in seeds:
        aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
        if hasattr(aspace, 'rng'):
            aspace.rng = np.random.default_rng(seed)

        # Random
        env = UAVISACEnv(config=cfg, seed=seed)
        m = run_episode(env, make_random_fn(aspace))
        for key in results['Random']:
            results['Random'][key].append(m[key])

        # P0-Fixed (radius/center scaled to region so the circle stays inside the area)
        env = UAVISACEnv(config=cfg, seed=seed)
        region = cfg.scenario.region_size
        center = np.array([region[0] / 2, region[1] / 2])
        radius = min(region) / 4.0   # e.g. 100 m in a 400x400 area -> circle within bounds
        agents = [P0FixedAgent(agent_id=k, K=cfg.scenario.K, center=center,
                               radius=radius, action_space=aspace,
                               position_scale=region[0])
                  for k in range(cfg.scenario.K)]
        m = run_episode(env, make_p0fixed_fn(agents))
        for key in results['P0-Fixed']:
            results['P0-Fixed'][key].append(m[key])

        # Greedy-Approach
        env = UAVISACEnv(config=cfg, seed=seed)
        m = run_episode(env, make_greedy_fn(aspace))
        for key in results['Greedy-Approach']:
            results['Greedy-Approach'][key].append(m[key])

    # ---- report ----
    print(f"\n{'='*78}\nBaseline comparison over {len(seeds)} seeds (mean +/- std)\n{'='*78}")
    print(f"{'Method':<18} {'avg_P_D':>14} {'steady_P_D':>14} {'steady_worst':>14} {'Jain':>8}")
    print("  (steady = last-20-frame average, i.e. ceiling after the approach transient)")
    for m in methods:
        a = np.array(results[m]['avg_P_D'])
        s = np.array(results[m]['steady_avg_P_D'])
        sw = np.array(results[m]['steady_worst_P_D'])
        j = np.array(results[m]['jain_fairness'])
        print(f"{m:<18} {a.mean():>7.4f}+/-{a.std():<5.3f} "
              f"{s.mean():>7.4f}+/-{s.std():<5.3f} {sw.mean():>7.4f}+/-{sw.std():<5.3f} {j.mean():>7.3f}")

    # paired t-test Greedy vs Random on STEADY avg_P_D (true ceiling)
    g = np.array(results['Greedy-Approach']['steady_avg_P_D'])
    r = np.array(results['Random']['steady_avg_P_D'])
    t, p = stats.ttest_rel(g, r)
    gain = (g.mean() / max(r.mean(), 1e-9))
    abs_gain = g.mean() - r.mean()
    print(f"\nGreedy vs Random  steady avg_P_D:  {r.mean():.4f} -> {g.mean():.4f} "
          f"({gain:.1f}x, +{abs_gain:.4f}),  paired t={t:.2f}, p={p:.4g}")
    print(f"建议: 把 config 的 detection.P_D_min 设为略低于 Greedy 的 steady_P_D "
          f"(当前≈{g.mean():.2f}) 的一个可达值, 例如 {max(0.1, round(g.mean()*0.6,1))}")
    # require a MEANINGFUL effect (not just p<0.05 on numerical noise):
    # significant AND >=2x AND absolute improvement >= 0.05
    if p < 0.05 and gain >= 2.0 and abs_gain >= 0.05:
        print("PASS: 非随机策略显著且实质性拉开 P_D 差距 (存在可学习间隙)")
    elif g.mean() <= r.mean() + 1e-6:
        print("DEAD-ENV?: 各方法 P_D 几乎相同且接近 P_FA — 很可能在跑修复前的旧包"
              "(检查 cfg.otfs.g_tx_dBi 是否存在、清 __pycache__ 后重跑)")
    else:
        print("WARN: 有差距但偏弱 — 可上调 n_cpi/增益, 或缩小区域/增大候选几何覆盖")


if __name__ == '__main__':
    main()
