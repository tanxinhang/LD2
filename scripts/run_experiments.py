#!/usr/bin/env python
"""Phase-1 main results: multi-seed MAPPO vs IPPO (+ non-learning baselines).

For each seed it:
  1. trains MAPPO (centralized critic / CTDE) to convergence (early-stop),
  2. trains IPPO  (decentralized local-obs critic) to convergence,
  3. evaluates Random / P0-Fixed / Greedy (no training) on the same seed,
then aggregates mean +/- std and runs a paired t-test (MAPPO vs IPPO) on the
per-seed best steady_P_D -> answers "cooperation (CTDE) > independence".

Results are saved incrementally to results/experiments.json (resumable).
Greedy uses privileged target positions -> reported as an ORACLE upper bound.

NOTE: training is torch-heavy; run on the training machine. Reduce N_SEEDS or
config.marl.num_episodes for a quick smoke test.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

import argparse
from config.params import get_default_config, load_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer
from uav_isac.agents.p0_fixed_agent import P0FixedAgent
# reuse the baseline policies + episode runner
from scripts.run_baselines import (
    run_episode, make_random_fn, make_p0fixed_fn, make_greedy_fn,
)

N_SEEDS = 1  # smoke test
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "experiments.json")


def train_one(cfg, seed, centralized: bool) -> float:
    """Train one policy (MAPPO if centralized else IPPO); return best steady_P_D."""
    set_seed(seed)
    env = UAVISACEnv(config=cfg, seed=seed)
    K = cfg.scenario.K
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    obs_dim = env.core.obs_builder.get_obs_dim()
    global_dim = env.core.obs_builder.get_global_state_dim()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agents = [
        MAPPOAgent(agent_id=k, obs_dim=obs_dim, global_state_dim=global_dim,
                   action_space=aspace, num_agents=K,
                   num_targets=cfg.scenario.Q,
                   hidden_layers=cfg.marl.hidden_layers, lr=cfg.marl.lr,
                   max_grad_norm=cfg.marl.max_grad_norm, device=device,
                   centralized_critic=centralized)
        for k in range(K)
    ]
    trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device=device)
    trainer.train(num_episodes=cfg.marl.num_episodes, log_interval=50)
    env.close()
    return float(trainer.best_score)


def eval_baseline(cfg, seed, policy: str) -> float:
    """Return steady_P_D of a non-learning baseline on the given seed."""
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                         learn_roles=cfg.marl.learn_roles)
    aspace.rng = np.random.default_rng(seed)
    env = UAVISACEnv(config=cfg, seed=seed)
    if policy == "Random":
        fn = make_random_fn(aspace)
    elif policy == "Greedy-Approach":
        fn = make_greedy_fn(aspace)
    elif policy == "P0-Fixed":
        region = cfg.scenario.region_size
        agents = [P0FixedAgent(agent_id=k, K=cfg.scenario.K,
                               center=np.array([region[0] / 2, region[1] / 2]),
                               radius=min(region) / 4.0, action_space=aspace,
                               position_scale=region[0])
                  for k in range(cfg.scenario.K)]
        fn = make_p0fixed_fn(agents)
    else:
        raise ValueError(policy)
    m = run_episode(env, fn)
    return float(m['steady_avg_P_D'])


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_results(res):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(res, f, indent=2)


def main():
    global RESULTS_PATH
    ap = argparse.ArgumentParser(description="Multi-seed MAPPO vs IPPO + baselines.")
    ap.add_argument("--config", default=None,
                    help="path to a config YAML (default: config/default.yaml). "
                         "Use config/exp_800_q4.yaml for the large-area scenario.")
    ap.add_argument("--seeds", type=int, default=N_SEEDS,
                    help=f"number of seeds to run (default {N_SEEDS}; full run = 5).")
    ap.add_argument("--results", default=RESULTS_PATH,
                    help="results JSON path (default results/experiments.json).")
    ap.add_argument("--p0-belief", action="store_true",
                    help="DEPLOYABLE mode: P0 ranks on fused belief (default OFF = oracle).")
    ap.add_argument("--detect-sample", action="store_true",
                    help="Bernoulli detection gating for belief updates (default OFF = optimistic).")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else get_default_config()
    cfg.marl.oracle_obs = False  # restored: use real belief observations
    if args.p0_belief:
        cfg.marl.p0_uses_belief = True
    if args.detect_sample:
        cfg.marl.belief_detection_sampling = True
    RESULTS_PATH = args.results
    seeds = cfg.seeds[:args.seeds]
    mode = ("DEPLOYABLE" if cfg.marl.p0_uses_belief else "ORACLE") + \
           ("+detect_sampling" if cfg.marl.belief_detection_sampling else "")
    print(f"config: {args.config or 'default.yaml'} | region={cfg.scenario.region_size} "
          f"K={cfg.scenario.K} Q={cfg.scenario.Q} learn_roles={cfg.marl.learn_roles} "
          f"| P0={mode} | seeds={seeds}")
    res = load_results()

    for seed in seeds:
        key = f"seed_{seed}"
        res.setdefault(key, {})
        if "MAPPO" not in res[key]:
            print(f"\n===== seed {seed}: training MAPPO (CTDE) =====")
            res[key]["MAPPO"] = train_one(cfg, seed, centralized=True); save_results(res)
        if "IPPO" not in res[key]:
            print(f"\n===== seed {seed}: training IPPO (local critic) =====")
            res[key]["IPPO"] = train_one(cfg, seed, centralized=False); save_results(res)
        for b in ["Random", "P0-Fixed", "Greedy-Approach"]:
            if b not in res[key]:
                res[key][b] = eval_baseline(cfg, seed, b); save_results(res)
        print(f"seed {seed}: " + ", ".join(f"{k}={v:.3f}" for k, v in res[key].items()))

    # ---- aggregate ----
    methods = ["Random", "P0-Fixed", "IPPO", "MAPPO", "Greedy-Approach"]
    print(f"\n{'='*62}\nPhase-1 results: best steady_P_D over {len(seeds)} seeds\n{'='*62}")
    print(f"{'Method':<18}{'mean':>10}{'std':>10}   note")
    agg = {}
    for m in methods:
        vals = np.array([res[f"seed_{s}"][m] for s in seeds])
        agg[m] = vals
        note = "ORACLE upper bound (privileged target pos)" if m == "Greedy-Approach" else \
               ("CTDE / centralized critic" if m == "MAPPO" else
                ("independent / local critic" if m == "IPPO" else "non-learning"))
        print(f"{m:<18}{vals.mean():>10.3f}{vals.std():>10.3f}   {note}")

    t, p = stats.ttest_rel(agg["MAPPO"], agg["IPPO"])
    print(f"\nMAPPO vs IPPO (cooperation>independence): "
          f"{agg['IPPO'].mean():.3f} -> {agg['MAPPO'].mean():.3f}, "
          f"paired t={t:.2f}, p={p:.4g}")
    if p < 0.05 and agg["MAPPO"].mean() > agg["IPPO"].mean():
        print("=> CTDE significantly beats independent learning (协同 > 独立).")
    else:
        print("=> No significant gap (CTDE ~ IPPO here; still a valid de-Witt-style finding).")
    save_results(res)
    print(f"\nsaved -> {os.path.normpath(RESULTS_PATH)}")


if __name__ == "__main__":
    main()
