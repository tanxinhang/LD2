#!/usr/bin/env python
"""Train IPPO baseline — same protocol as MAPPO, only critic differs.

IPPO = same PPO-clip actor + D1 warm-start + comm-off + per-module LR,
but DECENTRALIZED critic (local obs instead of global state).

Protocol matches run_mappo.py exactly:
  - D1 direct warm-start (no ResidualActor wrap)
  - learned_comm_mode='off'
  - per-module LR (encoder=1e-5, attention=1e-5, head=5e-5)
  - freeze_attention per config
  - 300 episodes, auto result saving

Only difference: centralized_critic=False.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config.params import load_config, get_default_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer


def main():
    ap = argparse.ArgumentParser(description="Train IPPO (decentralized critic)")
    ap.add_argument("--config", default="config/exp_800_q4_full.yaml")
    ap.add_argument("--warm-start", default="results/dagger_variants/dagger_D1.pt")
    ap.add_argument("--warm-start-mode", default="direct", choices=["direct", "residual"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else get_default_config()
    cfg.marl.centralized_critic = False  # IPPO: local-obs critic

    seed = args.seed
    set_seed(seed)

    env = UAVISACEnv(config=cfg, seed=seed)
    K, Q = cfg.scenario.K, cfg.scenario.Q
    action_space = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                               learn_roles=cfg.marl.learn_roles)
    action_space.num_targets = Q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 64

    obs_dim = env.core.obs_builder.get_obs_dim()
    global_dim = env.core.obs_builder.get_global_state_dim()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[IPPO] obs_dim={obs_dim} global_dim={global_dim} K={K} Q={Q} device={device}")
    print(f"[IPPO] centralized_critic=False (critic input = LOCAL obs, dim={obs_dim})")

    train_lr = cfg.marl.lr
    agents = [
        MAPPOAgent(agent_id=k, obs_dim=obs_dim, global_state_dim=global_dim,
                   action_space=action_space, num_agents=K, num_targets=Q,
                   hidden_layers=cfg.marl.hidden_layers, lr=train_lr,
                   critic_lr_mult=cfg.marl.critic_lr_mult,
                   max_grad_norm=cfg.marl.max_grad_norm, device=device,
                   centralized_critic=False)
        for k in range(K)
    ]

    # D1 direct warm-start
    if args.warm_start and args.warm_start_mode == "direct":
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        actor_state = ckpt.get('actor', ckpt)
        agents[0].actor.load_state_dict(actor_state, strict=False)
        agents[0].actor.zero_init_new_layers(set(actor_state.keys()))
        print(f"Warm-start mode=direct: actor loaded from {args.warm_start}")

    trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device=device)
    trainer._bc_actor = None

    n_eps = args.episodes if args.episodes is not None else cfg.marl.num_episodes
    print(f"\n[IPPO] training {n_eps} episodes...")
    metrics_history = trainer.train(num_episodes=n_eps, log_interval=50)

    print("\n" + "=" * 60)
    print("[IPPO] training complete!")
    print(f"Total frames: {trainer.total_frames}")
    print(f"Best eval steady_P_D: {trainer.best_score:.4f}"
          + (f" (converged @ ep {trainer.converged_episode})" if trainer.converged_episode else ""))

    # Save results (same format as run_mappo.py)
    import csv, json as _json, subprocess as _sp
    config_stem = os.path.splitext(os.path.basename(args.config or "config/default.yaml"))[0]
    variant = "ippo_" + config_stem.replace("exp_800_q4_", "") if "exp_800_q4_" in config_stem else "ippo"
    out_dir = args.out_dir or os.path.join("results", "ippo", variant, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        commit = _sp.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    manifest = {
        "git_commit": commit, "config": args.config, "seed": seed,
        "K": K, "Q": Q, "warm_start": args.warm_start,
        "warm_start_mode": args.warm_start_mode,
        "centralized_critic": False,
        "learned_comm_mode": cfg.marl.learned_comm_mode,
        "best_steady_P_D": float(trainer.best_score),
        "total_frames": trainer.total_frames,
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        _json.dump(manifest, f, indent=2)

    if metrics_history:
        keys = sorted(metrics_history[0].keys())
        with open(os.path.join(out_dir, "train_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for m in metrics_history:
                w.writerow({k: m.get(k, "") for k in keys})

    paired_seeds = [30001 + i for i in range(20)]
    try:
        trainer.agents[0].actor.eval()
        ev = trainer._evaluate(n_episodes=len(paired_seeds), eval_seeds=paired_seeds)
        with open(os.path.join(out_dir, "fixed_bank_eval.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(sorted(ev.keys()))
            w.writerow([ev.get(k, "") for k in sorted(ev.keys())])
    except Exception as e:
        print(f"  [warn] eval save failed: {e}")

    print(f"Results saved → {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
