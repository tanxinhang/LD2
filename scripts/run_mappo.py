#!/usr/bin/env python
"""Train Hierarchical MAPPO for Phase 1 UAV-ISAC system."""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config.params import get_default_config, load_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer


def main():
    ap = argparse.ArgumentParser(description="Train MAPPO (optionally warm-started).")
    ap.add_argument("--config", default=None, help="config YAML (default: config/default.yaml)")
    ap.add_argument("--warm-start", default=None,
                    help="path to an actor state_dict (e.g. results/warmstart_actor.pt) to "
                         "initialize the shared actor before PPO (see scripts/dagger_warmstart.py)")
    ap.add_argument("--warm-start-mode", default="direct",
                    choices=["direct", "residual"],
                    help="direct: load into StructuredActorNetwork (Full/EH default). "
                         "residual: wrap in ResidualActor (legacy safe fine-tuning).")
    ap.add_argument("--warmstart-lr", type=float, default=None,
                    help="override LR when warm-starting (default: config.marl.lr). "
                         "3e-5 recommended for BC warmstart to keep KL within trust region.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=None, help="override marl.num_episodes")
    args = ap.parse_args()

    # Config: single source of truth (default.yaml) or an explicit --config.
    config = load_config(args.config) if args.config else get_default_config()

    seed = args.seed
    set_seed(seed)

    # Create environment
    env = UAVISACEnv(config=config, seed=seed)
    K = config.scenario.K

    # Create action space
    action_space = ActionSpace(
        v_max=config.uav.v_max,
        dt=config.scenario.dt,
        learn_roles=config.marl.learn_roles,
    )
    action_space.num_targets = config.scenario.Q
    action_space.structured_actor = True   # relational with 2-frame parsing
    action_space.structured_entity_dim = 64

    # Create agents — use actual obs dim (includes history stacking)
    obs_test, _ = env.reset(seed=seed)
    obs_dim = obs_test['0'].shape[0]
    global_dim = env.core.obs_builder.get_global_state_dim()

    print(f"Observation dim: {obs_dim}")
    print(f"Global state dim: {global_dim}")
    print(f"K={K}, Q={config.scenario.Q}, T={config.scenario.T}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_lr = args.warmstart_lr if (args.warm_start and args.warmstart_lr) else config.marl.lr
    agents = []
    for k in range(K):
        agent = MAPPOAgent(
            agent_id=k,
            obs_dim=obs_dim,
            global_state_dim=global_dim,
            action_space=action_space,
            num_agents=K,
            num_targets=config.scenario.Q,
            hidden_layers=config.marl.hidden_layers,
            lr=train_lr,
            critic_lr_mult=config.marl.critic_lr_mult,
            max_grad_norm=config.marl.max_grad_norm,
            device=device,
            centralized_critic=getattr(config.marl, 'centralized_critic', True),
        )
        agents.append(agent)

    # Warm-start: load a pretrained (e.g. DAgger-cloned) actor into the SHARED actor.
    if args.warm_start:
        ckpt = torch.load(args.warm_start, map_location=device)
        actor_state = ckpt.get('actor', ckpt)  # unwrap if dict
        # Allow missing aux head keys (added after warmstart was generated)
        missing, unexpected = agents[0].actor.load_state_dict(actor_state, strict=False)
        if missing:
            print(f"warm-start: {len(missing)} new keys initialised (e.g. pd_aux_head)")
        if 'critic' in ckpt:
            agents[0].critic.load_state_dict(ckpt['critic'])
        # Keep default log_std=0 (σ=1) for warm-start with KL BC anchor.
        # KL anchor constrains σ in probability space — no need to force low exploration.
        print(f"warm-started actor{'+critic' if 'critic' in ckpt else ''} from {args.warm_start}"
              f"  (σ=1, MSE BC β={config.marl.bc_beta_init})")

    # Warm-start mode
    if args.warm_start:
        if args.warm_start_mode == "residual":
            from uav_isac.agents.residual_actor import ResidualActor
            base_actor = agents[0].actor
            residual = ResidualActor(base_actor, max_dp=config.uav.v_max * config.scenario.dt, delta_max=0.06)
            for agent in agents:
                agent.actor = residual
                agent.actor_optimizer = torch.optim.Adam(
                    residual.residual.parameters(), lr=train_lr)
            print(f"ResidualActor: δ_max=0.03, base frozen, {sum(p.numel() for p in residual.residual.parameters())} trainable params")
        else:
            # direct: already loaded into StructuredActorNetwork, no wrapping
            print(f"Warm-start mode=direct: actor loaded as-is (no ResidualActor wrap)")

    # BC anchor: skip for warm-started actors
    bc_actor = None
    if args.warm_start and not action_space.structured_actor:
        bc_actor = MAPPOAgent(agent_id=0, obs_dim=obs_dim, global_state_dim=global_dim,
                              action_space=action_space, num_agents=K,
                              hidden_layers=config.marl.hidden_layers,
                              lr=config.marl.lr, max_grad_norm=config.marl.max_grad_norm,
                              device=device)
        bc_actor.actor.load_state_dict(agents[0].actor.state_dict())
        for p in bc_actor.actor.parameters():
            p.requires_grad = False
        bc_actor.actor.eval()
        print(f"BC anchor actor frozen (β_init={config.marl.bc_beta_init})")

    # Create trainer
    trainer = MAPPTrainer(
        env=env,
        agents=agents,
        config=config,
        device=device,
    )
    trainer._bc_actor = bc_actor.actor if bc_actor else None

    print(f"\nStarting training for {config.marl.num_episodes} episodes...")
    print(f"Rollout steps: {config.marl.rollout_steps}")
    print(f"Gamma: {config.marl.gamma}, GAE lambda: {config.marl.gae_lambda}")
    print(f"PPO clip: {config.marl.ppo_clip}, Epochs: {config.marl.ppo_epochs}")
    print("-" * 60)

    # Diagnostic: confirm actual LR values
    actor_lr = agents[0].actor_optimizer.param_groups[0]['lr']
    critic_lr = agents[0].critic_optimizer.param_groups[0]['lr']
    print(f"[LR] actor={actor_lr:.1e} critic={critic_lr:.1e} bc_beta={config.marl.bc_beta_init} "
          f"entropy_coef={trainer.entropy_coef:.3f} ppo_epochs={config.marl.ppo_epochs} "
          f"ppo_clip={config.marl.ppo_clip} gae_lambda={config.marl.gae_lambda}")

    # Train
    metrics_history = trainer.train(
        num_episodes=args.episodes if args.episodes is not None else config.marl.num_episodes,
        log_interval=10,
    )

    # Save final metrics summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total frames: {trainer.total_frames}")

    if metrics_history:
        final_window = metrics_history[-50:]
        avg_pd_values = [m.get('avg_P_D', 0) for m in final_window if 'avg_P_D' in m]
        if avg_pd_values:
            print(f"Final 50-ep avg P_D: {np.mean(avg_pd_values):.4f}")

        actor_losses = [m.get('actor_loss', 0) for m in metrics_history]
        print(f"Initial actor loss: {actor_losses[0]:.4f}" if actor_losses else "")
        print(f"Final actor loss: {actor_losses[-1]:.4f}" if actor_losses else "")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
