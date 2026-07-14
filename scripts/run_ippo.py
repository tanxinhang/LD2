#!/usr/bin/env python
"""Train IPPO (Independent PPO) baseline for the "cooperation > independence" study.

IPPO = same PPO-clip actor as MAPPO, but a DECENTRALIZED critic: each agent's
value function conditions only on its OWN local observation (no global state).
This is the canonical independent-learning baseline; the only difference from our
CTDE-MAPPO is the critic's information (local obs vs global state).

References:
  - de Witt et al., "Is Independent Learning All You Need in the StarCraft
    Multi-Agent Challenge?", 2020 (arXiv:2011.09533)  -- IPPO
  - Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
    Games", NeurIPS 2022 (arXiv:2103.01955)            -- MAPPO (centralized critic)

Note: like canonical IPPO (de Witt 2020, SMAC), homogeneous agents share network
parameters; the defining IPPO property is the LOCAL value function, not the lack
of parameter sharing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config.params import get_default_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer


def main():
    config = get_default_config()
    config.marl.centralized_critic = False   # <-- the only change vs run_mappo: IPPO
    # (all other settings come from config = single source of truth)

    seed = 42
    set_seed(seed)

    env = UAVISACEnv(config=config, seed=seed)
    K = config.scenario.K
    action_space = ActionSpace(v_max=config.uav.v_max, dt=config.scenario.dt,
                               learn_roles=config.marl.learn_roles)
    obs_dim = env.core.obs_builder.get_obs_dim()
    global_dim = env.core.obs_builder.get_global_state_dim()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[IPPO] obs_dim={obs_dim} global_dim={global_dim} K={K} device={device}")
    print(f"[IPPO] centralized_critic={config.marl.centralized_critic} "
          f"(critic input = LOCAL obs, dim={obs_dim})")

    agents = [
        MAPPOAgent(
            agent_id=k, obs_dim=obs_dim, global_state_dim=global_dim,
            action_space=action_space, num_agents=K,
            hidden_layers=config.marl.hidden_layers, lr=config.marl.lr,
            max_grad_norm=config.marl.max_grad_norm, device=device,
            centralized_critic=False,   # IPPO: local-obs critic
        )
        for k in range(K)
    ]

    trainer = MAPPTrainer(env=env, agents=agents, config=config, device=device)
    print(f"\n[IPPO] training {config.marl.num_episodes} episodes (early-stop enabled)...")
    metrics_history = trainer.train(num_episodes=config.marl.num_episodes, log_interval=10)

    print("\n" + "=" * 60)
    print("[IPPO] training complete!")
    print(f"Total frames: {trainer.total_frames}")
    print(f"Best eval steady_P_D: {trainer.best_score:.4f}"
          + (f" (converged @ ep {trainer.converged_episode})" if trainer.converged_episode else ""))
    env.close()


if __name__ == "__main__":
    main()
