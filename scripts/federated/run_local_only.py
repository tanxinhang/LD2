#!/usr/bin/env python
"""Local-only matched control: same budget as federated, no aggregation."""
import sys, os, copy, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer

# Same region configs as federated
REGION_CONFIGS = [
    ('config/fed_region_A.yaml', 'A: standard'),
    ('config/fed_region_B.yaml', 'B: fast targets'),
    ('config/fed_region_C.yaml', 'C: poor comm'),
]

def train_local(cfg_path, seed, warmup=40, rounds=8, local_eps=15):
    cfg = load_config(cfg_path)
    env = UAVISACEnv(config=cfg, seed=seed)
    K, Q = cfg.scenario.K, cfg.scenario.Q
    action_space = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)
    action_space.num_targets = Q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 64

    obs_dim = env.core.obs_builder.get_obs_dim()
    gs_dim = env.core.obs_builder.get_global_state_dim()

    agents = []
    for k in range(K):
        agent = MAPPOAgent(k, obs_dim, gs_dim, action_space, K,
                          hidden_layers=cfg.marl.hidden_layers,
                          lr=cfg.marl.lr, critic_lr_mult=cfg.marl.critic_lr_mult,
                          max_grad_norm=cfg.marl.max_grad_norm, device='cuda')
        agents.append(agent)
    for k in range(1, K):
        agents[k].actor = agents[0].actor
        agents[k].critic = agents[0].critic
        agents[k].actor_optimizer = agents[0].actor_optimizer
        agents[k].critic_optimizer = agents[0].critic_optimizer

    trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device='cuda')
    trainer._bc_actor = None

    # Warmup
    for ep in range(warmup):
        trainer.train_episode()
        trainer._oracle_ep_count += 1
    ev_warmup = trainer._evaluate(5)['eval_steady_P_D']

    # Continue local training (same budget as federated rounds)
    history = [ev_warmup]
    for r in range(rounds):
        for ep in range(local_eps):
            trainer.train_episode()
            trainer._oracle_ep_count += 1
        ev = trainer._evaluate(5)['eval_steady_P_D']
        history.append(ev)

    ev_final = trainer._evaluate(5)
    env.close()
    return {
        'warmup_pd': ev_warmup,
        'history': history,
        'final_pd': ev_final['eval_steady_P_D'],
        'final_worst': ev_final['eval_worst_P_D'],
        'final_weak3': ev_final['eval_weak3_P_D'],
        'final_tstd': ev_final['eval_target_std'],
    }

def main():
    print('Local-only matched control (same budget, no aggregation)')
    print('=' * 60)
    for cfg_path, label in REGION_CONFIGS:
        result = train_local(cfg_path, seed=42)
        print(f'\n{label}:')
        print(f'  Warmup: {result["warmup_pd"]:.3f}')
        for r, pd in enumerate(result['history']):
            marker = ' ←' if r == 0 else ''
            print(f'  Round {r}: {pd:.3f}{marker}')
        print(f'  Final:  steady={result["final_pd"]:.3f} '
              f'worst={result["final_worst"]:.3f} '
              f'weak3={result["final_weak3"]:.3f} '
              f'tstd={result["final_tstd"]:.3f}')

if __name__ == '__main__':
    main()
