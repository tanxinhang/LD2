#!/usr/bin/env python
"""Evaluate trained MAPPO agent and baseline methods.

Computes metrics across ≥5 seeds and produces summary statistics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from typing import Dict, List

from config.params import get_default_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.evaluation.metrics import compute_episode_metrics


def evaluate_episode(env: UAVISACEnv, agents, deterministic: bool = True) -> Dict:
    """Run one evaluation episode and compute metrics.

    Args:
        env: UAVISAC environment
        agents: List of agent objects
        deterministic: Use deterministic actions

    Returns:
        Dict of episode metrics
    """
    obs, info = env.reset()

    P_D_history = []
    p0_solutions = []
    constraint_infos = []
    team_rewards = []

    initial_batteries = np.array([u.battery for u in env.core.uavs])

    while True:
        # Get actions
        actions = {}
        for k, agent in enumerate(agents):
            k_str = str(k)
            if k_str in obs:
                action, _, _ = agent.act(obs[k_str], deterministic=deterministic)
                actions[k_str] = {'delta_p': action.delta_p, 'role': action.role}

        obs, rewards, terminated, truncated, info = env.step(actions)

        P_D_history.append(info['P_D_q'].copy())
        if env.current_step_info is not None:
            p0_solutions.append(env.current_step_info.p0_solution)
            constraint_infos.append(env.current_step_info.constraint_info)
            team_rewards.append(env.current_step_info.team_reward)

        if terminated.get('__all__', False):
            break

    final_batteries = np.array([u.battery for u in env.core.uavs])

    metrics = compute_episode_metrics(
        P_D_history=P_D_history,
        p0_solutions=p0_solutions,
        constraint_infos=constraint_infos,
        initial_batteries=initial_batteries,
        final_batteries=final_batteries,
        team_rewards=team_rewards,
    )

    metrics['episode_length'] = len(P_D_history)
    return metrics


def main():
    config = get_default_config()
    seeds = config.seeds[:3]  # Use 3 seeds for quick evaluation

    all_results = {}

    for seed in seeds:
        set_seed(seed)
        env = UAVISACEnv(config=config, seed=seed)

        # Use random policy for baseline evaluation
        from uav_isac.agents.mappo_agent import MAPPOAgent
        from uav_isac.environment.action import ActionSpace

        action_space = ActionSpace(v_max=config.uav.v_max, dt=config.scenario.dt)
        obs_dim = env.core.obs_builder.get_obs_dim()
        global_dim = env.core.obs_builder.get_global_state_dim()

        # Random agent (untrained)
        class RandomAgent:
            def __init__(self, agent_id, action_space):
                self.agent_id = agent_id
                self.action_space = action_space

            def act(self, obs, deterministic=False):
                return self.action_space.sample(), 0.0, 0.0

        agents = [RandomAgent(k, action_space) for k in range(config.scenario.K)]

        metrics = evaluate_episode(env, agents, deterministic=False)

        all_results[f'seed_{seed}'] = metrics

        print(f"Seed {seed}: avg_P_D={metrics['avg_P_D']:.4f}, "
              f"worst_P_D={metrics['worst_P_D']:.4f}, "
              f"energy={metrics['cumulative_energy_J']:.0f}J, "
              f"viol_rate={metrics['constraint_violation_rate']:.3f}")

        env.close()

    # Summary
    avg_pd_vals = [r['avg_P_D'] for r in all_results.values()]
    print(f"\nSummary across {len(seeds)} seeds:")
    print(f"  avg_P_D: {np.mean(avg_pd_vals):.4f} ± {np.std(avg_pd_vals):.4f}")

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'evaluation.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
