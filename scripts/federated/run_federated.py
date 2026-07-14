#!/usr/bin/env python
"""Federated C3-Stable: multi-region training with encoder-only aggregation.

Each client = full 8-UAV swarm in one region.
Aggregates: self_enc, target_enc, neighbor_gru, neighbor_proj, attn, attn_norm
Keeps local: dp_head, comm_head, role_head, dp_log_std, comm_proj, gate, critic
"""

import sys, os, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from config.params import get_default_config, load_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer


# ── Which params to aggregate (FedPer: encoder-only) ──
SHARED_KEYS = {'self_enc', 'target_enc', 'neighbor_gru', 'neighbor_proj',
               'attn', 'attn_norm', 'global_enc'}


def get_encoder_params(agent):
    """Extract shared encoder state_dict."""
    full = agent.actor.state_dict()
    return {k: v.clone() for k, v in full.items()
            if any(k.startswith(prefix) for prefix in SHARED_KEYS)}


def set_encoder_params(agent, enc_state):
    """Replace shared encoder params, keep local heads."""
    full = agent.actor.state_dict()
    full.update(enc_state)
    agent.actor.load_state_dict(full)


def federated_average(enc_states, weights=None):
    """Weighted average of encoder parameters."""
    if weights is None:
        weights = [1.0 / len(enc_states)] * len(enc_states)
    avg = {}
    for k in enc_states[0]:
        avg[k] = sum(w * s[k] for w, s in zip(weights, enc_states))
    return avg


class FederatedTrainer:
    """Orchestrates multi-region federated training."""

    def __init__(self, config_paths, base_seed=42, device='cuda'):
        self.device = device
        self.n_clients = len(config_paths)
        self.base_seed = base_seed

        # Create trainers for each client
        self.clients = []
        for i, cfg_path in enumerate(config_paths):
            cfg = load_config(cfg_path)
            env = UAVISACEnv(config=cfg, seed=base_seed + i * 100)
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
                                  max_grad_norm=cfg.marl.max_grad_norm, device=device)
                agents.append(agent)

            # Share actor/critic across agents
            for k in range(1, K):
                agents[k].actor = agents[0].actor
                agents[k].critic = agents[0].critic
                agents[k].actor_optimizer = agents[0].actor_optimizer
                agents[k].critic_optimizer = agents[0].critic_optimizer

            trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device=device)
            trainer._bc_actor = None  # from scratch
            self.clients.append({
                'cfg': cfg, 'env': env, 'agents': agents, 'trainer': trainer,
                'name': os.path.basename(cfg_path).replace('.yaml', '')
            })

    def local_train(self, client_idx, n_episodes):
        """Run local training for n_episodes."""
        c = self.clients[client_idx]
        # Ensure encoder is in sync before training
        if hasattr(self, 'global_encoder'):
            set_encoder_params(c['agents'][0], self.global_encoder)
        trainer = c['trainer']
        for ep in range(n_episodes):
            metrics = trainer.train_episode()
            trainer._oracle_ep_count += 1
        # Evaluate
        ev = trainer._evaluate(5)
        return ev['eval_steady_P_D']

    def evaluate_all(self):
        """Evaluate all clients."""
        results = {}
        for i, c in enumerate(self.clients):
            ev = c['trainer']._evaluate(5)
            results[c['name']] = ev
        return results

    def run(self, warmup_eps=40, fed_rounds=8, local_eps=15, server_ema=0.3):
        """Run federated training loop."""
        print(f'Federated C3-Stable: {self.n_clients} clients, '
              f'warmup={warmup_eps}, rounds={fed_rounds}, local_eps={local_eps}')
        print('=' * 60)

        # ── Phase 1: Local warmup ──
        print('[Phase 1] Local warmup...')
        for i in range(self.n_clients):
            pd = self.local_train(i, warmup_eps)
            print(f'  {self.clients[i]["name"]}: P_D={pd:.3f}')

        # ── Phase 2: Federated rounds ──
        print('[Phase 2] Federated training...')
        enc_states = [get_encoder_params(c['agents'][0]) for c in self.clients]
        self.global_encoder = federated_average(enc_states)

        for r in range(fed_rounds):
            # Distribute global encoder
            for c in self.clients:
                set_encoder_params(c['agents'][0], self.global_encoder)

            # Local training
            round_pds = []
            for i in range(self.n_clients):
                pd = self.local_train(i, local_eps)
                round_pds.append(pd)

            # Collect encoder states
            enc_states = []
            weights = []
            for i, c in enumerate(self.clients):
                enc_states.append(get_encoder_params(c['agents'][0]))
                weights.append(1.0 / self.n_clients)

            # Server aggregation with EMA
            new_global = federated_average(enc_states, weights)
            for k in self.global_encoder:
                self.global_encoder[k] = (1 - server_ema) * self.global_encoder[k] + server_ema * new_global[k]

            # Eval
            mean_pd = np.mean(round_pds)
            ev = self.evaluate_all()
            pd_vals = [v['eval_steady_P_D'] for v in ev.values()]
            print(f'  Round {r+1}/{fed_rounds}: mean={mean_pd:.3f} '
                  f'clients={[f"{p:.3f}" for p in pd_vals]} '
                  f'worst={min(pd_vals):.3f}')

        # ── Final eval ──
        print('\n[Final] All-client evaluation:')
        ev = self.evaluate_all()
        for name, e in ev.items():
            print(f'  {name}: steady={e["eval_steady_P_D"]:.3f} '
                  f'worst={e["eval_worst_P_D"]:.3f} '
                  f'weak3={e["eval_weak3_P_D"]:.3f} '
                  f'tstd={e["eval_target_std"]:.3f}')
        return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=None,
                    help='config files per client (default: 3 regions)')
    ap.add_argument('--warmup', type=int, default=40)
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--local-eps', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.configs is None:
        # Default: 3 non-IID regions
        base = 'config/exp_800_k8q8.yaml'
        # Create variant configs by modifying the base
        import yaml
        with open(base) as f:
            base_cfg = yaml.safe_load(f)

        configs = []
        for i, (speed_range, los_a, los_b) in enumerate([
            ([0, 5], 4.88, 0.43),     # Region A: standard
            ([2, 10], 4.88, 0.43),    # Region B: fast targets
            ([0, 5], 7.0, 0.30),      # Region C: poor comm (more NLoS)
        ]):
            cfg = copy.deepcopy(base_cfg)
            cfg['target']['speed_range'] = list(speed_range)
            cfg['channel']['los_a'] = los_a
            cfg['channel']['los_b'] = los_b
            path = f'config/fed_region_{chr(65+i)}.yaml'
            with open(path, 'w') as f:
                yaml.dump(cfg, f)
            configs.append(path)
            print(f'Created {path}: speed={speed_range} los_a={los_a} los_b={los_b}')
    else:
        configs = args.configs

    ft = FederatedTrainer(configs, base_seed=args.seed)
    ft.run(warmup_eps=args.warmup, fed_rounds=args.rounds,
           local_eps=args.local_eps)


if __name__ == '__main__':
    import yaml
    main()
