#!/usr/bin/env python
"""Re-evaluate all available checkpoints with the new unified metrics."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.networks import StructuredActorNetwork

CHECKPOINTS = {
    'BC Warmstart': ('results/warmstart_2frame_rel.pt', 'config/exp_800_k8q8.yaml', 454, 64),
    'DAgger v1': ('results/warmstart_k8q8.pt', 'config/exp_800_k8q8.yaml', 248, 128),
    'DAgger v2': ('results/warmstart_k8q8_v2.pt', 'config/exp_800_k8q8.yaml', 251, 128),
    'GRU Dagger': ('results/warmstart_gru_dagger.pt', 'config/exp_800_k8q8.yaml', 454, 64),
}

def evaluate_checkpoint(ckpt_path, cfg_path, obs_dim, entity_dim=64, n_eps=10, steady_w=20):
    """Evaluate with new metrics: episode-wise steady-window aggregation."""
    cfg = load_config(cfg_path)
    K, Q = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    device = 'cuda'

    actor = StructuredActorNetwork(obs_dim=obs_dim, K=K, Q=Q, entity_dim=entity_dim, max_dp=max_dp).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'actor' in ckpt:
        actor.load_state_dict(ckpt['actor'], strict=False)
    else:
        actor.load_state_dict(ckpt, strict=False)
    actor.eval()
    aspace = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt)

    seeds = list(range(20001, 20001 + n_eps))
    ep_steady_means, ep_worst, ep_weak3, ep_tstd = [], [], [], []
    ep_per_target = []

    for seed in seeds:
        env = UAVISACEnv(config=cfg, seed=seed)
        obs, _ = env.reset(seed=seed)
        pd_per_target = []
        while True:
            ob = np.stack([obs[str(k)] for k in range(K)])
            with torch.no_grad():
                dpm, _, _, _, _, _ = actor(torch.as_tensor(ob, dtype=torch.float32, device=device))
            dpm = dpm.cpu().numpy()
            acts = {}
            for k in range(K):
                a, _ = aspace.decode(dpm[k], np.zeros(2), np.zeros(3), dp_deterministic=True)
                acts[str(k)] = {'delta_p': a.delta_p, 'role': 0}
            obs, _, t, tr, info = env.step(acts)
            pd_per_target.append(info['P_D_q'].copy())
            if t.get('__all__') or tr.get('__all__'): break
        env.close()

        w = min(steady_w, len(pd_per_target))
        steady = np.array(pd_per_target[-w:])  # (w, Q)
        steady_pt = steady.mean(axis=0)  # (Q,)
        ep_steady_means.append(float(steady_pt.mean()))
        ep_per_target.append(steady_pt)
        sorted_q = np.sort(steady_pt)
        ep_worst.append(float(sorted_q[0]))
        ep_weak3.append(float(np.mean(sorted_q[:3])))
        ep_tstd.append(float(steady_pt.std()))

    return {
        'steady_mean': float(np.mean(ep_steady_means)),
        'worst_epwise': float(np.mean(ep_worst)),
        'weak3_epwise': float(np.mean(ep_weak3)),
        'tstd_epwise': float(np.mean(ep_tstd)),
        'worst_std': float(np.std(ep_worst)),
        'per_target': np.array(ep_per_target).mean(axis=0).tolist() if ep_per_target else [],
    }

def main():
    print(f'Re-evaluating {len(CHECKPOINTS)} checkpoints with new metrics')
    print(f'{"Checkpoint":<20} {"steady":>8} {"worst":>8} {"weak3":>8} {"tstd":>8} {"worst_std":>8}')
    print('-' * 64)
    for name, (path, cfg, obs_dim, ent_dim) in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f'{name:<20} {"(missing)":>40}')
            continue
        r = evaluate_checkpoint(path, cfg, obs_dim, ent_dim)
        print(f'{name:<20} {r["steady_mean"]:8.4f} {r["worst_epwise"]:8.4f} '
              f'{r["weak3_epwise"]:8.4f} {r["tstd_epwise"]:8.4f} {r["worst_std"]:8.4f}')
        if r['per_target']:
            pts = [f'{p:.3f}' for p in r['per_target']]
            print(f'  per-target: {pts}')

if __name__ == '__main__':
    main()
