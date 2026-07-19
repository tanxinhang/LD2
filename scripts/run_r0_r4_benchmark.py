#!/usr/bin/env python
"""R0-R4 Belief Fusion Safety System — Full Benchmark with D1 Policy.

Loads the frozen D1 policy checkpoint and evaluates the belief-processing
pipeline under R0-R4 configurations. The policy is held constant; only
belief calibration/gating/scheduling/recovery mechanisms vary.

Usage:
    python scripts/run_r0_r4_benchmark.py --seeds 20
    python scripts/run_r0_r4_benchmark.py --levels R0,R2,R4 --seeds 50
"""

import argparse
import copy
import json
import os
import sys
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config.params import load_config
from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.agents.networks import StructuredActorNetwork
from uav_isac.utils.types import Action


# ── D1 checkpoint path ──
D1_CHECKPOINT = "results/dagger_variants/dagger_D1.pt"

# ── R0-R4 flag overrides ──
LEVEL_OVERRIDES = {
    "R0": {
        "desc": "CI+B3, no safety (B4 equiv)",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.0,
        "p0_eta_aoi": 0.0,
        "belief_nis_enabled": False,
        "trust_gate_enabled": False,
        "p0_safe_fallback": False,
        "active_probe_enabled": False,
    },
    "R1": {
        "desc": "+NIS calibration",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.005,
        "p0_eta_aoi": 0.005,
        "belief_nis_enabled": True,
        "trust_gate_enabled": False,
        "p0_safe_fallback": False,
        "active_probe_enabled": False,
    },
    "R2": {
        "desc": "+Disagreement gate",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.005,
        "p0_eta_aoi": 0.005,
        "belief_nis_enabled": True,
        "trust_gate_enabled": True,
        "p0_safe_fallback": False,
        "active_probe_enabled": False,
    },
    "R3": {
        "desc": "+Safe P0 fallback",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.005,
        "p0_eta_aoi": 0.005,
        "belief_nis_enabled": True,
        "trust_gate_enabled": True,
        "p0_safe_fallback": True,
        "active_probe_enabled": False,
    },
    "R4": {
        "desc": "+Active probe+feedback",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.005,
        "p0_eta_aoi": 0.005,
        "belief_nis_enabled": True,
        "trust_gate_enabled": True,
        "p0_safe_fallback": True,
        "active_probe_enabled": True,
    },
    # Reference: no fusion at all (B0 baseline)
    "B0": {
        "desc": "No fusion (B0 ref)",
        "neighbor_belief_fusion": False,
        "p0_beta_uncertainty": 0.0,
        "p0_eta_aoi": 0.0,
        "belief_nis_enabled": False,
        "trust_gate_enabled": False,
        "p0_safe_fallback": False,
        "active_probe_enabled": False,
    },
}


def apply_overrides(cfg, overrides):
    """Apply flat key overrides to config.marl fields."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key == "desc":
            continue
        if hasattr(cfg.marl, key):
            setattr(cfg.marl, key, value)
    return cfg


def load_d1_actor(cfg, device="cpu"):
    """Load frozen D1 actor from checkpoint.

    The D1 checkpoint was trained with rel_features=True and use_p0_sinr_gated=True.
    We must match this config to get compatible observation dimensions.

    Returns:
        actor, K, Q, max_dp
    """
    ckpt = torch.load(D1_CHECKPOINT, map_location=device, weights_only=False)
    actor_state = ckpt.get('actor', ckpt)

    K = cfg.scenario.K
    Q_cfg = cfg.scenario.Q

    # D1 training config: rel_features + P0 info
    # target_enc.0.weight: [64, 18] → 18 dims per target
    # neighbor_gru.weight_ih_l0: [192, 9] → 9 dims per neighbor
    # self_enc.0.weight: [64, 11] → 11 self dims (8 raw + 3 physics)

    entity_dim = 64  # from self_enc.0.weight shape[0]

    # Create a temporary env to get the correct single_frame_dim
    # We need to match the D1 training config for the observation builder
    from uav_isac.environment.env_core import EnvironmentCore
    import numpy as np

    # Override config to match D1 training settings
    cfg_d1 = copy.deepcopy(cfg)
    cfg_d1.marl.rel_features = True
    cfg_d1.marl.use_p0_sinr_gated = True
    cfg_d1.marl.learned_comm_mode = 'off'

    rng = np.random.default_rng(42)
    env_tmp = EnvironmentCore(cfg_d1, rng=rng)
    single_frame_dim = env_tmp.obs_builder.get_single_frame_dim()
    obs_dim_full = env_tmp.obs_builder.get_obs_dim()

    print(f"  D1 compat: single_frame_dim={single_frame_dim}, obs_dim={obs_dim_full}")

    max_dp = cfg.uav.v_max * cfg.scenario.dt  # 25 * 0.1 = 2.5
    actor = StructuredActorNetwork(
        obs_dim=obs_dim_full, K=K, Q=Q_cfg,
        entity_dim=entity_dim, max_dp=max_dp,
        single_frame_dim=single_frame_dim,
    ).to(device)

    # Load weights (allow missing keys for new layers)
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        print(f"  [D1 load] missing keys: {len(missing)} (new layers zero-inited)")
    # Zero-init any new layers
    actor.zero_init_new_layers(set(actor_state.keys()))

    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)

    return actor, K, Q_cfg, max_dp, cfg_d1


def run_episode(env, actor, K, Q, max_dp, device="cpu",
                T=150, steady_window=20):
    """Run one evaluation episode with frozen D1 policy.

    Returns:
        dict with steady_P_D, weak3_P_D, worst_P_D, nis_ema, trust_mean, etc.
    """
    obs_dict, _ = env.reset()

    # Streaming GRU hidden state per agent
    gru_hidden = {}  # k -> (D,) tensor or None

    # Per-frame metrics
    all_pd = []          # (T, Q)
    all_nis_ema = []     # scalar per frame
    all_inflate = []     # scalar per frame
    all_trust = []       # dict per frame
    probe_count = 0

    for t in range(T):
        # Build batched observation
        obs_batch = []
        agent_ids = sorted(obs_dict.keys())
        for k in agent_ids:
            obs_batch.append(obs_dict[k])
        obs_tensor = torch.as_tensor(np.stack(obs_batch), dtype=torch.float32).to(device)

        # Forward through actor (deterministic)
        with torch.no_grad():
            output = actor(obs_tensor, h_prev=None)
            # output: dp_mean, log_std, role_logits, comm_msg, pd_pred, h_new
            mean_actions = output[0]  # (B, 2)

        # Decode actions
        actions = {}
        for idx, k in enumerate(agent_ids):
            dp = mean_actions[idx, :2].cpu().numpy().astype(np.float64)
            # Clamp to max_dp
            norm = np.linalg.norm(dp)
            if norm > max_dp:
                dp = dp / norm * max_dp
            actions[k] = Action(delta_p=dp, role=2)  # role=idle, P0 assigns

        next_obs, rewards, dones, step_info = env.step(actions)

        # Collect per-frame P_D
        if hasattr(step_info, 'P_D_q') and step_info.P_D_q is not None:
            all_pd.append(step_info.P_D_q.copy())

        # Collect NIS diagnostics
        if (env.belief_mgr is not None
                and env.belief_mgr.nis_enabled
                and hasattr(env.belief_mgr, 'nis_ema')):
            all_nis_ema.append(float(np.mean(env.belief_mgr.nis_ema)))
            all_inflate.append(float(np.mean(env.belief_mgr.inflate_factor)))

        # Collect trust diagnostics
        if env._trust_manager is not None:
            all_trust.append(env._trust_manager.get_trust_summary())

        # Count probe triggers
        if hasattr(step_info, 'probe_triggered') and step_info.probe_triggered:
            probe_count += 1

        obs_dict = next_obs
        if dones.get('__all__', False):
            break

    # ── Compute steady-window metrics ──
    if len(all_pd) == 0:
        return {'steady_P_D': 0.0, 'weak3_P_D': 0.0, 'worst_P_D': 0.0}

    pd_array = np.array(all_pd)  # (T_actual, Q)
    n_frames = pd_array.shape[0]
    sw_start = max(0, n_frames - steady_window)
    pd_steady = pd_array[sw_start:]  # (window, Q)

    # Per-target mean over steady window
    per_target_steady = pd_steady.mean(axis=0)  # (Q,)
    steady = float(np.mean(per_target_steady))
    weak3 = float(np.mean(np.sort(per_target_steady)[:3]))
    worst = float(np.min(per_target_steady))

    result = {
        'steady_P_D': steady,
        'weak3_P_D': weak3,
        'worst_P_D': worst,
    }

    if all_nis_ema:
        result['nis_ema_mean'] = float(np.mean(all_nis_ema))
        result['inflate_factor_mean'] = float(np.mean(all_inflate))
    if all_trust:
        result['trust_mean'] = float(np.mean([t['fusion_trust_mean'] for t in all_trust]))
        result['quarantine_mean'] = float(np.mean([t['quarantine_count'] for t in all_trust]))
        result['rejection_rate'] = float(np.mean([t['fusion_rejection_rate'] for t in all_trust]))
    if probe_count > 0:
        result['probe_count'] = probe_count

    return result


def main():
    parser = argparse.ArgumentParser(description="R0-R4 Belief Safety Benchmark (D1 Policy)")
    parser.add_argument('--config', default='config/bench_medium.yaml',
                       help='Base config')
    parser.add_argument('--levels', default='B0,R0,R1,R2,R3,R4',
                       help='Comma-separated levels')
    parser.add_argument('--seeds', type=int, default=20,
                       help='Number of evaluation seeds')
    parser.add_argument('--checkpoint', default=D1_CHECKPOINT,
                       help='Path to D1 policy checkpoint')
    parser.add_argument('--results', default='results/r0_r4_benchmark.json',
                       help='Output JSON path')
    parser.add_argument('--device', default='cpu',
                       help='Device for actor inference')
    args = parser.parse_args()

    levels = [l.strip() for l in args.levels.split(',')]
    base_cfg = load_config(args.config)
    eval_seeds = list(range(30001, 30001 + args.seeds))
    T = base_cfg.scenario.T

    # Load D1 policy once — forces D1-compatible config (rel_features + P0 info)
    print("Loading D1 policy...")
    actor, K, Q, max_dp, cfg_d1 = load_d1_actor(base_cfg, device=args.device)
    print(f"  K={K}, Q={Q}, max_dp={max_dp:.2f}")

    # Use D1-compatible config as the base for ALL levels
    # (belief-processing flags vary by R-level, but observation structure is fixed)
    base_cfg = cfg_d1

    results = []
    print(f"\n{'='*75}")
    print(f"R0-R4 Belief Safety Benchmark (D1 Frozen Policy)")
    print(f"Config: {args.config} | Levels: {levels} | Seeds: {args.seeds}")
    print(f"{'='*75}")

    for level_name in levels:
        if level_name not in LEVEL_OVERRIDES:
            print(f"  SKIP {level_name}: unknown")
            continue

        overrides = LEVEL_OVERRIDES[level_name]
        cfg = apply_overrides(base_cfg, overrides)

        print(f"\n{'─'*55}")
        print(f"  {level_name}: {overrides['desc']}")
        print(f"  NIS={cfg.marl.belief_nis_enabled} "
              f"Gate={cfg.marl.trust_gate_enabled} "
              f"SafeP0={cfg.marl.p0_safe_fallback} "
              f"Probe={cfg.marl.active_probe_enabled}")

        level_results = []
        t0 = time.time()

        for seed_idx, seed in enumerate(eval_seeds):
            rng = np.random.default_rng(seed)
            env = EnvironmentCore(cfg, rng=rng)

            ep_result = run_episode(
                env, actor, K, Q, max_dp,
                device=args.device, T=T,
            )
            level_results.append(ep_result)

            if (seed_idx + 1) % 5 == 0:
                elapsed = time.time() - t0
                recent = level_results[-5:]
                avg_steady = np.mean([r['steady_P_D'] for r in recent])
                print(f"    seed {seed_idx+1}/{args.seeds} "
                      f"(steady={avg_steady:.4f}, {elapsed:.0f}s)")

        elapsed = time.time() - t0

        # Aggregate
        steadies = [r['steady_P_D'] for r in level_results]
        weak3s = [r['weak3_P_D'] for r in level_results]
        worsts = [r['worst_P_D'] for r in level_results]

        agg = {
            'level': level_name,
            'description': overrides['desc'],
            'num_seeds': len(level_results),
            'steady_P_D_mean': float(np.mean(steadies)),
            'steady_P_D_std': float(np.std(steadies)),
            'weak3_P_D_mean': float(np.mean(weak3s)),
            'weak3_P_D_std': float(np.std(weak3s)),
            'worst_P_D_mean': float(np.mean(worsts)),
            'worst_P_D_std': float(np.std(worsts)),
            'elapsed_s': elapsed,
        }

        # Diagnostic means
        for diag_key in ['nis_ema_mean', 'inflate_factor_mean',
                         'trust_mean', 'quarantine_mean',
                         'rejection_rate', 'probe_count']:
            vals = [r[diag_key] for r in level_results if diag_key in r]
            if vals:
                agg[diag_key] = float(np.mean(vals))

        results.append(agg)

        print(f"  => steady: {agg['steady_P_D_mean']:.4f} ± {agg['steady_P_D_std']:.4f}")
        print(f"     weak3:  {agg['weak3_P_D_mean']:.4f} ± {agg['weak3_P_D_std']:.4f}")
        print(f"     worst:  {agg['worst_P_D_mean']:.4f} ± {agg['worst_P_D_std']:.4f}")
        if 'nis_ema_mean' in agg:
            print(f"     NIS: {agg['nis_ema_mean']:.3f}  "
                  f"Trust: {agg.get('trust_mean', 'N/A')}")
        print(f"     time: {elapsed:.0f}s ({elapsed/len(level_results):.1f}s/ep)")

    # ── Comparison table ──
    print(f"\n{'='*75}")
    print(f"{'Level':<6} {'steady':>10} {'weak3':>10} {'worst':>10}  "
          f"{'Δ steady':>10}  Description")
    print(f"{'─'*75}")

    b0_steady = None
    r0_steady = None
    for r in results:
        if r['level'] == 'B0':
            b0_steady = r['steady_P_D_mean']
        if r['level'] == 'R0':
            r0_steady = r['steady_P_D_mean']

    for r in results:
        s = r['steady_P_D_mean']
        w = r['weak3_P_D_mean']
        wr = r['worst_P_D_mean']
        level = r['level']

        # Delta vs R0
        delta_r0 = ""
        if r0_steady is not None and s is not None and level != 'R0':
            delta_r0 = f"{s - r0_steady:+.4f}"

        # Delta vs B0
        delta_b0 = ""
        if b0_steady is not None and s is not None and level != 'B0':
            delta_b0 = f"(vs B0: {s - b0_steady:+.4f})"

        print(f"{level:<6} {s:>10.4f} {w:>10.4f} {wr:>10.4f}  "
              f"{delta_r0:>10}  {r['description']} {delta_b0}")

    # ── Save ──
    os.makedirs(os.path.dirname(args.results) if os.path.dirname(args.results) else 'results',
               exist_ok=True)
    output = {
        'timestamp': datetime.now().isoformat(),
        'base_config': args.config,
        'checkpoint': args.checkpoint,
        'num_seeds': args.seeds,
        'results': results,
    }
    with open(args.results, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {args.results}")


if __name__ == '__main__':
    main()
