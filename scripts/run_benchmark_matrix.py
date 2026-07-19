#!/usr/bin/env python
"""R0-R4 benchmark matrix: measure belief fusion safety system improvements.

R0: CI fusion + B3, NO safety (current B4 equivalent, worst case)
R1: R0 + NIS-driven covariance calibration (Layer 1)
R2: R1 + Disagreement-gated CI fusion (Layer 2)
R3: R2 + Safe P0 with bounded correction (Layer 3)
R4: R3 + Active probing + trust feedback (Layer 4)

Usage:
    python scripts/run_benchmark_matrix.py --seeds 20 --config config/bench_medium.yaml
    python scripts/run_benchmark_matrix.py --levels R0,R1,R2 --seeds 5
"""

import argparse
import copy
import json
import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config, MasterConfig


# ── R0-R4 flag overrides ──
# Each dict specifies the flags that differ from bench_medium.yaml
# All other flags are inherited from the base config.
LEVEL_OVERRIDES = {
    "R0": {
        "description": "CI fusion + B3, NO safety (B4 equivalent, worst case)",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.0,
        "p0_eta_aoi": 0.0,
        "belief_nis_enabled": False,
        "trust_gate_enabled": False,
        "p0_safe_fallback": False,
        "active_probe_enabled": False,
    },
    "R1": {
        "description": "R0 + NIS calibration (calibrated covariance)",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.005,
        "p0_eta_aoi": 0.005,
        "belief_nis_enabled": True,
        "trust_gate_enabled": False,
        "p0_safe_fallback": False,
        "active_probe_enabled": False,
    },
    "R2": {
        "description": "R1 + Disagreement-gated CI fusion",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.005,
        "p0_eta_aoi": 0.005,
        "belief_nis_enabled": True,
        "trust_gate_enabled": True,
        "p0_safe_fallback": False,
        "active_probe_enabled": False,
    },
    "R3": {
        "description": "R2 + Safe P0 with bounded correction",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.005,
        "p0_eta_aoi": 0.005,
        "belief_nis_enabled": True,
        "trust_gate_enabled": True,
        "p0_safe_fallback": True,
        "active_probe_enabled": False,
    },
    "R4": {
        "description": "R3 + Active probing + trust feedback",
        "neighbor_belief_fusion": True,
        "p0_beta_uncertainty": 0.005,
        "p0_eta_aoi": 0.005,
        "belief_nis_enabled": True,
        "trust_gate_enabled": True,
        "p0_safe_fallback": True,
        "active_probe_enabled": True,
    },
}


def apply_overrides(cfg: MasterConfig, overrides: dict) -> MasterConfig:
    """Apply flat key overrides to config.marl fields."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key == "description":
            continue
        if hasattr(cfg.marl, key):
            setattr(cfg.marl, key, value)
    return cfg


def run_single_config(cfg: MasterConfig, level_name: str, seeds: list,
                      results_dir: str) -> dict:
    """Run evaluation for a single R-level config.

    Uses the D1 frozen policy for evaluation (belief-only processing).
    Returns aggregated metrics.
    """
    from uav_isac.environment.env_core import EnvironmentCore

    # Use evaluation seeds
    eval_seeds = getattr(cfg.marl, 'eval_seeds', [10001, 10002, 10003, 10004, 10005])
    if seeds and len(seeds) > 0:
        eval_seeds = seeds

    all_steady = []
    all_weak3 = []
    all_worst = []
    nis_stats = []
    trust_stats = []

    for seed_idx, seed in enumerate(eval_seeds):
        rng = np.random.default_rng(seed)
        env = EnvironmentCore(cfg, rng=rng)
        obs, info = env.reset()

        ep_steady = []
        ep_weak3 = []
        ep_worst = []
        ep_nis_ema = []
        ep_inflate = []
        ep_trust = []

        for t in range(cfg.scenario.T):
            # Random policy (same as B0-B4 benchmark)
            actions = {}
            for k in range(cfg.scenario.K):
                from uav_isac.utils.types import Action
                delta_p = rng.uniform(-2.5, 2.5, size=2)
                delta_p = np.clip(delta_p, -2.5, 2.5)
                actions[k] = Action(delta_p=delta_p.astype(np.float64), role=2)

            next_obs, rewards, dones, step_info = env.step(actions)

            # Collect per-frame diagnostics
            if hasattr(step_info, 'P_D_q') and step_info.P_D_q is not None:
                pd = step_info.P_D_q
                if t >= cfg.scenario.T - 20:  # steady window
                    ep_steady.append(float(np.mean(pd)))
                    ep_weak3.append(float(np.mean(np.sort(pd)[:3])))
                    ep_worst.append(float(np.min(pd)))

            # Collect NIS state if calibration is enabled
            if env.belief_mgr is not None and env.belief_mgr.nis_enabled:
                status = env.belief_mgr.get_all_nis_status()
                ep_nis_ema.append(float(np.mean(status['nis_ema'])))
                ep_inflate.append(float(np.mean(status['inflate_factor'])))

            # Collect trust state if gating is enabled
            if env._trust_manager is not None:
                ts = env._trust_manager.get_trust_summary()
                ep_trust.append(ts)

            if dones.get('__all__', False):
                break

        if ep_steady:
            all_steady.append(np.mean(ep_steady))
            all_weak3.append(np.mean(ep_weak3))
            all_worst.append(np.mean(ep_worst))
        if ep_nis_ema:
            nis_stats.append({
                'nis_ema_mean': float(np.mean(ep_nis_ema)),
                'inflate_mean': float(np.mean(ep_inflate)),
            })
        if ep_trust:
            trust_stats.append({
                k: float(np.mean([t[k] for t in ep_trust]))
                for k in ep_trust[0]
            })

    result = {
        'level': level_name,
        'description': LEVEL_OVERRIDES[level_name]['description'],
        'num_seeds': len(eval_seeds),
        'steady_P_D_mean': float(np.mean(all_steady)) if all_steady else None,
        'steady_P_D_std': float(np.std(all_steady)) if all_steady else None,
        'weak3_P_D_mean': float(np.mean(all_weak3)) if all_weak3 else None,
        'weak3_P_D_std': float(np.std(all_weak3)) if all_weak3 else None,
        'worst_P_D_mean': float(np.mean(all_worst)) if all_worst else None,
        'worst_P_D_std': float(np.std(all_worst)) if all_worst else None,
    }

    # Add diagnostic metrics
    if nis_stats:
        result['nis_ema_mean'] = float(np.mean([s['nis_ema_mean'] for s in nis_stats]))
    if trust_stats:
        for k in trust_stats[0]:
            result[k] = float(np.mean([s[k] for s in trust_stats]))

    return result


def main():
    parser = argparse.ArgumentParser(description="R0-R4 benchmark matrix")
    parser.add_argument('--config', default='config/bench_medium.yaml',
                       help='Base config (default: bench_medium.yaml)')
    parser.add_argument('--levels', default='R0,R1,R2,R3,R4',
                       help='Comma-separated levels to run (default: all)')
    parser.add_argument('--seeds', type=int, default=20,
                       help='Number of evaluation seeds (default: 20)')
    parser.add_argument('--results', default='results/r0_r4_matrix.json',
                       help='Output JSON path')
    parser.add_argument('--csv', default=None,
                       help='Optional CSV output path')
    args = parser.parse_args()

    levels = [l.strip() for l in args.levels.split(',')]
    base_cfg = load_config(args.config)

    # Generate deterministic eval seeds
    eval_seeds = list(range(30001, 30001 + args.seeds))

    results = []
    print(f"{'='*70}")
    print(f"R0-R4 Benchmark Matrix: Calibrate–Gate–Schedule–Recover")
    print(f"Base config: {args.config}")
    print(f"Levels: {levels}")
    print(f"Seeds: {args.seeds}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    for level_name in levels:
        if level_name not in LEVEL_OVERRIDES:
            print(f"  SKIP {level_name}: unknown level")
            continue

        overrides = LEVEL_OVERRIDES[level_name]
        cfg = apply_overrides(base_cfg, overrides)
        desc = overrides['description']

        print(f"\n{'─'*50}")
        print(f"  {level_name}: {desc}")
        print(f"  Flags: nis={cfg.marl.belief_nis_enabled}, "
              f"gate={cfg.marl.trust_gate_enabled}, "
              f"safe_p0={cfg.marl.p0_safe_fallback}, "
              f"probe={cfg.marl.active_probe_enabled}")
        print(f"  B3: beta={cfg.marl.p0_beta_uncertainty}, eta={cfg.marl.p0_eta_aoi}")

        try:
            result = run_single_config(cfg, level_name, eval_seeds, args.results)
            results.append(result)

            # Print summary
            if result['steady_P_D_mean'] is not None:
                print(f"  steady: {result['steady_P_D_mean']:.4f} ± {result['steady_P_D_std']:.4f}")
                print(f"  weak3:  {result['weak3_P_D_mean']:.4f} ± {result['weak3_P_D_std']:.4f}")
                print(f"  worst:  {result['worst_P_D_mean']:.4f} ± {result['worst_P_D_std']:.4f}")
            if 'nis_ema_mean' in result:
                print(f"  NIS EMA: {result['nis_ema_mean']:.3f}")
            if 'fusion_trust_mean' in result:
                print(f"  Trust: {result['fusion_trust_mean']:.3f}, "
                      f"Quarantine: {result.get('quarantine_count', 'N/A')}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Print comparison table ──
    print(f"\n{'='*70}")
    print(f"Summary Matrix")
    print(f"{'='*70}")
    header = f"{'Level':<6} {'steady':>8} {'weak3':>8} {'worst':>8}  Description"
    print(header)
    print(f"{'─'*70}")

    r0_steady = None
    for r in results:
        s = r.get('steady_P_D_mean')
        w = r.get('weak3_P_D_mean')
        wr = r.get('worst_P_D_mean')
        level = r['level']
        if level == 'R0' and s is not None:
            r0_steady = s

        delta = ""
        if r0_steady is not None and s is not None and level != 'R0':
            delta = f"Δ={s - r0_steady:+.4f}"

        print(f"{level:<6} {s or 'N/A':>8} {w or 'N/A':>8} {wr or 'N/A':>8}  "
              f"{r['description'][:50]} {delta}")

    # ── Save results ──
    os.makedirs(os.path.dirname(args.results) if os.path.dirname(args.results) else 'results',
               exist_ok=True)
    output = {
        'timestamp': datetime.now().isoformat(),
        'base_config': args.config,
        'num_seeds': args.seeds,
        'results': results,
    }
    with open(args.results, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.results}")

    # Optional CSV
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            if results:
                all_keys = set()
                for r in results:
                    all_keys.update(r.keys())
                all_keys = sorted(all_keys)
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(results)
        print(f"CSV saved to: {args.csv}")


if __name__ == '__main__':
    main()
