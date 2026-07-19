#!/usr/bin/env python
"""Fault injection tests for CG-SR safety system.

Injects four controlled faults and measures:
  - T_detect:  frames from fault to SUSPECT state
  - T_recover: frames from fault to 90% performance recovery
  - peak_drop: maximum steady_P_D drop
  - cum_regret: cumulative detection regret

Fault types:
  1. Covariance collapse: P_j ← 1e-3 * P_j at t_fault
  2. Belief mean bias:   x̂_j ← x̂_j + [50, 50]^T at t_fault
  3. Stale neighbor:     freeze neighbor belief for 20 frames
  4. Common-mode mismatch: all nodes CV, targets CA

Usage:
    python scripts/fault_injection_test.py --fault 1 --seeds 10
    python scripts/fault_injection_test.py --fault all --seeds 20 --config config/bench_hard_clean.yaml
"""

import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import load_config
from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.utils.types import Action
import copy


# ── Fault injection functions ──

def inject_fault_cov_collapse(env, fault_uav, fault_target, t, t_fault):
    """Fault 1: Covariance collapse — P ← 1e-3 * P at t=t_fault."""
    if t == t_fault and env.belief_mgr is not None:
        for q in range(env.Q):
            env.belief_mgr.cov[fault_uav, q] *= 0.001

def inject_fault_belief_bias(env, fault_uav, fault_target, t, t_fault):
    """Fault 2: Belief mean bias — x̂ ← x̂ + [50, 50]^T at t=t_fault."""
    if t == t_fault and env.belief_mgr is not None:
        for q in range(env.Q):
            env.belief_mgr.mean[fault_uav, q, :2] += [50.0, 50.0]

def inject_fault_stale_neighbor(env, fault_uav, fault_target, t, t_fault):
    """Fault 3: Freeze neighbor belief from t_fault to t_fault+20."""
    if t_fault <= t < t_fault + 20 and env.belief_mgr is not None:
        # Store initial state at t_fault
        if not hasattr(env, '_fault_frozen_mean'):
            env._fault_frozen_mean = {}
            env._fault_frozen_cov = {}
            for q in range(env.Q):
                env._fault_frozen_mean[(fault_uav, q)] = env.belief_mgr.mean[fault_uav, q].copy()
                env._fault_frozen_cov[(fault_uav, q)] = env.belief_mgr.cov[fault_uav, q].copy()
            env._fault_frozen_aoi = {}
        # Restore frozen state after each step
        for q in range(env.Q):
            if (fault_uav, q) in env._fault_frozen_mean:
                env.belief_mgr.mean[fault_uav, q] = env._fault_frozen_mean[(fault_uav, q)].copy()
                env.belief_mgr.cov[fault_uav, q] = env._fault_frozen_cov[(fault_uav, q)].copy()

# ── Test harness ──

def run_fault_test(cfg, fault_type, fault_injector, seeds=10, T=150):
    """Run fault injection test and measure recovery metrics.

    Args:
        cfg: MasterConfig (should have safety layers enabled)
        fault_type: int 1-4
        fault_injector: callable(env, fault_uav, fault_target, t, t_fault)
        seeds: number of seeds
        T: episode length

    Returns:
        dict with recovery metrics
    """
    t_fault = T // 3  # inject fault at 1/3 of episode

    all_t_detect = []
    all_t_recover = []
    all_peak_drop = []
    all_cum_regret = []
    all_baselines = []
    all_fault_nis = []
    all_fault_infl = []

    for seed in range(30001, 30001 + seeds):
        rng = np.random.default_rng(seed)
        env = EnvironmentCore(cfg, rng=rng)
        env.reset()

        fault_uav = 1  # inject fault on UAV 1
        fault_target = 0

        pd_per_frame = []        # per-frame mean P_D
        nis_per_frame = []       # per-frame mean NIS EMA (if enabled)
        infl_per_frame = []      # per-frame mean inflation
        state_per_frame = []     # per-frame fault UAV state (0/1/2)

        # Pre-fault baseline (first 30 frames of steady state)
        pre_fault_pd = []

        detected = False
        t_detect = -1
        t_recover = -1

        for t_frame in range(T):
            # Homing policy
            actions = {}
            for k in range(cfg.scenario.K):
                uav_pos = env.uavs[k].pos[:2]
                best_d, best_dir = float('inf'), np.zeros(2)
                for q in range(cfg.scenario.Q):
                    b = env.belief_mgr.get_belief(k, q)
                    d = np.linalg.norm(b.mean[:2] - uav_pos)
                    if d < best_d:
                        best_d = d
                        best_dir = (b.mean[:2] - uav_pos) / max(d, 1e-6)
                dp = best_dir * 1.25
                actions[k] = Action(delta_p=dp.astype(np.float64), role=2)

            # Inject fault
            fault_injector(env, fault_uav, fault_target, t_frame, t_fault)

            # Step
            _, _, dones, info = env.step(actions)

            if info.P_D_q is not None:
                pd_per_frame.append(float(np.mean(info.P_D_q)))

            if env.belief_mgr is not None and env.belief_mgr.nis_enabled:
                nis_per_frame.append(float(np.mean(env.belief_mgr.nis_ema)))
                infl_per_frame.append(float(np.mean(env.belief_mgr.inflate_factor)))
                state_per_frame.append(int(env.belief_mgr.nis_state[fault_uav, fault_target]))

            if t_frame < t_fault:
                if info.P_D_q is not None:
                    pre_fault_pd.append(float(np.mean(info.P_D_q)))

            # Detection: first frame where fault UAV enters SUSPECT state
            if not detected and t_frame >= t_fault:
                if (env.belief_mgr is not None and env.belief_mgr.nis_enabled
                        and env.belief_mgr.nis_state[fault_uav, fault_target] >= 1):
                    detected = True
                    t_detect = t_frame - t_fault

            if dones.get('__all__', False):
                break

        # ── Compute metrics ──
        if len(pre_fault_pd) >= 10:
            baseline_pd = np.mean(pre_fault_pd[-10:])
        elif len(pre_fault_pd) > 0:
            baseline_pd = np.mean(pre_fault_pd)
        else:
            baseline_pd = 0.0

        pd_arr = np.array(pd_per_frame)
        post_fault = pd_arr[t_fault:]

        if len(post_fault) > 0:
            peak_drop = baseline_pd - np.min(post_fault)
            cum_regret = np.sum(np.maximum(0, baseline_pd - post_fault))
        else:
            peak_drop = 0.0
            cum_regret = 0.0

        # Recovery: first frame after fault where P_D returns to 90% of baseline
        if len(post_fault) > 0:
            recovery_mask = post_fault >= 0.9 * baseline_pd
            if np.any(recovery_mask):
                t_recover = np.argmax(recovery_mask)
            else:
                t_recover = len(post_fault)  # never recovered

        all_t_detect.append(t_detect if t_detect >= 0 else float('inf'))
        all_peak_drop.append(peak_drop)
        all_cum_regret.append(cum_regret)
        all_baselines.append(baseline_pd)
        if t_recover >= 0:
            all_t_recover.append(t_recover)
        if nis_per_frame:
            all_fault_nis.append(np.mean(nis_per_frame[t_fault:]) if len(nis_per_frame) > t_fault else 0)
            all_fault_infl.append(np.mean(infl_per_frame[t_fault:]) if len(infl_per_frame) > t_fault else 0)

    # Aggregate
    finite_detect = [d for d in all_t_detect if d < float('inf')]
    return {
        'fault_type': fault_type,
        'num_seeds': seeds,
        'baseline_P_D': float(np.mean(all_baselines)),
        'T_detect_mean': float(np.mean(finite_detect)) if finite_detect else float('inf'),
        'T_detect_std': float(np.std(finite_detect)) if finite_detect else 0,
        'detection_rate': len(finite_detect) / seeds,
        'T_recover_mean': float(np.mean(all_t_recover)) if all_t_recover else float('inf'),
        'T_recover_std': float(np.std(all_t_recover)) if all_t_recover else 0,
        'recovery_rate': len(all_t_recover) / seeds,
        'peak_drop_mean': float(np.mean(all_peak_drop)),
        'peak_drop_std': float(np.std(all_peak_drop)),
        'cum_regret_mean': float(np.mean(all_cum_regret)),
        'cum_regret_std': float(np.std(all_cum_regret)),
        'nis_post_fault_mean': float(np.mean(all_fault_nis)) if all_fault_nis else None,
        'infl_post_fault_mean': float(np.mean(all_fault_infl)) if all_fault_infl else None,
    }


def main():
    parser = argparse.ArgumentParser(description="CG-SR Fault Injection Tests")
    parser.add_argument('--fault', default='all',
                       help='Fault type: 1,2,3,4, or all')
    parser.add_argument('--seeds', type=int, default=10,
                       help='Number of evaluation seeds')
    parser.add_argument('--config', default='config/bench_medium.yaml',
                       help='Base config')
    parser.add_argument('--hard', action='store_true',
                       help='Use Hard scenario (CA targets)')
    args = parser.parse_args()

    if args.hard:
        args.config = 'config/bench_hard_clean.yaml'

    base = load_config(args.config)
    base.marl.rel_features = True
    base.marl.use_p0_sinr_gated = True

    # Enable full safety (no gate for homing policy)
    base.marl.neighbor_belief_fusion = True
    base.marl.belief_nis_enabled = True
    base.marl.trust_gate_enabled = False  # no gate
    base.marl.p0_safe_fallback = True
    base.marl.active_probe_enabled = True
    base.marl.p0_beta_uncertainty = 0.005
    base.marl.p0_eta_aoi = 0.005

    faults = {
        1: ("Cov collapse (P←1e-3·P)", inject_fault_cov_collapse),
        2: ("Belief bias (+50m)", inject_fault_belief_bias),
        3: ("Stale neighbor (20fr)", inject_fault_stale_neighbor),
    }

    if args.fault == 'all':
        fault_list = list(faults.keys())
    else:
        fault_list = [int(args.fault)]

    print(f"{'='*70}")
    print(f"CG-SR Fault Injection Tests")
    print(f"Config: {args.config} | Seeds: {args.seeds}")
    print(f"{'='*70}")

    for ft in fault_list:
        name, injector = faults[ft]
        print(f"\n{'─'*50}")
        print(f"  Fault {ft}: {name}")

        t0 = time.time()
        result = run_fault_test(base, ft, injector, seeds=args.seeds)
        elapsed = time.time() - t0

        print(f"  Baseline P_D:        {result['baseline_P_D']:.4f}")
        print(f"  Detection rate:       {result['detection_rate']:.1%}")
        print(f"  T_detect (frames):    {result['T_detect_mean']:.1f} ± {result['T_detect_std']:.1f}")
        print(f"  Recovery rate:        {result['recovery_rate']:.1%}")
        print(f"  T_recover (frames):   {result['T_recover_mean']:.1f} ± {result['T_recover_std']:.1f}")
        print(f"  Peak P_D drop:        {result['peak_drop_mean']:.4f} ± {result['peak_drop_std']:.4f}")
        print(f"  Cumulative regret:    {result['cum_regret_mean']:.3f} ± {result['cum_regret_std']:.3f}")
        if result['nis_post_fault_mean'] is not None:
            print(f"  NIS post-fault:       {result['nis_post_fault_mean']:.3f}")
            print(f"  Infl post-fault:      {result['infl_post_fault_mean']:.3f}")
        print(f"  Elapsed:              {elapsed:.0f}s")


if __name__ == '__main__':
    main()
