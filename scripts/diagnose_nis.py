#!/usr/bin/env python
"""Step 1: Verify NIS calibration detects and responds to miscalibration.

Runs a single episode with controlled model mismatch and tracks:
  - NIS EMA per target
  - Inflation factor λ
  - Covariance trace (should expand under mismatch)
  - Recovery when mismatch is removed

Test scenarios:
  A. Well-calibrated (sigma_a matches true process noise)
  B. Severe Q mismatch (filter sigma_a=0.1, true sigma_a=3.0)
  C. Recovery test (start with mismatch, then fix it)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import load_config
from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.utils.types import Action


def run_diagnostic(cfg, label, T=150, inject_mismatch=False,
                   mismatch_sigma_a=3.0, recovery_frame=None):
    """Run one episode with optional model mismatch injection.

    Args:
        cfg: MasterConfig (should have belief_nis_enabled=True)
        label: Human-readable label for this run
        T: Episode length
        inject_mismatch: If True, add extra process noise after Kalman predict
        mismatch_sigma_a: True process noise sigma (filter uses cfg.target.sigma_a)
        recovery_frame: If set, stop injecting mismatch after this frame

    Returns:
        dict with per-frame metrics
    """
    rng = np.random.default_rng(42)
    env = EnvironmentCore(cfg, rng=rng)
    obs, info = env.reset()

    # Tracking
    nis_ema_history = []      # (T, Q) mean NIS per target
    inflate_history = []       # (T,) mean inflation
    cov_trace_history = []     # (T,) mean cov trace
    pd_history = []            # (T, Q) per-target P_D
    aoi_history = []           # (T,) mean AoI

    for t in range(T):
        # Simple random-homing policy: move toward nearest target belief
        actions = {}
        for k in range(cfg.scenario.K):
            # Get this UAV's belief about nearest target
            best_dist = float('inf')
            best_dir = np.zeros(2)
            uav_pos = env.uavs[k].pos[:2]
            for q in range(cfg.scenario.Q):
                b = env.belief_mgr.get_belief(k, q)
                tgt_pos = b.mean[:2]
                d = np.linalg.norm(tgt_pos - uav_pos)
                if d < best_dist:
                    best_dist = d
                    best_dir = (tgt_pos - uav_pos) / max(d, 1e-6)

            # Move toward nearest target (half max speed)
            delta_p = best_dir * cfg.uav.v_max * cfg.scenario.dt * 0.5
            actions[k] = Action(delta_p=delta_p.astype(np.float64), role=2)

        # ── Inject model mismatch: corrupt position prediction ──
        # CV Kalman Q is tiny (q_p ~ dt^4), so NIS is insensitive to small
        # continuous drifts when measurement noise R=225m² dominates S.
        # Instead, inject sudden position jumps that simulate unmodeled maneuvers
        # or CA motion → large innovation → NIS should detect it.
        if inject_mismatch and (recovery_frame is None or t < recovery_frame):
            # Every 10 frames, inject a large position jump (simulates CA turn)
            if t % 10 == 0:
                jump_magnitude = 80.0  # 80m position jump → innovation ~80m >> σ_R=15m
                for k in range(cfg.scenario.K):
                    for q in range(cfg.scenario.Q):
                        angle = rng.uniform(0, 2 * np.pi)
                        env.belief_mgr.mean[k, q, 0] += jump_magnitude * np.cos(angle)
                        env.belief_mgr.mean[k, q, 1] += jump_magnitude * np.sin(angle)

        next_obs, rewards, dones, step_info = env.step(actions)

        # Collect NIS diagnostics
        if env.belief_mgr.nis_enabled:
            nis_ema_history.append(env.belief_mgr.nis_ema.mean(axis=0).copy())  # per-target
            inflate_history.append(float(np.mean(env.belief_mgr.inflate_factor)))
            cov_trace_history.append(float(np.mean([
                np.trace(env.belief_mgr.cov[k, q])
                for k in range(cfg.scenario.K)
                for q in range(cfg.scenario.Q)
            ])))

        if hasattr(step_info, 'P_D_q') and step_info.P_D_q is not None:
            pd_history.append(step_info.P_D_q.copy())

        aoi_history.append(float(np.mean(env.belief_mgr.aoi)))

        obs = next_obs
        if dones.get('__all__', False):
            break

    # Aggregate
    nis_arr = np.array(nis_ema_history)  # (T_actual, Q)
    result = {
        'label': label,
        'frames': len(nis_arr),
        'nis_ema_per_target': nis_arr.mean(axis=0).tolist(),  # per-target mean over time
        'nis_ema_overall': float(nis_arr.mean()),
        'nis_ema_final': float(nis_arr[-10:].mean()) if len(nis_arr) >= 10 else float(nis_arr.mean()),
        'inflate_mean': float(np.mean(inflate_history)) if inflate_history else None,
        'inflate_final': float(np.mean(inflate_history[-10:])) if len(inflate_history) >= 10 else None,
        'cov_trace_mean': float(np.mean(cov_trace_history)) if cov_trace_history else None,
        'cov_trace_final': float(np.mean(cov_trace_history[-10:])) if len(cov_trace_history) >= 10 else None,
        'pd_steady': float(np.mean([np.mean(p) for p in pd_history[-20:]])) if len(pd_history) >= 20 else None,
        'aoi_mean': float(np.mean(aoi_history)),
        'nis_time_series': [float(x) for x in np.array(nis_ema_history).mean(axis=1)],
        'inflate_time_series': [float(x) for x in inflate_history],
    }
    return result


def main():
    base_cfg = load_config('config/bench_medium.yaml')
    # Enable NIS but nothing else
    base_cfg.marl.belief_nis_enabled = True
    base_cfg.marl.neighbor_belief_fusion = False
    base_cfg.marl.trust_gate_enabled = False
    base_cfg.marl.p0_safe_fallback = False
    base_cfg.marl.active_probe_enabled = False
    # Use rel_features + P0 info for D1 compat, but we use homing policy
    base_cfg.marl.rel_features = True
    base_cfg.marl.use_p0_sinr_gated = True

    T = 150

    print("=" * 70)
    print("Step 1: NIS Calibration Diagnostic")
    print(f"Filter sigma_a: {base_cfg.target.sigma_a}")
    print("=" * 70)

    # Test A: Well-calibrated (filter sigma_a matches true process)
    cfg_a = base_cfg
    cfg_a.target.sigma_a = 3.0  # match bench_medium
    r_a = run_diagnostic(cfg_a, "A: Well-calibrated (σ_a=3.0)", T=T,
                         inject_mismatch=False)
    print(f"\n{'─'*50}")
    print(f"Test A: Well-calibrated (σ_a=3.0, no injected mismatch)")
    print(f"  NIS EMA: overall={r_a['nis_ema_overall']:.3f}, "
          f"final={r_a['nis_ema_final']:.3f}")
    print(f"  Inflation λ: mean={r_a['inflate_mean']:.3f}, "
          f"final={r_a['inflate_final']:.3f}")
    print(f"  Cov trace: mean={r_a['cov_trace_mean']:.1f}, "
          f"final={r_a['cov_trace_final']:.1f}")
    print(f"  P_D steady: {r_a['pd_steady']:.4f}")
    print(f"  AoI mean: {r_a['aoi_mean']:.1f}")
    print(f"  NIS per target: {[f'{x:.3f}' for x in r_a['nis_ema_per_target']]}")

    # Test B: Severe Q mismatch (filter σ_a=0.5, true σ_a=3.0)
    cfg_b = load_config('config/bench_medium.yaml')
    cfg_b.marl.belief_nis_enabled = True
    cfg_b.marl.neighbor_belief_fusion = False
    cfg_b.marl.trust_gate_enabled = False
    cfg_b.marl.p0_safe_fallback = False
    cfg_b.marl.active_probe_enabled = False
    cfg_b.marl.rel_features = True
    cfg_b.marl.use_p0_sinr_gated = True
    cfg_b.target.sigma_a = 0.5  # filter Q is tiny, true process is large

    r_b = run_diagnostic(cfg_b, "B: Q mismatch (filter σ_a=0.5, true σ≈3.0)", T=T,
                         inject_mismatch=True, mismatch_sigma_a=3.0)
    print(f"\n{'─'*50}")
    print(f"Test B: Q Mismatch (filter σ_a=0.5, injected σ=3.0)")
    print(f"  NIS EMA: overall={r_b['nis_ema_overall']:.3f}, "
          f"final={r_b['nis_ema_final']:.3f}")
    print(f"  Inflation λ: mean={r_b['inflate_mean']:.3f}, "
          f"final={r_b['inflate_final']:.3f}")
    print(f"  Cov trace: mean={r_b['cov_trace_mean']:.1f}, "
          f"final={r_b['cov_trace_final']:.1f}")
    print(f"  P_D steady: {r_b['pd_steady']:.4f}")
    print(f"  AoI mean: {r_b['aoi_mean']:.1f}")

    # Test C: Recovery (mismatch first 80 frames, then fix)
    cfg_c = load_config('config/bench_medium.yaml')
    cfg_c.marl.belief_nis_enabled = True
    cfg_c.marl.neighbor_belief_fusion = False
    cfg_c.marl.trust_gate_enabled = False
    cfg_c.marl.p0_safe_fallback = False
    cfg_c.marl.active_probe_enabled = False
    cfg_c.marl.rel_features = True
    cfg_c.marl.use_p0_sinr_gated = True
    cfg_c.target.sigma_a = 0.5

    r_c = run_diagnostic(cfg_c, "C: Recovery (mismatch→fixed at t=80)", T=T,
                         inject_mismatch=True, mismatch_sigma_a=3.0,
                         recovery_frame=80)
    print(f"\n{'─'*50}")
    print(f"Test C: Recovery (mismatch → fixed at t=80)")
    print(f"  NIS EMA: overall={r_c['nis_ema_overall']:.3f}, "
          f"final={r_c['nis_ema_final']:.3f}")
    print(f"  Inflation λ: mean={r_c['inflate_mean']:.3f}, "
          f"final={r_c['inflate_final']:.3f}")
    print(f"  Cov trace: mean={r_c['cov_trace_mean']:.1f}, "
          f"final={r_c['cov_trace_final']:.1f}")
    print(f"  P_D steady: {r_c['pd_steady']:.4f}")

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("Analysis")
    print(f"{'='*70}")

    # NIS should be ~1.0 for well-calibrated, >>1.0 for mismatched
    nis_ok_a = 0.7 < r_a['nis_ema_final'] < 1.5
    nis_ok_b = r_b['nis_ema_final'] > r_a['nis_ema_final'] * 1.5
    nis_ok_c = r_c['nis_ema_final'] < r_c['nis_ema_overall']  # recovered

    print(f"\n  1. NIS detection:")
    print(f"     A (calibrated): NIS={r_a['nis_ema_final']:.3f} "
          f"{'[OK] ~1.0' if nis_ok_a else '[FAIL] out of range'}")
    print(f"     B (mismatched): NIS={r_b['nis_ema_final']:.3f} "
          f"{'[OK] > calibrated' if nis_ok_b else '[FAIL] not elevated'}")
    print(f"     C (recovered):  NIS={r_c['nis_ema_final']:.3f} "
          f"{'[OK] decayed' if nis_ok_c else '[FAIL] did not recover'}")

    # Inflation should respond to NIS
    inflate_ok = r_b['inflate_final'] > 1.5
    print(f"\n  2. Covariance inflation:")
    print(f"     A (calibrated): lambda={r_a['inflate_final']:.3f} (expect ~1.0)")
    print(f"     B (mismatched): lambda={r_b['inflate_final']:.3f} "
          f"{'[OK] lambda>1.5' if inflate_ok else '[FAIL] no inflation response'}")

    # Covariance should be larger under mismatch
    cov_ok = r_b['cov_trace_final'] > r_a['cov_trace_final'] * 1.2
    print(f"\n  3. Covariance expansion:")
    print(f"     A (calibrated): trace={r_a['cov_trace_final']:.1f}")
    print(f"     B (mismatched): trace={r_b['cov_trace_final']:.1f} "
          f"{'[OK] expanded' if cov_ok else '[FAIL] no expansion'}")

    all_ok = nis_ok_a and nis_ok_b and nis_ok_c and inflate_ok and cov_ok
    print(f"\n  Overall: {'[OK] ALL CHECKS PASSED' if all_ok else '[FAIL] SOME CHECKS FAILED'}")


if __name__ == '__main__':
    main()
