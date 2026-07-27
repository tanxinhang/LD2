#!/usr/bin/env python
"""Offline validation: does probing the most ambiguous target reduce regret?

For high-ambiguity frames, we simulate two counterfactuals:
  1. ACTUAL: the observation that actually occurred (P0's selected pairings)
  2. PROBE:  if we had instead forced an observation of the most ambiguous target

We then compare the regret in the SUBSEQUENT frame to determine
whether probing the ambiguous target would have reduced regret.

This is a BEFORE-IMPLEMENTATION gate: if predicted regret reduction
does not correlate with actual regret reduction, stop this direction.

Usage:
    python scripts/offline_validate_probe.py --config config/bench_medium.yaml --seeds 10
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import load_config
from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.utils.types import Action
from uav_isac.physical.detection import compute_target_utilities
from uav_isac.utils.math_utils import marginal_utility_gain as mug
from scipy.stats import spearmanr
import copy


def collect_frames(cfg, seeds=10, T=150):
    """Collect per-frame diagnostics across multiple episodes."""
    frames = []
    for seed in range(30001, 30001 + seeds):
        rng = np.random.default_rng(seed)
        env = EnvironmentCore(cfg, rng=rng)
        env.reset()
        for t in range(T):
            actions = {}
            for k in range(cfg.scenario.K):
                p = env.uavs[k].pos[:2]
                best_d, best_dir = 1e9, np.zeros(2)
                for q in range(cfg.scenario.Q):
                    b = env.belief_mgr.get_belief(k, q)
                    d = np.linalg.norm(b.mean[:2] - p)
                    if d < best_d:
                        best_d = d
                        best_dir = (b.mean[:2] - p) / max(d, 1e-6)
                actions[k] = Action(delta_p=best_dir * 1.25, role=2)

            # Snapshot pre-step belief state
            if env.belief_mgr is not None:
                mean_before = env.belief_mgr.mean.copy()
                cov_before = env.belief_mgr.cov.copy()
                aoi_before = env.belief_mgr.aoi.copy()
            else:
                mean_before = cov_before = aoi_before = None

            _, _, dones, info = env.step(actions)

            if info.P_D_q is None:
                continue

            pd = float(np.mean(info.P_D_q))

            # Compute ambiguity A_q and P0 regret for this frame
            K, Q = cfg.scenario.K, cfg.scenario.Q
            P_FA = cfg.detection.P_FA
            cap = cfg.p0_solver.capacity_per_rx
            B_q = cfg.detection.B_q
            K_q_max = cfg.detection.K_q_max

            valid = [e for e in info.deflection_entries if e.d_eff > 0]

            # Truth-P0 (counterfactual)
            truth_sel = set()
            D_truth = np.zeros(Q)
            rem_cap = {j: float(cap) for j in range(K)}
            tgt_cnt = {q: 0 for q in range(Q)}
            while True:
                best_g, best_e = -1.0, None
                for e in valid:
                    if (e.i, e.j, e.q) in truth_sel: continue
                    if B_q > rem_cap.get(e.j, 0): continue
                    if tgt_cnt.get(e.q, 0) >= K_q_max: continue
                    g = mug(D_truth[e.q], e.d_eff, P_FA)
                    if g > best_g: best_g = g; best_e = e
                if best_e is None or best_g <= 1e-12: break
                truth_sel.add((best_e.i, best_e.j, best_e.q))
                D_truth[best_e.q] += best_e.d_eff
                rem_cap[best_e.j] -= B_q; tgt_cnt[best_e.q] += 1

            # Belief-P0 selections
            belief_sel = set(info.p0_solution.selected_set)
            D_belief = info.p0_solution.D_q_star

            # P0 regret
            U_b = compute_target_utilities(D_belief, P_FA)
            U_t = compute_target_utilities(D_truth, P_FA)
            regret = float(np.sum(np.maximum(0, U_t - U_b))) / (float(np.sum(U_t)) + 1e-8)

            # Ambiguity per target
            amb_q = np.zeros(Q)
            for q in range(Q):
                qe = [(e, mug(D_truth[q], e.d_eff, P_FA), e.d_eff)
                      for e in valid if e.q == q]
                qe.sort(key=lambda x: -x[1])
                if len(qe) >= 2:
                    delta_mu = abs(qe[0][1] - qe[1][1])
                    cov_tr = float(np.mean([
                        np.trace(cov_before[k, q, :4, :4])
                        for k in range(K)
                    ])) if cov_before is not None else 100.0
                    sigma_pos = np.sqrt(max(cov_tr, 1e-8))
                    sigma_u = sigma_pos * (qe[0][1] / (qe[0][2] + 1e-6))
                    amb_q[q] = sigma_u / (delta_mu + 1e-8)

            # Most ambiguous target
            q_star = int(np.argmax(amb_q))

            # Was q_star already observed (by belief-P0)?
            belief_q_star = {(i, j) for (i, j, tgt) in belief_sel if tgt == q_star}
            was_observed = len(belief_q_star) > 0

            # Best available probe for q_star (max d_eff among valid entries)
            best_probe = None
            best_d = -1.0
            for e in valid:
                if e.q == q_star and e.d_eff > best_d:
                    best_d = e.d_eff
                    best_probe = e

            # Compute "potential regret reduction" = ambiguity × d_eff of best probe
            # This is our PREDICTOR of whether probing q_star would help
            potential_reduction = amb_q[q_star] * (best_d if best_probe else 0.0)

            frames.append({
                't': t, 'seed': seed,
                'P_D': pd,
                'regret': regret,
                'amb_max': float(np.max(amb_q)),
                'amb_mean': float(np.mean(amb_q)),
                'q_star': q_star,
                'was_observed': was_observed,
                'potential_reduction': potential_reduction,
                'n_selected': len(belief_sel),
            })

            if dones.get('__all__', False):
                break

    return frames


def main():
    parser = argparse.ArgumentParser(description="Offline Probe Validation")
    parser.add_argument('--config', default='config/bench_medium.yaml')
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.marl.rel_features = True
    cfg.marl.use_p0_sinr_gated = True
    cfg.marl.neighbor_belief_fusion = False
    cfg.marl.belief_nis_enabled = False
    cfg.marl.p0_uses_belief = True
    cfg.marl.belief_detection_sampling = True
    cfg.marl.p0_beta_uncertainty = 0.0
    cfg.marl.p0_eta_aoi = 0.0

    print(f"Collecting frames from {args.config} ({args.seeds} seeds)...")
    frames = collect_frames(cfg, seeds=args.seeds)
    print(f"  {len(frames)} frames collected")

    # ── Analysis ──
    # Split into high-ambiguity (Q4) and rest
    amb_values = [f['amb_max'] for f in frames]
    amb_q3 = np.percentile(amb_values, 75)

    high_amb = [f for f in frames if f['amb_max'] >= amb_q3]
    low_amb = [f for f in frames if f['amb_max'] < amb_q3]

    print(f"\n{'='*60}")
    print(f"Offline Probe Validation")
    print(f"{'='*60}")
    print(f"\n  High-ambiguity frames (Q4, amb >= {amb_q3:.1f}): {len(high_amb)}")
    print(f"  Low-ambiguity frames (Q1-Q3): {len(low_amb)}")

    # Key comparison: in high-amb frames where q_star WAS observed,
    # was regret LOWER in the next frame?
    # We need consecutive frames from the same episode.

    # Pair consecutive frames
    paired = []
    for i in range(len(frames) - 1):
        if frames[i]['seed'] == frames[i+1]['seed']:
            paired.append((frames[i], frames[i+1]))

    high_paired = [(f0, f1) for f0, f1 in paired if f0['amb_max'] >= amb_q3]

    if len(high_paired) < 10:
        print("\n  [INCONCLUSIVE] Not enough consecutive high-ambiguity frames")
        return

    # Split high-amb frames by whether q_star was observed
    obs_yes = [(f0, f1) for f0, f1 in high_paired if f0['was_observed']]
    obs_no = [(f0, f1) for f0, f1 in high_paired if not f0['was_observed']]

    # Regret CHANGE: regret(f0) - regret(f1). Positive = regret decreased (good).
    regret_change_yes = [f0['regret'] - f1['regret'] for f0, f1 in obs_yes]
    regret_change_no = [f0['regret'] - f1['regret'] for f0, f1 in obs_no]
    pd_change_yes = [f1['P_D'] - f0['P_D'] for f0, f1 in obs_yes]
    pd_change_no = [f1['P_D'] - f0['P_D'] for f0, f1 in obs_no]

    print(f"\n  q* observed in next frame:     {len(obs_yes)} pairs")
    if regret_change_yes:
        print(f"    mean regret change: {np.mean(regret_change_yes):+.4f}  (+ = regret decreased)")
        print(f"    mean P_D change:    {np.mean(pd_change_yes):+.4f}")

    print(f"\n  q* NOT observed in next frame: {len(obs_no)} pairs")
    if regret_change_no:
        print(f"    mean regret change: {np.mean(regret_change_no):+.4f}  (+ = regret decreased)")
        print(f"    mean P_D change:    {np.mean(pd_change_no):+.4f}")

    # ── Key test: does potential_reduction predict actual regret reduction? ──
    pot_red = np.array([f0['potential_reduction'] for f0, f1 in high_paired])
    act_red = np.array([f0['regret'] - f1['regret'] for f0, f1 in high_paired])

    r, p = spearmanr(pot_red, act_red)

    print(f"\n{'='*60}")
    print(f"Key Test: potential_reduction vs actual regret reduction")
    print(f"{'='*60}")
    print(f"  Spearman rho = {r:.4f}, p = {p:.4f}")
    print(f"  n = {len(high_paired)} consecutive high-ambiguity pairs")

    if r > 0.05 and p < 0.05:
        print(f"\n  [PASS] Predicted reduction correlates with actual reduction")
        print(f"  -> Decision-space information gain IS predictive.")
        print(f"  -> Exploit/Resolve discrete switching is justified.")
    else:
        print(f"\n  [FAIL] No significant correlation")
        print(f"  -> Predicted reduction does NOT predict actual regret change.")
        print(f"  -> STOP this direction. Do not implement Resolve-P0.")

    # ── Secondary: compare observed vs unobserved ──
    if regret_change_yes and regret_change_no:
        from scipy.stats import mannwhitneyu
        u, p_mw = mannwhitneyu(regret_change_yes, regret_change_no, alternative='greater')
        print(f"\n  Secondary: observed vs unobserved regret reduction")
        print(f"    Mann-Whitney U = {u:.0f}, p = {p_mw:.4f}")
        if p_mw < 0.05:
            print(f"    [PASS] Observing q* significantly increases regret reduction")
        else:
            print(f"    [FAIL] No significant difference")


if __name__ == '__main__':
    main()
