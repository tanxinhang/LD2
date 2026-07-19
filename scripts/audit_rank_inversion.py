#!/usr/bin/env python
"""P0 Rank Inversion Causal Audit.

Measures whether belief errors cause P0 rank inversion, and whether
rank inversion causes P_D degradation. No new algorithms — pure measurement.

Three causal relationships tested:
  H1: A_q (ambiguity) ↑ → rank inversion rate ↑
  H2: rank inversion ↑ → P0 regret ↑
  H3: P0 regret ↑ → P_D ↓

If all three hold, DU-P0 (Decision-Uncertainty-aware P0) is justified.
If any fails, stop algorithm development on this direction.

Usage:
    python scripts/audit_rank_inversion.py --config config/bench_medium.yaml --seeds 20
    python scripts/audit_rank_inversion.py --config config/bench_hard_clean.yaml --seeds 20 --output results/audit_hard.json
"""

import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import load_config
from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.physical.detection import compute_detection_probabilities
from uav_isac.utils.types import Action
import copy


def compute_utility_sigma_points(belief_mean, belief_cov, deflection_entries,
                                  target_q, P_FA, num_sigma=7):
    """Compute utility-space uncertainty σ_c for each candidate of target q.

    Uses sigma points: x^(ℓ) ~ N(mean, cov), compute U_c(x^(ℓ)),
    then σ²_c = Var[U_c].

    Returns:
        mu_c: dict (i,j) -> mean utility
        sigma_c: dict (i,j) -> utility std
    """
    d = 4  # [x,y,vx,vy]
    mean_4d = belief_mean[:4]
    cov_4d = belief_cov[:4, :4]

    # Generate sigma points via Cholesky
    try:
        L = np.linalg.cholesky(cov_4d)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_4d + 1e-6 * np.eye(4))

    # Symmetric sigma points: mean ± sqrt(d) * L_i
    points = [mean_4d.copy()]
    weights = [0.0]  # mean weight handled separately

    sqrt_d = np.sqrt(d)
    for i in range(d):
        delta = sqrt_d * L[:, i]
        points.append(mean_4d + delta)
        points.append(mean_4d - delta)
        weights.append(1.0 / (2 * d))
        weights.append(1.0 / (2 * d))

    # Compute utility for each sigma point
    mu_c = {}
    sigma_c = {}

    for e in deflection_entries:
        if e.q != target_q or e.d_eff <= 0:
            continue
        key = (e.i, e.j)
        utils = []
        for pt in points:
            # Recompute d_eff using sigma point position
            # Simplified: use ratio of distances
            # Full recomputation would need full deflection recompute
            # For audit, approximate: utility ∝ d_eff which ∝ 1/distance^4
            utils.append(e.d_eff)  # placeholder — in full version, recompute geometry
        if utils:
            mu_c[key] = np.mean(utils)
            sigma_c[key] = np.std(utils)

    return mu_c, sigma_c


def audit_episode(env, T=150, steady_window=20):
    """Run one episode and collect per-frame P0 ranking diagnostics.

    For each frame, we compute BOTH belief-P0 ranking and truth-P0 ranking,
    then measure rank inversion, regret, and ambiguity.
    """
    obs, _ = env.reset()

    frames = []

    for t in range(T):
        # Homing policy
        actions = {}
        K = env.K
        for k in range(K):
            p = env.uavs[k].pos[:2]
            best_d, best_dir = 1e9, np.zeros(2)
            for q in range(env.Q):
                b = env.belief_mgr.get_belief(k, q)
                d = np.linalg.norm(b.mean[:2] - p)
                if d < best_d:
                    best_d = d
                    best_dir = (b.mean[:2] - p) / max(d, 1e-6)
            actions[k] = Action(delta_p=best_dir * 1.25, role=2)

        _, _, dones, info = env.step(actions)

        # ── Collect per-frame diagnostics ──
        frame_data = {'t': t}

        if info.P_D_q is not None:
            frame_data['P_D_mean'] = float(np.mean(info.P_D_q))
        else:
            frame_data['P_D_mean'] = 0.0

        # P0 solution from belief
        belief_selected = set(info.p0_solution.selected_set)
        frame_data['n_selected'] = len(belief_selected)

        # Belief-P0 per-target deflection
        D_belief = info.p0_solution.D_q_star

        # Truth-P0: what would P0 select with true target positions?
        # We approximate by using the true deflection entries
        # (which use true geometry) and re-running P0 greedily
        true_entries = info.deflection_entries
        # Filter to valid entries
        valid = [e for e in true_entries if e.d_eff > 0]

        # Greedy truth-P0: same algorithm as InnerSolver but with true geometry
        truth_selected = set()
        D_truth = np.zeros(env.Q, dtype=np.float64)
        remaining_cap = {j: float(env.cfg.p0_solver.capacity_per_rx) for j in range(K)}
        target_counts = {q: 0 for q in range(env.Q)}
        P_FA = env.cfg.detection.P_FA

        while True:
            best_gain = -1.0
            best_e = None
            for e in valid:
                key = (e.i, e.j, e.q)
                if key in truth_selected:
                    continue
                if env.cfg.detection.B_q > remaining_cap.get(e.j, 0):
                    continue
                if target_counts.get(e.q, 0) >= env.cfg.detection.K_q_max:
                    continue

                from uav_isac.utils.math_utils import marginal_utility_gain
                gain = marginal_utility_gain(D_truth[e.q], e.d_eff, P_FA)
                if gain > best_gain:
                    best_gain = gain
                    best_e = e

            if best_e is None or best_gain <= 1e-12:
                break

            truth_selected.add((best_e.i, best_e.j, best_e.q))
            D_truth[best_e.q] += best_e.d_eff
            remaining_cap[best_e.j] -= env.cfg.detection.B_q
            target_counts[best_e.q] += 1

        # ── Rank inversion metrics ──
        # Per-target: does belief-P0 select the same candidates as truth-P0?
        n_inversions = 0
        n_pairs = 0
        for q in range(env.Q):
            belief_q = {(i, j) for (i, j, tgt) in belief_selected if tgt == q}
            truth_q = {(i, j) for (i, j, tgt) in truth_selected if tgt == q}
            n_pairs += max(len(belief_q), len(truth_q), 1)

            # Rank inversion: pairs selected by truth but NOT by belief
            missed = truth_q - belief_q
            n_inversions += len(missed)

        frame_data['rank_inversion_rate'] = n_inversions / max(n_pairs, 1)

        # P0 regret: utility difference truth - belief
        from uav_isac.physical.detection import compute_target_utilities
        U_belief = compute_target_utilities(D_belief, P_FA)
        U_truth = compute_target_utilities(D_truth, P_FA)
        regret = float(np.sum(U_truth - U_belief))
        frame_data['P0_regret'] = max(0.0, regret)

        # Per-target ambiguity A_q for top-2 candidates
        ambiguities = []
        for q in range(env.Q):
            # Get all valid entries for this target, sorted by d_eff
            q_entries = [(e, e.d_eff) for e in valid if e.q == q]
            q_entries.sort(key=lambda x: -x[1])
            if len(q_entries) >= 2:
                # Top-2 d_eff values
                d1 = q_entries[0][1]
                d2 = q_entries[1][1]
                delta = abs(d1 - d2)
                # σ from belief covariance trace for this target
                cov_trace = float(np.mean([
                    np.trace(env.belief_mgr.cov[k, q])
                    for k in range(K)
                ]))
                sigma = np.sqrt(max(cov_trace, 1e-8))
                # Ambiguity: σ_Δ / (|Δμ| + ε)
                amb = sigma / (delta + 1e-6)
                ambiguities.append(amb)
        frame_data['ambiguity_mean'] = float(np.mean(ambiguities)) if ambiguities else 0.0
        frame_data['ambiguity_max'] = float(np.max(ambiguities)) if ambiguities else 0.0

        # Cov trace
        frame_data['cov_trace_mean'] = float(np.mean([
            np.trace(env.belief_mgr.cov[k, q])
            for k in range(K) for q in range(env.Q)
        ]))

        frames.append(frame_data)

        if dones.get('__all__', False):
            break

    return frames


def main():
    parser = argparse.ArgumentParser(description="P0 Rank Inversion Causal Audit")
    parser.add_argument('--config', default='config/bench_medium.yaml')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--output', default=None)
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

    T = cfg.scenario.T
    steady_window = 20

    all_frames = []
    print(f"{'='*60}")
    print(f"P0 Rank Inversion Audit: {args.config}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}")

    for seed_idx in range(args.seeds):
        seed = 30001 + seed_idx
        rng = np.random.default_rng(seed)
        env = EnvironmentCore(cfg, rng=rng)
        frames = audit_episode(env, T=T, steady_window=steady_window)
        all_frames.extend(frames)
        if (seed_idx + 1) % 5 == 0:
            print(f"  seed {seed_idx+1}/{args.seeds} ({len(frames)} frames)")

    # ── Causal Analysis ──
    # Split frames into steady window (last 20) vs transient
    n_frames = len(all_frames)
    steady_start = max(0, n_frames - args.seeds * steady_window)

    # Bin frames by ambiguity quartiles
    amb_values = [f['ambiguity_mean'] for f in all_frames]
    amb_thresholds = np.percentile(amb_values, [25, 50, 75])

    print(f"\n{'='*60}")
    print(f"Causal Analysis ({len(all_frames)} frames total)")
    print(f"{'='*60}")

    for label, mask_fn in [
        ("Low ambiguity (Q1)", lambda a: a <= amb_thresholds[0]),
        ("Q2", lambda a: amb_thresholds[0] < a <= amb_thresholds[1]),
        ("Q3", lambda a: amb_thresholds[1] < a <= amb_thresholds[2]),
        ("High ambiguity (Q4)", lambda a: a > amb_thresholds[2]),
    ]:
        subset = [f for f in all_frames if mask_fn(f['ambiguity_mean'])]
        if len(subset) < 10:
            continue
        inv_rate = np.mean([f['rank_inversion_rate'] for f in subset])
        regret = np.mean([f['P0_regret'] for f in subset])
        pd_mean = np.mean([f['P_D_mean'] for f in subset])
        amb_mean = np.mean([f['ambiguity_mean'] for f in subset])
        print(f"\n  {label} (n={len(subset)}, amb={amb_mean:.2f}):")
        print(f"    rank_inversion_rate = {inv_rate:.4f}")
        print(f"    P0_regret           = {regret:.4f}")
        print(f"    P_D_mean            = {pd_mean:.4f}")

    # ── Correlation checks ──
    amb_arr = np.array(amb_values)
    inv_arr = np.array([f['rank_inversion_rate'] for f in all_frames])
    regret_arr = np.array([f['P0_regret'] for f in all_frames])
    pd_arr = np.array([f['P_D_mean'] for f in all_frames])

    # Spearman rank correlation
    from scipy.stats import spearmanr
    r_amb_inv, p_amb_inv = spearmanr(amb_arr, inv_arr)
    r_inv_reg, p_inv_reg = spearmanr(inv_arr, regret_arr)
    r_reg_pd, p_reg_pd = spearmanr(regret_arr, pd_arr)

    print(f"\n{'='*60}")
    print(f"Hypothesis Tests (Spearman ρ)")
    print(f"{'='*60}")
    print(f"\n  H1: A_q ↑ → rank inversion ↑")
    print(f"      ρ = {r_amb_inv:.4f}, p = {p_amb_inv:.6f}  {'[PASS]' if r_amb_inv > 0.05 and p_amb_inv < 0.05 else '[FAIL]'}")

    print(f"\n  H2: rank inversion ↑ → P0 regret ↑")
    print(f"      ρ = {r_inv_reg:.4f}, p = {p_inv_reg:.6f}  {'[PASS]' if r_inv_reg > 0.05 and p_inv_reg < 0.05 else '[FAIL]'}")

    print(f"\n  H3: P0 regret ↑ → P_D ↓")
    print(f"      ρ = {r_reg_pd:.4f}, p = {p_reg_pd:.6f}  {'[PASS]' if r_reg_pd < -0.05 and p_reg_pd < 0.05 else '[FAIL]'}")

    all_pass = (
        r_amb_inv > 0.05 and p_amb_inv < 0.05 and
        r_inv_reg > 0.05 and p_inv_reg < 0.05 and
        r_reg_pd < -0.05 and p_reg_pd < 0.05
    )

    print(f"\n  {'[ALL PASS] → DU-P0 justified' if all_pass else '[FAIL] → stop algorithm development on this direction'}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        result = {
            'config': args.config,
            'seeds': args.seeds,
            'n_frames': n_frames,
            'H1': {'rho': float(r_amb_inv), 'p': float(p_amb_inv)},
            'H2': {'rho': float(r_inv_reg), 'p': float(p_inv_reg)},
            'H3': {'rho': float(r_reg_pd), 'p': float(p_reg_pd)},
            'all_pass': all_pass,
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
