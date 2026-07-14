#!/usr/bin/env python
"""B8 audit: how close is the P0 greedy to the exhaustive optimum?

The current detection utility U(D) = -log(1-P_D(D)) is NOT concave in D
(see docs/KNOWN_ISSUES.md B8), so the greedy P0 has no (1-1/e) submodular
guarantee. This script empirically measures the greedy optimality gap by
comparing inner_solver.solve (greedy) against inner_solver.solve_exhaustive
(brute force) on real per-frame deflection candidate sets.

CAVEAT 1 (scale): solve_exhaustive enumerates 2^n subsets and is only feasible
for n <= 20 candidate links. Role-agnostic K=4,Q=2 already yields 4*3*2=24
links, so this audit runs on a REDUCED scenario (K=3, Q=2 -> <=12 links). The
structural greedy-vs-optimal behavior of this utility does not depend strongly
on K,Q, so the result is indicative of the full scenario.

CAVEAT 2 (constraints): compute_greedy_gap compares greedy-vs-exhaustive under
capacity + cardinality constraints only; neither side enforces the env's
one-role-per-UAV rule. So this bounds the gap of the *base* selection problem.

Usage:
    python scripts/greedy_gap_audit.py [--frames 300] [--region 800]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import get_default_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--region", type=int, default=800)
    ap.add_argument("--seed", type=int, default=10001)
    args = ap.parse_args()

    cfg = get_default_config()
    cfg.scenario.K = 3          # keep candidate count <= 20 for exhaustive feasibility
    cfg.scenario.Q = 2
    cfg.scenario.region_size = (args.region, args.region)
    cfg.target.omega_q = [0.5, 0.5]
    K, Q = cfg.scenario.K, cfg.scenario.Q

    env = UAVISACEnv(config=cfg, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    solver = env.core.inner_solver

    gaps, optimal_flags, n_entries = [], [], []
    frames_with_candidates = 0
    for _ in range(args.frames):
        acts = {str(k): {'delta_p': rng.normal(0, 1.5, 2), 'role': 0} for k in range(K)}
        obs, _, term, trunc, info = env.step(acts)
        entries = env.current_step_info.deflection_entries
        valid = [e for e in entries if e.d_eff > 0]
        if len(valid) == 0:
            if term.get('__all__') or trunc.get('__all__'):
                obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            continue
        if len(valid) > 20:
            # Should not happen at K=3,Q=2, but guard anyway.
            continue
        res = solver.compute_greedy_gap(entries, Q=Q, K=K)
        gaps.append(res['relative_gap'])
        optimal_flags.append(res['greedy_is_optimal'])
        n_entries.append(len(valid))
        frames_with_candidates += 1
        if term.get('__all__') or trunc.get('__all__'):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

    gaps = np.array(gaps)
    print("=" * 60)
    print(f"P0 greedy optimality audit  (K={K}, Q={Q}, region={args.region})")
    print(f"frames with >=1 candidate: {frames_with_candidates}")
    if len(gaps) == 0:
        print("no candidate frames; try larger --region or more --frames")
        return
    print(f"mean candidate links/frame: {np.mean(n_entries):.1f} (max {max(n_entries)})")
    print(f"greedy == exhaustive optimum: {100*np.mean(optimal_flags):.1f}% of frames")
    print(f"relative gap (0=optimal): mean={gaps.mean():.4f}  "
          f"p95={np.percentile(gaps,95):.4f}  max={gaps.max():.4f}")
    print(f"=> greedy achieves on avg {100*(1-gaps.mean()):.2f}% of the optimal weighted utility")
    print("=" * 60)
    print("NOTE: base problem (capacity+cardinality), single-role rule not in exhaustive;")
    print("exhaustive limited to <=20 links so scenario reduced to K=3,Q=2 (see file header).")


if __name__ == "__main__":
    main()
