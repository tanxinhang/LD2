#!/usr/bin/env python
"""Audit P0 greedy solver optimality: exhaustive vs greedy comparison.

Uses the built-in solve_exhaustive() method in InnerSolver.
Measures how much utility the greedy heuristic leaves on the table.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.params import load_config
from uav_isac.environment.env_core import EnvironmentCore
from uav_isac.physical.inner_solver import InnerSolver
from uav_isac.physical.detection import compute_target_utilities
from uav_isac.utils.math_utils import utility_from_D


def utility_from_solution(sol, K, Q, P_FA, omega_q):
    """Extract weighted utility from a P0Solution."""
    D_q = np.zeros(Q, dtype=np.float64)
    for (i, j, q) in sol.selected_set:
        # d_eff was already accumulated; use sol.D_q_star
        pass
    if sol.D_q_star is not None and len(sol.D_q_star) > 0:
        D_q = sol.D_q_star
    U_q = utility_from_D(np.array(D_q, dtype=np.float64), P_FA)
    return float(np.dot(omega_q, U_q))


def main():
    cfg = load_config("config/exp_800_q4.yaml")
    K, Q = cfg.scenario.K, cfg.scenario.Q

    env = EnvironmentCore(cfg)
    dc = env.deflection_computer
    omega_q = env.reward_computer.omega_q
    P_FA = cfg.detection.P_FA

    inner_solver = InnerSolver(
        K_q_max=cfg.detection.K_q_max,
        B_q=cfg.detection.B_q,
        capacity_per_rx=cfg.p0_solver.capacity_per_rx,
        latency_max=cfg.p0_solver.latency_max,
        omega_q=omega_q,
        P_FA=P_FA,
        P_D_min=cfg.detection.P_D_min,
    )

    n_tests = 100
    gaps_pct = []
    greedy_better = 0
    exhaust_better = 0
    ties = 0
    n_exhaust_too_large = 0

    print(f"P0 Optimality Audit: K={K}, Q={Q}, {n_tests} random geometries")
    print("=" * 60)

    np.random.seed(12345)
    region = cfg.scenario.region_size

    for test_idx in range(n_tests):
        # Random geometry
        uav_pos = np.random.uniform(0, region[0], size=(K, 3))
        uav_pos[:, 2] = cfg.scenario.height
        uav_vel = np.zeros((K, 3))
        tgt_pos = np.random.uniform(0, region[0], size=(Q, 3))
        tgt_pos[:, 2] = 0
        tgt_vel = np.zeros((Q, 3))
        roles = np.zeros(K, dtype=np.int32)
        fc_pos = np.array([region[0]/2, region[1]/2, cfg.scenario.height])

        # Compute deflection entries (role-agnostic: any UAV can tx/rx)
        entries = dc.compute(uav_pos, uav_vel, tgt_pos, tgt_vel,
                            roles, fc_pos, role_agnostic=True)
        valid = [e for e in entries if e.d_eff > 0]

        if len(valid) == 0:
            continue

        # Limit to top-18 entries by d_eff for exhaustive search feasibility
        valid_sorted = sorted(valid, key=lambda e: -e.d_eff)
        valid_for_search = valid_sorted[:18]
        n_valid = len(valid_for_search)

        # Greedy
        greedy_sol = inner_solver.solve(valid, Q=Q, K=K)
        greedy_util = utility_from_solution(greedy_sol, K, Q, P_FA, omega_q)

        # Exhaustive (limited to top-N entries for 2^N feasibility)
        if n_valid <= 18:
            try:
                exhaust_sol = inner_solver.solve_exhaustive(valid_for_search, Q=Q, K=K)
                exhaust_util = utility_from_solution(exhaust_sol, K, Q, P_FA, omega_q)

                if exhaust_util > 0:
                    gap = (exhaust_util - greedy_util) / max(exhaust_util, 1e-10) * 100
                    gaps_pct.append(gap)

                    if greedy_util > exhaust_util + 1e-6:
                        greedy_better += 1
                    elif exhaust_util > greedy_util + 1e-6:
                        exhaust_better += 1
                    else:
                        ties += 1
            except Exception:
                n_exhaust_too_large += 1
        else:
            n_exhaust_too_large += 1

        if (test_idx + 1) % 20 == 0:
            mean_gap = np.mean(gaps_pct) if gaps_pct else 0
            max_gap = np.max(gaps_pct) if gaps_pct else 0
            print(f"  [{test_idx+1}/{n_tests}] mean_gap={mean_gap:.1f}% "
                  f"max_gap={max_gap:.1f}% "
                  f"(ex>gr:{exhaust_better} gr>ex:{greedy_better} tie:{ties} "
                  f"skip:{n_exhaust_too_large})")

    print("\n" + "=" * 60)
    n_valid = len(gaps_pct)
    if n_valid == 0:
        print("No valid tests. All geometries had too many entries (>20).")
        return

    print(f"RESULTS ({n_valid} comparable tests, {n_exhaust_too_large} too-large)")
    print(f"  Mean gap:        {np.mean(gaps_pct):.2f}%")
    print(f"  Median gap:      {np.median(gaps_pct):.2f}%")
    print(f"  Max gap:         {np.max(gaps_pct):.2f}%")
    print(f"  Exhaust better:  {exhaust_better}/{n_valid} "
          f"({100*exhaust_better/n_valid:.0f}%)")
    print(f"  Greedy better:   {greedy_better}/{n_valid} "
          f"({100*greedy_better/n_valid:.0f}%)")
    print(f"  Ties:            {ties}/{n_valid} ({100*ties/n_valid:.0f}%)")

    mean_gap = np.mean(gaps_pct)
    if mean_gap > 10:
        print("\n🔴 CRITICAL: Greedy solver loses >10% utility on average!")
        print("   P0 is the performance ceiling. Fix before improving RL.")
    elif mean_gap > 3:
        print(f"\n🟡 MODERATE: Greedy loses {mean_gap:.1f}% utility on average.")
        print("   Worth improving, but not the main bottleneck.")
    elif mean_gap > 0.5:
        print(f"\n🟢 ACCEPTABLE: Greedy loses {mean_gap:.1f}% — near-optimal.")
    else:
        print(f"\n✅ OPTIMAL: Greedy is essentially optimal ({mean_gap:.2f}% gap).")
        print("   P0 solver is NOT the bottleneck.")


if __name__ == "__main__":
    main()
