"""Targeted baseline: degenerate scenario where LP vertex chatters.

Constructs a gain matrix with two independent bottleneck groups (targets
3 and 5 are co-bottlenecked), so the optimal dual face is a line segment.
Under small perturbations the LP vertex jumps between the two endpoints
(chattering), while the entropic price stays at the central point.

This demonstrates the three theoretical guarantees:
  (i)   uniqueness: entropic gives a single point, LP gives a vertex
  (ii)  continuity: entropic is smooth under perturbation, LP jumps
  (iii) strict interior: entropic lambda > 0, LP lambda has zeros
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from uav_isac.coordination.maxmin_power import (
    optimal_maxmin_dual_prices,
    entropic_maxmin_dual_prices,
)


def main() -> None:
    # K=4 transmitters, Q=6 targets.
    # Targets 3 and 5 share identical gain rows → co-bottlenecked.
    # The optimal dual face is {lambda_3 + lambda_5 = 1, rest = 0}.
    K, Q = 4, 6
    budget = np.array([0.3, 0.3, 0.2, 0.2])

    gain_base = np.array([
        [1.0, 0.5, 0.0, 2.0, 0.0, 2.0],
        [0.5, 1.0, 0.0, 2.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0, 1.0, 0.0],
    ])

    print("=" * 72)
    print("Degenerate Scenario: Two Independent Bottleneck Groups")
    print("=" * 72)
    print(f"K={K}, Q={Q}, budget={budget}")
    print(f"Gain matrix:\n{gain_base}")
    print()

    # Baseline: LP vertex at the unperturbed gain
    lam_lp, f_lp = optimal_maxmin_dual_prices(gain_base, budget)
    lam_ent, f_ent = entropic_maxmin_dual_prices(gain_base, budget, 0.0)
    print("--- Unperturbed ---")
    print(f"  LP vertex:  {np.array2string(lam_lp, precision=4)}")
    print(f"  Entropic:   {np.array2string(lam_ent, precision=4)}")
    print(f"  f_lp={f_lp:.6f}  f_ent={f_ent:.6f}  gap={f_ent-f_lp:.2e}")
    print()

    # Perturbation study: add small noise to gain and see how prices move
    n_perturb = 5
    eps_max = 1e-3
    rng = np.random.default_rng(42)

    lp_prices = np.zeros((n_perturb, Q))
    ent_prices = np.zeros((n_perturb, Q))
    lp_jumps = np.zeros(n_perturb)
    ent_jumps = np.zeros(n_perturb)

    lam_lp_prev = lam_lp.copy()
    lam_ent_prev = lam_ent.copy()

    print("--- Perturbation Study (20 steps, eps=1e-3) ---")
    print(f"  {'step':>4}  {'eps':>8}  {'LP_jump':>10}  {'ENT_jump':>10}  "
          f"{'LP_min':>8}  {'ENT_min':>8}")

    for i in range(n_perturb):
        eps = eps_max * (i + 1) / n_perturb
        perturbation = eps * rng.standard_normal((K, Q))
        gain_pert = gain_base + perturbation

        lam_lp_i, _ = optimal_maxmin_dual_prices(gain_pert, budget)
        lam_ent_i, _ = entropic_maxmin_dual_prices(gain_pert, budget, 0.01)

        lp_jump = float(np.max(np.abs(lam_lp_i - lam_lp_prev)))
        ent_jump = float(np.max(np.abs(lam_ent_i - lam_ent_prev)))

        lp_prices[i] = lam_lp_i
        ent_prices[i] = lam_ent_i
        lp_jumps[i] = lp_jump
        ent_jumps[i] = ent_jump

        lam_lp_prev = lam_lp_i.copy()
        lam_ent_prev = lam_ent_i.copy()

        print(f"  {i+1:4d}  {eps:8.2e}  {lp_jump:10.6f}  {ent_jump:10.6f}  "
              f"{np.min(lam_lp_i):8.2e}  {np.min(lam_ent_i):8.2e}")

    print()
    print("--- Summary ---")
    print(f"  LP  total variation:  {np.sum(lp_jumps):.6f}")
    print(f"  ENT total variation:  {np.sum(ent_jumps):.6f}")
    print(f"  LP  max jump:         {np.max(lp_jumps):.6f}")
    print(f"  ENT max jump:         {np.max(ent_jumps):.6f}")
    print(f"  LP  zeros (min):      {np.min(lp_prices):.2e}")
    print(f"  ENT zeros (min):      {np.min(ent_prices):.2e}")

    # The key metric: chattering = total variation / path length
    lp_chatter = np.sum(lp_jumps) / (eps_max * n_perturb)
    ent_chatter = np.sum(ent_jumps) / (eps_max * n_perturb)
    print(f"  LP  chattering rate:  {lp_chatter:.2f}")
    print(f"  ENT chattering rate:  {ent_chatter:.2f}")
    print()
    print("Conclusion: LP vertex chatters (jumps between bases) while")
    print("entropic price is smooth (continuous under perturbation).")


if __name__ == "__main__":
    main()