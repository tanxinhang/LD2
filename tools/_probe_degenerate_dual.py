"""Probe: when is the max-min dual lambda* genuinely non-unique?

The dual is  min_{lambda in simplex} sum_i b_i max_q lambda_q a_iq.  Its optimum
is a singleton face unless the gain support has a "dedicated target" structure.
"""

from __future__ import annotations

import numpy as np

from uav_isac.coordination.maxmin_power import (
    entropic_maxmin_dual_prices,
    optimal_maxmin_dual_prices,
    canonical_maxmin_dual_prices,
)


def run(name: str, gain: np.ndarray, budget: np.ndarray) -> None:
    lam_opt, f = optimal_maxmin_dual_prices(gain, budget)
    lam_can, _ = canonical_maxmin_dual_prices(gain, budget)
    lam_ent, _ = entropic_maxmin_dual_prices(gain, budget, 1e-3)
    print(f"\n[{name}]  f*={f:.6f}")
    print(f"  LP vertex : {np.round(lam_opt, 4)}")
    print(f"  canonical : {np.round(lam_can, 4)}")
    print(f"  entropic  : {np.round(lam_ent, 4)}")
    print(f"  LP-vs-can ||.||1={np.sum(np.abs(lam_opt-lam_can)):.3e}  "
          f"can-vs-ent={np.sum(np.abs(lam_can-lam_ent)):.3e}")


if __name__ == "__main__":
    # (a) dedicated targets: each target reachable by exactly one UAV
    gain = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    run("dedicated (block-diagonal)", gain, np.ones(3))

    # (b) sparse random: many UAV-target pairs unreachable (DD gate / distance)
    rng = np.random.default_rng(7)
    K, Q = 8, 8
    g = rng.lognormal(0.0, 1.0, size=(K, Q))
    g[rng.random((K, Q)) < 0.6] = 0.0  # 60% unreachable
    run("sparse random (60% zero)", g, np.ones(K))

    # (c) two independent clusters of dedicated targets
    g2 = np.zeros((4, 4))
    g2[0, 0] = g2[1, 0] = 1.0
    g2[0, 1] = g2[1, 1] = 1.0
    g2[2, 2] = g2[3, 2] = 1.0
    g2[2, 3] = g2[3, 3] = 1.0
    run("two clusters (thresholded)", g2, np.ones(4))