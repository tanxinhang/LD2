"""Search for gains where the min-L2 canonical price and the max-entropy
price diverge on the same optimal dual face.  A large divergence is the
replacement case for the entropy-regularized (central path) price.
"""

from __future__ import annotations

import numpy as np

from uav_isac.coordination.maxmin_power import (
    entropic_maxmin_dual_prices,
    optimal_maxmin_dual_prices,
    canonical_maxmin_dual_prices,
)


def dual_obj(gain: np.ndarray, budget: np.ndarray, lam: np.ndarray) -> float:
    return float(np.sum(budget * np.max(lam[None, :] * gain, axis=1)))


def main() -> None:
    rng = np.random.default_rng(2026)
    best = (0.0, None)
    for trial in range(2000):
        K = int(rng.integers(3, 7))
        Q = int(rng.integers(3, 7))
        g = rng.lognormal(0.0, 1.5, size=(K, Q))
        # inject dedicated/sparse structure to create multi-bottleneck faces
        g[rng.random((K, Q)) < rng.uniform(0.0, 0.5)] = 0.0
        budget = rng.uniform(0.5, 1.5, size=K)
        try:
            lam_can, f_can = canonical_maxmin_dual_prices(g, budget)
            lam_ent, f_ent = entropic_maxmin_dual_prices(g, budget, 1e-4)
        except Exception:
            continue
        if not np.isfinite(f_can) or not np.isfinite(f_ent):
            continue
        spread = float(np.max(np.abs(lam_can - lam_ent)))
        if spread > best[0]:
            best = (spread, (g, budget, lam_can, lam_ent, f_can, f_ent))
        if trial < 5:
            print(f"trial {trial}: spread={spread:.4f}")

    spread, data = best
    print("\n=== best divergence ===")
    g, budget, lam_can, lam_ent, f_can, f_ent = data
    print(f"max |canonical - entropic| = {spread:.4f}")
    print(f"f*_can = {f_can:.6f}  f*_ent = {f_ent:.6f}")
    print(f"gain =\n{np.round(g, 3)}")
    print(f"budget = {np.round(budget, 3)}")
    print(f"canonical = {np.round(lam_can, 4)}")
    print(f"entropic  = {np.round(lam_ent, 4)}")
    # which components does min-L2 zero out but entropy keep positive?
    zero_can = np.where(lam_can < 1e-6)[0]
    zero_ent = np.where(lam_ent < 1e-6)[0]
    print(f"zero components: canonical {zero_can.tolist()}  entropic {zero_ent.tolist()}")


if __name__ == "__main__":
    main()