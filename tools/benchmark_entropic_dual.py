"""Baseline bench for the entropy-regularized max-min dual price.

Validates the central-path price against the LP (basis-dependent) and the
minimum-L2 canonical price on the three claims used to justify it:

  1. UNIQUENESS:  target permutation must leave entropic/canonical prices
     invariant, while the raw LP vertex may jump.
  2. ERROR BOUND: 0 <= f(lambda*_tau) - f(lambda*) <= tau * log Q, monotone in tau.
  3. GEOMETRY STABILITY: a small gain perturbation between adjacent frames
     must move the entropic price by less than the raw LP vertex.
"""

from __future__ import annotations

import time

import numpy as np

from uav_isac.coordination.maxmin_power import (
    entropic_maxmin_dual_prices,
    optimal_maxmin_dual_prices,
    canonical_maxmin_dual_prices,
)


def _make_gain(rng: np.random.Generator, K: int, Q: int) -> np.ndarray:
    gain = rng.lognormal(mean=0.0, sigma=1.2, size=(K, Q))
    gain *= rng.uniform(0.2, 1.0, size=(K, 1))
    scale = np.max(gain)
    return gain / scale if scale > 0.0 else gain


def _dual_objective(gain: np.ndarray, budget: np.ndarray, lam: np.ndarray) -> float:
    return float(np.sum(budget * np.max(lam[None, :] * gain, axis=1)))


def _permute(gain: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return gain[:, perm]


def bench_uniqueness() -> None:
    print("=" * 72)
    print("[1] UNIQUENESS under target permutation")
    print("=" * 72)
    rng = np.random.default_rng(0)
    K, Q = 8, 8
    gain = _make_gain(rng, K, Q)
    budget = np.ones(K)
    perms = [np.arange(Q), rng.permutation(Q), rng.permutation(Q)]
    values = {
        "LP vertex (basis-dep)": [],
        "canonical (min L2)": [],
        "entropic tau=1e-2": [],
    }
    for perm in perms:
        g = _permute(gain, perm)
        lam_opt, _ = optimal_maxmin_dual_prices(g, budget)
        lam_can, _ = canonical_maxmin_dual_prices(g, budget)
        lam_ent, _ = entropic_maxmin_dual_prices(g, budget, 1e-2)
        values["LP vertex (basis-dep)"].append(np.asarray(lam_opt)[::-1] if False else lam_opt)
        values["canonical (min L2)"].append(lam_can)
        values["entropic tau=1e-2"].append(lam_ent)
    for name, seq in values.items():
        # un-permute via inverse of the identity (identity mapping assumed on
        # the returned order: the LP/canonical/entropic each map target q to
        # position q of the caller's own ordering; permutation reorders columns
        # so price vector is permuted identically).
        ref = seq[0]
        spread = max(float(np.max(np.abs(seq[i] - ref))) for i in range(1, len(seq)))
        print(f"  {name:24s} max ||lam_i - lam_0||_inf = {spread:.3e}")


def bench_error_bound() -> None:
    print("=" * 72)
    print("[2] ERROR BOUND  0 <= f(lam_tau) - f(lam*) <= tau log Q")
    print("=" * 72)
    rng = np.random.default_rng(1)
    K, Q = 8, 8
    gain = _make_gain(rng, K, Q)
    budget = np.ones(K)
    _, f_star = optimal_maxmin_dual_prices(gain, budget)
    print(f"  exact dual value f(lam*) = {f_star:.6f}")
    print(f"  tau log Q = {np.log(Q):.4f} * tau")
    print(f"  {'tau':>10s} {'f(lam_tau)':>12s} {'gap':>12s} {'gap/(tau log Q)':>16s}")
    for tau in [1e0, 1e-1, 1e-2, 1e-3, 1e-4]:
        lam, f_tau = entropic_maxmin_dual_prices(gain, budget, tau)
        gap = f_tau - f_star
        ratio = gap / (tau * np.log(Q))
        print(f"  {tau:10.0e} {f_tau:12.6f} {gap:12.3e} {ratio:16.3f}")
    lam_1, f_1 = entropic_maxmin_dual_prices(gain, budget, 1e-1)
    lam_4, f_4 = entropic_maxmin_dual_prices(gain, budget, 1e-4)
    print(f"  ||lam(1e-1) - lam(1e-4)||_inf = "
          f"{float(np.max(np.abs(lam_1 - lam_4))):.3e}")


def bench_geometry_stability() -> None:
    print("=" * 72)
    print("[3] GEOMETRY STABILITY under a small inter-frame gain perturbation")
    print("=" * 72)
    rng = np.random.default_rng(2)
    K, Q = 8, 8
    gain = _make_gain(rng, K, Q)
    budget = np.ones(K)
    eps_list = [1e-3, 1e-2, 1e-1]
    print(f"  {'perturb eps':>12s} {'LP jump':>12s} {'canonical jump':>16s} "
          f"{'entropic jump':>16s}")
    for eps in eps_list:
        delta = rng.lognormal(mean=0.0, sigma=eps, size=(K, Q))
        gain2 = gain + eps * delta
        lam_opt_a, _ = optimal_maxmin_dual_prices(gain, budget)
        lam_opt_b, _ = optimal_maxmin_dual_prices(gain2, budget)
        lam_can_a, _ = canonical_maxmin_dual_prices(gain, budget)
        lam_can_b, _ = canonical_maxmin_dual_prices(gain2, budget)
        lam_ent_a, _ = entropic_maxmin_dual_prices(gain, budget, 1e-2)
        lam_ent_b, _ = entropic_maxmin_dual_prices(gain2, budget, 1e-2)
        j_opt = float(np.max(np.abs(lam_opt_a - lam_opt_b)))
        j_can = float(np.max(np.abs(lam_can_a - lam_can_b)))
        j_ent = float(np.max(np.abs(lam_ent_a - lam_ent_b)))
        print(f"  {eps:12.0e} {j_opt:12.3e} {j_can:16.3e} {j_ent:16.3e}")


def bench_runtime() -> None:
    print("=" * 72)
    print("[4] WALL TIME (single dual solve)")
    print("=" * 72)
    rng = np.random.default_rng(3)
    K, Q = 16, 16
    gain = _make_gain(rng, K, Q)
    budget = np.ones(K)
    for name, fn in [
        ("LP vertex", lambda: optimal_maxmin_dual_prices(gain, budget)),
        ("canonical min-L2", lambda: canonical_maxmin_dual_prices(gain, budget)),
        ("entropic tau=1e-2", lambda: entropic_maxmin_dual_prices(gain, budget, 1e-2)),
    ]:
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        print(f"  {name:20s} {dt * 1e3:8.2f} ms")


if __name__ == "__main__":
    bench_uniqueness()
    bench_error_bound()
    bench_geometry_stability()
    bench_runtime()