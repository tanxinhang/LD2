"""Tests for L3-G2: capability-sensitivity vs baselines."""

import numpy as np

from uav_isac.coordination.capability import (
    capability_gauge_pwl_lp_full,
    capability_geometry_gradient,
)
from uav_isac.coordination.pwl_pd import (
    chord_lower_bound,
    curvature_breakpoints,
)
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)

P_FA = 0.001
XI = (0.60, 0.70, 0.80, 3)


def _setup(rng, K=5, Q=4):
    uav = rng.uniform(0.0, 100.0, size=(K, 2))
    target = rng.uniform(0.0, 100.0, size=(Q, 2))
    owner = np.array([q % K for q in range(Q)])
    scale = 1e8

    def gain_of(p):
        rtx = np.linalg.norm(p[:, None, :] - target[None, :, :], axis=2)
        rrx = np.linalg.norm(p[owner] - target, axis=1)
        return scale / (rtx ** 2 * rrx[None, :] ** 2)

    gain = gain_of(uav)
    budget = np.full(K, 3.0)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([0.60]), P_FA)[0])
    d_max = float(np.max(np.sum(gain * budget[:, None], axis=0))) + 1.0
    bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=1e-3)
    cs, ci = chord_lower_bound(P_FA, bps)
    return uav, target, owner, gain_of, gain, budget, d_min, cs, ci


def _gamma(gain, budget, d_min, cs, ci):
    out = capability_gauge_pwl_lp_full(gain, budget, P_FA, XI, cs, ci, d_min)
    return None if out is None else out[0]


def _worst_target(gain, budget, d_min, p_fa):
    # Worst target under the equal-split deflection (a cheap proxy).
    d = np.sum(gain * (budget[:, None] / gain.shape[1]), axis=0)
    pd = compute_detection_probabilities(d, p_fa)
    return int(np.argmin(pd))


def _baseline_directions(uav, target, gain, budget, d_min, kind, power, prices):
    K, dim = uav.shape
    dirs = np.zeros_like(uav)
    if kind == "no_move":
        return dirs
    if kind == "toward_worst":
        q = _worst_target(gain, budget, d_min, P_FA)
        for k in range(K):
            v = target[q] - uav[k]
            n = np.linalg.norm(v)
            dirs[k] = v / max(n, 1e-9)
    elif kind == "friis_distance":
        # Move toward the nearest target (pure "get closer" heuristic).
        for k in range(K):
            q = int(np.argmin(np.linalg.norm(uav[k] - target, axis=1)))
            v = target[q] - uav[k]
            n = np.linalg.norm(v)
            dirs[k] = v / max(n, 1e-9)
    elif kind == "capability_dual":
        g = capability_geometry_gradient(
            gain, np.array([q % K for q in range(gain.shape[1])]),
            uav, target, power, prices)
        for k in range(K):
            n = np.linalg.norm(g[k])
            dirs[k] = -g[k] / max(n, 1e-9) if n > 1e-12 else 0.0
    return dirs


def _one_step_delta(gain_of, uav, target, gain, budget, d_min, cs, ci, kind,
                    power, prices, step=2.0):
    dirs = _baseline_directions(
        uav, target, gain, budget, d_min, kind, power, prices)
    new_uav = uav + step * dirs
    g0 = _gamma(gain, budget, d_min, cs, ci)
    g1 = _gamma(gain_of(new_uav), budget, d_min, cs, ci)
    if g0 is None or g1 is None:
        return None
    return g0 - g1  # positive = decrease


def _capability_dual_baseline_means():
    rng = np.random.default_rng(70)
    results = {k: [] for k in
               ("no_move", "toward_worst", "friis_distance", "capability_dual")}
    for _ in range(30):
        uav, target, owner, gain_of, gain, budget, d_min, cs, ci = _setup(rng)
        out = capability_gauge_pwl_lp_full(
            gain, budget, P_FA, XI, cs, ci, d_min)
        if out is None:
            continue
        _, power, prices = out
        for kind in results:
            d = _one_step_delta(gain_of, uav, target, gain, budget, d_min,
                                cs, ci, kind, power, prices)
            if d is not None:
                results[kind].append(d)

    return {k: float(np.mean(v)) for k, v in results.items() if v}


def test_capability_dual_beats_simple_baselines():
    mean = _capability_dual_baseline_means()
    # Capability-dual must be at least as good as every baseline on average.
    assert mean["capability_dual"] >= mean["toward_worst"] - 1e-9
    assert mean["capability_dual"] >= mean["friis_distance"] - 1e-9
    assert mean["capability_dual"] >= mean["no_move"] - 1e-9
    # And it must do strictly better than no movement.
    assert mean["capability_dual"] > 0.0


if __name__ == "__main__":
    means = _capability_dual_baseline_means()
    print(json := __import__("json").dumps(means, indent=2))
