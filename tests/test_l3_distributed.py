"""Tests for D0.94-L3D T2: distributed local gradient == centralized oracle."""

import numpy as np

from uav_isac.coordination.capability import (
    capability_gauge_pwl_lp_full,
    capability_geometry_gradient,
    local_capability_gradient_k,
)
from uav_isac.coordination.pwl_pd import (
    chord_lower_bound,
    curvature_breakpoints,
)
from uav_isac.physical.detection import (
    minimum_deflection_for_detection_probability,
)

P_FA = 0.001
XI = (0.60, 0.70, 0.80, 3)


def _setup(rng, K=5, Q=4):
    uav = rng.uniform(0.0, 100.0, size=(K, 2))
    target = rng.uniform(0.0, 100.0, size=(Q, 2))
    owner = np.array([q % K for q in range(Q)])
    scale = 1e8
    gain = scale / (
        np.linalg.norm(uav[:, None, :] - target[None, :, :], axis=2) ** 2
        * np.linalg.norm(uav[owner] - target, axis=1)[None, :] ** 2
    )
    budget = np.full(K, 3.0)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([0.60]), P_FA)[0])
    d_max = float(np.max(np.sum(gain * budget[:, None], axis=0))) + 1.0
    bps = curvature_breakpoints(P_FA, d_min, d_max, epsilon=1e-3)
    cs, ci = chord_lower_bound(P_FA, bps)
    return uav, target, owner, gain, budget, d_min, cs, ci


def test_local_gradient_matches_oracle():
    rng = np.random.default_rng(80)
    K, Q = 5, 4
    uav, target, owner, gain, budget, d_min, cs, ci = _setup(rng, K, Q)
    out = capability_gauge_pwl_lp_full(gain, budget, P_FA, XI, cs, ci, d_min)
    assert out is not None
    _, power, prices = out

    oracle = capability_geometry_gradient(gain, owner, uav, target, power, prices)

    # Build the delivered "support" records for each target (ideal communication).
    rtx = np.linalg.norm(uav[:, None, :] - target[None, :, :], axis=2)
    support = {q: {} for q in range(Q)}
    for q in range(Q):
        for i in range(K):
            support[q][i] = (power[i, q], gain[i, q])

    # Each UAV computes its local gradient from local info + delivered records.
    local = np.zeros_like(oracle)
    for k in range(K):
        own_gain = gain[k]  # UAV k's own fixed-owner gain row (local).
        own_power = power[k]
        local[k] = local_capability_gradient_k(
            k, owner, prices, own_power, own_gain, uav[k], target, support)

    err = np.linalg.norm(local - oracle) / (np.linalg.norm(oracle) + 1e-12)
    assert err < 1e-6, f"local vs oracle gradient error {err}"


def test_local_gradient_fail_closed_on_missing_support():
    """A missing transmitter record must not be reconstructed from truth."""
    rng = np.random.default_rng(81)
    K, Q = 5, 4
    uav, target, owner, gain, budget, d_min, cs, ci = _setup(rng, K, Q)
    out = capability_gauge_pwl_lp_full(gain, budget, P_FA, XI, cs, ci, d_min)
    assert out is not None
    _, power, prices = out

    # Drop one transmitter record for an owned target: the local gradient must
    # differ from the full-support gradient (fail-closed, no truth reconstruction).
    k = int(owner[0])
    support_full = {q: {i: (power[i, q], gain[i, q]) for i in range(K)}
                    for q in range(Q)}
    # Pick a transmitter of target 0 that actually carries non-negligible power.
    drop_i = max(range(K), key=lambda i: power[i, 0])
    support_partial = {q: dict(support_full[q]) for q in range(Q)}
    support_partial[0].pop(drop_i)  # a support transmitter not delivered

    g_full = local_capability_gradient_k(
        k, owner, prices, power[k], gain[k], uav[k], target, support_full)
    g_partial = local_capability_gradient_k(
        k, owner, prices, power[k], gain[k], uav[k], target, support_partial)
    assert not np.allclose(g_full, g_partial), (
        "missing support record must change the local gradient (fail-closed)")
