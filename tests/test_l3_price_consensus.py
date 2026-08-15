"""Tests for D0.94-L3D T0: peer-to-peer price consensus == centralized dual."""

import numpy as np

from uav_isac.coordination.maxmin_power import optimal_maxmin_dual_prices


def _metropolis_weights(adjacency):
    """Doubly-stochastic Metropolis-Hastings consensus matrix on a graph."""
    K = adjacency.shape[0]
    deg = adjacency.sum(axis=1)
    W = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            if i != j and adjacency[i, j]:
                W[i, j] = 1.0 / (max(deg[i], deg[j]) + 1)
    for i in range(K):
        W[i, i] = 1.0 - np.sum(W[i, :])
    return W


def _project_simplex(v):
    """Euclidean projection onto the probability simplex (O(Q log Q))."""
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, v.size + 1)
    cond = u - cssv / ind > 0
    rho = int(np.where(cond)[0][-1])
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def test_peer_to_peer_price_consensus_matches_centralized():
    rng = np.random.default_rng(90)
    K, Q = 6, 5
    gain = rng.uniform(0.1, 3.0, size=(K, Q))
    budget = rng.uniform(0.3, 1.0, size=K)

    # Centralized optimal dual price lambda*.
    lam_star, value = optimal_maxmin_dual_prices(gain, budget)

    # Ring graph: purely local peer-to-peer connectivity.
    adjacency = np.zeros((K, K), dtype=bool)
    for i in range(K):
        adjacency[i, (i + 1) % K] = True
        adjacency[(i + 1) % K, i] = True
    W = _metropolis_weights(adjacency)

    # Distributed dual subgradient: each UAV holds a local simplex price.
    prices = np.full((K, Q), 1.0 / Q, dtype=np.float64)
    rounds = 20000
    alpha0 = 0.1
    for r in range(1, rounds + 1):
        alpha = alpha0 / np.sqrt(r)
        # Consensus step (each UAV averages its neighbours).
        consensus = W @ prices
        # Local subgradient: b_i e_{argmax_q lambda_q a_iq}.
        for i in range(K):
            q = int(np.argmax(consensus[i] * gain[i]))
            g = np.zeros(Q)
            g[q] = budget[i]
            prices[i] = _project_simplex(consensus[i] - alpha * g)

    lam_avg = np.mean(prices, axis=0)
    lam_avg /= np.sum(lam_avg)
    err = np.linalg.norm(lam_avg - lam_star)
    assert err < 0.05, (
        f"consensus price diverged from centralized dual: err {err}, "
        f"avg {lam_avg}, star {lam_star}")
    # Dual value at the consensus price must approximate the centralized value.
    dual_value = float(np.sum(budget * np.max(
        lam_avg[None, :] * gain, axis=1)))
    assert abs(dual_value - value) < 0.05 * max(1.0, value)


def test_consensus_matrix_doubly_stochastic():
    rng = np.random.default_rng(91)
    K = 8
    adjacency = np.zeros((K, K), dtype=bool)
    for i in range(K):
        adjacency[i, (i + 1) % K] = True
        adjacency[(i + 1) % K, i] = True
    W = _metropolis_weights(adjacency)
    # Row-stochastic (1 W = 1) and column-stochastic (1^T W = 1^T).
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.allclose(W.sum(axis=0), 1.0)
    # Non-negative, self-loops non-negative.
    assert np.all(W >= 0.0)
