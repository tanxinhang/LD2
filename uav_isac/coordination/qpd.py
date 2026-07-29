"""Queue-driven distributed primal-dual coordination primitives.

The functions in this module deliberately have no environment or policy
dependencies.  Each UAV can therefore run the same update from its local
state plus the (possibly delayed) protocol packets that it actually received.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QPDLocalUpdate:
    """Result of one UAV's locally executable primal-dual update."""

    primal: np.ndarray
    target_price: np.ndarray
    kkt_residual: float
    estimated_target_load: np.ndarray


def project_capped_simplex(
    values: np.ndarray,
    capacity: float,
) -> np.ndarray:
    """Project onto ``{x: 0 <= x <= 1, sum(x) <= capacity}``.

    The inequality (rather than equality) lets the mechanism leave resource
    unused when every marginal bid is below its price.
    """
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.all(np.isfinite(raw)):
        raise ValueError("values must be finite")
    capacity = float(capacity)
    if not np.isfinite(capacity) or capacity < 0.0:
        raise ValueError("capacity must be finite and non-negative")

    clipped = np.clip(raw, 0.0, 1.0)
    if float(np.sum(clipped)) <= capacity + 1e-12:
        return clipped
    if capacity <= 0.0:
        return np.zeros_like(clipped)

    # Euclidean capped-simplex projection: x = clip(v - tau, 0, 1).
    lower = float(np.min(raw) - 1.0)
    upper = float(np.max(raw))
    for _ in range(80):
        tau = 0.5 * (lower + upper)
        projected = np.clip(raw - tau, 0.0, 1.0)
        if float(np.sum(projected)) > capacity:
            lower = tau
        else:
            upper = tau
    return np.clip(raw - upper, 0.0, 1.0)


def update_virtual_queue(
    queue: np.ndarray,
    local_detection: np.ndarray,
    qos_floor: float,
    step_size: float,
    queue_max: float,
) -> np.ndarray:
    """Update a QoS deficit queue.

    The plus sign is essential: detection below the floor raises the price,
    while a safe target drains it.
    """
    z = np.asarray(queue, dtype=np.float64)
    pd = np.asarray(local_detection, dtype=np.float64)
    if z.shape != pd.shape:
        raise ValueError("queue and local_detection must have the same shape")
    updated = z + float(step_size) * (float(qos_floor) - pd)
    return np.clip(updated, 0.0, float(queue_max))


def local_primal_dual_update(
    own_bid: np.ndarray,
    peer_primal: np.ndarray,
    previous_primal: np.ndarray,
    previous_target_price: np.ndarray,
    *,
    row_capacity: float,
    target_capacity: float,
    primal_step: float,
    dual_step: float,
    rounds: int,
    price_max: float,
    exploration_floor: float = 0.0,
) -> QPDLocalUpdate:
    """Run locally unrolled primal-dual iterations for one UAV.

    ``peer_primal`` is the sum of peer commitments visible in this UAV's
    physical inbox.  Missing or expired packets contribute zero; no global
    matrix is consulted.
    """
    bid = np.asarray(own_bid, dtype=np.float64)
    peer = np.asarray(peer_primal, dtype=np.float64)
    primal = np.asarray(previous_primal, dtype=np.float64).copy()
    price = np.asarray(previous_target_price, dtype=np.float64).copy()
    if not (bid.shape == peer.shape == primal.shape == price.shape):
        raise ValueError("all primal-dual vectors must have the same shape")
    if bid.ndim != 1:
        raise ValueError("primal-dual vectors must be one-dimensional")

    n_rounds = max(1, int(rounds))
    for _ in range(n_rounds):
        load = peer + primal
        price = np.clip(
            price + float(dual_step) * (load - float(target_capacity)),
            -float(price_max),
            float(price_max),
        )
        exponent = np.clip(
            float(primal_step) * (bid - price), -20.0, 20.0)
        candidate = (
            np.maximum(primal, 0.0)
            + max(1e-6, float(exploration_floor))
        ) * np.exp(exponent)
        primal = project_capped_simplex(candidate, row_capacity)

    load = peer + primal
    stationarity = np.abs(
        primal - project_capped_simplex(
            (primal + max(1e-6, float(exploration_floor))) * np.exp(np.clip(
                float(primal_step) * (bid - price), -20.0, 20.0)),
            row_capacity,
        )
    )
    overload = np.maximum(load - float(target_capacity), 0.0)
    residual = float(max(
        np.max(stationarity, initial=0.0),
        np.max(overload, initial=0.0),
    ))
    return QPDLocalUpdate(
        primal=primal,
        target_price=price,
        kkt_residual=residual,
        estimated_target_load=load,
    )
