"""Fixed-shape labels for a frozen centralized structure teacher.

The centralized max-min P0 selects directed ``(tx, rx, target)`` hyperedges.
Distillation must preserve transmitter and receiver ownership instead of
collapsing the decision to one target class per UAV. Privileged candidate
physics is materialized separately from deployable student observations.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from uav_isac.utils.types import DeflectionEntry


def structure_teacher_labels(
    selected_set: Sequence[Tuple[int, int, int]],
    num_uavs: int,
    num_targets: int,
) -> dict[str, np.ndarray]:
    """Convert directed teacher hyperedges to endpoint and role labels."""
    K = int(num_uavs)
    Q = int(num_targets)
    if K < 1 or Q < 1:
        raise ValueError("num_uavs and num_targets must be positive")

    pair = np.zeros((K, K, Q), dtype=np.uint8)
    tx_target = np.zeros((K, Q), dtype=np.uint8)
    rx_target = np.zeros((K, Q), dtype=np.uint8)
    receiver_owner = np.full(Q, -1, dtype=np.int16)

    for i_raw, j_raw, q_raw in selected_set:
        i, j, q = int(i_raw), int(j_raw), int(q_raw)
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q):
            raise ValueError("selected_set contains an out-of-range index")
        if i == j:
            raise ValueError("a bistatic edge cannot use the same UAV twice")
        pair[i, j, q] = 1
        tx_target[i, q] = 1
        rx_target[j, q] = 1
        previous = int(receiver_owner[q])
        if previous not in {-1, j}:
            receiver_owner[q] = -2
        else:
            receiver_owner[q] = j

    endpoint_target = np.maximum(tx_target, rx_target)
    role = np.full(K, -1, dtype=np.int8)
    has_tx = tx_target.any(axis=1)
    has_rx = rx_target.any(axis=1)
    role[has_tx & ~has_rx] = 0
    role[has_rx & ~has_tx] = 1
    role[has_tx & has_rx] = 2
    return {
        "pair": pair,
        "tx_target": tx_target,
        "rx_target": rx_target,
        "endpoint_target": endpoint_target,
        "receiver_owner": receiver_owner,
        "role": role,
    }


def deflection_entry_tensors(
    entries: Iterable[DeflectionEntry],
    num_uavs: int,
    num_targets: int,
) -> dict[str, np.ndarray]:
    """Materialize the privileged full candidate graph for audit only."""
    K = int(num_uavs)
    Q = int(num_targets)
    shape = (K, K, Q)
    tensors = {
        "candidate": np.zeros(shape, dtype=np.uint8),
        # Preserve solver precision so a trace replay is an exact instrument
        # check rather than a float32 near-tie perturbation.
        "d_eff": np.zeros(shape, dtype=np.float64),
        "d_raw": np.zeros(shape, dtype=np.float64),
        "alpha": np.zeros(shape, dtype=np.float64),
        "g_dd": np.zeros(shape, dtype=np.float64),
        "chi_rep": np.zeros(shape, dtype=np.float64),
    }
    for entry in entries:
        i, j, q = int(entry.i), int(entry.j), int(entry.q)
        if not (0 <= i < K and 0 <= j < K and 0 <= q < Q):
            raise ValueError("deflection entries contain an out-of-range index")
        tensors["candidate"][i, j, q] = 1
        tensors["d_eff"][i, j, q] = float(entry.d_eff)
        tensors["d_raw"][i, j, q] = float(entry.d_raw)
        tensors["alpha"][i, j, q] = float(entry.alpha)
        tensors["g_dd"][i, j, q] = float(entry.g_dd)
        tensors["chi_rep"][i, j, q] = float(entry.chi_rep)
    return tensors
