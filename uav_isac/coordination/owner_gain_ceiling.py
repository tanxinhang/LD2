"""Owner-gain ceiling primitive shared by all owner-election mechanisms (P2-2).

Blueprint mapping (CODE_STRUCTURE_MAP_2026-08-25, P2-2): the shared primitive
``owner_gain_ceiling(receiver, target, gain, budget, pair_limit)`` is
realised here as a row-level function — the receiver/target indices are the
calling context (each caller iterates (receiver, target) pairs and passes the
matching ``(K,)`` gain row).  This keeps the exact arithmetic of the
historical inline owner-bid envelope: scale the row by the budget vector,
sort, take the top-``pair_limit`` entries, sum — bit-identical to the code
it replaces, so behaviour does not move by a single ULP.

Semantics:
    ceiling = sum( sort(gain_row * budget_vec)[-pair_limit:] )
with ``budget_vec`` either a scalar broadcast to (K,) or an explicit (K,)
per-transmitter budget vector (the historical envelope multiplied the
per-transmitter gain row by the per-transmitter budget vector).  Returns 0.0
when ``pair_limit < 1`` or the row is empty.  Rejects non-finite/negative
gains and non-finite/negative budget entries.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def owner_gain_ceiling(
    gain_row: np.ndarray,
    *,
    budget_w: Union[float, int, np.floating, np.ndarray],
    pair_limit: Union[int, np.integer],
) -> float:
    """Return the top-``pair_limit`` budget-weighted gain ceiling of the row.

    Args:
        gain_row: 1-D ``(K,)`` per-transmitter gain for one (receiver,
            target) pair; finite and non-negative.
        budget_w: receiver sensing budget — either a scalar (same budget for
            every transmitter) or a ``(K,)`` vector (per-transmitter budget,
            matching the historical envelope semantics); finite and >= 0.
        pair_limit: maximum admitted transmitter pairs per target
            (senders-per-target capacity), >= 1.

    Returns:
        ``sum of the top-``pair_limit`` values of (gain_row * budget_vec)``
        as float; ``0.0`` when ``pair_limit < 1`` or ``gain_row`` is empty.
    """
    row = np.asarray(gain_row, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError("owner_gain_ceiling gain_row must be 1-D (K,)")
    if np.any(~np.isfinite(row)) or np.any(row < 0.0):
        raise ValueError("owner_gain_ceiling gain_row must be finite and >= 0")
    budget_arr = np.asarray(budget_w, dtype=np.float64)
    if budget_arr.ndim == 0:
        budget_vec = np.full(row.shape, float(budget_arr), dtype=np.float64)
    elif budget_arr.ndim == 1:
        if budget_arr.shape != row.shape:
            raise ValueError(
                "owner_gain_ceiling budget_w vector must match gain_row (K,)")
        budget_vec = budget_arr
    else:
        raise ValueError(
            "owner_gain_ceiling budget_w must be scalar or a (K,) vector")
    if np.any(~np.isfinite(budget_vec)) or np.any(budget_vec < 0.0):
        raise ValueError(
            "owner_gain_ceiling budget_w must be finite and >= 0")
    limit = int(pair_limit)
    if limit < 1 or row.size < 1:
        return 0.0
    scaled = row * budget_vec
    k = min(limit, row.size)
    return float(np.sum(np.sort(scaled)[-k:]))