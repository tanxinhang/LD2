"""Certified structural screens for the conservative ISAC repair model.

The first S0 layer intentionally implements only context-free statements.  A
zero coefficient is interpreted exactly, without a numerical threshold.  Such
an edge cannot improve any target Deflection and consumes only upper-bounded
resources, so fixing it to zero preserves feasibility.  It preserves the
minimum-intervention lexicographic optimum context-free only when the edge was
not selected in the reference structure; otherwise its removal itself creates
a repair toggle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Edge = tuple[int, int, int]


@dataclass(frozen=True)
class ContextFreeScreen:
    feasibility_preserving: tuple[Edge, ...]
    optimality_preserving: tuple[Edge, ...]
    selected_zero_gain: tuple[Edge, ...]
    candidate_edges: int


def context_free_zero_gain_screen(
    gain_per_watt: np.ndarray,
    current_selected: np.ndarray,
) -> ContextFreeScreen:
    """Return exact-zero D-F and D-O screens under separate UAV resources.

    No tolerance is accepted: a small positive coefficient remains capable of
    contributing when multiplied by an independent UAV power budget.  The
    theorem therefore applies only to coefficients whose model upper bound is
    exactly zero.
    """
    gain = np.asarray(gain_per_watt, dtype=np.float64)
    selected = np.asarray(current_selected, dtype=bool)
    if gain.ndim != 3 or gain.shape[0] != gain.shape[1]:
        raise ValueError("gain_per_watt must have shape (K,K,Q)")
    if selected.shape != gain.shape:
        raise ValueError("current_selected shape must match gain_per_watt")
    if np.any(~np.isfinite(gain)) or np.any(gain < 0.0):
        raise ValueError("gain_per_watt must be finite and nonnegative")
    K, _, Q = gain.shape
    edges = tuple(
        (i, j, q) for i in range(K) for j in range(K) if i != j
        for q in range(Q)
    )
    feasibility = tuple(edge for edge in edges if gain[edge] == 0.0)
    selected_zero = tuple(edge for edge in feasibility if selected[edge])
    optimality = tuple(edge for edge in feasibility if not selected[edge])
    return ContextFreeScreen(
        feasibility_preserving=feasibility,
        optimality_preserving=optimality,
        selected_zero_gain=selected_zero,
        candidate_edges=len(edges),
    )
