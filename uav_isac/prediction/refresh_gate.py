"""Fail-closed scheduling policy for predictive boundary refreshes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class RefreshPath(str, Enum):
    HOLD_SELECTED_ONLY = "hold_selected_only"
    PREDICTED_SPARSE_REFRESH = "predicted_sparse_refresh"
    FULL_EXACT_FALLBACK = "full_exact_fallback"


@dataclass(frozen=True)
class RefreshDecision:
    path: RefreshPath
    candidate_edges: tuple[tuple[int, int, int], ...]
    reason: str


class PredictiveRefreshGate:
    """Keep hold frames cheap and fail closed at protocol boundaries.

    The gate does not claim analytical validity. Callers may accept a sparse
    refresh only after selected-edge physics, omitted-edge bounds and solver
    residual checks all return true.
    """

    def __init__(self, hold_period: int = 5) -> None:
        if hold_period < 1:
            raise ValueError("hold_period must be positive")
        self.hold_period = int(hold_period)

    def is_refresh_boundary(self, source_frame: int) -> bool:
        return int(source_frame) % self.hold_period == 0

    def decide(
        self,
        source_frame: int,
        incumbent_edges: Iterable[tuple[int, int, int]],
        predicted_edges: Iterable[tuple[int, int, int]] = (),
        *,
        selected_physics_valid: bool = False,
        omitted_edge_bound_valid: bool = False,
        solver_residual_valid: bool = False,
    ) -> RefreshDecision:
        incumbent = tuple(sorted(set(tuple(edge) for edge in incumbent_edges)))
        if not self.is_refresh_boundary(source_frame):
            return RefreshDecision(
                RefreshPath.HOLD_SELECTED_ONLY,
                incumbent,
                "protocol hold fixes edge indices; recompute selected physics only",
            )
        candidates = tuple(sorted(
            set(incumbent) | set(tuple(edge) for edge in predicted_edges)))
        if (
            selected_physics_valid
            and omitted_edge_bound_valid
            and solver_residual_valid
        ):
            return RefreshDecision(
                RefreshPath.PREDICTED_SPARSE_REFRESH,
                candidates,
                "all analytical gates passed",
            )
        return RefreshDecision(
            RefreshPath.FULL_EXACT_FALLBACK,
            candidates,
            "one or more analytical gates failed or were not supplied",
        )
