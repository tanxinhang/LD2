"""Stateful Gate C1.7 coordinator for dynamic local structure maintenance.

The coordinator consumes only the frozen local Student edge graph and the
public candidate mask.  It deliberately has no access to true geometry,
realized deflection, target state, or the centralized belief state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from uav_isac.coordination.learned_move_ranker import FrozenLocalMoveRanker
from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    assert_local_feasible,
    oracle_best_improvement,
    ranked_first_improvement,
    rebuild_structure,
    role_first_initial_structure,
    role_owner_from_structure,
)
from uav_isac.coordination.local_move_ranker import local_move_features
from uav_isac.coordination.factor_graph_coordinator import (
    FiniteRoundFactorGraphCoordinator,
    decode_lightweight_feasible,
    factor_graph_edge_features,
)
from uav_isac.physical.feasibility_oracle import (
    solve_maxmin_single_role_pairs,
)
from uav_isac.utils.checkpoint_loading import safe_torch_load
from uav_isac.utils.types import DeflectionEntry
import torch


@dataclass(frozen=True)
class DynamicLocalSearchResult:
    selected: np.ndarray
    diagnostics: dict[str, float | int | str]


class DynamicLocalSearchCoordinator:
    """Cold deterministic search plus stateful warm local maintenance."""

    MODES = {"previous", "oracle", "hybrid", "replicated"}

    def __init__(
        self,
        mode: str,
        *,
        ranker_checkpoint: str | Path | None = None,
        cold_initializer: str = "role_first",
        factor_graph_checkpoint: str | Path | None = None,
        cold_rounds: int = 8,
        warm_rounds: int = 8,
        warm_top_m: int = 3,
        rebootstrap_mode: str = "off",
        rebootstrap_interval_frames: int = 0,
        rebootstrap_rounds: int = 8,
        rebootstrap_require_deficit: bool = False,
    ) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in self.MODES:
            raise ValueError(
                f"dynamic local-search mode must be one of {self.MODES}")
        if normalized == "hybrid" and ranker_checkpoint is None:
            raise ValueError("hybrid local search requires a ranker checkpoint")
        cold_initializer = str(cold_initializer).strip().lower()
        if cold_initializer not in {"role_first", "factor_graph"}:
            raise ValueError("cold initializer must be role_first or factor_graph")
        if cold_initializer == "factor_graph" and factor_graph_checkpoint is None:
            raise ValueError(
                "factor_graph cold initialization requires a checkpoint")
        self.mode = normalized
        self.cold_rounds = max(0, int(cold_rounds))
        self.warm_rounds = max(0, int(warm_rounds))
        self.warm_top_m = max(1, int(warm_top_m))
        rebootstrap_mode = str(rebootstrap_mode).strip().lower()
        if rebootstrap_mode not in {"off", "periodic", "oracle"}:
            raise ValueError(
                "rebootstrap mode must be off, periodic, or oracle")
        if (rebootstrap_mode == "periodic"
                and int(rebootstrap_interval_frames) <= 0):
            raise ValueError(
                "periodic rebootstrap requires a positive frame interval")
        self.rebootstrap_mode = rebootstrap_mode
        self.rebootstrap_interval_frames = max(
            0, int(rebootstrap_interval_frames))
        self.rebootstrap_rounds = max(0, int(rebootstrap_rounds))
        self.rebootstrap_require_deficit = bool(
            rebootstrap_require_deficit)
        self.cold_initializer = cold_initializer
        self.ranker = (
            FrozenLocalMoveRanker(ranker_checkpoint)
            if normalized == "hybrid" else None
        )
        self.factor_graph = self._load_factor_graph(
            factor_graph_checkpoint)
        self.reset()

    @staticmethod
    def _load_factor_graph(checkpoint: str | Path | None):
        if checkpoint is None:
            return None
        payload = safe_torch_load(
            Path(checkpoint),
            map_location="cpu",
            description="factor-graph coordinator checkpoint",
            required_keys=("hidden_dim", "rounds"),
            state_dict_keys=("state_dict",),
        )
        model = FiniteRoundFactorGraphCoordinator(
            edge_feature_dim=int(payload.get("edge_feature_dim", 4)),
            hidden_dim=int(payload["hidden_dim"]),
            rounds=int(payload["rounds"]),
            coupling_strength=float(payload.get("coupling_strength", 1.0)),
            coupling_rounds=int(payload.get("coupling_rounds", 0)),
            use_global_context=bool(payload.get("use_global_context", False)),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    def _factor_graph_initial(
        self,
        value: np.ndarray,
        candidate: np.ndarray,
        *,
        target_pair_limit: int,
        reports_per_receiver: int,
    ) -> np.ndarray:
        if self.factor_graph is None:
            raise RuntimeError("factor-graph initializer is unavailable")
        with torch.inference_mode():
            value_tensor = torch.as_tensor(
                value[None], dtype=torch.float32)
            mask_tensor = torch.as_tensor(
                candidate[None], dtype=torch.bool)
            output = self.factor_graph(
                factor_graph_edge_features(value_tensor, mask_tensor),
                mask_tensor,
            )
        decoded = decode_lightweight_feasible(
            output.edge_logits[0].cpu().numpy(),
            output.owner_logits[0].cpu().numpy(),
            output.role_logits[0].cpu().numpy(),
            candidate,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
            edge_value=value,
        )
        return decoded.selected

    def reset(self) -> None:
        self._selected: np.ndarray | None = None
        self._role: np.ndarray | None = None
        self._last_edge_value: np.ndarray | None = None
        self._last_candidate_mask: np.ndarray | None = None
        self._forced_selected_once: np.ndarray | None = None
        self._forced_role_once: np.ndarray | None = None
        self._resolve_index = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "selected": (
                None if self._selected is None else self._selected.copy()),
            "role": None if self._role is None else self._role.copy(),
            "last_edge_value": (
                None if self._last_edge_value is None
                else self._last_edge_value.copy()),
            "last_candidate_mask": (
                None if self._last_candidate_mask is None
                else self._last_candidate_mask.copy()),
            "forced_selected_once": (
                None if self._forced_selected_once is None
                else self._forced_selected_once.copy()),
            "forced_role_once": (
                None if self._forced_role_once is None
                else self._forced_role_once.copy()),
            "resolve_index": int(self._resolve_index),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        selected = state.get("selected")
        role = state.get("role")
        last_edge_value = state.get("last_edge_value")
        last_candidate_mask = state.get("last_candidate_mask")
        forced_selected_once = state.get("forced_selected_once")
        forced_role_once = state.get("forced_role_once")
        self._selected = (
            None if selected is None
            else np.asarray(selected, dtype=bool).copy())
        self._role = (
            None if role is None
            else np.asarray(role, dtype=np.int8).copy())
        self._last_edge_value = (
            None if last_edge_value is None
            else np.asarray(last_edge_value, dtype=np.float64).copy())
        self._last_candidate_mask = (
            None if last_candidate_mask is None
            else np.asarray(last_candidate_mask, dtype=bool).copy())
        self._forced_selected_once = (
            None if forced_selected_once is None
            else np.asarray(forced_selected_once, dtype=bool).copy())
        self._forced_role_once = (
            None if forced_role_once is None
            else np.asarray(forced_role_once, dtype=np.int8).copy())
        self._resolve_index = int(state.get("resolve_index", 0))

    def last_problem(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the exact public value/mask pair used at the last resolve."""
        if self._last_edge_value is None or self._last_candidate_mask is None:
            return None
        return (
            self._last_edge_value.copy(),
            self._last_candidate_mask.copy(),
        )

    def force_next_move_for_audit(self, move: LocalMove) -> None:
        """Install one atomic move for a same-state counterfactual replay.

        This hook is intentionally explicit and one-shot.  Feasibility against
        the branch's freshly reconstructed public candidate graph is asserted
        inside ``resolve``; deployment code never calls this method.
        """
        self._forced_selected_once = np.asarray(
            move.selected, dtype=bool).copy()
        self._forced_role_once = np.asarray(move.role, dtype=np.int8).copy()

    def _score_moves(self, selected, role, owner, moves, edge_value):
        if self.ranker is None:
            raise RuntimeError("learned move scorer is unavailable")
        return self.ranker.predict(np.stack([
            local_move_features(selected, role, owner, move, edge_value)
            for move in moves
        ]))

    def resolve(
        self,
        edge_value: np.ndarray,
        candidate_mask: np.ndarray,
        *,
        target_pair_limit: int,
        reports_per_receiver: int,
        p_fa: float = 1.0e-3,
        p_d_floor: float = 0.60,
        target_priority: np.ndarray | None = None,
        target_deficit: np.ndarray | None = None,
        frame_index: int | None = None,
    ) -> DynamicLocalSearchResult:
        value = np.asarray(edge_value, dtype=np.float64)
        candidate = np.asarray(candidate_mask, dtype=bool)
        if value.shape != candidate.shape or value.ndim != 3:
            raise ValueError("edge value/mask must have shape (K,K,Q)")
        K = value.shape[0]
        candidate = candidate & (value > 0.0)
        candidate[np.arange(K), np.arange(K), :] = False
        self._last_edge_value = value.copy()
        self._last_candidate_mask = candidate.copy()
        cold = self._selected is None or self._role is None
        if self._forced_selected_once is not None:
            selected = self._forced_selected_once.copy()
            forced_role = self._forced_role_once
            self._forced_selected_once = None
            self._forced_role_once = None
            assert_local_feasible(
                selected,
                candidate,
                target_pair_limit=target_pair_limit,
                reports_per_receiver=reports_per_receiver,
            )
            role, _ = role_owner_from_structure(
                selected,
                fallback_role=(
                    forced_role
                    if forced_role is not None
                    else self._role),
            )
            self._selected = selected.copy()
            self._role = role.copy()
            self._resolve_index += 1
            return DynamicLocalSearchResult(selected, {
                "local_search_mode": self.mode,
                "local_search_cold_start": float(cold),
                "local_search_candidate_count": 0,
                "local_search_exact_verifications": 0,
                "local_search_accepted_moves": 1,
                "local_search_selected_edges": int(np.sum(selected)),
                "local_search_resolve_index": int(self._resolve_index),
                "local_search_audit_forced_move": 1.0,
            })
        if not np.any(candidate):
            selected = np.zeros_like(candidate)
            self._selected = selected
            self._role = None
            self._resolve_index += 1
            return DynamicLocalSearchResult(selected, {
                "local_search_mode": self.mode,
                "local_search_cold_start": float(cold),
                "local_search_candidate_count": 0,
                "local_search_exact_verifications": 0,
                "local_search_accepted_moves": 0,
            })

        if self.mode == "replicated":
            entries = [
                DeflectionEntry(
                    i=int(i), j=int(j), q=int(q),
                    tau=0.0, nu=0.0, alpha=0.0,
                    d_raw=float(value[i, j, q]),
                    g_dd=1.0, chi_rep=1.0,
                    d_eff=float(value[i, j, q]),
                )
                for i, j, q in np.argwhere(candidate)
            ]
            selected_edges, _ = solve_maxmin_single_role_pairs(
                entries,
                num_uavs=value.shape[0],
                num_targets=value.shape[2],
                target_pair_limit=target_pair_limit,
                reports_per_receiver=reports_per_receiver,
                p_fa=float(p_fa),
                p_d_floor=float(p_d_floor),
                target_priority=(
                    np.ones(value.shape[2], dtype=np.float64)
                    if target_priority is None
                    else np.asarray(target_priority, dtype=np.float64)),
                fusion_mode="local_only",
            )
            selected = np.zeros_like(candidate)
            for edge in selected_edges:
                selected[edge] = True
            self._selected = selected.copy()
            self._role, _ = role_owner_from_structure(selected)
            self._resolve_index += 1
            return DynamicLocalSearchResult(selected, {
                "local_search_mode": self.mode,
                "local_search_cold_start": float(cold),
                "local_search_candidate_count": int(np.sum(candidate)),
                "local_search_exact_verifications": int(np.sum(candidate)),
                "local_search_accepted_moves": 0,
                "local_search_selected_edges": int(np.sum(selected)),
                "local_search_resolve_index": int(self._resolve_index),
            })

        role_first, fallback_role, _ = role_first_initial_structure(
            value,
            candidate,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
        )
        rebootstrap_attempted = False
        rebootstrap_accepted = False
        rebootstrap_candidates = 0
        rebootstrap_accepted_moves = 0
        if cold:
            # N5 crosses the coupled role/owner/support barrier; N1--N3 then
            # finish inexpensive local corrections in the same deterministic
            # cold-start transaction.
            cold_initial = (
                self._factor_graph_initial(
                    value,
                    candidate,
                    target_pair_limit=target_pair_limit,
                    reports_per_receiver=reports_per_receiver,
                )
                if self.cold_initializer == "factor_graph"
                else role_first
            )
            result = oracle_best_improvement(
                cold_initial,
                value,
                candidate,
                rounds=self.cold_rounds,
                neighborhoods=("N1", "N2", "N3", "N5"),
                target_pair_limit=target_pair_limit,
                reports_per_receiver=reports_per_receiver,
                initial_role=fallback_role,
            )
            exact = int(sum(result.candidate_counts))
        else:
            _, previous_owner = role_owner_from_structure(
                self._selected, fallback_role=self._role)
            initial = rebuild_structure(
                value,
                candidate,
                self._role,
                previous_owner,
                target_pair_limit=target_pair_limit,
                reports_per_receiver=reports_per_receiver,
            )
            if self.mode == "previous":
                result = oracle_best_improvement(
                    initial,
                    value,
                    candidate,
                    rounds=0,
                    neighborhoods=(),
                    target_pair_limit=target_pair_limit,
                    reports_per_receiver=reports_per_receiver,
                    initial_role=self._role,
                )
                exact = 0
            elif self.mode == "oracle":
                result = oracle_best_improvement(
                    initial,
                    value,
                    candidate,
                    rounds=self.warm_rounds,
                    neighborhoods=("N1", "N2", "N3"),
                    target_pair_limit=target_pair_limit,
                    reports_per_receiver=reports_per_receiver,
                    initial_role=self._role,
                )
                exact = int(sum(result.candidate_counts))
            else:
                result = ranked_first_improvement(
                    initial,
                    value,
                    candidate,
                    rounds=self.warm_rounds,
                    neighborhoods=("N1", "N2", "N3"),
                    target_pair_limit=target_pair_limit,
                    reports_per_receiver=reports_per_receiver,
                    ranking_method="learned",
                    top_m=self.warm_top_m,
                    initial_role=self._role,
                    ranking_scorer=self._score_moves,
                    collect_full_diagnostics=False,
                )
                exact = int(sum(result.verification_counts))

            maintenance_candidates = int(sum(result.candidate_counts))
            maintenance_accepted_moves = int(len(result.accepted_kinds))
            rebootstrap_scheduled = (
                self.rebootstrap_mode == "oracle"
                or (
                    self.rebootstrap_mode == "periodic"
                    and frame_index is not None
                    and int(frame_index) > 1
                    and int(frame_index)
                    % self.rebootstrap_interval_frames == 0
                )
            )
            deficit = (
                None
                if target_deficit is None
                else np.asarray(target_deficit, dtype=np.float64).reshape(-1)
            )
            if deficit is not None and deficit.shape != (value.shape[2],):
                raise ValueError(
                    "target_deficit must have shape (num_targets,)")
            deficit_present = bool(
                deficit is not None and np.max(deficit) > 1.0e-9)
            rebootstrap_blocked_no_deficit = bool(
                rebootstrap_scheduled
                and self.rebootstrap_require_deficit
                and not deficit_present
            )
            rebootstrap_attempted = bool(
                rebootstrap_scheduled
                and not rebootstrap_blocked_no_deficit)
            if rebootstrap_attempted:
                # The normal N1--N3 maintenance result is the feasible warm
                # start.  N5 is then allowed to change a coupled 2--3 UAV /
                # 1--2 target block atomically, crossing the role-owner-edge
                # barrier without invoking the replicated global solver.
                rebootstrap_result = oracle_best_improvement(
                    result.selected,
                    value,
                    candidate,
                    rounds=self.rebootstrap_rounds,
                    neighborhoods=("N5",),
                    target_pair_limit=target_pair_limit,
                    reports_per_receiver=reports_per_receiver,
                    initial_role=result.role,
                )
                rebootstrap_candidates = int(sum(
                    rebootstrap_result.candidate_counts))
                rebootstrap_accepted_moves = int(len(
                    rebootstrap_result.accepted_kinds))
                rebootstrap_accepted = rebootstrap_accepted_moves > 0
                exact += rebootstrap_candidates
                result = rebootstrap_result

        if cold:
            maintenance_candidates = int(sum(result.candidate_counts))
            maintenance_accepted_moves = int(len(result.accepted_kinds))

        assert_local_feasible(
            result.selected,
            candidate,
            target_pair_limit=target_pair_limit,
            reports_per_receiver=reports_per_receiver,
        )
        self._selected = result.selected.copy()
        self._role = result.role.copy()
        self._resolve_index += 1
        return DynamicLocalSearchResult(result.selected.copy(), {
            "local_search_mode": self.mode,
            "local_search_cold_initializer": self.cold_initializer,
            "local_search_cold_start": float(cold),
            "local_search_candidate_count": int(
                maintenance_candidates + rebootstrap_candidates),
            "local_search_exact_verifications": exact,
            "local_search_accepted_moves": int(
                maintenance_accepted_moves + rebootstrap_accepted_moves),
            "local_search_selected_edges": int(np.sum(result.selected)),
            "local_search_resolve_index": int(self._resolve_index),
            "local_search_rebootstrap_mode": self.rebootstrap_mode,
            "local_search_rebootstrap_require_deficit": float(
                self.rebootstrap_require_deficit),
            "local_search_rebootstrap_deficit_present": float(
                deficit_present if not cold else False),
            "local_search_rebootstrap_blocked_no_deficit": float(
                rebootstrap_blocked_no_deficit if not cold else False),
            "local_search_rebootstrap_attempted": float(
                rebootstrap_attempted),
            "local_search_rebootstrap_accepted": float(
                rebootstrap_accepted),
            "local_search_rebootstrap_candidate_count": int(
                rebootstrap_candidates),
            "local_search_rebootstrap_accepted_moves": int(
                rebootstrap_accepted_moves),
        })
