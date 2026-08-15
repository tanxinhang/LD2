"""Same-state one-frame CRN audit for atomic N5 reconfiguration moves."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import numpy as np

from uav_isac.coordination.local_exchange_oracle import (
    LocalMove,
    enumerate_local_moves,
    objective_strictly_improves,
    oracle_best_improvement,
    role_owner_from_structure,
    structure_objective_key,
)


def _qos(pd: np.ndarray) -> dict[str, float]:
    values = np.asarray(pd, dtype=np.float64).reshape(-1)
    ordered = np.sort(values)
    return {
        "steady": float(np.mean(values)) if values.size else 0.0,
        "weak3": float(np.mean(ordered[:min(3, values.size)]))
        if values.size else 0.0,
        "worst": float(ordered[0]) if values.size else 0.0,
    }


def protected_target_pd(
    baseline_pd: np.ndarray,
    p_d_floor: float,
) -> np.ndarray:
    """Per-target no-op protection threshold used by Gate D0.5."""
    baseline = np.asarray(baseline_pd, dtype=np.float64).reshape(-1)
    return np.minimum(baseline, float(p_d_floor))


def targetwise_no_op_safe(
    candidate_pd: np.ndarray,
    baseline_pd: np.ndarray,
    *,
    p_d_floor: float,
    tolerance: float = 1.0e-9,
) -> bool:
    """Whether a physical candidate satisfies the target-wise safety set."""
    candidate = np.asarray(candidate_pd, dtype=np.float64).reshape(-1)
    protected = protected_target_pd(baseline_pd, p_d_floor)
    if candidate.shape != protected.shape:
        raise ValueError("candidate and baseline P_D must have the same shape")
    return bool(np.all(candidate + float(tolerance) >= protected))


def cumulative_detection_deficit(
    pd: np.ndarray,
    *,
    p_d_floor: float,
) -> np.ndarray:
    """Worst-first cumulative detection-deficit curve for equal targets."""
    values = np.asarray(pd, dtype=np.float64).reshape(-1)
    deficit = np.maximum(float(p_d_floor) - values, 0.0)
    return np.cumsum(np.sort(deficit)[::-1])


def tail_deficit_dominates(
    candidate_pd: np.ndarray,
    baseline_pd: np.ndarray,
    *,
    p_d_floor: float,
    tolerance: float = 1.0e-9,
) -> bool:
    """Whether every worst-k cumulative deficit is non-increasing.

    With equal-priority targets, element ``k-1`` of this curve is ``k`` times
    the empirical upper-tail CVaR of the QoS deficit.  Dominance therefore
    protects the bottleneck (k=1), every discrete deficit tail, and total
    deficit (k=Q), while remaining invariant to target labels.
    """
    candidate = cumulative_detection_deficit(
        candidate_pd, p_d_floor=p_d_floor)
    baseline = cumulative_detection_deficit(
        baseline_pd, p_d_floor=p_d_floor)
    if candidate.shape != baseline.shape:
        raise ValueError("candidate and baseline P_D must have the same shape")
    return bool(np.all(candidate <= baseline + float(tolerance)))


def _geometry(core) -> tuple[np.ndarray, list[np.ndarray]]:
    uavs = np.stack([uav.get_state().pos for uav in core.uavs])
    targets = [np.asarray(target.state, dtype=np.float64).copy()
               for target in core.targets]
    return uavs, targets


def _geometry_max_error(
        reference: tuple[np.ndarray, list[np.ndarray]], core) -> float:
    uavs, targets = _geometry(core)
    error = float(np.max(np.abs(uavs - reference[0])))
    for actual, expected in zip(targets, reference[1]):
        error = max(error, float(np.max(np.abs(actual - expected))))
    return error


def has_proxy_positive_atomic_n5(baseline_env) -> bool:
    """Whether the last local resolve exposes an atomic proxy-positive N5."""
    core = baseline_env.core
    coordinator = core._dynamic_local_search_coordinator
    if coordinator is None or coordinator.last_problem() is None:
        return False
    edge_value, candidate_mask = coordinator.last_problem()
    selected = np.asarray(
        core._cached_p0_solution.z_selected, dtype=bool).copy()
    state = coordinator.get_state()
    if state.get("role") is None:
        return False
    role, owner = role_owner_from_structure(
        selected,
        fallback_role=np.asarray(state["role"], dtype=np.int8),
    )
    reports_per_receiver = (
        max(1, int(
            core.cfg.p0_solver.capacity_per_rx
            // max(core.cfg.detection.B_q, 1)))
        if core.ground_communication_enabled
        else core.Q * core.cfg.detection.K_q_max
    )
    current_key = structure_objective_key(selected, edge_value)
    moves = enumerate_local_moves(
        selected,
        edge_value,
        candidate_mask,
        role,
        owner,
        neighborhoods=("N5",),
        target_pair_limit=int(core.cfg.detection.K_q_max),
        reports_per_receiver=int(reports_per_receiver),
        n5_target_mode="proxy_weak",
    )
    return any(objective_strictly_improves(
        structure_objective_key(move.selected, edge_value), current_key)
        for move in moves)


def audit_atomic_n5_event(
    pre_step_env,
    baseline_env,
    actions: Mapping,
    baseline_info: Mapping,
    *,
    episode_seed: int,
    frame: int,
    target_mode: str = "both",
    max_candidates: int = 0,
    weak_target_budgets: tuple[int, ...] = (1, 2, 3),
    steady_floor: float = 0.80,
    weak3_floor: float = 0.70,
    p_d_floor: float = 0.60,
    n5_rebuild_scope: str = "global",
    neighborhood: str = "N5",
) -> dict[str, object]:
    """Replay all unique atomic N5 moves against one untouched baseline step.

    ``pre_step_env`` must be copied after communication/resource submissions
    and immediately before the baseline ``step``.  The baseline environment
    must contain the resulting resolved local structure.  Every branch is a
    fresh deep copy of the pre-step environment, so channel, motion and noise
    streams use common random numbers.
    """
    normalized = str(target_mode).strip().lower()
    if normalized not in {"proxy_weak", "all", "both"}:
        raise ValueError("target_mode must be proxy_weak, all, or both")
    normalized_rebuild_scope = str(n5_rebuild_scope).strip().lower()
    if normalized_rebuild_scope not in {"global", "target_block"}:
        raise ValueError(
            "n5_rebuild_scope must be global or target_block")
    normalized_neighborhood = str(neighborhood).strip().upper()
    if normalized_neighborhood not in {"N5", "N6"}:
        raise ValueError("neighborhood must be N5 or N6")
    core = baseline_env.core
    coordinator = core._dynamic_local_search_coordinator
    if coordinator is None:
        raise RuntimeError("N5 audit requires dynamic local search")
    problem = coordinator.last_problem()
    if problem is None:
        raise RuntimeError("dynamic coordinator has no resolved public problem")
    edge_value, candidate_mask = problem
    selected = np.asarray(
        core._cached_p0_solution.z_selected, dtype=bool).copy()
    state = coordinator.get_state()
    role = np.asarray(state["role"], dtype=np.int8)
    role, owner = role_owner_from_structure(
        selected, fallback_role=role)
    reports_per_receiver = (
        max(1, int(
            core.cfg.p0_solver.capacity_per_rx
            // max(core.cfg.detection.B_q, 1)))
        if core.ground_communication_enabled
        else core.Q * core.cfg.detection.K_q_max
    )
    kwargs = dict(
        neighborhoods=(normalized_neighborhood,),
        target_pair_limit=int(core.cfg.detection.K_q_max),
        reports_per_receiver=int(reports_per_receiver),
        n5_rebuild_scope=normalized_rebuild_scope,
    )
    requested_budgets = tuple(sorted(set(
        min(max(1, int(value)), selected.shape[2])
        for value in weak_target_budgets
    )))
    if 2 not in requested_budgets:
        requested_budgets = tuple(sorted((*requested_budgets, 2)))
    budget_moves = {
        budget: enumerate_local_moves(
            selected,
            edge_value,
            candidate_mask,
            role,
            owner,
            n5_target_mode="proxy_weak",
            n5_weak_target_count=int(budget),
            **kwargs,
        )
        for budget in requested_budgets
    }
    proxy_moves = budget_moves[2]
    all_moves = (
        enumerate_local_moves(
            selected, edge_value, candidate_mask, role, owner,
            n5_target_mode="all", **kwargs)
        if normalized in {"all", "both"} else proxy_moves)
    proxy_keys = {move.selected.tobytes() for move in proxy_moves}
    budget_keys = {
        int(budget): {move.selected.tobytes() for move in pool}
        for budget, pool in budget_moves.items()
    }
    moves = proxy_moves if normalized == "proxy_weak" else all_moves
    current_key = structure_objective_key(selected, edge_value)
    ordered = sorted(
        moves,
        key=lambda move: structure_objective_key(move.selected, edge_value),
        reverse=True,
    )
    truncated = int(max_candidates) > 0 and len(ordered) > int(max_candidates)
    if int(max_candidates) > 0:
        ordered = ordered[:int(max_candidates)]
    atomic_evaluated_count = len(ordered)
    proxy_sequence = oracle_best_improvement(
        selected,
        edge_value,
        candidate_mask,
        rounds=8,
        neighborhoods=(normalized_neighborhood,),
        target_pair_limit=int(core.cfg.detection.K_q_max),
        reports_per_receiver=int(reports_per_receiver),
        initial_role=role,
        n5_rebuild_scope=normalized_rebuild_scope,
    )
    proxy_sequence_key: bytes | None = None
    proxy_sequence_record_index: int | None = None
    if proxy_sequence.accepted_kinds:
        proxy_sequence_key = proxy_sequence.selected.tobytes()
        proxy_sequence_record_index = next((
            index for index, move in enumerate(ordered)
            if move.selected.tobytes() == proxy_sequence_key
        ), None)
        if proxy_sequence_record_index is None:
            ordered.append(LocalMove(
                "N5_sequence",
                proxy_sequence.selected,
                proxy_sequence.role,
                proxy_sequence.owner,
            ))
            proxy_sequence_record_index = len(ordered) - 1

    baseline_pd = np.asarray(baseline_info["P_D_q"], dtype=np.float64)
    baseline_qos = _qos(baseline_pd)
    target_proxy_value = np.sum(
        np.where(selected, np.maximum(edge_value, 0.0), 0.0),
        axis=(0, 1),
    )
    target_proxy_order = np.argsort(
        target_proxy_value, kind="stable")
    baseline_geometry = _geometry(core)
    records: list[dict[str, object]] = []
    for index, move in enumerate(ordered):
        proxy_key = structure_objective_key(move.selected, edge_value)
        branch_env = deepcopy(pre_step_env)
        try:
            branch_coordinator = (
                branch_env.core._dynamic_local_search_coordinator)
            if branch_coordinator is None:
                raise RuntimeError("branch lost dynamic coordinator")
            branch_coordinator.force_next_move_for_audit(move)
            _, _, terminated, truncated_step, info = branch_env.step(actions)
            if not bool(info.get("p0_resolved", False)):
                raise RuntimeError("forced N5 branch did not resolve P0")
            actual = np.asarray(
                branch_env.core._cached_p0_solution.z_selected, dtype=bool)
            if not np.array_equal(actual, move.selected):
                raise AssertionError("forced branch executed a different move")
            pd = np.asarray(info["P_D_q"], dtype=np.float64)
            qos = _qos(pd)
            changed_target_ids = np.flatnonzero(np.any(
                selected ^ move.selected, axis=(0, 1)))
            changed_role_ids = np.flatnonzero(role != move.role)
            changed_owner_ids = np.flatnonzero(owner != move.owner)
            records.append({
                "candidate_index": int(index),
                "is_atomic_candidate": bool(
                    index < atomic_evaluated_count),
                "is_proxy_sequence": bool(
                    proxy_sequence_record_index == index),
                "in_proxy_weak_pool": bool(
                    move.selected.tobytes() in proxy_keys),
                "in_weak_target_budget": {
                    str(budget): bool(
                        move.selected.tobytes() in keys)
                    for budget, keys in budget_keys.items()
                },
                "proxy_positive": bool(
                    objective_strictly_improves(proxy_key, current_key)),
                "proxy_key": [float(value) for value in proxy_key[:4]],
                "proxy_primary_delta": float(
                    proxy_key[0] - current_key[0]),
                "changed_edges": int(np.count_nonzero(
                    selected ^ move.selected)),
                "changed_roles": int(np.count_nonzero(role != move.role)),
                "changed_owners": int(np.count_nonzero(owner != move.owner)),
                "changed_target_ids": [
                    int(value) for value in changed_target_ids],
                "changed_target_count": int(len(changed_target_ids)),
                "changed_role_ids": [
                    int(value) for value in changed_role_ids],
                "changed_owner_targets": [
                    int(value) for value in changed_owner_ids],
                "pd": pd.tolist(),
                "delta_pd": (pd - baseline_pd).tolist(),
                "steady": qos["steady"],
                "weak3": qos["weak3"],
                "worst": qos["worst"],
                "delta_steady": qos["steady"] - baseline_qos["steady"],
                "delta_weak3": qos["weak3"] - baseline_qos["weak3"],
                "delta_worst": qos["worst"] - baseline_qos["worst"],
                "geometry_max_error": _geometry_max_error(
                    baseline_geometry, branch_env.core),
                "power_balance_error_w": float(info.get(
                    "isac_max_power_balance_error_w", 0.0)),
                "done": bool(
                    terminated.get("__all__", False)
                    or truncated_step.get("__all__", False)),
            })
        finally:
            branch_env.close()

    def best_record(pool: str) -> dict[str, object] | None:
        eligible = [
            record for record in records
            if record["is_atomic_candidate"]
            and (pool == "all" or record["in_proxy_weak_pool"])]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda record: (
                record["worst"], record["weak3"], record["steady"]),
        )

    def budget_records(budget: int) -> list[dict[str, object]]:
        key = str(int(budget))
        return [
            record for record in records
            if record["is_atomic_candidate"]
            and bool(record["in_weak_target_budget"].get(key, False))
        ]

    protected_steady = min(
        float(baseline_qos["steady"]), float(steady_floor))
    protected_weak3 = min(
        float(baseline_qos["weak3"]), float(weak3_floor))
    protected_pd = protected_target_pd(baseline_pd, float(p_d_floor))

    def no_op_safe_oracle(
        candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        # The no-op is an explicit feasible action.  A reconfiguration is
        # admissible only if it improves the physical bottleneck while not
        # worsening a metric that is already below its floor and not crossing
        # a floor that is currently satisfied.
        admissible = [
            record for record in candidates
            if float(record["delta_worst"]) > 1.0e-4
            and float(record["steady"]) + 1.0e-9 >= protected_steady
            and float(record["weak3"]) + 1.0e-9 >= protected_weak3
        ]
        if not admissible:
            return {
                "choice": "no_op",
                "candidate_index": None,
                **baseline_qos,
                "delta_steady": 0.0,
                "delta_weak3": 0.0,
                "delta_worst": 0.0,
            }
        chosen = max(
            admissible,
            key=lambda record: (
                record["worst"], record["weak3"], record["steady"]),
        )
        return {
            "choice": "candidate",
            "candidate_index": int(chosen["candidate_index"]),
            "steady": float(chosen["steady"]),
            "weak3": float(chosen["weak3"]),
            "worst": float(chosen["worst"]),
            "delta_steady": float(chosen["delta_steady"]),
            "delta_weak3": float(chosen["delta_weak3"]),
            "delta_worst": float(chosen["delta_worst"]),
        }

    def no_op_vector_safe_oracle(
        candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        """Exact-P_D ceiling for the proposed target-wise safety rule.

        This remains an offline physical Oracle.  It is useful for deciding
        whether a future owner-local lower-bound certificate has any attainable
        headroom, but it is not itself available to a deployed coordinator.
        """
        admissible = [
            record for record in candidates
            if float(record["delta_worst"]) > 1.0e-4
            and targetwise_no_op_safe(
                np.asarray(record["pd"], dtype=np.float64),
                baseline_pd,
                p_d_floor=float(p_d_floor),
            )
        ]
        if not admissible:
            return {
                "choice": "no_op",
                "candidate_index": None,
                **baseline_qos,
                "delta_steady": 0.0,
                "delta_weak3": 0.0,
                "delta_worst": 0.0,
                "admissible_positive_count": 0,
            }
        chosen = max(
            admissible,
            key=lambda record: (
                record["worst"], record["weak3"], record["steady"]),
        )
        return {
            "choice": "candidate",
            "candidate_index": int(chosen["candidate_index"]),
            "steady": float(chosen["steady"]),
            "weak3": float(chosen["weak3"]),
            "worst": float(chosen["worst"]),
            "delta_steady": float(chosen["delta_steady"]),
            "delta_weak3": float(chosen["delta_weak3"]),
            "delta_worst": float(chosen["delta_worst"]),
            "admissible_positive_count": int(len(admissible)),
        }

    def no_op_tail_deficit_oracle(
        candidates: list[dict[str, object]],
    ) -> dict[str, object]:
        admissible = [
            record for record in candidates
            if float(record["delta_worst"]) > 1.0e-4
            and tail_deficit_dominates(
                np.asarray(record["pd"], dtype=np.float64),
                baseline_pd,
                p_d_floor=float(p_d_floor),
            )
        ]
        if not admissible:
            return {
                "choice": "no_op",
                "candidate_index": None,
                **baseline_qos,
                "delta_steady": 0.0,
                "delta_weak3": 0.0,
                "delta_worst": 0.0,
                "admissible_positive_count": 0,
            }
        chosen = max(
            admissible,
            key=lambda record: (
                record["worst"], record["weak3"], record["steady"]),
        )
        return {
            "choice": "candidate",
            "candidate_index": int(chosen["candidate_index"]),
            "steady": float(chosen["steady"]),
            "weak3": float(chosen["weak3"]),
            "worst": float(chosen["worst"]),
            "delta_steady": float(chosen["delta_steady"]),
            "delta_weak3": float(chosen["delta_weak3"]),
            "delta_worst": float(chosen["delta_worst"]),
            "admissible_positive_count": int(len(admissible)),
            "candidate_deficit_curve": cumulative_detection_deficit(
                np.asarray(chosen["pd"], dtype=np.float64),
                p_d_floor=float(p_d_floor),
            ).tolist(),
        }

    evaluated_budgets = (
        tuple(budget for budget in requested_budgets if budget <= 2)
        if normalized == "proxy_weak"
        else requested_budgets
    )
    budget_summaries: dict[str, object] = {}
    for budget in evaluated_budgets:
        pool = budget_records(budget)
        unconstrained = (
            max(
                pool,
                key=lambda record: (
                    record["worst"], record["weak3"], record["steady"]),
            )
            if pool else None
        )
        budget_summaries[str(budget)] = {
            "candidate_count": int(len(budget_moves[budget])),
            "evaluated_candidate_count": int(len(pool)),
            "physical_positive_count": int(sum(
                float(record["delta_worst"]) > 1.0e-4
                for record in pool
            )),
            "proxy_positive_count": int(sum(
                bool(record["proxy_positive"]) for record in pool
            )),
            "unconstrained_physical_oracle": unconstrained,
            "no_op_safe_physical_oracle": no_op_safe_oracle(pool),
            "no_op_vector_safe_physical_oracle": (
                no_op_vector_safe_oracle(pool)),
            "no_op_tail_deficit_physical_oracle": (
                no_op_tail_deficit_oracle(pool)),
        }

    proxy_positive_records = [
        record for record in records
        if record["is_atomic_candidate"]
        and record["in_proxy_weak_pool"]
        and record["proxy_positive"]]
    proxy_choice = (
        max(proxy_positive_records, key=lambda record: record["proxy_key"])
        if proxy_positive_records else None)
    proxy_negative_records = [
        record for record in records
        if record["is_atomic_candidate"]
        and record["in_proxy_weak_pool"]
        and not record["proxy_positive"]]
    physical_positive_records = [
        record for record in records
        if record["is_atomic_candidate"]
        and record["in_proxy_weak_pool"]
        and float(record["delta_worst"]) > 1.0e-4]
    missed_positive_records = [
        record for record in proxy_negative_records
        if float(record["delta_worst"]) > 1.0e-4]
    material = [
        record for record in records
        if abs(float(record["delta_worst"])) > 1.0e-4]
    return {
        "episode_seed": int(episode_seed),
        "frame": int(frame),
        "target_mode": normalized,
        "n5_rebuild_scope": normalized_rebuild_scope,
        "neighborhood": normalized_neighborhood,
        "atomic_only": True,
        "candidate_count_proxy_weak": int(len(proxy_moves)),
        "candidate_count_all": int(len(all_moves)),
        "evaluated_atomic_candidate_count": int(atomic_evaluated_count),
        "evaluated_candidate_count": int(len(records)),
        "candidate_truncated": bool(truncated),
        "baseline_pd": baseline_pd.tolist(),
        "baseline": baseline_qos,
        "qos_safety_rule": {
            "steady_floor": float(steady_floor),
            "weak3_floor": float(weak3_floor),
            "protected_steady": float(protected_steady),
            "protected_weak3": float(protected_weak3),
            "rule": (
                "improve physical worst; preserve the baseline when it is "
                "below a floor, otherwise remain above that floor; compare "
                "against an explicit no-op"
            ),
        },
        "vector_qos_safety_rule": {
            "p_d_floor": float(p_d_floor),
            "protected_pd": protected_pd.tolist(),
            "rule": (
                "improve physical worst and preserve every target at its "
                "baseline when below the floor, otherwise remain above the "
                "floor; compare against an explicit no-op"
            ),
            "scope": (
                "offline exact-P_D headroom ceiling for a future calibrated "
                "owner-local lower-bound certificate"
            ),
        },
        "tail_deficit_safety_rule": {
            "p_d_floor": float(p_d_floor),
            "baseline_deficit_curve": cumulative_detection_deficit(
                baseline_pd, p_d_floor=float(p_d_floor)).tolist(),
            "rule": (
                "sort positive detection deficits from largest to smallest; "
                "for every k, the candidate cumulative worst-k deficit must "
                "not exceed no-op, and physical worst must strictly improve"
            ),
            "guarantees": (
                "for equal-priority targets this preserves maximum deficit, "
                "every discrete upper-tail deficit CVaR, and total deficit"
            ),
        },
        "target_proxy_value": target_proxy_value.tolist(),
        "target_proxy_weak_order": [
            int(value) for value in target_proxy_order],
        "weak_target_budget_summaries": budget_summaries,
        "current_proxy_key": [float(value) for value in current_key[:4]],
        "proxy_choice": proxy_choice,
        "proxy_sequence": next((
            record for record in records
            if record["is_proxy_sequence"]), None),
        "proxy_sequence_accepted_moves": int(len(
            proxy_sequence.accepted_kinds)),
        "oracle_proxy_pool": best_record("proxy"),
        "oracle_all_target_pool": best_record("all"),
        "material_sign_agreement": float(np.mean([
            bool(record["proxy_positive"])
            == (float(record["delta_worst"]) > 0.0)
            for record in material
        ])) if material else 0.0,
        "false_accept_rate": float(np.mean([
            float(record["delta_worst"]) < -1.0e-4
            for record in proxy_positive_records
        ])) if proxy_positive_records else 0.0,
        "false_miss_rate": float(np.mean([
            float(record["delta_worst"]) > 1.0e-4
            for record in proxy_negative_records
        ])) if proxy_negative_records else 0.0,
        "physical_positive_coverage": float(
            1.0 - len(missed_positive_records)
            / max(len(physical_positive_records), 1)),
        "max_geometry_error": float(max(
            [record["geometry_max_error"] for record in records]
            or [0.0])),
        "max_power_balance_error_w": float(max(
            [record["power_balance_error_w"] for record in records]
            or [0.0])),
        "affected_target_summary": {
            "mean": float(np.mean([
                record["changed_target_count"] for record in records
                if record["is_atomic_candidate"]
            ])) if atomic_evaluated_count else 0.0,
            "maximum": int(max([
                record["changed_target_count"] for record in records
                if record["is_atomic_candidate"]
            ] or [0])),
            "over_two_count": int(sum(
                int(record["changed_target_count"]) > 2
                for record in records
                if record["is_atomic_candidate"]
            )),
        },
        "candidates": records,
    }
