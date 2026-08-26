#!/usr/bin/env python
"""M4-D1: exact referee for a previously frozen oracle-free candidate pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.capability import (  # noqa: E402
    minimum_total_power_pwl_lp,
)
from uav_isac.coordination.dependency_commit import (  # noqa: E402
    DependencyCommitLayout,
    dependency_closure,
)
from uav_isac.coordination.local_exchange_oracle import LocalMove  # noqa: E402
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
)
from uav_isac.coordination.minimum_intervention_repair import (  # noqa: E402
    solve_minimum_intervention_task_repair_milp,
)
from uav_isac.coordination.oracle_free_candidate_locator import (  # noqa: E402
    structure_digest,
    structure_state,
)
from uav_isac.coordination.pwl_pd import (  # noqa: E402
    saturating_chord_lower_bound,
)
from uav_isac.evaluation.layer_provenance import (  # noqa: E402
    audit_layer_trace_provenance,
    canonical_json_sha256,
)
from uav_isac.physical.detection import (  # noqa: E402
    minimum_deflection_for_detection_probability,
)
from uav_isac.utils.provenance import sha256_file  # noqa: E402


POWER_TOLERANCE_W = 1.0e-8


def _mask(edges, K: int, Q: int) -> np.ndarray:
    selected = np.zeros((K, K, Q), dtype=bool)
    for edge in edges:
        selected[tuple(map(int, edge))] = True
    return selected


def _prefix(
    selected0: np.ndarray, role0: np.ndarray, owner0: np.ndarray,
    candidate: dict, layout: DependencyCommitLayout,
) -> tuple[int, int]:
    move = LocalMove(
        kind="frozen_oracle_free",
        selected=_mask(candidate["edges"], *selected0.shape[::2]),
        role=np.asarray(candidate["role"], dtype=np.int8),
        owner=np.asarray(candidate["owner"], dtype=np.int64),
    )
    closure = dependency_closure(selected0, role0, owner0, move)
    bits = layout.prepare_bits(
        changed_roles=closure.changed_roles,
        changed_owners=closure.changed_owners,
        toggled_edges=closure.toggled_edges)
    return len(closure.affected_targets) + len(closure.participants), bits


def _restricted_optimum(
    candidates: list[dict],
    *,
    coefficient: np.ndarray,
    budget: np.ndarray,
    selected0: np.ndarray,
    role0: np.ndarray,
    owner0: np.ndarray,
    layout: DependencyCommitLayout,
    p_fa: float,
    floors: tuple[float, float, float, int],
    slopes: np.ndarray,
    intercepts: np.ndarray,
    d_min: float,
) -> tuple[tuple[int, int, float] | None, list[str], int]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for candidate in candidates:
        groups[_prefix(selected0, role0, owner0, candidate, layout)].append(
            candidate)
    evaluated = 0
    for prefix in sorted(groups):
        feasible: list[tuple[float, str]] = []
        for candidate in groups[prefix]:
            evaluated += 1
            try:
                gain, _ = fixed_owner_gain_matrix(
                    coefficient,
                    [tuple(map(int, edge)) for edge in candidate["edges"]])
            except ValueError:
                continue
            solved = minimum_total_power_pwl_lp(
                gain, budget, p_fa, floors,
                slopes, intercepts, d_min)
            if solved is not None:
                feasible.append((float(solved[0]), str(candidate["digest"])))
        if feasible:
            best_power = min(item[0] for item in feasible)
            equivalent = sorted(
                digest for power, digest in feasible
                if abs(power - best_power) <= POWER_TOLERANCE_W)
            return (prefix[0], prefix[1], best_power), equivalent, evaluated
    return None, [], evaluated


def audit(
    trace_path: Path,
    config_path: Path,
    frozen_path: Path,
    *,
    time_limit_s: float,
) -> dict[str, object]:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("stage") != "M4-D0_ORACLE_FREE_CANDIDATE_FREEZE":
        raise ValueError("input is not an M4-D0 candidate freeze")
    if canonical_json_sha256(frozen["rows"]) != frozen.get(
        "candidate_payload_sha256"):
        raise ValueError("frozen candidate payload hash mismatch")
    locator_file = ROOT / "uav_isac/coordination/oracle_free_candidate_locator.py"
    if sha256_file(locator_file) != frozen.get("locator_code_sha256"):
        raise ValueError("locator code changed after candidate freeze")
    referee_started_ns = time.time_ns()
    referee_started_utc = datetime.now(timezone.utc).isoformat()
    if int(frozen["candidate_freeze_unix_ns"]) >= referee_started_ns:
        raise ValueError("candidate freeze is not earlier than exact referee")

    p0 = audit_layer_trace_provenance(trace_path)
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    trace_provenance = json.loads(str(
        np.asarray(data["provenance_json"]).reshape(-1)[0]))
    cfg = load_config(str(config_path))
    resolved = asdict(cfg) if is_dataclass(cfg) else cfg
    if canonical_json_sha256(resolved) != trace_provenance["config_sha256"]:
        raise ValueError("supplied config does not match trace config hash")

    frame_lookup = {
        (int(data["seed"][index]), int(data["frame"][index])): index
        for index in range(np.asarray(data["frame"]).size)
    }
    floors = tuple(cfg.marl.task_constrained_qos_floors)
    p_fa = float(cfg.detection.P_FA)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([float(floors[0])]), p_fa)[0])
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    receiver_limit = int(
        np.asarray(data["reports_per_receiver"]).reshape(-1)[0])
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    layout = DependencyCommitLayout(K, Q)
    rows = []
    started = time.perf_counter()
    for frozen_row in frozen["rows"]:
        key = (int(frozen_row["seed"]), int(frozen_row["frame"]))
        if key not in frame_lookup:
            raise ValueError(f"frozen frame {key} is absent from trace")
        index = frame_lookup[key]
        coefficient = np.asarray(
            data["per_watt_coefficient"][index], dtype=np.float64)
        budget = np.asarray(
            data["frame_sensing_budget_w"][index], dtype=np.float64)
        selected0 = np.asarray(
            data["deployed_selected"][index], dtype=bool)
        role0 = np.asarray(data["deployed_role"][index], dtype=np.int8)
        owner0 = np.asarray(
            data["deployed_receiver_owner"][index], dtype=np.int64)
        # Candidate integrity is revalidated before the oracle is allowed to
        # judge it; malformed records cannot become accidental competitors.
        for candidate in frozen_row["candidates"]:
            state = structure_state(candidate["edges"], K, Q)
            if state is None:
                raise ValueError(f"illegal frozen candidate on frame {key}")
            canonical, role, owner = state
            if (
                structure_digest(canonical, K, Q) != candidate["digest"]
                or list(role) != candidate["role"]
                or list(owner) != candidate["owner"]
            ):
                raise ValueError(f"candidate digest/state mismatch on frame {key}")

        exact = solve_minimum_intervention_task_repair_milp(
            coefficient, budget, selected0,
            role0 == 0, role0 == 1, owner0,
            p_fa=p_fa, task_floors=floors,
            target_pair_limit=pair_limit,
            reports_per_receiver=receiver_limit,
            pwl_epsilon=1.0e-3,
            time_limit_s=float(time_limit_s),
        )
        if not exact.feasible:
            rows.append({
                "seed": key[0], "frame": key[1],
                "exact_status": "UNRESOLVED_OR_INFEASIBLE",
                "solver_status": exact.solver_status,
            })
            continue
        global_objective = (
            int(exact.closure_cardinality), int(exact.prepare_bits),
            float(exact.total_sensing_power_w))
        if exact.closure_cardinality == 0:
            rows.append({
                "seed": key[0], "frame": key[1],
                "exact_status": "NOT_GENUINE_L2_CLOSURE_ZERO",
                "global_objective": list(global_objective),
            })
            continue

        relaxed_d = np.sum(
            np.max(coefficient, axis=1) * budget[:, None], axis=0)
        d_max = max(float(np.max(relaxed_d)), d_min + 1.0e-9)
        slopes, intercepts, _ = saturating_chord_lower_bound(
            p_fa, d_min, d_max, 1.0e-3)
        pool_a = [
            candidate for candidate in frozen_row["candidates"]
            if candidate["origin_pool"] == "A_DUAL_NESTED"
        ]
        pool_b = list(frozen_row["candidates"])
        result_a, equivalent_a, evaluated_a = _restricted_optimum(
            pool_a, coefficient=coefficient, budget=budget,
            selected0=selected0, role0=role0, owner0=owner0, layout=layout,
            p_fa=p_fa, floors=floors, slopes=slopes,
            intercepts=intercepts, d_min=d_min)
        result_b, equivalent_b, evaluated_b = _restricted_optimum(
            pool_b, coefficient=coefficient, budget=budget,
            selected0=selected0, role0=role0, owner0=owner0, layout=layout,
            p_fa=p_fa, floors=floors, slopes=slopes,
            intercepts=intercepts, d_min=d_min)

        def covered(result) -> bool:
            return bool(
                result is not None
                and result[0] == global_objective[0]
                and result[1] == global_objective[1]
                and abs(result[2] - global_objective[2]) <= POWER_TOLERANCE_W)

        rows.append({
            "seed": key[0], "frame": key[1],
            "exact_status": "GENUINE_L2",
            "global_objective": list(global_objective),
            "global_witness_digest": structure_digest(
                exact.selected_set, K, Q),
            "pool_a_candidate_count": len(pool_a),
            "pool_b_candidate_count": len(pool_b),
            "pool_a_restricted_objective": (
                None if result_a is None else list(result_a)),
            "pool_b_restricted_objective": (
                None if result_b is None else list(result_b)),
            "pool_a_optimal_equivalent_digests": equivalent_a,
            "pool_b_optimal_equivalent_digests": equivalent_b,
            "pool_a_evaluated_candidates": evaluated_a,
            "pool_b_evaluated_candidates": evaluated_b,
            "pool_a_coverage": covered(result_a),
            "pool_b_coverage": covered(result_b),
            "classification": (
                "A_COVERS_INFORMATION_RATE_READY" if covered(result_a)
                else "LOCALIZATION_GAP" if covered(result_b)
                else "MOVE_GRAMMAR_GAP"),
        })

    l2_rows = [row for row in rows if row["exact_status"] == "GENUINE_L2"]
    counts = Counter(row["classification"] for row in l2_rows)
    return {
        "schema_version": 1,
        "stage": "M4-D1_CANDIDATE_SUFFICIENCY_REFEREE",
        "candidate_artifact": str(frozen_path),
        "candidate_payload_sha256": frozen["candidate_payload_sha256"],
        "candidate_freeze_unix_ns": int(frozen["candidate_freeze_unix_ns"]),
        "referee_started_unix_ns": referee_started_ns,
        "temporal_separation_pass": (
            int(frozen["candidate_freeze_unix_ns"]) < referee_started_ns),
        "trace_p0": p0,
        "referee_code_sha256": sha256_file(Path(__file__)),
        "referee_started_utc": referee_started_utc,
        "power_equivalence_tolerance_w": POWER_TOLERANCE_W,
        "sampled_frames": len(rows),
        "genuine_l2_frames": len(l2_rows),
        "candidate_sufficiency_counts": dict(sorted(counts.items())),
        "pool_a_coverage_rate": float(np.mean([
            row["pool_a_coverage"] for row in l2_rows])) if l2_rows else None,
        "pool_b_coverage_rate": float(np.mean([
            row["pool_b_coverage"] for row in l2_rows])) if l2_rows else None,
        "d2_eligible_frames": sum(
            bool(row["pool_a_coverage"]) for row in l2_rows),
        "stop_rule": (
            "ENTER_D2_ON_POOL_A_COVERED_FRAMES"
            if any(row["pool_a_coverage"] for row in l2_rows)
            else "STOP_AT_CANDIDATE_LOCALIZATION"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.trace, args.config, args.candidates,
        time_limit_s=args.time_limit_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in result.items() if key != "rows"},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
