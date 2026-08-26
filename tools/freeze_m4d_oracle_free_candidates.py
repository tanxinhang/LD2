#!/usr/bin/env python
"""M4-D0: freeze oracle-free candidate pools before any new exact referee."""

from __future__ import annotations

import argparse
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
    capability_gauge_pwl_lp_certificate,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    fixed_owner_gain_matrix,
)
from uav_isac.coordination.oracle_free_candidate_locator import (  # noqa: E402
    generate_oracle_free_candidate_pools,
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


def freeze(
    trace_path: Path,
    config_path: Path,
    *,
    stride: int,
    nested_budgets: tuple[int, ...],
) -> dict[str, object]:
    if stride < 1:
        raise ValueError("stride must be positive")
    p0 = audit_layer_trace_provenance(trace_path)
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    provenance = json.loads(str(np.asarray(data["provenance_json"]).reshape(-1)[0]))
    cfg = load_config(str(config_path))
    resolved = asdict(cfg) if is_dataclass(cfg) else cfg
    if canonical_json_sha256(resolved) != provenance["config_sha256"]:
        raise ValueError("supplied config does not match trace config hash")
    floors = tuple(cfg.marl.task_constrained_qos_floors)
    p_fa = float(cfg.detection.P_FA)
    d_min = float(minimum_deflection_for_detection_probability(
        np.asarray([float(floors[0])]), p_fa)[0])
    d_saturation = float(minimum_deflection_for_detection_probability(
        np.asarray([0.999]), p_fa)[0])
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    receiver_limit = int(
        np.asarray(data["reports_per_receiver"]).reshape(-1)[0])
    indices = [
        index for index, frame in enumerate(np.asarray(data["frame"]))
        if int(frame) % stride == 0
    ]
    started = time.perf_counter()
    rows = []
    for index in indices:
        coefficient = np.asarray(
            data["per_watt_coefficient"][index], dtype=np.float64)
        budget = np.asarray(
            data["frame_sensing_budget_w"][index], dtype=np.float64)
        selected = [
            tuple(map(int, edge))
            for edge in np.argwhere(data["deployed_selected"][index])
        ]
        fixed_gain, owners = fixed_owner_gain_matrix(coefficient, selected)
        relaxed_d = np.sum(
            np.max(coefficient, axis=1) * budget[:, None], axis=0)
        d_max = max(
            2.0 * float(np.max(relaxed_d)),
            d_saturation + 1.0,
            d_min + 1.0,
        )
        slopes, intercepts, _ = saturating_chord_lower_bound(
            p_fa, d_min, d_max, 1.0e-3)
        certificate = capability_gauge_pwl_lp_certificate(
            fixed_gain, budget, p_fa, floors,
            slopes, intercepts, d_min)
        if certificate is None:
            raise RuntimeError("fixed capability certificate is undefined")
        gamma, power, target_price, scarcity_price = certificate
        candidates = generate_oracle_free_candidate_pools(
            coefficient, budget, selected,
            np.asarray(data["deployed_role"][index], dtype=np.int8),
            target_price, scarcity_price,
            target_pair_limit=pair_limit,
            reports_per_receiver=receiver_limit,
            nested_budgets=nested_budgets,
        )
        encoded = [{
            "digest": candidate.digest,
            "edges": [list(edge) for edge in candidate.edges],
            "role": list(candidate.role),
            "owner": list(candidate.owner),
            "origin_pool": candidate.pool,
            "nested_budget": candidate.nested_budget,
            "score_mode": candidate.score_mode,
            "support_size": candidate.support_size,
            "owner_rank": candidate.owner_rank,
        } for candidate in candidates]
        rows.append({
            "seed": int(data["seed"][index]),
            "episode": int(data["episode"][index]),
            "frame": int(data["frame"][index]),
            "pre_oracle_state_sha256": canonical_json_sha256({
                "selected": selected,
                "coefficient_sha256": canonical_json_sha256(
                    coefficient.tolist()),
                "budget": budget.tolist(),
                "gamma_fixed": float(gamma),
                "target_price": target_price.tolist(),
                "scarcity_price": scarcity_price.tolist(),
                "l1_power": power.tolist(),
            }),
            "gamma_fixed": float(gamma),
            "target_price": target_price.tolist(),
            "scarcity_price": scarcity_price.tolist(),
            "budget_dual_stationarity": float(
                np.dot(scarcity_price, budget)),
            "deployed_owner": owners.tolist(),
            "candidate_count": len(encoded),
            "pool_a_count": sum(
                item["origin_pool"] == "A_DUAL_NESTED" for item in encoded),
            # Pool B is a diagnostic superset: all candidates frozen here.
            "pool_b_union_count": len(encoded),
            "candidates": encoded,
        })

    frozen_payload = {
        "schema_version": 1,
        "stage": "M4-D0_ORACLE_FREE_CANDIDATE_FREEZE",
        "scope": "all systematic frames before L2-label filtering",
        "trace": str(trace_path),
        "trace_p0": p0,
        "config": str(config_path),
        "sampling": {
            "stride": int(stride),
            "frames": len(rows),
            "rule": "frame modulo stride equals zero; no P1/L2 input",
        },
        "nested_budgets": list(nested_budgets),
        "locator_inputs": [
            "deployed_structure", "per_watt_coefficient", "residual_budget",
            "gamma_fixed", "canonical_l1_target_price",
            "canonical_l1_budget_price", "hard_structure_constraints",
        ],
        "forbidden_inputs_absent": [
            "exact_witness", "witness_closure", "MILP_variables",
            "MILP_reduced_cost", "feasibility_query_feedback",
            "B2b_fixing_conflict", "exact_optimum", "P1_layer_label",
        ],
        "locator_code_sha256": sha256_file(
            ROOT / "uav_isac/coordination/oracle_free_candidate_locator.py"),
        "freeze_tool_sha256": sha256_file(Path(__file__)),
        "rows": rows,
    }
    frozen_payload["candidate_payload_sha256"] = canonical_json_sha256(rows)
    frozen_payload["candidate_freeze_utc"] = datetime.now(
        timezone.utc).isoformat()
    frozen_payload["candidate_freeze_unix_ns"] = time.time_ns()
    frozen_payload["elapsed_seconds"] = time.perf_counter() - started
    return frozen_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--nested-budgets", default="1,2,3")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    budgets = tuple(int(item) for item in args.nested_budgets.split(",") if item)
    result = freeze(
        args.trace, args.config, stride=args.stride,
        nested_budgets=budgets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "stage": result["stage"],
        "frames": result["sampling"]["frames"],
        "candidate_payload_sha256": result["candidate_payload_sha256"],
        "candidate_freeze_utc": result["candidate_freeze_utc"],
        "candidate_count_min": min(row["candidate_count"] for row in result["rows"]),
        "candidate_count_median": float(np.median([
            row["candidate_count"] for row in result["rows"]])),
        "candidate_count_max": max(row["candidate_count"] for row in result["rows"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
