"""Summarize development-only structure-consensus reconciliation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(audit_path: Path) -> dict[str, object]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    shadow = audit.get("shadow_router", {})
    if not bool(shadow.get(
        "structure_consensus_horizon_power_enabled", False
    )):
        raise ValueError("structure-consensus reconciliation was not enabled")
    if bool(shadow.get("commit_authority", True)):
        raise ValueError("development reconciliation unexpectedly had authority")

    rows = [
        row for row in audit.get("rows", [])
        if bool(row.get("horizon_diagnostic_evaluated", False))
    ]
    eligible = [
        row for row in rows
        if bool(row.get("structure_accepted", False))
        and bool(row.get("horizon_diagnostic_accept", False))
        and row.get("shadow_router_structure_identity_match") is True
    ]
    if not rows or not eligible:
        raise ValueError("development audit has no evaluated/eligible event")
    tolerance = 1.0e-12
    target_deltas: list[float] = []
    for row in eligible:
        candidate = np.asarray(
            row["horizon_future_realized_candidate_pd"], dtype=np.float64)
        noop = np.asarray(
            row["horizon_future_realized_noop_pd"], dtype=np.float64)
        target_deltas.extend((candidate - noop).reshape(-1).tolist())
    late = int(sum(
        float(row["horizon_diagnostic_end_to_end_latency_s"])
        > 0.1 + tolerance
        for row in eligible
    ))
    harm = int(np.sum(np.asarray(target_deltas) < -tolerance))
    communication_failures = int(sum(
        row.get("horizon_ranking_comm_bound_valid") is False
        for row in eligible
    ))
    full_action_matches = int(sum(
        row.get("shadow_router_candidate_identity_match") is True
        for row in eligible
    ))
    eligible_seeds = sorted({int(row["seed"]) for row in eligible})

    return {
        "schema_version": 1,
        "status": "development_rule_nonempty_not_independently_validated",
        "evaluated_event_count": len(rows),
        "discrete_structure_match_count": int(sum(
            row.get("shadow_router_structure_identity_match") is True
            for row in rows
        )),
        "joint_accept_structure_match_count": len(eligible),
        "joint_accept_structure_match_episode_count": len(eligible_seeds),
        "joint_accept_structure_match_seeds": eligible_seeds,
        "full_action_match_count": full_action_matches,
        "eligible_latency_s": [
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            for row in eligible
        ],
        "eligible_deadline_failure_count": late,
        "eligible_communication_failure_count": communication_failures,
        "eligible_target_step_count": len(target_deltas),
        "eligible_target_no_harm_failure_count": harm,
        "eligible_min_realized_target_pd_delta": float(min(target_deltas)),
        "shadow_selected_candidate_count": int(sum(
            row.get("shadow_router_action") in {
                "consensus_candidate", "horizon_power_candidate",
            }
            for row in rows
        )),
        "resource_accounting_complete": bool(
            shadow.get("resource_accounting_complete", False)),
        "commit_authority": False,
        "interpretation": (
            "The rule is nonempty and its horizon action retains the existing "
            "physical/communication/deadline certificate, but all eligible "
            "events come from too few independent episodes and incomplete "
            "resource accounting still forces No-op."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
