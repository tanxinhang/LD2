"""Summarize the development-only D0.23 set-consensus oracle diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(audit_path: Path) -> dict[str, object]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    rows = [
        row for row in audit.get("rows", [])
        if int(row.get("horizon_raw_candidate_oracle_candidate_count", 0)) > 0
    ]
    if not rows:
        raise ValueError("audit contains no raw-candidate oracle events")
    owner_safe = [
        row for row in rows
        if bool(row.get(
            "horizon_owner_proposal_oracle_structure_match_any_accept", False))
    ]
    raw_safe = [
        row for row in rows
        if bool(row.get(
            "horizon_raw_candidate_oracle_structure_match_any_accept", False))
    ]
    current_eligible = [
        row for row in rows
        if bool(row.get("structure_accepted", False))
        and bool(row.get("horizon_diagnostic_accept", False))
        and row.get("shadow_router_structure_identity_match") is True
    ]
    current_keys = {
        (int(row["seed"]), int(row["frame"])) for row in current_eligible
    }
    additional = [
        row for row in raw_safe
        if (int(row["seed"]), int(row["frame"])) not in current_keys
    ]
    return {
        "schema_version": 1,
        "status": "development_oracle_promising_not_resource_certified",
        "evaluated_event_count": len(rows),
        "owner_shortlist_safe_intersection_event_count": len(owner_safe),
        "owner_shortlist_safe_intersection_episode_count": len({
            int(row["seed"]) for row in owner_safe
        }),
        "raw_candidate_safe_intersection_event_count": len(raw_safe),
        "raw_candidate_safe_intersection_episode_count": len({
            int(row["seed"]) for row in raw_safe
        }),
        "current_reconciliation_event_count": len(current_eligible),
        "additional_raw_intersection_event_count": len(additional),
        "additional_raw_intersection_episode_count": len({
            int(row["seed"]) for row in additional
        }),
        "additional_raw_intersection_seeds": sorted({
            int(row["seed"]) for row in additional
        }),
        "maximum_raw_candidate_count": max(
            int(row["horizon_raw_candidate_oracle_candidate_count"])
            for row in rows
        ),
        "resource_certified": False,
        "commit_authority": False,
        "interpretation": (
            "The raw causal candidate set contains additional safe common "
            "structures, but broadcasting all raw candidates is not an "
            "admissible implementation.  Test a bounded digest rendezvous "
            "with full suffix compute/transport/energy accounting next."
        ),
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
