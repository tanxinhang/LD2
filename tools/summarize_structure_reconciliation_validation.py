"""Summarize the preregistered D0.22 structure-reconciliation validation.

The exchangeability unit is an episode seed.  Event and target-step counts are
reported as diagnostics only; they are never treated as independent trials.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_paired_horizon_confirmation import _sha256, _wilson


def _single_seed_split(manifest: dict[str, object]) -> list[int]:
    splits = list(dict(manifest.get("splits", {})).values())
    if len(splits) != 1:
        raise ValueError("D0.22 seed manifest must contain exactly one split")
    return [int(seed) for seed in splits[0]]


def summarize(
    audit_path: Path,
    seed_manifest_path: Path,
    trace_path: Path,
) -> dict[str, object]:
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    manifest = json.loads(
        Path(seed_manifest_path).read_text(encoding="utf-8"))
    seeds = _single_seed_split(manifest)
    if [int(seed) for seed in audit.get("seed_order", [])] != seeds:
        raise ValueError("audit seed order differs from preregistration")

    design = dict(audit.get("horizon_diagnostic", {}))
    expected = {
        "steps": 3,
        "proposal_ranking": "distributed_screened_margin",
        "transition_gate": "common_box_paired_deflection",
        "power_plan": "common_robust_weights",
        "weak_target_count": 3,
        "local_shortlist_per_owner": 1,
        "concurrent_owner_workers": 4,
        "deadline_accounting": "complete_ranking_wall_plus_transport",
    }
    mismatches = {
        key: {"expected": value, "observed": design.get(key)}
        for key, value in expected.items()
        if design.get(key) != value
    }
    if mismatches:
        raise ValueError(f"D0.22 frozen design mismatch: {mismatches}")

    shadow = dict(audit.get("shadow_router", {}))
    if not bool(shadow.get(
        "structure_consensus_horizon_power_enabled", False
    )):
        raise ValueError("structure-consensus reconciliation was not enabled")
    if bool(shadow.get("commit_authority", True)):
        raise ValueError("D0.22 shadow route unexpectedly had authority")

    rows = [
        row for row in audit.get("rows", [])
        if bool(row.get("horizon_diagnostic_evaluated", False))
    ]
    if not rows:
        raise ValueError("D0.22 audit has no evaluated horizon event")
    structure_matches = [
        row for row in rows
        if row.get("shadow_router_structure_identity_match") is True
    ]
    joint_accepts = [
        row for row in rows
        if bool(row.get("structure_accepted", False))
        and bool(row.get("horizon_diagnostic_accept", False))
    ]
    eligible = [
        row for row in joint_accepts
        if row.get("shadow_router_structure_identity_match") is True
    ]

    tolerance = 1.0e-12
    target_deltas: list[float] = []
    for row in eligible:
        if not bool(row.get("horizon_future_outcome_valid", False)):
            raise ValueError("eligible event lacks a true-future outcome")
        candidate = np.asarray(
            row["horizon_future_realized_candidate_pd"], dtype=np.float64)
        noop = np.asarray(
            row["horizon_future_realized_noop_pd"], dtype=np.float64)
        target_deltas.extend((candidate - noop).reshape(-1).tolist())

    eligible_seed_set = {int(row["seed"]) for row in eligible}
    structure_seed_set = {int(row["seed"]) for row in structure_matches}
    joint_seed_set = {int(row["seed"]) for row in joint_accepts}
    eligible_fail_seeds = {
        int(row["seed"])
        for row in eligible
        if (
            float(row["horizon_diagnostic_end_to_end_latency_s"])
            > 0.1 + tolerance
            or row.get("horizon_ranking_comm_bound_valid") is False
            or bool(row.get("horizon_future_candidate_bound_failure", False))
            or bool(row.get("horizon_future_noop_bound_failure", False))
            or row.get("horizon_future_realized_target_no_harm") is False
        )
    }
    target_failure_count = int(np.sum(
        np.asarray(target_deltas, dtype=np.float64) < -tolerance
    )) if target_deltas else 0
    occurrence_count = len(eligible_seed_set)
    resource_complete = bool(
        shadow.get("resource_accounting_complete", False))

    # This is deliberately descriptive.  No minimum occurrence rate or
    # non-inferiority threshold was preregistered for D0.22, so observing a
    # nonempty rule must not be retroactively promoted to a formal pass.
    status = (
        "validation_observed_safe_structure_consensus_shadow_only"
        if eligible and not eligible_fail_seeds and target_failure_count == 0
        else (
            "validation_no_structure_consensus"
            if not eligible else "validation_structure_consensus_unsafe"
        )
    )

    return {
        "schema_version": 1,
        "status": status,
        "frozen_design": {
            **expected,
            "structure_consensus_horizon_power_enabled": True,
            "commit_authority": False,
        },
        "independent_validation": {
            "episode_count": len(seeds),
            "evaluated_event_count": len(rows),
            "discrete_structure_match_event_count": len(structure_matches),
            "discrete_structure_match_episode_count": len(structure_seed_set),
            "joint_accept_event_count": len(joint_accepts),
            "joint_accept_episode_count": len(joint_seed_set),
            "reconciliation_eligible_event_count": len(eligible),
            "reconciliation_eligible_episode_count": occurrence_count,
            "reconciliation_eligible_seeds": sorted(eligible_seed_set),
            "episode_occurrence_rate_wilson95": _wilson(
                occurrence_count, len(seeds)),
            "full_action_identity_match_count": int(sum(
                row.get("shadow_router_candidate_identity_match") is True
                for row in rows
            )),
        },
        "eligible_safety": {
            "eligible_complete_latency_s": [
                float(row["horizon_diagnostic_end_to_end_latency_s"])
                for row in eligible
            ],
            "deadline_failure_event_count": int(sum(
                float(row["horizon_diagnostic_end_to_end_latency_s"])
                > 0.1 + tolerance
                for row in eligible
            )),
            "communication_failure_event_count": int(sum(
                row.get("horizon_ranking_comm_bound_valid") is False
                for row in eligible
            )),
            "coefficient_or_pd_bound_failure_event_count": int(sum(
                bool(row.get("horizon_future_candidate_bound_failure", False))
                or bool(row.get("horizon_future_noop_bound_failure", False))
                for row in eligible
            )),
            "target_step_count": len(target_deltas),
            "target_no_harm_failure_count": target_failure_count,
            "minimum_realized_target_pd_delta": (
                float(min(target_deltas)) if target_deltas else None),
            "episode_any_failure_count": len(eligible_fail_seeds),
            "episode_any_failure_rate_wilson95": (
                _wilson(len(eligible_fail_seeds), occurrence_count)
                if occurrence_count else None
            ),
        },
        "authority": {
            "shadow_selected_candidate_count": int(sum(
                row.get("shadow_router_action") in {
                    "consensus_candidate", "horizon_power_candidate",
                }
                for row in rows
            )),
            "resource_accounting_complete": resource_complete,
            "commit_authority": False,
        },
        "interpretation": (
            "The reconciliation rule recurred on fresh episodes and retained "
            "the horizon action's hard physical, communication, and deadline "
            "checks.  Its occurrence evidence is only episode-level and no "
            "minimum coverage threshold was preregistered, so this is a "
            "safe nonempty shadow observation rather than a deployment pass."
        ),
        "system_certificate_ready": False,
        "remaining_blockers": [
            "only a small number of independent episodes contain an eligible reconciliation",
            "common preprocessing and power-only compute are not fully metered",
            "CPU wattage is not hardware calibrated",
            "resource-incomplete shadow routing deterministically selects No-op",
            "deployment scheduling jitter is not measured on distributed UAV hardware",
        ],
        "provenance": {
            "audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "seed_manifest": {
                "path": str(seed_manifest_path),
                "sha256": _sha256(seed_manifest_path),
            },
            "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.audit, args.seed_manifest, args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "independent_validation": result["independent_validation"],
        "eligible_safety": result["eligible_safety"],
        "authority": result["authority"],
    }, indent=2))


if __name__ == "__main__":
    main()
