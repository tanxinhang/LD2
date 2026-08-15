#!/usr/bin/env python
"""Summarize Gate D0.4 nested weak-target budget headroom."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--frozen-b2", type=Path, required=True)
    parser.add_argument("--frozen-b6", type=Path, required=True)
    parser.add_argument("--seed291-control", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    event_files = sorted(args.event_dir.glob("seed*_frame*.json"))
    event_files = [path for path in event_files if "frozen" not in path.name]
    if len(event_files) != 3:
        raise ValueError(f"expected three crisis-event audits, found {len(event_files)}")

    events = [json.loads(path.read_text(encoding="utf-8")) for path in event_files]
    budgets = sorted(int(key) for key in events[0]["event"][
        "weak_target_budget_summaries"].keys())
    rows = []
    for payload in events:
        event = payload["event"]
        for budget in budgets:
            item = event["weak_target_budget_summaries"][str(budget)]
            unconstrained = item["unconstrained_physical_oracle"]
            safe = item["no_op_safe_physical_oracle"]
            rows.append({
                "seed": int(event["episode_seed"]),
                "frame": int(event["frame"]),
                "budget": int(budget),
                "candidate_count": int(item["candidate_count"]),
                "physical_positive_count": int(item["physical_positive_count"]),
                "proxy_positive_count": int(item["proxy_positive_count"]),
                "unconstrained_delta_worst": float(unconstrained["delta_worst"]),
                "unconstrained_delta_weak3": float(unconstrained["delta_weak3"]),
                "unconstrained_delta_steady": float(unconstrained["delta_steady"]),
                "safe_choice": str(safe["choice"]),
                "safe_delta_worst": float(safe["delta_worst"]),
                "safe_delta_weak3": float(safe["delta_weak3"]),
                "safe_delta_steady": float(safe["delta_steady"]),
            })

    by_budget: dict[str, object] = {}
    for budget in budgets:
        selected = [row for row in rows if row["budget"] == budget]
        by_budget[str(budget)] = {
            "event_count": int(len(selected)),
            "mean_candidate_count": float(np.mean([
                row["candidate_count"] for row in selected])),
            "total_candidate_count": int(sum(
                row["candidate_count"] for row in selected)),
            "safe_positive_event_count": int(sum(
                row["safe_choice"] == "candidate" for row in selected)),
            "mean_no_op_safe_delta_worst": float(np.mean([
                row["safe_delta_worst"] for row in selected])),
            "mean_unconstrained_delta_worst": float(np.mean([
                row["unconstrained_delta_worst"] for row in selected])),
            "total_physical_positive_candidates": int(sum(
                row["physical_positive_count"] for row in selected)),
        }

    frozen_b2 = json.loads(args.frozen_b2.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_b6.read_text(encoding="utf-8"))
    if (
        int(frozen_b2["seed"]) != int(frozen["seed"])
        or int(frozen_b2["event_frame"]) != int(frozen["event_frame"])
    ):
        raise ValueError("B=2 and B=Q frozen replays must match seed/frame")
    frozen_summary = frozen["paired"]
    seed291_payload = json.loads(
        args.seed291_control.read_text(encoding="utf-8"))
    seed291 = seed291_payload["events"][0]
    b2_frozen_future_mean = float(
        frozen_b2["paired"]["future"]["mean_delta_worst"])
    b6_frozen_future_mean = float(
        frozen_summary["future"]["mean_delta_worst"])
    b2_total = by_budget["2"]["total_candidate_count"]
    bq_total = by_budget[str(max(budgets))]["total_candidate_count"]

    report = {
        "protocol": "gate_d0_4_nested_weak_target_budget_v1",
        "scope": (
            "three development crisis events with exact same-state CRN full-pool audits, "
            "plus the previously audited seed-291 no-op control"
        ),
        "mathematical_contract": {
            "candidate_budget": (
                "B restricts singleton/two-target N5 blocks to the B weakest public-proxy "
                "targets; candidate pools are nested and B=Q is diagnostic only"
            ),
            "no_op_safety": (
                "accept only a physical-worst improvement that preserves the baseline "
                "when steady/weak3 are below their floor, or remains above the floor "
                "when it is already satisfied"
            ),
            "hard_physics": (
                "every replay uses the delivered candidate graph, exact bistatic detection, "
                "common random numbers, and the per-UAV 1 W communication+sensing simplex"
            ),
        },
        "budget_aggregate": by_budget,
        "event_rows": rows,
        "seed291_no_op_control": {
            "seed": 291,
            "frame": 95,
            "all_target_candidate_count": int(seed291["candidate_count_all"]),
            "physical_positive_candidate_count": int(sum(
                float(record["delta_worst"]) > 1.0e-4
                for record in seed291["candidates"]
                if record["is_atomic_candidate"]
            )),
            "no_op_delta_worst": 0.0,
        },
        "seed483_persistence": {
            "b2_immediate_delta_worst": 0.08206544669830435,
            "bq_immediate_delta_worst": float(
                frozen_summary["immediate"]["delta_worst"]),
            "b2_frozen_future5_mean_delta_worst": b2_frozen_future_mean,
            "bq_frozen_future5_mean_delta_worst": b6_frozen_future_mean,
            "incremental_frozen_future5_mean_headroom": float(
                b6_frozen_future_mean - b2_frozen_future_mean),
            "bq_frozen_endpoint_delta_worst": float(
                frozen_summary["future"]["endpoint_delta_worst"]),
            "note": (
                "the large same-frame B=Q gain is mostly transient and disappears at the "
                "next ordinary structural resolve; feedback variants are excluded because "
                "the available feedback trace was generated by the B=2 intervention"
            ),
        },
        "complexity": {
            "b2_total_candidates_three_events": int(b2_total),
            "bq_total_candidates_three_events": int(bq_total),
            "bq_over_b2_candidate_ratio": float(bq_total / max(b2_total, 1)),
        },
        "decision": {
            "uniform_budget_expansion": "reject",
            "reason": (
                "B=Q costs about 7.5x more candidates, does not create a safe action in "
                "two of three crisis events, and adds only about 0.0028 frozen-future mean "
                "worst beyond B=2 in the sole headroom event"
            ),
            "retain": (
                "nested budgets as an adaptive diagnostic/search mechanism, always with "
                "an explicit no-op and a calibrated QoS safety certificate"
            ),
            "next": (
                "collect owner-local certificate residuals and test a confidence-constrained "
                "event-triggered LNS before changing the deployed coordinator"
            ),
        },
        "test_split_protocol": (
            "development/selection traces only; no additional final-test seed consumed"
        ),
        "fresh_test_consumed": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "event_budget_rows.csv").open(
        "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
