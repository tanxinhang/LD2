#!/usr/bin/env python
"""Audit worst-k deficit dominance over N5 transition horizons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_n5_frozen_controller import _trace_rows
from uav_isac.evaluation.n5_counterfactual_audit import (
    cumulative_detection_deficit,
)


def _variant_summary(
    variant: dict[str, object],
    trace,
    rows: dict[int, int],
    *,
    p_d_floor: float,
) -> dict[str, object]:
    frame_rows = []
    for record in variant["frame_rows"]:
        frame = int(record["frame"])
        baseline_pd = np.asarray(
            trace["physical_pd"][rows[frame]], dtype=np.float64)
        forced_pd = baseline_pd + np.asarray(
            record["delta_pd"], dtype=np.float64)
        baseline_curve = cumulative_detection_deficit(
            baseline_pd, p_d_floor=p_d_floor)
        forced_curve = cumulative_detection_deficit(
            forced_pd, p_d_floor=p_d_floor)
        violation = np.maximum(forced_curve - baseline_curve, 0.0)
        frame_rows.append({
            "frame": frame,
            "delta_worst": float(record["delta_worst"]),
            "dominates_no_op": bool(np.all(violation <= 1.0e-9)),
            "maximum_cumulative_deficit_violation": float(
                np.max(violation, initial=0.0)),
            "baseline_deficit_curve": baseline_curve.tolist(),
            "forced_deficit_curve": forced_curve.tolist(),
        })
    immediate = frame_rows[0]
    future = frame_rows[1:]
    return {
        "immediate_dominates": bool(immediate["dominates_no_op"]),
        "future_frame_count": int(len(future)),
        "future_all_frames_dominate": bool(all(
            row["dominates_no_op"] for row in future)),
        "future_dominance_rate": float(np.mean([
            row["dominates_no_op"] for row in future
        ])) if future else 1.0,
        "maximum_horizon_cumulative_deficit_violation": float(max([
            row["maximum_cumulative_deficit_violation"]
            for row in frame_rows
        ] or [0.0])),
        "minimum_horizon_delta_worst": float(min([
            row["delta_worst"] for row in frame_rows
        ] or [0.0])),
        "endpoint_delta_worst": float(frame_rows[-1]["delta_worst"]),
        "frame_rows": frame_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--b2-replay", type=Path, required=True)
    parser.add_argument("--bq-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--p-d-floor", type=float, default=0.60)
    args = parser.parse_args()

    b2 = json.loads(args.b2_replay.read_text(encoding="utf-8"))
    bq = json.loads(args.bq_replay.read_text(encoding="utf-8"))
    if int(b2["seed"]) != int(bq["seed"]):
        raise ValueError("B=2 and B=Q replays must use the same seed")
    seed = int(b2["seed"])
    with np.load(args.trace, allow_pickle=False) as trace:
        rows = _trace_rows(trace, seed)
        variants = {
            "b2_frozen_noop_path": _variant_summary(
                b2["paired"], trace, rows,
                p_d_floor=float(args.p_d_floor)),
            "b2_recorded_closed_loop": _variant_summary(
                b2["mediation_variants"]["all_recorded_feedback"],
                trace,
                rows,
                p_d_floor=float(args.p_d_floor),
            ),
            "bq_frozen_noop_path": _variant_summary(
                bq["paired"], trace, rows,
                p_d_floor=float(args.p_d_floor)),
        }

    report = {
        "protocol": "gate_d0_6_transition_tail_deficit_dominance_v1",
        "scope": (
            "seed-483 development-event path-specific H=5 audit; B=2 has a "
            "matched recorded intervention feedback path, while B=Q is valid "
            "only under the frozen no-op controller path"
        ),
        "seed": seed,
        "event_frame": int(b2["event_frame"]),
        "p_d_floor": float(args.p_d_floor),
        "safety_contract": (
            "at every transition frame, every worst-k cumulative positive "
            "detection deficit must not exceed the paired no-op path"
        ),
        "variants": variants,
        "decision": {
            "same_frame_tail_certificate_is_sufficient": "reject",
            "require_candidate_dependent_transition_certificate": True,
            "reason": (
                "the same-frame B=2 move satisfies tail-deficit dominance, "
                "but its matched recorded closed-loop feedback path violates "
                "the horizon contract"
            ),
            "next": (
                "calibrate a joint event-level lower bound on the maximum "
                "worst-k deficit violation over candidates, targets/tails, "
                "and h=0..H; fail closed to no-op before deployment"
            ),
        },
        "limitations": [
            "This is one path-specific development event, not a population failure rate.",
            "The B=Q recorded-feedback variant is excluded because the available feedback trace was generated by the B=2 intervention.",
            "Frozen-path results are causal diagnostics, not a deployable closed-loop controller.",
        ],
        "fresh_test_consumed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
