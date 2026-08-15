#!/usr/bin/env python
"""Summarize Gate D0.5 locality and target-wise certificate alignment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _event_files(directory: Path, suffix: str) -> list[Path]:
    paths = sorted(directory.glob(f"seed*_frame*{suffix}.json"))
    return [path for path in paths if "frozen" not in path.name]


def _records_for_budget(event: dict[str, object], budget: int) -> list[dict]:
    key = str(int(budget))
    return [
        record for record in event["candidates"]
        if bool(record["is_atomic_candidate"])
        and bool(record["in_weak_target_budget"].get(key, False))
    ]


def _targetwise_oracle(
    event: dict[str, object],
    budget: int,
    p_d_floor: float,
) -> dict[str, object]:
    baseline = np.asarray(event["baseline_pd"], dtype=np.float64)
    protected = np.minimum(baseline, float(p_d_floor))
    admissible = [
        record for record in _records_for_budget(event, budget)
        if float(record["delta_worst"]) > 1.0e-4
        and np.all(
            np.asarray(record["pd"], dtype=np.float64) + 1.0e-9
            >= protected
        )
    ]
    chosen = max(
        admissible,
        key=lambda record: (
            record["worst"], record["weak3"], record["steady"]),
        default=None,
    )
    return {
        "admissible_positive_count": int(len(admissible)),
        "choice": "candidate" if chosen is not None else "no_op",
        "candidate_index": (
            int(chosen["candidate_index"]) if chosen is not None else None),
        "delta_worst": (
            float(chosen["delta_worst"]) if chosen is not None else 0.0),
        "delta_weak3": (
            float(chosen["delta_weak3"]) if chosen is not None else 0.0),
        "delta_steady": (
            float(chosen["delta_steady"]) if chosen is not None else 0.0),
    }


def _deficit_curve(pd: object, p_d_floor: float) -> np.ndarray:
    values = np.asarray(pd, dtype=np.float64)
    deficit = np.maximum(float(p_d_floor) - values, 0.0)
    return np.cumsum(np.sort(deficit)[::-1])


def _tail_deficit_oracle(
    event: dict[str, object],
    budget: int,
    p_d_floor: float,
) -> dict[str, object]:
    baseline_curve = _deficit_curve(event["baseline_pd"], p_d_floor)
    admissible = []
    for record in _records_for_budget(event, budget):
        if float(record["delta_worst"]) <= 1.0e-4:
            continue
        candidate_curve = _deficit_curve(record["pd"], p_d_floor)
        if np.all(candidate_curve <= baseline_curve + 1.0e-9):
            admissible.append(record)
    chosen = max(
        admissible,
        key=lambda record: (
            record["worst"], record["weak3"], record["steady"]),
        default=None,
    )
    return {
        "admissible_positive_count": int(len(admissible)),
        "choice": "candidate" if chosen is not None else "no_op",
        "candidate_index": (
            int(chosen["candidate_index"]) if chosen is not None else None),
        "delta_worst": (
            float(chosen["delta_worst"]) if chosen is not None else 0.0),
        "delta_weak3": (
            float(chosen["delta_weak3"]) if chosen is not None else 0.0),
        "delta_steady": (
            float(chosen["delta_steady"]) if chosen is not None else 0.0),
    }


def _affected_count(record: dict[str, object]) -> int:
    if "changed_target_count" in record:
        return int(record["changed_target_count"])
    return int(len(record["changed_target_ids"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-event-dir", type=Path, required=True)
    parser.add_argument("--strict-event-dir", type=Path, required=True)
    parser.add_argument("--n6-event-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p-d-floor", type=float, default=0.60)
    args = parser.parse_args()

    global_paths = _event_files(args.global_event_dir, "")
    strict_paths = _event_files(args.strict_event_dir, "_target_block")
    n6_paths = _event_files(args.n6_event_dir, "_n6")
    if (len(global_paths) != 3 or len(strict_paths) != 3
            or len(n6_paths) != 3):
        raise ValueError(
            "Gate D0.5 expects three global, target-block, and N6 events; "
            f"found {len(global_paths)}, {len(strict_paths)}, and {len(n6_paths)}"
        )
    global_events = {
        int(payload["event"]["episode_seed"]): payload["event"]
        for payload in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in global_paths
        )
    }
    strict_events = {
        int(payload["event"]["episode_seed"]): payload["event"]
        for payload in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in strict_paths
        )
    }
    n6_events = {
        int(payload["event"]["episode_seed"]): payload["event"]
        for payload in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in n6_paths
        )
    }
    if not (global_events.keys() == strict_events.keys() == n6_events.keys()):
        raise ValueError("global, target-block, and N6 event seeds do not match")
    budgets = sorted(int(key) for key in next(iter(global_events.values()))[
        "weak_target_budget_summaries"].keys())

    rows: list[dict[str, object]] = []
    for seed in sorted(global_events):
        global_event = global_events[seed]
        strict_event = strict_events[seed]
        n6_event = n6_events[seed]
        for budget in budgets:
            global_budget = global_event[
                "weak_target_budget_summaries"][str(budget)]
            strict_budget = strict_event[
                "weak_target_budget_summaries"][str(budget)]
            n6_budget = n6_event[
                "weak_target_budget_summaries"][str(budget)]
            global_vector = _targetwise_oracle(
                global_event, budget, float(args.p_d_floor))
            global_tail = _tail_deficit_oracle(
                global_event, budget, float(args.p_d_floor))
            strict_tail = _tail_deficit_oracle(
                strict_event, budget, float(args.p_d_floor))
            n6_tail = _tail_deficit_oracle(
                n6_event, budget, float(args.p_d_floor))
            strict_vector = strict_budget[
                "no_op_vector_safe_physical_oracle"]
            n6_vector = n6_budget[
                "no_op_vector_safe_physical_oracle"]
            rows.append({
                "seed": int(seed),
                "frame": int(global_event["frame"]),
                "budget": int(budget),
                "global_candidate_count": int(
                    global_budget["candidate_count"]),
                "global_physical_positive_count": int(
                    global_budget["physical_positive_count"]),
                "global_vector_safe_positive_count": int(
                    global_vector["admissible_positive_count"]),
                "global_vector_safe_delta_worst": float(
                    global_vector["delta_worst"]),
                "global_tail_safe_positive_count": int(
                    global_tail["admissible_positive_count"]),
                "global_tail_safe_delta_worst": float(
                    global_tail["delta_worst"]),
                "strict_candidate_count": int(
                    strict_budget["candidate_count"]),
                "strict_physical_positive_count": int(
                    strict_budget["physical_positive_count"]),
                "strict_vector_safe_positive_count": int(
                    strict_vector["admissible_positive_count"]),
                "strict_vector_safe_delta_worst": float(
                    strict_vector["delta_worst"]),
                "strict_tail_safe_positive_count": int(
                    strict_tail["admissible_positive_count"]),
                "strict_tail_safe_delta_worst": float(
                    strict_tail["delta_worst"]),
                "n6_candidate_count": int(n6_budget["candidate_count"]),
                "n6_physical_positive_count": int(
                    n6_budget["physical_positive_count"]),
                "n6_vector_safe_positive_count": int(
                    n6_vector["admissible_positive_count"]),
                "n6_vector_safe_delta_worst": float(
                    n6_vector["delta_worst"]),
                "n6_tail_safe_positive_count": int(
                    n6_tail["admissible_positive_count"]),
                "n6_tail_safe_delta_worst": float(
                    n6_tail["delta_worst"]),
            })

    def locality(events: dict[int, dict[str, object]]) -> dict[str, object]:
        counts = [
            _affected_count(record)
            for event in events.values()
            for record in event["candidates"]
            if bool(record["is_atomic_candidate"])
        ]
        return {
            "candidate_count": int(len(counts)),
            "mean_affected_targets": float(np.mean(counts)) if counts else 0.0,
            "maximum_affected_targets": int(max(counts, default=0)),
            "over_two_target_count": int(sum(value > 2 for value in counts)),
            "over_two_target_rate": float(np.mean(
                np.asarray(counts) > 2)) if counts else 0.0,
        }

    by_budget: dict[str, object] = {}
    for budget in budgets:
        selected = [row for row in rows if row["budget"] == budget]
        by_budget[str(budget)] = {
            "event_count": int(len(selected)),
            "global_candidate_count": int(sum(
                row["global_candidate_count"] for row in selected)),
            "strict_candidate_count": int(sum(
                row["strict_candidate_count"] for row in selected)),
            "n6_candidate_count": int(sum(
                row["n6_candidate_count"] for row in selected)),
            "strict_over_global_candidate_fraction": float(
                sum(row["strict_candidate_count"] for row in selected)
                / max(sum(
                    row["global_candidate_count"] for row in selected), 1)),
            "global_vector_safe_event_count": int(sum(
                row["global_vector_safe_positive_count"] > 0
                for row in selected)),
            "strict_vector_safe_event_count": int(sum(
                row["strict_vector_safe_positive_count"] > 0
                for row in selected)),
            "n6_vector_safe_event_count": int(sum(
                row["n6_vector_safe_positive_count"] > 0
                for row in selected)),
            "global_tail_safe_event_count": int(sum(
                row["global_tail_safe_positive_count"] > 0
                for row in selected)),
            "strict_tail_safe_event_count": int(sum(
                row["strict_tail_safe_positive_count"] > 0
                for row in selected)),
            "n6_tail_safe_event_count": int(sum(
                row["n6_tail_safe_positive_count"] > 0
                for row in selected)),
            "global_mean_vector_safe_delta_worst": float(np.mean([
                row["global_vector_safe_delta_worst"] for row in selected
            ])),
            "strict_mean_vector_safe_delta_worst": float(np.mean([
                row["strict_vector_safe_delta_worst"] for row in selected
            ])),
            "n6_mean_vector_safe_delta_worst": float(np.mean([
                row["n6_vector_safe_delta_worst"] for row in selected
            ])),
            "global_mean_tail_safe_delta_worst": float(np.mean([
                row["global_tail_safe_delta_worst"] for row in selected
            ])),
            "strict_mean_tail_safe_delta_worst": float(np.mean([
                row["strict_tail_safe_delta_worst"] for row in selected
            ])),
            "n6_mean_tail_safe_delta_worst": float(np.mean([
                row["n6_tail_safe_delta_worst"] for row in selected
            ])),
        }

    report = {
        "protocol": "gate_d0_5_n5_locality_certificate_alignment_v1",
        "scope": (
            "three development crisis events; same-state exact-physical CRN "
            "comparison of historical global-rebuild N5 and dependency-closed "
            "one/two-target N5/N6"
        ),
        "p_d_floor": float(args.p_d_floor),
        "mathematical_contract": {
            "targetwise_no_op_safety": (
                "P_D_q(S') >= min(P_D_q(S), P_D_floor) for every q and "
                "Delta physical worst > 1e-4"
            ),
            "strict_locality": (
                "all selected edges outside the declared one/two-target block "
                "are frozen; incompatible role/owner assignments are rejected"
            ),
            "role_preserving_locality": (
                "N6 keeps the role partition fixed and rebuilds only the "
                "declared one/two-target owner/support block"
            ),
            "tail_deficit_dominance": (
                "sort [P_D_floor-P_D_q]_+ from largest to smallest and "
                "require every worst-k cumulative deficit to be no greater "
                "than no-op; for equal targets this protects worst deficit, "
                "all discrete deficit CVaRs, and total deficit"
            ),
            "oracle_scope": (
                "exact physical P_D is an offline headroom ceiling only and "
                "is unavailable to the deployed coordinator"
            ),
        },
        "locality": {
            "historical_global_rebuild": locality(global_events),
            "strict_target_block": locality(strict_events),
            "role_preserving_n6": locality(n6_events),
        },
        "budget_aggregate": by_budget,
        "event_budget_rows": rows,
        "decision": {
            "claim_two_target_atomicity_for_historical_n5": "reject",
            "train_targetwise_certificate_for_strict_n5_now": "reject",
            "promote_role_preserving_n6": "reject",
            "retain_tail_deficit_dominance_for_role_closure": "pilot",
            "reason": (
                "historical N5 obtains its headroom through a multi-target "
                "role-change dependency closure, while both strict target-block "
                "N5 and role-preserving N6 have neither targetwise-safe nor "
                "tail-deficit-safe positive actions in these events"
            ),
            "next": (
                "treat role-changing N5 as a rare multi-target dependency "
                "closure; audit tail-deficit dominance over candidate-dependent "
                "short-horizon transitions before any uncertainty calibration"
            ),
        },
        "limitations": [
            "Three development events establish a mechanism-level No-Go, not a population failure rate.",
            "This gate is same-frame; candidate-dependent H-step controller feedback remains a separate requirement.",
            "Candidate counts are not runtime, message, bit, latency, or energy measurements.",
        ],
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
