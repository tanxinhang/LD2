"""Compare two strict-pilot acceleration golden traces.

Structure/protocol fields must match exactly. Numerical fields are checked
with explicit absolute and relative tolerances and reported independently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FRAME_EXACT_FIELDS = (
    "frame",
    "hold_active",
    "received_last_seen",
    "visible_views",
    "mutual",
    "stable",
    "selected",
    "public_full_views",
    "consensus_streak",
    "local_cache_valid",
)
FRAME_NUMERIC_FIELDS = (
    "public_gain_views",
    "certificate_gain_views",
    "certificate_gain_upper_views",
    "executed_power_w",
    "local_power_cache_w",
    "local_prices",
)
TRACE_NUMERIC_FIELDS = (
    "detection",
    "detection_deflection",
    "sensing_power_w",
    "certificate_safe_gain_per_watt",
)


def _numeric_error(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if expected.shape != actual.shape:
        return {
            "shape_match": False,
            "reference_shape": list(expected.shape),
            "candidate_shape": list(actual.shape),
            "max_abs": float("inf"),
            "max_rel": float("inf"),
            "within_tolerance": False,
        }
    if expected.size == 0:
        return {
            "shape_match": True,
            "reference_shape": list(expected.shape),
            "candidate_shape": list(actual.shape),
            "max_abs": 0.0,
            "max_rel": 0.0,
            "within_tolerance": True,
        }
    finite_match = np.array_equal(
        np.isfinite(expected), np.isfinite(actual))
    if not finite_match:
        max_abs = max_rel = float("inf")
        within_tolerance = False
    else:
        finite = np.isfinite(expected)
        difference = np.abs(actual[finite] - expected[finite])
        max_abs = float(np.max(difference, initial=0.0))
        denominator = np.maximum(np.abs(expected[finite]), 1.0e-300)
        max_rel = float(np.max(
            difference / denominator, initial=0.0))
        within_tolerance = bool(np.all(
            difference <= float(atol) + float(rtol) * np.abs(expected[finite])
        ))
    return {
        "shape_match": True,
        "reference_shape": list(expected.shape),
        "candidate_shape": list(actual.shape),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "within_tolerance": within_tolerance,
    }


def _merge_error(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["shape_match"] = bool(
        target["shape_match"] and source["shape_match"])
    target["max_abs"] = max(float(target["max_abs"]), float(source["max_abs"]))
    target["max_rel"] = max(float(target["max_rel"]), float(source["max_rel"]))
    target["within_tolerance"] = bool(
        target["within_tolerance"] and source["within_tolerance"])


def compare_results(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    reference_episodes = {
        int(episode["seed"]): episode for episode in reference["episodes"]}
    candidate_episodes = {
        int(episode["seed"]): episode for episode in candidate["episodes"]}
    exact_mismatches: list[str] = []
    if set(reference_episodes) != set(candidate_episodes):
        exact_mismatches.append("episode seed sets differ")

    numerical = {
        field: {
            "shape_match": True,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "within_tolerance": True,
        }
        for field in (*TRACE_NUMERIC_FIELDS, *FRAME_NUMERIC_FIELDS,
                      "local_plan_proxy_target_value", "local_plan_proxy_scores")
    }
    compared_frames = 0
    for seed in sorted(set(reference_episodes) & set(candidate_episodes)):
        expected_trace = reference_episodes[seed].get("trace")
        actual_trace = candidate_episodes[seed].get("trace")
        if expected_trace is None or actual_trace is None:
            exact_mismatches.append(f"seed={seed}: missing trace")
            continue
        for field in TRACE_NUMERIC_FIELDS:
            _merge_error(numerical[field], _numeric_error(
                expected_trace.get(field, []), actual_trace.get(field, []),
                atol=atol, rtol=rtol))
        expected_frames = expected_trace.get("acceleration_golden", [])
        actual_frames = actual_trace.get("acceleration_golden", [])
        if len(expected_frames) != len(actual_frames):
            exact_mismatches.append(
                f"seed={seed}: acceleration frame counts differ")
        for frame_index, (expected, actual) in enumerate(zip(
            expected_frames, actual_frames
        )):
            compared_frames += 1
            prefix = f"seed={seed},frame={frame_index}"
            for field in FRAME_EXACT_FIELDS:
                if expected.get(field) != actual.get(field):
                    exact_mismatches.append(f"{prefix}: {field} differs")
            for field in FRAME_NUMERIC_FIELDS:
                _merge_error(numerical[field], _numeric_error(
                    expected.get(field, []), actual.get(field, []),
                    atol=atol, rtol=rtol))

            expected_plans = expected.get("local_plans", [])
            actual_plans = actual.get("local_plans", [])
            if len(expected_plans) != len(actual_plans):
                exact_mismatches.append(
                    f"{prefix}: local plan counts differ")
            for viewer, (expected_plan, actual_plan) in enumerate(zip(
                expected_plans, actual_plans
            )):
                plan_prefix = f"{prefix},viewer={viewer}"
                if expected_plan.get("selected") != actual_plan.get("selected"):
                    exact_mismatches.append(
                        f"{plan_prefix}: local selected differs")
                expected_scores = expected_plan.get("proxy_scores", [])
                actual_scores = actual_plan.get("proxy_scores", [])
                if [item[0] for item in expected_scores] != [
                    item[0] for item in actual_scores
                ]:
                    exact_mismatches.append(
                        f"{plan_prefix}: proxy score edge keys differ")
                _merge_error(
                    numerical["local_plan_proxy_target_value"],
                    _numeric_error(
                        expected_plan.get("proxy_target_value", []),
                        actual_plan.get("proxy_target_value", []),
                        atol=atol,
                        rtol=rtol,
                    ),
                )
                _merge_error(
                    numerical["local_plan_proxy_scores"],
                    _numeric_error(
                        [item[1] for item in expected_scores],
                        [item[1] for item in actual_scores],
                        atol=atol,
                        rtol=rtol,
                    ),
                )

    numerical_failures = []
    for field, error in numerical.items():
        if (
            not error["shape_match"] or not error["within_tolerance"]
        ):
            numerical_failures.append(field)
    return {
        "schema_version": "acceleration-golden-comparison/v1",
        "reference_seeds": sorted(reference_episodes),
        "candidate_seeds": sorted(candidate_episodes),
        "compared_frames": compared_frames,
        "atol": float(atol),
        "rtol": float(rtol),
        "exact_mismatch_count": len(exact_mismatches),
        "exact_mismatches": exact_mismatches[:100],
        "numerical": numerical,
        "numerical_failures": numerical_failures,
        "passed": not exact_mismatches and not numerical_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare strict-pilot acceleration golden traces")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    with Path(args.reference).open("r", encoding="utf-8") as handle:
        reference = json.load(handle)
    with Path(args.candidate).open("r", encoding="utf-8") as handle:
        candidate = json.load(handle)
    report = compare_results(
        reference, candidate, atol=args.atol, rtol=args.rtol)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
