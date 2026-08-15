#!/usr/bin/env python
"""Audit fixed-owner distributed power repair over all development events.

This audit is deliberately same-state and privileged: it measures whether the
continuous RF allocation layer is worth pursuing before causal estimation and
calibration are introduced.  It never claims deployable performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _entries,
    _ordered_unique,
    _recorded_power,
)
from uav_isac.coordination.maxmin_power import (  # noqa: E402
    distributed_column_generation_maxmin_power,
    fixed_owner_gain_matrix,
    solve_fixed_structure_maxmin_power_lp,
)
from uav_isac.coordination.geometry_gain_predictor import (  # noqa: E402
    predict_fixed_owner_gain_from_geometry,
)
from uav_isac.evaluation.physical_oracle_audit import (  # noqa: E402
    per_watt_deflection_tensor,
)
from uav_isac.physical.detection import (  # noqa: E402
    compute_detection_probabilities,
)


def _worst_pd(deflection: np.ndarray, p_fa: float) -> float:
    return float(np.min(compute_detection_probabilities(deflection, p_fa)))


def _episode_mean(rows: list[dict[str, float]], key: str) -> float:
    seeds = sorted({int(row["seed"]) for row in rows})
    return float(np.mean([
        np.mean([float(row[key]) for row in rows if int(row["seed"]) == seed])
        for seed in seeds
    ]))


def _split_conformal_upper(scores: list[float], alpha: float) -> float:
    """Finite-sample one-sided split-conformal order statistic."""
    values = np.sort(np.maximum(np.asarray(scores, dtype=np.float64), 0.0))
    if values.size < 1:
        raise ValueError("at least one calibration score is required")
    rank = int(np.ceil((values.size + 1) * (1.0 - float(alpha)))) - 1
    return float(values[min(max(rank, 0), values.size - 1)])


def _diagnostic_feedback_gate(
    rows: list[dict[str, object]],
    *,
    prefix: str,
    seed_order: list[int],
    qos_floor: float,
    alpha: float = 0.10,
) -> dict[str, object]:
    split = max(1, len(seed_order) // 2)
    calibration_seeds = set(seed_order[:split])
    validation_seeds = set(seed_order[split:])
    eligible = [
        row for row in rows
        if bool(row[f"{prefix}_available"])
        and bool(row["noop_feedback_available"])
    ]
    calibration_rows = [
        row for row in eligible if int(row["seed"]) in calibration_seeds
    ]
    validation_rows = [
        row for row in eligible if int(row["seed"]) in validation_seeds
    ]
    candidate_margin = _split_conformal_upper([
        float(row[f"{prefix}_calibration_score"])
        for row in calibration_rows
    ], alpha)
    noop_margin = _split_conformal_upper([
        float(row["noop_feedback_calibration_score"])
        for row in calibration_rows
    ], alpha)
    accepted = []
    policy_worst = []
    violations = []
    for row in validation_rows:
        candidate_lower = max(
            float(row[f"{prefix}_estimated_worst"]) - candidate_margin,
            0.0,
        )
        noop_lower = max(
            float(row["noop_feedback_worst"]) - noop_margin,
            0.0,
        )
        choose = candidate_lower >= min(noop_lower, qos_floor)
        candidate_true = float(row[f"{prefix}_worst"])
        noop_true = float(row["deployed_worst"])
        accepted.append(choose)
        policy_worst.append(candidate_true if choose else noop_true)
        violations.append(
            choose and candidate_true + 1.0e-12 < min(noop_true, qos_floor)
        )
    return {
        "alpha": float(alpha),
        "calibration_seed_order": list(seed_order[:split]),
        "validation_seed_order": list(seed_order[split:]),
        "calibration_event_count": len(calibration_rows),
        "validation_event_count": len(validation_rows),
        "candidate_pd_overprediction_margin": float(candidate_margin),
        "noop_pd_overprediction_margin": float(noop_margin),
        "validation_accept_rate": float(np.mean(accepted)),
        "validation_policy_mean_worst": float(np.mean(policy_worst)),
        "validation_noop_mean_worst": float(np.mean([
            float(row["deployed_worst"]) for row in validation_rows
        ])),
        "validation_constraint_violation_rate": float(np.mean(violations)),
        "certificate_ready": False,
        "certificate_blocker": (
            f"events are clustered within only {len(calibration_seeds)} "
            "calibration episodes; the order statistic is diagnostic and "
            "not an independence-valid deployment certificate"
        ),
        "switch_bit_delay_cost_included": False,
    }


def audit(
    trace_path: Path,
    *,
    seed_limit: int,
    rounds: tuple[int, ...],
    price_bits: int,
    feedback_bits: int,
    qos_floor: float,
    causal_rounds: int,
) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    seed_mask = np.isin(seeds, seed_order)
    indices = np.flatnonzero(resolved & seed_mask)
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    rows: list[dict[str, object]] = []
    previous_coefficient: dict[int, np.ndarray] = {}
    previous_uav_positions: dict[int, np.ndarray] = {}
    previous_target_states: dict[int, np.ndarray] = {}
    started = time.perf_counter()
    for row in indices:
        row = int(row)
        comm_power, sensing_power, recorded_error = _recorded_power(data, row)
        d_eff = np.asarray(data["privileged_d_eff"][row], dtype=np.float64)
        pair = np.asarray(data["teacher_pair"][row], dtype=bool)
        selected = tuple(
            tuple(int(value) for value in edge) for edge in np.argwhere(pair)
        )
        deployed_deflection = np.max(np.sum(d_eff * pair, axis=0), axis=0)
        deployed_pd = compute_detection_probabilities(deployed_deflection, p_fa)
        coord_valid = bool(np.asarray(data["coord_pd_ema_valid"])[row])
        coord_pd = np.asarray(data["coord_pd_ema"][row], dtype=np.float64)
        coefficient = per_watt_deflection_tensor(
            _entries(data, row), sensing_power, K, Q)
        gain, _ = fixed_owner_gain_matrix(coefficient, selected)
        budget = 1.0 - comm_power
        exact = solve_fixed_structure_maxmin_power_lp(gain, budget)
        report: dict[str, object] = {
            "seed": int(seeds[row]),
            "frame": int(frames[row]),
            "deployed_worst": float(np.min(deployed_pd)),
            "noop_feedback_available": coord_valid,
            "noop_feedback_worst": float(np.min(coord_pd)),
            "noop_feedback_calibration_score": float(
                max(float(np.max(coord_pd - deployed_pd)), 0.0)
                if coord_valid else 0.0
            ),
            "exact_worst": _worst_pd(exact.deflection, p_fa),
            "exact_worst_deflection": float(exact.worst_deflection),
            "recorded_power_balance_error_w": float(recorded_error),
            "exact_power_balance_error_w": float(np.max(np.abs(
                np.sum(exact.power_w, axis=1) - budget
            ))),
            "exact_power_total_variation": float(np.mean(
                0.5 * np.sum(np.abs(exact.power_w - sensing_power), axis=1)
            )),
        }
        for count in rounds:
            finite = distributed_column_generation_maxmin_power(
                gain, budget, rounds=int(count), price_bits=int(price_bits))
            prefix = f"round_{count}"
            report[f"{prefix}_worst"] = _worst_pd(finite.deflection, p_fa)
            report[f"{prefix}_deflection_fraction"] = float(
                finite.worst_deflection
                / max(exact.worst_deflection, 1.0e-300)
            )
            report[f"{prefix}_power_balance_error_w"] = float(np.max(np.abs(
                np.sum(finite.power_w, axis=1) - budget
            )))
            report[f"{prefix}_power_total_variation"] = float(np.mean(
                0.5 * np.sum(np.abs(finite.power_w - sensing_power), axis=1)
            ))
            report[f"{prefix}_actual_rounds"] = int(finite.rounds)
        seed = int(seeds[row])
        lag_coefficient = previous_coefficient.get(seed)
        if lag_coefficient is None:
            report.update({
                "causal_lag_available": False,
                "causal_lag_worst": report["deployed_worst"],
                "causal_lag_estimated_worst": report["deployed_worst"],
                "causal_lag_harmed": False,
                "causal_lag_owner_pd_abs_error": 0.0,
                "causal_lag_owner_pd_overprediction": 0.0,
                "causal_lag_calibration_score": 0.0,
                "causal_lag_power_total_variation": 0.0,
            })
        else:
            lag_gain, _ = fixed_owner_gain_matrix(lag_coefficient, selected)
            causal = distributed_column_generation_maxmin_power(
                lag_gain,
                budget,
                rounds=int(causal_rounds),
                price_bits=int(price_bits),
            )
            realized_deflection = np.sum(gain * causal.power_w, axis=0)
            estimated_pd = compute_detection_probabilities(
                causal.deflection, p_fa)
            realized_pd = compute_detection_probabilities(
                realized_deflection, p_fa)
            report.update({
                "causal_lag_available": True,
                "causal_lag_worst": float(np.min(realized_pd)),
                "causal_lag_estimated_worst": float(np.min(estimated_pd)),
                "causal_lag_harmed": bool(
                    float(np.min(realized_pd))
                    < float(report["deployed_worst"]) - 1.0e-12
                ),
                "causal_lag_owner_pd_abs_error": float(np.mean(
                    np.abs(estimated_pd - realized_pd)
                )),
                "causal_lag_owner_pd_overprediction": float(np.max(
                    estimated_pd - realized_pd
                )),
                "causal_lag_calibration_score": float(max(
                    float(np.max(estimated_pd - realized_pd)), 0.0
                )),
                "causal_lag_power_total_variation": float(np.mean(
                    0.5 * np.sum(
                        np.abs(causal.power_w - sensing_power), axis=1
                    )
                )),
            })
        if lag_coefficient is None:
            report.update({
                "causal_geometry_available": False,
                "causal_geometry_worst": report["deployed_worst"],
                "causal_geometry_estimated_worst": report["deployed_worst"],
                "causal_geometry_harmed": False,
                "causal_geometry_owner_pd_abs_error": 0.0,
                "causal_geometry_owner_pd_overprediction": 0.0,
                "causal_geometry_calibration_score": 0.0,
                "causal_geometry_noop_estimated_worst": report[
                    "deployed_worst"],
                "causal_geometry_noop_calibration_score": 0.0,
                "causal_geometry_delta_calibration_score": 0.0,
                "causal_geometry_power_total_variation": 0.0,
                "causal_geometry_direct_edge_count": 0,
                "causal_geometry_fallback_edge_count": 0,
                "causal_geometry_invariant_log_mad": 0.0,
            })
        else:
            geometry = predict_fixed_owner_gain_from_geometry(
                lag_coefficient,
                previous_uav_positions[seed],
                np.asarray(data["uav_positions"][row], dtype=np.float64),
                previous_target_states[seed],
                np.asarray(data["target_states"][row], dtype=np.float64),
                selected,
            )
            causal_geometry = distributed_column_generation_maxmin_power(
                geometry.gain_per_watt,
                budget,
                rounds=int(causal_rounds),
                price_bits=int(price_bits),
            )
            geometry_realized_deflection = np.sum(
                gain * causal_geometry.power_w, axis=0)
            geometry_estimated_pd = compute_detection_probabilities(
                causal_geometry.deflection, p_fa)
            geometry_realized_pd = compute_detection_probabilities(
                geometry_realized_deflection, p_fa)
            geometry_noop_estimated_deflection = np.sum(
                geometry.gain_per_watt * sensing_power, axis=0)
            geometry_noop_estimated_pd = compute_detection_probabilities(
                geometry_noop_estimated_deflection, p_fa)
            predicted_delta = float(
                np.min(geometry_estimated_pd)
                - np.min(geometry_noop_estimated_pd)
            )
            realized_delta = float(
                np.min(geometry_realized_pd) - np.min(deployed_pd)
            )
            report.update({
                "causal_geometry_available": True,
                "causal_geometry_worst": float(np.min(geometry_realized_pd)),
                "causal_geometry_estimated_worst": float(
                    np.min(geometry_estimated_pd)),
                "causal_geometry_harmed": bool(
                    float(np.min(geometry_realized_pd))
                    < float(report["deployed_worst"]) - 1.0e-12
                ),
                "causal_geometry_owner_pd_abs_error": float(np.mean(
                    np.abs(geometry_estimated_pd - geometry_realized_pd)
                )),
                "causal_geometry_owner_pd_overprediction": float(np.max(
                    geometry_estimated_pd - geometry_realized_pd
                )),
                "causal_geometry_calibration_score": float(max(
                    float(np.max(
                        geometry_estimated_pd - geometry_realized_pd)), 0.0
                )),
                "causal_geometry_noop_estimated_worst": float(
                    np.min(geometry_noop_estimated_pd)),
                "causal_geometry_noop_calibration_score": float(max(
                    float(np.max(
                        geometry_noop_estimated_pd - deployed_pd)), 0.0
                )),
                "causal_geometry_delta_calibration_score": float(max(
                    predicted_delta - realized_delta, 0.0
                )),
                "causal_geometry_power_total_variation": float(np.mean(
                    0.5 * np.sum(
                        np.abs(causal_geometry.power_w - sensing_power), axis=1
                    )
                )),
                "causal_geometry_direct_edge_count": int(
                    geometry.direct_edge_count),
                "causal_geometry_fallback_edge_count": int(
                    geometry.fallback_edge_count),
                "causal_geometry_invariant_log_mad": float(
                    geometry.invariant_log_mad),
            })
        previous_coefficient[seed] = coefficient.copy()
        previous_uav_positions[seed] = np.asarray(
            data["uav_positions"][row], dtype=np.float64).copy()
        previous_target_states[seed] = np.asarray(
            data["target_states"][row], dtype=np.float64).copy()
        rows.append(report)
    elapsed = time.perf_counter() - started

    methods = {
        "deployed": "deployed_worst",
        "exact": "exact_worst",
        **{f"round_{count}": f"round_{count}_worst" for count in rounds},
    }
    summary = {
        name: {
            "episode_mean_worst": _episode_mean(rows, key),
            "event_mean_worst": float(np.mean([float(row[key]) for row in rows])),
            "event_qos_feasible_rate": float(np.mean([
                float(row[key]) >= qos_floor for row in rows
            ])),
        }
        for name, key in methods.items()
    }
    for count in rounds:
        name = f"round_{count}"
        summary[name].update({
            "event_mean_exact_deflection_fraction": float(np.mean([
                float(row[f"{name}_deflection_fraction"]) for row in rows
            ])),
            "event_min_exact_deflection_fraction": float(np.min([
                float(row[f"{name}_deflection_fraction"]) for row in rows
            ])),
            "event_mean_power_total_variation": float(np.mean([
                float(row[f"{name}_power_total_variation"]) for row in rows
            ])),
            "max_power_balance_error_w": float(np.max([
                float(row[f"{name}_power_balance_error_w"]) for row in rows
            ])),
            "mean_actual_rounds": float(np.mean([
                float(row[f"{name}_actual_rounds"]) for row in rows
            ])),
            "payload_bits_per_resolve_upper": int(
                Q * feedback_bits
                + int(count) * Q * (price_bits + feedback_bits)
            ),
        })
    lag_rows = [row for row in rows if bool(row["causal_lag_available"])]
    causal_summary = {
        "rounds": int(causal_rounds),
        "available_event_count": len(lag_rows),
        "episode_mean_worst": _episode_mean(lag_rows, "causal_lag_worst"),
        "event_mean_worst": float(np.mean([
            float(row["causal_lag_worst"]) for row in lag_rows
        ])),
        "event_qos_feasible_rate": float(np.mean([
            float(row["causal_lag_worst"]) >= qos_floor for row in lag_rows
        ])),
        "harm_rate_against_same_event_noop": float(np.mean([
            bool(row["causal_lag_harmed"]) for row in lag_rows
        ])),
        "owner_pd_mean_absolute_error": float(np.mean([
            float(row["causal_lag_owner_pd_abs_error"]) for row in lag_rows
        ])),
        "owner_pd_mean_max_overprediction": float(np.mean([
            float(row["causal_lag_owner_pd_overprediction"])
            for row in lag_rows
        ])),
        "owner_pd_p95_max_overprediction": float(np.quantile([
            float(row["causal_lag_owner_pd_overprediction"])
            for row in lag_rows
        ], 0.95)),
        "event_mean_power_total_variation": float(np.mean([
            float(row["causal_lag_power_total_variation"]) for row in lag_rows
        ])),
    }
    geometry_rows = [
        row for row in rows if bool(row["causal_geometry_available"])
    ]
    geometry_summary = {
        "rounds": int(causal_rounds),
        "available_event_count": len(geometry_rows),
        "episode_mean_worst": _episode_mean(
            geometry_rows, "causal_geometry_worst"),
        "event_mean_worst": float(np.mean([
            float(row["causal_geometry_worst"]) for row in geometry_rows
        ])),
        "event_qos_feasible_rate": float(np.mean([
            float(row["causal_geometry_worst"]) >= qos_floor
            for row in geometry_rows
        ])),
        "harm_rate_against_same_event_noop": float(np.mean([
            bool(row["causal_geometry_harmed"]) for row in geometry_rows
        ])),
        "owner_pd_mean_absolute_error": float(np.mean([
            float(row["causal_geometry_owner_pd_abs_error"])
            for row in geometry_rows
        ])),
        "owner_pd_mean_max_overprediction": float(np.mean([
            float(row["causal_geometry_owner_pd_overprediction"])
            for row in geometry_rows
        ])),
        "owner_pd_p95_max_overprediction": float(np.quantile([
            float(row["causal_geometry_owner_pd_overprediction"])
            for row in geometry_rows
        ], 0.95)),
        "event_mean_power_total_variation": float(np.mean([
            float(row["causal_geometry_power_total_variation"])
            for row in geometry_rows
        ])),
        "mean_direct_edge_count": float(np.mean([
            float(row["causal_geometry_direct_edge_count"])
            for row in geometry_rows
        ])),
        "mean_fallback_edge_count": float(np.mean([
            float(row["causal_geometry_fallback_edge_count"])
            for row in geometry_rows
        ])),
    }
    feedback_gate = _diagnostic_feedback_gate(
        rows,
        prefix="causal_lag",
        seed_order=seed_order,
        qos_floor=qos_floor,
    )
    geometry_feedback_gate = _diagnostic_feedback_gate(
        rows,
        prefix="causal_geometry",
        seed_order=seed_order,
        qos_floor=qos_floor,
    )
    return {
        "schema_version": 1,
        "scope": (
            "selection development episodes; all structural resolve events; "
            "same-state privileged per-watt gains; diagnostic potential only"
        ),
        "trace": str(trace_path),
        "seed_order": seed_order,
        "event_count": len(rows),
        "qos_floor": float(qos_floor),
        "price_bits": int(price_bits),
        "feedback_bits": int(feedback_bits),
        "causal_feedback_baseline": (
            "previous structural-resolve per-watt coefficient, remapped to the "
            "current fixed owner graph; first event per episode remains No-op"
        ),
        "causal_geometry_predictor": (
            "current mission-known target/UAV geometry plus previous owner "
            "per-watt feedback; persistent edges use the exact bistatic "
            "range-product-square ratio and new edges use a robust invariant"
        ),
        "payload_scope": (
            "target-cover feedback plus price/Deflection payload only; headers, "
            "contention, retransmission and causal estimator packets excluded"
        ),
        "elapsed_seconds": float(elapsed),
        "summary": summary,
        "causal_lag_summary": causal_summary,
        "causal_geometry_summary": geometry_summary,
        "feedback_gate": feedback_gate,
        "geometry_feedback_gate": geometry_feedback_gate,
        "max_recorded_power_balance_error_w": float(np.max([
            float(row["recorded_power_balance_error_w"]) for row in rows
        ])),
        "max_exact_power_balance_error_w": float(np.max([
            float(row["exact_power_balance_error_w"]) for row in rows
        ])),
        "rows": rows,
        "fresh_test_consumed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--seed-limit", type=int, default=10)
    parser.add_argument("--rounds", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--price-bits", type=int, default=6)
    parser.add_argument("--feedback-bits", type=int, default=16)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument("--causal-rounds", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rounds = tuple(sorted({max(1, int(value)) for value in args.rounds}))
    result = audit(
        args.trace,
        seed_limit=max(1, int(args.seed_limit)),
        rounds=rounds,
        price_bits=int(args.price_bits),
        feedback_bits=int(args.feedback_bits),
        qos_floor=float(args.qos_floor),
        causal_rounds=max(1, int(args.causal_rounds)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "event_count": result["event_count"],
        "elapsed_seconds": result["elapsed_seconds"],
        "summary": result["summary"],
        "causal_lag_summary": result["causal_lag_summary"],
        "causal_geometry_summary": result["causal_geometry_summary"],
        "feedback_gate": result["feedback_gate"],
        "geometry_feedback_gate": result["geometry_feedback_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
