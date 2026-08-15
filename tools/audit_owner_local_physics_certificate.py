#!/usr/bin/env python
"""Audit an owner-local two-part physical certificate on frozen traces."""

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

from tools.audit_power_repair_transport_trace import (  # noqa: E402
    _coefficient_from_trace,
    _communication_model,
)
from tools.audit_structure_teacher_trace import (  # noqa: E402
    _infer_observation_slices,
)
from tools.audit_structure_trace_physical_bottleneck import (  # noqa: E402
    _ordered_unique,
    _recorded_power,
)
from uav_isac.coordination.owner_local_physics import (  # noqa: E402
    decode_owner_local_kinematics,
    owner_local_dd_effectiveness,
    predict_coefficients_from_lagged_feedback,
    reciprocal_token_candidate_mask,
)
from uav_isac.evaluation.episode_joint_conformal import (  # noqa: E402
    split_conformal_upper,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    delivered_target_tokens,
)


def _episode_max(
    rows: list[dict[str, object]], key: str,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for row in rows:
        seed = int(row["seed"])
        result[seed] = max(result.get(seed, 0.0), float(row[key]))
    return result


def audit(
    trace_path: Path,
    config_path: Path,
    *,
    seed_limit: int,
    alpha: float = 0.05,
    frozen_joint_margin: float | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg, _ = _communication_model(config_path)
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    if K != int(cfg.scenario.K) or Q != int(cfg.scenario.Q):
        raise ValueError("trace and configuration dimensions disagree")
    risk = float(alpha)
    if not (0.0 < risk < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    if frozen_joint_margin is not None and (
        not np.isfinite(float(frozen_joint_margin))
        or float(frozen_joint_margin) < 0.0
    ):
        raise ValueError("frozen_joint_margin must be finite non-negative")

    seeds = np.asarray(data["seed"], dtype=np.int64)
    frames = np.asarray(data["frame"], dtype=np.int64)
    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    seed_order = _ordered_unique(seeds)[:max(1, int(seed_limit))]
    indices = np.flatnonzero(resolved & np.isin(seeds, seed_order))
    local_obs = np.asarray(data["local_obs"], dtype=np.float32)
    slices = _infer_observation_slices(local_obs.shape[-1], K, Q)
    token_visible, _ = delivered_target_tokens(local_obs, slices)
    token_candidate = reciprocal_token_candidate_mask(token_visible)
    area_size = tuple(float(value) for value in cfg.scenario.region_size)

    previous_coefficient: dict[int, np.ndarray] = {}
    previous_observed: dict[int, np.ndarray] = {}
    previous_state: dict[int, object] = {}
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for index in indices:
        row_id = int(index)
        seed = int(seeds[row_id])
        state = decode_owner_local_kinematics(
            local_obs[row_id], slices,
            area_size_m=area_size,
            height_m=float(cfg.scenario.height),
        )
        coefficient = _coefficient_from_trace(data, row_id, cfg)
        physical_support = (
            np.asarray(data["privileged_candidate"][row_id], dtype=bool)
            & (
                np.asarray(data["privileged_g_dd"][row_id], dtype=np.float64)
                >= float(cfg.detection.g_min)
            )
        )
        candidate = np.asarray(token_candidate[row_id], dtype=bool)
        selected = np.asarray(data["teacher_pair"][row_id], dtype=bool)
        _, sensing_power, _ = _recorded_power(data, row_id)
        dd_prediction = owner_local_dd_effectiveness(
            state,
            carrier_hz=float(cfg.otfs.fc),
            delta_f_hz=float(cfg.otfs.delta_f),
            symbol_period_s=float(cfg.otfs.T_sym),
            delay_bins=int(cfg.otfs.M),
            doppler_bins=int(cfg.otfs.N),
        )
        row: dict[str, object] = {
            "seed": seed,
            "frame": int(frames[row_id]),
            "causal_available": seed in previous_coefficient,
            "token_candidate_edge_count": int(np.sum(candidate)),
            "token_candidate_physical_edge_count": int(np.sum(
                candidate & physical_support)),
            "token_candidate_false_support_edge_count": int(np.sum(
                candidate & ~physical_support)),
        }
        if seed in previous_coefficient:
            prediction = predict_coefficients_from_lagged_feedback(
                previous_coefficient[seed],
                previous_observed[seed],
                previous_state[seed],
                state,
                current_support=candidate,
            )
            predicted_coefficient = prediction.coefficient_per_watt
            positive_prediction = predicted_coefficient > 0.0
            positive_truth = coefficient > 0.0
            comparable = (
                candidate & physical_support
                & positive_prediction & positive_truth
            )
            false_support_score = float(np.max(np.where(
                candidate & ~physical_support,
                np.maximum(dd_prediction - float(cfg.detection.g_min), 0.0),
                0.0,
            )))
            log_score = (
                float(np.max(np.abs(np.log(
                    predicted_coefficient[comparable]
                    / coefficient[comparable]
                ))))
                if np.any(comparable) else 0.0
            )
            unpredicted_positive = int(np.sum(
                candidate & physical_support & positive_truth
                & ~positive_prediction
            ))
            joint_score = max(false_support_score, log_score)
            row.update({
                "dd_false_support_score": false_support_score,
                "coefficient_abs_log_score": log_score,
                "joint_physics_score": joint_score,
                "prediction_direct_edge_count": int(
                    prediction.direct_edge_count),
                "prediction_target_fallback_edge_count": int(
                    prediction.target_fallback_edge_count),
                "prediction_unavailable_edge_count": int(
                    prediction.unavailable_edge_count),
                "unpredicted_positive_edge_count": unpredicted_positive,
            })
            records.append({
                "row": row,
                "candidate": candidate,
                "physical_support": physical_support,
                "dd_prediction": dd_prediction,
                "comparable": comparable,
                "predicted_coefficient": predicted_coefficient,
                "coefficient": coefficient,
            })
        previous_coefficient[seed] = coefficient.copy()
        previous_observed[seed] = (
            selected & (sensing_power[:, None, :] > 1.0e-12)
        )
        previous_state[seed] = state
        rows.append(row)

    causal_rows = [row for row in rows if bool(row["causal_available"])]
    if not causal_rows:
        raise ValueError("trace contains no causal resolve transitions")
    dd_scores = _episode_max(causal_rows, "dd_false_support_score")
    log_scores = _episode_max(causal_rows, "coefficient_abs_log_score")
    joint_scores = _episode_max(causal_rows, "joint_physics_score")
    dd_margin, dd_coverage, dd_rank = split_conformal_upper(
        dd_scores.values(), alpha=risk)
    log_margin, log_coverage, log_rank = split_conformal_upper(
        log_scores.values(), alpha=risk)
    joint_margin, joint_coverage, joint_rank = split_conformal_upper(
        joint_scores.values(), alpha=risk)
    active_margin = (
        float(joint_margin)
        if frozen_joint_margin is None else float(frozen_joint_margin)
    )

    admitted_count = 0
    admitted_physical_count = 0
    token_physical_count = 0
    support_violation_by_seed: dict[int, bool] = {}
    interval_violation_by_seed: dict[int, bool] = {}
    joint_violation_by_seed: dict[int, bool] = {}
    for record in records:
        row = record["row"]
        candidate = np.asarray(record["candidate"], dtype=bool)
        physical_support = np.asarray(record["physical_support"], dtype=bool)
        dd_prediction = np.asarray(record["dd_prediction"], dtype=np.float64)
        comparable = np.asarray(record["comparable"], dtype=bool)
        predicted_coefficient = np.asarray(
            record["predicted_coefficient"], dtype=np.float64)
        coefficient = np.asarray(record["coefficient"], dtype=np.float64)
        # Strict inequality is required.  Equality is a calibration boundary,
        # not evidence that a zero-gain edge is safe to excite.
        admitted = candidate & (
            dd_prediction - active_margin > float(cfg.detection.g_min)
        )
        support_violation = bool(np.any(admitted & ~physical_support))
        interval_mask = admitted & comparable
        lower = predicted_coefficient * np.exp(-active_margin)
        upper = predicted_coefficient * np.exp(active_margin)
        interval_violation = bool(np.any(
            interval_mask
            & ((coefficient < lower) | (coefficient > upper))
        ))
        missing_violation = bool(np.any(
            admitted & physical_support & (coefficient > 0.0)
            & (predicted_coefficient <= 0.0)
        ))
        joint_violation = bool(
            support_violation or interval_violation or missing_violation)
        seed = int(row["seed"])
        support_violation_by_seed[seed] = (
            support_violation_by_seed.get(seed, False) or support_violation)
        interval_violation_by_seed[seed] = (
            interval_violation_by_seed.get(seed, False)
            or interval_violation or missing_violation)
        joint_violation_by_seed[seed] = (
            joint_violation_by_seed.get(seed, False) or joint_violation)
        admitted_count += int(np.sum(admitted))
        admitted_physical_count += int(np.sum(admitted & physical_support))
        token_physical_count += int(np.sum(candidate & physical_support))
        row.update({
            "active_joint_margin": active_margin,
            "admitted_edge_count": int(np.sum(admitted)),
            "admitted_false_support_edge_count": int(np.sum(
                admitted & ~physical_support)),
            "support_certificate_violation": support_violation,
            "coefficient_interval_violation": interval_violation,
            "missing_coefficient_violation": missing_violation,
            "joint_physics_certificate_violation": joint_violation,
        })

    episode_count = len(joint_scores)
    summary = {
        "causal_event_count": len(causal_rows),
        "independent_episode_count": episode_count,
        "mean_token_candidate_edge_count": float(np.mean([
            int(row["token_candidate_edge_count"]) for row in causal_rows
        ])),
        "token_candidate_physical_precision": float(np.sum([
            int(row["token_candidate_physical_edge_count"])
            for row in causal_rows
        ]) / max(np.sum([
            int(row["token_candidate_edge_count"]) for row in causal_rows
        ]), 1)),
        "dd_only_diagnostic_margin": float(dd_margin),
        "dd_only_diagnostic_coverage_floor": float(dd_coverage),
        "dd_only_diagnostic_rank_one_based": int(dd_rank),
        "coefficient_only_diagnostic_log_margin": float(log_margin),
        "coefficient_only_diagnostic_coverage_floor": float(log_coverage),
        "coefficient_only_diagnostic_rank_one_based": int(log_rank),
        "joint_physics_margin": float(joint_margin),
        "joint_physics_multiplicative_lower_factor": float(np.exp(
            -joint_margin)),
        "joint_physics_multiplicative_upper_factor": float(np.exp(
            joint_margin)),
        "joint_physics_coverage_floor": float(joint_coverage),
        "joint_physics_rank_one_based": int(joint_rank),
        "active_frozen_joint_margin": float(active_margin),
        "admitted_mean_edge_count": float(admitted_count / len(causal_rows)),
        "admitted_physical_precision": float(
            admitted_physical_count / max(admitted_count, 1)),
        "admitted_recall_within_token_physical_graph": float(
            admitted_physical_count / max(token_physical_count, 1)),
        "episode_any_support_violation_rate": float(np.mean(list(
            support_violation_by_seed.values()))),
        "episode_any_coefficient_interval_violation_rate": float(np.mean(list(
            interval_violation_by_seed.values()))),
        "episode_any_joint_physics_violation_rate": float(np.mean(list(
            joint_violation_by_seed.values()))),
        "max_unpredicted_positive_edge_count": int(max([
            int(row["unpredicted_positive_edge_count"])
            for row in causal_rows
        ], default=0)),
    }
    return {
        "schema_version": 1,
        "gate": "D0.16-owner-local-physics",
        "scope": (
            "owner-local observability and two-part physical certificate; "
            "not an end-to-end controller performance claim"
        ),
        "trace": str(trace_path),
        "config": str(config_path),
        "seed_order": seed_order,
        "alpha": risk,
        "margin_source": (
            "same-split development calibration"
            if frozen_joint_margin is None else "externally frozen"
        ),
        "frozen_joint_margin": (
            None if frozen_joint_margin is None
            else float(frozen_joint_margin)
        ),
        "information_contract": {
            "proposal_graph": "reciprocal actually delivered target Tokens",
            "kinematics": "endpoint self-state plus receiver-local target belief",
            "gain_feedback": "lag-1 coefficients on selected powered edges only",
            "privileged_candidate_use": "outcome label only",
            "privileged_g_dd_use": "outcome label only",
            "current_true_target_state_use": False,
            "current_full_coefficient_use": "outcome label only",
        },
        "certificate_ready": False,
        "certificate_blockers": [
            "selected-edge coefficient feedback is replayed as noiseless",
            "the target-invariant feedback field and its transport cost are not yet encoded",
            "packet loss and CSI residuals remain deterministic in this replay",
            "the owner-local certificate is not yet connected to structural LNS",
        ],
        "summary": summary,
        "fresh_test_consumed": False,
        "elapsed_seconds": float(time.perf_counter() - started),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed-limit", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--frozen-joint-margin", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.trace,
        args.config,
        seed_limit=max(1, int(args.seed_limit)),
        alpha=float(args.alpha),
        frozen_joint_margin=args.frozen_joint_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "summary": result["summary"],
        "certificate_ready": result["certificate_ready"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
