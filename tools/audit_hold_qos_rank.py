#!/usr/bin/env python
"""Calibrate a hold-aware oracle ranker, then evaluate it on unseen seeds.

This is a falsification/upper-bound audit.  The ranker reads future realized
``d_eff`` inside the current hold segment and therefore is not deployable.
Its purpose is narrower: determine whether temporal aggregation can improve
episode worst/QoS before implementing a learned predictor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_oracle_ladder import (  # noqa: E402
    _checkpoint_training_seeds,
)
from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    episode_detection_summary,
    held_pair_sequence,
    realized_pd_history,
)
from uav_isac.evaluation.temporal_performance import (  # noqa: E402
    summarize_multiframe_detection,
)


def _subset(data: dict[str, np.ndarray], seeds: list[int]) -> dict[str, np.ndarray]:
    frame_count = len(data["seed"])
    keep = np.isin(data["seed"], np.asarray(seeds, dtype=np.int64))
    if not np.any(keep):
        raise ValueError("selected seeds have no frames")
    return {
        key: value[keep] if value.ndim > 0 and value.shape[0] == frame_count else value
        for key, value in data.items()
    }


def hold_aware_score(
    realized_d_eff: np.ndarray,
    episode: np.ndarray,
    resolved: np.ndarray,
    *,
    tail_weight: float,
    quantile: float,
) -> np.ndarray:
    """Blend hold-segment mean and lower-tail edge evidence at resolve frames."""
    values = np.asarray(realized_d_eff, dtype=np.float64)
    episodes = np.asarray(episode, dtype=np.int64)
    resolve = np.asarray(resolved, dtype=bool)
    if values.ndim != 4 or episodes.shape != (len(values),) or resolve.shape != episodes.shape:
        raise ValueError("incompatible d_eff/episode/resolved shapes")
    weight = float(np.clip(tail_weight, 0.0, 1.0))
    q = float(np.clip(quantile, 0.0, 1.0))
    score = values.copy()
    for ep in np.unique(episodes):
        indices = np.flatnonzero(episodes == ep)
        resolve_indices = indices[resolve[indices]]
        for position, frame in enumerate(resolve_indices):
            stop = (
                int(resolve_indices[position + 1])
                if position + 1 < len(resolve_indices)
                else int(indices[-1]) + 1
            )
            segment = np.maximum(values[int(frame):stop], 0.0)
            mean = np.mean(segment, axis=0)
            tail = np.quantile(segment, q, axis=0)
            score[int(frame)] = (1.0 - weight) * mean + weight * tail
    return score


def _multiframe_episode_summary(
    target_deflection: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    p_fa: float,
    worst_floor: float,
    frame_duration_s: float,
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for ep in np.unique(np.asarray(data["episode"], dtype=np.int64)):
        indices = np.flatnonzero(data["episode"] == ep)
        report = summarize_multiframe_detection(
            target_deflection[indices],
            p_fa,
            windows=(1, 2, 3, 5, 10),
            tail_window=20,
            floors=(0.80, 0.70, float(worst_floor)),
            frame_duration_s=float(frame_duration_s),
        )
        for width, item in report["windows"].items():
            rows.setdefault(width, []).append({
                "episode": int(ep),
                "seed": int(data["seed"][indices[0]]),
                **item,
            })
    summary: dict[str, Any] = {}
    for width, items in rows.items():
        worst = np.asarray([item["worst"] for item in items], dtype=np.float64)
        summary[width] = {
            "episodes": len(items),
            "observation_time_s": items[0]["observation_time_s"],
            "steady_mean": float(np.mean([item["steady"] for item in items])),
            "weak3_mean": float(np.mean([item["weak3"] for item in items])),
            "worst_mean": float(np.mean(worst)),
            "worst_min": float(np.min(worst)),
            "worst_p05": float(np.percentile(worst, 5)),
            "worst_floor_pass_rate": float(np.mean(worst >= float(worst_floor))),
            "qos_rate": float(np.mean([item["qos_success"] for item in items])),
            "episode_rows": items,
        }
    eligible = [
        int(width) for width, item in summary.items()
        if item["worst_mean"] >= float(worst_floor)
    ]
    all_episode_eligible = [
        int(width) for width, item in summary.items()
        if item["worst_min"] >= float(worst_floor)
    ]
    return {
        "worst_floor": float(worst_floor),
        "minimum_window_with_mean_worst_pass": min(eligible) if eligible else None,
        "minimum_window_with_all_episodes_pass": (
            min(all_episode_eligible) if all_episode_eligible else None
        ),
        "windows": summary,
    }


def _evaluate(
    data: dict[str, np.ndarray],
    score: np.ndarray,
    *,
    worst_floor: float = 0.75,
    frame_duration_s: float = 0.1,
) -> dict[str, Any]:
    physical = np.asarray(data["privileged_candidate"], dtype=bool)
    pairs = held_pair_sequence(score, physical, data)
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    realized = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    pd = realized_pd_history(pairs, realized, p_fa)
    receiver_d = np.sum(realized * pairs, axis=1)
    target_d = np.max(receiver_d, axis=1)
    summary, rows = episode_detection_summary(
        pd,
        data["episode"],
        data["seed"],
        steady_window=20,
        qos_thresholds=(0.80, 0.70, 0.60),
    )
    return {
        "summary": summary,
        "episodes": rows,
        "multiframe_worst_075": _multiframe_episode_summary(
            target_d,
            data,
            p_fa=p_fa,
            worst_floor=worst_floor,
            frame_duration_s=frame_duration_s,
        ),
    }


def _selection_key(result: dict[str, Any]) -> tuple[float, ...]:
    summary = result["summary"]
    return (
        float(summary["qos_feasible"]),
        float(summary["worst"]),
        float(summary["weak3"]),
        float(summary["steady"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.trace, allow_pickle=False) as loaded:
        raw = {key: loaded[key] for key in loaded.files}
    student = FrozenStructureStudent(args.student_checkpoint)
    calibration_seeds = sorted(_checkpoint_training_seeds(student))
    all_seeds = list(dict.fromkeys(int(v) for v in np.asarray(raw["seed"])))
    evaluation_seeds = [seed for seed in all_seeds if seed not in calibration_seeds]
    if args.max_evaluation_seeds > 0:
        evaluation_seeds = evaluation_seeds[:args.max_evaluation_seeds]
    if not calibration_seeds or not evaluation_seeds:
        raise ValueError("both calibration and evaluation seed sets are required")

    calibration = _subset(raw, calibration_seeds)
    evaluation = _subset(raw, evaluation_seeds)
    candidates: dict[str, dict[str, Any]] = {}
    parameter_grid: list[tuple[float, float]] = []
    for quantile in args.quantiles:
        for weight in args.tail_weights:
            parameter_grid.append((float(quantile), float(weight)))
    for quantile, weight in parameter_grid:
        name = f"q{quantile:.2f}_w{weight:.2f}"
        score = hold_aware_score(
            calibration["privileged_d_eff"],
            calibration["episode"],
            calibration["p0_resolved"],
            tail_weight=weight,
            quantile=quantile,
        )
        candidates[name] = {
            "quantile": quantile,
            "tail_weight": weight,
            **_evaluate(
                calibration,
                score,
                worst_floor=args.worst_floor,
                frame_duration_s=args.frame_duration_s,
            ),
        }
    selected_name = max(candidates, key=lambda name: _selection_key(candidates[name]))
    selected = candidates[selected_name]

    current_frame = _evaluate(
        evaluation,
        np.asarray(evaluation["privileged_d_eff"], dtype=np.float64),
        worst_floor=args.worst_floor,
        frame_duration_s=args.frame_duration_s,
    )
    selected_score = hold_aware_score(
        evaluation["privileged_d_eff"],
        evaluation["episode"],
        evaluation["p0_resolved"],
        tail_weight=float(selected["tail_weight"]),
        quantile=float(selected["quantile"]),
    )
    hold_aware = _evaluate(
        evaluation,
        selected_score,
        worst_floor=args.worst_floor,
        frame_duration_s=args.frame_duration_s,
    )
    metrics = ("steady", "weak3", "worst", "cvar", "qos_feasible")
    delta = {
        metric: float(hold_aware["summary"][metric])
        - float(current_frame["summary"][metric])
        for metric in metrics
    }
    return {
        "schema": "hold-qos-rank-oracle-v1",
        "scope": "future-d_eff upper-bound audit; not deployable performance",
        "trace": str(args.trace),
        "student_checkpoint": str(args.student_checkpoint),
        "calibration_seeds": calibration_seeds,
        "evaluation_seeds": evaluation_seeds,
        "selection_order": ["qos_feasible", "worst", "weak3", "steady"],
        "calibration_grid": candidates,
        "selected": {
            "name": selected_name,
            "quantile": selected["quantile"],
            "tail_weight": selected["tail_weight"],
        },
        "evaluation": {
            "current_frame_oracle_rank": current_frame,
            "hold_aware_oracle_rank": hold_aware,
            "delta": delta,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evaluation-seeds", type=int, default=18)
    parser.add_argument("--worst-floor", type=float, default=0.75)
    parser.add_argument("--frame-duration-s", type=float, default=0.1)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    parser.add_argument("--tail-weights", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": report["selected"],
        "current": report["evaluation"]["current_frame_oracle_rank"]["summary"],
        "hold_aware": report["evaluation"]["hold_aware_oracle_rank"]["summary"],
        "delta": report["evaluation"]["delta"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
