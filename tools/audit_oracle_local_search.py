#!/usr/bin/env python
"""Gate C1.5 oracle local-neighborhood headroom audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.train_factor_graph_coordinator import (  # noqa: E402
    _aligned_data,
    _edge_values,
    _load_npz,
)
from uav_isac.coordination.local_exchange_oracle import (  # noqa: E402
    oracle_best_improvement,
    rebuild_structure,
    role_first_initial_structure,
    role_owner_from_structure,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    episode_detection_summary,
    realized_pd_history,
)


def _controller_summary(
    pair: np.ndarray,
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pd = realized_pd_history(
        pair,
        data["privileged_d_eff"],
        float(data["p_fa"][0]),
    )
    return episode_detection_summary(
        pd,
        data["episode"],
        data["seed"],
        steady_window=args.steady_window,
        qos_thresholds=(
            args.steady_floor, args.weak3_floor, args.worst_floor),
    )


def _run_variant(
    initial_name: str,
    neighborhoods: tuple[str, ...],
    initial_sequence: np.ndarray,
    edge_value: np.ndarray,
    candidate_mask: np.ndarray,
    data: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    frames, K, _, Q = candidate_mask.shape
    output = np.zeros_like(candidate_mask, dtype=bool)
    current = np.zeros((K, K, Q), dtype=bool)
    previous_final = np.zeros_like(current)
    previous_role: np.ndarray | None = None
    previous_episode: int | None = None
    accepted_counts = []
    candidate_counts = []
    move_counts = {kind: 0 for kind in ("N1", "N2", "N3", "N5")}
    resolve_count = 0
    monotonic = True
    for frame in range(frames):
        episode = int(data["episode"][frame])
        if episode != previous_episode:
            current.fill(False)
            previous_final.fill(False)
            previous_role = None
            previous_episode = episode
        if bool(data["p0_resolved"][frame]):
            current.fill(False)
            if np.any(candidate_mask[frame]):
                role_first, fallback_role, _ = role_first_initial_structure(
                    edge_value[frame],
                    candidate_mask[frame],
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(
                        data["reports_per_receiver"][0]),
                )
                if initial_name == "factor_graph":
                    initial = initial_sequence[frame]
                    initial_role = fallback_role
                elif initial_name == "role_first":
                    initial = role_first
                    initial_role = fallback_role
                elif initial_name == "previous":
                    if np.any(previous_final) and previous_role is not None:
                        _, owner = role_owner_from_structure(
                            previous_final, fallback_role=previous_role)
                        initial = rebuild_structure(
                            edge_value[frame],
                            candidate_mask[frame],
                            previous_role,
                            owner,
                            target_pair_limit=int(
                                data["target_pair_limit"][0]),
                            reports_per_receiver=int(
                                data["reports_per_receiver"][0]),
                        )
                        initial_role = previous_role
                    else:
                        initial = role_first
                        initial_role = fallback_role
                else:
                    raise ValueError(f"unknown initial: {initial_name}")
                result = oracle_best_improvement(
                    initial,
                    edge_value[frame],
                    candidate_mask[frame],
                    rounds=args.rounds,
                    neighborhoods=neighborhoods,
                    target_pair_limit=int(data["target_pair_limit"][0]),
                    reports_per_receiver=int(
                        data["reports_per_receiver"][0]),
                    initial_role=initial_role,
                )
                current[:] = result.selected
                previous_final[:] = result.selected
                previous_role = result.role.copy()
                accepted_counts.append(len(result.accepted_kinds))
                candidate_counts.append(sum(result.candidate_counts))
                for kind in result.accepted_kinds:
                    move_counts[kind] += 1
                monotonic &= all(
                    right > left
                    for left, right in zip(
                        result.objective_history,
                        result.objective_history[1:],
                    )
                )
                resolve_count += 1
        output[frame] = current
    diagnostics: dict[str, object] = {
        "resolve_nonempty": resolve_count,
        "accepted_moves_mean": float(np.mean(accepted_counts))
        if accepted_counts else 0.0,
        "accepted_moves_max": int(max(accepted_counts, default=0)),
        "candidate_evaluations_mean": float(np.mean(candidate_counts))
        if candidate_counts else 0.0,
        "positive_move_frame_rate": float(np.mean(
            np.asarray(accepted_counts) > 0)) if accepted_counts else 0.0,
        "objective_monotonic": bool(monotonic),
        "move_counts": move_counts,
    }
    return output, diagnostics


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    data, candidate = _aligned_data(args.trace, args.candidate)
    initial_artifact = _load_npz(args.initial_artifact)
    initial_sequence = initial_artifact[args.initial_key].astype(bool)
    if initial_sequence.shape != candidate["candidate_mask"].shape:
        raise ValueError("initial pair sequence does not align with candidate")
    reference_artifact = _load_npz(args.reference_artifact)
    reference_sequence = reference_artifact[args.reference_key].astype(bool)
    if reference_sequence.shape != initial_sequence.shape:
        raise ValueError("reference pair sequence does not align")
    edge_value = _edge_values(data, candidate, args.student_checkpoint)
    candidate_mask = candidate["candidate_mask"].astype(bool)
    initial_summary, _ = _controller_summary(initial_sequence, data, args)
    reference_summary, _ = _controller_summary(reference_sequence, data, args)

    variants: dict[str, dict[str, object]] = {}
    initial_metrics: dict[str, dict[str, object]] = {}
    outputs: dict[str, np.ndarray] = {}
    episode_rows = []
    neighborhood_sets = {
        "N1": ("N1",),
        "N1N2": ("N1", "N2"),
        "N1N2N3": ("N1", "N2", "N3"),
        "N1N2N3N5": ("N1", "N2", "N3", "N5"),
    }
    for initial_name in args.initials:
        baseline_pair, baseline_diagnostics = _run_variant(
            initial_name,
            tuple(),
            initial_sequence,
            edge_value,
            candidate_mask,
            data,
            args,
        )
        baseline_summary, baseline_rows = _controller_summary(
            baseline_pair, data, args)
        initial_metrics[initial_name] = {
            "metrics": baseline_summary,
            "diagnostics": baseline_diagnostics,
        }
        outputs[f"pairs_{initial_name}_initial"] = baseline_pair.astype(
            np.uint8)
        for row in baseline_rows:
            episode_rows.append({
                "controller": f"{initial_name}_initial", **row})
        for neighborhood_name, neighborhoods in neighborhood_sets.items():
            name = f"{initial_name}_{neighborhood_name}"
            pair, diagnostics = _run_variant(
                initial_name,
                neighborhoods,
                initial_sequence,
                edge_value,
                candidate_mask,
                data,
                args,
            )
            summary, rows = _controller_summary(pair, data, args)
            worst_gap = float(reference_summary["worst"]) - float(
                summary["worst"])
            baseline_worst = float(baseline_summary["worst"])
            denominator = float(reference_summary["worst"]) - baseline_worst
            recovery = (
                (float(summary["worst"]) - baseline_worst)
                / denominator
                if denominator > 1.0e-12 else 1.0
            )
            checks = {
                "worst_at_least_052": float(summary["worst"]) >= 0.52,
                "worst_reference_gap": worst_gap <= args.max_reference_gap,
                "gap_recovery": recovery >= args.min_gap_recovery,
                "cvar_non_decreasing": (
                    float(summary["cvar"]) >= float(baseline_summary["cvar"])
                    - 1.0e-12),
                "objective_monotonic": bool(
                    diagnostics["objective_monotonic"]),
                "round_budget": float(diagnostics["accepted_moves_mean"])
                <= args.max_mean_moves,
            }
            variants[name] = {
                "initial": initial_name,
                "neighborhoods": list(neighborhoods),
                "metrics": summary,
                "diagnostics": diagnostics,
                "reference_worst_gap": worst_gap,
                "gap_recovery": recovery,
                "gate_c1_5": {
                    "checks": checks,
                    "pass": bool(all(checks.values())),
                },
            }
            outputs[f"pairs_{name}"] = pair.astype(np.uint8)
            for row in rows:
                episode_rows.append({"controller": name, **row})
            print(name, json.dumps(variants[name]), flush=True)
    result = {
        "protocol": "gate_c1_5_oracle_local_search_v1",
        "seeds": [int(value) for value in np.unique(data["seed"])],
        "rounds": args.rounds,
        "initial_factor_graph": initial_summary,
        "initial_metrics": initial_metrics,
        "replicated_teacher_reference": reference_summary,
        "variants": variants,
        "fresh_test_consumed": False,
        "runtime_s": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "per_frame.npz",
        episode=data["episode"],
        seed=data["seed"],
        frame=data["frame"],
        **outputs,
    )
    with (args.output_dir / "episode_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_teacher_trace_selection20/teacher_trace.npz"))
    parser.add_argument(
        "--candidate", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_candidate_value_lu4_lq4_r1_gate_a2_holdout10/per_frame.npz"))
    parser.add_argument(
        "--initial-artifact", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_factor_graph_gate_c1_screen/per_frame.npz"))
    parser.add_argument("--initial-key", type=str, default="eval_prediction")
    parser.add_argument(
        "--reference-artifact", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_replicated_consensus_holdout10/per_frame.npz"))
    parser.add_argument(
        "--reference-key", type=str, default="pairs_replicated_consensus")
    parser.add_argument(
        "--student-checkpoint", type=Path, default=Path(
            "results/architecture_v2_teacher_cleanreset_trace_gate100/frozen_structure_student_endpoint8.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_oracle_local_search_gate_c1_5"))
    parser.add_argument(
        "--initials", nargs="+", default=["factor_graph", "role_first", "previous"])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-reference-gap", type=float, default=0.02)
    parser.add_argument("--min-gap-recovery", type=float, default=0.80)
    parser.add_argument("--max-mean-moves", type=float, default=8.0)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    parser.add_argument("--steady-window", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
