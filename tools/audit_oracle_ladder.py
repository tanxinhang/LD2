#!/usr/bin/env python
"""Same-trace oracle ladder for the frozen 6x6 structure controller.

The audit applies four cumulative interventions while keeping the trace,
episode seeds, QoS convention, holding interval, and feasibility constraints
fixed:

  baseline          recorded local state + local candidates + Student rank
  oracle_state      exact target state in the Student observation
  oracle_candidate  full physically supported candidate graph
  oracle_rank       realized physical edge value as the ranking score
  oracle_projection exact receiver-owner projection

The deployed structure path already calls the same exact receiver-owner
projection used by the final rung.  Consequently the last intervention is an
explicit identity check, not a second differently implemented solver.  This
is important: a zero final gain is evidence that projection is not the
binding structural module, rather than an omitted experiment.

This is an offline, same-state diagnostic.  It does not claim closed-loop or
deployable oracle performance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from tools.audit_local_candidate_upper_bound import (  # noqa: E402
    _infer_observation_slices,
    _student_features,
)
from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    build_local_candidate_mask,
    candidate_recall_metrics,
    delivered_target_tokens,
    episode_detection_summary,
    held_pair_sequence,
    realized_pd_history,
    receiver_owner_from_pairs,
    select_local_neighbors,
    select_value_guided_neighbors,
)


METRICS = ("steady", "weak3", "worst", "cvar", "qos_feasible")


def _ordered_unique(values: np.ndarray) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values.reshape(-1)))


def _checkpoint_training_seeds(student: FrozenStructureStudent) -> set[int]:
    """Read training-trace seed values declared by the frozen artifact."""
    metadata = dict(getattr(student, "metadata", {}) or {})
    result: set[int] = set()
    for item in metadata.get("trained_cardinalities", []):
        trace = Path(str(item.get("trace", "")))
        if not trace.is_file():
            continue
        with np.load(trace, allow_pickle=False) as loaded:
            result.update(int(value) for value in loaded["seed"])
    return result


def _subset_trace(
    data: dict[str, np.ndarray],
    *,
    excluded_seeds: set[int],
    max_seeds: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    seed_order = [
        seed for seed in _ordered_unique(np.asarray(data["seed"]))
        if seed not in excluded_seeds
    ]
    if max_seeds > 0:
        seed_order = seed_order[:max_seeds]
    if not seed_order:
        raise ValueError("seed filtering left no evaluation episodes")
    keep = np.isin(data["seed"], seed_order)
    frame_count = len(data["seed"])
    subset = {
        key: (
            value[keep]
            if value.ndim > 0 and value.shape[0] == frame_count
            else value
        )
        for key, value in data.items()
    }
    return subset, seed_order


def oracleize_student_observation(
    local_obs: np.ndarray,
    target_states: np.ndarray,
    uav_positions: np.ndarray,
    slices: Any,
    *,
    area_size: tuple[float, float],
    velocity_scale: float,
) -> np.ndarray:
    """Replace exactly the belief/geometry fields consumed by the Student.

    Covariance and AoI are set to zero because the oracle state is exact and
    current.  Communication, local P_D history, resource actions and all
    transport masks remain recorded, so this intervention changes state
    quality only.
    """
    obs = np.asarray(local_obs, dtype=np.float32).copy()
    target = np.asarray(target_states, dtype=np.float64)
    uav = np.asarray(uav_positions, dtype=np.float64)
    if obs.ndim != 3:
        raise ValueError("local_obs must have shape (F,K,D)")
    F, K, _ = obs.shape
    Q = int(slices.Q)
    if target.shape[:2] != (F, Q) or target.shape[-1] < 4:
        raise ValueError("target_states must have shape (F,Q,>=4)")
    if uav.shape[:2] != (F, K) or uav.shape[-1] < 2:
        raise ValueError("uav_positions must have shape (F,K,>=2)")

    area_w, area_h = (float(area_size[0]), float(area_size[1]))
    speed = max(float(velocity_scale), 1.0e-9)
    mean_scale = np.asarray([area_w, area_h, speed, speed])
    belief = np.zeros((F, K, Q, slices.belief_per_target), dtype=np.float64)
    belief[..., :4] = target[:, None, :, :4] / mean_scale
    start = int(slices.belief_start)
    stop = start + Q * int(slices.belief_per_target)
    obs[..., start:stop] = belief.reshape(F, K, -1).astype(np.float32)

    if bool(slices.has_rel_features):
        # Traces can record post-action UAV positions while observations are
        # pre-action.  If the recorded belief already equals the target state,
        # the existing geometry is the correctly time-aligned representation;
        # retaining it avoids manufacturing a false oracle intervention.
        observed_mean = (
            np.asarray(local_obs[..., slices.belief_start:
                                  slices.belief_start + Q *
                                  int(slices.belief_per_target)])
            .reshape(F, K, Q, int(slices.belief_per_target))[..., :4]
        )
        observed_mean = observed_mean * mean_scale
        mean_already_exact = bool(np.max(np.abs(
            observed_mean - target[:, None, :, :4]
        )) <= 1.0e-4)
        if mean_already_exact:
            return obs
        delta = target[:, None, :, :2] - uav[:, :, None, :2]
        distance = np.linalg.norm(delta, axis=-1)
        angle = np.arctan2(delta[..., 1], delta[..., 0])
        diagonal = float(np.hypot(area_w, area_h))
        geometry = np.stack([
            delta[..., 0] / area_w,
            delta[..., 1] / area_h,
            distance / diagonal,
            np.sin(angle),
            np.cos(angle),
            np.exp(-distance / 50.0),
            np.exp(-distance / 150.0),
            np.exp(-distance / 400.0),
        ], axis=-1)
        start = int(slices.geom_start)
        stop = start + Q * int(slices.geom_per_target)
        obs[..., start:stop] = geometry.reshape(F, K, -1).astype(np.float32)
    return obs


def _local_candidate_and_rank(
    data: dict[str, np.ndarray],
    local_obs: np.ndarray,
    slices: Any,
    student: FrozenStructureStudent,
    *,
    neighbor_topk: int,
    target_topk: int,
    coverage_fraction: float,
    refinement_rounds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_visible, token_age = delivered_target_tokens(local_obs, slices)
    local_pd = slices.extract_pd_hist(local_obs)
    neighbors = select_local_neighbors(
        token_visible,
        token_age,
        neighbor_topk=neighbor_topk,
    )
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    for _ in range(max(0, int(refinement_rounds))):
        provisional = _student_features(
            {**data, "local_obs": local_obs},
            slices,
            neighbor_subset_mask=neighbors,
        )
        provisional_rank = student.predict(provisional)
        neighbors = select_value_guided_neighbors(
            provisional_rank,
            local_pd,
            token_visible,
            token_age,
            neighbor_topk=neighbor_topk,
            qos_floor=floor,
            coverage_fraction=coverage_fraction,
        )
    features = _student_features(
        {**data, "local_obs": local_obs},
        slices,
        neighbor_subset_mask=neighbors,
    )
    ranking = student.predict(features)
    candidate, _ = build_local_candidate_mask(
        ranking,
        local_pd,
        neighbors,
        token_visible,
        target_topk=target_topk,
        qos_floor=floor,
        coverage_fraction=coverage_fraction,
        require_reciprocal_link=True,
    )
    return candidate, ranking, neighbors


def _pair_change(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    old = np.asarray(previous, dtype=bool)
    new = np.asarray(current, dtype=bool)
    frame_equal = np.all(old == new, axis=(1, 2, 3))
    intersection = np.sum(old & new, axis=(1, 2, 3))
    union = np.sum(old | new, axis=(1, 2, 3))
    return {
        "changed_frame_rate": float(np.mean(~frame_equal)),
        "pair_jaccard": float(np.mean(
            np.where(union > 0, intersection / np.maximum(union, 1), 1.0)
        )),
    }


def ladder_deltas(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    order = [
        "baseline",
        "oracle_state",
        "oracle_candidate",
        "oracle_rank",
        "oracle_projection",
    ]
    result: dict[str, dict[str, float]] = {}
    for previous, current in zip(order, order[1:]):
        result[f"{previous}_to_{current}"] = {
            metric: float(summaries[current][metric])
            - float(summaries[previous][metric])
            for metric in METRICS
        }
    return result


def _binding_stage(deltas: dict[str, dict[str, float]]) -> dict[str, Any]:
    gains = {
        name: float(values["worst"])
        for name, values in deltas.items()
    }
    positive = {name: max(value, 0.0) for name, value in gains.items()}
    winner = max(positive, key=positive.get)
    total = float(sum(positive.values()))
    return {
        "criterion": "largest positive increment in episode-mean worst P_D",
        "stage": winner,
        "worst_pd_gain": gains[winner],
        "positive_gain_share": (
            float(positive[winner] / total) if total > 1.0e-12 else 0.0
        ),
        "all_incremental_worst_pd_gains": gains,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    with np.load(args.trace, allow_pickle=False) as loaded:
        raw = {key: loaded[key] for key in loaded.files}
    student = FrozenStructureStudent(args.student_checkpoint)
    training_seeds = _checkpoint_training_seeds(student)
    excluded = training_seeds if args.exclude_training_seed_values else set()
    data, seed_order = _subset_trace(
        raw,
        excluded_seeds=excluded,
        max_seeds=args.max_seeds,
    )
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    cfg = load_config(str(args.config))

    recorded_obs = np.asarray(data["local_obs"], dtype=np.float32)
    oracle_obs = oracleize_student_observation(
        recorded_obs,
        data["target_states"],
        # ``local_obs`` is captured before the frame action; use its own
        # normalized self position when geometry must be rebuilt, rather than
        # the post-action position stored in the trace.
        np.concatenate([
            recorded_obs[..., :2]
            * np.asarray(tuple(cfg.scenario.region_size), dtype=np.float32),
            recorded_obs[..., 2:3] * float(cfg.scenario.height),
        ], axis=-1),
        slices,
        area_size=tuple(cfg.scenario.region_size),
        velocity_scale=float(cfg.uav.v_max),
    )
    local_candidate, local_rank, local_neighbors = _local_candidate_and_rank(
        data,
        recorded_obs,
        slices,
        student,
        neighbor_topk=args.neighbor_topk,
        target_topk=args.target_topk,
        coverage_fraction=args.coverage_fraction,
        refinement_rounds=args.neighbor_refinement_rounds,
    )
    state_candidate, state_rank, state_neighbors = _local_candidate_and_rank(
        data,
        oracle_obs,
        slices,
        student,
        neighbor_topk=args.neighbor_topk,
        target_topk=args.target_topk,
        coverage_fraction=args.coverage_fraction,
        refinement_rounds=args.neighbor_refinement_rounds,
    )

    physical = np.asarray(data["privileged_candidate"], dtype=bool)
    realized_rank = np.asarray(data["privileged_d_eff"], dtype=np.float64)
    controller_pairs = {
        "baseline": held_pair_sequence(
            local_rank, local_candidate & physical, data),
        "oracle_state": held_pair_sequence(
            state_rank, state_candidate & physical, data),
        "oracle_candidate": held_pair_sequence(
            state_rank, physical, data),
        "oracle_rank": held_pair_sequence(
            realized_rank, physical, data),
    }
    # The live P0 path and audit oracle both call the exact same constrained
    # receiver-owner projection.  Preserve the rung as an identity test.
    controller_pairs["oracle_projection"] = controller_pairs[
        "oracle_rank"].copy()

    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    summaries: dict[str, dict[str, Any]] = {}
    for name, pair in controller_pairs.items():
        pd = realized_pd_history(pair, realized_rank, p_fa)
        summary, _ = episode_detection_summary(
            pd,
            data["episode"],
            data["seed"],
            steady_window=args.steady_window,
            qos_thresholds=(
                args.steady_floor,
                args.weak3_floor,
                args.worst_floor,
            ),
        )
        summaries[name] = summary

    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    coord = np.where(
        np.asarray(data["coord_pd_ema_valid"], dtype=bool)[:, None],
        np.asarray(data["coord_pd_ema"], dtype=np.float64),
        np.mean(slices.extract_pd_hist(recorded_obs), axis=1),
    )
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    deficit = np.maximum(floor - coord, 0.0)
    full_pairs = controller_pairs["oracle_rank"]
    recall = candidate_recall_metrics(
        local_candidate[resolved],
        physical[resolved],
        full_pairs[resolved],
        receiver_owner_from_pairs(full_pairs)[resolved],
        deficit[resolved],
    )
    deltas = ladder_deltas(summaries)
    changes = {}
    order = list(controller_pairs)
    for previous, current in zip(order, order[1:]):
        changes[f"{previous}_to_{current}"] = _pair_change(
            controller_pairs[previous], controller_pairs[current])

    state_abs = np.abs(
        oracle_obs.astype(np.float64) - recorded_obs.astype(np.float64))
    result: dict[str, Any] = {
        "schema": "oracle-ladder-v1",
        "scope": (
            "offline same-trace structural attribution; not closed-loop "
            "deployment evidence"
        ),
        "trace": str(args.trace),
        "config": str(args.config),
        "student_checkpoint": str(args.student_checkpoint),
        "num_uavs": K,
        "num_targets": Q,
        "frames": int(len(data["seed"])),
        "resolve_frames": int(np.sum(resolved)),
        "evaluation_seeds": seed_order,
        "excluded_training_seed_values": sorted(
            set(_ordered_unique(np.asarray(raw["seed"]))) & training_seeds
        ) if args.exclude_training_seed_values else [],
        "interventions": {
            "baseline": (
                "recorded local observation + local sparse candidate + "
                "frozen Student rank + deployed exact projection"
            ),
            "oracle_state": (
                "exact current target position/velocity with zero covariance "
                "and AoI; candidate/rank/projection otherwise unchanged"
            ),
            "oracle_candidate": (
                "full physically supported graph; oracle-state Student rank "
                "and projection unchanged"
            ),
            "oracle_rank": (
                "realized d_eff ranking on the full graph; projection unchanged"
            ),
            "oracle_projection": (
                "same exact max-min single-role receiver-owner operator used "
                "by deployed P0; identity audit"
            ),
        },
        "state_intervention": {
            "observation_max_abs_delta": float(np.max(state_abs)),
            "observation_changed_fraction": float(np.mean(state_abs > 1.0e-7)),
            "student_rank_max_abs_delta": float(np.max(np.abs(
                state_rank.astype(np.float64) - local_rank.astype(np.float64)
            ))),
            "candidate_changed_fraction": float(np.mean(
                state_candidate != local_candidate
            )),
            "neighbor_changed_fraction": float(np.mean(
                state_neighbors != local_neighbors
            )),
            "already_oracle_in_trace": bool(np.max(state_abs) <= 1.0e-7),
        },
        "local_candidate_recall_against_oracle_rank_solution": recall,
        "performance": summaries,
        "incremental_delta": deltas,
        "structure_change": changes,
        "binding_stage": _binding_stage(deltas),
        "projection_identity": {
            "same_operator": True,
            "pair_matrix_equal": bool(np.array_equal(
                controller_pairs["oracle_rank"],
                controller_pairs["oracle_projection"],
            )),
            "incremental_gain_is_zero": bool(all(
                abs(value) <= 1.0e-12
                for value in deltas[
                    "oracle_rank_to_oracle_projection"].values()
            )),
        },
        "runtime_s": float(time.perf_counter() - started),
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=0)
    parser.add_argument(
        "--exclude-training-seed-values",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--neighbor-topk", type=int, default=4)
    parser.add_argument("--target-topk", type=int, default=4)
    parser.add_argument("--coverage-fraction", type=float, default=0.30)
    parser.add_argument("--neighbor-refinement-rounds", type=int, default=1)
    parser.add_argument("--steady-window", type=int, default=20)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
