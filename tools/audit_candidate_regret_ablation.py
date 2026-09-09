#!/usr/bin/env python
"""Orthogonal candidate-provenance × task-regret-rank ablation.

The candidate mask is generated once from the deployed CE Student and then
held fixed while rankers change.  This makes the 2x2 comparison causal:

  candidate pool: local sparse vs full physical support
  ranker:         CE Student vs task-regret Student

All four controllers use the same trace, exact receiver-owner projection,
hold interval and realized physical deflection.  The full physical pool is an
oracle upper-bound condition; this tool is an offline same-state diagnostic.
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

from tools.audit_oracle_ladder import (  # noqa: E402
    _checkpoint_training_seeds,
    _local_candidate_and_rank,
    _subset_trace,
    _infer_observation_slices,
)
from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    candidate_recall_metrics,
    episode_detection_summary,
    held_pair_sequence,
    realized_pd_history,
    receiver_owner_from_pairs,
)
from uav_isac.evaluation.episode_regret import (  # noqa: E402
    candidate_support_by_episode,
    episode_qos_regret,
)


METRICS = ("steady", "weak3", "worst", "cvar", "qos_feasible")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    with np.load(args.trace, allow_pickle=False) as loaded:
        raw = {key: loaded[key] for key in loaded.files}
    ce = FrozenStructureStudent(args.ce_checkpoint)
    regret = FrozenStructureStudent(args.task_regret_checkpoint)
    training_seeds = (
        _checkpoint_training_seeds(ce)
        | _checkpoint_training_seeds(regret)
    )
    data, seeds = _subset_trace(
        raw,
        excluded_seeds=training_seeds if args.exclude_training_seed_values
        else set(),
        max_seeds=args.max_seeds,
    )
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    local_candidate, ce_rank, neighbors = _local_candidate_and_rank(
        data,
        np.asarray(data["local_obs"], dtype=np.float32),
        slices,
        ce,
        neighbor_topk=args.neighbor_topk,
        target_topk=args.target_topk,
        coverage_fraction=args.coverage_fraction,
        refinement_rounds=args.neighbor_refinement_rounds,
    )
    # Rank features are computed on the same full local view for both rankers;
    # the CE-derived sparse candidate mask is deliberately not recomputed.
    from uav_isac.agents.frozen_structure_student import (
        build_structure_student_features,
    )
    features = build_structure_student_features(
        np.asarray(data["local_obs"], dtype=np.float32),
        slices,
        outgoing_message=data["outgoing_message"],
        outgoing_token_mask=data["outgoing_token_mask"],
        outgoing_rate=data["outgoing_rate"],
        comm_fraction=data["comm_fraction"],
        sensing_weights=data["sensing_weights"],
        rate_scale=max(float(np.max(data["outgoing_rate"])), 1.0),
    )
    regret_rank = regret.predict(features)
    physical = np.asarray(data["privileged_candidate"], dtype=bool)
    pair_by_condition = {
        "local_ce": held_pair_sequence(
            ce_rank, local_candidate & physical, data),
        "full_ce": held_pair_sequence(
            ce_rank, physical, data),
        "local_task_regret": held_pair_sequence(
            regret_rank, local_candidate & physical, data),
        "full_task_regret": held_pair_sequence(
            regret_rank, physical, data),
        "full_oracle_rank": held_pair_sequence(
            np.asarray(data["privileged_d_eff"], dtype=np.float64),
            physical,
            data,
        ),
    }
    p_fa = float(np.asarray(data["p_fa"]).reshape(-1)[0])
    summaries: dict[str, dict[str, Any]] = {}
    episode_rows: dict[str, list[dict[str, Any]]] = {}
    for name, pairs in pair_by_condition.items():
        pd = realized_pd_history(
            pairs,
            np.asarray(data["privileged_d_eff"], dtype=np.float64),
            p_fa,
        )
        summaries[name], episode_rows[name] = episode_detection_summary(
            pd,
            data["episode"],
            data["seed"],
            steady_window=args.steady_window,
            qos_thresholds=(args.steady_floor, args.weak3_floor,
                            args.worst_floor),
        )

    resolved = np.asarray(data["p0_resolved"], dtype=bool)
    floor = float(np.asarray(data["qos_floor"]).reshape(-1)[0])
    coord = np.where(
        np.asarray(data["coord_pd_ema_valid"], dtype=bool)[:, None],
        np.asarray(data["coord_pd_ema"], dtype=np.float64),
        np.mean(slices.extract_pd_hist(data["local_obs"]), axis=1),
    )
    recall = candidate_recall_metrics(
        local_candidate[resolved],
        physical[resolved],
        pair_by_condition["full_oracle_rank"][resolved],
        receiver_owner_from_pairs(
            pair_by_condition["full_oracle_rank"]
        )[resolved],
        np.maximum(floor - coord, 0.0)[resolved],
    )

    def difference(left: str, right: str) -> dict[str, float]:
        return {
            metric: float(summaries[right][metric])
            - float(summaries[left][metric])
            for metric in METRICS
        }

    candidate_at_ce = difference("local_ce", "full_ce")
    candidate_at_regret = difference("local_task_regret", "full_task_regret")
    regret_at_local = difference("local_ce", "local_task_regret")
    regret_at_full = difference("full_ce", "full_task_regret")
    interaction = {
        metric: float(candidate_at_regret[metric]
                     - candidate_at_ce[metric])
        for metric in METRICS
    }
    # Compare every learned condition with the realized-d_eff reference.  The
    # support map is strict and episode-level: a single missing reference edge
    # on any resolved frame marks that episode unsupported.  This is an audit
    # decomposition, not a training target or a causal proof of rank failure.
    oracle_rows = episode_rows["full_oracle_rank"]
    local_support = candidate_support_by_episode(
        local_candidate & physical,
        pair_by_condition["full_oracle_rank"],
        np.asarray(data["episode"], dtype=np.int64),
        resolved=resolved,
    )
    full_support = candidate_support_by_episode(
        physical,
        pair_by_condition["full_oracle_rank"],
        np.asarray(data["episode"], dtype=np.int64),
        resolved=resolved,
    )
    episode_regret = {
        "local_ce": episode_qos_regret(
            oracle_rows, episode_rows["local_ce"],
            candidate_supported=local_support),
        "full_ce": episode_qos_regret(
            oracle_rows, episode_rows["full_ce"],
            candidate_supported=full_support),
        "local_task_regret": episode_qos_regret(
            oracle_rows, episode_rows["local_task_regret"],
            candidate_supported=local_support),
        "full_task_regret": episode_qos_regret(
            oracle_rows, episode_rows["full_task_regret"],
            candidate_supported=full_support),
    }

    result: dict[str, Any] = {
        "schema": "candidate-regret-ablation-v1",
        "scope": "offline same-trace orthogonal ablation; not deployment evidence",
        "trace": str(args.trace),
        "ce_checkpoint": str(args.ce_checkpoint),
        "task_regret_checkpoint": str(args.task_regret_checkpoint),
        "frames": int(len(data["seed"])),
        "resolve_frames": int(np.sum(resolved)),
        "evaluation_seeds": seeds,
        "excluded_training_seed_values": sorted(
            set(int(v) for v in np.unique(raw["seed"])) & training_seeds
        ) if args.exclude_training_seed_values else [],
        "factor_design": {
            "candidate_pool": {
                "local": "CE Student sparse candidate, held fixed for both rankers",
                "full": "privileged physical support graph (oracle upper bound)",
            },
            "ranker": {
                "ce": "multiscale CE Student",
                "task_regret": "task-regret Student",
                "oracle": "realized d_eff (reference only)",
            },
        },
        "candidate_recall_against_oracle_rank": recall,
        "performance": summaries,
        "episode_rows": episode_rows,
        "episode_qos_regret_vs_full_oracle": episode_regret,
        "contrasts": {
            "candidate_gain_at_ce_rank": candidate_at_ce,
            "candidate_gain_at_task_regret_rank": candidate_at_regret,
            "task_regret_gain_at_local_candidate": regret_at_local,
            "task_regret_gain_at_full_candidate": regret_at_full,
            "oracle_rank_headroom_at_full_candidate": difference(
                "full_task_regret", "full_oracle_rank"),
        },
        "candidate_rank_interaction": interaction,
        "structure_change": {
            "local_ce_to_full_ce": {
                "changed_frame_rate": float(np.mean(
                    np.any(pair_by_condition["local_ce"]
                           != pair_by_condition["full_ce"], axis=(1, 2, 3))))
            },
            "local_task_regret_to_full_task_regret": {
                "changed_frame_rate": float(np.mean(
                    np.any(pair_by_condition["local_task_regret"]
                           != pair_by_condition["full_task_regret"],
                           axis=(1, 2, 3))))
            },
            "local_ce_to_local_task_regret": {
                "changed_frame_rate": float(np.mean(
                    np.any(pair_by_condition["local_ce"]
                           != pair_by_condition["local_task_regret"],
                           axis=(1, 2, 3))))
            },
            "full_ce_to_full_task_regret": {
                "changed_frame_rate": float(np.mean(
                    np.any(pair_by_condition["full_ce"]
                           != pair_by_condition["full_task_regret"],
                           axis=(1, 2, 3))))
            },
        },
        "interpretation_guard": (
            "candidate and rank contrasts are orthogonal only because the local "
            "mask is generated once from CE and then held fixed"
        ),
        "runtime_s": float(time.perf_counter() - started),
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--ce-checkpoint", type=Path, required=True)
    parser.add_argument("--task-regret-checkpoint", type=Path, required=True)
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
