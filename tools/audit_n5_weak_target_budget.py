#!/usr/bin/env python
"""Audit nested weak-target budgets for one same-state local-exchange event.

This is an offline headroom experiment.  It reconstructs the exact no-op
controller trajectory from a recorded development trace, then evaluates every
unique atomic N5 candidate with common random numbers.  Budget ``B`` means
that singleton/two-target blocks may be drawn from the ``B`` weakest public
proxy targets; it does not enlarge the atomic target block beyond two.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import load_config
from tools.audit_n5_frozen_controller import (
    _actions,
    _observation_slices,
    _submit_frozen_controller,
    _trace_rows,
)
from uav_isac.agents.frozen_structure_student import FrozenStructureStudent
from uav_isac.coordination.dynamic_local_search import (
    DynamicLocalSearchCoordinator,
)
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.evaluation.n5_counterfactual_audit import (
    audit_atomic_n5_event,
)


def _parse_budgets(text: str, num_targets: int) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(text).split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        value = num_targets if normalized in {"all", "q"} else int(normalized)
        values.append(min(max(1, value), num_targets))
    if not values:
        raise ValueError("at least one weak-target budget is required")
    return tuple(sorted(set(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--ranker-checkpoint", required=True)
    parser.add_argument("--factor-graph-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--event-frame", type=int, required=True)
    parser.add_argument("--weak-target-budgets", default="1,2,3,all")
    parser.add_argument(
        "--neighborhood",
        choices=("N5", "N6"),
        default="N5",
        help=(
            "N5 changes roles and may require a dependency closure; N6 keeps "
            "roles fixed and exchanges owner/support on one/two targets"
        ),
    )
    parser.add_argument("--neighbor-topk", type=int, default=4)
    parser.add_argument("--target-topk", type=int, default=4)
    parser.add_argument("--coverage-fraction", type=float, default=0.30)
    parser.add_argument("--cold-rounds", type=int, default=8)
    parser.add_argument("--warm-rounds", type=int, default=8)
    parser.add_argument("--warm-top-m", type=int, default=5)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--p-d-floor", type=float, default=0.60)
    parser.add_argument(
        "--n5-rebuild-scope",
        choices=("global", "target_block"),
        default="global",
        help=(
            "global preserves the historical full rebuild; target_block "
            "freezes all outside-target edges for a genuine one/two-target "
            "dependency closure"
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    budgets = _parse_budgets(args.weak_target_budgets, int(cfg.scenario.Q))
    student = FrozenStructureStudent(args.student_checkpoint)
    trace_path = Path(args.trace)
    with np.load(trace_path, allow_pickle=False) as trace:
        rows = _trace_rows(trace, args.seed)
        missing = [
            frame for frame in range(1, int(args.event_frame) + 1)
            if frame not in rows
        ]
        if missing:
            raise ValueError(f"trace is missing prefix frames: {missing}")

        env = UAVISACEnv(config=cfg, seed=int(args.seed))
        env.core.configure_dynamic_local_search(
            DynamicLocalSearchCoordinator(
                "hybrid",
                ranker_checkpoint=args.ranker_checkpoint,
                cold_initializer="factor_graph",
                factor_graph_checkpoint=args.factor_graph_checkpoint,
                cold_rounds=max(0, int(args.cold_rounds)),
                warm_rounds=max(0, int(args.warm_rounds)),
                warm_top_m=max(1, int(args.warm_top_m)),
                rebootstrap_mode="off",
            )
        )
        slices = _observation_slices(env)
        observations, _ = env.reset(seed=int(args.seed))
        qos_floor = float(getattr(
            cfg.marl, "comm_qos_worst_min", 0.60))
        pre_step_env = None
        baseline_info = None
        event_actions = None
        observation_max_error = 0.0
        physical_pd_max_error = 0.0
        try:
            for frame in range(1, int(args.event_frame) + 1):
                row = rows[frame]
                actual_obs = np.stack([
                    observations[str(k)] for k in range(env.core.K)])
                observation_max_error = max(
                    observation_max_error,
                    float(np.max(np.abs(
                        actual_obs - np.asarray(
                            trace["local_obs"][row], dtype=np.float64)))),
                )
                _submit_frozen_controller(
                    env,
                    trace,
                    row,
                    student,
                    slices,
                    neighbor_topk=args.neighbor_topk,
                    target_topk=args.target_topk,
                    coverage_fraction=args.coverage_fraction,
                    qos_floor=qos_floor,
                )
                actions = _actions(trace, row)
                if frame == int(args.event_frame):
                    pre_step_env = deepcopy(env)
                    event_actions = actions
                observations, _, terminated, truncated, info = env.step(actions)
                physical_pd_max_error = max(
                    physical_pd_max_error,
                    float(np.max(np.abs(
                        np.asarray(info["P_D_q"], dtype=np.float64)
                        - np.asarray(
                            trace["physical_pd"][row], dtype=np.float64)))),
                )
                if frame == int(args.event_frame):
                    baseline_info = info
                if (terminated.get("__all__", False)
                        or truncated.get("__all__", False)):
                    raise RuntimeError("baseline ended before the event")

            if pre_step_env is None or baseline_info is None or event_actions is None:
                raise RuntimeError("failed to capture the event state")
            event = audit_atomic_n5_event(
                pre_step_env,
                env,
                event_actions,
                baseline_info,
                episode_seed=int(args.seed),
                frame=int(args.event_frame),
                target_mode="both",
                weak_target_budgets=budgets,
                steady_floor=float(args.steady_floor),
                weak3_floor=float(args.weak3_floor),
                p_d_floor=float(args.p_d_floor),
                n5_rebuild_scope=str(args.n5_rebuild_scope),
                neighborhood=str(args.neighborhood),
            )
        finally:
            if pre_step_env is not None:
                pre_step_env.close()
            env.close()

    payload = {
        "schema_version": 1,
        "protocol": "local_exchange_nested_weak_target_budget_crn_v2",
        "scope": (
            "development event, exact no-op-controller prefix replay, "
            "same-state one-frame common-random-number physical audit"
        ),
        "budget_definition": (
            "B selects the B weakest public-proxy targets as the universe "
            "for singleton/two-target atomic N5 blocks; B=Q is the diagnostic "
            "all-target-universe ceiling"
        ),
        "n5_rebuild_scope": str(args.n5_rebuild_scope),
        "neighborhood": str(args.neighborhood),
        "prefix_replay": {
            "observation_max_error": float(observation_max_error),
            "physical_pd_max_error": float(physical_pd_max_error),
        },
        "event": event,
        "limitations": [
            "B=Q is an offline diagnostic ceiling, not a claimed low-cost deployment mode.",
            "This gate tests same-frame structural headroom; closed-loop persistence is a separate gate.",
            "The physical Oracle is used only to label headroom and cannot be exposed to the deployed agents.",
        ],
        "fresh_test_consumed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "seed": int(args.seed),
        "event_frame": int(args.event_frame),
        "budgets": list(budgets),
        "prefix_replay": payload["prefix_replay"],
        "budget_summaries": event["weak_target_budget_summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
