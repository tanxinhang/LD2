#!/usr/bin/env python
"""Audit a componentwise hold-Pareto no-regret structure gate.

This is a noncausal upper-bound experiment.  Future realized detection is
used only to decide whether an exact-feasible proposal Pareto-dominates the
baseline over its complete hold segment.  It tests whether such a label has
enough value to justify learning a causal acceptance gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_causal_hold_qos_rank import (  # noqa: E402
    _pair_summary,
    _subset,
)
from tools.audit_hold_qos_rank import hold_aware_score  # noqa: E402
from tools.audit_local_candidate_upper_bound import (  # noqa: E402
    _infer_observation_slices,
)
from tools.audit_oracle_ladder import _local_candidate_and_rank  # noqa: E402
from uav_isac.agents.frozen_structure_student import (  # noqa: E402
    FrozenStructureStudent,
)
from uav_isac.evaluation.local_candidate_audit import (  # noqa: E402
    held_pair_sequence,
)
from uav_isac.evaluation.pareto_no_regret import (  # noqa: E402
    componentwise_hold_pareto_gate,
)


def _metric_delta(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    return {
        key: float(candidate["summary"][key]) - float(baseline["summary"][key])
        for key in ("steady", "weak3", "worst", "cvar", "qos_feasible")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.trace, allow_pickle=False) as loaded:
        raw = {key: loaded[key] for key in loaded.files}
    ordered_seeds = list(dict.fromkeys(int(value) for value in raw["seed"]))
    seeds = ordered_seeds[-int(args.test_seeds):]
    data = _subset(raw, seeds)
    K = int(np.asarray(data["num_uavs"]).reshape(-1)[0])
    Q = int(np.asarray(data["num_targets"]).reshape(-1)[0])
    slices = _infer_observation_slices(data["local_obs"].shape[-1], K, Q)
    student = FrozenStructureStudent(args.student_checkpoint)
    local_candidate, baseline_rank, _ = _local_candidate_and_rank(
        data,
        np.asarray(data["local_obs"], dtype=np.float32),
        slices,
        student,
        neighbor_topk=args.neighbor_topk,
        target_topk=args.target_topk,
        coverage_fraction=args.coverage_fraction,
        refinement_rounds=args.neighbor_refinement_rounds,
    )
    physical = np.asarray(data["privileged_candidate"], dtype=bool)
    admitted = np.asarray(local_candidate, dtype=bool) & physical
    hold_score = hold_aware_score(
        data["privileged_d_eff"],
        data["episode"],
        data["p0_resolved"],
        tail_weight=1.0,
        quantile=args.quantile,
    )
    baseline_pairs = held_pair_sequence(baseline_rank, admitted, data)
    proposal_pairs = held_pair_sequence(hold_score, admitted, data)
    gated_pairs, gate = componentwise_hold_pareto_gate(
        data,
        baseline_pairs,
        proposal_pairs,
        tolerance=args.tolerance,
        minimum_gain=args.minimum_gain,
    )
    common = dict(
        steady_window=args.steady_window,
        steady_floor=args.steady_floor,
        weak3_floor=args.weak3_floor,
        worst_floor=args.worst_floor,
        frame_duration_s=args.frame_duration_s,
    )
    baseline = _pair_summary(data, baseline_pairs, **common)
    proposal = _pair_summary(data, proposal_pairs, **common)
    gated = _pair_summary(data, gated_pairs, **common)
    return {
        "schema": "componentwise-hold-pareto-no-regret-oracle-v1",
        "scope": (
            "noncausal label upper bound; future outcomes are forbidden at "
            "deployment; same local candidate and exact-feasible actions"
        ),
        "seeds": seeds,
        "definition": {
            "accept": (
                "mean_hold_PD(proposal,q) >= mean_hold_PD(baseline,q) - "
                "tolerance for every target q, and at least one target gains"
            ),
            "tolerance": float(args.tolerance),
            "minimum_gain": float(args.minimum_gain),
        },
        "gate": gate,
        "baseline": baseline,
        "ungated_hold_oracle": proposal,
        "pareto_gated_hold_oracle": gated,
        "ungated_delta": _metric_delta(proposal, baseline),
        "gated_delta": _metric_delta(gated, baseline),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-seeds", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--tolerance", type=float, default=1.0e-9)
    parser.add_argument("--minimum-gain", type=float, default=1.0e-6)
    parser.add_argument("--neighbor-topk", type=int, default=4)
    parser.add_argument("--target-topk", type=int, default=4)
    parser.add_argument("--coverage-fraction", type=float, default=0.30)
    parser.add_argument("--neighbor-refinement-rounds", type=int, default=1)
    parser.add_argument("--steady-window", type=int, default=20)
    parser.add_argument("--steady-floor", type=float, default=0.80)
    parser.add_argument("--weak3-floor", type=float, default=0.70)
    parser.add_argument("--worst-floor", type=float, default=0.60)
    parser.add_argument("--frame-duration-s", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "gate": report["gate"],
        "gated_delta": report["gated_delta"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
