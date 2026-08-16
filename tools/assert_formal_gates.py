#!/usr/bin/env python
"""Batch Medium-gate assertion over the formal evaluation results.

Registry-driven re-assertion of every formal result cited in the docs, so
"Gate passed" is verifiable in one command instead of living only in
markdown tables.  Results whose seeds are still under quarantine are listed
but never asserted (they cannot be formal evidence until the banks are
regenerated -- see docs/KNOWN_ISSUES.md).

Usage:
    python tools/assert_formal_gates.py                # all results, table out
    python tools/assert_formal_gates.py --json-output x.json
Exit code 0 = every non-quarantined result cleared its gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

from tools.assert_gate_thresholds import assert_gate_from_csv

RESULTS_ROOT = "results/"


@dataclass(frozen=True)
class FormalResult:
    name: str
    csv: str
    qos_tol: float = 0.0
    quarantined: bool = False
    require_lcb: bool = False
    note: str = ""


FORMAL_RESULTS: List[FormalResult] = [
    FormalResult(
        name="4/4 frozen deployment (100 seeds, test split)",
        csv=("architecture_v2_structure_student_u2u_resolve_bw50k_adaptive_"
             "b4b8_failclosed_gate100/paired_eval.csv"),
        note="docs/CURRENT_SYSTEM_STATUS.md 4/4 正式版本"),
    FormalResult(
        name="4/4 anchor-preserving residual safety gate (10 seeds)",
        csv=("architecture_v2_structure_student_cardinality_residual46_"
             "bw50k_gate10/paired_eval.csv"),
    ),
    FormalResult(
        name="8/8 analytical stack e2e D0.95 (20 seeds)",
        csv=("architecture_v2_scale_k8q8_analytical_power_l0l1_movement/"
             "paired_eval.csv"),
        qos_tol=1e-6,
        note="0.60-floor float artifact (QoS 0.65 at tol=0), see D1_1A"),
    FormalResult(
        name="8/8 lexicographic L1 live (D1.1-A, 20 seeds)",
        csv="_d095_lex20/paired_eval.csv",
        note="live eval, same seeds/warm-start as D0.95 baseline"),
    FormalResult(
        name="8/8 lex + multi-candidate L3 live (D1.1-B, 20 seeds)",
        csv="_d095_lexcand20/paired_eval.csv",
        note="deployment candidate baseline (D1.1-B, 0.975)"),
    FormalResult(
        name="8/8 lex + multi-candidate L3 (D1.1-B, 20 seeds) -- LCB enforced",
        csv="_d095_lexcand20/paired_eval.csv",
        require_lcb=True,
        note="audit-recommended Wilson LCB standard"),
    FormalResult(
        name="6/6 cardinality residual (10 seeds) -- QUARANTINED",
        csv=("architecture_v2_scale_k6q6_structure_student_cardinality_"
             "residual46_gate10/paired_eval.csv"),
        quarantined=True,
        note=("test split 前 10 个种子含全部 5 个隔离种子；bank 回填前不得作为"
              "正式证据 (docs/KNOWN_ISSUES.md)"),
    ),
    FormalResult(
        name="6/6 cross-scale student (10 seeds) -- QUARANTINED",
        csv=("architecture_v2_scale_k6q6_structure_student_adaptive_b4b8_"
             "gate10/paired_eval.csv"),
        quarantined=True,
        note="同 980_k6q6 test split 污染；不可作为正式证据",
    ),
    FormalResult(
        name="8/8 frozen deployment candidate D1.5 blind (100 seeds)",
        csv="_d1_5_blind100/paired_eval.csv",
        note=("100 全新 blind seed 认证：QoS 点估计 0.730 过门（advice 013），"
              "docs/OPTIMIZATION_LOG.md D1.5"),
    ),
    FormalResult(
        name="8/8 frozen deployment candidate D1.5 blind (100 seeds) -- LCB enforced",
        csv="_d1_5_blind100/paired_eval.csv",
        require_lcb=True,
        note=("Wilson LCB 0.636 < 0.70：N=100 统计功效不足，须在论文中如实披露"
              "（点估计 + 置信区间口径）"),
    ),
]


def assert_formal_gates(
    results: Optional[List[FormalResult]] = None,
    require_lcb: bool = False,
) -> Dict[str, Dict[str, object]]:
    results = results if results is not None else FORMAL_RESULTS
    report: Dict[str, Dict[str, object]] = {}
    for entry in results:
        item: Dict[str, object] = {"name": entry.name,
                                   "quarantined": entry.quarantined}
        if entry.quarantined:
            item["status"] = "QUARANTINED"
            item["note"] = entry.note
            report[entry.name] = item
            continue
        path = entry.csv if os.path.isabs(entry.csv) else RESULTS_ROOT + entry.csv
        try:
            aggregates = assert_gate_from_csv(
                path, qos_tol=entry.qos_tol,
                require_wilson_lcb=require_lcb or entry.require_lcb)
            item["status"] = "PASS"
            item.update(aggregates)
        except (AssertionError, ValueError, OSError) as exc:
            item["status"] = "FAIL"
            item["error"] = str(exc).splitlines()[0]
        item["note"] = entry.note
        report[entry.name] = item
    return report


def render_table(report: Dict[str, Dict[str, object]]) -> str:
    lines = ["| result | status | steady | weak3 | worst | QoS | LCB |",
             "|---|---|---|---|---|---|---|"]
    for name, item in report.items():
        if item["status"] == "QUARANTINED":
            lines.append(f"| {name} | **QUARANTINED** | n/a | n/a | n/a | n/a | n/a |")
        elif item["status"] == "FAIL":
            lines.append(f"| {name} | **FAIL** | n/a | n/a | n/a | n/a | n/a | "
                         f"({item.get('error', '')[:60]})")
        else:
            lines.append(
                f"| {name} | PASS | {item['steady']:.4f} | {item['weak3']:.4f} "
                f"| {item['worst']:.4f} | {item['qos_feasible']:.2f} "
                f"| {item['qos_wilson_lcb']:.3f} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-lcb", action="store_true",
                        help="also require the QoS Wilson LCB to clear 0.70")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args(argv)

    report = assert_formal_gates(require_lcb=args.require_lcb)
    print(render_table(report))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    failed = [n for n, item in report.items() if item["status"] == "FAIL"]
    if failed:
        print(f"\nFAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
