#!/usr/bin/env python
"""Gate 0 (advice/016): summarize the V3-C0 re-certification A/B vs history.

Compares, per seed, the V3-C0-fixed-code certification (new) against the
historical pre-fix certification (old), on the SAME seeds:

  8/8: results/_gate0_8x8_v3c0_indep100/paired_eval.csv
       vs results/_d1_10_blind100_indep/paired_eval.csv
  6/6: results/_gate0_6x6_v3c0_indep100/paired_eval.csv
       vs results/_6x6_blind100_indep/paired_eval.csv

Reports QoS feasible rate + Wilson LCB for each arm, per-seed worst/steady
deltas, and the seed-level flip inventory (PASS->FAIL / FAIL->PASS / same).
Preserves the historical numbers as-is (never overwrites them).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.assert_gate_thresholds import wilson_lower  # noqa: E402

FLOORS = {"steady": 0.8, "weak3": 0.7, "worst": 0.6}


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "eval_episode_seeds" not in rows[0]:
        raise ValueError(f"{path} not ready (still running or missing eval)")
    r = rows[0]
    return {
        "seeds": [int(s) for s in ast.literal_eval(r["eval_episode_seeds"])],
        "steady": np.asarray(ast.literal_eval(r["eval_episode_steady_P_D"])),
        "weak3": np.asarray(ast.literal_eval(r["eval_episode_weak3_P_D"])),
        "worst": np.asarray(ast.literal_eval(r["eval_episode_worst_P_D"])),
        "qos": float(r["eval_qos_feasible_rate"]),
        "lcb": float(r["eval_qos_feasible_wilson_lcb"]),
    }


def _feasible(a: dict) -> np.ndarray:
    return np.logical_and.reduce([
        a["steady"] >= FLOORS["steady"],
        a["weak3"] >= FLOORS["weak3"],
        a["worst"] >= FLOORS["worst"],
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-8", required=True, type=Path)
    ap.add_argument("--old-8", required=True, type=Path)
    ap.add_argument("--new-6", default=None, type=Path)
    ap.add_argument("--old-6", default=None, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    report = {"schema": "gate0-v3c0-summary", "arms": {}}
    arms = [("8x8", args.new_8, args.old_8)]
    if args.new_6 is not None and args.old_6 is not None:
        arms.append(("6x6", args.new_6, args.old_6))
    for label, new_p, old_p in arms:
        new, old = _load(new_p), _load(old_p)
        # align on the historical seed order
        idx = {s: i for i, s in enumerate(old["seeds"])}
        order = [idx[s] for s in new["seeds"] if s in idx]
        missing = [s for s in new["seeds"] if s not in idx]
        old_feas = _feasible(old)[order]
        new_feas = _feasible(new)
        # flip inventory (only over seeds present in BOTH)
        n_paired = len(order)
        flipped_to_pass = int(np.sum(~old_feas & new_feas))
        flipped_to_fail = int(np.sum(old_feas & ~new_feas))
        kept_pass = int(np.sum(old_feas & new_feas))
        kept_fail = int(np.sum(~old_feas & ~new_feas))
        worst_delta = new["worst"] - old["worst"][order]
        steady_delta = new["steady"] - old["steady"][order]
        arm = {
            "label": label,
            "old": {
                "qos": float(old["qos"]),
                "lcb": float(old["lcb"]),
                "worst_mean": float(np.mean(old["worst"])),
                "steady_mean": float(np.mean(old["steady"])),
            },
            "new": {
                "qos": float(new["qos"]),
                "lcb": float(new["lcb"]),
                "worst_mean": float(np.mean(new["worst"])),
                "steady_mean": float(np.mean(new["steady"])),
            },
            "paired": {
                "n": n_paired,
                "kept_pass": kept_pass,
                "kept_fail": kept_fail,
                "flipped_to_pass": flipped_to_pass,
                "flipped_to_fail": flipped_to_fail,
                "worst_delta_mean": float(np.mean(worst_delta)),
                "worst_delta_min": float(np.min(worst_delta)),
                "worst_delta_max": float(np.max(worst_delta)),
                "worst_degraded_over_0p05": int(
                    np.sum(worst_delta < -0.05)),
                "worst_improved_over_0p05": int(
                    np.sum(worst_delta > 0.05)),
                "steady_delta_mean": float(np.mean(steady_delta)),
                "missing_seeds_in_new": missing,
            },
        }
        # per-seed table
        rows = []
        for j, s in enumerate(new["seeds"]):
            if s not in idx:
                continue
            o = order.index(idx[s])
            rows.append({
                "seed": s,
                "old_feasible": bool(old_feas[o]),
                "new_feasible": bool(new_feas[j]),
                "old_worst": float(old["worst"][idx[s]]),
                "new_worst": float(new["worst"][j]),
                "worst_delta": float(worst_delta[o]),
                "old_steady": float(old["steady"][idx[s]]),
                "new_steady": float(new["steady"][j]),
            })
        arm["per_seed"] = sorted(rows, key=lambda r: r["worst_delta"])
        report["arms"][label] = arm
        print(f"\n=== {label}: V3-C0 vs historical (paired n={n_paired}) ===")
        print(f"  old QoS={arm['old']['qos']:.3f} LCB={arm['old']['lcb']:.3f}"
              f"  worst={arm['old']['worst_mean']:.4f}")
        print(f"  new QoS={arm['new']['qos']:.3f} LCB={arm['new']['lcb']:.3f}"
              f"  worst={arm['new']['worst_mean']:.4f}")
        print(f"  flips: PASS->FAIL={flipped_to_fail} "
              f"FAIL->PASS={flipped_to_pass} kept={kept_pass + kept_fail}")
        print(f"  worst delta mean={np.mean(worst_delta):+.4f} "
              f"min={np.min(worst_delta):+.4f} max={np.max(worst_delta):+.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
