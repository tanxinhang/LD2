#!/usr/bin/env python
"""D1.5 blind-certification left-tail analysis (reproducible).

Reads a paired_eval.csv (single-row convention) plus the blind bank metadata
and reports the failure decomposition that motivates the D1.9 initial-transient
optimization:

  - QoS pass rate and Wilson LCB (point estimate vs LCB-enforced gate)
  - failure mode: scene-level collapse (steady ~ weak3 ~ worst) vs target-level
  - dominant factor: initial `worst_nearest` (bank metadata) vs realized
    worst UAV-target distance (from eval), and the physical time budget
    (v_max * T * dt) reachability boundary

Usage:
    python tools/analyze_blind_tail.py --csv results/_d1_5_blind100/paired_eval.csv
    python tools/analyze_blind_tail.py --csv ... --bank config/stratified_seeds_1130_k8q8_blind.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def _wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a Binomial proportion."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - half) / denom)


def _load_arrays(row: Dict[str, str], *cols: str) -> List[np.ndarray]:
    out = []
    for col in cols:
        raw = row.get(col, "")
        out.append(np.asarray(ast.literal_eval(raw), dtype=np.float64)
                   if raw else np.zeros(0))
    return out


def analyze(csv_path: str, bank_path: Optional[str] = None) -> Dict[str, object]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    seeds, steady, weak3, worst = _load_arrays(
        row, "eval_episode_seeds", "eval_episode_steady_P_D",
        "eval_episode_weak3_P_D", "eval_episode_worst_P_D")
    n = len(seeds)
    qos = (steady >= 0.80) & (weak3 >= 0.70) & (worst >= 0.60)
    k_pass = int(qos.sum())

    realized = None
    if "eval_episode_worst_nearest_distance_m" in row:
        realized = _load_arrays(
            row, "eval_episode_worst_nearest_distance_m")[0]

    meta: Dict[str, Dict[str, float]] = {}
    if bank_path:
        with open(bank_path) as f:
            bank = json.load(f)
        meta = bank.get("seed_metadata", {})

    report: Dict[str, object] = {
        "n_seeds": n,
        "qos_pass": k_pass,
        "qos_rate": float(k_pass / n) if n else 0.0,
        "wilson_lcb": _wilson_lower(k_pass, n),
        "steady": {"mean": float(steady.mean()), "median": float(np.median(steady))},
        "weak3": {"mean": float(weak3.mean()), "median": float(np.median(weak3))},
        "worst": {"mean": float(worst.mean()), "median": float(np.median(worst))},
        "failure_mode": _failure_mode(steady, weak3, worst, qos),
    }

    if len(meta):
        worst_nearest = np.asarray(
            [meta.get(str(int(s)), {}).get("worst_nearest_m", np.nan)
             for s in seeds], dtype=np.float64)
        report["worst_nearest"] = _bucket_table(worst_nearest, qos, worst)

    if realized is not None and len(realized) == n:
        report["realized_distance"] = _bucket_table(realized, qos, worst)
        report["corr_realized_worst_pd"] = _pearson_correlation(realized, worst)
    return report


def _pearson_correlation(x: np.ndarray, y: np.ndarray):
    """Pearson r with pairwise finite filtering and an explicit domain guard."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("correlation inputs must have equal length")
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return None
    x_centered = x[finite] - np.mean(x[finite])
    y_centered = y[finite] - np.mean(y[finite])
    x_energy = float(np.sum(x_centered * x_centered))
    y_energy = float(np.sum(y_centered * y_centered))
    if x_energy == 0.0 or y_energy == 0.0:
        return None
    covariance = float(np.sum(x_centered * y_centered))
    return covariance / np.sqrt(x_energy * y_energy)


def _failure_mode(
    steady: np.ndarray, weak3: np.ndarray, worst: np.ndarray,
    qos: np.ndarray,
) -> Dict[str, object]:
    fail = ~qos
    if fail.sum() == 0:
        return {"scene_level_failures": 0, "target_level_failures": 0}
    # scene-level: all three floors collapse together (weak3/worst track steady)
    scene = fail & (np.abs(weak3 - worst) < 0.02) & (steady < 0.80)
    return {
        "scene_level_failures": int(scene.sum()),
        "target_level_failures": int(fail.sum() - scene.sum()),
        "worst_failing_seed_indices": [
            int(i) for i in np.argsort(worst[fail])[:10].tolist()],
    }


def _bucket_table(
    x: np.ndarray, qos: np.ndarray, worst: np.ndarray,
) -> List[Dict[str, object]]:
    buckets = [(0, 250), (250, 350), (350, 450), (450, 550), (550, 1e9)]
    out = []
    for lo, hi in buckets:
        m = (x >= lo) & (x < hi)
        if m.sum() == 0:
            continue
        out.append({
            "range_m": f"[{lo},{hi})",
            "n": int(m.sum()),
            "qos_rate": float(qos[m].mean()),
            "worst_median": float(np.median(worst[m])),
        })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="paired_eval.csv path")
    ap.add_argument("--bank", default=None,
                    help="blind bank json (seed_metadata for worst_nearest)")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args(argv)

    report = analyze(args.csv, args.bank)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
