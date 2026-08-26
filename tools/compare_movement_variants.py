"""Compare movement-strategy variants on the decisive seeds (three-phase audit).

Reads paired_eval.csv files for the frozen hybrid base, gap-coverage v1,
pursuit-250, near-field focus variants, and prints a per-seed worst-P_D table
plus gate counts, so the three-phase policy progression is transparent.

Usage:
    python tools/compare_movement_variants.py
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

VARIANTS = [
    ("frozen_hybrid", "_conf25_hybrid", "frozen hybrid (audit freeze)"),
    ("gapcov_v1", "_conf25_gapcov", "gap-coverage v1 (product-order)"),
    ("gapcov_p250", "_gapcov_pursuit250_11seed", "pursuit-250"),
    ("gapcov_focus350", "_gapcov_focus_12seed", "near-field focus 350"),
    ("gapcov_focus450", "_gapcov_focus450_12seed", "near-field focus 450"),
]


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return dict(zip(rows[0], rows[1]))


def arr(value) -> np.ndarray:
    return np.asarray(ast.literal_eval(value), dtype=np.float64)


def main() -> None:
    runs = {}
    for key, dirname, label in VARIANTS:
        csv_path = RESULTS / dirname / "paired_eval.csv"
        if not csv_path.exists():
            print(f"[skip] {key}: {csv_path} missing")
            continue
        runs[key] = (label, load(csv_path))

    base_label, base = runs["frozen_hybrid"]
    base_seeds = [int(s) for s in ast.literal_eval(base["eval_episode_seeds"])]
    base_worst = arr(base["eval_episode_worst_P_D"])
    base_index = {seed: i for i, seed in enumerate(base_seeds)}

    # Intersect seeds across all variants that have the 25-seed protocol
    # seeds; focus runs use a 12-seed subset, so align by seed id.
    seeds = sorted(base_seeds)
    header = f"{'seed':>12} | {'hybrid':>7} | " + " | ".join(
        f"{key:>9}" for key in runs if key != "frozen_hybrid")
    print(header)
    print("-" * len(header))
    for seed in seeds:
        row = f"{seed:>12} | {base_worst[base_index[seed]]:7.4f} | "
        for key in runs:
            if key == "frozen_hybrid":
                continue
            _label, data = runs[key]
            ep_seeds = [int(s) for s in
                        ast.literal_eval(data["eval_episode_seeds"])]
            if seed not in ep_seeds:
                row += f"{'—':>9} | "
                continue
            idx = ep_seeds.index(seed)
            worst = arr(data["eval_episode_worst_P_D"])[idx]
            row += f"{worst:9.4f} | "
        print(row)

    # Gate counts for runs that cover the full 25 protocol seeds.
    print()
    for key in ("gapcov_v1",):
        if key not in runs:
            continue
        _label, data = runs[key]
        if len(ast.literal_eval(data["eval_episode_seeds"])) != 25:
            print(f"[{key}] not a full 25-seed run; skip gate count")
            continue
        w = arr(data["eval_episode_worst_P_D"])
        w3 = arr(data["eval_episode_weak3_P_D"])
        st = arr(data["eval_episode_steady_P_D"])
        gate = int(np.sum((w >= 0.6) & (w3 >= 0.7) & (st >= 0.8)))
        print(f"[{key}] gate pass {gate}/25")
    base_gate = int(np.sum(
        (base_worst >= 0.6)
        & (arr(base["eval_episode_weak3_P_D"]) >= 0.7)
        & (arr(base["eval_episode_steady_P_D"]) >= 0.8)))
    print(f"[frozen_hybrid] gate pass {base_gate}/25")


if __name__ == "__main__":
    main()