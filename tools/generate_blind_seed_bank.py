#!/usr/bin/env python
"""D1.5 blind seed bank generation (advice 013).

The deployment candidate (L0-KKT + Lex-L1 + P0-L2 + multi-candidate L3) was
tuned on the 20-seed selection split, which has been reused for margin
selection and multiple validation rounds -- it can no longer serve as a
blind test bank.  This tool builds a BLIND bank for the 8/8 (1130 x 1130)
environment from the bank's own seed_metadata (the originally sampled
geometries), excluding:
  - the quarantined seeds {795, 747, 105, 860, 2};
  - every seed that has EVER appeared in any results/ paired_eval.csv
    episode seed list (any split, any run -- these have been "seen");
  - every seed in the 1130_k8q6 / 800_q4 / 980_k6q6 bank splits
    (selection/confirmation/test/stress).

The surviving seeds were never evaluated by any run in this repo, so they
form a valid blind test bank for the frozen candidate.

Usage: python tools/generate_blind_seed_bank.py [--size 100] [--bank 1130_k8q8]
Output: config/stratified_seeds_1130_k8q8_blind.json
"""

from __future__ import annotations

import ast
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUARANTINED = {795, 747, 105, 860, 2}
BANKS = ("stratified_seeds_1130_k8q8.json", "stratified_seeds_800_q4.json",
         "stratified_seeds_980_k6q6.json", "stratified_seeds_980_k6q6_v2.json")


def collect_exposed_seeds() -> set[int]:
    """Every seed that appears in any paired_eval.csv episode list."""
    exposed: set[int] = set()
    n_csv = 0
    for path in glob.glob(os.path.join(ROOT, "results", "*", "paired_eval.csv")):
        try:
            with open(path, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if not rows or "eval_episode_seeds" not in rows[0]:
                continue
            seeds = ast.literal_eval(rows[0]["eval_episode_seeds"])
            exposed.update(int(s) for s in seeds)
            n_csv += 1
        except (ValueError, SyntaxError, OSError):
            continue
    print(f"scanned {n_csv} paired_eval.csv files; "
          f"{len(exposed)} exposed seeds")
    return exposed


def collect_bank_seeds() -> set[int]:
    used: set[int] = set()
    for name in BANKS:
        path = os.path.join(ROOT, "config", name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            bank = json.load(handle)
        for values in bank.get("splits", {}).values():
            used.update(int(s) for s in values)
    return used


def main() -> int:
    size = 100
    if len(sys.argv) > 2 and sys.argv[1] == "--size":
        size = int(sys.argv[2])
    bank_path = os.path.join(ROOT, "config", "stratified_seeds_1130_k8q8.json")
    with open(bank_path, encoding="utf-8") as handle:
        bank = json.load(handle)
    metadata = {int(k): v for k, v in bank["seed_metadata"].items()}

    exposed = collect_exposed_seeds()
    bank_seeds = collect_bank_seeds()
    forbidden = QUARANTINED | exposed | bank_seeds
    candidates = [s for s in metadata if s not in forbidden]
    print(f"candidates after exclusion: {len(candidates)} "
          f"(need {size})")
    if len(candidates) < size:
        print("not enough blind candidates; widen the metadata pool",
              file=sys.stderr)
        return 1

    scores = np.asarray([metadata[s]["difficulty_score_m"] for s in candidates])
    q25, q75 = (float(x) for x in np.quantile(scores, [0.25, 0.75]))
    tiers = {"easy": [], "medium": [], "hard": []}
    for seed in candidates:
        score = metadata[seed]["difficulty_score_m"]
        tier = "easy" if score <= q25 else "hard" if score >= q75 else "medium"
        tiers[tier].append(seed)
    counts = {"easy": size // 4, "medium": size // 2, "hard": size // 4}
    rng = np.random.default_rng(20260827)  # frozen draw seed
    chosen: list[int] = []
    for tier in ("easy", "medium", "hard"):
        pool = list(rng.permutation(tiers[tier]))
        if len(pool) < counts[tier]:
            print(f"not enough {tier}: need {counts[tier]}, have {len(pool)}",
                  file=sys.stderr)
            return 1
        chosen.extend(pool[:counts[tier]])
    blind = [int(x) for x in rng.permutation(chosen)]

    out = {
        "schema_version": 1,
        "scenario_fingerprint": bank["scenario_fingerprint"],
        "source_config": bank["source_config"],
        "purpose": "D1.5 blind certification bank (advice 013)",
        "blind_draw_seed": 20260827,
        "excluded": {
            "quarantined": sorted(QUARANTINED),
            "exposed_in_results": len(exposed),
            "bank_splits": len(bank_seeds),
        },
        "tier_quantiles_m": {"q25": q25, "q75": q75},
        "tier_counts": {t: counts[t] for t in counts},
        "splits": {"test": blind},
        "seed_metadata": {
            # Keep ALL sampled geometries: the training pool is built from
            # this metadata minus the reserved (test/blind) seeds, so blind
            # seeds are automatically excluded from training -- exactly the
            # separation D1.5 requires -- while the pool stays non-empty.
            str(s): metadata[s] for s in sorted(metadata)
        },
    }
    out_path = os.path.join(ROOT, "config",
                            "stratified_seeds_1130_k8q8_blind.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {out_path}")
    print(f"blind test: {len(blind)} seeds, first 10: {blind[:10]}")
    hits = [s for s in blind if s in forbidden]
    print(f"forbidden seeds in blind: {hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
