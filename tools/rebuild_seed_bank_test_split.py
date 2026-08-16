#!/usr/bin/env python
"""Rebuild the 980_k6q6 TEST split with quarantined seeds excluded (2026-08-16).

The original test split's first five seeds are the quarantined set
{795, 747, 105, 860, 2} (docs/KNOWN_ISSUES.md).  This writes a v2 bank that
keeps selection/confirmation/stress UNCHANGED (so historical run manifests
stay valid) and replaces the test split with 100 clean, tier-balanced seeds
drawn from the bank's own seed_metadata (the original sampled geometries),
excluding the quarantined seeds and every seed already used by the other
splits.

Usage: python tools/rebuild_seed_bank_test_split.py
Output: config/stratified_seeds_980_k6q6_v2.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK_PATH = os.path.join(ROOT, "config", "stratified_seeds_980_k6q6.json")
OUT_PATH = os.path.join(ROOT, "config", "stratified_seeds_980_k6q6_v2.json")
QUARANTINED = {795, 747, 105, 860, 2}
TEST_SIZE = 100


def main() -> int:
    with open(BANK_PATH, "r", encoding="utf-8") as handle:
        bank = json.load(handle)

    metadata = {int(k): v for k, v in bank["seed_metadata"].items()}
    used = {int(s) for values in bank["splits"].values() for s in values}
    candidates = [
        seed for seed in metadata
        if seed not in QUARANTINED and seed not in used
    ]
    if len(candidates) < TEST_SIZE:
        print(f"only {len(candidates)} clean unused candidates, "
              f"need {TEST_SIZE}", file=sys.stderr)
        return 1

    scores = np.asarray([metadata[s]["difficulty_score_m"] for s in candidates])
    q25, q75 = (float(x) for x in np.quantile(scores, [0.25, 0.75]))
    tiers = {"easy": [], "medium": [], "hard": []}
    for seed in candidates:
        score = metadata[seed]["difficulty_score_m"]
        tier = "easy" if score <= q25 else "hard" if score >= q75 else "medium"
        tiers[tier].append(seed)

    # 1:2:1 tier balance (same convention as the original generator).
    counts = {"easy": TEST_SIZE // 4, "medium": TEST_SIZE // 2,
              "hard": TEST_SIZE // 4}
    chosen: list[int] = []
    rng = np.random.default_rng(int(bank.get("sampling_seed", 20260721)) + 1)
    for tier in ("easy", "medium", "hard"):
        pool = list(rng.permutation(tiers[tier]))
        if len(pool) < counts[tier]:
            print(f"not enough {tier} seeds: need {counts[tier]}, "
                  f"have {len(pool)}", file=sys.stderr)
            return 1
        chosen.extend(pool[:counts[tier]])
    test_split = [int(x) for x in rng.permutation(chosen)]

    new_bank = dict(bank)
    new_bank["schema_version"] = 2
    new_bank["splits"] = dict(bank["splits"])
    new_bank["splits"]["test"] = test_split
    new_bank["quarantined_excluded"] = sorted(QUARANTINED)
    new_bank["test_rebuild_note"] = (
        "2026-08-16: test split rebuilt excluding the quarantined seeds "
        "{795,747,105,860,2}; selection/confirmation/stress unchanged; "
        "seeds drawn from the original seed_metadata (same geometries).")

    with open(OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(new_bank, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {OUT_PATH}")
    print(f"new test split: {TEST_SIZE} seeds, first 10: {test_split[:10]}")
    hits = [s for s in test_split if s in QUARANTINED]
    print(f"quarantined in new test: {hits}")
    overlap = set(test_split) & used
    print(f"overlap with other splits: {sorted(overlap)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
