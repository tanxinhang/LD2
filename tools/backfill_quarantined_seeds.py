#!/usr/bin/env python
"""P2 (advice 014): backfill quarantined seeds out of 800_q4 / 1130_k8q8 splits.

The quarantined set {795, 747, 105, 860, 2} (docs/KNOWN_ISSUES.md) leaked into
the selection/confirmation splits of the 800_q4 and 1130_k8q8 banks.  This tool
writes v2 banks that replace every quarantined seed with a clean, unused seed
drawn deterministically from the same bank's seed_metadata (preferring the same
difficulty tier), leaving all non-quarantined seeds untouched so historical
run manifests stay valid.  The output passes the strict loader
(load_stratified_seed_split(strict=True)) which fails closed on quarantined
seeds.

Usage: python tools/backfill_quarantined_seeds.py
Output: config/stratified_seeds_800_q4_v2.json,
        config/stratified_seeds_1130_k8q8_v2.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUARANTINED = {795, 747, 105, 860, 2}
BANKS = {
    "800_q4": "stratified_seeds_800_q4.json",
    "1130_k8q8": "stratified_seeds_1130_k8q8.json",
}


def _difficulty_tier(score: float, q25: float, q75: float) -> str:
    if score <= q25:
        return "easy"
    if score >= q75:
        return "hard"
    return "medium"


def backfill(bank_name: str, src_name: str) -> int:
    src = os.path.join(ROOT, "config", src_name)
    with open(src, "r", encoding="utf-8") as handle:
        bank = json.load(handle)
    metadata = {int(k): v for k, v in bank["seed_metadata"].items()}
    used = {int(s) for values in bank["splits"].values() for s in values}
    candidates = [
        s for s in metadata if s not in QUARANTINED and s not in used
    ]
    if not candidates:
        print(f"{bank_name}: no clean candidates", file=sys.stderr)
        return 1
    scores = np.asarray([metadata[s]["difficulty_score_m"]
                         for s in candidates])
    q25, q75 = (float(x) for x in np.quantile(scores, [0.25, 0.75]))
    tiers: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
    for s in candidates:
        tiers[_difficulty_tier(
            metadata[s]["difficulty_score_m"], q25, q75)].append(s)
    draw = np.random.default_rng(int(bank.get("sampling_seed", 20260721)) + 2)

    new_splits: dict[str, list[int]] = {}
    replaced: dict[str, list[int]] = {}
    for split, seeds in bank["splits"].items():
        out: list[int] = []
        for raw in seeds:
            s = int(raw)
            if s in QUARANTINED:
                tier = _difficulty_tier(
                    metadata.get(s, {"difficulty_score_m": q75})
                    .get("difficulty_score_m", q75), q25, q75)
                pool = [c for c in tiers[tier] if c not in used]
                if not pool:
                    pool = [c for c in candidates if c not in used]
                if not pool:
                    print(f"{bank_name}/{split}: no replacement for {s}",
                          file=sys.stderr)
                    return 1
                choice = int(draw.choice(sorted(pool)))
                used.add(choice)
                out.append(choice)
                replaced.setdefault(split, []).append((s, choice))
            else:
                out.append(s)
        new_splits[split] = out

    new_bank = dict(bank)
    new_bank["schema_version"] = 2
    new_bank["splits"] = new_splits
    new_bank["quarantined_excluded"] = sorted(QUARANTINED)
    new_bank["backfill_note"] = (
        "2026-08-17 (P2, advice 014): quarantined seeds replaced in "
        "selection/confirmation with clean unused same-tier draws; all other "
        "seeds unchanged; passes the strict loader.")
    out_name = src_name.replace(".json", "_v2.json")
    out_path = os.path.join(ROOT, "config", out_name)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(new_bank, handle, indent=2, sort_keys=True)
        handle.write("\n")
    hits = {
        split: sorted(set(int(s) for s in seeds) & QUARANTINED)
        for split, seeds in new_splits.items()
    }
    hits = {k: v for k, v in hits.items() if v}
    print(f"{bank_name}: wrote {out_path}; replaced: {replaced}; "
          f"remaining quarantined: {hits}")
    return 0


def main() -> int:
    code = 0
    for bank_name, src_name in BANKS.items():
        code |= backfill(bank_name, src_name)
    return code


if __name__ == "__main__":
    sys.exit(main())
