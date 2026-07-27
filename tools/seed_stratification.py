#!/usr/bin/env python
"""Generate reproducible geometry-stratified evaluation seed banks.

The generator never changes the environment distribution.  It samples normal
environment resets, records their initial geometry, and writes disjoint seed
splits for checkpoint selection, candidate confirmation, final testing, and
stress testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Iterable, List

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from uav_isac.environment.env_wrapper import UAVISACEnv


def geometry_difficulty(uav_xy: np.ndarray, target_xy: np.ndarray) -> Dict[str, float]:
    """Return nearest, second-endpoint, and bottleneck-matching distances."""
    uav_xy = np.asarray(uav_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    distance = np.linalg.norm(
        uav_xy[:, None, :] - target_xy[None, :, :], axis=-1)
    ordered = np.sort(distance, axis=0)
    d1 = float(np.max(ordered[0]))
    d2 = float(np.max(ordered[min(1, ordered.shape[0] - 1)]))
    rows, cols = linear_sum_assignment(distance)
    bottleneck = float(np.max(distance[rows, cols]))
    # All terms remain in metres. D2 represents formation of a bistatic pair,
    # while bottleneck prevents one UAV from appearing responsible everywhere.
    score = float(max(bottleneck, 0.5 * (d1 + d2)))
    return {
        "worst_nearest_m": d1,
        "worst_second_nearest_m": d2,
        "bottleneck_matching_m": bottleneck,
        "difficulty_score_m": score,
    }


def _tier_counts(total: int) -> Dict[str, int]:
    easy = total // 4
    hard = total // 4
    return {"easy": easy, "medium": total - easy - hard, "hard": hard}


def build_disjoint_splits(
    tiers: Dict[str, List[int]],
    split_sizes: Dict[str, int],
    rng: np.random.Generator,
) -> Dict[str, List[int]]:
    """Draw 1:2:1 tier-balanced splits without seed overlap."""
    remaining = {
        tier: list(rng.permutation(np.asarray(values, dtype=np.int64)).astype(int))
        for tier, values in tiers.items()
    }
    result: Dict[str, List[int]] = {}
    for split_name, total in split_sizes.items():
        counts = _tier_counts(int(total))
        chosen: List[int] = []
        for tier in ("easy", "medium", "hard"):
            count = counts[tier]
            if len(remaining[tier]) < count:
                raise ValueError(
                    f"not enough {tier} seeds for {split_name}: "
                    f"need {count}, have {len(remaining[tier])}")
            chosen.extend(remaining[tier][:count])
            del remaining[tier][:count]
        result[split_name] = [int(x) for x in rng.permutation(chosen)]
    return result


def _scenario_fingerprint(config_path: str, cfg) -> str:
    payload = {
        "config": os.path.normpath(config_path),
        "region_size": list(cfg.scenario.region_size),
        "K": int(cfg.scenario.K),
        "Q": int(cfg.scenario.Q),
        "T": int(cfg.scenario.T),
        "dt": float(cfg.scenario.dt),
        "v_max": float(cfg.uav.v_max),
        "tracking_enabled": bool(cfg.marl.tracking_enabled),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def generate_seed_bank(
    config_path: str,
    candidate_seeds: Iterable[int],
    nominal_d1_max_m: float,
    split_sizes: Dict[str, int],
    stress_size: int,
    sampling_seed: int,
) -> dict:
    cfg = load_config(config_path)
    env = UAVISACEnv(config=cfg, seed=0)
    metadata: Dict[int, Dict[str, float]] = {}
    try:
        for seed in candidate_seeds:
            _, info = env.reset(seed=int(seed))
            metadata[int(seed)] = geometry_difficulty(
                np.asarray(info["uav_positions"])[:, :2],
                np.asarray(info["target_positions"])[:, :2],
            )
    finally:
        env.close()

    nominal = [
        seed for seed, values in metadata.items()
        if values["worst_nearest_m"] <= float(nominal_d1_max_m)
    ]
    if not nominal:
        raise ValueError("nominal feasibility filter removed every candidate seed")
    scores = np.asarray(
        [metadata[seed]["difficulty_score_m"] for seed in nominal])
    q25, q75 = (float(x) for x in np.quantile(scores, [0.25, 0.75]))
    tiers = {"easy": [], "medium": [], "hard": []}
    for seed in nominal:
        score = metadata[seed]["difficulty_score_m"]
        tier = "easy" if score <= q25 else "hard" if score >= q75 else "medium"
        tiers[tier].append(seed)

    rng = np.random.default_rng(int(sampling_seed))
    splits = build_disjoint_splits(tiers, split_sizes, rng)
    used = {seed for values in splits.values() for seed in values}
    stress_candidates = sorted(
        (seed for seed in metadata if seed not in used),
        key=lambda seed: metadata[seed]["difficulty_score_m"],
        reverse=True,
    )
    splits["stress"] = stress_candidates[:int(stress_size)]
    return {
        "schema_version": 1,
        "scenario_fingerprint": _scenario_fingerprint(config_path, cfg),
        "source_config": os.path.normpath(config_path),
        "candidate_count": len(metadata),
        "sampling_seed": int(sampling_seed),
        "nominal_filter": {
            "metric": "worst_nearest_m",
            "max_m": float(nominal_d1_max_m),
            "note": "benchmark scope filter, not a proof of physical feasibility",
        },
        "tier_metric": "difficulty_score_m=max(bottleneck_matching_m, "
                       "0.5*(worst_nearest_m+worst_second_nearest_m))",
        "tier_quantiles_m": {"q25": q25, "q75": q75},
        "tier_counts": {tier: len(values) for tier, values in tiers.items()},
        "splits": splits,
        "seed_metadata": {
            # Retain every sampled geometry, not only evaluation seeds.  This
            # lets training construct a disjoint prioritized pool from the same
            # frozen distribution without contaminating any evaluation split.
            str(seed): metadata[seed] for seed in sorted(metadata)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/exp_800_q4_u2u.yaml")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-count", type=int, default=1000)
    parser.add_argument("--nominal-d1-max-m", type=float, default=350.0)
    parser.add_argument("--selection-size", type=int, default=20)
    parser.add_argument("--confirmation-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--stress-size", type=int, default=50)
    parser.add_argument("--sampling-seed", type=int, default=20260721)
    parser.add_argument(
        "--output", default="config/stratified_seeds_800_q4.json")
    args = parser.parse_args()

    bank = generate_seed_bank(
        config_path=args.config,
        candidate_seeds=range(args.seed_start, args.seed_start + args.seed_count),
        nominal_d1_max_m=args.nominal_d1_max_m,
        split_sizes={
            "selection": args.selection_size,
            "confirmation": args.confirmation_size,
            "test": args.test_size,
        },
        stress_size=args.stress_size,
        sampling_seed=args.sampling_seed,
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(bank, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output": output,
        "candidate_count": bank["candidate_count"],
        "tier_counts": bank["tier_counts"],
        "split_sizes": {k: len(v) for k, v in bank["splits"].items()},
        "tier_quantiles_m": bank["tier_quantiles_m"],
    }, indent=2))


if __name__ == "__main__":
    main()
