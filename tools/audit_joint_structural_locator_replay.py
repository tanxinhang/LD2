#!/usr/bin/env python
"""Development-only replay of the joint structural locator.

This tool deliberately consumes previously exposed exact witness digests.  It
therefore measures grammar recall on the development trace and MUST NOT be
used as blind evidence or as the M4-D0/M4-D1 temporal-separation gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_isac.coordination.oracle_free_candidate_locator import (  # noqa: E402
    generate_oracle_free_candidate_pools,
)
from uav_isac.utils.provenance import sha256_file  # noqa: E402


def replay(trace_path: Path, frozen_path: Path, referee_path: Path) -> dict:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    referee = json.loads(referee_path.read_text(encoding="utf-8"))
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    frame_lookup = {
        (int(data["seed"][i]), int(data["frame"][i])): i
        for i in range(np.asarray(data["frame"]).size)
    }
    frozen_lookup = {
        (int(row["seed"]), int(row["frame"])): row
        for row in frozen["rows"]
    }
    exact = {
        (int(row["seed"]), int(row["frame"])): str(
            row["global_witness_digest"])
        for row in referee["rows"]
        if row.get("exact_status") == "GENUINE_L2"
    }
    pair_limit = int(np.asarray(data["target_pair_limit"]).reshape(-1)[0])
    receiver_limit = int(
        np.asarray(data["reports_per_receiver"]).reshape(-1)[0])
    started = time.perf_counter()
    rows = []
    for key, witness_digest in exact.items():
        index = frame_lookup[key]
        old = frozen_lookup[key]
        candidates = generate_oracle_free_candidate_pools(
            np.asarray(data["per_watt_coefficient"][index], dtype=np.float64),
            np.asarray(data["frame_sensing_budget_w"][index], dtype=np.float64),
            [tuple(map(int, edge)) for edge in np.argwhere(
                data["deployed_selected"][index])],
            np.asarray(data["deployed_role"][index], dtype=np.int8),
            np.asarray(old["target_price"], dtype=np.float64),
            np.asarray(old["scarcity_price"], dtype=np.float64),
            target_pair_limit=pair_limit,
            reports_per_receiver=receiver_limit,
            nested_budgets=tuple(map(int, frozen["nested_budgets"])),
        )
        match = next(
            (candidate for candidate in candidates
             if candidate.digest == witness_digest), None)
        rows.append({
            "seed": key[0],
            "frame": key[1],
            "candidate_count": len(candidates),
            "pool_a_count": sum(
                item.pool == "A_DUAL_NESTED" for item in candidates),
            "exact_witness_digest": witness_digest,
            "exact_digest_recalled": match is not None,
            "pool_a_recalled": bool(
                match is not None and match.pool == "A_DUAL_NESTED"),
            "matched_score_mode": (
                None if match is None else match.score_mode),
            "matched_nested_budget": (
                None if match is None else match.nested_budget),
        })
    counts = np.asarray([row["candidate_count"] for row in rows], dtype=float)
    return {
        "schema_version": 1,
        "stage": "JOINT_STRUCTURAL_LOCATOR_DEVELOPMENT_REPLAY",
        "evidence_class": "DEVELOPMENT_ONLY_NOT_BLIND",
        "forbidden_claims": [
            "independent_confirmation", "blind_generalization",
            "M4-D0/M4-D1 temporal separation",
        ],
        "trace": str(trace_path),
        "prior_candidate_artifact": str(frozen_path),
        "prior_exact_referee_artifact": str(referee_path),
        "locator_code_sha256": sha256_file(
            ROOT / "uav_isac/coordination/oracle_free_candidate_locator.py"),
        "tool_code_sha256": sha256_file(Path(__file__)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "genuine_l2_frames": len(rows),
        "exact_digest_recall": sum(
            row["exact_digest_recalled"] for row in rows),
        "pool_a_exact_digest_recall": sum(
            row["pool_a_recalled"] for row in rows),
        "candidate_count_min": int(np.min(counts)) if rows else None,
        "candidate_count_median": float(np.median(counts)) if rows else None,
        "candidate_count_max": int(np.max(counts)) if rows else None,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--prior-candidates", required=True, type=Path)
    parser.add_argument("--prior-referee", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay(args.trace, args.prior_candidates, args.prior_referee)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(
        {key: value for key, value in result.items() if key != "rows"},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
