#!/usr/bin/env python
"""S0-D spectral audit of dense bistatic coefficient structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import load_config  # noqa: E402
from uav_isac.coordination.coefficient_structure import (  # noqa: E402
    exact_bistatic_factorization, spectral_structure,
)


def audit(trace_path: Path, config_path: Path, g4a_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    cfg = load_config(str(config_path))
    g4a = json.loads(g4a_path.read_text(encoding="utf-8"))
    region = {(int(row["seed"]), int(row["frame"])) for row in g4a["rows"]}
    records = []
    stages = (
        "ideal_path_completion", "raw_path_bistatic",
        "reported_bistatic", "dd_operational")
    for index in range(np.asarray(data["seed"]).size):
        key = (int(data["seed"][index]), int(data["frame"][index]))
        if key not in region:
            continue
        alpha2 = np.asarray(data["privileged_alpha"][index], dtype=np.float64) ** 2
        report = np.asarray(data["privileged_chi_rep"][index], dtype=np.float64)
        dd = np.asarray(data["privileged_g_dd"][index], dtype=np.float64)
        K, _, Q = alpha2.shape
        off_diagonal = ~np.eye(K, dtype=bool)
        active = dd >= float(cfg.detection.g_min)
        uav_position = np.asarray(data["uav_positions"][index], dtype=np.float64)
        target_state = np.asarray(data["target_states"][index], dtype=np.float64)
        ideal = np.empty_like(alpha2)
        for q in range(Q):
            target_position = np.array(
                [target_state[q, 0], target_state[q, 1], 0.0], dtype=np.float64)
            inverse_range_squared = 1.0 / np.maximum(
                np.sum((uav_position - target_position) ** 2, axis=1), 1.0e-12)
            ideal[:, :, q] = np.outer(
                inverse_range_squared, inverse_range_squared)
        tensors = {
            "ideal_path_completion": ideal,
            "raw_path_bistatic": alpha2 * off_diagonal[:, :, None],
            "reported_bistatic": alpha2 * report * off_diagonal[:, :, None],
            "dd_operational": alpha2 * report * active * off_diagonal[:, :, None],
        }
        for q in range(Q):
            factorization = exact_bistatic_factorization(
                tensors["raw_path_bistatic"][:, :, q], report[:, :, q],
                active[:, :, q])
            reconstructed = factorization.dense()
            operational = tensors["dd_operational"][:, :, q]
            scale = max(float(np.max(np.abs(operational))), 1.0e-300)
            reconstruction_error = float(
                np.max(np.abs(reconstructed - operational)) / scale)
            for stage in stages:
                structure = spectral_structure(tensors[stage][:, :, q])
                records.append({
                    "seed": key[0], "frame": key[1], "target": q,
                    "stage": stage,
                    "singular_values": list(structure.singular_values),
                    "rank_90": structure.rank_90,
                    "rank_95": structure.rank_95,
                    "rank_99": structure.rank_99,
                    "stable_rank": structure.stable_rank,
                    "entropy_effective_rank": structure.entropy_effective_rank,
                    "rank1_relative_frobenius_error": (
                        structure.rank1_relative_frobenius_error),
                    "exact_factorization_relative_max_error": reconstruction_error,
                    "dd_sparse_exceptions": len(
                        factorization.inactive_offdiagonal),
                })

    summary = {}
    for stage in stages:
        selected = [record for record in records if record["stage"] == stage]
        summary[stage] = {
            name: {
                "min": float(np.min([record[name] for record in selected])),
                "median": float(np.median([record[name] for record in selected])),
                "max": float(np.max([record[name] for record in selected])),
            }
            for name in (
                "rank_90", "rank_95", "rank_99", "stable_rank",
                "entropy_effective_rank", "rank1_relative_frobenius_error")
        }
    operational = [record for record in records if record["stage"] == "dd_operational"]
    low_rank_95 = sum(record["rank_95"] <= 2 for record in operational)
    return {
        "gate": "S0-D-dense-low-rank-coefficient-audit",
        "scope": "21_postG2_region_ii_frames_x_6_targets",
        "rank_definition": "minimum rank retaining threshold fraction of squared singular-value energy",
        "stage_semantics": {
            "ideal_path_completion": (
                "inverse-range outer product reconstructed from geometry, including diagonal"),
            "raw_path_bistatic": "raw path after monostatic diagonal exclusion",
            "reported_bistatic": "reported path after monostatic diagonal exclusion",
            "dd_operational": "reported bistatic path after hard DD gate",
        },
        "frames": len(region),
        "target_matrices": len(operational),
        "summary": summary,
        "operational_rank95_le_2": low_rank_95,
        "operational_rank95_le_2_ratio": low_rank_95 / len(operational),
        "exact_factorization_max_relative_error": max(
            record["exact_factorization_relative_max_error"]
            for record in operational),
        "dd_sparse_exceptions": sum(
            record["dd_sparse_exceptions"] for record in operational),
        "dd_sparse_exception_ratio": sum(
            record["dd_sparse_exceptions"] for record in operational)
            / (len(operational) * 6 * 5),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--g4a", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.trace, args.config, args.g4a)
    payload = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
