"""Freeze an episode-level H-step ISAC coefficient-envelope margin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.evaluation.episode_joint_conformal import (  # noqa: E402
    split_conformal_upper,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate(
    audit_paths: list[Path],
    *,
    alpha: float,
) -> dict[str, object]:
    """Combine disjoint episode audits without treating events as IID."""
    if not audit_paths:
        raise ValueError("at least one audit is required")
    risk = float(alpha)
    if not np.isfinite(risk) or not 0.0 < risk < 1.0:
        raise ValueError("alpha must lie in (0,1)")

    episode_scores: dict[int, float] = {}
    source_records = []
    reference_design = None
    total_events = 0
    total_coefficients = 0
    for raw_path in audit_paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) < 6:
            raise ValueError(f"{path} predates the future-outcome audit")
        horizon = payload.get("horizon_diagnostic", {})
        diagnostic = payload.get(
            "horizon_future_calibration_diagnostic", {})
        summary = payload.get("summary", {})
        design = {
            "steps": int(horizon.get("steps", 0)),
            "base_current_log_margin": float(
                horizon.get("base_current_log_margin", np.nan)),
            "uav_reachable_position_radius_m": list(
                horizon.get("uav_reachable_position_radius_m", [])),
            "uav_reachable_velocity_radius_mps": list(
                horizon.get("uav_reachable_velocity_radius_mps", [])),
            "future_outcome_policy": horizon.get("future_outcome_policy"),
            "score": diagnostic.get("score"),
            "exchangeability_unit": diagnostic.get(
                "exchangeability_unit"),
        }
        if design["steps"] <= 0:
            raise ValueError(f"{path} has no enabled horizon audit")
        if not np.isclose(
            float(horizon.get(
                "frozen_transition_residual_log_margin", np.nan)),
            0.0,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"{path} is not a raw zero-transition-margin audit")
        if bool(diagnostic.get("future_information_used_by_controller", True)):
            raise ValueError(f"{path} leaks future outcomes to the controller")
        infinite = [
            int(seed) for seed in diagnostic.get("infinite_score_seeds", [])
        ]
        if infinite:
            raise ValueError(
                f"{path} has support errors requiring infinite margin: "
                f"{infinite}")
        if reference_design is None:
            reference_design = design
        elif design != reference_design:
            raise ValueError("calibration audits use different horizon designs")

        seeds = [int(seed) for seed in payload.get("seed_order", [])]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"{path} has an invalid seed_order")
        score_map = {
            int(seed): float(score)
            for seed, score in diagnostic.get(
                "finite_episode_scores", {}).items()
        }
        unknown = set(score_map) - set(seeds)
        if unknown:
            raise ValueError(f"{path} scores unknown seeds: {sorted(unknown)}")
        overlap = set(seeds) & set(episode_scores)
        if overlap:
            raise ValueError(
                f"episode overlap across calibration audits: {sorted(overlap)}")
        for seed in seeds:
            score = float(score_map.get(seed, 0.0))
            if not np.isfinite(score) or score < 0.0:
                raise ValueError(f"invalid episode score for seed {seed}")
            episode_scores[seed] = score
        event_count = int(summary.get(
            "horizon_future_outcome_event_count", 0))
        coefficient_count = int(summary.get(
            "horizon_future_coefficient_audit_count", 0))
        if event_count <= 0 or coefficient_count <= 0:
            raise ValueError(f"{path} has no future physical outcomes")
        total_events += event_count
        total_coefficients += coefficient_count
        source_records.append({
            "path": str(path),
            "sha256": _sha256(path),
            "episode_count": len(seeds),
            "future_event_count": event_count,
            "audited_coefficient_count": coefficient_count,
        })

    margin, coverage_floor, rank = split_conformal_upper(
        episode_scores.values(), alpha=risk)
    coverage_target = 1.0 - risk
    coverage_target_met = bool(
        coverage_floor + 1.0e-15 >= coverage_target)
    return {
        "schema_version": 1,
        "method": "episode_level_split_conformal_simultaneous_log_envelope",
        "alpha": risk,
        "target_coverage": coverage_target,
        "finite_sample_coverage_floor": float(coverage_floor),
        "coverage_target_met": coverage_target_met,
        "rank_one_based": int(rank),
        "calibration_episode_count": len(episode_scores),
        "future_event_count": int(total_events),
        "audited_coefficient_count": int(total_coefficients),
        "frozen_transition_residual_log_margin": float(margin),
        "multiplicative_lower_factor": float(np.exp(-margin)),
        "multiplicative_upper_factor": float(np.exp(margin)),
        "horizon_design": reference_design,
        "episode_scores": {
            str(seed): float(score)
            for seed, score in sorted(episode_scores.items())
        },
        "sources": source_records,
        "envelope_calibration_ready": coverage_target_met,
        "system_certificate_ready": False,
        "limitations": [
            "coverage requires exchangeability with future deployment episodes",
            "future labels use a frozen open-loop movement-action tape",
            "this calibrates only the H-step coefficient envelope, not link, "
            "runtime, energy, or live commit risks",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate(args.audit, alpha=float(args.alpha))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "episode_count": result["calibration_episode_count"],
        "margin": result["frozen_transition_residual_log_margin"],
        "coverage_floor": result["finite_sample_coverage_floor"],
        "ready": result["envelope_calibration_ready"],
    }, indent=2))


if __name__ == "__main__":
    main()
