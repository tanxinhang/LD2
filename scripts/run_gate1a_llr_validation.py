"""Monte Carlo validation for Gate 1a stochastic receiver evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uav_isac.physical.evidence import (
    detect_from_llr,
    sample_gaussian_llr,
    theoretical_detection_probability,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--p-fa", type=float, default=0.001)
    parser.add_argument(
        "--deflection",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate1a_llr_validation/summary.json"),
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    d_grid = np.asarray(args.deflection, dtype=np.float64)
    if np.any(d_grid <= 0.0):
        raise ValueError("validation deflections must be strictly positive")

    rng = np.random.default_rng(args.seed)
    shape = (args.samples, d_grid.size)
    d = np.broadcast_to(d_grid, shape)
    common_noise_h0 = rng.standard_normal(shape)
    common_noise_h1 = rng.standard_normal(shape)
    llr_h0 = sample_gaussian_llr(
        d, 0, standard_normal=common_noise_h0)
    llr_h1 = sample_gaussian_llr(
        d, 1, standard_normal=common_noise_h1)
    empirical_pfa = np.mean(
        detect_from_llr(llr_h0, d, args.p_fa), axis=0)
    empirical_pd = np.mean(
        detect_from_llr(llr_h1, d, args.p_fa), axis=0)
    theoretical_pd = theoretical_detection_probability(d_grid, args.p_fa)

    rows = []
    for index, deflection in enumerate(d_grid):
        rows.append({
            "deflection": float(deflection),
            "empirical_pfa": float(empirical_pfa[index]),
            "target_pfa": float(args.p_fa),
            "pfa_abs_error": float(
                abs(empirical_pfa[index] - args.p_fa)),
            "empirical_pd": float(empirical_pd[index]),
            "theoretical_pd": float(theoretical_pd[index]),
            "pd_abs_error": float(
                abs(empirical_pd[index] - theoretical_pd[index])),
        })

    pfa_tolerance = max(2.0e-4, 5.0 * np.sqrt(
        args.p_fa * (1.0 - args.p_fa) / args.samples))
    pd_tolerance = 5.0 * float(np.max(np.sqrt(
        theoretical_pd * (1.0 - theoretical_pd) / args.samples)))
    summary = {
        "gate": "G1a",
        "samples_per_deflection_hypothesis": int(args.samples),
        "seed": int(args.seed),
        "target_pfa": float(args.p_fa),
        "max_pfa_abs_error": float(
            np.max(np.abs(empirical_pfa - args.p_fa))),
        "max_pd_abs_error": float(
            np.max(np.abs(empirical_pd - theoretical_pd))),
        "pfa_tolerance": float(pfa_tolerance),
        "pd_tolerance": float(pd_tolerance),
        "gate_pass": bool(
            np.max(np.abs(empirical_pfa - args.p_fa)) <= pfa_tolerance
            and np.max(np.abs(empirical_pd - theoretical_pd))
            <= pd_tolerance
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
