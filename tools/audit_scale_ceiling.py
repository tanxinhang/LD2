#!/usr/bin/env python
"""Scale-aware same-geometry ceiling decomposition.

For a frozen policy trace at any (K, Q), answer one decisive question: is the
deployed worst-target detection probability limited by *physics* or by
*coordination/power allocation*?  The answer comes from a same-geometry
waterfall of progressively stronger benchmarks:

    deployed  ->  power-only   (re-optimize sensing power, fix structure)
               ->  single      (joint structure+power, one role per UAV)
               ->  duplex      (joint structure+power, full duplex)
               ->  relaxed     (same-geometry per-target ceiling)

``relaxed`` relaxes cross-target power coupling, common receiver owner, role
and capacity, so it is a *componentwise upper bound* on every feasible
same-geometry allocation.  If it is below the QoS floor, no coordination
algorithm can reach the floor on those geometries; if it is well above the
deployed value, the remaining gap is coordination, not physics.

The decomposition reuses the audited machinery in
``tools/audit_structure_trace_physical_bottleneck.py`` and only adds the
waterfall attribution, oracle-gap recovery and the feasibility verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_structure_trace_physical_bottleneck import audit  # noqa: E402


def ceiling_decomposition(result: dict, qos_floor: float = 0.60) -> dict:
    """Turn the per-seed benchmark rows into a ceiling waterfall."""
    mean = result["mean"]
    deployed = float(mean["deployed_worst"])
    power = float(mean["power_only_worst"])
    single = float(mean["single_worst"])
    duplex = float(mean["duplex_worst"])
    relaxed = float(result["same_geometry_relaxed_upper_mean_worst"])

    spans = [
        ("deployed", deployed),
        ("power_only", power),
        ("single", single),
        ("duplex", duplex),
        ("relaxed_ceiling", relaxed),
    ]
    gaps = {
        "power_gap": power - deployed,
        "structure_gap": single - power,
        "duplex_gap": duplex - single,
        "relaxation_gap": relaxed - duplex,
    }
    reachable = max(relaxed - deployed, 0.0)
    recovery = {
        key: (value / reachable if reachable > 1e-12 else 0.0)
        for key, value in (
            ("power_recovery", power - deployed),
            ("single_recovery", single - deployed),
            ("duplex_recovery", duplex - deployed),
        )
    }

    floor = float(qos_floor)
    if relaxed < floor:
        verdict = (
            "physics_limited: relaxed same-geometry ceiling "
            f"{relaxed:.4f} < floor {floor:.4f}"
        )
        explanation = (
            "Even with cross-target power coupling, common owner, role and "
            "capacity all relaxed, no same-geometry allocation reaches the "
            "QoS floor. Optimization must reframe as oracle-gap recovery, not "
            "absolute feasibility."
        )
    elif deployed >= floor:
        verdict = f"already_feasible: deployed {deployed:.4f} >= floor"
        explanation = (
            "Deployed worst already meets the floor on these frames; the "
            "ceiling decomposition is a headroom report."
        )
    else:
        verdict = "coordination_limited"
        explanation = (
            f"Deployed {deployed:.4f} is below floor {floor:.4f} but the "
            f"single-role oracle reaches {single:.4f} and full-duplex "
            f"{duplex:.4f}. The gap is coordination/power allocation, not "
            "physics."
        )

    dominant = max(gaps, key=lambda key: float(gaps[key]))
    return {
        "qos_floor": floor,
        "verdict": verdict,
        "explanation": explanation,
        "waterfall": {key: value for key, value in spans},
        "gaps": gaps,
        "dominant_gap": dominant,
        "oracle_gap_recovery": recovery,
        "qos_feasible_rate": result.get("qos_feasible_rate"),
        "fixed_owner_exact_mean_worst": result.get(
            "fixed_owner_exact_mean_worst"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed-limit", type=int, default=10)
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument(
        "--dual-rounds", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--price-bits", type=int, default=6)
    parser.add_argument("--feedback-bits", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rounds = tuple(sorted({max(1, int(value)) for value in args.dual_rounds}))
    result = audit(
        args.trace,
        max(1, int(args.seed_limit)),
        dual_rounds=rounds,
        price_bits=int(args.price_bits),
        feedback_bits=int(args.feedback_bits),
        config_path=args.config,
    )
    decomposition = ceiling_decomposition(result, float(args.qos_floor))
    payload = {
        "schema_version": 1,
        "trace": str(args.trace),
        "config": str(args.config) if args.config is not None else None,
        "seed_limit": int(result["seed_limit"]),
        "decomposition": decomposition,
        "distributed_column_generation": result.get("distributed_mean"),
        "raw_mean": result["mean"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "verdict": decomposition["verdict"],
        "waterfall": decomposition["waterfall"],
        "gaps": decomposition["gaps"],
        "dominant_gap": decomposition["dominant_gap"],
        "oracle_gap_recovery": decomposition["oracle_gap_recovery"],
        "qos_feasible_rate": decomposition["qos_feasible_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
