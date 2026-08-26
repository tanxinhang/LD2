#!/usr/bin/env python
"""P0 fail-closed audit for a current-model layer-provenance trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_isac.evaluation.layer_provenance import (  # noqa: E402
    audit_layer_trace_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--expected-physics-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_layer_trace_provenance(
            args.trace,
            expected_physics_sha256=args.expected_physics_sha256,
        )
        result["trace"] = str(args.trace)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result = {
            "gate": "FAIL",
            "trace": str(args.trace),
            "reason": str(error),
            "scientific_status": "BLOCKED_BY_LAYER_PROVENANCE",
        }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if result["gate"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
