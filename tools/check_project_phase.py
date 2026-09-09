#!/usr/bin/env python
"""Inspect or enforce the repository project-phase gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uav_isac.governance.project_phase import (
    OperationBlockedError,
    assert_operation_allowed,
    load_project_phase,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        help="fail unless this operation is explicitly enabled",
    )
    args = parser.parse_args()

    try:
        phase = (
            assert_operation_allowed(args.operation)
            if args.operation
            else load_project_phase()
        )
    except (OSError, ValueError, OperationBlockedError) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print(json.dumps({
        "phase": phase.phase,
        "completed_gates": phase.completed_gates,
        "operations": phase.operations,
        "source": str(phase.source),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
