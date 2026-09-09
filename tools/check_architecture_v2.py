#!/usr/bin/env python
"""Fail when architecture V2 dependency or module-size rules are violated."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uav_isac.governance.architecture import audit_architecture


def main() -> int:
    violations = audit_architecture()
    if violations:
        for violation in violations:
            print(violation.format())
        print(f"FAILED: {len(violations)} architecture violation(s)")
        return 1
    print("PASS: architecture V2 boundaries are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

