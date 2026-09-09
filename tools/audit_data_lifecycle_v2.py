#!/usr/bin/env python
"""Print a read-only lifecycle inventory for legacy result data."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uav_isac.governance.data_inventory import build_data_inventory
from uav_isac.governance.project_phase import assert_operation_allowed


def main() -> int:
    assert_operation_allowed("architecture_inventory")
    inventory = build_data_inventory(REPOSITORY_ROOT)
    print(json.dumps(dataclasses.asdict(inventory), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

