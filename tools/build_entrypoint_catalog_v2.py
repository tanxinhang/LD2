#!/usr/bin/env python
"""Write the immutable V2 classification of every Python entrypoint."""

from __future__ import annotations

import dataclasses
import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uav_isac.governance.entrypoint_inventory import build_entrypoint_inventory
from uav_isac.governance.project_phase import assert_operation_allowed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="legacy/entrypoints.jsonl")
    args = parser.parse_args(argv)
    assert_operation_allowed("architecture_inventory")
    output = (REPOSITORY_ROOT / "artifacts" / args.output).resolve()
    artifact_root = (REPOSITORY_ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifact_root) or output == artifact_root:
        raise ValueError("entrypoint catalog must stay below artifacts/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite entrypoint catalog: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = build_entrypoint_inventory(REPOSITORY_ROOT)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")
    counts = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    print(json.dumps({
        "output": str(output),
        "entrypoints": len(records),
        "statuses": counts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
