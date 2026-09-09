#!/usr/bin/env python
"""Create immutable metadata catalogs for legacy results; never delete data."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uav_isac.governance.data_inventory import build_data_files
from uav_isac.governance.project_phase import assert_operation_allowed


def _safe_output(relative: str) -> Path:
    root = (REPOSITORY_ROOT / "artifacts").resolve()
    output = (root / relative).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("catalog output must stay below artifacts/")
    return output


def _write_new_json_lines(path: Path, rows) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite catalog: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale catalog temporary exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"catalog appeared during write: {path}")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="legacy/catalog.jsonl")
    parser.add_argument("--candidates", default="cleanup/candidates.jsonl")
    args = parser.parse_args(argv)
    assert_operation_allowed("architecture_inventory")
    files = build_data_files(REPOSITORY_ROOT)
    catalog_rows = (
        {**dataclasses.asdict(item), "sha256": None, "hash_status": "pending"}
        for item in files
    )
    candidate_rows = (
        {
            **dataclasses.asdict(item),
            "decision": "review_required",
            "deletion_authorized": False,
        }
        for item in files
        if item.lifecycle == "scratch_candidate"
    )
    catalog = _safe_output(args.catalog)
    candidates = _safe_output(args.candidates)
    _write_new_json_lines(catalog, catalog_rows)
    _write_new_json_lines(candidates, candidate_rows)
    print(json.dumps({
        "catalog": str(catalog),
        "catalog_files": len(files),
        "candidates": str(candidates),
        "candidate_files": sum(
            item.lifecycle == "scratch_candidate" for item in files
        ),
        "deletion_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

