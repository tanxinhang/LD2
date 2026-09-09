#!/usr/bin/env python
"""Hash legacy result files and report exact duplicates without deleting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uav_isac.governance.project_phase import assert_operation_allowed


def _safe_artifact(relative: str) -> Path:
    root = (REPOSITORY_ROOT / "artifacts").resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("output must stay below artifacts/")
    return path


def _write_new_lines(path: Path, rows) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(f"output appeared during write: {path}")
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _hash_row(row):
    source = (REPOSITORY_ROOT / "results" / row["path"]).resolve()
    result_root = (REPOSITORY_ROOT / "results").resolve()
    if not source.is_relative_to(result_root) or not source.is_file():
        raise FileNotFoundError(f"catalog source is unavailable: {source}")
    before = source.stat()
    if before.st_size != row["size_bytes"] or before.st_mtime_ns != row["modified_ns"]:
        raise RuntimeError(f"source changed after metadata catalog: {row['path']}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise RuntimeError(f"source changed while hashing: {row['path']}")
    return {**row, "sha256": digest.hexdigest(), "hash_status": "verified"}


def _canonical_duplicate(rows):
    return min(
        rows,
        key=lambda row: (
            row["lifecycle"] != "referenced",
            row["lifecycle"] == "scratch_candidate",
            len(row["path"]),
            row["path"],
        ),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="legacy/catalog.jsonl")
    parser.add_argument("--output", default="legacy/catalog.sha256.jsonl")
    parser.add_argument("--duplicates", default="cleanup/exact_duplicates.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    assert_operation_allowed("architecture_inventory")
    source = _safe_artifact(args.input)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        hashed = list(pool.map(_hash_row, rows))
    hashed.sort(key=lambda row: row["path"])

    groups = defaultdict(list)
    for row in hashed:
        groups[(row["size_bytes"], row["sha256"])].append(row)
    duplicate_rows = []
    for (_, digest), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        canonical = _canonical_duplicate(members)
        for member in members:
            if member["path"] == canonical["path"]:
                continue
            duplicate_rows.append({
                "path": member["path"],
                "size_bytes": member["size_bytes"],
                "sha256": digest,
                "canonical_path": canonical["path"],
                "lifecycle": member["lifecycle"],
                "decision": "review_required",
                "deletion_authorized": False,
            })
    duplicate_rows.sort(key=lambda row: row["path"])
    output = _safe_artifact(args.output)
    duplicates = _safe_artifact(args.duplicates)
    _write_new_lines(output, hashed)
    _write_new_lines(duplicates, duplicate_rows)
    print(json.dumps({
        "verified_files": len(hashed),
        "verified_bytes": sum(row["size_bytes"] for row in hashed),
        "exact_duplicate_files": len(duplicate_rows),
        "exact_duplicate_bytes": sum(row["size_bytes"] for row in duplicate_rows),
        "catalog": str(output),
        "duplicates": str(duplicates),
        "deletion_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

