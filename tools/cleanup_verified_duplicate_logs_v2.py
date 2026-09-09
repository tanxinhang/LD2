#!/usr/bin/env python
"""Delete only reviewed scratch logs whose exact content remains elsewhere."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uav_isac.governance.data_inventory import build_data_files
from uav_isac.governance.project_phase import assert_operation_allowed


def _artifact(relative: str) -> Path:
    root = (REPOSITORY_ROOT / "artifacts").resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("artifact path must stay below artifacts/")
    return path


def _result(relative: str) -> Path:
    root = (REPOSITORY_ROOT / "results").resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("result path must stay below results/")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_candidates(source: Path) -> list[dict]:
    current = {item.path: item for item in build_data_files(REPOSITORY_ROOT)}
    candidates = []
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        relative = row["path"]
        if row.get("lifecycle") != "scratch_candidate":
            continue
        if Path(relative).suffix.lower() != ".log":
            continue
        item = current.get(relative)
        if item is None or item.lifecycle != "scratch_candidate":
            raise RuntimeError(f"candidate disappeared or became referenced: {relative}")
        target = _result(relative)
        canonical = _result(row["canonical_path"])
        if not target.is_file() or not canonical.is_file():
            raise FileNotFoundError(f"duplicate pair is incomplete: {relative}")
        if target.stat().st_size != row["size_bytes"]:
            raise RuntimeError(f"candidate size changed: {relative}")
        expected = row["sha256"]
        if _sha256(target) != expected or _sha256(canonical) != expected:
            raise RuntimeError(f"duplicate hash verification failed: {relative}")
        candidates.append({
            "path": relative,
            "canonical_path": row["canonical_path"],
            "size_bytes": row["size_bytes"],
            "sha256": expected,
        })
    return candidates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cleanup/exact_duplicates.jsonl")
    parser.add_argument("--report", default="cleanup/redundant_logs_v2.jsonl")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", default=None)
    args = parser.parse_args(argv)
    assert_operation_allowed("destructive_data_cleanup")
    if args.execute and not args.authorization:
        parser.error("--execute requires a non-empty --authorization record")

    candidates = _verified_candidates(_artifact(args.input))
    summary = {
        "mode": "execute" if args.execute else "dry_run",
        "files": len(candidates),
        "bytes": sum(item["size_bytes"] for item in candidates),
    }
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return 0

    report = _artifact(args.report)
    if report.exists():
        raise FileExistsError(f"refusing to overwrite cleanup report: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_name(report.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale cleanup journal exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        header = {
            "record_type": "cleanup_start",
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "authorization": args.authorization,
            **summary,
        }
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        for item in candidates:
            target = _result(item["path"])
            canonical = _result(item["canonical_path"])
            if _sha256(target) != item["sha256"] or _sha256(canonical) != item["sha256"]:
                raise RuntimeError(f"pre-delete hash verification failed: {item['path']}")
            target.unlink()
            record = {
                "record_type": "deleted",
                **item,
                "content_recoverable_from": item["canonical_path"],
            }
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(report)
    print(json.dumps({**summary, "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
