#!/usr/bin/env python
"""Audit the results/ tree governance (audit remediation 2026-08-16).

Every experiment directory should be reproducible from its own files: a
paired_eval.csv (per-episode aggregates), a run_manifest.json (git commit +
full config) and, for gates, a summary.json.  This tool scans the tree and
reports:
  - totals and per-artifact coverage;
  - directories missing all three artifacts (unreproducible);
  - scratch/`_`-prefixed directories mixed with formal results;
  - summary.json schema heterogeneity (distinct top-level key sets).

It is read-only: it never moves or modifies anything under results/.
Physical archiving of scratch dirs is a separate, user-confirmed step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

ARTIFACTS = ("summary.json", "paired_eval.csv", "run_manifest.json")


def scan_tree(root: str) -> Dict[str, object]:
    dirs = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    coverage = Counter()
    missing_all: List[str] = []
    scratch: List[str] = []
    schemas: Counter = Counter()
    schema_examples: Dict[str, str] = {}
    per_dir: Dict[str, List[str]] = {}
    for name in sorted(dirs):
        path = os.path.join(root, name)
        present = [a for a in ARTIFACTS
                   if os.path.isfile(os.path.join(path, a))]
        coverage.update(present)
        per_dir[name] = present
        if name.startswith("_"):
            scratch.append(name)
        if not present:
            missing_all.append(name)
        if "summary.json" in present:
            try:
                with open(os.path.join(path, "summary.json"),
                          encoding="utf-8") as handle:
                    keys = tuple(sorted(json.load(handle).keys()))
                schemas[keys] += 1
                schema_examples.setdefault(keys, name)
            except (OSError, ValueError):
                schemas["<unreadable>"] += 1
                schema_examples.setdefault("<unreadable>", name)
    return {
        "root": os.path.abspath(root),
        "total_dirs": len(dirs),
        "artifact_coverage": dict(coverage),
        "missing_all": missing_all,
        "scratch_dirs": scratch,
        "schema_variants": len(schemas),
        "schema_counts": {str(k): v for k, v in schemas.items()},
        "schema_examples": schema_examples,
        "per_dir": per_dir,
    }


def render(report: Dict[str, object]) -> str:
    lines = [
        f"results/ tree: {report['total_dirs']} dirs",
        f"  artifacts: {report['artifact_coverage']}",
        f"  dirs missing all three artifacts: {len(report['missing_all'])}",
        f"  scratch (_-prefixed) dirs: {len(report['scratch_dirs'])}",
        f"  summary.json schema variants: {report['schema_variants']}",
    ]
    if report["missing_all"]:
        lines.append("  missing-all sample: "
                     + ", ".join(report["missing_all"][:10]))
    if report["scratch_dirs"]:
        lines.append("  scratch sample: "
                     + ", ".join(report["scratch_dirs"][:10]))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args(argv)

    report = scan_tree(args.root)
    print(render(report))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
