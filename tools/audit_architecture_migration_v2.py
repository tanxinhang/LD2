#!/usr/bin/env python
"""Run the fail-closed architecture V2 migration audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uav_isac.adapters import (
    HoldPositionPolicy,
    build_legacy_environment,
    load_registered_configuration,
)
from uav_isac.application import EpisodeRunner
from uav_isac.domain import EpisodeSpec
from uav_isac.governance import (
    audit_architecture,
    load_characterization_baselines,
    load_project_phase,
    load_runtime_profiles,
    semantic_fingerprint,
)
from uav_isac.governance.entrypoint_inventory import build_entrypoint_inventory


def _check_phase():
    phase = load_project_phase()
    assert phase.phase in {"architecture_audit", "reproduction", "result_refresh"}
    expected = (
        "migration_freeze_declared",
        "baseline_characterized",
        "architecture_migrated",
        "architecture_audited",
        "reproduction_passed",
        "result_refresh_approved",
    )
    assert phase.required_gate_order == expected
    if phase.phase == "result_refresh":
        assert phase.completed_gates == expected
    else:
        assert phase.completed_gates == expected[:len(phase.completed_gates)]


def _check_baselines():
    for baseline in load_characterization_baselines():
        result = EpisodeRunner(
            build_legacy_environment(baseline.config, baseline.seed),
            HoldPositionPolicy(),
        ).run(EpisodeSpec(baseline.seed, baseline.frames))
        assert semantic_fingerprint(result) == baseline.semantic_sha256


def _check_profiles():
    profiles = load_runtime_profiles()
    assert profiles
    for profile in profiles:
        assert len(load_registered_configuration(profile.name).sha256) == 64


def _check_packaging():
    payload = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text("utf-8"))
    assert payload["project"]["scripts"]["uav-isac"] == "uav_isac.interfaces.cli:main"
    assert payload["project"]["requires-python"] == ">=3.10"


def _check_data_catalog():
    source = (
        REPOSITORY_ROOT
        / "artifacts/legacy/catalog.post_classified_cleanup.sha256.jsonl"
    )
    rows = [json.loads(line) for line in source.read_text("utf-8").splitlines()]
    assert rows and all(row["hash_status"] == "verified" for row in rows)
    assert all(len(row["sha256"]) == 64 for row in rows)
    result_root = (REPOSITORY_ROOT / "results").resolve()
    for row in rows:
        path = (result_root / row["path"]).resolve()
        assert path.is_relative_to(result_root) and path.is_file()
        stat = path.stat()
        assert stat.st_size == row["size_bytes"]
        assert stat.st_mtime_ns == row["modified_ns"]
    cleanup_report = REPOSITORY_ROOT / "artifacts/cleanup/redundant_logs_v2.jsonl"
    cleanup_rows = [
        json.loads(line) for line in cleanup_report.read_text("utf-8").splitlines()
    ]
    assert cleanup_rows[0]["record_type"] == "cleanup_start"
    deleted = cleanup_rows[1:]
    assert len(deleted) == cleanup_rows[0]["files"]
    assert sum(row["size_bytes"] for row in deleted) == cleanup_rows[0]["bytes"]
    for row in deleted:
        assert row["record_type"] == "deleted"
        assert not (result_root / row["path"]).exists()
        canonical = (result_root / row["content_recoverable_from"]).resolve()
        assert canonical.is_relative_to(result_root) and canonical.is_file()
    duplicates = [
        json.loads(line)
        for line in (
            REPOSITORY_ROOT / "artifacts/cleanup/exact_duplicates.post_cleanup.jsonl"
        ).read_text("utf-8").splitlines()
    ]
    assert all(not row["deletion_authorized"] for row in duplicates)


def _check_entrypoints():
    source = REPOSITORY_ROOT / "artifacts/legacy/entrypoints.v3.jsonl"
    rows = [json.loads(line) for line in source.read_text("utf-8").splitlines()]
    assert sum(row["status"] == "canonical" for row in rows) == 1
    assert sum(row["status"] == "adapter_backend" for row in rows) == 3
    assert all(row["owner"] and row["operation"] for row in rows)
    current = [
        {
            "path": row.path,
            "status": row.status,
            "owner": row.owner,
            "operation": row.operation,
        }
        for row in build_entrypoint_inventory(REPOSITORY_ROOT)
    ]
    assert rows == current, "entrypoint catalog is stale; regenerate entrypoints.v3.jsonl"


def _run_full_tests():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout[-8000:])
    lines = completed.stdout.strip().splitlines()
    return lines[-1] if lines else "pytest completed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="include full pytest")
    parser.add_argument("--output", default="audits/architecture_migration_v2.json")
    args = parser.parse_args(argv)
    checks = (
        ("phase", _check_phase),
        ("dependency_boundaries", lambda: (
            None if not audit_architecture()
            else (_ for _ in ()).throw(AssertionError("architecture violations found"))
        )),
        ("semantic_baselines", _check_baselines),
        ("runtime_profiles", _check_profiles),
        ("packaging", _check_packaging),
        ("data_catalog", _check_data_catalog),
        ("entrypoint_catalog", _check_entrypoints),
    )
    results = []
    for name, check in checks:
        try:
            check()
            results.append({"name": name, "status": "pass"})
        except Exception as exc:
            results.append({"name": name, "status": "fail", "detail": str(exc)})
    if args.full:
        try:
            detail = _run_full_tests()
            results.append({"name": "full_pytest", "status": "pass", "detail": detail})
        except Exception as exc:
            results.append({"name": "full_pytest", "status": "fail", "detail": str(exc)})

    output = (REPOSITORY_ROOT / "artifacts" / args.output).resolve()
    artifact_root = (REPOSITORY_ROOT / "artifacts").resolve()
    if not output.is_relative_to(artifact_root) or output == artifact_root:
        raise ValueError("audit output must stay below artifacts/")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "passed": all(item["status"] == "pass" for item in results),
        "checks": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
