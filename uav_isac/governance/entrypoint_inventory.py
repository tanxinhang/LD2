"""Classify Python entrypoints during the V2 strangler migration."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


CANONICAL = {"uav_isac/interfaces/cli.py"}
GOVERNANCE = {
    "tools/audit_data_lifecycle_v2.py",
    "tools/build_legacy_data_catalog_v2.py",
    "tools/build_entrypoint_catalog_v2.py",
    "tools/check_architecture_v2.py",
    "tools/check_project_phase.py",
    "tools/hash_legacy_data_catalog_v2.py",
    "tools/audit_architecture_migration_v2.py",
    "tools/cleanup_verified_duplicate_logs_v2.py",
}
BACKENDS = {
    "scripts/run_mappo.py": "algorithm_optimization",
    "tools/run_strict_distributed_pilot.py": "migration_audit",
    "tools/run_strict_distributed_bank.py": "full_result_refresh",
}


@dataclass(frozen=True)
class EntrypointRecord:
    path: str
    status: str
    owner: str
    operation: str


def _has_main_guard(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
        ):
            return True
    return False


def _legacy_operation(path: str) -> str:
    name = Path(path).stem.lower()
    if name.startswith(("audit", "check", "verify", "diagnose", "probe")):
        return "legacy_audit"
    if name.startswith(("train", "dagger", "pretrain")):
        return "legacy_training"
    if name.startswith(("benchmark", "run_")):
        return "legacy_experiment"
    if name.startswith("test"):
        return "legacy_probe"
    return "legacy_utility"


def build_entrypoint_inventory(repository_root: Path | str) -> Tuple[EntrypointRecord, ...]:
    repository = Path(repository_root).resolve()
    records = []
    for root_name in ("uav_isac", "scripts", "tools"):
        root = repository / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if not _has_main_guard(path):
                continue
            relative = path.relative_to(repository).as_posix()
            if relative in CANONICAL:
                status, owner, operation = "canonical", "architecture", "multi"
            elif relative in GOVERNANCE:
                status, owner, operation = "governance", "architecture", "migration"
            elif relative in BACKENDS:
                status, owner, operation = "adapter_backend", "legacy_runtime", BACKENDS[relative]
            else:
                status, owner, operation = (
                    "legacy_retained",
                    "legacy_reproduction",
                    _legacy_operation(relative),
                )
            records.append(EntrypointRecord(relative, status, owner, operation))
    return tuple(sorted(records, key=lambda item: item.path))
