"""Machine-checkable gate between research refactoring and algorithm work."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Mapping, Tuple

from .architecture import REPOSITORY_ROOT, audit_architecture
from .data_inventory import build_data_inventory
from .entrypoint_inventory import build_entrypoint_inventory
from .project_phase import load_project_phase
from .research_programs import load_research_programs


@dataclass(frozen=True)
class ResearchReadiness:
    ready_for_algorithm_optimization: bool
    blockers: Tuple[str, ...]
    metrics: Mapping[str, int]


def _tracked(repository: Path, relative: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def audit_research_readiness(
    repository_root: Path | str = REPOSITORY_ROOT,
) -> ResearchReadiness:
    """Return explicit blockers; never mutate phase or research artifacts."""
    repository = Path(repository_root).resolve()
    blockers = []

    violations = audit_architecture()
    if violations:
        blockers.append(f"architecture violations remain: {len(violations)}")

    entrypoints = build_entrypoint_inventory(repository)
    adapter_backends = tuple(
        row.path for row in entrypoints if row.status == "adapter_backend")
    if adapter_backends:
        blockers.append(
            "canonical commands still delegate legacy backends: "
            + ", ".join(adapter_backends))

    inventory = build_data_inventory(repository)
    ambiguous = sum(
        bucket.files for bucket in inventory.buckets
        if bucket.lifecycle == "unclassified"
    )
    if ambiguous:
        blockers.append(f"result files remain unclassified: {ambiguous}")

    programs = load_research_programs()
    missing_program_assets = []
    for program in programs:
        for relative in (program.implementation, program.profile, program.evidence):
            path = (repository / relative).resolve()
            if not path.is_relative_to(repository) or not path.exists():
                missing_program_assets.append(f"{program.name}:{relative}")
    if missing_program_assets:
        blockers.append(
            "active research registry has missing assets: "
            + ", ".join(missing_program_assets))

    registry_path = repository / "formal_evidence" / "registry.json"
    portable_assets = 0
    missing_evidence = []
    if registry_path.is_file():
        document = json.loads(registry_path.read_text(encoding="utf-8"))
        for entry in document.get("entries", []):
            for field in ("csv", "run_manifest"):
                relative = str(entry.get(field, "")).replace("\\", "/")
                path = (repository / relative).resolve()
                if (
                    relative.startswith("artifacts/")
                    and path.is_relative_to(repository)
                    and path.is_file()
                    and _tracked(repository, relative)
                ):
                    portable_assets += 1
                else:
                    missing_evidence.append(f"{field}:{relative}")
    else:
        missing_evidence.append("formal_evidence/registry.json")
    if missing_evidence:
        blockers.append(
            "formal evidence is not portable and tracked: "
            + ", ".join(missing_evidence))

    phase = load_project_phase()
    if blockers and phase.allows("algorithm_optimization"):
        blockers.append("algorithm optimization was enabled before refactor readiness")

    return ResearchReadiness(
        ready_for_algorithm_optimization=not blockers,
        blockers=tuple(blockers),
        metrics={
            "entrypoints": len(entrypoints),
            "legacy_adapter_backends": len(adapter_backends),
            "active_research_programs": len(programs),
            "ambiguous_result_files": ambiguous,
            "portable_formal_assets": portable_assets,
        },
    )
