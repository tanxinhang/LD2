"""Storage-neutral run and artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple


RUN_STATES = {"created", "running", "completed", "failed", "audited"}


@dataclass(frozen=True)
class RunManifest:
    """Minimum identity required before a V2 run may write artifacts."""

    run_id: str
    run_type: str
    state: str
    created_at: str
    command: Tuple[str, ...]
    config_sha256: str
    code_identity: str
    seeds: Tuple[int, ...]
    inputs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or any(char in self.run_id for char in "/\\"):
            raise ValueError("run_id must be a non-empty path segment")
        if not self.run_type:
            raise ValueError("run_type must be non-empty")
        if self.state not in RUN_STATES:
            raise ValueError(f"invalid run state: {self.state!r}")
        if len(self.config_sha256) != 64:
            raise ValueError("config_sha256 must be a SHA-256 hex digest")
        if not self.code_identity:
            raise ValueError("code_identity must be non-empty")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("manifest seeds must be unique")


@dataclass(frozen=True)
class ArtifactRecord:
    run_id: str
    relative_path: str
    sha256: str
    size_bytes: int

