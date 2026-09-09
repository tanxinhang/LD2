"""Artifact persistence port used by application services."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from uav_isac.domain.artifacts import ArtifactRecord, RunManifest


class ArtifactStore(Protocol):
    def create_run(self, manifest: RunManifest) -> ArtifactRecord:
        ...

    def write_json(
        self,
        run_id: str,
        relative_path: str,
        payload: Mapping[str, Any],
    ) -> ArtifactRecord:
        ...

