"""Fail-closed filesystem implementation of the V2 artifact store."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from uav_isac.domain.artifacts import ArtifactRecord, RunCompletion, RunManifest


class FileSystemArtifactStore:
    """Write immutable JSON artifacts below one explicit root directory."""

    def __init__(self, root: Path | str):
        self._root = Path(root).resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _run_directory(self, run_id: str) -> Path:
        if not run_id or any(char in run_id for char in "/\\") or run_id in {".", ".."}:
            raise ValueError("run_id must be one safe path segment")
        return self._root / "runs" / run_id

    def _artifact_path(self, run_id: str, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            raise ValueError("artifact path must be a safe relative path")
        run_directory = self._run_directory(run_id)
        resolved = (run_directory / candidate).resolve()
        if not resolved.is_relative_to(run_directory.resolve()):
            raise ValueError("artifact path escapes its run directory")
        return resolved

    @staticmethod
    def _encode(payload: Mapping[str, Any]) -> bytes:
        return (json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n").encode("utf-8")

    @staticmethod
    def _record(run_id: str, relative_path: str, data: bytes) -> ArtifactRecord:
        return ArtifactRecord(
            run_id=run_id,
            relative_path=relative_path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    @staticmethod
    def _write_new(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite artifact: {path}") from exc

    def create_run(self, manifest: RunManifest) -> ArtifactRecord:
        if manifest.state != "created":
            raise ValueError("a new run manifest must start in created state")
        run_directory = self._run_directory(manifest.run_id)
        run_directory.mkdir(parents=True, exist_ok=False)
        relative_path = "manifest.json"
        data = self._encode(dataclasses.asdict(manifest))
        self._write_new(run_directory / relative_path, data)
        return self._record(manifest.run_id, relative_path, data)

    def write_json(
        self,
        run_id: str,
        relative_path: str,
        payload: Mapping[str, Any],
    ) -> ArtifactRecord:
        run_directory = self._run_directory(run_id)
        if not (run_directory / "manifest.json").is_file():
            raise FileNotFoundError(f"run has no manifest: {run_id}")
        path = self._artifact_path(run_id, relative_path)
        data = self._encode(payload)
        self._write_new(path, data)
        return self._record(run_id, relative_path, data)

    def complete_run(self, completion: RunCompletion) -> ArtifactRecord:
        """Write a terminal record only after verifying every bound artifact."""
        run_directory = self._run_directory(completion.run_id)
        if not (run_directory / "manifest.json").is_file():
            raise FileNotFoundError(f"run has no manifest: {completion.run_id}")
        for relative_path, expected_sha256 in completion.artifacts.items():
            path = self._artifact_path(completion.run_id, relative_path)
            if not path.is_file():
                raise FileNotFoundError(f"completion artifact is missing: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise ValueError(
                    f"completion artifact hash mismatch: {relative_path}")
        data = self._encode(dataclasses.asdict(completion))
        relative_path = "completion.json"
        self._write_new(run_directory / relative_path, data)
        return self._record(completion.run_id, relative_path, data)
