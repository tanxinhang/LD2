"""Deterministic Git/source identity for reproducible V2 manifests."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATHS = (
    "uav_isac",
    "config",
    "scripts",
    "tools",
    "requirements.txt",
    "constraints-ci.txt",
)


@dataclass(frozen=True)
class RepositoryIdentity:
    commit: str
    dirty: bool
    sha256: str


def _git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def build_repository_identity(
    root: Path | str = REPOSITORY_ROOT,
) -> RepositoryIdentity:
    """Hash the commit plus tracked and untracked runtime source changes."""

    repository = Path(root).resolve()
    commit = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    tracked_patch = _git(
        repository,
        "diff",
        "--binary",
        "HEAD",
        "--",
        *RUNTIME_PATHS,
    )
    untracked_raw = _git(
        repository,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *RUNTIME_PATHS,
    ).decode("utf-8")
    untracked = tuple(sorted(line for line in untracked_raw.splitlines() if line))

    digest = hashlib.sha256()
    digest.update(b"commit\0" + commit.encode("ascii") + b"\0")
    digest.update(b"tracked-patch\0" + tracked_patch + b"\0")
    for relative in untracked:
        path = (repository / relative).resolve()
        if not path.is_relative_to(repository) or not path.is_file():
            raise ValueError(f"unsafe untracked source path: {relative}")
        digest.update(b"untracked\0" + relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return RepositoryIdentity(
        commit=commit,
        dirty=bool(tracked_patch or untracked),
        sha256=digest.hexdigest(),
    )

