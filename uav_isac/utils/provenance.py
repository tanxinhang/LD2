"""Content-addressed provenance for training and evaluation artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile


_SOURCE_PATTERNS = (
    "uav_isac/**/*.py",
    "config/**/*.py",
    "config/**/*.yaml",
    "config/**/*.json",
    "scripts/**/*.py",
    "tools/**/*.py",
)
_ROOT_FILES = (
    "requirements.txt",
    "constraints-ci.txt",
    "pytest.ini",
    ".gitignore",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(root: Path) -> tuple[Path, ...]:
    files: set[Path] = set()
    for pattern in _SOURCE_PATTERNS:
        files.update(path for path in root.glob(pattern) if path.is_file())
    files.update(root / name for name in _ROOT_FILES if (root / name).is_file())
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def write_source_snapshot(root: str | Path, output: str | Path) -> dict:
    """Write a deterministic ZIP containing the exact executable source tree."""
    root_path = Path(root).resolve()
    output_path = Path(output)
    files = _source_files(root_path)
    with zipfile.ZipFile(
            output_path, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(root_path).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "file_count": len(files),
    }


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_run_provenance(
    *,
    root: str | Path,
    config: object,
    source_snapshot: dict,
    checkpoint_paths: tuple[str | None, ...] = (),
) -> dict:
    """Return a JSON-safe record binding code, configuration and runtime."""
    root_path = Path(root).resolve()
    resolved = asdict(config) if is_dataclass(config) else config
    encoded = json.dumps(
        resolved, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    packages = {}
    for distribution in (
            "numpy", "scipy", "torch", "PyYAML", "gymnasium",
            "scikit-learn", "psutil", "pytest"):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None
    checkpoints = []
    for raw_path in checkpoint_paths:
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        checkpoints.append({
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        })
    commit = _git(root_path, "rev-parse", "HEAD")
    status = _git(root_path, "status", "--porcelain=v1", "--untracked-files=all")
    git_available = commit is not None and status is not None
    return {
        "schema_version": 1,
        "git_commit": commit,
        # A missing Git executable/repository is not evidence of a clean tree.
        "git_available": git_available,
        "git_dirty": (not git_available) or bool(status),
        "git_status_porcelain": status.splitlines() if status else [],
        "source_snapshot": source_snapshot,
        "resolved_config": resolved,
        "resolved_config_sha256": hashlib.sha256(encoded).hexdigest(),
        "checkpoints": checkpoints,
        "command": [str(value) for value in sys.argv],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
    }
