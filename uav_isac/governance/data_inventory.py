"""Read-only lifecycle inventory for legacy result data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


REFERENCE_ROOTS = ("docs", "paper", "frozen", "config")
REFERENCE_SUFFIXES = {".md", ".yaml", ".yml", ".json", ".txt", ".py"}
SCRATCH_MARKERS = (
    "smoke",
    "crash",
    "debug",
    "probe",
    "scratch",
    "tmp",
    "test",
    "console",
)


@dataclass(frozen=True)
class DataBucket:
    lifecycle: str
    files: int
    size_bytes: int


@dataclass(frozen=True)
class LargeResult:
    path: str
    size_bytes: int
    lifecycle: str


@dataclass(frozen=True)
class DataFile:
    path: str
    size_bytes: int
    modified_ns: int
    lifecycle: str


@dataclass(frozen=True)
class DataInventory:
    result_root: str
    total_files: int
    total_size_bytes: int
    buckets: Tuple[DataBucket, ...]
    largest: Tuple[LargeResult, ...]


def _reference_text(repository: Path) -> str:
    chunks = []
    for root_name in REFERENCE_ROOTS:
        root = repository / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in REFERENCE_SUFFIXES:
                try:
                    chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    continue
    return "\n".join(chunks).replace("\\", "/")


def _lifecycle(relative: str, references: str) -> str:
    normalized = relative.replace("\\", "/")
    if normalized in references:
        return "referenced"
    parts = Path(normalized).parts
    if parts and parts[0].lower() == "_archive":
        return "legacy"
    lowered = normalized.lower()
    if any(marker in lowered for marker in SCRATCH_MARKERS):
        return "scratch_candidate"
    # `results/` is the explicitly quarantined pre-V2 store. Files that are
    # neither cited nor obvious scratch are retained as historical research
    # inputs, but lack a V2 run manifest and therefore cannot support a new
    # claim. This is a lifecycle decision, not deletion authorization.
    return "legacy_unmanifested"


def build_data_inventory(
    repository_root: Path | str,
    result_directory: str = "results",
    largest_count: int = 20,
) -> DataInventory:
    """Classify result files without modifying them.

    `scratch_candidate` is only a review label. It never grants deletion
    authority. `legacy_unmanifested` is read-only historical material and may
    not be promoted into a new result without an explicit V2 manifest.
    """

    repository = Path(repository_root).resolve()
    result_root = (repository / result_directory).resolve()
    if not result_root.is_relative_to(repository):
        raise ValueError("result directory escapes repository")
    data_files = build_data_files(repository, result_directory)
    records = [
        (item.path, item.size_bytes, item.lifecycle)
        for item in data_files
    ]

    bucket_values = {}
    for _, size, lifecycle in records:
        count, total = bucket_values.get(lifecycle, (0, 0))
        bucket_values[lifecycle] = (count + 1, total + size)
    buckets = tuple(
        DataBucket(name, values[0], values[1])
        for name, values in sorted(bucket_values.items())
    )
    largest = tuple(
        LargeResult(path, size, lifecycle)
        for path, size, lifecycle in sorted(
            records, key=lambda item: item[1], reverse=True
        )[:max(0, largest_count)]
    )
    return DataInventory(
        result_root=str(result_root),
        total_files=len(records),
        total_size_bytes=sum(item[1] for item in records),
        buckets=buckets,
        largest=largest,
    )


def build_data_files(
    repository_root: Path | str,
    result_directory: str = "results",
) -> Tuple[DataFile, ...]:
    """Return the immutable file-level input for lifecycle catalogs."""

    repository = Path(repository_root).resolve()
    result_root = (repository / result_directory).resolve()
    if not result_root.is_relative_to(repository):
        raise ValueError("result directory escapes repository")
    references = _reference_text(repository)
    records = []
    if result_root.exists():
        for path in result_root.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            relative = path.relative_to(result_root).as_posix()
            records.append(DataFile(
                path=relative,
                size_bytes=stat.st_size,
                modified_ns=stat.st_mtime_ns,
                lifecycle=_lifecycle(relative, references),
            ))
    return tuple(sorted(records, key=lambda item: item.path))
