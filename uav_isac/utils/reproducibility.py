"""Fail-loud reproducibility metadata for formal experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "strict-distributed-run-manifest/v2"
SCENARIO_FINGERPRINT_VERSION = "reset-distribution/v2"
FORMAL_RUNTIME_PACKAGES = (
    "numpy", "scipy", "threadpoolctl", "torch", "gymnasium",
    "PyYAML", "psutil", "networkx", "scikit-learn",
)


def scenario_fingerprint(cfg: Any, config_identity: str) -> str:
    """Fingerprint every config field that changes reset/dynamics identity."""
    payload = {
        "fingerprint_version": SCENARIO_FINGERPRINT_VERSION,
        "config": str(config_identity).replace("\\", "/"),
        "region_size": list(cfg.scenario.region_size),
        "height": float(cfg.scenario.height),
        "K": int(cfg.scenario.K),
        "Q": int(cfg.scenario.Q),
        "T": int(cfg.scenario.T),
        "dt": float(cfg.scenario.dt),
        "v_max": float(cfg.uav.v_max),
        "d_safe": float(cfg.uav.d_safe),
        "tracking_enabled": bool(cfg.marl.tracking_enabled),
        "target_motion_model": str(cfg.target.motion_model),
        "target_speed_range": list(cfg.target.speed_range),
        "target_sigma_a": float(cfg.target.sigma_a),
        "target_ct_turn_rate": float(cfg.target.ct_turn_rate),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _reject_duplicate_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON document contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(value: str) -> Any:
    """Decode JSON without accepting ambiguous duplicate object keys."""
    return json.loads(value, object_pairs_hook=_reject_duplicate_json_object)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _normalise_source_bytes(content: bytes) -> bytes:
    """Make text-source hashes invariant to Git checkout line endings."""
    return content.replace(b"\r\n", b"\n")


def _sha256_source_file(path: Path) -> str:
    return _sha256_bytes(_normalise_source_bytes(path.read_bytes()))


def _portable_repo_path(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _is_source_path(relative: str) -> bool:
    path = Path(relative)
    normalized = path.as_posix()
    if normalized in {
        "requirements.txt", "constraints-ci.txt", "pytest.ini", ".gitignore",
    }:
        return True
    return (
        bool(path.parts)
        and path.parts[0] in {"config", "scripts", "tools", "uav_isac"}
        and path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}
    )


def _hash_source_records(records: Iterable[tuple[str, bytes]]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for relative_text, raw_content in sorted(records, key=lambda item: item[0]):
        relative = relative_text.encode("utf-8")
        content = _normalise_source_bytes(raw_content)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        count += 1
    return digest.hexdigest(), count


def _source_tree_hash(root: Path) -> tuple[str, int]:
    """Hash tracked and untracked experiment source, excluding result data."""
    candidates: list[Path] = []
    for source_root in ("config", "scripts", "tools", "uav_isac"):
        base = root / source_root
        if base.is_dir():
            candidates.extend(path for path in base.rglob("*") if path.is_file())
    candidates.extend(
        root / name for name in (
            "requirements.txt", "constraints-ci.txt", "pytest.ini", ".gitignore",
        )
        if (root / name).is_file()
    )
    records = (
        (relative, path.read_bytes())
        for path in candidates
        for relative in (path.relative_to(root).as_posix(),)
        if _is_source_path(relative)
    )
    return _hash_source_records(records)


def source_tree_hash_at_commit(root: Path, commit: str) -> tuple[str, int]:
    """Hash the experiment source from immutable Git blobs."""
    completed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", str(commit)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    paths = [
        line.strip().replace("\\", "/")
        for line in completed.stdout.splitlines()
        if line.strip() and _is_source_path(line.strip().replace("\\", "/"))
    ]
    records: list[tuple[str, bytes]] = []
    for relative in paths:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        records.append((relative, blob))
    return _hash_source_records(records)


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def build_run_manifest(
    cfg: Any,
    *,
    config_path: str,
    seeds: Iterable[int],
    algorithm_version: str,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Bind a result to code, effective config, seeds, runtime and hardware."""
    workspace = Path(root or Path(__file__).resolve().parents[2]).resolve()
    effective = asdict(cfg) if is_dataclass(cfg) else cfg
    effective_hash = _sha256_bytes(_canonical_json(effective))
    source_hash, source_files = _source_tree_hash(workspace)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = workspace / config_file
    seed_bank_path = str(getattr(
        getattr(cfg, "marl", object()), "eval_seed_bank_path", ""))
    seed_bank = Path(seed_bank_path) if seed_bank_path else None
    if seed_bank is not None and not seed_bank.is_absolute():
        seed_bank = workspace / seed_bank
    seed_bank_metadata: dict[str, Any] = {}
    if seed_bank is not None and seed_bank.is_file():
        try:
            raw_bank = _strict_json_loads(
                seed_bank.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raw_bank = None
        if isinstance(raw_bank, dict):
            split_name = str(getattr(
                getattr(cfg, "marl", object()),
                "final_eval_seed_split",
                "test",
            ))
            seed_bank_metadata = {
                "schema_version": raw_bank.get("schema_version"),
                "fingerprint_version": raw_bank.get("fingerprint_version"),
                "scenario_fingerprint": raw_bank.get("scenario_fingerprint"),
                "source_config": raw_bank.get("source_config"),
                "split": split_name,
            }
    status = _git(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status)
    try:
        branch = _git(workspace, "symbolic-ref", "--short", "HEAD")
    except subprocess.CalledProcessError:
        branch = "DETACHED"
    return {
        "schema_version": SCHEMA_VERSION,
        "workspace_root": str(workspace),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_version": str(algorithm_version),
        "formal_result_eligible": not dirty,
        "formal_result_ineligibility_reason": (
            "dirty workspace; commit/freeze before confirmatory execution"
            if dirty else ""
        ),
        "git": {
            "commit": _git(workspace, "rev-parse", "HEAD"),
            "branch": branch,
            "dirty": dirty,
            "status_sha256": _sha256_bytes(status.encode("utf-8")),
        },
        "source_tree": {
            "sha256": source_hash,
            "file_count": source_files,
            "scope": [
                "config/**/*.{py,yaml,yml,json}",
                "scripts/**/*.py",
                "tools/**/*.py",
                "uav_isac/**/*.py",
                "requirements.txt",
                "constraints-ci.txt",
                "pytest.ini",
                ".gitignore",
            ],
        },
        "config": {
            "path": _portable_repo_path(config_file, workspace),
            "source_sha256": (
                _sha256_source_file(config_file)
                if config_file.is_file() else "MISSING"
            ),
            "effective_sha256": effective_hash,
            "effective_snapshot": effective,
        },
        "seeds": {
            "values": [int(seed) for seed in seeds],
            "library_path": (
                _portable_repo_path(seed_bank, workspace) if seed_bank else ""),
            "library_sha256": (
                _sha256_source_file(seed_bank)
                if seed_bank is not None and seed_bank.is_file()
                else "MISSING" if seed_bank is not None else ""
            ),
            "library_metadata": seed_bank_metadata,
        },
        "runtime": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "packages": _package_versions(FORMAL_RUNTIME_PACKAGES),
            "thread_environment": {
                name: os.environ.get(name, "UNSET")
                for name in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            },
        },
    }


def validate_formal_run(
    cfg: Any,
    seeds: Iterable[int],
    manifest: dict[str, Any],
    *,
    minimum_seeds: int = 100,
) -> None:
    """Reject a confirmatory run unless its frozen identity is exact."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"formal run requires manifest schema {SCHEMA_VERSION}")
    if not bool(manifest.get("formal_result_eligible", False)):
        raise RuntimeError(
            "formal run refused: workspace is dirty or identity is incomplete")
    seed_values = [int(seed) for seed in seeds]
    if len(seed_values) < int(minimum_seeds):
        raise RuntimeError(
            f"formal run requires at least {int(minimum_seeds)} seeds")
    if len(seed_values) != len(set(seed_values)):
        raise RuntimeError("formal run seed list contains duplicates")
    workspace_value = str(manifest.get("workspace_root", "")).strip()
    if not workspace_value:
        raise RuntimeError("formal run manifest workspace_root is missing")
    workspace = Path(workspace_value).resolve()
    bank_path = Path(manifest["seeds"]["library_path"])
    if not bank_path.is_absolute():
        bank_path = workspace / bank_path
    if not bank_path.is_file():
        raise RuntimeError("formal run requires an existing frozen seed bank")
    bank = _strict_json_loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema_version") != 2:
        raise RuntimeError("formal run requires seed-bank schema_version 2")
    if str(bank.get("fingerprint_version", "")) != (
        SCENARIO_FINGERPRINT_VERSION
    ):
        raise RuntimeError(
            "formal run requires a reset-distribution/v2 seed bank; "
            "legacy bank must be regenerated from the frozen run config")
    bank_identity = str(bank.get("source_config", ""))
    recorded_bank_metadata = manifest.get("seeds", {}).get(
        "library_metadata", {})
    expected_bank_metadata = {
        "schema_version": bank.get("schema_version"),
        "fingerprint_version": bank.get("fingerprint_version"),
        "scenario_fingerprint": bank.get("scenario_fingerprint"),
        "source_config": bank.get("source_config"),
        "split": str(getattr(
            getattr(cfg, "marl", object()),
            "final_eval_seed_split",
            "test",
        )),
    }
    if recorded_bank_metadata != expected_bank_metadata:
        raise RuntimeError("formal run seed-bank metadata binding changed")
    expected_fingerprint = scenario_fingerprint(cfg, bank_identity)
    if str(bank.get("scenario_fingerprint", "")) != expected_fingerprint:
        raise RuntimeError(
            "formal run seed-bank scenario fingerprint does not match the "
            "effective K/Q/region/dynamics configuration")
    source_identity = str(bank.get("source_config", "")).strip()
    source_path = Path(source_identity.replace("\\", os.sep))
    if not source_path.is_absolute():
        source_path = workspace / source_path
    if not source_path.is_file():
        raise RuntimeError("formal run seed-bank source_config is missing")
    from config.params import load_config

    source_cfg = load_config(str(source_path))
    if scenario_fingerprint(source_cfg, source_identity) != str(
        bank.get("scenario_fingerprint", "")
    ):
        raise RuntimeError(
            "formal run seed-bank fingerprint does not match source_config")
    split_name = str(getattr(
        getattr(cfg, "marl", object()), "final_eval_seed_split", "test"))
    expected = [int(seed) for seed in bank.get("splits", {}).get(
        split_name, [])]
    if seed_values != expected:
        raise RuntimeError(
            "formal run seeds must exactly match the configured frozen split "
            f"{split_name!r}, including order")
    excluded = set(int(seed) for seed in bank.get("quarantined_excluded", []))
    excluded.update(int(seed) for seed in bank.get("development_excluded", []))
    legacy_excluded = bank.get("excluded", {})
    if isinstance(legacy_excluded, dict):
        excluded.update(int(seed) for seed in legacy_excluded.get(
            "quarantined", []))
    overlap = excluded.intersection(seed_values)
    if overlap:
        raise RuntimeError(
            "formal run seed split contains quarantined/development seeds: "
            f"{sorted(overlap)}")
    splits = bank.get("splits", {})
    if not isinstance(splits, dict):
        raise RuntimeError("formal run seed-bank splits must be a mapping")
    for other_name, other_values in splits.items():
        if str(other_name) == split_name:
            continue
        if not isinstance(other_values, list):
            raise RuntimeError(
                f"formal run seed-bank split {other_name!r} must be a list")
        leaked = set(seed_values).intersection(int(seed) for seed in other_values)
        if leaked:
            raise RuntimeError(
                "formal run seed split overlaps another split "
                f"{other_name!r}: {sorted(leaked)}")
    thread_environment = manifest["runtime"]["thread_environment"]
    non_unit = {
        name: value for name, value in thread_environment.items()
        if str(value) != "1"
    }
    if non_unit:
        raise RuntimeError(
            "formal run requires deterministic single-thread numerical "
            f"environment; set all thread variables to 1: {non_unit}")

    # Recompute every recorded content binding immediately before execution;
    # callers cannot manufacture eligibility by flipping one manifest flag.
    if not workspace.is_dir():
        raise RuntimeError("formal run manifest workspace_root is invalid")
    current_status = _git(
        workspace, "status", "--porcelain=v1", "--untracked-files=all")
    current_commit = _git(workspace, "rev-parse", "HEAD")
    git_record = manifest.get("git", {})
    if current_status or git_record.get("dirty") is not False:
        raise RuntimeError("formal run refused: workspace is dirty")
    if git_record.get("commit") != current_commit or not re.fullmatch(
        r"[0-9a-f]{40}", str(current_commit)
    ):
        raise RuntimeError("formal run Git commit binding is invalid")
    if git_record.get("status_sha256") != _sha256_bytes(b""):
        raise RuntimeError("formal run Git status hash is not the clean-tree hash")

    source_hash, source_count = _source_tree_hash(workspace)
    source_record = manifest.get("source_tree", {})
    if (
        source_record.get("sha256") != source_hash
        or int(source_record.get("file_count", -1)) != source_count
    ):
        raise RuntimeError("formal run source-tree hash changed")
    commit_source_hash, commit_source_count = source_tree_hash_at_commit(
        workspace, current_commit)
    if (
        commit_source_hash != source_hash
        or commit_source_count != source_count
    ):
        raise RuntimeError(
            "formal run source tree does not match immutable Git blobs")
    effective = asdict(cfg) if is_dataclass(cfg) else cfg
    if manifest.get("config", {}).get("effective_sha256") != _sha256_bytes(
        _canonical_json(effective)
    ):
        raise RuntimeError("formal run effective configuration hash changed")

    config_path = Path(manifest.get("config", {}).get("path", ""))
    if not config_path.is_absolute():
        config_path = workspace / config_path
    config_path = config_path.resolve()
    bound_bank_path = Path(
        manifest.get("seeds", {}).get("library_path", ""))
    if not bound_bank_path.is_absolute():
        bound_bank_path = workspace / bound_bank_path
    bound_bank_path = bound_bank_path.resolve()
    for label, path, recorded_hash in (
        ("config", config_path,
         manifest.get("config", {}).get("source_sha256")),
        ("seed bank", bound_bank_path,
         manifest.get("seeds", {}).get("library_sha256")),
    ):
        try:
            relative = path.relative_to(workspace).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"formal run {label} must reside inside the repository") from exc
        if not path.is_file() or recorded_hash != _sha256_source_file(path):
            raise RuntimeError(f"formal run {label} content hash changed")
        try:
            _git(workspace, "ls-files", "--error-unmatch", relative)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"formal run {label} is not tracked by Git") from exc

    package_versions = manifest.get("runtime", {}).get("packages", {})
    missing_records = [
        name for name in FORMAL_RUNTIME_PACKAGES
        if name not in package_versions
    ]
    if missing_records:
        raise RuntimeError(
            "formal run manifest omits required dependencies: "
            + ", ".join(missing_records))
    missing_packages = [
        name for name, version in package_versions.items()
        if version in (None, "", "NOT_INSTALLED")
    ]
    if missing_packages:
        raise RuntimeError(
            "formal run has missing recorded dependencies: "
            + ", ".join(sorted(missing_packages)))
