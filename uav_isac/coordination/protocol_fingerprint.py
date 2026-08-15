"""Deterministic fingerprint of the U2U protocol implementation and config."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import yaml


DEFAULT_IMPLEMENTATION_PATHS = (
    "uav_isac/environment/communication.py",
    "uav_isac/coordination/power_repair_transport.py",
    "uav_isac/coordination/owner_proposal_transport.py",
    "uav_isac/coordination/target_invariant_transport.py",
    "uav_isac/coordination/dependency_commit.py",
    "uav_isac/coordination/structure_sequence_transport.py",
    "uav_isac/coordination/digest_rendezvous.py",
    "uav_isac/coordination/bounded_repetition.py",
    "uav_isac/evaluation/link_reliability_calibration.py",
    "uav_isac/evaluation/finite_sample_feasibility.py",
    "tools/calibrate_link_reliability_epoch.py",
)


@dataclass(frozen=True)
class FingerprintedFile:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ProtocolImplementationFingerprint:
    schema_version: int
    sha256: str
    files: tuple[FingerprintedFile, ...]
    config_chain: tuple[str, ...]


@dataclass(frozen=True)
class ControllerImplementationFingerprint:
    schema_version: int
    sha256: str
    files: tuple[FingerprintedFile, ...]
    config_chain: tuple[str, ...]


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"fingerprinted path is outside workspace: {path}") from exc


def _config_chain(config_path: Path, root: Path) -> tuple[Path, ...]:
    seen: set[Path] = set()

    def visit(path: Path) -> tuple[Path, ...]:
        resolved = path.resolve()
        _inside(root, resolved)
        if resolved in seen:
            raise ValueError(f"cyclic config inheritance involving {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        seen.add(resolved)
        document = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(document, dict):
            raise ValueError(f"config root must be a mapping: {resolved}")
        parent = document.get("extends")
        if parent is None:
            chain = (resolved,)
        else:
            if not isinstance(parent, str):
                raise ValueError("config extends must be a path string")
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = resolved.parent / parent_path
            chain = visit(parent_path) + (resolved,)
        seen.remove(resolved)
        return chain

    return visit(config_path)


def _fingerprint_implementation(
    config_path: Path,
    *,
    workspace_root: Path,
    implementation_paths: Sequence[str],
    domain: bytes,
    result_type: type[
        ProtocolImplementationFingerprint | ControllerImplementationFingerprint
    ],
) -> ProtocolImplementationFingerprint | ControllerImplementationFingerprint:
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError("workspace_root must be an existing directory")
    relative_sources = tuple(str(value).replace("\\", "/") for value in (
        implementation_paths))
    if not relative_sources or len(set(relative_sources)) != len(relative_sources):
        raise ValueError("implementation_paths must be nonempty and unique")
    source_paths = tuple(root / value for value in relative_sources)
    config_paths = _config_chain(Path(config_path), root)
    ordered_paths = source_paths + config_paths
    digest = hashlib.sha256()
    digest.update(domain + b"\x00")
    records = []
    for path in ordered_paths:
        relative = _inside(root, path).as_posix()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        file_digest = hashlib.sha256(payload).hexdigest()
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        records.append(FingerprintedFile(
            path=relative,
            sha256=file_digest,
            size_bytes=len(payload),
        ))
    config_chain = tuple(
        _inside(root, path).as_posix() for path in config_paths)
    return result_type(
        schema_version=1,
        sha256=digest.hexdigest(),
        files=tuple(records),
        config_chain=config_chain,
    )


def fingerprint_protocol_implementation(
    config_path: Path,
    *,
    workspace_root: Path,
    implementation_paths: Sequence[str] = DEFAULT_IMPLEMENTATION_PATHS,
) -> ProtocolImplementationFingerprint:
    """Hash ordered U2U source bytes and the complete config chain."""
    result = _fingerprint_implementation(
        config_path,
        workspace_root=workspace_root,
        implementation_paths=implementation_paths,
        domain=b"uav-isac-u2u-protocol-fingerprint-v1",
        result_type=ProtocolImplementationFingerprint,
    )
    assert isinstance(result, ProtocolImplementationFingerprint)
    return result


def fingerprint_controller_implementation(
    config_path: Path,
    *,
    workspace_root: Path,
    implementation_paths: Sequence[str] | None = None,
) -> ControllerImplementationFingerprint:
    """Hash the complete controller Python tree and inherited configuration.

    The default manifest is intentionally conservative: every Python source
    below ``uav_isac`` is bound, even if a particular run does not enter that
    module.  Adding or changing controller code therefore invalidates an old
    hardware/runtime energy epoch instead of silently reusing it.
    """
    root = Path(workspace_root).resolve()
    if implementation_paths is None:
        package_root = root / "uav_isac"
        if not package_root.is_dir():
            raise ValueError("workspace has no uav_isac controller package")
        discovered = {
            path.relative_to(root).as_posix()
            for path in package_root.rglob("*.py")
            if path.is_file()
        }
        discovered.update((
            "tools/audit_dual_guided_structure_trace.py",
            "tools/calibrate_compute_energy_epoch.py",
            "tools/calibrate_runtime_latency_epoch.py",
        ))
        implementation_paths = tuple(sorted(discovered))
    result = _fingerprint_implementation(
        config_path,
        workspace_root=root,
        implementation_paths=implementation_paths,
        domain=b"uav-isac-complete-controller-fingerprint-v1",
        result_type=ControllerImplementationFingerprint,
    )
    assert isinstance(result, ControllerImplementationFingerprint)
    return result
