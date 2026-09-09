"""Registry of the small set of active V2 runtime profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_FILE = Path(__file__).resolve().parent / "data" / "runtime_profiles.yaml"


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    path: str
    purpose: str
    formal: bool
    status: str


def load_runtime_profiles(
    path: Path | str = DEFAULT_PROFILE_FILE,
) -> Tuple[RuntimeProfile, ...]:
    with Path(path).resolve().open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported runtime profile registry schema")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("runtime profile registry must not be empty")
    profiles = []
    for name, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            raise ValueError(f"profile {name!r} must be a mapping")
        profile_path = raw.get("path")
        purpose = raw.get("purpose")
        formal = raw.get("formal")
        status = raw.get("status")
        if not isinstance(profile_path, str) or not profile_path.startswith("config/"):
            raise ValueError(f"profile {name!r} must point below config/")
        resolved = (REPOSITORY_ROOT / profile_path).resolve()
        config_root = (REPOSITORY_ROOT / "config").resolve()
        if not resolved.is_relative_to(config_root) or not resolved.is_file():
            raise ValueError(f"profile {name!r} path is unavailable")
        if not isinstance(purpose, str) or not purpose:
            raise ValueError(f"profile {name!r} must declare purpose")
        if not isinstance(formal, bool) or status not in {"active", "paused"}:
            raise ValueError(f"profile {name!r} has invalid formal/status fields")
        profiles.append(RuntimeProfile(
            name=str(name),
            path=profile_path,
            purpose=purpose,
            formal=formal,
            status=status,
        ))
    return tuple(profiles)


def get_runtime_profile(name: str) -> RuntimeProfile:
    matches = [item for item in load_runtime_profiles() if item.name == name]
    if not matches:
        raise KeyError(f"unregistered runtime profile: {name!r}")
    profile = matches[0]
    if profile.status != "active":
        raise RuntimeError(f"runtime profile is not active: {name!r}")
    return profile
