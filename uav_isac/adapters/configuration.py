"""Legacy YAML loading behind a resolved V2 configuration identity."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from uav_isac.governance.fingerprint import canonical_sha256
from uav_isac.governance.profiles import get_runtime_profile


@dataclass(frozen=True)
class ResolvedConfiguration:
    """A validated runtime config plus its fully resolved content hash."""

    value: Any
    source: str
    payload: Mapping[str, Any]
    sha256: str


def load_resolved_configuration(path: str | None = None) -> ResolvedConfiguration:
    """Load inherited YAML and hash the resulting dataclass, including defaults."""

    from config.params import get_default_config, load_config

    if path is None:
        value = get_default_config()
        source = str((Path(__file__).resolve().parents[2] / "config" / "default.yaml"))
    else:
        source = str(Path(path).resolve())
        value = load_config(source)
    payload = dataclasses.asdict(value)
    return ResolvedConfiguration(
        value=value,
        source=source,
        payload=payload,
        sha256=canonical_sha256(payload),
    )


def load_registered_configuration(name: str) -> ResolvedConfiguration:
    """Resolve one active profile; arbitrary YAML is not accepted here."""

    profile = get_runtime_profile(name)
    repository = Path(__file__).resolve().parents[2]
    return load_resolved_configuration(str(repository / profile.path))
