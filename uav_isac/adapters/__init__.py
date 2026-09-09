"""Infrastructure and legacy adapters for architecture V2."""

from .legacy_environment import (
    LegacyGymEnvironmentAdapter,
    build_environment_from_resolved,
    build_legacy_environment,
)
from .policies import HoldPositionPolicy
from .filesystem_artifacts import FileSystemArtifactStore
from .configuration import (
    ResolvedConfiguration,
    load_registered_configuration,
    load_resolved_configuration,
)
from .repository_identity import RepositoryIdentity, build_repository_identity
from .legacy_process import LegacyPythonProcessRunner, ManagedLegacyPythonProcessRunner

__all__ = [
    "HoldPositionPolicy",
    "LegacyGymEnvironmentAdapter",
    "build_legacy_environment",
    "build_environment_from_resolved",
    "FileSystemArtifactStore",
    "ResolvedConfiguration",
    "load_resolved_configuration",
    "load_registered_configuration",
    "RepositoryIdentity",
    "build_repository_identity",
    "LegacyPythonProcessRunner",
    "ManagedLegacyPythonProcessRunner",
]
