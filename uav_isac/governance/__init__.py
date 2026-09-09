"""Project-level governance contracts used during architecture migration."""

from .project_phase import (
    OperationBlockedError,
    ProjectPhase,
    assert_operation_allowed,
    assert_managed_executor,
    load_project_phase,
)
from .research_programs import ResearchProgram, load_research_programs
from .research_readiness import ResearchReadiness, audit_research_readiness
from .architecture import audit_architecture, load_architecture_rules
from .fingerprint import canonical_sha256, semantic_differences, semantic_fingerprint
from .baseline import load_characterization_baselines
from .profiles import get_runtime_profile, load_runtime_profiles

__all__ = [
    "OperationBlockedError",
    "ProjectPhase",
    "assert_operation_allowed",
    "assert_managed_executor",
    "load_project_phase",
    "audit_architecture",
    "load_architecture_rules",
    "semantic_fingerprint",
    "semantic_differences",
    "canonical_sha256",
    "load_characterization_baselines",
    "get_runtime_profile",
    "load_runtime_profiles",
    "ResearchProgram",
    "load_research_programs",
    "ResearchReadiness",
    "audit_research_readiness",
]
