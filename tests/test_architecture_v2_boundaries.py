from pathlib import Path

from uav_isac.governance.architecture import (
    ArchitectureRules,
    LayerRule,
    audit_architecture,
)


def test_repository_v2_modules_respect_dependency_rules():
    assert audit_architecture() == ()


def test_forbidden_dependency_is_reported(tmp_path: Path):
    domain = tmp_path / "domain"
    domain.mkdir()
    (domain / "bad.py").write_text(
        "from uav_isac.environment.env_core import EnvironmentCore\n",
        encoding="utf-8",
    )
    rules = ArchitectureRules(
        repository_root=tmp_path,
        max_module_lines=500,
        layers=(LayerRule(
            name="domain",
            paths=(domain,),
            forbidden_imports=("uav_isac.environment",),
        ),),
    )

    violations = audit_architecture(rules)

    assert len(violations) == 1
    assert violations[0].rule == "forbidden_import"
    assert "env_core" in violations[0].detail


def test_registered_legacy_dependency_is_allowed_but_only_at_exact_path(tmp_path: Path):
    coordination = tmp_path / "coordination"
    coordination.mkdir()
    allowed = coordination / "allowed.py"
    allowed.write_text(
        "from uav_isac.environment.communication import Model\n",
        encoding="utf-8",
    )
    rules = ArchitectureRules(
        repository_root=tmp_path,
        max_module_lines=500,
        layers=(LayerRule(
            name="legacy",
            paths=(coordination,),
            forbidden_imports=("uav_isac.environment",),
            allowed_imports=(("coordination/allowed.py", "uav_isac.environment.communication"),),
            enforce_module_size=False,
        ),),
    )
    assert audit_architecture(rules) == ()

    (coordination / "new.py").write_text(
        "from uav_isac.environment.communication import Model\n",
        encoding="utf-8",
    )
    violations = audit_architecture(rules)
    assert len(violations) == 1
    assert violations[0].path.name == "new.py"


def test_stale_legacy_allowlist_entry_is_reported(tmp_path: Path):
    coordination = tmp_path / "coordination"
    coordination.mkdir()
    (coordination / "clean.py").write_text("value = 1\n", encoding="utf-8")
    rules = ArchitectureRules(
        repository_root=tmp_path,
        max_module_lines=500,
        layers=(LayerRule(
            name="legacy",
            paths=(coordination,),
            forbidden_imports=("uav_isac.environment",),
            allowed_imports=(("coordination/clean.py", "uav_isac.environment"),),
            enforce_module_size=False,
        ),),
    )

    violations = audit_architecture(rules)

    assert len(violations) == 1
    assert violations[0].rule == "stale_allowlist"
