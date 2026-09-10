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


def test_source_root_reports_unmapped_module(tmp_path: Path):
    source_root = tmp_path / "uav_isac"
    source_root.mkdir()
    orphan = source_root / "orphan.py"
    orphan.write_text("value = 1\n", encoding="utf-8")
    rules = ArchitectureRules(
        repository_root=tmp_path,
        source_root=source_root,
        max_module_lines=500,
        layers=(LayerRule(
            name="owned_elsewhere",
            paths=(source_root / "owned",),
            forbidden_imports=(),
        ),),
    )

    violations = audit_architecture(rules)

    assert [item.rule for item in violations] == ["unmapped_module"]
    assert violations[0].path == orphan


def test_duplicate_layer_ownership_is_reported(tmp_path: Path):
    source_root = tmp_path / "uav_isac"
    shared = source_root / "shared"
    shared.mkdir(parents=True)
    module = shared / "module.py"
    module.write_text("value = 1\n", encoding="utf-8")
    rules = ArchitectureRules(
        repository_root=tmp_path,
        source_root=source_root,
        max_module_lines=500,
        layers=(
            LayerRule("first", (shared,), ()),
            LayerRule("second", (shared,), ()),
        ),
    )

    violations = audit_architecture(rules)

    assert [item.rule for item in violations] == [
        "duplicate_layer_ownership"]
    assert "first, second" in violations[0].detail


def test_relative_import_is_resolved_before_dependency_check(tmp_path: Path):
    source_root = tmp_path / "uav_isac"
    feature = source_root / "feature"
    environment = source_root / "environment"
    feature.mkdir(parents=True)
    environment.mkdir()
    bad = feature / "bad.py"
    bad.write_text(
        "from ..environment import state\n",
        encoding="utf-8",
    )
    (environment / "state.py").write_text("value = 1\n", encoding="utf-8")
    rules = ArchitectureRules(
        repository_root=tmp_path,
        source_root=source_root,
        max_module_lines=500,
        layers=(
            LayerRule(
                "feature",
                (feature,),
                ("uav_isac.environment",),
            ),
            LayerRule("environment", (environment,), ()),
        ),
    )

    violations = audit_architecture(rules)

    forbidden = [item for item in violations if item.rule == "forbidden_import"]
    assert len(forbidden) == 1
    assert forbidden[0].path == bad
    assert "uav_isac.environment" in forbidden[0].detail


def test_layer_dependency_cycle_is_reported(tmp_path: Path):
    source_root = tmp_path / "uav_isac"
    first = source_root / "first"
    second = source_root / "second"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "a.py").write_text(
        "from uav_isac.second import b\n", encoding="utf-8")
    (second / "b.py").write_text(
        "from uav_isac.first import a\n", encoding="utf-8")
    rules = ArchitectureRules(
        repository_root=tmp_path,
        source_root=source_root,
        max_module_lines=500,
        layers=(
            LayerRule("first", (first,), ()),
            LayerRule("second", (second,), ()),
        ),
    )

    violations = audit_architecture(rules)

    cycles = [item for item in violations if item.rule == "dependency_cycle"]
    assert len(cycles) == 1
    assert "first -> second -> first" in cycles[0].detail
