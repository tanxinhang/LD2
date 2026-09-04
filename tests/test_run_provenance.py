from pathlib import Path
import zipfile

from config.params import MasterConfig
from uav_isac.utils.provenance import (
    build_run_provenance,
    write_source_snapshot,
)
import uav_isac.utils.provenance as provenance_module


def test_source_snapshot_is_content_deterministic(tmp_path):
    root = tmp_path / "workspace"
    (root / "uav_isac").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "uav_isac" / "model.py").write_text("x = 1\n", encoding="utf-8")
    (root / "config" / "run.yaml").write_text("scenario: {}\n", encoding="utf-8")
    first = write_source_snapshot(root, tmp_path / "first.zip")
    second = write_source_snapshot(root, tmp_path / "second.zip")
    assert first["sha256"] == second["sha256"]
    with zipfile.ZipFile(tmp_path / "first.zip") as archive:
        assert archive.namelist() == ["config/run.yaml", "uav_isac/model.py"]


def test_source_snapshot_includes_dependency_constraints(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "constraints-ci.txt").write_text(
        "numpy==2.5.0\n", encoding="utf-8")
    result = write_source_snapshot(root, tmp_path / "snapshot.zip")
    assert result["file_count"] == 1
    with zipfile.ZipFile(tmp_path / "snapshot.zip") as archive:
        assert archive.namelist() == ["constraints-ci.txt"]


def test_provenance_binds_resolved_config_and_missing_packages(tmp_path):
    snapshot = {"path": "source.zip", "sha256": "a" * 64, "file_count": 2}
    result = build_run_provenance(
        root=Path(__file__).resolve().parents[1],
        config=MasterConfig(),
        source_snapshot=snapshot,
    )
    assert len(result["resolved_config_sha256"]) == 64
    assert result["resolved_config"]["scenario"]["K"] == 4
    assert result["source_snapshot"] == snapshot
    assert "scikit-learn" in result["runtime"]["packages"]


def test_missing_git_identity_is_never_reported_as_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance_module, "_git", lambda *args: None)
    result = build_run_provenance(
        root=tmp_path,
        config={"alpha": 1},
        source_snapshot={"path": "source.zip", "sha256": "0" * 64},
    )
    assert result["git_available"] is False
    assert result["git_dirty"] is True
    assert result["git_commit"] is None
