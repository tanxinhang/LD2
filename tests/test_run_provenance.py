from pathlib import Path
import zipfile

from config.params import MasterConfig
from uav_isac.utils.provenance import (
    build_run_provenance,
    write_source_snapshot,
)


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
