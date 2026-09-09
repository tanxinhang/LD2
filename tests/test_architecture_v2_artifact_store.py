import json

import pytest

from uav_isac.adapters import FileSystemArtifactStore
from uav_isac.domain import RunManifest


def _manifest(run_id="run-001"):
    return RunManifest(
        run_id=run_id,
        run_type="characterization",
        state="created",
        created_at="2026-09-09T00:00:00Z",
        command=("python", "-m", "uav_isac.interfaces.cli"),
        config_sha256="a" * 64,
        code_identity="git:test-dirty",
        seeds=(451,),
    )


def test_store_requires_manifest_and_never_overwrites(tmp_path):
    store = FileSystemArtifactStore(tmp_path / "artifacts")

    with pytest.raises(FileNotFoundError):
        store.write_json("run-001", "derived/metrics.json", {"value": 1})

    manifest_record = store.create_run(_manifest())
    result_record = store.write_json(
        "run-001", "derived/metrics.json", {"value": 1})

    assert manifest_record.relative_path == "manifest.json"
    assert len(result_record.sha256) == 64
    assert json.loads(
        (tmp_path / "artifacts/runs/run-001/derived/metrics.json").read_text()
    ) == {"value": 1}
    with pytest.raises(FileExistsError):
        store.write_json("run-001", "derived/metrics.json", {"value": 2})
    with pytest.raises(FileExistsError):
        store.create_run(_manifest())


@pytest.mark.parametrize("path", ["../escape.json", "/absolute.json"])
def test_store_rejects_paths_outside_run(tmp_path, path):
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    store.create_run(_manifest())

    with pytest.raises(ValueError):
        store.write_json("run-001", path, {"unsafe": True})

