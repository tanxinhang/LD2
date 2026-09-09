import json

import pytest

from uav_isac.adapters import FileSystemArtifactStore
from uav_isac.domain import RunCompletion, RunManifest


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


def test_store_rejects_manifest_that_claims_completion_before_outputs(tmp_path):
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    premature = RunManifest(
        run_id="run-premature",
        run_type="characterization",
        state="completed",
        created_at="2026-09-09T00:00:00Z",
        command=("python", "-m", "uav_isac.interfaces.cli"),
        config_sha256="a" * 64,
        code_identity="git:test-dirty",
        seeds=(451,),
    )

    with pytest.raises(ValueError, match="must start in created state"):
        store.create_run(premature)


@pytest.mark.parametrize("path", ["../escape.json", "/absolute.json"])
def test_store_rejects_paths_outside_run(tmp_path, path):
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    store.create_run(_manifest())

    with pytest.raises(ValueError):
        store.write_json("run-001", path, {"unsafe": True})


def test_completion_binds_existing_artifacts_and_never_overwrites(tmp_path):
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    store.create_run(_manifest())
    artifact = store.write_json("run-001", "derived/metrics.json", {"value": 1})
    completion = RunCompletion(
        run_id="run-001",
        state="completed",
        completed_at="2026-09-09T00:01:00Z",
        artifacts={artifact.relative_path: artifact.sha256},
    )

    record = store.complete_run(completion)

    assert record.relative_path == "completion.json"
    with pytest.raises(FileExistsError):
        store.complete_run(completion)


def test_completion_rejects_missing_or_mismatched_artifact(tmp_path):
    store = FileSystemArtifactStore(tmp_path / "artifacts")
    store.create_run(_manifest())

    missing = RunCompletion(
        run_id="run-001",
        state="completed",
        completed_at="2026-09-09T00:01:00Z",
        artifacts={"derived/missing.json": "a" * 64},
    )
    with pytest.raises(FileNotFoundError):
        store.complete_run(missing)

    artifact = store.write_json("run-001", "derived/metrics.json", {"value": 1})
    mismatched = RunCompletion(
        run_id="run-001",
        state="completed",
        completed_at="2026-09-09T00:01:00Z",
        artifacts={artifact.relative_path: "b" * 64},
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        store.complete_run(mismatched)
