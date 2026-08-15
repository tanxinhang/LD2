import json
from pathlib import Path

import pytest

from tools.audit_n5_dependency_commit import verify_trace_provenance


def _manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    paths = {
        name: tmp_path / name
        for name in ("trace.npz", "config.yaml", "student.pt", "ranker.pt", "factor.pt")
    }
    payload = {
        "git_commit": "deadbeef",
        "config": str(paths["config.yaml"]),
        "structure_teacher_trace_output": str(paths["trace.npz"]),
        "structure_student_checkpoint": str(paths["student.pt"]),
        "dynamic_local_search": {
            "mode": "hybrid",
            "ranker_checkpoint": str(paths["ranker.pt"]),
            "cold_initializer": "factor_graph",
            "factor_graph_checkpoint": str(paths["factor.pt"]),
            "neighbor_topk": 4,
            "target_topk": 4,
            "coverage_fraction": 0.3,
            "cold_rounds": 8,
            "warm_rounds": 8,
            "warm_top_m": 5,
            "rebootstrap_mode": "off",
        },
    }
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, paths


def _verify(manifest: Path, paths: dict[str, Path], **overrides):
    values = {
        "trace_path": paths["trace.npz"],
        "config_path": paths["config.yaml"],
        "student_checkpoint": paths["student.pt"],
        "ranker_checkpoint": paths["ranker.pt"],
        "factor_graph_checkpoint": paths["factor.pt"],
        "neighbor_topk": 4,
        "target_topk": 4,
        "coverage_fraction": 0.3,
        "cold_rounds": 8,
        "warm_rounds": 8,
        "warm_top_m": 5,
    }
    values.update(overrides)
    return verify_trace_provenance(manifest, **values)


def test_exact_trace_pipeline_provenance_is_accepted(tmp_path):
    manifest, paths = _manifest(tmp_path)
    result = _verify(manifest, paths)
    assert result["verified"] is True
    assert result["git_commit"] == "deadbeef"
    assert "ranker_checkpoint" in result["bound_fields"]


def test_wrong_checkpoint_is_rejected_before_residual_interpretation(tmp_path):
    manifest, paths = _manifest(tmp_path)
    with pytest.raises(RuntimeError, match="trace provenance mismatch") as error:
        _verify(manifest, paths, ranker_checkpoint=tmp_path / "wrong-ranker.pt")
    assert "ranker_checkpoint" in str(error.value)
    assert "not interpretable" in str(error.value)


def test_search_hyperparameter_mismatch_is_rejected(tmp_path):
    manifest, paths = _manifest(tmp_path)
    with pytest.raises(RuntimeError, match="neighbor_topk"):
        _verify(manifest, paths, neighbor_topk=5)


def test_missing_manifest_fails_closed(tmp_path):
    missing = tmp_path / "missing.json"
    paths = {
        name: tmp_path / name
        for name in ("trace.npz", "config.yaml", "student.pt", "ranker.pt", "factor.pt")
    }
    with pytest.raises(RuntimeError, match="manifest is missing"):
        _verify(missing, paths)
