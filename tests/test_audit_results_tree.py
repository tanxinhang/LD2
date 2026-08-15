"""Tests for tools/audit_results_tree.py (2026-08-16)."""

import json

from tools.audit_results_tree import render, scan_tree


def _make_tree(root):
    # Complete artifact dir.
    d = root / "gate_a"
    d.mkdir(parents=True)
    (d / "paired_eval.csv").write_text("x\n", encoding="utf-8")
    (d / "run_manifest.json").write_text("{}", encoding="utf-8")
    (d / "summary.json").write_text(
        json.dumps({"schema_version": 1, "mean": 1.0}), encoding="utf-8")
    # Missing-all dir.
    (root / "old_smoke").mkdir()
    # Scratch dir with only a summary.
    s = root / "_scratch_run"
    s.mkdir()
    (s / "summary.json").write_text(
        json.dumps({"trace": [], "mean": {}}), encoding="utf-8")
    return root


def test_scan_tree_counts_artifacts_and_flags(tmp_path):
    root = _make_tree(tmp_path / "results")
    report = scan_tree(str(root))
    assert report["total_dirs"] == 3
    assert report["artifact_coverage"]["summary.json"] == 2
    assert report["artifact_coverage"]["paired_eval.csv"] == 1
    assert report["artifact_coverage"]["run_manifest.json"] == 1
    assert report["missing_all"] == ["old_smoke"]
    assert report["scratch_dirs"] == ["_scratch_run"]
    assert report["schema_variants"] == 2  # gate_a vs _scratch_run


def test_render_is_text_table(tmp_path):
    root = _make_tree(tmp_path / "results")
    text = render(scan_tree(str(root)))
    assert "results/" in text
    assert "3 dirs" in text
    assert "scratch" in text
