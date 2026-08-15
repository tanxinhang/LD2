"""Tests for tools/assert_formal_gates.py (2026-08-16)."""

import pytest

from tools.assert_formal_gates import (
    FORMAL_RESULTS,
    FormalResult,
    assert_formal_gates,
    render_table,
)


def test_registry_covers_documented_formal_results():
    names = [r.name for r in FORMAL_RESULTS]
    assert any("4/4 frozen deployment" in n for n in names)
    assert any("8/8 analytical stack" in n for n in names)
    # Quarantined 6/6 rows are present but flagged, never asserted.
    quarantined = [r for r in FORMAL_RESULTS if r.quarantined]
    assert len(quarantined) == 2
    assert all("6/6" in r.name for r in quarantined)


def test_assert_formal_gates_marks_quarantined_without_asserting():
    report = assert_formal_gates()
    for name, item in report.items():
        if "6/6" in name:
            assert item["status"] == "QUARANTINED"
        else:
            assert item["status"] == "PASS", name
            assert item["steady"] > 0
            assert item["worst"] > 0


def test_custom_result_failure_is_reported(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    entry = FormalResult(name="synthetic", csv=str(bad))
    report = assert_formal_gates(results=[entry])
    assert report["synthetic"]["status"] == "FAIL"
    assert "lacks column" in report["synthetic"]["error"]


def test_render_table_lists_every_result():
    report = assert_formal_gates()
    table = render_table(report)
    for name in report:
        assert name in table
    assert "QUARANTINED" in table
    assert "PASS" in table
