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
        elif "LCB enforced" in name:
            # LCB-enforced rows are audit-standard checks; a FAIL is a valid,
            # honest disclosure of insufficient statistical power (e.g. D1.5
            # blind 100: LCB 0.636 < 0.70), so it must not crash the batch.
            assert item["status"] in ("PASS", "FAIL")
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


def test_d1_5_blind_registry_point_estimate_passes_lcb_disclosed():
    """D1.5 blind 100: point estimate passes the QoS gate; the LCB-enforced
    row is present and honestly reports FAIL (0.636 < 0.70), matching the
    documented statistical-power limitation (docs/KNOWN_ISSUES.md)."""
    point = [r for r in FORMAL_RESULTS
             if "D1.5 blind (100 seeds)" in r.name and not r.require_lcb]
    lcb = [r for r in FORMAL_RESULTS
           if "D1.5 blind (100 seeds)" in r.name and r.require_lcb]
    assert len(point) == 1 and len(lcb) == 1
    report = assert_formal_gates(results=point + lcb)
    assert report[point[0].name]["status"] == "PASS"
    assert report[point[0].name]["qos_feasible"] >= 0.70
    assert report[lcb[0].name]["status"] == "FAIL"
    assert report[lcb[0].name]["error"]  # non-empty failure detail


def test_d1_9_blind_registry_passes_point_and_lcb():
    """D1.9 bottleneck-lookahead blind 100: both the point-estimate and the
    LCB-enforced rows must PASS (QoS 0.950 / LCB 0.888 >= 0.70), closing the
    D1.5 statistical-power gap (docs/OPTIMIZATION_LOG.md D1.9)."""
    point = [r for r in FORMAL_RESULTS
             if "D1.9 bottleneck-lookahead blind (100 seeds)" in r.name
             and not r.require_lcb]
    lcb = [r for r in FORMAL_RESULTS
           if "D1.9 bottleneck-lookahead blind (100 seeds)" in r.name
           and r.require_lcb]
    assert len(point) == 1 and len(lcb) == 1
    report = assert_formal_gates(results=point + lcb)
    assert report[point[0].name]["status"] == "PASS"
    assert report[point[0].name]["qos_feasible"] >= 0.90
    assert report[lcb[0].name]["status"] == "PASS"
    assert report[lcb[0].name]["qos_wilson_lcb"] >= 0.70
