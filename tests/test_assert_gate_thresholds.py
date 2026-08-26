"""Tests for tools/assert_gate_thresholds.py (2026-08-16).

Turns "Gate passed" into script-enforced checks: aggregate arrays must clear
the Medium floors (steady>=0.80, weak3>=0.70, mean worst>=0.60, QoS>=0.70),
with the Wilson LCB reported and optionally enforced.  The CSV format and
Wilson convention match the project's summarize scripts (single aggregate
row, eval_episode_*_P_D literal lists, z=1.96).
"""

import ast
import csv
import json

import pytest

from tools.assert_gate_thresholds import (
    assert_gate_from_csv,
    assert_medium_gate,
    medium_gate_checks,
    wilson_lower,
)


def _write_csv(path, steady, weak3, worst):
    cols = ["eval_episode_seeds", "eval_episode_steady_P_D",
            "eval_episode_weak3_P_D", "eval_episode_worst_P_D"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerow({
            "eval_episode_seeds": str(list(range(len(steady)))),
            "eval_episode_steady_P_D": repr(steady),
            "eval_episode_weak3_P_D": repr(weak3),
            "eval_episode_worst_P_D": repr(worst),
        })


def test_wilson_lower_matches_project_convention():
    # 72/100 successes, z=1.96 -> LCB ~0.6251 (same formula as
    # tools/summarize_final_paper_suite.py::wilson_lower).
    lcb = wilson_lower(72, 100)
    assert 0.62 < lcb < 0.63
    # Consistency with the existing implementation.
    from tools.summarize_final_paper_suite import wilson_lower as project_lower
    assert lcb == pytest.approx(project_lower(72, 100), abs=1e-12)


def test_medium_gate_passes_when_all_floors_clear():
    checks = medium_gate_checks(
        steady=0.913, weak3=0.885, worst=0.739,
        qos_feasible=0.72, qos_wilson_lcb=None)
    assert all(c["passed"] for c in checks)
    assert_medium_gate(0.913, 0.885, 0.739, 0.72)  # no raise


def test_medium_gate_fails_on_each_missing_floor():
    with pytest.raises(AssertionError, match="steady"):
        assert_medium_gate(0.79, 0.88, 0.74, 0.80)
    with pytest.raises(AssertionError, match="weak3"):
        assert_medium_gate(0.90, 0.65, 0.74, 0.80)
    with pytest.raises(AssertionError, match="mean_worst"):
        assert_medium_gate(0.90, 0.88, 0.55, 0.80)
    with pytest.raises(AssertionError, match="qos_feasible"):
        assert_medium_gate(0.90, 0.88, 0.74, 0.60)


def test_wilson_lcb_enforcement_is_optional():
    # The LCB is enforced only when the caller passes it: QoS 0.72 (72/100)
    # clears the point-estimate floor, but its z=1.96 LCB ~0.63 does not
    # clear 0.70 -- analogous to the current 4/4 formal result.
    assert_medium_gate(0.913, 0.885, 0.739, 0.72)  # no LCB -> passes
    with pytest.raises(AssertionError, match="qos_wilson_lcb"):
        assert_medium_gate(0.913, 0.885, 0.739, 0.72, qos_wilson_lcb=0.63)


def test_assert_gate_from_csv_recomputes_aggregates(tmp_path):
    # All-feasible CSV: aggregates must match and the gate must pass.
    path = tmp_path / "paired_eval.csv"
    steady = [0.90] * 100
    weak3 = [0.80] * 100
    worst = [0.65] * 100
    _write_csv(path, steady, weak3, worst)
    aggregates = assert_gate_from_csv(str(path))
    assert aggregates["episodes"] == 100
    assert aggregates["steady"] == pytest.approx(0.90)
    assert aggregates["weak3"] == pytest.approx(0.80)
    assert aggregates["worst"] == pytest.approx(0.65)
    assert aggregates["qos_feasible"] == pytest.approx(1.0)
    assert aggregates["qos_wilson_lcb"] > 0.90

    # Mixed CSV: mean worst below 0.60 -> gate must fail.
    bad = tmp_path / "bad_mix.csv"
    steady_b, weak3_b, worst_b = [], [], []
    for i in range(100):
        if i < 72:
            steady_b.append(0.90); weak3_b.append(0.80); worst_b.append(0.65)
        else:
            steady_b.append(0.70); weak3_b.append(0.60); worst_b.append(0.40)
    _write_csv(bad, steady_b, weak3_b, worst_b)
    with pytest.raises(AssertionError, match="mean_worst"):
        assert_gate_from_csv(str(bad))


def test_assert_gate_from_csv_on_real_4x4_formal_result():
    # The frozen 4/4 formal result must pass the point-estimate Medium gate.
    path = ("results/architecture_v2_structure_student_u2u_resolve_bw50k_"
            "adaptive_b4b8_failclosed_gate100/paired_eval.csv")
    aggregates = assert_gate_from_csv(path)
    assert aggregates["episodes"] == 100
    assert aggregates["steady"] == pytest.approx(0.9132, abs=1e-3)
    assert aggregates["weak3"] == pytest.approx(0.8848, abs=1e-3)
    assert aggregates["worst"] == pytest.approx(0.7393, abs=1e-3)
    assert aggregates["qos_feasible"] == pytest.approx(0.72, abs=1e-9)
    assert aggregates["qos_wilson_lcb"] == pytest.approx(
        wilson_lower(72, 100), abs=1e-9)


def test_cli_rejects_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("a,b\n1,2\n")
    from tools.assert_gate_thresholds import main
    assert main([str(path)]) == 2


def test_gate_rejects_misaligned_episode_arrays(tmp_path):
    path = tmp_path / "misaligned.csv"
    _write_csv(path, [0.9, 0.9], [0.8], [0.7, 0.7])
    with pytest.raises(ValueError, match="misaligned episode arrays"):
        assert_gate_from_csv(str(path))


def test_gate_rejects_duplicate_episode_seeds(tmp_path):
    path = tmp_path / "duplicates.csv"
    _write_csv(path, [0.9, 0.9], [0.8, 0.8], [0.7, 0.7])
    text = path.read_text(encoding="utf-8").replace("[0, 1]", "[7, 7]")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate evaluation seeds"):
        assert_gate_from_csv(str(path))


def test_gate_rejects_nonfinite_or_out_of_range_metrics(tmp_path):
    path = tmp_path / "out_of_range.csv"
    _write_csv(path, [0.9], [0.8], [1.2])
    with pytest.raises(ValueError, match="out-of-range worst"):
        assert_gate_from_csv(str(path))
