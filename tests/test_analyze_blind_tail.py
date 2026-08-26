"""Tests for tools/analyze_blind_tail.py (2026-08-16, D1.5 left-tail analysis)."""

import csv
import json

import numpy as np

from tools.analyze_blind_tail import (
    _pearson_correlation,
    _wilson_lower,
    analyze,
)


def _write_synthetic_csv(path, seeds, steady, weak3, worst, realized=None):
    rows = {
        "eval_episode_seeds": seeds.tolist(),
        "eval_episode_steady_P_D": steady.tolist(),
        "eval_episode_weak3_P_D": weak3.tolist(),
        "eval_episode_worst_P_D": worst.tolist(),
    }
    if realized is not None:
        rows["eval_episode_worst_nearest_distance_m"] = realized.tolist()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows))
        w.writeheader()
        w.writerow(rows)


def test_wilson_lower_matches_reference():
    # Classic reference: 73/100 -> LCB ~0.636 (matches D1.5 blind100 report).
    assert abs(_wilson_lower(73, 100) - 0.6357) < 1e-3
    assert _wilson_lower(0, 10) == 0.0


def test_pearson_correlation_has_defined_degenerate_behavior():
    assert _pearson_correlation([1, 2, 3], [3, 2, 1]) == -1.0
    assert _pearson_correlation([1, 1, 1], [1, 2, 3]) is None
    assert _pearson_correlation([1, np.nan, 3], [3, 2, 1]) == -1.0


def test_analyze_reports_qos_and_failure_mode(tmp_path):
    rng = np.random.default_rng(7)
    n = 40
    seeds = rng.integers(1, 10000, size=n)
    steady = np.clip(rng.normal(0.85, 0.25, n), 0, 1)
    weak3 = np.clip(rng.normal(0.82, 0.28, n), 0, 1)
    worst = np.minimum(weak3, np.clip(rng.normal(0.80, 0.3, n), 0, 1))
    p = tmp_path / "synth.csv"
    _write_synthetic_csv(p, seeds, steady, weak3, worst)
    rep = analyze(str(p))
    assert rep["n_seeds"] == n
    assert 0 <= rep["qos_rate"] <= 1
    assert 0 <= rep["wilson_lcb"] <= rep["qos_rate"]
    fm = rep["failure_mode"]
    n_pass = int(((steady >= 0.8) & (weak3 >= 0.7) & (worst >= 0.6)).sum())
    assert fm["scene_level_failures"] + fm["target_level_failures"] == n - n_pass
    assert "realized_distance" not in rep  # absent column -> no bucket table


def test_analyze_uses_bank_metadata_and_realized_distance(tmp_path):
    rng = np.random.default_rng(11)
    n = 20
    seeds = np.arange(1, n + 1)
    steady = np.where(seeds % 2, 0.95, 0.30)
    weak3 = np.where(seeds % 2, 0.94, 0.28)
    worst = np.where(seeds % 2, 0.93, 0.25)
    realized = np.where(seeds % 2, 120.0, 420.0)
    p = tmp_path / "synth.csv"
    _write_synthetic_csv(p, seeds, steady, weak3, worst, realized)
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({
        "seed_metadata": {
            str(int(s)): {"worst_nearest_m": float(90 + 10 * s)}
            for s in seeds},
    }), encoding="utf-8")
    rep = analyze(str(p), str(bank))
    assert rep["qos_rate"] == 0.5
    assert rep["corr_realized_worst_pd"] < -0.9  # strong negative link
    assert rep["worst_nearest"][0]["n"] > 0  # bucket table populated
    assert rep["realized_distance"][-1]["n"] > 0
