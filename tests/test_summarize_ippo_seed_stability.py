from __future__ import annotations

import numpy as np

from tools.summarize_ippo_seed_stability import aggregate


def test_aggregate_reports_seed_level_sample_standard_deviation() -> None:
    rows = []
    for value in (0.7, 0.8, 0.9):
        row = {f"ippo_{name}": value for name in (
            "steady", "weak3", "worst", "worst_lcb", "cvar20",
            "qos_feasible", "qos_wilson_lcb", "bits_per_frame")}
        row["ippo_worst_delta_vs_frozen"] = value - 0.8
        rows.append(row)
    result = aggregate(rows, "ippo")
    assert np.isclose(result["worst"]["mean"], 0.8)
    assert np.isclose(result["worst"]["sample_std"], 0.1)
    assert np.isclose(result["worst_delta_vs_frozen"]["mean"], 0.0)
