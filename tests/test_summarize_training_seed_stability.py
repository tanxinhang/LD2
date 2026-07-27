from __future__ import annotations

import numpy as np

from tools.summarize_training_seed_stability import aggregate


def test_aggregate_uses_sample_standard_deviation() -> None:
    rows = []
    for value in (0.7, 0.8, 0.9):
        rows.append({name: value for name in (
            "steady", "weak3", "worst", "worst_lcb", "cvar20",
            "qos_feasible", "qos_wilson_lcb", "bits_per_frame")})
    result = aggregate(rows)
    assert np.isclose(result["worst"]["mean"], 0.8)
    assert np.isclose(result["worst"]["sample_std"], 0.1)
    assert result["worst"]["minimum"] == 0.7
    assert result["worst"]["maximum"] == 0.9
