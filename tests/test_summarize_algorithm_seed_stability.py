from __future__ import annotations

import numpy as np

from tools.summarize_algorithm_seed_stability import (
    aggregate_delta,
    aggregate_method,
)


def test_method_and_paired_delta_aggregates_use_seed_sample_std() -> None:
    rows = []
    for value in (0.7, 0.8, 0.9):
        row = {f"mappo_{name}": value for name in (
            "steady", "weak3", "worst", "worst_lcb", "cvar20",
            "qos_feasible", "qos_wilson_lcb", "bits_per_frame")}
        row["mappo_minus_frozen_worst_delta"] = value - 0.8
        rows.append(row)
    method = aggregate_method(rows, "mappo")
    delta = aggregate_delta(rows, "mappo_minus_frozen")
    assert np.isclose(method["worst"]["mean"], 0.8)
    assert np.isclose(method["worst"]["sample_std"], 0.1)
    assert np.isclose(delta["mean"], 0.0)
    assert np.isclose(delta["sample_std"], 0.1)
