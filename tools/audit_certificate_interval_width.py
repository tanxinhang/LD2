"""Audit selected-edge lower/nominal/upper widths in a teacher NPZ."""

from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    data = np.load(args.dataset)
    resident = {
        key: data[key]
        for key in (
            "frame", "edge_index", "edge_mask", "lower_gain",
            "nominal_gain", "upper_gain", "power", "visible",
        )
    }
    values = []
    for frame in range(len(resident["frame"])):
        edges = resident["edge_index"][frame][resident["edge_mask"][frame]]
        for tx, _rx, target in edges:
            values.append((
                resident["lower_gain"][frame, tx, tx, target],
                resident["nominal_gain"][frame, tx, tx, target],
                resident["upper_gain"][frame, tx, tx, target],
                resident["power"][frame, tx, tx, target],
                resident["visible"][frame, tx, tx, target],
            ))
    array = np.asarray(values, dtype=np.float64)
    lower, nominal, upper, power, visible = array.T
    positive = nominal > 0.0
    powered = power > 0.0
    lower_ratio = np.divide(
        lower, nominal, out=np.zeros_like(lower), where=positive)
    upper_ratio = np.divide(
        upper, nominal, out=np.full_like(upper, np.inf), where=positive)
    relative_width = np.divide(
        upper - lower,
        np.maximum(nominal, np.finfo(np.float64).tiny),
    )
    active_positive = powered & positive
    output = {
        "selected_samples": len(array),
        "visible_fraction": float(np.mean(visible)),
        "positive_nominal_fraction": float(np.mean(positive)),
        "positive_lower_fraction": float(np.mean(lower > 0.0)),
        "powered_fraction": float(np.mean(powered)),
        "lower_over_nominal": {
            "p05": float(np.percentile(lower_ratio[positive], 5)),
            "p50": float(np.median(lower_ratio[positive])),
            "mean": float(np.mean(lower_ratio[positive])),
        },
        "upper_over_nominal": {
            "p50": float(np.median(upper_ratio[positive])),
            "p95": float(np.percentile(upper_ratio[positive], 95)),
            "mean": float(np.mean(upper_ratio[positive])),
        },
        "relative_width": {
            "p50": float(np.median(relative_width[positive])),
            "p95": float(np.percentile(relative_width[positive], 95)),
        },
        "powered": {
            "positive_lower_fraction": float(np.mean(lower[powered] > 0.0)),
            "lower_over_nominal_p50": float(np.median(
                lower_ratio[active_positive])),
            "upper_over_nominal_p50": float(np.median(
                upper_ratio[active_positive])),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
