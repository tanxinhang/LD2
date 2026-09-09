"""Measure a DD-aware physical upper on an exported K16 teacher trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.coordination.composable_certificate import (  # noqa: E402
    optimal_simplex_dual_upper,
)
from uav_isac.coordination.hyperedge import (  # noqa: E402
    reconstruct_bistatic_coefficient_dd_upper_from_public_state,
    reconstruct_bistatic_coefficient_upper_from_public_state,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    source = np.load(args.dataset)
    keys = (
        "frame", "endpoint_position", "endpoint_velocity", "target_position",
        "target_velocity", "target_position_uncertainty",
        "target_velocity_uncertainty", "visible", "upper_gain", "lower_gain",
        "power", "edge_index", "edge_mask",
    )
    data = {key: source[key] for key in keys}
    frames, viewers, nodes, targets = data["upper_gain"].shape
    position_radius = 0.5 * np.hypot(1130.0 / 255.0, 1130.0 / 255.0)
    velocity_radius = np.sqrt(2.0) * 25.0 / 255.0
    budget = np.full(nodes, 0.0251, dtype=np.float64)
    old_uppers = []
    tight_uppers = []
    optimal_uppers = []
    lowers = []
    selected_ratios = []
    for frame in range(frames):
        tight_views = np.zeros_like(data["upper_gain"][frame])
        for viewer in range(viewers):
            common = dict(
                uav_height_m=20.0,
                fc_hz=28.0e9,
                rcs_m2=1.0,
                coefficient_scale=1.0,
                position_uncertainty_m=position_radius,
                target_position_uncertainty_m=(
                    data["target_position_uncertainty"][frame, viewer]),
            )
            loose = reconstruct_bistatic_coefficient_upper_from_public_state(
                data["endpoint_position"][frame, viewer],
                data["target_position"][frame, viewer],
                data["visible"][frame, viewer],
                **common,
            )
            tight = reconstruct_bistatic_coefficient_dd_upper_from_public_state(
                data["endpoint_position"][frame, viewer],
                data["endpoint_velocity"][frame, viewer],
                data["target_position"][frame, viewer],
                data["target_velocity"][frame, viewer],
                data["visible"][frame, viewer],
                delta_f_hz=15625.0,
                symbol_period_s=6.4e-5,
                delay_bins=64,
                doppler_bins=16,
                dd_gate_min=0.5,
                velocity_uncertainty_mps=velocity_radius,
                target_velocity_uncertainty_mps=(
                    data["target_velocity_uncertainty"][frame, viewer]),
                dd_gain_mode="continuous",
                **common,
            )
            factor = np.divide(
                tight, loose, out=np.zeros_like(tight), where=loose > 0.0)
            factor_gain = np.zeros((nodes, targets), dtype=np.float64)
            for tx, rx, target in data["edge_index"][frame][
                data["edge_mask"][frame]
            ]:
                factor_gain[tx, target] = factor[tx, rx, target]
            tight_views[viewer] = (
                data["upper_gain"][frame, viewer] * factor_gain)
        old_rows = np.stack([
            data["upper_gain"][frame, node, node] for node in range(nodes)])
        tight_rows = np.stack([
            tight_views[node, node] for node in range(nodes)])
        lower_rows = np.stack([
            data["lower_gain"][frame, node, node] for node in range(nodes)])
        power_rows = np.stack([
            data["power"][frame, node, node] for node in range(nodes)])
        old_uppers.append(float(np.min(np.sum(
            budget[:, None] * old_rows, axis=0))))
        tight_uppers.append(float(np.min(np.sum(
            budget[:, None] * tight_rows, axis=0))))
        optimal_uppers.append(optimal_simplex_dual_upper(
            tight_rows, budget)[0])
        lowers.append(float(np.min(np.sum(lower_rows * power_rows, axis=0))))
        for tx, _rx, target in data["edge_index"][frame][
            data["edge_mask"][frame]
        ]:
            old = data["upper_gain"][frame, tx, tx, target]
            if old > 0.0:
                selected_ratios.append(tight_views[tx, tx, target] / old)
    old = np.asarray(old_uppers)
    tight = np.asarray(tight_uppers)
    optimal = np.asarray(optimal_uppers)
    lower = np.asarray(lowers)
    valid = (old > 0.0) & (tight > 0.0) & (optimal > 0.0)
    print(json.dumps({
        "frames": frames,
        "selected_upper_tight_over_old_p50": float(np.median(selected_ratios)),
        "selected_upper_tight_over_old_p95": float(np.percentile(
            selected_ratios, 95)),
        "onehot_old_upper_mean": float(np.mean(old)),
        "onehot_dd_tight_upper_mean": float(np.mean(tight)),
        "optimal_dd_tight_upper_mean": float(np.mean(optimal)),
        "dd_tight_over_old_mean": float(np.mean(tight[valid] / old[valid])),
        "dd_tight_onehot_ratio_mean": float(np.mean(np.clip(
            lower[valid] / tight[valid], 0.0, 1.0))),
        "dd_tight_optimal_ratio_mean": float(np.mean(np.clip(
            lower[valid] / optimal[valid], 0.0, 1.0))),
    }, indent=2))


if __name__ == "__main__":
    main()
