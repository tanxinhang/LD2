#!/usr/bin/env python
"""Geometry-grounded single-Tx/passive-Rx ISAC performance audit.

Exactly one UAV radiates one target-illuminating OTFS waveform.  Every other
UAV is a passive receiver, so adding receivers does not duplicate sensing
power or sensing airtime.  The all-passive sum assumes receiver-local thermal
noise is independent and contains no common clutter; it is therefore an
explicit diagnostic reference, not a field-performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.params import get_default_config
from uav_isac.physical.bistatic_waveform import (
    bistatic_geometry_to_waveform,
    ideal_coherent_h0_deflection,
    real_equivalent_complex_noise_variance,
)
from uav_isac.physical.channel import compute_noise_power
from uav_isac.physical.detection import compute_detection_probabilities
from uav_isac.physical.waveform_evidence import MinimalOTFSWaveform
from uav_isac.utils.math_utils import Q_inverse


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, data.size, size=(4000, data.size))
    means = np.mean(data[draws], axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def audit(
    *, geometries: int, uavs: int, region_size_m: float, seed: int,
) -> dict:
    if int(geometries) < 20 or int(uavs) < 2:
        raise ValueError("geometries>=20 and uavs>=2 are required")
    region_length = float(region_size_m)
    if not np.isfinite(region_length) or region_length <= 0.0:
        raise ValueError("region_size_m must be finite and positive")
    cfg = get_default_config()
    waveform_noise = real_equivalent_complex_noise_variance(
        compute_noise_power(cfg.channel.kT, cfg.otfs.B, cfg.channel.NF))
    waveform = MinimalOTFSWaveform(
        delay_bins=int(cfg.otfs.M),
        doppler_bins=int(cfg.otfs.N),
        delta_f_hz=float(cfg.otfs.delta_f),
        noise_variance=waveform_noise,
    )
    rng = np.random.default_rng(int(seed))
    K = int(uavs)
    roles = np.ones(K, dtype=np.int64)
    roles[0] = 0
    power = np.zeros((K, 1), dtype=np.float64)
    power[0, 0] = float(cfg.uav.P_sense)
    best_pd = []
    passive_pd = []
    feasible_receiver_counts = []
    unmet_floor_geometries = 0
    alias_failures = 0
    region = np.full(2, region_length, dtype=np.float64)
    target_speed_max = float(max(cfg.target.speed_range))
    for _ in range(int(geometries)):
        uav_xy = rng.random((K, 2)) * region
        uav_xyz = np.column_stack((uav_xy, np.full(K, cfg.scenario.height)))
        target_xy = rng.random((1, 2)) * region
        target_xyz = np.column_stack((target_xy, np.zeros(1)))
        uav_speed = rng.uniform(0.0, float(cfg.uav.v_max), size=K)
        uav_angle = rng.uniform(0.0, 2.0 * np.pi, size=K)
        uav_velocity = np.column_stack((
            uav_speed * np.cos(uav_angle),
            uav_speed * np.sin(uav_angle),
            np.zeros(K),
        ))
        target_speed = rng.uniform(0.0, target_speed_max)
        target_angle = rng.uniform(0.0, 2.0 * np.pi)
        target_velocity = np.array([[
            target_speed * np.cos(target_angle),
            target_speed * np.sin(target_angle),
            0.0,
        ]])
        physical = bistatic_geometry_to_waveform(
            uav_xyz,
            uav_velocity,
            target_xyz,
            target_velocity,
            roles,
            power,
            carrier_hz=float(cfg.otfs.fc),
            waveform=waveform,
            rcs_m2=float(cfg.target.rcs),
            tx_gain_dbi=float(cfg.otfs.g_tx_dBi),
            rx_gain_dbi=float(cfg.otfs.g_rx_dBi),
        )
        edge_mask = physical.valid_edge_mask[0, :, 0]
        usable = edge_mask & physical.unambiguous_edge_mask[0, :, 0]
        alias_failures += int(np.any(edge_mask & ~usable))
        edge_d = ideal_coherent_h0_deflection(
            physical.received_target_amplitude[0, usable, 0],
            waveform=waveform,
            complex_noise_variance=waveform_noise,
            n_cpi=int(cfg.otfs.n_cpi),
        )
        ordered = np.sort(np.asarray(edge_d, dtype=np.float64))[::-1]
        best_d = float(ordered[0]) if ordered.size else 0.0
        all_d = float(np.sum(ordered))
        pd = compute_detection_probabilities(
            np.asarray([best_d, all_d]), float(cfg.detection.P_FA))
        best_pd.append(float(pd[0]))
        passive_pd.append(float(pd[1]))
        cumulative_pd = compute_detection_probabilities(
            np.cumsum(ordered), float(cfg.detection.P_FA))
        enough = np.flatnonzero(cumulative_pd >= float(cfg.detection.P_D_min))
        if enough.size:
            feasible_receiver_counts.append(int(enough[0] + 1))
        else:
            unmet_floor_geometries += 1

    best = np.asarray(best_pd)
    passive = np.asarray(passive_pd)
    gain = passive - best
    gain_ci = _bootstrap_mean_ci(gain, seed=int(seed) + 991)
    floor = float(cfg.detection.P_D_min)
    return {
        "status": "PASS",
        "evidence_class": "IDEAL_SINGLE_TX_PASSIVE_RX_DIAGNOSTIC",
        "geometry_distribution": (
            "independent uniform UAV/target xy in frozen square; fixed UAV height; "
            "uniform speed and heading within frozen limits"),
        "assumptions": [
            "one target waveform and one sensing-power charge per geometry",
            "perfect continuous delay-Doppler template knowledge",
            "independent receiver thermal noise",
            "no common clutter, blockage, packet loss, or evidence quantization",
        ],
        "geometries": int(geometries),
        "uavs": K,
        "region_size_m": [region_length, region_length],
        "p_fa": float(cfg.detection.P_FA),
        "pd_floor": floor,
        "best_single_pd_mean": float(np.mean(best)),
        "all_passive_pd_mean": float(np.mean(passive)),
        "best_single_floor_rate": float(np.mean(best >= floor)),
        "all_passive_floor_rate": float(np.mean(passive >= floor)),
        "paired_pd_gain_mean": float(np.mean(gain)),
        "paired_pd_gain_ci95": list(gain_ci),
        "minimum_receivers_median_when_feasible": float(
            np.median(feasible_receiver_counts)),
        "minimum_receivers_p95_when_feasible": float(
            np.percentile(feasible_receiver_counts, 95)),
        "all_passive_unmet_floor_geometries": int(unmet_floor_geometries),
        "performance_assessment": (
            "IDEAL_REFERENCE_HAS_TAIL_FAILURES"
            if unmet_floor_geometries else "IDEAL_REFERENCE_MEETS_FLOOR"
        ),
        "geometries_with_any_dd_alias": int(alias_failures),
        "normal_quantile_at_pfa": float(Q_inverse(cfg.detection.P_FA)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometries", type=int, default=200)
    parser.add_argument("--uavs", type=int, default=8)
    parser.add_argument("--region-size-m", type=float, default=1130.0)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    print(json.dumps(
        audit(
            geometries=args.geometries,
            uavs=args.uavs,
            region_size_m=args.region_size_m,
            seed=args.seed,
        ),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
