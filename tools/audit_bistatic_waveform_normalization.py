#!/usr/bin/env python
"""Cross-check geometry, analytical Deflection, and waveform Monte Carlo."""

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
from uav_isac.physical.deflection import compute_raw_deflection
from uav_isac.physical.detection import (
    compute_detection_probabilities,
    minimum_deflection_for_detection_probability,
)
from uav_isac.physical.waveform_evidence import (
    MinimalOTFSWaveform,
    WaveformEvidenceScenario,
    generate_local_evidence_trace,
)


def audit(*, distances_m: tuple[float, ...], trials: int, seed: int) -> dict:
    cfg = get_default_config()
    waveform = MinimalOTFSWaveform(
        delay_bins=int(cfg.otfs.M),
        doppler_bins=int(cfg.otfs.N),
        delta_f_hz=float(cfg.otfs.delta_f),
        noise_variance=real_equivalent_complex_noise_variance(
            compute_noise_power(cfg.channel.kT, cfg.otfs.B, cfg.channel.NF)),
    )
    real_noise = 0.5 * float(waveform.noise_variance)
    rows = []
    for index, leg in enumerate(distances_m):
        distance = float(leg)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("distances_m must be finite and positive")
        uav = np.array([[-distance, 0.0, 0.0], [distance, 0.0, 0.0]])
        velocity = np.zeros_like(uav)
        target = np.zeros((1, 3), dtype=np.float64)
        target_velocity = np.zeros_like(target)
        roles = np.array([0, 1])
        power = np.array([[float(cfg.uav.P_sense)], [0.0]])
        physical = bistatic_geometry_to_waveform(
            uav,
            velocity,
            target,
            target_velocity,
            roles,
            power,
            carrier_hz=float(cfg.otfs.fc),
            waveform=waveform,
            rcs_m2=float(cfg.target.rcs),
            tx_gain_dbi=float(cfg.otfs.g_tx_dBi),
            rx_gain_dbi=float(cfg.otfs.g_rx_dBi),
            require_unambiguous=True,
        )
        edge = (0, 1, 0)
        analytical = compute_raw_deflection(
            float(physical.path_amplitude[edge]),
            float(cfg.uav.P_sense),
            float(cfg.otfs.T_sym),
            int(cfg.otfs.M),
            int(cfg.otfs.N),
            real_noise,
            antenna_gain=10.0 ** (
                (float(cfg.otfs.g_tx_dBi) + float(cfg.otfs.g_rx_dBi)) / 10.0),
            n_cpi=int(cfg.otfs.n_cpi),
            c_det=float(cfg.detection.c_det),
        )
        waveform_expected = float(ideal_coherent_h0_deflection(
            physical.received_target_amplitude[edge],
            waveform=waveform,
            complex_noise_variance=float(waveform.noise_variance),
            n_cpi=int(cfg.otfs.n_cpi),
        ))
        scenario = WaveformEvidenceScenario(
            target_delay_bin=np.array([physical.delay_bin[edge]]),
            target_doppler_bin=np.array([physical.doppler_bin[edge]]),
            target_amplitude=np.array([
                physical.received_target_amplitude[edge]]),
            common_clutter_delay_bin=np.zeros(1),
            common_clutter_doppler_bin=np.zeros(1),
            common_clutter_loading=np.ones(1),
            common_clutter_std=0.0,
            local_clutter_delay_bin=np.zeros(1),
            local_clutter_doppler_bin=np.zeros(1),
            local_clutter_std=np.zeros(1),
        )
        h0 = generate_local_evidence_trace(
            waveform, scenario, trials=trials, hypothesis=0,
            seed=int(seed) + 2 * index)
        h1 = generate_local_evidence_trace(
            waveform, scenario, trials=trials, hypothesis=1,
            seed=int(seed) + 2 * index + 1)
        empirical = float(
            (np.mean(h1) - np.mean(h0)) ** 2 / np.var(h0, ddof=1))
        analytical_pd = float(compute_detection_probabilities(
            np.asarray([analytical]), float(cfg.detection.P_FA))[0])
        rows.append({
            "leg_distance_m": distance,
            "analytical_deflection": float(analytical),
            "waveform_expected_deflection": waveform_expected,
            "empirical_deflection": empirical,
            "analytical_relative_error": float(
                abs(waveform_expected - analytical) / max(analytical, 1.0e-15)),
            "empirical_relative_error": float(
                abs(empirical - analytical) / max(analytical, 1.0e-15)),
            "analytical_pd": analytical_pd,
            "meets_single_edge_pd_floor": bool(
                analytical_pd >= float(cfg.detection.P_D_min)),
        })
    maximum_exact_error = max(row["analytical_relative_error"] for row in rows)
    maximum_mc_error = max(row["empirical_relative_error"] for row in rows)
    required_d = float(minimum_deflection_for_detection_probability(
        np.asarray([cfg.detection.P_D_min]), float(cfg.detection.P_FA))[0])
    # In the symmetric audit geometry alpha^2, and hence D, scales as r^-4.
    reference = rows[0]
    single_edge_leg_limit = float(
        reference["leg_distance_m"]
        * (reference["analytical_deflection"] / required_d) ** 0.25)
    return {
        "status": "PASS" if maximum_exact_error <= 1.0e-12 and maximum_mc_error <= 0.10 else "FAIL",
        "detector_convention": "real_gaussian_shift_c_det_1",
        "complex_noise_variance_rule": "sigma_complex_squared=2*real_equivalent_noise_power",
        "trials_per_hypothesis": int(trials),
        "maximum_analytical_relative_error": float(maximum_exact_error),
        "maximum_monte_carlo_relative_error": float(maximum_mc_error),
        "p_fa": float(cfg.detection.P_FA),
        "single_edge_pd_floor": float(cfg.detection.P_D_min),
        "deflection_required_for_pd_floor": required_d,
        "symmetric_single_edge_leg_limit_m": single_edge_leg_limit,
        "all_audit_ranges_meet_single_edge_pd_floor": bool(all(
            row["meets_single_edge_pd_floor"] for row in rows)),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances-m", default="100,300,500,800")
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    distances = tuple(float(item) for item in args.distances_m.split(","))
    if args.trials < 1000:
        raise ValueError("trials must be at least 1000")
    result = audit(distances_m=distances, trials=args.trials, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(bool(args.strict and result["status"] != "PASS"))


if __name__ == "__main__":
    raise SystemExit(main())
