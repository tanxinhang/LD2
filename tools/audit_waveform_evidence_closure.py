"""G1--G3 falsification audit for the minimal OTFS evidence closure."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.physical.evidence_calibration import (  # noqa: E402
    calibrate_gaussian_evidence,
    choose_detection_covariance,
    covariance_relative_error,
    fixed_linear_detector_validation,
    subset_surrogate_rank_validation,
)
from uav_isac.physical.waveform_evidence import (  # noqa: E402
    MinimalOTFSWaveform,
    WaveformEvidenceScenario,
    generate_local_evidence_trace,
)


def _scenario(config: MinimalOTFSWaveform) -> WaveformEvidenceScenario:
    sample_count = config.sample_count
    desired_local_deflection = np.array([8.0, 7.2, 6.0, 5.0])
    amplitude = np.sqrt(
        desired_local_deflection
        * float(config.noise_variance)
        / (2.0 * sample_count)
    )
    return WaveformEvidenceScenario(
        target_delay_bin=np.array([2.0, 2.3, 6.4, 10.1]),
        target_doppler_bin=np.array([1.0, 1.2, -2.1, 2.7]),
        target_amplitude=amplitude,
        common_clutter_delay_bin=np.full(4, 2.15),
        common_clutter_doppler_bin=np.full(4, 1.1),
        common_clutter_loading=np.array([1.0, 0.9, 0.15, 0.05]),
        common_clutter_std=0.25,
        local_clutter_delay_bin=np.array([3.1, 3.3, 7.2, 9.4]),
        local_clutter_doppler_bin=np.array([0.2, 0.3, -1.3, 3.4]),
        local_clutter_std=np.full(4, 0.04),
        target_secondary_relative_gain=0.25 + 0.1j,
        target_secondary_delay_offset_bin=0.7,
        target_secondary_doppler_offset_bin=-0.35,
    )


def run_audit(*, samples: int, p_fa: float, seed: int) -> dict[str, object]:
    started = perf_counter()
    config = MinimalOTFSWaveform()
    scenario = _scenario(config)
    cal0 = generate_local_evidence_trace(
        config, scenario, trials=samples, hypothesis=0, seed=seed)
    cal1 = generate_local_evidence_trace(
        config, scenario, trials=samples, hypothesis=1, seed=seed + 1)
    val0 = generate_local_evidence_trace(
        config, scenario, trials=samples, hypothesis=0, seed=seed + 2)
    val1 = generate_local_evidence_trace(
        config, scenario, trials=samples, hypothesis=1, seed=seed + 3)
    calibration = calibrate_gaussian_evidence(cal0, cal1)
    validation_calibration = calibrate_gaussian_evidence(val0, val1)
    choice = choose_detection_covariance(
        calibration,
        equal_covariance_tolerance=0.15,
        maximum_condition_number=1.0e4,
    )
    weights = np.linalg.solve(choice.covariance, calibration.mean_shift)
    detector = fixed_linear_detector_validation(
        cal0, val0, val1, weights, p_fa=p_fa)

    snr_scale = (0.4, 0.7, 1.0, 1.3)
    pd_sweep: list[float] = []
    for index, scale in enumerate(snr_scale):
        scaled = replace(
            scenario,
            target_amplitude=np.asarray(scenario.target_amplitude) * scale,
        )
        scaled_h1 = generate_local_evidence_trace(
            config,
            scaled,
            trials=samples,
            hypothesis=1,
            seed=seed + 20 + index,
        )
        result = fixed_linear_detector_validation(
            cal0, val0, scaled_h1, weights, p_fa=p_fa)
        pd_sweep.append(float(result["validation_pd"]))

    correlation_h0 = np.corrcoef(cal0, rowvar=False)
    upper = correlation_h0[np.triu_indices(4, k=1)]
    covariance_stability = covariance_relative_error(
        calibration.covariance_h0,
        validation_calibration.covariance_h0,
    )
    ranking = subset_surrogate_rank_validation(
        calibration,
        choice,
        cal0,
        val0,
        val1,
        p_fa=p_fa,
    )

    # A target-amplitude fluctuation is a deliberate counterexample to the
    # equal-covariance model.  Passing this control means the policy falls back
    # to Sigma_0 instead of silently pooling unequal covariances.
    fluctuating = replace(scenario, target_fluctuation_std=0.25)
    control0 = generate_local_evidence_trace(
        config, fluctuating, trials=samples, hypothesis=0, seed=seed + 40)
    control1 = generate_local_evidence_trace(
        config, fluctuating, trials=samples, hypothesis=1, seed=seed + 41)
    control_calibration = calibrate_gaussian_evidence(control0, control1)
    control_choice = choose_detection_covariance(
        control_calibration,
        equal_covariance_tolerance=0.15,
        maximum_condition_number=1.0e4,
    )

    checks = {
        "g1_h0_threshold_controls_pfa": bool(
            0.5 * p_fa <= float(detector["validation_pfa"])
            <= p_fa + 0.005),
        "g1_pd_monotone_with_target_amplitude": bool(
            np.all(np.diff(pd_sweep) >= -0.005)),
        "g2_equal_covariance_supported": bool(
            calibration.equal_covariance_relative_error <= 0.15),
        "g2_heterogeneous_correlation_emerges": bool(
            correlation_h0[0, 1] - correlation_h0[0, 3] >= 0.2
            and float(np.max(upper) - np.min(upper)) >= 0.2),
        "g2_held_out_covariance_stable": bool(covariance_stability <= 0.15),
        "g3_surrogate_rank_valid": bool(float(ranking["spearman_r"]) >= 0.8),
        "unequal_covariance_control_uses_h0_fallback": bool(
            control_choice.policy == "h0_fixed_pfa_fallback"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failures else "FAIL",
        "evidence_class": "DIAGNOSTIC_ONLY",
        "scope": "ideal_cyclic_coherent_otfs_offline_calibration",
        "nonclaims": [
            "not an RF/hardware-faithful transceiver",
            "no CP-insufficiency, pulse-shaping, synchronization or RF impairments",
            "no unknown-phase GLRT calibration",
            "no online-controller or packet-delivery change",
        ],
        "config": {
            "delay_bins": config.delay_bins,
            "doppler_bins": config.doppler_bins,
            "delta_f_hz": config.delta_f_hz,
            "samples_per_hypothesis_split": samples,
            "p_fa": p_fa,
            "calibration_seed": seed,
            "validation_seed": seed + 2,
        },
        "checks": checks,
        "failure_reasons": failures,
        "g1": {
            "validation_pfa": float(detector["validation_pfa"]),
            "validation_pd": float(detector["validation_pd"]),
            "target_amplitude_scale": list(snr_scale),
            "validation_pd_sweep": pd_sweep,
        },
        "g2": {
            "calibration_correlation_h0": correlation_h0.tolist(),
            "equal_covariance_relative_error": float(
                calibration.equal_covariance_relative_error),
            "held_out_covariance_relative_error": float(covariance_stability),
            "covariance_policy": choice.policy,
            "condition_number_before": choice.condition_number_before,
            "condition_number_after": choice.condition_number_after,
            "diagonal_shrinkage": choice.shrinkage_to_diagonal,
            "unequal_covariance_control_relative_error": float(
                control_calibration.equal_covariance_relative_error),
            "unequal_covariance_control_policy": control_choice.policy,
        },
        "g3": ranking,
        "elapsed_s": float(perf_counter() - started),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--p-fa", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=52000)
    parser.add_argument("--output")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.samples < 1000 or not 0.0 < args.p_fa < 1.0:
        raise ValueError("samples must be >=1000 and p_fa must lie in (0,1)")
    result = run_audit(samples=args.samples, p_fa=args.p_fa, seed=args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(bool(args.strict and result["status"] != "PASS"))


if __name__ == "__main__":
    raise SystemExit(main())
