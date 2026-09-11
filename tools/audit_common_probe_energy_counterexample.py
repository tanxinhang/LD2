#!/usr/bin/env python
"""Equal-resource counterexample to per-target accounting of a common probe.

This is an ideal-cyclic, coherent, single-transmitter falsification case.  A
single broadcast probe is reflected by both hypothesis cells, whereas two
perfectly isolated target-directed spatial streams split the same sensing
energy.  The result is not a claim that common probing is generally superior.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.physical.sensing_resource import (
    ResourceOccupancy,
    SensingMode,
    SensingResourceCaps,
    audit_sensing_resource_conservation,
    certify_target_separable_lp_specialization,
    separation_diagnostics,
)
from uav_isac.physical.waveform_evidence import (
    MinimalOTFSWaveform,
    dd_path_response,
)


def _occupancy(mode: SensingMode) -> ResourceOccupancy:
    streams = 1 if mode is SensingMode.COMMON_PROBE else 2
    mask = np.zeros((1, streams, 4, 16), dtype=bool)
    mask[:, :, 1:, :] = True
    return ResourceOccupancy(
        mode=mode,
        mask=mask,
        stream_ids=("common",) if streams == 1 else ("cell-0", "cell-1"),
        hypothesis_ids=(None,) if streams == 1 else (0, 1),
        spatial_ids=() if streams == 1 else ("beam-0", "beam-1"),
    )


def _ideal_awgn_capacity_bits(
    *, bandwidth_hz: float, duration_s: float, receive_snr_linear: float,
) -> float:
    """Shannon capacity bound, not a finite-blocklength throughput claim."""
    values = (float(bandwidth_hz), float(duration_s), float(receive_snr_linear))
    if any(not np.isfinite(value) for value in values):
        raise ValueError("communication inputs must be finite")
    if bandwidth_hz <= 0.0 or duration_s <= 0.0 or receive_snr_linear < 0.0:
        raise ValueError("communication inputs lie outside their physical domain")
    return float(bandwidth_hz * duration_s * np.log2(1.0 + receive_snr_linear))


def audit() -> dict:
    waveform = MinimalOTFSWaveform(
        delay_bins=16,
        doppler_bins=8,
        delta_f_hz=15_625.0,
        noise_variance=1.0,
    )
    pilot = np.zeros((waveform.doppler_bins, waveform.delay_bins), dtype=complex)
    pilot[0, 0] = 1.0
    signatures = np.stack([
        dd_path_response(pilot, delay_bin=1.0, doppler_bin=1.0).reshape(-1),
        dd_path_response(pilot, delay_bin=4.0, doppler_bin=3.0).reshape(-1),
    ], axis=1)
    signature_diagnostic = separation_diagnostics(signatures)

    duration_s = 1.0
    bandwidth_hz = 1.0e6
    sensing_energy_j = 0.3
    communication_energy_j = 0.1
    peak_power_w = 0.4
    caps = SensingResourceCaps(
        epoch_duration_s=duration_s,
        bandwidth_hz=bandwidth_hz,
        peak_rf_power_w=np.array([peak_power_w]),
        sensing_energy_cap_j=np.array([sensing_energy_j]),
        total_rf_energy_cap_j=np.array([
            sensing_energy_j + communication_energy_j]),
        maximum_tf_fraction=np.array([0.75]),
    )
    communication_trace = np.array([[peak_power_w, 0.0, 0.0, 0.0]])
    common = _occupancy(SensingMode.COMMON_PROBE)
    separable = _occupancy(SensingMode.TARGET_SEPARABLE)
    common_audit = audit_sensing_resource_conservation(
        common,
        stream_energy_j=np.array([[sensing_energy_j]]),
        stream_power_envelope_w=np.array([[peak_power_w]]),
        caps=caps,
        communication_energy_j=np.array([communication_energy_j]),
        communication_power_envelope_w=communication_trace,
    )
    separable_energy = np.full((1, 2), sensing_energy_j / 2.0)
    separable_power = np.full((1, 2), peak_power_w / 2.0)
    separable_certificate = certify_target_separable_lp_specialization(
        separable,
        stream_energy_j=separable_energy,
        stream_power_envelope_w=separable_power,
        caps=caps,
        hypothesis_signatures=signatures,
        maximum_mutual_coherence=1.0e-10,
        minimum_eigenvalue=1.0 - 1.0e-10,
        communication_energy_j=np.array([communication_energy_j]),
        communication_power_envelope_w=communication_trace,
    )
    common_certificate = certify_target_separable_lp_specialization(
        common,
        stream_energy_j=np.array([[sensing_energy_j]]),
        stream_power_envelope_w=np.array([[peak_power_w]]),
        caps=caps,
        hypothesis_signatures=signatures,
        maximum_mutual_coherence=1.0e-10,
        minimum_eigenvalue=1.0 - 1.0e-10,
        communication_energy_j=np.array([communication_energy_j]),
        communication_power_envelope_w=communication_trace,
    )

    # For sqrt(E)*h*s plus CN(0,sigma^2), sqrt(2)Re matched detection has
    # D0=2*E*|h|^2/sigma^2.  The common broadcast illuminates both cells with
    # E; the ideal isolated streams use E/2 each and have no beamforming gain.
    channel_amplitude = np.array([1.0, 0.7])
    common_deflection = (
        2.0 * sensing_energy_j * np.square(channel_amplitude)
        / waveform.noise_variance)
    separable_deflection = (
        2.0 * separable_energy.reshape(-1) * np.square(channel_amplitude)
        / waveform.noise_variance)

    communication_capacity = _ideal_awgn_capacity_bits(
        bandwidth_hz=bandwidth_hz,
        duration_s=duration_s / 4.0,
        receive_snr_linear=10.0,
    )
    required_payload_bits = 500_000.0
    fair_resources = bool(
        np.allclose(common_audit.sensing_energy_j,
                    separable_certificate.resource_audit.sensing_energy_j)
        and np.allclose(common_audit.total_rf_energy_j,
                        separable_certificate.resource_audit.total_rf_energy_j)
        and np.allclose(common_audit.occupied_tf_fraction,
                        separable_certificate.resource_audit.occupied_tf_fraction)
        and np.allclose(common_audit.maximum_concurrent_power_envelope_w,
                        separable_certificate.resource_audit
                        .maximum_concurrent_power_envelope_w)
        and not np.any(
            np.any(common.mask, axis=(1, 3))
            & (communication_trace > 0.0))
    )
    passed = bool(
        common_audit.passed
        and separable_certificate.passed
        and not common_certificate.passed
        and fair_resources
        and communication_capacity >= required_payload_bits
        and np.allclose(common_deflection, 2.0 * separable_deflection)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "IDEAL_COMMON_PROBE_ENERGY_ACCOUNTING_COUNTEREXAMPLE",
        "online_behavior_changed": False,
        "resource_equality": {
            "passed": fair_resources,
            "sensing_energy_j": sensing_energy_j,
            "communication_energy_j": communication_energy_j,
            "total_energy_j": sensing_energy_j + communication_energy_j,
            "maximum_total_rf_power_w": peak_power_w,
            "sensing_tf_fraction": 0.75,
            "communication_tf_fraction": 0.25,
            "total_tf_fraction": 1.0,
            "communication_and_sensing_time_disjoint": True,
        },
        "communication_qos": {
            "evidence": "ideal_awgn_shannon_capacity_only",
            "receive_snr_linear": 10.0,
            "required_payload_bits": required_payload_bits,
            "capacity_upper_bound_bits": communication_capacity,
            "same_for_both_modes": True,
            "finite_blocklength_qos_certified": False,
        },
        "signature_separation": {
            "mutual_coherence": signature_diagnostic.mutual_coherence,
            "minimum_eigenvalue": signature_diagnostic.minimum_eigenvalue,
        },
        "common_probe_deflection": common_deflection.tolist(),
        "ideal_target_separable_deflection": separable_deflection.tolist(),
        "common_to_separable_ratio": (
            common_deflection / separable_deflection).tolist(),
        "common_probe_lp_mapping_rejected": not common_certificate.passed,
        "limitations": [
            "ideal cyclic OTFS and integer delay-Doppler cells",
            "coherent phase and unit normalized path signatures",
            "perfectly isolated spatial streams with no beamforming gain",
            "no clutter, blockage, leakage, PAPR, or RF impairments",
            "communication check is Shannon capacity, not finite-blocklength QoS",
        ],
        "interpretation": (
            "A common broadcast joule can illuminate multiple cells; it cannot "
            "be converted into per-cell LP energy by splitting that joule. The "
            "2x result is a controlled counterexample, not a universal gain."),
    }


def main() -> int:
    result = audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
