#!/usr/bin/env python
"""Falsify unsafe mappings from general sensing resources to ``p_iq``."""

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
    certify_target_separable_lp_specialization,
)


def _case(
    mode: SensingMode,
    signatures: np.ndarray,
    *,
    communication_power_w: float = 0.0,
) -> dict:
    occupancy = ResourceOccupancy(
        mode=mode,
        mask=np.ones((1, 2, 4, 4), dtype=bool),
        stream_ids=("cell-0", "cell-1"),
        hypothesis_ids=(0, 1) if mode is SensingMode.TARGET_SEPARABLE else (None, None),
    )
    caps = SensingResourceCaps(
        epoch_duration_s=1.0,
        bandwidth_hz=100.0,
        peak_rf_power_w=np.array([0.5]),
        sensing_energy_cap_j=np.array([0.4]),
        total_rf_energy_cap_j=np.array([0.6]),
        maximum_tf_fraction=np.array([1.0]),
    )
    communication_energy = (
        None if communication_power_w == 0.0
        else np.array([communication_power_w]))
    communication_trace = (
        None if communication_power_w == 0.0
        else np.full((1, 4), communication_power_w))
    certificate = certify_target_separable_lp_specialization(
        occupancy,
        stream_energy_j=np.array([[0.2, 0.2]]),
        stream_power_envelope_w=np.array([[0.2, 0.2]]),
        caps=caps,
        hypothesis_signatures=signatures,
        maximum_mutual_coherence=0.2,
        minimum_eigenvalue=0.5,
        communication_energy_j=communication_energy,
        communication_power_envelope_w=communication_trace,
    )
    diagnostic = certificate.separation_diagnostics
    return {
        "passed": certificate.passed,
        "failure_reasons": list(certificate.failure_reasons),
        "mutual_coherence": (
            None if diagnostic is None else diagnostic.mutual_coherence),
        "minimum_eigenvalue": (
            None if diagnostic is None else diagnostic.minimum_eigenvalue),
        "equivalent_average_power_w": (
            None if certificate.equivalent_average_power_w is None
            else certificate.equivalent_average_power_w.tolist()),
    }


def audit() -> dict:
    valid = _case(SensingMode.TARGET_SEPARABLE, np.eye(2))
    common = _case(SensingMode.COMMON_PROBE, np.eye(2))
    redundant = _case(
        SensingMode.TARGET_SEPARABLE,
        np.array([[1.0, 1.0], [0.0, 0.01]]),
    )
    peak_overload = _case(
        SensingMode.TARGET_SEPARABLE,
        np.eye(2),
        communication_power_w=0.2,
    )
    passed = (
        valid["passed"]
        and not common["passed"]
        and "common_probe_has_no_target_separable_lp_mapping"
        in common["failure_reasons"]
        and not redundant["passed"]
        and "separation_not_certified" in redundant["failure_reasons"]
        and not peak_overload["passed"]
        and "peak_rf_power_exceeded" in peak_overload["failure_reasons"]
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "OFFLINE_LP_SPECIALIZATION_CONTRACT",
        "online_behavior_changed": False,
        "cases": {
            "valid_target_separable": valid,
            "common_probe_counterexample": common,
            "redundant_signature_counterexample": redundant,
            "communication_peak_counterexample": peak_overload,
        },
    }


def main() -> int:
    result = audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
