"""Fail-closed provenance contract for current-model layer audits.

The layer label (deployed/L1/L2/L3) is meaningful only when the executed
decision, physical coefficients, RF budgets and detector convention refer to
the same frame and the same implementation.  This module defines the compact,
versioned contract used by fresh P0 traces.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LAYER_TRACE_SCHEMA_VERSION = 3

REQUIRED_FRAME_ARRAYS = (
    "episode",
    "seed",
    "frame",
    "uav_positions",
    "uav_velocities",
    "target_positions",
    "target_velocities",
    "deployed_selected",
    "deployed_role",
    "deployed_receiver_owner",
    "deployed_sensing_power_w",
    "deployed_comm_power_w",
    "per_watt_coefficient",
    "privileged_g_dd",
    "privileged_chi_rep",
    "physical_pd",
    "task_mean_pd",
    "task_weak3_pd",
    "task_worst_pd",
    "frame_sensing_budget_w",
)

REQUIRED_PROVENANCE_HASHES = (
    "code_sha256",
    "config_sha256",
    "physics_sha256",
    "checkpoint_sha256",
)


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON data without depending on whitespace or dictionary order."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_state_dict(state: Mapping[str, Any]) -> str:
    """Hash a model state by tensor name, dtype, shape and exact value bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def physics_contract_from_environment(core: Any, config: Any) -> dict[str, Any]:
    """Resolve the detector/OTFS/RF convention actually used by an env."""
    dc = core.deflection_computer
    return {
        "contract_version": 1,
        "deflection": {
            "definition": "matched_filter_signal_energy_over_noise_psd",
            "c_det": float(dc.c_det),
            "n_cpi": int(dc.n_cpi),
            # No additional effective-look multiplier exists in the current
            # detector.  Recording one explicitly prevents a hidden L_eff.
            "l_eff": 1,
            "M": int(dc.M),
            "N": int(dc.N),
            "T_sym_s": float(dc.T_sym),
            "implied_bandwidth_hz": float(dc.M / dc.T_sym),
            "noise_power_w": float(dc.noise_power),
        },
        "rf": {
            "total_power_cap_w_per_uav": float(core._isac_total_power_w),
            "sensing_power_cap_w_per_uav": float(
                core._sensing_power_cap_w),
            "configured_P_sense_max_w": float(config.uav.P_sense_max),
        },
        "detector": {
            "name": "gaussian_deflection_q_function",
            "p_fa": float(config.detection.P_FA),
            "fusion_mode": str(core._detection_fusion_mode),
        },
        "delay_doppler": {
            "g_min": float(dc.g_min),
            "effectiveness": "sinc2_delay_times_sinc2_doppler",
        },
        "reporting_reliability": {
            "enabled": bool(dc.use_report_link),
            "coefficient": "chi_rep",
        },
    }


def build_layer_trace_provenance(
    *,
    run_binding: Mapping[str, Any],
    physics_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a trace to executable code, config, physics and policy state."""
    payload = dict(run_binding)
    payload["physics_contract"] = dict(physics_contract)
    payload["physics_sha256"] = canonical_json_sha256(physics_contract)
    missing = [
        key for key in REQUIRED_PROVENANCE_HASHES
        if not isinstance(payload.get(key), str) or not payload[key]
    ]
    if missing:
        raise ValueError("trace provenance lacks hashes: " + ", ".join(missing))
    return payload


def _scalar_int(data: Mapping[str, Any], name: str) -> int:
    values = np.asarray(data[name]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{name} must be scalar")
    return int(values[0])


def audit_layer_trace_provenance(
    trace: str | Path | Mapping[str, Any],
    *,
    expected_physics_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a P0 trace, rejecting incomplete and cross-version evidence."""
    close = False
    if isinstance(trace, (str, Path)):
        data = np.load(Path(trace), allow_pickle=False)
        close = True
    else:
        data = trace
    try:
        fields = set(data.files if hasattr(data, "files") else data.keys())
        if "schema_version" not in fields:
            raise ValueError("trace has no schema_version")
        version = _scalar_int(data, "schema_version")
        if version != LAYER_TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"layer trace schema {version} is not current schema "
                f"{LAYER_TRACE_SCHEMA_VERSION}")
        missing = sorted(set(REQUIRED_FRAME_ARRAYS) - fields)
        if missing:
            raise ValueError("layer trace lacks frame arrays: " + ", ".join(missing))
        if "provenance_json" not in fields:
            raise ValueError("layer trace lacks provenance_json")
        raw = np.asarray(data["provenance_json"]).reshape(-1)
        if raw.size != 1:
            raise ValueError("provenance_json must be scalar")
        provenance = json.loads(str(raw[0]))
        missing_hashes = [
            key for key in REQUIRED_PROVENANCE_HASHES
            if not isinstance(provenance.get(key), str)
            or len(provenance[key]) != 64
        ]
        if missing_hashes:
            raise ValueError(
                "trace provenance lacks valid SHA-256 values: "
                + ", ".join(missing_hashes))
        physics = provenance.get("physics_contract")
        if not isinstance(physics, dict):
            raise ValueError("trace provenance lacks physics_contract")
        actual_physics_hash = canonical_json_sha256(physics)
        if actual_physics_hash != provenance["physics_sha256"]:
            raise ValueError("physics contract hash does not match its payload")
        if (
            expected_physics_sha256 is not None
            and actual_physics_hash != expected_physics_sha256
        ):
            raise ValueError("trace physics does not match the requested model")

        frame_count = int(np.asarray(data["frame"]).shape[0])
        if frame_count < 1:
            raise ValueError("layer trace contains no frames")
        bad_lengths = [
            name for name in REQUIRED_FRAME_ARRAYS
            if int(np.asarray(data[name]).shape[0]) != frame_count
        ]
        if bad_lengths:
            raise ValueError(
                "frame-axis length mismatch: " + ", ".join(bad_lengths))
        for name in (
            "deployed_sensing_power_w", "deployed_comm_power_w",
            "per_watt_coefficient", "privileged_g_dd",
            "privileged_chi_rep", "physical_pd", "task_mean_pd",
            "task_weak3_pd", "task_worst_pd", "frame_sensing_budget_w",
        ):
            values = np.asarray(data[name])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(np.asarray(data["deployed_sensing_power_w"]) < -1.0e-12):
            raise ValueError("deployed sensing power is negative")
        if np.any(np.asarray(data["deployed_comm_power_w"]) < -1.0e-12):
            raise ValueError("deployed communication power is negative")
        if np.any(np.asarray(data["frame_sensing_budget_w"]) < -1.0e-12):
            raise ValueError("frame sensing budget is negative")
        return {
            "gate": "PASS",
            "schema_version": version,
            "frame_count": frame_count,
            "physics_sha256": actual_physics_hash,
            "code_sha256": provenance["code_sha256"],
            "config_sha256": provenance["config_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
        }
    finally:
        if close:
            data.close()
