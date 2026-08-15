"""Fail-closed binding of runtime-latency epochs to controller evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution


@dataclass(frozen=True)
class BoundRuntimeLatencyEpoch:
    complete_compute_latency_bound_s: float
    metadata: Mapping[str, object]
    runtime_latency_certificate_candidate: bool


def _ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def bind_runtime_latency_epoch(
    calibration: Mapping[str, object],
    *,
    current_controller_sha256: str,
    controller_episode_ids: Sequence[str],
) -> BoundRuntimeLatencyEpoch:
    recommended = calibration.get("recommended_controller_parameters")
    epoch = calibration.get("epoch")
    provenance = calibration.get("provenance")
    authority = calibration.get("authority")
    if not all(isinstance(value, Mapping) for value in (
        recommended, epoch, provenance, authority,
    )):
        raise ValueError("runtime calibration epoch has invalid structure")
    if bool(authority.get("system_commit_authority", False)):
        raise ValueError("runtime artifact cannot grant commit authority")
    if not bool(recommended.get("usable_for_shadow_controller", False)):
        raise ValueError("runtime calibration epoch is not finite/usable")
    if recommended.get("network_latency_included") is not False:
        raise ValueError(
            "runtime epoch must exclude network latency to avoid double counting")
    bound = float(recommended[
        "complete_controller_compute_latency_bound_s"])
    if (
        not bool(epoch.get("finite", False))
        or not np.isfinite(bound) or bound < 0.0
    ):
        raise ValueError("runtime calibration bound is not finite")
    frozen = str(provenance.get("controller_implementation_sha256", ""))
    current = str(current_controller_sha256)
    if frozen != current:
        raise ValueError(
            "runtime epoch controller/config fingerprint mismatch: "
            f"frozen={frozen}, current={current}")
    if provenance.get("measurement_scope") != (
        "complete_parallel_controller_compute_critical_path_excluding_network"
    ):
        raise ValueError("runtime epoch has the wrong measurement scope")
    if provenance.get("clock_semantics") != (
        "monotonic_wall_clock_includes_preemption"
    ):
        raise ValueError("runtime epoch does not use monotonic wall time")
    for name in (
        "hardware_id", "runtime_id", "clock_id", "clock_calibration_id",
    ):
        if not str(provenance.get(name, "")).strip():
            raise ValueError(f"runtime epoch has no stable {name}")

    training = _ids(epoch.get("training_episode_ids", ()),
                    "training_episode_ids")
    calibration_ids = _ids(epoch.get("calibration_episode_ids", ()),
                           "calibration_episode_ids")
    validation_document = calibration.get("validation")
    validation_ids = _ids(
        (validation_document or {}).get("validation_episode_ids", ()),
        "validation_episode_ids",
    ) if isinstance(validation_document, Mapping) else ()
    split_sets = (set(training), set(calibration_ids), set(validation_ids))
    if (
        split_sets[0] & split_sets[1]
        or split_sets[0] & split_sets[2]
        or split_sets[1] & split_sets[2]
    ):
        raise ValueError("runtime artifact has internal episode leakage")
    overlap = {str(value) for value in controller_episode_ids} & set().union(
        *split_sets)
    if overlap:
        raise ValueError(
            "runtime epoch/controller evaluation episode leakage: "
            f"{sorted(overlap)}")

    exact_upper = None
    validation_supports = False
    if validation_ids:
        if not isinstance(validation_document, Mapping):
            raise ValueError("runtime validation IDs require validation metadata")
        failure_ids = _ids(
            validation_document.get("failure_episode_ids", ()),
            "failure_episode_ids")
        if not set(failure_ids) <= set(validation_ids):
            raise ValueError("runtime validation failures are outside its split")
        failure_count = int(validation_document.get("failure_count", -1))
        episode_count = int(validation_document.get("episode_count", -1))
        if failure_count != len(failure_ids) or episode_count != len(
            validation_ids
        ):
            raise ValueError("runtime validation counts are inconsistent")
        exact_upper = (
            1.0 if failure_count == episode_count else float(
                beta_distribution.ppf(
                    0.95, failure_count + 1,
                    episode_count - failure_count)))
        reported = float(validation_document.get(
            "failure_rate_clopper_pearson_one_sided95_upper", -1.0))
        if not np.isclose(exact_upper, reported, rtol=0.0, atol=1.0e-12):
            raise ValueError("runtime validation exact upper is inconsistent")
        validation_supports = bool(
            exact_upper <= float(epoch["miscoverage"]) + 1.0e-15)

    deployment_source = str(provenance.get("source_kind")) == (
        "deployment_monotonic_clock")
    uncertainty_accounted = bool(provenance.get(
        "timer_difference_uncertainty_accounted", False))
    independent = bool(validation_ids) and bool(
        authority.get("independent_validation_present", False))
    candidate = bool(
        deployment_source and uncertainty_accounted and independent
        and validation_supports
        and authority.get("runtime_latency_certificate_candidate", False))
    metadata = {
        "epoch_id": str(epoch.get("epoch_id")),
        "source_kind": str(provenance.get("source_kind")),
        "controller_implementation_sha256": frozen,
        "current_controller_implementation_sha256": current,
        "hardware_id": str(provenance.get("hardware_id")),
        "runtime_id": str(provenance.get("runtime_id")),
        "clock_id": str(provenance.get("clock_id")),
        "clock_calibration_id": str(provenance.get("clock_calibration_id")),
        "complete_compute_latency_bound_s": bound,
        "miscoverage": float(epoch["miscoverage"]),
        "coverage_floor": float(epoch["coverage_floor"]),
        "training_episode_ids": list(training),
        "calibration_episode_ids": list(calibration_ids),
        "validation_episode_ids": list(validation_ids),
        "validation_failure_rate_exact_one_sided95_upper": exact_upper,
        "validation_supports_declared_risk": validation_supports,
        "deployment_timing_source": deployment_source,
        "timer_difference_uncertainty_accounted": uncertainty_accounted,
        "independent_validation_present": independent,
        "runtime_latency_certificate_candidate": candidate,
        "input": provenance.get("input"),
        "artifact": provenance.get("calibration_artifact"),
    }
    return BoundRuntimeLatencyEpoch(
        complete_compute_latency_bound_s=bound,
        metadata=metadata,
        runtime_latency_certificate_candidate=candidate,
    )
