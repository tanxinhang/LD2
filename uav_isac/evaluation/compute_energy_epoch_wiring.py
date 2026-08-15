"""Fail-closed binding of a frozen compute-energy epoch to one controller run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution


@dataclass(frozen=True)
class BoundComputeEnergyEpoch:
    package_energy_bound_j: float
    metadata: Mapping[str, object]
    energy_certificate_candidate: bool


def _ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def bind_compute_energy_epoch(
    calibration: Mapping[str, object],
    *,
    current_controller_sha256: str,
    controller_episode_ids: Sequence[str],
) -> BoundComputeEnergyEpoch:
    """Validate provenance, split isolation and the once-only package bound."""
    recommended = calibration.get("recommended_controller_parameters")
    epoch = calibration.get("epoch")
    provenance = calibration.get("provenance")
    authority = calibration.get("authority")
    if not all(isinstance(value, Mapping) for value in (
        recommended, epoch, provenance, authority,
    )):
        raise ValueError("compute-energy calibration epoch has invalid structure")
    if bool(authority.get("system_commit_authority", False)):
        raise ValueError("compute-energy artifact cannot grant commit authority")
    if not bool(recommended.get("usable_for_shadow_controller", False)):
        raise ValueError("compute-energy calibration epoch is not finite/usable")
    if recommended.get("rf_energy_included") is not False:
        raise ValueError(
            "compute-energy epoch must exclude RF energy to avoid double counting")
    bound = float(recommended["complete_controller_package_energy_bound_j"])
    if (
        not bool(epoch.get("finite", False))
        or not np.isfinite(bound)
        or bound < 0.0
    ):
        raise ValueError("compute-energy calibration bound is not finite")

    frozen_fingerprint = str(
        provenance.get("controller_implementation_sha256", ""))
    current_fingerprint = str(current_controller_sha256)
    if frozen_fingerprint != current_fingerprint:
        raise ValueError(
            "compute-energy epoch controller/config fingerprint mismatch: "
            f"frozen={frozen_fingerprint}, current={current_fingerprint}")
    if provenance.get("meter_scope") != "complete_controller_package":
        raise ValueError("compute-energy epoch does not meter the complete package")
    for name in (
        "hardware_id", "runtime_id", "meter_id", "meter_calibration_id",
    ):
        if not str(provenance.get(name, "")).strip():
            raise ValueError(f"compute-energy epoch has no stable {name}")

    training_ids = _ids(epoch.get("training_episode_ids", ()),
                        "training_episode_ids")
    calibration_ids = _ids(epoch.get("calibration_episode_ids", ()),
                           "calibration_episode_ids")
    validation_document = calibration.get("validation")
    validation_ids = _ids(
        (validation_document or {}).get("validation_episode_ids", ()),
        "validation_episode_ids",
    ) if isinstance(validation_document, Mapping) else ()
    split_sets = (set(training_ids), set(calibration_ids), set(validation_ids))
    if (
        split_sets[0] & split_sets[1]
        or split_sets[0] & split_sets[2]
        or split_sets[1] & split_sets[2]
    ):
        raise ValueError("compute-energy artifact has internal episode leakage")
    controller_ids = {str(value) for value in controller_episode_ids}
    source_ids = set().union(*split_sets)
    overlap = controller_ids & source_ids
    if overlap:
        raise ValueError(
            "compute-energy epoch/controller evaluation episode leakage: "
            f"{sorted(overlap)}")

    source_kind = str(provenance.get("source_kind", "unspecified"))
    empirical = source_kind in {"rapl_package", "external_cpu_rail"}
    uncertainty_accounted = bool(provenance.get(
        "meter_difference_uncertainty_accounted", False))
    independent = bool(validation_ids) and bool(
        authority.get("independent_validation_present", False))
    validation_supports_risk = False
    exact_validation_upper = None
    if validation_ids:
        if not isinstance(validation_document, Mapping):
            raise ValueError("validation episode IDs require validation metadata")
        failure_ids = _ids(
            validation_document.get("failure_episode_ids", ()),
            "failure_episode_ids")
        if not set(failure_ids) <= set(validation_ids):
            raise ValueError("energy validation failures are outside its split")
        failure_count = int(validation_document.get("failure_count", -1))
        episode_count = int(validation_document.get("episode_count", -1))
        if failure_count != len(failure_ids) or episode_count != len(
            validation_ids
        ):
            raise ValueError("energy validation counts are inconsistent")
        exact_validation_upper = (
            1.0 if failure_count == episode_count else float(
                beta_distribution.ppf(
                    0.95, failure_count + 1,
                    episode_count - failure_count)))
        reported_upper = float(validation_document.get(
            "failure_rate_clopper_pearson_one_sided95_upper", -1.0))
        if not np.isclose(
            exact_validation_upper, reported_upper, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("energy validation exact upper bound is inconsistent")
        validation_supports_risk = bool(
            exact_validation_upper
            <= float(epoch["miscoverage"]) + 1.0e-15)
    candidate = bool(
        empirical
        and uncertainty_accounted
        and independent
        and validation_supports_risk
        and authority.get("compute_energy_certificate_candidate", False)
    )
    metadata = {
        "epoch_id": str(epoch.get("epoch_id")),
        "source_kind": source_kind,
        "controller_implementation_sha256": frozen_fingerprint,
        "current_controller_implementation_sha256": current_fingerprint,
        "hardware_id": str(provenance.get("hardware_id")),
        "runtime_id": str(provenance.get("runtime_id")),
        "meter_id": str(provenance.get("meter_id")),
        "meter_calibration_id": str(provenance.get("meter_calibration_id")),
        "meter_difference_uncertainty_accounted": uncertainty_accounted,
        "meter_scope": str(provenance.get("meter_scope")),
        "package_energy_bound_j": bound,
        "miscoverage": float(epoch["miscoverage"]),
        "coverage_floor": float(epoch["coverage_floor"]),
        "calibration_episode_ids": list(calibration_ids),
        "training_episode_ids": list(training_ids),
        "validation_episode_ids": list(validation_ids),
        "independent_validation_present": independent,
        "validation_failure_rate_exact_one_sided95_upper": (
            exact_validation_upper),
        "validation_supports_declared_risk": validation_supports_risk,
        "empirical_hardware_source": empirical,
        "compute_energy_certificate_candidate": candidate,
        "input": provenance.get("input"),
        "artifact": provenance.get("calibration_artifact"),
    }
    return BoundComputeEnergyEpoch(
        package_energy_bound_j=bound,
        metadata=metadata,
        energy_certificate_candidate=candidate,
    )
