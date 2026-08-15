"""Calibrate a frozen episode-level controller package-energy epoch from JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

from scipy.stats import beta as beta_distribution


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_paired_horizon_confirmation import _sha256, _wilson
from uav_isac.evaluation.compute_energy_calibration import (
    ComputeEnergyEventObservation,
    calibrate_frozen_compute_energy_epoch,
    validate_frozen_compute_energy_epoch,
)


def _required_nonempty(document: Mapping[str, object], name: str) -> str:
    value = str(document.get(name, "")).strip()
    if not value:
        raise ValueError(f"{name} must be a non-empty stable identifier")
    return value


def _lower_hex_sha256(document: Mapping[str, object], name: str) -> str:
    value = str(document.get(name, ""))
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hex digits")
    return value


def _observations(
    values: Sequence[Mapping[str, object]],
) -> tuple[ComputeEnergyEventObservation, ...]:
    result = []
    for index, item in enumerate(values):
        try:
            censored_raw = item.get("window_censored", False)
            if not isinstance(censored_raw, bool):
                raise ValueError("window_censored must be boolean")
            wrap_raw = item.get("counter_wrap_count", 0)
            if isinstance(wrap_raw, bool):
                raise ValueError("counter_wrap_count must be an integer")
            wrap_count = int(wrap_raw)
            if wrap_count != wrap_raw:
                raise ValueError("counter_wrap_count must be an integer")
            result.append(ComputeEnergyEventObservation(
                episode_id=str(item["episode_id"]),
                event_id=str(item["event_id"]),
                counter_start_j=(
                    None if item.get("counter_start_j") is None
                    else float(item["counter_start_j"])),
                counter_end_j=(
                    None if item.get("counter_end_j") is None
                    else float(item["counter_end_j"])),
                counter_resolution_j=(
                    None if item.get("counter_resolution_j") is None
                    else float(item["counter_resolution_j"])),
                counter_wrap_count=wrap_count,
                counter_modulus_j=(
                    None if item.get("counter_modulus_j") is None
                    else float(item["counter_modulus_j"])),
                window_censored=bool(censored_raw),
                counter_difference_uncertainty_j=(
                    None if item.get(
                        "counter_difference_uncertainty_j") is None
                    else float(item[
                        "counter_difference_uncertainty_j"])),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid compute-energy event at index {index}: {exc}"
            ) from exc
    return tuple(result)


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _clopper_pearson_one_sided_upper(
    failures: int, total: int, *, confidence: float = 0.95,
) -> float:
    """Exact binomial one-sided upper confidence bound."""
    k = int(failures)
    n = int(total)
    level = float(confidence)
    if n <= 0 or k < 0 or k > n:
        raise ValueError("invalid failure count for exact binomial bound")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    if k == n:
        return 1.0
    return float(beta_distribution.ppf(level, k + 1, n - k))


def calibrate_document(document: Mapping[str, object]) -> dict[str, object]:
    if int(document.get("schema_version", -1)) != 1:
        raise ValueError("compute-energy log schema_version must equal 1")
    if document.get("meter_scope") != "complete_controller_package":
        raise ValueError(
            "meter_scope must equal complete_controller_package")
    if document.get("counter_semantics") != "monotone_quantized_energy_joule":
        raise ValueError(
            "counter_semantics must equal monotone_quantized_energy_joule")

    source_kind = str(document.get("source_kind", "unspecified"))
    allowed_sources = {
        "unspecified", "synthetic_unit_test", "rapl_package",
        "external_cpu_rail",
    }
    if source_kind not in allowed_sources:
        raise ValueError(f"unsupported source_kind: {source_kind}")
    controller_fingerprint = _lower_hex_sha256(
        document, "controller_implementation_sha256")
    hardware_id = _required_nonempty(document, "hardware_id")
    runtime_id = _required_nonempty(document, "runtime_id")
    meter_id = _required_nonempty(document, "meter_id")
    meter_calibration_id = _required_nonempty(
        document, "meter_calibration_id")
    uncertainty_accounted_raw = document.get(
        "meter_difference_uncertainty_accounted", False)
    if not isinstance(uncertainty_accounted_raw, bool):
        raise ValueError(
            "meter_difference_uncertainty_accounted must be boolean")
    uncertainty_accounted = bool(uncertainty_accounted_raw)

    risk = document.get("risk")
    if not isinstance(risk, Mapping):
        raise ValueError("compute-energy log requires a risk object")
    training_ids = tuple(
        str(value) for value in document.get("training_episode_ids", ()))
    calibration_ids = tuple(
        str(value) for value in document.get("calibration_episode_ids", ()))
    validation_ids = tuple(
        str(value) for value in document.get("validation_episode_ids", ()))
    calibration_values = document.get("calibration_events")
    validation_values = document.get("validation_events", ())
    if not isinstance(calibration_values, Sequence) or isinstance(
        calibration_values, (str, bytes)
    ):
        raise ValueError("calibration_events must be an array")
    if not isinstance(validation_values, Sequence) or isinstance(
        validation_values, (str, bytes)
    ):
        raise ValueError("validation_events must be an array")

    calibration_events = _observations(calibration_values)
    validation_events = _observations(validation_values)
    if source_kind in {"rapl_package", "external_cpu_rail"}:
        if not uncertainty_accounted:
            raise ValueError(
                "hardware energy sources require meter difference-uncertainty "
                "accounting")
        if any(
            item.counter_difference_uncertainty_j is None
            for item in calibration_events + validation_events
            if not item.window_censored
        ):
            raise ValueError(
                "every uncensored hardware window requires "
                "counter_difference_uncertainty_j")

    epoch = calibrate_frozen_compute_energy_epoch(
        calibration_events,
        calibration_episode_ids=calibration_ids,
        training_episode_ids=training_ids,
        miscoverage=float(risk["miscoverage"]),
        epoch_id=str(document["epoch_id"]),
    )
    validation = None
    if validation_ids:
        validation = validate_frozen_compute_energy_epoch(
            epoch,
            validation_events,
            validation_episode_ids=validation_ids,
        )
    elif validation_values:
        raise ValueError(
            "validation events require a frozen validation episode list")

    finite = bool(epoch.finite)
    status = (
        "compute_energy_calibration_unresolved_fail_closed"
        if not finite else (
            "frozen_compute_energy_epoch_validation_complete"
            if validation is not None else
            "frozen_compute_energy_epoch_calibrated_pending_validation"
        )
    )
    empirical_hardware_source = source_kind in {
        "rapl_package", "external_cpu_rail"}
    exact_validation_upper = None
    validation_supports_declared_risk = False
    if validation is not None:
        exact_validation_upper = _clopper_pearson_one_sided_upper(
            validation.failure_count, validation.episode_count)
        validation_supports_declared_risk = bool(
            exact_validation_upper <= epoch.miscoverage + 1.0e-15)
    candidate = bool(
        finite
        and empirical_hardware_source
        and uncertainty_accounted
        and validation is not None
        and validation_supports_declared_risk)
    output: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "epoch": {
            "epoch_id": epoch.epoch_id,
            "exchangeability_unit": "complete_episode",
            "miscoverage": epoch.miscoverage,
            "calibration_episode_count": len(
                epoch.calibration_episode_ids),
            "calibration_episode_ids": list(
                epoch.calibration_episode_ids),
            "training_episode_ids": list(epoch.training_episode_ids),
            "package_energy_episode_scores_j": [
                _finite_or_none(value) for value in epoch.episode_scores_j
            ],
            "package_energy_bound_j": _finite_or_none(
                epoch.package_energy_bound_j),
            "rank_one_based": epoch.rank_one_based,
            "coverage_floor": epoch.coverage_floor,
            "finite": finite,
        },
        "recommended_controller_parameters": {
            "complete_controller_package_energy_bound_j": (
                _finite_or_none(epoch.package_energy_bound_j)),
            "usable_for_shadow_controller": finite,
            "rf_energy_included": False,
            "complete_event_energy_formula": (
                "E_control_upper = E_package_upper + E_RF_current_worst"),
        },
        "validation": None,
        "provenance": {
            "source_kind": source_kind,
            "controller_implementation_sha256": controller_fingerprint,
            "hardware_id": hardware_id,
            "runtime_id": runtime_id,
            "meter_id": meter_id,
            "meter_calibration_id": meter_calibration_id,
            "meter_scope": "complete_controller_package",
            "counter_semantics": "monotone_quantized_energy_joule",
            "counter_difference_error_bound": (
                "one declared counter resolution plus one declared meter "
                "difference-uncertainty bound per complete window"),
            "meter_difference_uncertainty_accounted": uncertainty_accounted,
            "package_energy_measured_once_for_parallel_branches": True,
        },
        "authority": {
            "empirical_hardware_source": empirical_hardware_source,
            "independent_validation_present": validation is not None,
            "validation_supports_declared_risk": (
                validation_supports_declared_risk),
            "compute_energy_certificate_candidate": candidate,
            "system_commit_authority": False,
        },
        "remaining_blockers": [],
    }
    if validation is not None:
        output["validation"] = {
            "episode_count": validation.episode_count,
            "validation_episode_ids": list(
                validation.validation_episode_ids),
            "failure_episode_ids": list(validation.failure_episode_ids),
            "failure_count": validation.failure_count,
            "failure_rate": (
                validation.failure_count / validation.episode_count),
            "failure_rate_wilson95": _wilson(
                validation.failure_count, validation.episode_count),
            "failure_rate_clopper_pearson_one_sided95_upper": (
                exact_validation_upper),
            "supports_declared_miscoverage_at_one_sided95": (
                validation_supports_declared_risk),
            "events_are_not_counted_as_independent_trials": True,
        }

    blockers: list[str] = []
    if not finite:
        blockers.append(
            "finite-sample quantile is unresolved or contains a censored "
            "calibration episode")
    if validation is None:
        blockers.append("no disjoint episode-level validation split")
    elif not validation_supports_declared_risk:
        blockers.append(
            "independent validation one-sided 95% exact failure-rate upper "
            "bound exceeds the declared miscoverage")
    if not empirical_hardware_source:
        blockers.append("source is not a measured hardware energy counter")
    elif not uncertainty_accounted:
        blockers.append(
            "hardware meter difference uncertainty is not accounted")
    blockers.extend((
        "RF transmit energy must be added without double counting",
        "DVFS, thermal state, workload and runtime identity must remain in epoch",
        "system-level energy cap and live fail-closed meter path are not verified",
    ))
    output["remaining_blockers"] = blockers
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = json.loads(args.input.read_text(encoding="utf-8"))
    result = calibrate_document(document)
    result["provenance"]["input"] = {
        "path": str(args.input),
        "sha256": _sha256(args.input),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
