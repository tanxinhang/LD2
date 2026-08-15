"""Calibrate a frozen complete-controller compute-latency epoch from JSON."""

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
from uav_isac.evaluation.runtime_latency_calibration import (
    RuntimeLatencyEventObservation,
    calibrate_frozen_runtime_latency_epoch,
    validate_frozen_runtime_latency_epoch,
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
) -> tuple[RuntimeLatencyEventObservation, ...]:
    result = []
    for index, item in enumerate(values):
        try:
            censored_raw = item.get("window_censored", False)
            if not isinstance(censored_raw, bool):
                raise ValueError("window_censored must be boolean")
            result.append(RuntimeLatencyEventObservation(
                episode_id=str(item["episode_id"]),
                event_id=str(item["event_id"]),
                observed_complete_compute_latency_s=(
                    None if item.get(
                        "observed_complete_compute_latency_s") is None
                    else float(item[
                        "observed_complete_compute_latency_s"])),
                timer_resolution_s=(
                    None if item.get("timer_resolution_s") is None
                    else float(item["timer_resolution_s"])),
                timer_difference_uncertainty_s=(
                    None if item.get(
                        "timer_difference_uncertainty_s") is None
                    else float(item[
                        "timer_difference_uncertainty_s"])),
                window_censored=bool(censored_raw),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid runtime event at index {index}: {exc}") from exc
    return tuple(result)


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _exact_upper(failures: int, total: int) -> float:
    if total <= 0 or failures < 0 or failures > total:
        raise ValueError("invalid runtime validation failure count")
    if failures == total:
        return 1.0
    return float(beta_distribution.ppf(
        0.95, failures + 1, total - failures))


def calibrate_document(document: Mapping[str, object]) -> dict[str, object]:
    if int(document.get("schema_version", -1)) != 1:
        raise ValueError("runtime log schema_version must equal 1")
    if document.get("measurement_scope") != (
        "complete_parallel_controller_compute_critical_path_excluding_network"
    ):
        raise ValueError("runtime measurement_scope is not the complete compute path")
    if document.get("clock_semantics") != (
        "monotonic_wall_clock_includes_preemption"
    ):
        raise ValueError("runtime clock must be monotonic wall time")

    source_kind = str(document.get("source_kind", "unspecified"))
    allowed = {
        "unspecified", "synthetic_unit_test", "workstation_wall_clock",
        "deployment_monotonic_clock",
    }
    if source_kind not in allowed:
        raise ValueError(f"unsupported source_kind: {source_kind}")
    fingerprint = _lower_hex_sha256(
        document, "controller_implementation_sha256")
    hardware_id = _required_nonempty(document, "hardware_id")
    runtime_id = _required_nonempty(document, "runtime_id")
    clock_id = _required_nonempty(document, "clock_id")
    clock_calibration_id = _required_nonempty(document, "clock_calibration_id")
    uncertainty_raw = document.get(
        "timer_difference_uncertainty_accounted", False)
    if not isinstance(uncertainty_raw, bool):
        raise ValueError(
            "timer_difference_uncertainty_accounted must be boolean")
    uncertainty_accounted = bool(uncertainty_raw)

    risk = document.get("risk")
    if not isinstance(risk, Mapping):
        raise ValueError("runtime log requires a risk object")
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
    if source_kind == "deployment_monotonic_clock":
        if not uncertainty_accounted:
            raise ValueError(
                "deployment timing requires timer difference-uncertainty accounting")
        if any(
            item.timer_difference_uncertainty_s is None
            for item in calibration_events + validation_events
            if not item.window_censored
        ):
            raise ValueError(
                "every deployment runtime window requires timer uncertainty")

    epoch = calibrate_frozen_runtime_latency_epoch(
        calibration_events,
        calibration_episode_ids=calibration_ids,
        training_episode_ids=training_ids,
        miscoverage=float(risk["miscoverage"]),
        epoch_id=str(document["epoch_id"]),
    )
    validation = None
    if validation_ids:
        validation = validate_frozen_runtime_latency_epoch(
            epoch, validation_events,
            validation_episode_ids=validation_ids)
    elif validation_events:
        raise ValueError(
            "validation events require a frozen validation episode list")

    finite = bool(epoch.finite)
    exact_upper = None
    validation_supports_risk = False
    if validation is not None:
        exact_upper = _exact_upper(
            validation.failure_count, validation.episode_count)
        validation_supports_risk = bool(
            exact_upper <= epoch.miscoverage + 1.0e-15)
    deployment_source = source_kind == "deployment_monotonic_clock"
    candidate = bool(
        finite and deployment_source and uncertainty_accounted
        and validation is not None and validation_supports_risk)
    status = (
        "runtime_latency_calibration_unresolved_fail_closed"
        if not finite else (
            "frozen_runtime_latency_epoch_validation_complete"
            if validation is not None else
            "frozen_runtime_latency_epoch_calibrated_pending_validation"))
    output: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "epoch": {
            "epoch_id": epoch.epoch_id,
            "exchangeability_unit": "complete_episode",
            "miscoverage": epoch.miscoverage,
            "calibration_episode_count": len(epoch.calibration_episode_ids),
            "calibration_episode_ids": list(epoch.calibration_episode_ids),
            "training_episode_ids": list(epoch.training_episode_ids),
            "complete_compute_episode_scores_s": [
                _finite_or_none(value) for value in epoch.episode_scores_s],
            "complete_compute_latency_bound_s": _finite_or_none(
                epoch.complete_compute_latency_bound_s),
            "rank_one_based": epoch.rank_one_based,
            "coverage_floor": epoch.coverage_floor,
            "finite": finite,
        },
        "recommended_controller_parameters": {
            "complete_controller_compute_latency_bound_s": _finite_or_none(
                epoch.complete_compute_latency_bound_s),
            "usable_for_shadow_controller": finite,
            "network_latency_included": False,
            "deadline_formula": (
                "T_upper = T_complete_compute_upper + "
                "T_repeated_network_upper + J_link_upper"),
        },
        "validation": None,
        "provenance": {
            "source_kind": source_kind,
            "controller_implementation_sha256": fingerprint,
            "hardware_id": hardware_id,
            "runtime_id": runtime_id,
            "clock_id": clock_id,
            "clock_calibration_id": clock_calibration_id,
            "measurement_scope": document["measurement_scope"],
            "clock_semantics": document["clock_semantics"],
            "timer_difference_uncertainty_accounted": uncertainty_accounted,
            "complete_compute_path_measured_once_for_parallel_branches": True,
        },
        "authority": {
            "deployment_timing_source": deployment_source,
            "independent_validation_present": validation is not None,
            "validation_supports_declared_risk": validation_supports_risk,
            "runtime_latency_certificate_candidate": candidate,
            "system_commit_authority": False,
        },
        "remaining_blockers": [],
    }
    if validation is not None:
        output["validation"] = {
            "episode_count": validation.episode_count,
            "validation_episode_ids": list(validation.validation_episode_ids),
            "failure_episode_ids": list(validation.failure_episode_ids),
            "failure_count": validation.failure_count,
            "failure_rate": validation.failure_count / validation.episode_count,
            "failure_rate_wilson95": _wilson(
                validation.failure_count, validation.episode_count),
            "failure_rate_clopper_pearson_one_sided95_upper": exact_upper,
            "supports_declared_miscoverage_at_one_sided95": (
                validation_supports_risk),
            "events_are_not_counted_as_independent_trials": True,
        }
    blockers = []
    if not finite:
        blockers.append(
            "finite-sample runtime quantile is unresolved or censored")
    if validation is None:
        blockers.append("no disjoint episode-level runtime validation split")
    elif not validation_supports_risk:
        blockers.append(
            "runtime validation exact upper exceeds declared miscoverage")
    if not deployment_source:
        blockers.append("source is not deployment monotonic wall-clock timing")
    blockers.extend((
        "network repetition/queue latency must be added without double counting",
        "live deadline fail-closed path remains to be hardware verified",
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
        "path": str(args.input), "sha256": _sha256(args.input)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
