"""Calibrate and validate an episode-level U2U reliability epoch from JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.summarize_paired_horizon_confirmation import _sha256, _wilson
from uav_isac.evaluation.link_reliability_calibration import (
    LinkReliabilityEventObservation,
    calibrate_frozen_link_reliability_epoch,
    validate_frozen_link_reliability_epoch,
)
from uav_isac.evaluation.finite_sample_feasibility import (
    clopper_pearson_one_sided_upper,
)


def _observations(
    values: Sequence[Mapping[str, object]],
) -> tuple[LinkReliabilityEventObservation, ...]:
    result = []
    for index, item in enumerate(values):
        try:
            censored_raw = item.get("delivery_censored", False)
            if not isinstance(censored_raw, bool):
                raise ValueError("delivery_censored must be boolean")
            censored = bool(censored_raw)
            result.append(LinkReliabilityEventObservation(
                episode_id=str(item["episode_id"]),
                event_id=str(item["event_id"]),
                erased_copies_before_success=item[
                    "erased_copies_before_success"],
                observed_complete_protocol_latency_s=(
                    None if item.get(
                        "observed_complete_protocol_latency_s") is None
                    else float(item[
                        "observed_complete_protocol_latency_s"])),
                observed_snr_modeled_protocol_latency_s=(
                    None if item.get(
                        "observed_snr_modeled_protocol_latency_s") is None
                    else float(item[
                        "observed_snr_modeled_protocol_latency_s"])),
                delivery_censored=censored,
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid link event at index {index}: {exc}") from exc
    return tuple(result)


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def calibrate_document(document: Mapping[str, object]) -> dict[str, object]:
    if int(document.get("schema_version", -1)) != 1:
        raise ValueError("link log schema_version must equal 1")
    risk = document.get("risk")
    if not isinstance(risk, Mapping):
        raise ValueError("link log requires a risk object")
    calibration_ids = tuple(
        str(value) for value in document.get("calibration_episode_ids", ()))
    validation_ids = tuple(
        str(value) for value in document.get("validation_episode_ids", ()))
    training_ids = tuple(
        str(value) for value in document.get("training_episode_ids", ()))
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
    epoch = calibrate_frozen_link_reliability_epoch(
        calibration_events,
        calibration_episode_ids=calibration_ids,
        training_episode_ids=training_ids,
        total_miscoverage=float(risk["total_miscoverage"]),
        erasure_miscoverage=float(risk["erasure_miscoverage"]),
        queue_miscoverage=float(risk["queue_miscoverage"]),
        epoch_id=str(document["epoch_id"]),
    )
    validation = None
    if validation_ids:
        validation = validate_frozen_link_reliability_epoch(
            epoch,
            validation_events,
            validation_episode_ids=validation_ids,
        )
    elif validation_events:
        raise ValueError(
            "validation events require a frozen validation episode list")
    source_kind = str(document.get("source_kind", "unspecified"))
    allowed_sources = {
        "unspecified", "synthetic_unit_test", "emulated_network",
        "hardware_u2u",
    }
    if source_kind not in allowed_sources:
        raise ValueError(f"unsupported source_kind: {source_kind}")
    protocol_fingerprint = str(document.get(
        "protocol_implementation_sha256", ""))
    if (
        len(protocol_fingerprint) != 64
        or any(character not in "0123456789abcdef"
               for character in protocol_fingerprint)
    ):
        raise ValueError(
            "protocol_implementation_sha256 must be 64 lowercase hex digits")
    finite = bool(epoch.finite)
    exact_validation_upper = None
    validation_supports_declared_risk = False
    if validation is not None:
        exact_validation_upper = clopper_pearson_one_sided_upper(
            validation.joint_failure_count, validation.episode_count)
        validation_supports_declared_risk = bool(
            exact_validation_upper
            <= epoch.total_miscoverage + 1.0e-15)
    status = (
        "link_reliability_calibration_unresolved_fail_closed"
        if not finite else (
            "frozen_link_reliability_epoch_validation_complete"
            if validation is not None else
            "frozen_link_reliability_epoch_calibrated_pending_validation"
        )
    )
    output: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "epoch": {
            "epoch_id": epoch.epoch_id,
            "exchangeability_unit": "complete_episode",
            "total_miscoverage": epoch.total_miscoverage,
            "erasure_miscoverage": epoch.erasure_miscoverage,
            "queue_miscoverage": epoch.queue_miscoverage,
            "allocated_miscoverage": epoch.allocated_miscoverage,
            "joint_coverage_floor_union_bound": epoch.joint_coverage_floor,
            "calibration_episode_count": len(
                epoch.calibration_episode_ids),
            "calibration_episode_ids": list(
                epoch.calibration_episode_ids),
            "training_episode_ids": list(epoch.training_episode_ids),
            "erasure_episode_scores": [
                _finite_or_none(value)
                for value in epoch.erasure_episode_scores
            ],
            "queue_episode_scores_s": [
                _finite_or_none(value)
                for value in epoch.queue_episode_scores_s
            ],
            "erasure_bound": _finite_or_none(epoch.erasure_bound),
            "queue_bound_s": _finite_or_none(epoch.queue_bound_s),
            "erasure_rank_one_based": epoch.erasure_rank_one_based,
            "queue_rank_one_based": epoch.queue_rank_one_based,
            "erasure_coverage_floor": epoch.erasure_coverage_floor,
            "queue_coverage_floor": epoch.queue_coverage_floor,
            "finite": finite,
        },
        "recommended_controller_parameters": {
            "horizon_network_repetition_count": (
                epoch.recommended_repetition_count),
            "horizon_network_excess_queue_bound_s": (
                _finite_or_none(epoch.queue_bound_s)),
            "usable_for_shadow_controller": finite,
        },
        "validation": None,
        "provenance": {
            "source_kind": source_kind,
            "protocol_implementation_sha256": protocol_fingerprint,
            "observed_snr_serialization_removed_before_queue_score": True,
            "instrumentation_id": document.get("instrumentation_id"),
        },
        "authority": {
            "empirical_hardware_source": source_kind == "hardware_u2u",
            "independent_validation_present": validation is not None,
            "validation_supports_declared_risk": (
                validation_supports_declared_risk),
            "link_certificate_candidate": bool(
                finite
                and source_kind == "hardware_u2u"
                and validation is not None
                and validation_supports_declared_risk
            ),
            "system_commit_authority": False,
        },
        "remaining_blockers": [],
    }
    if validation is not None:
        output["validation"] = {
            "episode_count": validation.episode_count,
            "validation_episode_ids": list(
                validation.validation_episode_ids),
            "erasure_failure_episode_ids": list(
                validation.erasure_failure_episode_ids),
            "queue_failure_episode_ids": list(
                validation.queue_failure_episode_ids),
            "joint_failure_episode_ids": list(
                validation.joint_failure_episode_ids),
            "joint_failure_count": validation.joint_failure_count,
            "joint_failure_rate": (
                validation.joint_failure_count / validation.episode_count),
            "joint_failure_rate_wilson95": _wilson(
                validation.joint_failure_count, validation.episode_count),
            "joint_failure_rate_clopper_pearson_one_sided95_upper": (
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
            "link validation exact upper exceeds declared total link risk")
    if source_kind != "hardware_u2u":
        blockers.append("source is not measured hardware U2U traffic")
    blockers.extend((
        "compute/package energy is not included",
        "live duplicate suppression and packet parser are not verified",
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
