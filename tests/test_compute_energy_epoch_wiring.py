import copy

import pytest

from tools.calibrate_compute_energy_epoch import calibrate_document
from tests.test_calibrate_compute_energy_epoch_tool import _document
from uav_isac.evaluation.compute_energy_epoch_wiring import (
    bind_compute_energy_epoch,
)


def _artifact(*, source="external_cpu_rail"):
    document = _document(source=source)
    if source == "external_cpu_rail":
        validation_ids = [f"val-supported-{index}" for index in range(80)]
        document["validation_episode_ids"] = validation_ids
        document["validation_events"] = [
            {
                "episode_id": episode,
                "event_id": "0",
                "counter_start_j": 2.0,
                "counter_end_j": 2.11,
                "counter_resolution_j": 0.0001,
                "counter_wrap_count": 0,
                "window_censored": False,
                "counter_difference_uncertainty_j": 0.0002,
            }
            for episode in validation_ids
        ]
    return calibrate_document(document)


def test_energy_epoch_binding_accepts_exact_fingerprint_and_isolated_splits():
    artifact = _artifact()
    bound = bind_compute_energy_epoch(
        artifact,
        current_controller_sha256="b" * 64,
        controller_episode_ids=("audit-0", "audit-1"),
    )
    assert bound.package_energy_bound_j == pytest.approx(0.1183)
    assert bound.energy_certificate_candidate
    assert bound.metadata["hardware_id"] == "synthetic-cpu-v1"
    assert bound.metadata["meter_difference_uncertainty_accounted"]


def test_energy_epoch_binding_rejects_stale_code_leakage_and_rf_double_count():
    artifact = _artifact()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        bind_compute_energy_epoch(
            artifact,
            current_controller_sha256="c" * 64,
            controller_episode_ids=("audit-0",),
        )
    with pytest.raises(ValueError, match="evaluation episode leakage"):
        bind_compute_energy_epoch(
            artifact,
            current_controller_sha256="b" * 64,
            controller_episode_ids=("cal-0",),
        )
    includes_rf = copy.deepcopy(artifact)
    includes_rf["recommended_controller_parameters"][
        "rf_energy_included"] = True
    with pytest.raises(ValueError, match="exclude RF energy"):
        bind_compute_energy_epoch(
            includes_rf,
            current_controller_sha256="b" * 64,
            controller_episode_ids=("audit-0",),
        )
    inconsistent = copy.deepcopy(artifact)
    inconsistent["validation"]["failure_count"] = 1
    with pytest.raises(ValueError, match="counts are inconsistent"):
        bind_compute_energy_epoch(
            inconsistent,
            current_controller_sha256="b" * 64,
            controller_episode_ids=("audit-0",),
        )


def test_synthetic_energy_epoch_binds_but_is_not_a_certificate_candidate():
    bound = bind_compute_energy_epoch(
        _artifact(source="synthetic_unit_test"),
        current_controller_sha256="b" * 64,
        controller_episode_ids=("audit-0",),
    )
    assert not bound.energy_certificate_candidate
