import copy

import pytest

from tests.test_calibrate_runtime_latency_epoch_tool import _document
from tools.calibrate_runtime_latency_epoch import calibrate_document
from uav_isac.evaluation.runtime_latency_epoch_wiring import (
    bind_runtime_latency_epoch,
)


def _artifact(*, source="deployment_monotonic_clock"):
    document = _document(source=source)
    if source == "deployment_monotonic_clock":
        validation_ids = [f"val-supported-{index}" for index in range(80)]
        document["validation_episode_ids"] = validation_ids
        document["validation_events"] = [
            {
                "episode_id": episode, "event_id": "0",
                "observed_complete_compute_latency_s": 0.02,
                "timer_resolution_s": 1.0e-6,
                "timer_difference_uncertainty_s": 2.0e-6,
                "window_censored": False,
            }
            for episode in validation_ids
        ]
    return calibrate_document(document)


def test_runtime_binding_accepts_exact_identity_and_isolated_splits():
    bound = bind_runtime_latency_epoch(
        _artifact(), current_controller_sha256="d" * 64,
        controller_episode_ids=("audit-0",))
    assert bound.complete_compute_latency_bound_s == pytest.approx(0.028003)
    assert bound.runtime_latency_certificate_candidate


def test_runtime_binding_rejects_stale_leaky_and_network_inclusive_artifacts():
    artifact = _artifact()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        bind_runtime_latency_epoch(
            artifact, current_controller_sha256="e" * 64,
            controller_episode_ids=("audit-0",))
    with pytest.raises(ValueError, match="evaluation episode leakage"):
        bind_runtime_latency_epoch(
            artifact, current_controller_sha256="d" * 64,
            controller_episode_ids=("cal-0",))
    inclusive = copy.deepcopy(artifact)
    inclusive["recommended_controller_parameters"][
        "network_latency_included"] = True
    with pytest.raises(ValueError, match="exclude network latency"):
        bind_runtime_latency_epoch(
            inclusive, current_controller_sha256="d" * 64,
            controller_episode_ids=("audit-0",))
    inconsistent = copy.deepcopy(artifact)
    inconsistent["validation"]["failure_count"] = 1
    with pytest.raises(ValueError, match="counts are inconsistent"):
        bind_runtime_latency_epoch(
            inconsistent, current_controller_sha256="d" * 64,
            controller_episode_ids=("audit-0",))


def test_workstation_epoch_binds_but_is_not_deployment_candidate():
    bound = bind_runtime_latency_epoch(
        _artifact(source="workstation_wall_clock"),
        current_controller_sha256="d" * 64,
        controller_episode_ids=("audit-0",))
    assert not bound.runtime_latency_certificate_candidate
