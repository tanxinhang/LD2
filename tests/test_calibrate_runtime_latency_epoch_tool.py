import pytest

from tools.calibrate_runtime_latency_epoch import calibrate_document


def _document(count=19, *, with_validation=True, source="synthetic_unit_test"):
    calibration_ids = [f"cal-{index}" for index in range(count)]
    validation_ids = ["val-0", "val-1"] if with_validation else []
    deployment = source == "deployment_monotonic_clock"
    return {
        "schema_version": 1,
        "epoch_id": "synthetic-runtime-epoch",
        "source_kind": source,
        "controller_implementation_sha256": "d" * 64,
        "hardware_id": "synthetic-cpu-v1",
        "runtime_id": "python-test-runtime-v1",
        "clock_id": "perf-counter-test-v1",
        "clock_calibration_id": "synthetic-clock-calibration-none",
        "measurement_scope": (
            "complete_parallel_controller_compute_critical_path_excluding_network"),
        "clock_semantics": "monotonic_wall_clock_includes_preemption",
        "timer_difference_uncertainty_accounted": deployment,
        "training_episode_ids": ["train-0"],
        "calibration_episode_ids": calibration_ids,
        "validation_episode_ids": validation_ids,
        "risk": {"miscoverage": 0.05},
        "calibration_events": [
            {
                "episode_id": episode, "event_id": "0",
                "observed_complete_compute_latency_s": 0.01 + index * 0.001,
                "timer_resolution_s": 1.0e-6,
                "timer_difference_uncertainty_s": (
                    2.0e-6 if deployment else None),
                "window_censored": False,
            }
            for index, episode in enumerate(calibration_ids)
        ],
        "validation_events": [
            {
                "episode_id": episode, "event_id": "0",
                "observed_complete_compute_latency_s": 0.02 + index * 0.02,
                "timer_resolution_s": 1.0e-6,
                "timer_difference_uncertainty_s": (
                    2.0e-6 if deployment else None),
                "window_censored": False,
            }
            for index, episode in enumerate(validation_ids)
        ],
    }


def test_runtime_tool_emits_bound_but_synthetic_is_not_candidate():
    result = calibrate_document(_document())
    assert result["status"] == "frozen_runtime_latency_epoch_validation_complete"
    assert result["epoch"]["complete_compute_latency_bound_s"] == pytest.approx(
        0.028001)
    assert result["validation"]["failure_count"] == 1
    assert not result["authority"]["runtime_latency_certificate_candidate"]
    assert not result["authority"]["system_commit_authority"]


def test_runtime_tool_unresolved_quantile_is_json_null():
    result = calibrate_document(_document(count=18, with_validation=False))
    assert result["status"] == "runtime_latency_calibration_unresolved_fail_closed"
    assert result["epoch"]["complete_compute_latency_bound_s"] is None
    assert result["recommended_controller_parameters"][
        "complete_controller_compute_latency_bound_s"] is None


def test_deployment_candidate_requires_exact_validation_support():
    document = _document(source="deployment_monotonic_clock")
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
    result = calibrate_document(document)
    assert result["validation"]["failure_count"] == 0
    assert result["validation"][
        "failure_rate_clopper_pearson_one_sided95_upper"] < 0.05
    assert result["authority"]["runtime_latency_certificate_candidate"]
    assert not result["authority"]["system_commit_authority"]


def test_deployment_timing_requires_uncertainty_and_scope_identity():
    document = _document(source="deployment_monotonic_clock")
    document["timer_difference_uncertainty_accounted"] = False
    with pytest.raises(ValueError, match="difference-uncertainty"):
        calibrate_document(document)
    document = _document()
    document["measurement_scope"] = "one_branch_only"
    with pytest.raises(ValueError, match="measurement_scope"):
        calibrate_document(document)
    document = _document()
    document["controller_implementation_sha256"] = "D" * 64
    with pytest.raises(ValueError, match="lowercase hex"):
        calibrate_document(document)
