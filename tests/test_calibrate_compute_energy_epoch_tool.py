import pytest

from tools.calibrate_compute_energy_epoch import calibrate_document


def _document(count=19, *, with_validation=True, source="synthetic_unit_test"):
    calibration_ids = [f"cal-{index}" for index in range(count)]
    validation_ids = ["val-0", "val-1"] if with_validation else []
    return {
        "schema_version": 1,
        "epoch_id": "synthetic-compute-energy-epoch",
        "source_kind": source,
        "controller_implementation_sha256": "b" * 64,
        "hardware_id": "synthetic-cpu-v1",
        "runtime_id": "python-test-runtime-v1",
        "meter_id": "synthetic-counter-v1",
        "meter_calibration_id": "synthetic-calibration-none",
        "meter_difference_uncertainty_accounted": source in {
            "rapl_package", "external_cpu_rail"},
        "meter_scope": "complete_controller_package",
        "counter_semantics": "monotone_quantized_energy_joule",
        "training_episode_ids": ["train-0"],
        "calibration_episode_ids": calibration_ids,
        "validation_episode_ids": validation_ids,
        "risk": {"miscoverage": 0.05},
        "calibration_events": [
            {
                "episode_id": episode,
                "event_id": "0",
                "counter_start_j": 1.0,
                "counter_end_j": 1.1 + index * 0.001,
                "counter_resolution_j": 0.0001,
                "counter_wrap_count": 0,
                "window_censored": False,
                "counter_difference_uncertainty_j": (
                    0.0002 if source in {
                        "rapl_package", "external_cpu_rail"} else None),
            }
            for index, episode in enumerate(calibration_ids)
        ],
        "validation_events": [
            {
                "episode_id": episode,
                "event_id": "0",
                "counter_start_j": 2.0,
                "counter_end_j": 2.11 + index * 0.02,
                "counter_resolution_j": 0.0001,
                "counter_wrap_count": 0,
                "window_censored": False,
                "counter_difference_uncertainty_j": (
                    0.0002 if source in {
                        "rapl_package", "external_cpu_rail"} else None),
            }
            for index, episode in enumerate(validation_ids)
        ],
    }


def test_tool_emits_frozen_package_bound_and_independent_validation():
    result = calibrate_document(_document())
    assert result["status"] == (
        "frozen_compute_energy_epoch_validation_complete")
    assert result["epoch"]["finite"]
    assert result["recommended_controller_parameters"][
        "complete_controller_package_energy_bound_j"] == pytest.approx(
            0.1181)
    assert not result["recommended_controller_parameters"][
        "rf_energy_included"]
    assert result["validation"]["episode_count"] == 2
    assert result["validation"][
        "events_are_not_counted_as_independent_trials"]
    assert not result["validation"][
        "supports_declared_miscoverage_at_one_sided95"]
    assert not result["authority"]["compute_energy_certificate_candidate"]
    assert not result["authority"]["system_commit_authority"]


def test_tool_serializes_unresolved_energy_as_null():
    result = calibrate_document(_document(count=18, with_validation=False))
    assert result["status"] == (
        "compute_energy_calibration_unresolved_fail_closed")
    assert result["epoch"]["package_energy_bound_j"] is None
    assert result["recommended_controller_parameters"][
        "complete_controller_package_energy_bound_j"] is None
    assert not result["recommended_controller_parameters"][
        "usable_for_shadow_controller"]


def test_tool_requires_meter_domain_and_bound_identities():
    for field in (
        "hardware_id", "runtime_id", "meter_id", "meter_calibration_id",
    ):
        document = _document()
        document[field] = ""
        with pytest.raises(ValueError, match=field):
            calibrate_document(document)
    document = _document()
    document["meter_scope"] = "one_branch_only"
    with pytest.raises(ValueError, match="meter_scope"):
        calibrate_document(document)
    document = _document()
    document["controller_implementation_sha256"] = "B" * 64
    with pytest.raises(ValueError, match="lowercase hex"):
        calibrate_document(document)


def test_hardware_source_without_statistical_validation_support_is_not_candidate():
    result = calibrate_document(_document(source="rapl_package"))
    assert not result["authority"]["compute_energy_certificate_candidate"]
    assert not result["authority"]["system_commit_authority"]


def test_hardware_candidate_requires_exact_validation_upper_below_risk():
    document = _document(source="rapl_package")
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
    result = calibrate_document(document)
    assert result["validation"]["failure_count"] == 0
    assert result["validation"][
        "failure_rate_clopper_pearson_one_sided95_upper"] < 0.05
    assert result["authority"]["compute_energy_certificate_candidate"]
    assert not result["authority"]["system_commit_authority"]


def test_hardware_source_requires_meter_uncertainty_accounting():
    document = _document(source="rapl_package")
    document["meter_difference_uncertainty_accounted"] = False
    with pytest.raises(ValueError, match="difference-uncertainty"):
        calibrate_document(document)
