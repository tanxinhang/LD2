from tools.calibrate_link_reliability_epoch import calibrate_document


def _document(count=19, *, with_validation=True, validation_count=2):
    calibration_ids = [f"cal-{index}" for index in range(count)]
    validation_ids = (
        [f"val-{index}" for index in range(validation_count)]
        if with_validation else [])
    return {
        "schema_version": 1,
        "epoch_id": "synthetic-link-epoch",
        "source_kind": "synthetic_unit_test",
        "protocol_implementation_sha256": "a" * 64,
        "instrumentation_id": "unit-test-observed-snr-model-v1",
        "training_episode_ids": ["train-0"],
        "calibration_episode_ids": calibration_ids,
        "validation_episode_ids": validation_ids,
        "risk": {
            "total_miscoverage": 0.10,
            "erasure_miscoverage": 0.05,
            "queue_miscoverage": 0.05,
        },
        "calibration_events": [
            {
                "episode_id": episode,
                "event_id": "0",
                "erased_copies_before_success": index % 2,
                "observed_complete_protocol_latency_s": 0.02 + index * 1e-5,
                "observed_snr_modeled_protocol_latency_s": 0.02,
                "delivery_censored": False,
            }
            for index, episode in enumerate(calibration_ids)
        ],
        "validation_events": [
            {
                "episode_id": episode,
                "event_id": "0",
                "erased_copies_before_success": index % 2,
                "observed_complete_protocol_latency_s": 0.0201,
                "observed_snr_modeled_protocol_latency_s": 0.02,
                "delivery_censored": False,
            }
            for index, episode in enumerate(validation_ids)
        ],
    }


def test_tool_emits_frozen_finite_shadow_parameters_and_validation():
    result = calibrate_document(_document())
    assert result["status"] == (
        "frozen_link_reliability_epoch_validation_complete")
    assert result["epoch"]["finite"]
    assert result["recommended_controller_parameters"][
        "horizon_network_repetition_count"] == 2
    assert result["recommended_controller_parameters"][
        "usable_for_shadow_controller"]
    assert result["validation"]["episode_count"] == 2
    assert result["validation"][
        "events_are_not_counted_as_independent_trials"]
    assert result["validation"][
        "joint_failure_rate_clopper_pearson_one_sided95_upper"] > 0.10
    assert not result["authority"]["validation_supports_declared_risk"]
    assert not result["authority"]["link_certificate_candidate"]
    assert not result["authority"]["system_commit_authority"]


def test_tool_serializes_unresolved_quantiles_as_null_parameters():
    result = calibrate_document(_document(count=18, with_validation=False))
    assert result["status"] == (
        "link_reliability_calibration_unresolved_fail_closed")
    assert result["epoch"]["erasure_bound"] is None
    assert result["epoch"]["queue_bound_s"] is None
    assert result["recommended_controller_parameters"][
        "horizon_network_repetition_count"] is None
    assert not result["recommended_controller_parameters"][
        "usable_for_shadow_controller"]
    assert result["validation"] is None


def test_hardware_link_candidate_requires_exact_validation_support():
    document = _document(validation_count=29)
    document["source_kind"] = "hardware_u2u"
    result = calibrate_document(document)
    assert result["validation"]["joint_failure_count"] == 0
    assert result["validation"][
        "joint_failure_rate_clopper_pearson_one_sided95_upper"] <= 0.10
    assert result["authority"]["validation_supports_declared_risk"]
    assert result["authority"]["link_certificate_candidate"]


def test_tool_rejects_missing_or_malformed_protocol_fingerprint():
    missing = _document()
    del missing["protocol_implementation_sha256"]
    try:
        calibrate_document(missing)
    except ValueError as exc:
        assert "protocol_implementation_sha256" in str(exc)
    else:
        raise AssertionError("missing protocol fingerprint was accepted")

    malformed = _document()
    malformed["protocol_implementation_sha256"] = "A" * 64
    try:
        calibrate_document(malformed)
    except ValueError as exc:
        assert "lowercase hex" in str(exc)
    else:
        raise AssertionError("malformed protocol fingerprint was accepted")
