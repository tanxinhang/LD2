import numpy as np
import pytest

from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.evaluation.channel_margin import (
    calibrate_frozen_joint_safety_epoch,
    calibrate_frozen_channel_margin_epoch,
    event_joint_channel_underestimate_score,
    event_joint_transport_residual_score,
)


def _model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 8],
        header_bits=64,
        bandwidth_hz=1.0e6,
        deadline_s=0.02,
        processing_delay_s=2.0e-4,
        snr_threshold_db=-20.0,
        antenna_gain_dbi=0.0,
        carrier_hz=2.4e9,
        tx_power_w=0.2,
        kT=4.0e-21,
        noise_figure_db=5.0,
        dt=0.1,
    )


def test_robust_link_budget_recomputes_shannon_rate_and_latency():
    model = _model()
    sender = np.asarray([0.0, 0.0, 100.0])
    receiver = np.asarray([100.0, 0.0, 100.0])
    nominal = model.link_budget(sender, receiver, 1000, 1.0e6, 0.1)
    zero_margin = model.robust_link_budget(
        sender, receiver, 1000, 1.0e6, 0.1)
    np.testing.assert_allclose(zero_margin, nominal, rtol=1e-14, atol=1e-14)

    robust = model.robust_link_budget(
        sender,
        receiver,
        1000,
        1.0e6,
        0.1,
        snr_margin_db=6.0,
        latency_margin_s=5.0e-4,
    )
    assert robust[0] == pytest.approx(nominal[0] - 6.0)
    assert robust[1] < nominal[1]
    assert robust[2] > nominal[2]
    assert robust[3] > nominal[3] + 5.0e-4


def test_joint_channel_score_is_one_maximum_per_event():
    predicted_snr = np.asarray([10.0, 10.0, 8.0])
    observed_snr = np.asarray([9.0, 11.0, 8.0])
    predicted_latency = np.asarray([1.0e-3, 1.0e-3, 1.0e-3])
    observed_latency = np.asarray([1.0e-3, 1.5e-3, 0.5e-3])
    score = event_joint_channel_underestimate_score(
        predicted_snr,
        observed_snr,
        predicted_latency,
        observed_latency,
        snr_resolution_db=1.0,
        latency_resolution_s=2.5e-4,
    )
    assert score == pytest.approx(2.0)

    # Repeating dependent packets within this event cannot create more
    # calibration observations or change its event-level maximum.
    repeated = event_joint_channel_underestimate_score(
        np.tile(predicted_snr, 100),
        np.tile(observed_snr, 100),
        np.tile(predicted_latency, 100),
        np.tile(observed_latency, 100),
        snr_resolution_db=1.0,
        latency_resolution_s=2.5e-4,
    )
    assert repeated == score


def test_transport_score_separates_snr_serialization_from_excess_delay():
    predicted_snr = np.asarray([10.0])
    observed_snr = np.asarray([7.0])
    bits = np.asarray([1000.0])
    bandwidth = np.asarray([1.0e6])
    processing = 2.0e-4
    observed_rate = bandwidth * np.log2(
        1.0 + 10.0 ** (observed_snr / 10.0))
    observed_latency = bits / observed_rate + processing + 3.0e-4
    score = event_joint_transport_residual_score(
        predicted_snr,
        observed_snr,
        observed_latency,
        bits,
        bandwidth,
        processing_delay_s=processing,
        snr_resolution_db=1.0,
        excess_latency_resolution_s=1.0e-4,
    )
    assert score == pytest.approx(3.0)

    robust = _model().robust_link_budget(
        np.asarray([0.0, 0.0, 100.0]),
        np.asarray([100.0, 0.0, 100.0]),
        1000,
        1.0e6,
        0.1,
        snr_margin_db=3.0,
        latency_margin_s=3.0e-4,
    )
    nominal = _model().link_budget(
        np.asarray([0.0, 0.0, 100.0]),
        np.asarray([100.0, 0.0, 100.0]),
        1000,
        1.0e6,
        0.1,
    )
    # Exactly one SNR degradation enters Shannon serialization; the separate
    # delay term is only the calibrated non-Shannon excess.
    assert robust[0] == pytest.approx(nominal[0] - 3.0)
    assert robust[3] == pytest.approx(
        robust[2] + _model().processing_delay_s + 3.0e-4)


def test_missing_required_channel_packet_is_an_infinite_event_score():
    score = event_joint_channel_underestimate_score(
        np.asarray([10.0, 10.0]),
        np.asarray([10.0, 10.0]),
        np.asarray([1.0e-3, 1.0e-3]),
        np.asarray([1.0e-3, 1.0e-3]),
        snr_resolution_db=1.0,
        latency_resolution_s=1.0e-4,
        delivered_mask=np.asarray([True, False]),
    )
    assert np.isinf(score)


def _calibration_events(count):
    predicted_snr = [np.asarray([0.0]) for _ in range(count)]
    observed_snr = [np.asarray([0.0]) for _ in range(count)]
    predicted_latency = [np.asarray([0.0]) for _ in range(count)]
    observed_latency = [
        np.asarray([float(index) * 1.0e-3]) for index in range(count)
    ]
    return predicted_snr, observed_snr, predicted_latency, observed_latency


def test_channel_margin_uses_finite_sample_event_quantile_and_fails_closed():
    events = _calibration_events(19)
    epoch = calibrate_frozen_channel_margin_epoch(
        *events,
        event_ids=[f"event-{index}" for index in range(19)],
        miscoverage=0.05,
        snr_resolution_db=2.0,
        latency_resolution_s=1.0e-3,
        epoch_id="channel-1",
    )
    assert epoch.multiplier == pytest.approx(18.0)
    assert epoch.snr_margin_db == pytest.approx(36.0)
    assert epoch.latency_margin_s == pytest.approx(18.0e-3)
    assert epoch.finite

    unresolved_events = _calibration_events(18)
    unresolved = calibrate_frozen_channel_margin_epoch(
        *unresolved_events,
        event_ids=[f"short-{index}" for index in range(18)],
        miscoverage=0.05,
        snr_resolution_db=2.0,
        latency_resolution_s=1.0e-3,
        epoch_id="channel-short",
    )
    assert np.isinf(unresolved.multiplier)
    assert not unresolved.finite

    missing_packet = calibrate_frozen_channel_margin_epoch(
        *events,
        event_ids=[f"loss-{index}" for index in range(19)],
        miscoverage=0.05,
        snr_resolution_db=2.0,
        latency_resolution_s=1.0e-3,
        epoch_id="channel-loss",
        delivered_mask_events=[
            np.asarray([index != 18]) for index in range(19)
        ],
    )
    assert np.isinf(missing_packet.multiplier)
    assert not missing_packet.finite


def test_joint_epoch_spends_one_risk_budget_without_independence_assumption():
    transition_scores = np.arange(19, dtype=np.float64)
    channel_scores = transition_scores[::-1] + 0.5
    epoch = calibrate_frozen_joint_safety_epoch(
        transition_scores,
        channel_scores,
        event_ids=[f"joint-{index}" for index in range(19)],
        miscoverage=0.05,
        snr_resolution_db=1.0,
        latency_resolution_s=1.0e-4,
        epoch_id="joint-1",
        training_event_ids=["train-0", "train-1"],
    )
    expected_joint = np.maximum(transition_scores, channel_scores)
    assert epoch.joint_event_scores == pytest.approx(expected_joint)
    assert epoch.multiplier == pytest.approx(np.max(expected_joint))
    assert epoch.snr_margin_db == pytest.approx(epoch.multiplier)
    assert epoch.latency_margin_s == pytest.approx(
        epoch.multiplier * 1.0e-4)
    assert epoch.finite

    with pytest.raises(ValueError, match="leakage"):
        calibrate_frozen_joint_safety_epoch(
            [0.0],
            [0.0],
            event_ids=["train-0"],
            miscoverage=0.05,
            snr_resolution_db=1.0,
            latency_resolution_s=1.0e-4,
            epoch_id="leaky",
            training_event_ids=["train-0"],
        )
