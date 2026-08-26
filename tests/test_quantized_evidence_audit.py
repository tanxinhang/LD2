import numpy as np

from uav_isac.evaluation.quantized_evidence_audit import (
    analytic_trace_bounds,
    calibrate_standardized_threshold,
    episode_detection_metrics,
    evidence_inclusion,
    mean_structured_bits_per_frame,
    simulate_quantized_detection,
    structured_broadcast_transport,
    summarize_paired_detection_delta,
)
from uav_isac.environment.communication import InterUAVCommunicationModel
from uav_isac.physical.evidence import EvidencePacketLayout


def _trace():
    return np.asarray([
        [[4.0, 1.0], [2.0, 3.0], [1.0, 2.0]],
        [[3.0, 2.0], [1.0, 4.0], [2.0, 1.0]],
    ])


def _owners(frames):
    return np.tile(np.asarray([[0, 1]], dtype=np.int64), (frames, 1))


def test_evidence_inclusion_keeps_owner_and_selected_peers():
    routing = evidence_inclusion(
        _trace(), topk=1, fusion_owner=_owners(2))
    np.testing.assert_allclose(
        routing["fused_deflection"],
        np.asarray([[4.0, 5.0], [5.0, 4.0]]),
    )


def test_owner_aware_selection_does_not_spend_slots_on_local_owner_evidence():
    receiver_d = np.asarray([[
        [100.0, 4.0, 3.0],
        [2.0, 90.0, 1.0],
        [1.0, 2.0, 80.0],
    ]])
    naive = evidence_inclusion(
        receiver_d, topk=1, fusion_owner=np.asarray([[0, 1, 2]]),
        owner_aware=False)
    aware = evidence_inclusion(
        receiver_d, topk=1, fusion_owner=np.asarray([[0, 1, 2]]),
        owner_aware=True)
    assert np.sum(naive["peer_mask"]) == 0
    assert np.sum(aware["peer_mask"]) == 3
    assert np.sum(
        aware["fused_deflection"] - naive["fused_deflection"]) > 0.0


def test_threshold_deflection_uses_finite_peer_confidence_only():
    receiver_d = np.asarray([[
        [10.0, 2.0],
        [4.0, 8.0],
    ]])
    estimate = np.full_like(receiver_d, 1.5)
    routing = evidence_inclusion(
        receiver_d,
        topk=1,
        fusion_owner=np.asarray([[0, 1]]),
        owner_aware=True,
        peer_deflection_estimate=estimate,
    )
    # Each target owner keeps exact local quality and receives one 1.5-level
    # peer confidence rather than the peer's exact 4.0/2.0 deflection.
    np.testing.assert_allclose(
        routing["threshold_deflection"], np.asarray([[11.5, 9.5]]))


def test_quantized_detection_calibrates_and_improves_over_local():
    calibration = np.tile(_trace(), (20, 1, 1))
    threshold = calibrate_standardized_threshold(
        calibration,
        fusion_owner=_owners(calibration.shape[0]),
        topk=2,
        bits=8,
        clip_max=20.0,
        p_fa=0.01,
        samples=100_000,
        seed=1,
    )
    result = simulate_quantized_detection(
        calibration,
        fusion_owner=_owners(calibration.shape[0]),
        topk=2,
        bits=8,
        clip_max=20.0,
        standardized_threshold=threshold["standardized_threshold"],
        p_fa=0.01,
        draws_per_frame=4000,
        batch_frames=8,
        seed=2,
    )
    bounds = analytic_trace_bounds(calibration, 0.01)
    assert abs(result["aggregate_pfa"] - 0.01) < 0.002
    assert np.mean(result["pd"]) > np.mean(bounds["local_pd"])


def test_global_llr_threshold_needs_no_per_frame_deflection_at_decision():
    calibration = np.tile(_trace(), (20, 1, 1))
    threshold = calibrate_standardized_threshold(
        calibration,
        fusion_owner=_owners(calibration.shape[0]),
        topk=1,
        bits=8,
        clip_max=20.0,
        p_fa=0.01,
        threshold_mode="global_llr",
        samples=100_000,
        seed=7,
    )
    result = simulate_quantized_detection(
        calibration,
        fusion_owner=_owners(calibration.shape[0]),
        topk=1,
        bits=8,
        clip_max=20.0,
        standardized_threshold=threshold["threshold_value"],
        p_fa=0.01,
        threshold_mode="global_llr",
        draws_per_frame=4000,
        batch_frames=8,
        seed=8,
    )
    assert threshold["threshold_mode"] == "global_llr"
    assert abs(result["aggregate_pfa"] - 0.01) < 0.002


def test_episode_metrics_use_each_episode_steady_window():
    pd = np.asarray([
        [0.1, 0.2], [0.4, 0.6],
        [0.3, 0.5], [0.7, 0.9],
    ])
    episode = np.asarray([0, 0, 1, 1])
    metrics = episode_detection_metrics(pd, episode, steady_window=1)
    np.testing.assert_allclose(
        metrics,
        np.asarray([[0.5, 0.5, 0.4], [0.8, 0.8, 0.7]]),
    )


def test_structured_bits_share_header_across_top2_entries():
    selected = np.ones((3, 4, 2), dtype=bool)
    layout = EvidencePacketLayout(4, 4)
    bits, active = mean_structured_bits_per_frame(
        selected, layout, llr_bits=8)
    assert bits == 4 * 102
    assert active == 4.0


def test_structured_broadcast_transport_respects_deadline_and_snr():
    model = InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 8],
        header_bits=64,
        bandwidth_hz=100_000,
        deadline_s=0.005,
        processing_delay_s=0.0002,
        snr_threshold_db=0.0,
        antenna_gain_dbi=0.0,
        carrier_hz=5.0e9,
        tx_power_w=0.25,
        kT=4.0e-21,
        noise_figure_db=4.0,
        dt=0.1,
        message_dim=1,
    )
    positions = np.asarray([[
        [0.0, 0.0, 20.0],
        [50.0, 0.0, 20.0],
        [0.0, 50.0, 20.0],
    ]])
    power = np.full((1, 3), 0.25)
    selected = np.ones((1, 3, 1), dtype=bool)
    result = structured_broadcast_transport(
        positions,
        power,
        selected,
        EvidencePacketLayout(3, 1),
        llr_bits=8,
        model=model,
    )
    assert result["attempted_links"] == 6
    assert result["delivery_rate"] == 1.0
    assert result["mean_latency_s"] < 0.005


def test_paired_detection_delta_preserves_episode_pairing():
    episode = np.repeat(np.arange(12), 20)
    first = np.full((240, 4), 0.60)
    second = np.full((240, 4), 0.55)
    summary = summarize_paired_detection_delta(
        first, second, episode, bootstrap_samples=20)
    np.testing.assert_allclose(summary["worst_delta"], 0.05)
    assert summary["worst_delta_ci95"][0] > 0.0
