import numpy as np

from uav_isac.physical.evidence import (
    EvidencePacketLayout,
    calibrate_deflection_confidence,
    calibrate_llr_clip,
    detect_from_llr,
    gather_owner_values,
    llr_threshold,
    local_quality_topk_mask,
    pre_evidence_fusion_owner,
    quantize_llr,
    sample_gaussian_llr,
    theoretical_detection_probability,
)


def test_gaussian_llr_moments_match_deflection_model():
    rng = np.random.default_rng(20260723)
    d = np.full(300_000, 4.0)

    llr_h0 = sample_gaussian_llr(d, hypothesis=0, rng=rng)
    llr_h1 = sample_gaussian_llr(d, hypothesis=1, rng=rng)

    assert abs(np.mean(llr_h0) + 2.0) < 0.015
    assert abs(np.mean(llr_h1) - 2.0) < 0.015
    assert abs(np.var(llr_h0) - 4.0) < 0.04
    assert abs(np.var(llr_h1) - 4.0) < 0.04


def test_llr_detector_matches_target_pfa_and_existing_pd_formula():
    rng = np.random.default_rng(1729)
    p_fa = 0.001
    samples = 400_000
    d_grid = np.asarray([1.0, 4.0, 12.0])
    d = np.broadcast_to(d_grid, (samples, d_grid.size))

    llr_h0 = sample_gaussian_llr(d, 0, rng=rng)
    llr_h1 = sample_gaussian_llr(d, 1, rng=rng)
    empirical_pfa = np.mean(detect_from_llr(llr_h0, d, p_fa), axis=0)
    empirical_pd = np.mean(detect_from_llr(llr_h1, d, p_fa), axis=0)
    theoretical_pd = theoretical_detection_probability(d_grid, p_fa)

    np.testing.assert_allclose(empirical_pfa, p_fa, atol=1.8e-4)
    np.testing.assert_allclose(empirical_pd, theoretical_pd, atol=1.8e-3)


def test_independent_receiver_llrs_add_to_central_llr_on_same_samples():
    rng = np.random.default_rng(81)
    receiver_d = np.asarray([[1.0, 3.0], [4.0, 2.0], [2.0, 0.5]])
    noise = rng.standard_normal(receiver_d.shape)
    local_llr = sample_gaussian_llr(
        receiver_d, 1, standard_normal=noise)

    central_llr = np.sum(local_llr, axis=0)
    central_d = np.sum(receiver_d, axis=0)
    expected = 0.5 * central_d + np.sum(
        np.sqrt(receiver_d) * noise, axis=0)

    np.testing.assert_allclose(central_llr, expected)


def test_fusion_owner_is_selected_from_quality_not_realized_llr():
    receiver_d = np.asarray([
        [2.0, 5.0, 1.0],
        [4.0, 1.0, 3.0],
        [4.0, 2.0, 2.0],
    ])
    owner = pre_evidence_fusion_owner(receiver_d)
    np.testing.assert_array_equal(owner, np.asarray([1, 0, 1]))

    realized_llr = np.asarray([
        [100.0, -2.0, 1.0],
        [-5.0, 50.0, 2.0],
        [80.0, 20.0, 99.0],
    ])
    np.testing.assert_array_equal(
        gather_owner_values(realized_llr, owner),
        np.asarray([-5.0, -2.0, 2.0]),
    )


def test_zero_deflection_has_no_deterministic_llr_detection():
    threshold = llr_threshold(np.asarray([0.0]), 0.001)
    assert np.isinf(threshold[0])
    llr = sample_gaussian_llr(
        np.asarray([0.0]), 1, standard_normal=np.asarray([10.0]))
    assert not detect_from_llr(llr, np.asarray([0.0]), 0.001)[0]


def test_structured_packet_bit_accounting_is_explicit():
    layout = EvidencePacketLayout(
        num_agents=4,
        num_targets=4,
        header_bits=64,
        timestamp_bits=16,
    )
    assert layout.source_bits == 2
    assert layout.target_bits == 2
    assert layout.broadcast_bits(num_entries=1, llr_bits=8) == 92
    assert layout.broadcast_bits(num_entries=2, llr_bits=8) == 102
    assert layout.broadcast_bits(num_entries=0, llr_bits=8) == 0
    assert layout.broadcast_bits(num_entries=2, llr_bits=0) == 0


def test_llr_quantization_clips_and_improves_with_precision():
    values = np.asarray([-20.0, -1.3, 0.0, 2.7, 20.0])
    q4 = quantize_llr(values, bits=4, clip_max=8.0)
    q8 = quantize_llr(values, bits=8, clip_max=8.0)
    assert np.max(np.abs(q4)) <= 8.0
    assert np.max(np.abs(q8)) <= 8.0
    assert q4[2] == 0.0
    assert q8[2] == 0.0
    interior = np.abs(values) < 8.0
    assert np.mean((q8[interior] - values[interior]) ** 2) < np.mean(
        (q4[interior] - values[interior]) ** 2)


def test_local_quality_topk_mask_supports_frame_batches():
    receiver_d = np.asarray([
        [[4.0, 1.0, 0.0], [2.0, 3.0, 0.0]],
        [[0.0, 5.0, 2.0], [0.0, 0.0, 0.0]],
    ])
    mask = local_quality_topk_mask(receiver_d, topk=1)
    np.testing.assert_array_equal(
        mask,
        np.asarray([
            [[True, False, False], [False, True, False]],
            [[False, True, False], [False, False, False]],
        ]),
    )


def test_clip_calibration_is_reproducible_and_positive():
    receiver_d = np.asarray([[1.0, 4.0], [2.0, 8.0]])
    first = calibrate_llr_clip(
        receiver_d, samples=20_000, seed=17)
    second = calibrate_llr_clip(
        receiver_d, samples=20_000, seed=17)
    assert first == second
    assert first > 0.0


def test_two_bit_confidence_quantizer_has_four_monotone_levels():
    calibration = np.geomspace(0.1, 100.0, 1000)
    quantizer = calibrate_deflection_confidence(
        calibration, bits=2)
    decoded = quantizer.quantize(
        np.asarray([0.0, 0.2, 2.0, 20.0, 80.0]))
    assert quantizer.representatives.size == 4
    assert decoded[0] == 0.0
    assert np.all(np.diff(decoded[1:]) >= 0.0)
