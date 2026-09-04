"""Certified stopping tests for progressive information transport."""

import numpy as np
import pytest

from uav_isac.coordination.progressive_information_transport import (
    InformationRefinementLayer,
    ProgressiveCandidate,
    certified_progressive_transport,
    quantized_psd_logdet_interval,
)


def _candidate(identifier, intervals):
    return ProgressiveCandidate(identifier, tuple(
        InformationRefinementLayer(lo, hi, bits, f"l{index}")
        for index, (lo, hi, bits) in enumerate(intervals)
    ))


def test_base_layer_stops_when_it_already_proves_the_decision():
    candidates = (
        _candidate(0, [(0.8, 1.0, 8), (0.9, 0.9, 32)]),
        _candidate(1, [(0.0, 0.5, 8), (0.4, 0.4, 32)]),
    )
    result = certified_progressive_transport(
        candidates, incumbent_candidate_id=1, link_rate_bps=1000.0,
        deadline_s=1.0, transmit_power_w=0.5,
    )
    assert result.certified
    assert result.chosen_candidate_id == 0
    assert result.transmitted_bits == 16
    assert result.refinement_count == 0


def test_overlapping_candidates_are_refined_until_certified():
    candidates = (
        _candidate(0, [(0.0, 1.0, 8), (0.95, 0.95, 32)]),
        _candidate(1, [(0.0, 0.9, 8), (0.6, 0.6, 32)]),
    )
    result = certified_progressive_transport(
        candidates, incumbent_candidate_id=1, link_rate_bps=1000.0,
        deadline_s=1.0, transmit_power_w=0.5,
    )
    assert result.certified
    assert result.chosen_candidate_id == 0
    assert result.transmitted_bits < result.full_transport_bits
    assert result.selected_layers == (1, 0)


def test_tight_budget_returns_incumbent_without_false_certificate():
    candidates = (
        _candidate(0, [(0.0, 1.0, 8), (0.8, 0.8, 32)]),
        _candidate(1, [(0.0, 0.9, 8), (0.6, 0.6, 32)]),
    )
    result = certified_progressive_transport(
        candidates, incumbent_candidate_id=1, link_rate_bps=1000.0,
        deadline_s=1.0, transmit_power_w=0.5, bit_budget=16,
    )
    assert not result.certified
    assert result.used_incumbent_fallback
    assert result.chosen_candidate_id == 1


def test_anytime_budget_uses_best_received_lower_bound_without_certifying():
    candidates = (
        _candidate(0, [(0.7, 1.0, 8), (0.8, 0.8, 32)]),
        _candidate(1, [(0.2, 0.9, 8), (0.6, 0.6, 32)]),
    )
    result = certified_progressive_transport(
        candidates, incumbent_candidate_id=1, link_rate_bps=1000.0,
        deadline_s=1.0, transmit_power_w=0.5, bit_budget=16,
        execute_best_lower_on_budget_exhaustion=True,
    )

    assert not result.certified
    assert not result.used_incumbent_fallback
    assert result.chosen_candidate_id == 0


def test_random_nested_final_intervals_recover_exact_best():
    rng = np.random.default_rng(20260827)
    for _ in range(30):
        gain = rng.uniform(0.1, 2.0, size=20)
        candidates = tuple(
            _candidate(index, [
                (0.0, float(value + 1.0), 8),
                (max(0.0, float(value - 0.2)), float(value + 0.2), 24),
                (float(value), float(value), 64),
            ])
            for index, value in enumerate(gain)
        )
        result = certified_progressive_transport(
            candidates, incumbent_candidate_id=0,
            link_rate_bps=1.0e6, deadline_s=1.0,
            transmit_power_w=0.25,
        )
        assert result.certified
        assert result.chosen_candidate_id == int(np.argmax(gain))


def test_non_nested_intervals_are_rejected():
    invalid = (_candidate(0, [(0.2, 0.5, 8), (0.1, 0.4, 8)]),)
    with pytest.raises(ValueError, match="nested"):
        certified_progressive_transport(
            invalid, incumbent_candidate_id=0,
            link_rate_bps=1.0, deadline_s=1.0, transmit_power_w=0.0,
        )


def test_quantized_psd_interval_contains_exact_logdet_without_label_input():
    rng = np.random.default_rng(81)
    for bits in (4, 8, 12):
        for _ in range(50):
            factor = rng.normal(size=(4, 2))
            gram = factor.T @ factor
            exact = float(np.linalg.slogdet(np.eye(2) + gram)[1])
            lower, upper = quantized_psd_logdet_interval(
                gram, bits_per_entry=bits)
            assert lower <= exact + 1.0e-12
            assert exact <= upper + 1.0e-12
