import numpy as np

from uav_isac.coordination.certified_factor_message import (
    certify_capability_route,
    certify_fixed_plan_three_floors,
    dense_capability_payload_bits,
    factor_capability_payload_bits,
    factor_coefficient_envelope,
    linear_interval_quantize_positive,
    log_interval_quantize,
    task_equivalent_payload_bits,
)


def test_capability_route_certificate_uses_monotone_gauge_bounds():
    # gamma(A)=2/sum(A) is a simple decreasing capability gauge.
    gauge = lambda gain: 2.0 / float(np.sum(gain))
    feasible = certify_capability_route(
        np.full((2, 2), 0.6), np.full((2, 2), 0.7), gauge)
    assert feasible.status == "CERT_L1_FEASIBLE"
    assert feasible.gamma_lower <= feasible.gamma_upper <= 1.0
    fallback = certify_capability_route(
        np.full((2, 2), 0.3), np.full((2, 2), 0.4), gauge)
    assert fallback.status == "CERT_L1_FALLBACK"
    unresolved = certify_capability_route(
        np.full((2, 2), 0.45), np.full((2, 2), 0.55), gauge)
    assert unresolved.status == "UNRESOLVED"


def test_capability_route_certificate_fails_closed_on_missing_or_bad_gauge():
    lower = np.ones((2, 2))
    upper = lower * 2.0
    missing = certify_capability_route(lower, upper, lambda gain: None)
    assert missing.status == "UNRESOLVED"
    # Deliberately increasing 'gauge' violates the required monotonicity.
    broken = certify_capability_route(
        lower, upper, lambda gain: float(np.sum(gain)))
    assert broken.status == "UNRESOLVED"


def test_log_interval_quantizer_contains_dynamic_range_endpoints():
    value = np.geomspace(1.0e-12, 1.0e4, 100)
    code = log_interval_quantize(value, 3)
    assert np.all(code.lower <= value)
    assert np.all(value <= code.upper)


def test_factor_envelope_contains_active_product_and_zeros_support():
    tx = np.array([1.0, 2.0, 4.0])
    rx = np.array([0.5, 3.0, 5.0])
    active = np.ones((3, 3), dtype=bool)
    active[0, 2] = False
    lower, upper = factor_coefficient_envelope(
        log_interval_quantize(tx, 4), log_interval_quantize(rx, 4), active)
    truth = np.outer(tx, rx)
    truth[np.eye(3, dtype=bool) | ~active] = 0.0
    assert np.all(lower <= truth)
    assert np.all(truth <= upper)
    assert lower[0, 2] == upper[0, 2] == 0.0


def test_payload_accounting_includes_adaptive_dd_codec():
    dense = dense_capability_payload_bits(6, 6, 8)
    sparse = factor_capability_payload_bits(6, 6, 8, 4)
    dense_dd = factor_capability_payload_bits(6, 6, 8, 180)
    assert dense.total_bits == 25 + 32 + 180 * 8
    assert sparse.dd_bits == 8 + 4 * 8
    assert dense_dd.dd_bits == 180
    assert sparse.total_bits < dense.total_bits


def test_three_floor_certificate_is_fail_closed_between_bounds():
    lower = np.zeros((3, 3, 3))
    upper = np.zeros_like(lower)
    for q in range(3):
        lower[0, 1, q] = 15.0
        upper[0, 1, q] = 20.0
    feasible = certify_fixed_plan_three_floors(
        lower, upper, np.ones(3, dtype=int), np.ones((3, 3)),
        p_fa=0.1, task_floors=(0.55, 0.55, 0.55, 2))
    assert feasible.status == "CERT_FEASIBLE"
    unresolved = certify_fixed_plan_three_floors(
        lower * 0.1, upper, np.ones(3, dtype=int), np.ones((3, 3)),
        p_fa=0.1, task_floors=(0.55, 0.55, 0.55, 2))
    assert unresolved.status == "UNRESOLVED"
    infeasible = certify_fixed_plan_three_floors(
        lower * 0.0, upper * 0.01, np.ones(3, dtype=int), np.ones((3, 3)),
        p_fa=0.1, task_floors=(0.55, 0.55, 0.55, 2))
    assert infeasible.status == "CERT_INFEASIBLE"


def test_linear_interval_and_fair_payload_baselines():
    value = np.geomspace(1.0e-9, 2.0, 50)
    code = linear_interval_quantize_positive(value, 8)
    assert np.all(code.lower <= value)
    assert np.all(value <= code.upper)
    factor = task_equivalent_payload_bits("factor_log", 6, 6, 8, 4)
    dense_log = task_equivalent_payload_bits("dense_log", 6, 6, 8, 4)
    dense_linear = task_equivalent_payload_bits("dense_linear", 6, 6, 8, 4)
    assert factor.dd_bits == dense_log.dd_bits == dense_linear.dd_bits
    assert dense_log.scale_bits == factor.scale_bits == 64
    assert dense_linear.scale_bits == 32
