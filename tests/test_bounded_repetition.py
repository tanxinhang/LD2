import numpy as np
import pytest

from uav_isac.coordination.bounded_repetition import (
    certify_bounded_repetition_path,
)


def _certificate(**overrides):
    arguments = dict(
        common_compute_s=0.010,
        parallel_branch_compute_s=(0.012, 0.020),
        parallel_branch_protocol_latency_s=(0.018, 0.008),
        serial_compute_s=0.006,
        serial_protocol_latency_s=(0.003, 0.007),
        nominal_over_air_bits=1000,
        nominal_rf_energy_j=0.02,
        repetition_count=1,
        excess_queue_bound_s=0.0,
        deadline_s=0.1,
        nominal_transport_feasible=True,
    )
    arguments.update(overrides)
    return certify_bounded_repetition_path(**arguments)


def test_r1_zero_queue_is_exact_nominal_path():
    result = _certificate()
    expected = 0.010 + max(0.012 + 0.018, 0.020 + 0.008)
    expected += 0.003
    expected += 0.007
    expected += 0.006
    assert result.feasible
    assert result.nominal_total_latency_s == expected
    assert result.worst_case_total_latency_s == expected
    assert result.worst_case_over_air_bits == 1000
    assert result.worst_case_rf_energy_j == 0.02


def test_repetition_multiplies_only_protocol_terms_and_rf_resources():
    result = _certificate(repetition_count=2, excess_queue_bound_s=0.004)
    expected = 0.010 + max(0.012 + 2 * 0.018, 0.020 + 2 * 0.008)
    expected += 2 * 0.003
    expected += 2 * 0.007
    expected += 0.006
    expected += 0.004
    assert result.worst_case_total_latency_s == expected
    assert result.worst_case_over_air_bits == 2000
    assert result.worst_case_rf_energy_j == 0.04
    assert result.tolerated_erasures_per_logical_packet == 1


def test_repetition_deadline_and_nominal_transport_fail_closed():
    late = _certificate(repetition_count=4)
    broken = _certificate(nominal_transport_feasible=False)
    assert not late.feasible
    assert "repetition:end_to_end_deadline" in late.reasons
    assert not broken.feasible
    assert "transport:nominal_infeasible" in broken.reasons


def test_repetition_envelope_is_monotone_in_repetition_and_queue():
    baseline = _certificate()
    repeated = _certificate(repetition_count=2)
    queued = _certificate(repetition_count=2, excess_queue_bound_s=0.005)
    assert baseline.worst_case_total_latency_s < repeated.worst_case_total_latency_s
    assert repeated.worst_case_total_latency_s < queued.worst_case_total_latency_s
    assert repeated.worst_case_over_air_bits >= baseline.worst_case_over_air_bits
    assert repeated.worst_case_rf_energy_j >= baseline.worst_case_rf_energy_j


@pytest.mark.parametrize(
    "override",
    [
        {"repetition_count": 0},
        {"excess_queue_bound_s": -1.0},
        {"nominal_rf_energy_j": np.nan},
        {"nominal_over_air_bits": 1.5},
        {"parallel_branch_compute_s": ()},
        {"parallel_branch_protocol_latency_s": (0.1,)},
    ],
)
def test_invalid_envelopes_are_rejected(override):
    with pytest.raises(ValueError):
        _certificate(**override)
