from dataclasses import replace

import numpy as np

from uav_isac.coordination.correlation_candidate_commit import (
    build_correlation_candidate_record,
    canonical_correlation_candidate_bytes,
    certify_correlation_candidate_commit,
    correlation_candidate_commit_layout,
    correlation_candidate_digest,
    verify_reconstructed_correlation_candidates,
)
from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.environment.communication import InterUAVCommunicationModel


def _fixture():
    K, Q = 4, 2
    coefficient = np.zeros((K, K, Q), dtype=np.float64)
    current = np.zeros_like(coefficient, dtype=bool)
    proposal = np.zeros_like(current)
    for edge, value in {
        (0, 2, 0): 10.0,
        (1, 2, 0): 10.0,
        (0, 2, 1): 8.0,
        (0, 3, 0): 8.0,
        (1, 3, 0): 8.0,
        (0, 3, 1): 8.0,
    }.items():
        coefficient[edge] = value
    current[0, 2, 0] = True
    current[1, 2, 0] = True
    current[0, 2, 1] = True
    proposal[0, 3, 0] = True
    proposal[1, 3, 0] = True
    proposal[0, 3, 1] = True
    role = np.asarray([1, 1, 0, 0], dtype=np.int8)
    old_owner = np.asarray([2, 2], dtype=np.int64)
    move = LocalMove(
        kind="receiver_exchange",
        selected=proposal,
        role=role.copy(),
        owner=np.asarray([3, 3], dtype=np.int64),
    )
    tau = np.zeros_like(coefficient)
    nu = np.zeros_like(coefficient)
    tau[1, 3, 0] = 1.0 / (64.0 * 15_625.0)
    record = build_correlation_candidate_record(
        move,
        coefficient,
        np.full(K, 0.8),
        tau,
        nu,
        generation_id=17,
        dependency_versions=np.asarray([11, 14, 9, 12], dtype=np.uint64),
        delay_size=64,
        doppler_size=16,
        delta_f_hz=15_625.0,
        symbol_time_s=64e-6,
    )
    positions = np.asarray([
        [0.0, 0.0, 100.0],
        [10.0, 0.0, 100.0],
        [20.0, 0.0, 100.0],
        [30.0, 0.0, 100.0],
    ])
    return current, role, old_owner, move, record, positions


def _model(**overrides):
    values = dict(
        rate_bits_per_dim=[0, 8],
        header_bits=64,
        bandwidth_hz=1.0e6,
        deadline_s=0.05,
        processing_delay_s=2.0e-4,
        snr_threshold_db=-20.0,
        antenna_gain_dbi=0.0,
        carrier_hz=2.4e9,
        tx_power_w=0.2,
        kT=4.0e-21,
        noise_figure_db=5.0,
        dt=0.1,
    )
    values.update(overrides)
    return InterUAVCommunicationModel(**values)


def test_identical_reconstructions_have_a_stable_full_sha256_identity():
    _, _, _, _, record, _ = _fixture()
    first = canonical_correlation_candidate_bytes(record)
    second = canonical_correlation_candidate_bytes(record)
    assert first == second
    assert correlation_candidate_digest(record).bit_length() <= 256
    certificate = verify_reconstructed_correlation_candidates(
        [record, record, record, record], [0, 1, 2, 3])
    assert certificate.feasible
    assert certificate.digest == correlation_candidate_digest(record)
    assert certificate.canonical_bytes == len(first)


def test_one_ulp_replica_difference_forces_abort():
    _, _, _, _, record, _ = _fixture()
    factors = record.correlation_factors.copy()
    factors[0] = np.nextafter(factors[0], np.inf)
    changed = replace(record, correlation_factors=factors)
    certificate = verify_reconstructed_correlation_candidates(
        [record, changed, record, record], [0, 1, 2, 3])
    assert not certificate.feasible
    assert "protocol:candidate_digest:uav:1" in certificate.reasons
    assert "protocol:canonical_record:uav:1" in certificate.reasons


def test_full_record_comparison_remains_mandatory_under_a_hash_collision():
    _, _, _, _, record, _ = _fixture()
    factors = record.correlation_factors.copy()
    factors[0] = np.nextafter(factors[0], np.inf)
    changed = replace(record, correlation_factors=factors)
    certificate = verify_reconstructed_correlation_candidates(
        [record, changed], [0, 1], _digest_fn=lambda _: bytes(32))
    assert not certificate.feasible
    assert not any("candidate_digest" in reason for reason in certificate.reasons)
    assert "protocol:canonical_record:uav:1" in certificate.reasons


def test_internally_inconsistent_power_plan_is_rejected():
    _, _, _, _, record, _ = _fixture()
    power = record.sensing_power_w.copy()
    power[0, 0] += 1.0
    changed = replace(record, sensing_power_w=power)
    certificate = verify_reconstructed_correlation_candidates([changed], [0])
    assert not certificate.feasible
    assert certificate.reasons == (
        "protocol:invalid_candidate:uav:0:ValueError",)


def test_generation_disagreement_forces_abort():
    _, _, _, _, record, _ = _fixture()
    stale = replace(record, generation_id=16)
    certificate = verify_reconstructed_correlation_candidates(
        [record, record, stale, record], [0, 1, 2, 3])
    assert not certificate.feasible
    assert "protocol:candidate_digest:uav:2" in certificate.reasons
    assert "protocol:canonical_record:uav:2" in certificate.reasons


def test_semantic_and_physical_certificates_must_both_pass():
    selected, role, owner, move, record, positions = _fixture()
    certificate = certify_correlation_candidate_commit(
        selected,
        role,
        owner,
        move,
        replica_records=[record, record, record, record],
        positions=positions,
        comm_power_w=np.full(4, 0.2),
        communication_model=_model(),
    )
    assert certificate.feasible
    assert certificate.reconstruction.feasible
    assert certificate.transport.feasible
    layout = correlation_candidate_commit_layout(4, 2)
    assert layout.digest_bits == 256
    assert certificate.transport.prepare_bits == layout.prepare_bits(
        changed_roles=0, changed_owners=2, toggled_edges=6)
    assert certificate.transport.total_over_air_bits > 0

    link_failure = certify_correlation_candidate_commit(
        selected,
        role,
        owner,
        move,
        replica_records=[record, record, record, record],
        positions=positions * np.asarray([1.0e7, 1.0e7, 1.0]),
        comm_power_w=np.full(4, 0.2),
        communication_model=_model(deadline_s=1.0e-5, snr_threshold_db=40.0),
    )
    assert not link_failure.feasible
    assert link_failure.reconstruction.feasible
    assert not link_failure.transport.feasible
