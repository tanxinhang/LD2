"""Tests for the decoupled belief-freshness schedule and the unified MAC
protocol slot accounting (audit priorities #3 and #5).

Priority #3: the posterior broadcast must not ride blindly on the evidence
top-k selection.  A freshness schedule ranks each source's own posteriors by a
joint AoI / uncertainty / task-loss score, skips unfusable (stale or
owned-target) entries, and only the scheduled entries pay the posterior
payload bits.

Priority #5: the coordination broadcast and the evidence/belief broadcast
share one frame window; when serialized in two sub-slots their summed latency
must fit the frame deadline for the protocol to be certificate-valid.
"""

import numpy as np
import pytest

from uav_isac.environment.communication import CommunicationStepStats
from uav_isac.physical.evidence import (
    EvidencePacketLayout,
    route_structured_evidence,
)


def _layout(feedback_bits=88):
    return EvidencePacketLayout(
        num_agents=12,
        num_targets=12,
        header_bits=64,
        timestamp_bits=16,
        confidence_bits=2,
        feedback_bits_per_entry=feedback_bits,
    )


def _fake_link_model():
    class _Link:
        message_dim = 192
        header_bits = 64

        def transmit(self, messages, rate_indices, xyz, tx_powers_w,
                     extra_payload_bits, base_payload_dimensions,
                     suppress_message_payload,
                     header_bits_by_sender=None,
                     service_envelope_payload_bits_by_sender=None):
            deliveries = []
            stats = CommunicationStepStats()
            for sender, message in messages.items():
                for receiver in range(xyz.shape[0]):
                    if receiver == sender:
                        continue
                    deliveries.append(_Delivery(sender, receiver))
                    stats.attempted_links += 1
                    stats.delivered_links += 1
                    stats.total_bits += float(
                        extra_payload_bits.get(sender, 0))
                    stats.mean_latency_s = 0.002
                    stats.p95_latency_s = 0.0025
                    stats.max_latency_s = 0.0025
                    stats.active_senders = len(messages)
            return deliveries, stats

    class _Delivery:
        def __init__(self, sender, receiver):
            self.sender = sender
            self.receiver = receiver
            self.rate_index = 0
            self.latency_s = 0.002
            self.snr_db = 30.0
            self.tx_power_w = 0.05
            self.token_mask = np.ones(12)
            self.message = np.zeros(192)
            self.delay_frames = 1

    return _Link()


def test_broadcast_bits_legacy_coupled_unchanged():
    layout = _layout()
    base = EvidencePacketLayout(
        num_agents=12, num_targets=12, header_bits=64,
        timestamp_bits=16, confidence_bits=2, feedback_bits_per_entry=0)
    assert layout.broadcast_bits(3, 8) - base.broadcast_bits(3, 8) == 3 * 88


def test_broadcast_bits_decoupled_belief_entries():
    layout = _layout()
    # 3 evidence entries + 2 extra belief entries: only the 2 belief entries
    # pay the 88-bit posterior payload beyond the evidence header cost.
    base = EvidencePacketLayout(
        num_agents=12, num_targets=12, header_bits=64,
        timestamp_bits=16, confidence_bits=2, feedback_bits_per_entry=0)
    with_union = layout.broadcast_bits(3, 8, belief_entries=2)
    assert with_union - base.broadcast_bits(3, 8) == 2 * 88


def test_route_evidence_carries_decoupled_belief_mask():
    rng = np.random.default_rng(3)
    K, Q = 12, 12
    receiver_d = rng.uniform(0.0, 50.0, (K, Q))
    positions = rng.uniform(0.0, 1386.0, (K, 3))
    power = np.full(K, 0.05)
    owner = np.full(Q, -1, dtype=np.int64)
    for q in range(Q):
        owner[q] = int(q % K)
    evidence_mask = np.zeros((K, Q), dtype=bool)
    for k in range(K):
        evidence_mask[k, int(k % Q)] = True
    belief_mask = np.zeros((K, Q), dtype=bool)
    for k in range(K):
        belief_mask[k, int((k + 5) % Q)] = True

    transport = route_structured_evidence(
        receiver_d,
        positions,
        power,
        observation_frame=1,
        topk=1,
        llr_bits=8,
        layout=_layout(),
        link_model=_fake_link_model(),
        fusion_owner=owner,
        owner_aware=True,
        belief_selected_mask=belief_mask,
    )
    assert transport.belief_selected_mask is not None
    # The supplied posterior schedule remains explicit: evidence for another
    # target cannot silently reveal that target's 4D posterior.
    assert np.array_equal(transport.belief_schedule_mask, belief_mask)
    # The belief schedule is decoupled: it is not a copy of the evidence
    # selection, and the evidence selection stays owner-aware (a receiver
    # never broadcasts evidence about a target it owns).
    assert not np.array_equal(
        transport.selected_mask, transport.belief_schedule_mask)
    for k in range(K):
        assert not transport.selected_mask[k, int(k)]
    # The belief entries pay their posterior bits; target metadata is shared
    # through the union packet but the 88-bit state is never free.
    layout = _layout()
    for s in range(K):
        union_count = int(np.sum(
            transport.selected_mask[s] | belief_mask[s]))
        belief_count = int(np.sum(belief_mask[s]))
        assert transport.payload_bits_by_sender[s] == (
            layout.broadcast_bits(
                union_count, 8, belief_entries=belief_count))
    # The fake link model broadcasts each sender to all K-1 peers and charges
    # the payload remainder beyond the physical header per delivery.
    assert transport.total_bits == float(
        11 * (int(np.sum(transport.payload_bits_by_sender)) - K * 64))


def test_route_evidence_legacy_coupled_belief_default():
    rng = np.random.default_rng(5)
    K, Q = 12, 12
    receiver_d = rng.uniform(0.0, 50.0, (K, Q))
    positions = rng.uniform(0.0, 1386.0, (K, 3))
    power = np.full(K, 0.05)
    owner = np.full(Q, -1, dtype=np.int64)
    for q in range(Q):
        owner[q] = int(q % K)
    transport = route_structured_evidence(
        receiver_d,
        positions,
        power,
        observation_frame=1,
        topk=1,
        llr_bits=8,
        layout=_layout(),
        link_model=_fake_link_model(),
        fusion_owner=owner,
        owner_aware=True,
    )
    assert np.array_equal(
        transport.belief_schedule_mask, transport.selected_mask)


def test_decoupled_posterior_pays_even_when_target_id_overlaps_evidence():
    rng = np.random.default_rng(17)
    K, Q = 12, 12
    receiver_d = rng.uniform(0.0, 50.0, (K, Q))
    positions = rng.uniform(0.0, 1386.0, (K, 3))
    power = np.full(K, 0.05)
    owner = np.arange(Q, dtype=np.int64) % K
    no_feedback = route_structured_evidence(
        receiver_d, positions, power,
        observation_frame=1, topk=1, llr_bits=8,
        layout=_layout(feedback_bits=0), link_model=_fake_link_model(),
        fusion_owner=owner, owner_aware=True,
        belief_selected_mask=np.zeros((K, Q), dtype=bool),
    )
    overlap = no_feedback.selected_mask.copy()
    charged = route_structured_evidence(
        receiver_d, positions, power,
        observation_frame=1, topk=1, llr_bits=8,
        layout=_layout(feedback_bits=88), link_model=_fake_link_model(),
        fusion_owner=owner, owner_aware=True,
        belief_selected_mask=overlap,
    )
    np.testing.assert_array_equal(charged.belief_schedule_mask, overlap)
    np.testing.assert_array_equal(
        charged.payload_bits_by_sender - no_feedback.payload_bits_by_sender,
        88 * np.sum(overlap, axis=1),
    )


def test_serialized_protocol_accounting():
    from uav_isac.environment.env_core import EnvironmentCore

    class _FakeCore:
        _active_comm_deadline_s = 0.004

    core = _FakeCore()

    coord = CommunicationStepStats()
    coord.total_bits = 1440.0
    coord.mean_latency_s = 0.00293
    coord.p95_latency_s = 0.0038
    coord.max_latency_s = 0.004

    class _Ev:
        mean_latency_s = 0.00244
        p95_latency_s = 0.0031
        total_bits = 982.0

    out = EnvironmentCore._serialized_protocol_accounting(
        core, coord, _Ev())
    assert abs(out['protocol_total_latency_s'] - 0.00537) < 1.0e-6
    assert abs(out['protocol_serialized_bits'] - 2422.0) < 1.0e-6
    assert out['protocol_total_deadline_violation'] == 1.0
    # A MAC that keeps both classes inside the frame has zero violation.
    coord2 = CommunicationStepStats()
    coord2.total_bits = 720.0
    coord2.mean_latency_s = 0.0015
    coord2.p95_latency_s = 0.0018
    ev2 = type('_Ev', (), {
        'mean_latency_s': 0.0016, 'p95_latency_s': 0.0019,
        'total_bits': 500.0})()
    out2 = EnvironmentCore._serialized_protocol_accounting(
        core, coord2, ev2)
    assert out2['protocol_total_latency_s'] < 0.004
    assert out2['protocol_total_deadline_violation'] == 0.0
    # No evidence broadcast -> only the coordination slot is counted.
    out3 = EnvironmentCore._serialized_protocol_accounting(core, coord, None)
    assert abs(out3['protocol_total_latency_s'] - 0.00293) < 1.0e-9
