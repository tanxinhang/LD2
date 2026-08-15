import numpy as np
import pytest

from uav_isac.coordination.digest_rendezvous import (
    StructureDigestRendezvousLayout,
    certify_deferred_horizon_prefix,
    certify_rendezvous_candidate_suffix,
    certify_structure_digest_rendezvous,
)
from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.owner_proposal_transport import RankedOwnerProposal
from uav_isac.coordination.owner_proposal_transport import OwnerProposalWireLayout
from uav_isac.coordination.power_repair_transport import PowerRepairWireLayout
from uav_isac.environment.communication import InterUAVCommunicationModel


def _model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4],
        header_bits=64,
        bandwidth_hz=100_000.0,
        deadline_s=0.005,
        processing_delay_s=0.0002,
        snr_threshold_db=0.0,
        antenna_gain_dbi=0.0,
        carrier_hz=28.0e9,
        tx_power_w=0.25,
        kT=4.0e-21,
        noise_figure_db=4.0,
        dt=0.1,
    )


def _case():
    selected = np.zeros((4, 4, 2), dtype=bool)
    selected[0, 2, 0] = True
    selected[1, 2, 1] = True
    role = np.asarray([1, 1, 0, 0], dtype=np.int8)
    owner = np.asarray([2, 2], dtype=np.int64)
    candidate = selected.copy()
    candidate[1, 2, 1] = False
    candidate[1, 3, 1] = True
    move = LocalMove(
        kind="N6",
        selected=candidate,
        role=role.copy(),
        owner=np.asarray([2, 3], dtype=np.int64),
    )
    proposal = RankedOwnerProposal(
        proposer=2,
        move=move,
        lower=0.25,
        upper=0.75,
        score=0.5,
    )
    positions = np.asarray([
        [0.0, 0.0, 20.0],
        [30.0, 0.0, 20.0],
        [0.0, 30.0, 20.0],
        [30.0, 30.0, 20.0],
    ])
    return selected, role, owner, proposal, positions


def test_rendezvous_is_bounded_and_physically_feasible():
    selected, role, owner, proposal, positions = _case()
    layout = StructureDigestRendezvousLayout(
        4, 2, target_pair_limit=1, digest_bits=256)
    result = certify_structure_digest_rendezvous(
        selected,
        role,
        owner,
        proposal,
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        full_structure_verified=True,
        communication_model=_model(),
        layout=layout,
        control_period_s=0.1,
    )

    assert result.feasible
    assert result.full_structure_verified
    assert result.digest_bits == 256
    assert result.collision_probability_upper == 2.0 ** -256
    assert result.request_bits == layout.request_packet_bits
    assert result.reply_bits == 3 * layout.reply_packet_bits
    assert result.total_over_air_bits == (
        result.request_bits + result.reply_bits + result.proposal_bits)
    assert result.total_protocol_latency_s <= 0.1
    assert np.all(result.projected_comm_power_w < 1.0)


def test_hash_match_cannot_authorize_without_full_structure_equality():
    selected, role, owner, proposal, positions = _case()
    result = certify_structure_digest_rendezvous(
        selected,
        role,
        owner,
        proposal,
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        full_structure_verified=False,
        communication_model=_model(),
        layout=StructureDigestRendezvousLayout(
            4, 2, target_pair_limit=1),
        control_period_s=0.1,
    )

    assert not result.feasible
    assert "identity:full_structure_mismatch" in result.reasons


def test_total_serial_suffix_must_fit_the_control_period():
    selected, role, owner, proposal, positions = _case()
    result = certify_structure_digest_rendezvous(
        selected,
        role,
        owner,
        proposal,
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        full_structure_verified=True,
        communication_model=_model(),
        layout=StructureDigestRendezvousLayout(
            4, 2, target_pair_limit=1),
        control_period_s=1.0e-6,
    )

    assert not result.feasible
    assert "protocol:control_period" in result.reasons


def test_candidate_suffix_rechecks_power_transport_commit_and_balance():
    selected, role, owner, proposal, positions = _case()
    result = certify_rendezvous_candidate_suffix(
        selected,
        role,
        owner,
        proposal.move,
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        final_sensing_weights=np.full((4, 2), 0.5),
        communication_model=_model(),
        power_layout=PowerRepairWireLayout(4, 2, rounds=2),
        control_period_s=0.1,
    )

    assert result.feasible
    assert result.verification_bits > 0
    assert result.commit_bits > 0
    assert result.total_over_air_bits == (
        result.verification_bits + result.commit_bits)
    assert result.total_protocol_latency_s <= 0.1
    assert result.max_isac_power_balance_error_w <= 1.0e-12
    assert np.allclose(
        result.projected_comm_power_w
        + np.sum(result.projected_sensing_power_w, axis=1),
        1.0,
    )


def test_exact_parallel_structure_commit_can_be_reused_but_power_is_rechecked():
    selected, role, owner, proposal, positions = _case()
    common = dict(
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        final_sensing_weights=np.full((4, 2), 0.5),
        communication_model=_model(),
        power_layout=PowerRepairWireLayout(4, 2, rounds=2),
        control_period_s=0.1,
    )
    repeated = certify_rendezvous_candidate_suffix(
        selected, role, owner, proposal.move, **common)
    reused = certify_rendezvous_candidate_suffix(
        selected,
        role,
        owner,
        proposal.move,
        parallel_committed_move=proposal.move,
        parallel_commit_feasible=True,
        **common,
    )

    assert reused.feasible
    assert reused.dependency_commit_reused
    assert reused.verification_bits == repeated.verification_bits
    assert reused.commit_bits == 0
    assert reused.total_over_air_bits < repeated.total_over_air_bits
    assert reused.total_protocol_latency_s < repeated.total_protocol_latency_s


def test_parallel_commit_reuse_rejects_a_different_transition():
    selected, role, owner, proposal, positions = _case()
    different = LocalMove(
        kind=proposal.move.kind,
        selected=selected.copy(),
        role=role.copy(),
        owner=owner.copy(),
    )
    with pytest.raises(ValueError, match="exact same"):
        certify_rendezvous_candidate_suffix(
            selected,
            role,
            owner,
            proposal.move,
            positions=positions,
            existing_comm_power_w=np.zeros(4),
            final_sensing_weights=np.full((4, 2), 0.5),
            communication_model=_model(),
            power_layout=PowerRepairWireLayout(4, 2, rounds=2),
            control_period_s=0.1,
            parallel_committed_move=different,
            parallel_commit_feasible=True,
        )


def test_deferred_prefix_charges_baseline_and_proposal_but_no_candidate_commit():
    selected, role, owner, proposal, positions = _case()
    result = certify_deferred_horizon_prefix(
        selected,
        role,
        owner,
        (proposal,),
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        communication_model=_model(),
        power_layout=PowerRepairWireLayout(4, 2, rounds=2),
        proposal_layout=OwnerProposalWireLayout(
            4, 2, target_pair_limit=1, proposal_budget=1),
        control_period_s=0.1,
    )

    assert result.feasible
    assert result.invariant_bits == 0
    assert result.baseline_bits > 0
    assert result.proposal_count == 1
    assert result.proposal_bits >= 0
    assert result.total_over_air_bits == (
        result.baseline_bits + result.proposal_bits)
    assert result.total_protocol_latency_s == (
        result.baseline_latency_s + result.proposal_latency_s)
    assert np.all(result.projected_comm_power_w < 1.0)
