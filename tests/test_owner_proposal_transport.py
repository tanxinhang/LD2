import numpy as np
import pytest

from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.owner_proposal_transport import (
    OwnerProposalWireLayout,
    RankedOwnerProposal,
    certify_owner_proposal_transport,
    quantize_nonnegative_float16_lower,
    quantize_nonnegative_float16_upper,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


def _model():
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4], header_bits=64,
        bandwidth_hz=100_000.0, deadline_s=0.005,
        processing_delay_s=0.0002, snr_threshold_db=0.0,
        antenna_gain_dbi=0.0, carrier_hz=28.0e9,
        tx_power_w=0.25, kT=4.0e-21, noise_figure_db=4.0,
        dt=0.1,
    )


def _case():
    selected = np.zeros((4, 4, 2), dtype=bool)
    selected[0, 2, 0] = True
    selected[1, 2, 1] = True
    role = np.asarray([1, 1, 0, 0], dtype=np.int8)
    owner = np.asarray([2, 2], dtype=np.int64)
    proposal = selected.copy()
    proposal[1, 2, 1] = False
    proposal[1, 3, 1] = True
    move = LocalMove(
        kind="N6", selected=proposal, role=role.copy(),
        owner=np.asarray([2, 3], dtype=np.int64))
    positions = np.asarray([
        [0.0, 0.0, 20.0], [30.0, 0.0, 20.0],
        [0.0, 30.0, 20.0], [30.0, 30.0, 20.0],
    ])
    ranked = RankedOwnerProposal(
        proposer=2, move=move, lower=0.25, upper=0.75, score=0.5)
    return selected, role, owner, ranked, positions


def test_binary16_interval_quantization_is_directional():
    value = 0.3333
    lower = quantize_nonnegative_float16_lower(value)
    upper = quantize_nonnegative_float16_upper(value)
    assert lower <= value <= upper
    assert upper - lower <= 2.0e-3


def test_owner_proposal_packet_and_transport_are_bounded_and_feasible():
    selected, role, owner, proposal, positions = _case()
    layout = OwnerProposalWireLayout(4, 2, target_pair_limit=1)
    bits = layout.proposal_bits(selected, role, owner, proposal)
    assert bits > 0
    result = certify_owner_proposal_transport(
        selected,
        role,
        owner,
        [proposal],
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        communication_model=_model(),
        layout=layout,
    )
    assert result.feasible
    assert result.proposal_count == 1
    assert result.total_protocol_latency_s <= _model().deadline_s
    assert np.all(result.projected_comm_power_w < 1.0)


def test_owner_can_emit_at_most_one_proposal_per_round():
    selected, role, owner, proposal, positions = _case()
    with pytest.raises(ValueError, match="at most one"):
        certify_owner_proposal_transport(
            selected,
            role,
            owner,
            [proposal, proposal],
            positions=positions,
            existing_comm_power_w=np.zeros(4),
            communication_model=_model(),
            layout=OwnerProposalWireLayout(4, 2, target_pair_limit=1),
        )


def test_owner_proposal_layout_rejects_unimplemented_score_codec():
    with pytest.raises(ValueError, match="binary16"):
        OwnerProposalWireLayout(
            4, 2, target_pair_limit=1, score_bits=8)


def test_owner_proposal_layout_requires_one_bit_proof_flag():
    with pytest.raises(ValueError, match="exactly one bit"):
        OwnerProposalWireLayout(
            4, 2, target_pair_limit=1, proof_flag_bits=0)


def test_proposal_round_cannot_exceed_target_budget():
    selected, role, owner, proposal, positions = _case()
    proposals = [
        RankedOwnerProposal(
            proposer=sender,
            move=proposal.move,
            lower=proposal.lower,
            upper=proposal.upper,
            score=proposal.score,
        )
        for sender in (1, 2, 3)
    ]
    with pytest.raises(ValueError, match="owner-token budget"):
        certify_owner_proposal_transport(
            selected,
            role,
            owner,
            proposals,
            positions=positions,
            existing_comm_power_w=np.zeros(4),
            communication_model=_model(),
            layout=OwnerProposalWireLayout(4, 2, target_pair_limit=1),
        )


def test_round_budget_is_independent_of_atomic_target_budget():
    selected, role, owner, proposal, positions = _case()
    proposals = [
        RankedOwnerProposal(
            proposer=sender,
            move=proposal.move,
            lower=proposal.lower,
            upper=proposal.upper,
            score=proposal.score,
        )
        for sender in (1, 2, 3)
    ]
    result = certify_owner_proposal_transport(
        selected,
        role,
        owner,
        proposals,
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        communication_model=_model(),
        layout=OwnerProposalWireLayout(
            4,
            2,
            target_pair_limit=1,
            target_budget=2,
            proposal_budget=3,
        ),
    )

    assert result.proposal_count == 3
