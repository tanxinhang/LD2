import numpy as np
import pytest

from uav_isac.coordination.local_exchange_oracle import LocalMove
from uav_isac.coordination.power_repair_transport import PowerRepairWireLayout
from uav_isac.coordination.owner_proposal_transport import (
    OwnerProposalWireLayout,
    RankedOwnerProposal,
)
from uav_isac.coordination.structure_sequence_transport import (
    certify_top1_structure_sequence_transport,
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
    return selected, role, owner, move, positions


def test_top1_sequence_charges_verification_and_commit_and_preserves_one_watt():
    selected, role, owner, move, positions = _case()
    result = certify_top1_structure_sequence_transport(
        selected, role, owner, [move], [move],
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        final_sensing_weights=np.full((4, 2), 0.5),
        communication_model=_model(),
        power_layout=PowerRepairWireLayout(4, 2, rounds=4),
        control_period_s=0.1,
        reserve_upper_w=0.25,
        owner_proposal_rounds=[[
            RankedOwnerProposal(
                proposer=2, move=move, lower=0.25, upper=0.75, score=0.5)
        ]],
        owner_proposal_layout=OwnerProposalWireLayout(
            4, 2, target_pair_limit=1),
    )
    assert result.feasible
    assert result.proposal_count == 1
    assert result.verification_count == 2
    assert result.baseline_verification_bits > 0
    assert result.verification_bits > result.baseline_verification_bits
    assert result.commit_bits > 0
    assert result.max_packet_latency_s <= _model().deadline_s
    assert result.total_protocol_latency_s <= 0.1
    assert result.max_isac_power_balance_error_w <= 1.0e-12


def test_top1_contract_rejects_uncounted_best_of_many_verifications():
    selected, role, owner, move, positions = _case()
    with pytest.raises(ValueError, match="Top-1"):
        certify_top1_structure_sequence_transport(
            selected, role, owner, [move, move], [],
            positions=positions,
            existing_comm_power_w=np.zeros(4),
            final_sensing_weights=np.full((4, 2), 0.5),
            communication_model=_model(),
            power_layout=PowerRepairWireLayout(4, 2, rounds=4),
            control_period_s=0.1,
        )


def test_proven_improvement_outranks_larger_exploratory_score():
    selected, role, owner, move, positions = _case()
    exploratory_selected = selected.copy()
    exploratory_selected[0, 2, 0] = False
    exploratory_selected[0, 3, 0] = True
    exploratory = LocalMove(
        kind="N6",
        selected=exploratory_selected,
        role=role.copy(),
        owner=np.asarray([3, 2], dtype=np.int64),
    )
    result = certify_top1_structure_sequence_transport(
        selected, role, owner, [move], [],
        positions=positions,
        existing_comm_power_w=np.zeros(4),
        final_sensing_weights=np.full((4, 2), 0.5),
        communication_model=_model(),
        power_layout=PowerRepairWireLayout(4, 2, rounds=4),
        control_period_s=0.1,
        owner_proposal_rounds=[[
            RankedOwnerProposal(
                proposer=2, move=move, lower=0.6, upper=0.7, score=0.6,
                certified_improvement=True),
            RankedOwnerProposal(
                proposer=3, move=exploratory, lower=0.2, upper=1.0,
                score=1.0, certified_improvement=False),
        ]],
        owner_proposal_layout=OwnerProposalWireLayout(
            4, 2, target_pair_limit=1),
    )
    assert result.feasible
