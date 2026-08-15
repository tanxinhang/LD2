import numpy as np

from uav_isac.coordination.owner_bid_transport import (
    OwnerBidWireLayout,
    certify_owner_bid_transport,
    owner_bid_sparse_candidate_mask,
    owner_capacity_bid_envelope,
)
from uav_isac.environment.communication import InterUAVCommunicationModel


def _model() -> InterUAVCommunicationModel:
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


def test_owner_bid_wire_layout_counts_fixed_records():
    layout = OwnerBidWireLayout(
        num_agents=6, num_targets=6, owners_per_target=3)
    assert layout.shared_bits == 150
    assert layout.bid_packet_bits == 360
    assert layout.decision_packet_bits == 240


def test_owner_bids_are_directional_and_candidates_are_sparse():
    lower = np.zeros((4, 4, 2), dtype=np.float64)
    lower[0, 2] = [4.0, 1.0]
    lower[1, 2] = [3.0, 2.0]
    lower[0, 3] = [2.0, 4.0]
    lower[1, 3] = [1.0, 3.0]
    upper = 1.001 * lower
    budget = np.asarray([0.8, 0.7, 0.9, 0.6])
    bid_lower, bid_upper = owner_capacity_bid_envelope(
        lower, upper, budget, target_pair_limit=2)
    assert np.all(bid_lower <= bid_upper)
    result = owner_bid_sparse_candidate_mask(
        lower,
        upper,
        budget,
        target_pair_limit=2,
        owners_per_target=1,
    )
    assert result.selected_owner_groups == ((2, 0), (3, 1))
    assert result.candidate_edge_count < result.full_edge_count


def test_owner_candidate_selection_is_permutation_equivariant():
    rng = np.random.default_rng(7)
    upper = rng.uniform(0.1, 2.0, size=(4, 4, 3))
    diagonal = np.arange(4)
    upper[diagonal, diagonal, :] = 0.0
    lower = 0.9 * upper
    budget = rng.uniform(0.4, 0.9, size=4)
    base = owner_bid_sparse_candidate_mask(
        lower,
        upper,
        budget,
        target_pair_limit=2,
        owners_per_target=2,
    )
    uav_permutation = np.asarray([2, 0, 3, 1])
    target_permutation = np.asarray([1, 2, 0])
    permuted = owner_bid_sparse_candidate_mask(
        lower[
            uav_permutation[:, None, None],
            uav_permutation[None, :, None],
            target_permutation[None, None, :],
        ],
        upper[
            uav_permutation[:, None, None],
            uav_permutation[None, :, None],
            target_permutation[None, None, :],
        ],
        budget[uav_permutation],
        target_pair_limit=2,
        owners_per_target=2,
    )
    assert base.candidate_edge_count == permuted.candidate_edge_count
    np.testing.assert_allclose(
        np.sort(base.owner_lower_bid, axis=None),
        np.sort(permuted.owner_lower_bid, axis=None),
    )


def test_owner_bid_transport_meets_deadlines_and_control_period():
    positions = np.asarray([
        [0.0, 0.0, 20.0], [40.0, 0.0, 20.0],
        [80.0, 0.0, 20.0], [0.0, 40.0, 20.0],
        [40.0, 40.0, 20.0], [80.0, 40.0, 20.0],
    ])
    result = certify_owner_bid_transport(
        positions,
        np.full(6, 0.25),
        communication_model=_model(),
        layout=OwnerBidWireLayout(
            num_agents=6, num_targets=6, owners_per_target=3),
        control_period_s=0.1,
        snr_margin_db=3.0,
        latency_margin_s=5.0e-4,
    )
    assert result.feasible
    assert result.max_packet_latency_s <= 0.005 + 1.0e-12
    assert result.total_protocol_latency_s <= 0.1
    assert result.total_over_air_bits == 5 * 360 + 240
    assert np.all(result.projected_comm_power_w < 1.0)
