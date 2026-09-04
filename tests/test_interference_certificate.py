"""Tests for the fail-closed U2U spatial-reuse certificate."""

import numpy as np

from uav_isac.environment.interference_certificate import (
    certified_spatial_reuse_schedule,
)


COMMON = {
    "carrier_hz": 5.0e9,
    "bandwidth_hz": 1.0e6,
    "kT": 4.0e-21,
    "noise_figure_db": 4.0,
    "antenna_gain_dbi": 0.0,
    "required_sinr_db": 0.0,
}


def test_all_to_all_half_duplex_broadcast_cannot_reuse_a_slot():
    positions = np.asarray([
        [0.0, 0.0, 20.0],
        [100.0, 0.0, 20.0],
        [0.0, 100.0, 20.0],
    ])
    active = (0, 1, 2)
    receivers = {
        sender: tuple(node for node in active if node != sender)
        for sender in active
    }
    result = certified_spatial_reuse_schedule(
        positions, active, receivers, {sender: 0.25 for sender in active},
        half_duplex=True, **COMMON)

    assert result.all_links_certified
    assert result.groups == ((0,), (1,), (2,))
    assert result.max_concurrency == 1
    assert result.reuse_factor == 1.0
    assert np.all(result.pairwise_conflict[np.triu_indices(3, 1)])


def test_disjoint_short_links_can_reuse_when_cumulative_sinr_is_safe():
    # Two short desired links separated by 10 km have negligible cross-talk.
    positions = np.asarray([
        [0.0, 0.0], [10.0, 0.0],
        [10_000.0, 0.0], [10_010.0, 0.0],
    ])
    result = certified_spatial_reuse_schedule(
        positions,
        active_senders=(0, 2),
        receiver_sets={0: (1,), 2: (3,)},
        tx_powers_w={0: 0.25, 2: 0.25},
        half_duplex=True,
        **COMMON,
    )

    assert result.all_links_certified
    assert result.groups == ((0, 2),)
    assert result.max_concurrency == 2
    assert result.reuse_factor == 2.0
    assert result.minimum_sinr_margin_db > 0.0


def test_cumulative_interference_is_rechecked_after_pairwise_admission():
    # The exact geometry is less important than the invariant: every returned
    # group must pass the complete multi-interferer calculation.
    positions = np.asarray([
        [0.0, 0.0], [10.0, 0.0],
        [400.0, 0.0], [410.0, 0.0],
        [800.0, 0.0], [810.0, 0.0],
    ])
    active = (0, 2, 4)
    receivers = {0: (1,), 2: (3,), 4: (5,)}
    result = certified_spatial_reuse_schedule(
        positions, active, receivers, {sender: 0.25 for sender in active},
        interference_gain_margin_db=3.0,
        half_duplex=True,
        **COMMON,
    )

    assert result.all_links_certified
    assert result.minimum_sinr_margin_db >= -1.0e-10


def test_singleton_link_below_threshold_fails_closed():
    positions = np.asarray([[0.0, 0.0], [1.0e7, 0.0]])
    result = certified_spatial_reuse_schedule(
        positions,
        active_senders=(0,),
        receiver_sets={0: (1,)},
        tx_powers_w={0: 1.0e-9},
        half_duplex=True,
        **COMMON,
    )

    assert not result.all_links_certified
    assert result.uncertified_links == ((0, 1),)
    assert result.minimum_sinr_margin_db < 0.0


def test_robust_margins_never_improve_the_certificate():
    positions = np.asarray([
        [0.0, 0.0], [10.0, 0.0],
        [1000.0, 0.0], [1010.0, 0.0],
    ])
    kwargs = dict(
        positions=positions,
        active_senders=(0, 2),
        receiver_sets={0: (1,), 2: (3,)},
        tx_powers_w={0: 0.25, 2: 0.25},
        half_duplex=True,
        **COMMON,
    )
    nominal = certified_spatial_reuse_schedule(**kwargs)
    robust = certified_spatial_reuse_schedule(
        **kwargs, desired_gain_margin_db=3.0,
        interference_gain_margin_db=3.0)

    assert robust.minimum_sinr_margin_db <= nominal.minimum_sinr_margin_db
    assert robust.max_concurrency <= nominal.max_concurrency
