import numpy as np

from uav_isac.evaluation.local_candidate_audit import (
    build_local_candidate_mask,
    candidate_equivalence_metrics,
    candidate_recall_metrics,
    delivered_target_tokens,
    episode_detection_summary,
    receiver_owner_from_pairs,
    select_local_neighbors,
    select_value_guided_neighbors,
)
from uav_isac.environment.observation_slices import ObservationSlices


def _token_observation(K=3, Q=2, token_dim=7):
    slices = ObservationSlices.from_config(
        K=K,
        Q=Q,
        use_p0=False,
        use_rel_features=True,
        use_comm_tokens=True,
        comm_token_dim=token_dim,
        comm_tokens_per_sender=Q,
    )
    obs = np.zeros((1, K, slices.total_dim), dtype=np.float32)
    return slices, obs


def test_delivered_target_tokens_maps_peer_axes_and_age():
    slices, obs = _token_observation()
    # Viewer 0 peer order is [1, 2]. Deliver both targets from peer 2.
    mask = np.zeros((1, 3, 4), dtype=np.float32)
    mask[0, 0, 2:4] = 1.0
    obs[..., slices.comm_mask_start:slices.comm_mask_start + 4] = mask
    tokens = np.zeros((1, 3, 4, 7), dtype=np.float32)
    tokens[0, 0, 2, -1] = 0.2
    tokens[0, 0, 3, -1] = 0.4
    obs[..., slices.comm_token_start:slices.comm_mask_start] = tokens.reshape(
        1, 3, -1)

    visible, age = delivered_target_tokens(obs, slices)
    assert visible.shape == (1, 3, 3, 2)
    assert visible[0, 0, 2].tolist() == [True, True]
    assert not np.any(visible[0, 0, 1])
    np.testing.assert_allclose(age[0, 0, 2], [0.2, 0.4])


def test_select_local_neighbors_prefers_delivery_count_then_aoi():
    visible = np.zeros((1, 4, 4, 3), dtype=bool)
    age = np.full((1, 4, 4, 3), np.inf)
    visible[0, 0, 1, :2] = True
    age[0, 0, 1, :2] = 0.5
    visible[0, 0, 2, :2] = True
    age[0, 0, 2, :2] = 0.1
    visible[0, 0, 3, 0] = True
    age[0, 0, 3, 0] = 0.0
    selected = select_local_neighbors(
        visible, age, neighbor_topk=2)
    assert selected[0, 0].tolist() == [False, True, True, False]


def test_value_guided_neighbors_reserve_weak_target_then_fill_by_value():
    values = np.zeros((1, 4, 4, 2), dtype=np.float64)
    values[0, 0, 1] = [0.20, 0.95]
    values[0, 0, 2] = [0.90, 0.10]
    values[0, 0, 3] = [0.40, 0.20]
    pd = np.full((1, 4, 2), 0.80, dtype=np.float64)
    pd[0, 0, 1] = 0.10
    visible = np.zeros((1, 4, 4, 2), dtype=bool)
    visible[0, 0, 1:, :] = True
    age = np.where(visible, 0.1, np.inf)
    selected = select_value_guided_neighbors(
        values,
        pd,
        visible,
        age,
        neighbor_topk=2,
        qos_floor=0.60,
        coverage_fraction=0.50,
    )
    assert selected[0, 0, 1]
    assert selected[0, 0, 2]
    assert not selected[0, 0, 3]


def test_candidate_mask_reserves_a_weak_target_slot():
    values = np.zeros((1, 3, 3, 3), dtype=np.float64)
    values[0, 0, 1] = [9.0, 8.0, 0.1]
    pd = np.full((1, 3, 3), 0.8)
    pd[0, 0, 2] = 0.1
    neighbors = np.zeros((1, 3, 3), dtype=bool)
    neighbors[0, 0, 1] = True
    visible = np.zeros((1, 3, 3, 3), dtype=bool)
    visible[0, 0, 1, 0] = True
    visible[0, 1, 0, 0] = True

    candidate, targets = build_local_candidate_mask(
        values,
        pd,
        neighbors,
        visible,
        target_topk=2,
        qos_floor=0.6,
        coverage_fraction=0.5,
    )
    assert targets[0, 0].tolist() == [True, False, True]
    assert candidate[0, 0, 1].tolist() == [True, False, True]


def test_candidate_recall_separates_owner_and_empty_target_failures():
    shape = (1, 3, 3, 2)
    physical = np.ones(shape, dtype=bool)
    candidate = np.zeros(shape, dtype=bool)
    teacher = np.zeros(shape, dtype=bool)
    teacher[0, 0, 1, 0] = True
    teacher[0, 2, 1, 1] = True
    candidate[0, 0, 1, 0] = True
    owner = np.array([[1, 1]])
    metrics = candidate_recall_metrics(
        candidate,
        physical,
        teacher,
        owner,
        np.array([[0.1, 0.9]]),
    )
    assert metrics["edge_recall"] == 0.5
    assert metrics["owner_recall"] == 0.5
    assert metrics["target_empty_rate"] == 0.5
    assert metrics["deficit_weighted_edge_recall"] == 0.1


def test_equivalence_recall_accepts_same_owner_substitute_edge():
    shape = (1, 3, 3, 1)
    physical = np.ones(shape, dtype=bool)
    teacher = np.zeros(shape, dtype=bool)
    teacher[0, 0, 1, 0] = True
    candidate = np.zeros(shape, dtype=bool)
    candidate[0, 2, 1, 0] = True
    d_eff = np.zeros(shape, dtype=np.float64)
    d_eff[0, 0, 1, 0] = 1.0
    d_eff[0, 2, 1, 0] = 0.96
    metrics = candidate_equivalence_metrics(
        candidate,
        physical,
        teacher,
        np.array([[1]]),
        d_eff,
        target_pair_limit=1,
        evidence_ratio=0.95,
    )
    assert metrics["same_owner_equivalent_recall"] == 1.0
    assert metrics["any_owner_equivalent_recall"] == 1.0
    assert metrics["teacher_owner_candidate_rate"] == 1.0


def test_receiver_owner_from_pairs_recovers_unique_receiver():
    pairs = np.zeros((2, 3, 3, 2), dtype=bool)
    pairs[0, 0, 2, 0] = True
    pairs[0, 1, 2, 0] = True
    pairs[1, 2, 0, 1] = True
    owners = receiver_owner_from_pairs(pairs)
    np.testing.assert_array_equal(owners, [[2, -1], [-1, 0]])


def test_episode_summary_uses_last_twenty_frames_per_target():
    pd = np.zeros((25, 4), dtype=np.float64)
    pd[:5] = 0.1
    pd[5:] = np.array([0.9, 0.8, 0.7, 0.6])
    summary, rows = episode_detection_summary(
        pd,
        np.zeros(25, dtype=np.int64),
        np.full(25, 17, dtype=np.int64),
    )
    assert len(rows) == 1
    assert np.isclose(rows[0]["steady"], 0.75)
    assert np.isclose(rows[0]["weak3"], 0.7)
    assert np.isclose(rows[0]["worst"], 0.6)
    assert np.isclose(summary["worst"], 0.6)
