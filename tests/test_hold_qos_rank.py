import numpy as np

from tools.audit_hold_qos_rank import _multiframe_episode_summary, hold_aware_score


def test_hold_aware_score_uses_only_current_hold_segment():
    values = np.zeros((4, 1, 1, 1), dtype=np.float64)
    values[:, 0, 0, 0] = [1.0, 0.2, 0.8, 0.4]
    episode = np.zeros(4, dtype=np.int64)
    resolved = np.array([True, False, True, False])

    score = hold_aware_score(
        values, episode, resolved, tail_weight=1.0, quantile=0.0)

    assert score[0, 0, 0, 0] == 0.2
    assert score[2, 0, 0, 0] == 0.4
    # Non-resolve frames are irrelevant to the selector and remain unchanged.
    assert score[1, 0, 0, 0] == 0.2


def test_hold_aware_score_does_not_cross_episode_boundary():
    values = np.zeros((3, 1, 1, 1), dtype=np.float64)
    values[:, 0, 0, 0] = [0.9, 0.7, 0.1]
    episode = np.array([0, 0, 1])
    resolved = np.array([True, False, True])

    score = hold_aware_score(
        values, episode, resolved, tail_weight=0.0, quantile=0.0)

    assert score[0, 0, 0, 0] == 0.8
    assert score[2, 0, 0, 0] == 0.1


def test_multiframe_summary_distinguishes_mean_and_all_episode_gate():
    # One easy and one hard episode: two-frame mean passes 0.75, but the hard
    # episode does not. Five frames give both enough accumulated evidence.
    target_deflection = np.array([
        [16.0], [16.0], [16.0], [16.0], [16.0],
        [4.0], [4.0], [4.0], [4.0], [4.0],
    ])
    data = {
        "episode": np.array([0] * 5 + [1] * 5),
        "seed": np.array([10] * 5 + [20] * 5),
    }

    result = _multiframe_episode_summary(
        target_deflection,
        data,
        p_fa=1e-3,
        worst_floor=0.75,
        frame_duration_s=0.1,
    )

    assert result["minimum_window_with_mean_worst_pass"] is not None
    assert result["minimum_window_with_all_episodes_pass"] is not None
    assert (
        result["minimum_window_with_all_episodes_pass"]
        > result["minimum_window_with_mean_worst_pass"]
    )
