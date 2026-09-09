import numpy as np
import pytest

from uav_isac.evaluation.episode_regret import (
    candidate_support_by_episode,
    episode_qos_regret,
)


def _row(ep, steady, weak3, worst, feasible):
    return {
        "episode": ep,
        "steady": steady,
        "weak3": weak3,
        "worst": worst,
        "qos_feasible": feasible,
    }


def test_episode_qos_regret_preserves_vector_and_decomposes_flips():
    reference = [_row(0, .90, .80, .70, True), _row(1, .90, .80, .70, True)]
    method = [_row(0, .85, .75, .55, False), _row(1, .89, .79, .69, False)]
    result = episode_qos_regret(
        reference, method, candidate_supported={0: False, 1: True})

    assert result["regret_order"] == ["steady", "weak3", "worst"]
    assert result["episodes"][0]["regret"] == {
        "steady": pytest.approx(.05),
        "weak3": pytest.approx(.05),
        "worst": pytest.approx(.15),
    }
    assert result["aggregate"]["qos_flip_rate"] == pytest.approx(1.0)
    assert result["aggregate"]["candidate_miss_qos_regret_rate"] == pytest.approx(.5)
    assert result["aggregate"]["rank_or_execution_qos_regret_rate"] == pytest.approx(.5)


def test_candidate_support_requires_all_resolved_reference_edges():
    reference = np.zeros((3, 1, 1, 2), dtype=bool)
    reference[0, 0, 0, 0] = True
    reference[1, 0, 0, 0] = True
    reference[2, 0, 0, 1] = True
    candidate = reference.copy()
    candidate[1, 0, 0, 0] = False
    episodes = np.array([0, 0, 1])

    assert candidate_support_by_episode(
        candidate, reference, episodes, resolved=np.ones(3, dtype=bool)
    ) == {0: False, 1: True}


def test_episode_qos_regret_rejects_mismatched_episode_sets():
    with pytest.raises(ValueError, match="same non-empty episodes"):
        episode_qos_regret(
            [_row(0, .9, .8, .7, True)],
            [_row(1, .9, .8, .7, True)],
        )

