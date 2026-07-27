import numpy as np

from uav_isac.evaluation.responsibility_audit import (
    classify_responsibility_frame,
    node_removal_marginal_pd,
    summarize_responsibility_episodes,
)
from uav_isac.utils.types import DeflectionEntry


def _entry(i, j, q, d_eff):
    return DeflectionEntry(
        i=i, j=j, q=q, tau=0.0, nu=0.0, alpha=1.0,
        d_raw=d_eff, g_dd=1.0, chi_rep=1.0, d_eff=d_eff)


def test_node_removal_marginal_identifies_selected_endpoints():
    entries = [_entry(0, 1, 0, 20.0), _entry(2, 1, 1, 10.0)]
    result = node_removal_marginal_pd(
        [(0, 1, 0), (2, 1, 1)], entries, 3, 2, 0.001)
    marginal = result["node_marginal_pd"]
    assert marginal[0, 0] > 0.0
    assert marginal[1, 0] > 0.0
    assert marginal[2, 0] == 0.0
    assert marginal[2, 1] > 0.0
    assert marginal[0, 1] == 0.0


def test_responsibility_classification_separates_useful_and_wasted_overlap():
    entries = [_entry(0, 1, 0, 20.0), _entry(2, 3, 1, 20.0)]
    record = classify_responsibility_frame(
        target_choices=np.array([0, 0, 0, 1]),
        active_movement=np.ones(4, dtype=bool),
        selected_set=[(0, 1, 0), (2, 3, 1)],
        deflection_entries=entries,
        num_targets=2,
        p_fa=0.001,
    )
    # UAV0/UAV1 form a useful bistatic pair on target 0; UAV2 is a
    # zero-marginal third mover toward that same target.
    assert record["effective_overlap_target_rate"] == 0.5
    assert record["ineffective_duplicate_target_rate"] == 0.5
    assert record["ineffective_duplicate_responsibility_rate"] == 1.0 / 3.0
    assert record["productive_duplicate_marginal_pd"] > 0.0


def test_episode_cluster_summary_uses_complete_five_frame_windows():
    pd = np.array([
        [0.2, 0.8],
        [0.3, 0.8],
        [0.4, 0.8],
        [0.5, 0.8],
        [0.6, 0.8],
        [0.7, 0.8],
    ])
    base = {
        "frame_index": 0,
        "pre_pd": np.array([0.1, 0.8]),
        "duplicate_target_rate": 0.5,
        "duplicate_frame": 1.0,
        "duplicate_responsibility_rate": 1.0,
        "ineffective_duplicate_responsibility_rate": 0.5,
        "ineffective_duplicate_target_rate": 0.5,
        "effective_overlap_target_rate": 0.5,
        "same_role_overlap_target_rate": 0.0,
        "uncovered_target_rate": 0.5,
        "uncovered_but_sensed_target_rate": 0.0,
        "uncovered_and_unsensed_target_rate": 0.5,
        "productive_duplicate_marginal_pd": 0.2,
        "ineffective_duplicate_marginal_pd": 0.0,
        "single_responsibility_marginal_pd": 0.1,
        "mean_node_marginal_pd": 0.1,
    }
    summary = summarize_responsibility_episodes(
        [[base]], [pd], horizon=5, bootstrap_samples=4)
    assert summary["eval_resp_boundary_count"] == 1
    assert summary["eval_resp_duplicate_frame"] == 1.0
    assert summary["eval_resp_gate_pass"] is False
