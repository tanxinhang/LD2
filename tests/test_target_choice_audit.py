from copy import deepcopy

import numpy as np

from uav_isac.evaluation.target_choice_audit import (
    blend_sensing_allocation,
    summarize_joint_sensing_pair_interventions,
    summarize_sensing_choice_interventions,
    summarize_target_choice_interventions,
)


def test_target_choice_summary_reports_controllability_gate():
    row = {
        "worst_value_std": 0.04,
        "worst_value_range": 0.12,
        "worst_oracle_gap": 0.06,
        "actual_is_best": 0.0,
        "weak3_value_range": 0.08,
        "weak3_oracle_gap": 0.04,
    }
    episodes = [[deepcopy(row)] for _ in range(12)]
    summary = summarize_target_choice_interventions(
        episodes, bootstrap_samples=20)
    assert summary["eval_target_cf_intervention_count"] == 12
    assert summary["eval_target_cf_gate_pass"] is True
    assert summary["eval_target_cf_opportunity_gt_002_rate"] == 1.0


def test_target_choice_summary_rejects_small_action_spread():
    row = {
        "worst_value_std": 0.001,
        "worst_value_range": 0.004,
        "worst_oracle_gap": 0.002,
        "actual_is_best": 0.0,
        "weak3_value_range": 0.003,
        "weak3_oracle_gap": 0.001,
    }
    episodes = [[deepcopy(row)] for _ in range(12)]
    summary = summarize_target_choice_interventions(
        episodes, bootstrap_samples=20)
    assert summary["eval_target_cf_gate_pass"] is False


def test_target_choice_summary_stratifies_ineffective_duplicates():
    row = {
        "worst_value_std": 0.05,
        "worst_value_range": 0.15,
        "worst_oracle_gap": 0.07,
        "actual_is_best": 0.0,
        "weak3_value_range": 0.10,
        "weak3_oracle_gap": 0.05,
        "actual_is_ineffective_duplicate": 1.0,
    }
    episodes = [[deepcopy(row), deepcopy(row)] for _ in range(10)]
    summary = summarize_target_choice_interventions(
        episodes, bootstrap_samples=20)
    assert summary["eval_target_cf_ineffective_duplicate_count"] == 20
    assert summary["eval_target_cf_ineffective_repair_gate_pass"] is True


def test_sensing_residual_preserves_simplex_and_is_bounded():
    candidate = blend_sensing_allocation(
        np.array([0.1, 0.2, 0.3, 0.4]),
        target_index=1,
        residual_blend=0.25,
    )
    np.testing.assert_allclose(candidate, [0.075, 0.4, 0.225, 0.3])
    assert np.isclose(np.sum(candidate), 1.0)
    assert np.all(candidate >= 0.0)


def test_sensing_summary_uses_distinct_metric_namespace():
    row = {
        "worst_value_std": 0.04,
        "worst_value_range": 0.12,
        "worst_oracle_gap": 0.06,
        "actual_is_best": 0.0,
        "weak3_value_range": 0.08,
        "weak3_oracle_gap": 0.04,
    }
    episodes = [[deepcopy(row)] for _ in range(10)]
    summary = summarize_sensing_choice_interventions(
        episodes, bootstrap_samples=20)
    assert summary["eval_sensing_cf_intervention_count"] == 10
    assert summary["eval_sensing_cf_gate_pass"] is True
    assert "eval_target_cf_intervention_count" not in summary


def test_joint_sensing_summary_uses_distinct_metric_namespace():
    row = {
        "worst_value_std": 0.04,
        "worst_value_range": 0.12,
        "worst_oracle_gap": 0.06,
        "actual_is_best": 0.0,
        "weak3_value_range": 0.08,
        "weak3_oracle_gap": 0.04,
    }
    episodes = [[deepcopy(row)] for _ in range(10)]
    summary = summarize_joint_sensing_pair_interventions(
        episodes, bootstrap_samples=20)
    assert summary["eval_joint_sensing_cf_intervention_count"] == 10
    assert summary["eval_joint_sensing_cf_gate_pass"] is True
    assert "eval_sensing_cf_intervention_count" not in summary
