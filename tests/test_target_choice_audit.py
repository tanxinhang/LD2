from copy import deepcopy

import numpy as np

from uav_isac.evaluation.target_choice_audit import (
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
