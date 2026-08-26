import pytest

from tools.audit_g2_1a_batch import audit_row, paired_delta


def _row(seeds="[1, 2]", values="[1.0, 0.5]"):
    return {
        "eval_episode_seeds": seeds,
        "eval_episode_steady_P_D": values,
        "eval_episode_weak3_P_D": values,
        "eval_episode_worst_P_D": values,
        "eval_steady_P_D": "0.75",
        "eval_weak3_P_D": "0.75",
        "eval_worst_P_D": "0.75",
        "eval_qos_tol": "0.0",
        "eval_qos_feasible_rate": "0.5",
        "eval_qos_feasible_wilson_lcb": "0.09452865480086614",
        "eval_comm_bits_per_frame": "100",
        "eval_comm_mean_latency_s": "0.001",
        "eval_comm_delivery_rate": "0.8",
        "eval_comm_deadline_violation_rate": "0.2",
        "eval_isac_sensing_power_w_per_frame": "0.1004",
        "eval_isac_comm_power_w_per_frame": "0.2",
        "eval_isac_max_power_balance_error_w": "0.8",
    }


def test_audit_recomputes_and_marks_small_batch_in_progress():
    result = audit_row("4x4", _row())
    assert result["status"] == "IN_PROGRESS"
    assert result["all_checks_pass"]
    assert result["metrics"]["qos_successes"] == 1


def test_paired_delta_requires_identical_order_and_uses_ce_minus_baseline():
    baseline = _row(values="[0.2, 0.4]")
    ce = _row(values="[0.3, 0.1]")
    result = paired_delta(baseline, ce)
    assert result["steady"]["mean_delta"] == pytest.approx(-0.1)
    assert result["steady"]["ce_wins"] == 1
    ce["eval_episode_seeds"] = "[2, 1]"
    with pytest.raises(ValueError, match="identical ordered seeds"):
        paired_delta(baseline, ce)
