import pytest

from tools.crash_isolated_seed_eval import _merge_rows, _wilson_lower


def test_merge_recomputes_multiseed_metrics_from_episode_arrays():
    rows = [
        {
            "eval_episode_seeds": "[2]",
            "eval_episode_steady_P_D": "[0.9]",
            "eval_episode_weak3_P_D": "[0.8]",
            "eval_episode_worst_P_D": "[0.7]",
            "eval_steady_P_D": "0.9",
            "eval_weak3_P_D": "0.8",
            "eval_worst_P_D": "0.7",
            "eval_qos_feasible_rate": "1.0",
            "eval_qos_feasible_wilson_lcb": "0.2",
            "eval_qos_tol": "0.0",
            "eval_comm_bits_per_frame": "10.0",
            "eval_isac_max_power_balance_error_w": "1e-12",
        },
        {
            "eval_episode_seeds": "[1]",
            "eval_episode_steady_P_D": "[0.7]",
            "eval_episode_weak3_P_D": "[0.6]",
            "eval_episode_worst_P_D": "[0.5]",
            "eval_steady_P_D": "0.7",
            "eval_weak3_P_D": "0.6",
            "eval_worst_P_D": "0.5",
            "eval_qos_feasible_rate": "0.0",
            "eval_qos_feasible_wilson_lcb": "0.0",
            "eval_qos_tol": "0.0",
            "eval_comm_bits_per_frame": "30.0",
            "eval_isac_max_power_balance_error_w": "4e-12",
        },
    ]
    merged = _merge_rows(rows)
    assert merged["eval_episode_seeds"] == "[2, 1]"
    assert float(merged["eval_steady_P_D"]) == pytest.approx(0.8)
    assert float(merged["eval_weak3_P_D"]) == pytest.approx(0.7)
    assert float(merged["eval_worst_P_D"]) == pytest.approx(0.6)
    assert float(merged["eval_qos_feasible_rate"]) == pytest.approx(0.5)
    assert float(merged["eval_qos_feasible_wilson_lcb"]) == pytest.approx(
        _wilson_lower(1, 2))
    assert float(merged["eval_comm_bits_per_frame"]) == pytest.approx(20.0)
    assert float(merged["eval_isac_max_power_balance_error_w"]) == pytest.approx(
        4e-12)


def test_wilson_lower_is_bounded_and_increases_with_successes():
    values = [_wilson_lower(successes, 100) for successes in (0, 50, 100)]
    assert 0.0 <= values[0] < values[1] < values[2] <= 1.0


def test_merge_preserves_minimum_distance_and_sums_safety_counts():
    rows = [
        {
            "eval_inter_uav_min_distance_m": "31.0",
            "eval_inter_uav_continuous_min_distance_m": "29.0",
            "eval_movement_safety_fail_closed_calls": "2.0",
            "eval_movement_safety_fail_closed_outside_invariant_calls": "2.0",
        },
        {
            "eval_inter_uav_min_distance_m": "23.0",
            "eval_inter_uav_continuous_min_distance_m": "21.0",
            "eval_movement_safety_fail_closed_calls": "3.0",
            "eval_movement_safety_fail_closed_outside_invariant_calls": "1.0",
        },
    ]
    merged = _merge_rows(rows)
    assert float(merged["eval_inter_uav_min_distance_m"]) == 23.0
    assert float(merged[
        "eval_inter_uav_continuous_min_distance_m"]) == 21.0
    assert float(merged[
        "eval_movement_safety_fail_closed_calls"]) == 5.0
    assert float(merged[
        "eval_movement_safety_fail_closed_outside_invariant_calls"]) == 3.0
