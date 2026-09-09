"""Behavior tests for ``uav_isac.evaluation.metrics``.

Audit 2026-09-04 §10 B1: ``evaluation/metrics.py`` had no dedicated tests
(only ``scripts/run_baselines.py`` referenced it).  These tests lock the
metric semantics used by baseline evaluation runs.
"""

import numpy as np
import pytest

from uav_isac.evaluation.metrics import (
    compute_avg_PD,
    compute_constraint_violation_rate,
    compute_cumulative_energy,
    compute_episode_metrics,
    compute_jain_fairness,
    compute_total_communication,
    compute_worst_PD,
)


def test_avg_and_worst_PD():
    pd_q = np.array([0.9, 0.7, 0.5])
    assert compute_avg_PD(pd_q) == pytest.approx(0.7)
    assert compute_worst_PD(pd_q) == pytest.approx(0.5)


def test_avg_and_worst_PD_single_target():
    pd_q = np.array([0.42])
    assert compute_avg_PD(pd_q) == pytest.approx(0.42)
    assert compute_worst_PD(pd_q) == pytest.approx(0.42)


def test_jain_fairness_is_one_for_equal_values():
    assert compute_jain_fairness(np.full(4, 0.6)) == pytest.approx(1.0)


def test_jain_fairness_single_target_is_one():
    assert compute_jain_fairness(np.array([0.4])) == pytest.approx(1.0)


def test_jain_fairness_empty_and_all_zero_are_one():
    # Empty and degenerate all-zero inputs decode as "perfectly fair",
    # matching the documented fallback (1.0).
    assert compute_jain_fairness(np.empty(0)) == pytest.approx(1.0)
    assert compute_jain_fairness(np.zeros(3)) == pytest.approx(1.0)


def test_jain_fairness_unequal_values_strictly_between_1_Q_and_1():
    pd_q = np.array([0.9, 0.1, 0.5])
    jain = compute_jain_fairness(pd_q)
    assert 1.0 / 3.0 < jain < 1.0


def test_cumulative_energy_is_total_depletion():
    initial = np.array([100.0, 50.0])
    final = np.array([70.0, 20.0])
    assert compute_cumulative_energy(initial, final) == pytest.approx(60.0)


def test_cumulative_energy_zero_when_unchanged():
    batteries = np.array([80.0, 40.0])
    assert compute_cumulative_energy(batteries, batteries.copy()) == 0.0


def test_total_communication_sums_solution_bits():
    class _Solution:
        def __init__(self, bits):
            self.total_bits = bits

    solutions = [_Solution(10.0), _Solution(5.5), _Solution(0.0)]
    assert compute_total_communication(solutions) == pytest.approx(15.5)


def test_total_communication_empty_is_zero():
    assert compute_total_communication([]) == 0.0


def test_constraint_violation_rate_empty_is_zero():
    assert compute_constraint_violation_rate([]) == 0.0


def test_constraint_violation_rate_is_violation_fraction():
    infos = [
        {"any_violation": True},
        {"any_violation": False},
        {"any_violation": True},
    ]
    assert compute_constraint_violation_rate(infos) == pytest.approx(2.0 / 3.0)


def test_episode_metrics_empty_history_is_empty_dict():
    assert compute_episode_metrics(
        [], [], [], np.zeros(2), np.zeros(2), []) == {}


def test_episode_metrics_returns_full_schema():
    class _Solution:
        def __init__(self, bits):
            self.total_bits = bits

    history = [
        np.array([0.8, 0.6, 0.4]),
        np.array([0.9, 0.7, 0.5]),
        np.array([0.95, 0.8, 0.6]),
    ]
    solutions = [_Solution(8.0), _Solution(9.0), _Solution(10.0)]
    infos = [{"any_violation": False}] * 3
    initial = np.full(2, 50.0)
    final = np.full(2, 30.0)
    rewards = [1.0, 2.0, 3.0]

    metrics = compute_episode_metrics(
        history, solutions, infos, initial, final, rewards)

    assert set(metrics) == {
        "avg_P_D", "worst_P_D",
        "final_avg_P_D", "final_worst_P_D",
        "steady_avg_P_D", "steady_worst_P_D",
        "jain_fairness",
        "cumulative_energy_J", "total_communication_bits",
        "constraint_violation_rate",
        "mean_team_reward", "total_team_reward",
    }
    assert metrics["avg_P_D"] == pytest.approx(
        np.mean([np.mean(pd) for pd in history]))
    assert metrics["worst_P_D"] == pytest.approx(
        np.mean([np.min(pd) for pd in history]))
    assert metrics["final_avg_P_D"] == pytest.approx(np.mean(history[-1]))
    assert metrics["final_worst_P_D"] == pytest.approx(np.min(history[-1]))
    assert metrics["cumulative_energy_J"] == pytest.approx(40.0)
    assert metrics["total_communication_bits"] == pytest.approx(27.0)
    assert metrics["constraint_violation_rate"] == 0.0
    assert metrics["mean_team_reward"] == pytest.approx(2.0)
    assert metrics["total_team_reward"] == pytest.approx(6.0)


def test_episode_metrics_steady_window_slices_tail():
    class _Solution:
        def __init__(self, bits):
            self.total_bits = bits

    history = [
        np.array([0.2, 0.2, 0.2]),   # approach transient
        np.array([0.8, 0.6, 0.4]),
        np.array([0.9, 0.8, 0.7]),
    ]
    solutions = [_Solution(b) for b in (1.0, 2.0, 3.0)]
    infos = [{"any_violation": False}] * 3
    batteries = np.full(2, 10.0)

    metrics = compute_episode_metrics(
        history, solutions, infos, batteries, batteries.copy(),
        [1.0, 1.0, 1.0], steady_window=2)

    tail_avg = np.mean([np.mean(pd) for pd in history[-2:]])
    tail_worst = np.mean([np.min(pd) for pd in history[-2:]])
    assert metrics["steady_avg_P_D"] == pytest.approx(tail_avg)
    assert metrics["steady_worst_P_D"] == pytest.approx(tail_worst)
    # The tail is strictly better than the full-horizon average: the
    # transient frame must not pollute the steady window.
    assert metrics["steady_avg_P_D"] > metrics["avg_P_D"]