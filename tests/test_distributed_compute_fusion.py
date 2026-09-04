"""Resource-closure tests for distributed UAV compute fusion."""

import numpy as np
import pytest

from uav_isac.coordination.distributed_compute_fusion import (
    ComputeNodeProfile,
    ComputeTask,
    schedule_compute_tasks,
)


def _nodes():
    return (
        ComputeNodeProfile(10.0, 2.0e-9),
        ComputeNodeProfile(100.0, 1.0e-9),
    )


def test_fast_neighbor_is_used_only_when_transport_improves_completion():
    tasks = [ComputeTask("a", 0, 10.0, 1, 1, 2.0)]
    fast = schedule_compute_tasks(tasks, _nodes(), np.full((2, 2), np.inf))
    slow = schedule_compute_tasks(tasks, _nodes(), np.asarray([
        [np.inf, 1.0], [1.0, np.inf],
    ]))

    assert fast.assignments[0].executor == 1
    assert fast.assignments[0].completion_s == pytest.approx(0.1)
    assert slow.assignments[0].executor == 0


def test_nonmigratable_task_never_leaves_its_data_owner():
    result = schedule_compute_tasks(
        [ComputeTask("private", 0, 10.0, 0, 0, 2.0, migratable=False)],
        _nodes(), np.full((2, 2), np.inf),
    )
    assert result.assignments[0].executor == 0
    assert result.total_communication_bits == 0


def test_offload_accounts_for_bidirectional_bits_and_both_energy_sources():
    result = schedule_compute_tasks(
        [ComputeTask("public", 0, 10.0, 20, 30, 2.0)],
        _nodes(), np.full((2, 2), 1_000.0), transmit_power_w=0.5,
    )
    item = result.assignments[0]
    assert item.executor == 1
    assert item.communication_bits == 50
    assert item.compute_energy_j == pytest.approx(10.0e-9)
    assert item.communication_energy_j == pytest.approx(0.025)


def test_deadline_miss_is_reported_instead_of_hidden_by_offload():
    result = schedule_compute_tasks(
        [ComputeTask("late", 0, 10.0, 100, 100, 0.05)],
        _nodes(), np.full((2, 2), 1_000.0),
    )
    assert result.deadline_miss_rate == 1.0
    assert not result.assignments[0].met_deadline


def test_invalid_zero_rate_is_rejected():
    with pytest.raises(ValueError, match="off-diagonal"):
        schedule_compute_tasks(
            [], _nodes(), np.asarray([[np.inf, 0.0], [1.0, np.inf]]))


def test_fixed_executor_replays_a_route_without_reoptimizing_on_result_size():
    tasks = [ComputeTask("a", 0, 10.0, 1, 1, 2.0)]
    rates = np.full((2, 2), np.inf)
    result = schedule_compute_tasks(
        tasks, _nodes(), rates, fixed_executors={"a": 1})
    assert result.assignments[0].executor == 1
