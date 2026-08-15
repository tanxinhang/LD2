from threading import Barrier
import time

import pytest

from uav_isac.coordination.owner_parallel_runtime import (
    execute_owner_jobs_concurrently,
)


def test_owner_jobs_run_concurrently_and_return_in_owner_order():
    barrier = Barrier(2)

    def job(value):
        def run():
            barrier.wait(timeout=1.0)
            time.sleep(0.01)
            return value
        return run

    result = execute_owner_jobs_concurrently(
        {7: job("seven"), 2: job("two")}, max_workers=2)

    assert [item.owner for item in result.measurements] == [2, 7]
    assert result.results == {2: "two", 7: "seven"}
    assert result.worker_count == 2
    assert result.wall_latency_s < result.sum_worker_wall_s
    assert result.process_cpu_s >= 0.0
    assert result.sum_worker_thread_cpu_s >= 0.0


def test_compute_energy_is_cpu_time_charge_not_claimed_meter_reading():
    result = execute_owner_jobs_concurrently({0: lambda: 3}, max_workers=1)
    assert result.charged_compute_energy_j(0.0) == 0.0
    assert result.charged_compute_energy_j(10.0) == pytest.approx(
        10.0 * result.process_cpu_s)
    with pytest.raises(ValueError, match="power"):
        result.charged_compute_energy_j(-1.0)


def test_owner_runtime_rejects_empty_or_invalid_worker_count():
    with pytest.raises(ValueError, match="at least one"):
        execute_owner_jobs_concurrently({})
    with pytest.raises(ValueError, match="positive"):
        execute_owner_jobs_concurrently({0: lambda: None}, max_workers=0)
