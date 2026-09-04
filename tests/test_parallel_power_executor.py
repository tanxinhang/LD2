"""Determinism and fail-closed tests for private-LP process execution."""

from time import perf_counter

import numpy as np

from config.params import load_config
from uav_isac.coordination.maxmin_power import (
    replicated_local_row_maxmin_power,
    solve_fixed_structure_maxmin_power_lp,
    sparse_harmonic_row_power,
)
from uav_isac.coordination.parallel_power_executor import (
    ParallelPowerBatch,
    ReplicatedPowerProcessExecutor,
)
from uav_isac.environment.env_core import EnvironmentCore


class _DeterministicExecutor:
    worker_count = 2

    def solve_many(self, gains, budget):
        started = perf_counter()
        solved = tuple(
            solve_fixed_structure_maxmin_power_lp(gain, budget)
            for gain in gains)
        return ParallelPowerBatch(
            results=solved,
            node_compute_time_s=tuple(0.001 for _ in solved),
            batch_wall_time_s=perf_counter() - started,
            worker_count=self.worker_count,
        )


class _FailingExecutor:
    def solve_many(self, gains, budget):
        raise RuntimeError("injected worker failure")


def _private_views():
    base = np.asarray([
        [2.0, 0.4, 0.2],
        [0.3, 1.5, 0.8],
        [0.2, 0.5, 1.8],
    ])
    return np.stack((base, base * 1.1, base * 0.9)), np.asarray(
        [0.8, 0.7, 0.9])


def test_parallel_dispatch_is_exactly_equivalent_to_serial_assembly():
    views, budget = _private_views()
    serial = replicated_local_row_maxmin_power(views, budget)
    parallel = replicated_local_row_maxmin_power(
        views, budget, parallel_executor=_DeterministicExecutor())

    np.testing.assert_array_equal(parallel.power_w, serial.power_w)
    np.testing.assert_array_equal(
        parallel.local_full_power_w, serial.local_full_power_w)
    np.testing.assert_array_equal(parallel.local_prices, serial.local_prices)
    assert parallel.parallel_execution_used
    assert parallel.parallel_worker_count == 2
    assert not parallel.parallel_fallback_to_serial


def test_worker_failure_falls_back_to_identical_serial_solution():
    views, budget = _private_views()
    serial = replicated_local_row_maxmin_power(views, budget)
    fallback = replicated_local_row_maxmin_power(
        views,
        budget,
        parallel_executor=_FailingExecutor(),
        parallel_fallback_to_serial=True,
    )

    np.testing.assert_array_equal(fallback.power_w, serial.power_w)
    np.testing.assert_array_equal(
        fallback.local_full_power_w, serial.local_full_power_w)
    assert not fallback.parallel_execution_used
    assert fallback.parallel_fallback_to_serial


def test_worker_failure_uses_certified_cached_incumbent_without_new_lp():
    views, budget = _private_views()
    initial = replicated_local_row_maxmin_power(views, budget)
    fallback = replicated_local_row_maxmin_power(
        views,
        budget,
        previous_local_power_w=initial.local_full_power_w,
        previous_local_prices=initial.local_prices,
        previous_local_cache_valid=initial.local_cache_valid,
        parallel_executor=_FailingExecutor(),
        parallel_fallback_to_serial=True,
        parallel_failure_mode="cached_or_uniform",
        deadline_incumbent_relative_tolerance=0.05,
    )

    np.testing.assert_array_equal(fallback.power_w, initial.power_w)
    assert not np.any(fallback.local_resolved)
    assert fallback.parallel_fallback_to_serial
    assert fallback.deadline_incumbent_used_fraction == 1.0
    assert fallback.deadline_incumbent_certificate_fraction == 1.0
    assert fallback.deadline_uniform_fallback_fraction == 0.0


def test_worker_failure_without_cache_uses_budget_feasible_uniform_rows():
    views, budget = _private_views()
    fallback = replicated_local_row_maxmin_power(
        views,
        budget,
        parallel_executor=_FailingExecutor(),
        parallel_fallback_to_serial=True,
        parallel_failure_mode="cached_or_uniform",
    )

    np.testing.assert_allclose(
        fallback.power_w,
        budget[:, None] * np.full((3, 3), 1.0 / 3.0),
        atol=1.0e-15,
    )
    assert not np.any(fallback.local_resolved)
    assert fallback.deadline_incumbent_used_fraction == 0.0
    assert fallback.deadline_incumbent_certificate_fraction == 0.0
    assert fallback.deadline_uniform_fallback_fraction == 1.0


def test_latched_worker_failure_keeps_using_incumbent_without_executor():
    views, budget = _private_views()
    initial = replicated_local_row_maxmin_power(views, budget)
    fallback = replicated_local_row_maxmin_power(
        views,
        budget,
        previous_local_power_w=initial.local_full_power_w,
        previous_local_prices=initial.local_prices,
        previous_local_cache_valid=initial.local_cache_valid,
        parallel_failure_mode="cached_or_uniform",
        force_deadline_fallback=True,
    )

    np.testing.assert_array_equal(fallback.power_w, initial.power_w)
    assert fallback.parallel_fallback_to_serial
    assert fallback.deadline_incumbent_used_fraction == 1.0
    assert not np.any(fallback.local_resolved)


def test_harmonic_deadline_fallback_equalizes_safe_row_contributions():
    views, public_budget = _private_views()
    executed_budget = 0.5 * public_budget
    safe_gain = np.asarray([
        [2.0, 1.0, 0.5],
        [0.4, 0.8, 1.6],
        [1.5, 0.75, 0.3],
    ])
    fallback = replicated_local_row_maxmin_power(
        views,
        public_budget,
        executed_budget,
        parallel_executor=_FailingExecutor(),
        parallel_fallback_to_serial=True,
        parallel_failure_mode="cached_or_harmonic",
        deadline_safe_row_gain_per_watt=safe_gain,
    )

    expected = np.empty_like(fallback.power_w)
    for node in range(public_budget.size):
        inverse = 1.0 / safe_gain[node]
        expected[node] = executed_budget[node] * inverse / np.sum(inverse)
    np.testing.assert_allclose(fallback.power_w, expected, rtol=1.0e-14)
    contributions = safe_gain * fallback.power_w
    np.testing.assert_allclose(
        contributions,
        np.repeat(contributions[:, :1], contributions.shape[1], axis=1),
        rtol=1.0e-14,
        atol=1.0e-15,
    )
    assert fallback.deadline_harmonic_fallback_fraction == 1.0
    assert fallback.deadline_incumbent_used_fraction == 0.0
    assert fallback.deadline_uniform_fallback_fraction == 0.0
    assert fallback.deadline_composable_deflection_floor == float(
        np.min(np.sum(contributions, axis=0)))


def test_sparse_harmonic_fallback_covers_union_of_reachable_targets():
    views, budget = _private_views()
    safe_gain = np.asarray([
        [2.0, 1.0, 0.0],
        [0.0, 0.5, 1.5],
        [1.0, 0.0, 2.0],
    ])
    fallback = replicated_local_row_maxmin_power(
        views,
        budget,
        parallel_executor=_FailingExecutor(),
        parallel_failure_mode="cached_or_harmonic",
        deadline_safe_row_gain_per_watt=safe_gain,
    )

    assert fallback.deadline_harmonic_fallback_fraction == 1.0
    assert np.all(fallback.power_w[safe_gain == 0.0] == 0.0)
    target_lower = np.sum(safe_gain * fallback.power_w, axis=0)
    assert np.all(target_lower > 0.0)
    assert fallback.deadline_composable_deflection_floor == float(
        np.min(target_lower))


def test_sparse_harmonic_helper_is_row_feasible_and_support_local():
    gain = np.asarray([
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    budget = np.asarray([0.6, 0.4])
    fallback = np.asarray([
        [0.2, 0.2, 0.2],
        [0.3, 0.1, 0.0],
    ])
    power = sparse_harmonic_row_power(
        gain,
        budget,
        zero_support_fallback_power_w=fallback,
    )

    np.testing.assert_allclose(np.sum(power, axis=1), budget)
    np.testing.assert_allclose(gain[0, :2] * power[0, :2], [0.4, 0.4])
    assert power[0, 2] == 0.0
    np.testing.assert_allclose(power[1], [0.3, 0.1, 0.0])


def test_harmonic_mode_keeps_current_certified_incumbent():
    views, budget = _private_views()
    initial = replicated_local_row_maxmin_power(views, budget)
    fallback = replicated_local_row_maxmin_power(
        views,
        budget,
        previous_local_power_w=initial.local_full_power_w,
        previous_local_prices=initial.local_prices,
        previous_local_cache_valid=initial.local_cache_valid,
        parallel_executor=_FailingExecutor(),
        parallel_failure_mode="cached_or_harmonic",
        deadline_incumbent_relative_tolerance=0.05,
        deadline_safe_row_gain_per_watt=np.ones((3, 3)),
    )

    np.testing.assert_array_equal(fallback.power_w, initial.power_w)
    assert fallback.deadline_incumbent_certificate_fraction == 1.0
    assert fallback.deadline_harmonic_fallback_fraction == 0.0


def test_real_isolated_process_executor_preserves_lp_solution():
    views, budget = _private_views()
    expected = tuple(
        solve_fixed_structure_maxmin_power_lp(view, budget)
        for view in views)
    with ReplicatedPowerProcessExecutor(worker_count=2) as executor:
        assert executor.warm_up(3, 3) > 0.0
        batch = executor.solve_many(tuple(views), budget)

    assert batch.worker_count == 2
    assert batch.batch_wall_time_s >= 0.0
    for reference, actual in zip(expected, batch.results):
        np.testing.assert_array_equal(actual.power_w, reference.power_w)
        assert actual.worst_deflection == reference.worst_deflection


def test_real_process_executor_preserves_reserve_constrained_lp_solution():
    views, budget = _private_views()
    reserves = (
        np.asarray([0.2, 0.2, 0.2]),
        np.asarray([0.25, 0.2, 0.2]),
        np.asarray([0.2, 0.25, 0.2]),
    )
    expected = tuple(
        solve_fixed_structure_maxmin_power_lp(
            view, budget, minimum_deflection=reserve)
        for view, reserve in zip(views, reserves)
    )
    with ReplicatedPowerProcessExecutor(worker_count=2) as executor:
        executor.warm_up(3, 3)
        batch = executor.solve_many(
            tuple(views), budget, minimum_deflections=reserves)

    for reference, actual in zip(expected, batch.results):
        np.testing.assert_array_equal(actual.power_w, reference.power_w)
        np.testing.assert_array_equal(actual.deflection, reference.deflection)


def test_environment_rejects_parallel_power_without_distributed_solver():
    cfg = load_config("config/default.yaml")
    cfg.marl.distributed_replicated_power_process_parallel_enabled = True
    cfg.marl.distributed_replicated_power_enabled = False

    try:
        EnvironmentCore(cfg)
    except ValueError as error:
        assert "requires distributed replicated power" in str(error)
    else:
        raise AssertionError("invalid parallel configuration was accepted")
