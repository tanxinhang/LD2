"""Conservative task-level compute fusion for distributed UAV control.

This module accounts for compute cycles, compute energy, transport bits and
end-to-end completion time.  It never assumes that another UAV can evaluate a
task unless the task is explicitly migratable.  Link and processor queues are
serial and deterministic, making the latency estimate conservative for a
single-radio/single-processor implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ComputeNodeProfile:
    cycles_per_second: float
    joules_per_cycle: float


@dataclass(frozen=True)
class ComputeTask:
    task_id: str
    owner: int
    cycles: float
    input_bits: int
    output_bits: int
    deadline_s: float
    priority: float = 0.0
    migratable: bool = True
    extra_transport_delay_s: float = 0.0


@dataclass(frozen=True)
class ComputeAssignment:
    task_id: str
    owner: int
    executor: int
    completion_s: float
    met_deadline: bool
    communication_bits: int
    compute_energy_j: float
    communication_energy_j: float


@dataclass(frozen=True)
class ComputeFusionResult:
    assignments: tuple[ComputeAssignment, ...]
    makespan_s: float
    deadline_miss_rate: float
    offloaded_fraction: float
    total_communication_bits: int
    total_compute_energy_j: float
    total_communication_energy_j: float
    per_node_compute_cycles: tuple[float, ...]


def _validate_profile(profile: ComputeNodeProfile, index: int) -> None:
    rate = float(profile.cycles_per_second)
    energy = float(profile.joules_per_cycle)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"node {index} cycles_per_second must be finite and > 0")
    if not np.isfinite(energy) or energy < 0.0:
        raise ValueError(f"node {index} joules_per_cycle must be finite and >= 0")


def schedule_compute_tasks(
    tasks: Sequence[ComputeTask],
    nodes: Sequence[ComputeNodeProfile],
    link_rates_bps: np.ndarray,
    *,
    transmit_power_w: float | Sequence[float] = 0.25,
    allow_offload: bool = True,
    initial_node_ready_s: float | Sequence[float] = 0.0,
    link_processing_delay_s: float = 0.0,
    fixed_executors: Mapping[str, int] | None = None,
) -> ComputeFusionResult:
    """Schedule owner-local or migratable tasks on a distributed compute pool.

    Tasks are considered by descending priority, then earliest deadline and
    deterministic task id.  For each task, the scheduler selects the executor
    with the earliest conservative response-arrival time.  Remote execution
    includes request transport, executor queue/compute, and result transport.
    The function does not change sensing utility; callers must separately
    require every task needed by an analytical certificate to meet its
    deadline before accepting the corresponding action.
    """
    profiles = tuple(nodes)
    count = len(profiles)
    if count < 1:
        raise ValueError("at least one compute node is required")
    for index, profile in enumerate(profiles):
        _validate_profile(profile, index)
    rates = np.asarray(link_rates_bps, dtype=np.float64)
    if rates.shape != (count, count) or np.any(np.isnan(rates)):
        raise ValueError("link_rates_bps must have finite-or-inf shape (K,K)")
    if np.any(rates[~np.eye(count, dtype=bool)] <= 0.0):
        raise ValueError("off-diagonal link rates must be positive")
    powers = np.asarray(transmit_power_w, dtype=np.float64)
    if powers.ndim == 0:
        powers = np.full(count, float(powers), dtype=np.float64)
    powers = powers.reshape(-1)
    if (powers.shape != (count,) or np.any(~np.isfinite(powers))
            or np.any(powers < 0.0)):
        raise ValueError("transmit_power_w must be non-negative per node")
    initial_ready = np.asarray(initial_node_ready_s, dtype=np.float64)
    if initial_ready.ndim == 0:
        initial_ready = np.full(
            count, float(initial_ready), dtype=np.float64)
    initial_ready = initial_ready.reshape(-1)
    if (initial_ready.shape != (count,)
            or np.any(~np.isfinite(initial_ready))
            or np.any(initial_ready < 0.0)):
        raise ValueError("initial_node_ready_s must be finite and non-negative")
    processing_delay = float(link_processing_delay_s)
    if not np.isfinite(processing_delay) or processing_delay < 0.0:
        raise ValueError("link_processing_delay_s must be finite and non-negative")

    normalized: list[ComputeTask] = []
    identifiers: set[str] = set()
    for task in tasks:
        if task.task_id in identifiers:
            raise ValueError("task ids must be unique")
        identifiers.add(task.task_id)
        if not 0 <= int(task.owner) < count:
            raise ValueError(f"task {task.task_id} owner is out of range")
        values = np.asarray([
            task.cycles, task.deadline_s, task.priority,
            task.extra_transport_delay_s,
        ], dtype=np.float64)
        if (np.any(~np.isfinite(values)) or task.cycles <= 0.0
                or task.deadline_s < 0.0
                or task.extra_transport_delay_s < 0.0):
            raise ValueError(f"task {task.task_id} has invalid scalar fields")
        if (int(task.input_bits) != task.input_bits or task.input_bits < 0
                or int(task.output_bits) != task.output_bits
                or task.output_bits < 0):
            raise ValueError(f"task {task.task_id} bit counts are invalid")
        normalized.append(task)

    ordered = sorted(
        normalized,
        key=lambda task: (
            -float(task.priority), float(task.deadline_s), task.task_id),
    )
    node_ready = initial_ready.copy()
    link_ready = np.zeros((count, count), dtype=np.float64)
    node_cycles = np.zeros(count, dtype=np.float64)
    assignments: list[ComputeAssignment] = []

    for task in ordered:
        owner = int(task.owner)
        forced_executor = (
            None if fixed_executors is None
            else fixed_executors.get(task.task_id)
        )
        if forced_executor is not None and not 0 <= int(forced_executor) < count:
            raise ValueError(f"task {task.task_id} fixed executor is out of range")
        choices: list[tuple[float, int, float, float, float, float]] = []
        local_compute_start = float(node_ready[owner])
        local_compute_end = (
            local_compute_start
            + float(task.cycles) / profiles[owner].cycles_per_second)
        if forced_executor is None or int(forced_executor) == owner:
            choices.append((
                local_compute_end, owner, local_compute_start,
                local_compute_end, 0.0, 0.0))

        if bool(allow_offload) and bool(task.migratable):
            for executor in range(count):
                if executor == owner:
                    continue
                if (forced_executor is not None
                        and executor != int(forced_executor)):
                    continue
                request_rate = float(rates[owner, executor])
                response_rate = float(rates[executor, owner])
                request_start = float(link_ready[owner, executor])
                request_time = float(task.input_bits) / request_rate
                request_end = request_start + request_time + processing_delay
                compute_start = max(float(node_ready[executor]), request_end)
                compute_end = (
                    compute_start
                    + float(task.cycles)
                    / profiles[executor].cycles_per_second)
                response_start = max(
                    compute_end, float(link_ready[executor, owner]))
                response_time = float(task.output_bits) / response_rate
                completion = (
                    response_start + response_time + processing_delay
                    + float(task.extra_transport_delay_s))
                communication_energy = (
                    powers[owner] * request_time
                    + powers[executor] * response_time)
                choices.append((
                    completion, executor, compute_start, compute_end,
                    request_end, float(communication_energy)))

        if not choices:
            raise ValueError(
                f"task {task.task_id} fixed executor requires offload support")

        completion, executor, compute_start, compute_end, request_end, comm_e = min(
            choices, key=lambda item: (item[0], item[1]))
        executor = int(executor)
        communication_bits = 0
        if executor != owner:
            communication_bits = int(task.input_bits + task.output_bits)
            link_ready[owner, executor] = float(request_end)
            response_rate = float(rates[executor, owner])
            response_start = float(completion) - (
                float(task.output_bits) / response_rate + processing_delay
                + float(task.extra_transport_delay_s))
            link_ready[executor, owner] = float(completion)
            # The response cannot start before this compute completes.
            if response_start + 1.0e-15 < compute_end:
                raise RuntimeError("invalid response schedule")
        node_ready[executor] = float(compute_end)
        node_cycles[executor] += float(task.cycles)
        compute_energy = (
            float(task.cycles) * profiles[executor].joules_per_cycle)
        assignments.append(ComputeAssignment(
            task_id=task.task_id,
            owner=owner,
            executor=executor,
            completion_s=float(completion),
            met_deadline=bool(completion <= task.deadline_s + 1.0e-12),
            communication_bits=communication_bits,
            compute_energy_j=float(compute_energy),
            communication_energy_j=float(comm_e),
        ))

    if not assignments:
        return ComputeFusionResult(
            assignments=tuple(), makespan_s=0.0, deadline_miss_rate=0.0,
            offloaded_fraction=0.0, total_communication_bits=0,
            total_compute_energy_j=0.0, total_communication_energy_j=0.0,
            per_node_compute_cycles=tuple(0.0 for _ in profiles),
        )
    return ComputeFusionResult(
        assignments=tuple(assignments),
        makespan_s=float(max(item.completion_s for item in assignments)),
        deadline_miss_rate=float(np.mean([
            not item.met_deadline for item in assignments])),
        offloaded_fraction=float(np.mean([
            item.executor != item.owner for item in assignments])),
        total_communication_bits=int(sum(
            item.communication_bits for item in assignments)),
        total_compute_energy_j=float(sum(
            item.compute_energy_j for item in assignments)),
        total_communication_energy_j=float(sum(
            item.communication_energy_j for item in assignments)),
        per_node_compute_cycles=tuple(node_cycles.tolist()),
    )
