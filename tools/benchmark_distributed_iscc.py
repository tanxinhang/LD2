#!/usr/bin/env python
"""Shadow A/B for communication--sensing--compute candidate processing.

The sensing candidates and AI teacher are synthetic SCIR-style low-rank
experiments.  Compute cycles, weak-UAV throughput, bidirectional offload bits,
RF energy and a hard compute deadline are explicitly charged.  This is a
resource-closure diagnostic, not an end-to-end deployment claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_certified_ai_screening import _candidates, _train
from uav_isac.coordination.ai_candidate_screener import (
    CandidateScoreNetwork,
    certified_ai_screen,
)
from uav_isac.coordination.distributed_compute_fusion import (
    ComputeNodeProfile,
    ComputeTask,
    schedule_compute_tasks,
)
from uav_isac.coordination.progressive_information_transport import (
    InformationRefinementLayer,
    ProgressiveCandidate,
    certified_progressive_transport,
    quantized_psd_logdet_interval,
)


class FirstFeatureScorer(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


@dataclass(frozen=True)
class Case:
    name: str
    K: int
    Q: int


def _summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def _node_profiles(K: int, base_cycles_per_second: float):
    # Deterministic heterogeneous weak onboard processors.  Rotating the
    # factors avoids assigning every case's weakest device the same role.
    factors = np.resize(
        np.asarray([0.55, 0.70, 0.85, 1.00, 1.15, 0.65, 0.90, 1.10]), K)
    return tuple(
        ComputeNodeProfile(
            cycles_per_second=float(base_cycles_per_second * factor),
            joules_per_cycle=float(3.0e-10 * (0.85 + 0.30 * factor)),
        )
        for factor in factors
    )


def _owner_outcome(
    gain: np.ndarray,
    *,
    hard_ready: bool,
    chosen_index: int,
    performance_certified: bool,
) -> tuple[float, float, float, float]:
    """Separate physical readiness from an optional optimality proof."""
    chosen = int(chosen_index) if hard_ready else int(np.argmin(gain))
    return (
        float(hard_ready),
        float(hard_ready and performance_certified),
        float(hard_ready and not performance_certified),
        float(gain[chosen] / max(np.max(gain), 1.0e-12)),
    )


def _progressive_layers(
    candidate_ids: tuple[int, ...],
    gram: np.ndarray,
) -> tuple[ProgressiveCandidate, ...]:
    candidates = []
    for index in candidate_ids:
        matrix = np.asarray(gram[index], dtype=np.float64)
        previous_lower = 0.0
        previous_upper = np.inf
        layers = []
        previous_depth = 0
        for depth, label in (
            (4, "summary"),
            (8, "coarse_gram"),
            (12, "fine_gram"),
        ):
            lower, upper_value = quantized_psd_logdet_interval(
                matrix, bits_per_entry=depth)
            lower = max(previous_lower, lower)
            upper_value = min(previous_upper, upper_value)
            incremental_bits = (
                32 + 3 * depth if previous_depth == 0
                else 3 * (depth - previous_depth))
            layers.append(InformationRefinementLayer(
                lower, upper_value, incremental_bits, label))
            previous_lower, previous_upper = lower, upper_value
            previous_depth = depth
        exact = float(np.linalg.slogdet(np.eye(matrix.shape[0]) + matrix)[1])
        layers.append(InformationRefinementLayer(
            exact, exact, 3 * (64 - previous_depth),
            "full_sufficient_statistic"))
        candidates.append(ProgressiveCandidate(index, tuple(layers)))
    return tuple(candidates)


def _run_strategy(
    strategy: str,
    owner_data: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ai_model: CandidateScoreNetwork,
    nodes: tuple[ComputeNodeProfile, ...],
    *,
    deadline_s: float,
    link_rate_bps: float,
    exact_cycles: float,
    ai_cycles_per_candidate: float,
    bound_sort_cycles_per_compare: float,
    topk: int,
    request_bits: int,
    response_bits: int,
    offload_hold_frames: int,
    max_exact_evaluations: int | None,
) -> dict[str, float]:
    K = len(owner_data)
    tasks: list[ComputeTask] = []
    routing_tasks: list[ComputeTask] = []
    results = []
    selected_by_owner: list[tuple[int, ...]] = []
    initial_cycles = np.zeros(K, dtype=np.float64)
    exhaustive = strategy == "exhaustive_local"
    allow_offload = strategy in {
        "ai_fused_full", "ai_fused_progressive"}
    response_by_owner: list[int] = []
    full_response_by_owner: list[int] = []
    transport_result_by_owner = []
    for owner, (features, gram, gain, upper) in enumerate(owner_data):
        incumbent = int(np.argmin(gain))
        if exhaustive:
            selected = tuple(range(gain.size))
            results.append(None)
        elif strategy == "bound_local":
            bound_features = upper[:, None].astype(np.float32)
            result = certified_ai_screen(
                FirstFeatureScorer(), bound_features, upper,
                lambda indices, values=gain: values[indices],
                topk=1, incumbent_index=incumbent,
                max_exact_evaluations=max_exact_evaluations,
                verification_batch_size=16, device="cpu",
                execute_best_verified_on_budget_exhaustion=True,
            )
            selected = result.evaluated_indices
            results.append(result)
            initial_cycles[owner] = (
                bound_sort_cycles_per_compare * gain.size
                * max(np.log2(max(gain.size, 2)), 1.0))
        else:
            result = certified_ai_screen(
                ai_model, features, upper,
                lambda indices, values=gain: values[indices],
                topk=topk, incumbent_index=incumbent,
                max_exact_evaluations=max_exact_evaluations,
                verification_batch_size=16, device="cpu",
                execute_best_verified_on_budget_exhaustion=True,
            )
            selected = result.evaluated_indices
            results.append(result)
            initial_cycles[owner] = ai_cycles_per_candidate * gain.size
        selected_by_owner.append(tuple(int(index) for index in selected))
        # 32-bit shared scale plus three symmetric Gram entries refined from
        # 4 to 64 bits: 32 + 3*64 = 224 bits per candidate.
        full_response = int(224 * len(selected))
        transport_result = None
        response = int(response_bits)
        if strategy == "ai_fused_full":
            response = full_response
        elif strategy == "ai_fused_progressive":
            transport_result = certified_progressive_transport(
                _progressive_layers(tuple(selected), gram),
                incumbent_candidate_id=incumbent,
                link_rate_bps=link_rate_bps,
                deadline_s=deadline_s,
                transmit_power_w=0.25,
                bit_budget=full_response,
                processing_delay_s=2.0e-4,
                refinements_per_packet=4,
                execute_best_lower_on_budget_exhaustion=True,
            )
            response = int(transport_result.transmitted_bits)
        response_by_owner.append(response)
        full_response_by_owner.append(full_response)
        transport_result_by_owner.append(transport_result)
        extra_transport_delay = 0.0
        if transport_result is not None:
            single_packet_latency = response / link_rate_bps + 2.0e-4
            extra_transport_delay = max(
                transport_result.latency_s - single_packet_latency, 0.0)
        actual_task = ComputeTask(
            task_id=f"{owner}:batch",
            owner=owner,
            cycles=exact_cycles * len(selected),
            input_bits=request_bits,
            output_bits=response,
            deadline_s=deadline_s,
            priority=float(np.max(upper[list(selected)])),
            # The batch is addressed by one deterministic candidate-set hash;
            # public tokens let the executor reconstruct every member.
            migratable=True,
            extra_transport_delay_s=extra_transport_delay,
        )
        tasks.append(actual_task)
        routing_tasks.append(ComputeTask(
            task_id=actual_task.task_id,
            owner=actual_task.owner,
            cycles=actual_task.cycles,
            input_bits=actual_task.input_bits,
            output_bits=(
                full_response
                if strategy == "ai_fused_progressive"
                else actual_task.output_bits),
            deadline_s=actual_task.deadline_s,
            priority=actual_task.priority,
            migratable=actual_task.migratable,
        ))

    initial_ready = np.asarray([
        initial_cycles[index] / nodes[index].cycles_per_second
        for index in range(K)
    ])
    rates = np.full((K, K), float(link_rate_bps), dtype=np.float64)
    np.fill_diagonal(rates, np.inf)
    routed = schedule_compute_tasks(
        routing_tasks, nodes, rates, allow_offload=allow_offload,
        transmit_power_w=0.25, initial_node_ready_s=initial_ready,
        link_processing_delay_s=2.0e-4,
    )
    fixed_route = {
        item.task_id: item.executor for item in routed.assignments
    }
    scheduled = schedule_compute_tasks(
        tasks, nodes, rates, allow_offload=allow_offload,
        transmit_power_w=0.25, initial_node_ready_s=initial_ready,
        link_processing_delay_s=2.0e-4,
        fixed_executors=fixed_route,
    )
    deadline_map = {
        item.task_id: item.met_deadline for item in scheduled.assignments
    }
    assignment_map = {
        item.owner: item for item in scheduled.assignments
    }
    hard_feasibility = []
    certificates = []
    uncertified_executions = []
    utility_ratios = []
    for owner, (_features, _gram, gain, _upper) in enumerate(owner_data):
        if exhaustive:
            complete = bool(deadline_map[f"{owner}:batch"])
            hard_feasibility.append(float(complete))
            certificates.append(float(complete))
            uncertified_executions.append(0.0)
            chosen = int(np.argmax(gain)) if complete else int(np.argmin(gain))
            utility_ratios.append(float(
                gain[chosen] / max(np.max(gain), 1.0e-12)))
        else:
            transport_result = transport_result_by_owner[owner]
            assignment_ready = bool(deadline_map[f"{owner}:batch"])
            remote_progressive = bool(
                assignment_map[owner].executor != owner
                and transport_result is not None)
            transport_ready = bool(
                not remote_progressive
                or all(index >= 0 for index in transport_result.selected_layers))
            hard_ready = bool(assignment_ready and transport_ready)
            if remote_progressive:
                chosen_index = int(transport_result.chosen_candidate_id)
                performance_certified = bool(
                    results[owner].certified and transport_result.certified)
            else:
                chosen_index = int(results[owner].chosen_index)
                performance_certified = bool(results[owner].certified)
            hard, certificate, uncertified, ratio = _owner_outcome(
                gain,
                hard_ready=hard_ready,
                chosen_index=chosen_index,
                performance_certified=performance_certified,
            )
            hard_feasibility.append(hard)
            certificates.append(certificate)
            uncertified_executions.append(uncertified)
            utility_ratios.append(ratio)
    initial_energy = float(sum(
        initial_cycles[index] * nodes[index].joules_per_cycle
        for index in range(K)
    ))
    offloaded_owners = [
        owner for owner, item in assignment_map.items()
        if item.executor != owner
    ]
    actual_response_bits = float(sum(
        response_by_owner[owner] for owner in offloaded_owners))
    actual_full_response_bits = float(sum(
        full_response_by_owner[owner] for owner in offloaded_owners))
    progressive_results = [
        transport_result_by_owner[owner] for owner in offloaded_owners
        if transport_result_by_owner[owner] is not None
    ]
    return {
        "hard_feasibility_rate": float(np.mean(hard_feasibility)),
        "certificate_rate": float(np.mean(certificates)),
        "uncertified_execution_rate": float(np.mean(uncertified_executions)),
        "oracle_utility_ratio": float(np.mean(utility_ratios)),
        "makespan_ms": float(1000.0 * scheduled.makespan_s),
        "task_deadline_miss_rate": scheduled.deadline_miss_rate,
        "offloaded_fraction": scheduled.offloaded_fraction,
        "communication_bits": float(scheduled.total_communication_bits),
        "amortized_communication_bits_per_frame": float(
            scheduled.total_communication_bits / offload_hold_frames),
        "compute_energy_j": float(
            initial_energy + scheduled.total_compute_energy_j),
        "offload_rf_energy_j": scheduled.total_communication_energy_j,
        "exact_tasks": float(sum(map(len, selected_by_owner))),
        "transport_response_bits": actual_response_bits,
        "full_transport_response_bits": actual_full_response_bits,
        "transport_bit_ratio_to_full": float(
            actual_response_bits / actual_full_response_bits
            if actual_full_response_bits > 0.0 else 0.0),
        "full_transmission_fraction": float(np.mean([
            result.full_transmission_fraction
            for result in progressive_results
        ])) if progressive_results else 0.0,
    }


def benchmark(
    *,
    cases: list[Case],
    scenes: int,
    calibration_scenes: int,
    seed: int,
    deadline_ms: float,
    base_cycles_per_second: float,
    exact_cycles: float,
    link_rate_bps: float,
    topk: int,
    max_exact_evaluations: int | None,
) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_x, _gram, train_y, _upper = _candidates(rng, 20_000)
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained = CandidateScoreNetwork(train_x.shape[1], hidden_dim=64)
    training_ms = _train(
        trained, train_x, train_y, epochs=80, device=train_device)
    ai_model = trained.to("cpu").eval()
    base_strategies = (
        "exhaustive_local", "bound_local", "ai_local",
        "ai_fused_full", "ai_fused_progressive")
    case_reports = []
    for case in cases:
        nodes = _node_profiles(case.K, base_cycles_per_second)
        rows = {name: [] for name in (*base_strategies, "adaptive_iscc")}
        adaptive_modes = (
            "bound_local", "ai_local", "ai_fused_progressive")
        adaptive_selection = {name: 0 for name in adaptive_modes}

        def generate_scene():
            owner_data = []
            nominal = max((case.K - 1) * case.Q, 1)
            load = rng.lognormal(mean=-0.5 * 0.35**2, sigma=0.35, size=case.K)
            for owner in range(case.K):
                count = max(8, int(round(nominal * load[owner])))
                features, gram, gain, upper = _candidates(rng, count)
                owner_data.append((features, gram, gain, upper))
            return owner_data, float(np.max(load))

        def evaluate_scene(owner_data):
            samples = {}
            for strategy in base_strategies:
                samples[strategy] = _run_strategy(
                    strategy, owner_data, ai_model, nodes,
                    deadline_s=deadline_ms / 1000.0,
                    link_rate_bps=link_rate_bps,
                    exact_cycles=exact_cycles,
                    ai_cycles_per_candidate=4_736.0,
                    bound_sort_cycles_per_compare=24.0,
                    topk=topk,
                    request_bits=64,
                    response_bits=64,
                    offload_hold_frames=5,
                    max_exact_evaluations=max_exact_evaluations,
                )
            return samples

        # Offline calibration is disjoint from the reported evaluation scenes.
        # Current-scene outcomes never enter routing; only the observable load
        # bin selects a frozen mode table.
        calibration = []
        for _ in range(calibration_scenes):
            owner_data, load_metric = generate_scene()
            calibration.append((load_metric, evaluate_scene(owner_data)))
        cuts = np.quantile(
            [item[0] for item in calibration], [1.0 / 3.0, 2.0 / 3.0])

        def bin_index(load_metric: float) -> int:
            return int(np.searchsorted(cuts, load_metric, side="right"))

        frozen_mode_by_bin: list[str] = []
        for target_bin in range(3):
            records = [
                samples for load_metric, samples in calibration
                if bin_index(load_metric) == target_bin
            ]
            if not records:
                frozen_mode_by_bin.append("bound_local")
                continue
            summaries = {
                mode: {
                    key: _summary([record[mode][key] for record in records])
                    for key in records[0][mode]
                }
                for mode in adaptive_modes
            }

            def calibration_feasible(mode: str) -> bool:
                sample = summaries[mode]
                return bool(
                    sample["hard_feasibility_rate"]["mean"] >= 0.99
                    and sample["oracle_utility_ratio"]["mean"] >= 0.99
                    and sample["makespan_ms"]["p95"] <= deadline_ms
                    and sample["task_deadline_miss_rate"]["mean"] <= 0.01
                    and sample[
                        "amortized_communication_bits_per_frame"
                    ]["mean"] <= 320.0
                )

            valid_modes = [
                mode for mode in adaptive_modes
                if calibration_feasible(mode)
            ]
            if valid_modes:
                selected_mode = min(
                    valid_modes,
                    key=lambda mode: (
                        summaries[mode]["compute_energy_j"]["mean"]
                        + summaries[mode]["offload_rf_energy_j"]["mean"],
                        summaries[mode]["makespan_ms"]["p95"],
                        mode,
                    ),
                )
            else:
                selected_mode = max(
                    adaptive_modes,
                    key=lambda mode: (
                        summaries[mode]["oracle_utility_ratio"]["mean"],
                        summaries[mode]["hard_feasibility_rate"]["mean"],
                        -summaries[mode]["makespan_ms"]["p95"],
                    ),
                )
            frozen_mode_by_bin.append(selected_mode)

        for _ in range(scenes):
            owner_data, load_metric = generate_scene()
            samples = evaluate_scene(owner_data)
            for strategy in base_strategies:
                rows[strategy].append(samples[strategy])
            selected_name = frozen_mode_by_bin[bin_index(load_metric)]
            selected = samples[selected_name]
            adaptive_selection[selected_name] += 1
            rows["adaptive_iscc"].append(dict(selected))
        summarized = {}
        for strategy, samples in rows.items():
            summarized[strategy] = {
                key: _summary([sample[key] for sample in samples])
                for key in samples[0]
            }
        summarized["adaptive_iscc"]["selection_rate"] = {
            name: float(count / scenes)
            for name, count in adaptive_selection.items()
        }
        fused = summarized["adaptive_iscc"]
        gate = {
            "hard_feasibility_rate_ge_0_99": (
                fused["hard_feasibility_rate"]["mean"] >= 0.99),
            "oracle_utility_ratio_ge_0_99": (
                fused["oracle_utility_ratio"]["mean"] >= 0.99),
            "p95_makespan_within_deadline": (
                fused["makespan_ms"]["p95"] <= deadline_ms),
            "task_deadline_miss_rate_le_0_01": (
                fused["task_deadline_miss_rate"]["mean"] <= 0.01),
            "offload_bits_per_frame_le_320": (
                fused["amortized_communication_bits_per_frame"]["mean"]
                <= 320.0),
        }
        case_reports.append({
            "case": asdict(case),
            "causal_router": {
                "observable": "current candidate-load bin only",
                "load_bin_upper_edges": [float(value) for value in cuts],
                "frozen_mode_by_bin": frozen_mode_by_bin,
                "uses_current_scene_outcomes": False,
            },
            "strategies": summarized,
            "adaptive_iscc_gate": gate,
            "passes_all_gates": bool(all(gate.values())),
        })
    return {
            "status": "SHADOW_ISCC_RESOURCE_MODEL_ONLY",
            "physical_action_mutation": False,
            "certificate_policy": (
                "hard_feasibility_gates_execution; performance_optimality_"
                "certificate_is_optional_anytime_diagnostic"),
        "teacher_data": "synthetic_SCIR_low_rank_candidates",
        "seed": seed,
        "scenes": scenes,
        "calibration_scenes": calibration_scenes,
        "assumptions": {
            "compute_deadline_ms": deadline_ms,
            "base_cycles_per_second": base_cycles_per_second,
            "exact_candidate_cycles": exact_cycles,
            "link_rate_bps": link_rate_bps,
            "offload_request_bits": 64,
            "offload_response_bits": 64,
            "offload_assignment_hold_frames": 5,
            "max_exact_evaluations_per_owner": max_exact_evaluations,
            "link_processing_delay_s_per_packet": 2.0e-4,
            "offload_batch_semantics": (
                "one deterministic candidate-set hash and one certified "
                "batch result per owner/executor"),
            "ai_macs_per_candidate_charged_as_cycles": 4_736,
            "ai_training_device": str(train_device),
            "ai_training_ms": training_ms,
        },
        "cases": case_reports,
        "interpretation": (
            "Only compute/transport closure is tested.  Detection probability "
            "and communication QoS remain those of the strict pilot until a "
            "physical H/R candidate adapter drives deployed actions."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--calibration-scenes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--deadline-ms", type=float, default=50.0)
    parser.add_argument("--base-cycles-per-second", type=float, default=1.2e8)
    parser.add_argument("--exact-cycles", type=float, default=250_000.0)
    parser.add_argument("--link-rate-bps", type=float, default=100_000.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--max-exact-evaluations", type=int, default=0,
        help=("per-owner anytime performance-proof budget; 0 keeps the "
              "historical unlimited verifier"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if (args.scenes < 1 or args.calibration_scenes < 3
            or args.deadline_ms <= 0.0
            or args.base_cycles_per_second <= 0.0
            or args.exact_cycles <= 0.0 or args.link_rate_bps <= 0.0
            or args.topk < 1 or args.max_exact_evaluations < 0):
        parser.error("all resource arguments must be positive")
    report = benchmark(
        cases=[
            Case("k8_q8", 8, 8),
            Case("k8_q12", 8, 12),
            Case("k12_q8", 12, 8),
            Case("k12_q12", 12, 12),
        ],
        scenes=args.scenes,
        calibration_scenes=args.calibration_scenes,
        seed=args.seed,
        deadline_ms=args.deadline_ms,
        base_cycles_per_second=args.base_cycles_per_second,
        exact_cycles=args.exact_cycles,
        link_rate_bps=args.link_rate_bps,
        topk=args.topk,
        max_exact_evaluations=(
            None if args.max_exact_evaluations == 0
            else args.max_exact_evaluations),
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
