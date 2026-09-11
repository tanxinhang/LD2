"""Reproducible small-scale Top-M audit for correlation-aware exchange."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.coordination.correlation_aware_exchange import (  # noqa: E402
    correlation_aware_dual_pruned_exact_exchange,
)
from uav_isac.coordination.dependency_commit import (  # noqa: E402
    certify_best_dependency_commit,
)
from uav_isac.coordination.correlation_candidate_commit import (  # noqa: E402
    correlation_candidate_commit_layout,
)
from uav_isac.environment.communication import (  # noqa: E402
    InterUAVCommunicationModel,
)
from uav_isac.coordination.local_exchange_oracle import (  # noqa: E402
    role_owner_from_structure,
)


def _case(seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(int(seed))
    K, Q = 6, 4
    coefficient = np.zeros((K, K, Q), dtype=np.float64)
    support = np.zeros_like(coefficient, dtype=bool)
    tau = np.zeros_like(coefficient)
    nu = np.zeros_like(coefficient)
    for transmitter in range(3):
        for receiver in range(3, 6):
            for target in range(Q):
                edge = (transmitter, receiver, target)
                # Receiver 3 is the incumbent, receiver 4 is a feasible
                # improvement, and receiver 5 is often the stronger sensing
                # option but has no control reserve below. This forces Top-M
                # ranking to operate before a nontrivial physical gate.
                receiver_scale = {3: 0.65, 4: 1.15, 5: 1.45}[receiver]
                coefficient[edge] = (
                    receiver_scale * rng.lognormal(0.0, 0.7))
                support[edge] = True
                tau[edge] = rng.uniform(0.0, 5.0) / (64.0 * 15_625.0)
                nu[edge] = rng.uniform(-4.0, 4.0) / (16.0 * 64.0e-6)
    selected = np.zeros_like(support)
    for target in range(Q):
        selected[:3, 3, target] = True
    return {
        "initial_selected": selected,
        "coefficient_per_watt": coefficient,
        "candidate_mask": support,
        "initial_role": np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int8),
        "sensing_budget_w": np.full(K, 0.70, dtype=np.float64),
        "delay_s": tau,
        "doppler_hz": nu,
        "delay_size": 64,
        "doppler_size": 16,
        "delta_f_hz": 15_625.0,
        "symbol_time_s": 64.0e-6,
        "target_pair_limit": 3,
        "reports_per_receiver": 30,
        "weak_target_count": 4,
        # Isolate role-preserving owner/support exchange so communication
        # infeasibility is caused by the new owner closure, not an unrelated
        # N5 role swap.
        "neighborhoods": ("N6",),
    }, {
        "positions": np.asarray([
            [0.0, 0.0, 100.0],
            [20.0, 0.0, 100.0],
            [40.0, 0.0, 100.0],
            [100.0, 0.0, 100.0],
            [120.0 + rng.uniform(-10.0, 10.0), 0.0, 100.0],
            [180.0 + rng.uniform(-10.0, 10.0), 0.0, 100.0],
        ], dtype=np.float64),
        # Receiver 5 has no control reserve in this stress case. Owner
        # exchanges that need it must fail the physical vote round, while
        # receiver-4 alternatives remain admissible.
        "comm_power_w": np.asarray(
            [0.30, 0.30, 0.30, 0.30, 0.30, 0.0], dtype=np.float64),
    }


def _communication_model() -> InterUAVCommunicationModel:
    return InterUAVCommunicationModel(
        rate_bits_per_dim=[0, 4],
        header_bits=64,
        bandwidth_hz=500_000.0,
        deadline_s=0.005,
        processing_delay_s=2.0e-4,
        snr_threshold_db=0.0,
        antenna_gain_dbi=0.0,
        carrier_hz=28.0e9,
        tx_power_w=0.30,
        kT=4.0e-21,
        noise_figure_db=4.0,
        dt=0.1,
    )


def audit(cases: int, seed_offset: int, top_m_values: tuple[int, ...]) -> dict:
    started = time.perf_counter()
    samples = []
    communication_model = _communication_model()
    for index in range(int(cases)):
        inputs, physical = _case(int(seed_offset) + index)
        initial_selected = inputs["initial_selected"]
        initial_role = inputs["initial_role"]
        _, initial_owner = role_owner_from_structure(
            initial_selected, fallback_role=initial_role)
        physical_failure_reasons: dict[str, int] = {}

        def physical_certificate(move, sensing_power):
            certificate = certify_best_dependency_commit(
                initial_selected,
                initial_role,
                initial_owner,
                move,
                positions=physical["positions"],
                comm_power_w=physical["comm_power_w"],
                sensing_power_w=sensing_power,
                state_versions=np.ones(6, dtype=np.uint64),
                certificate_epoch_ids=np.ones(6, dtype=np.uint64),
                certificate_digests=np.ones(6, dtype=object),
                communication_model=communication_model,
                total_power_w=1.0,
                total_deadline_s=0.095,
                layout=correlation_candidate_commit_layout(6, 4),
            )
            for reason in certificate.reasons:
                physical_failure_reasons[reason] = (
                    physical_failure_reasons.get(reason, 0) + 1)
            return certificate.feasible

        def physical_prefilter(move):
            # The fixed-structure LP saturates each available row budget.
            # Hence the shared RF constraint and transport topology can be
            # certified before any Gram matrix or exact LP is evaluated.
            return physical_certificate(
                move, inputs["sensing_budget_w"])

        def physical_gate(move, _gain, _factors, result):
            # Defense-in-depth replay on the exact power matrix. A rejection
            # here would falsify the prefilter's sufficiency claim.
            return physical_certificate(move, result.power_w)

        exhaustive = correlation_aware_dual_pruned_exact_exchange(
            **inputs, top_m=100_000,
            pre_candidate_gate=physical_prefilter,
            exact_candidate_gate=physical_gate)
        variants = {}
        optimum = max(exhaustive.power_result.worst_deflection, 1.0e-15)
        for top_m in top_m_values:
            result = correlation_aware_dual_pruned_exact_exchange(
                **inputs, top_m=int(top_m),
                pre_candidate_gate=physical_prefilter,
                exact_candidate_gate=physical_gate)
            ratio = float(result.power_result.worst_deflection / optimum)
            variants[str(top_m)] = {
                "exact_recall": bool(abs(ratio - 1.0) <= 1.0e-9),
                "neighborhood_optimum_ratio": ratio,
                "exact_lp_count": int(result.exact_verification_count),
                "physical_gate_rejected_count": int(
                    result.exact_gate_rejected_count),
                "physical_prefilter_rejected_count": int(
                    result.physical_prefilter_rejected_count),
            }
        samples.append({
            "seed": int(seed_offset) + index,
            "exhaustive_exact_lp_count": int(
                exhaustive.exact_verification_count),
            "dual_upper_pruned_count": int(
                exhaustive.dual_upper_pruned_count),
            "eligible_candidate_count": int(exhaustive.candidate_count),
            "exhaustive_physical_gate_rejected_count": int(
                exhaustive.exact_gate_rejected_count),
            "exhaustive_physical_prefilter_rejected_count": int(
                exhaustive.physical_prefilter_rejected_count),
            "physical_failure_reasons": dict(sorted(
                physical_failure_reasons.items())),
            "top_m": variants,
        })
    summary = {}
    for top_m in top_m_values:
        items = [sample["top_m"][str(top_m)] for sample in samples]
        ratios = np.asarray([
            item["neighborhood_optimum_ratio"] for item in items
        ], dtype=np.float64)
        summary[str(top_m)] = {
            "exact_recall_rate": float(np.mean([
                item["exact_recall"] for item in items])),
            "mean_neighborhood_optimum_ratio": float(np.mean(ratios)),
            "minimum_neighborhood_optimum_ratio": float(np.min(ratios)),
            "mean_exact_lp_count": float(np.mean([
                item["exact_lp_count"] for item in items])),
            "mean_physical_gate_rejected_count": float(np.mean([
                item["physical_gate_rejected_count"] for item in items])),
            "mean_physical_prefilter_rejected_count": float(np.mean([
                item["physical_prefilter_rejected_count"] for item in items])),
        }
    summary["exhaustive"] = {
        "mean_exact_lp_count": float(np.mean([
            sample["exhaustive_exact_lp_count"] for sample in samples])),
        "mean_dual_upper_pruned_count": float(np.mean([
            sample["dual_upper_pruned_count"] for sample in samples])),
        "mean_eligible_candidate_count": float(np.mean([
            sample["eligible_candidate_count"] for sample in samples])),
        "mean_physical_gate_rejected_count": float(np.mean([
            sample["exhaustive_physical_gate_rejected_count"]
            for sample in samples])),
        "mean_physical_prefilter_rejected_count": float(np.mean([
            sample["exhaustive_physical_prefilter_rejected_count"]
            for sample in samples])),
    }
    return {
        "status": "DIAGNOSTIC_ONLY",
        "cases": int(cases),
        "seed_offset": int(seed_offset),
        "summary": summary,
        "elapsed_s": float(time.perf_counter() - started),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--top-m", default="4,8,12")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.cases < 1:
        raise ValueError("cases must be positive")
    top_m = tuple(int(value) for value in args.top_m.split(","))
    if not top_m or any(value < 1 for value in top_m):
        raise ValueError("top-m values must be positive")
    result = audit(args.cases, args.seed_offset, top_m)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
