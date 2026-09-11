"""Falsifiable Correlation x Budget audit for the scientific core.

This diagnostic intentionally removes trajectory, atomic commit and provenance
effects.  It asks one narrow question: when high-quality evidence becomes
redundant, does conditional-Deflection selection outperform a strong
correlation-unaware baseline at the same transmitted bits while approaching a
small-scale exhaustive reference?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.physical.correlated_soft_evidence import (
    conditional_deflection_greedy,
    correlation_unaware_greedy,
    exhaustive_budgeted_reference,
    optimal_linear_soft_fusion,
)
from uav_isac.physical.detection import compute_detection_probabilities


def _controlled_model(rho: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return four independent view pairs with controlled within-pair overlap."""
    correlation = float(rho)
    if not 0.0 <= correlation < 1.0:
        raise ValueError("rho must lie in [0,1)")
    # Adjacent sources have similar local quality and a common view.  Pairwise
    # lognormal scaling adds case diversity without destroying the intended
    # alignment between a strong source and its redundant partner.
    local_deflection = np.array(
        [8.0, 7.2, 6.0, 5.6, 4.8, 4.4, 3.6, 3.2], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    pair_scale = np.exp(rng.normal(0.0, 0.08, size=4))
    delta = np.sqrt(local_deflection) * np.repeat(pair_scale, 2)
    covariance = np.eye(8, dtype=np.float64)
    for start in range(0, 8, 2):
        covariance[start, start + 1] = correlation
        covariance[start + 1, start] = correlation
    return delta, covariance


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 4000,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return float("nan"), float("nan")
    if np.all(data == data[0]):
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, data.size, size=(int(draws), data.size))
    means = np.mean(data[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def audit(
    *,
    cases: int,
    correlations: tuple[float, ...],
    budget_evidence_counts: tuple[int, ...],
    bits_per_evidence: int,
    p_fa: float,
    target_pd: float,
    seed_offset: int,
) -> dict[str, object]:
    started = perf_counter()
    n_sources = 8
    all_bits = int(bits_per_evidence) * n_sources
    rows: list[dict[str, object]] = []
    mechanism_failures: list[str] = []
    for rho_index, rho in enumerate(correlations):
        for budget_index, evidence_count in enumerate(budget_evidence_counts):
            budget = int(evidence_count) * int(bits_per_evidence)
            proposed_pd: list[float] = []
            unaware_pd: list[float] = []
            reference_pd: list[float] = []
            single_pd: list[float] = []
            all_neighbor_pd: list[float] = []
            proposed_d: list[float] = []
            unaware_d: list[float] = []
            reference_d: list[float] = []
            single_d: list[float] = []
            all_neighbor_d: list[float] = []
            exact_matches = 0
            for case in range(int(cases)):
                delta, covariance = _controlled_model(
                    rho, int(seed_offset) + case)
                bits = np.full(n_sources, int(bits_per_evidence), dtype=np.int64)
                proposed = conditional_deflection_greedy(
                    delta, covariance, bits, budget)
                unaware = correlation_unaware_greedy(
                    delta, covariance, bits, budget)
                reference = exhaustive_budgeted_reference(
                    delta, covariance, bits, budget)
                local_values = np.square(delta) / np.diag(covariance)
                single_value = float(np.max(local_values))
                all_value, _ = optimal_linear_soft_fusion(
                    delta, covariance, tuple(range(n_sources)))
                if proposed.deflection > reference.deflection + 1.0e-9:
                    mechanism_failures.append(
                        "proposed exceeds exhaustive reference at "
                        f"rho={rho}, evidence_count={evidence_count}, case={case}")
                exact_matches += int(
                    abs(proposed.deflection - reference.deflection) <= 1.0e-9)
                p_values = compute_detection_probabilities(
                    np.array([
                        proposed.deflection,
                        unaware.deflection,
                        reference.deflection,
                        single_value,
                        all_value,
                    ]),
                    float(p_fa),
                )
                proposed_pd.append(float(p_values[0]))
                unaware_pd.append(float(p_values[1]))
                reference_pd.append(float(p_values[2]))
                single_pd.append(float(p_values[3]))
                all_neighbor_pd.append(float(p_values[4]))
                proposed_d.append(float(proposed.deflection))
                unaware_d.append(float(unaware.deflection))
                reference_d.append(float(reference.deflection))
                single_d.append(single_value)
                all_neighbor_d.append(all_value)
            pd_delta = np.asarray(proposed_pd) - np.asarray(unaware_pd)
            d_delta = np.asarray(proposed_d) - np.asarray(unaware_d)
            ci = _bootstrap_mean_ci(
                pd_delta,
                seed=99173 + 101 * rho_index + budget_index,
            )
            rows.append({
                "rho": float(rho),
                "budget_evidence_count": int(evidence_count),
                "budget_bits": int(budget),
                "budget_fraction": float(budget / all_bits),
                "proposed_pd_mean": float(np.mean(proposed_pd)),
                "correlation_unaware_pd_mean": float(np.mean(unaware_pd)),
                "exhaustive_reference_pd_mean": float(np.mean(reference_pd)),
                "single_uav_pd_mean": float(np.mean(single_pd)),
                "all_neighbor_pd_mean": float(np.mean(all_neighbor_pd)),
                "proposed_minus_unaware_pd_mean": float(np.mean(pd_delta)),
                "proposed_minus_unaware_pd_ci95": [ci[0], ci[1]],
                "proposed_deflection_mean": float(np.mean(proposed_d)),
                "correlation_unaware_deflection_mean": float(np.mean(unaware_d)),
                "exhaustive_reference_deflection_mean": float(
                    np.mean(reference_d)),
                "single_uav_deflection_mean": float(np.mean(single_d)),
                "all_neighbor_deflection_mean": float(np.mean(all_neighbor_d)),
                "all_neighbor_communication_bits": int(all_bits),
                "proposed_minus_unaware_deflection_mean": float(np.mean(d_delta)),
                "proposed_reference_exact_match_rate": float(exact_matches / int(cases)),
            })

    pareto: list[dict[str, object]] = []
    for rho in correlations:
        rho_rows = sorted(
            (row for row in rows if row["rho"] == rho),
            key=lambda row: int(row["budget_bits"]),
        )

        def first_bits(field: str) -> int | None:
            return next((
                int(row["budget_bits"]) for row in rho_rows
                if float(row[field]) >= float(target_pd)
            ), None)

        proposed_bits = first_bits("proposed_pd_mean")
        unaware_bits = first_bits("correlation_unaware_pd_mean")
        reference_bits = first_bits("exhaustive_reference_pd_mean")
        pareto.append({
            "rho": float(rho),
            "target_pd": float(target_pd),
            "proposed_minimum_grid_bits": proposed_bits,
            "correlation_unaware_minimum_grid_bits": unaware_bits,
            "exhaustive_reference_minimum_grid_bits": reference_bits,
            "proposed_bit_saving_vs_unaware": (
                None if proposed_bits is None or unaware_bits is None
                else int(unaware_bits - proposed_bits)
            ),
        })

    low_rows = [row for row in rows if row["rho"] == min(correlations)]
    low_limit = max(
        abs(float(row["proposed_minus_unaware_pd_mean"])) for row in low_rows)
    if min(correlations) == 0.0 and low_limit > 1.0e-12:
        mechanism_failures.append(
            f"independent-limit ablation gap is {low_limit:.3e}, expected zero")
    high_rho = max(correlations)
    high_interior = [
        row for row in rows
        if row["rho"] == high_rho
        and 0.0 < float(row["budget_fraction"]) < 1.0
        and int(row["budget_bits"]) >= 2 * int(bits_per_evidence)
    ]
    if not any(
        float(row["proposed_minus_unaware_pd_ci95"][0]) > 0.0
        for row in high_interior
    ):
        mechanism_failures.append(
            "high-correlation interior budget has no positive paired CI")
    high_pareto = next(row for row in pareto if row["rho"] == high_rho)
    high_saving = high_pareto["proposed_bit_saving_vs_unaware"]
    if high_saving is None or int(high_saving) <= 0:
        mechanism_failures.append(
            "high-correlation grid does not save bits at target P_D")

    return {
        "status": "PASS" if not mechanism_failures else "FAIL",
        "evidence_class": "DIAGNOSTIC_ONLY",
        "scientific_question": (
            "conditional-Deflection evidence selection versus correlation-"
            "unaware selection at equal delivered bits"),
        "model": {
            "sources": n_sources,
            "covariance": "four independent 2x2 blocks with within-pair rho",
            "bits_per_evidence": int(bits_per_evidence),
            "p_fa": float(p_fa),
            "target_pd": float(target_pd),
            "cases": int(cases),
            "seed_offset": int(seed_offset),
        },
        "checks": {
            "rho_zero_reduces_to_unaware": bool(low_limit <= 1.0e-12),
            "high_rho_has_positive_paired_ci": bool(any(
                float(row["proposed_minus_unaware_pd_ci95"][0]) > 0.0
                for row in high_interior)),
            "high_rho_saves_bits_at_target_pd": bool(
                high_saving is not None and int(high_saving) > 0),
            "proposed_never_exceeds_exhaustive_reference": not any(
                "exceeds exhaustive reference" in failure
                for failure in mechanism_failures),
        },
        "failure_reasons": mechanism_failures,
        "pareto_target": pareto,
        "rows": rows,
        "elapsed_s": float(perf_counter() - started),
    }


def _csv_tuple(value: str, cast) -> tuple:
    result = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("grid cannot be empty")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--correlations", default="0,0.2,0.5,0.8,0.95")
    parser.add_argument(
        "--budget-evidence-counts", default="1,2,3,4,5,6,7,8",
        help="exact numbers of fixed-size evidence entries allowed")
    parser.add_argument("--bits-per-evidence", type=int, default=64)
    parser.add_argument("--p-fa", type=float, default=1.0e-3)
    parser.add_argument("--target-pd", type=float, default=0.9)
    parser.add_argument("--seed-offset", type=int, default=31000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    correlations = _csv_tuple(args.correlations, float)
    evidence_counts = _csv_tuple(args.budget_evidence_counts, int)
    if (
        args.cases < 1 or args.bits_per_evidence < 1
        or any(value < 0.0 or value >= 1.0 for value in correlations)
        or any(value < 1 or value > 8 for value in evidence_counts)
        or len(set(evidence_counts)) != len(evidence_counts)
        or not 0.0 < args.p_fa < 1.0
        or not args.p_fa < args.target_pd < 1.0
    ):
        raise ValueError("audit arguments lie outside supported ranges")
    result = audit(
        cases=args.cases,
        correlations=correlations,
        budget_evidence_counts=evidence_counts,
        bits_per_evidence=args.bits_per_evidence,
        p_fa=args.p_fa,
        target_pd=args.target_pd,
        seed_offset=args.seed_offset,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return int(bool(args.strict and result["status"] != "PASS"))


if __name__ == "__main__":
    raise SystemExit(main())
