#!/usr/bin/env python
"""Synthetic benchmark for AI-proposed, analytically certified screening."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.coordination.ai_candidate_screener import (
    CandidateScoreNetwork,
    certified_ai_screen,
)


def _candidates(rng: np.random.Generator, count: int):
    state_dim = 4
    rank = 2
    prior_diag = np.exp(rng.normal(0.0, 0.6, size=(count, state_dim)))
    coupling = rng.uniform(0.0, 0.95, size=count)
    raw_factor = rng.normal(0.0, 0.7, size=(count, state_dim, rank))
    factor = raw_factor / np.sqrt(1.0 + coupling)[:, None, None]
    gram = np.einsum(
        "nir,ni,nis->nrs", factor, prior_diag, factor,
        optimize=True)
    identity = np.eye(rank, dtype=np.float64)[None, :, :]
    gain = np.linalg.slogdet(identity + gram)[1]
    trace = np.trace(gram, axis1=1, axis2=2)
    upper = rank * np.log1p(trace / rank)
    energy = rng.uniform(0.01, 1.0, size=count)
    bits = rng.integers(0, 257, size=count) / 256.0
    aoi = rng.integers(0, 11, size=count) / 10.0
    deficit = rng.uniform(0.0, 1.0, size=count)
    prior_mean = np.mean(prior_diag, axis=1)
    prior_spread = np.std(np.log(prior_diag), axis=1)
    factor_energy = np.sum(factor * factor, axis=(1, 2))
    features = np.column_stack((
        np.log1p(upper),
        coupling,
        energy,
        bits,
        aoi,
        deficit,
        np.log1p(prior_mean),
        prior_spread,
        np.log1p(factor_energy),
    )).astype(np.float32)
    return features, gram, gain.astype(np.float64), upper.astype(np.float64)


def _train(
    model: CandidateScoreNetwork,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    device: torch.device,
) -> float:
    model.to(device)
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1.0e-6)
    model.set_normalization(mean, scale)
    x = torch.as_tensor(features, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    started = perf_counter()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.mean((prediction - y) ** 2)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (perf_counter() - started)


def _latency_summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(data)),
        "p50_ms": float(np.percentile(data, 50)),
        "p95_ms": float(np.percentile(data, 95)),
        "max_ms": float(np.max(data)),
    }


def benchmark(
    *,
    train_candidates: int,
    scenes: int,
    candidates_per_scene: int,
    topk: int,
    exact_budget: int,
    verification_batch_size: int,
    epochs: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    train_x, _train_gram, train_y, _train_upper = _candidates(
        rng, train_candidates)
    cuda = torch.cuda.is_available()
    train_device = torch.device("cuda" if cuda else "cpu")
    gpu_model = CandidateScoreNetwork(train_x.shape[1], hidden_dim=64)
    training_ms = _train(
        gpu_model, train_x, train_y, epochs=epochs, device=train_device)
    gpu_model.eval()
    cpu_model = copy.deepcopy(gpu_model).to("cpu").eval()

    # Warm both dispatch paths before measuring.
    warm_x, _g, _y, _u = _candidates(rng, candidates_per_scene)
    with torch.inference_mode():
        cpu_model(torch.as_tensor(warm_x))
        if cuda:
            gpu_model(torch.as_tensor(warm_x, device=train_device))
            torch.cuda.synchronize(train_device)

    cpu_inference = []
    gpu_inference = []
    cpu_total_screening = []
    gpu_total_screening = []
    full_exact = []
    evaluated_fraction = []
    proposed_recall = []
    upper_bound_recall = []
    bounded_certified = []
    bounded_fallback = []
    exact_match = []
    for _ in range(scenes):
        features, gram, gain, upper = _candidates(
            rng, candidates_per_scene)
        incumbent = int(np.argmin(upper))

        started = perf_counter()
        dense_gain = np.linalg.slogdet(
            np.eye(gram.shape[-1])[None, :, :] + gram)[1]
        full_exact.append(1000.0 * (perf_counter() - started))
        teacher = int(np.argmax(dense_gain))
        upper_bound_recall.append(float(
            teacher in np.argsort(-upper, kind="stable")[:topk]))

        cpu_result = certified_ai_screen(
            cpu_model, features, upper,
            lambda indices, values=gain: values[indices],
            topk=topk, incumbent_index=incumbent, device="cpu",
            verification_batch_size=verification_batch_size)
        cpu_inference.append(cpu_result.inference_time_ms)
        cpu_total_screening.append(cpu_result.total_screening_time_ms)
        evaluated_fraction.append(
            len(cpu_result.evaluated_indices) / candidates_per_scene)
        proposed_recall.append(float(teacher in cpu_result.proposed_indices))
        exact_match.append(float(cpu_result.chosen_index == teacher))

        bounded = certified_ai_screen(
            cpu_model, features, upper,
            lambda indices, values=gain: values[indices],
            topk=topk, incumbent_index=incumbent,
            max_exact_evaluations=exact_budget, device="cpu",
            verification_batch_size=verification_batch_size)
        bounded_certified.append(float(bounded.certified))
        bounded_fallback.append(float(bounded.used_incumbent_fallback))

        if cuda:
            gpu_result = certified_ai_screen(
                gpu_model, features, upper,
                lambda indices, values=gain: values[indices],
                topk=topk, incumbent_index=incumbent,
                device=train_device,
                verification_batch_size=verification_batch_size)
            torch.cuda.synchronize(train_device)
            gpu_inference.append(gpu_result.inference_time_ms)
            gpu_total_screening.append(gpu_result.total_screening_time_ms)

    mean_evaluated = float(np.mean(evaluated_fraction))
    cpu_mean = float(np.mean(cpu_total_screening))
    candidates_saved = candidates_per_scene * (1.0 - mean_evaluated)
    break_even_us = float(
        1000.0 * cpu_mean / max(candidates_saved, 1.0e-12))
    return {
        "status": "SYNTHETIC_SCREENING_ONLY",
        "device": str(train_device),
        "gpu_name": (
            torch.cuda.get_device_name(train_device) if cuda else None),
        "training": {
            "candidates": train_candidates,
            "epochs": epochs,
            "time_ms": training_ms,
        },
        "evaluation": {
            "scenes": scenes,
            "candidates_per_scene": candidates_per_scene,
            "topk": topk,
            "exact_budget": exact_budget,
            "verification_batch_size": verification_batch_size,
            "ai_topk_teacher_recall": float(np.mean(proposed_recall)),
            "upper_bound_topk_teacher_recall": float(
                np.mean(upper_bound_recall)),
            "certified_exact_match_rate": float(np.mean(exact_match)),
            "mean_exact_candidate_fraction": mean_evaluated,
            "mean_exact_candidates": float(
                mean_evaluated * candidates_per_scene),
            "bounded_certification_rate": float(
                np.mean(bounded_certified)),
            "bounded_incumbent_fallback_rate": float(
                np.mean(bounded_fallback)),
            "full_batched_logdet_latency": _latency_summary(full_exact),
            "cpu_ai_inference_latency": _latency_summary(cpu_inference),
            "cpu_total_certified_screening_latency": _latency_summary(
                cpu_total_screening),
            "gpu_ai_inference_latency": (
                _latency_summary(gpu_inference) if gpu_inference else None),
            "gpu_total_certified_screening_latency": (
                _latency_summary(gpu_total_screening)
                if gpu_total_screening else None),
            "cpu_break_even_exact_cost_per_saved_candidate_us": (
                break_even_us),
        },
        "interpretation": (
            "AI is worthwhile only when full H/R construction plus exact "
            "verification costs more per pruned candidate than the reported "
            "break-even value; it is not intended to accelerate a lone 2x2 "
            "log-det already available as one batched tensor operation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=int, default=20_000)
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--candidates-per-scene", type=int, default=512)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--exact-budget", type=int, default=64)
    parser.add_argument("--verification-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = benchmark(
        train_candidates=args.train_candidates,
        scenes=args.scenes,
        candidates_per_scene=args.candidates_per_scene,
        topk=args.topk,
        exact_budget=args.exact_budget,
        verification_batch_size=args.verification_batch_size,
        epochs=args.epochs,
        seed=args.seed,
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
