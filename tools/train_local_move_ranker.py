#!/usr/bin/env python
"""Train the Gate C1.6 shared local-move ranking scorer."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_isac.coordination.learned_move_ranker import LocalMoveRanker
from uav_isac.coordination.local_move_ranker import FEATURE_NAMES


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _group_metrics(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    group_indices: np.ndarray,
) -> dict[str, float]:
    ptr = data["group_ptr"]
    opportunity = 0
    top1_positive = 0
    top3_positive = 0
    top1_best = 0
    top3_best = 0
    primary_regret = []
    for group in group_indices:
        start, stop = int(ptr[group]), int(ptr[group + 1])
        positive = data["positive"][start:stop].astype(bool)
        if not np.any(positive):
            continue
        best = data["best"][start:stop].astype(bool)
        order = np.argsort(-scores[start:stop], kind="stable")
        top3 = order[:min(3, len(order))]
        opportunity += 1
        top1_positive += int(positive[order[0]])
        top3_positive += int(np.any(positive[top3]))
        top1_best += int(best[order[0]])
        top3_best += int(np.any(best[top3]))
        best_delta = float(np.max(
            data["objective_delta"][start:stop, 0]))
        selected_delta = float(data["objective_delta"][start + order[0], 0])
        primary_regret.append(max(0.0, best_delta - selected_delta))
    denominator = float(max(opportunity, 1))
    return {
        "positive_groups": int(opportunity),
        "top1_positive_rate": top1_positive / denominator,
        "top3_positive_rate": top3_positive / denominator,
        "top1_best_rate": top1_best / denominator,
        "top3_best_rate": top3_best / denominator,
        "primary_regret_mean": float(np.mean(primary_regret))
        if primary_regret else 0.0,
    }


def _selection_score(metrics: dict[str, float]) -> float:
    return (
        2.0 * metrics["top3_positive_rate"]
        + metrics["top1_positive_rate"]
        + 0.5 * metrics["top3_best_rate"]
        + 0.25 * metrics["top1_best_rate"]
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = _load(args.dataset)
    features_np = data["features"].astype(np.float32)
    if features_np.shape[1] != len(FEATURE_NAMES):
        raise ValueError("dataset feature width does not match scorer")
    seeds = np.unique(data["group_seed"])
    if len(seeds) < 3:
        raise ValueError("at least three seeds are required for group split")
    val_seed_count = min(max(1, int(args.val_seeds)), len(seeds) - 1)
    val_seeds = seeds[-val_seed_count:]
    train_seeds = seeds[:-val_seed_count]
    train_groups = np.flatnonzero(
        np.isin(data["group_seed"], train_seeds))
    val_groups = np.flatnonzero(np.isin(data["group_seed"], val_seeds))
    ptr = data["group_ptr"]
    train_move_mask = np.zeros(len(features_np), dtype=bool)
    for group in train_groups:
        train_move_mask[int(ptr[group]):int(ptr[group + 1])] = True
    feature_mean = np.mean(features_np[train_move_mask], axis=0)
    feature_std = np.std(features_np[train_move_mask], axis=0)
    model = LocalMoveRanker(
        feature_mean, feature_std, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    features = torch.from_numpy(features_np)
    positive = torch.from_numpy(data["positive"].astype(np.float32))
    rank_target = torch.from_numpy(data["rank_target"].astype(np.float32))
    train_mask_t = torch.from_numpy(train_move_mask)
    train_positive = positive[train_mask_t]
    positives = float(torch.sum(train_positive).item())
    negatives = float(len(train_positive) - positives)
    pos_weight = torch.tensor(min(
        max(negatives / max(positives, 1.0), 1.0), args.max_pos_weight))
    best_score = -float("inf")
    best_epoch = -1
    best_alpha = 1.0
    best_state = None
    best_metrics: dict[str, float] = {}
    stale = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rank_score, positive_logit = model(features)
        bce = F.binary_cross_entropy_with_logits(
            positive_logit[train_mask_t],
            positive[train_mask_t],
            pos_weight=pos_weight,
        )
        listwise_terms = []
        combined = rank_score + positive_logit
        for group in train_groups:
            start, stop = int(ptr[group]), int(ptr[group + 1])
            group_positive = positive[start:stop] > 0.5
            if not bool(torch.any(group_positive)):
                continue
            log_probability = F.log_softmax(combined[start:stop], dim=0)
            quality = rank_target[start:stop][group_positive]
            target = F.softmax(quality / args.target_temperature, dim=0)
            listwise_terms.append(-torch.sum(
                target * log_probability[group_positive]))
        listwise = torch.stack(listwise_terms).mean()
        loss = args.bce_weight * bce + args.listwise_weight * listwise
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_rank, eval_positive = model(features)
        alpha_metrics = {}
        epoch_best_score = -float("inf")
        epoch_best_alpha = 1.0
        epoch_best_metrics = {}
        for alpha in args.alpha:
            scores = (eval_rank + float(alpha) * eval_positive).numpy()
            metrics = _group_metrics(scores, data, val_groups)
            alpha_metrics[str(alpha)] = metrics
            score = _selection_score(metrics)
            if score > epoch_best_score:
                epoch_best_score = score
                epoch_best_alpha = float(alpha)
                epoch_best_metrics = metrics
        history.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "bce": float(bce.item()),
            "listwise": float(listwise.item()),
            "val_score": epoch_best_score,
            "alpha": epoch_best_alpha,
        })
        if epoch_best_score > best_score + 1.0e-9:
            best_score = epoch_best_score
            best_epoch = epoch
            best_alpha = epoch_best_alpha
            best_metrics = epoch_best_metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch % args.log_every == 0:
            print(json.dumps(history[-1]), flush=True)
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rank_score, positive_logit = model(features)
        final_scores = (
            rank_score + best_alpha * positive_logit).numpy()
    train_metrics = _group_metrics(final_scores, data, train_groups)
    val_metrics = _group_metrics(final_scores, data, val_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "hidden_dim": args.hidden_dim,
        "alpha": best_alpha,
        "feature_names": FEATURE_NAMES,
        "train_seeds": train_seeds.tolist(),
        "val_seeds": val_seeds.tolist(),
    }, args.output)
    result = {
        "protocol": "gate_c1_6_local_move_ranker_v1",
        "dataset": str(args.dataset),
        "checkpoint": str(args.output),
        "parameters": int(sum(
            parameter.numel() for parameter in model.parameters())),
        "train_seeds": [int(value) for value in train_seeds],
        "val_seeds": [int(value) for value in val_seeds],
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "alpha": best_alpha,
        "selection_score": best_score,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "positive_class_weight": float(pos_weight.item()),
        "fresh_test_consumed": False,
        "runtime_s": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps({**result, "history": history}, indent=2),
        encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_move_rank_train10/previous_n123.npz"))
    parser.add_argument(
        "--output", type=Path, default=Path(
            "results/architecture_v2_scale_k6q6_local_move_ranker_gate_c1_6/previous_n123_ranker.pt"))
    parser.add_argument("--val-seeds", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--listwise-weight", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=0.15)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--alpha", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--seed", type=int, default=8127)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
