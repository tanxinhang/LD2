#!/usr/bin/env python
"""Pretrain only the observational receiver-side token semantic decoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from tools.pretrain_qos_commitment import build_agent
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.utils.checkpoint_loading import safe_torch_load, validate_state_dict
from uav_isac.utils.seeding import set_seed


def load_rows(path: str):
    with np.load(path) as data:
        return (
            np.asarray(data["tokens"][:, 1:], dtype=np.float32),
            np.asarray(data["current_local_pd"], dtype=np.float32),
        )


def metrics(active_true, evidence_probability, pd_true, pd_prediction):
    from sklearn.metrics import (
        balanced_accuracy_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    active = np.asarray(active_true, dtype=np.int64)
    selector = active > 0
    return {
        "active_roc_auc": float(roc_auc_score(active, evidence_probability)),
        "active_balanced_accuracy": float(balanced_accuracy_score(
            active, evidence_probability >= 0.5)),
        "pd_given_active_mae": float(mean_absolute_error(
            pd_true[selector], pd_prediction[selector])),
        "pd_given_active_r2": float(r2_score(
            pd_true[selector], pd_prediction[selector])),
        "active_rate": float(np.mean(active)),
        "rows": int(active.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "config/exp_800_q4_u2u_hierarchical_multistatic_"
            "distributed_matching_hybrid50_semantic_decoder_eval.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default="results/qos_pretrained_soft_gated_move50_test100/best_restored.pt",
    )
    parser.add_argument(
        "--train-data",
        default="results/token_semantic_probe/datasets/probe_train.npz")
    parser.add_argument(
        "--test-data",
        default="results/token_semantic_probe/datasets/probe_test.npz")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--output-dir", default="results/token_semantic_decoder_pretrain")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    if not bool(cfg.marl.comm_semantic_decoder_enabled):
        raise ValueError("config must enable comm_semantic_decoder_enabled")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = UAVISACEnv(config=cfg, seed=args.seed)
    agent = build_agent(cfg, env, device)
    checkpoint = safe_torch_load(
        args.checkpoint,
        map_location=device,
        description="token semantic-decoder source checkpoint",
        optional_mapping_keys=("runtime",),
        optional_state_dict_keys=("actor",),
    )
    actor_state = checkpoint.get("actor", checkpoint)
    validate_state_dict(
        actor_state, description="token semantic-decoder actor state_dict")
    agent.load_actor_state_dict_compatible(actor_state)
    runtime = checkpoint.get("runtime", {})
    if hasattr(agent.actor, "set_capacity_matching_blend"):
        agent.actor.set_capacity_matching_blend(float(runtime.get(
            "capacity_matching_blend", cfg.marl.capacity_matching_blend_start)))
    if hasattr(agent.actor, "set_target_allocation_movement_blend"):
        agent.actor.set_target_allocation_movement_blend(float(runtime.get(
            "target_allocation_movement_blend",
            cfg.marl.target_allocation_movement_blend)))
    decoder = agent.actor.comm_semantic_decoder
    # Compatible loading zeroes modules absent from a legacy checkpoint.  A
    # fully zero MLP cannot propagate gradients through ReLUs, so explicitly
    # initialize this isolated new head before its offline pretraining.
    for module in decoder.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()
    for parameter in agent.actor.parameters():
        parameter.requires_grad = False
    for parameter in decoder.parameters():
        parameter.requires_grad = True

    x_train_np, pd_train_np = load_rows(args.train_data)
    x_test_np, pd_test_np = load_rows(args.test_data)
    x_train = torch.as_tensor(x_train_np, device=device)
    pd_train = torch.as_tensor(pd_train_np, device=device)
    active_train = (pd_train > 0.0).to(torch.float32)
    pos_weight = torch.as_tensor(
        float((active_train.numel() - active_train.sum()).item())
        / max(float(active_train.sum().item()), 1.0),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    batch_size = min(512, x_train.shape[0])
    decoder.train()
    for _ in range(max(1, args.epochs)):
        order = torch.randperm(
            x_train.shape[0], generator=generator, device="cpu")
        for start in range(0, x_train.shape[0], batch_size):
            index = order[start:start + batch_size].to(device)
            decoded = decoder(x_train[index])
            active = active_train[index]
            evidence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                decoded[:, 0], active, pos_weight=pos_weight)
            conditional = active > 0.5
            pd_prediction = torch.sigmoid(decoded[:, 1])
            pd_loss = torch.nn.functional.smooth_l1_loss(
                pd_prediction[conditional], pd_train[index][conditional])
            loss = evidence_loss + 2.0 * pd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    decoder.eval()
    with torch.inference_mode():
        test_output = decoder(torch.as_tensor(x_test_np, device=device))
        evidence = torch.sigmoid(test_output[:, 0]).cpu().numpy()
        pd_prediction = torch.sigmoid(test_output[:, 1]).cpu().numpy()
    report_metrics = metrics(
        pd_test_np > 0.0, evidence, pd_test_np, pd_prediction)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": agent.actor.state_dict(),
        "runtime": runtime,
        "semantic_decoder_metrics": report_metrics,
        "source_checkpoint": args.checkpoint,
    }, output_dir / "semantic_decoder.pt")
    report = {
        "config": args.config,
        "source_checkpoint": args.checkpoint,
        "train_data": args.train_data,
        "test_data": args.test_data,
        "epochs": int(args.epochs),
        "decoder_parameters": int(sum(
            p.numel() for p in decoder.parameters())),
        "metrics": report_metrics,
        "behavioural_authority": 0.0,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
