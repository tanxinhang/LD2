#!/usr/bin/env python
"""Offline gate: can decoded token semantics identify low-QoS target frames?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uav_isac.utils.checkpoint_loading import safe_torch_load  # noqa: E402


def load_decoder(checkpoint_path: str):
    checkpoint = safe_torch_load(
        checkpoint_path,
        map_location="cpu",
        description="crisis-gate semantic decoder checkpoint",
        state_dict_keys=("actor",),
    )
    state = checkpoint["actor"]
    decoder = torch.nn.Sequential(
        torch.nn.Linear(15, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 16), torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )
    prefix = "comm_semantic_decoder."
    decoder.load_state_dict({
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    })
    decoder.eval()
    return decoder


def load_gate_rows(path: str, decoder) -> dict:
    with np.load(path) as data:
        required = {
            "tokens", "episode_seed", "frame_index", "target_id",
            "frame_seed", "frame_step", "frame_target_id",
            "frame_claim_count", "frame_next_team_pd",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"dataset predates crisis fields, regenerate it: {sorted(missing)}")
        tokens = np.asarray(data["tokens"], dtype=np.float32)
        with torch.inference_mode():
            output = decoder(torch.as_tensor(tokens[:, 1:]))
            evidence = torch.sigmoid(output[:, 0]).numpy()
            pd_value = torch.sigmoid(output[:, 1]).numpy()
        row_quality = evidence * pd_value
        row_key = list(zip(
            np.asarray(data["episode_seed"], dtype=np.int64),
            np.asarray(data["frame_index"], dtype=np.int64),
            np.asarray(data["target_id"], dtype=np.int64),
        ))
        quality_by_key = {}
        bid_by_key = {}
        for index, key in enumerate(row_key):
            quality_by_key[key] = max(
                quality_by_key.get(key, 0.0), float(row_quality[index]))
            normalized_bid = float(np.clip((tokens[index, 0] + 1.0) / 2.0, 0, 1))
            bid_by_key[key] = max(bid_by_key.get(key, 0.0), normalized_bid)
        frame_seed = np.asarray(data["frame_seed"], dtype=np.int64)
        frame_step = np.asarray(data["frame_step"], dtype=np.int64)
        frame_target = np.asarray(data["frame_target_id"], dtype=np.int64)
        keys = list(zip(frame_seed, frame_step, frame_target))
        semantic_quality = np.asarray([
            quality_by_key.get(key, 0.0) for key in keys])
        contract_bid = np.asarray([bid_by_key.get(key, 0.0) for key in keys])
        claim_count = np.asarray(data["frame_claim_count"], dtype=np.float64)
        endpoint_deficit = np.clip(2.0 - claim_count, 0.0, 2.0) / 2.0
        semantic_crisis = 1.0 - semantic_quality
        contract_crisis = np.maximum(endpoint_deficit, 1.0 - contract_bid)
        combined_crisis = np.maximum(endpoint_deficit, semantic_crisis)
        return {
            "seed": frame_seed,
            "step": frame_step,
            "target": frame_target,
            "claim_count": claim_count,
            "team_pd": np.asarray(data["frame_next_team_pd"], dtype=np.float64),
            "endpoint_deficit": endpoint_deficit,
            "contract_crisis": contract_crisis,
            "semantic_crisis": semantic_crisis,
            "combined_crisis": combined_crisis,
        }


def choose_threshold(score: np.ndarray, label: np.ndarray) -> float:
    score, label = _validate_binary_inputs(score, label)
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    best = (-np.inf, float(np.median(score)))
    for threshold in candidates:
        value = _balanced_accuracy(label, score >= threshold)
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]


def evaluate(score, label, threshold) -> dict:
    score, label = _validate_binary_inputs(score, label)
    prediction = score >= threshold
    true_positive = int(np.count_nonzero(prediction & label))
    predicted_positive = int(np.count_nonzero(prediction))
    actual_positive = int(np.count_nonzero(label))
    return {
        "roc_auc": _binary_roc_auc(label, score),
        "balanced_accuracy": _balanced_accuracy(label, prediction),
        "precision": (
            true_positive / predicted_positive if predicted_positive else 0.0),
        "recall": true_positive / actual_positive,
        "activation_rate": float(np.mean(prediction)),
        "threshold_from_train": float(threshold),
    }


def _validate_binary_inputs(score, label) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=bool).reshape(-1)
    if score.size == 0 or score.size != label.size:
        raise ValueError("score and label must be non-empty and equally sized")
    if not np.all(np.isfinite(score)):
        raise ValueError("score must contain only finite values")
    if not np.any(label) or np.all(label):
        raise ValueError("both binary classes are required")
    return score, label


def _balanced_accuracy(label: np.ndarray, prediction: np.ndarray) -> float:
    """Binary macro recall: one half of sensitivity plus specificity."""
    positive = label
    negative = ~label
    sensitivity = np.mean(prediction[positive])
    specificity = np.mean(~prediction[negative])
    return float(0.5 * (sensitivity + specificity))


def _binary_roc_auc(label: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC with half credit for tied positive/negative scores."""
    positive = score[label]
    negative = score[~label]
    wins = np.count_nonzero(positive[:, None] > negative[None, :])
    ties = np.count_nonzero(positive[:, None] == negative[None, :])
    return float((wins + 0.5 * ties) / (positive.size * negative.size))


def temporal_hysteresis_gate(
    rows: dict,
    semantic_threshold: float,
    stagnation_frames: int = 3,
    improvement_epsilon: float = 0.02,
    ema_alpha: float = 0.30,
) -> np.ndarray:
    """Immediate endpoint repair plus delayed low-quality semantic crisis."""
    gate = rows["claim_count"] < 2
    quality = 1.0 - rows["semantic_crisis"]
    for episode_seed in np.unique(rows["seed"]):
        for target in np.unique(rows["target"]):
            index = np.flatnonzero(
                (rows["seed"] == episode_seed)
                & (rows["target"] == target))
            index = index[np.argsort(rows["step"][index])]
            if index.size <= stagnation_frames:
                continue
            ema = np.empty(index.size, dtype=np.float64)
            ema[0] = quality[index[0]]
            for position in range(1, index.size):
                ema[position] = (
                    ema_alpha * quality[index[position]]
                    + (1.0 - ema_alpha) * ema[position - 1])
            for position in range(stagnation_frames, index.size):
                row = index[position]
                low_quality = rows["semantic_crisis"][row] >= semantic_threshold
                improvement = ema[position] - ema[position - stagnation_frames]
                stagnant = improvement <= improvement_epsilon
                gate[row] = bool(gate[row] or (low_quality and stagnant))
    return gate


def choose_temporal_config(
    rows: dict,
    label: np.ndarray,
    activation_cap: float = 0.25,
) -> dict:
    from sklearn.metrics import balanced_accuracy_score

    score = rows["semantic_crisis"]
    candidates = np.unique(np.quantile(score, np.linspace(0.50, 0.999, 100)))
    feasible = []
    fallback = []
    for stagnation_frames in (3, 5, 8, 10):
        for improvement_epsilon in (0.0, 0.005, 0.01, 0.02):
            for threshold in candidates:
                gate = temporal_hysteresis_gate(
                    rows,
                    float(threshold),
                    stagnation_frames=stagnation_frames,
                    improvement_epsilon=improvement_epsilon,
                    ema_alpha=0.30,
                )
                value = float(balanced_accuracy_score(label, gate))
                activation = float(np.mean(gate))
                candidate = (
                    value, -activation, -stagnation_frames,
                    -improvement_epsilon, float(threshold),
                )
                fallback.append(candidate)
                if activation <= activation_cap:
                    feasible.append(candidate)
    selected = max(feasible if feasible else fallback)
    return {
        "threshold": selected[4],
        "stagnation_frames": int(-selected[2]),
        "improvement_epsilon": float(-selected[3]),
        "ema_alpha": 0.30,
        "train_activation_cap": float(activation_cap),
        "train_balanced_accuracy": float(selected[0]),
        "train_activation_rate": float(-selected[1]),
    }


def evaluate_binary_gate(gate: np.ndarray, label: np.ndarray) -> dict:
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    gate = np.asarray(gate, dtype=bool)
    return {
        "roc_auc_binary": float(roc_auc_score(label, gate)),
        "balanced_accuracy": float(balanced_accuracy_score(label, gate)),
        "precision": float(precision_score(label, gate, zero_division=0)),
        "recall": float(recall_score(label, gate, zero_division=0)),
        "activation_rate": float(np.mean(gate)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder-checkpoint",
        default="results/token_semantic_decoder_pretrain/semantic_decoder.pt")
    parser.add_argument(
        "--train-data",
        default="results/token_semantic_probe/datasets/probe_train.npz")
    parser.add_argument(
        "--test-data",
        default="results/token_semantic_probe/datasets/probe_test.npz")
    parser.add_argument("--qos-floor", type=float, default=0.60)
    parser.add_argument(
        "--output", default="results/token_semantic_probe/crisis_gate.json")
    args = parser.parse_args()

    decoder = load_decoder(args.decoder_checkpoint)
    train = load_gate_rows(args.train_data, decoder)
    test = load_gate_rows(args.test_data, decoder)
    train_label = train["team_pd"] < args.qos_floor
    test_label = test["team_pd"] < args.qos_floor
    if np.unique(train_label).size < 2 or np.unique(test_label).size < 2:
        raise RuntimeError("crisis data must contain both QoS classes")
    results = {}
    for name in (
        "endpoint_deficit", "contract_crisis", "semantic_crisis",
        "combined_crisis",
    ):
        threshold = choose_threshold(train[name], train_label)
        results[name] = evaluate(test[name], test_label, threshold)
    temporal_config = choose_temporal_config(train, train_label)
    temporal_test_gate = temporal_hysteresis_gate(
        test,
        temporal_config["threshold"],
        stagnation_frames=temporal_config["stagnation_frames"],
        improvement_epsilon=temporal_config["improvement_epsilon"],
        ema_alpha=temporal_config["ema_alpha"],
    )
    results["temporal_hysteresis_gate"] = {
        **evaluate_binary_gate(temporal_test_gate, test_label),
        "semantic_threshold_from_train": temporal_config["threshold"],
        "stagnation_frames": temporal_config["stagnation_frames"],
        "improvement_epsilon": temporal_config["improvement_epsilon"],
        "ema_alpha": temporal_config["ema_alpha"],
        "train_activation_cap": temporal_config["train_activation_cap"],
        "train_balanced_accuracy": temporal_config[
            "train_balanced_accuracy"],
        "train_activation_rate": temporal_config["train_activation_rate"],
    }
    report = {
        "qos_floor": float(args.qos_floor),
        "train_rows": int(train_label.size),
        "test_rows": int(test_label.size),
        "train_crisis_rate": float(np.mean(train_label)),
        "test_crisis_rate": float(np.mean(test_label)),
        "test_underload_rate": float(np.mean(test["claim_count"] < 2)),
        "results": results,
        "decision": {
            "semantic_auc_gain_over_contract": float(
                results["combined_crisis"]["roc_auc"]
                - results["contract_crisis"]["roc_auc"]),
            "gate_is_predictive": bool(
                results["combined_crisis"]["roc_auc"] >= 0.65
                and results["combined_crisis"]["balanced_accuracy"] >= 0.60),
            "temporal_gate_is_selective": bool(
                results["temporal_hysteresis_gate"]["activation_rate"] <= 0.40
                and results["temporal_hysteresis_gate"][
                    "balanced_accuracy"] >= 0.60),
            "note": (
                "This validates crisis detection only; it does not yet prove "
                "that acting on the gate improves QoS."),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
