#!/usr/bin/env python
"""Probe information carried by a frozen target-token policy.

The probe never changes the actor or environment.  It rolls out a frozen
checkpoint, applies the same physical quantizer as the U2U channel, and asks
whether token dimensions 1..D-1 add information beyond protocol metadata and
the existing dimension-0 movement bid.

Episode seeds, rather than individual rows, define the train/test split.  This
prevents adjacent frames from the same trajectory leaking into both sets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from tools.pretrain_qos_commitment import build_agent
from uav_isac.agents.trainer import load_stratified_seed_split
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.utils.checkpoint_loading import safe_torch_load, validate_state_dict
from uav_isac.utils.seeding import set_seed


@dataclass(frozen=True)
class ProbeDataset:
    """Target-token rows and aligned sender/outcome labels."""

    metadata: np.ndarray
    tokens: np.ndarray
    current_local_pd: np.ndarray
    next_local_pd: np.ndarray
    next_team_pd: np.ndarray
    distance_m: np.ndarray
    episode_seed: np.ndarray
    sender_id: np.ndarray | None = None
    target_id: np.ndarray | None = None
    frame_index: np.ndarray | None = None
    claim_count: np.ndarray | None = None
    frame_seed: np.ndarray | None = None
    frame_step: np.ndarray | None = None
    frame_target_id: np.ndarray | None = None
    frame_claim_count: np.ndarray | None = None
    frame_next_team_pd: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def quantize_target_tokens(
    messages: np.ndarray,
    rates: np.ndarray,
    token_masks: np.ndarray,
    comm_model,
    q_count: int,
    token_dim: int,
) -> np.ndarray:
    """Use the environment's exact quantizer and sparse-token masking."""
    result = np.zeros_like(messages, dtype=np.float64)
    for sender in range(messages.shape[0]):
        quantized = comm_model.quantize(messages[sender], int(rates[sender]))
        view = quantized.reshape(q_count, token_dim)
        view[token_masks[sender] <= 0.5] = 0.0
        result[sender] = view.reshape(-1)
    return result


def build_feature_views(dataset: ProbeDataset) -> Dict[str, np.ndarray]:
    """Return protocol, contract, semantic, and complete feature views."""
    if dataset.tokens.ndim != 2 or dataset.tokens.shape[1] < 2:
        raise ValueError("target tokens must have at least bid + one semantic dim")
    bid = dataset.tokens[:, :1]
    semantic = dataset.tokens[:, 1:]
    return {
        "metadata": dataset.metadata,
        "contract": np.concatenate([dataset.metadata, bid], axis=1),
        "semantic_only": semantic,
        "metadata_semantic": np.concatenate(
            [dataset.metadata, semantic], axis=1),
        "full": np.concatenate([dataset.metadata, bid, semantic], axis=1),
    }


def intervene_semantic(
    full_features: np.ndarray,
    metadata_dim: int,
    mode: str,
    target_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Zero or target-conditionally permute semantic dimensions only.

    The bid channel immediately following metadata is preserved.  Conditional
    permutation keeps each target's marginal semantic distribution intact.
    """
    altered = np.asarray(full_features, dtype=np.float64).copy()
    semantic_start = int(metadata_dim) + 1
    if mode == "zero":
        altered[:, semantic_start:] = 0.0
    elif mode == "permute":
        rng = np.random.default_rng(seed)
        for q in np.unique(target_ids.astype(np.int64)):
            idx = np.flatnonzero(target_ids == q)
            if idx.size > 1:
                altered[idx, semantic_start:] = altered[
                    rng.permutation(idx), semantic_start:]
    elif mode != "none":
        raise ValueError(f"unknown semantic intervention: {mode}")
    return altered


def _local_pd_from_observation(actor, obs_rows: np.ndarray) -> np.ndarray:
    slices = getattr(actor, "_obs_slices", None)
    if slices is None or slices.pd_hist_len <= 0:
        raise RuntimeError("actor observation parser does not expose local P_D")
    pd = slices.extract_pd_hist(obs_rows)
    return np.asarray(pd, dtype=np.float64).reshape(obs_rows.shape[0], -1)


def _target_distances(env: UAVISACEnv) -> np.ndarray:
    uav_xy = np.stack([u.pos[:2] for u in env.core.uavs])
    target_xy = np.stack([
        target.get_position_3d()[:2] for target in env.core.targets])
    return np.linalg.norm(uav_xy[:, None, :] - target_xy[None, :, :], axis=-1)


def collect_dataset(
    cfg,
    checkpoint: Mapping,
    seeds: Sequence[int],
    device: str,
) -> ProbeDataset:
    """Roll out the frozen policy and collect physically quantized tokens."""
    set_seed(0)
    env = UAVISACEnv(config=cfg, seed=0)
    agent = build_agent(cfg, env, device)
    actor_state = checkpoint.get("actor", checkpoint)
    validate_state_dict(
        actor_state, description="token-semantics actor state_dict")
    missing, unexpected = agent.load_actor_state_dict_compatible(actor_state)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint actor keys: {unexpected[:5]}")
    if missing:
        print(f"[probe] compatible load initialized {len(missing)} new actor keys")
    runtime = checkpoint.get("runtime", {}) if isinstance(checkpoint, Mapping) else {}
    if hasattr(agent.actor, "set_capacity_matching_blend"):
        agent.actor.set_capacity_matching_blend(float(runtime.get(
            "capacity_matching_blend",
            cfg.marl.capacity_matching_blend_start)))
    if hasattr(agent.actor, "set_target_allocation_movement_blend"):
        agent.actor.set_target_allocation_movement_blend(float(runtime.get(
            "target_allocation_movement_blend",
            cfg.marl.target_allocation_movement_blend)))
    actor = agent.actor.eval()
    aspace = agent.action_space

    k_count, q_count = cfg.scenario.K, cfg.scenario.Q
    token_dim = int(cfg.marl.comm_target_token_dim)
    if str(cfg.marl.comm_payload_mode).lower() != "target_tokens":
        raise ValueError("semantic probe requires target_tokens payload mode")
    if env.core._inter_uav_comm is None:
        raise ValueError("semantic probe requires cost-aware U2U communication")
    rate_levels = np.asarray(cfg.marl.comm_rate_bits_per_dim, dtype=np.int64)
    movement_interval = max(1, int(cfg.marl.movement_decision_interval))

    metadata_rows: List[np.ndarray] = []
    token_rows: List[np.ndarray] = []
    current_pd_rows: List[float] = []
    next_pd_rows: List[float] = []
    team_pd_rows: List[float] = []
    distance_rows: List[float] = []
    episode_rows: List[int] = []
    sender_rows: List[int] = []
    target_rows: List[int] = []
    frame_rows: List[int] = []
    claim_count_rows: List[int] = []
    frame_seed_rows: List[int] = []
    frame_step_rows: List[int] = []
    frame_target_rows: List[int] = []
    frame_claim_count_rows: List[int] = []
    frame_team_pd_rows: List[float] = []

    for ep_index, episode_seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(episode_seed))
        h_prev = None
        movement_phase = 0
        held_actions = None
        done = False
        step_index = 0
        while not done:
            obs_batch = np.stack([obs[str(k)] for k in range(k_count)])
            current_local_pd = _local_pd_from_observation(actor, obs_batch)
            current_distance = _target_distances(env)
            phase_value = float(movement_phase) / max(movement_interval - 1, 1)
            with torch.inference_mode():
                obs_t = torch.as_tensor(
                    obs_batch, dtype=torch.float32, device=device)
                phase_t = torch.full(
                    (k_count,), phase_value,
                    dtype=torch.float32, device=device)
                identity_t = torch.arange(
                    k_count, dtype=torch.long, device=device)
                dp_mean, dp_log_std, role_logits, comm_mean, _, h_new = actor(
                    obs_t,
                    h_prev,
                    comm_round_phase=phase_t,
                    agent_identity=identity_t,
                )
                token_mask_t = getattr(actor, "last_outgoing_token_mask", None)
                if token_mask_t is None:
                    token_mask_t = torch.ones(
                        (k_count, q_count), dtype=comm_mean.dtype, device=device)
                comm_action, rate_action, _, _ = agent.sample_communication(
                    comm_mean, deterministic=True)
                if bool(cfg.marl.joint_isac_power_enabled):
                    (_, _, comm_fraction, sensing_weights, _, _) = (
                        agent.sample_isac_resources(
                            comm_mean,
                            rate_action,
                            deterministic=True,
                            comm_fraction_min=float(
                                cfg.marl.comm_power_fraction_min),
                            comm_fraction_max=float(
                                cfg.marl.comm_power_fraction_max),
                        ))
                h_prev = h_new

            messages = comm_action.detach().cpu().numpy()
            rates = rate_action.detach().cpu().numpy().astype(np.int64)
            masks = token_mask_t.detach().cpu().numpy()
            quantized = quantize_target_tokens(
                messages,
                rates,
                masks,
                env.core._inter_uav_comm,
                q_count,
                token_dim,
            ).reshape(k_count, q_count, token_dim)

            dpm = dp_mean.detach().cpu().numpy()
            dps = dp_log_std.detach().cpu().numpy()
            roles = role_logits.detach().cpu().numpy()
            actions = {}
            for k in range(k_count):
                # StructuredActorNetwork exposes one shared 2-D log standard
                # deviation, while some legacy actors expand it per row.
                dp_std_k = dps if dps.ndim == 1 else dps[k]
                action, _ = aspace.decode(
                    dpm[k], dp_std_k, roles[k],
                    dp_deterministic=True,
                    role_deterministic=True,
                )
                actions[str(k)] = {
                    "delta_p": np.asarray(action.delta_p).copy(),
                    "role": int(action.role),
                }
            if movement_phase == 0 or held_actions is None:
                held_actions = {
                    key: {
                        "delta_p": value["delta_p"].copy(),
                        "role": value["role"],
                    }
                    for key, value in actions.items()
                }
            else:
                actions = {
                    key: {
                        "delta_p": value["delta_p"].copy(),
                        "role": value["role"],
                    }
                    for key, value in held_actions.items()
                }

            message_map = {k: messages[k].copy() for k in range(k_count)}
            rate_map = {k: int(rates[k]) for k in range(k_count)}
            mask_map = {k: masks[k].copy() for k in range(k_count)}
            if bool(cfg.marl.joint_isac_power_enabled):
                env.core.submit_learned_communications(
                    message_map,
                    rate_map,
                    {k: float(comm_fraction[k].item()) for k in range(k_count)},
                    {k: sensing_weights[k].detach().cpu().numpy()
                     for k in range(k_count)},
                    token_masks=mask_map,
                )
            else:
                env.core.submit_learned_communications(
                    message_map, rate_map, token_masks=mask_map)

            next_obs, _, terminated, truncated, info = env.step(actions)
            next_obs_batch = np.stack([
                next_obs[str(k)] for k in range(k_count)])
            next_local_pd = _local_pd_from_observation(actor, next_obs_batch)
            next_team_pd = np.asarray(info["P_D_q"], dtype=np.float64)

            per_target_claim_count = np.sum(masks > 0.5, axis=0).astype(np.int64)
            for q in range(q_count):
                frame_seed_rows.append(int(episode_seed))
                frame_step_rows.append(int(step_index))
                frame_target_rows.append(int(q))
                frame_claim_count_rows.append(int(per_target_claim_count[q]))
                frame_team_pd_rows.append(float(next_team_pd[q]))

            for sender in range(k_count):
                if rates[sender] <= 0:
                    continue
                active_targets = np.flatnonzero(masks[sender] > 0.5)
                for q in active_targets:
                    metadata_rows.append(np.asarray([
                        sender / max(k_count - 1, 1),
                        q / max(q_count - 1, 1),
                        phase_value,
                        rate_levels[rates[sender]] / max(rate_levels[-1], 1),
                    ], dtype=np.float64))
                    token_rows.append(quantized[sender, q].copy())
                    current_pd_rows.append(float(current_local_pd[sender, q]))
                    next_pd_rows.append(float(next_local_pd[sender, q]))
                    team_pd_rows.append(float(next_team_pd[q]))
                    distance_rows.append(float(current_distance[sender, q]))
                    episode_rows.append(int(episode_seed))
                    sender_rows.append(int(sender))
                    target_rows.append(int(q))
                    frame_rows.append(int(step_index))
                    claim_count_rows.append(int(per_target_claim_count[q]))

            obs = next_obs
            done = bool(
                terminated.get("__all__", False)
                or truncated.get("__all__", False))
            movement_phase = (movement_phase + 1) % movement_interval
            step_index += 1

        print(
            f"[probe] episode {ep_index + 1}/{len(seeds)} "
            f"seed={episode_seed} rows={len(token_rows)}")

    env.close()
    return ProbeDataset(
        metadata=np.stack(metadata_rows),
        tokens=np.stack(token_rows),
        current_local_pd=np.asarray(current_pd_rows),
        next_local_pd=np.asarray(next_pd_rows),
        next_team_pd=np.asarray(team_pd_rows),
        distance_m=np.asarray(distance_rows),
        episode_seed=np.asarray(episode_rows, dtype=np.int64),
        sender_id=np.asarray(sender_rows, dtype=np.int64),
        target_id=np.asarray(target_rows, dtype=np.int64),
        frame_index=np.asarray(frame_rows, dtype=np.int64),
        claim_count=np.asarray(claim_count_rows, dtype=np.int64),
        frame_seed=np.asarray(frame_seed_rows, dtype=np.int64),
        frame_step=np.asarray(frame_step_rows, dtype=np.int64),
        frame_target_id=np.asarray(frame_target_rows, dtype=np.int64),
        frame_claim_count=np.asarray(
            frame_claim_count_rows, dtype=np.int64),
        frame_next_team_pd=np.asarray(frame_team_pd_rows, dtype=np.float64),
    )


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    weights = np.ones_like(y, dtype=np.float64)
    for cls in (0, 1):
        if counts[cls] > 0:
            weights[y == cls] = y.size / (2.0 * counts[cls])
    return weights


def fit_probe_suite(
    train: ProbeDataset,
    test: ProbeDataset,
    seed: int,
) -> Dict[str, dict]:
    """Fit identical nonlinear probes for every feature view."""
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.metrics import (
        balanced_accuracy_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    train_views = build_feature_views(train)
    test_views = build_feature_views(test)
    targets = {
        "current_active": (
            (train.current_local_pd > 0.0).astype(np.int64),
            (test.current_local_pd > 0.0).astype(np.int64),
            "classification",
        ),
        "next_active": (
            (train.next_local_pd > 0.0).astype(np.int64),
            (test.next_local_pd > 0.0).astype(np.int64),
            "classification",
        ),
        "current_pd_given_active": (
            train.current_local_pd,
            test.current_local_pd,
            "conditional_regression",
        ),
        "next_pd_given_active": (
            train.next_local_pd,
            test.next_local_pd,
            "conditional_regression",
        ),
        "next_team_pd": (
            train.next_team_pd,
            test.next_team_pd,
            "regression",
        ),
        "distance_m": (
            train.distance_m,
            test.distance_m,
            "regression",
        ),
    }
    results: Dict[str, dict] = {}
    fitted_full: Dict[str, object] = {}
    for view_name, x_train in train_views.items():
        x_test = test_views[view_name]
        view_results: Dict[str, dict] = {}
        for target_name, (y_train, y_test, kind) in targets.items():
            train_selector = np.ones(len(train), dtype=bool)
            test_selector = np.ones(len(test), dtype=bool)
            if kind == "conditional_regression":
                train_selector = y_train > 0.0
                test_selector = y_test > 0.0
            if kind == "classification":
                if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                    view_results[target_name] = {"status": "single_class"}
                    continue
                model = HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=120,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=seed,
                )
                model.fit(
                    x_train,
                    y_train,
                    sample_weight=_balanced_sample_weights(y_train),
                )
                probability = model.predict_proba(x_test)[:, 1]
                prediction = probability >= 0.5
                metrics = {
                    "roc_auc": float(roc_auc_score(y_test, probability)),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_test, prediction)),
                    "positive_rate": float(np.mean(y_test)),
                }
            else:
                if train_selector.sum() < 20 or test_selector.sum() < 10:
                    view_results[target_name] = {"status": "insufficient_rows"}
                    continue
                model = HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=120,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=seed,
                )
                model.fit(
                    x_train[train_selector], y_train[train_selector])
                prediction = model.predict(x_test[test_selector])
                metrics = {
                    "mae": float(mean_absolute_error(
                        y_test[test_selector], prediction)),
                    "r2": float(r2_score(y_test[test_selector], prediction)),
                    "rows": int(test_selector.sum()),
                }
            view_results[target_name] = metrics
            if view_name == "full":
                fitted_full[target_name] = (model, kind, y_test, test_selector)
        results[view_name] = view_results

    # Evaluate the already-fitted full probes after semantic-only interventions.
    full_test = test_views["full"]
    target_ids = np.rint(
        test.metadata[:, 1] * max(int(np.max(test.metadata[:, 1]) > 0) * 3, 1)
    ).astype(np.int64)
    # Recover target ids robustly from the normalized metadata values.
    unique_target_values = np.unique(test.metadata[:, 1])
    target_lookup = {value: idx for idx, value in enumerate(unique_target_values)}
    target_ids = np.asarray([target_lookup[x] for x in test.metadata[:, 1]])
    for mode in ("zero", "permute"):
        x_intervened = intervene_semantic(
            full_test,
            metadata_dim=test.metadata.shape[1],
            mode=mode,
            target_ids=target_ids,
            seed=seed + 17,
        )
        mode_results: Dict[str, dict] = {}
        for target_name, (model, kind, y_test, selector) in fitted_full.items():
            if kind == "classification":
                probability = model.predict_proba(x_intervened)[:, 1]
                mode_results[target_name] = {
                    "roc_auc": float(roc_auc_score(y_test, probability)),
                    "balanced_accuracy": float(balanced_accuracy_score(
                        y_test, probability >= 0.5)),
                    "positive_rate": float(np.mean(y_test)),
                }
            else:
                prediction = model.predict(x_intervened[selector])
                mode_results[target_name] = {
                    "mae": float(mean_absolute_error(
                        y_test[selector], prediction)),
                    "r2": float(r2_score(y_test[selector], prediction)),
                    "rows": int(selector.sum()),
                }
        results[f"full_semantic_{mode}"] = mode_results
    return results


def fit_lightweight_semantic_decoder(
    train: ProbeDataset,
    test: ProbeDataset,
    seed: int,
    epochs: int = 80,
) -> dict:
    """Train the proposed lightweight token-only neural decoder offline.

    The decoder sees only quantized dimensions 1..D-1.  Its two outputs are an
    evidence-validity logit and conditional local P_D.  No actor parameter is
    touched, so this is a before-integration feasibility gate.
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.as_tensor(
        train.tokens[:, 1:], dtype=torch.float32, device=device)
    x_test = torch.as_tensor(
        test.tokens[:, 1:], dtype=torch.float32, device=device)
    pd_train = torch.as_tensor(
        train.current_local_pd, dtype=torch.float32, device=device)
    active_train = (pd_train > 0.0).to(torch.float32)
    active_count = float(active_train.sum().item())
    inactive_count = float(active_train.numel() - active_count)
    pos_weight = torch.as_tensor(
        inactive_count / max(active_count, 1.0),
        dtype=torch.float32,
        device=device,
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = min(512, len(train))
    final_loss = float("nan")
    model.train()
    for _ in range(max(1, int(epochs))):
        permutation = torch.randperm(
            len(train), generator=generator, device="cpu")
        for start in range(0, len(train), batch_size):
            index = permutation[start:start + batch_size].to(device)
            output = model(x_train[index])
            active = active_train[index]
            pd_value = pd_train[index]
            evidence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output[:, 0], active, pos_weight=pos_weight)
            pd_prediction = torch.sigmoid(output[:, 1])
            active_mask = active > 0.5
            if active_mask.any():
                pd_loss = torch.nn.functional.smooth_l1_loss(
                    pd_prediction[active_mask], pd_value[active_mask])
            else:
                pd_loss = torch.zeros((), device=device)
            loss = evidence_loss + 2.0 * pd_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.inference_mode():
        output = model(x_test)
        evidence_probability = torch.sigmoid(output[:, 0]).cpu().numpy()
        pd_prediction = torch.sigmoid(output[:, 1]).cpu().numpy()
    active_test = (test.current_local_pd > 0.0).astype(np.int64)
    active_selector = active_test > 0
    metrics = {
        "architecture": "15-32-16-2 MLP",
        "epochs": int(epochs),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "final_minibatch_loss": final_loss,
        "current_active_roc_auc": float(
            roc_auc_score(active_test, evidence_probability)),
        "current_active_balanced_accuracy": float(
            balanced_accuracy_score(
                active_test, evidence_probability >= 0.5)),
        "current_pd_given_active_mae": float(mean_absolute_error(
            test.current_local_pd[active_selector],
            pd_prediction[active_selector])),
        "current_pd_given_active_r2": float(r2_score(
            test.current_local_pd[active_selector],
            pd_prediction[active_selector])),
    }
    return metrics


def save_dataset_npz(path: Path, dataset: ProbeDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "metadata": dataset.metadata,
        "tokens": dataset.tokens,
        "current_local_pd": dataset.current_local_pd,
        "next_local_pd": dataset.next_local_pd,
        "next_team_pd": dataset.next_team_pd,
        "distance_m": dataset.distance_m,
        "episode_seed": dataset.episode_seed,
    }
    for name in (
        "sender_id", "target_id", "frame_index", "claim_count",
        "frame_seed", "frame_step", "frame_target_id",
        "frame_claim_count", "frame_next_team_pd",
    ):
        value = getattr(dataset, name)
        if value is not None:
            values[name] = value
    np.savez_compressed(path, **values)


def _summarize_decision(results: Mapping[str, dict]) -> dict:
    """Apply a conservative descriptive gate to incremental semantics."""
    full = results["full"]
    contract = results["contract"]
    zero = results["full_semantic_zero"]
    comparisons = {}
    informative_votes = 0
    for name in ("current_active", "next_active"):
        if "roc_auc" not in full[name]:
            continue
        gain = full[name]["roc_auc"] - contract[name]["roc_auc"]
        zero_drop = full[name]["roc_auc"] - zero[name]["roc_auc"]
        comparisons[name] = {
            "full_minus_contract_auc": float(gain),
            "full_minus_zero_auc": float(zero_drop),
        }
        informative_votes += int(gain >= 0.02 and zero_drop >= 0.02)
    for name in ("next_team_pd", "distance_m"):
        if "r2" not in full[name]:
            continue
        gain = full[name]["r2"] - contract[name]["r2"]
        zero_drop = full[name]["r2"] - zero[name]["r2"]
        comparisons[name] = {
            "full_minus_contract_r2": float(gain),
            "full_minus_zero_r2": float(zero_drop),
        }
        informative_votes += int(gain >= 0.05 and zero_drop >= 0.05)
    return {
        "comparisons": comparisons,
        "informative_votes": int(informative_votes),
        "verdict": (
            "existing_semantic_stream_is_informative"
            if informative_votes >= 2
            else "existing_semantic_stream_not_yet_reliably_informative"
        ),
        "note": "Thresholds are predeclared engineering gates, not p-values.",
    }


def _dataset_summary(dataset: ProbeDataset) -> dict:
    return {
        "rows": len(dataset),
        "episodes": int(np.unique(dataset.episode_seed).size),
        "current_active_rate": float(np.mean(dataset.current_local_pd > 0.0)),
        "next_active_rate": float(np.mean(dataset.next_local_pd > 0.0)),
        "mean_next_team_pd": float(np.mean(dataset.next_team_pd)),
        "mean_distance_m": float(np.mean(dataset.distance_m)),
        "zero_claim_frame_target_rate": (
            float(np.mean(dataset.frame_claim_count == 0))
            if dataset.frame_claim_count is not None else None),
        "underloaded_frame_target_rate": (
            float(np.mean(dataset.frame_claim_count < 2))
            if dataset.frame_claim_count is not None else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-policy target-token semantic probe")
    parser.add_argument(
        "--config",
        default=(
            "config/exp_800_q4_u2u_hierarchical_multistatic_"
            "distributed_matching_hybrid50_eval.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default="results/qos_pretrained_soft_gated_move50_test100/best_restored.pt",
    )
    parser.add_argument(
        "--seed-bank", default="config/stratified_seeds_800_q4.json")
    parser.add_argument("--train-split", default="selection")
    parser.add_argument("--test-split", default="confirmation")
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--output", default="results/token_semantic_probe/probe_smoke.json")
    parser.add_argument(
        "--dataset-dir", default="results/token_semantic_probe/datasets")
    parser.add_argument("--decoder-epochs", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = safe_torch_load(
        args.checkpoint,
        map_location="cpu",
        description="token-semantics probe checkpoint",
        optional_mapping_keys=("runtime",),
        optional_state_dict_keys=("actor",),
    )
    train_seeds = load_stratified_seed_split(
        args.seed_bank, args.train_split)[:max(1, args.train_episodes)]
    test_seeds = load_stratified_seed_split(
        args.seed_bank, args.test_split)[:max(1, args.test_episodes)]
    overlap = set(train_seeds) & set(test_seeds)
    if overlap:
        raise ValueError(f"train/test episode seeds overlap: {sorted(overlap)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe] device={device} train={train_seeds} test={test_seeds}")

    train = collect_dataset(cfg, checkpoint, train_seeds, device)
    test = collect_dataset(cfg, checkpoint, test_seeds, device)
    results = fit_probe_suite(train, test, args.seed)
    neural_decoder = fit_lightweight_semantic_decoder(
        train, test, args.seed, epochs=args.decoder_epochs)
    decision = _summarize_decision(results)
    dataset_dir = Path(args.dataset_dir)
    train_dataset_path = dataset_dir / "probe_train.npz"
    test_dataset_path = dataset_dir / "probe_test.npz"
    save_dataset_npz(train_dataset_path, train)
    save_dataset_npz(test_dataset_path, test)
    report = {
        "schema_version": 1,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed_bank": args.seed_bank,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_seeds": [int(x) for x in train_seeds],
        "test_seeds": [int(x) for x in test_seeds],
        "train": _dataset_summary(train),
        "test": _dataset_summary(test),
        "features": {
            "metadata": ["sender", "target", "movement_phase", "rate_bits"],
            "contract": "metadata + quantized token dimension 0 bid",
            "semantic": "quantized token dimensions 1..15",
        },
        "probe_model": (
            "HistGradientBoosting, episode-disjoint train/test, "
            "max_iter=120, max_leaf_nodes=15"),
        "results": results,
        "lightweight_semantic_decoder": neural_decoder,
        "datasets": {
            "train": str(train_dataset_path),
            "test": str(test_dataset_path),
        },
        "decision": decision,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"[probe] report={output}")


if __name__ == "__main__":
    main()
