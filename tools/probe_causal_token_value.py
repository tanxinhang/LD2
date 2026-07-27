#!/usr/bin/env python
"""Probe whether decentralized observations predict a token's causal QoS value.

This is deliberately an offline diagnostic, not a new deployed policy.  At a
frozen-policy state it changes exactly one sender/target token mask and rolls
the two branches forward with common initial simulator and policy state.  The
resulting finite-horizon QoS difference is then predicted from either strictly
local information or privileged centralized information.

The first-stage question is intentionally narrow: is sender-token marginal
value learnable at all?  A sequential token gate should only be added to the
actor if this probe succeeds on episode-disjoint seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from tools.pretrain_qos_commitment import build_agent
from tools.probe_token_semantics import quantize_target_tokens
from uav_isac.agents.trainer import load_stratified_seed_split
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.utils.seeding import set_seed


QOS_TARGETS = np.asarray([0.80, 0.70, 0.60], dtype=np.float64)


@dataclass
class PolicyRuntime:
    """Policy state held outside the simulator by the evaluation loop."""

    h_prev: Optional[torch.Tensor]
    movement_phase: int
    held_actions: Optional[Dict[str, Dict[str, object]]]

    def clone(self) -> "PolicyRuntime":
        return PolicyRuntime(
            h_prev=(None if self.h_prev is None else self.h_prev.detach().clone()),
            movement_phase=int(self.movement_phase),
            held_actions=copy.deepcopy(self.held_actions),
        )


@dataclass
class PolicyDecision:
    """One deterministic joint action before communication-mask intervention."""

    actions: Dict[str, Dict[str, object]]
    messages: np.ndarray
    rates: np.ndarray
    token_masks: np.ndarray
    claim_scores: np.ndarray
    policy_latent: np.ndarray
    comm_fractions: Optional[np.ndarray]
    sensing_weights: Optional[np.ndarray]
    next_runtime: PolicyRuntime


@dataclass(frozen=True)
class CausalDataset:
    """Episode-disjoint rows for decentralized and centralized probes."""

    protocol: np.ndarray
    local_raw: np.ndarray
    local: np.ndarray
    central: np.ndarray
    delta_qos: np.ndarray
    delta_worst: np.ndarray
    delta_weak3: np.ndarray
    delta_steady: np.ndarray
    delta_feasible: np.ndarray
    delta_bits: np.ndarray
    send_pd: np.ndarray
    drop_pd: np.ndarray
    delta_pd: np.ndarray
    send_worst_id: np.ndarray
    drop_worst_id: np.ndarray
    episode_seed: np.ndarray
    frame: np.ndarray
    sender: np.ndarray
    target: np.ndarray
    rank: np.ndarray

    def __len__(self) -> int:
        return int(self.delta_qos.size)


class FrozenPolicyRunner:
    """Exact deterministic action path used for both intervention branches."""

    def __init__(self, cfg, checkpoint: Mapping, device: str):
        self.cfg = cfg
        self.device = device
        self.env_template = UAVISACEnv(config=cfg, seed=0)
        self.agent = build_agent(cfg, self.env_template, device)
        actor_state = checkpoint.get("actor", checkpoint)
        missing, unexpected = self.agent.load_actor_state_dict_compatible(actor_state)
        if unexpected:
            raise RuntimeError(
                f"unexpected checkpoint actor keys: {unexpected[:5]}")
        if missing:
            print(f"[causal-value] compatible load initialized {len(missing)} keys")
        runtime = checkpoint.get("runtime", {}) if isinstance(
            checkpoint, Mapping) else {}
        actor = self.agent.actor
        if hasattr(actor, "set_capacity_matching_blend"):
            actor.set_capacity_matching_blend(float(runtime.get(
                "capacity_matching_blend",
                cfg.marl.capacity_matching_blend_start,
            )))
        if hasattr(actor, "set_target_allocation_movement_blend"):
            actor.set_target_allocation_movement_blend(float(runtime.get(
                "target_allocation_movement_blend",
                cfg.marl.target_allocation_movement_blend,
            )))
        self.actor = actor.eval()
        self.action_space = self.agent.action_space
        self.K = int(cfg.scenario.K)
        self.Q = int(cfg.scenario.Q)
        self.token_dim = int(cfg.marl.comm_target_token_dim)
        self.movement_interval = max(
            1, int(cfg.marl.movement_decision_interval))
        self.rate_levels = np.asarray(
            cfg.marl.comm_rate_bits_per_dim, dtype=np.int64)

    def new_env(self, seed: int) -> Tuple[UAVISACEnv, Dict[str, np.ndarray]]:
        env = UAVISACEnv(config=self.cfg, seed=int(seed))
        obs, _ = env.reset(seed=int(seed))
        return env, obs

    def decide(
        self,
        obs: Dict[str, np.ndarray],
        runtime: PolicyRuntime,
    ) -> PolicyDecision:
        obs_batch = np.stack([obs[str(k)] for k in range(self.K)])
        phase_value = float(runtime.movement_phase) / max(
            self.movement_interval - 1, 1)
        with torch.inference_mode():
            obs_t = torch.as_tensor(
                obs_batch, dtype=torch.float32, device=self.device)
            phase_t = torch.full(
                (self.K,), phase_value, dtype=torch.float32,
                device=self.device)
            identity_t = torch.arange(
                self.K, dtype=torch.long, device=self.device)
            (dp_mean, dp_log_std, role_logits, comm_mean, _, h_new) = (
                self.actor(
                    obs_t,
                    runtime.h_prev,
                    comm_round_phase=phase_t,
                    agent_identity=identity_t,
                )
            )
            token_mask_t = getattr(
                self.actor, "last_outgoing_token_mask", None)
            if token_mask_t is None:
                token_mask_t = torch.ones(
                    (self.K, self.Q), dtype=comm_mean.dtype,
                    device=self.device)
            claim_scores_t = getattr(
                self.actor, "last_sparse_claim_scores", None)
            if claim_scores_t is None:
                claim_scores_t = token_mask_t
            comm_action, rate_action, _, _ = (
                self.agent.sample_communication(
                    comm_mean, deterministic=True))
            if bool(self.cfg.marl.joint_isac_power_enabled):
                (_, _, comm_fraction, sensing_weights, _, _) = (
                    self.agent.sample_isac_resources(
                        comm_mean,
                        rate_action,
                        deterministic=True,
                        comm_fraction_min=float(
                            self.cfg.marl.comm_power_fraction_min),
                        comm_fraction_max=float(
                            self.cfg.marl.comm_power_fraction_max),
                    )
                )
            else:
                comm_fraction = sensing_weights = None

        dpm = dp_mean.detach().cpu().numpy()
        dps = dp_log_std.detach().cpu().numpy()
        roles = role_logits.detach().cpu().numpy()
        proposed: Dict[str, Dict[str, object]] = {}
        for k in range(self.K):
            dp_std_k = dps if dps.ndim == 1 else dps[k]
            action, _ = self.action_space.decode(
                dpm[k], dp_std_k, roles[k],
                dp_deterministic=True,
                role_deterministic=True,
            )
            proposed[str(k)] = {
                "delta_p": np.asarray(action.delta_p).copy(),
                "role": int(action.role),
            }

        if runtime.movement_phase == 0 or runtime.held_actions is None:
            held_actions = copy.deepcopy(proposed)
            actions = copy.deepcopy(proposed)
        else:
            held_actions = copy.deepcopy(runtime.held_actions)
            actions = copy.deepcopy(runtime.held_actions)

        next_runtime = PolicyRuntime(
            h_prev=(None if h_new is None else h_new.detach().clone()),
            movement_phase=(runtime.movement_phase + 1) % self.movement_interval,
            held_actions=held_actions,
        )
        return PolicyDecision(
            actions=actions,
            messages=comm_action.detach().cpu().numpy(),
            rates=rate_action.detach().cpu().numpy().astype(np.int64),
            token_masks=token_mask_t.detach().cpu().numpy(),
            claim_scores=claim_scores_t.detach().cpu().numpy(),
            policy_latent=self.actor.last_policy_latent.detach().cpu().numpy(),
            comm_fractions=(None if comm_fraction is None else
                            comm_fraction.detach().cpu().numpy()),
            sensing_weights=(None if sensing_weights is None else
                             sensing_weights.detach().cpu().numpy()),
            next_runtime=next_runtime,
        )

    def step(
        self,
        env: UAVISACEnv,
        obs: Dict[str, np.ndarray],
        runtime: PolicyRuntime,
        token_override: Optional[Tuple[int, int, bool]] = None,
    ) -> Tuple[Dict[str, np.ndarray], PolicyRuntime, dict, bool, PolicyDecision]:
        decision = self.decide(obs, runtime)
        masks = decision.token_masks.copy()
        if token_override is not None:
            sender, target, enabled = token_override
            masks[int(sender), int(target)] = float(bool(enabled))

        messages = {
            k: decision.messages[k].copy() for k in range(self.K)}
        rates = {k: int(decision.rates[k]) for k in range(self.K)}
        mask_map = {k: masks[k].copy() for k in range(self.K)}
        if bool(self.cfg.marl.joint_isac_power_enabled):
            assert decision.comm_fractions is not None
            assert decision.sensing_weights is not None
            env.core.submit_learned_communications(
                messages,
                rates,
                {k: float(decision.comm_fractions[k])
                 for k in range(self.K)},
                {k: decision.sensing_weights[k].copy()
                 for k in range(self.K)},
                token_masks=mask_map,
            )
        else:
            env.core.submit_learned_communications(
                messages, rates, token_masks=mask_map)
        next_obs, _, terminated, truncated, info = env.step(decision.actions)
        done = bool(
            terminated.get("__all__", False)
            or truncated.get("__all__", False))
        return next_obs, decision.next_runtime, info, done, decision

    def close(self) -> None:
        self.env_template.close()


def qos_components(pd: np.ndarray) -> Tuple[float, float, float, float]:
    """Return steady, weak3, strict worst, and Medium-QoS feasibility."""
    values = np.asarray(pd, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    ordered = np.sort(values)
    steady = float(np.mean(values))
    weak3 = float(np.mean(ordered[:min(3, ordered.size)]))
    worst = float(ordered[0])
    feasible = float(
        steady >= QOS_TARGETS[0]
        and weak3 >= QOS_TARGETS[1]
        and worst >= QOS_TARGETS[2])
    return steady, weak3, worst, feasible


def summarize_branch(
    infos: Sequence[Mapping],
    gamma: float = 0.95,
) -> Dict[str, object]:
    """Discount a finite branch without mixing resource cost into QoS value."""
    if not infos:
        return {
            "steady": 0.0, "weak3": 0.0, "worst": 0.0,
            "feasible": 0.0, "qos": 0.0, "bits": 0.0,
            "pd": np.zeros(0, dtype=np.float64), "worst_id": -1,
        }
    weights = np.power(float(gamma), np.arange(len(infos), dtype=np.float64))
    weights /= max(float(np.sum(weights)), 1e-12)
    pd_rows = np.stack([
        np.asarray(info["P_D_q"], dtype=np.float64) for info in infos])
    discounted_pd = np.sum(pd_rows * weights[:, None], axis=0)
    rows = np.asarray([
        qos_components(pd) for pd in pd_rows
    ], dtype=np.float64)
    steady, weak3, worst, feasible = np.sum(
        rows * weights[:, None], axis=0)
    # Feasibility dominates; the continuous terms provide resolution when both
    # branches have the same feasible-frame fraction.
    qos = 2.0 * feasible + worst + 0.25 * weak3 + 0.10 * steady
    bits = float(np.sum([
        float(info.get("learned_comm_bits", 0.0)) for info in infos]))
    return {
        "steady": float(steady),
        "weak3": float(weak3),
        "worst": float(worst),
        "feasible": float(feasible),
        "qos": float(qos),
        "bits": bits,
        "pd": discounted_pd,
        "worst_id": int(np.argmin(discounted_pd)),
    }


def rollout_branch(
    runner: FrozenPolicyRunner,
    env: UAVISACEnv,
    obs: Dict[str, np.ndarray],
    runtime: PolicyRuntime,
    horizon: int,
    first_override: Tuple[int, int, bool],
    gamma: float,
) -> Dict[str, object]:
    """Roll a deep-copied branch; later actions react to intervened messages."""
    branch_env = copy.deepcopy(env)
    branch_obs = {key: value.copy() for key, value in obs.items()}
    branch_runtime = runtime.clone()
    infos: List[Mapping] = []
    try:
        for step in range(max(1, int(horizon))):
            override = first_override if step == 0 else None
            branch_obs, branch_runtime, info, done, _ = runner.step(
                branch_env,
                branch_obs,
                branch_runtime,
                token_override=override,
            )
            infos.append(info)
            if done:
                break
    finally:
        branch_env.close()
    return summarize_branch(infos, gamma=gamma)


def token_rank(claim_scores: np.ndarray, sender: int, target: int) -> int:
    order = np.argsort(-np.asarray(claim_scores[sender]), kind="stable")
    return int(np.flatnonzero(order == int(target))[0]) + 1


def choose_candidate(
    decision: PolicyDecision,
    sample_index: int,
) -> Optional[Tuple[int, int, int]]:
    """Cycle deterministically across all physically active sender-token edges."""
    active: List[Tuple[int, int, int]] = []
    for sender in range(decision.token_masks.shape[0]):
        if decision.rates[sender] <= 0:
            continue
        for target in np.flatnonzero(decision.token_masks[sender] > 0.5):
            active.append((
                int(sender), int(target),
                token_rank(decision.claim_scores, sender, int(target)),
            ))
    if not active:
        return None
    return active[int(sample_index) % len(active)]


def build_features(
    runner: FrozenPolicyRunner,
    env: UAVISACEnv,
    obs: Dict[str, np.ndarray],
    runtime: PolicyRuntime,
    decision: PolicyDecision,
    sender: int,
    target: int,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build protocol, strictly local, and privileged centralized views."""
    quantized = quantize_target_tokens(
        decision.messages,
        decision.rates,
        np.ones_like(decision.token_masks),
        env.core._inter_uav_comm,
        runner.Q,
        runner.token_dim,
    )
    rate_bits = int(runner.rate_levels[int(decision.rates[sender])])
    phase = float(runtime.movement_phase) / max(
        runner.movement_interval - 1, 1)
    protocol = np.asarray([
        sender / max(runner.K - 1, 1),
        target / max(runner.Q - 1, 1),
        rank / max(runner.Q, 1),
        phase,
        rate_bits / max(int(runner.rate_levels[-1]), 1),
        float(np.sum(decision.token_masks[sender])) / max(runner.Q, 1),
    ], dtype=np.float64)
    # The sender knows its local observation, all locally generated candidates,
    # and its own selected set.  It does not receive other agents' hidden state.
    local_raw = np.concatenate([
        protocol,
        np.asarray(obs[str(sender)], dtype=np.float64).reshape(-1),
        quantized[sender].reshape(-1),
        decision.token_masks[sender].astype(np.float64),
    ])
    # This compact state is the actor's own decentralized policy embedding.
    # It has already compressed local geometry, message history, and target
    # context, avoiding an ill-posed raw-observation fit on a small probe set.
    local = np.concatenate([
        protocol,
        decision.policy_latent[sender].astype(np.float64).reshape(-1),
        quantized[sender].reshape(-1),
        decision.token_masks[sender].astype(np.float64),
    ])
    # Central view is a diagnostic ceiling, never a proposed execution input.
    central = np.concatenate([
        local,
        np.asarray(env.core.get_global_state(), dtype=np.float64).reshape(-1),
        quantized.reshape(-1),
        decision.token_masks.astype(np.float64).reshape(-1),
    ])
    return protocol, local_raw, local, central


def collect_dataset(
    runner: FrozenPolicyRunner,
    seeds: Sequence[int],
    horizon: int,
    intervention_stride: int,
    max_samples_per_episode: int,
    gamma: float,
) -> CausalDataset:
    rows: Dict[str, List] = {
        name: [] for name in (
            "protocol", "local_raw", "local", "central", "delta_qos", "delta_worst",
            "delta_weak3", "delta_steady", "delta_feasible", "delta_bits",
            "send_pd", "drop_pd", "delta_pd", "send_worst_id",
            "drop_worst_id", "episode_seed", "frame", "sender", "target",
            "rank",
        )
    }
    sample_index = 0
    stride = max(1, int(intervention_stride))
    for episode_index, episode_seed in enumerate(seeds):
        env, obs = runner.new_env(int(episode_seed))
        runtime = PolicyRuntime(None, 0, None)
        done = False
        frame = 0
        episode_samples = 0
        try:
            while not done:
                preview = runner.decide(obs, runtime)
                should_sample = (
                    frame % stride == 0
                    and (max_samples_per_episode <= 0
                         or episode_samples < max_samples_per_episode)
                )
                candidate = (
                    choose_candidate(preview, sample_index)
                    if should_sample else None)
                if candidate is not None:
                    sender, target, rank = candidate
                    protocol, local_raw, local, central = build_features(
                        runner, env, obs, runtime, preview,
                        sender, target, rank)
                    send = rollout_branch(
                        runner, env, obs, runtime, horizon,
                        (sender, target, True), gamma)
                    drop = rollout_branch(
                        runner, env, obs, runtime, horizon,
                        (sender, target, False), gamma)
                    rows["protocol"].append(protocol)
                    rows["local_raw"].append(local_raw)
                    rows["local"].append(local)
                    rows["central"].append(central)
                    for name in (
                        "qos", "worst", "weak3", "steady", "feasible",
                    ):
                        rows[f"delta_{name}"].append(send[name] - drop[name])
                    rows["delta_bits"].append(send["bits"] - drop["bits"])
                    send_pd = np.asarray(send["pd"], dtype=np.float64)
                    drop_pd = np.asarray(drop["pd"], dtype=np.float64)
                    rows["send_pd"].append(send_pd)
                    rows["drop_pd"].append(drop_pd)
                    rows["delta_pd"].append(send_pd - drop_pd)
                    rows["send_worst_id"].append(int(send["worst_id"]))
                    rows["drop_worst_id"].append(int(drop["worst_id"]))
                    rows["episode_seed"].append(int(episode_seed))
                    rows["frame"].append(int(frame))
                    rows["sender"].append(int(sender))
                    rows["target"].append(int(target))
                    rows["rank"].append(int(rank))
                    sample_index += 1
                    episode_samples += 1

                # Continue the untouched frozen-policy trajectory.  Branches
                # were deep copies, so intervention outcomes cannot leak here.
                obs, runtime, _, done, _ = runner.step(env, obs, runtime)
                frame += 1
        finally:
            env.close()
        print(
            f"[causal-value] episode {episode_index + 1}/{len(seeds)} "
            f"seed={episode_seed} samples={episode_samples}")

    if not rows["delta_qos"]:
        raise RuntimeError("no active sender-token candidates were collected")
    float_matrix = (
        "protocol", "local_raw", "local", "central",
        "send_pd", "drop_pd", "delta_pd",
    )
    int_vector = (
        "send_worst_id", "drop_worst_id", "episode_seed", "frame",
        "sender", "target", "rank",
    )
    values = {}
    for name, value in rows.items():
        if name in float_matrix:
            values[name] = np.stack(value).astype(np.float64)
        elif name in int_vector:
            values[name] = np.asarray(value, dtype=np.int64)
        else:
            values[name] = np.asarray(value, dtype=np.float64)
    return CausalDataset(**values)


def _rank_correlation(y: np.ndarray, prediction: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.unique(y).size < 2 or np.unique(prediction).size < 2:
        return 0.0
    value = spearmanr(y, prediction).statistic
    return float(value) if np.isfinite(value) else 0.0


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = np.ones(labels.size, dtype=np.float64)
    for cls in (0, 1):
        if counts[cls] > 0:
            weights[labels == cls] = labels.size / (2.0 * counts[cls])
    return weights


def _choose_gate_threshold(prediction: np.ndarray, delta: np.ndarray) -> float:
    candidates = np.unique(np.quantile(
        prediction, np.linspace(0.0, 1.0, 101)))
    best = (-np.inf, float(np.median(prediction)))
    for threshold in candidates:
        utility = float(np.mean(delta * (prediction > threshold)))
        if utility > best[0]:
            best = (utility, float(threshold))
    return best[1]


def fit_probe_view(
    x_train: np.ndarray,
    x_test: np.ndarray,
    train: CausalDataset,
    test: CausalDataset,
    seed: int,
    neutral_epsilon: float,
) -> dict:
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

    reg = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        random_state=seed,
    )
    reg.fit(x_train, train.delta_qos)
    pred_train = reg.predict(x_train)
    pred_test = reg.predict(x_test)

    train_label = (train.delta_qos > neutral_epsilon).astype(np.int64)
    test_label = (test.delta_qos > neutral_epsilon).astype(np.int64)
    classification = {"status": "single_class"}
    if np.unique(train_label).size == 2 and np.unique(test_label).size == 2:
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=seed,
        )
        classifier.fit(
            x_train,
            train_label,
            sample_weight=_balanced_weights(train_label),
        )
        probability = classifier.predict_proba(x_test)[:, 1]
        classification = {
            "roc_auc": float(roc_auc_score(test_label, probability)),
            "balanced_accuracy": float(balanced_accuracy_score(
                test_label, probability >= 0.5)),
            "positive_rate": float(np.mean(test_label)),
        }

    threshold = _choose_gate_threshold(pred_train, train.delta_qos)
    gate = pred_test > threshold
    oracle_gain = float(np.mean(np.maximum(test.delta_qos, 0.0)))
    gate_gain = float(np.mean(test.delta_qos * gate))
    always_send_gain = float(np.mean(test.delta_qos))
    positive = test.delta_qos > neutral_epsilon
    critical_cut = float(np.quantile(
        test.delta_qos[positive], 0.75)) if np.any(positive) else np.inf
    critical = test.delta_qos >= critical_cut
    critical_recall = (
        float(np.mean(gate[critical])) if np.any(critical) else None)
    return {
        "regression": {
            "mae": float(mean_absolute_error(test.delta_qos, pred_test)),
            "r2": float(r2_score(test.delta_qos, pred_test)),
            "spearman": _rank_correlation(test.delta_qos, pred_test),
        },
        "classification": classification,
        "gate": {
            "threshold_from_train": float(threshold),
            "activation_rate": float(np.mean(gate)),
            "always_send_gain_vs_drop": always_send_gain,
            "predicted_gate_gain_vs_drop": gate_gain,
            "hindsight_oracle_gain_vs_drop": oracle_gain,
            "oracle_regret": float(oracle_gain - gate_gain),
            "recovered_oracle_fraction": (
                float(gate_gain / oracle_gain)
                if oracle_gain > 1e-12 else None),
            "critical_positive_recall": critical_recall,
        },
    }


def fit_probe_suite(
    train: CausalDataset,
    test: CausalDataset,
    seed: int,
    neutral_epsilon: float,
) -> Dict[str, dict]:
    views = {
        "protocol": (train.protocol, test.protocol),
        "local_raw": (train.local_raw, test.local_raw),
        "local": (train.local, test.local),
        "central": (train.central, test.central),
    }
    return {
        name: fit_probe_view(
            x_train, x_test, train, test, seed, neutral_epsilon)
        for name, (x_train, x_test) in views.items()
    }


def dataset_summary(data: CausalDataset, neutral_epsilon: float) -> dict:
    delta = data.delta_qos
    return {
        "rows": len(data),
        "episodes": int(np.unique(data.episode_seed).size),
        "positive_rate": float(np.mean(delta > neutral_epsilon)),
        "negative_rate": float(np.mean(delta < -neutral_epsilon)),
        "neutral_rate": float(np.mean(np.abs(delta) <= neutral_epsilon)),
        "delta_qos_mean": float(np.mean(delta)),
        "delta_qos_std": float(np.std(delta)),
        "delta_worst_mean": float(np.mean(data.delta_worst)),
        "delta_worst_std": float(np.std(data.delta_worst)),
        "delta_feasible_mean": float(np.mean(data.delta_feasible)),
        "delta_bits_mean": float(np.mean(data.delta_bits)),
        "delta_pd_mae": float(np.mean(np.abs(data.delta_pd))),
        "worst_identity_change_rate": float(np.mean(
            data.send_worst_id != data.drop_worst_id)),
        "rank_counts": {
            str(int(rank)): int(np.sum(data.rank == rank))
            for rank in np.unique(data.rank)
        },
    }


def save_dataset(path: Path, data: CausalDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{
        name: getattr(data, name) for name in data.__dataclass_fields__
    })


def build_verdict(results: Mapping[str, dict]) -> dict:
    local = results["local"]
    central = results["central"]
    local_auc = local["classification"].get("roc_auc")
    local_spearman = local["regression"]["spearman"]
    local_fraction = local["gate"]["recovered_oracle_fraction"]
    learnable = bool(
        local_auc is not None
        and local_auc >= 0.65
        and local_spearman >= 0.20
        and local_fraction is not None
        and local_fraction >= 0.50)
    return {
        "verdict": (
            "local_causal_value_is_promising"
            if learnable else
            "do_not_integrate_gate_yet"
        ),
        "predeclared_gates": {
            "local_auc_min": 0.65,
            "local_spearman_min": 0.20,
            "recovered_oracle_fraction_min": 0.50,
        },
        "local": {
            "roc_auc": local_auc,
            "spearman": local_spearman,
            "recovered_oracle_fraction": local_fraction,
        },
        "central_minus_local": {
            "spearman": float(
                central["regression"]["spearman"] - local_spearman),
            "roc_auc": (
                None if local_auc is None
                or central["classification"].get("roc_auc") is None
                else float(
                    central["classification"]["roc_auc"] - local_auc)),
        },
        "note": (
            "These are engineering feasibility gates on a small diagnostic, "
            "not statistical proof or a deployable score."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sender-token paired-intervention causal-value probe")
    parser.add_argument(
        "--config",
        default=(
            "config/exp_800_q4_u2u_hierarchical_multistatic_"
            "distributed_matching_hybrid50_eval.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "results/qos_pretrained_soft_gated_move50_test100/"
            "best_restored.pt"),
    )
    parser.add_argument(
        "--seed-bank", default="config/stratified_seeds_800_q4.json")
    parser.add_argument("--train-split", default="selection")
    parser.add_argument("--test-split", default="confirmation")
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--test-episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--intervention-stride", type=int, default=5)
    parser.add_argument("--max-samples-per-episode", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--neutral-epsilon", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--output",
        default="results/causal_token_value_probe/report.json")
    parser.add_argument(
        "--dataset-dir",
        default="results/causal_token_value_probe/datasets")
    args = parser.parse_args()

    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    cfg = load_config(args.config)
    if str(cfg.marl.comm_payload_mode).lower() != "target_tokens":
        raise ValueError("causal token probe requires target_tokens payload")
    if str(cfg.marl.learned_comm_mode).lower() != "cost_aware":
        raise ValueError("causal token probe requires cost-aware communication")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False)
    train_seeds = load_stratified_seed_split(
        args.seed_bank, args.train_split)[:max(1, args.train_episodes)]
    test_seeds = load_stratified_seed_split(
        args.seed_bank, args.test_split)[:max(1, args.test_episodes)]
    overlap = set(train_seeds) & set(test_seeds)
    if overlap:
        raise ValueError(f"train/test episode seeds overlap: {sorted(overlap)}")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[causal-value] device={device} H={args.horizon} "
        f"train={train_seeds} test={test_seeds}")
    runner = FrozenPolicyRunner(cfg, checkpoint, device)
    try:
        train = collect_dataset(
            runner, train_seeds, args.horizon, args.intervention_stride,
            args.max_samples_per_episode, args.gamma)
        test = collect_dataset(
            runner, test_seeds, args.horizon, args.intervention_stride,
            args.max_samples_per_episode, args.gamma)
    finally:
        runner.close()

    results = fit_probe_suite(
        train, test, args.seed, args.neutral_epsilon)
    verdict = build_verdict(results)
    dataset_dir = Path(args.dataset_dir)
    train_path = dataset_dir / "train.npz"
    test_path = dataset_dir / "test.npz"
    save_dataset(train_path, train)
    save_dataset(test_path, test)
    report = {
        "schema_version": 2,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "seed_bank": args.seed_bank,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "train_seeds": [int(x) for x in train_seeds],
        "test_seeds": [int(x) for x in test_seeds],
        "intervention": {
            "unit": "one sender-target token mask",
            "paired_common_state": True,
            "horizon": int(args.horizon),
            "gamma": float(args.gamma),
            "stride": int(args.intervention_stride),
            "neutral_epsilon": float(args.neutral_epsilon),
            "qos_targets": QOS_TARGETS.tolist(),
            "qos_scalar": "2*feasible + worst + 0.25*weak3 + 0.10*steady",
        },
        "train": dataset_summary(train, args.neutral_epsilon),
        "test": dataset_summary(test, args.neutral_epsilon),
        "results": results,
        "verdict": verdict,
        "datasets": {"train": str(train_path), "test": str(test_path)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    print(f"[causal-value] report={output}")


if __name__ == "__main__":
    main()
