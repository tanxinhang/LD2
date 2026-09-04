#!/usr/bin/env python
"""Offline pretrain the distributed slow commitment head with QoS labels.

The teacher sees centralized geometry and the previous team P_D only while
creating labels.  The optimized head receives the unchanged decentralized
actor observation; communication/resource/movement output heads are frozen.
"""

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
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import (
    compute_qos_bistatic_assignment_teacher,
    load_training_seed_pool,
)
from uav_isac.environment.action import ActionSpace
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.utils.checkpoint_loading import safe_torch_load, validate_state_dict
from uav_isac.utils.seeding import set_seed


def build_agent(cfg, env: UAVISACEnv, device: str) -> MAPPOAgent:
    """Construct the same structured actor used by ``run_mappo.py``."""
    k_count, q_count = cfg.scenario.K, cfg.scenario.Q
    payload_mode = str(cfg.marl.comm_payload_mode).lower()
    target_tokens = payload_mode == 'target_tokens'
    target_dim = int(cfg.marl.comm_target_token_dim)
    payload_dim = q_count * target_dim if target_tokens else 16
    receiver_token_dim = target_dim + 6 if target_tokens else 21
    tokens_per_sender = q_count if target_tokens else 1
    action_space = ActionSpace(
        cfg.uav.v_max, cfg.scenario.dt, learn_roles=cfg.marl.learn_roles)
    action_space.num_targets = q_count
    action_space.structured_actor = True
    action_space.structured_entity_dim = 64
    obs, _ = env.reset(seed=0)
    obs_dim = obs['0'].shape[-1]
    schedule_enabled = int(getattr(
        cfg.marl, 'target_allocation_movement_blend_anneal_frames', 0)) > 0
    movement_blend = (
        cfg.marl.target_allocation_movement_blend_start
        if schedule_enabled else cfg.marl.target_allocation_movement_blend)
    return MAPPOAgent(
        agent_id=0,
        obs_dim=obs_dim,
        global_state_dim=env.core.obs_builder.get_global_state_dim(),
        action_space=action_space,
        num_agents=k_count,
        num_targets=q_count,
        hidden_layers=cfg.marl.hidden_layers,
        lr=cfg.marl.lr,
        critic_lr_mult=cfg.marl.critic_lr_mult,
        max_grad_norm=cfg.marl.max_grad_norm,
        device=device,
        centralized_critic=cfg.marl.centralized_critic,
        comm_num_rate_levels=len(cfg.marl.comm_rate_bits_per_dim),
        comm_log_std_init=cfg.marl.comm_message_log_std_init,
        comm_entropy_scale=cfg.marl.comm_entropy_scale,
        isac_power_log_std_init=cfg.marl.isac_power_log_std_init,
        sensing_allocation_log_std_init=(
            cfg.marl.sensing_allocation_log_std_init),
        use_comm_cross_attention=cfg.marl.comm_cross_attention_enabled,
        comm_token_dim=receiver_token_dim,
        comm_tokens_per_sender=tokens_per_sender,
        comm_payload_dim=payload_dim,
        comm_target_token_enabled=target_tokens,
        comm_target_token_dim=target_dim,
        use_target_allocation=bool(
            cfg.marl.target_allocation_enabled
            or cfg.marl.target_allocation_teacher_enabled),
        use_team_sinkhorn=cfg.marl.target_allocation_sinkhorn_enabled,
        capacity_matching_enabled=cfg.marl.capacity_matching_enabled,
        capacity_matching_row_capacity=cfg.marl.capacity_matching_row_capacity,
        capacity_matching_column_capacity=(
            cfg.marl.capacity_matching_column_capacity),
        capacity_matching_temperature=cfg.marl.capacity_matching_temperature,
        capacity_matching_iterations=cfg.marl.capacity_matching_iterations,
        capacity_matching_blend=cfg.marl.capacity_matching_blend_start,
        target_allocation_temperature=cfg.marl.target_allocation_temperature,
        target_allocation_straight_through=(
            cfg.marl.target_allocation_straight_through),
        target_allocation_movement_blend=movement_blend,
        target_allocation_movement_confidence_gating_enabled=bool(getattr(
            cfg.marl,
            'target_allocation_movement_confidence_gating_enabled', False)),
        target_allocation_movement_confidence_floor=float(getattr(
            cfg.marl,
            'target_allocation_movement_confidence_floor', 0.0)),
        target_allocation_movement_confidence_power=float(getattr(
            cfg.marl,
            'target_allocation_movement_confidence_power', 2.0)),
        hierarchical_dual_assignment_enabled=(
            cfg.marl.hierarchical_dual_assignment_enabled),
        movement_team_matching_enabled=bool(getattr(
            cfg.marl, 'movement_team_matching_enabled', False)),
        movement_team_matching_temperature=float(getattr(
            cfg.marl, 'movement_team_matching_temperature', 0.35)),
        movement_team_matching_iterations=int(getattr(
            cfg.marl, 'movement_team_matching_iterations', 16)),
        movement_team_matching_blend=float(getattr(
            cfg.marl, 'movement_team_matching_blend', 0.0)),
        movement_team_matching_intrinsic_bid_mix=float(getattr(
            cfg.marl, 'movement_team_matching_intrinsic_bid_mix', 0.0)),
        target_allocation_resource_blend=(
            cfg.marl.target_allocation_resource_blend),
        round_negotiation_enabled=cfg.marl.round_negotiation_enabled,
        round_negotiation_strength=cfg.marl.round_negotiation_strength,
        round_negotiation_temperature=cfg.marl.round_negotiation_temperature,
        sparse_claim_enabled=cfg.marl.sparse_claim_enabled,
        sparse_claim_share_topk=cfg.marl.sparse_claim_share_topk,
        sparse_claim_desired_endpoints=cfg.marl.sparse_claim_desired_endpoints,
        sparse_claim_full_penalty=cfg.marl.sparse_claim_full_penalty,
        sparse_claim_vacant_bonus=cfg.marl.sparse_claim_vacant_bonus,
        sparse_claim_temperature=cfg.marl.sparse_claim_temperature,
        comm_aided_sensing_enabled=cfg.marl.comm_aided_sensing_enabled,
        comm_aided_sensing_blend=cfg.marl.comm_aided_sensing_blend,
        comm_semantic_decoder_enabled=bool(getattr(
            cfg.marl, 'comm_semantic_decoder_enabled', False)),
        comm_semantic_capacity_bid_enabled=bool(getattr(
            cfg.marl, 'comm_semantic_capacity_bid_enabled', False)),
        comm_semantic_capacity_bid_gain=float(getattr(
            cfg.marl, 'comm_semantic_capacity_bid_gain', 0.0)),
        comm_semantic_extra_token_enabled=bool(getattr(
            cfg.marl, 'comm_semantic_extra_token_enabled', False)),
        comm_semantic_extra_token_threshold=float(getattr(
            cfg.marl, 'comm_semantic_extra_token_threshold', 0.10)),
        semantic_kinematic_field_enabled=(
            cfg.marl.semantic_kinematic_field_enabled),
        semantic_kinematic_field_gain=cfg.marl.semantic_kinematic_field_gain,
        target_conditioned_movement_enabled=(
            cfg.marl.target_conditioned_movement_enabled),
        target_conditioned_movement_gain=(
            cfg.marl.target_conditioned_movement_gain),
        scale_equivariant_comm_heads_enabled=bool(getattr(
            cfg.marl, 'scale_equivariant_comm_heads_enabled', False)),
        permutation_equivariant_round_encoding_enabled=bool(getattr(
            cfg.marl,
            'permutation_equivariant_round_encoding_enabled', False)),
        set_risk_critic_enabled=bool(getattr(
            cfg.marl, 'set_risk_critic_enabled', False)),
        risk_critic_hidden_dim=int(getattr(
            cfg.marl, 'risk_critic_hidden_dim', 128)),
        risk_critic_num_quantiles=int(getattr(
            cfg.marl, 'risk_critic_num_quantiles', 16)),
        risk_critic_cvar_alpha=float(getattr(
            cfg.marl, 'risk_critic_cvar_alpha', 0.20)),
        risk_critic_monotonic_quantiles_enabled=bool(getattr(
            cfg.marl,
            'risk_critic_monotonic_quantiles_enabled', False)),
    )


def choose_geometry_spread(
    seeds: np.ndarray,
    difficulty: np.ndarray,
    count: int,
) -> np.ndarray:
    """Deterministically cover the full training-pool difficulty range."""
    order = np.argsort(difficulty)
    count = min(max(1, int(count)), len(order))
    indices = np.linspace(0, len(order) - 1, count).round().astype(int)
    return seeds[order[indices]].astype(np.int64)


def generate_dataset(cfg, seeds, commitments: int, hold_frames: int):
    """Roll out persistent QoS labels through the exact physical environment."""
    # Label generation must observe every physically viable reporting edge;
    # the deployed commitment filter is a learned execution constraint and is
    # deliberately disabled only in this offline teacher simulator.
    cfg.marl.distributed_target_commitment_enabled = False
    env = UAVISACEnv(cfg, seed=0)
    observations = []
    labels = []
    seed_ids = []
    k_count, q_count = cfg.scenario.K, cfg.scenario.Q
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    zero_messages = {
        k: np.zeros(q_count * cfg.marl.comm_target_token_dim,
                    dtype=np.float64)
        for k in range(k_count)}
    silent_rates = {k: 0 for k in range(k_count)}
    zero_fractions = {k: 0.0 for k in range(k_count)}
    uniform_sensing = {
        k: np.full(q_count, 1.0 / q_count, dtype=np.float64)
        for k in range(k_count)}
    try:
        for seed_index, seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(seed))
            state_history = []
            done = False
            for commitment_index in range(int(commitments)):
                state_history.append(env.core.get_global_state().copy())
                history = torch.as_tensor(
                    np.stack(state_history), dtype=torch.float32)
                teacher_labels, _ = compute_qos_bistatic_assignment_teacher(
                    history,
                    k_count,
                    q_count,
                    cfg.scenario.region_size,
                    max_dp,
                    commitment_frames=hold_frames,
                    qos_floor=cfg.marl.target_allocation_teacher_qos_floor,
                    qos_weight=cfg.marl.target_allocation_teacher_qos_weight,
                    height_m=cfg.marl.target_allocation_teacher_height_m,
                    switching_penalty_m=(
                        cfg.marl.target_allocation_teacher_switching_penalty_m),
                )
                current_labels = teacher_labels.reshape(-1, k_count)[-1].numpy()
                observations.append(np.stack([
                    obs[str(k)] for k in range(k_count)]).astype(np.float32))
                labels.append(current_labels.astype(np.int64))
                seed_ids.append(np.full(k_count, int(seed), dtype=np.int64))

                for _ in range(int(hold_frames)):
                    target_xy = np.stack([
                        target.get_position_3d()[:2]
                        for target in env.core.targets])
                    actions = {}
                    for k in range(k_count):
                        delta = target_xy[current_labels[k]] - env.core.uavs[k].pos[:2]
                        norm = float(np.linalg.norm(delta))
                        move = (delta / norm * min(max_dp, norm)
                                if norm > 1e-9 else np.zeros(2))
                        actions[str(k)] = {'delta_p': move, 'role': 2}
                    env.core.submit_learned_communications(
                        zero_messages, silent_rates,
                        zero_fractions, uniform_sensing)
                    obs, _, terminated, truncated, _ = env.step(actions)
                    if (terminated.get('__all__', False)
                            or truncated.get('__all__', False)):
                        done = True
                        break
                if done:
                    break
            print(
                f'[{seed_index + 1}/{len(seeds)}] seed={seed} '
                f'snapshots={len(state_history)}', flush=True)
    finally:
        env.close()
    return (
        np.concatenate(observations, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(seed_ids, axis=0),
    )


@torch.no_grad()
def evaluate_head(actor, obs, labels, device, batch_size=512):
    actor.eval()
    total_correct = 0
    total_nll = 0.0
    total = 0
    k_count = actor.K
    for start in range(0, len(obs), batch_size):
        batch = torch.as_tensor(
            obs[start:start + batch_size], dtype=torch.float32, device=device)
        target = torch.as_tensor(
            labels[start:start + batch_size], dtype=torch.long, device=device)
        identity = torch.arange(start, start + len(batch), device=device) % k_count
        actor(batch, agent_identity=identity)
        probs = actor.last_movement_assignment
        total_correct += int((probs.argmax(-1) == target).sum().item())
        total_nll += float(torch.nn.functional.nll_loss(
            torch.log(probs.clamp_min(1e-8)), target,
            reduction='sum').item())
        total += len(batch)
    return {'accuracy': total_correct / max(total, 1),
            'nll': total_nll / max(total, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='config/exp_800_q4_u2u_hierarchical_multistatic_stable_eval.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--seed-bank', default='config/stratified_seeds_800_q4.json')
    parser.add_argument('--max-seeds', type=int, default=30)
    parser.add_argument('--commitments', type=int, default=15)
    parser.add_argument('--hold-frames', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=20260722)
    parser.add_argument('--runtime-frames', type=int, default=3072)
    parser.add_argument(
        '--output-dir', default='results/qos_commitment_pretrain')
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    seeds, difficulty = load_training_seed_pool(
        args.seed_bank,
        max_nearest_m=cfg.marl.training_seed_max_nearest_m)
    selected_seeds = choose_geometry_spread(
        np.asarray(seeds), np.asarray(difficulty), args.max_seeds)
    obs, labels, seed_ids = generate_dataset(
        cfg, selected_seeds, args.commitments, args.hold_frames)

    # Split by seed so validation snapshots never share a trajectory with train.
    validation_seeds = set(int(x) for x in selected_seeds[::5])
    validation_mask = np.asarray([
        int(seed) in validation_seeds for seed in seed_ids], dtype=bool)
    train_mask = ~validation_mask

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_env = UAVISACEnv(cfg, seed=0)
    agent = build_agent(cfg, eval_env, device)
    eval_env.close()
    checkpoint = safe_torch_load(
        args.checkpoint,
        map_location=device,
        description="QoS commitment pretraining source checkpoint",
        optional_state_dict_keys=("actor",),
    )
    actor_state = checkpoint.get('actor', checkpoint)
    validate_state_dict(
        actor_state, description="QoS commitment pretraining actor state_dict")
    agent.load_actor_state_dict_compatible(actor_state)
    actor = agent.actor
    if not hasattr(actor, 'movement_commitment_head'):
        raise RuntimeError('config must enable hierarchical dual assignment')
    for parameter in actor.parameters():
        parameter.requires_grad = False
    for parameter in actor.movement_commitment_head.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(
        actor.movement_commitment_head.parameters(), lr=args.lr)

    initial_train = evaluate_head(
        actor, obs[train_mask], labels[train_mask], device)
    initial_validation = evaluate_head(
        actor, obs[validation_mask], labels[validation_mask], device)
    rng = np.random.default_rng(args.seed)
    train_indices = np.flatnonzero(train_mask)
    history = []
    for epoch in range(args.epochs):
        actor.train()
        rng.shuffle(train_indices)
        for start in range(0, len(train_indices), args.batch_size):
            index = train_indices[start:start + args.batch_size]
            batch = torch.as_tensor(
                obs[index], dtype=torch.float32, device=device)
            target = torch.as_tensor(
                labels[index], dtype=torch.long, device=device)
            identity = torch.as_tensor(
                index % cfg.scenario.K, dtype=torch.long, device=device)
            actor(batch, agent_identity=identity)
            probs = actor.last_movement_assignment
            loss = torch.nn.functional.nll_loss(
                torch.log(probs.clamp_min(1e-8)), target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.movement_commitment_head.parameters(), 1.0)
            optimizer.step()
        train_metrics = evaluate_head(
            actor, obs[train_mask], labels[train_mask], device)
        validation_metrics = evaluate_head(
            actor, obs[validation_mask], labels[validation_mask], device)
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'validation': validation_metrics,
        })
        print(
            f'epoch={epoch + 1} train_acc={train_metrics["accuracy"]:.3f} '
            f'val_acc={validation_metrics["accuracy"]:.3f} '
            f'val_nll={validation_metrics["nll"]:.3f}', flush=True)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = checkpoint.get('runtime', {}) if isinstance(checkpoint, dict) else {}
    runtime = dict(runtime)
    runtime.setdefault('total_frames', int(args.runtime_frames))
    runtime.setdefault(
        'capacity_matching_blend',
        float(cfg.marl.capacity_matching_blend_start))
    runtime.setdefault(
        'target_allocation_movement_blend',
        float(cfg.marl.target_allocation_movement_blend))
    torch.save({
        'actor': actor.state_dict(),
        'runtime': runtime,
        'pretraining': {
            'teacher': 'qos_bistatic',
            'trainable_module': 'movement_commitment_head',
            'seed': int(args.seed),
            'max_seeds': int(args.max_seeds),
            'commitments': int(args.commitments),
            'hold_frames': int(args.hold_frames),
            'epochs': int(args.epochs),
            'learning_rate': float(args.lr),
        },
    }, output / 'qos_commitment_pretrained.pt')
    summary = {
        'config': args.config,
        'source_checkpoint': args.checkpoint,
        'seed_bank': args.seed_bank,
        'seed': int(args.seed),
        'max_seeds': int(args.max_seeds),
        'commitments': int(args.commitments),
        'hold_frames': int(args.hold_frames),
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'learning_rate': float(args.lr),
        'selected_seeds': selected_seeds.tolist(),
        'validation_seeds': sorted(validation_seeds),
        'num_rows': int(len(obs)),
        'num_train_rows': int(train_mask.sum()),
        'num_validation_rows': int(validation_mask.sum()),
        'initial_train': initial_train,
        'initial_validation': initial_validation,
        'history': history,
        'runtime': runtime,
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
