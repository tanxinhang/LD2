#!/usr/bin/env python
"""Closed-loop DAgger for decentralized QoS movement commitments.

The deployed actor, including its physical U2U channel and learned ISAC
resources, generates the visited states.  A centralized QoS-bistatic teacher
labels only those states.  Optimization is restricted to the movement-only
residual adapter and commitment head, so communication content, rate, power
and sensing-resource heads remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.params import load_config
from tools.pretrain_qos_commitment import (
    build_agent,
    choose_geometry_spread,
    evaluate_head,
)
from uav_isac.agents.trainer import (
    build_differentiable_u2u_inbox,
    compute_qos_bistatic_assignment_teacher,
    load_training_seed_pool,
)
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.utils.checkpoint_loading import safe_torch_load, validate_state_dict
from uav_isac.utils.seeding import set_seed


@torch.no_grad()
def collect_actor_dataset(
    cfg,
    agent,
    seeds: np.ndarray,
    commitments: int,
    device: str,
    policy_alignment_m: float = 0.0,
):
    """Collect teacher labels on states visited by the deterministic actor."""
    env = UAVISACEnv(cfg, seed=0)
    actor = agent.actor
    actor.eval()
    k_count, q_count = cfg.scenario.K, cfg.scenario.Q
    interval = max(1, int(cfg.marl.movement_decision_interval))
    max_dp = cfg.uav.v_max * cfg.scenario.dt
    observations = []
    labels = []
    seed_ids = []
    episode_summaries = []
    total_match = 0
    total_labels = 0
    try:
        for seed_index, seed in enumerate(seeds):
            obs, _ = env.reset(seed=int(seed))
            state_history = []
            probability_history = []
            h_prev = None
            held_actions = None
            phase = 0
            decisions = 0
            pd_history = []
            done = False
            while not done and decisions < int(commitments):
                ob = np.stack([obs[str(k)] for k in range(k_count)])
                phase_value = float(phase) / max(interval - 1, 1)
                round_phase = torch.full(
                    (k_count,), phase_value, dtype=torch.float32,
                    device=device)
                identity = torch.arange(
                    k_count, dtype=torch.long, device=device)
                ob_t = torch.as_tensor(
                    ob, dtype=torch.float32, device=device)
                (dp_mean, dp_log_std, role_logits, comm_mean,
                 _, h_new) = actor(
                    ob_t,
                    h_prev,
                    comm_round_phase=round_phase,
                    agent_identity=identity,
                )
                h_prev = h_new
                movement_decision = bool(phase == 0 or held_actions is None)

                if movement_decision:
                    state_history.append(env.core.get_global_state().copy())
                    probability_history.append(
                        actor.last_movement_assignment.detach().cpu().numpy())
                    history = torch.as_tensor(
                        np.stack(state_history), dtype=torch.float32)
                    teacher_labels, _ = compute_qos_bistatic_assignment_teacher(
                        history,
                        k_count,
                        q_count,
                        cfg.scenario.region_size,
                        max_dp,
                        commitment_frames=interval,
                        qos_floor=(
                            cfg.marl.target_allocation_teacher_qos_floor),
                        qos_weight=(
                            cfg.marl.target_allocation_teacher_qos_weight),
                        height_m=(
                            cfg.marl.target_allocation_teacher_height_m),
                        switching_penalty_m=(
                            cfg.marl
                            .target_allocation_teacher_switching_penalty_m),
                        policy_assignment_probs=torch.as_tensor(
                            np.stack(probability_history),
                            dtype=torch.float32),
                        policy_alignment_m=policy_alignment_m,
                    )
                    current_labels = teacher_labels.reshape(
                        -1, k_count)[-1].cpu().numpy()
                    observations.append(ob.astype(np.float32))
                    labels.append(current_labels.astype(np.int64))
                    seed_ids.append(np.full(
                        k_count, int(seed), dtype=np.int64))
                    actor_choice = actor.last_movement_assignment.argmax(
                        dim=-1).detach().cpu().numpy()
                    total_match += int(np.sum(actor_choice == current_labels))
                    total_labels += k_count
                    decisions += 1

                dpm = dp_mean.detach().cpu().numpy()
                dps = dp_log_std.detach().cpu().numpy()
                roles = role_logits.detach().cpu().numpy()
                actions = {}
                for k in range(k_count):
                    action, _ = agent.action_space.decode(
                        dpm[k], dps, roles[k],
                        dp_deterministic=True,
                        role_deterministic=True,
                    )
                    actions[str(k)] = {
                        'delta_p': action.delta_p,
                        'role': action.role,
                    }
                if movement_decision:
                    held_actions = {
                        key: {
                            'delta_p': np.asarray(value['delta_p']).copy(),
                            'role': int(value['role']),
                        }
                        for key, value in actions.items()
                    }
                else:
                    actions = {
                        key: {
                            'delta_p': np.asarray(value['delta_p']).copy(),
                            'role': int(value['role']),
                        }
                        for key, value in held_actions.items()
                    }

                comm_action, rate_action, _, _ = agent.sample_communication(
                    comm_mean, deterministic=True)
                (_, _, comm_fraction, sensing_weights, _, _) = (
                    agent.sample_isac_resources(
                        comm_mean,
                        rate_action,
                        deterministic=True,
                        comm_fraction_min=(
                            cfg.marl.comm_power_fraction_min),
                        comm_fraction_max=(
                            cfg.marl.comm_power_fraction_max),
                    )
                )
                token_mask = getattr(actor, 'last_outgoing_token_mask', None)
                messages = {
                    k: comm_action[k].detach().cpu().numpy()
                    for k in range(k_count)
                }
                rates = {
                    k: int(rate_action[k].item()) for k in range(k_count)
                }
                fractions = {
                    k: float(comm_fraction[k].item())
                    for k in range(k_count)
                }
                weights = {
                    k: sensing_weights[k].detach().cpu().numpy()
                    for k in range(k_count)
                }
                if bool(cfg.marl.sparse_claim_enabled):
                    masks = {
                        k: token_mask[k].detach().cpu().numpy()
                        for k in range(k_count)
                    }
                    env.core.submit_learned_communications(
                        messages, rates, fractions, weights,
                        token_masks=masks)
                else:
                    env.core.submit_learned_communications(
                        messages, rates, fractions, weights)

                obs, _, terminated, truncated, info = env.step(actions)
                pd_history.append(np.asarray(info['P_D_q'], dtype=np.float64))
                done = bool(
                    terminated.get('__all__', False)
                    or truncated.get('__all__', False))
                phase = (phase + 1) % interval

            if pd_history:
                steady = np.mean(np.stack(pd_history)[-20:], axis=0)
                episode_summaries.append({
                    'seed': int(seed),
                    'steady': float(np.mean(steady)),
                    'weak3': float(np.mean(np.sort(steady)[:min(3, q_count)])),
                    'worst': float(np.min(steady)),
                    'decisions': int(decisions),
                })
            print(
                f'collect [{seed_index + 1}/{len(seeds)}] seed={int(seed)} '
                f'decisions={decisions}', flush=True)
    finally:
        env.close()

    if not observations:
        raise RuntimeError('actor DAgger collector produced no observations')
    return {
        'obs': np.concatenate(observations, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'seed_ids': np.concatenate(seed_ids, axis=0),
        'teacher_match': total_match / max(total_labels, 1),
        'episodes': episode_summaries,
    }


def movement_parameters(actor, train_communication: bool = False):
    modules = [actor.movement_commitment_head]
    if hasattr(actor, 'movement_feature_adapter'):
        modules.insert(0, actor.movement_feature_adapter)
    if train_communication:
        modules.append(actor.comm_target_token_head)
    for parameter in actor.parameters():
        parameter.requires_grad = False
    params = []
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
            params.append(parameter)
    return params


@torch.no_grad()
def evaluate_comm_exchange(actor, obs, labels, device, rate_levels):
    """Evaluate one differentiable token exchange on complete UAV teams."""
    actor.eval()
    k_count = actor.K
    total_correct = 0
    total_nll = 0.0
    total = 0
    teams = len(obs) // k_count
    for team_start in range(0, teams, 64):
        team_stop = min(team_start + 64, teams)
        index = np.arange(
            team_start * k_count, team_stop * k_count, dtype=np.int64)
        batch = torch.as_tensor(
            obs[index], dtype=torch.float32, device=device)
        target = torch.as_tensor(
            labels[index], dtype=torch.long, device=device)
        identity = torch.arange(
            len(batch), device=device, dtype=torch.long) % k_count
        _, _, _, sender_comm, _, _ = actor(
            batch, agent_identity=identity)
        token_masks = getattr(actor, 'last_outgoing_token_mask', None)
        exchange = build_differentiable_u2u_inbox(
            batch,
            sender_comm,
            actor,
            num_agents=k_count,
            rate_index=2,
            rate_bits=rate_levels[2],
            num_rate_levels=len(rate_levels),
            sender_token_masks=token_masks,
        )
        actor(exchange, agent_identity=identity)
        probs = actor.last_movement_assignment
        total_correct += int((probs.argmax(-1) == target).sum().item())
        total_nll += float(torch.nn.functional.nll_loss(
            torch.log(probs.clamp_min(1e-8)), target,
            reduction='sum').item())
        total += len(batch)
    return {
        'accuracy': total_correct / max(total, 1),
        'nll': total_nll / max(total, 1),
    }


def train_aggregate(
    actor,
    obs,
    labels,
    validation,
    device,
    *,
    epochs: int,
    batch_size: int,
    adapter_lr: float,
    head_lr: float,
    message_lr: float,
    train_communication: bool,
    rate_levels,
    seed: int,
):
    params = movement_parameters(actor, train_communication)
    groups = []
    if hasattr(actor, 'movement_feature_adapter'):
        groups.append({
            'params': list(actor.movement_feature_adapter.parameters()),
            'lr': float(adapter_lr),
        })
    groups.append({
        'params': list(actor.movement_commitment_head.parameters()),
        'lr': float(head_lr),
    })
    message_anchor = None
    if train_communication:
        groups.append({
            'params': list(actor.comm_target_token_head.parameters()),
            'lr': float(message_lr),
        })
        message_anchor = [
            parameter.detach().clone()
            for parameter in actor.comm_target_token_head.parameters()
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    rng = np.random.default_rng(int(seed))
    if len(obs) % actor.K != 0:
        raise ValueError('DAgger rows must contain complete UAV teams')
    team_indices = np.arange(len(obs) // actor.K, dtype=np.int64)
    best_state = deepcopy(actor.state_dict())
    if train_communication:
        evaluate_fn = lambda values, targets: evaluate_comm_exchange(
            actor, values, targets, device, rate_levels)
    else:
        evaluate_fn = lambda values, targets: evaluate_head(
            actor, values, targets, device)
    initial_validation = evaluate_fn(
        validation['obs'], validation['labels'])
    best_key = (
        float(initial_validation['accuracy']),
        -float(initial_validation['nll']),
    )
    history = []
    k_count = actor.K
    for epoch in range(int(epochs)):
        actor.train()
        rng.shuffle(team_indices)
        batch_teams = max(1, int(batch_size) // actor.K)
        for start in range(0, len(team_indices), batch_teams):
            selected_teams = team_indices[start:start + batch_teams]
            index = (
                selected_teams[:, None] * actor.K
                + np.arange(actor.K, dtype=np.int64)[None, :]
            ).reshape(-1)
            batch = torch.as_tensor(
                obs[index], dtype=torch.float32, device=device)
            target = torch.as_tensor(
                labels[index], dtype=torch.long, device=device)
            identity = torch.as_tensor(
                index % k_count, dtype=torch.long, device=device)
            _, _, _, sender_comm, _, _ = actor(
                batch, agent_identity=identity)
            if train_communication:
                token_masks = getattr(
                    actor, 'last_outgoing_token_mask', None)
                batch = build_differentiable_u2u_inbox(
                    batch,
                    sender_comm,
                    actor,
                    num_agents=actor.K,
                    rate_index=2,
                    rate_bits=rate_levels[2],
                    num_rate_levels=len(rate_levels),
                    sender_token_masks=token_masks,
                )
                actor(batch, agent_identity=identity)
            logits = torch.log(
                actor.last_movement_assignment.clamp_min(1e-8))
            loss = torch.nn.functional.cross_entropy(
                logits, target, label_smoothing=0.02)
            if message_anchor is not None:
                anchor_loss = sum(
                    (parameter - anchor).pow(2).mean()
                    for parameter, anchor in zip(
                        actor.comm_target_token_head.parameters(),
                        message_anchor)
                )
                loss = loss + 1e-3 * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        train_metrics = evaluate_fn(obs, labels)
        validation_metrics = evaluate_fn(
            validation['obs'], validation['labels'])
        key = (
            float(validation_metrics['accuracy']),
            -float(validation_metrics['nll']),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = deepcopy(actor.state_dict())
        item = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'validation': validation_metrics,
        }
        history.append(item)
        print(
            f'epoch={epoch + 1} train_acc={train_metrics["accuracy"]:.3f} '
            f'val_acc={validation_metrics["accuracy"]:.3f} '
            f'val_nll={validation_metrics["nll"]:.3f}', flush=True)
    actor.load_state_dict(best_state)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=(
            'config/exp_800_q4_u2u_hierarchical_multistatic_'
            'soft_commitment_gated_move50_eval.yaml'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--seed-bank', default='config/stratified_seeds_800_q4.json')
    parser.add_argument('--max-seeds', type=int, default=40)
    parser.add_argument('--validation-stride', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--commitments', type=int, default=20)
    parser.add_argument('--epochs-per-round', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--adapter-lr', type=float, default=1e-4)
    parser.add_argument('--head-lr', type=float, default=3e-5)
    parser.add_argument('--message-lr', type=float, default=1e-5)
    parser.add_argument('--train-communication', action='store_true')
    parser.add_argument('--policy-alignment-m', type=float, default=50.0)
    parser.add_argument('--seed', type=int, default=20260722)
    parser.add_argument(
        '--output-dir', default='results/qos_commitment_dagger')
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    seeds, difficulty = load_training_seed_pool(
        args.seed_bank,
        max_nearest_m=cfg.marl.training_seed_max_nearest_m)
    selected = choose_geometry_spread(
        np.asarray(seeds), np.asarray(difficulty), args.max_seeds)
    stride = max(2, int(args.validation_stride))
    validation_seeds = selected[::stride]
    validation_set = set(int(seed) for seed in validation_seeds)
    training_seeds = np.asarray([
        seed for seed in selected if int(seed) not in validation_set
    ], dtype=np.int64)
    round_seeds = np.array_split(training_seeds, max(1, int(args.rounds)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = UAVISACEnv(cfg, seed=0)
    agent = build_agent(cfg, env, device)
    env.close()
    checkpoint = safe_torch_load(
        args.checkpoint,
        map_location=device,
        description="QoS commitment DAgger source checkpoint",
        optional_state_dict_keys=("actor",),
    )
    actor_state = checkpoint.get('actor', checkpoint)
    validate_state_dict(
        actor_state, description="QoS commitment DAgger actor state_dict")
    missing, unexpected = agent.load_actor_state_dict_compatible(actor_state)
    print(
        f'checkpoint loaded missing={len(missing)} '
        f'unexpected={len(unexpected)}', flush=True)
    actor = agent.actor
    if not hasattr(actor, 'movement_feature_adapter'):
        raise RuntimeError('config must enable the movement residual adapter')

    validation = collect_actor_dataset(
        cfg, agent, validation_seeds, args.commitments, device,
        policy_alignment_m=args.policy_alignment_m)
    initial_validation = deepcopy(validation)
    aggregate_obs = []
    aggregate_labels = []
    round_history = []
    for round_index, seeds_this_round in enumerate(round_seeds):
        collected = collect_actor_dataset(
            cfg, agent, seeds_this_round, args.commitments, device,
            policy_alignment_m=args.policy_alignment_m)
        aggregate_obs.append(collected['obs'])
        aggregate_labels.append(collected['labels'])
        obs = np.concatenate(aggregate_obs, axis=0)
        labels = np.concatenate(aggregate_labels, axis=0)
        before = evaluate_head(actor, obs, labels, device)
        epoch_history = train_aggregate(
            actor,
            obs,
            labels,
            validation,
            device,
            epochs=args.epochs_per_round,
            batch_size=args.batch_size,
            adapter_lr=args.adapter_lr,
            head_lr=args.head_lr,
            message_lr=args.message_lr,
            train_communication=args.train_communication,
            rate_levels=list(cfg.marl.comm_rate_bits_per_dim),
            seed=args.seed + round_index,
        )
        live_validation = collect_actor_dataset(
            cfg, agent, validation_seeds, args.commitments, device,
            policy_alignment_m=args.policy_alignment_m)
        after = evaluate_head(actor, obs, labels, device)
        round_history.append({
            'round': round_index + 1,
            'seeds': seeds_this_round.tolist(),
            'rows_added': int(len(collected['obs'])),
            'aggregate_rows': int(len(obs)),
            'collector_teacher_match': float(collected['teacher_match']),
            'before': before,
            'after': after,
            'fixed_validation': evaluate_head(
                actor, validation['obs'], validation['labels'], device),
            'live_validation_teacher_match': float(
                live_validation['teacher_match']),
            'live_validation_episodes': live_validation['episodes'],
            'epochs': epoch_history,
        })
        validation = live_validation

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = checkpoint.get('runtime', {}) if isinstance(checkpoint, dict) else {}
    runtime = dict(runtime)
    runtime['capacity_matching_blend'] = float(
        cfg.marl.capacity_matching_blend_start)
    runtime['target_allocation_movement_blend'] = float(
        cfg.marl.target_allocation_movement_blend)
    torch.save({
        'actor': actor.state_dict(),
        'runtime': runtime,
        'pretraining': {
            'teacher': 'closed_loop_qos_bistatic_dagger',
            'trainable_modules': [
                'movement_feature_adapter',
                'movement_commitment_head',
            ] + (['comm_target_token_head']
                 if args.train_communication else []),
        },
    }, output / 'qos_commitment_dagger.pt')
    summary = {
        'config': args.config,
        'source_checkpoint': args.checkpoint,
        'seed_bank': args.seed_bank,
        'selected_seeds': selected.tolist(),
        'training_seeds': training_seeds.tolist(),
        'validation_seeds': validation_seeds.tolist(),
        'initial_validation_teacher_match': float(
            initial_validation['teacher_match']),
        'initial_validation_episodes': initial_validation['episodes'],
        'rounds': round_history,
        'policy_alignment_m': float(args.policy_alignment_m),
        'train_communication': bool(args.train_communication),
        'runtime': runtime,
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
