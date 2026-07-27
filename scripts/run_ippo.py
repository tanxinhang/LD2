#!/usr/bin/env python
"""Train IPPO baseline — same protocol as MAPPO, only critic differs.

IPPO uses the actor, communication mode, warm start, and optimizer settings
from the supplied config, but a DECENTRALIZED critic (local observation rather
than the global state).  This makes critic centralization the controlled
algorithmic difference from run_mappo.py.

Protocol matches run_mappo.py exactly:
  - D1 direct warm-start (no ResidualActor wrap)
  - selection/confirmation seeds from the configured stratified bank
  - complete fixed-bank final evaluation
  - restored-best actor/critic checkpoint saving

Only difference: centralized_critic=False.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config.params import load_config, get_default_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer, load_stratified_seed_split


def main():
    ap = argparse.ArgumentParser(description="Train IPPO (decentralized critic)")
    ap.add_argument("--config", default="config/exp_800_q4_full.yaml")
    ap.add_argument("--warm-start", default="results/dagger_variants/dagger_D1.pt")
    ap.add_argument("--warm-start-mode", default="direct", choices=["direct", "residual"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--ignore-warm-runtime", action="store_true")
    ap.add_argument("--final-eval-split", default=None)
    ap.add_argument("--max-final-eval-seeds", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else get_default_config()
    cfg.marl.centralized_critic = False  # IPPO: local-obs critic

    seed = args.seed
    set_seed(seed)

    env = UAVISACEnv(config=cfg, seed=seed)
    K, Q = cfg.scenario.K, cfg.scenario.Q
    action_space = ActionSpace(v_max=cfg.uav.v_max, dt=cfg.scenario.dt,
                               learn_roles=cfg.marl.learn_roles)
    action_space.num_targets = Q
    action_space.structured_actor = True
    action_space.structured_entity_dim = 64

    obs_dim = env.core.obs_builder.get_obs_dim()
    global_dim = env.core.obs_builder.get_global_state_dim()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[IPPO] obs_dim={obs_dim} global_dim={global_dim} K={K} Q={Q} device={device}")
    print(f"[IPPO] centralized_critic=False (critic input = LOCAL obs, dim={obs_dim})")

    train_lr = cfg.marl.lr
    payload_mode = str(getattr(
        cfg.marl, 'comm_payload_mode', 'aggregate')).lower()
    target_token_dim = int(getattr(
        cfg.marl, 'comm_target_token_dim', 16))
    target_tokens = payload_mode == 'target_tokens'
    comm_payload_dim = (Q * target_token_dim if target_tokens else 16)
    receiver_token_dim = target_token_dim + 6 if target_tokens else 21
    tokens_per_sender = Q if target_tokens else 1
    agents = [
        MAPPOAgent(agent_id=k, obs_dim=obs_dim, global_state_dim=global_dim,
                   action_space=action_space, num_agents=K, num_targets=Q,
                   hidden_layers=cfg.marl.hidden_layers, lr=train_lr,
                   critic_lr_mult=cfg.marl.critic_lr_mult,
                   max_grad_norm=cfg.marl.max_grad_norm, device=device,
                   centralized_critic=False,
                   comm_num_rate_levels=len(getattr(
                       cfg.marl, 'comm_rate_bits_per_dim', [0, 4, 8, 16])),
                   comm_log_std_init=float(getattr(
                       cfg.marl, 'comm_message_log_std_init', -1.0)),
                   comm_entropy_scale=float(getattr(
                       cfg.marl, 'comm_entropy_scale', 1.0)),
                   isac_power_log_std_init=float(getattr(
                       cfg.marl, 'isac_power_log_std_init', -1.0)),
                   sensing_allocation_log_std_init=float(getattr(
                       cfg.marl, 'sensing_allocation_log_std_init', -1.0)),
                   use_comm_cross_attention=bool(getattr(
                       cfg.marl, 'comm_cross_attention_enabled', False)),
                   comm_token_dim=receiver_token_dim,
                   comm_tokens_per_sender=tokens_per_sender,
                   comm_payload_dim=comm_payload_dim,
                   comm_target_token_enabled=target_tokens,
                   comm_target_token_dim=target_token_dim,
                   use_target_allocation=bool(
                       getattr(cfg.marl, 'target_allocation_enabled', False)
                       or getattr(cfg.marl,
                                  'target_allocation_teacher_enabled', False)),
                    use_team_sinkhorn=bool(getattr(
                        cfg.marl, 'target_allocation_sinkhorn_enabled', False)),
                    capacity_matching_enabled=bool(getattr(
                        cfg.marl, 'capacity_matching_enabled', False)),
                    capacity_matching_row_capacity=int(getattr(
                        cfg.marl, 'capacity_matching_row_capacity', 2)),
                    capacity_matching_column_capacity=int(getattr(
                        cfg.marl, 'capacity_matching_column_capacity', 2)),
                    capacity_matching_temperature=float(getattr(
                        cfg.marl, 'capacity_matching_temperature', 0.35)),
                    capacity_matching_iterations=int(getattr(
                        cfg.marl, 'capacity_matching_iterations', 32)),
                    capacity_matching_blend=float(getattr(
                        cfg.marl, 'capacity_matching_blend_start', 0.0)),
                    target_allocation_temperature=float(getattr(
                        cfg.marl, 'target_allocation_temperature', 1.0)),
                    target_allocation_straight_through=bool(getattr(
                        cfg.marl, 'target_allocation_straight_through', False)),
                    target_allocation_movement_blend=float(getattr(
                        cfg.marl, 'target_allocation_movement_blend_start', 0.0)
                        if int(getattr(
                            cfg.marl,
                            'target_allocation_movement_blend_anneal_frames',
                            0)) > 0
                        else getattr(
                            cfg.marl, 'target_allocation_movement_blend', 0.0)),
                    target_allocation_movement_confidence_gating_enabled=bool(
                        getattr(
                            cfg.marl,
                            'target_allocation_movement_confidence_gating_enabled',
                            False)),
                    target_allocation_movement_confidence_floor=float(getattr(
                        cfg.marl,
                        'target_allocation_movement_confidence_floor', 0.0)),
                    target_allocation_movement_confidence_power=float(getattr(
                        cfg.marl,
                        'target_allocation_movement_confidence_power', 2.0)),
                    hierarchical_dual_assignment_enabled=bool(getattr(
                        cfg.marl, 'hierarchical_dual_assignment_enabled',
                        False)),
                    movement_team_matching_enabled=bool(getattr(
                        cfg.marl, 'movement_team_matching_enabled', False)),
                    movement_team_matching_temperature=float(getattr(
                        cfg.marl, 'movement_team_matching_temperature', 0.35)),
                    movement_team_matching_iterations=int(getattr(
                        cfg.marl, 'movement_team_matching_iterations', 16)),
                    movement_team_matching_blend=float(getattr(
                        cfg.marl, 'movement_team_matching_blend', 0.0)),
                    movement_team_matching_intrinsic_bid_mix=float(getattr(
                        cfg.marl, 'movement_team_matching_intrinsic_bid_mix',
                        0.0)),
                    target_allocation_resource_blend=float(getattr(
                        cfg.marl, 'target_allocation_resource_blend', 0.0)),
                    round_negotiation_enabled=bool(getattr(
                        cfg.marl, 'round_negotiation_enabled', False)),
                    round_negotiation_strength=float(getattr(
                        cfg.marl, 'round_negotiation_strength', 0.5)),
                    round_negotiation_temperature=float(getattr(
                        cfg.marl, 'round_negotiation_temperature', 0.5)),
                    sparse_claim_enabled=bool(getattr(
                        cfg.marl, 'sparse_claim_enabled', False)),
                    sparse_claim_share_topk=int(getattr(
                        cfg.marl, 'sparse_claim_share_topk', 2)),
                    sparse_claim_desired_endpoints=int(getattr(
                        cfg.marl, 'sparse_claim_desired_endpoints', 2)),
                    sparse_claim_full_penalty=float(getattr(
                        cfg.marl, 'sparse_claim_full_penalty', 2.0)),
                    sparse_claim_vacant_bonus=float(getattr(
                        cfg.marl, 'sparse_claim_vacant_bonus', 0.5)),
                    sparse_claim_temperature=float(getattr(
                        cfg.marl, 'sparse_claim_temperature', 0.25)),
                    comm_aided_sensing_enabled=bool(getattr(
                        cfg.marl, 'comm_aided_sensing_enabled', False)),
                    comm_aided_sensing_blend=float(getattr(
                        cfg.marl, 'comm_aided_sensing_blend', 1.0)),
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
                    semantic_kinematic_field_enabled=bool(getattr(
                        cfg.marl, 'semantic_kinematic_field_enabled', False)),
                    semantic_kinematic_field_gain=float(getattr(
                        cfg.marl, 'semantic_kinematic_field_gain', 0.15)),
                    target_conditioned_movement_enabled=bool(getattr(
                        cfg.marl, 'target_conditioned_movement_enabled', False)),
                    target_conditioned_movement_gain=float(getattr(
                        cfg.marl, 'target_conditioned_movement_gain', 0.15)))
        for k in range(K)
    ]

    # D1 direct warm-start
    warm_runtime = None
    if args.warm_start and args.warm_start_mode == "direct":
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        actor_state = ckpt.get('actor', ckpt)
        if isinstance(ckpt, dict):
            warm_runtime = ckpt.get('runtime')
        agents[0].load_actor_state_dict_compatible(actor_state)
        agents[0].actor.zero_init_new_layers(set(actor_state.keys()))
        print(f"Warm-start mode=direct: actor loaded from {args.warm_start}")

    trainer = MAPPTrainer(env=env, agents=agents, config=cfg, device=device)
    trainer._bc_actor = None
    if warm_runtime and not args.ignore_warm_runtime:
        trainer.restore_policy_runtime_state(warm_runtime)
        print(
            "[IPPO] restored runtime: "
            f"frames={warm_runtime.get('total_frames', 0)} "
            f"capacity_blend={warm_runtime.get('capacity_matching_blend', 0.0):.5f} "
            f"movement_blend={warm_runtime.get('target_allocation_movement_blend', 0.0):.5f}")

    n_eps = args.episodes if args.episodes is not None else cfg.marl.num_episodes
    print(f"\n[IPPO] training {n_eps} episodes...")
    metrics_history = trainer.train(num_episodes=n_eps, log_interval=50)

    print("\n" + "=" * 60)
    print("[IPPO] training complete!")
    print(f"Total frames: {trainer.total_frames}")
    print(f"Best eval steady_P_D: {trainer.best_score:.4f}"
          + (f" (converged @ ep {trainer.converged_episode})" if trainer.converged_episode else ""))

    # Save results (same format as run_mappo.py)
    import csv, json as _json, subprocess as _sp
    config_stem = os.path.splitext(os.path.basename(args.config or "config/default.yaml"))[0]
    variant = "ippo_" + config_stem.replace("exp_800_q4_", "") if "exp_800_q4_" in config_stem else "ippo"
    out_dir = args.out_dir or os.path.join("results", "ippo", variant, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        commit = _sp.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    manifest = {
        "git_commit": commit, "config": args.config, "seed": seed,
        "K": K, "Q": Q, "warm_start": args.warm_start,
        "warm_start_mode": args.warm_start_mode,
        "warm_runtime_restored": bool(
            warm_runtime and not args.ignore_warm_runtime),
        "centralized_critic": False,
        "learned_comm_mode": cfg.marl.learned_comm_mode,
        "tracking_enabled": cfg.marl.tracking_enabled,
        "ground_communication_enabled": cfg.marl.ground_communication_enabled,
        "comm_only_neighbor_information": cfg.marl.comm_only_neighbor_information,
        "comm_rate_bits_per_dim": cfg.marl.comm_rate_bits_per_dim,
        "comm_tx_power_w": cfg.marl.comm_tx_power_w,
        "comm_cross_attention_enabled": cfg.marl.comm_cross_attention_enabled,
        "comm_entropy_scale": cfg.marl.comm_entropy_scale,
        "comm_qos_constrained": cfg.marl.comm_qos_constrained,
        "comm_encouragement_enabled": cfg.marl.comm_encouragement_enabled,
        "comm_encouragement_weight": cfg.marl.comm_encouragement_weight,
        "comm_encouragement_floor_ratio": (
            cfg.marl.comm_encouragement_floor_ratio),
        "comm_rate_bonus_enabled": cfg.marl.comm_rate_bonus_enabled,
        "comm_rate_bonus_weight": cfg.marl.comm_rate_bonus_weight,
        "comm_rate_bonus_target_bits": cfg.marl.comm_rate_bonus_target_bits,
        "comm_rate_bonus_aux_coef": cfg.marl.comm_rate_bonus_aux_coef,
        "comm_rate_bonus_aux_lr": cfg.marl.comm_rate_bonus_aux_lr,
        "comm_silence_penalty_enabled": (
            cfg.marl.comm_silence_penalty_enabled),
        "comm_silence_grace_decisions": (
            cfg.marl.comm_silence_grace_decisions),
        "comm_silence_penalty_per_decision": (
            cfg.marl.comm_silence_penalty_per_decision),
        "comm_silence_penalty_max": cfg.marl.comm_silence_penalty_max,
        "comm_eval_force_silence": cfg.marl.comm_eval_force_silence,
        "comm_eval_message_ablation": cfg.marl.comm_eval_message_ablation,
        "eval_centralized_assignment_movement": (
            cfg.marl.eval_centralized_assignment_movement),
        "comm_qos_min_rate_bits": cfg.marl.comm_qos_min_rate_bits,
        "target_allocation_enabled": cfg.marl.target_allocation_enabled,
        "target_allocation_teacher_enabled": (
            cfg.marl.target_allocation_teacher_enabled),
        "best_steady_P_D": float(trainer.best_score),
        "best_weak3_P_D": float(getattr(
            trainer, "_best_eval_metrics", {}).get("eval_weak3_P_D", 0.0)),
        "best_worst_P_D": float(getattr(
            trainer, "_best_eval_metrics", {}).get("eval_worst_P_D", 0.0)),
        "eval_seed_bank_path": cfg.marl.eval_seed_bank_path,
        "eval_seed_split": cfg.marl.eval_seed_split,
        "final_eval_seed_split": (
            args.final_eval_split or cfg.marl.final_eval_seed_split),
        "checkpoint_selection": (
            "wilson_feasible_lcb,bootstrap_worst_lcb,worst_cvar,"
            "strict_worst,trimmed_worst,weak3,steady,-bits"),
        "total_frames": trainer.total_frames,
        "total_episodes": len(metrics_history),
        "max_final_eval_seeds": max(0, int(args.max_final_eval_seeds)),
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        _json.dump(manifest, f, indent=2)

    if metrics_history:
        keys = sorted(metrics_history[0].keys())
        with open(os.path.join(out_dir, "train_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for m in metrics_history:
                w.writerow({k: m.get(k, "") for k in keys})

    final_split = args.final_eval_split or cfg.marl.final_eval_seed_split
    if cfg.marl.eval_seed_bank_path:
        paired_seeds = load_stratified_seed_split(
            cfg.marl.eval_seed_bank_path, final_split)
    else:
        paired_seeds = [30001 + i for i in range(100)]
    if args.max_final_eval_seeds > 0:
        paired_seeds = paired_seeds[:args.max_final_eval_seeds]
    try:
        trainer.agents[0].actor.eval()
        ev = trainer._evaluate(n_episodes=len(paired_seeds), eval_seeds=paired_seeds)
        with open(os.path.join(out_dir, "paired_eval.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(sorted(ev.keys()))
            w.writerow([ev.get(k, "") for k in sorted(ev.keys())])
        torch.save({
            'actor': trainer.agents[0].actor.state_dict(),
            'critic': trainer.agents[0].critic.state_dict(),
            'runtime': trainer.get_policy_runtime_state(),
            'centralized_critic': False,
        }, os.path.join(out_dir, "best_restored.pt"))
    except Exception as e:
        print(f"  [warn] eval save failed: {e}")

    print(f"Results saved → {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
