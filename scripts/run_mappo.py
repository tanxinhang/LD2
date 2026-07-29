#!/usr/bin/env python
"""Train Hierarchical MAPPO for Phase 1 UAV-ISAC system."""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config.params import get_default_config, load_config
from uav_isac.utils.seeding import set_seed
from uav_isac.environment.env_wrapper import UAVISACEnv
from uav_isac.environment.action import ActionSpace
from uav_isac.agents.mappo_agent import MAPPOAgent
from uav_isac.agents.trainer import MAPPTrainer, load_stratified_seed_split


def main():
    ap = argparse.ArgumentParser(description="Train MAPPO (optionally warm-started).")
    ap.add_argument("--config", default=None, help="config YAML (default: config/default.yaml)")
    ap.add_argument("--warm-start", default=None,
                    help="path to an actor state_dict (e.g. results/warmstart_actor.pt) to "
                         "initialize the shared actor before PPO (see scripts/dagger_warmstart.py)")
    ap.add_argument("--warm-start-mode", default="direct",
                    choices=["direct", "residual", "risk_residual"],
                    help="direct: load into StructuredActorNetwork (Full/EH default). "
                         "residual: legacy ResidualActor. risk_residual: freeze "
                         "the foundation and train a local gated movement adapter.")
    ap.add_argument("--warmstart-lr", type=float, default=None,
                    help="override LR when warm-starting (default: config.marl.lr). "
                         "3e-5 recommended for BC warmstart to keep KL within trust region.")
    ap.add_argument("--ignore-warm-runtime", action="store_true",
                    help="load actor tensors but retain runtime controller values "
                         "from the evaluation config (diagnostic only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=None, help="override marl.num_episodes")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for results (default: results/full_eh/seed_{seed})")
    ap.add_argument("--final-eval-split", default=None,
                    help="override the stratified seed-bank split used for final evaluation")
    ap.add_argument("--max-final-eval-seeds", type=int, default=0,
                    help="limit final fixed-bank episodes for diagnostic screening; "
                         "0 evaluates the complete split")
    ap.add_argument("--target-choice-audit-stride", type=int, default=0,
                    help="at movement boundaries, run a common-random-number "
                         "single-UAV target-direction intervention every N "
                         "frames; 0 disables the expensive diagnostic")
    ap.add_argument("--sensing-choice-audit-stride", type=int, default=0,
                    help="run a common-random-number bounded single-UAV "
                         "sensing-allocation intervention every N frames; "
                         "0 disables the expensive diagnostic")
    ap.add_argument("--sensing-residual-blend", type=float, default=0.25,
                    help="fraction of one UAV's sensing mass redirected to "
                         "each candidate target during the causal audit")
    ap.add_argument("--sensing-audit-horizon", type=int, default=0,
                    help="candidate rollout horizon in frames; 0 uses the "
                         "movement decision interval")
    ap.add_argument("--sensing-oracle-control", action="store_true",
                    help="execute the best future-horizon sensing candidate "
                         "between audits; noncausal diagnostic upper bound")
    ap.add_argument("--joint-sensing-pair-audit-stride", type=int, default=0,
                    help="enumerate no-op-controlled two-UAV sensing residuals "
                         "toward the weakest target every N frames")
    ap.add_argument("--physical-oracle-stride", type=int, default=0,
                    help="solve same-geometry single-role and full-duplex "
                         "pair/power upper bounds every N evaluation frames")
    ap.add_argument(
        "--evidence-trace-output",
        default=None,
        help="optional .npz path for receiver-target deflection traces from "
             "the final paired evaluation; used by offline Gate 1b only",
    )
    args = ap.parse_args()

    # Config: single source of truth (default.yaml) or an explicit --config.
    config = load_config(args.config) if args.config else get_default_config()

    seed = args.seed
    set_seed(seed)

    # Create environment
    env = UAVISACEnv(config=config, seed=seed)
    K = config.scenario.K

    # Create action space
    action_space = ActionSpace(
        v_max=config.uav.v_max,
        dt=config.scenario.dt,
        learn_roles=config.marl.learn_roles,
    )
    action_space.num_targets = config.scenario.Q
    action_space.structured_actor = True   # relational with 2-frame parsing
    action_space.structured_entity_dim = 64

    # Create agents — use actual obs dim (includes history stacking)
    obs_test, _ = env.reset(seed=seed)
    obs_dim = obs_test['0'].shape[0]
    global_dim = env.core.obs_builder.get_global_state_dim()

    print(f"Observation dim: {obs_dim}")
    print(f"Global state dim: {global_dim}")
    print(f"K={K}, Q={config.scenario.Q}, T={config.scenario.T}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_lr = args.warmstart_lr if (args.warm_start and args.warmstart_lr) else config.marl.lr
    payload_mode = str(getattr(
        config.marl, 'comm_payload_mode', 'aggregate')).lower()
    target_token_dim = int(getattr(
        config.marl, 'comm_target_token_dim', 16))
    target_tokens = payload_mode == 'target_tokens'
    comm_payload_dim = (config.scenario.Q * target_token_dim
                        if target_tokens else 16)
    receiver_token_dim = target_token_dim + 6 if target_tokens else 21
    tokens_per_sender = config.scenario.Q if target_tokens else 1
    agents = []
    for k in range(K):
        agent = MAPPOAgent(
            agent_id=k,
            obs_dim=obs_dim,
            global_state_dim=global_dim,
            action_space=action_space,
            num_agents=K,
            num_targets=config.scenario.Q,
            hidden_layers=config.marl.hidden_layers,
            lr=train_lr,
            critic_lr_mult=config.marl.critic_lr_mult,
            max_grad_norm=config.marl.max_grad_norm,
            device=device,
            centralized_critic=getattr(config.marl, 'centralized_critic', True),
            comm_num_rate_levels=len(getattr(
                config.marl, 'comm_rate_bits_per_dim', [0, 4, 8, 16])),
            comm_log_std_init=float(getattr(
                config.marl, 'comm_message_log_std_init', -1.0)),
            comm_entropy_scale=float(getattr(
                config.marl, 'comm_entropy_scale', 1.0)),
            isac_power_log_std_init=float(getattr(
                config.marl, 'isac_power_log_std_init', -1.0)),
            sensing_allocation_log_std_init=float(getattr(
                config.marl, 'sensing_allocation_log_std_init', -1.0)),
            use_comm_cross_attention=bool(getattr(
                config.marl, 'comm_cross_attention_enabled', False)),
            comm_token_dim=receiver_token_dim,
            comm_tokens_per_sender=tokens_per_sender,
            comm_payload_dim=comm_payload_dim,
            comm_target_token_enabled=target_tokens,
            comm_target_token_dim=target_token_dim,
            use_target_allocation=bool(
                getattr(config.marl, 'target_allocation_enabled', False)
                or getattr(config.marl,
                           'target_allocation_teacher_enabled', False)),
            use_team_sinkhorn=bool(getattr(
                config.marl, 'target_allocation_sinkhorn_enabled', False)),
            capacity_matching_enabled=bool(getattr(
                config.marl, 'capacity_matching_enabled', False)),
            capacity_matching_row_capacity=int(getattr(
                config.marl, 'capacity_matching_row_capacity', 2)),
            capacity_matching_column_capacity=int(getattr(
                config.marl, 'capacity_matching_column_capacity', 2)),
            capacity_matching_temperature=float(getattr(
                config.marl, 'capacity_matching_temperature', 0.35)),
            capacity_matching_iterations=int(getattr(
                config.marl, 'capacity_matching_iterations', 32)),
            capacity_matching_blend=float(getattr(
                config.marl, 'capacity_matching_blend_start', 0.0)),
            target_allocation_temperature=float(getattr(
                config.marl, 'target_allocation_temperature', 1.0)),
            target_allocation_straight_through=bool(getattr(
                config.marl, 'target_allocation_straight_through', False)),
            target_allocation_movement_blend=float(getattr(
                config.marl,
                'target_allocation_movement_blend_start', 0.0)
                if int(getattr(
                    config.marl,
                    'target_allocation_movement_blend_anneal_frames', 0)) > 0
                else getattr(
                    config.marl, 'target_allocation_movement_blend', 0.0)),
            target_allocation_movement_confidence_gating_enabled=bool(getattr(
                config.marl,
                'target_allocation_movement_confidence_gating_enabled',
                False)),
            target_allocation_movement_confidence_floor=float(getattr(
                config.marl,
                'target_allocation_movement_confidence_floor', 0.0)),
            target_allocation_movement_confidence_power=float(getattr(
                config.marl,
                'target_allocation_movement_confidence_power', 2.0)),
            hierarchical_dual_assignment_enabled=bool(getattr(
                config.marl, 'hierarchical_dual_assignment_enabled', False)),
            movement_team_matching_enabled=bool(getattr(
                config.marl, 'movement_team_matching_enabled', False)),
            movement_team_matching_temperature=float(getattr(
                config.marl, 'movement_team_matching_temperature', 0.35)),
            movement_team_matching_iterations=int(getattr(
                config.marl, 'movement_team_matching_iterations', 16)),
            movement_team_matching_blend=float(getattr(
                config.marl, 'movement_team_matching_blend', 0.0)),
            movement_team_matching_intrinsic_bid_mix=float(getattr(
                config.marl,
                'movement_team_matching_intrinsic_bid_mix', 0.0)),
            target_allocation_resource_blend=float(getattr(
                config.marl, 'target_allocation_resource_blend', 0.0)),
            round_negotiation_enabled=bool(getattr(
                config.marl, 'round_negotiation_enabled', False)),
            round_negotiation_strength=float(getattr(
                config.marl, 'round_negotiation_strength', 0.5)),
            round_negotiation_temperature=float(getattr(
                config.marl, 'round_negotiation_temperature', 0.5)),
            sparse_claim_enabled=bool(getattr(
                config.marl, 'sparse_claim_enabled', False)),
            sparse_claim_share_topk=int(getattr(
                config.marl, 'sparse_claim_share_topk', 2)),
            sparse_claim_desired_endpoints=int(getattr(
                config.marl, 'sparse_claim_desired_endpoints', 2)),
            sparse_claim_full_penalty=float(getattr(
                config.marl, 'sparse_claim_full_penalty', 2.0)),
            sparse_claim_vacant_bonus=float(getattr(
                config.marl, 'sparse_claim_vacant_bonus', 0.5)),
            sparse_claim_temperature=float(getattr(
                config.marl, 'sparse_claim_temperature', 0.25)),
            comm_aided_sensing_enabled=bool(getattr(
                config.marl, 'comm_aided_sensing_enabled', False)),
            comm_aided_sensing_blend=float(getattr(
                config.marl, 'comm_aided_sensing_blend', 1.0)),
            comm_semantic_decoder_enabled=bool(getattr(
                config.marl, 'comm_semantic_decoder_enabled', False)),
            comm_semantic_capacity_bid_enabled=bool(getattr(
                config.marl, 'comm_semantic_capacity_bid_enabled', False)),
            comm_semantic_capacity_bid_gain=float(getattr(
                config.marl, 'comm_semantic_capacity_bid_gain', 0.0)),
            comm_semantic_extra_token_enabled=bool(getattr(
                config.marl, 'comm_semantic_extra_token_enabled', False)),
            comm_semantic_extra_token_threshold=float(getattr(
                config.marl, 'comm_semantic_extra_token_threshold', 0.10)),
            semantic_kinematic_field_enabled=bool(getattr(
                config.marl, 'semantic_kinematic_field_enabled', False)),
            semantic_kinematic_field_gain=float(getattr(
                config.marl, 'semantic_kinematic_field_gain', 0.15)),
            target_conditioned_movement_enabled=bool(getattr(
                config.marl, 'target_conditioned_movement_enabled', False)),
            target_conditioned_movement_gain=float(getattr(
                config.marl, 'target_conditioned_movement_gain', 0.15)),
            architecture_v2_enabled=bool(getattr(
                config.marl, 'architecture_v2_enabled', False)),
            architecture_v2_prior_gain=float(getattr(
                config.marl, 'architecture_v2_prior_gain', 1.0)),
            architecture_v2_distance_weight=float(getattr(
                config.marl, 'architecture_v2_distance_weight', 0.25)),
            architecture_v2_qos_floor=float(getattr(
                config.marl, 'architecture_v2_qos_floor', 0.60)),
            architecture_v2_comm_prior_gain=float(getattr(
                config.marl, 'architecture_v2_comm_prior_gain', 2.0)),
            architecture_v2_comm_crisis_threshold=float(getattr(
                config.marl,
                'architecture_v2_comm_crisis_threshold', 0.25)),
            architecture_v2_consensus_enabled=bool(getattr(
                config.marl, 'architecture_v2_consensus_enabled', True)),
            architecture_v2_matching_temperature=float(getattr(
                config.marl,
                'architecture_v2_matching_temperature', 0.35)),
            architecture_v2_movement_consensus_blend=float(getattr(
                config.marl,
                'architecture_v2_movement_consensus_blend', 1.0)),
            architecture_v2_endpoint_consensus_gain=float(getattr(
                config.marl,
                'architecture_v2_endpoint_consensus_gain', 2.0)),
            architecture_v2_bid_residual_scale=float(getattr(
                config.marl,
                'architecture_v2_bid_residual_scale', 0.25)),
            architecture_v2_sensing_aligned_claims_enabled=bool(getattr(
                config.marl,
                'architecture_v2_sensing_aligned_claims_enabled', False)),
            architecture_v2_modular_coordination_enabled=bool(getattr(
                config.marl,
                'architecture_v2_modular_coordination_enabled', False)),
            architecture_v2_modular_num_experts=int(getattr(
                config.marl, 'architecture_v2_modular_num_experts', 3)),
            architecture_v2_modular_gain=float(getattr(
                config.marl, 'architecture_v2_modular_gain', 0.25)),
            architecture_v2_modular_temperature=float(getattr(
                config.marl,
                'architecture_v2_modular_temperature', 0.75)),
            scale_equivariant_comm_heads_enabled=bool(getattr(
                config.marl,
                'scale_equivariant_comm_heads_enabled', False)),
            permutation_equivariant_round_encoding_enabled=bool(getattr(
                config.marl,
                'permutation_equivariant_round_encoding_enabled', False)),
            comm_channel_feedback_rate_enabled=bool(getattr(
                config.marl,
                'comm_channel_feedback_rate_enabled', False)),
            comm_channel_feedback_dim=int(getattr(
                config.marl, 'comm_channel_feedback_dim', 6)),
            equivariant_value_critic_enabled=bool(getattr(
                config.marl,
                'equivariant_value_critic_enabled', False)),
            set_risk_critic_enabled=bool(getattr(
                config.marl, 'set_risk_critic_enabled', False)),
            risk_critic_hidden_dim=int(getattr(
                config.marl, 'risk_critic_hidden_dim', 128)),
            risk_critic_num_quantiles=int(getattr(
                config.marl, 'risk_critic_num_quantiles', 16)),
            risk_critic_cvar_alpha=float(getattr(
                config.marl, 'risk_critic_cvar_alpha', 0.20)),
            risk_critic_monotonic_quantiles_enabled=bool(getattr(
                config.marl,
                'risk_critic_monotonic_quantiles_enabled', False)),
        )
        agents.append(agent)

    # Warm-start: load a pretrained (e.g. DAgger-cloned) actor into the SHARED actor.
    warm_runtime = None
    if args.warm_start:
        ckpt = torch.load(args.warm_start, map_location=device)
        actor_state = ckpt.get('actor', ckpt)  # unwrap if dict
        if isinstance(ckpt, dict):
            warm_runtime = ckpt.get('runtime')
        # Allow missing aux head keys (added after warmstart was generated)
        missing, unexpected = agents[0].load_actor_state_dict_compatible(actor_state)
        if missing:
            print(f"warm-start: {len(missing)} new keys initialised (e.g. pd_aux_head)")
        if bool(getattr(
                config.marl, 'adaptive_topk_reset_rate_head', False)):
            if not bool(getattr(
                    config.marl, 'adaptive_topk_from_rate_enabled', False)):
                raise ValueError(
                    'adaptive_topk_reset_rate_head requires adaptive top-k')
            rate_head = (
                agents[0].actor.comm_set_rate_head
                if getattr(
                    agents[0].actor,
                    '_scale_equivariant_comm_heads_enabled', False)
                else agents[0].actor.comm_rate_head)
            with torch.no_grad():
                rate_head.weight.zero_()
                rate_head.bias.zero_()
                feedback_head = getattr(
                    agents[0].actor, 'comm_rate_feedback_head', None)
                if feedback_head is not None:
                    feedback_head.weight.zero_()
                    feedback_head.bias.zero_()
                # Silence remains available but does not consume one third of
                # early exploration. Active k choices start equiprobable and
                # can immediately acquire state-dependent weight gradients.
                rate_head.bias[0] = -4.0
            print('adaptive top-k rate head reset: active k logits balanced')
        if 'critic' in ckpt:
            critic_state = ckpt['critic']
            risk_critic_enabled = bool(getattr(
                config.marl, 'set_risk_critic_enabled', False))
            if not risk_critic_enabled:
                # A training checkpoint may contain CTDE-only set-risk heads
                # that are deliberately absent from the deployment/evaluation
                # critic. Keep the complete compatible base critic and discard
                # only keys that do not exist in the current architecture.
                current_critic_state = agents[0].critic.state_dict()
                dropped_critic_keys = sorted(
                    key for key in critic_state
                    if key not in current_critic_state)
                critic_state = {
                    key: value for key, value in critic_state.items()
                    if (key in current_critic_state
                        and value.shape == current_critic_state[key].shape)
                }
                if dropped_critic_keys:
                    print(
                        'warm-start: omitted '
                        f'{len(dropped_critic_keys)} CTDE-only critic tensors '
                        'from deployment evaluation')
            critic_result = agents[0].critic.load_state_dict(
                critic_state,
                strict=not risk_critic_enabled)
            if critic_result.missing_keys:
                print(
                    'warm-start: initialized '
                    f'{len(critic_result.missing_keys)} new risk-critic tensors')
        # Keep default log_std=0 (σ=1) for warm-start with KL BC anchor.
        # KL anchor constrains σ in probability space — no need to force low exploration.
        print(f"warm-started actor{'+critic' if 'critic' in ckpt else ''} from {args.warm_start}"
              f"  (σ=1, MSE BC β={config.marl.bc_beta_init})")

    # Warm-start mode
    if args.warm_start:
        if args.warm_start_mode == "residual":
            from uav_isac.agents.residual_actor import ResidualActor
            base_actor = agents[0].actor
            residual = ResidualActor(base_actor, max_dp=config.uav.v_max * config.scenario.dt, delta_max=0.06)
            for agent in agents:
                agent.actor = residual
                agent.actor_optimizer = torch.optim.Adam(
                    residual.residual.parameters(), lr=train_lr)
            print(f"ResidualActor: delta_max=0.06, base frozen, {sum(p.numel() for p in residual.residual.parameters())} trainable params")
        elif args.warm_start_mode == "risk_residual":
            from uav_isac.agents.residual_actor import RiskGatedResidualActor
            base_actor = agents[0].actor
            residual = RiskGatedResidualActor(
                base_actor,
                delta_max=config.marl.risk_residual_delta_max,
                hidden_units=config.marl.risk_residual_hidden_dim,
                gate_bias=config.marl.risk_residual_gate_bias,
                risk_floor=config.marl.risk_target_floor,
                directional_basis_enabled=(
                    config.marl.risk_residual_directional_basis_enabled),
            ).to(device)
            residual_lr = float(config.marl.risk_residual_learning_rate)
            shared_optimizer = torch.optim.Adam(
                residual.trainable_params, lr=residual_lr)
            for agent in agents:
                agent.actor = residual
                agent.actor_optimizer = shared_optimizer
            print(
                "RiskGatedResidualActor: "
                f"delta_max={residual.delta_max:.3f}, gate_bias="
                f"{config.marl.risk_residual_gate_bias:.2f}, base frozen, "
                f"{sum(p.numel() for p in residual.trainable_params)} "
                f"trainable params, lr={residual_lr:.1e}")
        else:
            # direct: already loaded into StructuredActorNetwork, no wrapping
            print(f"Warm-start mode=direct: actor loaded as-is (no ResidualActor wrap)")

    # BC anchor: skip for warm-started actors
    bc_actor = None
    if args.warm_start and not action_space.structured_actor:
        bc_actor = MAPPOAgent(agent_id=0, obs_dim=obs_dim, global_state_dim=global_dim,
                              action_space=action_space, num_agents=K,
                               hidden_layers=config.marl.hidden_layers,
                               lr=config.marl.lr, max_grad_norm=config.marl.max_grad_norm,
                               device=device,
                               comm_num_rate_levels=len(config.marl.comm_rate_bits_per_dim),
                               comm_log_std_init=config.marl.comm_message_log_std_init,
                               comm_entropy_scale=config.marl.comm_entropy_scale,
                               use_comm_cross_attention=config.marl.comm_cross_attention_enabled,
                               comm_token_dim=receiver_token_dim,
                               comm_tokens_per_sender=tokens_per_sender,
                               comm_payload_dim=comm_payload_dim,
                               comm_target_token_enabled=target_tokens,
                               comm_target_token_dim=target_token_dim,
                               use_target_allocation=bool(
                                   config.marl.target_allocation_enabled
                                   or config.marl.target_allocation_teacher_enabled),
                               use_team_sinkhorn=bool(
                                   config.marl.target_allocation_sinkhorn_enabled))
        bc_actor.actor.load_state_dict(agents[0].actor.state_dict())
        for p in bc_actor.actor.parameters():
            p.requires_grad = False
        bc_actor.actor.eval()
        print(f"BC anchor actor frozen (β_init={config.marl.bc_beta_init})")

    # Optional zero-initialized modules consume RNG while being constructed.
    # Reset after the complete network/warm-start setup so paired variants use
    # identical stochastic action streams; otherwise a no-op architecture can
    # appear to change performance solely through shifted sampling noise.
    set_seed(seed)

    # Create trainer
    trainer = MAPPTrainer(
        env=env,
        agents=agents,
        config=config,
        device=device,
    )
    trainer._bc_actor = bc_actor.actor if bc_actor else None
    if warm_runtime and not args.ignore_warm_runtime:
        trainer.restore_policy_runtime_state(warm_runtime)
        print('restored policy runtime state: '
              f"frames={trainer.total_frames} "
              f"capacity_blend={warm_runtime.get('capacity_matching_blend', 0.0):.5f} "
              f"movement_blend={warm_runtime.get('target_allocation_movement_blend', 0.0):.5f}")

    print(f"\nStarting training for {config.marl.num_episodes} episodes...")
    print(f"Rollout steps: {config.marl.rollout_steps}")
    print(f"Gamma: {config.marl.gamma}, GAE lambda: {config.marl.gae_lambda}")
    print(f"PPO clip: {config.marl.ppo_clip}, Epochs: {config.marl.ppo_epochs}")
    print("-" * 60)

    # Diagnostic: confirm actual LR values
    actor_lr = agents[0].actor_optimizer.param_groups[0]['lr']
    critic_lr = agents[0].critic_optimizer.param_groups[0]['lr']
    print(f"[LR] actor={actor_lr:.1e} critic={critic_lr:.1e} bc_beta={config.marl.bc_beta_init} "
          f"entropy_coef={trainer.entropy_coef:.3f} ppo_epochs={config.marl.ppo_epochs} "
          f"ppo_clip={config.marl.ppo_clip} gae_lambda={config.marl.gae_lambda}")

    # Train
    metrics_history = trainer.train(
        num_episodes=args.episodes if args.episodes is not None else config.marl.num_episodes,
        log_interval=10,
    )
    # Save final metrics summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total frames: {trainer.total_frames}")

    if metrics_history:
        final_window = metrics_history[-50:]
        avg_pd_values = [m.get('avg_P_D', 0) for m in final_window if 'avg_P_D' in m]
        if avg_pd_values:
            print(f"Final 50-ep avg P_D: {np.mean(avg_pd_values):.4f}")

        actor_losses = [m.get('actor_loss', 0) for m in metrics_history]
        print(f"Initial actor loss: {actor_losses[0]:.4f}" if actor_losses else "")
        print(f"Final actor loss: {actor_losses[-1]:.4f}" if actor_losses else "")

    # Save results for reproducibility and paired bootstrap
    import csv, json as _json, subprocess as _sp
    # Auto-derive variant from config name: exp_800_q4_full → full
    config_stem = os.path.splitext(os.path.basename(args.config or "config/default.yaml"))[0]
    variant = config_stem.replace("exp_800_q4_", "") if "exp_800_q4_" in config_stem else config_stem
    out_dir = args.out_dir or os.path.join("results", "full_eh", variant, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    if trainer.last_unrestored_params is not None:
        # Diagnostic only. best_restored.pt below remains the sole deployable
        # checkpoint selected by QoS feasibility/worst/weak3/steady/-bits.
        torch.save(
            trainer.last_unrestored_params['actor'],
            os.path.join(out_dir, "last_unrestored_diagnostic.pt"))
        if bool(getattr(
                trainer.agents[0].critic,
                'set_risk_critic_enabled', False)):
            # Risk calibration runs may fail during a later evaluation
            # diagnostic. Preserve the completed training state before final
            # evaluation so the critic never needs to be refitted.
            torch.save({
                'actor': trainer.last_unrestored_params['actor'],
                'critic': trainer.last_unrestored_params['critic'],
                'runtime': trainer.get_policy_runtime_state(),
            }, os.path.join(out_dir, "risk_critic_final.pt"))

    # Commit hash
    try:
        commit = _sp.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    # Manifest
    manifest = {
        "git_commit": commit,
        "config": args.config or "config/default.yaml",
        "seed": seed,
        "K": config.scenario.K, "Q": config.scenario.Q,
        "warm_start": args.warm_start,
        "warm_start_mode": args.warm_start_mode,
        "warm_runtime_restored": bool(
            warm_runtime and not args.ignore_warm_runtime),
        "risk_residual": {
            "enabled": args.warm_start_mode == "risk_residual",
            "directional_weak_target_basis": (
                config.marl.risk_residual_directional_basis_enabled),
            "delta_max": config.marl.risk_residual_delta_max,
            "hidden_dim": config.marl.risk_residual_hidden_dim,
            "gate_bias": config.marl.risk_residual_gate_bias,
            "learning_rate": config.marl.risk_residual_learning_rate,
            "risk_floor": config.marl.risk_target_floor,
        },
        "scale_equivariant_comm_heads_enabled": (
            config.marl.scale_equivariant_comm_heads_enabled),
        "permutation_equivariant_round_encoding_enabled": (
            config.marl.permutation_equivariant_round_encoding_enabled),
        "set_risk_critic": {
            "enabled": config.marl.set_risk_critic_enabled,
            "hidden_dim": config.marl.risk_critic_hidden_dim,
            "num_quantiles": config.marl.risk_critic_num_quantiles,
            "cvar_alpha": config.marl.risk_critic_cvar_alpha,
            "monotonic_quantiles_enabled": (
                config.marl.risk_critic_monotonic_quantiles_enabled),
            "qos_floor": config.marl.risk_critic_qos_floor,
            "quantile_coef": config.marl.risk_critic_quantile_coef,
            "constraint_coef": config.marl.risk_critic_constraint_coef,
            "constraint_positive_weight": (
                config.marl.risk_critic_constraint_positive_weight),
            "critic_only_training": (
                config.marl.risk_critic_only_training),
        },
        "learned_comm_mode": config.marl.learned_comm_mode,
        "tracking_enabled": config.marl.tracking_enabled,
        "ground_communication_enabled": config.marl.ground_communication_enabled,
        "comm_only_neighbor_information": config.marl.comm_only_neighbor_information,
        "comm_rate_bits_per_dim": config.marl.comm_rate_bits_per_dim,
        "comm_tx_power_w": config.marl.comm_tx_power_w,
        "comm_deadline_s": config.marl.comm_deadline_s,
        "comm_snr_threshold_db": config.marl.comm_snr_threshold_db,
        "comm_channel_randomization": {
            "enabled": (
                config.marl.comm_channel_randomization_enabled),
            "snr_threshold_db_values": (
                config.marl
                .comm_channel_randomization_snr_threshold_db_values),
            "deadline_s_values": (
                config.marl.comm_channel_randomization_deadline_s_values),
            "profile_weights": (
                config.marl.comm_channel_randomization_profile_weights),
        },
        "comm_cross_attention_enabled": config.marl.comm_cross_attention_enabled,
        "comm_message_ttl_frames": config.marl.comm_message_ttl_frames,
        "comm_bit_cost_weight": config.marl.comm_bit_cost_weight,
        "comm_energy_cost_weight": config.marl.comm_energy_cost_weight,
        "comm_delay_cost_weight": config.marl.comm_delay_cost_weight,
        "comm_entropy_scale": config.marl.comm_entropy_scale,
        "comm_payload_mode": config.marl.comm_payload_mode,
        "comm_target_token_dim": config.marl.comm_target_token_dim,
        "comm_payload_dim": comm_payload_dim,
        "joint_isac_power_enabled": config.marl.joint_isac_power_enabled,
        "movement_decision_interval": config.marl.movement_decision_interval,
        "learn_roles": config.marl.learn_roles,
        "multistatic_subslot_enabled": (
            config.marl.multistatic_subslot_enabled),
        "distributed_target_commitment_enabled": (
            config.marl.distributed_target_commitment_enabled),
        "distributed_target_commitment_topk": (
            config.marl.distributed_target_commitment_topk),
        "distributed_target_commitment_require_receiver": (
            config.marl.distributed_target_commitment_require_receiver),
        "distributed_target_commitment_mode": (
            config.marl.distributed_target_commitment_mode),
        "distributed_target_commitment_soft_floor": (
            config.marl.distributed_target_commitment_soft_floor),
        "distributed_target_commitment_uncertainty_relief": (
            config.marl.distributed_target_commitment_uncertainty_relief),
        "distributed_target_commitment_source": (
            config.marl.distributed_target_commitment_source),
        "p0_maxmin_pairing_enabled": (
            config.marl.p0_maxmin_pairing_enabled),
        "p0_maxmin_bypass_commitment_filter": (
            config.marl.p0_maxmin_bypass_commitment_filter),
        "p0_maxmin_pairing_hold_frames": (
            config.marl.p0_maxmin_pairing_hold_frames),
        "p0_maxmin_local_fusion_enabled": (
            config.marl.p0_maxmin_local_fusion_enabled),
        "p0_maxmin_deficit_priority_gain": (
            config.marl.p0_maxmin_deficit_priority_gain),
        "hyperedge_negotiation": {
            "enabled": config.marl.hyperedge_negotiation_enabled,
            "share_topk": config.marl.hyperedge_share_topk,
            "distance_scale_m": config.marl.hyperedge_distance_scale_m,
            "capability_mode": config.marl.hyperedge_capability_mode,
            "deficit_gain": config.marl.hyperedge_deficit_gain,
            "proxy_floor": config.marl.hyperedge_proxy_floor,
            "pair_score_mode": config.marl.hyperedge_pair_score_mode,
            "state_stream_enabled": (
                config.marl.hyperedge_state_stream_enabled),
            "consensus_rounds": config.marl.hyperedge_consensus_rounds,
            "assignment_hold_frames": (
                config.marl.hyperedge_assignment_hold_frames),
            "min_target_coverage": (
                config.marl.hyperedge_min_target_coverage),
            "safety_fallback_enabled": (
                config.marl.hyperedge_safety_fallback_enabled),
        },
        "round_negotiation_enabled": config.marl.round_negotiation_enabled,
        "round_negotiation_strength": config.marl.round_negotiation_strength,
        "round_negotiation_temperature": (
            config.marl.round_negotiation_temperature),
        "sparse_claim_enabled": config.marl.sparse_claim_enabled,
        "sparse_claim_share_topk": config.marl.sparse_claim_share_topk,
        "sparse_claim_commit_topk": config.marl.sparse_claim_commit_topk,
        "sparse_claim_desired_endpoints": (
            config.marl.sparse_claim_desired_endpoints),
        "sparse_claim_full_penalty": config.marl.sparse_claim_full_penalty,
        "sparse_claim_vacant_bonus": config.marl.sparse_claim_vacant_bonus,
        "sparse_claim_temperature": config.marl.sparse_claim_temperature,
        "sparse_claim_underload_coef": (
            config.marl.sparse_claim_underload_coef),
        "sparse_claim_overload_coef": (
            config.marl.sparse_claim_overload_coef),
        "adaptive_topk_from_rate_enabled": (
            config.marl.adaptive_topk_from_rate_enabled),
        "adaptive_topk_rate_mapping": (
            config.marl.adaptive_topk_rate_mapping),
        "adaptive_topk_rate_only_training": (
            config.marl.adaptive_topk_rate_only_training),
        "adaptive_topk_reset_rate_head": (
            config.marl.adaptive_topk_reset_rate_head),
        "adaptive_topk_worst_credit_coef": (
            config.marl.adaptive_topk_worst_credit_coef),
        "comm_rate_metadata_denominator": (
            config.marl.comm_rate_metadata_denominator),
        "comm_channel_feedback_rate_enabled": (
            config.marl.comm_channel_feedback_rate_enabled),
        "comm_channel_feedback_dim": (
            config.marl.comm_channel_feedback_dim),
        "comm_sender_delivery_penalty_enabled": (
            config.marl.comm_sender_delivery_penalty_enabled),
        "comm_sender_delivery_penalty_weight": (
            config.marl.comm_sender_delivery_penalty_weight),
        "comm_aided_sensing_enabled": (
            config.marl.comm_aided_sensing_enabled),
        "comm_aided_sensing_blend": (
            config.marl.comm_aided_sensing_blend),
        "comm_aided_sensing_aux_coef": (
            config.marl.comm_aided_sensing_aux_coef),
        "comm_aided_sensing_counterfactual_coef": (
            config.marl.comm_aided_sensing_counterfactual_coef),
        "comm_aided_sensing_temperature": (
            config.marl.comm_aided_sensing_temperature),
        "comm_aided_sensing_margin": (
            config.marl.comm_aided_sensing_margin),
        "semantic_kinematic_field_enabled": (
            config.marl.semantic_kinematic_field_enabled),
        "semantic_kinematic_field_gain": (
            config.marl.semantic_kinematic_field_gain),
        "target_conditioned_movement_enabled": (
            config.marl.target_conditioned_movement_enabled),
        "target_conditioned_movement_gain": (
            config.marl.target_conditioned_movement_gain),
        "architecture_v2_enabled": (
            config.marl.architecture_v2_enabled),
        "architecture_v2_prior_gain": (
            config.marl.architecture_v2_prior_gain),
        "architecture_v2_distance_weight": (
            config.marl.architecture_v2_distance_weight),
        "architecture_v2_qos_floor": (
            config.marl.architecture_v2_qos_floor),
        "architecture_v2_comm_prior_gain": (
            config.marl.architecture_v2_comm_prior_gain),
        "architecture_v2_comm_crisis_threshold": (
            config.marl.architecture_v2_comm_crisis_threshold),
        "architecture_v2_consensus_enabled": (
            config.marl.architecture_v2_consensus_enabled),
        "architecture_v2_matching_temperature": (
            config.marl.architecture_v2_matching_temperature),
        "architecture_v2_movement_consensus_blend": (
            config.marl.architecture_v2_movement_consensus_blend),
        "architecture_v2_endpoint_consensus_gain": (
            config.marl.architecture_v2_endpoint_consensus_gain),
        "architecture_v2_bid_residual_scale": (
            config.marl.architecture_v2_bid_residual_scale),
        "architecture_v2_modular_coordination_enabled": (
            config.marl.architecture_v2_modular_coordination_enabled),
        "architecture_v2_modular_num_experts": (
            config.marl.architecture_v2_modular_num_experts),
        "architecture_v2_modular_gain": (
            config.marl.architecture_v2_modular_gain),
        "architecture_v2_modular_temperature": (
            config.marl.architecture_v2_modular_temperature),
        "architecture_v2_modular_balance_coef": (
            config.marl.architecture_v2_modular_balance_coef),
        "architecture_v2_modular_specialization_coef": (
            config.marl.architecture_v2_modular_specialization_coef),
        "architecture_v2_modular_lr_scale": (
            config.marl.architecture_v2_modular_lr_scale),
        "equivariant_value_critic_enabled": (
            config.marl.equivariant_value_critic_enabled),
        "P_isac_total": config.uav.P_isac_total,
        "comm_power_fraction_bounds": [
            config.marl.comm_power_fraction_min,
            config.marl.comm_power_fraction_max,
        ],
        "comm_qos_constrained": config.marl.comm_qos_constrained,
        "eval_seed_bank_path": config.marl.eval_seed_bank_path,
        "eval_seed_split": config.marl.eval_seed_split,
        "final_eval_seed_split": (
            args.final_eval_split or config.marl.final_eval_seed_split),
        "eval_seed_count": len(trainer.eval_seeds),
        "checkpoint_confidence_alpha": (
            config.marl.checkpoint_confidence_alpha),
        "checkpoint_bootstrap_samples": (
            config.marl.checkpoint_bootstrap_samples),
        "checkpoint_cvar_fraction": config.marl.checkpoint_cvar_fraction,
        "checkpoint_confirmation_enabled": (
            config.marl.checkpoint_confirmation_enabled),
        "checkpoint_confirmation_split": (
            config.marl.checkpoint_confirmation_split),
        "comm_qos_reward_scale": config.marl.comm_qos_reward_scale,
        "comm_qos_floors": {
            "steady": config.marl.comm_qos_steady_min,
            "weak3": config.marl.comm_qos_weak3_min,
            "worst": config.marl.comm_qos_worst_min,
        },
        "coord_reward_enabled": config.marl.coord_reward_enabled,
        "coord_reward_stage": config.marl.coord_reward_stage,
        "coord_reward_ema_alpha": config.marl.coord_reward_ema_alpha,
        "coord_reward_floors": {
            "steady": config.marl.coord_reward_steady_floor,
            "weak3": config.marl.coord_reward_weak3_floor,
            "worst": config.marl.coord_reward_worst_floor,
        },
        "coord_reward_weights": {
            "worst": config.marl.coord_reward_worst_weight,
            "duplicate": config.marl.coord_reward_duplicate_weight,
            "weak3": config.marl.coord_reward_weak3_weight,
            "steady": config.marl.coord_reward_steady_weight,
        },
        "headwise_credit_enabled": config.marl.headwise_credit_enabled,
        "headwise_credit_vf_coef": config.marl.headwise_credit_vf_coef,
        "headwise_credit_coefs": {
            "movement": config.marl.headwise_movement_coef,
            "message": config.marl.headwise_message_coef,
            "rate": config.marl.headwise_rate_coef,
            "resource": config.marl.headwise_resource_coef,
        },
        "headwise_message_credit_delay_decisions": (
            1 if config.marl.headwise_credit_enabled else 0),
        "headwise_detach_rate_resource_from_token": bool(
            config.marl.headwise_credit_enabled),
        "causal_ccp_enabled": config.marl.causal_ccp_enabled,
        "causal_ccp": {
            "teacher": "paired_one_step_common_random_number_intervention",
            "unit": "sender_target_token",
            "hidden_dim": config.marl.causal_ccp_hidden_dim,
            "lr": config.marl.causal_ccp_lr,
            "epochs": config.marl.causal_ccp_epochs,
            "intervention_stride": (
                config.marl.causal_ccp_intervention_stride),
            "tail_temperature": config.marl.causal_ccp_tail_temperature,
            "message_mix": config.marl.causal_ccp_message_mix,
        },
        "comm_encouragement_enabled": (
            config.marl.comm_encouragement_enabled),
        "comm_encouragement_weight": config.marl.comm_encouragement_weight,
        "comm_encouragement_floor_ratio": (
            config.marl.comm_encouragement_floor_ratio),
        "comm_rate_bonus_enabled": config.marl.comm_rate_bonus_enabled,
        "comm_rate_bonus_weight": config.marl.comm_rate_bonus_weight,
        "comm_rate_bonus_target_bits": config.marl.comm_rate_bonus_target_bits,
        "comm_rate_bonus_aux_coef": config.marl.comm_rate_bonus_aux_coef,
        "comm_rate_bonus_aux_lr": config.marl.comm_rate_bonus_aux_lr,
        "comm_silence_penalty_enabled": (
            config.marl.comm_silence_penalty_enabled),
        "comm_silence_grace_decisions": (
            config.marl.comm_silence_grace_decisions),
        "comm_silence_penalty_per_decision": (
            config.marl.comm_silence_penalty_per_decision),
        "comm_silence_penalty_max": config.marl.comm_silence_penalty_max,
        "comm_eval_force_silence": config.marl.comm_eval_force_silence,
        "comm_eval_force_rate_bits": (
            config.marl.comm_eval_force_rate_bits),
        "comm_eval_message_ablation": (
            config.marl.comm_eval_message_ablation),
        "eval_centralized_assignment_movement": (
            config.marl.eval_centralized_assignment_movement),
        "eval_qos_bistatic_assignment_movement": (
            config.marl.eval_qos_bistatic_assignment_movement),
        "comm_qos_min_rate_bits": config.marl.comm_qos_min_rate_bits,
        "target_allocation_enabled": config.marl.target_allocation_enabled,
        "target_allocation_teacher_enabled": (
            config.marl.target_allocation_teacher_enabled),
        "target_allocation_temperature": (
            config.marl.target_allocation_temperature),
        "target_allocation_straight_through": (
            config.marl.target_allocation_straight_through),
        "target_allocation_movement_blend": (
            config.marl.target_allocation_movement_blend),
        "target_allocation_movement_blend_schedule": [
            config.marl.target_allocation_movement_blend_start,
            config.marl.target_allocation_movement_blend_end,
            config.marl.target_allocation_movement_blend_anneal_frames,
        ],
        "target_allocation_movement_confidence_gating_enabled": (
            config.marl.target_allocation_movement_confidence_gating_enabled),
        "target_allocation_movement_confidence_floor": (
            config.marl.target_allocation_movement_confidence_floor),
        "target_allocation_movement_confidence_power": (
            config.marl.target_allocation_movement_confidence_power),
        "hierarchical_dual_assignment_enabled": (
            config.marl.hierarchical_dual_assignment_enabled),
        "movement_team_matching_enabled": (
            config.marl.movement_team_matching_enabled),
        "movement_team_matching_temperature": (
            config.marl.movement_team_matching_temperature),
        "movement_team_matching_iterations": (
            config.marl.movement_team_matching_iterations),
        "movement_team_matching_blend": (
            config.marl.movement_team_matching_blend),
        "movement_team_matching_intrinsic_bid_mix": (
            config.marl.movement_team_matching_intrinsic_bid_mix),
        "target_allocation_resource_blend": (
            config.marl.target_allocation_resource_blend),
        "target_allocation_differentiable_comm": (
            config.marl.target_allocation_differentiable_comm),
        "target_allocation_temporal_coef": (
            config.marl.target_allocation_temporal_coef),
        "target_allocation_aux_epochs": (
            config.marl.target_allocation_aux_epochs),
        "target_allocation_aux_lr": config.marl.target_allocation_aux_lr,
        "capacity_matching_enabled": config.marl.capacity_matching_enabled,
        "capacity_matching_capacities": [
            config.marl.capacity_matching_row_capacity,
            config.marl.capacity_matching_column_capacity,
        ],
        "capacity_matching_temperature": (
            config.marl.capacity_matching_temperature),
        "capacity_matching_iterations": (
            config.marl.capacity_matching_iterations),
        "capacity_matching_blend_schedule": [
            config.marl.capacity_matching_blend_start,
            config.marl.capacity_matching_blend_end,
            config.marl.capacity_matching_anneal_frames,
        ],
        "advantage_mode": config.marl.advantage_mode,
        "risk_tail_fraction": config.marl.risk_tail_fraction,
        "training_seed_replay_enabled": (
            config.marl.training_seed_replay_enabled),
        "freeze_attention": getattr(config.marl, "freeze_attention", False),
        "use_per_module_lr": getattr(config.marl, "use_per_module_lr", False),
        "best_steady_P_D": float(trainer.best_score),
        "best_weak3_P_D": float(getattr(
            trainer, "_best_eval_metrics", {}).get("eval_weak3_P_D", 0.0)),
        "best_worst_P_D": float(getattr(
            trainer, "_best_eval_metrics", {}).get("eval_worst_P_D", 0.0)),
        "best_worst_lcb": float(getattr(
            trainer, "_best_eval_metrics", {}).get("eval_worst_lcb", 0.0)),
        "best_worst_cvar": float(getattr(
            trainer, "_best_eval_metrics", {}).get("eval_worst_cvar", 0.0)),
        "best_qos_feasible_wilson_lcb": float(getattr(
            trainer, "_best_eval_metrics", {}).get(
                "eval_qos_feasible_wilson_lcb", 0.0)),
        "checkpoint_selection": (
            "wilson_feasible_lcb,bootstrap_worst_lcb,worst_cvar,"
            "strict_worst,trimmed_worst,weak3,steady,-bits"),
        "total_frames": trainer.total_frames,
        "policy_runtime": trainer.get_policy_runtime_state(),
        "total_episodes": len(metrics_history),
        "max_final_eval_seeds": max(0, int(args.max_final_eval_seeds)),
        "target_choice_audit_stride": max(
            0, int(args.target_choice_audit_stride)),
        "sensing_choice_audit_stride": max(
            0, int(args.sensing_choice_audit_stride)),
        "sensing_residual_blend": float(np.clip(
            args.sensing_residual_blend, 0.0, 1.0)),
        "sensing_audit_horizon": max(
            0, int(args.sensing_audit_horizon)),
        "sensing_oracle_control": bool(args.sensing_oracle_control),
        "joint_sensing_pair_audit_stride": max(
            0, int(args.joint_sensing_pair_audit_stride)),
        "physical_oracle_stride": max(
            0, int(args.physical_oracle_stride)),
        "evidence_trace_output": args.evidence_trace_output,
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as f:
        _json.dump(manifest, f, indent=2)

    # Training metrics CSV
    if metrics_history:
        keys = sorted(metrics_history[0].keys())
        with open(os.path.join(out_dir, "train_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for m in metrics_history:
                w.writerow({k: m.get(k, "") for k in keys})

    # Per-episode evaluation on FIXED test bank for paired bootstrap
    final_split = args.final_eval_split or config.marl.final_eval_seed_split
    if config.marl.eval_seed_bank_path:
        paired_seeds = load_stratified_seed_split(
            config.marl.eval_seed_bank_path, final_split)
    else:
        paired_seeds = [30001, 30002, 30003, 30004, 30005,
                        30006, 30007, 30008, 30009, 30010,
                        30011, 30012, 30013, 30014, 30015,
                        30016, 30017, 30018, 30019, 30020]
    if args.max_final_eval_seeds > 0:
        paired_seeds = paired_seeds[:args.max_final_eval_seeds]
    try:
        trainer.agents[0].actor.eval()
        # Use _evaluate with streaming GRU (already fixed in P0-3)
        ev = trainer._evaluate(
            n_episodes=len(paired_seeds),
            eval_seeds=paired_seeds,
            target_choice_audit_stride=max(
                0, int(args.target_choice_audit_stride)),
            sensing_choice_audit_stride=max(
                0, int(args.sensing_choice_audit_stride)),
            sensing_residual_blend=float(np.clip(
                args.sensing_residual_blend, 0.0, 1.0)),
            sensing_audit_horizon=max(
                0, int(args.sensing_audit_horizon)),
            sensing_oracle_control=bool(args.sensing_oracle_control),
            joint_sensing_pair_audit_stride=max(
                0, int(args.joint_sensing_pair_audit_stride)),
            physical_oracle_stride=max(
                0, int(args.physical_oracle_stride)),
            evidence_trace_output=args.evidence_trace_output,
        )
        eval_keys = sorted(ev.keys())
        with open(os.path.join(out_dir, "paired_eval.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(eval_keys)
            w.writerow([ev.get(k, "") for k in eval_keys])
        # Save best checkpoint
        checkpoint = {
            'actor': trainer.agents[0].actor.state_dict(),
            # Persist the standard and headwise critics even when the optional
            # set-risk branch is disabled.  Actor-only historical checkpoints
            # made post-hoc temporal-credit audits impossible.
            'critic': trainer.agents[0].critic.state_dict(),
            'runtime': trainer.get_policy_runtime_state(),
            'centralized_critic': bool(trainer.centralized_critic),
        }
        torch.save(checkpoint, os.path.join(out_dir, "best_restored.pt"))
    except Exception as e:
        print(f"  [warn] paired eval failed: {e}")

    print(f"Results saved → {out_dir}")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
