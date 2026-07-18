"""Configuration parameter dataclasses and YAML loader."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml
import os


@dataclass
class ScenarioParams:
    region_size: Tuple[float, float] = (400.0, 400.0)  # Device-free MARL ISAC (TVT 2024): 400x400 m
    height: float = 20.0                                # TVT 2024: H=20 m
    K: int = 4
    Q: int = 2
    T: int = 150                                        # traversable: 2.5*150=375m > half-diag 283m
    dt: float = 0.1
    C: int = 1


@dataclass
class UAVParams:
    v_max: float = 25.0
    d_safe: float = 20.0
    P_sense: float = 0.0251   # 14 dBm — Device-free MARL ISAC (TVT 2024)
    P_report: float = 0.25    # W — TVT 2024 comm power set {0.25..1 W}
    B_max: float = 50000.0
    P_fly_static: float = 80.0
    P_fly_coeff: float = 0.05


@dataclass
class TargetParams:
    motion_model: str = "CV"
    # Stage-1 curriculum: slow targets [0,5] so UAV (v_max=25) can hold a tight
    # bistatic pair -> task is learnable, P_D can approach the ~1.0 ceiling.
    # Ramp up to (0,20) only after MAPPO reaches high P_D on slow targets.
    speed_range: Tuple[float, float] = (0.0, 5.0)
    sigma_a: float = 0.5
    rcs: float = 1.0
    omega_q: List[float] = field(default_factory=lambda: [0.5, 0.5])


@dataclass
class OTFSParams:
    fc: float = 2.8e+10
    B: float = 1.0e+6        # 1 MHz — Device-free MARL ISAC (TVT 2024)
    delta_f: float = 1.5625e+4  # B/M = 1e6/64
    M: int = 64
    N: int = 16
    T_sym: float = 6.4e-5    # 1/delta_f
    g_tx_dBi: float = 16.0   # 64-elem UAV phased array (lit: 8 dBi single / "large arrays")
    g_rx_dBi: float = 16.0   # rx array gain
    n_cpi: int = 128         # coherent integration -> sensing radius ~120 m (matches 400x400)


@dataclass
class ChannelParams:
    NF: float = 4.0          # kT*B*NF = 4e-21*1e6*2.51 = 1e-14 W = -110 dBm (TVT 2024 N0)
    kT: float = 4.0e-21
    ric_K: float = 6.0
    # Low-altitude (H=20m) report-link blockage: Al-Hourani LoS/NLoS (suburban). ON.
    use_los_prob: bool = True
    los_a: float = 4.88
    los_b: float = 0.43
    eta_los_dB: float = 0.1
    eta_nlos_dB: float = 21.0
    # Swerling-II RCS fading on the sensing return (default OFF).
    use_swerling: bool = False


@dataclass
class DetectionParams:
    P_FA: float = 0.001
    g_min: float = 0.5
    K_q_max: int = 3
    B_q: int = 64
    # Long-term fairness floor (constraint D4). 0.8 was unreachable even for the
    # best scripted policy (Greedy steady-state P_D ~0.2-0.4), which pinned the
    # Lagrangian multiplier and biased training. Set to a reachable-but-binding
    # value; re-tune from run_baselines.py 'steady_P_D' output.
    P_D_min: float = 0.2
    T_report: float = 0.005


@dataclass
class P0SolverParams:
    capacity_per_rx: int = 256
    latency_max: float = 0.005


@dataclass
class MARLParams:
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95  # 0.98→0.95: per-agent reward adds variance, lower λ reduces advantage noise
    ppo_clip: float = 0.1     # 0.2→0.1: gentler updates near BC anchor
    ppo_epochs: int = 2       # 4→2: fewer gradient steps/rollout, less drift
    rollout_steps: int = 2048
    minibatch_size: int = 256
    entropy_init: float = 0.03  # 0.08→0.03: lower entropy preserves BC behavior
    entropy_final: float = 0.005  # 0.08→0.005: decay to near-deterministic
    entropy_decay_frames: int = 500_000  # faster decay
    eta_mc: float = 0.5
    eta_sense: float = 0.0  # per-agent sensing: 0=off (team-only baseline), 0.1=on
    use_centered_marginal: bool = False  # centered marginal contribution shaping
    use_difference_reward: bool = True   # fixed-assignment no-op difference reward
    team_weight: float = 0.7             # team reward weight (E3 baseline)
    diff_weight: float = 0.3             # difference reward weight (E3 baseline)
    comm_cost_weight: float = 0.001     # ||comm_msg||^2 penalty (ISAC resource constraint)
    use_distance_shaping: bool = False  # potential-based "approach target" shaping
    shape_w: float = 0.01               # shaping weight (action signal ~ shape_w * 2.5/step)
    # Communication-cost weight in reward. MUST be small vs detection utility
    # (U_q=-log(1-P_D) ~ 0.3-2). At 1e-3, lambda*bits (~0.4) exceeded utility,
    # so "do nothing" paid better than detecting -> P_D collapsed. Comms is also
    # hard-capped by p0_solver.capacity_per_rx, so this is only a light regularizer.
    lambda_report: float = 1.0e-5
    alpha_pd: float = 0.0                    # direct P_D reward weight (0=utility-only, 0.5=hybrid)
    lambda_tail: float = 0.0                 # bottom-3 bonus weight
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lagrangian_lr: float = 0.002     # lowered: binary any_violation made λ climb monotonically -> non-stationary returns -> critic blow-up
    max_violation_rate: float = 0.1  # Lagrangian target: max fraction of steps with constraint violations
    lagrangian_max: float = 1.0      # lowered cap so λ can't dominate the reward
    target_kl: float = 0.02          # 0.03 -> 0.02: tighter per-rollout policy drift bound
    num_episodes: int = 1000         # max episodes (hard cap; early-stop usually stops sooner)
    num_envs: int = 8  # parallel envs for GPU batching (tuned; stable MAPPO)
    assignment_hold_frames: int = 1  # 1=every frame; 5=hold P0 for 5 frames
    actor_decision_interval: int = 1  # 1=per-frame; 5=macro-action (hold action 5 frames)
    obs_history_frames: int = 1  # 1=current only; 2=stack prev+current frame obs
    oracle_obs: bool = False  # diagnostic: feed true target pos (not beliefs) to actor
    rel_features: bool = True  # explicit per-target (dx,dy,dist,bearing) in actor obs
    structured_actor: bool = True  # entity-attention actor (vs flat MLP)
    centralized_actor: bool = False  # diagnostic: actor sees global state (upper bound)
    critic_lr_mult: float = 5.0  # critic LR = lr * this (critic needs to track moving returns)
    bc_beta_init: float = 0.05  # BC anchor strength; sweet spot: prevents collapse, allows improvement
    use_p0_sinr_gated: bool = False  # gate P0 features by SINR threshold
    freeze_actor_after: int = 0       # freeze Actor at this episode (0=disabled)
    # Role assignment. learn_roles=False (default) DROPS the policy role head from
    # the objective and lets the inner P0 solver assign tx/rx: deflection enumerates
    # all i!=j ordered pairs (any UAV may tx or rx), P0 selects (i,j,q) with a
    # one-role-per-UAV constraint, and roles are derived from the selection. This
    # removes the degenerate all-same-role argmax collapse (diagnosed: deterministic
    # role argmax -> 95% frames single role -> zero tx-rx pairing -> P_D collapse).
    # learn_roles=True restores the original learned-role behavior (for comparison).
    learn_roles: bool = False
    # P0 information source (B6). False (default) = ORACLE inner scheduler: P0 ranks
    # candidates on TRUE target geometry (upper bound). True = DEPLOYABLE: P0 ranks on
    # the fused belief estimate, while the realized deflection/P_D of the selected
    # pairs still uses TRUE geometry (the physical echo). See docs/KNOWN_ISSUES.md B6.
    p0_uses_belief: bool = False
    # Belief–P_D coupling (B7). False (default) = optimistic (selected pair always
    # updates belief). True = Bernoulli detection gating: target q's belief updates
    # only if a detection event delta_q ~ Bernoulli(P_D_q) fires; else predict-only
    # and AoI keeps growing. See docs/KNOWN_ISSUES.md B7.
    belief_detection_sampling: bool = False
    # Fixed evaluation scenarios (reused every eval + across decode modes).
    eval_seeds: List[int] = field(default_factory=lambda: [10001, 10002, 10003, 10004, 10005])
    # Convergence-based early stopping (deterministic eval on a plateau)
    early_stop: bool = True
    eval_interval: int = 50          # run a deterministic eval every N episodes
    eval_episodes: int = 3           # deterministic eval episodes to average
    early_stop_patience: int = 12    # stop after this many evals with no improvement
    early_stop_min_delta: float = 0.005  # min steady_P_D gain counted as improvement
    # CTDE centralized critic (MAPPO, critic sees global state) vs decentralized
    # critic (IPPO, critic sees only local obs). de Witt et al. 2020 / Yu et al. 2022.
    centralized_critic: bool = True
    # Communication mode. 'off' = zero comm messages, freeze comm-related heads,
    # no comm loss / intent loss. Used for Full/EH isolation experiments.
    # 'on' (default) = learned communication.
    learned_comm_mode: str = 'on'
    # Per-module LR: encoder=1e-5, attention=1e-5, head=5e-5 (Full).
    # When freeze_attention=True: attention LR→0. False = single LR for all.
    use_per_module_lr: bool = False
    # Neighbor belief fusion via multi-head attention + CI.
    neighbor_belief_fusion: bool = False
    # Freeze attention (attn.* + attn_norm.*) — EH mode.
    # Only meaningful when use_per_module_lr=True.
    freeze_attention: bool = False
    # Advantage mode: 'scalar' (default) or 'target_wise' (S4).
    # target_wise: per-target advantages aggregated via UAV-target
    # responsibility weights (inverse-distance softmax, detached).
    advantage_mode: str = 'scalar'
    # Temperature for inverse-distance target responsibility (meters).
    # Only used when advantage_mode='target_wise'.
    target_responsibility_tau_m: float = 50.0


@dataclass
class MasterConfig:
    scenario: ScenarioParams = field(default_factory=ScenarioParams)
    uav: UAVParams = field(default_factory=UAVParams)
    target: TargetParams = field(default_factory=TargetParams)
    otfs: OTFSParams = field(default_factory=OTFSParams)
    channel: ChannelParams = field(default_factory=ChannelParams)
    detection: DetectionParams = field(default_factory=DetectionParams)
    p0_solver: P0SolverParams = field(default_factory=P0SolverParams)
    marl: MARLParams = field(default_factory=MARLParams)
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    def to_small_config(self) -> "MasterConfig":
        """Return a reduced config for fast smoke tests."""
        small = MasterConfig()
        small.scenario.K = 2
        small.scenario.Q = 1
        small.scenario.T = 10
        small.marl.hidden_layers = [32, 32]
        small.marl.rollout_steps = 64
        small.marl.minibatch_size = 16
        small.marl.num_episodes = 10
        small.marl.ppo_epochs = 2
        return small


def _dict_to_dataclass(cls, d: dict):
    """Recursively convert dict to dataclass instance."""
    import dataclasses
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in d.items():
        if key in field_types:
            ft = field_types[key]
            if dataclasses.is_dataclass(ft) and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(ft, value)
            elif hasattr(ft, '__origin__') and ft.__origin__ in (list, List):
                kwargs[key] = value
            elif hasattr(ft, '__origin__') and ft.__origin__ in (tuple, Tuple):
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str) -> MasterConfig:
    """Load configuration from YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return _dict_to_dataclass(MasterConfig, raw)


def get_default_config() -> MasterConfig:
    """Load the default config from the package."""
    default_path = os.path.join(os.path.dirname(__file__), 'default.yaml')
    return load_config(default_path)
