"""Neural network architectures for actor and critic.

Actor: shared MLP [256, 256] → heads: dp_mean(2), dp_log_std(2), role_logits(3)
  (shared = obs → 256 → 256, then head: 256 → output)
Critic: MLP [256, 256] → scalar value + per-target value heads
"""

import torch
import torch.nn as nn
import numpy as np
import itertools
import threading
from typing import Optional, Tuple


def sinkhorn_normalize(
    logits: torch.Tensor,
    iterations: int = 48,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Differentiable approximately doubly-stochastic team assignment."""
    if logits.ndim != 3:
        raise ValueError('Sinkhorn logits must have shape (batch, K, Q)')
    log_p = logits / max(float(temperature), 1e-4)
    for _ in range(max(int(iterations), 1)):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return torch.exp(log_p)


def exact_permutation_assignment(
    logits: torch.Tensor,
    temperature: float = 0.35,
    max_exact_agents: int = 6,
) -> torch.Tensor:
    """Exact decentralized one-to-one projection with soft gradients.

    If every UAV evaluates this function on the same globally indexed U2U
    claim matrix, every UAV obtains the same permutation without a coordinator.
    The forward value is hard; a Gibbs distribution over feasible permutations
    supplies a straight-through gradient during training.
    """
    if logits.ndim != 3:
        raise ValueError('assignment logits must have shape (batch, K, Q)')
    _, num_agents, num_targets = logits.shape
    if num_agents != num_targets or num_agents > max_exact_agents:
        return sinkhorn_normalize(logits, temperature=temperature)

    permutations = torch.tensor(
        list(itertools.permutations(range(num_targets))),
        dtype=torch.long,
        device=logits.device,
    )
    rows = torch.arange(num_agents, device=logits.device)
    permutation_scores = logits[:, rows, permutations].sum(dim=-1)
    soft_weights = torch.softmax(
        permutation_scores / max(float(temperature), 1e-4), dim=-1)
    permutation_matrices = torch.nn.functional.one_hot(
        permutations, num_classes=num_targets).to(logits.dtype)
    soft_assignment = torch.einsum(
        'bp,pkq->bkq', soft_weights, permutation_matrices)
    hard_assignment = permutation_matrices[
        permutation_scores.argmax(dim=-1)]
    return hard_assignment + soft_assignment - soft_assignment.detach()


def _solve_capacity_duals_bisection(
    scaled: torch.Tensor,
    row_capacity: float,
    column_capacity: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the bounded transport dual with the canonical fixed iteration."""
    row_dual = torch.zeros_like(scaled[:, :, :1])
    column_dual = torch.zeros_like(scaled[:, :1, :])

    def solve_dual(
        fixed: torch.Tensor,
        target: float,
        reduce_dim: int,
        template: torch.Tensor,
    ) -> torch.Tensor:
        # Monotone bisection is deliberately used instead of an unconstrained
        # Newton update: sparse top-k claims can saturate sigmoid derivatives
        # and make a Newton step jump to the wrong boundary.
        lower = torch.full_like(template, -60.0)
        upper = torch.full_like(template, 60.0)
        for _ in range(16):
            midpoint = 0.5 * (lower + upper)
            load = torch.sigmoid(fixed + midpoint).sum(
                dim=reduce_dim, keepdim=True)
            below = load < target
            lower = torch.where(below, midpoint, lower)
            upper = torch.where(below, upper, midpoint)
        return 0.5 * (lower + upper)

    for _ in range(max(int(iterations), 1)):
        row_dual = solve_dual(
            scaled + column_dual,
            row_capacity,
            -1,
            row_dual,
        )
        column_dual = solve_dual(
            scaled + row_dual,
            column_capacity,
            -2,
            column_dual,
        )
    return row_dual, column_dual


class _CapacityDualCudaGraph:
    """Static-buffer CUDA graph for one small dual-projection shape."""

    def __init__(
        self,
        example: torch.Tensor,
        row_capacity: float,
        column_capacity: float,
        iterations: int,
    ) -> None:
        self._input = torch.empty_like(example)
        self._graph = torch.cuda.CUDAGraph()
        self._lock = threading.Lock()
        warm_stream = torch.cuda.Stream(device=example.device)
        current_stream = torch.cuda.current_stream(example.device)
        warm_stream.wait_stream(current_stream)
        with torch.cuda.stream(warm_stream):
            for _ in range(2):
                _solve_capacity_duals_bisection(
                    self._input,
                    row_capacity,
                    column_capacity,
                    iterations,
                )
        current_stream.wait_stream(warm_stream)
        with torch.cuda.graph(self._graph):
            self._row_dual, self._column_dual = (
                _solve_capacity_duals_bisection(
                    self._input,
                    row_capacity,
                    column_capacity,
                    iterations,
                )
            )

    def solve(
        self,
        scaled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Clone before returning so a later replay cannot overwrite constants
        # saved by autograd for the current forward pass. Commands for a given
        # stream are enqueued under one lock and therefore retain this order.
        with self._lock:
            self._input.copy_(scaled)
            self._graph.replay()
            return self._row_dual.clone(), self._column_dual.clone()


_CAPACITY_DUAL_GRAPH_CACHE: dict[tuple, _CapacityDualCudaGraph | None] = {}
_CAPACITY_DUAL_GRAPH_CACHE_LOCK = threading.Lock()
_CAPACITY_DUAL_GRAPH_CACHE_LIMIT = 16


def _cuda_graph_capacity_duals(
    scaled: torch.Tensor,
    row_capacity: float,
    column_capacity: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return graph-replayed duals, or ``None`` for a fail-safe fallback."""
    stream = torch.cuda.current_stream(scaled.device)
    key = (
        int(scaled.device.index or 0),
        str(scaled.dtype),
        tuple(int(value) for value in scaled.shape),
        float(row_capacity),
        float(column_capacity),
        int(iterations),
        int(stream.cuda_stream),
    )
    with _CAPACITY_DUAL_GRAPH_CACHE_LOCK:
        if key not in _CAPACITY_DUAL_GRAPH_CACHE:
            if len(_CAPACITY_DUAL_GRAPH_CACHE) >= _CAPACITY_DUAL_GRAPH_CACHE_LIMIT:
                return None
            try:
                _CAPACITY_DUAL_GRAPH_CACHE[key] = _CapacityDualCudaGraph(
                    scaled,
                    row_capacity,
                    column_capacity,
                    iterations,
                )
            except RuntimeError:
                # CUDA graph support depends on the surrounding execution
                # context. Remember failure and preserve the original path.
                _CAPACITY_DUAL_GRAPH_CACHE[key] = None
        graph = _CAPACITY_DUAL_GRAPH_CACHE[key]
    return None if graph is None else graph.solve(scaled)


def capacity_sinkhorn_normalize(
    logits: torch.Tensor,
    row_capacity: float,
    column_capacity: float,
    iterations: int = 32,
    temperature: float = 0.35,
    use_cuda_graph: Optional[bool] = None,
) -> torch.Tensor:
    """Project team bids onto a soft capacitated bipartite matching.

    Unlike a doubly-stochastic one-to-one Sinkhorn matrix, bistatic sensing
    needs more than one endpoint per target.  This alternating KL projection
    preserves the requested row/column loads while keeping entries in [0, 1].
    The total requested capacity must agree on both sides.
    """
    if logits.ndim != 3:
        raise ValueError('capacity logits must have shape (batch, K, Q)')
    _, num_agents, num_targets = logits.shape
    row_capacity = float(row_capacity)
    column_capacity = float(column_capacity)
    if row_capacity <= 0.0 or column_capacity <= 0.0:
        raise ValueError('matching capacities must be positive')
    total_rows = row_capacity * num_agents
    total_columns = column_capacity * num_targets
    if not np.isclose(total_rows, total_columns, rtol=1e-5, atol=1e-5):
        raise ValueError(
            'row and column matching capacities must have equal total load')

    # The box-constrained entropic transport optimum has the logistic form
    #   X_ij = sigmoid(logit_ij / T + u_i + v_j).
    # Alternating Newton updates of its row/column dual variables enforce both
    # marginals without the post-hoc clipping that breaks capacity conservation.
    # This is the bounded analogue of Sinkhorn scaling for binary endpoint
    # reservations; every edge remains differentiable and lies strictly in
    # (0, 1).
    scaled = torch.clamp(
        logits / max(float(temperature), 1e-4), -30.0, 30.0)
    # The dual variables only enforce feasibility; treating their numerical
    # solve as a stop-gradient operation avoids backpropagating through hundreds
    # of bisection kernels.  The final logistic edge probabilities still carry
    # direct gradients to every bid logit (a straight-through dual projection).
    with torch.no_grad():
        detached_scaled = scaled.detach()
        graph_allowed = bool(
            scaled.is_cuda
            and scaled.dtype == torch.float32
            and scaled.numel() <= 4096
            and not torch.cuda.is_current_stream_capturing()
            and use_cuda_graph is not False
        )
        graph_duals = (
            _cuda_graph_capacity_duals(
                detached_scaled,
                row_capacity,
                column_capacity,
                iterations,
            )
            if graph_allowed else None
        )
        if graph_duals is None:
            row_dual, column_dual = _solve_capacity_duals_bisection(
                detached_scaled,
                row_capacity,
                column_capacity,
                iterations,
            )
        else:
            row_dual, column_dual = graph_duals
    return torch.sigmoid(scaled + row_dual + column_dual)


def apply_semantic_capacity_bid_correction(
    neighbor_logits: torch.Tensor,
    semantic_evidence: torch.Tensor,
    semantic_pd: torch.Tensor,
    valid: torch.Tensor,
    gain: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine sparse peer bids with decoded quality before projection.

    The correction is row-centred: it can change which of one sender's target
    claims survives a capacity conflict, but cannot inflate that sender's
    aggregate authority. Invalid or silent token edges remain untouched. This
    path therefore affects sensing endpoints without directly changing motion
    or the per-UAV communication/sensing power split.
    """
    if neighbor_logits.ndim != 3:
        raise ValueError('neighbor logits must have shape (batch, peers, Q)')
    expected_shape = neighbor_logits.shape
    for name, value in (
        ('semantic evidence', semantic_evidence),
        ('semantic P_D', semantic_pd),
        ('valid mask', valid),
    ):
        if value.shape != expected_shape:
            raise ValueError(f'{name} must match neighbor logits')
    valid_bool = valid.to(torch.bool)
    valid_float = valid_bool.to(neighbor_logits.dtype)
    quality = (
        semantic_evidence.to(neighbor_logits.dtype)
        * semantic_pd.to(neighbor_logits.dtype)
        * valid_float
    )
    valid_count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    row_mean = quality.sum(dim=-1, keepdim=True) / valid_count
    correction = (quality - row_mean) * valid_float
    corrected = neighbor_logits + max(float(gain), 0.0) * correction
    return corrected, correction


def mlp(input_dim: int, hidden_dims: list, output_dim: int,
        activation=nn.ReLU, output_activation=None) -> nn.Sequential:
    """Build an MLP with configurable hidden layers."""
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(activation())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ActorNetwork(nn.Module):
    """MAPPO actor network.

    Input: local observation (obs_dim,)
    Output:
      - dp_mean: (batch, 2) latent Gaussian mean (tanh-squashed in distribution)
      - dp_log_std: (2,) learnable log-std (broadcast over batch)
      - role_logits: (batch, 3) logits for tx/rx/idle
    """

    def __init__(self, obs_dim: int, hidden_layers: list = [256, 256],
                 max_dp: float = 2.5, comm_num_rate_levels: int = 4,
                 comm_log_std_init: float = -1.0, num_targets: int = 4,
                 comm_payload_dim: int = 16,
                 isac_power_log_std_init: float = -1.0,
                 sensing_allocation_log_std_init: float = -1.0):
        super().__init__()
        self.max_dp = max_dp
        self.comm_payload_dim = max(1, int(comm_payload_dim))

        # Shared feature extractor: 2 hidden ReLU layers (obs→256→256→256).
        # FIX: previously hidden_layers[:-1] dropped the 2nd hidden layer
        # (only 1 effective ReLU). Use full hidden_layers so depth matches docs.
        self.shared = mlp(obs_dim, hidden_layers, hidden_layers[-1])

        # Heads
        self.dp_mean_head = nn.Linear(hidden_layers[-1], 2)
        self.comm_head = nn.Linear(hidden_layers[-1], self.comm_payload_dim)
        self.comm_rate_head = nn.Linear(
            self.comm_payload_dim, comm_num_rate_levels)
        self.comm_log_std = nn.Parameter(
            torch.full((self.comm_payload_dim,), float(comm_log_std_init)))
        self.isac_power_mean_head = nn.Linear(self.comm_payload_dim, 1)
        self.isac_sensing_mean_head = nn.Linear(
            self.comm_payload_dim, int(num_targets))
        self.isac_power_log_std = nn.Parameter(torch.tensor(
            [float(isac_power_log_std_init)]))
        self.isac_sensing_log_std = nn.Parameter(torch.full(
            (int(num_targets),), float(sensing_allocation_log_std_init)))
        self.pd_aux_head = nn.Linear(hidden_layers[-1], 1) # auxiliary P_D predictor
        # init 0: with range (-1,1), tanh(0)=0 -> log_std=0 -> sigma=1 (matches the
        # high-entropy regime that learned early).
        self.dp_log_std = nn.Parameter(torch.zeros(2))  # learnable, shared
        self.role_head = nn.Linear(hidden_layers[-1], 3)

        # Initialize weights
        self._init_weights()
        nn.init.zeros_(self.comm_rate_head.weight)
        nn.init.zeros_(self.comm_rate_head.bias)
        nn.init.zeros_(self.isac_power_mean_head.weight)
        nn.init.zeros_(self.isac_power_mean_head.bias)
        nn.init.zeros_(self.isac_sensing_mean_head.weight)
        nn.init.zeros_(self.isac_sensing_mean_head.bias)

    def _init_weights(self):
        """PPO-style init: hidden layers sqrt(2), output heads small."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'head' in name or name.endswith('role_head'):
                    # Policy output: small weights for stable initial exploration
                    nn.init.orthogonal_(module.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor, h_prev: torch.Tensor = None,
                detach_h_new: bool = True, window_mask: torch.Tensor = None,
                comm_round_phase: torch.Tensor = None,
                agent_identity: torch.Tensor = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (batch, obs_dim)

        Returns:
            dp_mean: (batch, 2) latent Gaussian mean (unbounded; tanh-squash in distribution)
            dp_log_std: (2,) learnable log-std parameter (broadcast over batch)
            role_logits: (batch, 3) logits for tx/rx/idle
        """
        h = self.shared(obs)
        dp_mean = self.dp_mean_head(h)  # no tanh — handled by tanh-squash distribution
        role_logits = self.role_head(h)
        comm_msg = torch.tanh(self.comm_head(h))  # no saturation
        pd_pred = torch.sigmoid(self.pd_aux_head(h))  # aux P_D prediction (0~1)
        LOG_STD_MIN, LOG_STD_MAX = -1.0, 1.0
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(self.dp_log_std) + 1.0)
        return dp_mean, log_std, role_logits, comm_msg, pd_pred, None

    def communication_parameters(self, comm_mean: torch.Tensor):
        """Distribution parameters for the stochastic communication action."""
        comm_log_std = torch.clamp(self.comm_log_std, -4.0, 1.0)
        rate_logits = self.comm_rate_head(comm_mean)
        return comm_log_std, rate_logits

    def isac_resource_parameters(self, comm_mean: torch.Tensor):
        """Logistic-normal parameters for power and multi-target sensing."""
        power_mean = self.isac_power_mean_head(comm_mean).squeeze(-1)
        sensing_mean = self.isac_sensing_mean_head(comm_mean)
        power_log_std = torch.clamp(self.isac_power_log_std, -4.0, 1.0)
        sensing_log_std = torch.clamp(
            self.isac_sensing_log_std, -4.0, 1.0)
        return power_mean, power_log_std, sensing_mean, sensing_log_std


class StructuredActorNetwork(nn.Module):
    """Entity-based actor with cross-attention and gated communication.

    Instead of a flat MLP over 248-dim obs, this encodes self, targets, and
    neighbors as SEPARATE entities, pools them via cross-attention, and gates
    the communication channel. This:
      - eliminates permutation sensitivity (targets/neighbors are sets)
      - lets attention automatically ignore distant/irrelevant entities
      - protects the BC-learned physical policy from comm noise via a gate

    P1 FIX: PD_hist (previous-frame per-target detection probability) is now
    projected and added to the target entity encoding, so the Actor can
    directly condition on which targets had low/high detection.

    NOTE: PD_hist in the local observation IS the UAV's own LOCAL detection
    confidence — P_D computed from deflection of bistatic pairs where THIS UAV
    is tx or rx (not global fused P_D). Each UAV sees different PD_hist values.
    This respects the decentralized information boundary: no free global
    fusion-centre broadcast. See env_core.py prev_P_D_local (2026-07-14 fix).
    """

    def __init__(self, obs_dim: int, K: int = 8, Q: int = 8,
                 entity_dim: int = 128, max_dp: float = 2.5,
                 single_frame_dim: int = 0,
                 use_corrected_parser: bool = False,
                 use_p0: bool = False,
                 comm_num_rate_levels: int = 4,
                 comm_log_std_init: float = -1.0,
                 use_comm_cross_attention: bool = False,
                 comm_token_dim: int = 21,
                 comm_tokens_per_sender: int = 1,
                 comm_payload_dim: int = 16,
                 comm_target_token_enabled: bool = False,
                 comm_target_token_dim: int = 16,
                 use_target_allocation: bool = False,
                 use_team_sinkhorn: bool = False,
                 capacity_matching_enabled: bool = False,
                 capacity_matching_row_capacity: int = 2,
                 capacity_matching_column_capacity: int = 2,
                 capacity_matching_temperature: float = 0.35,
                 capacity_matching_iterations: int = 32,
                 capacity_matching_blend: float = 0.0,
                 target_allocation_temperature: float = 1.0,
                 target_allocation_straight_through: bool = False,
                 target_allocation_movement_blend: float = 0.0,
                 target_allocation_movement_confidence_gating_enabled: bool = False,
                 target_allocation_movement_confidence_floor: float = 0.0,
                 target_allocation_movement_confidence_power: float = 2.0,
                 hierarchical_dual_assignment_enabled: bool = False,
                 movement_team_matching_enabled: bool = False,
                 movement_team_matching_temperature: float = 0.35,
                 movement_team_matching_iterations: int = 16,
                 movement_team_matching_blend: float = 0.0,
                 movement_team_matching_intrinsic_bid_mix: float = 0.0,
                 target_allocation_resource_blend: float = 0.0,
                 round_negotiation_enabled: bool = False,
                 round_negotiation_strength: float = 0.5,
                 round_negotiation_temperature: float = 0.5,
                 sparse_claim_enabled: bool = False,
                 sparse_claim_share_topk: int = 2,
                 sparse_claim_desired_endpoints: int = 2,
                 sparse_claim_full_penalty: float = 2.0,
                 sparse_claim_vacant_bonus: float = 0.5,
                 sparse_claim_temperature: float = 0.25,
                 comm_aided_sensing_enabled: bool = False,
                 comm_aided_sensing_blend: float = 1.0,
                 comm_semantic_decoder_enabled: bool = False,
                 comm_semantic_capacity_bid_enabled: bool = False,
                 comm_semantic_capacity_bid_gain: float = 0.0,
                 comm_semantic_extra_token_enabled: bool = False,
                 comm_semantic_extra_token_threshold: float = 0.10,
                 semantic_kinematic_field_enabled: bool = False,
                 semantic_kinematic_field_gain: float = 0.15,
                 target_conditioned_movement_enabled: bool = False,
                 target_conditioned_movement_gain: float = 0.15,
                 architecture_v2_enabled: bool = False,
                 architecture_v2_prior_gain: float = 1.0,
                 architecture_v2_distance_weight: float = 0.25,
                 architecture_v2_qos_floor: float = 0.60,
                 architecture_v2_comm_prior_gain: float = 2.0,
                 architecture_v2_comm_crisis_threshold: float = 0.25,
                 architecture_v2_consensus_enabled: bool = True,
                 architecture_v2_matching_temperature: float = 0.35,
                 architecture_v2_movement_consensus_blend: float = 1.0,
                 architecture_v2_endpoint_consensus_gain: float = 2.0,
                 architecture_v2_bid_residual_scale: float = 0.25,
                 architecture_v2_sensing_aligned_claims_enabled: bool = False,
                 architecture_v2_modular_coordination_enabled: bool = False,
                 architecture_v2_modular_num_experts: int = 3,
                 architecture_v2_modular_gain: float = 0.25,
                 architecture_v2_modular_temperature: float = 0.75,
                 scale_equivariant_comm_heads_enabled: bool = False,
                 permutation_equivariant_round_encoding_enabled: bool = False,
                 comm_channel_feedback_rate_enabled: bool = False,
                 comm_channel_feedback_dim: int = 6,
                 isac_power_log_std_init: float = -1.0,
                 sensing_allocation_log_std_init: float = -1.0):
        super().__init__()
        self.K, self.Q = K, Q
        self.max_dp = max_dp
        D = entity_dim
        self.single_frame_dim = single_frame_dim
        # Audit 2026-08-17 (P0): the historical hand parser `_parse_one` reads
        # per-target INTERLEAVED belief(9)+geometry(8), but ObservationBuilder
        # emits BLOCK-wise fields (all Q beliefs, then all Q geometry) -- this
        # has been the layout since the initial commit.  With the legacy parser
        # every target's "geometry" was read from the NEXT target's belief block
        # (Q>=2), silently corrupting the target-conditioned policy inputs in
        # configs without cross-attention (e.g. the 4/4 frozen deployment).
        # The corrected slice-descriptor parser is authoritative; the legacy
        # path is now unreachable dead code.
        self._use_corrected_parser = True
        self._use_p0 = use_p0
        self._use_comm_cross_attention = bool(use_comm_cross_attention)
        self._use_target_allocation = bool(use_target_allocation)
        self._use_team_sinkhorn = bool(
            use_team_sinkhorn and use_target_allocation)
        self._capacity_matching_enabled = bool(
            capacity_matching_enabled and use_target_allocation
            and comm_target_token_enabled and use_comm_cross_attention)
        self._capacity_matching_row_capacity = max(
            1, int(capacity_matching_row_capacity))
        self._capacity_matching_column_capacity = max(
            1, int(capacity_matching_column_capacity))
        self._capacity_matching_temperature = max(
            1e-3, float(capacity_matching_temperature))
        self._capacity_matching_iterations = max(
            1, int(capacity_matching_iterations))
        self._capacity_matching_blend = float(np.clip(
            capacity_matching_blend, 0.0, 1.0))
        if self._capacity_matching_enabled:
            row_total = self.K * self._capacity_matching_row_capacity
            column_total = self.Q * self._capacity_matching_column_capacity
            if row_total != column_total:
                raise ValueError(
                    'capacity matching requires K*row_capacity == '
                    'Q*column_capacity')
        self._target_allocation_temperature = max(
            float(target_allocation_temperature), 1e-3)
        self._target_allocation_straight_through = bool(
            target_allocation_straight_through)
        self._target_allocation_movement_blend = float(np.clip(
            target_allocation_movement_blend, 0.0, 1.0))
        self._target_allocation_movement_confidence_gating_enabled = bool(
            target_allocation_movement_confidence_gating_enabled)
        self._target_allocation_movement_confidence_floor = float(np.clip(
            target_allocation_movement_confidence_floor, 0.0, 1.0))
        self._target_allocation_movement_confidence_power = max(
            0.0, float(target_allocation_movement_confidence_power))
        self._hierarchical_dual_assignment_enabled = bool(
            hierarchical_dual_assignment_enabled
            and self._capacity_matching_enabled)
        self._movement_team_matching_enabled = bool(
            movement_team_matching_enabled and use_target_allocation
            and comm_target_token_enabled and use_comm_cross_attention)
        self._movement_team_matching_temperature = max(
            1e-3, float(movement_team_matching_temperature))
        self._movement_team_matching_iterations = max(
            1, int(movement_team_matching_iterations))
        self._movement_team_matching_blend = float(np.clip(
            movement_team_matching_blend, 0.0, 1.0))
        self._movement_team_matching_intrinsic_bid_mix = float(np.clip(
            movement_team_matching_intrinsic_bid_mix, 0.0, 1.0))
        self._target_allocation_resource_blend = max(
            0.0, float(target_allocation_resource_blend))
        self._round_negotiation_enabled = bool(
            round_negotiation_enabled and use_target_allocation
            and comm_target_token_enabled)
        self._round_negotiation_strength = max(
            0.0, float(round_negotiation_strength))
        self._round_negotiation_temperature = max(
            1e-3, float(round_negotiation_temperature))
        self._sparse_claim_enabled = bool(
            sparse_claim_enabled and use_target_allocation
            and comm_target_token_enabled)
        self._sparse_claim_share_topk = int(np.clip(
            sparse_claim_share_topk, 1, Q))
        self._sparse_claim_desired_endpoints = max(
            1, int(sparse_claim_desired_endpoints))
        self._sparse_claim_full_penalty = max(
            0.0, float(sparse_claim_full_penalty))
        self._sparse_claim_vacant_bonus = max(
            0.0, float(sparse_claim_vacant_bonus))
        self._sparse_claim_temperature = max(
            1e-3, float(sparse_claim_temperature))
        self._comm_aided_sensing_enabled = bool(
            comm_aided_sensing_enabled and use_comm_cross_attention)
        self._comm_aided_sensing_blend = max(
            0.0, float(comm_aided_sensing_blend))
        self._comm_semantic_decoder_enabled = bool(
            comm_semantic_decoder_enabled
            and use_comm_cross_attention
            and comm_target_token_enabled
            and int(comm_target_token_dim) >= 2)
        self._comm_semantic_capacity_bid_enabled = bool(
            comm_semantic_capacity_bid_enabled
            and self._comm_semantic_decoder_enabled
            and self._capacity_matching_enabled)
        self._comm_semantic_capacity_bid_gain = max(
            0.0, float(comm_semantic_capacity_bid_gain))
        self._comm_semantic_extra_token_enabled = bool(
            comm_semantic_extra_token_enabled
            and self._comm_semantic_decoder_enabled
            and self._sparse_claim_enabled)
        self._comm_semantic_extra_token_threshold = max(
            0.0, float(comm_semantic_extra_token_threshold))
        self._semantic_kinematic_field_enabled = bool(
            semantic_kinematic_field_enabled
            and self._sparse_claim_enabled)
        self._semantic_kinematic_field_gain = max(
            0.0, float(semantic_kinematic_field_gain))
        self._target_conditioned_movement_enabled = bool(
            target_conditioned_movement_enabled and use_target_allocation)
        self._target_conditioned_movement_gain = max(
            0.0, float(target_conditioned_movement_gain))
        self._architecture_v2_enabled = bool(architecture_v2_enabled)
        self._architecture_v2_prior_gain = max(
            0.0, float(architecture_v2_prior_gain))
        self._architecture_v2_distance_weight = max(
            0.0, float(architecture_v2_distance_weight))
        self._architecture_v2_qos_floor = float(np.clip(
            architecture_v2_qos_floor, 1e-3, 1.0))
        self._architecture_v2_comm_prior_gain = max(
            0.0, float(architecture_v2_comm_prior_gain))
        self._architecture_v2_comm_crisis_threshold = float(np.clip(
            architecture_v2_comm_crisis_threshold, 0.0, 1.0))
        self._architecture_v2_consensus_enabled = bool(
            architecture_v2_enabled and architecture_v2_consensus_enabled)
        self._architecture_v2_matching_temperature = max(
            1e-3, float(architecture_v2_matching_temperature))
        self._architecture_v2_movement_consensus_blend = float(np.clip(
            architecture_v2_movement_consensus_blend, 0.0, 1.0))
        self._architecture_v2_endpoint_consensus_gain = max(
            0.0, float(architecture_v2_endpoint_consensus_gain))
        self._architecture_v2_bid_residual_scale = max(
            0.0, float(architecture_v2_bid_residual_scale))
        self._architecture_v2_sensing_aligned_claims_enabled = bool(
            architecture_v2_enabled
            and architecture_v2_sensing_aligned_claims_enabled)
        self._architecture_v2_modular_coordination_enabled = bool(
            architecture_v2_enabled
            and architecture_v2_modular_coordination_enabled)
        self._architecture_v2_modular_num_experts = max(
            2, int(architecture_v2_modular_num_experts))
        self._architecture_v2_modular_gain = max(
            0.0, float(architecture_v2_modular_gain))
        self._architecture_v2_modular_temperature = max(
            1e-3, float(architecture_v2_modular_temperature))
        if self._architecture_v2_enabled and not (
                use_target_allocation
                and comm_target_token_enabled
                and use_comm_cross_attention):
            raise ValueError(
                'architecture_v2_enabled requires target allocation, '
                'per-target token communication and cross-attention')
        if (self._semantic_kinematic_field_enabled
                and self._target_conditioned_movement_enabled):
            raise ValueError(
                'semantic field and target-conditioned movement are mutually exclusive')
        self.comm_token_dim = int(comm_token_dim)
        self.comm_tokens_per_sender = max(1, int(comm_tokens_per_sender))
        self.comm_payload_dim = max(1, int(comm_payload_dim))
        self._comm_channel_feedback_rate_enabled = bool(
            comm_channel_feedback_rate_enabled)
        self.comm_channel_feedback_dim = max(
            1, int(comm_channel_feedback_dim))
        if (self._comm_channel_feedback_rate_enabled
                and self.comm_channel_feedback_dim != 6):
            raise ValueError(
                'comm_channel_feedback_dim must be 6 for the current '
                'local channel summary')
        self._comm_target_token_enabled = bool(comm_target_token_enabled)
        self.comm_target_token_dim = max(1, int(comm_target_token_dim))
        self._scale_equivariant_comm_heads_enabled = bool(
            (scale_equivariant_comm_heads_enabled
             or self._architecture_v2_enabled)
            and self._comm_target_token_enabled)
        self._permutation_equivariant_round_encoding_enabled = bool(
            (permutation_equivariant_round_encoding_enabled
             or self._architecture_v2_enabled)
            and self._round_negotiation_enabled)
        self.last_comm_attention = None
        self.last_target_assignment = None
        self.last_movement_assignment = None
        self.last_target_assignment_st = None
        self.last_movement_confidence = None
        self.last_effective_movement_blend = None
        self.last_capacity_assignment = None
        self.last_movement_team_assignment = None
        self.last_round_peer_claims = None
        self.last_peer_claim_load = None
        self.last_outgoing_token_mask = None
        self.last_sparse_claim_scores = None
        # Read-only diagnostic used by offline communication-value probes.
        # It is never fed back into the actor and therefore cannot change the
        # deployed policy or checkpoint compatibility.
        self.last_policy_latent = None
        self.last_comm_sensing_logits = None
        self.last_comm_semantic_evidence = None
        self.last_comm_semantic_pd = None
        self.last_comm_semantic_capacity_bias = None
        self.last_comm_semantic_extra_token_mask = None
        self.last_v2_target_latent = None
        self.last_v2_sensing_logits = None
        self.last_v2_movement_gate = None
        self.last_v2_physics_prior = None
        self.last_v2_comm_crisis = None
        self.last_v2_local_bids = None
        self.last_v2_local_bid_logits = None
        self.last_v2_peer_bids = None
        self.last_v2_bid_agreement = None
        self.last_v2_movement_consensus = None
        self.last_v2_endpoint_consensus = None
        self.last_v2_consensus_available = None
        self.last_v2_module_routing = None
        self.last_v2_module_residual_norm = None
        self.last_semantic_field_vector = None
        self.last_semantic_field_weights = None
        self.last_semantic_field_delta = None
        self.last_target_movement_candidates = None
        self.last_target_movement_residual = None
        self.last_target_movement_delta = None
        self.last_target_movement_candidates = None
        self.last_target_movement_residual = None
        self.last_target_movement_delta = None
        self.last_comm_channel_feedback = None
        if self._use_corrected_parser:
            from uav_isac.environment.observation_slices import ObservationSlices
            self._obs_slices = ObservationSlices.from_config(
                K=K, Q=Q, use_p0=use_p0, use_rel_features=True,
                use_comm_tokens=self._use_comm_cross_attention,
                comm_token_dim=self.comm_token_dim,
                comm_tokens_per_sender=self.comm_tokens_per_sender,
                use_channel_feedback=(
                    self._comm_channel_feedback_rate_enabled),
                channel_feedback_dim=self.comm_channel_feedback_dim)

        # ── Entity encoders ──
        self.self_enc = nn.Sequential(
            nn.Linear(11, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D),
        )
        self.target_enc = nn.Sequential(
            nn.Linear(18, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D),
        )
        # PD_hist projector: maps scalar P_D (per target) → entity-dim signal
        # that modulates the target encoding, so Actor knows which targets
        # had low detection in the previous frame.
        self.pd_hist_proj = nn.Linear(1, D)

        # GRU for neighbor temporal encoding (handles window=1 gracefully)
        self.neighbor_gru = nn.GRU(input_size=9, hidden_size=D, batch_first=True)
        self.neighbor_proj = nn.Linear(D, D)  # project GRU output
        self.global_enc = nn.Linear(2, D)

        # ── Cross-attention ──
        self.attn = nn.MultiheadAttention(D, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(D)

        # ── Communication gate ──
        self.comm_proj = nn.Linear(16, D)
        self.gate = nn.Linear(D + D, 1)
        if self._use_comm_cross_attention:
            self.comm_token_enc = nn.Sequential(
                nn.Linear(self.comm_token_dim, D), nn.ReLU(),
                nn.Linear(D, D), nn.ReLU(),
            )
            comm_heads = 4 if D % 4 == 0 else 1
            self.comm_cross_attn = nn.MultiheadAttention(
                D, num_heads=comm_heads, batch_first=True)
            self.comm_cross_norm = nn.LayerNorm(D)
            self.comm_target_gate = nn.Linear(2 * D, D)
            if self._comm_aided_sensing_enabled:
                # Explicit receiver path: a target query and the message
                # context it retrieved jointly produce a residual on the
                # executed target-sensing logits.
                self.comm_sensing_gate = nn.Linear(2 * D, 1)
                self.comm_sensing_head = nn.Sequential(
                    nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))
            if self._comm_semantic_decoder_enabled:
                semantic_dim = self.comm_target_token_dim - 1
                self.comm_semantic_decoder = nn.Sequential(
                    nn.Linear(semantic_dim, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                    nn.Linear(16, 2),
                )

        if self._use_target_allocation:
            self.target_assignment_head = nn.Sequential(
                nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))
            if self._architecture_v2_enabled:
                # A single shared target trunk produces every target-dependent
                # action.  No parameter dimension depends on Q: permuting the
                # local target set permutes these outputs, and a checkpoint can
                # be reused at a different target cardinality.
                self.v2_target_policy = nn.Sequential(
                    nn.Linear(2 * D, D),
                    nn.LayerNorm(D),
                    nn.SiLU(),
                    nn.Linear(D, D),
                    nn.SiLU(),
                )
                self.v2_assignment_head = nn.Linear(D, 1)
                self.v2_sensing_head = nn.Linear(D, 1)
                self.v2_movement_head = nn.Linear(D, 2)
                self.v2_target_attention = nn.Linear(D, 1)
                self.v2_target_context = nn.Sequential(
                    nn.Linear(D, D), nn.LayerNorm(D), nn.SiLU())
                self.v2_movement_gate = nn.Linear(2 * D, 1)
            if self._hierarchical_dual_assignment_enabled:
                # Independent slow commitment head.  Endpoint-capacity PPO and
                # the one-target kinematic teacher no longer push the same
                # output weights in contradictory directions.
                # A zero-initialized residual adapter gives DAgger a private
                # feature path without modifying token, sensing-resource or
                # endpoint-assignment representations used by other heads.
                self.movement_feature_adapter = nn.Linear(2 * D, 2 * D)
                self.movement_commitment_head = nn.Sequential(
                    nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))
            self.allocation_proj = nn.Linear(D, D)
            self.allocation_gate = nn.Linear(2 * D, 1)
            if self._target_conditioned_movement_enabled:
                # Interpretable (radial, tangential) coefficients for every
                # locally represented target. Peer tokens condition ``te``
                # before this head is evaluated.
                self.target_movement_head = nn.Sequential(
                    nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 2))
            if (self._use_team_sinkhorn or self._capacity_matching_enabled
                    or self._movement_team_matching_enabled):
                self.neighbor_bid_msg_proj = nn.Linear(D, D)
                self.neighbor_bid_target_proj = nn.Linear(D, D)
        if self._round_negotiation_enabled:
            # A local synchronized phase bit is not inter-UAV information.  It
            # lets the same shared policy emit a proposal token in round 0 and
            # a response token after consuming the delivered proposal in round 1.
            if not self._architecture_v2_enabled:
                self.round_phase_enc = nn.Sequential(
                    nn.Linear(2 + self.K, D), nn.Tanh(), nn.Linear(D, D))
            if self._permutation_equivariant_round_encoding_enabled:
                # The phase is shared local protocol state, not an agent ID.
                # Removing the K-dimensional one-hot makes this branch
                # permutation equivariant and cardinality independent.
                self.round_phase_equivariant_enc = nn.Sequential(
                    nn.Linear(2, D), nn.Tanh(), nn.Linear(D, D))
            self.round_target_gate = nn.Linear(2 * D, 1)
            # The payload remains latent.  This shared decoder merely learns a
            # comparable per-target peer claim from each delivered token.
            self.round_peer_claim_head = nn.Sequential(
                nn.Linear(self.comm_target_token_dim, D), nn.ReLU(),
                nn.Linear(D, 1))

        # ── Output heads ──
        self.dp_head = nn.Linear(D, 2)
        self.comm_head = nn.Linear(D, 16)
        if self._comm_target_token_enabled:
            self.comm_target_token_head = nn.Linear(
                D, self.comm_target_token_dim)
        if not self._architecture_v2_enabled:
            self.comm_rate_head = nn.Linear(
                self.comm_payload_dim, comm_num_rate_levels)
        if self._comm_channel_feedback_rate_enabled:
            self.comm_rate_feedback_head = nn.Linear(
                self.comm_channel_feedback_dim, comm_num_rate_levels)
        comm_std_dim = (
            self.comm_target_token_dim
            if self._architecture_v2_enabled
            else self.comm_payload_dim)
        self.comm_log_std = nn.Parameter(
            torch.full((comm_std_dim,), float(comm_log_std_init)))
        if not self._architecture_v2_enabled:
            self.isac_power_mean_head = nn.Linear(self.comm_payload_dim, 1)
            self.isac_sensing_mean_head = nn.Linear(self.comm_payload_dim, Q)
        if self._scale_equivariant_comm_heads_enabled:
            # Pool only this UAV's own per-target outgoing tokens. No raw
            # state or hidden representation from another UAV is available.
            self.comm_set_attention = nn.Linear(
                self.comm_target_token_dim, 1)
            self.comm_set_rate_head = nn.Linear(
                self.comm_target_token_dim, comm_num_rate_levels)
            self.isac_set_power_head = nn.Linear(
                self.comm_target_token_dim, 1)
        self.isac_power_log_std = nn.Parameter(torch.tensor(
            [float(isac_power_log_std_init)]))
        sensing_std_dim = 1 if self._architecture_v2_enabled else Q
        self.isac_sensing_log_std = nn.Parameter(torch.full(
            (sensing_std_dim,), float(sensing_allocation_log_std_init)))
        if not self._architecture_v2_enabled:
            self.intent_head = nn.Linear(self.comm_payload_dim, Q)
        self.role_head = nn.Linear(D, 3)
        self.dp_log_std = nn.Parameter(torch.zeros(2))

        if self._architecture_v2_modular_coordination_enabled:
            # Register optional experts only after every historical actor
            # module. Combined with fork_rng, this makes all common parameters
            # bitwise identical for the same seed in enabled/disabled actors.
            # Thus a gate experiment cannot mistake constructor RNG drift for
            # a modular-routing gain.
            with torch.random.fork_rng(devices=[]):
                router_input_dim = 2 * D + 3
                self.v2_module_router = nn.Sequential(
                    nn.Linear(router_input_dim, D),
                    nn.SiLU(),
                    nn.Linear(
                        D, self._architecture_v2_modular_num_experts),
                )
                self.v2_coordination_experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(D, D),
                        nn.SiLU(),
                        nn.Linear(D, D),
                    )
                    for _ in range(
                        self._architecture_v2_modular_num_experts)
                ])

        self._init_weights()
        if hasattr(self, 'comm_rate_head'):
            nn.init.zeros_(self.comm_rate_head.weight)
            nn.init.zeros_(self.comm_rate_head.bias)
        if self._comm_channel_feedback_rate_enabled:
            nn.init.zeros_(self.comm_rate_feedback_head.weight)
            nn.init.zeros_(self.comm_rate_feedback_head.bias)
        if hasattr(self, 'isac_power_mean_head'):
            nn.init.zeros_(self.isac_power_mean_head.weight)
            nn.init.zeros_(self.isac_power_mean_head.bias)
        if hasattr(self, 'isac_sensing_mean_head'):
            nn.init.zeros_(self.isac_sensing_mean_head.weight)
            nn.init.zeros_(self.isac_sensing_mean_head.bias)
        if self._scale_equivariant_comm_heads_enabled:
            # Uniform set pooling is the neutral cardinality-equivariant
            # initialization. Compatible checkpoint loading replaces the two
            # output heads with block-summed legacy weights.
            nn.init.zeros_(self.comm_set_attention.weight)
            nn.init.zeros_(self.comm_set_attention.bias)
            nn.init.zeros_(self.comm_set_rate_head.weight)
            nn.init.zeros_(self.comm_set_rate_head.bias)
            nn.init.zeros_(self.isac_set_power_head.weight)
            nn.init.zeros_(self.isac_set_power_head.bias)
        if self._use_target_allocation:
            # Nearly identity-preserving for old physical-policy warm starts;
            # PPO and the allocation auxiliary can open the residual gate.
            nn.init.zeros_(self.allocation_gate.weight)
            nn.init.constant_(self.allocation_gate.bias, -4.0)
            if self._architecture_v2_enabled:
                # Start from the explicit geometry/QoS prior. The learned
                # target heads are residuals and can override it after PPO
                # observes communication-conditioned improvements.
                nn.init.zeros_(self.v2_assignment_head.weight)
                nn.init.zeros_(self.v2_assignment_head.bias)
                nn.init.zeros_(self.v2_sensing_head.weight)
                nn.init.zeros_(self.v2_sensing_head.bias)
                nn.init.zeros_(self.v2_movement_head.weight)
                with torch.no_grad():
                    self.v2_movement_head.bias.copy_(torch.tensor(
                        [1.0, 0.0],
                        dtype=self.v2_movement_head.bias.dtype))
                nn.init.zeros_(self.v2_movement_gate.weight)
                nn.init.constant_(self.v2_movement_gate.bias, 2.0)
                if self._architecture_v2_modular_coordination_enabled:
                    # Uniform routing makes the centered expert mixture an
                    # exact no-op for migrated V2 checkpoints. Different small
                    # expert bases still give the router a useful first-step
                    # gradient under direct-bid supervision.
                    nn.init.zeros_(self.v2_module_router[-1].weight)
                    nn.init.zeros_(self.v2_module_router[-1].bias)
                    for expert in self.v2_coordination_experts:
                        nn.init.orthogonal_(expert[-1].weight, gain=0.02)
                        nn.init.zeros_(expert[-1].bias)
            if self._target_conditioned_movement_enabled:
                # Exact no-op for old checkpoints. PPO must earn any physical
                # influence through target-wise movement credit.
                nn.init.zeros_(self.target_movement_head[-1].weight)
                nn.init.zeros_(self.target_movement_head[-1].bias)
        if self._round_negotiation_enabled:
            nn.init.zeros_(self.round_target_gate.weight)
            nn.init.constant_(self.round_target_gate.bias, -4.0)
            nn.init.zeros_(self.round_peer_claim_head[-1].weight)
            nn.init.zeros_(self.round_peer_claim_head[-1].bias)
        if self._comm_aided_sensing_enabled:
            # Neutral for old checkpoints; the auxiliary objective opens the
            # path only when received tokens improve target-level sensing.
            nn.init.zeros_(self.comm_sensing_head[-1].weight)
            nn.init.zeros_(self.comm_sensing_head[-1].bias)
            nn.init.zeros_(self.comm_sensing_gate.weight)
            nn.init.constant_(self.comm_sensing_gate.bias, -2.0)

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                g = 0.01 if 'head' in name else np.sqrt(2)
                nn.init.orthogonal_(m.weight, gain=g)
                nn.init.constant_(m.bias, 0.0)

    def zero_init_new_layers(self, known_keys: set):
        """Zero-initialize layers NOT present in an old checkpoint.

        When loading a DAgger checkpoint that predates pd_hist_proj (or any
        future layer addition), new layers get random orthogonal weights from
        __init__ → _init_weights. That random init changes the policy, so the
        "DAgger baseline" is no longer the true DAgger policy.

        Call this AFTER load_state_dict(..., strict=False) to zero out any
        parameter whose name is NOT in known_keys. For Linear layers, both
        weight and bias are zeroed, making them identity-through-zero:
        e_{kq} = e_{kq}^{base} + 0 = e_{kq}^{base}.

        Args:
            known_keys: set of parameter names present in the old checkpoint.
        """
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n not in known_keys:
                    p.zero_()
                    print(f'  [zero_init] {n} ← zeros (not in old checkpoint)')

    def _parse_obs(self, obs: torch.Tensor):
        """Parse flat obs. Returns entity tensors as sequences (B, N, W, D)."""
        B = obs.shape[0]
        obs_dim = obs.shape[1]
        single_dim = self.single_frame_dim if self.single_frame_dim > 0 else obs_dim

        # Audit 2026-08-25 (P1b): the legacy hand parser was unreachable dead
        # code (self._use_corrected_parser is hard-set True at __init__); its
        # definition was removed.  The corrected slice-descriptor parser is the
        # single authoritative parser.
        parse_fn = self._parse_one_corrected

        if obs_dim > single_dim + 20:
            w = obs_dim // single_dim
            frames = []
            for i in range(w):
                s, t, n, g, c, pd, mt, mm, cf = parse_fn(
                    obs[:, i*single_dim:(i+1)*single_dim], B)
                frames.append((s, t, n, g, c, pd, mt, mm, cf))
            s_seq = torch.stack([f[0] for f in frames], dim=-1)
            t_seq = torch.stack([f[1] for f in frames], dim=-1)
            n_seq = torch.stack([f[2] for f in frames], dim=-1)
            g_seq = torch.stack([f[3] for f in frames], dim=-1)
            pd_seq = torch.stack([f[5] for f in frames], dim=-1)
            return (s_seq, t_seq, n_seq, g_seq, frames[-1][4], w, pd_seq,
                    frames[-1][6], frames[-1][7], frames[-1][8])
        else:
            s, t, n, g, c, pd, mt, mm, cf = parse_fn(obs, B)
            return (s.unsqueeze(-1), t.unsqueeze(-1), n.unsqueeze(-1),
                    g.unsqueeze(-1), c, 1, pd.unsqueeze(-1), mt, mm, cf)

    def _parse_one_corrected(self, obs, B):
        """Parse using ObservationSlices — correct block-based layout."""
        K, Q = self.K, self.Q
        sl = self._obs_slices

        # Self: 8 + physics(3) = 11
        self_raw = obs[:, sl.self_start:sl.self_start + sl.self_len]
        phys = obs[:, sl.physics_start:sl.physics_start + sl.physics_len]
        self_state = torch.cat([self_raw, phys], dim=-1)  # (B, 11)

        # Beliefs block (Q × 9) → reshape
        beliefs = obs[:, sl.belief_start:sl.belief_start + Q * sl.belief_per_target]
        beliefs = beliefs.reshape(B, Q, sl.belief_per_target)  # (B, Q, 9)

        # Geometry block (Q × 8) when rel_features
        if sl.has_rel_features and sl.geom_per_target > 0:
            geometry = obs[:, sl.geom_start:sl.geom_start + Q * sl.geom_per_target]
            geometry = geometry.reshape(B, Q, sl.geom_per_target)  # (B, Q, 8)
        else:
            geometry = torch.zeros(B, Q, 0, device=obs.device)

        # Per-target: cat belief + geometry
        targets = []
        for q in range(Q):
            tq = torch.cat([beliefs[:, q, :], geometry[:, q, :]], dim=-1)  # 17 dims
            targets.append(tq)
        target_stack = torch.stack(targets, dim=1)  # (B, Q, 17)

        # Coverage (P0)
        if sl.has_p0 and sl.coverage_len > 0:
            cov = obs[:, sl.coverage_start:sl.coverage_start + sl.coverage_len]
            target_stack = torch.cat([target_stack, cov.unsqueeze(-1)], dim=-1)  # → 18
        if target_stack.shape[-1] < 18:
            target_stack = torch.cat([
                target_stack,
                torch.zeros(B, Q, 18 - target_stack.shape[-1], device=obs.device)
            ], dim=-1)

        # Neighbors block — always pad to 9
        n_dim = sl.neighbor_per_agent
        n_raw = obs[:, sl.neighbor_start:sl.neighbor_start + (K-1) * n_dim]
        neighbors = n_raw.reshape(B, K-1, n_dim)
        if n_dim == 8:
            neighbors = torch.cat([
                neighbors[:, :, :7],
                torch.zeros(B, K-1, 1, device=obs.device),
                neighbors[:, :, 7:]
            ], dim=-1)
        neighbor_stack = neighbors  # (B, K-1, 9)

        # Global
        if sl.has_p0 and sl.global_len > 0:
            global_feat = obs[:, sl.global_start:sl.global_start + sl.global_len]
        else:
            global_feat = torch.zeros(B, 2, device=obs.device)

        # PD_hist and comm
        pd_hist = obs[:, sl.pd_hist_start:sl.pd_hist_start + sl.pd_hist_len]
        comm_agg = obs[:, sl.comm_start:sl.comm_start + sl.comm_len]
        if sl.has_comm_tokens:
            token_count = (K - 1) * sl.comm_tokens_per_sender
            token_len = token_count * sl.comm_token_per_sender
            comm_tokens = obs[:, sl.comm_token_start:sl.comm_token_start + token_len]
            comm_tokens = comm_tokens.reshape(
                B, token_count, sl.comm_token_per_sender)
            comm_mask = obs[:, sl.comm_mask_start:sl.comm_mask_start + sl.comm_mask_len]
        else:
            token_count = (K - 1) * self.comm_tokens_per_sender
            comm_tokens = torch.zeros(B, token_count, self.comm_token_dim,
                                      device=obs.device)
            comm_mask = torch.zeros(B, token_count, device=obs.device)
        if sl.has_channel_feedback:
            channel_feedback = obs[
                :, sl.channel_feedback_start:
                sl.channel_feedback_start + sl.channel_feedback_len]
        else:
            channel_feedback = torch.zeros(
                B, self.comm_channel_feedback_dim, device=obs.device)

        return (self_state, target_stack, neighbor_stack, global_feat,
                comm_agg, pd_hist, comm_tokens, comm_mask,
                channel_feedback)


    def _route_v2_coordination_modules(
        self,
        target_latent: torch.Tensor,
        local_target_latent: torch.Tensor,
        qos_deficit: torch.Tensor,
    ):
        """Apply identity-free shared experts to the slow coordination latent.

        The routing context is a permutation-invariant summary of this UAV's
        local target set plus three local QoS-deficit statistics. The same
        routing is used for message-conditioned and pre-message latents, so the
        transmitted comparable bid remains receiver independent.

        Expert outputs are mixed with ``routing - uniform``. A newly enabled
        zero-initialized router is therefore an exact no-op while direct
        bid/movement supervision can still break the routing symmetry.
        """
        if not self._architecture_v2_modular_coordination_enabled:
            self.last_v2_module_routing = None
            self.last_v2_module_residual_norm = None
            return target_latent, local_target_latent

        pooled_mean = local_target_latent.mean(dim=1)
        pooled_max = local_target_latent.amax(dim=1)
        deficit_stats = torch.stack([
            qos_deficit.mean(dim=-1),
            qos_deficit.amax(dim=-1),
            qos_deficit.std(dim=-1, unbiased=False),
        ], dim=-1)
        router_input = torch.cat(
            [pooled_mean, pooled_max, deficit_stats], dim=-1)
        route_logits = self.v2_module_router(router_input)
        routing = torch.softmax(
            route_logits / self._architecture_v2_modular_temperature,
            dim=-1)
        centered_routing = routing - (
            1.0 / float(self._architecture_v2_modular_num_experts))

        def apply_experts(latent):
            expert_outputs = torch.stack(
                [expert(latent) for expert in self.v2_coordination_experts],
                dim=2,
            )
            residual = torch.einsum(
                'bm,bqmd->bqd', centered_routing, expert_outputs)
            residual = self._architecture_v2_modular_gain * residual
            return latent + residual, residual

        routed_target, target_residual = apply_experts(target_latent)
        routed_local, _ = apply_experts(local_target_latent)
        self.last_v2_module_routing = routing
        self.last_v2_module_residual_norm = (
            target_residual.norm(dim=-1).mean(dim=-1).detach())
        return routed_target, routed_local

    def forward(self, obs: torch.Tensor, h_prev: torch.Tensor = None,
                detach_h_new: bool = True, window_mask: torch.Tensor = None,
                comm_round_phase: torch.Tensor = None,
                agent_identity: torch.Tensor = None):
        """Forward with optional streaming GRU hidden state.

        Args:
            obs: (B, obs_dim) observation batch
            h_prev: (1, B*(K-1), D) GRU hidden states, or None for zero-init.
            detach_h_new: If True (default), detach h_new before returning.
            window_mask: ignored (interface compat with TICA).
                True for rollout, evaluation, and PPO single-step updates
                (prevents cross-timestep computation graphs).
                False for DAgger chunk BPTT training (allows gradient flow
                within a chunk; caller must detach at chunk boundaries).

        Returns:
            dp_mean, log_std, role_logits, comm_msg, pd_pred, h_new
        """
        result = self._parse_obs(obs)
        (self_s, targets, neighbors, global_f, comm_agg, n_frames, pd_hist,
         comm_tokens, comm_mask, channel_feedback) = result
        self.last_comm_channel_feedback = channel_feedback
        B = obs.shape[0]
        D = self.self_enc[0].out_features
        if agent_identity is None:
            identity_index = torch.zeros(
                B, dtype=torch.long, device=obs.device)
        else:
            identity_index = agent_identity.to(
                device=obs.device, dtype=torch.long).reshape(B)
            identity_index = identity_index.clamp(0, self.K - 1)
        LOG_STD_MIN, LOG_STD_MAX = -1.0, 1.0
        log_std = LOG_STD_MIN + 0.5*(LOG_STD_MAX-LOG_STD_MIN)*(torch.tanh(self.dp_log_std)+1.0)

        if targets is None:
            h = self.self_enc(torch.zeros(B, 11, device=obs.device))
            self.last_policy_latent = h.detach()
            comm_out = (torch.zeros(B, self.comm_payload_dim, device=obs.device)
                        if self._comm_target_token_enabled
                        else torch.tanh(self.comm_head(h)))
            return (self.dp_head(h), log_std, self.role_head(h), comm_out,
                    torch.zeros(B, 1, device=obs.device), None)

        # Encode entities: use last timestep (dim=-1 is seq)
        se = self.self_enc(self_s[..., -1]).unsqueeze(1)          # (B, 1, D)
        te_base = self.target_enc(targets[..., -1])                # (B, Q, D)
        ge = self.global_enc(global_f[..., -1]).unsqueeze(1)      # (B, 1, D)

        # P1 FIX: Project PD_hist into target entity encoding.
        # This gives Actor direct knowledge of which targets had low detection
        # in the previous frame — the most direct signal of target failure.
        # PD_hist shape: (B, Q, W) → use last timestep
        pd_last = pd_hist[..., -1]  # (B, Q)
        pd_feat = self.pd_hist_proj(pd_last.unsqueeze(-1))  # (B, Q, D)
        te = te_base + pd_feat  # residual modulation by detection history

        # Target-conditioned receiver: every target entity independently queries
        # the delivered per-sender messages. Invalid/silent/expired links are
        # masked, and all-missing batches are forced to a zero residual.
        msg_entities = None
        valid = None
        valid_any = None
        self.last_comm_sensing_logits = None
        self.last_comm_semantic_evidence = None
        self.last_comm_semantic_pd = None
        self.last_comm_semantic_capacity_bias = None
        self.last_comm_semantic_extra_token_mask = None
        self.last_v2_module_routing = None
        self.last_v2_module_residual_norm = None
        local_te = te
        if (self._use_comm_cross_attention
                and comm_tokens is not None and comm_tokens.shape[1] > 0):
            valid = comm_mask > 0.5
            valid_any = valid.any(dim=1)
            safe_valid = valid.clone()
            if (~valid_any).any():
                safe_valid[~valid_any, 0] = True
            token_input = comm_tokens * valid.unsqueeze(-1).to(comm_tokens.dtype)
            if self._comm_semantic_decoder_enabled:
                evidence_logits, semantic_pd = self.decode_comm_semantics(
                    token_input[..., :self.comm_target_token_dim])
                valid_float = valid.to(semantic_pd.dtype)
                self.last_comm_semantic_evidence = (
                    torch.sigmoid(evidence_logits) * valid_float)
                self.last_comm_semantic_pd = semantic_pd * valid_float
            msg_entities = self.comm_token_enc(token_input)
            msg_ctx, attn_weights = self.comm_cross_attn(
                te, msg_entities, msg_entities,
                key_padding_mask=~safe_valid,
                need_weights=True,
                average_attn_weights=False,
            )
            msg_ctx = msg_ctx * valid_any[:, None, None].to(msg_ctx.dtype)
            msg_gate = torch.sigmoid(self.comm_target_gate(
                torch.cat([te, msg_ctx], dim=-1)))
            # Normalize only the message residual. Normalizing the whole sum
            # would alter a warm-started physical policy merely because a
            # message arrived, even while the receiver branch is untrained.
            fused_te = te + msg_gate * self.comm_cross_norm(msg_ctx)
            # Preserve the physical target embedding for samples with no
            # delivered token, so silence is a genuinely neutral input.
            te = torch.where(valid_any[:, None, None], fused_te, te)
            if self._comm_aided_sensing_enabled:
                comm_sensing_input = torch.cat([local_te, msg_ctx], dim=-1)
                comm_sensing_gate = torch.sigmoid(
                    self.comm_sensing_gate(comm_sensing_input)).squeeze(-1)
                comm_sensing_logits = torch.tanh(
                    self.comm_sensing_head(
                        comm_sensing_input).squeeze(-1))
                self.last_comm_sensing_logits = (
                    comm_sensing_gate * comm_sensing_logits
                    * valid_any[:, None].to(comm_sensing_logits.dtype))
            # Kept only for diagnostics/figures; never fed back into training.
            self.last_comm_attention = (
                attn_weights.detach()
                * valid[:, None, None, :].to(attn_weights.dtype)
            )
        else:
            self.last_comm_attention = None

        # Phase-conditioned proposal/response refinement.  In round 1, ``te``
        # already contains the physically delivered round-0 token context, so
        # the outgoing response is causally conditioned on peer proposals.
        negotiation_te = te
        if self._round_negotiation_enabled:
            if comm_round_phase is None:
                phase = torch.zeros(B, 1, device=obs.device, dtype=te.dtype)
            else:
                phase = comm_round_phase.to(
                    device=obs.device, dtype=te.dtype).reshape(B, 1)
                phase = phase.clamp(0.0, 1.0)
            phase_only = torch.cat([1.0 - phase, phase], dim=-1)
            if self._permutation_equivariant_round_encoding_enabled:
                phase_ctx = self.round_phase_equivariant_enc(
                    phase_only).unsqueeze(1)
            else:
                identity_onehot = torch.nn.functional.one_hot(
                    identity_index, num_classes=self.K).to(te.dtype)
                phase_input = torch.cat(
                    [phase_only, identity_onehot], dim=-1)
                phase_ctx = self.round_phase_enc(phase_input).unsqueeze(1)
            phase_ctx = phase_ctx.expand(-1, self.Q, -1)
            phase_gate = torch.sigmoid(self.round_target_gate(torch.cat(
                [te, phase_ctx], dim=-1)))
            negotiation_te = te + phase_gate * phase_ctx

        outgoing_target_tokens = None
        if self._comm_target_token_enabled:
            outgoing_target_tokens = torch.tanh(
                self.comm_target_token_head(negotiation_te))

        # Decentralized target responsibility. Target embeddings already carry
        # the masked per-sender communication context, so the allocation remains
        # local at execution time while being communication-aware.
        assignment_context = None
        v2_local_bid_scores = None
        v2_movement_consensus = None
        v2_endpoint_consensus = None
        v2_consensus_available = None
        if self._use_target_allocation:
            self_for_targets = se.expand(-1, self.Q, -1)
            assignment_features = torch.cat(
                [negotiation_te, self_for_targets], dim=-1)
            if self._architecture_v2_enabled:
                v2_target_latent = self.v2_target_policy(
                    assignment_features)
                normalized_distance = targets[
                    ..., -1][:, :, 11].clamp_min(0.0)
                qos_deficit = torch.relu(
                    self._architecture_v2_qos_floor - pd_last
                ) / self._architecture_v2_qos_floor
                # A cardinality-invariant local crisis statistic supplies the
                # communication-rate head with a deployment-time prior.  It
                # removes the stochastic-training/deterministic-evaluation
                # tie at initialization while still allowing silence after
                # every locally observed target clears the QoS floor.
                self.last_v2_comm_crisis = qos_deficit.amax(dim=-1)
                physics_prior = (
                    qos_deficit
                    - self._architecture_v2_distance_weight
                    * normalized_distance)
                # The comparable bid must not depend on received messages:
                # otherwise different receivers reconstruct different team
                # matrices.  The shared scorer is evaluated on the pre-message
                # local target representation for a stable proposal round.
                local_assignment_features = torch.cat(
                    [local_te, self_for_targets], dim=-1)
                local_target_latent = self.v2_target_policy(
                    local_assignment_features)
                v2_target_latent, local_target_latent = (
                    self._route_v2_coordination_modules(
                        v2_target_latent,
                        local_target_latent,
                        qos_deficit,
                    ))
                learned_assignment = (
                    self._architecture_v2_bid_residual_scale
                    * torch.tanh(self.v2_assignment_head(
                        v2_target_latent).squeeze(-1)))
                local_assignment_logits = (
                    self._architecture_v2_bid_residual_scale
                    * torch.tanh(self.v2_assignment_head(
                        local_target_latent).squeeze(-1))
                    + self._architecture_v2_prior_gain * physics_prior)
                v2_local_bid_scores = torch.tanh(local_assignment_logits)
                # Non-detached logits are the causal variable encoded in the
                # comparable Token header. CTDE may supervise this local
                # intention directly instead of backpropagating through a
                # saturated hard team matching result.
                self.last_v2_local_bid_logits = local_assignment_logits
                self.last_v2_local_bids = (
                    v2_local_bid_scores.detach())
                assignment_logits = (
                    learned_assignment
                    + self._architecture_v2_prior_gain * physics_prior)
                self.last_v2_target_latent = v2_target_latent
                self.last_v2_sensing_logits = (
                    self.v2_sensing_head(
                        v2_target_latent).squeeze(-1)
                    + self._architecture_v2_prior_gain * physics_prior)
                self.last_v2_physics_prior = physics_prior.detach()
            else:
                self.last_v2_comm_crisis = None
                assignment_logits = self.target_assignment_head(
                    assignment_features).squeeze(-1)
            if self._hierarchical_dual_assignment_enabled:
                movement_features = assignment_features + torch.tanh(
                    self.movement_feature_adapter(assignment_features))
                movement_logits = self.movement_commitment_head(
                    movement_features).squeeze(-1)
            else:
                movement_logits = assignment_logits
            if (self._sparse_claim_enabled and valid is not None
                    and valid.shape[1] == (self.K - 1) * self.Q):
                # A delivered target-token mask is an explicit sparse proposal,
                # not prescribed token semantics. One peer claim is welcome as
                # the other bistatic endpoint; two or more fill the target and
                # suppress further competition. Unclaimed targets receive a
                # small vacancy bonus so remote targets are not abandoned.
                peer_claims = valid.reshape(
                    B, self.K - 1, self.Q).to(assignment_logits.dtype)
                peer_load = peer_claims.sum(dim=1)
                desired = float(self._sparse_claim_desired_endpoints)
                full_gate = torch.sigmoid(
                    (peer_load - (desired - 0.5))
                    / self._sparse_claim_temperature)
                vacant_gate = torch.sigmoid(
                    (0.5 - peer_load) / self._sparse_claim_temperature)
                coordination_bias = (
                    -self._sparse_claim_full_penalty * full_gate
                    + self._sparse_claim_vacant_bonus * vacant_gate)
                assignment_logits = assignment_logits + coordination_bias
                movement_logits = movement_logits + coordination_bias
                self.last_peer_claim_load = peer_load.detach()
            else:
                self.last_peer_claim_load = None
            if self._architecture_v2_enabled:
                # Reserve one token coordinate as a scale-free coordination
                # header.  Each node compares its local bid with delivered
                # peer bids using only its own inbox.  The remaining token
                # coordinates remain unconstrained latent semantics.
                if (msg_entities is not None and valid is not None
                        and msg_entities.shape[1]
                        == (self.K - 1) * self.Q):
                    peer_content = comm_tokens[
                        ..., :self.comm_target_token_dim].reshape(
                            B, self.K - 1, self.Q,
                            self.comm_target_token_dim)
                    peer_bids = peer_content[..., 0]
                    peer_valid = valid.reshape(B, self.K - 1, self.Q)
                    margin = (
                        v2_local_bid_scores.unsqueeze(1) - peer_bids
                    ) / self._round_negotiation_temperature
                    win_log = torch.nn.functional.logsigmoid(margin)
                    win_log = win_log * peer_valid.to(win_log.dtype)
                    peer_count = peer_valid.sum(
                        dim=1).clamp_min(1).to(win_log.dtype)
                    agreement = win_log.sum(dim=1) / peer_count
                    assignment_logits = (
                        assignment_logits
                        + self._round_negotiation_strength * agreement)
                    movement_logits = (
                        movement_logits
                        + self._round_negotiation_strength * agreement)
                    self.last_v2_peer_bids = peer_bids.detach()
                    self.last_v2_bid_agreement = agreement.detach()
                    if self._architecture_v2_consensus_enabled:
                        # Every receiver sees the same set of sender bids up to
                        # row permutation.  Exact assignment (small square
                        # teams) and Sinkhorn fallback are row-equivariant, so
                        # the receiver can select its first/local row without
                        # an absolute UAV ID or central coordinator.
                        # Use this UAV's physically transmitted previous bid,
                        # not a newly recomputed one.  Peers received that same
                        # bid, so every node reconstructs exactly the same
                        # sparse matrix up to row order despite message delay.
                        # The fixed local-memory block supports the intended
                        # Q<=8 scaling range (including the 8/8 experiment).
                        if comm_agg.shape[-1] >= 2 * self.Q:
                            own_memory_bid = comm_agg[:, :self.Q]
                            own_memory_valid = (
                                comm_agg[:, self.Q:2 * self.Q] > 0.5)
                        else:
                            own_memory_bid = v2_local_bid_scores
                            own_memory_valid = torch.zeros(
                                B, self.Q, dtype=torch.bool,
                                device=obs.device)
                        team_bids = torch.cat([
                            own_memory_bid.unsqueeze(1),
                            peer_bids,
                        ], dim=1)
                        team_valid = torch.cat([
                            own_memory_valid.unsqueeze(1),
                            peer_valid,
                        ], dim=1)
                        feasible_bids = torch.where(
                            team_valid,
                            team_bids,
                            torch.full_like(team_bids, -12.0))
                        movement_team = exact_permutation_assignment(
                            feasible_bids,
                            temperature=(
                                self._architecture_v2_matching_temperature),
                        )
                        v2_movement_consensus = movement_team[:, 0, :]

                        endpoint_capacity = min(
                            float(self._sparse_claim_desired_endpoints),
                            float(self.K))
                        row_capacity = (
                            endpoint_capacity * float(self.Q)
                            / float(self.K))
                        endpoint_team = capacity_sinkhorn_normalize(
                            feasible_bids,
                            row_capacity=row_capacity,
                            column_capacity=endpoint_capacity,
                            iterations=16,
                            temperature=(
                                self._architecture_v2_matching_temperature),
                        )
                        v2_endpoint_consensus = endpoint_team[:, 0, :]
                        v2_endpoint_consensus = (
                            v2_endpoint_consensus
                            / v2_endpoint_consensus.sum(
                                dim=-1, keepdim=True).clamp_min(1e-8))
                        v2_consensus_available = (
                            own_memory_valid.any(dim=-1, keepdim=True)
                            & peer_valid.any(
                                dim=-1).all(dim=-1, keepdim=True))
                        self.last_v2_movement_consensus = (
                            movement_team.detach())
                        self.last_v2_endpoint_consensus = (
                            endpoint_team.detach())
                        self.last_v2_consensus_available = (
                            v2_consensus_available.detach())
                else:
                    self.last_v2_peer_bids = None
                    self.last_v2_bid_agreement = None
                    self.last_v2_movement_consensus = None
                    self.last_v2_endpoint_consensus = None
                    self.last_v2_consensus_available = None
                self.last_round_peer_claims = None
            elif (self._round_negotiation_enabled
                    and msg_entities is not None and valid is not None
                    and msg_entities.shape[1]
                    == (self.K - 1) * self.Q):
                peer_content = comm_tokens[
                    ..., :self.comm_target_token_dim].reshape(
                        B, self.K - 1, self.Q,
                        self.comm_target_token_dim)
                peer_claims = self.round_peer_claim_head(
                    peer_content).squeeze(-1)
                own_claims = self.round_peer_claim_head(
                    outgoing_target_tokens).squeeze(-1)
                peer_valid = valid.reshape(B, self.K - 1, self.Q)
                # A UAV locally prefers targets for which its own claim outranks
                # delivered peer claims.  All operations use only its inbox.
                margin = (
                    own_claims.unsqueeze(1) - peer_claims
                ) / self._round_negotiation_temperature
                win_log = torch.nn.functional.logsigmoid(margin)
                win_log = win_log * peer_valid.to(win_log.dtype)
                peer_count = peer_valid.sum(dim=1).clamp_min(1).to(win_log.dtype)
                agreement = win_log.sum(dim=1) / peer_count
                assignment_logits = (
                    assignment_logits
                    + self._round_negotiation_strength * agreement)
                movement_logits = (
                    movement_logits
                    + self._round_negotiation_strength * agreement)
                self.last_round_peer_claims = peer_claims.detach()
            else:
                self.last_round_peer_claims = None
            assignment_probs = torch.softmax(
                assignment_logits / self._target_allocation_temperature,
                dim=-1)
            # The slow kinematic variable has a one-target semantics and an
            # independent head; the endpoint branch below is projected to
            # capacity two without changing these probabilities.
            movement_probs = torch.softmax(
                movement_logits / self._target_allocation_temperature,
                dim=-1)
            if (v2_movement_consensus is not None
                    and v2_consensus_available is not None):
                movement_blend = (
                    self._architecture_v2_movement_consensus_blend)
                consensus_movement = (
                    (1.0 - movement_blend) * movement_probs
                    + movement_blend * v2_movement_consensus)
                movement_probs = torch.where(
                    v2_consensus_available,
                    consensus_movement,
                    movement_probs)
            if (v2_endpoint_consensus is not None
                    and v2_consensus_available is not None):
                assignment_probs = torch.where(
                    v2_consensus_available,
                    v2_endpoint_consensus,
                    assignment_probs)
                endpoint_logits = torch.log(
                    v2_endpoint_consensus.clamp_min(1e-8))
                endpoint_logits = (
                    endpoint_logits
                    - endpoint_logits.mean(dim=-1, keepdim=True))
                self.last_v2_sensing_logits = (
                    self.last_v2_sensing_logits
                    + self._architecture_v2_endpoint_consensus_gain
                    * endpoint_logits
                    * v2_consensus_available.to(endpoint_logits.dtype))
            # Preserve the intrinsic local bid before any team projection.
            # This is what must be communicated in the next negotiation round;
            # rebroadcasting the projected one-hot winner creates a positive
            # feedback loop that permanently locks the first permutation.
            local_movement_probs = movement_probs
            self.last_capacity_assignment = None
            self.last_movement_team_assignment = None
            if ((self._use_team_sinkhorn or self._capacity_matching_enabled
                 or self._movement_team_matching_enabled)
                    and msg_entities is not None
                    and valid is not None):
                target_bids = self.neighbor_bid_target_proj(te)
                if msg_entities.shape[1] == (self.K - 1) * self.Q:
                    # Target-token payloads contain Q tokens per sender. They
                    # are Q target bids from ONE neighboring UAV, not Q extra
                    # agents. Preserve the target alignment and collapse only
                    # the sender dimension into Sinkhorn rows.
                    msg_bids = self.neighbor_bid_msg_proj(
                        msg_entities.reshape(
                            B, self.K - 1, self.Q, D))
                    neighbor_logits = torch.einsum(
                        'bnqd,bqd->bnq', msg_bids, target_bids
                    ) / np.sqrt(D)
                    neighbor_valid = valid.reshape(
                        B, self.K - 1, self.Q)
                else:
                    # Aggregate payload: one token already represents one
                    # neighboring UAV.
                    msg_bids = self.neighbor_bid_msg_proj(msg_entities)
                    neighbor_logits = torch.einsum(
                        'bnd,bqd->bnq', msg_bids, target_bids
                    ) / np.sqrt(D)
                    neighbor_valid = valid.unsqueeze(-1).expand(
                        -1, -1, self.Q)
                if self._comm_semantic_capacity_bid_enabled:
                    semantic_evidence = self.last_comm_semantic_evidence.reshape(
                        B, self.K - 1, self.Q)
                    semantic_pd = self.last_comm_semantic_pd.reshape(
                        B, self.K - 1, self.Q)
                    neighbor_logits, semantic_capacity_bias = (
                        apply_semantic_capacity_bid_correction(
                            neighbor_logits,
                            semantic_evidence,
                            semantic_pd,
                            neighbor_valid,
                            self._comm_semantic_capacity_bid_gain,
                        ))
                    self.last_comm_semantic_capacity_bias = (
                        semantic_capacity_bias.detach())
                team_logits = torch.cat(
                    [assignment_logits.unsqueeze(1), neighbor_logits], dim=1)
                team_valid = torch.cat([
                    torch.ones(
                        B, 1, self.Q, dtype=torch.bool,
                        device=valid.device),
                    neighbor_valid,
                ], dim=1)
                if self._movement_team_matching_enabled:
                    # Consensus claim graph. Peer token masks arrive in sender
                    # ID order and the otherwise-unused aggregate block stores
                    # this UAV's last transmitted mask. Scatter both into the
                    # true global UAV rows, so all receivers with complete U2U
                    # delivery evaluate the exact same matrix. Token content
                    # still conditions sensing and local logits above; the hard
                    # movement topology uses explicit sparse claims to avoid a
                    # receiver-dependent reinterpretation of the same token.
                    peer_scores = comm_tokens[
                        ..., :self.comm_target_token_dim].reshape(
                            B, self.K - 1, self.Q,
                            self.comm_target_token_dim)[..., 0]
                    peer_scores = torch.where(
                        neighbor_valid,
                        peer_scores,
                        torch.full_like(peer_scores, -12.0))
                    own_scores = comm_agg[:, :self.Q]
                    own_claims = comm_agg[
                        :, self.Q:2 * self.Q] > 0.5
                    own_scores = torch.where(
                        own_claims,
                        own_scores,
                        torch.full_like(own_scores, -12.0))
                    all_agent_ids = torch.arange(
                        self.K, device=obs.device).unsqueeze(0).expand(B, -1)
                    peer_agent_ids = all_agent_ids[
                        all_agent_ids != identity_index.unsqueeze(1)
                    ].reshape(B, self.K - 1)
                    global_claims = torch.full(
                        (B, self.K, self.Q), -12.0,
                        dtype=movement_logits.dtype,
                        device=obs.device)
                    global_claims.scatter_(
                        1,
                        peer_agent_ids.unsqueeze(-1).expand(-1, -1, self.Q),
                        peer_scores,
                    )
                    global_claims.scatter_(
                        1,
                        identity_index[:, None, None].expand(-1, 1, self.Q),
                        own_scores.unsqueeze(1),
                    )
                    # Claims dominate. A tiny lexicographic UAV/target bias
                    # resolves otherwise symmetric permutations identically at
                    # every node without materially changing claim preference.
                    agent_rank = torch.arange(
                        self.K, dtype=movement_logits.dtype,
                        device=obs.device).view(1, self.K, 1)
                    target_rank = torch.arange(
                        self.Q, dtype=movement_logits.dtype,
                        device=obs.device).view(1, 1, self.Q)
                    tie_break = -1e-3 * target_rank / torch.pow(
                        torch.as_tensor(
                            float(self.Q + 1), dtype=movement_logits.dtype,
                            device=obs.device),
                        agent_rank,
                    )
                    consensus_logits = global_claims + tie_break
                    movement_team_assignment = exact_permutation_assignment(
                        consensus_logits,
                        temperature=self._movement_team_matching_temperature,
                    )
                    batch_index = torch.arange(B, device=obs.device)
                    own_movement = movement_team_assignment[
                        batch_index, identity_index, :]
                    peer_available = neighbor_valid.any(dim=-1).all(
                        dim=-1, keepdim=True)
                    own_available = own_claims.any(
                        dim=-1, keepdim=True)
                    consensus_available = peer_available & own_available
                    movement_probs = torch.where(
                        consensus_available,
                        ((1.0 - self._movement_team_matching_blend)
                         * movement_probs
                         + self._movement_team_matching_blend
                         * own_movement),
                        movement_probs,
                    )
                    self.last_movement_team_assignment = (
                        movement_team_assignment.detach())
                if self._capacity_matching_enabled:
                    # Sparse top-k payloads intentionally leave most
                    # sender-target entries absent.  Old one-to-one Sinkhorn
                    # required valid.all(), so it never affected this path.
                    # Keep received claims as feasible edges and assign a low
                    # finite bid to absent edges so projection remains smooth.
                    sparse_team_logits = torch.where(
                        team_valid, team_logits,
                        torch.full_like(team_logits, -12.0))
                    capacity_assignment = capacity_sinkhorn_normalize(
                        sparse_team_logits,
                        row_capacity=(
                            self._capacity_matching_row_capacity),
                        column_capacity=(
                            self._capacity_matching_column_capacity),
                        iterations=self._capacity_matching_iterations,
                        temperature=self._capacity_matching_temperature,
                    )
                    own_capacity = capacity_assignment[:, 0, :]
                    own_capacity = own_capacity / own_capacity.sum(
                        dim=-1, keepdim=True).clamp_min(1e-8)
                    # A complete broadcast is unnecessary; require only one
                    # physically delivered sparse claim from every peer. This
                    # preserves decentralized operation under top-k payloads.
                    peer_available = neighbor_valid.any(dim=-1).all(
                        dim=-1, keepdim=True)
                    blend = self._capacity_matching_blend
                    blended_assignment = (
                        (1.0 - blend) * assignment_probs
                        + blend * own_capacity)
                    assignment_probs = torch.where(
                        peer_available,
                        blended_assignment,
                        assignment_probs,
                    )
                    self.last_capacity_assignment = (
                        capacity_assignment.detach())
                else:
                    team_assignment = sinkhorn_normalize(team_logits)
                    own_sinkhorn = team_assignment[:, 0, :]
                    own_sinkhorn = own_sinkhorn / own_sinkhorn.sum(
                        dim=-1, keepdim=True).clamp_min(1e-8)
                    assignment_probs = torch.where(
                        valid.all(dim=1, keepdim=True),
                        own_sinkhorn,
                        assignment_probs,
                    )
            self.last_target_assignment = assignment_probs
            if (not self._hierarchical_dual_assignment_enabled
                    and not self._architecture_v2_enabled):
                movement_probs = assignment_probs
            self.last_movement_assignment = movement_probs
            if self._target_allocation_straight_through:
                hard_assignment = torch.nn.functional.one_hot(
                    movement_probs.argmax(dim=-1),
                    num_classes=self.Q,
                ).to(movement_probs.dtype)
                movement_assignment = (
                    hard_assignment + movement_probs
                    - movement_probs.detach())
            else:
                movement_assignment = movement_probs
            self.last_target_assignment_st = movement_assignment
            assignment_context = torch.sum(
                movement_assignment.unsqueeze(-1) * negotiation_te, dim=1)
            if self._sparse_claim_enabled:
                if (self._architecture_v2_sensing_aligned_claims_enabled
                        and self.last_v2_sensing_logits is not None):
                    sensing_claim_logits = self.last_v2_sensing_logits
                    if (self._comm_aided_sensing_enabled
                            and self.last_comm_sensing_logits is not None
                            and self.last_comm_sensing_logits.shape
                            == sensing_claim_logits.shape):
                        sensing_claim_logits = (
                            sensing_claim_logits
                            + self._comm_aided_sensing_blend
                            * self.last_comm_sensing_logits)
                    shared_claim_probs = torch.softmax(
                        sensing_claim_logits, dim=-1)
                    team_bid_probs = shared_claim_probs
                else:
                    bid_mix = self._movement_team_matching_intrinsic_bid_mix
                    team_bid_probs = (
                        bid_mix * local_movement_probs
                        + (1.0 - bid_mix) * movement_probs)
                    shared_claim_probs = (
                        team_bid_probs
                        if self._movement_team_matching_enabled
                        else assignment_probs)
                self.last_sparse_claim_scores = shared_claim_probs.detach()
                top_idx = torch.topk(
                    shared_claim_probs,
                    k=self._sparse_claim_share_topk,
                    dim=-1,
                ).indices
                outgoing_mask = torch.zeros_like(assignment_probs)
                outgoing_mask.scatter_(1, top_idx, 1.0)
                if (self._comm_semantic_extra_token_enabled
                        and valid is not None
                        and valid.shape[1] == (self.K - 1) * self.Q):
                    peer_valid = valid.reshape(B, self.K - 1, self.Q)
                    peer_load = peer_valid.to(assignment_probs.dtype).sum(dim=1)
                    peer_quality = (
                        self.last_comm_semantic_evidence
                        * self.last_comm_semantic_pd
                    ).reshape(B, self.K - 1, self.Q)
                    peer_quality = torch.where(
                        peer_valid,
                        peer_quality,
                        torch.zeros_like(peer_quality),
                    ).amax(dim=1)
                    # Geometry feature 15 is exp(-distance/150 m), a bounded
                    # local capability measure available without ground truth.
                    local_capability = targets[..., -1][..., 15].clamp(0.0, 1.0)
                    endpoint_deficit = torch.relu(
                        float(self._sparse_claim_desired_endpoints) - peer_load)
                    extra_score = (
                        endpoint_deficit
                        * (1.0 - peer_quality)
                        * local_capability
                        * (1.0 - outgoing_mask)
                    )
                    extra_value, extra_index = extra_score.max(
                        dim=-1, keepdim=True)
                    extra_active = (
                        extra_value >= self._comm_semantic_extra_token_threshold
                    ).to(outgoing_mask.dtype)
                    extra_mask = torch.zeros_like(outgoing_mask)
                    extra_mask.scatter_(1, extra_index, extra_active)
                    outgoing_mask = torch.maximum(outgoing_mask, extra_mask)
                    self.last_comm_semantic_extra_token_mask = (
                        extra_mask.detach())
                self.last_outgoing_token_mask = outgoing_mask.detach()
                outgoing_target_tokens = (
                    outgoing_target_tokens * outgoing_mask.unsqueeze(-1))
                if self._architecture_v2_enabled:
                    # The comparable header is computed before peer agreement
                    # modifies the local responsibility.  Rebroadcasting the
                    # negotiated result would create a positive feedback loop.
                    bid_channel = v2_local_bid_scores * outgoing_mask
                    if self.comm_target_token_dim == 1:
                        outgoing_target_tokens = bid_channel.unsqueeze(-1)
                    else:
                        outgoing_target_tokens = torch.cat([
                            bid_channel.unsqueeze(-1),
                            outgoing_target_tokens[..., 1:],
                        ], dim=-1)
                elif self._movement_team_matching_enabled:
                    # Reserve one continuous channel in every learned target
                    # token for a comparable movement bid. The remaining token
                    # dimensions stay latent and continue to condition sensing.
                    bid_channel = (
                        (2.0 * team_bid_probs - 1.0) * outgoing_mask)
                    if self.comm_target_token_dim == 1:
                        outgoing_target_tokens = bid_channel.unsqueeze(-1)
                    else:
                        outgoing_target_tokens = torch.cat([
                            bid_channel.unsqueeze(-1),
                            outgoing_target_tokens[..., 1:],
                        ], dim=-1)
            else:
                self.last_outgoing_token_mask = None
                self.last_sparse_claim_scores = None
                if self._architecture_v2_enabled:
                    # Dense V2 communication uses the same coordination
                    # header; sparse top-k merely masks which headers leave
                    # the sender.
                    if self.comm_target_token_dim == 1:
                        outgoing_target_tokens = (
                            v2_local_bid_scores.unsqueeze(-1))
                    else:
                        outgoing_target_tokens = torch.cat([
                            v2_local_bid_scores.unsqueeze(-1),
                            outgoing_target_tokens[..., 1:],
                        ], dim=-1)
        else:
            self.last_target_assignment = None
            self.last_movement_assignment = None
            self.last_target_assignment_st = None
            self.last_capacity_assignment = None
            self.last_movement_team_assignment = None
            self.last_peer_claim_load = None
            self.last_outgoing_token_mask = None
            self.last_sparse_claim_scores = None
            self.last_v2_local_bids = None
            self.last_v2_local_bid_logits = None
            self.last_v2_movement_consensus = None
            self.last_v2_endpoint_consensus = None
            self.last_v2_consensus_available = None

        # Streaming GRU for neighbors: (B*Nn, 1, 9) with per-neighbor state
        Nn = neighbors.shape[1]
        n_feat = neighbors[..., -1]  # (B, Nn, 9) — last timestep
        n_input = n_feat.reshape(B * Nn, 1, 9)  # (B*Nn, 1, 9)
        if h_prev is not None:
            _, hn = self.neighbor_gru(n_input, h_prev)
        else:
            _, hn = self.neighbor_gru(n_input)
        ne = self.neighbor_proj(hn.squeeze(0)).reshape(B, Nn, -1)
        h_new = hn.detach() if detach_h_new else hn  # (1, B*Nn, D)

        entities = torch.cat([ge, te, ne], dim=1)
        ctx, _ = self.attn(se, entities, entities)
        h_physical = self.attn_norm(se + ctx).squeeze(1)

        if assignment_context is not None:
            allocation_gate = torch.sigmoid(self.allocation_gate(torch.cat(
                [h_physical, assignment_context], dim=-1)))
            h_physical = h_physical + allocation_gate * self.allocation_proj(
                assignment_context)

        if self._use_comm_cross_attention:
            h = h_physical
        else:
            cp = self.comm_proj(comm_agg)
            gate = torch.sigmoid(self.gate(torch.cat([h_physical, cp], dim=-1)))
            h = h_physical + gate * cp

        self.last_policy_latent = h.detach()
        dp_mean = self.dp_head(h)
        if (self._architecture_v2_enabled
                and assignment_context is not None
                and self.last_v2_target_latent is not None):
            # Shared target-conditioned kinematics.  Each target proposes a
            # radial/tangential action in the UAV's local frame; the
            # communication-refined responsibility distribution combines the
            # proposals.  Both the scorer and the aggregation are target
            # permutation equivariant and independent of Q.
            target_rel_xy = targets[..., -1][:, :, 9:11]
            target_rel_norm = target_rel_xy.norm(dim=-1, keepdim=True)
            radial_direction = (
                target_rel_xy / target_rel_norm.clamp_min(1e-8))
            radial_direction = torch.where(
                target_rel_norm > 1e-8,
                radial_direction,
                torch.zeros_like(radial_direction))
            tangent_direction = torch.stack(
                [-radial_direction[..., 1], radial_direction[..., 0]],
                dim=-1)
            raw_coefficients = self.v2_movement_head(
                self.last_v2_target_latent)
            # A committed UAV may learn how strongly to approach and how much
            # tangential baseline to create, but it cannot learn an action
            # that actively flees its assigned target.  This converts the
            # communication assignment into a kinematic safety constraint
            # rather than a weak hidden-state suggestion.
            coefficients = torch.cat([
                torch.sigmoid(raw_coefficients[..., :1]),
                0.5 * torch.tanh(raw_coefficients[..., 1:]),
            ], dim=-1)
            coefficient_norm = coefficients.norm(dim=-1, keepdim=True)
            coefficients = coefficients / coefficient_norm.clamp_min(1.0)
            candidates = (
                coefficients[..., :1] * radial_direction
                + coefficients[..., 1:] * tangent_direction)
            target_action = torch.sum(
                movement_probs.unsqueeze(-1) * candidates, dim=1)

            target_attention = torch.softmax(
                self.v2_target_attention(
                    self.last_v2_target_latent).squeeze(-1), dim=-1)
            pooled_target = torch.sum(
                target_attention.unsqueeze(-1)
                * self.last_v2_target_latent, dim=1)
            pooled_target = self.v2_target_context(pooled_target)
            movement_gate = torch.sigmoid(self.v2_movement_gate(
                torch.cat([h, pooled_target], dim=-1)))
            base_action = torch.tanh(dp_mean)
            blended_action = torch.clamp(
                (1.0 - movement_gate) * base_action
                + movement_gate * target_action,
                -0.999, 0.999)
            dp_mean = torch.atanh(blended_action)
            self.last_v2_movement_gate = movement_gate.detach()
            self.last_target_movement_candidates = candidates.detach()
            self.last_target_movement_residual = (
                blended_action - base_action).detach()
        self.last_semantic_field_vector = None
        self.last_semantic_field_weights = None
        self.last_semantic_field_delta = None
        if (assignment_context is not None
                and not self._architecture_v2_enabled
                and self._target_allocation_movement_blend > 0.0):
            # Geometry fields 9:11 are the normalized local dx/dy from this UAV
            # to each target.  A straight-through committed target therefore
            # has an immediate, differentiable effect on the physical action,
            # rather than relying on a weak hidden-state residual to discover
            # the connection indirectly.
            target_rel_xy = targets[..., -1][:, :, 9:11]
            committed_rel = torch.sum(
                movement_assignment.unsqueeze(-1) * target_rel_xy, dim=1)
            committed_norm = committed_rel.norm(dim=-1, keepdim=True)
            committed_direction = committed_rel / committed_norm.clamp_min(1e-8)
            committed_direction = torch.where(
                committed_norm > 1e-8,
                committed_direction,
                torch.zeros_like(committed_direction))
            guidance_raw = torch.atanh(
                committed_direction.clamp(-0.999, 0.999))
            blend = self._target_allocation_movement_blend
            self.last_movement_confidence = None
            self.last_effective_movement_blend = None
            if self._target_allocation_movement_confidence_gating_enabled:
                if self.Q > 1:
                    top2 = torch.topk(
                        movement_probs, k=2, dim=-1).values
                    confidence = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
                else:
                    confidence = torch.ones(
                        movement_probs.shape[0], dtype=movement_probs.dtype,
                        device=movement_probs.device)
                confidence_gate = (
                    self._target_allocation_movement_confidence_floor
                    + (1.0 - self._target_allocation_movement_confidence_floor)
                    * confidence.pow(
                        self._target_allocation_movement_confidence_power)
                )
                blend = blend * confidence_gate.unsqueeze(-1)
                self.last_movement_confidence = confidence.detach()
                self.last_effective_movement_blend = blend.detach()
            dp_mean = (1.0 - blend) * dp_mean + blend * guidance_raw
        if (self._semantic_kinematic_field_enabled
                and not self._architecture_v2_enabled
                and assignment_context is not None
                and self.last_peer_claim_load is not None
                and valid_any is not None):
            # ED-SKF: a target with missing peer endpoints attracts assistance;
            # a target already at bistatic capacity exerts no kinematic force.
            # Competition for full targets is suppressed in assignment logits
            # above.  A zero force is important here: actively moving away from
            # an already useful sensing geometry can improve role diversity but
            # catastrophically reduce worst-target P_D.
            target_rel_xy = targets[..., -1][:, :, 9:11]
            desired_endpoints = float(self._sparse_claim_desired_endpoints)
            endpoint_signal = torch.relu(
                desired_endpoints - self.last_peer_claim_load
            ) / desired_endpoints
            field_weights = movement_assignment * endpoint_signal
            target_rel_norm = target_rel_xy.norm(dim=-1, keepdim=True)
            target_direction = target_rel_xy / target_rel_norm.clamp_min(1e-8)
            target_direction = torch.where(
                target_rel_norm > 1e-8,
                target_direction,
                torch.zeros_like(target_direction))
            # Assignment probabilities sum to one and endpoint_signal lies in
            # [0, 1], so the raw weighted vector is already bounded. Preserve
            # its magnitude: weak or conflicting evidence must yield a weak
            # intervention instead of being normalized to full gain.
            field_vector = torch.sum(
                field_weights.unsqueeze(-1) * target_direction, dim=1
            )
            field_norm = field_vector.norm(dim=-1, keepdim=True)
            field_active = (
                valid_any.unsqueeze(-1) & (field_norm > 1e-8))
            field_vector = torch.where(
                field_active, field_vector,
                torch.zeros_like(field_vector))

            # Apply the semantic gradient as a bounded residual to the actual
            # normalized movement action, then map back to the Gaussian mean
            # parameter used by PPO. Silence is exactly neutral.
            base_action = torch.tanh(dp_mean)
            guided_action = torch.clamp(
                base_action
                + self._semantic_kinematic_field_gain * field_vector,
                -0.999, 0.999)
            field_delta = torch.where(
                field_active, guided_action - base_action,
                torch.zeros_like(base_action))
            dp_mean = torch.atanh(torch.clamp(
                base_action + field_delta, -0.999, 0.999))
            self.last_semantic_field_vector = field_vector.detach()
            self.last_semantic_field_weights = field_weights.detach()
            self.last_semantic_field_delta = field_delta.detach()
        if (self._target_conditioned_movement_enabled
                and not self._architecture_v2_enabled
                and assignment_context is not None):
            # CTMH: every communication-refined target entity proposes a local
            # radial/tangential motion. Negotiated responsibilities combine the
            # proposals, retaining a distributed actor at execution time.
            target_rel_xy = targets[..., -1][:, :, 9:11]
            target_rel_norm = target_rel_xy.norm(dim=-1, keepdim=True)
            radial_direction = (
                target_rel_xy / target_rel_norm.clamp_min(1e-8))
            radial_direction = torch.where(
                target_rel_norm > 1e-8,
                radial_direction,
                torch.zeros_like(radial_direction))
            tangent_direction = torch.stack(
                [-radial_direction[..., 1], radial_direction[..., 0]],
                dim=-1)
            target_self = se.expand(-1, self.Q, -1)
            coefficients = torch.tanh(self.target_movement_head(torch.cat(
                [negotiation_te, target_self], dim=-1)))
            coefficient_norm = coefficients.norm(dim=-1, keepdim=True)
            coefficients = coefficients / coefficient_norm.clamp_min(1.0)
            candidates = (
                coefficients[..., :1] * radial_direction
                + coefficients[..., 1:] * tangent_direction)
            movement_residual = torch.sum(
                movement_assignment.unsqueeze(-1) * candidates, dim=1)

            base_action = torch.tanh(dp_mean)
            guided_action = torch.clamp(
                base_action
                + self._target_conditioned_movement_gain * movement_residual,
                -0.999, 0.999)
            movement_delta = guided_action - base_action
            # Preserve bitwise-identical warm-start behavior while the
            # zero-initialized movement head is inactive. A tanh/atanh round
            # trip alone is enough to alter long stochastic trajectories. The
            # straight-through neutral branch keeps its forward value exact but
            # still lets PPO open the zero-initialized head on the first update.
            movement_active = movement_residual.abs().sum(
                dim=-1, keepdim=True) > 0.0
            guided_mean = torch.atanh(guided_action)
            logit_delta = guided_mean - dp_mean
            neutral_straight_through = (
                dp_mean + logit_delta - logit_delta.detach())
            dp_mean = torch.where(
                movement_active, guided_mean, neutral_straight_through)
            self.last_target_movement_candidates = candidates.detach()
            self.last_target_movement_residual = movement_residual.detach()
            self.last_target_movement_delta = movement_delta.detach()
        if self._comm_target_token_enabled:
            # Transmit the learned target-entity tokens themselves. Their
            # semantics are not prescribed; the receiver learns how to use the
            # sender-target token set through cross-attention.
            comm_msg = outgoing_target_tokens.reshape(B, -1)
        else:
            comm_msg = torch.tanh(self.comm_head(h))
        role_logits = self.role_head(h)

        return dp_mean, log_std, role_logits, comm_msg, torch.sigmoid(torch.zeros_like(dp_mean[:, :1])), h_new

    def communication_parameters(self, comm_mean: torch.Tensor):
        """Distribution parameters for learned message content and rate.

        Rate index zero is the silence action.  Message semantics remain fully
        learned; the environment only quantizes the sampled content according
        to the selected rate.
        """
        comm_log_std = torch.clamp(self.comm_log_std, -4.0, 1.0)
        if self._architecture_v2_enabled:
            # One learned uncertainty vector is shared by every target token.
            # Repetition changes only the sampled action width, not the model
            # parameters, so the same actor state dict works for any Q.
            comm_log_std = comm_log_std.repeat(self.Q)
        if self._scale_equivariant_comm_heads_enabled:
            summary = self._pool_local_target_tokens(comm_mean)
            rate_logits = self.comm_set_rate_head(summary)
        else:
            rate_logits = self.comm_rate_head(comm_mean)
        if (self._architecture_v2_enabled
                and self.last_v2_comm_crisis is not None
                and rate_logits.shape[-1] > 1):
            # General value-of-information prior: a node should communicate
            # while its local target set contains a material QoS deficit, but
            # the learned set head may override the prior when communication
            # is costly or redundant.  Centering the two groups preserves the
            # logit scale and works for any number of active precision levels.
            active_margin = self._architecture_v2_comm_prior_gain * (
                self.last_v2_comm_crisis
                - self._architecture_v2_comm_crisis_threshold)
            rate_logits = rate_logits.clone()
            rate_logits[:, 0] = (
                rate_logits[:, 0] - 0.5 * active_margin)
            rate_logits[:, 1:] = (
                rate_logits[:, 1:]
                + 0.5 * active_margin.unsqueeze(-1))
        feedback = self.last_comm_channel_feedback
        if (self._comm_channel_feedback_rate_enabled
                and feedback is not None
                and feedback.ndim == 2
                and feedback.shape[0] == comm_mean.shape[0]):
            rate_logits = (
                rate_logits + self.comm_rate_feedback_head(feedback))
        return comm_log_std, rate_logits

    def _pool_local_target_tokens(
        self, comm_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-pool one sender's target-token set.

        This is a local set operation: the batch row belongs to one UAV and
        contains only that UAV's outgoing target tokens. The result is
        invariant to target ordering and independent of Q.
        """
        expected = self.Q * self.comm_target_token_dim
        if comm_mean.ndim != 2 or comm_mean.shape[-1] != expected:
            raise ValueError(
                'scale-equivariant communication head expected '
                f'{expected} values, got {tuple(comm_mean.shape)}')
        tokens = comm_mean.reshape(
            comm_mean.shape[0], self.Q, self.comm_target_token_dim)
        logits = self.comm_set_attention(tokens).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        self.last_comm_set_attention = weights.detach()
        return torch.sum(weights.unsqueeze(-1) * tokens, dim=1)

    def set_capacity_matching_blend(self, blend: float) -> None:
        """Set the execution coupling used by the slow matching layer."""
        self._capacity_matching_blend = float(np.clip(blend, 0.0, 1.0))

    def set_target_allocation_movement_blend(self, blend: float) -> None:
        """Set slow-commitment authority at a rollout boundary.

        The trainer, rather than ``forward``, owns the schedule so every
        sample in one PPO rollout is generated by the same behaviour map.
        """
        self._target_allocation_movement_blend = float(np.clip(
            blend, 0.0, 1.0))

    def decode_comm_semantics(
        self, target_token_content: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode sender evidence and conditional P_D from latent dimensions.

        The explicit bid at dimension zero is deliberately excluded.  This
        helper has no path to any action head; it becomes behaviourally active
        only when a later, separately gated CA-CSR mechanism consumes it.
        """
        if not self._comm_semantic_decoder_enabled:
            raise RuntimeError('communication semantic decoder is disabled')
        if target_token_content.shape[-1] < self.comm_target_token_dim:
            raise ValueError('target token content is shorter than configured')
        semantic = target_token_content[
            ..., 1:self.comm_target_token_dim]
        decoded = self.comm_semantic_decoder(semantic)
        return decoded[..., 0], torch.sigmoid(decoded[..., 1])

    def isac_resource_parameters(self, comm_mean: torch.Tensor):
        """Logistic-normal parameters for joint power and target allocation.

        The optional responsibility adapter converts locally negotiated target
        probabilities to centered log-odds and adds them to the physical
        sensing logits. Token agreement therefore controls an executed ISAC
        resource instead of remaining an auxiliary representation.
        """
        if self._scale_equivariant_comm_heads_enabled:
            summary = self._pool_local_target_tokens(comm_mean)
            power_mean = self.isac_set_power_head(summary).squeeze(-1)
        else:
            power_mean = self.isac_power_mean_head(comm_mean).squeeze(-1)
        if self._architecture_v2_enabled:
            sensing_mean = self.last_v2_sensing_logits
            if (sensing_mean is None
                    or sensing_mean.shape != (comm_mean.shape[0], self.Q)):
                raise RuntimeError(
                    'architecture V2 resource parameters require a matching '
                    'actor forward pass before sensing allocation')
        else:
            sensing_mean = self.isac_sensing_mean_head(comm_mean)
        comm_sensing_logits = self.last_comm_sensing_logits
        if (self._comm_aided_sensing_enabled
                and comm_sensing_logits is not None
                and comm_sensing_logits.shape == sensing_mean.shape):
            sensing_mean = (
                sensing_mean
                + self._comm_aided_sensing_blend * comm_sensing_logits)
        assignment = self.last_target_assignment
        if (self._target_allocation_resource_blend > 0.0
                and assignment is not None
                and assignment.shape == sensing_mean.shape):
            allocation_logits = torch.log(assignment.clamp_min(1e-8))
            allocation_logits = allocation_logits - allocation_logits.mean(
                dim=-1, keepdim=True)
            sensing_mean = (
                sensing_mean
                + self._target_allocation_resource_blend * allocation_logits)
        power_log_std = torch.clamp(self.isac_power_log_std, -4.0, 1.0)
        sensing_log_std = torch.clamp(
            self.isac_sensing_log_std, -4.0, 1.0)
        if self._architecture_v2_enabled:
            sensing_log_std = sensing_log_std.expand(self.Q)
        return power_mean, power_log_std, sensing_mean, sensing_log_std


# ── Parameter group names for selective plasticity (S1) ──
# These are the canonical module prefixes. Use explicit inclusion (not
# string-exclusion) to avoid silently missing heads like intent_head, dp_log_std.
ENCODER_PARAM_PREFIXES = (
    'self_enc.', 'target_enc.', 'pd_hist_proj.',
    'neighbor_gru.', 'neighbor_proj.', 'global_enc.', 'comm_token_enc.',
    'round_phase_enc.', 'round_phase_equivariant_enc.',
)
HEAD_PARAM_PREFIXES = (
    'dp_head.', 'comm_head.', 'comm_target_token_head.', 'intent_head.',
    'comm_rate_head.', 'comm_log_std',
    'comm_rate_feedback_head.',
    'comm_set_rate_head.', 'comm_set_attention.',
    'isac_power_mean_head.', 'isac_sensing_mean_head.',
    'isac_set_power_head.',
    'isac_power_log_std', 'isac_sensing_log_std',
    'role_head.', 'comm_proj.', 'gate.', 'comm_target_gate.',
    'target_assignment_head.', 'movement_feature_adapter.',
    'movement_commitment_head.', 'allocation_gate.',
    'neighbor_bid_msg_proj.', 'neighbor_bid_target_proj.',
    'round_target_gate.', 'round_peer_claim_head.',
    'comm_sensing_gate.', 'comm_sensing_head.',
    'comm_semantic_decoder.',
    'target_movement_head.',
    'v2_target_policy.', 'v2_assignment_head.', 'v2_sensing_head.',
    'v2_movement_head.', 'v2_target_attention.', 'v2_target_context.',
    'v2_movement_gate.', 'v2_module_router.',
    'v2_coordination_experts.',
    'dp_log_std',  # nn.Parameter, no trailing dot
)
ATTENTION_PARAM_PREFIXES = (
    'attn.', 'attn_norm.', 'comm_cross_attn.', 'comm_cross_norm.',
)


def split_param_groups(named_params):
    """Split parameters into encoder, head, and attention groups.

    Returns:
        enc_params, head_params, attn_params
    """
    enc, head, attn = [], [], []
    for n, p in named_params:
        if n.startswith(ATTENTION_PARAM_PREFIXES):
            attn.append(p)
        elif n.startswith(HEAD_PARAM_PREFIXES):
            head.append(p)
        elif n.startswith(ENCODER_PARAM_PREFIXES):
            enc.append(p)
        else:
            # Unknown params default to encoder (conservative, low LR)
            enc.append(p)
    return enc, head, attn


class CausalContributionPredictor(nn.Module):
    """Amortized sender-token effects on per-target sensing QoS.

    The predictor is used only during centralized training. Its supervision is
    produced by paired simulator interventions that keep the post-transition
    state and random-number stream fixed while removing one sender from every
    receiver inbox. The deployed actor does not call this network.
    """

    def __init__(self, obs_dim: int, comm_dim: int, num_targets: int,
                 num_rate_levels: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.num_targets = int(num_targets)
        hidden = max(32, int(hidden_dim))
        rate_dim = min(16, hidden // 4)
        self.obs_encoder = nn.Sequential(
            nn.Linear(int(obs_dim), hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.message_encoder = nn.Sequential(
            nn.Linear(int(comm_dim), hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.rate_embedding = nn.Embedding(
            max(1, int(num_rate_levels)), rate_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden + rate_dim, hidden),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.effect_head = nn.Linear(hidden // 2, self.num_targets)
        self.apply(self._init_module)
        # Neutral prior before the first intervention batch is fitted.
        nn.init.zeros_(self.effect_head.weight)
        nn.init.zeros_(self.effect_head.bias)

    @staticmethod
    def _init_module(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor, message: torch.Tensor,
                rate_index: torch.Tensor) -> torch.Tensor:
        obs_h = self.obs_encoder(obs)
        msg_h = self.message_encoder(message)
        rate = rate_index.to(dtype=torch.long).clamp(
            min=0, max=self.rate_embedding.num_embeddings - 1)
        rate_h = self.rate_embedding(rate)
        effect = self.effect_head(self.fusion(torch.cat(
            [obs_h, msg_h, rate_h], dim=-1)))
        # A paired one-step difference in detection probability is in [-1, 1].
        return torch.tanh(effect)


class CriticNetwork(nn.Module):
    """MAPPO centralized critic network.

    Input: global state (state_dim + num_agents + comm_dim,)
    Output: scalar V(s) + optional per-target V_q(s)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_layers: list = [256, 256],
        num_agents: int = 4,
        comm_dim: int = 0,
        num_targets: int = 0,
        equivariant_value_critic_enabled: bool = False,
        set_risk_critic_enabled: bool = False,
        risk_hidden_dim: int = 128,
        risk_num_quantiles: int = 16,
        risk_cvar_alpha: float = 0.20,
        risk_monotonic_quantiles_enabled: bool = False,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_agents = num_agents
        self.comm_dim = comm_dim
        self.num_targets = num_targets
        self.equivariant_value_critic_enabled = bool(
            equivariant_value_critic_enabled)
        self.set_risk_critic_enabled = bool(set_risk_critic_enabled)
        self.risk_num_quantiles = max(2, int(risk_num_quantiles))
        self.risk_cvar_alpha = float(np.clip(risk_cvar_alpha, 1e-6, 1.0))
        self.risk_monotonic_quantiles_enabled = bool(
            risk_monotonic_quantiles_enabled)
        input_dim = state_dim + num_agents + comm_dim
        self.shared = None
        self.value_uav_encoder = None
        self.value_target_encoder = None
        self.value_uav_attention = None
        self.value_target_attention = None
        self.value_team_context = None
        self.value_target_context = None
        self.target_value_head = None
        if self.equivariant_value_critic_enabled:
            expected_state_dim = (
                8 * int(num_agents) + 8 * int(num_targets) + 1)
            if int(state_dim) != expected_state_dim:
                raise ValueError(
                    'equivariant value critic requires the centralized '
                    f'global-state layout of width {expected_state_dim}, '
                    f'got {state_dim}')
            value_hidden = max(16, int(hidden_layers[-1]))
            self.value_uav_encoder = nn.Sequential(
                nn.Linear(8, value_hidden),
                nn.LayerNorm(value_hidden),
                nn.SiLU(),
                nn.Linear(value_hidden, value_hidden),
                nn.SiLU(),
            )
            self.value_target_encoder = nn.Sequential(
                nn.Linear(8, value_hidden),
                nn.LayerNorm(value_hidden),
                nn.SiLU(),
                nn.Linear(value_hidden, value_hidden),
                nn.SiLU(),
            )
            self.value_uav_attention = nn.Linear(value_hidden, 1)
            self.value_target_attention = nn.Linear(value_hidden, 1)
            self.value_team_context = nn.Sequential(
                nn.Linear(
                    2 * value_hidden + 1 + int(comm_dim), value_hidden),
                nn.LayerNorm(value_hidden),
                nn.SiLU(),
            )
            self.value_target_context = nn.Sequential(
                nn.Linear(2 * value_hidden, value_hidden),
                nn.LayerNorm(value_hidden),
                nn.SiLU(),
            )
            self.target_value_head = nn.Linear(value_hidden, 1)
            last_dim = value_hidden
        else:
            # Legacy fixed-width value path retained for old checkpoints.
            self.shared = mlp(
                input_dim, hidden_layers, hidden_layers[-1])
            last_dim = hidden_layers[-1]
        self.value_head = nn.Linear(last_dim, 1)
        # CTDE-only action-head credit baselines. These do not enter the actor
        # observation or deployed policy; they only reduce variance for the
        # movement/message/rate/resource PPO objectives.
        self.credit_head_names = ('movement', 'message', 'rate', 'resource')
        self.credit_heads = nn.ModuleList([
            nn.Linear(last_dim, 1) for _ in self.credit_head_names
        ])
        # Per-target value heads (S3: target-wise critic)
        self.target_heads = None
        if num_targets > 0 and not self.equivariant_value_critic_enabled:
            self.target_heads = nn.ModuleList([
                nn.Linear(last_dim, 1) for _ in range(num_targets)
            ])

        # CTDE-only distributional risk critic.  The centralized state has the
        # layout K*[pos, vel, battery, role], Q*[pos, vel], time,
        # Q*uncertainty, Q*P_D.  Shared node encoders plus set pooling make the
        # UAV path permutation invariant and the target outputs permutation
        # equivariant.  In particular, the appended K-long agent one-hot is
        # intentionally ignored; it is useful to the legacy value critic but
        # must not become an identity shortcut for team risk prediction.
        self.risk_uav_encoder = None
        self.risk_target_encoder = None
        self.risk_uav_attention = None
        self.risk_target_attention = None
        self.risk_context = None
        self.risk_quantile_head = None
        self.risk_constraint_head = None
        if self.set_risk_critic_enabled:
            expected_state_dim = (
                8 * int(num_agents) + 8 * int(num_targets) + 1)
            if int(state_dim) != expected_state_dim:
                raise ValueError(
                    'set risk critic requires the centralized global-state '
                    f'layout of width {expected_state_dim}, got {state_dim}')
            risk_hidden_dim = max(16, int(risk_hidden_dim))
            self.risk_uav_encoder = nn.Sequential(
                nn.Linear(8, risk_hidden_dim),
                nn.LayerNorm(risk_hidden_dim),
                nn.ReLU(),
                nn.Linear(risk_hidden_dim, risk_hidden_dim),
                nn.ReLU(),
            )
            # Six target kinematics + uncertainty + previous P_D.
            self.risk_target_encoder = nn.Sequential(
                nn.Linear(8, risk_hidden_dim),
                nn.LayerNorm(risk_hidden_dim),
                nn.ReLU(),
                nn.Linear(risk_hidden_dim, risk_hidden_dim),
                nn.ReLU(),
            )
            self.risk_uav_attention = nn.Linear(risk_hidden_dim, 1)
            self.risk_target_attention = nn.Linear(risk_hidden_dim, 1)
            risk_context_dim = (
                3 * risk_hidden_dim + 1 + int(comm_dim))
            self.risk_context = nn.Sequential(
                nn.Linear(risk_context_dim, risk_hidden_dim),
                nn.LayerNorm(risk_hidden_dim),
                nn.ReLU(),
            )
            quantile_outputs = (
                self.risk_num_quantiles + 1
                if self.risk_monotonic_quantiles_enabled
                else self.risk_num_quantiles)
            self.risk_quantile_head = nn.Linear(
                risk_hidden_dim, quantile_outputs)
            self.risk_constraint_head = nn.Linear(risk_hidden_dim, 1)
        self._init_weights()
        # A random baseline would overwhelm the milliscale delayed message
        # reward on the first PPO update. Start all credit baselines at zero;
        # their supervised return losses learn the appropriate scales.
        for head in self.credit_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def _forward_equivariant_value_features(
        self, state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a centralized state without absolute node identities.

        Returns a permutation-invariant team feature and target-equivariant
        target features.  The deployed actor never calls this CTDE-only path.
        """
        if not self.equivariant_value_critic_enabled:
            raise RuntimeError('equivariant value critic is disabled')
        expected_width = self.state_dim + self.num_agents + self.comm_dim
        if state.ndim != 2 or state.shape[-1] != expected_width:
            raise ValueError(
                'equivariant value critic input must be [base global state, '
                'agent one-hot, communication summary] with width '
                f'{expected_width}')

        batch = state.shape[0]
        base = state[:, :self.state_dim]
        uav_end = 8 * self.num_agents
        target_motion_end = uav_end + 6 * self.num_targets
        uav = base[:, :uav_end].reshape(batch, self.num_agents, 8)
        target_motion = base[:, uav_end:target_motion_end].reshape(
            batch, self.num_targets, 6)
        time_fraction = base[:, target_motion_end:target_motion_end + 1]
        uncertainty_start = target_motion_end + 1
        uncertainty = base[
            :, uncertainty_start:uncertainty_start + self.num_targets]
        pd_start = uncertainty_start + self.num_targets
        previous_pd = base[:, pd_start:pd_start + self.num_targets]
        target = torch.cat([
            target_motion,
            uncertainty.unsqueeze(-1),
            previous_pd.unsqueeze(-1),
        ], dim=-1)

        uav_h = self.value_uav_encoder(uav)
        target_h = self.value_target_encoder(target)
        uav_weight = torch.softmax(
            self.value_uav_attention(uav_h), dim=1)
        target_weight = torch.softmax(
            self.value_target_attention(target_h), dim=1)
        uav_pool = torch.sum(uav_weight * uav_h, dim=1)
        target_pool = torch.sum(target_weight * target_h, dim=1)

        # The absolute-agent one-hot is deliberately skipped.  Communication
        # is already sender-averaged by the trainer.
        comm_start = self.state_dim + self.num_agents
        comm = state[:, comm_start:comm_start + self.comm_dim]
        team_h = self.value_team_context(torch.cat(
            [uav_pool, target_pool, time_fraction, comm], dim=-1))
        target_context = self.value_target_context(torch.cat([
            target_h,
            team_h.unsqueeze(1).expand(-1, self.num_targets, -1),
        ], dim=-1))
        return team_h, target_context

    def forward(self, state: torch.Tensor):
        """Returns scalar V(s) for backward compatibility."""
        if self.equivariant_value_critic_enabled:
            h, _ = self._forward_equivariant_value_features(state)
        else:
            h = self.shared(state)
        return self.value_head(h).squeeze(-1)

    def forward_with_targets(self, state: torch.Tensor):
        """Returns (scalar_v, per_target_v) for S3 diagnostics."""
        target_v = None
        if self.equivariant_value_critic_enabled:
            h, target_h = self._forward_equivariant_value_features(state)
            target_v = self.target_value_head(target_h).squeeze(-1)
        else:
            h = self.shared(state)
        scalar_v = self.value_head(h).squeeze(-1)
        if self.target_heads is not None:
            target_v = torch.stack([head(h).squeeze(-1) for head in self.target_heads], dim=-1)
        return scalar_v, target_v

    def forward_with_credit(self, state: torch.Tensor):
        """Return the legacy scalar value and four action-head baselines."""
        if self.equivariant_value_critic_enabled:
            h, _ = self._forward_equivariant_value_features(state)
        else:
            h = self.shared(state)
        scalar_v = self.value_head(h).squeeze(-1)
        credit_v = torch.stack([
            head(h).squeeze(-1) for head in self.credit_heads
        ], dim=-1)
        return scalar_v, credit_v

    def forward_with_auxiliaries(
        self,
        state: torch.Tensor,
        *,
        include_credit: bool = False,
    ):
        """Return scalar, optional credit, and target values from one encoding.

        All value heads are deterministic readouts of the same critic feature.
        Evaluating that feature once is therefore algebraically identical to
        calling ``forward``/``forward_with_credit`` and
        ``forward_with_targets`` separately, while removing a duplicate shared
        trunk pass from every rollout step.
        """
        if self.equivariant_value_critic_enabled:
            feature, target_feature = (
                self._forward_equivariant_value_features(state))
            target_value = self.target_value_head(
                target_feature).squeeze(-1)
        else:
            feature = self.shared(state)
            target_value = None
            if self.target_heads is not None:
                target_value = torch.stack([
                    head(feature).squeeze(-1)
                    for head in self.target_heads
                ], dim=-1)
        scalar_value = self.value_head(feature).squeeze(-1)
        credit_value = None
        if include_credit:
            credit_value = torch.stack([
                head(feature).squeeze(-1)
                for head in self.credit_heads
            ], dim=-1)
        return scalar_value, credit_value, target_value

    def forward_risk(self, state: torch.Tensor):
        """Return per-target next-P_D quantiles and QoS-violation logits.

        This branch is training-only.  It does not enter the decentralized
        actor forward pass or add any observation at deployment.
        """
        if not self.set_risk_critic_enabled:
            return None
        expected_width = self.state_dim + self.num_agents + self.comm_dim
        if state.ndim != 2 or state.shape[-1] != expected_width:
            raise ValueError(
                'risk critic input must be [base global state, agent one-hot, '
                f'communication summary] with width {expected_width}')

        batch = state.shape[0]
        base = state[:, :self.state_dim]
        uav_end = 8 * self.num_agents
        target_motion_end = uav_end + 6 * self.num_targets
        uav = base[:, :uav_end].reshape(batch, self.num_agents, 8)
        target_motion = base[:, uav_end:target_motion_end].reshape(
            batch, self.num_targets, 6)
        time_fraction = base[:, target_motion_end:target_motion_end + 1]
        uncertainty_start = target_motion_end + 1
        uncertainty = base[
            :, uncertainty_start:uncertainty_start + self.num_targets]
        pd_start = uncertainty_start + self.num_targets
        previous_pd = base[:, pd_start:pd_start + self.num_targets]
        target = torch.cat([
            target_motion,
            uncertainty.unsqueeze(-1),
            previous_pd.unsqueeze(-1),
        ], dim=-1)

        uav_h = self.risk_uav_encoder(uav)
        target_h = self.risk_target_encoder(target)
        uav_weight = torch.softmax(
            self.risk_uav_attention(uav_h), dim=1)
        target_weight = torch.softmax(
            self.risk_target_attention(target_h), dim=1)
        uav_pool = torch.sum(uav_weight * uav_h, dim=1)
        target_pool = torch.sum(target_weight * target_h, dim=1)

        # The communication summary is already averaged over senders by the
        # trainer.  Ignore the intervening absolute-agent one-hot block.
        comm_start = self.state_dim + self.num_agents
        comm = state[:, comm_start:comm_start + self.comm_dim]
        shared_context = torch.cat(
            [uav_pool, target_pool, time_fraction, comm], dim=-1)
        shared_context = shared_context.unsqueeze(1).expand(
            -1, self.num_targets, -1)
        context = self.risk_context(torch.cat(
            [target_h, shared_context], dim=-1))
        quantile_raw = self.risk_quantile_head(context)
        if self.risk_monotonic_quantiles_enabled:
            # A simplex over N+1 non-negative spacings parameterizes N ordered
            # points strictly inside [0, 1].  This prevents quantile crossing
            # structurally instead of relying on an auxiliary penalty.
            spacing = torch.softmax(quantile_raw, dim=-1)
            quantiles = torch.cumsum(spacing, dim=-1)[
                ..., :self.risk_num_quantiles]
        else:
            quantiles = quantile_raw
        constraint_logits = self.risk_constraint_head(context).squeeze(-1)
        return quantiles, constraint_logits

    def risk_cvar(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Lower-tail next-P_D estimate for each target."""
        outputs = self.forward_risk(state)
        if outputs is None:
            return None
        quantiles, _ = outputs
        tail_count = max(
            1, int(np.ceil(self.risk_cvar_alpha * self.risk_num_quantiles)))
        ordered = torch.sort(quantiles, dim=-1).values
        return ordered[..., :tail_count].mean(dim=-1)
