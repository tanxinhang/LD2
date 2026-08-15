"""Shared finite-round factor-graph coordinator for Gate C1.

The network reasons over UAV nodes, target nodes and directed candidate
hyperedges ``(tx, rx, target)``.  Parameters are shared across all entities and
rounds, so the model does not depend on a fixed UAV/target identity layout.
The decoder is deliberately lightweight: it only enforces roles, unique owner
selection and receiver capacity.  It never calls a dynamic program or MILP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def factor_graph_edge_features(
    edge_value: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Build scale-stable, permutation-equivariant candidate-edge features."""
    if edge_value.ndim != 4 or candidate_mask.shape != edge_value.shape:
        raise ValueError("edge_value/mask must have shape (B,K,K,Q)")
    if edge_value.shape[1] != edge_value.shape[2]:
        raise ValueError("edge_value must have equal UAV axes")
    mask = candidate_mask.to(dtype=torch.bool)
    value = torch.where(mask, edge_value.clamp_min(0), 0.0)
    log_value = torch.log1p(value)
    eps = torch.finfo(log_value.dtype).eps
    global_scale = log_value.amax(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    target_scale = log_value.amax(dim=(1, 2), keepdim=True).clamp_min(eps)
    tx_scale = log_value.amax(dim=(2, 3), keepdim=True).clamp_min(eps)
    rx_scale = log_value.amax(dim=(1, 3), keepdim=True).clamp_min(eps)
    features = torch.stack(
        (
            log_value / global_scale,
            log_value / target_scale,
            log_value / tx_scale,
            log_value / rx_scale,
        ),
        dim=-1,
    )
    return torch.where(mask[..., None], features, 0.0)


def _masked_mean_max(
    value: torch.Tensor,
    mask: torch.Tensor,
    dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = mask[..., None]
    count = expanded.sum(dim=dims).clamp_min(1)
    mean = torch.where(expanded, value, 0.0).sum(dim=dims) / count
    maximum = value.masked_fill(~expanded, -torch.inf).amax(dim=dims)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    return mean, maximum


@dataclass(frozen=True)
class FactorGraphOutput:
    edge_logits: torch.Tensor
    owner_logits: torch.Tensor
    role_logits: torch.Tensor
    stop_logit: torch.Tensor
    round_edge_logits: tuple[torch.Tensor, ...]


class FiniteRoundFactorGraphCoordinator(nn.Module):
    """Permutation-equivariant coordinator with shared recurrent updates."""

    def __init__(
        self,
        *,
        edge_feature_dim: int = 4,
        hidden_dim: int = 64,
        rounds: int = 3,
        coupling_strength: float = 1.0,
        coupling_rounds: int = 0,
        use_global_context: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.rounds = max(1, int(rounds))
        self.coupling_strength = float(coupling_strength)
        self.coupling_rounds = max(0, int(coupling_rounds))
        self.use_global_context = bool(use_global_context)
        H = self.hidden_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(int(edge_feature_dim), H), nn.SiLU(), nn.Linear(H, H))
        self.uav_encoder = nn.Sequential(
            nn.Linear(4 * H, H), nn.SiLU(), nn.Linear(H, H))
        self.target_encoder = nn.Sequential(
            nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, H))
        self.edge_update = nn.Sequential(
            nn.Linear(4 * H, 2 * H), nn.SiLU(), nn.Linear(2 * H, H))
        self.uav_update = nn.Sequential(
            nn.Linear(5 * H, 2 * H), nn.SiLU(), nn.Linear(2 * H, H))
        self.target_update = nn.Sequential(
            nn.Linear(3 * H, 2 * H), nn.SiLU(), nn.Linear(2 * H, H))
        if self.use_global_context:
            self.uav_global_update = nn.Sequential(
                nn.Linear(3 * H, H), nn.SiLU(), nn.Linear(H, H))
            self.target_global_update = nn.Sequential(
                nn.Linear(3 * H, H), nn.SiLU(), nn.Linear(H, H))
        self.edge_head = nn.Linear(H, 1)
        self.role_head = nn.Linear(H, 3)  # idle, Tx, Rx.
        self.owner_head = nn.Sequential(
            nn.Linear(4 * H, H), nn.SiLU(), nn.Linear(H, 1))
        self.no_owner_head = nn.Linear(H, 1)
        self.stop_head = nn.Sequential(
            nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, 1))

    @staticmethod
    def _pool_entities(
        edge: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tx_mean, tx_max = _masked_mean_max(edge, mask, (2, 3))
        rx_mean, rx_max = _masked_mean_max(edge, mask, (1, 3))
        uav = torch.cat((tx_mean, tx_max, rx_mean, rx_max), dim=-1)
        target_mean, target_max = _masked_mean_max(edge, mask, (1, 2))
        target = torch.cat((target_mean, target_max), dim=-1)
        return uav, target

    def _heads(
        self,
        edge: torch.Tensor,
        uav: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _, Q, H = edge.shape
        raw_edge_logits = self.edge_head(edge).squeeze(-1).masked_fill(
            ~mask, -20.0)
        base_role_logits = self.role_head(uav)
        incoming_mean, incoming_max = _masked_mean_max(edge, mask, (1,))
        # incoming pools are (B,rx,Q,H); owner logits are (B,Q,rx).
        incoming_mean = incoming_mean.permute(0, 2, 1, 3)
        incoming_max = incoming_max.permute(0, 2, 1, 3)
        target_expand = target[:, :, None, :].expand(B, Q, K, H)
        uav_expand = uav[:, None, :, :].expand(B, Q, K, H)
        owner_feature = torch.cat(
            (target_expand, uav_expand, incoming_mean, incoming_max), dim=-1)
        owner = self.owner_head(owner_feature).squeeze(-1)
        receiver_available = mask.any(dim=1).permute(0, 2, 1)
        owner = owner.masked_fill(~receiver_available, -20.0)
        no_owner = self.no_owner_head(target)
        base_owner_logits = torch.cat((owner, no_owner), dim=-1)
        role_logits = base_role_logits
        owner_logits = base_owner_logits

        def standardized(
            support: torch.Tensor,
            available: torch.Tensor,
            dimension: int,
        ) -> torch.Tensor:
            weight = available.to(support.dtype)
            count = weight.sum(dim=dimension, keepdim=True).clamp_min(1.0)
            mean = (support * weight).sum(
                dim=dimension, keepdim=True) / count
            variance = (((support - mean) * weight) ** 2).sum(
                dim=dimension, keepdim=True) / count
            return torch.where(
                available,
                (support - mean) / torch.sqrt(variance + 1.0e-4),
                0.0,
            )

        # Shared mean-field rounds exchange target-owner evidence and role
        # support. They are linear in the sparse dense-layout candidate table,
        # unlike enumerating all Tx/Rx partitions.
        receiver_available_kq = receiver_available.permute(0, 2, 1)
        tx_available = mask.any(dim=(2, 3))
        rx_available = mask.any(dim=(1, 3))
        for _ in range(self.coupling_rounds):
            role_log_probability = torch.log_softmax(role_logits, dim=-1)
            incoming = raw_edge_logits + self.coupling_strength * (
                role_log_probability[:, :, None, None, 1])
            incoming = incoming.masked_fill(~mask, -torch.inf)
            incoming = torch.logsumexp(incoming, dim=1)
            incoming = torch.where(
                receiver_available_kq, incoming, torch.zeros_like(incoming))
            incoming = standardized(
                incoming, receiver_available_kq, dimension=1)
            owner_logits = torch.cat((
                base_owner_logits[..., :-1]
                + self.coupling_strength * (
                    role_log_probability[:, None, :, 2]
                    + incoming.permute(0, 2, 1)),
                base_owner_logits[..., -1:],
            ), dim=-1)
            owner_log_probability = torch.log_softmax(owner_logits, dim=-1)
            owner_edge = owner_log_probability[..., :-1].permute(
                0, 2, 1)[:, None, :, :]
            tx_support = (raw_edge_logits + owner_edge
                          + role_log_probability[:, None, :, None, 2])
            tx_support = tx_support.masked_fill(~mask, -torch.inf)
            tx_support = torch.logsumexp(tx_support, dim=(2, 3))
            tx_support = torch.where(
                tx_available, tx_support, torch.zeros_like(tx_support))
            tx_support = standardized(
                tx_support, tx_available, dimension=1)
            rx_support = (raw_edge_logits + owner_edge
                          + role_log_probability[:, :, None, None, 1])
            rx_support = rx_support.masked_fill(~mask, -torch.inf)
            rx_support = torch.logsumexp(rx_support, dim=(1, 3))
            rx_support = torch.where(
                rx_available, rx_support, torch.zeros_like(rx_support))
            rx_support = standardized(
                rx_support, rx_available, dimension=1)
            role_logits = torch.stack((
                base_role_logits[..., 0],
                base_role_logits[..., 1]
                + self.coupling_strength * tx_support,
                base_role_logits[..., 2]
                + self.coupling_strength * rx_support,
            ), dim=-1)

        # Roles, owners and edges form one structured decision. The final
        # joint edge logit is used by both supervision and lightweight decode.
        role_log_probability = torch.log_softmax(role_logits, dim=-1)
        if self.coupling_rounds == 0:
            owner_logits = torch.cat((
                base_owner_logits[..., :-1]
                + self.coupling_strength
                * role_log_probability[:, None, :, 2],
                base_owner_logits[..., -1:],
            ), dim=-1)
        owner_log_probability = torch.log_softmax(owner_logits, dim=-1)
        edge_logits = raw_edge_logits + self.coupling_strength * (
            role_log_probability[:, :, None, None, 1]
            + role_log_probability[:, None, :, None, 2]
            + owner_log_probability[..., :-1].permute(0, 2, 1)[:, None, :, :]
        )
        edge_logits = edge_logits.masked_fill(~mask, -20.0)
        stop_logit = self.stop_head(torch.cat(
            (uav.mean(dim=1), target.mean(dim=1)), dim=-1)).squeeze(-1)
        return edge_logits, owner_logits, role_logits, stop_logit

    def forward(
        self,
        edge_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> FactorGraphOutput:
        if edge_features.ndim != 5:
            raise ValueError("edge_features must have shape (B,K,K,Q,F)")
        mask = candidate_mask.to(dtype=torch.bool)
        if edge_features.shape[:-1] != mask.shape:
            raise ValueError("edge feature/mask shapes do not align")
        edge = self.edge_encoder(edge_features)
        edge = torch.where(mask[..., None], edge, 0.0)
        uav_feature, target_feature = self._pool_entities(edge, mask)
        uav = self.uav_encoder(uav_feature)
        target = self.target_encoder(target_feature)
        round_logits: list[torch.Tensor] = []
        for _ in range(self.rounds):
            B, K, _, Q, H = edge.shape
            tx = uav[:, :, None, None, :].expand(B, K, K, Q, H)
            rx = uav[:, None, :, None, :].expand(B, K, K, Q, H)
            target_edge = target[:, None, None, :, :].expand(B, K, K, Q, H)
            edge = edge + self.edge_update(torch.cat(
                (edge, tx, rx, target_edge), dim=-1))
            edge = torch.where(mask[..., None], edge, 0.0)
            uav_feature, target_feature = self._pool_entities(edge, mask)
            uav = uav + self.uav_update(torch.cat(
                (uav, uav_feature), dim=-1))
            target = target + self.target_update(torch.cat(
                (target, target_feature), dim=-1))
            if self.use_global_context:
                target_context = torch.cat((
                    target.mean(dim=1),
                    target.amax(dim=1),
                    target.amin(dim=1),
                ), dim=-1)
                uav_context = torch.cat((
                    uav.mean(dim=1),
                    uav.amax(dim=1),
                    uav.amin(dim=1),
                ), dim=-1)
                uav = uav + self.uav_global_update(
                    target_context)[:, None, :]
                target = target + self.target_global_update(
                    uav_context)[:, None, :]
            round_logits.append(self.edge_head(edge).squeeze(-1).masked_fill(
                ~mask, -20.0))
        edge_logits, owner_logits, role_logits, stop_logit = self._heads(
            edge, uav, target, mask)
        return FactorGraphOutput(
            edge_logits=edge_logits,
            owner_logits=owner_logits,
            role_logits=role_logits,
            stop_logit=stop_logit,
            round_edge_logits=tuple(round_logits),
        )


@dataclass(frozen=True)
class LightweightProjectionResult:
    proposal: np.ndarray
    selected: np.ndarray
    role: np.ndarray
    owner: np.ndarray
    edge_change_rate: float
    score_delta: float
    role_fallback: bool
    owner_repairs: int


def _top_edges(
    score: np.ndarray,
    mask: np.ndarray,
    receiver: int,
    target: int,
    transmitters: np.ndarray,
    limit: int,
) -> list[tuple[int, int, int]]:
    values = [
        (float(score[tx, receiver, target]), int(tx))
        for tx in transmitters
        if tx != receiver and mask[tx, receiver, target]
    ]
    values.sort(key=lambda item: (-item[0], item[1]))
    return [
        (tx, int(receiver), int(target))
        for _, tx in values[:max(1, int(limit))]
    ]


def decode_lightweight_feasible(
    edge_score: np.ndarray,
    owner_logits: np.ndarray,
    role_logits: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    target_pair_limit: int,
    reports_per_receiver: int,
    edge_value: np.ndarray | None = None,
) -> LightweightProjectionResult:
    """Decode and locally repair a proposal without combinatorial search."""
    score = np.asarray(edge_score, dtype=np.float64)
    owner_score = np.asarray(owner_logits, dtype=np.float64)
    role_score = np.asarray(role_logits, dtype=np.float64)
    mask = np.asarray(candidate_mask, dtype=bool)
    if score.shape != mask.shape or score.ndim != 3:
        raise ValueError("edge score/mask must have shape (K,K,Q)")
    K, K2, Q = score.shape
    if K != K2 or owner_score.shape != (Q, K + 1):
        raise ValueError("owner logits must have shape (Q,K+1)")
    if role_score.shape != (K, 3):
        raise ValueError("role logits must have shape (K,3)")
    raw_role = np.argmax(role_score, axis=-1).astype(np.int8)
    raw_owner = np.argmax(owner_score, axis=-1).astype(np.int64)
    proposal = np.zeros_like(mask)
    proposed_tx = np.flatnonzero(raw_role == 1)
    for target in range(Q):
        receiver = int(raw_owner[target])
        if receiver >= K or raw_role[receiver] != 2:
            continue
        for edge in _top_edges(
                score, mask, receiver, target, proposed_tx,
                int(target_pair_limit)):
            proposal[edge] = True

    role = raw_role.copy()
    role_fallback = not (np.any(role == 1) and np.any(role == 2))
    selected = np.zeros_like(mask)
    owner = np.full(Q, -1, dtype=np.int64)
    owner_repairs = 0
    if not role_fallback:
        tx_nodes = np.flatnonzero(role == 1)
        rx_nodes = np.flatnonzero(role == 2)
        capacity = np.full(K, max(1, int(reports_per_receiver)), dtype=int)
        confidence = np.max(owner_score[:, :K], axis=-1)
        for target in sorted(range(Q), key=lambda q: (-confidence[q], q)):
            if raw_owner[target] >= K:
                continue
            receivers = sorted(
                (int(rx) for rx in rx_nodes),
                key=lambda rx: (-owner_score[target, rx], rx),
            )
            chosen_edges: list[tuple[int, int, int]] = []
            chosen_receiver = -1
            for receiver in receivers:
                available = min(
                    max(capacity[receiver], 0), max(1, int(target_pair_limit)))
                if available <= 0:
                    continue
                incoming = _top_edges(
                    score, mask, receiver, target, tx_nodes, available)
                if incoming:
                    chosen_receiver = receiver
                    chosen_edges = incoming
                    break
            if chosen_receiver < 0:
                continue
            owner_repairs += int(chosen_receiver != int(raw_owner[target]))
            owner[target] = chosen_receiver
            for edge in chosen_edges:
                selected[edge] = True
            capacity[chosen_receiver] -= len(chosen_edges)

    union = proposal | selected
    edge_change_rate = float(
        np.sum(proposal ^ selected) / max(int(np.sum(union)), 1))
    value = score if edge_value is None else np.asarray(
        edge_value, dtype=np.float64)
    if value.shape != score.shape:
        raise ValueError("edge_value must match edge_score")
    score_delta = float(np.sum(value * selected) - np.sum(value * proposal))
    return LightweightProjectionResult(
        proposal=proposal,
        selected=selected,
        role=role,
        owner=owner,
        edge_change_rate=edge_change_rate,
        score_delta=score_delta,
        role_fallback=role_fallback,
        owner_repairs=owner_repairs,
    )


def assert_hard_structure_invariants(
    selected: np.ndarray,
    *,
    reports_per_receiver: int,
) -> None:
    """Raise when a decoded structure violates a Gate C1 hard invariant."""
    pair = np.asarray(selected, dtype=bool)
    if pair.ndim != 3 or pair.shape[0] != pair.shape[1]:
        raise ValueError("selected must have shape (K,K,Q)")
    K = pair.shape[0]
    if np.any(pair[np.arange(K), np.arange(K), :]):
        raise AssertionError("self edges are forbidden")
    tx_used = np.any(pair, axis=(1, 2))
    rx_used = np.any(pair, axis=(0, 2))
    if np.any(tx_used & rx_used):
        raise AssertionError("a UAV cannot be both Tx and Rx")
    receiver_target = np.any(pair, axis=0)
    if np.any(np.sum(receiver_target, axis=0) > 1):
        raise AssertionError("each target must have a unique receiver owner")
    if np.any(np.sum(pair, axis=(0, 2)) > int(reports_per_receiver)):
        raise AssertionError("receiver report capacity exceeded")
