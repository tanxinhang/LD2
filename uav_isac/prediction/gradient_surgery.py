"""Gradient conflict handling for constraint-native multi-task training."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def physical_anchor_pcgrad(
    structure_loss: torch.Tensor,
    physical_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    epsilon: float = 1.0e-12,
) -> float:
    """Assign a summed gradient that never opposes the physical task.

    When the shared structural gradient has a negative dot product with the
    physical gradient, its conflicting component is projected away. Parameters
    owned by only one task retain their original gradients. The pre-projection
    cosine similarity is returned for diagnostics.
    """
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    structure_grad = torch.autograd.grad(
        structure_loss, trainable, allow_unused=True)
    physical_grad = torch.autograd.grad(
        physical_loss, trainable, allow_unused=True)
    shared = [
        (left, right)
        for left, right in zip(structure_grad, physical_grad)
        if left is not None and right is not None
    ]
    if shared:
        dot = sum(torch.sum(left * right) for left, right in shared)
        structure_norm = sum(torch.sum(left * left) for left, _ in shared)
        physical_norm = sum(torch.sum(right * right) for _, right in shared)
        cosine = dot / torch.sqrt(torch.clamp(
            structure_norm * physical_norm, min=float(epsilon)))
        coefficient = torch.minimum(
            dot / torch.clamp(physical_norm, min=float(epsilon)),
            torch.zeros_like(dot),
        )
    else:
        cosine = torch.zeros((), device=structure_loss.device)
        coefficient = torch.zeros((), device=structure_loss.device)
    for parameter, left, right in zip(
        trainable, structure_grad, physical_grad
    ):
        if left is None:
            combined = right
        elif right is None:
            combined = left
        else:
            combined = left - coefficient * right + right
        parameter.grad = None if combined is None else combined.detach()
    return float(cosine.detach())
