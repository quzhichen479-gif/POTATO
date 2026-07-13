from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .routing import SRTHOutput


@dataclass
class SRTHLossOutput:
    total: torch.Tensor
    route: torch.Tensor
    prior_consistency: torch.Tensor
    smoothness: torch.Tensor
    gate_budget: torch.Tensor


def _zero_like_output(output: SRTHOutput) -> torch.Tensor:
    first = next(iter(output.route_logits.values()))
    return first.sum() * 0.0


def _masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask.expand_as(logits)
    if mask.shape != logits.shape or target.shape != logits.shape:
        raise ValueError("route logits, targets and expanded masks must share a shape")
    if not mask.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask])


def _total_variation(value: torch.Tensor) -> torch.Tensor:
    horizontal = torch.abs(value[..., :, 1:] - value[..., :, :-1]).mean()
    vertical = torch.abs(value[..., 1:, :] - value[..., :-1, :]).mean()
    return horizontal + vertical


def compute_srth_v1_loss(
    output: SRTHOutput,
    route_targets: Mapping[str, torch.Tensor],
    route_masks: Mapping[str, torch.Tensor],
    *,
    route_weight: float = 1.0,
    prior_consistency_weight: float = 0.15,
    smoothness_weight: float = 0.02,
    gate_budget_weight: float = 0.02,
    max_gate_mean: float = 0.65,
) -> SRTHLossOutput:
    """Compute SRTH auxiliary losses; the detector loss remains owned by YOLO26."""
    if not 0 <= max_gate_mean <= 1:
        raise ValueError("max_gate_mean must be in [0, 1]")

    route_terms: list[torch.Tensor] = []
    prior_terms: list[torch.Tensor] = []
    smoothness_terms: list[torch.Tensor] = []
    budget_terms: list[torch.Tensor] = []

    for level, logits in output.route_logits.items():
        if level not in route_targets or level not in route_masks:
            raise KeyError(f"missing route target or mask for level {level}")
        target = route_targets[level].to(device=logits.device, dtype=logits.dtype)
        mask = route_masks[level].to(device=logits.device, dtype=torch.bool)
        route_terms.append(_masked_bce_with_logits(logits, target, mask))
        prior_terms.append(_masked_bce_with_logits(output.prior_logits[level], target, mask))
        gates = output.gates[level]
        smoothness_terms.append(_total_variation(gates))
        budget_terms.append(F.relu(gates.mean() - max_gate_mean).square())

    if not route_terms:
        zero = _zero_like_output(output)
        return SRTHLossOutput(zero, zero, zero, zero, zero)

    route = torch.stack(route_terms).mean()
    prior_consistency = torch.stack(prior_terms).mean()
    smoothness = torch.stack(smoothness_terms).mean()
    gate_budget = torch.stack(budget_terms).mean()
    total = (
        route_weight * route
        + prior_consistency_weight * prior_consistency
        + smoothness_weight * smoothness
        + gate_budget_weight * gate_budget
    )
    return SRTHLossOutput(
        total=total,
        route=route,
        prior_consistency=prior_consistency,
        smoothness=smoothness,
        gate_budget=gate_budget,
    )
