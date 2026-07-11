from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import ArbitratorOutput


@dataclass
class LossOutput:
    total: torch.Tensor
    valid: torch.Tensor
    iou: torch.Tensor
    pair: torch.Tensor
    rescue: torch.Tensor
    protect: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "valid": float(self.valid.detach()),
            "iou": float(self.iou.detach()),
            "pair": float(self.pair.detach()),
            "rescue": float(self.rescue.detach()),
            "protect": float(self.protect.detach()),
        }


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    logits = logits[mask]
    target = target[mask]
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1 - probability) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


def compute_loss(
    output: ArbitratorOutput,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> LossOutput:
    token_mask = batch["mask"]
    modality = batch["modality"]

    valid_loss = focal_bce_with_logits(
        output.valid_logit,
        batch["valid_target"],
        token_mask,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    if token_mask.any():
        iou_loss = F.smooth_l1_loss(
            output.iou_pred[token_mask],
            batch["iou_target"][token_mask],
            beta=0.1,
        )
    else:
        iou_loss = output.iou_pred.sum() * 0.0

    pair_loss = focal_bce_with_logits(
        output.pair_same_logit,
        batch["pair_same_target"],
        batch["pair_mask"],
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    rescue_mask = token_mask & (modality == 1)
    rescue_loss = focal_bce_with_logits(
        output.rescue_logit,
        batch["rescue_target"],
        rescue_mask,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    protect_mask = token_mask & (modality == 0)
    protect_loss = focal_bce_with_logits(
        output.protect_logit,
        batch["protect_target"],
        protect_mask,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    total = (
        float(weights.get("valid_weight", 1.0)) * valid_loss
        + float(weights.get("iou_weight", 1.0)) * iou_loss
        + float(weights.get("pair_weight", 1.0)) * pair_loss
        + float(weights.get("rescue_weight", 1.0)) * rescue_loss
        + float(weights.get("protect_weight", 1.0)) * protect_loss
    )
    return LossOutput(
        total=total,
        valid=valid_loss,
        iou=iou_loss,
        pair=pair_loss,
        rescue=rescue_loss,
        protect=protect_loss,
    )
