from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .e11_losses import bounded_residual_score, pairwise_ranking_loss
from .e12_model import RollbackTransformerOutput


@dataclass
class E12LossOutput:
    total: torch.Tensor
    setwise: torch.Tensor
    quality: torch.Tensor
    restore_score: torch.Tensor
    edge: torch.Tensor
    safe_score: torch.Tensor
    safe_ranking: torch.Tensor
    delta_reg: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "setwise": float(self.setwise.detach()),
            "quality": float(self.quality.detach()),
            "restore_score": float(self.restore_score.detach()),
            "edge": float(self.edge.detach()),
            "safe_score": float(self.safe_score.detach()),
            "safe_ranking": float(self.safe_ranking.detach()),
            "delta_reg": float(self.delta_reg.detach()),
        }


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    if not mask.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[mask], target[mask], beta=beta)


def _balanced_restore_score_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    negative_weight: float,
) -> torch.Tensor:
    if not mask.any():
        return prediction.sum() * 0.0
    element = F.smooth_l1_loss(prediction, target, beta=0.1, reduction="none")
    positive = mask & (target > 0)
    negative = mask & ~positive
    terms: list[torch.Tensor] = []
    if positive.any():
        terms.append(element[positive].mean())
    if negative.any():
        terms.append(float(negative_weight) * element[negative].mean())
    if not terms:
        return prediction.sum() * 0.0
    return torch.stack(terms).sum()


def compute_e12_loss(
    output: RollbackTransformerOutput,
    batch: dict[str, torch.Tensor],
    config: dict[str, float],
) -> E12LossOutput:
    rollback_mask = batch["rollback_mask"]
    safe_mask = batch["safe_mask"]
    alpha = float(config.get("residual_alpha", 0.5))

    setwise_per_image = F.cross_entropy(
        output.restore_logits,
        batch["restore_target_index"],
        label_smoothing=float(config.get("label_smoothing", 0.0)),
        reduction="none",
    )
    target_is_none = batch["restore_target_index"] == 0
    setwise_weight = torch.where(
        target_is_none,
        torch.full_like(setwise_per_image, float(config.get("none_image_weight", 0.25))),
        torch.full_like(setwise_per_image, float(config.get("restore_image_weight", 1.0))),
    )
    setwise_loss = (setwise_per_image * setwise_weight).sum() / setwise_weight.sum().clamp_min(1e-8)

    quality_loss = _masked_smooth_l1(
        output.rollback_quality,
        batch["rollback_iou_target"],
        rollback_mask,
    )
    restore_score_prediction = torch.sigmoid(output.restore_score_logit)
    restore_score_loss = _balanced_restore_score_loss(
        restore_score_prediction,
        batch["restore_score_target"],
        rollback_mask,
        negative_weight=float(config.get("restore_score_negative_weight", 0.25)),
    )

    if rollback_mask.any():
        edge_loss = F.cross_entropy(
            output.edge_logits[rollback_mask], batch["edge_target"][rollback_mask]
        )
    else:
        edge_loss = output.edge_logits.sum() * 0.0

    adjusted_safe_score = bounded_residual_score(
        batch["safe_scores"], output.safe_score_delta, alpha
    )
    safe_score_loss = _masked_smooth_l1(
        adjusted_safe_score,
        batch["safe_iou_target"],
        safe_mask,
    )
    safe_ranking_loss = pairwise_ranking_loss(
        adjusted_safe_score,
        batch["safe_rank_positive"] & safe_mask,
        batch["safe_rank_negative"] & safe_mask,
        margin=float(config.get("ranking_margin", 0.1)),
    )
    if safe_mask.any():
        delta_reg = torch.tanh(output.safe_score_delta[safe_mask]).pow(2).mean()
    else:
        delta_reg = output.safe_score_delta.sum() * 0.0

    total = (
        float(config.get("set_weight", 1.0)) * setwise_loss
        + float(config.get("quality_weight", 1.0)) * quality_loss
        + float(config.get("restore_score_weight", 1.0)) * restore_score_loss
        + float(config.get("edge_weight", 0.25)) * edge_loss
        + float(config.get("safe_score_weight", 0.5)) * safe_score_loss
        + float(config.get("safe_ranking_weight", 0.5)) * safe_ranking_loss
        + float(config.get("delta_reg_weight", 0.02)) * delta_reg
    )
    return E12LossOutput(
        total=total,
        setwise=setwise_loss,
        quality=quality_loss,
        restore_score=restore_score_loss,
        edge=edge_loss,
        safe_score=safe_score_loss,
        safe_ranking=safe_ranking_loss,
        delta_reg=delta_reg,
    )
