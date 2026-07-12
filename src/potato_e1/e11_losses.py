from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .e11_model import ResidualArbitratorOutput
from .losses import focal_bce_with_logits


@dataclass
class E11LossOutput:
    total: torch.Tensor
    quality: torch.Tensor
    score: torch.Tensor
    ranking: torch.Tensor
    rescue: torch.Tensor
    delta_reg: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "quality": float(self.quality.detach()),
            "score": float(self.score.detach()),
            "ranking": float(self.ranking.detach()),
            "rescue": float(self.rescue.detach()),
            "delta_reg": float(self.delta_reg.detach()),
        }


def bounded_residual_score(
    raw_score: torch.Tensor,
    score_delta: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply a bounded correction in logit space.

    The correction magnitude cannot exceed ``alpha`` regardless of the network output.
    """
    score = raw_score.clamp(1e-5, 1 - 1e-5)
    logit = torch.log(score) - torch.log1p(-score)
    return torch.sigmoid(logit + float(alpha) * torch.tanh(score_delta))


def pairwise_ranking_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for batch_index in range(scores.shape[0]):
        positive = scores[batch_index][positive_mask[batch_index]]
        negative = scores[batch_index][negative_mask[batch_index]]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        pair_loss = F.relu(float(margin) - positive[:, None] + negative[None, :])
        losses.append(pair_loss.mean())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def compute_e11_loss(
    output: ResidualArbitratorOutput,
    batch: dict[str, torch.Tensor],
    config: dict[str, float],
) -> E11LossOutput:
    token_mask = batch["mask"]
    base_keep = batch["base_keep_mask"] & token_mask
    admission_mask = batch["e11_admission_mask"] & token_mask
    alpha = float(config.get("residual_alpha", 0.5))

    if token_mask.any():
        quality_loss = F.smooth_l1_loss(
            output.iou_pred[token_mask], batch["iou_target"][token_mask], beta=0.1
        )
    else:
        quality_loss = output.iou_pred.sum() * 0.0

    adjusted_score = bounded_residual_score(batch["scores"], output.score_delta, alpha)
    if base_keep.any():
        score_loss = F.smooth_l1_loss(
            adjusted_score[base_keep], batch["iou_target"][base_keep], beta=0.1
        )
        delta_reg = torch.tanh(output.score_delta[base_keep]).pow(2).mean()
    else:
        score_loss = adjusted_score.sum() * 0.0
        delta_reg = output.score_delta.sum() * 0.0

    ranking_loss = pairwise_ranking_loss(
        adjusted_score,
        batch["e11_rank_positive"] & token_mask,
        batch["e11_rank_negative"] & token_mask,
        margin=float(config.get("ranking_margin", 0.1)),
    )

    rescue_loss = focal_bce_with_logits(
        output.rescue_logit,
        batch["e11_rescue_target"],
        admission_mask,
        alpha=float(config.get("focal_alpha", 0.25)),
        gamma=float(config.get("focal_gamma", 2.0)),
    )

    total = (
        float(config.get("quality_weight", 1.0)) * quality_loss
        + float(config.get("score_weight", 1.0)) * score_loss
        + float(config.get("ranking_weight", 1.0)) * ranking_loss
        + float(config.get("rescue_weight", 1.0)) * rescue_loss
        + float(config.get("delta_reg_weight", 0.05)) * delta_reg
    )
    return E11LossOutput(
        total=total,
        quality=quality_loss,
        score=score_loss,
        ranking=ranking_loss,
        rescue=rescue_loss,
        delta_reg=delta_reg,
    )
