from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .model import _geometry_features


@dataclass
class ResidualArbitratorOutput:
    score_delta: torch.Tensor
    iou_pred: torch.Tensor
    rescue_logit: torch.Tensor
    encoded: torch.Tensor


class CandidateResidualTransformer(nn.Module):
    """Transformer residual scorer anchored to the fixed cross-modal NMS baseline.

    It has no delete/replace/protect head. Non-degradation is enforced by inference:
    every fixed-NMS safe candidate is retained, while the network only adjusts scores
    within a bounded logit residual and optionally admits a small number of POL extras.
    """

    def __init__(
        self,
        appearance_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.appearance_dim = int(appearance_dim)
        self.base_projection = nn.Sequential(
            nn.Linear(11, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.modality_embedding = nn.Embedding(2, d_model)
        self.appearance_projection = (
            nn.Sequential(nn.Linear(appearance_dim, d_model), nn.LayerNorm(d_model))
            if appearance_dim > 0
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

        def scalar_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        self.score_delta_head = scalar_head()
        self.iou_head = scalar_head()
        self.rescue_head = scalar_head()

    def forward(self, batch: dict[str, torch.Tensor]) -> ResidualArbitratorOutput:
        boxes = batch["boxes"]
        scores = batch["scores"]
        modality = batch["modality"]
        mask = batch["mask"]

        token = self.base_projection(_geometry_features(boxes, scores))
        token = token + self.modality_embedding(modality.clamp(0, 1))
        if self.appearance_projection is not None:
            token = token + self.appearance_projection(batch["appearance"])

        encoded = self.encoder(token, src_key_padding_mask=~mask)
        encoded = encoded * mask.unsqueeze(-1)
        score_delta = self.score_delta_head(encoded).squeeze(-1) * mask
        iou_pred = torch.sigmoid(self.iou_head(encoded).squeeze(-1)) * mask
        rescue_logit = self.rescue_head(encoded).squeeze(-1)
        rescue_logit = rescue_logit.masked_fill(~mask, -20.0)
        return ResidualArbitratorOutput(
            score_delta=score_delta,
            iou_pred=iou_pred,
            rescue_logit=rescue_logit,
            encoded=encoded,
        )
