from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from .targets import box_iou


@dataclass
class ArbitratorOutput:
    valid_logit: torch.Tensor
    iou_pred: torch.Tensor
    rescue_logit: torch.Tensor
    protect_logit: torch.Tensor
    pair_same_logit: torch.Tensor
    encoded: torch.Tensor


def _geometry_features(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    width = (x2 - x1).clamp_min(1e-6)
    height = (y2 - y1).clamp_min(1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    area = width * height
    score = scores.clamp(1e-5, 1 - 1e-5)
    score_logit = torch.log(score) - torch.log1p(-score)
    return torch.stack(
        (x1, y1, x2, y2, cx, cy, width, height, area, score, score_logit), dim=-1
    )


def _pair_geometry(boxes: torch.Tensor) -> torch.Tensor:
    batch, tokens, _ = boxes.shape
    features: list[torch.Tensor] = []
    for batch_index in range(batch):
        current = boxes[batch_index]
        iou = box_iou(current, current)
        x1, y1, x2, y2 = current.unbind(dim=-1)
        width = (x2 - x1).clamp_min(1e-6)
        height = (y2 - y1).clamp_min(1e-6)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dx = (cx[:, None] - cx[None, :]).abs()
        dy = (cy[:, None] - cy[None, :]).abs()
        log_wr = (torch.log(width[:, None]) - torch.log(width[None, :])).abs()
        log_hr = (torch.log(height[:, None]) - torch.log(height[None, :])).abs()
        features.append(torch.stack((iou, dx, dy, log_wr, log_hr), dim=-1))
    if not features:
        return boxes.new_zeros((0, tokens, tokens, 5))
    return torch.stack(features, dim=0)


class CandidateTransformerArbitrator(nn.Module):
    """Contextual candidate scorer with cross-modal pair matching.

    The model never receives GT-derived features at inference. Candidate tokens are built
    from normalized geometry, detector score, modality, optional appearance features and
    optional candidate-region physical statistics.
    """

    def __init__(
        self,
        appearance_dim: int = 0,
        physics_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pair_dim: int = 64,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.appearance_dim = int(appearance_dim)
        self.physics_dim = int(physics_dim)
        self.d_model = int(d_model)

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
        self.physics_projection = (
            nn.Sequential(nn.Linear(physics_dim, d_model), nn.LayerNorm(d_model))
            if physics_dim > 0
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
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        def scalar_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        self.valid_head = scalar_head()
        self.iou_head = scalar_head()
        self.rescue_head = scalar_head()
        self.protect_head = scalar_head()

        self.pair_query = nn.Linear(d_model, pair_dim, bias=False)
        self.pair_key = nn.Linear(d_model, pair_dim, bias=False)
        self.pair_geometry = nn.Sequential(
            nn.Linear(5, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ArbitratorOutput:
        boxes = batch["boxes"]
        scores = batch["scores"]
        modality = batch["modality"]
        mask = batch["mask"]

        token = self.base_projection(_geometry_features(boxes, scores))
        token = token + self.modality_embedding(modality.clamp(0, 1))
        if self.appearance_projection is not None:
            token = token + self.appearance_projection(batch["appearance"])
        if self.physics_projection is not None:
            # RGB physical vectors are intentionally zero; POL receives candidate-region values.
            token = token + self.physics_projection(batch["physics"])

        encoded = self.encoder(token, src_key_padding_mask=~mask)
        encoded = encoded * mask.unsqueeze(-1)

        valid_logit = self.valid_head(encoded).squeeze(-1)
        iou_pred = torch.sigmoid(self.iou_head(encoded).squeeze(-1))
        rescue_logit = self.rescue_head(encoded).squeeze(-1)
        protect_logit = self.protect_head(encoded).squeeze(-1)

        query = self.pair_query(encoded)
        key = self.pair_key(encoded)
        pair_content = torch.matmul(query, key.transpose(1, 2)) / sqrt(query.shape[-1])
        pair_bias = self.pair_geometry(_pair_geometry(boxes)).squeeze(-1)
        pair_same_logit = pair_content + pair_bias

        valid_pair = mask[:, :, None] & mask[:, None, :]
        pair_same_logit = pair_same_logit.masked_fill(~valid_pair, -20.0)

        return ArbitratorOutput(
            valid_logit=valid_logit,
            iou_pred=iou_pred,
            rescue_logit=rescue_logit,
            protect_logit=protect_logit,
            pair_same_logit=pair_same_logit,
            encoded=encoded,
        )
