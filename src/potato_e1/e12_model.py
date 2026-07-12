from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RollbackTransformerOutput:
    restore_logits: torch.Tensor
    rollback_quality: torch.Tensor
    restore_score_logit: torch.Tensor
    edge_logits: torch.Tensor
    safe_score_delta: torch.Tensor
    safe_encoded: torch.Tensor
    rollback_encoded: torch.Tensor


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


def _effective_mask_and_token(
    token: torch.Tensor,
    mask: torch.Tensor,
    null_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prevent all-padding Transformer rows without turning dummy rows into candidates."""
    effective = mask.clone()
    empty = ~effective.any(dim=1)
    if empty.any():
        effective[empty, 0] = True
        token = token.clone()
        token[empty, 0] = null_token
    return token, effective


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(value.dtype)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class RollbackTransformer(nn.Module):
    """Set-wise K=1 selector over NMS-suppressed and low-confidence candidates."""

    def __init__(
        self,
        *,
        appearance_dim: int = 0,
        edge_dim: int = 8,
        edge_classes: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        safe_layers: int = 1,
        rollback_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.appearance_dim = int(appearance_dim)
        self.edge_dim = int(edge_dim)
        self.d_model = int(d_model)

        self.geometry_projection = nn.Sequential(
            nn.Linear(11, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.modality_embedding = nn.Embedding(2, d_model)
        self.source_embedding = nn.Embedding(2, d_model)
        self.appearance_projection = (
            nn.Sequential(nn.Linear(appearance_dim, d_model), nn.LayerNorm(d_model))
            if appearance_dim > 0
            else None
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        def encoder(layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(
                layer, num_layers=layers, norm=nn.LayerNorm(d_model)
            )

        self.safe_encoder = encoder(safe_layers)
        self.rollback_encoder = encoder(rollback_layers)
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        self.safe_null = nn.Parameter(torch.zeros(d_model))
        self.rollback_null = nn.Parameter(torch.zeros(d_model))
        self.none_token = nn.Parameter(torch.zeros(d_model))

        def scalar_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        self.restore_choice_head = scalar_head()
        self.none_head = scalar_head()
        self.rollback_quality_head = scalar_head()
        self.restore_score_head = scalar_head()
        self.safe_delta_head = scalar_head()
        self.edge_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, edge_classes),
        )

        nn.init.normal_(self.safe_null, std=0.02)
        nn.init.normal_(self.rollback_null, std=0.02)
        nn.init.normal_(self.none_token, std=0.02)

    def _base_token(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        modality: torch.Tensor,
        appearance: torch.Tensor,
    ) -> torch.Tensor:
        token = self.geometry_projection(_geometry_features(boxes, scores))
        token = token + self.modality_embedding(modality.clamp(0, 1))
        if self.appearance_projection is not None:
            token = token + self.appearance_projection(appearance)
        return token

    def forward(self, batch: dict[str, torch.Tensor]) -> RollbackTransformerOutput:
        safe_mask = batch["safe_mask"]
        rollback_mask = batch["rollback_mask"]

        safe_token = self._base_token(
            batch["safe_boxes"],
            batch["safe_scores"],
            batch["safe_modality"],
            batch["safe_appearance"],
        )
        safe_token, effective_safe = _effective_mask_and_token(
            safe_token, safe_mask, self.safe_null
        )
        safe_encoded = self.safe_encoder(
            safe_token, src_key_padding_mask=~effective_safe
        )

        rollback_token = self._base_token(
            batch["rollback_boxes"],
            batch["rollback_scores"],
            batch["rollback_modality"],
            batch["rollback_appearance"],
        )
        rollback_token = rollback_token + self.source_embedding(
            batch["rollback_source"].clamp(0, 1)
        )
        rollback_token = rollback_token + self.edge_projection(batch["rollback_edge"])
        rollback_token, effective_rollback = _effective_mask_and_token(
            rollback_token, rollback_mask, self.rollback_null
        )
        rollback_encoded = self.rollback_encoder(
            rollback_token, src_key_padding_mask=~effective_rollback
        )

        attended, _ = self.cross_attention(
            query=rollback_encoded,
            key=safe_encoded,
            value=safe_encoded,
            key_padding_mask=~effective_safe,
            need_weights=False,
        )
        rollback_context = self.cross_norm(rollback_encoded + attended)

        candidate_choice = self.restore_choice_head(rollback_context).squeeze(-1)
        candidate_choice = candidate_choice.masked_fill(~rollback_mask, -20.0)
        safe_pool = _masked_mean(safe_encoded, effective_safe)
        rollback_pool = _masked_mean(rollback_context, effective_rollback)
        none_context = self.none_token.unsqueeze(0) + safe_pool + rollback_pool
        none_logit = self.none_head(none_context).squeeze(-1)
        restore_logits = torch.cat((none_logit[:, None], candidate_choice), dim=1)

        rollback_quality = torch.sigmoid(
            self.rollback_quality_head(rollback_context).squeeze(-1)
        )
        restore_score_logit = self.restore_score_head(rollback_context).squeeze(-1)
        edge_logits = self.edge_head(rollback_context)
        safe_score_delta = self.safe_delta_head(safe_encoded).squeeze(-1)

        rollback_quality = rollback_quality * rollback_mask
        restore_score_logit = restore_score_logit.masked_fill(~rollback_mask, -20.0)
        safe_score_delta = safe_score_delta * safe_mask
        return RollbackTransformerOutput(
            restore_logits=restore_logits,
            rollback_quality=rollback_quality,
            restore_score_logit=restore_score_logit,
            edge_logits=edge_logits,
            safe_score_delta=safe_score_delta,
            safe_encoded=safe_encoded * safe_mask.unsqueeze(-1),
            rollback_encoded=rollback_context * rollback_mask.unsqueeze(-1),
        )
