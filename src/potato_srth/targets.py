from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass
class OracleRouteTargets:
    dif: torch.Tensor
    pol: torch.Tensor
    best_expert: torch.Tensor
    qualities: torch.Tensor


def detector_quality(
    iou: torch.Tensor,
    confidence: torch.Tensor,
    *,
    hit_iou_threshold: float = 0.5,
    iou_weight: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Combine localization and confidence into a bounded per-GT quality score."""
    if iou.shape != confidence.shape:
        raise ValueError("iou and confidence must have identical shapes")
    if not 0 <= hit_iou_threshold <= 1:
        raise ValueError("hit_iou_threshold must be in [0, 1]")
    if not 0 <= iou_weight <= 1:
        raise ValueError("iou_weight must be in [0, 1]")
    iou = iou.clamp(0, 1)
    confidence = confidence.clamp(eps, 1)
    hit = (iou >= hit_iou_threshold).to(iou.dtype)
    return hit * iou.pow(iou_weight) * confidence.pow(1.0 - iou_weight)


def oracle_route_targets(
    rgb_quality: torch.Tensor,
    dif_quality: torch.Tensor,
    pol_quality: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> OracleRouteTargets:
    """Create soft independent DIF/POL-vs-RGB routing targets from OOF qualities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not (rgb_quality.shape == dif_quality.shape == pol_quality.shape):
        raise ValueError("all quality tensors must have the same shape")
    qualities = torch.stack((rgb_quality, dif_quality, pol_quality), dim=-1)
    return OracleRouteTargets(
        dif=torch.sigmoid((dif_quality - rgb_quality) / temperature),
        pol=torch.sigmoid((pol_quality - rgb_quality) / temperature),
        best_expert=qualities.argmax(dim=-1),
        qualities=qualities,
    )


def rasterize_route_targets(
    boxes_xyxy_normalized: torch.Tensor,
    batch_indices: torch.Tensor,
    dif_targets: torch.Tensor,
    pol_targets: torch.Tensor,
    *,
    batch_size: int,
    feature_size: tuple[int, int],
    center_fraction: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize per-GT route labels to a feature map without supervising background.

    Overlapping target regions are averaged. Boxes are expected in normalized xyxy format.
    """
    if boxes_xyxy_normalized.ndim != 2 or boxes_xyxy_normalized.shape[-1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    n = boxes_xyxy_normalized.shape[0]
    if not (batch_indices.shape == dif_targets.shape == pol_targets.shape == (n,)):
        raise ValueError("batch indices and route targets must have shape [N]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < center_fraction <= 1:
        raise ValueError("center_fraction must be in (0, 1]")
    height, width = feature_size
    if height <= 0 or width <= 0:
        raise ValueError("feature_size must be positive")

    device = boxes_xyxy_normalized.device
    dtype = dif_targets.dtype
    accum = torch.zeros((batch_size, 2, height, width), device=device, dtype=dtype)
    count = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)

    boxes = boxes_xyxy_normalized.clamp(0, 1)
    for index in range(n):
        batch = int(batch_indices[index].item())
        if batch < 0 or batch >= batch_size:
            raise ValueError("batch index out of range")
        x1, y1, x2, y2 = boxes[index]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        half_w = (x2 - x1).clamp_min(1.0 / width) * 0.5 * center_fraction
        half_h = (y2 - y1).clamp_min(1.0 / height) * 0.5 * center_fraction

        left = max(0, min(width - 1, int(torch.floor((cx - half_w) * width).item())))
        right = max(left + 1, min(width, int(torch.ceil((cx + half_w) * width).item())))
        top = max(0, min(height - 1, int(torch.floor((cy - half_h) * height).item())))
        bottom = max(top + 1, min(height, int(torch.ceil((cy + half_h) * height).item())))

        target = torch.stack((dif_targets[index], pol_targets[index]))[:, None, None]
        accum[batch, :, top:bottom, left:right] += target
        count[batch, :, top:bottom, left:right] += 1

    mask = count > 0
    targets = accum / count.clamp_min(1)
    return targets, mask


def rasterize_targets_for_levels(
    boxes_xyxy_normalized: torch.Tensor,
    batch_indices: torch.Tensor,
    route_targets: OracleRouteTargets,
    *,
    batch_size: int,
    level_sizes: Mapping[str, tuple[int, int]],
    center_fraction: float = 0.6,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for level, feature_size in level_sizes.items():
        target, mask = rasterize_route_targets(
            boxes_xyxy_normalized,
            batch_indices,
            route_targets.dif,
            route_targets.pol,
            batch_size=batch_size,
            feature_size=feature_size,
            center_fraction=center_fraction,
        )
        targets[level] = target
        masks[level] = mask
    return targets, masks
