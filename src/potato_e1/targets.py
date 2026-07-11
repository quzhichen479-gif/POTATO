from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TargetSet:
    valid: torch.Tensor
    iou: torch.Tensor
    rescue: torch.Tensor
    protect: torch.Tensor
    matched_gt: torch.Tensor
    pair_same: torch.Tensor
    pair_mask: torch.Tensor


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp_min(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp_min(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp_min(1e-8)


def _match_candidates(
    boxes: torch.Tensor,
    classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = boxes.shape[0]
    if n == 0 or gt_boxes.shape[0] == 0:
        return boxes.new_zeros(n), torch.full((n,), -1, dtype=torch.long, device=boxes.device)

    iou = box_iou(boxes, gt_boxes)
    same_class = classes[:, None] == gt_classes[None, :]
    iou = torch.where(same_class, iou, torch.zeros_like(iou))
    max_iou, matched = iou.max(dim=1)
    return max_iou, matched


def build_targets(
    rgb_boxes: torch.Tensor,
    rgb_classes: torch.Tensor,
    pol_boxes: torch.Tensor,
    pol_classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    positive_iou: float = 0.5,
    pair_candidate_iou: float = 0.05,
) -> TargetSet:
    rgb_iou, rgb_gt = _match_candidates(rgb_boxes, rgb_classes, gt_boxes, gt_classes)
    pol_iou, pol_gt = _match_candidates(pol_boxes, pol_classes, gt_boxes, gt_classes)

    rgb_valid = rgb_iou >= positive_iou
    pol_valid = pol_iou >= positive_iou

    g = gt_boxes.shape[0]
    rgb_covered = torch.zeros(g, dtype=torch.bool, device=gt_boxes.device)
    pol_covered = torch.zeros(g, dtype=torch.bool, device=gt_boxes.device)
    if rgb_valid.any():
        rgb_covered[rgb_gt[rgb_valid]] = True
    if pol_valid.any():
        pol_covered[pol_gt[pol_valid]] = True

    rgb_protect = rgb_valid.clone()
    if rgb_valid.any() and g > 0:
        rgb_protect[rgb_valid] = ~pol_covered[rgb_gt[rgb_valid]]

    pol_rescue = pol_valid.clone()
    if pol_valid.any() and g > 0:
        pol_rescue[pol_valid] = ~rgb_covered[pol_gt[pol_valid]]

    valid = torch.cat((rgb_valid, pol_valid)).float()
    iou = torch.cat((rgb_iou, pol_iou)).float()
    rescue = torch.cat((torch.zeros_like(rgb_valid), pol_rescue)).float()
    protect = torch.cat((rgb_protect, torch.zeros_like(pol_valid))).float()
    matched_gt = torch.cat((rgb_gt, pol_gt))

    n_rgb, n_pol = rgb_boxes.shape[0], pol_boxes.shape[0]
    total = n_rgb + n_pol
    pair_same = torch.zeros((total, total), dtype=torch.float32, device=gt_boxes.device)
    pair_mask = torch.zeros((total, total), dtype=torch.bool, device=gt_boxes.device)

    if n_rgb and n_pol:
        cross_iou = box_iou(rgb_boxes, pol_boxes)
        same_class = rgb_classes[:, None] == pol_classes[None, :]
        same_gt = (
            rgb_valid[:, None]
            & pol_valid[None, :]
            & (rgb_gt[:, None] == pol_gt[None, :])
            & same_class
        )
        candidate_pair = same_class & ((cross_iou >= pair_candidate_iou) | same_gt)
        pair_same[:n_rgb, n_rgb:] = same_gt.float()
        pair_same[n_rgb:, :n_rgb] = same_gt.transpose(0, 1).float()
        pair_mask[:n_rgb, n_rgb:] = candidate_pair
        pair_mask[n_rgb:, :n_rgb] = candidate_pair.transpose(0, 1)

    return TargetSet(
        valid=valid,
        iou=iou,
        rescue=rescue,
        protect=protect,
        matched_gt=matched_gt,
        pair_same=pair_same,
        pair_mask=pair_mask,
    )
