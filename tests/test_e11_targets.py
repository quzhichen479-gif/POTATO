from __future__ import annotations

import torch

from potato_e1.e11_dataset import fixed_baseline_keep_mask
from potato_e1.targets import box_iou


def _rescue_target(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    modality: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_keep = fixed_baseline_keep_mask(
        boxes, scores, classes, base_conf=0.25, base_nms_iou=0.30
    )
    overlap = box_iou(boxes, gt_boxes)
    overlap = torch.where(
        classes[:, None] == gt_classes[None, :], overlap, torch.zeros_like(overlap)
    )
    best_iou, matched_gt = overlap.max(dim=1)
    valid = best_iou >= 0.5
    gt_covered = torch.zeros(len(gt_boxes), dtype=torch.bool)
    safe_valid = base_keep & valid
    if safe_valid.any():
        gt_covered[matched_gt[safe_valid]] = True
    admission = (modality == 1) & ~base_keep
    target = torch.zeros(len(boxes))
    positive = admission & valid
    target[positive] = (~gt_covered[matched_gt[positive]]).float()
    return base_keep, target


def test_pol_extra_is_rescue_only_for_gt_uncovered_by_safe_set() -> None:
    boxes = torch.tensor(
        [
            [0.10, 0.10, 0.30, 0.30],
            [0.11, 0.11, 0.31, 0.31],
            [0.60, 0.60, 0.80, 0.80],
        ]
    )
    scores = torch.tensor([0.90, 0.10, 0.10])
    classes = torch.zeros(3, dtype=torch.long)
    modality = torch.tensor([0, 1, 1])
    gt_boxes = torch.tensor(
        [[0.10, 0.10, 0.30, 0.30], [0.60, 0.60, 0.80, 0.80]]
    )
    gt_classes = torch.zeros(2, dtype=torch.long)

    base_keep, target = _rescue_target(
        boxes, scores, classes, modality, gt_boxes, gt_classes
    )
    assert base_keep.tolist() == [True, False, False]
    assert target.tolist() == [0.0, 0.0, 1.0]
