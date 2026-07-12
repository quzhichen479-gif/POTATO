from __future__ import annotations

from typing import Any

import torch

from .arbitration import classwise_nms
from .dataset import CandidateCacheDataset, collate_candidate_batch
from .targets import box_iou


def fixed_baseline_keep_mask(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    base_conf: float,
    base_nms_iou: float,
) -> torch.Tensor:
    """Return the immutable fixed-NMS safe set used by E1.1.

    The mask is defined only from cached detector outputs. It never uses GT.
    """
    keep_mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    eligible = torch.nonzero(scores >= base_conf, as_tuple=False).flatten()
    if eligible.numel() == 0:
        return keep_mask
    local_keep = classwise_nms(
        boxes[eligible], scores[eligible], classes[eligible], iou_threshold=base_nms_iou
    )
    keep_mask[eligible[local_keep]] = True
    return keep_mask


def _match_candidates_to_gt(
    boxes: torch.Tensor,
    classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return (
            boxes.new_zeros((len(boxes),)),
            torch.full((len(boxes),), -1, dtype=torch.long, device=boxes.device),
        )
    overlap = box_iou(boxes, gt_boxes)
    overlap = torch.where(
        classes[:, None] == gt_classes[None, :], overlap, torch.zeros_like(overlap)
    )
    return overlap.max(dim=1)


class E11CandidateCacheDataset(CandidateCacheDataset):
    """Candidate cache dataset with fixed-baseline-relative E1.1 targets."""

    def __init__(
        self,
        *args: Any,
        base_conf: float = 0.25,
        base_nms_iou: float = 0.30,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_conf = float(base_conf)
        self.base_nms_iou = float(base_nms_iou)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        boxes = sample["boxes"]
        scores = sample["scores"]
        classes = sample["classes"]
        modality = sample["modality"]
        gt_boxes = sample["gt_boxes"]
        gt_classes = sample["gt_classes"]

        base_keep = fixed_baseline_keep_mask(
            boxes, scores, classes, self.base_conf, self.base_nms_iou
        )
        candidate_iou, matched_gt = _match_candidates_to_gt(
            boxes, classes, gt_boxes, gt_classes
        )
        candidate_valid = candidate_iou >= self.positive_iou

        gt_covered = torch.zeros(len(gt_boxes), dtype=torch.bool)
        safe_valid = base_keep & candidate_valid
        if safe_valid.any():
            gt_covered[matched_gt[safe_valid]] = True

        admission_mask = (modality == 1) & ~base_keep
        rescue_target = torch.zeros(len(boxes), dtype=torch.float32)
        rescue_positive = admission_mask & candidate_valid
        if rescue_positive.any() and len(gt_boxes):
            rescue_target[rescue_positive] = (
                ~gt_covered[matched_gt[rescue_positive]]
            ).float()

        sample.update(
            {
                "base_keep_mask": base_keep,
                "e11_admission_mask": admission_mask,
                "e11_rescue_target": rescue_target,
                "e11_rank_positive": base_keep & candidate_valid,
                "e11_rank_negative": base_keep & ~candidate_valid,
            }
        )
        return sample


def collate_e11_candidate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_candidate_batch(samples)
    batch_size, max_tokens = batch["mask"].shape
    for key, dtype in (
        ("base_keep_mask", torch.bool),
        ("e11_admission_mask", torch.bool),
        ("e11_rescue_target", torch.float32),
        ("e11_rank_positive", torch.bool),
        ("e11_rank_negative", torch.bool),
    ):
        batch[key] = torch.zeros((batch_size, max_tokens), dtype=dtype)

    for batch_index, sample in enumerate(samples):
        count = len(sample["boxes"])
        for key in (
            "base_keep_mask",
            "e11_admission_mask",
            "e11_rescue_target",
            "e11_rank_positive",
            "e11_rank_negative",
        ):
            batch[key][batch_index, :count] = sample[key]
    return batch
