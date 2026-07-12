from __future__ import annotations

from dataclasses import dataclass

import torch

from .arbitration import ArbitrationResult, classwise_nms
from .e11_losses import bounded_residual_score
from .e11_model import ResidualArbitratorOutput
from .targets import box_iou


@dataclass(frozen=True)
class E11Thresholds:
    base_conf: float = 0.25
    base_nms_iou: float = 0.30
    residual_alpha: float = 0.50
    rescue_threshold: float = 0.80
    extra_quality_threshold: float = 0.50
    extra_score_floor: float = 0.10
    extra_overlap_iou: float = 0.30
    max_extra_per_image: int = 1


@dataclass
class E11ArbitrationResult(ArbitrationResult):
    base_count: int
    extra_count: int
    source_indices: torch.Tensor


def _empty_result(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
) -> E11ArbitrationResult:
    return E11ArbitrationResult(
        boxes=boxes.new_zeros((0, 4)),
        scores=scores.new_zeros((0,)),
        classes=classes.new_zeros((0,), dtype=torch.long),
        provenance=classes.new_zeros((0,), dtype=torch.long),
        base_count=0,
        extra_count=0,
        source_indices=classes.new_zeros((0,), dtype=torch.long),
    )


def fixed_baseline_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    thresholds: E11Thresholds,
) -> torch.Tensor:
    eligible = torch.nonzero(scores >= thresholds.base_conf, as_tuple=False).flatten()
    if eligible.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    local_keep = classwise_nms(
        boxes[eligible],
        scores[eligible],
        classes[eligible],
        iou_threshold=thresholds.base_nms_iou,
    )
    return eligible[local_keep]


def fixed_baseline_single(
    batch: dict[str, torch.Tensor],
    batch_index: int,
    thresholds: E11Thresholds,
) -> ArbitrationResult:
    count = int(batch["mask"][batch_index].sum())
    boxes = batch["boxes"][batch_index, :count]
    scores = batch["scores"][batch_index, :count]
    classes = batch["classes"][batch_index, :count]
    modality = batch["modality"][batch_index, :count]
    keep = fixed_baseline_indices(boxes, scores, classes, thresholds)
    return ArbitrationResult(
        boxes=boxes[keep],
        scores=scores[keep],
        classes=classes[keep],
        provenance=modality[keep],
    )


def arbitrate_e11_single(
    output: ResidualArbitratorOutput,
    batch: dict[str, torch.Tensor],
    batch_index: int,
    thresholds: E11Thresholds,
) -> E11ArbitrationResult:
    """Run E1.1 with a hard safe-set inclusion guarantee.

    Fixed-NMS candidates are copied into the final set unchanged in box/class identity.
    The Transformer may only rerank them through a bounded score residual and append a
    bounded number of non-overlapping POL candidates.
    """
    count = int(batch["mask"][batch_index].sum())
    boxes = batch["boxes"][batch_index, :count]
    raw_scores = batch["scores"][batch_index, :count]
    classes = batch["classes"][batch_index, :count]
    modality = batch["modality"][batch_index, :count]
    if count == 0:
        return _empty_result(boxes, raw_scores, classes)

    adjusted_scores = bounded_residual_score(
        raw_scores,
        output.score_delta[batch_index, :count],
        thresholds.residual_alpha,
    )
    quality = output.iou_pred[batch_index, :count]
    rescue = torch.sigmoid(output.rescue_logit[batch_index, :count])

    safe_indices = fixed_baseline_indices(boxes, raw_scores, classes, thresholds)
    safe_mask = torch.zeros(count, dtype=torch.bool, device=boxes.device)
    safe_mask[safe_indices] = True

    selected_extra: list[int] = []
    if thresholds.max_extra_per_image > 0:
        extra_candidates = torch.nonzero(
            (modality == 1)
            & ~safe_mask
            & (adjusted_scores >= thresholds.extra_score_floor)
            & (quality >= thresholds.extra_quality_threshold)
            & (rescue >= thresholds.rescue_threshold),
            as_tuple=False,
        ).flatten()
        if extra_candidates.numel():
            priority = adjusted_scores[extra_candidates] * rescue[extra_candidates]
            for local_index in priority.argsort(descending=True):
                candidate = int(extra_candidates[local_index])
                reference = torch.cat(
                    (
                        safe_indices,
                        torch.tensor(selected_extra, device=boxes.device, dtype=torch.long),
                    )
                )
                if reference.numel():
                    same_class = classes[reference] == classes[candidate]
                    if same_class.any():
                        overlap = box_iou(
                            boxes[candidate : candidate + 1], boxes[reference[same_class]]
                        ).squeeze(0)
                        if (overlap > thresholds.extra_overlap_iou).any():
                            continue
                selected_extra.append(candidate)
                if len(selected_extra) >= thresholds.max_extra_per_image:
                    break

    extra_indices = torch.tensor(selected_extra, device=boxes.device, dtype=torch.long)
    source_indices = torch.cat((safe_indices, extra_indices))
    if source_indices.numel() == 0:
        return _empty_result(boxes, raw_scores, classes)

    base_scores = adjusted_scores[safe_indices]
    extra_scores = adjusted_scores[extra_indices] * rescue[extra_indices]
    result_scores = torch.cat((base_scores, extra_scores))
    result_provenance = torch.cat(
        (
            modality[safe_indices],
            torch.full(
                (len(extra_indices),), 3, dtype=torch.long, device=boxes.device
            ),
        )
    )
    return E11ArbitrationResult(
        boxes=boxes[source_indices],
        scores=result_scores,
        classes=classes[source_indices],
        provenance=result_provenance,
        base_count=len(safe_indices),
        extra_count=len(extra_indices),
        source_indices=source_indices,
    )
