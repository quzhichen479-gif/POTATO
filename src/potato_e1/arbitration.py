from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ArbitratorOutput
from .targets import box_iou


@dataclass(frozen=True)
class ArbitrationThresholds:
    rgb_raw_score_floor: float = 0.25
    rgb_valid_threshold: float = 0.50
    rgb_protect_threshold: float = 0.50
    pol_valid_threshold: float = 0.50
    pol_rescue_threshold: float = 0.50
    pair_same_threshold: float = 0.50
    replace_margin: float = 0.05
    final_nms_iou: float = 0.60
    merge_duplicates: bool = True


@dataclass
class ArbitrationResult:
    boxes: torch.Tensor
    scores: torch.Tensor
    classes: torch.Tensor
    provenance: torch.Tensor


def classwise_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    kept: list[torch.Tensor] = []
    for class_id in classes.unique(sorted=True):
        indices = torch.nonzero(classes == class_id, as_tuple=False).flatten()
        order = indices[scores[indices].argsort(descending=True)]
        while order.numel():
            current = order[0]
            kept.append(current)
            if order.numel() == 1:
                break
            remaining = order[1:]
            overlaps = box_iou(boxes[current : current + 1], boxes[remaining]).squeeze(0)
            order = remaining[overlaps <= iou_threshold]
    return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)


def arbitrate_single(
    output: ArbitratorOutput,
    batch: dict[str, torch.Tensor],
    batch_index: int,
    thresholds: ArbitrationThresholds,
) -> ArbitrationResult:
    token_mask = batch["mask"][batch_index]
    count = int(token_mask.sum())
    boxes = batch["boxes"][batch_index, :count].clone()
    raw_scores = batch["scores"][batch_index, :count]
    classes = batch["classes"][batch_index, :count]
    modality = batch["modality"][batch_index, :count]

    valid = torch.sigmoid(output.valid_logit[batch_index, :count])
    rescue = torch.sigmoid(output.rescue_logit[batch_index, :count])
    protect = torch.sigmoid(output.protect_logit[batch_index, :count])
    iou_quality = output.iou_pred[batch_index, :count]
    pair_same = torch.sigmoid(output.pair_same_logit[batch_index, :count, :count])

    # Blend calibrated validity, localization quality and original detector evidence.
    priority = torch.sqrt((raw_scores * valid).clamp_min(0)) * (0.5 + 0.5 * iou_quality)

    rgb_indices = torch.nonzero(modality == 0, as_tuple=False).flatten()
    pol_indices = torch.nonzero(modality == 1, as_tuple=False).flatten()
    keep_rgb_mask = (
        (raw_scores[rgb_indices] >= thresholds.rgb_raw_score_floor)
        | (valid[rgb_indices] >= thresholds.rgb_valid_threshold)
        | (protect[rgb_indices] >= thresholds.rgb_protect_threshold)
    )
    selected: list[int] = rgb_indices[keep_rgb_mask].tolist()
    provenance: list[int] = [0 for _ in selected]  # 0 RGB, 1 POL, 2 merged

    for pol_index_tensor in pol_indices[priority[pol_indices].argsort(descending=True)]:
        pol_index = int(pol_index_tensor)
        if valid[pol_index] < thresholds.pol_valid_threshold:
            continue

        duplicate_position: int | None = None
        duplicate_pair_score = -1.0
        for position, selected_index in enumerate(selected):
            if classes[selected_index] != classes[pol_index]:
                continue
            score = float(pair_same[selected_index, pol_index])
            if score >= thresholds.pair_same_threshold and score > duplicate_pair_score:
                duplicate_pair_score = score
                duplicate_position = position

        if duplicate_position is None:
            if rescue[pol_index] >= thresholds.pol_rescue_threshold:
                selected.append(pol_index)
                provenance.append(1)
            continue

        rgb_or_merged_index = selected[duplicate_position]
        pol_better = (
            iou_quality[pol_index]
            > iou_quality[rgb_or_merged_index] + thresholds.replace_margin
        )
        rgb_is_protected = protect[rgb_or_merged_index] >= thresholds.rgb_protect_threshold

        if pol_better and not rgb_is_protected:
            selected[duplicate_position] = pol_index
            provenance[duplicate_position] = 1
        elif thresholds.merge_duplicates:
            rgb_weight = priority[rgb_or_merged_index].clamp_min(1e-6)
            pol_weight = priority[pol_index].clamp_min(1e-6)
            boxes[rgb_or_merged_index] = (
                rgb_weight * boxes[rgb_or_merged_index] + pol_weight * boxes[pol_index]
            ) / (rgb_weight + pol_weight)
            priority[rgb_or_merged_index] = torch.maximum(
                priority[rgb_or_merged_index], priority[pol_index]
            )
            provenance[duplicate_position] = 2

    if not selected:
        return ArbitrationResult(
            boxes=boxes.new_zeros((0, 4)),
            scores=raw_scores.new_zeros((0,)),
            classes=classes.new_zeros((0,), dtype=torch.long),
            provenance=classes.new_zeros((0,), dtype=torch.long),
        )

    selected_tensor = torch.tensor(selected, device=boxes.device, dtype=torch.long)
    result_boxes = boxes[selected_tensor]
    result_scores = priority[selected_tensor]
    result_classes = classes[selected_tensor]
    result_provenance = torch.tensor(provenance, device=boxes.device, dtype=torch.long)

    kept = classwise_nms(
        result_boxes,
        result_scores,
        result_classes,
        iou_threshold=thresholds.final_nms_iou,
    )
    return ArbitrationResult(
        boxes=result_boxes[kept],
        scores=result_scores[kept],
        classes=result_classes[kept],
        provenance=result_provenance[kept],
    )
