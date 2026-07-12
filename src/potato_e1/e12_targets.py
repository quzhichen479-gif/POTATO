from __future__ import annotations

from dataclasses import dataclass

import torch

from .targets import box_iou

SOURCE_NMS = 0
SOURCE_LOW_CONF = 1

EDGE_DUPLICATE = 0
EDGE_DISTINCT = 1
EDGE_BACKGROUND = 2
EDGE_LOW_CONF_VALID = 3


@dataclass
class NmsTrace:
    safe_indices: torch.Tensor
    suppressed_by: torch.Tensor


@dataclass
class RollbackTargets:
    safe_indices: torch.Tensor
    rollback_indices: torch.Tensor
    rollback_source: torch.Tensor
    rollback_context_indices: torch.Tensor
    safe_iou: torch.Tensor
    safe_valid: torch.Tensor
    rollback_iou: torch.Tensor
    rollback_matched_gt: torch.Tensor
    rollback_positive: torch.Tensor
    restore_score_target: torch.Tensor
    edge_target: torch.Tensor
    restore_target_index: torch.Tensor


def _stable_score_order(indices: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return indices
    order = torch.argsort(scores[indices], descending=True, stable=True)
    return indices[order]


def trace_classwise_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    score_threshold: float,
    iou_threshold: float,
) -> NmsTrace:
    """Run deterministic class-wise NMS and retain the actual suppressor edge.

    ``suppressed_by[j]`` is the global token index of the kept candidate that
    suppressed candidate ``j``. Safe or ineligible candidates have value -1.
    """
    count = boxes.shape[0]
    suppressed_by = torch.full((count,), -1, dtype=torch.long, device=boxes.device)
    eligible = torch.nonzero(scores >= float(score_threshold), as_tuple=False).flatten()
    safe: list[torch.Tensor] = []

    for class_id in classes[eligible].unique(sorted=True):
        class_indices = eligible[classes[eligible] == class_id]
        order = _stable_score_order(class_indices, scores)
        while order.numel():
            current = order[0]
            safe.append(current)
            if order.numel() == 1:
                break
            remaining = order[1:]
            overlap = box_iou(boxes[current : current + 1], boxes[remaining]).squeeze(0)
            suppress = overlap > float(iou_threshold)
            if suppress.any():
                suppressed_by[remaining[suppress]] = current
            order = remaining[~suppress]

    if safe:
        safe_indices = torch.stack(safe)
        safe_indices = _stable_score_order(safe_indices, scores)
    else:
        safe_indices = torch.empty(0, dtype=torch.long, device=boxes.device)
    return NmsTrace(safe_indices=safe_indices, suppressed_by=suppressed_by)


def match_candidates(
    boxes: torch.Tensor,
    classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return (
            boxes.new_zeros((boxes.shape[0],)),
            torch.full((boxes.shape[0],), -1, dtype=torch.long, device=boxes.device),
        )
    overlap = box_iou(boxes, gt_boxes)
    overlap = torch.where(
        classes[:, None] == gt_classes[None, :], overlap, torch.zeros_like(overlap)
    )
    best_iou, matched_gt = overlap.max(dim=1)
    return best_iou, matched_gt


def _nearest_safe_context(
    candidate_index: int,
    safe_indices: torch.Tensor,
    boxes: torch.Tensor,
    classes: torch.Tensor,
) -> int:
    if safe_indices.numel() == 0:
        return -1
    same_class = safe_indices[classes[safe_indices] == classes[candidate_index]]
    pool = same_class if same_class.numel() else safe_indices
    overlap = box_iou(boxes[candidate_index : candidate_index + 1], boxes[pool]).squeeze(0)
    if overlap.max() > 0:
        return int(pool[overlap.argmax()])

    candidate_box = boxes[candidate_index]
    candidate_center = (candidate_box[:2] + candidate_box[2:]) * 0.5
    pool_center = (boxes[pool, :2] + boxes[pool, 2:]) * 0.5
    distance = (pool_center - candidate_center).pow(2).sum(dim=1)
    return int(pool[distance.argmin()])


def _choose_restore_target(
    positive: torch.Tensor,
    iou: torch.Tensor,
    scores: torch.Tensor,
    global_indices: torch.Tensor,
) -> torch.Tensor:
    candidates = torch.nonzero(positive, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return torch.tensor(0, dtype=torch.long, device=positive.device)

    # Stable deterministic lexicographic order: IoU, raw score, then lower global index.
    chosen = int(candidates[0])
    for candidate in candidates[1:]:
        index = int(candidate)
        if iou[index] > iou[chosen]:
            chosen = index
        elif iou[index] == iou[chosen] and scores[index] > scores[chosen]:
            chosen = index
        elif (
            iou[index] == iou[chosen]
            and scores[index] == scores[chosen]
            and global_indices[index] < global_indices[chosen]
        ):
            chosen = index
    # Class zero is NONE; rollback candidates start at one.
    return torch.tensor(chosen + 1, dtype=torch.long, device=positive.device)


def build_rollback_targets(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    *,
    modality: torch.Tensor | None = None,
    candidate_conf_min: float = 0.01,
    base_conf: float = 0.25,
    base_nms_iou: float = 0.30,
    positive_iou: float = 0.50,
    max_safe_candidates: int | None = None,
    max_rollback_candidates: int | None = None,
    include_nms: bool = True,
    include_low_conf: bool = True,
    include_rgb: bool = True,
    include_pol: bool = True,
) -> RollbackTargets:
    """Build the immutable safe set and configurable bidirectional rollback pool.

    Pool construction uses predictions only. GT is used solely to construct training
    targets. Source and modality switches exist for T1/T2 and RGB/POL ablations.
    """
    trace = trace_classwise_nms(
        boxes,
        scores,
        classes,
        score_threshold=base_conf,
        iou_threshold=base_nms_iou,
    )
    safe_indices = trace.safe_indices
    if max_safe_candidates is not None:
        safe_indices = safe_indices[: int(max_safe_candidates)]

    safe_mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    safe_mask[safe_indices] = True
    nms_indices = (
        torch.nonzero(trace.suppressed_by >= 0, as_tuple=False).flatten()
        if include_nms
        else torch.empty(0, dtype=torch.long, device=boxes.device)
    )
    low_conf_indices = (
        torch.nonzero(
            (scores >= float(candidate_conf_min))
            & (scores < float(base_conf))
            & ~safe_mask,
            as_tuple=False,
        ).flatten()
        if include_low_conf
        else torch.empty(0, dtype=torch.long, device=boxes.device)
    )

    rollback_indices = torch.cat((nms_indices, low_conf_indices))
    rollback_source = torch.cat(
        (
            torch.full_like(nms_indices, SOURCE_NMS),
            torch.full_like(low_conf_indices, SOURCE_LOW_CONF),
        )
    )
    if modality is not None and rollback_indices.numel():
        allowed = torch.zeros(len(rollback_indices), dtype=torch.bool, device=boxes.device)
        if include_rgb:
            allowed |= modality[rollback_indices] == 0
        if include_pol:
            allowed |= modality[rollback_indices] == 1
        rollback_indices = rollback_indices[allowed]
        rollback_source = rollback_source[allowed]
    elif not include_rgb and not include_pol:
        rollback_indices = rollback_indices[:0]
        rollback_source = rollback_source[:0]

    if rollback_indices.numel():
        order = torch.argsort(scores[rollback_indices], descending=True, stable=True)
        rollback_indices = rollback_indices[order]
        rollback_source = rollback_source[order]
    if max_rollback_candidates is not None:
        limit = int(max_rollback_candidates)
        rollback_indices = rollback_indices[:limit]
        rollback_source = rollback_source[:limit]

    contexts: list[int] = []
    for local_index, global_index_tensor in enumerate(rollback_indices):
        global_index = int(global_index_tensor)
        if int(rollback_source[local_index]) == SOURCE_NMS:
            contexts.append(int(trace.suppressed_by[global_index]))
        else:
            contexts.append(_nearest_safe_context(global_index, safe_indices, boxes, classes))
    rollback_context_indices = torch.tensor(
        contexts, dtype=torch.long, device=boxes.device
    )

    safe_iou, safe_gt = match_candidates(
        boxes[safe_indices], classes[safe_indices], gt_boxes, gt_classes
    )
    safe_valid = safe_iou >= float(positive_iou)
    covered_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=boxes.device)
    if safe_valid.any():
        covered_gt[safe_gt[safe_valid]] = True

    rollback_iou, rollback_gt = match_candidates(
        boxes[rollback_indices], classes[rollback_indices], gt_boxes, gt_classes
    )
    rollback_valid = rollback_iou >= float(positive_iou)
    rollback_positive = rollback_valid.clone()
    if rollback_valid.any():
        rollback_positive[rollback_valid] = ~covered_gt[rollback_gt[rollback_valid]]

    restore_score_target = torch.where(
        rollback_positive, rollback_iou, torch.zeros_like(rollback_iou)
    )
    edge_target = torch.full(
        (len(rollback_indices),), EDGE_BACKGROUND, dtype=torch.long, device=boxes.device
    )
    for local_index in range(len(rollback_indices)):
        if not rollback_valid[local_index]:
            edge_target[local_index] = EDGE_BACKGROUND
        elif rollback_positive[local_index]:
            edge_target[local_index] = (
                EDGE_DISTINCT
                if int(rollback_source[local_index]) == SOURCE_NMS
                else EDGE_LOW_CONF_VALID
            )
        else:
            edge_target[local_index] = EDGE_DUPLICATE

    restore_target_index = _choose_restore_target(
        rollback_positive,
        rollback_iou,
        scores[rollback_indices],
        rollback_indices,
    )
    return RollbackTargets(
        safe_indices=safe_indices,
        rollback_indices=rollback_indices,
        rollback_source=rollback_source,
        rollback_context_indices=rollback_context_indices,
        safe_iou=safe_iou,
        safe_valid=safe_valid,
        rollback_iou=rollback_iou,
        rollback_matched_gt=rollback_gt,
        rollback_positive=rollback_positive,
        restore_score_target=restore_score_target,
        edge_target=edge_target,
        restore_target_index=restore_target_index,
    )


def build_edge_features(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    appearance: torch.Tensor,
    rollback_indices: torch.Tensor,
    context_indices: torch.Tensor,
) -> torch.Tensor:
    """Return relation features to the true suppressor or nearest safe context."""
    features = boxes.new_zeros((len(rollback_indices), 8))
    for local_index, global_index_tensor in enumerate(rollback_indices):
        global_index = int(global_index_tensor)
        context_index = int(context_indices[local_index])
        if context_index < 0:
            continue
        candidate = boxes[global_index]
        context = boxes[context_index]
        candidate_wh = (candidate[2:] - candidate[:2]).clamp_min(1e-6)
        context_wh = (context[2:] - context[:2]).clamp_min(1e-6)
        candidate_center = (candidate[:2] + candidate[2:]) * 0.5
        context_center = (context[:2] + context[2:]) * 0.5
        candidate_area = candidate_wh.prod()
        context_area = context_wh.prod()
        cosine = boxes.new_tensor(0.0)
        if appearance.shape[1] > 0:
            a = appearance[global_index]
            b = appearance[context_index]
            denominator = a.norm() * b.norm()
            if denominator > 0:
                cosine = torch.dot(a, b) / denominator
        features[local_index] = torch.stack(
            (
                box_iou(candidate[None], context[None])[0, 0],
                (candidate_center[0] - context_center[0]).abs(),
                (candidate_center[1] - context_center[1]).abs(),
                (torch.log(candidate_wh[0]) - torch.log(context_wh[0])).abs(),
                (torch.log(candidate_wh[1]) - torch.log(context_wh[1])).abs(),
                scores[global_index] - scores[context_index],
                torch.log(candidate_area) - torch.log(context_area),
                cosine,
            )
        )
    return features
