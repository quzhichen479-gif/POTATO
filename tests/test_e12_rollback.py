from __future__ import annotations

import torch

from potato_e1.e12_arbitration import E12Thresholds, arbitrate_e12_single
from potato_e1.e12_dataset import collate_e12_candidate_batch
from potato_e1.e12_losses import compute_e12_loss
from potato_e1.e12_model import RollbackTransformer, RollbackTransformerOutput
from potato_e1.e12_targets import (
    EDGE_DISTINCT,
    EDGE_DUPLICATE,
    EDGE_LOW_CONF_VALID,
    SOURCE_LOW_CONF,
    SOURCE_NMS,
    build_edge_features,
    build_rollback_targets,
)


def _problem() -> tuple[torch.Tensor, ...]:
    boxes = torch.tensor(
        [
            [0.00, 0.00, 0.40, 0.40],
            [0.01, 0.01, 0.39, 0.39],
            [0.20, 0.00, 0.60, 0.40],
            [0.65, 0.00, 0.95, 0.30],
        ]
    )
    scores = torch.tensor([0.95, 0.90, 0.80, 0.10])
    classes = torch.zeros(4, dtype=torch.long)
    modality = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    appearance = torch.zeros((4, 0))
    gt_boxes = torch.tensor(
        [
            [0.00, 0.00, 0.40, 0.40],
            [0.20, 0.00, 0.60, 0.40],
            [0.65, 0.00, 0.95, 0.30],
        ]
    )
    gt_classes = torch.zeros(3, dtype=torch.long)
    return boxes, scores, classes, modality, appearance, gt_boxes, gt_classes


def _sample() -> dict[str, object]:
    boxes, scores, classes, modality, appearance, gt_boxes, gt_classes = _problem()
    targets = build_rollback_targets(
        boxes,
        scores,
        classes,
        gt_boxes,
        gt_classes,
        candidate_conf_min=0.01,
        base_conf=0.25,
        base_nms_iou=0.30,
        positive_iou=0.50,
    )
    edge = build_edge_features(
        boxes,
        scores,
        appearance,
        targets.rollback_indices,
        targets.rollback_context_indices,
    )
    safe = targets.safe_indices
    rollback = targets.rollback_indices
    return {
        "sample_id": "toy",
        "group": "exp00",
        "safe_boxes": boxes[safe],
        "safe_scores": scores[safe],
        "safe_classes": classes[safe],
        "safe_modality": modality[safe],
        "safe_appearance": appearance[safe],
        "safe_iou_target": targets.safe_iou,
        "safe_valid_target": targets.safe_valid.float(),
        "safe_rank_positive": targets.safe_valid,
        "safe_rank_negative": ~targets.safe_valid,
        "safe_source_indices": safe,
        "rollback_boxes": boxes[rollback],
        "rollback_scores": scores[rollback],
        "rollback_classes": classes[rollback],
        "rollback_modality": modality[rollback],
        "rollback_source": targets.rollback_source,
        "rollback_appearance": appearance[rollback],
        "rollback_edge": edge,
        "rollback_iou_target": targets.rollback_iou,
        "rollback_positive_target": targets.rollback_positive.float(),
        "restore_score_target": targets.restore_score_target,
        "edge_target": targets.edge_target,
        "rollback_source_indices": rollback,
        "rollback_context_indices": targets.rollback_context_indices,
        "restore_target_index": targets.restore_target_index,
        "all_boxes": boxes,
        "all_scores": scores,
        "all_classes": classes,
        "all_modality": modality,
        "gt_boxes": gt_boxes,
        "gt_classes": gt_classes,
    }


def test_rollback_pool_contains_both_sources_and_modalities() -> None:
    boxes, scores, classes, _, _, gt_boxes, gt_classes = _problem()
    targets = build_rollback_targets(
        boxes,
        scores,
        classes,
        gt_boxes,
        gt_classes,
        candidate_conf_min=0.01,
        base_conf=0.25,
        base_nms_iou=0.30,
        positive_iou=0.50,
    )
    assert targets.safe_indices.tolist() == [0]
    assert targets.rollback_indices.tolist() == [1, 2, 3]
    assert targets.rollback_source.tolist() == [SOURCE_NMS, SOURCE_NMS, SOURCE_LOW_CONF]
    assert targets.edge_target.tolist() == [
        EDGE_DUPLICATE,
        EDGE_DISTINCT,
        EDGE_LOW_CONF_VALID,
    ]
    assert targets.rollback_positive.tolist() == [False, True, True]
    assert int(targets.restore_target_index) == 2
    assert int(targets.rollback_context_indices[0]) == 0
    assert int(targets.rollback_context_indices[1]) == 0


def test_e12_forward_loss_and_backward() -> None:
    batch = collate_e12_candidate_batch([_sample(), _sample()])
    model = RollbackTransformer(
        appearance_dim=0,
        edge_dim=8,
        d_model=32,
        nhead=4,
        safe_layers=1,
        rollback_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    output = model(batch)
    assert output.restore_logits.shape == (2, batch["rollback_mask"].shape[1] + 1)
    assert output.rollback_quality.shape == batch["rollback_mask"].shape
    assert output.edge_logits.shape[-1] == 4
    losses = compute_e12_loss(
        output,
        batch,
        {
            "residual_alpha": 0.5,
            "set_weight": 1.0,
            "quality_weight": 1.0,
            "restore_score_weight": 1.0,
            "edge_weight": 0.25,
            "safe_score_weight": 0.5,
            "safe_ranking_weight": 0.5,
            "delta_reg_weight": 0.02,
        },
    )
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_e12_never_changes_safe_boxes_or_classes() -> None:
    batch = collate_e12_candidate_batch([_sample()])
    rollback_count = batch["rollback_mask"].shape[1]
    safe_count = batch["safe_mask"].shape[1]
    restore_logits = torch.full((1, rollback_count + 1), -5.0)
    restore_logits[0, 2] = 5.0  # choose local rollback index 1
    output = RollbackTransformerOutput(
        restore_logits=restore_logits,
        rollback_quality=torch.full((1, rollback_count), 0.9),
        restore_score_logit=torch.full((1, rollback_count), 4.0),
        edge_logits=torch.zeros((1, rollback_count, 4)),
        safe_score_delta=torch.zeros((1, safe_count)),
        safe_encoded=torch.zeros((1, safe_count, 8)),
        rollback_encoded=torch.zeros((1, rollback_count, 8)),
    )
    safe_boxes = batch["safe_boxes"][0][batch["safe_mask"][0]].clone()
    safe_classes = batch["safe_classes"][0][batch["safe_mask"][0]].clone()
    result = arbitrate_e12_single(
        output,
        batch,
        0,
        E12Thresholds(
            restore_probability_threshold=0.5,
            restore_quality_threshold=0.5,
            restore_score_threshold=0.1,
        ),
    )
    assert result.restore_count == 1
    assert result.base_count == len(safe_boxes)
    assert torch.equal(result.boxes[: result.base_count], safe_boxes)
    assert torch.equal(result.classes[: result.base_count], safe_classes)
    assert len(result.boxes) == len(safe_boxes) + 1
