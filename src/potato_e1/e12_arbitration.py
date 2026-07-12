from __future__ import annotations

from dataclasses import dataclass

import torch

from .arbitration import ArbitrationResult
from .e11_losses import bounded_residual_score
from .e12_model import RollbackTransformerOutput


@dataclass(frozen=True)
class E12Thresholds:
    residual_alpha: float = 0.50
    restore_probability_threshold: float = 0.50
    restore_quality_threshold: float = 0.50
    restore_score_threshold: float = 0.10
    max_restore_per_image: int = 1


@dataclass
class E12ArbitrationResult(ArbitrationResult):
    base_count: int
    restore_count: int
    restore_local_index: int
    restore_source: int
    restore_modality: int
    restore_probability: float
    restore_quality: float
    restore_score: float
    source_indices: torch.Tensor


def fixed_baseline_e12_single(
    batch: dict[str, torch.Tensor], batch_index: int
) -> ArbitrationResult:
    mask = batch["safe_mask"][batch_index]
    return ArbitrationResult(
        boxes=batch["safe_boxes"][batch_index][mask],
        scores=batch["safe_scores"][batch_index][mask],
        classes=batch["safe_classes"][batch_index][mask],
        provenance=batch["safe_modality"][batch_index][mask],
    )


def arbitrate_e12_single(
    output: RollbackTransformerOutput,
    batch: dict[str, torch.Tensor],
    batch_index: int,
    thresholds: E12Thresholds,
) -> E12ArbitrationResult:
    """Append at most one rollback candidate while preserving every safe box/class."""
    if thresholds.max_restore_per_image not in (0, 1):
        raise ValueError("E1.2 currently supports max_restore_per_image in {0, 1}")

    safe_mask = batch["safe_mask"][batch_index]
    rollback_mask = batch["rollback_mask"][batch_index]
    safe_boxes = batch["safe_boxes"][batch_index][safe_mask]
    safe_raw_scores = batch["safe_scores"][batch_index][safe_mask]
    safe_classes = batch["safe_classes"][batch_index][safe_mask]
    safe_modality = batch["safe_modality"][batch_index][safe_mask]
    safe_source_indices = batch["safe_source_indices"][batch_index][safe_mask]
    safe_delta = output.safe_score_delta[batch_index][safe_mask]
    safe_scores = bounded_residual_score(
        safe_raw_scores, safe_delta, thresholds.residual_alpha
    )

    boxes = safe_boxes
    scores = safe_scores
    classes = safe_classes
    provenance = safe_modality
    source_indices = safe_source_indices

    selected_local = -1
    selected_source = -1
    selected_modality = -1
    selected_probability = 0.0
    selected_quality = 0.0
    selected_score = 0.0

    if thresholds.max_restore_per_image == 1 and rollback_mask.any():
        probability = torch.softmax(output.restore_logits[batch_index], dim=0)
        choice = int(probability.argmax())
        if choice > 0:
            local_index = choice - 1
            candidate_probability = float(probability[choice])
            candidate_quality = float(
                output.rollback_quality[batch_index, local_index]
            )
            candidate_score = float(
                torch.sigmoid(output.restore_score_logit[batch_index, local_index])
            )
            candidate_is_real = bool(rollback_mask[local_index])
            passes = (
                candidate_is_real
                and candidate_probability
                >= thresholds.restore_probability_threshold
                and candidate_quality >= thresholds.restore_quality_threshold
                and candidate_score >= thresholds.restore_score_threshold
            )
            if passes:
                restored_box = batch["rollback_boxes"][batch_index, local_index][None]
                restored_class = batch["rollback_classes"][batch_index, local_index][None]
                restored_modality = int(
                    batch["rollback_modality"][batch_index, local_index]
                )
                restored_source = int(
                    batch["rollback_source"][batch_index, local_index]
                )
                # Cross-image comparable confidence; quality discourages poorly localized boxes.
                final_restore_score = torch.sqrt(
                    torch.tensor(
                        candidate_score * candidate_quality,
                        dtype=scores.dtype,
                        device=scores.device,
                    ).clamp_min(0.0)
                )[None]
                boxes = torch.cat((boxes, restored_box), dim=0)
                scores = torch.cat((scores, final_restore_score), dim=0)
                classes = torch.cat((classes, restored_class), dim=0)
                provenance = torch.cat(
                    (
                        provenance,
                        torch.tensor(
                            [4 + restored_modality],
                            dtype=torch.long,
                            device=provenance.device,
                        ),
                    )
                )
                source_indices = torch.cat(
                    (
                        source_indices,
                        batch["rollback_source_indices"][
                            batch_index, local_index
                        ][None],
                    )
                )
                selected_local = local_index
                selected_source = restored_source
                selected_modality = restored_modality
                selected_probability = candidate_probability
                selected_quality = candidate_quality
                selected_score = candidate_score

    return E12ArbitrationResult(
        boxes=boxes,
        scores=scores,
        classes=classes,
        provenance=provenance,
        base_count=len(safe_boxes),
        restore_count=int(selected_local >= 0),
        restore_local_index=selected_local,
        restore_source=selected_source,
        restore_modality=selected_modality,
        restore_probability=selected_probability,
        restore_quality=selected_quality,
        restore_score=selected_score,
        source_indices=source_indices,
    )
