from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .e12_targets import build_edge_features, build_rollback_targets
from .schema import CacheRecord, load_cache, record_from_json


class RollbackCacheDataset(Dataset[dict[str, Any]]):
    """Candidate-cache dataset for E1.2 suppression-confidence rollback."""

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        split: str,
        *,
        topk_per_modality: int = 64,
        max_safe_candidates: int = 64,
        max_rollback_candidates: int = 128,
        appearance_dim: int = 0,
        candidate_conf_min: float = 0.01,
        base_conf: float = 0.25,
        base_nms_iou: float = 0.30,
        positive_iou: float = 0.50,
        include_nms: bool = True,
        include_low_conf: bool = True,
        include_rgb: bool = True,
        include_pol: bool = True,
    ) -> None:
        self.root = Path(root)
        self.topk = int(topk_per_modality)
        self.max_safe = int(max_safe_candidates)
        self.max_rollback = int(max_rollback_candidates)
        self.appearance_dim = int(appearance_dim)
        self.candidate_conf_min = float(candidate_conf_min)
        self.base_conf = float(base_conf)
        self.base_nms_iou = float(base_nms_iou)
        self.positive_iou = float(positive_iou)
        self.include_nms = bool(include_nms)
        self.include_low_conf = bool(include_low_conf)
        self.include_rgb = bool(include_rgb)
        self.include_pol = bool(include_pol)

        records: list[CacheRecord] = []
        with Path(manifest).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = record_from_json(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"invalid manifest line {line_no}: {exc}") from exc
                if record.split == split:
                    records.append(record)
        if not records:
            raise ValueError(f"manifest contains no records for split={split!r}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _fit_feature_dim(value: np.ndarray | None, rows: int, dim: int) -> np.ndarray:
        if dim <= 0:
            return np.zeros((rows, 0), dtype=np.float32)
        result = np.zeros((rows, dim), dtype=np.float32)
        if value is None or value.size == 0:
            return result
        width = min(dim, value.shape[1])
        result[:, :width] = value[:, :width]
        return result

    def _select_modality(
        self, data: dict[str, np.ndarray], modality: str
    ) -> dict[str, np.ndarray]:
        scores = data[f"{modality}_scores"]
        order = np.argsort(-scores, kind="stable")[: self.topk]
        features = data.get(f"{modality}_features")
        return {
            "boxes": data[f"{modality}_boxes"][order],
            "scores": scores[order],
            "classes": data[f"{modality}_classes"][order],
            "features": self._fit_feature_dim(
                None if features is None else features[order], len(order), self.appearance_dim
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        data = load_cache(self.root / record.cache)
        rgb = self._select_modality(data, "rgb")
        pol = self._select_modality(data, "pol")

        boxes = torch.from_numpy(np.concatenate((rgb["boxes"], pol["boxes"]), axis=0)).float()
        scores = torch.from_numpy(
            np.concatenate((rgb["scores"], pol["scores"]), axis=0)
        ).float()
        classes = torch.from_numpy(
            np.concatenate((rgb["classes"], pol["classes"]), axis=0)
        ).long()
        modality = torch.cat(
            (
                torch.zeros(len(rgb["boxes"]), dtype=torch.long),
                torch.ones(len(pol["boxes"]), dtype=torch.long),
            )
        )
        appearance = torch.from_numpy(
            np.concatenate((rgb["features"], pol["features"]), axis=0)
        ).float()
        gt_boxes = torch.from_numpy(data["gt_boxes"]).float()
        gt_classes = torch.from_numpy(data["gt_classes"]).long()

        targets = build_rollback_targets(
            boxes,
            scores,
            classes,
            gt_boxes,
            gt_classes,
            modality=modality,
            candidate_conf_min=self.candidate_conf_min,
            base_conf=self.base_conf,
            base_nms_iou=self.base_nms_iou,
            positive_iou=self.positive_iou,
            max_safe_candidates=self.max_safe,
            max_rollback_candidates=self.max_rollback,
            include_nms=self.include_nms,
            include_low_conf=self.include_low_conf,
            include_rgb=self.include_rgb,
            include_pol=self.include_pol,
        )
        edge_features = build_edge_features(
            boxes,
            scores,
            appearance,
            targets.rollback_indices,
            targets.rollback_context_indices,
        )

        safe = targets.safe_indices
        rollback = targets.rollback_indices
        safe_rank_positive = targets.safe_iou >= self.positive_iou
        safe_rank_negative = ~safe_rank_positive

        return {
            "sample_id": record.sample_id,
            "group": record.group,
            "safe_boxes": boxes[safe],
            "safe_scores": scores[safe],
            "safe_classes": classes[safe],
            "safe_modality": modality[safe],
            "safe_appearance": appearance[safe],
            "safe_iou_target": targets.safe_iou,
            "safe_valid_target": targets.safe_valid.float(),
            "safe_rank_positive": safe_rank_positive,
            "safe_rank_negative": safe_rank_negative,
            "safe_source_indices": safe,
            "rollback_boxes": boxes[rollback],
            "rollback_scores": scores[rollback],
            "rollback_classes": classes[rollback],
            "rollback_modality": modality[rollback],
            "rollback_source": targets.rollback_source,
            "rollback_appearance": appearance[rollback],
            "rollback_edge": edge_features,
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


def collate_e12_candidate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_safe = max(1, max(sample["safe_boxes"].shape[0] for sample in samples))
    max_rollback = max(1, max(sample["rollback_boxes"].shape[0] for sample in samples))
    max_all = max(1, max(sample["all_boxes"].shape[0] for sample in samples))
    appearance_dim = samples[0]["safe_appearance"].shape[1]
    edge_dim = samples[0]["rollback_edge"].shape[1]

    def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    batch: dict[str, Any] = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "group": [sample["group"] for sample in samples],
        "safe_boxes": zeros(batch_size, max_safe, 4),
        "safe_scores": zeros(batch_size, max_safe),
        "safe_classes": zeros(batch_size, max_safe, dtype=torch.long),
        "safe_modality": zeros(batch_size, max_safe, dtype=torch.long),
        "safe_appearance": zeros(batch_size, max_safe, appearance_dim),
        "safe_mask": zeros(batch_size, max_safe, dtype=torch.bool),
        "safe_iou_target": zeros(batch_size, max_safe),
        "safe_valid_target": zeros(batch_size, max_safe),
        "safe_rank_positive": zeros(batch_size, max_safe, dtype=torch.bool),
        "safe_rank_negative": zeros(batch_size, max_safe, dtype=torch.bool),
        "safe_source_indices": zeros(batch_size, max_safe, dtype=torch.long),
        "rollback_boxes": zeros(batch_size, max_rollback, 4),
        "rollback_scores": zeros(batch_size, max_rollback),
        "rollback_classes": zeros(batch_size, max_rollback, dtype=torch.long),
        "rollback_modality": zeros(batch_size, max_rollback, dtype=torch.long),
        "rollback_source": zeros(batch_size, max_rollback, dtype=torch.long),
        "rollback_appearance": zeros(batch_size, max_rollback, appearance_dim),
        "rollback_edge": zeros(batch_size, max_rollback, edge_dim),
        "rollback_mask": zeros(batch_size, max_rollback, dtype=torch.bool),
        "rollback_iou_target": zeros(batch_size, max_rollback),
        "rollback_positive_target": zeros(batch_size, max_rollback),
        "restore_score_target": zeros(batch_size, max_rollback),
        "edge_target": zeros(batch_size, max_rollback, dtype=torch.long),
        "rollback_source_indices": zeros(batch_size, max_rollback, dtype=torch.long),
        "rollback_context_indices": torch.full(
            (batch_size, max_rollback), -1, dtype=torch.long
        ),
        "restore_target_index": torch.stack(
            [sample["restore_target_index"] for sample in samples]
        ),
        "all_boxes": zeros(batch_size, max_all, 4),
        "all_scores": zeros(batch_size, max_all),
        "all_classes": zeros(batch_size, max_all, dtype=torch.long),
        "all_modality": zeros(batch_size, max_all, dtype=torch.long),
        "all_mask": zeros(batch_size, max_all, dtype=torch.bool),
        "gt_boxes": [sample["gt_boxes"] for sample in samples],
        "gt_classes": [sample["gt_classes"] for sample in samples],
    }

    safe_keys = (
        "safe_boxes",
        "safe_scores",
        "safe_classes",
        "safe_modality",
        "safe_appearance",
        "safe_iou_target",
        "safe_valid_target",
        "safe_rank_positive",
        "safe_rank_negative",
        "safe_source_indices",
    )
    rollback_keys = (
        "rollback_boxes",
        "rollback_scores",
        "rollback_classes",
        "rollback_modality",
        "rollback_source",
        "rollback_appearance",
        "rollback_edge",
        "rollback_iou_target",
        "rollback_positive_target",
        "restore_score_target",
        "edge_target",
        "rollback_source_indices",
        "rollback_context_indices",
    )
    all_keys = ("all_boxes", "all_scores", "all_classes", "all_modality")

    for batch_index, sample in enumerate(samples):
        safe_count = sample["safe_boxes"].shape[0]
        rollback_count = sample["rollback_boxes"].shape[0]
        all_count = sample["all_boxes"].shape[0]
        batch["safe_mask"][batch_index, :safe_count] = True
        batch["rollback_mask"][batch_index, :rollback_count] = True
        batch["all_mask"][batch_index, :all_count] = True
        for key in safe_keys:
            batch[key][batch_index, :safe_count] = sample[key]
        for key in rollback_keys:
            batch[key][batch_index, :rollback_count] = sample[key]
        for key in all_keys:
            batch[key][batch_index, :all_count] = sample[key]
    return batch
