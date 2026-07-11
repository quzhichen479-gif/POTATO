from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import CacheRecord, load_cache, record_from_json
from .targets import build_targets


class CandidateCacheDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        split: str,
        topk_per_modality: int = 64,
        appearance_dim: int = 0,
        physics_dim: int = 0,
        positive_iou: float = 0.5,
        pair_candidate_iou: float = 0.05,
    ) -> None:
        self.root = Path(root)
        self.topk = int(topk_per_modality)
        self.appearance_dim = int(appearance_dim)
        self.physics_dim = int(physics_dim)
        self.positive_iou = float(positive_iou)
        self.pair_candidate_iou = float(pair_candidate_iou)

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
    def _select_topk(data: dict[str, np.ndarray], modality: str, topk: int) -> dict[str, np.ndarray]:
        scores = data[f"{modality}_scores"]
        order = np.argsort(-scores, kind="stable")[:topk]
        selected = {
            "boxes": data[f"{modality}_boxes"][order],
            "scores": scores[order],
            "classes": data[f"{modality}_classes"][order],
        }
        feature_key = f"{modality}_features"
        if feature_key in data:
            selected["features"] = data[feature_key][order]
        if modality == "pol" and "pol_physics" in data:
            selected["physics"] = data["pol_physics"][order]
        return selected

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        data = load_cache(self.root / record.cache)
        rgb = self._select_topk(data, "rgb", self.topk)
        pol = self._select_topk(data, "pol", self.topk)

        rgb_boxes = torch.from_numpy(rgb["boxes"]).float()
        rgb_scores = torch.from_numpy(rgb["scores"]).float()
        rgb_classes = torch.from_numpy(rgb["classes"]).long()
        pol_boxes = torch.from_numpy(pol["boxes"]).float()
        pol_scores = torch.from_numpy(pol["scores"]).float()
        pol_classes = torch.from_numpy(pol["classes"]).long()
        gt_boxes = torch.from_numpy(data["gt_boxes"]).float()
        gt_classes = torch.from_numpy(data["gt_classes"]).long()

        targets = build_targets(
            rgb_boxes=rgb_boxes,
            rgb_classes=rgb_classes,
            pol_boxes=pol_boxes,
            pol_classes=pol_classes,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            positive_iou=self.positive_iou,
            pair_candidate_iou=self.pair_candidate_iou,
        )

        rgb_appearance = self._fit_feature_dim(
            rgb.get("features"), len(rgb_boxes), self.appearance_dim
        )
        pol_appearance = self._fit_feature_dim(
            pol.get("features"), len(pol_boxes), self.appearance_dim
        )
        rgb_physics = np.zeros((len(rgb_boxes), self.physics_dim), dtype=np.float32)
        pol_physics = self._fit_feature_dim(
            pol.get("physics"), len(pol_boxes), self.physics_dim
        )

        return {
            "sample_id": record.sample_id,
            "group": record.group,
            "boxes": torch.cat((rgb_boxes, pol_boxes), dim=0),
            "scores": torch.cat((rgb_scores, pol_scores), dim=0),
            "classes": torch.cat((rgb_classes, pol_classes), dim=0),
            "modality": torch.cat(
                (
                    torch.zeros(len(rgb_boxes), dtype=torch.long),
                    torch.ones(len(pol_boxes), dtype=torch.long),
                )
            ),
            "appearance": torch.from_numpy(
                np.concatenate((rgb_appearance, pol_appearance), axis=0)
            ).float(),
            "physics": torch.from_numpy(
                np.concatenate((rgb_physics, pol_physics), axis=0)
            ).float(),
            "valid_target": targets.valid,
            "iou_target": targets.iou,
            "rescue_target": targets.rescue,
            "protect_target": targets.protect,
            "pair_same_target": targets.pair_same,
            "pair_mask": targets.pair_mask,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
            "n_rgb": len(rgb_boxes),
            "n_pol": len(pol_boxes),
        }


def collate_candidate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_tokens = max(sample["boxes"].shape[0] for sample in samples)
    appearance_dim = samples[0]["appearance"].shape[1]
    physics_dim = samples[0]["physics"].shape[1]

    def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    batch: dict[str, Any] = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "group": [sample["group"] for sample in samples],
        "boxes": zeros(batch_size, max_tokens, 4),
        "scores": zeros(batch_size, max_tokens),
        "classes": zeros(batch_size, max_tokens, dtype=torch.long),
        "modality": zeros(batch_size, max_tokens, dtype=torch.long),
        "appearance": zeros(batch_size, max_tokens, appearance_dim),
        "physics": zeros(batch_size, max_tokens, physics_dim),
        "mask": zeros(batch_size, max_tokens, dtype=torch.bool),
        "valid_target": zeros(batch_size, max_tokens),
        "iou_target": zeros(batch_size, max_tokens),
        "rescue_target": zeros(batch_size, max_tokens),
        "protect_target": zeros(batch_size, max_tokens),
        "pair_same_target": zeros(batch_size, max_tokens, max_tokens),
        "pair_mask": zeros(batch_size, max_tokens, max_tokens, dtype=torch.bool),
        "gt_boxes": [sample["gt_boxes"] for sample in samples],
        "gt_classes": [sample["gt_classes"] for sample in samples],
        "n_rgb": torch.tensor([sample["n_rgb"] for sample in samples], dtype=torch.long),
        "n_pol": torch.tensor([sample["n_pol"] for sample in samples], dtype=torch.long),
    }

    token_keys = (
        "boxes",
        "scores",
        "classes",
        "modality",
        "appearance",
        "physics",
        "valid_target",
        "iou_target",
        "rescue_target",
        "protect_target",
    )
    for batch_index, sample in enumerate(samples):
        count = sample["boxes"].shape[0]
        batch["mask"][batch_index, :count] = True
        for key in token_keys:
            batch[key][batch_index, :count] = sample[key]
        batch["pair_same_target"][batch_index, :count, :count] = sample[
            "pair_same_target"
        ]
        batch["pair_mask"][batch_index, :count, :count] = sample["pair_mask"]

    return batch
