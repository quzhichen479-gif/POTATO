from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = (
    "rgb_boxes",
    "rgb_scores",
    "rgb_classes",
    "pol_boxes",
    "pol_scores",
    "pol_classes",
    "gt_boxes",
    "gt_classes",
)


@dataclass(frozen=True)
class CacheRecord:
    sample_id: str
    cache: str
    split: str
    group: str
    oof_fold: int | None = None


class CacheFormatError(ValueError):
    pass


def _as_2d(array: np.ndarray, width: int, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] != width:
        raise CacheFormatError(f"{name} must have shape [N,{width}], got {array.shape}")
    return array


def _as_1d(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 1:
        raise CacheFormatError(f"{name} must have shape [N], got {array.shape}")
    return array


def _check_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise CacheFormatError(f"{name} contains NaN or Inf")


def validate_boxes(name: str, boxes: np.ndarray) -> None:
    _check_finite(name, boxes)
    if boxes.size == 0:
        return
    if (boxes < 0).any() or (boxes > 1).any():
        raise CacheFormatError(f"{name} must be normalized to [0,1]")
    if (boxes[:, 2] < boxes[:, 0]).any() or (boxes[:, 3] < boxes[:, 1]).any():
        raise CacheFormatError(f"{name} contains invalid xyxy ordering")


def load_cache(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as loaded:
        missing = [key for key in REQUIRED_KEYS if key not in loaded]
        if missing:
            raise CacheFormatError(f"{path}: missing keys {missing}")
        data = {key: np.asarray(loaded[key]) for key in loaded.files}

    for modality in ("rgb", "pol"):
        boxes = _as_2d(data[f"{modality}_boxes"], 4, f"{modality}_boxes").astype(
            np.float32, copy=False
        )
        scores = _as_1d(data[f"{modality}_scores"], f"{modality}_scores").astype(
            np.float32, copy=False
        )
        classes = _as_1d(data[f"{modality}_classes"], f"{modality}_classes").astype(
            np.int64, copy=False
        )
        if not (len(boxes) == len(scores) == len(classes)):
            raise CacheFormatError(f"{path}: {modality} arrays have inconsistent lengths")
        validate_boxes(f"{modality}_boxes", boxes)
        _check_finite(f"{modality}_scores", scores)
        if (scores < 0).any() or (scores > 1).any():
            raise CacheFormatError(f"{modality}_scores must be in [0,1]")
        data[f"{modality}_boxes"] = boxes
        data[f"{modality}_scores"] = scores
        data[f"{modality}_classes"] = classes

    gt_boxes = _as_2d(data["gt_boxes"], 4, "gt_boxes").astype(np.float32, copy=False)
    gt_classes = _as_1d(data["gt_classes"], "gt_classes").astype(np.int64, copy=False)
    if len(gt_boxes) != len(gt_classes):
        raise CacheFormatError(f"{path}: gt arrays have inconsistent lengths")
    validate_boxes("gt_boxes", gt_boxes)
    data["gt_boxes"] = gt_boxes
    data["gt_classes"] = gt_classes

    for optional_key, count_key in (
        ("rgb_features", "rgb_boxes"),
        ("pol_features", "pol_boxes"),
        ("pol_physics", "pol_boxes"),
    ):
        if optional_key not in data:
            continue
        value = np.asarray(data[optional_key], dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != data[count_key].shape[0]:
            raise CacheFormatError(
                f"{path}: {optional_key} must have shape [N,D] aligned with {count_key}"
            )
        _check_finite(optional_key, value)
        data[optional_key] = value

    return data


def record_from_json(obj: dict[str, Any]) -> CacheRecord:
    required = ("sample_id", "cache", "split", "group")
    missing = [key for key in required if key not in obj]
    if missing:
        raise CacheFormatError(f"manifest record missing keys {missing}")
    fold = obj.get("oof_fold")
    return CacheRecord(
        sample_id=str(obj["sample_id"]),
        cache=str(obj["cache"]),
        split=str(obj["split"]),
        group=str(obj["group"]),
        oof_fold=None if fold is None else int(fold),
    )
