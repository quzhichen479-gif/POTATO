from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def jitter_box(rng: np.random.Generator, box: np.ndarray, scale: float = 0.015) -> np.ndarray:
    jittered = box + rng.normal(0.0, scale, size=4)
    jittered[:2] = np.clip(jittered[:2], 0.0, 0.95)
    jittered[2:] = np.clip(jittered[2:], jittered[:2] + 0.01, 1.0)
    return jittered.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--appearance-dim", type=int, default=16)
    parser.add_argument("--physics-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    root = Path(args.output)
    sample_dir = root / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest_path = root / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index in range(args.samples):
            gt_count = int(rng.integers(1, 5))
            xy = rng.uniform(0.05, 0.75, size=(gt_count, 2))
            wh = rng.uniform(0.05, 0.18, size=(gt_count, 2))
            gt_boxes = np.concatenate((xy, np.minimum(xy + wh, 0.98)), axis=1).astype(np.float32)
            gt_classes = np.zeros(gt_count, dtype=np.int64)

            rgb_boxes: list[np.ndarray] = []
            pol_boxes: list[np.ndarray] = []
            rgb_scores: list[float] = []
            pol_scores: list[float] = []
            for gt_box in gt_boxes:
                mode = int(rng.choice(4, p=[0.60, 0.18, 0.12, 0.10]))
                if mode in (0, 1):
                    rgb_boxes.append(jitter_box(rng, gt_box))
                    rgb_scores.append(float(rng.uniform(0.55, 0.95)))
                if mode in (0, 2):
                    pol_boxes.append(jitter_box(rng, gt_box))
                    pol_scores.append(float(rng.uniform(0.55, 0.95)))

            for _ in range(int(rng.integers(0, 3))):
                x, y = rng.uniform(0.0, 0.85, size=2)
                w, h = rng.uniform(0.03, 0.12, size=2)
                target = rgb_boxes if rng.random() < 0.5 else pol_boxes
                target_scores = rgb_scores if target is rgb_boxes else pol_scores
                target.append(np.array([x, y, min(x + w, 1), min(y + h, 1)], dtype=np.float32))
                target_scores.append(float(rng.uniform(0.02, 0.45)))

            def stack_boxes(values: list[np.ndarray]) -> np.ndarray:
                return np.stack(values).astype(np.float32) if values else np.zeros((0, 4), np.float32)

            rgb_boxes_array = stack_boxes(rgb_boxes)
            pol_boxes_array = stack_boxes(pol_boxes)
            rgb_features = rng.normal(size=(len(rgb_boxes_array), args.appearance_dim)).astype(np.float32)
            pol_features = rng.normal(size=(len(pol_boxes_array), args.appearance_dim)).astype(np.float32)
            pol_physics = rng.normal(size=(len(pol_boxes_array), args.physics_dim)).astype(np.float32)

            sample_id = f"toy_{index:05d}"
            relative = f"samples/{sample_id}.npz"
            np.savez_compressed(
                root / relative,
                rgb_boxes=rgb_boxes_array,
                rgb_scores=np.asarray(rgb_scores, dtype=np.float32),
                rgb_classes=np.zeros(len(rgb_boxes_array), dtype=np.int64),
                rgb_features=rgb_features,
                pol_boxes=pol_boxes_array,
                pol_scores=np.asarray(pol_scores, dtype=np.float32),
                pol_classes=np.zeros(len(pol_boxes_array), dtype=np.int64),
                pol_features=pol_features,
                pol_physics=pol_physics,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
            )
            split = "train" if index < int(args.samples * 0.75) else "val"
            record = {
                "sample_id": sample_id,
                "cache": relative,
                "split": split,
                "group": f"toy_exp{index % 4 + 1:02d}",
                "oof_fold": index % 4 if split == "train" else None,
            }
            manifest.write(json.dumps(record) + "\n")

    print(manifest_path)


if __name__ == "__main__":
    main()
