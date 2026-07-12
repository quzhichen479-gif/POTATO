from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from potato_e1.e12_dataset import collate_e12_candidate_batch
from potato_e1.evaluate import greedy_match
from potato_e1.train import apply_overrides
from potato_e1.train_e12 import build_e12_dataset


def _prediction_rows(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    provenance: torch.Tensor,
) -> list[dict[str, Any]]:
    return [
        {
            "box": box.tolist(),
            "score": float(score),
            "class": int(class_id),
            "provenance": int(source),
        }
        for box, score, class_id, source in zip(
            boxes, scores, classes, provenance
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    apply_overrides(config, args.overrides)
    if args.split == config["data"].get("test_split", "test"):
        raise RuntimeError("Oracle AP audit is validation-only and refuses test")

    dataset = build_e12_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        collate_fn=collate_e12_candidate_batch,
    )
    match_iou = float(config["evaluation"].get("match_iou", 0.5))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "oracle_selection_raw_score.val.jsonl"
    quality_path = output_dir / "oracle_selection_quality_score.val.jsonl"

    raw_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    raw_tp = raw_fp = quality_tp = quality_fp = total_gt = restored = 0

    for batch in tqdm(loader, desc="e12-oracle-ap-audit"):
        safe_mask = batch["safe_mask"][0]
        rollback_mask = batch["rollback_mask"][0]
        safe_boxes = batch["safe_boxes"][0][safe_mask]
        safe_raw_scores = batch["safe_scores"][0][safe_mask]
        safe_quality_scores = batch["safe_iou_target"][0][safe_mask]
        safe_classes = batch["safe_classes"][0][safe_mask]
        safe_provenance = batch["safe_modality"][0][safe_mask]

        target = int(batch["restore_target_index"][0])
        if target > 0:
            local_index = target - 1
            if not bool(rollback_mask[local_index]):
                raise AssertionError("restore target points to padded rollback token")
            restored += 1
            restore_box = batch["rollback_boxes"][0, local_index][None]
            restore_class = batch["rollback_classes"][0, local_index][None]
            restore_provenance = torch.tensor(
                [4 + int(batch["rollback_modality"][0, local_index])], dtype=torch.long
            )
            raw_boxes = torch.cat((safe_boxes, restore_box))
            raw_scores = torch.cat(
                (safe_raw_scores, batch["rollback_scores"][0, local_index][None])
            )
            raw_classes = torch.cat((safe_classes, restore_class))
            raw_provenance = torch.cat((safe_provenance, restore_provenance))
            quality_boxes = raw_boxes
            quality_scores = torch.cat(
                (
                    safe_quality_scores,
                    batch["rollback_iou_target"][0, local_index][None],
                )
            )
            quality_classes = raw_classes
            quality_provenance = raw_provenance
        else:
            raw_boxes = safe_boxes
            raw_scores = safe_raw_scores
            raw_classes = safe_classes
            raw_provenance = safe_provenance
            quality_boxes = safe_boxes
            quality_scores = safe_quality_scores
            quality_classes = safe_classes
            quality_provenance = safe_provenance

        gt_boxes = batch["gt_boxes"][0]
        gt_classes = batch["gt_classes"][0]
        _, current_raw_tp, current_raw_fp = greedy_match(
            raw_boxes,
            raw_scores,
            raw_classes,
            gt_boxes,
            gt_classes,
            match_iou,
        )
        _, current_quality_tp, current_quality_fp = greedy_match(
            quality_boxes,
            quality_scores,
            quality_classes,
            gt_boxes,
            gt_classes,
            match_iou,
        )
        total_gt += len(gt_boxes)
        raw_tp += current_raw_tp
        raw_fp += current_raw_fp
        quality_tp += current_quality_tp
        quality_fp += current_quality_fp

        common = {
            "sample_id": batch["sample_id"][0],
            "group": batch["group"][0],
            "gt": len(gt_boxes),
            "oracle_restored": int(target > 0),
        }
        raw_rows.append(
            {
                **common,
                "predictions": _prediction_rows(
                    raw_boxes, raw_scores, raw_classes, raw_provenance
                ),
            }
        )
        quality_rows.append(
            {
                **common,
                "predictions": _prediction_rows(
                    quality_boxes,
                    quality_scores,
                    quality_classes,
                    quality_provenance,
                ),
            }
        )

    image_count = len(dataset)
    summary = {
        "split": args.split,
        "images": image_count,
        "gt": total_gt,
        "oracle_restored_images": restored,
        "O1_raw_score": {
            "tp": raw_tp,
            "fp": raw_fp,
            "recall": raw_tp / max(total_gt, 1),
            "fp_per_image": raw_fp / max(image_count, 1),
            "predictions": str(raw_path),
        },
        "O2_oracle_quality_score": {
            "tp": quality_tp,
            "fp": quality_fp,
            "recall": quality_tp / max(total_gt, 1),
            "fp_per_image": quality_fp / max(image_count, 1),
            "predictions": str(quality_path),
        },
        "warning": (
            "This script exports GT-assisted diagnostic predictions. Run the project's "
            "existing COCO evaluator to obtain AP50/AP75/AP50:95. Never deploy or run on test."
        ),
    }
    with raw_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
        for row in raw_rows:
            handle.write(json.dumps({"type": "sample", **row}, ensure_ascii=False) + "\n")
    with quality_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
        for row in quality_rows:
            handle.write(json.dumps({"type": "sample", **row}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
