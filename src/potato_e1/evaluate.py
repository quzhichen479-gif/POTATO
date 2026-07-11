from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .arbitration import ArbitrationThresholds, arbitrate_single
from .dataset import collate_candidate_batch
from .targets import box_iou
from .train import build_dataset, build_model, move_batch


def greedy_match(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, int, int]:
    covered = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
    if len(boxes) == 0:
        return covered, 0, 0
    order = scores.argsort(descending=True)
    true_positive = 0
    false_positive = 0
    for prediction_index in order:
        same_class = gt_classes == classes[prediction_index]
        available = same_class & ~covered
        if not available.any():
            false_positive += 1
            continue
        candidate_gt = torch.nonzero(available, as_tuple=False).flatten()
        overlap = box_iou(boxes[prediction_index : prediction_index + 1], gt_boxes[candidate_gt])[0]
        best_overlap, local_index = overlap.max(dim=0)
        if best_overlap >= iou_threshold:
            covered[candidate_gt[local_index]] = True
            true_positive += 1
        else:
            false_positive += 1
    return covered, true_positive, false_positive


def thresholds_from_config(config: dict[str, Any]) -> ArbitrationThresholds:
    values = config["arbitration"]
    return ArbitrationThresholds(**{key: values[key] for key in ArbitrationThresholds.__dataclass_fields__ if key in values})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", default=None, help="optional locked YAML override")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.thresholds:
        with Path(args.thresholds).open("r", encoding="utf-8") as handle:
            locked = yaml.safe_load(handle)
        config["arbitration"].update(locked.get("arbitration", locked))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        collate_fn=collate_candidate_batch,
    )
    thresholds = thresholds_from_config(config)
    match_iou = float(config["evaluation"].get("match_iou", 0.5))
    base_conf = float(config["evaluation"].get("base_conf", 0.25))

    total_gt = total_tp = total_fp = 0
    protected_gt = destroyed_gt = common_miss_gt = rescued_gt = 0
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate:{args.split}"):
            batch = move_batch(batch, device)
            output = model(batch)
            result = arbitrate_single(output, batch, 0, thresholds)
            gt_boxes = batch["gt_boxes"][0].to(device)
            gt_classes = batch["gt_classes"][0].to(device)

            final_covered, tp, fp = greedy_match(
                result.boxes,
                result.scores,
                result.classes,
                gt_boxes,
                gt_classes,
                match_iou,
            )

            token_mask = batch["mask"][0]
            boxes = batch["boxes"][0][token_mask]
            scores = batch["scores"][0][token_mask]
            classes = batch["classes"][0][token_mask]
            modality = batch["modality"][0][token_mask]

            raw_covered: list[torch.Tensor] = []
            for modality_id in (0, 1):
                selected = (modality == modality_id) & (scores >= base_conf)
                covered, _, _ = greedy_match(
                    boxes[selected],
                    scores[selected],
                    classes[selected],
                    gt_boxes,
                    gt_classes,
                    match_iou,
                )
                raw_covered.append(covered)

            protected = raw_covered[0] | raw_covered[1]
            common_miss = ~protected
            destroyed = protected & ~final_covered
            rescued = common_miss & final_covered

            total_gt += len(gt_boxes)
            total_tp += tp
            total_fp += fp
            protected_gt += int(protected.sum())
            destroyed_gt += int(destroyed.sum())
            common_miss_gt += int(common_miss.sum())
            rescued_gt += int(rescued.sum())

            rows.append(
                {
                    "sample_id": batch["sample_id"][0],
                    "group": batch["group"][0],
                    "gt": len(gt_boxes),
                    "tp": tp,
                    "fp": fp,
                    "protected": int(protected.sum()),
                    "destroyed": int(destroyed.sum()),
                    "common_miss": int(common_miss.sum()),
                    "rescued": int(rescued.sum()),
                    "predictions": [
                        {
                            "box": box.tolist(),
                            "score": float(score),
                            "class": int(class_id),
                            "provenance": int(source),
                        }
                        for box, score, class_id, source in zip(
                            result.boxes.cpu(),
                            result.scores.cpu(),
                            result.classes.cpu(),
                            result.provenance.cpu(),
                        )
                    ],
                }
            )

    image_count = len(dataset)
    metrics = {
        "split": args.split,
        "images": image_count,
        "gt": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "recall": total_tp / max(total_gt, 1),
        "precision": total_tp / max(total_tp + total_fp, 1),
        "fp_per_image": total_fp / max(image_count, 1),
        "destruction": destroyed_gt / max(protected_gt, 1),
        "rescue": rescued_gt / max(common_miss_gt, 1),
        "protected_gt": protected_gt,
        "destroyed_gt": destroyed_gt,
        "common_miss_gt": common_miss_gt,
        "rescued_gt": rescued_gt,
        "thresholds": thresholds.__dict__,
        "note": "AP50/AP50:95 must be computed by the project COCO evaluator from saved predictions.",
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / f"{args.split}_predictions.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "metrics", **metrics}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps({"type": "sample", **row}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
