from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .e11_arbitration import (
    E11Thresholds,
    arbitrate_e11_single,
    fixed_baseline_single,
)
from .e11_dataset import collate_e11_candidate_batch
from .evaluate import greedy_match
from .train import move_batch
from .train_e11 import build_e11_dataset, build_e11_model


@dataclass
class MetricAccumulator:
    images: int = 0
    gt: int = 0
    tp: int = 0
    fp: int = 0
    protected_gt: int = 0
    destroyed_gt: int = 0
    common_miss_gt: int = 0
    rescued_gt: int = 0

    def update(
        self,
        *,
        gt_count: int,
        tp: int,
        fp: int,
        protected: torch.Tensor,
        covered: torch.Tensor,
    ) -> None:
        common_miss = ~protected
        self.images += 1
        self.gt += gt_count
        self.tp += tp
        self.fp += fp
        self.protected_gt += int(protected.sum())
        self.destroyed_gt += int((protected & ~covered).sum())
        self.common_miss_gt += int(common_miss.sum())
        self.rescued_gt += int((common_miss & covered).sum())

    def as_dict(self) -> dict[str, float | int]:
        return {
            "images": self.images,
            "gt": self.gt,
            "tp": self.tp,
            "fp": self.fp,
            "recall": self.tp / max(self.gt, 1),
            "precision": self.tp / max(self.tp + self.fp, 1),
            "fp_per_image": self.fp / max(self.images, 1),
            "destruction": self.destroyed_gt / max(self.protected_gt, 1),
            "rescue": self.rescued_gt / max(self.common_miss_gt, 1),
            "protected_gt": self.protected_gt,
            "destroyed_gt": self.destroyed_gt,
            "common_miss_gt": self.common_miss_gt,
            "rescued_gt": self.rescued_gt,
        }


def thresholds_from_config(config: dict[str, Any]) -> E11Thresholds:
    values = config["arbitration"]
    return E11Thresholds(
        **{
            key: values[key]
            for key in E11Thresholds.__dataclass_fields__
            if key in values
        }
    )


def modal_union_coverage(
    batch: dict[str, Any],
    device: torch.device,
    match_iou: float,
    base_conf: float,
) -> torch.Tensor:
    token_mask = batch["mask"][0]
    boxes = batch["boxes"][0][token_mask]
    scores = batch["scores"][0][token_mask]
    classes = batch["classes"][0][token_mask]
    modality = batch["modality"][0][token_mask]
    gt_boxes = batch["gt_boxes"][0].to(device)
    gt_classes = batch["gt_classes"][0].to(device)

    covered_by_modality: list[torch.Tensor] = []
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
        covered_by_modality.append(covered)
    return covered_by_modality[0] | covered_by_modality[1]


def _prediction_rows(result: Any) -> list[dict[str, Any]]:
    return [
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
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", default=None, help="locked validation YAML")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    locked_payload: dict[str, Any] | None = None
    if args.thresholds:
        with Path(args.thresholds).open("r", encoding="utf-8") as handle:
            locked_payload = yaml.safe_load(handle)
        config["arbitration"].update(
            locked_payload.get("arbitration", locked_payload)
        )

    require_lock = bool(
        config["evaluation"].get("require_locked_thresholds_for_test", True)
    )
    if args.split == config["data"].get("test_split", "test") and require_lock:
        if locked_payload is None or locked_payload.get("locked") is not True:
            raise RuntimeError(
                "test evaluation requires --thresholds with top-level locked: true; "
                "do not consume test while tuning"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_e11_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_e11_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        collate_fn=collate_e11_candidate_batch,
    )
    thresholds = thresholds_from_config(config)
    match_iou = float(config["evaluation"].get("match_iou", 0.5))
    base_conf = float(config["evaluation"].get("base_conf", 0.25))

    baseline_metrics = MetricAccumulator()
    method_metrics = MetricAccumulator()
    rows: list[dict[str, Any]] = []
    total_extra = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate-e11:{args.split}"):
            batch = move_batch(batch, device)
            output = model(batch)
            baseline = fixed_baseline_single(batch, 0, thresholds)
            result = arbitrate_e11_single(output, batch, 0, thresholds)

            if result.base_count != len(baseline.boxes):
                raise AssertionError("E1.1 safe-set cardinality changed")
            if result.base_count:
                if not torch.allclose(result.boxes[: result.base_count], baseline.boxes):
                    raise AssertionError("E1.1 modified a fixed-NMS safe box")
                if not torch.equal(
                    result.classes[: result.base_count], baseline.classes
                ):
                    raise AssertionError("E1.1 modified a fixed-NMS safe class")

            gt_boxes = batch["gt_boxes"][0].to(device)
            gt_classes = batch["gt_classes"][0].to(device)
            protected = modal_union_coverage(
                batch, device, match_iou=match_iou, base_conf=base_conf
            )

            base_covered, base_tp, base_fp = greedy_match(
                baseline.boxes,
                baseline.scores,
                baseline.classes,
                gt_boxes,
                gt_classes,
                match_iou,
            )
            final_covered, final_tp, final_fp = greedy_match(
                result.boxes,
                result.scores,
                result.classes,
                gt_boxes,
                gt_classes,
                match_iou,
            )
            baseline_metrics.update(
                gt_count=len(gt_boxes),
                tp=base_tp,
                fp=base_fp,
                protected=protected,
                covered=base_covered,
            )
            method_metrics.update(
                gt_count=len(gt_boxes),
                tp=final_tp,
                fp=final_fp,
                protected=protected,
                covered=final_covered,
            )
            total_extra += result.extra_count

            rows.append(
                {
                    "sample_id": batch["sample_id"][0],
                    "group": batch["group"][0],
                    "gt": len(gt_boxes),
                    "base_count": result.base_count,
                    "extra_count": result.extra_count,
                    "baseline_predictions": _prediction_rows(baseline),
                    "e11_predictions": _prediction_rows(result),
                }
            )

    metrics = {
        "method": "E1.1 fixed-NMS-anchored residual Transformer",
        "split": args.split,
        "baseline": baseline_metrics.as_dict(),
        "e11": method_metrics.as_dict(),
        "delta": {
            key: method_metrics.as_dict()[key] - baseline_metrics.as_dict()[key]
            for key in ("recall", "precision", "fp_per_image", "destruction", "rescue")
        },
        "total_extra": total_extra,
        "thresholds": thresholds.__dict__,
        "safe_set_invariant": "verified for every image",
        "note": "Compute AP50/AP50:95 with the project COCO evaluator from saved predictions.",
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).parent / f"{args.split}_e11_predictions.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "metrics", **metrics}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps({"type": "sample", **row}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
