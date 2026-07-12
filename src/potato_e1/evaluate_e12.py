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

from .e12_arbitration import (
    E12Thresholds,
    arbitrate_e12_single,
    fixed_baseline_e12_single,
)
from .e12_dataset import collate_e12_candidate_batch
from .e12_targets import SOURCE_LOW_CONF, SOURCE_NMS, match_candidates
from .evaluate import greedy_match
from .train import apply_overrides, move_batch
from .train_e12 import build_e12_dataset, build_e12_model


@dataclass
class DetectionTotals:
    images: int = 0
    gt: int = 0
    tp: int = 0
    fp: int = 0

    def update(self, gt_count: int, tp: int, fp: int) -> None:
        self.images += 1
        self.gt += gt_count
        self.tp += tp
        self.fp += fp

    def as_dict(self) -> dict[str, float | int]:
        return {
            "images": self.images,
            "gt": self.gt,
            "tp": self.tp,
            "fp": self.fp,
            "recall": self.tp / max(self.gt, 1),
            "precision": self.tp / max(self.tp + self.fp, 1),
            "fp_per_image": self.fp / max(self.images, 1),
        }


def thresholds_from_config(config: dict[str, Any]) -> E12Thresholds:
    values = config["arbitration"]
    return E12Thresholds(
        **{
            key: values[key]
            for key in E12Thresholds.__dataclass_fields__
            if key in values
        }
    )


def modal_union_coverage(
    batch: dict[str, Any],
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    match_iou: float,
    base_conf: float,
) -> torch.Tensor:
    mask = batch["all_mask"][0]
    boxes = batch["all_boxes"][0][mask]
    scores = batch["all_scores"][0][mask]
    classes = batch["all_classes"][0][mask]
    modality = batch["all_modality"][0][mask]
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


def prediction_rows(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "box": box.tolist(),
            "score": float(score),
            "class": int(class_id),
            "provenance": int(source),
            "source_index": int(source_index),
        }
        for box, score, class_id, source, source_index in zip(
            result.boxes.cpu(),
            result.scores.cpu(),
            result.classes.cpu(),
            result.provenance.cpu(),
            result.source_indices.cpu(),
        )
    ]


def selected_restore_log(
    result: Any,
    batch: dict[str, Any],
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
) -> dict[str, Any] | None:
    if result.restore_count == 0:
        return None
    local_index = result.restore_local_index
    box = batch["rollback_boxes"][0, local_index][None]
    class_id = batch["rollback_classes"][0, local_index][None]
    iou, matched_gt = match_candidates(box, class_id, gt_boxes, gt_classes)
    source = result.restore_source
    context_index = int(batch["rollback_context_indices"][0, local_index])
    return {
        "restore_source": "nms_suppressed" if source == SOURCE_NMS else "low_conf",
        "modality": "rgb" if result.restore_modality == 0 else "pol",
        "suppressor_index": context_index if source == SOURCE_NMS else -1,
        "context_index": context_index,
        "candidate_index": int(
            batch["rollback_source_indices"][0, local_index]
        ),
        "candidate_iou_gt": float(iou[0]),
        "matched_gt": int(matched_gt[0]) if float(iou[0]) > 0 else -1,
        "candidate_raw_score": float(batch["rollback_scores"][0, local_index]),
        "restore_probability": result.restore_probability,
        "predicted_quality": result.restore_quality,
        "predicted_restore_score": result.restore_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    apply_overrides(config, args.overrides)

    locked_payload: dict[str, Any] | None = None
    if args.thresholds:
        with Path(args.thresholds).open("r", encoding="utf-8") as handle:
            locked_payload = yaml.safe_load(handle)
        config["arbitration"].update(
            locked_payload.get("arbitration", locked_payload)
        )

    test_split = config["data"].get("test_split", "test")
    if (
        args.split == test_split
        and config["evaluation"].get("require_locked_thresholds_for_test", True)
        and (locked_payload is None or locked_payload.get("locked") is not True)
    ):
        raise RuntimeError(
            "E1.2 test evaluation requires --thresholds with top-level locked: true"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_e12_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_e12_dataset(config, args.split)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        collate_fn=collate_e12_candidate_batch,
    )
    thresholds = thresholds_from_config(config)
    match_iou = float(config["evaluation"].get("match_iou", 0.5))
    base_conf = float(config["evaluation"].get("base_conf", 0.25))

    baseline_totals = DetectionTotals()
    method_totals = DetectionTotals()
    protected_total = 0
    baseline_destroyed = 0
    method_destroyed = 0
    restored_protected = 0
    newly_destroyed = 0
    restore_source_counts = {"nms_suppressed": 0, "low_conf": 0}
    restore_modality_counts = {"rgb": 0, "pol": 0}
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate-e12:{args.split}"):
            batch = move_batch(batch, device)
            output = model(batch)
            baseline = fixed_baseline_e12_single(batch, 0)
            result = arbitrate_e12_single(output, batch, 0, thresholds)

            if result.base_count != len(baseline.boxes):
                raise AssertionError("E1.2 changed safe-set cardinality")
            if result.base_count:
                if not torch.allclose(result.boxes[: result.base_count], baseline.boxes):
                    raise AssertionError("E1.2 moved a safe-set box")
                if not torch.equal(
                    result.classes[: result.base_count], baseline.classes
                ):
                    raise AssertionError("E1.2 changed a safe-set class")

            gt_boxes = batch["gt_boxes"][0].to(device)
            gt_classes = batch["gt_classes"][0].to(device)
            protected = modal_union_coverage(
                batch, gt_boxes, gt_classes, match_iou, base_conf
            )
            baseline_covered, baseline_tp, baseline_fp = greedy_match(
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
            baseline_totals.update(len(gt_boxes), baseline_tp, baseline_fp)
            method_totals.update(len(gt_boxes), final_tp, final_fp)

            protected_total += int(protected.sum())
            baseline_destroyed += int((protected & ~baseline_covered).sum())
            method_destroyed += int((protected & ~final_covered).sum())
            restored_protected += int(
                (protected & ~baseline_covered & final_covered).sum()
            )
            newly_destroyed += int(
                (protected & baseline_covered & ~final_covered).sum()
            )

            restore_log = selected_restore_log(result, batch, gt_boxes, gt_classes)
            if restore_log is not None:
                restore_source_counts[restore_log["restore_source"]] += 1
                restore_modality_counts[restore_log["modality"]] += 1

            rows.append(
                {
                    "sample_id": batch["sample_id"][0],
                    "group": batch["group"][0],
                    "gt": len(gt_boxes),
                    "base_count": result.base_count,
                    "restore_count": result.restore_count,
                    "restore": restore_log,
                    "baseline_predictions": [
                        {
                            "box": box.tolist(),
                            "score": float(score),
                            "class": int(class_id),
                            "provenance": int(source),
                        }
                        for box, score, class_id, source in zip(
                            baseline.boxes.cpu(),
                            baseline.scores.cpu(),
                            baseline.classes.cpu(),
                            baseline.provenance.cpu(),
                        )
                    ],
                    "e12_predictions": prediction_rows(result),
                }
            )

    baseline_metrics = baseline_totals.as_dict()
    method_metrics = method_totals.as_dict()
    baseline_destruction = baseline_destroyed / max(protected_total, 1)
    method_destruction = method_destroyed / max(protected_total, 1)
    metrics = {
        "method": "E1.2 bidirectional suppression-confidence rollback Transformer",
        "split": args.split,
        "baseline": {**baseline_metrics, "destruction_rate": baseline_destruction},
        "e12": {**method_metrics, "destruction_rate": method_destruction},
        "delta": {
            "recall": method_metrics["recall"] - baseline_metrics["recall"],
            "precision": method_metrics["precision"] - baseline_metrics["precision"],
            "fp_per_image": method_metrics["fp_per_image"] - baseline_metrics["fp_per_image"],
            "destruction_vs_r0": method_destruction - baseline_destruction,
            "net_tp": method_metrics["tp"] - baseline_metrics["tp"],
        },
        "protected_gt": protected_total,
        "restored_protected_gt": restored_protected,
        "newly_destroyed_protected_gt": newly_destroyed,
        "net_protected_gain": restored_protected - newly_destroyed,
        "restore_source_counts": restore_source_counts,
        "restore_modality_counts": restore_modality_counts,
        "thresholds": thresholds.__dict__,
        "safe_set_invariant": "verified for every image",
        "note": "Compute AP50/AP75/AP50:95 with the project COCO evaluator from saved predictions.",
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).parent / f"{args.split}_e12_predictions.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "metrics", **metrics}, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps({"type": "sample", **row}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
