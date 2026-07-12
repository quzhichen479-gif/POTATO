from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .e12_arbitration import E12Thresholds, arbitrate_e12_single
from .e12_dataset import collate_e12_candidate_batch
from .e12_model import RollbackTransformerOutput
from .evaluate import greedy_match
from .evaluate_e12 import DetectionTotals, modal_union_coverage
from .train import apply_overrides, move_batch
from .train_e12 import build_e12_dataset, build_e12_model


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _cpu_output(output: RollbackTransformerOutput) -> RollbackTransformerOutput:
    return RollbackTransformerOutput(
        restore_logits=output.restore_logits.detach().cpu(),
        rollback_quality=output.rollback_quality.detach().cpu(),
        restore_score_logit=output.restore_score_logit.detach().cpu(),
        edge_logits=output.edge_logits.detach().cpu(),
        safe_score_delta=output.safe_score_delta.detach().cpu(),
        safe_encoded=output.safe_encoded.detach().cpu(),
        rollback_encoded=output.rollback_encoded.detach().cpu(),
    )


def threshold_grid(config: dict[str, Any]) -> list[E12Thresholds]:
    search = config["search"]
    values = itertools.product(
        search["residual_alpha"],
        search["restore_probability_threshold"],
        search["restore_quality_threshold"],
        search["restore_score_threshold"],
    )
    return [
        E12Thresholds(
            residual_alpha=float(alpha),
            restore_probability_threshold=float(probability),
            restore_quality_threshold=float(quality),
            restore_score_threshold=float(score),
            max_restore_per_image=1,
        )
        for alpha, probability, quality, score in values
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    apply_overrides(config, args.overrides)
    if args.split == config["data"].get("test_split", "test"):
        raise RuntimeError("search_e12 is validation-only and refuses test")

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
    match_iou = float(config["evaluation"].get("match_iou", 0.5))
    base_conf = float(config["evaluation"].get("base_conf", 0.25))

    cached: list[tuple[dict[str, Any], RollbackTransformerOutput, torch.Tensor]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="cache-e12-validation"):
            batch = move_batch(batch, device)
            output = model(batch)
            gt_boxes = batch["gt_boxes"][0].to(device)
            gt_classes = batch["gt_classes"][0].to(device)
            protected = modal_union_coverage(
                batch, gt_boxes, gt_classes, match_iou, base_conf
            )
            cached.append((_to_cpu(batch), _cpu_output(output), protected.cpu()))

    search = config["search"]
    recall_floor = float(search["recall_floor"])
    fp_limit = float(search["fp_per_image_limit"])
    destruction_limit = float(search["destruction_limit"])
    rows: list[dict[str, Any]] = []

    for thresholds in tqdm(threshold_grid(config), desc="search-e12-operating-points"):
        totals = DetectionTotals()
        protected_total = 0
        destroyed_total = 0
        restored_total = 0
        source_nms = source_low = modality_rgb = modality_pol = 0
        for batch, output, protected in cached:
            result = arbitrate_e12_single(output, batch, 0, thresholds)
            gt_boxes = batch["gt_boxes"][0]
            gt_classes = batch["gt_classes"][0]
            covered, tp, fp = greedy_match(
                result.boxes,
                result.scores,
                result.classes,
                gt_boxes,
                gt_classes,
                match_iou,
            )
            totals.update(len(gt_boxes), tp, fp)
            protected_total += int(protected.sum())
            destroyed_total += int((protected & ~covered).sum())
            restored_total += result.restore_count
            if result.restore_count:
                if result.restore_source == 0:
                    source_nms += 1
                else:
                    source_low += 1
                if result.restore_modality == 0:
                    modality_rgb += 1
                else:
                    modality_pol += 1

        values = totals.as_dict()
        destruction = destroyed_total / max(protected_total, 1)
        feasible_without_ap = (
            values["recall"] >= recall_floor
            and values["fp_per_image"] <= fp_limit
            and destruction <= destruction_limit
        )
        rows.append(
            {
                **thresholds.__dict__,
                **values,
                "destruction_rate": destruction,
                "restore_count": restored_total,
                "restore_nms": source_nms,
                "restore_low_conf": source_low,
                "restore_rgb": modality_rgb,
                "restore_pol": modality_pol,
                "feasible_without_ap": feasible_without_ap,
                "ap50_95": None,
            }
        )

    rows.sort(
        key=lambda row: (
            not row["feasible_without_ap"],
            -float(row["recall"]),
            float(row["fp_per_image"]),
            float(row["destruction_rate"]),
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "split": args.split,
                "points": len(rows),
                "feasible_without_ap": sum(
                    bool(row["feasible_without_ap"]) for row in rows
                ),
                "output": str(output_path),
                "warning": (
                    "AP is intentionally left null. Run the project COCO evaluator "
                    "on validation predictions before locking thresholds."
                ),
                "top_points": rows[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
