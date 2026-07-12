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

from .e11_arbitration import E11Thresholds, arbitrate_e11_single
from .e11_dataset import collate_e11_candidate_batch
from .e11_model import ResidualArbitratorOutput
from .evaluate import greedy_match
from .evaluate_e11 import MetricAccumulator, modal_union_coverage
from .train import apply_overrides, move_batch
from .train_e11 import build_e11_dataset, build_e11_model


def _cpu_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _cpu_output(output: ResidualArbitratorOutput) -> ResidualArbitratorOutput:
    return ResidualArbitratorOutput(
        score_delta=output.score_delta.detach().cpu(),
        iou_pred=output.iou_pred.detach().cpu(),
        rescue_logit=output.rescue_logit.detach().cpu(),
        encoded=output.encoded.detach().cpu(),
    )


def _grid(config: dict[str, Any]) -> list[E11Thresholds]:
    search = config["search"]
    fixed = config["arbitration"]
    values = itertools.product(
        search["residual_alpha"],
        search["rescue_threshold"],
        search["extra_quality_threshold"],
        search["extra_score_floor"],
        search["max_extra_per_image"],
    )
    return [
        E11Thresholds(
            base_conf=float(fixed.get("base_conf", 0.25)),
            base_nms_iou=float(fixed.get("base_nms_iou", 0.30)),
            residual_alpha=float(alpha),
            rescue_threshold=float(rescue),
            extra_quality_threshold=float(quality),
            extra_score_floor=float(score_floor),
            extra_overlap_iou=float(fixed.get("extra_overlap_iou", 0.30)),
            max_extra_per_image=int(max_extra),
        )
        for alpha, rescue, quality, score_floor, max_extra in values
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
        raise RuntimeError("search_e11 is validation-only and refuses test")

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
    match_iou = float(config["evaluation"].get("match_iou", 0.5))
    base_conf = float(config["evaluation"].get("base_conf", 0.25))

    cached: list[tuple[dict[str, Any], ResidualArbitratorOutput, torch.Tensor]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="cache-e11-val-outputs"):
            batch = move_batch(batch, device)
            output = model(batch)
            protected = modal_union_coverage(
                batch, device, match_iou=match_iou, base_conf=base_conf
            )
            cached.append((_cpu_batch(batch), _cpu_output(output), protected.cpu()))

    search = config["search"]
    fp_limit = float(search["fp_per_image_limit"])
    destruction_limit = float(search["destruction_limit"])
    recall_floor = float(search["recall_floor"])
    rows: list[dict[str, Any]] = []

    for thresholds in tqdm(_grid(config), desc="search-e11-operating-points"):
        metrics = MetricAccumulator()
        total_extra = 0
        for batch, output, protected in cached:
            result = arbitrate_e11_single(output, batch, 0, thresholds)
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
            metrics.update(
                gt_count=len(gt_boxes),
                tp=tp,
                fp=fp,
                protected=protected,
                covered=covered,
            )
            total_extra += result.extra_count

        values = metrics.as_dict()
        feasible_without_ap = (
            values["recall"] >= recall_floor
            and values["fp_per_image"] <= fp_limit
            and values["destruction"] <= destruction_limit
        )
        rows.append(
            {
                **thresholds.__dict__,
                **values,
                "total_extra": total_extra,
                "feasible_without_ap": feasible_without_ap,
                "ap50_95": None,
            }
        )

    rows.sort(
        key=lambda row: (
            not row["feasible_without_ap"],
            -float(row["recall"]),
            float(row["fp_per_image"]),
            float(row["destruction"]),
        )
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "split": args.split,
        "points": len(rows),
        "feasible_without_ap": sum(bool(row["feasible_without_ap"]) for row in rows),
        "output": str(output_path),
        "warning": "AP is intentionally not approximated here; run the project COCO evaluator before locking a point.",
        "top_points": rows[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
