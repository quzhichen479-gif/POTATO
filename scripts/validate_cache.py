from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from potato_e1.schema import load_cache, record_from_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--require-oof-train", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    root = Path(args.root)
    seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    totals = Counter()
    appearance_dims: set[int] = set()
    physics_dims: set[int] = set()

    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = record_from_json(json.loads(line))
            if record.sample_id in seen:
                raise ValueError(f"duplicate sample_id at line {line_no}: {record.sample_id}")
            seen.add(record.sample_id)
            if args.require_oof_train and record.split == "train" and record.oof_fold is None:
                raise ValueError(f"train sample lacks oof_fold: {record.sample_id}")

            data = load_cache(root / record.cache)
            split_counts[record.split] += 1
            group_counts[record.group] += 1
            totals["rgb_candidates"] += len(data["rgb_boxes"])
            totals["pol_candidates"] += len(data["pol_boxes"])
            totals["gt"] += len(data["gt_boxes"])
            if len(data["rgb_boxes"]) + len(data["pol_boxes"]) == 0:
                raise ValueError(f"sample has no candidates in either modality: {record.sample_id}")
            if "rgb_features" in data:
                appearance_dims.add(data["rgb_features"].shape[1])
            if "pol_features" in data:
                appearance_dims.add(data["pol_features"].shape[1])
            if "pol_physics" in data:
                physics_dims.add(data["pol_physics"].shape[1])

    if len(appearance_dims) > 1:
        raise ValueError(f"inconsistent appearance feature dimensions: {appearance_dims}")
    if len(physics_dims) > 1:
        raise ValueError(f"inconsistent physical feature dimensions: {physics_dims}")

    report = {
        "samples": len(seen),
        "splits": dict(split_counts),
        "groups": dict(group_counts),
        "totals": dict(totals),
        "appearance_dim": next(iter(appearance_dims), 0),
        "physics_dim": next(iter(physics_dims), 0),
        "status": "OK",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
