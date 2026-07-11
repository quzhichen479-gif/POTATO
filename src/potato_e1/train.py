from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CandidateCacheDataset, collate_candidate_batch
from .losses import compute_loss
from .model import CandidateTransformerArbitrator


def parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        cursor = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = parse_value(raw_value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def build_dataset(config: dict[str, Any], split: str) -> CandidateCacheDataset:
    data = config["data"]
    if not data.get("manifest") or not data.get("root"):
        raise ValueError("data.manifest and data.root must be configured")
    return CandidateCacheDataset(
        manifest=data["manifest"],
        root=data["root"],
        split=split,
        topk_per_modality=int(data["topk_per_modality"]),
        appearance_dim=int(data.get("appearance_dim", 0)),
        physics_dim=int(data.get("physics_dim", 0)),
        positive_iou=float(data.get("positive_iou", 0.5)),
        pair_candidate_iou=float(data.get("pair_candidate_iou", 0.05)),
    )


def build_model(config: dict[str, Any]) -> CandidateTransformerArbitrator:
    data = config["data"]
    model = config["model"]
    return CandidateTransformerArbitrator(
        appearance_dim=int(data.get("appearance_dim", 0)),
        physics_dim=int(data.get("physics_dim", 0)),
        d_model=int(model["d_model"]),
        nhead=int(model["nhead"]),
        num_layers=int(model["num_layers"]),
        dim_feedforward=int(model["dim_feedforward"]),
        dropout=float(model["dropout"]),
        pair_dim=int(model["pair_dim"]),
    )


def run_epoch(
    model: CandidateTransformerArbitrator,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {key: 0.0 for key in ("loss", "valid", "iou", "pair", "rescue", "protect")}
    sample_count = 0

    progress = tqdm(loader, leave=False, desc="train" if training else "val")
    for batch in progress:
        batch = move_batch(batch, device)
        batch_size = len(batch["sample_id"])
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            enabled=bool(config["train"].get("amp", True) and device.type == "cuda"),
        ):
            output = model(batch)
            losses = compute_loss(
                output,
                batch,
                config["loss"],
                focal_alpha=float(config["loss"].get("focal_alpha", 0.25)),
                focal_gamma=float(config["loss"].get("focal_gamma", 2.0)),
            )

        if training:
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["train"].get("grad_clip_norm", 5.0))
            )
            scaler.step(optimizer)
            scaler.update()

        detached = losses.detached()
        for key, value in detached.items():
            totals[key] += value * batch_size
        sample_count += batch_size
        progress.set_postfix(loss=f"{detached['loss']:.4f}")

    if sample_count == 0:
        raise RuntimeError("empty data loader")
    return {key: value / sample_count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    apply_overrides(config, args.overrides)

    seed = int(config.get("seed", 47))
    seed_everything(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = build_dataset(config, config["data"].get("train_split", "train"))
    val_dataset = build_dataset(config, config["data"].get("val_split", "val"))
    loader_kwargs = {
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": int(config["data"].get("num_workers", 4)),
        "collate_fn": collate_candidate_batch,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["train"]["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = float("inf")
    stale_epochs = 0
    history_path = output_dir / "history.jsonl"
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, device, config, optimizer, scaler)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, config, None, scaler)
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["train"].get("early_stop_patience", 15)):
                print(f"early stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()
