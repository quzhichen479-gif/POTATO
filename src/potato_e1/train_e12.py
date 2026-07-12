from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .e12_dataset import RollbackCacheDataset, collate_e12_candidate_batch
from .e12_losses import compute_e12_loss
from .e12_model import RollbackTransformer
from .train import apply_overrides, move_batch, seed_everything


def build_e12_dataset(config: dict[str, Any], split: str) -> RollbackCacheDataset:
    data = config["data"]
    if not data.get("manifest") or not data.get("root"):
        raise ValueError("data.manifest and data.root must be configured")
    return RollbackCacheDataset(
        manifest=data["manifest"],
        root=data["root"],
        split=split,
        topk_per_modality=int(data.get("topk_per_modality", 64)),
        max_safe_candidates=int(data.get("max_safe_candidates", 64)),
        max_rollback_candidates=int(data.get("max_rollback_candidates", 128)),
        appearance_dim=int(data.get("appearance_dim", 0)),
        candidate_conf_min=float(data.get("candidate_conf_min", 0.01)),
        base_conf=float(data.get("base_conf", 0.25)),
        base_nms_iou=float(data.get("base_nms_iou", 0.30)),
        positive_iou=float(data.get("positive_iou", 0.50)),
    )


def build_e12_model(config: dict[str, Any]) -> RollbackTransformer:
    data = config["data"]
    model = config["model"]
    return RollbackTransformer(
        appearance_dim=int(data.get("appearance_dim", 0)),
        edge_dim=int(model.get("edge_dim", 8)),
        edge_classes=int(model.get("edge_classes", 4)),
        d_model=int(model.get("d_model", 128)),
        nhead=int(model.get("nhead", 4)),
        safe_layers=int(model.get("safe_layers", 1)),
        rollback_layers=int(model.get("rollback_layers", 2)),
        dim_feedforward=int(model.get("dim_feedforward", 256)),
        dropout=float(model.get("dropout", 0.1)),
    )


def run_e12_epoch(
    model: RollbackTransformer,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    keys = (
        "loss",
        "setwise",
        "quality",
        "restore_score",
        "edge",
        "safe_score",
        "safe_ranking",
        "delta_reg",
    )
    totals = {key: 0.0 for key in keys}
    sample_count = 0

    progress = tqdm(loader, leave=False, desc="train-e12" if training else "val-e12")
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
            losses = compute_e12_loss(output, batch, config["loss"])

        if training:
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["train"].get("grad_clip_norm", 5.0))
            )
            scaler.step(optimizer)
            scaler.update()

        detached = losses.detached()
        for key in keys:
            totals[key] += detached[key] * batch_size
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
    train_dataset = build_e12_dataset(config, config["data"].get("train_split", "train"))
    val_dataset = build_e12_dataset(config, config["data"].get("val_split", "val"))
    loader_kwargs = {
        "batch_size": int(config["train"].get("batch_size", 16)),
        "num_workers": int(config["data"].get("num_workers", 4)),
        "collate_fn": collate_e12_candidate_batch,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_e12_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"].get("lr", 3e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["train"].get("epochs", 80))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = float("inf")
    stale_epochs = 0
    history_path = output_dir / "history.jsonl"
    for epoch in range(1, int(config["train"].get("epochs", 80)) + 1):
        train_metrics = run_e12_epoch(model, train_loader, device, config, optimizer, scaler)
        with torch.no_grad():
            val_metrics = run_e12_epoch(model, val_loader, device, config, None, scaler)
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
            "method": "E1.2 bidirectional suppression-confidence rollback Transformer",
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
