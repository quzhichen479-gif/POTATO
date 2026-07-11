# Codex instructions for POTATO

Read `README.md` first, then `docs/E1_IMPLEMENTATION_SPEC.md`, and implement GitHub issue #1 in that order.

## Scope

Current scope is only E1: frozen RGB/POL detector candidate caching plus a lightweight Transformer candidate arbitrator. Do not introduce input-level fusion, P2 heads, SAHI, CARAFE, DySample, wavelets, new detector losses, or YOLO26 migration in this milestone.

## Non-negotiable rules

1. Never use test data to select thresholds, loss weights, NMS values, checkpoints, or feature sets.
2. Never use GT boxes to compute deployable physical features. Use predicted candidate boxes and their context.
3. Reproduce RGB-only, POL-only, and fixed cross-modal NMS from exactly the same cache before claiming a gain.
4. Complete A5 without physical features before A6 with physical features.
5. Keep failed ablations and full threshold curves.
6. Do not commit datasets, `.npz` caches, weights, checkpoints, or `runs/` artifacts.
7. Add tests for every change to target construction or arbitration logic.

## First commands

```bash
pip install -e .[dev]
pytest -q
python scripts/make_toy_cache.py --output /tmp/potato_e1_toy
python scripts/validate_cache.py --manifest /tmp/potato_e1_toy/manifest.jsonl --root /tmp/potato_e1_toy --require-oof-train
python -m potato_e1.train \
  --config configs/e1_transformer.yaml \
  data.manifest=/tmp/potato_e1_toy/manifest.jsonl \
  data.root=/tmp/potato_e1_toy \
  data.appearance_dim=16 \
  data.physics_dim=8 \
  data.num_workers=0 \
  train.epochs=2 \
  output_dir=/tmp/potato_e1_run
```

Before touching the real detector integration, make the toy-cache pipeline and tests pass.
