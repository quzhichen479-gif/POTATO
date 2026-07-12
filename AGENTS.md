# Codex instructions for POTATO

Read in this exact order:

1. `README.md`
2. `docs/E1_1_IMPLEMENTATION_SPEC.md`
3. the current E1.1 GitHub issue

The old `docs/E1_IMPLEMENTATION_SPEC.md` and `configs/e1_transformer.yaml` are retained only to reproduce the failed E1 A2–A6 ablations. Do not continue tuning the old full-authority arbitrator.

## Current scope

Current scope is only **E1.1: fixed-NMS-anchored residual Transformer**.

The fixed cross-modal NMS output is an immutable safe set. The learned model may:

- apply a bounded score residual for ranking;
- estimate IoU quality;
- admit at most one conservative POL extra per image.

The learned model may not:

- delete safe-set candidates;
- replace RGB/POL safe candidates;
- move or merge safe boxes;
- modify safe classes;
- use physical features.

## Non-negotiable rules

1. Never use test data to select thresholds, losses, checkpoints, NMS, alpha, features, or ablations.
2. Keep test sealed until a debug-validation version passes all gates, 14/14 detector OOF caches are complete, and validation thresholds are locked.
3. Reproduce same-cache fixed NMS=0.3 before comparing E1.1.
4. Run R0→R1→R2→R3→R4 in order; do not jump directly to a combined search.
5. Report all 3 arbitrator seeds, all operating points, and failed variants.
6. Use the project’s existing COCO evaluator for AP50/AP75/AP50:95.
7. Do not commit datasets, `.npz` caches, weights, checkpoints, predictions, or `runs/`.
8. Add or update tests for every change to safe-set construction, residual scoring, or extra admission.
9. Do not restore A6 physics; it was experimentally rejected.
10. Do not introduce P2, SAHI, CARAFE, DySample, wavelets, detector-loss changes, or YOLO26 migration in E1.1.

## First commands

```bash
pip install -e .[dev]
pytest -q
python scripts/make_toy_cache.py --output /tmp/potato_e1_toy
python scripts/validate_cache.py \
  --manifest /tmp/potato_e1_toy/manifest.jsonl \
  --root /tmp/potato_e1_toy
python -m potato_e1.train_e11 \
  --config configs/e1_1_residual.yaml \
  data.manifest=/tmp/potato_e1_toy/manifest.jsonl \
  data.root=/tmp/potato_e1_toy \
  data.appearance_dim=16 \
  data.num_workers=0 \
  train.epochs=2 \
  output_dir=/tmp/potato_e11_run
python -m potato_e1.evaluate_e11 \
  --config configs/e1_1_residual.yaml \
  --checkpoint /tmp/potato_e11_run/best.pt \
  --split val \
  data.manifest=/tmp/potato_e1_toy/manifest.jsonl \
  data.root=/tmp/potato_e1_toy
```

If the evaluation CLI does not yet accept dotted overrides, fix that before real experiments or provide a resolved config file. Do not work around it by editing the committed default paths.

## Required acceptance gates

```text
AP50:95 >= 0.4547
Recall >= 0.8596
FP/image <= 0.248
Destruction <= 0.0251
```

The program must also verify for every image that all fixed-NMS safe boxes and classes are present unchanged in the E1.1 output.
