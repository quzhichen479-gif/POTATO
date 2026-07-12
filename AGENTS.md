# Codex instructions for POTATO

Read in this exact order:

1. `README.md`
2. `docs/E1_2_IMPLEMENTATION_SPEC.md`
3. the current E1.2 GitHub issue

The old E1/E1.1 files are retained only to reproduce failed ablations. Do not continue tuning the full-authority E1 arbitrator or the POL-only E1.1 extra branch.

## Current scope

Current scope is only **E1.2: bidirectional suppression-confidence rollback Transformer**.

The fixed cross-modal NMS output is an immutable safe set. The learned model may:

- apply a bounded score residual to safe-set scores;
- inspect RGB/POL NMS-suppressed candidates;
- inspect RGB/POL low-confidence candidates;
- choose one rollback candidate or NONE with a set-wise selector;
- append at most one candidate per image.

The learned model may not:

- delete, replace, move, merge, or relabel safe-set candidates;
- restore more than one candidate per image;
- ban NMS rollback candidates merely because they overlap the suppressor;
- use GT-derived inference features;
- use physical features;
- introduce unrelated detector changes.

## Non-negotiable rules

1. Never use test data to select thresholds, losses, checkpoints, features, source pools, score formulas, or ablations.
2. Keep test sealed until debug validation passes all gates, 14/14 group-aware detector OOF caches are complete, and validation thresholds are locked.
3. Reproduce same-cache T0 fixed NMS=0.3 before comparing E1.2.
4. Run T0→T1→T2→T3→T4→T5→T6 in order.
5. Report seeds 41/47/53, all operating points, and all failed variants.
6. Use the project’s existing COCO evaluator for AP50/AP75/AP50:95.
7. Run the O1/O2 Oracle AP audit before claiming the selector is the remaining bottleneck.
8. Do not commit datasets, `.npz` caches, detector weights, checkpoints, predictions, or `runs/` artifacts.
9. Add or update tests for every change to NMS trace, rollback-pool construction, target selection, or safe-set preservation.
10. Do not restore A6 physics; it was experimentally rejected.
11. Do not introduce P2, SAHI, CARAFE, DySample, wavelets, detector-loss changes, or YOLO26 migration in E1.2.

## First commands

```bash
pip install -e .[dev]
pytest -q
python scripts/make_toy_cache.py --output /tmp/potato_e1_toy
python scripts/validate_cache.py \
  --manifest /tmp/potato_e1_toy/manifest.jsonl \
  --root /tmp/potato_e1_toy
python -m potato_e1.train_e12 \
  --config configs/e1_2_rollback.yaml \
  data.manifest=/tmp/potato_e1_toy/manifest.jsonl \
  data.root=/tmp/potato_e1_toy \
  data.appearance_dim=16 \
  data.num_workers=0 \
  train.epochs=2 \
  output_dir=/tmp/potato_e12_run
python -m potato_e1.evaluate_e12 \
  --config configs/e1_2_rollback.yaml \
  --checkpoint /tmp/potato_e12_run/best.pt \
  --split val \
  data.manifest=/tmp/potato_e1_toy/manifest.jsonl \
  data.root=/tmp/potato_e1_toy \
  data.appearance_dim=16 \
  data.num_workers=0
```

## Required invariants

For every image:

```text
safe output count == T0 safe count
safe output boxes == T0 safe boxes
safe output classes == T0 safe classes
restore_count in {0, 1}
```

Destruction must be a non-negative rate. Report separately:

```text
destruction_rate
delta_destruction_vs_r0
restored_protected_gt
newly_destroyed_protected_gt
net_protected_gain
```

## Required acceptance gates

```text
AP50:95 >= 0.4547
Recall >= 0.8596
FP/image <= 0.248
Destruction <= 0.0251
```

Do not start 14/14 detector OOF generation until one debug-validation E1.2 variant passes all four gates across the required three arbitrator seeds.
