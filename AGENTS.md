# Codex instructions for POTATO SRTH V1

Read in this exact order:

1. `CODEX_SRTH_V1_IMPLEMENTATION_README.md`
2. `docs/SRTH_V1_IMPLEMENTATION_SPEC.md`
3. `configs/srth_v1.yaml`
4. current SRTH GitHub issue/PR

## External project boundary

YOLO26 source, YAML/model files, weights, dataset code and the working CUDA environment are local in `PythonProject2`, not in POTATO.

Use the existing PythonProject2 CUDA interpreter. Install POTATO into that environment with:

```text
python -m pip install -e <path-to-POTATO> --no-deps
```

Never replace the working CUDA PyTorch build with a CPU package. Do not vendor the full PythonProject2/YOLO26 tree into POTATO. Do not commit absolute local paths.

## Current scope

Only implement SRTH V1:

- RGB complete primary path;
- lightweight DIF and POL adapters;
- P3/P4 independent spatial sigmoid gates;
- empirical Oracle route supervision from grouped OOF predictions;
- physical/statistical heuristic logit prior;
- RGB P5 passthrough;
- original YOLO26 neck/head/assignment/base loss unchanged.

Do not implement EfficientViM, Mamba, HyperACE, a hypergraph, a new detection head or late-fusion inference in this phase.

## Data and leakage rules

1. RGB/DIF/POL geometric transforms must remain aligned.
2. Route targets must come from grouped OOF single-modality predictions.
3. No detector may generate route labels for images used to train that detector fold.
4. Never use test to design modules, targets, thresholds, heuristics or ablations.
5. Never use a GT-derived input feature at inference.
6. Do not commit images, datasets, weights, OOF caches, checkpoints, predictions or `runs/`.

## Required invariants

```text
P3/P4 output spatial shapes == RGB P3/P4 shapes
P5 output is unchanged RGB P5
initial fused P3/P4 is numerically close to RGB
DIF and POL gates are independent sigmoid gates
YOLO26 Detect output contract is unchanged
original YOLO26 detection loss remains intact
```

## First local commands

Run from the existing PythonProject2 CUDA environment:

```text
python scripts/check_srth_cuda_env.py --yolo-root <path-to-PythonProject2>
python -m pip install -e <path-to-POTATO> --no-deps
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_srth_v1.py
python scripts/srth_v1_smoke.py --device cuda:0
```

## Experiment order

```text
S0 RGB-only
S1 RGB + DIF direct add
S2 RGB + POL direct add
S3 tri-modal concat/add
S4 learned gates, no heuristic
S5 heuristic prior, no Oracle route loss
S6 full SRTH V1
```

After a correct one-seed debug run, use seeds 41/47/53 and report all failed variants.
