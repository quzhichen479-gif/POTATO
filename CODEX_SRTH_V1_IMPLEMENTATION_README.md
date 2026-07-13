# Codex task — integrate SRTH V1 with local CUDA YOLO26 in PythonProject2

## Read this first

This repository contains the reusable SRTH V1 routing implementation and tests. It deliberately does **not** contain:

- YOLO26 source files;
- the local modified Ultralytics package;
- YOLO26 model YAML files or weights;
- PoTATO image data and OOF caches;
- the user's CUDA environment.

Those files already exist locally in **`PythonProject2`**. Do not assume that POTATO is a standalone detector repository. Do not copy or vendor the complete YOLO26 project into POTATO.

Expected local layout:

```text
<LOCAL_WORKSPACE>/
├─ POTATO/          # this GitHub repository
└─ PythonProject2/  # local YOLO26 source, model files, data code and CUDA environment
```

The absolute Windows path may differ. Discover it locally; do not commit an absolute path.

## Mandatory environment rule

Use the **existing local CUDA Python environment that already runs PythonProject2/YOLO26**. Do not create a CPU-only environment and do not reinstall PyTorch unless the user explicitly requests it.

From the PythonProject2 environment, first run:

```powershell
cd <LOCAL_WORKSPACE>\PythonProject2
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
nvidia-smi
```

Then validate the local project import using the script in POTATO:

```powershell
python <LOCAL_WORKSPACE>\POTATO\scripts\check_srth_cuda_env.py `
  --yolo-root <LOCAL_WORKSPACE>\PythonProject2 `
  --import-module ultralytics
```

The command must report `cuda_available: true`. Stop and report the environment issue instead of silently training on CPU.

Install only the reusable POTATO package into the active PythonProject2 environment:

```powershell
python -m pip install -e <LOCAL_WORKSPACE>\POTATO --no-deps
```

`--no-deps` is required so pip does not replace the user's working CUDA PyTorch installation.

## Source of truth

Read in this order:

1. `CODEX_SRTH_V1_IMPLEMENTATION_README.md`
2. `docs/SRTH_V1_IMPLEMENTATION_SPEC.md`
3. `configs/srth_v1.yaml`
4. `src/potato_srth/routing.py`
5. `src/potato_srth/targets.py`
6. `src/potato_srth/losses.py`
7. current SRTH GitHub issue/PR

The old E1/E1.1/E1.2 code is retained for diagnostic history. Do not extend the E1.2 post-processing Transformer in this task.

## Task objective

Integrate SRTH V1 into the actual local YOLO26 pipeline in PythonProject2:

- RGB stays the complete primary path.
- DIF and POL use lightweight local encoders/adapters.
- Route only P3 and P4.
- Keep P5 from RGB unchanged.
- Add independent DIF/POL spatial sigmoid gates.
- Add empirical-Oracle route supervision from grouped OOF predictions.
- Add physical/statistical heuristic maps as a soft logit prior.
- Keep the original YOLO26 neck, Detect head, assignment and base detection loss.

Do not implement EfficientViM, HyperACE, a hypergraph, Mamba, a new head or late-fusion NMS in V1.

## Step 1 — inspect the real local YOLO26 implementation

Do not assume upstream Ultralytics line numbers or module names. Inspect `PythonProject2` and record:

- local package version and commit/state;
- model construction entry point;
- backbone stage outputs corresponding to P3/P4/P5;
- exact channel widths and strides;
- neck input contract;
- dataset/batch structure for aligned RGB/DIF/POL;
- existing training loss entry point;
- AMP/DDP behavior;
- export/inference path.

Write the discovered contract to an untracked local note or to a concise committed integration note without absolute paths.

## Step 2 — create the local integration layer in PythonProject2

Prefer a thin local module such as:

```text
PythonProject2/
└─ ultralytics/nn/modules/srth_v1.py
```

or the equivalent path used by the local fork. The local module may import:

```python
from potato_srth import SRTHExternalFeatureBridge, SRTHMultiScaleV1
```

Do not duplicate `potato_srth` code into multiple YOLO files.

Instantiate the router only after discovering actual feature widths:

```python
router = SRTHMultiScaleV1(
    channels={
        "p3": (rgb_p3_channels, dif_p3_channels, pol_p3_channels),
        "p4": (rgb_p4_channels, dif_p4_channels, pol_p4_channels),
    },
    heuristic_channels=num_heuristic_channels,
    hidden_channels=64,
    residual_scale_init=1e-3,
)
```

Feed returned `fused_features["p3"]` and `fused_features["p4"]` into the existing neck; pass RGB P5 unchanged.

## Step 3 — aligned tri-modal batches

RGB, DIF and POL must refer to the same original capture and geometric augmentation. Any resize, crop, flip, mosaic or affine transformation must be synchronized.

For the first controlled experiment, disable augmentations that cannot be proven to preserve exact cross-modal alignment. Log a sample triplet overlay before training.

Do not commit images or data paths.

## Step 4 — heuristic maps

Construct the configured channels from available PoTATO metadata/raw derivatives:

- target-scale proxy during training or scale-level proxy during inference;
- local DoLP mean;
- local DoLP standard deviation;
- HSV saturation;
- `intensity_s0_std`;
- sensor/image saturation ratio.

No GT-derived feature may be used at inference. A GT scale value may supervise routing during training, but the inference tensor must use a feature-level or input-derived proxy.

Normalize each heuristic channel using train-split statistics only. Save statistics in a small YAML/JSON file, not in the dataset directory.

## Step 5 — OOF Oracle targets

Use grouped OOF predictions from the existing RGB, DIF and POL single-modality detectors. Group by recording session/day/sequence so a detector never labels an image used in its own training fold.

For each GT, store only compact target data required by `potato_srth.targets`:

```text
image_id, gt_box, rgb_iou, rgb_score, dif_iou, dif_score, pol_iou, pol_score
```

Do not regenerate targets from in-sample predictions. Do not use test predictions.

Use `detector_quality()` and `oracle_route_targets()` to produce soft DIF-vs-RGB and POL-vs-RGB labels. Rasterize them to the actual P3/P4 feature sizes.

## Step 6 — training loss

Keep the existing YOLO26 loss intact:

```python
loss = yolo_original_loss + lambda_srth * srth_loss.total
```

Use `compute_srth_v1_loss()` for the auxiliary term. Start with `configs/srth_v1.yaml` and log each component separately.

Recommended staged start:

1. reproduce RGB-only;
2. one epoch with RGB backbone frozen and router/adapters trainable;
3. two additional router warmup epochs;
4. unfreeze according to the existing local training policy;
5. train seeds 41/47/53 only after the smoke run is correct.

## Step 7 — mandatory tests

In POTATO, using the active local environment:

```powershell
cd <LOCAL_WORKSPACE>\POTATO
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
pytest -q tests/test_srth_v1.py
python scripts/srth_v1_smoke.py --device cuda:0
```

In PythonProject2 add local integration tests for:

- P3/P4/P5 shapes and strides;
- aligned three-modal batch loading;
- RGB numerical preservation at initialization;
- nonzero gradients in DIF/POL adapters and route heads;
- AMP forward/backward on CUDA;
- single-GPU training smoke test;
- unchanged RGB P5;
- unchanged original YOLO26 Detect output contract;
- checkpoint save/load;
- inference without GT-derived tensors.

## Step 8 — experiment matrix

Run in order with identical split, resolution, schedule and evaluator:

```text
S0  RGB-only exact baseline
S1  RGB + DIF direct add
S2  RGB + POL direct add
S3  RGB + DIF + POL concat/add
S4  learned SRTH gates, no heuristic prior
S5  heuristic prior, no Oracle route loss
S6  full SRTH V1
```

After a successful single-seed debug, run seeds `41, 47, 53`.

Report:

- AP50:95, AP50, AP75, APs;
- recall and FP/image;
- target-level rescue and destruction;
- FP-matched net TP;
- route AUROC/calibration;
- mean DIF/POL gate by scale and physical condition;
- parameters, FLOPs, peak CUDA memory and latency.

Keep the test split sealed until validation structure and thresholds are fixed.

## Forbidden actions

- Do not copy the full PythonProject2/YOLO26 tree into POTATO.
- Do not commit local absolute paths, data, model weights, OOF caches, checkpoints or runs.
- Do not install CPU-only PyTorch over the working CUDA build.
- Do not change YOLO26 Detect, assignment or original detection loss in V1.
- Do not use test to design routes, thresholds or ablations.
- Do not claim a mathematical theoretical upper bound; use “empirical Oracle upper bound”.
- Do not add a hypergraph/backbone replacement before SRTH V1 is evaluated.

## Completion report

When finished, report:

1. exact PythonProject2 files changed;
2. detected local feature widths/strides;
3. CUDA/PyTorch/GPU versions;
4. unit and CUDA smoke-test outputs;
5. S0–S6 validation table;
6. route/rescue/destruction diagnostics;
7. remaining failures without hiding negative results.
