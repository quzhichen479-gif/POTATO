# POTATO — SRTH V1 tri-modal spatial routing

The active research line is now:

> **SRTH V1: an RGB-preserving spatial router for aligned RGB, DIF and POL features, supervised by an empirical single-modality Oracle upper bound and guided by physical/statistical heuristics.**

YOLO26 source, model files, data and the CUDA environment are **not stored in this repository**. They are in the user's local `PythonProject2`. POTATO provides reusable routing, target and auxiliary-loss code plus the Codex integration contract.

## Diagnostic basis

| Diagnostic | Result |
|---|---:|
| RGB-only recall | 82.24% |
| RGB ∪ DIF ∪ POL Oracle recall | 90.17% |
| Complementarity | +7.93 pp, about +427 TP |
| RGB / DIF / POL exclusive hits | 176 / 108 / 184 |
| POL > RGB OOF AUROC | 0.6540 logistic, 0.6521 GBDT |
| FP-matched late-fusion gain | +261 TP |
| Simple-fusion destruction | 0.6%–3.4% |

These results justify learned routing. They do not yet justify a new backbone or hypergraph.

## V1 design

```text
RGB P3/P4 ──────────────────────────────────────────┐
DIF P3/P4 ─ lightweight projection ─ DIF gate ─────┤
POL P3/P4 ─ lightweight projection ─ POL gate ─────┼─ fused P3/P4 → local YOLO26 neck/head
heuristics ─ soft prior logits ─────────────────────┘
RGB P5 ─────────────────────────────────────────────── unchanged
```

- RGB identity path is always present.
- DIF and POL use independent sigmoid gates.
- Routing is applied only at P3 and P4 in V1.
- The original local YOLO26 neck, head, assignment and base loss remain unchanged.
- Test remains sealed.

## Repository layout

```text
configs/srth_v1.yaml
CODEX_SRTH_V1_IMPLEMENTATION_README.md
docs/SRTH_V1_IMPLEMENTATION_SPEC.md
src/potato_srth/
├─ routing.py     # multi-level RGB-preserving router
├─ targets.py     # OOF quality and soft Oracle route targets
├─ losses.py      # SRTH auxiliary routing losses
└─ external.py    # dependency-free PythonProject2 feature bridge
scripts/
├─ check_srth_cuda_env.py
└─ srth_v1_smoke.py
tests/test_srth_v1.py
```

## Local package tests

```bash
pip install -e .[dev]
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_srth_v1.py
python scripts/srth_v1_smoke.py --device cpu
```

For real integration and training, activate the existing CUDA environment from `PythonProject2`, install POTATO with `--no-deps`, and follow `CODEX_SRTH_V1_IMPLEMENTATION_README.md`.

## Current boundaries

SRTH V1 does not include EfficientViM, Mamba, HyperACE, a hypergraph, a new Detect head, a changed assignment strategy or test-time late fusion. Those are considered only after the lightweight route baseline is validated.

The older E1/E1.1/E1.2 files remain for diagnostic history and reproducibility; they are no longer the active implementation line.
