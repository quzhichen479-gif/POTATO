# SRTH V1 implementation specification

## 1. Scope

SRTH V1 is the first **feature-level, spatially aware tri-modal router** for PoTATO. It uses:

- RGB, DIF and POL features;
- grouped out-of-fold single-modality predictions as empirical Oracle supervision;
- physical/statistical heuristic maps as routing priors;
- an immutable RGB identity path;
- independent DIF and POL sigmoid gates at P3 and P4;
- the original local YOLO26 neck, head, assignment and detection loss.

This version does **not** introduce a hypergraph, EfficientViM backbone, Mamba block, new detector head or test-time late fusion.

The phrase “Oracle upper bound” means an **empirical Oracle upper bound** measured from single-modality OOF predictions. It is not a mathematically proven theoretical upper bound.

## 2. Evidence motivating SRTH

| Item | Result |
|---|---:|
| RGB-only recall | 82.24% |
| RGB ∪ DIF ∪ POL Oracle recall | 90.17% |
| Complementarity gain | +7.93 pp, about +427 TP |
| RGB-exclusive targets | 176 |
| DIF-exclusive targets | 108 |
| POL-exclusive targets | 184 |
| POL > RGB OOF AUROC | 0.6540 logistic / 0.6521 GBDT |
| FP-matched late-fusion net gain | +261 TP |
| Simple-fusion destruction | 0.6%–3.4% |

The data supports learned spatial routing. It does not yet prove that a new backbone or high-order graph is necessary.

## 3. Architecture

```text
RGB image ─ local YOLO26 RGB encoder ─ RGB P3/P4/P5 ───────────────┐
                                                                    │
DIF image ─ lightweight local encoder/adapter ─ DIF P3/P4 ──┐      │
                                                             ├ SRTH ├ local YOLO26 neck/head
POL image ─ lightweight local encoder/adapter ─ POL P3/P4 ──┘      │
                                                                    │
scale/DoLP/S0/saturation maps ─ heuristic prior ────────────────────┘
```

For each routed level `l ∈ {P3, P4}`:

```text
F_out_l = F_rgb_l
        + alpha_dif_l * G_dif_l * Delta_dif_l
        + alpha_pol_l * G_pol_l * Delta_pol_l
```

`G_dif` and `G_pol` are independent sigmoid gates, not a softmax. Both auxiliary modalities may be enabled for one target. `alpha_dif` and `alpha_pol` are initialized near zero so the initial model stays close to the RGB baseline.

P5 remains the RGB feature in V1.

## 4. Routing logits

The learned router uses RGB, projected DIF, projected POL, absolute RGB–DIF/RGB–POL differences and resized heuristic maps.

```text
route_logits = learned_logits + lambda_h * heuristic_prior_logits
```

The heuristic prior guides but does not hard-code the decision because its current OOF AUROC is useful but limited.

## 5. Empirical Oracle supervision

For GT `i` and modality `m`, calculate a quality value from the best OOF prediction:

```text
q_m(i) = 1[IoU_m >= threshold] * IoU_m^gamma * confidence_m^(1-gamma)
```

Create independent soft route targets:

```text
y_dif(i) = sigmoid((q_dif(i) - q_rgb(i)) / tau)
y_pol(i) = sigmoid((q_pol(i) - q_rgb(i)) / tau)
```

Do not use winner-takes-all labels. DIF and POL may both be helpful. Route labels must come from grouped OOF predictions. Never generate route supervision from a detector prediction on an image used to train that same detector.

## 6. Feature contract with PythonProject2

POTATO does not contain YOLO26. The local integration code in `PythonProject2` must provide:

```python
rgb_features = {"p3": Tensor[B,C3,H3,W3], "p4": ..., "p5": ...}
dif_features = {"p3": Tensor[B,D3,H3,W3], "p4": ...}
pol_features = {"p3": Tensor[B,P3,H3,W3], "p4": ...}
heuristics = Tensor[B,Hc,H0,W0]  # or per-level dictionary
```

DIF/POL P3 and P4 must have the same spatial sizes as the corresponding RGB level before calling `SRTHExternalFeatureBridge`. Channel widths may differ and are projected internally.

The returned `fused_features` replace only P3/P4 inputs to the existing local neck. P5 passes through unchanged.

## 7. Loss ownership

The local YOLO26 implementation remains responsible for its original detection loss. SRTH V1 adds Oracle route BCE, heuristic-prior consistency, low-weight spatial smoothness and gate-budget regularization.

```text
L_total = L_yolo26_original + lambda_srth * L_srth_v1
```

Do not edit YOLO26 assignment, Detect head or base box/classification loss for V1.

## 8. Required experiment order

1. RGB-only exact reproduction.
2. RGB + DIF direct add.
3. RGB + POL direct add.
4. RGB + DIF + POL concat/add baseline.
5. SRTH learned gates without heuristic prior.
6. SRTH heuristic prior without Oracle route loss.
7. SRTH full V1.
8. Three seeds: 41, 47 and 53.

Only after V1 is stable should the project evaluate non-degradation distillation or a target-centric hypergraph.

## 9. Acceptance checks

- RGB baseline reproduced with the same local CUDA setup.
- P3/P4 features and channels logged from the actual local YOLO26 model.
- Initial fused feature is numerically close to RGB.
- Gradients reach DIF, POL and route heads.
- P5 is unchanged.
- Test remains sealed.
- AP/AP75/APs, rescue, destruction and FP-matched net TP are reported.
- No dataset, weights, cache or `runs/` artifacts are committed.
