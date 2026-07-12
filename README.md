# POTATO — E1.2 Bidirectional Suppression–Confidence Rollback Transformer

本仓库当前唯一主线是：

> **E1.2：以 fixed cross-modal NMS 为不可修改安全集合，从 RGB/POL 的 NMS-suppressed 与 low-confidence 统一候选池中，每图恢复最多 1 个候选，或选择 NONE。**

旧 E1 和 E1.1 已保留用于失败消融复现，不再继续调参。

---

## 1. 为什么进入 E1.2

### E1.1 validation 结果

| 方法 | AP50:95 | Recall | FP/image | Destruction |
|---|---:|---:|---:|---:|
| 严格门槛 | 0.4547 | 0.8596 | 0.248 | 2.51% |
| R0 fixed NMS=0.3 | 0.4399 | 0.8701 | 0.195 | 3.58% |
| E1.1 R4 | 0.4425 | 0.8701 | 0.195 | 3.58% |

E1.1 的 bounded score residual 能安全提高 AP，但 POL-only extra 实际 `Rescue=0`，未通过 AP 与 Destruction 门槛。

### 修正后的 K=1 Oracle

| 指标 | 结果 |
|---|---:|
| Recall | 0.9273 |
| FP/image | 0.1950 |
| Destruction | 0.0038 |
| Δ Destruction vs R0 | -0.0320 |
| Net TP restored | +99 |
| Restored protected GT | 50 |
| Newly destroyed protected GT | 0 |

恢复来源：

- `nms_suppressed`：52；
- `low_conf`：47；
- RGB：66；
- POL：33。

因此 E1.2 必须同时覆盖两种来源和两个模态，不能再做 POL-only 补充。

---

## 2. E1.2 结构

```text
RGB/POL cached candidates
        │
        ├─ fixed class-wise NMS@0.30 ──> immutable safe set
        │
        └─ rollback pool
             ├─ RGB/POL NMS-suppressed
             └─ RGB/POL low-confidence (0.01 <= score < 0.25)
                       │
             set-wise Transformer + NONE token
                       │
             restore at most one candidate
                       │
        safe set + optional rollback candidate
```

硬约束：

- 不删除安全候选；
- 不移动或融合安全框；
- 不修改安全类别；
- 每图最多恢复 1 个；
- 恢复候选允许与 suppressor 高 IoU；
- 不使用物理特征；
- test 继续封存。

---

## 3. 核心实现

```text
configs/e1_2_rollback.yaml

docs/E1_2_IMPLEMENTATION_SPEC.md

src/potato_e1/
├─ e12_targets.py       # NMS trace、统一 rollback pool、set-wise 标签
├─ e12_dataset.py       # safe/rollback 双集合缓存数据集
├─ e12_model.py         # safe encoder + rollback encoder + cross attention + NONE
├─ e12_losses.py        # set-wise、quality、edge、score、bounded residual
├─ e12_arbitration.py   # 不可修改 safe set + K=1 restore
├─ train_e12.py
├─ evaluate_e12.py
└─ search_e12.py        # validation-only operating curve

tests/test_e12_rollback.py
```

模型输出：

```text
restore_logits        [B, R+1]   # NONE + rollback candidates
rollback_quality      [B, R]
restore_score_logit   [B, R]     # 跨图 AP 排序
edge_logits           [B, R, 4]
safe_score_delta      [B, S]
```

---

## 4. 安装与基础检查

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pytest -q
```

Toy cache：

```bash
python scripts/make_toy_cache.py --output /tmp/potato_e1_toy
python scripts/validate_cache.py \
  --manifest /tmp/potato_e1_toy/manifest.jsonl \
  --root /tmp/potato_e1_toy
```

---

## 5. 训练

```bash
python -m potato_e1.train_e12 \
  --config configs/e1_2_rollback.yaml \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache \
  data.appearance_dim=<candidate_feature_dim> \
  data.num_workers=0 \
  seed=41 \
  output_dir=runs/e1_2_t6_seed41
```

正式 debug validation 使用 3 个 arbitrator seed：

```text
41, 47, 53
```

---

## 6. Validation 评估与阈值曲线

评估：

```bash
python -m potato_e1.evaluate_e12 \
  --config configs/e1_2_rollback.yaml \
  --checkpoint runs/e1_2_t6_seed41/best.pt \
  --split val \
  --output runs/e1_2_t6_seed41/val_predictions.jsonl \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache \
  data.appearance_dim=<candidate_feature_dim>
```

Validation-only 搜索：

```bash
python -m potato_e1.search_e12 \
  --config configs/e1_2_rollback.yaml \
  --checkpoint runs/e1_2_t6_seed41/best.pt \
  --split val \
  --output runs/e1_2_t6_seed41/operating_curve.val.jsonl \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache \
  data.appearance_dim=<candidate_feature_dim>
```

`search_e12` 会拒绝 test。AP 字段故意留空，Codex 必须接入项目已有 COCO evaluator 计算 AP50/AP75/AP50:95 后才能锁定工作点。

---

## 7. Test 保护

Test 必须使用 validation 生成的锁定文件：

```yaml
locked: true
arbitration:
  residual_alpha: 0.50
  restore_probability_threshold: 0.50
  restore_quality_threshold: 0.50
  restore_score_threshold: 0.10
  max_restore_per_image: 1
```

没有顶层 `locked: true` 时，`evaluate_e12 --split test` 会直接拒绝运行。

当前阶段禁止运行 test，也禁止提前生成 14/14 detector OOF；先在 `full_train_debug` validation 完成结构筛选。

---

## 8. 消融顺序

| 版本 | 内容 |
|---|---|
| T0 | R0 fixed NMS=0.3 |
| T1 | NMS-suppressed pool，RGB/POL，set-wise K=1 |
| T2 | low-conf pool，RGB/POL，set-wise K=1 |
| T3 | NMS + low-conf unified pool |
| T4 | T3 + IoU quality + restore score |
| T5 | T4 + edge auxiliary loss |
| T6 | T5 + bounded residual reranking |

额外对照：

- RGB-only / POL-only / bidirectional rollback；
- NMS-only / low-conf-only / unified pool；
- independent BCE / set-wise + NONE。

每个版本运行 3 seed，保存完整 operating curve 和所有失败结果。

---

## 9. 严格验收门槛

```text
AP50:95 >= 0.4547
Recall >= 0.8596
FP/image <= 0.248
Destruction <= 0.0251
```

还必须满足：

- 与 T0 使用同一 cache、split、evaluator；
- safe box/class 逐图不变；
- 正确输出 `destruction_rate`、`delta_destruction_vs_r0`、`restored_protected_gt`、`newly_destroyed_protected_gt`；
- 3 seed 方向一致；
- test 未运行。

只有 debug validation 通过全部门槛，才允许生成 14/14 group-aware detector OOF cache。

---

## 10. Codex 读取顺序

1. `README.md`
2. `AGENTS.md`
3. `docs/E1_2_IMPLEMENTATION_SPEC.md`
4. 当前 E1.2 GitHub issue

Codex 先运行测试和 toy-cache smoke test，再完成 Oracle AP 审计、T0–T6、项目 COCO evaluator 对接和 3-seed validation。不得运行 test。
