# E1.2 工程规范：双向抑制—低置信度统一回滚 Transformer

## 0. 当前结论

E1.1 已在 `full_train_debug` validation 上失败：

| 方法 | AP50:95 | Recall | FP/image | Destruction |
|---|---:|---:|---:|---:|
| R0 fixed NMS=0.3 | 0.4399 | 0.8701 | 0.195 | 3.58% |
| E1.1 R4 | 0.4425 | 0.8701 | 0.195 | 3.58% |
| 严格门槛 | 0.4547 | 0.8596 | 0.248 | 2.51% |

E1.1 的 bounded residual 能无损提高 AP，但 POL-only extra 与 NMS overlap 规则冲突，实际 `Rescue=0`。

修正后的 Suppression Recovery Oracle（每图最多恢复一个，K=1）为：

- Recall：0.9273；
- FP/image：0.1950；
- Destruction：0.0038；
- 相比 R0 的 Destruction 变化：-0.0320；
- 净恢复 TP：+99；
- restored protected GT：50；
- newly destroyed protected GT：0；
- 来源：NMS-suppressed 52，low-confidence 47；
- 模态：RGB 66，POL 33。

因此 E1.2 必须同时覆盖两种来源和两个模态，不能再做 POL-only 补充。

---

## 1. 方法边界

E1.2 的唯一目标是：

> 在不可修改 fixed-NMS 安全集合的前提下，从 RGB/POL 的 NMS-suppressed 与 low-confidence 统一候选池中，每图选择最多一个候选恢复，或选择 NONE。

E1.2 允许：

1. 对安全集合分数施加 bounded logit residual；
2. 恢复一个 RGB 或 POL 候选；
3. 恢复候选可以与 suppressor 高度重叠；
4. 使用预测候选 geometry、detector score、appearance feature、source/modality、suppression edge。

E1.2 禁止：

- 删除、替换、移动、融合安全集合 box；
- 修改安全集合 class；
- 每图恢复超过 1 个候选；
- 使用 GT 生成推理输入；
- 恢复 A6 物理特征；
- 使用 test 调阈值、损失、checkpoint 或特征；
- 同时引入 P2、SAHI、CARAFE、DySample、小波、检测器 loss 或 YOLO26。

---

## 2. 固定安全集合与 NMS trace

输入为同一缓存中的 RGB/POL 候选并集。第一版固定：

```text
candidate_conf_min = 0.01
base_conf = 0.25
base_nms_iou = 0.30
```

运行 deterministic class-wise NMS，并保存：

```text
safe_indices
suppressed_by[candidate_index]
```

对于每个 NMS-suppressed 候选，必须知道实际的 `suppressor_index`。不能只记录“未保留”。

安全集合约束：

```text
safe_set boxes/classes ⊆ final_set boxes/classes
```

每图评估时必须 assert：

- safe count 不变；
- safe box 坐标逐项不变；
- safe class 逐项不变。

---

## 3. 统一 rollback pool

```text
rollback_pool = nms_suppressed ∪ low_confidence
```

其中：

- `nms_suppressed`：score >= 0.25，且被 fixed NMS 抑制；
- `low_confidence`：0.01 <= score < 0.25；
- 两类均同时允许 RGB 与 POL。

每个 token 必须携带：

```text
box xyxy
raw score
class
modality: RGB/POL
source: NMS/LOW_CONF
appearance feature（若缓存提供）
source candidate index
context index
```

NMS 候选的 context 是实际 suppressor；low-conf 候选的 context 是同类最大 IoU safe candidate，若没有同类则为最近 safe candidate。

### 3.1 edge features

当前实现使用 8 维：

1. candidate-context IoU；
2. |Δcx|；
3. |Δcy|；
4. |log width ratio|；
5. |log height ratio|；
6. candidate score - context score；
7. log area ratio；
8. appearance cosine similarity。

low-conf 没有 suppressor，正式日志中 `suppressor_index=-1`，但保留 `context_index`。

---

## 4. 训练标签

先用 safe set 覆盖 GT。rollback candidate 只有同时满足：

```text
same class
IoU(candidate, GT) >= 0.5
GT not covered by safe_set
```

才是 restore positive。

### 4.1 set-wise target

模型输出：

```text
[NONE, candidate_0, candidate_1, ...]
```

每图只有一个 target：

- 无可恢复候选：NONE；
- 有多个正候选：按 IoU、raw score、低 candidate index 的稳定字典序选一个。

K=1 必须通过 set-wise softmax 建模，不能用多个独立 BCE 后再临时 top-1 代替主实验。

### 4.2 edge auxiliary labels

```text
0 duplicate_covered
1 nms_distinct_uncovered
2 background
3 low_conf_valid_uncovered
```

### 4.3 quality 与 restore score

- `rollback_iou_target`：候选与最佳同类 GT 的 IoU；
- `restore_score_target`：restore positive 使用 IoU，否则为 0；
- 该 score 用于跨图 AP 排序，不能直接使用 set-wise softmax 概率替代。

---

## 5. 模型

当前实现：

```text
safe encoder
rollback encoder
rollback-to-safe cross attention
NONE token
```

输出：

```text
restore_logits        [B, R+1]
rollback_quality      [B, R]
restore_score_logit   [B, R]
edge_logits           [B, R, 4]
safe_score_delta      [B, S]
```

推理：

1. set-wise softmax 选择 NONE 或一个 rollback candidate；
2. 同时检查 restore probability、quality、restore score；
3. 通过时追加一个候选；
4. 安全集合 box/class 完全不变；
5. restored final score 当前为 `sqrt(restore_score * quality)`；该规则只能在 validation 消融和锁定。

---

## 6. 损失

```text
L = L_set
  + λq L_quality
  + λs L_restore_score
  + λe L_edge
  + λb L_safe_score
  + λr L_safe_ranking
  + λd L_delta_reg
```

- `L_set`：NONE + candidates 的 cross entropy；
- `L_quality`：rollback IoU SmoothL1；
- `L_restore_score`：跨图可比较的 restore score；
- `L_edge`：四类 edge/source 辅助监督；
- `L_safe_score/ranking`：保留 E1.1 bounded residual 的排序价值；
- `L_delta_reg`：限制 safe residual。

---

## 7. 运行顺序

### 7.1 基础检查

```bash
pip install -e .[dev]
pytest -q
python scripts/make_toy_cache.py --output /tmp/potato_e1_toy
python scripts/validate_cache.py \
  --manifest /tmp/potato_e1_toy/manifest.jsonl \
  --root /tmp/potato_e1_toy
```

### 7.2 训练

```bash
python -m potato_e1.train_e12 \
  --config configs/e1_2_rollback.yaml \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache \
  data.appearance_dim=<feature_dim> \
  data.num_workers=0 \
  seed=41 \
  output_dir=runs/e1_2_t6_seed41
```

正式 debug validation 运行 seed：`41, 47, 53`。

### 7.3 validation evaluation

```bash
python -m potato_e1.evaluate_e12 \
  --config configs/e1_2_rollback.yaml \
  --checkpoint runs/e1_2_t6_seed41/best.pt \
  --split val \
  --output runs/e1_2_t6_seed41/val_predictions.jsonl \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache \
  data.appearance_dim=<feature_dim>
```

### 7.4 validation-only threshold curve

```bash
python -m potato_e1.search_e12 \
  --config configs/e1_2_rollback.yaml \
  --checkpoint runs/e1_2_t6_seed41/best.pt \
  --split val \
  --output runs/e1_2_t6_seed41/operating_curve.val.jsonl \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache \
  data.appearance_dim=<feature_dim>
```

`search_e12` 必须拒绝 test。其 AP 字段故意为 null；必须接入项目已有 COCO evaluator 后才能锁点。

---

## 8. Oracle AP 审计（训练前必须补齐）

在现有 `suppression_oracle.py` 基础上增加两个输出：

### O1 Oracle selection + raw score

Oracle 选中 K=1 候选，但保留 detector raw score，计算 AP50/AP75/AP50:95。

### O2 Oracle selection + oracle quality

Oracle 选中 K=1 候选，使用 GT IoU/正确性质量分排序，计算 AP50/AP75/AP50:95。

解释：

- O1 通过：selector 是主要问题；
- O1 不通过、O2 通过：selector + calibration 都是核心；
- O1/O2 均不通过：现有框定位质量不足，候选回滚不是完整解法。

Oracle 仅作诊断，不得作为部署输入。

---

## 9. 消融顺序

| 版本 | 内容 |
|---|---|
| T0 | R0 fixed NMS=0.3 |
| T1 | NMS-suppressed pool，RGB/POL，set-wise K=1 |
| T2 | low-conf pool，RGB/POL，set-wise K=1 |
| T3 | NMS + low-conf unified pool |
| T4 | T3 + IoU quality + restore score |
| T5 | T4 + edge auxiliary loss |
| T6 | T5 + bounded residual reranking |

额外必须报告：

- RGB-only rollback；
- POL-only rollback；
- bidirectional rollback；
- NMS-only；
- low-conf-only；
- unified pool；
- independent BCE（对照）；
- set-wise + NONE（主方法）。

所有版本均运行 3 seed，不得只报告最佳 seed。

---

## 10. 正确指标

Destruction 必须是非负率：

```text
destruction_rate
baseline_destruction_rate
delta_destruction_vs_r0
restored_protected_gt
newly_destroyed_protected_gt
net_protected_gain
```

恢复日志至少包括：

```text
sample_id
gt_id / matched_gt
restore_source
modality
suppressor_index
context_index
candidate_index
candidate_iou_gt
candidate_raw_score
restore_probability
predicted_quality
predicted_restore_score
```

---

## 11. 验收门槛

```text
AP50:95 >= 0.4547
Recall >= 0.8596
FP/image <= 0.248
Destruction <= 0.0251
```

还必须满足：

- 与 T0 使用同一 cache、split、evaluator；
- safe-set invariant 逐图通过；
- 3 seed 方向一致；
- 完整 operating curve 保存；
- test 未运行；
- 当前只使用 `full_train_debug` 做结构筛选。

只有 debug validation 通过全部门槛后，才允许生成 14/14 group-aware detector OOF cache。OOF 完成并在 validation 重新锁定阈值后，才允许唯一一次 test。
