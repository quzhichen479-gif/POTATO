# E1.1 工程规范：Fixed-NMS-Anchored Residual Transformer

## 1. 状态与动机

E1 A5/A6 已在 `full_train_debug` validation 缓存上完成 3 个 arbitrator seed。结果表明：

| 方法 | AP50:95 | Recall | FP/image | Destruction |
|---|---:|---:|---:|---:|
| 严格诊断门槛 | 0.4547 | 0.8596 | 0.248 | 2.51% |
| 同缓存 fixed NMS=0.3 | 0.4399 | 0.8701 | 0.195 | 3.58% |
| A5，无物理 | 0.4653 | 0.8482 | 0.2256 | 6.06% |
| A6，含物理 | 0.4529 | 0.8485 | 0.2311 | 6.08% |

E1 未通过门槛。A5 学到了排序信息，但获得删除、替换和合并权限后，以 Recall 和 Destruction 为代价提高 AP。A6 明显劣于 A5，因此物理特征已经被否决，不进入 E1.1。

E1.1 将“非退化”从 soft loss 改为 hard inference invariant：

> fixed cross-modal NMS 输出是不可删除、不可移动、不可改类别的安全集合；Transformer 只能做有界分数残差和少量 POL 增量准入。

## 2. 方法定义

### 2.1 安全集合

对缓存候选先执行固定基线：

```text
eligible = union(RGB, POL) with raw_score >= base_conf
safe_set = classwise_nms(eligible, IoU=base_nms_iou)
```

默认：

```yaml
base_conf: 0.25
base_nms_iou: 0.30
```

最终集合必须满足：

```text
safe_set ⊆ final_set
```

且安全集合的 box 和 class 必须逐项不变。允许改变的只有用于 AP 排序的 score。

### 2.2 有界残差重排序

Transformer 不直接产生替代分数，而预测 `score_delta`：

```text
final_score = sigmoid(logit(raw_score) + alpha * tanh(score_delta))
```

因此 logit 修正绝对值不会超过 `alpha`。首轮只在 validation 扫描：

```text
alpha ∈ {0.25, 0.50, 1.00}
```

### 2.3 定位质量

`iou_pred` 以候选和最佳同类 GT 的 IoU 为连续监督。它用于训练排序表征，也用于 POL extra 准入。E1.1 不移动或融合安全集合框。

### 2.4 POL 增量准入

只考虑未进入安全集合的 POL 候选。候选必须同时满足：

```text
rescue_probability >= rescue_threshold
predicted_iou >= extra_quality_threshold
adjusted_score >= extra_score_floor
same-class IoU with all selected boxes <= extra_overlap_iou
```

并满足每图增量上限：

```text
max_extra_per_image ∈ {0, 1}
```

`max_extra_per_image=0` 对应纯重排序版本 R1–R3；等于 1 才进入 R4。

### 2.5 E1.1 rescue 标签

旧 E1 的 rescue 标签定义为“POL 正确且 RGB 未覆盖”。E1.1 改为部署系统相对标签：

```text
candidate is POL
candidate is not in fixed-NMS safe set
candidate correctly matches a GT
that GT is not covered by the fixed-NMS safe set
```

这直接学习“是否能补充当前 fixed-NMS 系统”。

## 3. 模型输出与损失

模型文件：`src/potato_e1/e11_model.py`

输出仅保留：

```text
score_delta
unconstrained scalar, used through tanh

iou_pred
[0, 1] localization-quality estimate

rescue_logit
POL incremental admission estimate
```

E1.1 不存在 `protect_head`、删除头或替换头。

损失文件：`src/potato_e1/e11_losses.py`

```text
L = λq Lquality
  + λs Lscore
  + λr Lranking
  + λa Ladmission
  + λd Ldelta_reg
```

- `Lquality`：所有有效 token 的 IoU smooth-L1；
- `Lscore`：安全集合中 adjusted score 对 IoU target 的 smooth-L1；
- `Lranking`：同图安全集合中正候选应高于负候选；
- `Ladmission`：未进入安全集合的 POL 候选 rescue focal BCE；
- `Ldelta_reg`：限制不必要的大幅 score correction。

## 4. 代码入口

配置：

```text
configs/e1_1_residual.yaml
```

训练：

```bash
python -m potato_e1.train_e11 \
  --config configs/e1_1_residual.yaml \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache \
  data.appearance_dim=<cached_feature_dim> \
  output_dir=runs/e1_1_seed47
```

validation：

```bash
python -m potato_e1.evaluate_e11 \
  --config configs/e1_1_residual.yaml \
  --checkpoint runs/e1_1_seed47/best.pt \
  --split val \
  --output runs/e1_1_seed47/val_predictions.jsonl
```

test 只允许读取 validation 锁定文件：

```yaml
locked: true
arbitration:
  residual_alpha: 0.5
  rescue_threshold: 0.9
  extra_quality_threshold: 0.5
  extra_score_floor: 0.15
  max_extra_per_image: 1
```

```bash
python -m potato_e1.evaluate_e11 \
  --config configs/e1_1_residual.yaml \
  --checkpoint runs/e1_1_seed47/best.pt \
  --split test \
  --thresholds runs/e1_1_locked/thresholds.locked.yaml
```

缺少 `locked: true` 时，test 入口必须拒绝运行。

## 5. 消融顺序

不得一次性搜索所有组件。按以下顺序：

| 版本 | 组件 | 目标 |
|---|---|---|
| R0 | 同缓存 fixed NMS=0.3 | 安全基线 |
| R1 | R0 + bounded score residual | 验证排序价值 |
| R2 | R1 + IoU quality loss | 提升 AP50:95/AP75 |
| R3 | R2 + pairwise ranking | 强化正负排序 |
| R4 | R3 + max 1 POL extra | 使用剩余 FP 预算补召回 |

当前实现覆盖 R0–R4。R5 学习式重复匹配不在本轮范围内。

## 6. Validation 锁定规则

必须保存完整 operating curve，不得只保存最佳点。所有阈值只在 validation 选择。

严格门槛：

```text
AP50:95 >= 0.4547
Recall >= 0.8596
FP/image <= 0.248
Destruction <= 0.0251
```

阶段性规则：

- R1–R3：不得降低 fixed-NMS Recall 超过 0.1 个百分点；Destruction 不得高于 R0；AP 必须提高；
- R4：FP/image 不得超过 0.248，Recall 不下降，Destruction 应下降；
- 3 个 arbitrator seed 报告均值、标准差和逐 seed 结果；
- test 在正式 group-aware OOF detector cache 完成前保持封存。

## 7. 数据与 OOF 边界

当前 `full_train_debug` 结果只能用于工程筛选。正式实验必须完成 14/14 group-aware detector OOF predictions。不要为失败版本消耗 OOF 生成成本。

只有 debug validation 上 R1–R4 某版本满足全部门槛后，才执行：

1. 生成完整 detector OOF train/validation cache；
2. 重新训练 3 个 arbitrator seed；
3. 在 OOF validation 锁定阈值；
4. 冻结代码、配置、checkpoint hash 和 cache fingerprint；
5. 最后运行唯一一次 test。

## 8. 禁止事项

- 禁止恢复物理特征；
- 禁止让 Transformer 删除安全集合候选；
- 禁止移动、融合或替换安全集合框；
- 禁止用 test 搜索 alpha、rescue、quality、score floor 或 max-extra；
- 禁止把训练 loss 最低直接当作部署 operating point；
- 禁止只报告最佳 seed；
- 禁止同时引入 P2、SAHI、CARAFE、DySample、小波或新 detector loss。

## 9. Codex 验收输出

完成后必须提交：

```text
runs_manifest.json
cache fingerprint
checkpoint SHA256
R0–R4 per-seed table
validation full threshold curves
thresholds.locked.yaml（仅在通过后生成）
per-image predictions
COCO AP50/AP75/AP50:95
Recall / Precision / FP-image / Destruction / Rescue / net TP
cluster-bootstrap 95% CI
failure cases and ablation conclusions
```

大型 cache、dataset、checkpoint 和 `runs/` 不得提交到 Git。
