# POTATO — E1.1 Fixed-NMS-Anchored Residual Transformer

本仓库当前工作的唯一主线是：

> **E1.1：以 fixed cross-modal NMS 为不可删除安全集合的轻量残差 Transformer。**

旧 E1 的全权候选仲裁器已经完成 validation 诊断，但没有通过验收门槛。E1.1 不再允许 Transformer 删除、替换或移动 fixed-NMS 已保留候选，只允许：

1. 对安全集合做**有界分数残差重排序**；
2. 从安全集合之外保守接纳每图至多 1 个 POL 补充候选。

物理特征已经在 A6 中被实验证伪，不进入 E1.1。

---

## 1. E1 失败结论

`full_train_debug` validation 缓存上 3 个 arbitrator seed 的均值：

| 方法 | AP50:95 | Recall | FP/image | Destruction |
|---|---:|---:|---:|---:|
| 严格诊断门槛 | 0.4547 | 0.8596 | 0.248 | 2.51% |
| 同缓存 fixed NMS=0.3 | 0.4399 | 0.8701 | 0.195 | 3.58% |
| A5，无物理 | **0.4653** | 0.8482 | 0.2256 | 6.06% |
| A6，含物理 | 0.4529 | 0.8485 | 0.2311 | 6.08% |

结论：

- A5 提高 AP，但以 Recall 下降和 Destruction 上升为代价；
- A2–A6 没有任何搜索点同时满足 FP 与 Destruction 约束；
- A6 相比 A5，AP 下降、FP 上升、Recall 几乎不变、Destruction 不改善；
- 物理特征已否决；
- 当前 test 尚未运行，必须继续封存；
- 正式 group-aware detector OOF 仍为 0/14，当前只允许 debug validation 筛选结构。

旧 E1 代码保留用于失败消融复现，不再作为默认入口。

---

## 2. E1.1 核心结构

### 2.1 不可删除安全集合

```text
eligible = RGB/POL union with raw score >= 0.25
safe_set = class-wise NMS(eligible, IoU=0.30)
```

E1.1 强制：

```text
safe_set ⊆ final_set
```

安全集合的 box 和 class 不得改变。

### 2.2 有界分数残差

Transformer 输出 `score_delta`，最终分数为：

```text
adjusted_score = sigmoid(logit(raw_score) + alpha * tanh(score_delta))
```

因此 logit 修正绝对值永远不超过 `alpha`。默认 `alpha=0.5`，只允许在 validation 扫描 `{0.25, 0.5, 1.0}`。

### 2.3 POL 增量准入

未进入安全集合的 POL 候选只有同时满足以下条件才允许加入：

```text
rescue_probability >= rescue_threshold
predicted_iou >= extra_quality_threshold
adjusted_score >= extra_score_floor
same-class overlap with selected boxes <= extra_overlap_iou
```

并限制：

```text
max_extra_per_image <= 1
```

Transformer 没有 delete head、replace head 或 protect head。非退化由结构保证，而不是依赖 soft loss 学习。

---

## 3. 代码结构

```text
POTATO/
├─ AGENTS.md
├─ configs/
│  ├─ e1_transformer.yaml          # 旧 E1，仅供失败消融复现
│  └─ e1_1_residual.yaml           # 当前默认配置
├─ docs/
│  ├─ E1_IMPLEMENTATION_SPEC.md    # 旧 E1 规范
│  └─ E1_1_IMPLEMENTATION_SPEC.md  # 当前工程合同
├─ src/potato_e1/
│  ├─ e11_dataset.py
│  ├─ e11_model.py
│  ├─ e11_losses.py
│  ├─ e11_arbitration.py
│  ├─ train_e11.py
│  └─ evaluate_e11.py
└─ tests/test_e11_residual.py
```

---

## 4. 安装与基础检查

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
pytest -q
```

生成 toy cache：

```bash
python scripts/make_toy_cache.py --output /tmp/potato_e1_toy
python scripts/validate_cache.py \
  --manifest /tmp/potato_e1_toy/manifest.jsonl \
  --root /tmp/potato_e1_toy
```

---

## 5. E1.1 训练

```bash
python -m potato_e1.train_e11 \
  --config configs/e1_1_residual.yaml \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache \
  data.appearance_dim=<candidate_feature_dim> \
  data.num_workers=0 \
  output_dir=runs/e1_1_seed47
```

首轮必须先在 toy cache 做：

- 单 batch overfit；
- `quality/score/ranking/rescue/delta_reg` 五项损失均可计算；
- 总损失下降；
- safe-set invariant 测试通过。

---

## 6. Validation 评估

```bash
python -m potato_e1.evaluate_e11 \
  --config configs/e1_1_residual.yaml \
  --checkpoint runs/e1_1_seed47/best.pt \
  --split val \
  --output runs/e1_1_seed47/val_predictions.jsonl
```

评估会同时输出：

- 同缓存 fixed-NMS baseline；
- E1.1；
- Recall、Precision、FP/image、Destruction、Rescue；
- 每图 safe count 与 extra count；
- safe-set box/class 不变断言；
- 可供项目 COCO evaluator 计算 AP50/AP75/AP50:95 的逐图预测。

当前基础入口不替代项目已有 COCO AP 评估器。Codex 必须接入同一套官方评估脚本，不能自行使用不同口径。

---

## 7. Test 锁定保护

Test 不允许直接运行。必须提供 validation 生成的锁定文件：

```yaml
locked: true
arbitration:
  base_conf: 0.25
  base_nms_iou: 0.30
  residual_alpha: 0.50
  rescue_threshold: 0.90
  extra_quality_threshold: 0.50
  extra_score_floor: 0.15
  extra_overlap_iou: 0.30
  max_extra_per_image: 1
```

运行：

```bash
python -m potato_e1.evaluate_e11 \
  --config configs/e1_1_residual.yaml \
  --checkpoint runs/e1_1_seed47/best.pt \
  --split test \
  --thresholds runs/e1_1_locked/thresholds.locked.yaml
```

没有顶层 `locked: true` 时，程序必须拒绝 test。

当前阶段**禁止运行 test**，因为正式 group-aware OOF detector cache 尚未完成。

---

## 8. 必须按顺序完成的消融

| 版本 | 组件 | 目的 |
|---|---|---|
| R0 | 同缓存 fixed NMS=0.3 | 安全基线 |
| R1 | R0 + bounded score residual | 验证 A5 的排序价值 |
| R2 | R1 + IoU quality loss | 提升 AP50:95/AP75 |
| R3 | R2 + pairwise ranking loss | 强化候选排序 |
| R4 | R3 + 每图最多 1 个 POL extra | 在剩余 FP 预算内补召回 |

本轮不实现 R5 学习式重复匹配。

建议开关：

- R1：`quality_weight=0, ranking_weight=0, rescue_weight=0, max_extra_per_image=0`
- R2：开启 `quality_weight`；
- R3：开启 `ranking_weight`；
- R4：开启 `rescue_weight` 且 `max_extra_per_image=1`。

每个版本至少 3 个 arbitrator seed，保存逐 seed 与均值，不得只报告最佳 seed。

---

## 9. 验收门槛

严格门槛保持不变：

```text
AP50:95 >= 0.4547
Recall >= 0.8596
FP/image <= 0.248
Destruction <= 0.0251
```

阶段门槛：

- R1–R3：Recall 相比 R0 下降不超过 0.1 个百分点；Destruction 不高于 R0；AP 必须提高；
- R4：FP/image 仍不超过 0.248，Recall 不下降，Destruction 应下降；
- A/B 对比必须来自同一 cache、同一 evaluator、同一 split；
- 所有搜索点与失败结果完整保存。

只有 debug validation 上有版本通过全部门槛，才值得生成 14/14 group-aware detector OOF cache。

---

## 10. Codex 当前任务

Codex 读取顺序：

1. `README.md`
2. `AGENTS.md`
3. `docs/E1_1_IMPLEMENTATION_SPEC.md`
4. GitHub E1.1 issue

执行要求：

```text
先运行 pytest 和 toy-cache smoke test；
修复所有 E1.1 入口问题；
在 full_train_debug validation cache 上完成 R0→R4；
复现同缓存 fixed NMS=0.3；
用项目原 COCO evaluator 计算 AP50/AP75/AP50:95；
保存完整 operating curves；
三个 arbitrator seed 全部报告；
不得运行 test；
不得生成正式 OOF，除非 debug validation 已通过全部门槛。
```

---

## 11. 明确禁止

- 禁止恢复 A6 物理特征；
- 禁止让 Transformer 删除、替换、移动或融合 safe-set 候选；
- 禁止使用 test 选择任何阈值、损失权重、checkpoint 或特征；
- 禁止只报告最佳 seed 或最佳 operating point；
- 禁止同时引入 P2、SAHI、CARAFE、DySample、小波、检测器 loss 修改或 YOLO26 迁移；
- 禁止提交 dataset、cache、checkpoint、weights 或 `runs/`。

完整细节见 `docs/E1_1_IMPLEMENTATION_SPEC.md`。
