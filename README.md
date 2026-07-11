# POTATO — E1 Transformer Candidate Arbitrator

本仓库当前锁定 PoTATO RGB–POL 多模态检测的第一项工程：**E1：缓存候选上的轻量 Transformer 仲裁器**。

E1 不重新训练 RGB/POL 检测器，也不做输入级 early fusion。它读取两个冻结单模态专家导出的候选缓存，学习四件事：

1. 判断 RGB 与 POL 候选是否属于同一真实目标；
2. 预测每个候选的真实性与定位质量；
3. 识别值得接纳的 POL-only 补充候选；
4. 显式保护 RGB 已正确候选，降低融合破坏率。

当前研究结论已经排除“整图/整目标 RGB–POL 硬切换”作为主线。物理量只作为候选可靠性的辅助特征，不直接决定最终模态。

---

## 1. 已锁定的诊断基线

使用 seed 47 的 RGB/POL YOLOv5m，在官方 2,000 张测试图像、5,384 个 GT 上：

| 系统 | Recall | FP/image | AP50:95 | Destruction |
|---|---:|---:|---:|---:|
| RGB @ 0.25 | 0.8224 | 0.248 | 0.4410 | 0.0672 |
| POL @ 0.25 | 0.7762 | 0.240 | 0.4287 | 0.1197 |
| 固定 cross-modal NMS=0.3, conf=0.25 | 0.8601 | 0.252 | 0.4547 | 0.0244 |
| 严格 FP 匹配诊断上界 | 0.8596 | 0.248 | 0.4547 | 0.0251 |
| GT Oracle | 0.8817 | — | — | 0.0000 |

RGB-only 目标为 568 个，POL-only 目标为 319 个，两者共同漏检 637 个。

E1 的最低竞争对象不是 RGB，而是 **固定 cross-modal NMS=0.3**。所有阈值必须在验证集确定，测试集只运行一次锁定配置。

---

## 2. E1 方法定义

### 2.1 输入

每张图像包含两组经单模态后处理后的低阈值候选：

- RGB 候选：框、类别、原始分数、可选检测特征；
- POL 候选：框、类别、原始分数、可选检测特征、候选区域物理统计；
- 训练时额外提供 GT 框和类别。

建议缓存单模态 NMS 后、较低置信度阈值的候选，初版：

- `candidate_conf = 0.01`
- `single_modal_nms = 0.60`
- `topk_per_modality = 64`

不得使用测试 GT 生成推理特征。物理统计必须在预测候选框及其上下文区域上提取，而不是在 GT 框上提取。

### 2.2 Transformer 输出

对拼接后的 RGB/POL 候选 token，轻量 Transformer 预测：

- `valid_logit`：候选是否为真实目标；
- `iou_pred`：候选与匹配 GT 的定位质量；
- `rescue_logit`：POL 候选是否能补充 RGB 未覆盖的 GT；
- `protect_logit`：RGB 候选是否必须保护；
- `pair_same_logit[i,j]`：RGB/POL 候选是否对应同一 GT。

### 2.3 推理规则

推理顺序必须保持为：

1. **Protect RGB**：先保留高可信 RGB 候选；
2. **Match pairs**：利用 `pair_same_logit` 匹配跨模态重复候选；
3. **Merge/select**：重复候选按预测定位质量融合或选框，不能仅比较未校准原始分数；
4. **Admit POL**：未匹配 POL 只有在 `valid` 与 `rescue` 同时满足阈值时才准入；
5. **Final cleanup**：执行轻量最终去重，避免重新退化成普通 NMS 堆叠。

默认策略是 RGB 锚定。POL 候选不能仅因分数更高而替换被保护的 RGB 候选。

---

## 3. 仓库结构

```text
POTATO/
├─ configs/e1_transformer.yaml
├─ docs/E1_IMPLEMENTATION_SPEC.md
├─ scripts/validate_cache.py
├─ src/potato_e1/
│  ├─ schema.py
│  ├─ dataset.py
│  ├─ targets.py
│  ├─ model.py
│  ├─ losses.py
│  ├─ arbitration.py
│  ├─ train.py
│  └─ evaluate.py
└─ tests/test_model_shapes.py
```

---

## 4. 候选缓存格式

数据根目录包含一个 `manifest.jsonl` 与若干 `.npz`：

```json
{"sample_id":"exp03_frame_000123","cache":"samples/exp03_frame_000123.npz","split":"train","group":"exp03","oof_fold":0}
```

每个 `.npz` 至少包含：

```text
rgb_boxes       float32 [Nr, 4]  normalized xyxy
rgb_scores      float32 [Nr]
rgb_classes     int64   [Nr]
pol_boxes       float32 [Np, 4]
pol_scores      float32 [Np]
pol_classes     int64   [Np]
gt_boxes        float32 [G, 4]
gt_classes      int64   [G]
```

可选字段：

```text
rgb_features    float32 [Nr, Fa]
pol_features    float32 [Np, Fa]
pol_physics     float32 [Np, Fp]
```

约束：

- 框必须是 `[0,1]` 归一化 `xyxy`；
- 所有数组必须有限值；
- 候选按分数降序或由数据加载器统一排序；
- `train/val/test` 的候选必须来自同一固定检测器配置；
- 论文正式结果的仲裁器训练缓存应优先使用 detector OOF 预测，避免元学习器读取检测器对自身训练图像的过拟合输出；
- `test` 绝不能参与阈值、NMS 或损失权重选择。

---

## 5. 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

验证缓存：

```bash
python scripts/validate_cache.py \
  --manifest /path/to/cache/manifest.jsonl \
  --root /path/to/cache
```

训练：

```bash
python -m potato_e1.train \
  --config configs/e1_transformer.yaml \
  data.manifest=/path/to/cache/manifest.jsonl \
  data.root=/path/to/cache
```

评估：

```bash
python -m potato_e1.evaluate \
  --config configs/e1_transformer.yaml \
  --checkpoint runs/e1/best.pt \
  --split val
```

当前代码提供模型、标签、损失、缓存数据集、仲裁推理和基础评估骨架。检测器候选导出与 COCO AP 接口需要根据现有 YOLOv5 工程路径接入，具体要求见 `docs/E1_IMPLEMENTATION_SPEC.md`。

---

## 6. Codex 必须按顺序完成的工程任务

### P0：让端到端最小闭环跑通

- [ ] 接入现有 RGB/POL YOLOv5m 权重，导出 train/val/test 候选缓存；
- [ ] 将检测头对应候选位置的 neck/head 特征写入 `rgb_features/pol_features`；
- [ ] 在预测 POL 框和 1.5× 上下文区域提取物理统计，写入 `pol_physics`；
- [ ] 运行 `validate_cache.py`，保证所有样本无 NaN、shape 闭合、框合法；
- [ ] 运行单 batch overfit，确认总损失和五个子损失均能下降；
- [ ] 在 validation 上完成阈值搜索并冻结 `thresholds.locked.yaml`；
- [ ] 在 test 上只运行锁定阈值一次。

### P1：完成公平基线

必须由同一缓存复现：

- [ ] RGB-only；
- [ ] POL-only；
- [ ] union + fixed cross-modal NMS，至少扫描 0.3/0.4/0.5/0.6，但只在 val 选取；
- [ ] score calibration + NMS；
- [ ] Transformer 无物理特征；
- [ ] Transformer + 物理特征。

### P2：完成消融

按顺序启用：

- [ ] A2：仅 `valid + iou`；
- [ ] A3：A2 + `pair_same`；
- [ ] A4：A3 + `rescue`；
- [ ] A5：A4 + `protect`；
- [ ] A6：A5 + `pol_physics`。

物理特征若不能稳定优于 A5，不得为了“物理创新”强行保留。

---

## 7. 验收标准

E1 第一版必须满足：

1. 在 validation 锁定阈值后，test 上不重新调参；
2. 在 `FP/image <= RGB FP/image + 0.01` 条件下，与固定 cross-modal NMS=0.3 比较；
3. 至少满足以下一项，并且其余指标不显著恶化：
   - 同 FP 下 Recall 与 AP50:95 同时提高；
   - 同 Recall 下 FP/image 降低；
   - 同 AP 下 Destruction 明显降低；
4. 报告 Recall、Precision、FP/image、AP50、AP50:95、Destruction、Rescue、净 TP；
5. 按完整采集实验做 cluster bootstrap 95% CI；
6. 至少 3 个 arbitrator seed；
7. 保存所有失败消融和完整阈值曲线。

当前必须击败的固定基线：`Recall≈0.8601, FP/image≈0.252, AP50:95≈0.4547, Destruction≈0.0244`。

---

## 8. 明确禁止

- 禁止使用 test 选择置信度、pair 阈值、NMS 阈值或损失权重；
- 禁止把 GT 框物理量作为部署输入；
- 禁止把 RGB/POL 未校准原始置信度直接当概率比较；
- 禁止将共同漏检目标的 Rescue=0 误解为仲裁器失败：E1 只重组已有候选，不能创造新候选；
- 禁止在 E1 同时加入 P2、SAHI、CARAFE、DySample、小波、loss 替换等无关变量；
- 禁止只报告最佳 seed 或最佳阈值。

---

## 9. 下一阶段边界

只有 E1 在固定 FP 预算下稳定优于 fixed NMS，才进入 E2：将候选仲裁器迁移到 YOLO26s 的 one-to-one 推理路径。共同漏检的 637 个目标属于后续“特征交互生成新候选”问题，不在 E1 范围内。
