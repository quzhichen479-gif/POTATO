# E1 Transformer 候选仲裁器：工程实现规范

本文档是 Codex 的执行合同。除非发现明确 bug，不要改变研究问题、标签定义或评测口径。

## 1. 研究目标

冻结 RGB 与 POL 单模态检测器，在候选层学习一个轻量 Transformer，使最终系统在近似相同 FP 预算下优于固定 cross-modal NMS。

E1 只解决“已有候选的安全组合”，不负责创造 RGB/POL 均未产生的新候选。

## 2. 不得改变的对照

诊断基准：

- RGB：Recall 0.8224，FP/image 0.248，AP50:95 0.4410；
- fixed cross-modal NMS=0.3：Recall 0.8601，FP/image 0.252，AP50:95 0.4547，Destruction 0.0244；
- Oracle Recall：0.8817。

所有新结果必须与同一候选缓存上的 fixed NMS 比较。不得引用旧脚本中不同置信度、不同单模态 NMS 或不同预测导出的数值作为公平对照。

## 3. 候选导出器

### 3.1 输入模型

第一版使用已验证的 seed 47 RGB/POL YOLOv5m 权重。两模型：

- 输入图像一一对应；
- 使用相同 resize/letterbox；
- 输出坐标映射回同一原图坐标后归一化；
- 使用相同 `candidate_conf`、`single_modal_nms` 和 `topk`。

### 3.2 推荐候选点

初版先导出单模态 class-wise NMS 后的候选：

```text
candidate_conf = 0.01
single_modal_nms = 0.60
topk_per_modality = 64
```

必须同时保留原始 detector confidence。若后续导出 pre-NMS dense candidates，应作为独立实验，不得覆盖第一版缓存。

### 3.3 特征对齐

`rgb_features/pol_features` 必须与最终缓存候选逐行对齐。推荐：

1. 在检测头 decode 前保存各尺度位置特征；
2. 记录每个 decode candidate 的 scale index 与 spatial index；
3. NMS 后按保留索引取得对应特征；
4. 如不同尺度特征通道数不同，用固定的、冻结或可训练的线性投影统一维度；
5. 导出前做断言：候选数等于特征行数。

不得用 GT ROIAlign 生成候选特征。

### 3.4 OOF 规则

正式仲裁器训练优先使用 detector out-of-fold predictions：

- 按完整 recording session/group 分折；
- 每一折检测器不得在该折图像上训练；
- 将各折 held-out predictions 合并为仲裁器 train cache；
- val/test 使用最终冻结检测器输出；
- manifest 的 `oof_fold` 记录来源折。

可先用 full-train detector 在 train 上导出候选完成工程 smoke test，但这些结果只能标为 `leaky-debug`，不得进入论文主表。

## 4. 物理特征

物理特征从预测 POL 候选框和扩大 1.5 倍的上下文框提取。第一版优先实现：

### RGB radiometry

- luma mean/std/q05/q95；
- q95-q05 dynamic range；
- any-channel clipped ratio；
- HSV saturation mean/std；
- gradient mean/std；
- 候选框与上下文环的亮度/梯度差。

### Stokes/intensity reliability

- S0 mean/std/q05/q95；
- I0/I45/I90/I135 mean/std；
- feasibility excess：`max(0, sqrt(S1^2+S2^2)-S0)` 的 mean/max/relative max；
- 从 Stokes 重建四方向强度后的 reprojection residual；
- DoLP clipped quantiles；
- q/u 或归一化 S1/S2 的 mean/std。

要求：

- 特征顺序写入 `physics_feature_names.json`；
- 只在 train 统计均值/标准差，保存 scaler；
- val/test 使用同一 scaler；
- 缺失或无效值必须用显式 mask 或有限值替代，禁止 NaN；
- A5（无 physics）必须先完成，再运行 A6。

## 5. 标签定义

设 candidate 与同类别 GT 的最大 IoU 为 `q`，正候选阈值为 0.5。

### valid

```text
valid = 1[q >= 0.5]
```

### iou quality

```text
iou_target = q
```

### pair_same

RGB 与 POL 候选均为正，并匹配同一个 GT 时为 1。训练 pair 集合包括：

- 所有正 pair；
- 跨模态 IoU >= 0.05 的同类别 hard pairs。

不要用“跨模态框 IoU 高”直接作为正标签。

### rescue

仅对 POL token 计算：

```text
rescue = POL candidate is positive
         AND its matched GT has no positive RGB candidate
```

### protect

仅对 RGB token 计算：

```text
protect = RGB candidate is positive
          AND its matched GT has no positive POL candidate
```

第一版 protect 明确对应 RGB-only 目标。后续可增加“所有正确 RGB 均保护”的辅助标签，但必须作为消融。

## 6. 模型约束

默认模型：

```text
d_model=128
nhead=4
layers=2
ffn=256
dropout=0.1
topk=64+64
```

输入 token 包含：

- normalized xyxy；
- center/width/height/area；
- raw score 与 score logit；
- modality embedding；
- 可选 appearance projection；
- 可选 physics projection（RGB 位置为零向量）。

pair_same 使用内容相似度加几何偏置，不允许为每一对构造超大 4D Transformer token，避免 E1 失去轻量属性。

必须记录：参数量、候选缓存读取时间、模型 forward 延迟、完整仲裁延迟、峰值显存。

## 7. 训练顺序

### Stage S0：shape smoke test

- 随机 Nr/Np，包括某一模态为 0；
- forward 输出 shape 正确；
- padding token 不产生有效预测；
- loss finite；
- backward finite。

### Stage S1：单 batch overfit

固定 8–16 张图，训练至少 500 step：

- total loss 显著下降；
- valid/pair/rescue/protect 均不应恒定；
- 输出每类正负数量，避免某头没有正例。

### Stage S2：无特征基线

`appearance_dim=0, physics_dim=0`，验证仅几何、分数和上下文 Transformer 是否优于 fixed NMS。

### Stage S3：appearance

加入候选检测特征。

### Stage S4：双风险

依次加入 pair、rescue、protect，严格按 A2–A5 保存结果。

### Stage S5：physics

仅在 A5 稳定后加入物理特征形成 A6。

## 8. 阈值选择与锁定

在 validation 上搜索：

- rgb_valid_threshold；
- rgb_protect_threshold；
- pol_valid_threshold；
- pol_rescue_threshold；
- pair_same_threshold；
- replace_margin；
- final_nms_iou。

不要做无约束高维穷举。推荐：

1. 先固定 RGB 保护策略，搜索 pair threshold；
2. 再搜索 POL valid/rescue；
3. 再搜索 replace/merge；
4. 最后只微调 final NMS。

目标是 constrained selection：

```text
maximize AP50:95 or Recall
subject to FP/image <= RGB_FP + 0.01
and Destruction <= fixed_NMS_Destruction
```

将最终配置写入：

```text
runs/<experiment>/thresholds.locked.yaml
```

文件中必须包含：

- 数据缓存 fingerprint；
- detector weight SHA256；
- arbitrator checkpoint SHA256；
- val 指标；
- 所有阈值；
- 锁定时间。

测试脚本检测到 `split=test` 时必须要求 `--thresholds` 指向 locked 文件，防止隐式使用测试调参。

## 9. 公平基线实现

同一缓存上输出：

1. RGB-only；
2. POL-only；
3. union + fixed NMS；
4. temperature/isotonic calibration + fixed NMS；
5. A2 valid+iou；
6. A3 + pair；
7. A4 + rescue；
8. A5 + protect；
9. A6 + physics。

每个实验保存逐图预测，不只保存汇总指标。

## 10. 评测

必报：

- AP50:95、AP50、AP75；
- Recall、Precision、FP/image；
- Destruction；
- candidate-level POL admission count；
- 新增 TP、损失 RGB TP、净 TP；
- provenance：RGB/POL/merged 各自 TP 与 FP；
- 参数、延迟、显存。

Rescue 指标需区分：

1. `candidate rescue`：最终接纳了 RGB 未覆盖但 POL 已有的目标；
2. `common-miss rescue`：RGB/POL 原候选均没有但最终检出。

E1 理论上主要改善前者，后者接近 0 是预期行为。

不确定性：按完整 `group/exp` cluster bootstrap 10,000 次，至少 3 个 arbitrator seeds。不得把帧或 GT 当独立重采样单位。

## 11. 单元测试

至少包含：

- cache schema 成功与失败样例；
- target 四象限标签；
- pair_same 正负标签；
- model shape 与 backward；
- RGB-only 保护；
- POL-only 准入；
- duplicate merge；
- protected RGB 不被高分 POL 替换；
- deterministic inference。

## 12. 完成定义

Codex 完成后 README 的 P0/P1/P2 checklist 应按真实状态更新，并提交：

```text
runs/e1_transformer/
  resolved_config.yaml
  cache_fingerprint.json
  history.jsonl
  best.pt
  thresholds.locked.yaml
  val_metrics.json
  test_metrics.json
  predictions_val.jsonl
  predictions_test.jsonl
  ablation.csv
  latency.json
```

大型 checkpoint 与原始缓存不要提交 Git，使用 `.gitignore` 或外部制品存储；仓库中提交可复现实验命令、配置和摘要表。
