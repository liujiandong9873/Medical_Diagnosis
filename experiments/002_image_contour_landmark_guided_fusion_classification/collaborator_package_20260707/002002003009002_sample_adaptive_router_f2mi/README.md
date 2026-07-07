# 002002003009002: Sample-Adaptive-Router F2MI

创建日期: 2026-07-06

## 实验定位

本实验是 `002002003009` 的直接后续实验。

`002002003009` 的 F2MI-Transformer 已经证明 fractal-guided multimodal interaction 可以达到接近 `002002002` contour auxiliary baseline 的性能，但后续 router usage analysis 显示:

- router 权重在同一个 fold 内几乎不随样本变化;
- router 更像是在学习 fold-level modality prior;
- 因此 009 还不能充分支持“fractal/fractional 信息驱动样本级动态融合”的核心主张。

本实验的目标不是继续堆叠更大的 backbone，而是把模型创新集中在 router 机制上。

## 实验问题

核心问题:

```text
能否让 F2MI 的 modality router 从 fold-level prior 变成 sample-adaptive routing，
并在保持 raw X-ray only inference 的前提下，稳定达到或超过 002002002 contour auxiliary baseline?
```

## 数据与公平性原则

数据仍使用实验 002 的 matched ZYQ subset:

- 样本量: 667 rows;
- unique sample_id: 663;
- normal: 324;
- hypertrophy: 343;
- 交叉验证: 5-fold StratifiedGroupKFold;
- 推理阶段输入: 原始 X-ray 图像;
- 训练阶段监督: contour mask、128 点 y/e contour、b/Y/B landmarks、morphology vector、fractal token consistency。

公平性原则不变:

```text
人工 contour、landmark、line、A/N 等标注不能作为测试时直接输入。
它们只用于训练监督与分析。
```

## 模型设计

基础结构继承 `002002003009`:

```text
X-ray image
  -> ResNet18 encoder
  -> contour mask head
  -> contour points / landmark / morphology prediction heads
  -> fractal token encoder
  -> fractional contour operator
  -> multimodal token encoders
  -> fractal-guided interaction
  -> classifier
```

009002 的关键变化是 router。

009 原始 router:

```text
router input = fractal_pooled
```

009002 新 router:

```text
router input =
  fractal_pooled
  + image attended summary
  + contour attended summary
  + landmark attended summary
  + morphology attended summary
  + cross-modal summary std
```

这样 router 不再只看一个全局 fractal 表示，而是同时看到当前样本的多模态交互状态。

## 训练约束

总损失在 009 的基础上新增两项轻量 router regularization:

```text
total_loss =
  cls_loss
  + mask_loss
  + contour point loss
  + landmark loss
  + morphology loss
  + fractal consistency loss
  + router diversity loss
  + router entropy loss
```

其中:

- `router diversity loss`: 鼓励同一个 batch 内不同样本的 router 权重有最小差异，避免所有样本使用同一套权重;
- `router entropy loss`: 轻度避免 router 过早塌缩到单一路径;
- 默认权重很小，避免为了“看起来动态”而牺牲主要分类目标。

## 评价指标

主要分类指标:

- Acc
- Pre
- Recall
- F1

router 机制指标:

- router_image / router_contour / router_landmark / router_morphology;
- router_image_std / router_contour_std / router_landmark_std / router_morphology_std;
- router_entropy;
- 每 fold 的 predictions.csv 保留逐样本 router 权重。

## 成功判据

硬性性能目标:

```text
达到或超过 002002002 contour auxiliary baseline:
```

机制目标:

```text
router 在同一 fold 内应出现可观察的 sample-level variation。
如果 router std 仍接近 0，则说明 sample-adaptive routing 没有真正建立。
```

论文目标:

```text
如果分类性能接近 baseline，同时 router analysis 显示样本级动态融合成立，
则该实验可作为 Fractal and Fractional 期刊投稿中模型架构创新的主路线之一。
```

## 输出文件

训练过程中每个 fold 都会独立保存:

- `outputs/folds/f*/history_live.csv`
- `outputs/folds/f*/history.csv`
- `outputs/folds/f*/metrics.csv`
- `outputs/folds/f*/predictions.csv`

整体输出:

- `outputs/cv_metrics_by_fold.csv`
- `outputs/cv_predictions.csv`
- `outputs/training_history.csv`
- `outputs/cv_summary.csv`
- `outputs/training_progress.md`
- `outputs/run_summary.md`
