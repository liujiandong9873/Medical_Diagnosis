# 002002003009003: Recall-Preserving Sample-Adaptive F2MI

创建日期: 2026-07-06

## 实验定位

本实验是 `002002003009002_sample_adaptive_router_f2mi` 的后续实验。

`009002` 的分析结果显示:

- sample-adaptive router 已经被激活，fold 内 router 权重不再接近固定;
- 但固定 0.5 阈值下 recall 和 F1 低于 `002002002` 和原始 `009`;
- validation-based F1 threshold tuning 可以把 009002 的 test F1 从约 0.842 提高到约 0.873;
- 因此主要问题不是 sample-adaptive router 不工作，而是训练目标/模型选择使输出概率偏保守。

## 核心问题

```text
能否保留 009002 的 sample-adaptive router 机制，
同时通过更合理的 checkpoint 选择和更温和的 router 约束恢复 recall/F1?
```

## 与 009002 的区别

继承 009002:

- raw X-ray only inference;
- ResNet18 image encoder;
- contour / landmark / morphology prediction heads;
- fractal token encoder;
- fractional contour operator;
- sample-adaptive router input:
  - fractal pooled token;
  - image attended summary;
  - contour attended summary;
  - landmark attended summary;
  - morphology attended summary;
  - cross-modal summary std。

009003 修改:

- 默认移除 router diversity loss;
- 默认移除 router entropy loss;
- router std / entropy 只作为监测指标，不强行优化;
- 目标是提高固定 0.5 阈值下的 recall/F1，而不是依赖后处理阈值补救。

## 成功判据

主要目标:

```text
F1 和 recall 明显高于 009002 fixed-threshold 结果。
```

参考目标:


机制目标:

```text
sample-adaptive router 仍应保留可观察的 fold 内样本级变化。
```

## 输出

每个 fold:

- `outputs/folds/f*/history_live.csv`
- `outputs/folds/f*/history.csv`
- `outputs/folds/f*/metrics.csv`
- `outputs/folds/f*/predictions.csv`

整体:

- `outputs/cv_metrics_by_fold.csv`
- `outputs/cv_predictions.csv`
- `outputs/training_history.csv`
- `outputs/cv_summary.csv`
- `outputs/training_progress.md`
- `outputs/run_summary.md`

## 当前状态

更新日期: 2026-07-06

在完成 `002002003009004_calibrated_router_calibrated_router_f2mi` 后，当前路线决策为:

```text
002002003009003 回到主线，作为 recall-preserving sample-adaptive F2MI candidate。
```

原因:

- `009003` 修复了 `009002` 的 fixed-threshold low-recall 问题;
- `009003` 的 F1 与 `002002002` 和原始 `009` 基本持平;
- `009003` 的 recall 高于主要对照模型;

当前推荐定位:

```text
009003 is a recall-preserving candidate model, not a statistically superior final model.
```

详细决策见:

```text
CANDIDATE_MODEL_DECISION.md
```
