# 002002003010: Baseline-Preserving Fractal-Fractional Residual F2MI

创建日期: 2026-07-06

## 实验定位

本实验是 `002002003009003_recall_preserving_adaptive_f2mi` 之后的新路线。

前序结果显示:

- `002002002 contour auxiliary ResNet18` 是当前最强、最稳的内部 baseline;
- 因此，如果要让 proposed model 更稳妥地支撑一区 SCI 投稿，需要让 proposed model 在不牺牲 baseline 决策面的前提下获得增益。

本实验的核心思想是:

```text
不要让复杂 F2MI 分支重新学习整个分类决策。
先保留 002002002 contour auxiliary baseline 的强分类 logit，
再让 fractal/fractional F2MI 只学习一个有界 residual correction。
```

## 核心问题

```text
能否在保留 002002002 强 baseline 性能的基础上，
通过 fractal/fractional sample-adaptive residual correction
```

## 模型结构

模型名称:

```text
Baseline-preserving fractal-fractional residual F2MI
```

核心输出:

```text
final_logit = baseline_logit + residual_scale * residual_logit
```

其中:

| 分支 | 来源 | 作用 |
| --- | --- | --- |
| baseline_logit | ResNet18 global feature + baseline classifier | 复现/保持 `002002002` contour auxiliary 的强分类决策 |
| residual_logit | fractal/fractional F2MI branch | 只学习相对 baseline 的样本级修正 |
| final_logit | baseline + bounded residual | 最终用于二分类 |

默认:

```text
residual_scale = 0.35
lambda_baseline_cls = 0.5
lambda_residual_l2 = 0.02
```

## 初始化策略

默认从 `002002002_contour_auxiliary_resnet18` 的 fold checkpoint 初始化:

```text
Models_saved/Generated_Models/002002002_contour_auxiliary_resnet18/contour_aux_resnet18_fold{fold}.pt
```

映射内容:

| 002002002 checkpoint | 002002003010 model |
| --- | --- |
| ResNet18 encoder | shared encoder |
| contour decoder | contour decoder |
| classifier | baseline_classifier |

F2MI residual branch 保持随机初始化，然后在 baseline-preserving loss 下学习。

## 损失函数

```text
L_total =
  L_cls(final_logit, y)
  + lambda_baseline_cls * L_cls(baseline_logit, y)
  + lambda_residual_l2 * mean(residual_logit^2)
  + lambda_mask * L_contour_mask
  + lambda_points * L_contour_points
  + lambda_landmark * L_landmark
  + lambda_morphology * L_morphology
  + lambda_fractal * L_fractal_consistency
```

设计含义:

- `L_cls(final_logit)` 让最终模型追求分类性能;
- `L_cls(baseline_logit)` 防止 baseline head 在联合训练中被破坏;
- `residual_l2` 防止 residual 分支过度改写 baseline;
- 结构监督继续保持 training-only，不进入测试输入。

## 数据与公平性原则

复用 `002002002` 的 matched image-contour 数据:

| 项目 | 数值 |
| --- | --- |
| samples | 667 |
| normal | 324 |
| hypertrophy | 343 |
| unique sample_id | 663 |
| CV | 5-fold StratifiedGroupKFold |
| inference input | raw X-ray image only |

测试阶段仍然不输入人工轮廓、A/N、PAS、关键点或参考线。

## 成功标准

主成功标准建议预注册为:

```text
F1 >= 0.8596
Recall >= 0.8700
Pre >= 0.8600
```

更稳妥的论文成功标准:

```text
在 F1 不低于 002002002 的前提下，提高 recall;
```

失败判据:

```text
不能作为更强 proposed model。
```

## 输出文件

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

预测文件额外保存:

| 字段 | 含义 |
| --- | --- |
| `prob_hypertrophy` | final probability |
| `baseline_prob_hypertrophy` | baseline head probability |
| `residual_logit` | F2MI residual correction |
| `final_logit` | final decision logit |
| `router_*` | sample-adaptive router 权重 |
| `fractional_alpha_*` | 分数阶参数 |

## 运行命令

准备数据与写入 summary:

```powershell
python experiments\002_image_contour_landmark_guided_fusion_classification\002002_image_contour_prediction\002002003010_baseline_preserving_residual_f2mi\scripts\run_010.py --prepare-only
```

smoke test:

```powershell
python experiments\002_image_contour_landmark_guided_fusion_classification\002002_image_contour_prediction\002002003010_baseline_preserving_residual_f2mi\scripts\run_010.py --smoke-test --device cpu --no-amp
```

正式训练:

```powershell
python experiments\002_image_contour_landmark_guided_fusion_classification\002002_image_contour_prediction\002002003010_baseline_preserving_residual_f2mi\scripts\run_010.py
```

分 fold 训练:

```powershell
python experiments\002_image_contour_landmark_guided_fusion_classification\002002_image_contour_prediction\002002003010_baseline_preserving_residual_f2mi\scripts\run_010.py --folds 0 1
```

## 当前状态

```text
实验准备中。
已建立目录、README、实验设计和训练脚本。
下一步应先运行 prepare-only 和 smoke-test，确认 checkpoint 映射、数据读取和 forward/loss 正常。
```
