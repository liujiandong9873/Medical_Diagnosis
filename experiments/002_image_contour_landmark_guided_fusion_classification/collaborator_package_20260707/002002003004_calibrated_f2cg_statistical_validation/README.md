# 002002003004: Calibrated F2CG Statistical Validation

创建日期：2026-07-03

## 实验目的

`002002003003` 已经显示：

```text
beta calibration + default 0.5
isotonic calibration + default 0.5
```


本实验的目的，是把这个结果从“指标观察”推进到“统计证据”：

```text
1. 给主要指标提供bootstrap 95% CI。
2. 对关键模型差异做paired bootstrap。
3. 生成论文可用图表。
4. 判断calibrated F2CG是否能作为正式投稿主结果。
```

## 输入

```text
../002002002_image_contour_auxiliary_supervision/outputs/cv_predictions.csv
../002002003003_f2cg_probability_calibration/outputs/calibrated_predictions.csv
```

使用 `split=test` 的 667 条 paired prediction。

## 比较模型

| 模型 | 说明 |
| --- | --- |
| 002002002_contour_aux | contour auxiliary baseline |
| f2cg_raw_default | F2CG raw probability + threshold 0.5 |
| f2cg_beta_default | F2CG beta-calibrated probability + threshold 0.5 |
| f2cg_isotonic_default | F2CG isotonic-calibrated probability + threshold 0.5 |

## 统计方法

```text
clustered paired bootstrap by sample_id
bootstrap iterations = 2000
seed = 20260703
```

指标：

```text
Brier
NLL
Accuracy
Precision
Recall
F1
```

关键差异比较：

```text
beta-calibrated F2CG vs 002002002
isotonic-calibrated F2CG vs 002002002
raw F2CG vs 002002002
beta-calibrated F2CG vs raw F2CG
isotonic-calibrated F2CG vs raw F2CG
```

## 运行命令

```text
python scripts/run_statistical_validation.py
```

## 输出

```text
outputs/observed_metrics.csv
outputs/bootstrap_metric_ci.csv
outputs/paired_bootstrap_difference_tests.csv
outputs/statistical_validation_report.md
outputs/paired_model_predictions_long.csv
outputs/paired_model_predictions_wide.csv
outputs/bootstrap_metric_samples.csv
outputs/paired_bootstrap_difference_samples.csv
outputs/calibration_bins.csv
figures/roc_pr_model_comparison.png
figures/calibration_model_comparison.png
figures/decision_metric_bars.png
figures/confusion_matrices.png
```

## 判读原则

```text
则可以作为较强统计支持。

如果CI跨0但点估计更好，
则应表述为趋势性提升，并强调样本量限制。

beta calibration是参数化主方案；
isotonic calibration作为敏感性分析。
```

## 已完成结果

### 1. 主要指标及95% CI

| 模型 | Recall | F1 | Brier | NLL |
| --- | --- | --- | --- | --- |
| 002002002 contour auxiliary | 0.8367 [0.7954, 0.8750] | 0.8593 [0.8283, 0.8870] | 0.1143 [0.0942, 0.1367] | 0.4995 [0.3960, 0.6143] |
| F2CG raw + 0.5 | 0.7638 [0.7194, 0.8088] | 0.8291 [0.7961, 0.8602] | 0.1388 [0.1149, 0.1635] | 0.5519 [0.4540, 0.6534] |
| F2CG beta-calibrated + 0.5 | 0.8367 [0.7949, 0.8750] | 0.8697 [0.8395, 0.8965] | 0.1046 [0.0874, 0.1234] | 0.3588 [0.3083, 0.4141] |
| F2CG isotonic-calibrated + 0.5 | 0.8688 [0.8328, 0.9041] | 0.8726 [0.8437, 0.8973] | 0.0996 [0.0835, 0.1170] | 0.5490 [0.4047, 0.7221] |

说明：

```text
这里使用的是paired test predictions的pooled统计；
因此和各训练脚本中的fold mean ± std会有轻微差异。
```

### 2. 与002002002的paired bootstrap差异

| 比较 | 指标 | 差值 | 95% CI | p |
| --- | --- | --- | --- | --- |
| beta F2CG - 002002002 | F1 | +0.0104 | [-0.0192, 0.0396] | 0.513 |
| beta F2CG - 002002002 | NLL | -0.1407 | [-0.2373, -0.0470] | 0.002 |
| isotonic F2CG - 002002002 | F1 | +0.0133 | [-0.0142, 0.0414] | 0.357 |
| isotonic F2CG - 002002002 | Recall | +0.0321 | [-0.0060, 0.0727] | 0.130 |

解释：

```text
但差异CI跨0，不能宣称显著优于002002002。

beta calibration相对002002002的NLL显著降低，
说明概率质量有较明确改善。
```

### 3. 与raw F2CG的paired bootstrap差异

| 比较 | 指标 | 差值 | 95% CI | p |
| --- | --- | --- | --- | --- |
| beta F2CG - raw F2CG | F1 | +0.0406 | [0.0224, 0.0611] | 0.000 |
| beta F2CG - raw F2CG | Recall | +0.0729 | [0.0448, 0.1032] | 0.000 |
| beta F2CG - raw F2CG | Brier | -0.0342 | [-0.0473, -0.0221] | 0.000 |
| isotonic F2CG - raw F2CG | F1 | +0.0435 | [0.0175, 0.0712] | 0.001 |
| isotonic F2CG - raw F2CG | Recall | +0.1050 | [0.0662, 0.1454] | 0.000 |
| isotonic F2CG - raw F2CG | Brier | -0.0392 | [-0.0562, -0.0231] | 0.000 |

解释：

```text
概率校准相对raw F2CG的改善具有明确统计支持。
这说明F2CG-Net的问题主要不是结构无效，
而是原始输出概率尺度不适合直接使用默认0.5阈值。
```

### 4. 当前论文表述建议

推荐写法：

```text
```

中文解释：

```text
校准后的F2CG相对002002002是趋势性提升；
校准相对raw F2CG是有统计支持的改进。
```
