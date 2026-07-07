# 002002003003: F2CG Probability Calibration

创建日期：2026-07-03

## 实验目的

`002002003002_result_analysis` 已经证明：

```text
F2CG-Net 默认阈值0.5下偏保守；
```

本实验继续向前一步，正式验证：

```text
能否通过概率校准，让F2CG-Net的输出概率尺度更稳定，
从而把threshold calibration纳入正式推理流程？
```

## 输入

```text
../002002003002_f2cg_gated_fractal_fractional_fusion/outputs/cv_predictions.csv
```

使用方式：

```text
每个outer fold用该fold的validation预测拟合校准器；
然后把该校准器固定到同一fold的test预测上。
```

这样避免用test预测拟合校准器。

## 校准方法

| 方法 | 说明 |
| --- | --- |
| raw | 不做概率校准 |
| temperature | 对logit做temperature scaling |
| platt | logistic / sigmoid calibration |
| beta | beta calibration近似形式 |
| isotonic | 非参数isotonic regression |

## 输出指标

概率质量指标：

```text
Brier score
NLL
ECE-10
```

分类决策指标：

```text
Accuracy
Precision
Recall
F1
```

## 输出文件

```text
outputs/calibrated_predictions.csv
outputs/calibrator_parameters.csv
outputs/probability_calibration_summary.csv
outputs/probability_calibration_by_fold.csv
outputs/calibrated_decision_summary.csv
outputs/calibrated_decision_by_fold.csv
outputs/calibration_bins.csv
outputs/probability_calibration_report.md
figures/calibration_curves.png
figures/calibration_metric_bars.png
figures/roc_pr_after_calibration.png
```

## 运行命令

```text
python scripts/run_probability_calibration.py
```

## 判读规则

```text
如果某个校准方法显著降低Brier/ECE，说明概率可信度改善。
如果校准后默认阈值0.5即可接近或超过fold-wise threshold结果，
说明概率校准可以替代手动阈值调节。
如果仍需要validation-selected threshold，
则投稿时应把threshold calibration作为正式推理步骤描述。
```

## 已完成结果

### 1. 概率质量

| 方法 | Brier | NLL | ECE-10 |
| --- | --- | --- | --- |
| isotonic | 0.0996 | 0.5490 | 0.0588 |
| beta | 0.1046 | 0.3588 | 0.0428 |
| platt | 0.1084 | 0.3660 | 0.0575 |
| temperature | 0.1216 | 0.3952 | 0.0619 |
| raw | 0.1388 | 0.5519 | 0.1305 |

解释：

```text
isotonic 的 Brier score 最低，说明整体概率误差最小。
beta calibration 的 NLL 和 ECE-10 最低，说明概率可信度和分箱校准最好。
raw 概率的 ECE-10 明显偏高，验证了F2CG-Net原始概率尺度确实需要校准。
```

### 2. 默认阈值0.5下的分类表现

| 方法 | Recall | F1 |
| --- | --- | --- |
| beta + default 0.5 | 0.8367 | 0.8697 |
| isotonic + default 0.5 | 0.8688 | 0.8726 |
| platt + default 0.5 | 0.8192 | 0.8593 |
| raw + default 0.5 | 0.7638 | 0.8291 |
| 002002002 contour auxiliary | 0.8383 | 0.8596 |

关键结论：

```text
概率校准后，不需要再手动降低阈值到0.038或做fold-wise threshold search，
默认阈值0.5已经可以超过002002002。
```

其中：

```text
isotonic calibration 更适合追求 Recall / F1；
platt calibration 接近002002002，但优势不如 beta 和 isotonic 明显；
temperature scaling 对分类阈值没有改善。
```

### 3. 与上一阶段阈值校准的关系

上一阶段 `002002003002_result_analysis` 的最佳结果：

```text
fold-wise validation threshold:
F1           = 0.8673
```

本实验最佳结果：

```text
beta + default 0.5:
F1           = 0.8697

isotonic + default 0.5:
F1           = 0.8726
```

因此：

```text
probability calibration 比单纯 threshold calibration 更适合作为正式推理流程。
```

### 4. 当前推荐

论文和后续实验中建议采用：

```text
主推荐：beta calibration + default threshold 0.5
补充报告：isotonic calibration + default threshold 0.5
```

理由：

```text
beta calibration 是参数化方法，更容易在论文中描述和复现；
isotonic calibration 性能很好，但非参数方法在小样本validation上更容易被质疑过拟合。
```

## 统计验证补充

后续已完成：

```text
../002002003004_calibrated_f2cg_statistical_validation/
```

统计验证后的更严谨解释：

```text
beta/isotonic calibration相对raw F2CG有明确统计改善；
```

因此论文中应表述为：

```text
while probability calibration significantly improved the raw F2CG-Net decision performance.
```
