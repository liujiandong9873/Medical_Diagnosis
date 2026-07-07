# 002002003005: F2CG-Net Module Ablation Study

创建日期：2026-07-03

## 实验编号

```text
002002003005_f2cg_module_ablation
```

本实验接在以下实验之后：

```text
002002003001: direct predicted contour feature fusion
002002003002: full F2CG-Net
002002003003: F2CG probability calibration
002002003004: calibrated F2CG statistical validation
```

## 实验目的

前序实验已经说明：

```text
1. 直接预测轮廓concat不稳定。
2. F2CG-Net在ranking层面表现最好。
3. F2CG-Net默认阈值0.5下偏保守。
4. probability calibration可以显著改善raw F2CG的分类决策。
5. calibrated F2CG相对002002002是点估计提升，但统计上仍是趋势性提升。
```

现在最需要补充的是模块消融：

```text
F2CG-Net的性能提升到底来自哪里？
是gated contour fusion？
是multifractal embedding？
是fractional scale memory？
还是主要来自probability calibration？
```

这个问题对投稿 *Fractal and Fractional* 非常关键。因为如果不能证明分形/分数阶模块有独立贡献，文章会容易被认为只是普通多任务CNN加校准。

## 总体假设

### H1: Gated contour fusion 比 direct contour concat 更稳定

```text
如果 gated-only 模型优于 002002003001 direct fusion，
说明问题不是“轮廓融合没有价值”，而是直接concat方式不稳。
```

### H2: Multifractal scale embedding 提供额外结构复杂度信息

```text
如果 gated + fractal 优于 gated-only，
说明从预测轮廓中提取多尺度分形响应有实际贡献。
```

### H3: Fractional scale memory 改善尺度依赖建模

```text
如果 gated + fractal + fractional 优于 gated + fractal，
说明fractional-order memory对多尺度复杂度建模有贡献。
```

### H4: Probability calibration 是F2CG推理流程的必要组成部分

```text
如果full F2CG + beta calibration明显优于full F2CG raw，
说明校准不是附属技巧，而是最终推理流程的一部分。
```

## 数据与公平性

复用 `002002002` 准备好的 matched image-contour 数据：

```text
../002002002_image_contour_auxiliary_supervision/outputs/metadata.csv
../002002002_image_contour_auxiliary_supervision/outputs/folds.csv
../002002002_image_contour_auxiliary_supervision/outputs/outer_splits.csv
../002002002_image_contour_auxiliary_supervision/outputs/contour_targets_224_uint8.npz
```

样本：

| 项目 | 数值 |
| --- | --- |
| samples | 667 |
| unique sample_id | 663 |
| normal | 324 |
| hypertrophy | 343 |
| folds | 5-fold StratifiedGroupKFold |
| image size | 224 x 224 |
| contour channels | y, e |

公平性原则：

```text
所有新训练的ablation模型使用相同样本、相同fold、相同ResNet18 backbone。
测试阶段只能输入原始X-ray图像。
人工y/e轮廓只能作为训练监督、fractal consistency target或validation校准依据。
```

## 消融矩阵

### 已有对照实验

| 编号 | 模型 | Gated | Fractal | Fractional | Calibration | 是否新训练 | 作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 002002001 | matched image-only | 否 | 否 | 否 | 否 | 已完成 | 公平image-only基线 |
| 002002002 | contour auxiliary | 否 | 否 | 否 | 否 | 已完成 | 轮廓辅助监督基线 |
| 002002003001 | direct contour concat | 否 | 否 | 否 | 否 | 已完成 | 证明直接融合不稳定 |
| 002002003002 | full F2CG raw | 是 | 是 | 是 | 否 | 已完成 | 完整F2CG结构 |
| 002002003003 | full F2CG calibrated | 是 | 是 | 是 | 是 | 已完成 | 完整F2CG + 概率校准 |

### 新增消融子实验

| 编号 | 模型 | Gated | Fractal embedding | Fractional memory | Fractal consistency loss | Calibration | 目的 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 002002003005001 | gated-only F2CG | 是 | 否 | 否 | 否 | 后处理 | 单独验证门控轮廓融合 |
| 002002003005002 | fractal-only F2CG | 否 | 是 | 否 | 是 | 后处理 | 验证分形尺度特征，不使用门控融合 |
| 002002003005003 | fractional-only F2CG | 否 | 否 | 是 | 是 | 后处理 | 验证分数阶尺度记忆，不使用门控融合 |
| 002002003005004 | gated + fractal | 是 | 是 | 否 | 是 | 后处理 | 验证门控融合基础上分形模块贡献 |
| 002002003005005 | gated + fractional | 是 | 否 | 是 | 是 | 后处理 | 验证门控融合基础上分数阶记忆贡献 |
| 002002003005006 | fractal + fractional without gates | 否 | 是 | 是 | 是 | 后处理 | 验证分形/分数阶模块是否可以独立工作 |

说明：

```text
fractional-only 并不是完全不计算fractal scale sequence。
FractionalScaleMemory需要multifractal scale sequence作为输入，
但该模型不把fractal embedding直接送入分类器，只使用fractional memory embedding。
```

## 统一训练设置

建议沿用 `002002003002` 的主要配置：

| 参数 | 值 |
| --- | --- |
| backbone | ResNet18 |
| image size | 224 |
| epochs | 80 |
| patience | 10 |
| batch size | 20 |
| lr | 1e-4 |
| weight decay | 1e-4 |
| lambda_contour | 0.3 |
| lambda_fractal | 0.05 |
| contour_pos_weight | 50.0 |
| monitor | val_f1 |
| device | cuda |

训练后统一做：

```text
raw default threshold 0.5
fold-wise validation threshold
beta calibration + default threshold 0.5
isotonic calibration + default threshold 0.5
```

这样可以区分：

```text
模型结构贡献
阈值选择贡献
概率校准贡献
```

## 输出指标

分类指标：

```text
Accuracy
Precision
Recall
F1
Brier
NLL
ECE-10
```

结构指标：

```text
Mean Dice
y Dice
e Dice
Mean IoU
fractal consistency loss
fractional alpha mean/std
```

统计分析：

```text
bootstrap 95% CI
paired bootstrap difference test
primary comparison: each ablation vs 002002002
internal comparison: each ablation vs full F2CG raw
calibration comparison: raw vs beta-calibrated within same ablation
```

## 主要判读路径

### 1. 验证 gated fusion

关键比较：

```text
002002003005001 gated-only
vs 002002003001 direct contour concat
vs 002002002 contour auxiliary
```

如果 gated-only 明显优于 direct concat，则说明：

```text
中间特征门控比最终mask concat更适合contour fusion。
```

### 2. 验证 fractal 模块

关键比较：

```text
002002003005004 gated + fractal
vs 002002003005001 gated-only
```

如果提升明显，则支持：

```text
multifractal scale embedding提供了分类相关的结构复杂度信息。
```

### 3. 验证 fractional 模块

关键比较：

```text
002002003002 full F2CG
vs 002002003005004 gated + fractal
```

如果 full F2CG 更好，则支持：

```text
fractional-order scale memory对多尺度依赖建模有贡献。
```

如果没有提升，则需要调整文章表述：

```text
fractional module作为理论启发和可解释尺度记忆组件，
但当前数据上独立增益有限。
```

### 4. 验证 calibration

关键比较：

```text
same architecture raw default 0.5
vs beta calibration + default 0.5
```

如果校准稳定提升，则支持：

```text
probability calibration是F2CG-Net推理流程中的必要组成部分。
```

## 预期风险

1. 样本量只有667，多个消融模型可能出现fold间波动。
2. fractional alpha在full F2CG中几乎保持0.5，fractional模块可能难以证明独立贡献。
3. calibration可能贡献很大，容易掩盖结构模块贡献。
4. 如果所有消融差异不显著，论文主张应从“显著提升准确率”转向“结构建模与可解释校准框架”。

## 成功标准

强结果：

```text
gated + fractal + fractional > gated + fractal > gated-only > direct concat
```

可接受结果：

```text
gated-only或gated + fractal明显优于direct concat；
full F2CG相对raw F2CG经过校准后稳定改善；
相对002002002为趋势性提升。
```

负结果但仍可发表的结果：

```text
gated/fractal/fractional单模块提升不显著，
但完整流程显示预测轮廓融合必须结合概率校准；
文章主张改为“fractal-fractional contour-guided calibrated framework”。
```

## 当前状态

```text
实验编号已分配。
目录已建立。
README设计已完成。
统一训练脚本已建立。
prepare-only检查已通过。
6个variant的smoke test已通过。
002002003005001_gated_only 已完成正式5-fold训练。
002002003005002_fractal_only 已完成正式5-fold训练。
002002003005003_fractional_only 已完成正式5-fold训练。
002002003005004_gated_fractal 已完成正式5-fold训练。
002002003005005_gated_fractional 已完成正式5-fold训练。
002002003005006_fractal_fractional_no_gates 已完成正式5-fold训练。
6个variant已全部完成正式5-fold训练。
```

## 已完成正式训练结果

| Variant | Gated | Fractal embedding | Fractional embedding | F1 | Recall | Mean Dice |
| --- | --- | --- | --- | --- | --- | --- |
| 002002003005001_gated_only | 是 | 否 | 否 | 0.8398 ± 0.0805 | 0.8077 ± 0.1682 | 0.1258 ± 0.0770 |
| 002002003005002_fractal_only | 否 | 是 | 否 | 0.8348 ± 0.0489 | 0.7917 ± 0.1150 | 0.1815 ± 0.0227 |
| 002002003005003_fractional_only | 否 | 否 | 是 | 0.8593 ± 0.0204 | 0.8418 ± 0.0549 | 0.1423 ± 0.0297 |
| 002002003005004_gated_fractal | 是 | 是 | 否 | 0.8416 ± 0.0406 | 0.8133 ± 0.1160 | 0.1248 ± 0.0235 |
| 002002003005005_gated_fractional | 是 | 否 | 是 | 0.8412 ± 0.0583 | 0.8355 ± 0.0903 | 0.1045 ± 0.0608 |
| 002002003005006_fractal_fractional_no_gates | 否 | 是 | 是 | 0.8502 ± 0.0331 | 0.8326 ± 0.0789 | 0.1387 ± 0.0388 |

初步解读：

```text
说明门控轮廓融合可能改善ranking能力。

说明仅有门控还不足以构成完整的性能提升证据。

说明multifractal embedding可能改善ranking稳定性。

说明multifractal embedding本身确实携带分类相关的结构复杂度信息。

是本组消融中默认0.5阈值下分类指标最好的variant。
这说明fractional memory通道有独立分类贡献，
但其fractional alpha = 0.5001 ± 0.0001，
仍几乎固定在初始附近。

说明分形/分数阶模块脱离门控后仍能独立提供部分分类信息。

总体看，gated fusion不是唯一有效来源；
fractal embedding偏向提升ranking稳定性，
fractional memory偏向改善默认阈值分类，
但当前alpha缺乏动态变化，是后续方法学改进的关键问题。
```

输出文件：

```text
outputs/001_gated_only/run_summary.md
outputs/001_gated_only/cv_summary.csv
outputs/001_gated_only/cv_metrics_by_fold.csv
outputs/001_gated_only/cv_predictions.csv
outputs/001_gated_only/training_history.csv
outputs/002_fractal_only/run_summary.md
outputs/002_fractal_only/cv_summary.csv
outputs/002_fractal_only/cv_metrics_by_fold.csv
outputs/002_fractal_only/cv_predictions.csv
outputs/002_fractal_only/training_history.csv
outputs/003_fractional_only/run_summary.md
outputs/003_fractional_only/cv_summary.csv
outputs/003_fractional_only/cv_metrics_by_fold.csv
outputs/003_fractional_only/cv_predictions.csv
outputs/003_fractional_only/training_history.csv
outputs/004_gated_fractal/run_summary.md
outputs/004_gated_fractal/cv_summary.csv
outputs/004_gated_fractal/cv_metrics_by_fold.csv
outputs/004_gated_fractal/cv_predictions.csv
outputs/004_gated_fractal/training_history.csv
outputs/005_gated_fractional/run_summary.md
outputs/005_gated_fractional/cv_summary.csv
outputs/005_gated_fractional/cv_metrics_by_fold.csv
outputs/005_gated_fractional/cv_predictions.csv
outputs/005_gated_fractional/training_history.csv
outputs/006_ff_no_gates/run_summary.md
outputs/006_ff_no_gates/cv_summary.csv
outputs/006_ff_no_gates/cv_metrics_by_fold.csv
outputs/006_ff_no_gates/cv_predictions.csv
outputs/006_ff_no_gates/training_history.csv
```

## 统一训练脚本

```text
scripts/run_002002003005_ablation.py
```

该脚本通过 `--variant` 控制消融结构。

可选variant：

```text
002002003005001_gated_only
002002003005002_fractal_only
002002003005003_fractional_only
002002003005004_gated_fractal
002002003005005_gated_fractional
002002003005006_fractal_fractional_no_gates
```

输出目录使用短目录名以避免Windows路径过长：

| Variant | Output dir |
| --- | --- |
| 002002003005001_gated_only | `outputs/001_gated_only` |
| 002002003005002_fractal_only | `outputs/002_fractal_only` |
| 002002003005003_fractional_only | `outputs/003_fractional_only` |
| 002002003005004_gated_fractal | `outputs/004_gated_fractal` |
| 002002003005005_gated_fractional | `outputs/005_gated_fractional` |
| 002002003005006_fractal_fractional_no_gates | `outputs/006_ff_no_gates` |

模型权重目录：

```text
Models_saved/Generated_Models/002002003005_f2cg_module_ablation/<short_variant_dir>/
```

## 已完成检查

prepare-only：

```text
python scripts/run_002002003005_ablation.py --variant 002002003005001_gated_only --prepare-only --device cpu --no-pretrained --no-amp --allow-random-fallback
```

smoke test：

```text
6个variant均已完成smoke test。
所有variant均输出:
cls_logits_shape = [4]
contour_logits_shape = [4, 2, 224, 224]
fractal_scale_vector_shape = [4, 42]
fractal_embedding_shape = [4, 64]
fractional_memory_shape = [4, 5]
```

## 正式训练命令模板

```text
python scripts/run_002002003005_ablation.py --variant <VARIANT> --device cuda --epochs 80 --patience 10 --batch-size 20 --lr 1e-4 --weight-decay 1e-4 --lambda-contour 0.3 --lambda-fractal 0.05 --allow-random-fallback
```

建议训练顺序：

```text
1. 002002003005001_gated_only
2. 002002003005004_gated_fractal
3. 002002003005005_gated_fractional
4. 002002003005006_fractal_fractional_no_gates
5. 002002003005002_fractal_only
6. 002002003005003_fractional_only
```

理由：

```text
优先验证最关键的gated路线，再验证fractal/fractional增益。
fractal-only和fractional-only更像补充解释，可放在后面。
```
