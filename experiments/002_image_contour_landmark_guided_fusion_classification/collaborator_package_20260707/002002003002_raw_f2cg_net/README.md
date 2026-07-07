# 002002003002：F2CG-Net Gated Fractal-Fractional Contour Fusion

创建日期：2026-07-02

## 实验定位

`002002003002` 是对 `002002003001` 的改进实验，目标是让 contour fusion 这条路线更适合投稿 *Fractal and Fractional*。

`002002003001` 的直接预测轮廓特征融合没有超过 `002002002`，主要问题可能是：

```text
预测 contour probability map 噪声较大；
直接编码最终 mask 并 concat 到分类头不稳定；
fusion classifier 自由度增加后可能更容易过拟合。
```

因此本实验不再直接融合最终预测 mask，而是设计：

```text
F2CG-Net: Fractal-Fractional Contour-Guided Network
```

核心思想：

```text
用 contour decoder 的中间特征通过 gated fusion 调制图像特征；
再从预测轮廓概率图中提取 differentiable multifractal scale embedding；
最后用 fractional-order scale memory 建模多尺度复杂度之间的长程依赖。
```

## 与 Fractal and Fractional 的对应关系

该实验面向 *Fractal and Fractional* 的方法学贡献，而不是普通医学分类模型。

| 期刊关注点 | 本实验对应设计 |
| --- | --- |
| Fractal / multifractal analysis | Differentiable multifractal scale layer |
| Fractional-order modeling | Fractional-order scale memory module |
| Nonlinear / multi-scale methods | scale-wise contour complexity embedding |
| Application to life science / image analysis | 腺样体肥大侧位X-ray分类 |
| Interpretability | contour prediction、scale response、fractional order alpha |

## 模型结构

```text
Input X-ray
ResNet18 encoder
Multi-scale image features: C2, C3, C4, C5
Contour decoder
Multi-scale contour decoder features: D2, D3, D4
Contour-guided gated fusion
Predicted y/e contour probability maps
Differentiable multifractal scale layer
Fractional-order scale memory
Image feature + gated contour feature + fractal embedding + fractional embedding
Fusion classifier
normal / hypertrophy
```

## 关键模块

### 1. Gated Contour-Guided Fusion

不直接把最终预测 mask 拼接到分类头，而是在中间尺度上进行门控融合：

```text
G_l = sigmoid(Conv([C_l, D_l]))
F_l = C_l + alpha_l * G_l * D_l
```

其中：

```text
C_l = image encoder feature
D_l = contour decoder feature
G_l = contour-guided gate
alpha_l = learnable fusion strength
```

这样 contour 分支是结构引导项，而不是强行参与分类的噪声输入。

### 2. Differentiable Multifractal Scale Layer

对预测轮廓概率图 `P(x,y)` 做 soft box-counting：

```text
N(epsilon) = sum_b [1 - exp(-gamma * sum_{(x,y) in box b} P(x,y))]
```

得到：

```text
log N(epsilon)
scale-wise contour complexity vector
global slope-like fractal response
q-order partition responses
```

当前默认尺度：

```text
epsilon = 2, 4, 8, 16, 32
```

当前默认 q 阶：

```text
q = 0.5, 1.0, 2.0
```

### 3. Fractional-Order Scale Memory

在多尺度分形响应序列上使用可学习阶数 `alpha` 的分数阶差分：

```text
D^alpha Z_k = sum_j w_j(alpha) * Z_{k-j}
```

其中：

```text
alpha in (0, 1)
```

医学解释：

```text
粗尺度：整体腺样体膨隆和气道受压趋势
中尺度：轮廓形态变化
细尺度：局部边界复杂度和不规则性
```

fractional memory 用于建模这些尺度之间的依赖，而不是把各尺度独立看待。

## 损失函数

```text
L_total = L_cls
        + lambda_contour * L_contour
        + lambda_fractal * L_fractal_consistency
```

其中：

```text
L_cls = BCEWithLogitsLoss
L_contour = weighted BCE + Dice loss
L_fractal_consistency = SmoothL1(F_pred, F_manual)
```

`F_manual` 来自人工 y/e 轮廓 mask 计算出的同一套 multifractal scale vector，只在训练阶段作为约束使用。

## 复用数据

复用 `002002002` 的数据准备结果：

```text
../002002002_image_contour_auxiliary_supervision/outputs/metadata.csv
../002002002_image_contour_auxiliary_supervision/outputs/folds.csv
../002002002_image_contour_auxiliary_supervision/outputs/outer_splits.csv
../002002002_image_contour_auxiliary_supervision/outputs/contour_targets_224_uint8.npz
```

## 关键对比

| 对比 | 目的 |
| --- | --- |
| 002002002 vs 002002003002 | 判断 F2CG-Net 是否优于单纯 contour auxiliary |
| 002002003001 vs 002002003002 | 判断 gated + fractal/fractional 是否优于直接 contour feature concat |
| 002002003002 vs 后续 002005 | 判断该轻量分形/分数阶模块是否可作为最终 AFS-Net 基础 |

## 当前状态

```text
目录已建立。
README设计记录已完成。
训练脚本已完成。
prepare-only 检查已通过。
smoke test 已通过。
正式5-fold训练已完成。
```

## Smoke Test

已验证：

```text
image_shape = [4, 3, 224, 224]
mask_shape = [4, 2, 224, 224]
cls_logits_shape = [4]
contour_logits_shape = [4, 2, 224, 224]
fractal_scale_vector_shape = [4, 42]
fractal_embedding_shape = [4, 64]
fractional_memory_shape = [4, 5]
fractional_alpha = 0.5
status = ok
```

## 建议正式训练命令

```text
python scripts/run_002002003002_f2cg.py --device cuda --epochs 80 --patience 10 --batch-size 20 --lr 1e-4 --weight-decay 1e-4 --lambda-contour 0.3 --lambda-fractal 0.05 --allow-random-fallback
```

## 正式训练结果

训练日期：2026-07-03

输出文件：

```text
outputs/cv_summary.csv
outputs/cv_metrics_by_fold.csv
outputs/cv_predictions.csv
outputs/training_history.csv
outputs/run_summary.md
```

模型权重：

```text
Models_saved/Generated_Models/002002003002_f2cg_resnet18/
```

5-fold交叉验证汇总：

| 指标 | Mean | Std |
| --- | --- | --- |
| Accuracy | 0.8379 | 0.0325 |
| Precision | 0.9147 | 0.0484 |
| Recall | 0.7584 | 0.1163 |
| F1 | 0.8230 | 0.0541 |
| Mean Dice | 0.1358 | 0.0249 |
| Mean IoU | 0.0733 | 0.0142 |
| Fractional alpha | 0.5000 | 0.0003 |

与关键基线比较：

| 实验 | F1 | Mean Dice |
| --- | --- | --- |
| 002002001 image-only | 0.8321 | - |
| 002002002 contour auxiliary | 0.8596 | 0.1720 |
| 002002003001 direct contour fusion | 0.8484 | 0.0585 |
| 002002003002 F2CG-Net | 0.8230 | 0.1358 |

阶段性解释：

```text
说明门控轮廓融合 + 多分形/分数阶尺度建模提升了样本排序能力。

说明默认阈值下更保守，更少误判肥大，但会漏掉一部分肥大样本。

但还不能作为“阈值分类全面优于002002002”的最终结论。
```

## 阈值校准后的补充结论

后续已建立并完成：

```text
../002002003002_result_analysis/
```

该分析读取 `outputs/cv_predictions.csv`，比较默认阈值、全局validation阈值、fold-wise validation阈值和test-oracle阈值。

关键结果：

| 设置 | Recall | F1 |
| --- | --- | --- |
| 默认阈值0.5 | 0.7638 | 0.8291 |
| 全局validation阈值0.038 | 0.8571 | 0.8547 |
| fold-wise validation阈值 | 0.8571 | 0.8673 |
| 002002002 contour auxiliary | 0.8383 | 0.8596 |

更新后的解释：

```text
这说明模型架构本身有价值，但不同fold模型的概率尺度存在差异。
后续若将F2CG-Net作为投稿主模型，应把阈值选择或概率校准作为正式推理流程的一部分。
```

## 概率校准后的进一步结论

后续已建立并完成：

```text
../002002003003_f2cg_probability_calibration/
```

该实验使用每个fold的validation预测拟合概率校准器，再固定应用到对应fold的test预测。

关键结果：

| 设置 | Recall | F1 |
| --- | --- | --- |
| raw + default 0.5 | 0.7638 | 0.8291 |
| raw + fold-wise threshold | 0.8571 | 0.8673 |
| beta calibration + default 0.5 | 0.8367 | 0.8697 |
| isotonic calibration + default 0.5 | 0.8688 | 0.8726 |
| 002002002 contour auxiliary | 0.8383 | 0.8596 |

更新后的推荐推理流程：

```text
F2CG-Net forward
-> fold/model-specific probability calibration
-> default threshold 0.5
-> final hypertrophy prediction
```

推荐优先级：

```text
主推荐：beta calibration + default 0.5
补充报告：isotonic calibration + default 0.5
```
