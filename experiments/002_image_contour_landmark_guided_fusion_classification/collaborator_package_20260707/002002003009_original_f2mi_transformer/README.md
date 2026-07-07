# 002002003009: Fractal-Guided Multimodal Interaction

创建日期: 2026-07-04

## 实验定位

本实验是 `002002003008_dynamic_fractal_fractional_fusion` 之后的新主路线。

实验编号:

```text
002002003009
```

模型名称:

```text
F2MI-Transformer
Fractal-Fractional Multimodal Interaction Transformer
```

## 核心主张

本实验不再把 fractal/fractional 特征作为分类头前的附加拼接特征，而是将其提升为多模态交互机制的核心。

核心思想:

```text
Fractal tokens summarize multiscale anatomical shape complexity.
They condition cross-modal interaction among image, contour, landmark, and morphology tokens.
```

中文表述:

```text
分形 token 概括了解剖结构的多尺度几何复杂度。
模型使用分形 token 作为结构查询、动态路由变量和分数阶参数控制信号，
从而引导 image、contour、landmark、morphology 之间的信息交互。
```

## 与 002002003008 的区别

`002002003008` 的路线是:

```text
predicted contour -> fractal/fractional descriptors -> logit correction
```

其问题是:

```text
fractal/fractional 分支虽然活动，但没有形成有效分类增益。
```

`002002003009` 改为:

```text
image encoder
  -> predicted contour / landmark / morphology tokens
  -> fractal tokens as multimodal queries
  -> fractal router for adaptive modality weighting
  -> fractal-conditioned fractional contour operator
  -> classification
```

即:

```text
分形不再只修正最终 logit，而是参与中间层多模态交互。
```

## 输入与公平性

推理阶段默认只输入原始 X-ray 图像。

训练阶段可以使用已有标注作为监督信号:

| 标注 | 用途 |
| --- | --- |
| y/e contour linestrip | contour point target, contour mask target, fractal target |
| b/Y points | landmark target |
| B line | line/morphology target |
| A_N | morphology target |

重要原则:

```text
人工 contour、landmark、line、A_N 不直接作为测试输入。
它们只用于训练监督。
```

## 模型结构

```text
Input X-ray
ResNet18 image encoder + FPN-style decoder
Image tokens
Predicted contour mask + predicted 128-point y/e contours
Contour sequence tokens
Predicted b/Y/B landmarks and morphology vector
Landmark tokens + morphology tokens
Differentiable fractal token encoder
Fractal tokens
Fractal-guided cross-modal attention
Fractal router
Fractal-conditioned fractional contour operator
Normal / Hypertrophy prediction
```

## 关键模块

### 1. Fractal Tokens

从预测 contour probability map 中构造多尺度 fractal tokens:

```text
F = {F_2, F_4, F_8, F_16, F_32}
```

每个 token 包含:

```text
box-counting occupancy
multifractal q-moments
scale code
```

这些 token 描述不同尺度下的边界复杂度。

### 2. Fractal-Guided Cross-Modal Attention

fractal tokens 作为 queries:

```text
Z_fi = Attention(Q=F, K=Image, V=Image)
Z_fc = Attention(Q=F, K=Contour, V=Contour)
Z_fl = Attention(Q=F, K=Landmark, V=Landmark)
Z_fm = Attention(Q=F, K=Morphology, V=Morphology)
```

含义:

```text
由样本的多尺度形态复杂度决定从不同模态中读取什么信息。
```

### 3. Fractal Router

fractal tokens 同时生成模态路由权重:

```text
[w_image, w_contour, w_landmark, w_morphology] = Router(F)
```

最终融合:

```text
Z = w_image * Z_image
  + w_contour * Z_contour
  + w_landmark * Z_landmark
  + w_morphology * Z_morphology
```

### 4. Fractal-Conditioned Fractional Contour Operator

由 fractal tokens 预测 fractional order:

```text
alpha_y, alpha_e = AlphaNet(F)
```

再对 128-point contour sequence 做分数阶差分:

```text
D^alpha C
```

含义:

```text
fractal complexity controls fractional-order contour modeling.
```

## 损失函数

```text
L_total = L_cls
        + lambda_mask * L_contour_mask
        + lambda_points * L_contour_points
        + lambda_landmark * L_landmark
        + lambda_morphology * L_morphology
        + lambda_fractal * L_fractal_token
```

## 成功标准

目标不是只让模块活动，而是超过当前强基线:

| 对比对象 | 目标 |
| --- | --- |
| 002002003003 beta calibrated F2CG | Acc/F1 接近或超过 |
| 002002003008 DFF-Net | 明显超过，证明新交互机制优于 logit correction 路线 |

## 必做消融

| 消融 | 目的 |
| --- | --- |
| no fractal tokens | 验证分形 token 是否必要 |
| fractal concat only | 验证交互优于简单拼接 |
| no fractal router | 验证路由贡献 |
| no fractal query attention | 验证 cross-modal query 贡献 |
| no fractional contour operator | 验证 fraction 模块贡献 |
| shuffled fractal tokens | 验证样本级 fractal token 对应关系是否重要 |
| random router tokens | 排除只是参数量增加 |

## 当前状态

```text
Step 1: 实验编号已分配。
Step 2: 实验目录已建立。
Step 3: 架构设计 README 已完成。
Step 4: 模型脚本已实现。
Step 5: prepare-only 已通过。
Step 6: smoke test 已通过。
Step 7: 等待正式训练。
```

## 已完成的 smoke test

运行命令:

```text
python scripts/run_002002003009_f2mi_transformer.py --smoke-test --device cpu --no-pretrained --no-amp --allow-random-fallback --batch-size 4
```

关键输出:

| 项目 | 形状/结果 |
| --- | --- |
| image | [4, 3, 224, 224] |
| mask | [4, 2, 224, 224] |
| target contour points | [4, 2, 128, 2] |
| target landmarks | [4, 4, 2] |
| target morphology | [4, 10] |
| predicted contour points | [4, 2, 128, 2] |
| predicted landmarks | [4, 4, 2] |
| predicted morphology | [4, 10] |
| fractal tokens | [4, 5, 128] |
| image tokens | [4, 49, 128] |
| contour tokens | [4, 256, 128] |
| landmark tokens | [4, 4, 128] |
| morphology tokens | [4, 10, 128] |
| fractional alpha mean | 0.9021 |

初始 router 权重均值:

```text
image      0.2407
contour    0.2143
landmark   0.4177
morphology 0.1273
```

说明: 当前权重只是随机初始化模型的 smoke test 输出，不代表训练后模型偏好。
