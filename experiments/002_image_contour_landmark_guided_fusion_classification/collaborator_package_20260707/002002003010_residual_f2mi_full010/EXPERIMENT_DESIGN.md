# 002002003010 实验设计

## 1. 背景

当前 `002002002 contour auxiliary ResNet18` 是实验002中最稳定的强 baseline:

| Model | Acc | Pre | Recall | F1 |
| --- | --- | --- | --- | --- |
| 002002002 contour auxiliary | 0.8593 | 0.8845 | 0.8383 | 0.8596 |

`002002003009003 recall-preserving adaptive F2MI` 的结果为:

| Model | Acc | Pre | Recall | F1 |
| --- | --- | --- | --- | --- |
| 009003 | 0.8573 | 0.8538 | 0.8700 | 0.8609 |


因此 010 的设计目标是:

```text
让 proposed model 不再从头替代强 baseline，
而是在强 baseline 决策之上学习 fractal/fractional residual correction。
```

## 2. 假设

主假设:

```text
如果 F2MI 的 fractal/fractional 信息确实能补充 baseline，
那么 residual branch 应该只在 baseline 不确定或容易错判的样本上提供有限修正，
```

反假设:

```text
如果 residual branch 只是改变阈值工作点，
与 009004 的失败模式类似。
```

## 3. 实验模型

### 3.1 Baseline branch

使用 ResNet18 encoder 的 global feature:

```text
h = GAP(ResNet18(X))
baseline_logit = Linear(h)
```

该 branch 从 `002002002` checkpoint 初始化，并继续接受分类监督。

### 3.2 Residual F2MI branch

继承 009003 的结构:

- contour mask decoder;
- 128-point y/e contour regression;
- landmark and morphology heads;
- fractal token encoder;
- fractional contour operator;
- sample-adaptive multimodal router。

输出:

```text
residual_logit = F2MIResidual(image, contour, landmark, morphology, fractal, fractional)
```

### 3.3 Final classifier

```text
final_logit = baseline_logit + residual_scale * residual_logit
```

其中 residual_scale 默认 0.35，用于限制 residual 对 baseline 的破坏。

## 4. 训练策略

默认从每个 fold 的 002002002 checkpoint 初始化:

```text
contour_aux_resnet18_fold0.pt
contour_aux_resnet18_fold1.pt
contour_aux_resnet18_fold2.pt
contour_aux_resnet18_fold3.pt
contour_aux_resnet18_fold4.pt
```

训练目标:

```text
L_total =
  L_cls(final)
  + 0.5 * L_cls(baseline)
  + 0.02 * L2(residual_logit)
  + 0.25 * L_mask
  + 0.15 * L_points
  + 0.05 * L_landmark
  + 0.05 * L_morphology
  + 0.05 * L_fractal
```

checkpoint monitor:

```text
val_f1
```

理由:

- F1 是 precision/recall 的综合指标;
- 010 的主目标是超过或至少不弱于 baseline，而不是无限推高 recall。

## 5. 主要对照

| 对照 | 目的 |
| --- | --- |
| 002002002 contour auxiliary | 主要强 baseline |
| 009003 recall-preserving adaptive F2MI | 当前候选模型 |

## 6. 主要评价指标

主指标:

```text
Acc, Pre, Recall, F1
```

机制指标:

```text
delta_f1_vs_baseline_head
delta_recall_vs_baseline_head
residual_abs_logit
router weights
fractional alpha
```

关键判读:

```text
如果 final F1 > baseline head F1，且 residual_abs_logit 不过大，
说明 residual branch 可能真的补充了 baseline。

说明 residual branch 主要改变了工作点，而不是提升决策质量。
```

## 7. 预期结果

最理想:

```text
F1 > 0.8596
Recall >= 0.8700
Pre >= 0.8600
```

可以接受:

```text
F1 不低于 002002002，
Recall 明显高于 002002002，
```

不接受:

```text
```

## 8. 后续分析计划

完成训练后，应新增 `002002003010001` 分析:

1. 010 vs 002002002 paired bootstrap;
2. 010 vs 009003 paired bootstrap;
3. baseline head vs final head 的样本级修正分析;
4. residual_logit 在 correct/incorrect、false negative/false positive 上的分布;
5. router 权重和 residual 大小的相关分析;
6. 是否满足主成功标准。
