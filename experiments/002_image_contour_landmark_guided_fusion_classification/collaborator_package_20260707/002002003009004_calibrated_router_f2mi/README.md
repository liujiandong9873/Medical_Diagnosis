创建日期: 2026-07-06

## 实验定位

本实验是 `002002003009003_recall_preserving_adaptive_f2mi` 之后的架构优化实验。

前序结论:

- `009002` 激活了 sample-adaptive router，但 fixed 0.5 阈值下 recall/F1 偏低。
- `009003` 通过移除 router 正则并用 `val_f1` 选择 checkpoint，修复了 recall/F1。

因此 `009004` 的目标不是继续追求更高 recall，而是:

```text
保留 009003 的 recall/F1，
```

## 实验问题

核心问题:

```text
在 sample-adaptive F2MI 框架下，
能否通过 calibrated-router ranking objective 和 contour-aware calibrated routing，
同时保持排序能力、固定阈值 F1 和较高 recall?
```

## 数据与公平性原则

数据仍使用实验 002 的 matched ZYQ subset:

- 样本数: 667 rows;
- unique sample_id: 663;
- normal: 324;
- hypertrophy: 343;
- 交叉验证: 5-fold StratifiedGroupKFold;
- 推理阶段输入: raw X-ray image only;
- 训练阶段监督: contour mask、128-point y/e contour、landmark、morphology、fractal token consistency。

公平性原则不变:

```text
人工 contour、landmark、line、A/N 等标注不能作为测试时直接输入。
它们只用于训练监督与机制分析。
```

## 基础模型

009004 继承 009003 的主体:

```text
X-ray image
  -> ResNet18 image encoder
  -> contour mask head
  -> contour points / landmark / morphology prediction heads
  -> fractal token encoder
  -> fractional contour operator
  -> sample-adaptive multimodal router
  -> classifier
```

保留 009002/009003 的 sample-adaptive router input:

```text
router input =
  fractal_pooled
  + image attended summary
  + contour attended summary
  + landmark attended summary
  + morphology attended summary
  + cross-modal summary std
```

## 关键改动

### 1. Joint Monitor Checkpoint



009004 使用 joint monitor:

```text
val_score =
  + 0.30 * val_f1
  + 0.20 * val_recall
```

目的:

- 保持 fixed 0.5 threshold 下的 F1;
- 避免 recall 再次过低。

### 2. Weak Pairwise Ranking Loss


```text
rank_loss = mean(softplus(-(positive_logit - negative_logit)))
```

只在 batch 内同时有阳性和阴性样本时计算。

默认权重:

```text
lambda_rank = 0.02
```

作用:

- BCE 负责分类边界;
- ranking loss 负责阳性样本整体排序高于阴性样本;
- joint monitor 负责模型选择。

### 3. Weak Contour-Aware Router Preservation

009003 的错误分析提示:

```text
true positive 样本 contour weight 更高;
错误样本更 image-heavy。
```

因此 009004 加入非常轻的 contour floor regularization:

```text
router_contour_floor_loss =
  mean(max(0, contour_min - router_contour)^2)
```

默认:

```text
contour_min = 0.10
lambda_contour_floor = 0.001
```

注意:

```text
这不是回到 009002 的强制 router diversity/entropy。
它只是避免 contour 分支在训练中完全被 image 分支吞掉。
```

## 损失函数

基础损失继承 009003:

```text
base_loss =
  cls_loss
  + lambda_mask * mask_loss
  + lambda_points * points_loss
  + lambda_landmark * landmark_loss
  + lambda_morphology * morphology_loss
  + lambda_fractal * fractal_loss
```

009004 总损失:

```text
total_loss =
  base_loss
  + lambda_rank * rank_loss
  + lambda_contour_floor * contour_floor_loss
```

默认不使用:

```text
lambda_router_div = 0
lambda_router_entropy = 0
```

router diversity / entropy 只作为监测指标记录。

## 默认训练参数

继承 009003:

- epochs: 80
- patience: 10
- batch size: 12
- lr: 1e-4
- weight decay: 1e-4
- image size: 224
- d_model: 128
- n_heads: 4
- dropout: 0.2
- pretrained ResNet18: true
- AMP: true

009004 新参数:

```text
monitor = val_joint_score
monitor_f1_weight = 0.30
monitor_recall_weight = 0.20
lambda_rank = 0.02
lambda_contour_floor = 0.001
contour_router_min = 0.10
```

## 评价指标

主要分类指标:

- Acc
- Pre
- Recall
- F1

机制指标:

- router_image / router_contour / router_landmark / router_morphology;
- router branch std;
- router entropy;
- rank loss;
- contour floor loss;
- false negative / false positive decision counts。

## 成功判据

最低成功标准:

```text
F1 >= 0.860
Recall >= 0.850
Pre >= 0.860
```

更理想目标:

```text
F1 接近或超过 009003;
Recall 保持在 0.85 以上;
Precision 高于 009003。
```

## 对比对象

训练完成后必须与以下模型比较:

- `002002002 contour auxiliary`
- `002002003009 original F2MI`
- `002002003009002 sample-adaptive router F2MI`
- `002002003009003 recall-preserving adaptive F2MI`

统计分析编号建议:

```text
002002003009004001 paired statistical and error analysis
```

分析内容:

- pooled test metrics;
- fold-mean metrics;
- paired bootstrap;
- decision counts;
- false negative / false positive cases;
- router/error association;
- threshold tuning;

## 预期解释

如果 009004 成功，论文中可以表述为:

```text
The calibrated sample-adaptive F2MI preserves the high-recall behavior of the recall-preserving variant,
while recovering ranking discrimination and reducing false-positive inflation through weak contour-aware routing and pairwise ranking supervision.
```

中文表述:

```text
校准型样本自适应 F2MI 在保持高召回的同时，
通过弱 contour-aware routing 和 pairwise ranking supervision 恢复了排序判别能力，
减少了 009003 中为了提高召回而带来的 false positive 增加。
```

## 风险与备选方案

可能失败模式:

2. contour floor 过强，router 变成非自适应。
3. ranking loss 权重过高，概率校准变差。

备选处理:

- 如果 precision 仍低，增加 `lambda_rank` 到 0.03;
- 如果 router 变僵硬，降低 `lambda_contour_floor` 到 0.0005;
