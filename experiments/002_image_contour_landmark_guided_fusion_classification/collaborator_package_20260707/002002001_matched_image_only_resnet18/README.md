# 002002001：Matched Image-only Baseline

创建日期：2026-06-29

## 实验目的

`002002001` 是 `002002 Image + Contour Prediction` 实验组中的同条件纯图像分类基线。

本实验不引入新的结构模块，目的不是追求模型创新，而是为后续轮廓监督模型建立公平对照：

```text
在与 002002002/002002003001/002002003002 完全相同的数据、划分、backbone、训练策略和评价流程下，
只使用原始 X-ray 图像进行正常/肥大分类，能够达到什么性能？
```

后续如果加入轮廓监督或预测轮廓融合后性能发生变化，才能更可靠地归因于：

```text
轮廓监督
预测轮廓结构特征
多任务学习
```

而不是训练脚本、数据划分、学习率、checkpoint 选择或评价指标差异。

## 与实验002的关系

实验002的总体目标是比较：

```text
Image-only classification
vs
Image-based anatomy-structure-guided classification
```

`002002001` 对应其中的同条件 image-only 对照。

后续核心比较为：

```text
002002001 matched image-only baseline
002002002 image + contour auxiliary supervision
002002003001 image + predicted contour fusion
002002003002 F2CG gated fractal-fractional contour fusion
```

判断逻辑：

```text
如果 002002002 > 002002001：
说明轮廓预测辅助监督改善了图像表征。
```

```text
如果 002002003001 > 002002002：
说明直接预测轮廓编码特征在分类中提供了额外信息。
```

```text
如果 002002003002 > 002002002 且 002002003002 > 002002003001：
说明门控轮廓融合和分形/分数阶尺度建模比直接 contour concat 更有效。
```

## 输入与禁用信息

模型输入：

```text
Dataset/ZYQ_Dataset/images
```

分类标签：

```text
正常
肥大
```

禁止作为模型输入：

```text
人工 y/e 轮廓
b/Y/B 点线
A(mm)
N
A/N
PAS(mm)
```

人工轮廓和关键点可以用于样本筛选与后续监督任务构建，但本实验的模型训练和测试均不使用这些结构标注作为输入。

## 数据选择

为了保证与后续 `002002002/002002003001/002002003002` 完全可比，`002002001` 应使用后续轮廓监督实验同一批样本：

```text
1. 有明确正常/肥大分组
2. 有对应原始图像
3. 有可用 y contour 和 e contour 标注
```

这样可以避免出现：

```text
image-only baseline 使用全部图像样本，
而 contour model 只能使用有轮廓标注的子集
```

导致的不公平比较。

## 数据划分

建议从 `002002` 开始使用更严格的数据划分：

```text
StratifiedGroupKFold
group = sample_id
n_splits = 5
```

原因：

```text
002001 中发现极少数 sample_id 可能对应多张图像。
按 sample_id 分组划分可以避免同一病例/同一编号图像跨 train/test。
```

所有 `002002001-002002003002` 必须复用完全相同的 fold 划分。

## 模型设置

主模型：

```text
ResNet18
```

选择理由：

```text
1. 002001 中 ResNet18 表现强且稳定。
2. 模型轻量，适合作为后续多任务轮廓预测模型的共享 encoder。
3. 训练成本较低，便于快速进行结构模块消融。
```

可选补充模型：

```text
DenseNet121
```

但第一版建议只使用 ResNet18 跑通 `002002` 实验链条。

## 训练配置

建议初始配置：

```text
image_size = 224
optimizer = AdamW
lr = 1e-4
weight_decay = 1e-4
epochs = 80
patience = 10
loss = BCEWithLogitsLoss
AMP = yes
weighted sampler = no
pos_weight = yes
```

数据增强保持保守：

```text
small rotation, about +/-5 degrees
small translation / scale
brightness and contrast jitter
```

不建议使用：

```text
horizontal flip
large rotation
strong elastic deformation
random crop that may remove nasopharyngeal anatomy
```

## 输出文件

建议输出：

```text
metadata.csv
folds.csv
outer_splits.csv
cv_predictions.csv
cv_metrics_by_fold.csv
cv_summary.csv
training_history.csv
run_summary.md
```

## 评价指标

分类主指标：

```text
accuracy
precision
F1
loss
```

后续可选补充：

```text
Brier score
calibration curve
confusion matrix
```

## 成功标准

`002002001` 本身没有“必须超过 002001 最强模型”的要求。

它的成功标准是：

```text
1. 与后续轮廓监督模型使用相同样本和相同 fold。
2. 训练流程稳定，无明显类别塌缩。
4. 能作为 002002002/002002003 的公平直接对照。
```

## 预期解释

如果 `002002001` 明显低于 `002001` 最强 image-only baseline，可能原因包括：

```text
1. 使用了更严格的有轮廓标注子集。
2. 使用了 GroupKFold，避免了轻微样本泄漏。
3. 训练配置与 002001 不完全相同。
```

这并不一定是问题，因为 `002002001` 的核心作用是“同条件对照”，不是替代 `002001` 的总体 image-only 模型排名。

## 当前状态

```text
Step 1: 实验目的已明确
Step 2: 实验设计 README 已建立
Step 3: 等待创建运行脚本
Step 4: 等待运行实验
Step 5: 等待结果分析
```
## Run Completion Record

Completed date: 2026-07-02

Selected backbone from `002001`: `ResNet18`.

Selection basis:

```text
```

Implemented script:

```text
scripts/run_002002001_matched_resnet18.py
```

Matched subset:

```text
Total matched image-contour samples: 667
Unique sample_id count: 663
Normal samples: 324
Hypertrophy samples: 343
Required annotations: y/e linestrip contours
Splitter: StratifiedGroupKFold grouped by sample_id
Folds: 5
```

Training configuration:

```text
backbone = ResNet18
pretrained = ImageNet
image_size = 224
batch_size = 32
epochs = 80
patience = 10
optimizer = AdamW
lr = 1e-4
weight_decay = 1e-4
loss = BCEWithLogitsLoss with pos_weight
device = cuda
```

Output directory:

```text
outputs/
```

Generated files:

```text
metadata.csv
folds.csv
outer_splits.csv
cv_predictions.csv
cv_metrics_by_fold.csv
cv_summary.csv
training_history.csv
run_summary.md
```

Cross-validation test summary:

```text
accuracy         = 0.8378 +/- 0.0481
precision        = 0.8818 +/- 0.0518
F1               = 0.8321 +/- 0.0638
loss             = 0.4744 +/- 0.0778
```

Final status:

```text
Step 1: experiment purpose completed
Step 2: README completed
Step 3: run script completed
Step 4: training completed
Step 5: outputs and summary completed
```
