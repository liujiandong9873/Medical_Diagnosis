# 002002003001：Image + Predicted Contour Feature Fusion

创建日期：2026-07-02

## 实验目的

`002002003001` 是 `002002003` 轮廓融合路线中的第一个子实验，用于回答：

```text
模型预测出的 y/e 轮廓概率图本身是否包含可用于腺样体肥大分类的结构信息？
```

它和 `002002002` 的区别是：

```text
002002002：轮廓预测只作为辅助监督，分类主要来自图像编码器特征。
002002003001：轮廓预测不仅作为辅助监督，还会被编码成 contour feature，并与 image feature 融合后分类。
```

## 公平性原则

测试阶段仍然只输入原始 X-ray 图像。

人工 `y/e` 轮廓只用于训练阶段监督，不作为测试阶段输入。测试时的 contour feature 来自模型自己预测的轮廓概率图。

## 复用数据

本实验复用 `002002002` 已准备好的数据和轮廓目标：

```text
../002002002_image_contour_auxiliary_supervision/outputs/metadata.csv
../002002002_image_contour_auxiliary_supervision/outputs/folds.csv
../002002002_image_contour_auxiliary_supervision/outputs/outer_splits.csv
../002002002_image_contour_auxiliary_supervision/outputs/contour_targets_224_uint8.npz
```

这些文件会在运行脚本时复制到当前实验的 `outputs/` 下，保证本实验输出自洽。

## 模型结构

```text
Input X-ray image
Shared ResNet18 encoder
        +-- Image feature
        +-- Contour decoder
                -> predicted y/e contour probability map
                        -> contour feature encoder

Image feature + contour feature
Fusion classifier
normal / hypertrophy
```

## 损失函数

```text
L_total = L_cls + lambda_contour * L_contour
```

其中：

```text
L_cls = BCEWithLogitsLoss
L_contour = weighted contour BCE + Dice loss
```

默认参数：

```text
lambda_contour = 0.3
contour_pos_weight = 50.0
```

## 关键对比

| 对比 | 目的 |
| --- | --- |
| 002002001 vs 002002003001 | 判断轮廓预测与轮廓特征融合整体是否优于 image-only |
| 002002002 vs 002002003001 | 判断显式预测轮廓特征融合是否优于单纯轮廓辅助监督 |

## 预期结论

如果 `002002003001` 优于 `002002002`，可以说明：

```text
模型不仅因为轮廓辅助任务被正则化，
而且预测轮廓表征本身含有可用于分类的结构信息。
```

## 当前状态

```text
目录已建立。
训练脚本已准备。
数据输入复用 002002002。
prepare-only 检查已通过。
smoke test 已通过。
正式训练已完成。
```

## Smoke Test

已验证：

```text
image_shape = [4, 3, 224, 224]
mask_shape = [4, 2, 224, 224]
cls_logits_shape = [4]
contour_logits_shape = [4, 2, 224, 224]
contour_features_shape = [4, 128]
status = ok
```

## 建议正式训练命令

```text
python scripts/run_002002003001_contour_fusion.py --device cuda --epochs 80 --patience 10 --batch-size 24 --lr 1e-4 --weight-decay 1e-4 --lambda-contour 0.3 --allow-random-fallback
```

## 正式训练结果

训练完成时间：2026-07-02

配置：

```text
5-fold StratifiedGroupKFold
ResNet18 pretrained backbone
contour_feature_dim = 128
fusion_hidden_dim = 256
lambda_contour = 0.3
contour_pos_weight = 50.0
batch_size = 24
epochs = 80
patience = 10
device = cuda
```

交叉验证结果：

| 指标 | Mean | Std |
| --- | --- | --- |
| Accuracy | 0.8468 | 0.0327 |
| Precision | 0.8651 | 0.0831 |
| Recall | 0.8451 | 0.0934 |
| F1 | 0.8484 | 0.0392 |
| Mean Dice | 0.0585 | 0.0486 |
| y Dice | 0.0587 | 0.0339 |
| e Dice | 0.0584 | 0.0658 |
| Mean IoU | 0.0308 | 0.0263 |

逐折最佳 epoch：

| Fold | Best Epoch | F1 | Mean Dice |
| --- | --- | --- | --- |
| 0 | 13 | 0.8489 | 0.0289 |
| 1 | 12 | 0.7967 | 0.0237 |
| 2 | 8 | 0.8271 | 0.0235 |
| 3 | 23 | 0.8707 | 0.0841 |
| 4 | 32 | 0.8986 | 0.1325 |

## 与前序实验对比

| 实验 | 模型 | F1 | Mean Dice |
| --- | --- | --- | --- |
| 002002001 | matched image-only ResNet18 | 0.8321 | 不适用 |
| 002002002 | contour auxiliary ResNet18 | 0.8596 | 0.1720 |
| 002002003001 | predicted contour feature fusion | 0.8484 | 0.0585 |

阶段解释：

```text
002002003001 相比 matched image-only baseline 有小幅提升，
但没有超过 002002002 contour auxiliary supervision。
```

可能原因：

1. 当前显式融合使用的是模型预测轮廓概率图，早期预测轮廓较噪，融合后可能引入不稳定结构信息。
2. 002002003001 的 contour Dice 明显低于 002002002，说明该结构下 contour decoder 学到的细线轮廓质量不足。
3. fusion classifier 增加了模型自由度，在当前数据量下可能更容易过拟合验证集。
4. 当前 contour feature encoder 只编码预测概率图，尚未加入更强的结构约束、中心线距离损失或分形尺度约束。

当前结论：

```text
轮廓辅助监督本身是有效的；
但当前版本的显式预测轮廓特征融合尚未证明优于单纯辅助监督。
后续如果继续做 contour fusion，应改进轮廓质量或加入更稳定的结构/分形约束。
```
