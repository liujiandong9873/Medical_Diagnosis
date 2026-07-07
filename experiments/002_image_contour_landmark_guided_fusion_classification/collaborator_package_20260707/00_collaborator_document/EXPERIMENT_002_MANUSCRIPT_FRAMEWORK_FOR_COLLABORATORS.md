# 实验002论文协作者总框架

更新日期：2026-07-07

## 0. 执行摘要

本文件用于把当前 `Experiment 002` 整理成可交给合作者阅读和讨论的论文框架。它回答几个核心问题：

```text
1. 我们研究什么问题？
2. 我们使用什么数据？
3. proposed model 是什么？
4. baseline 和内部强对照是什么？
5. 各个模型的 Acc / Pre / Recall / F1 表现如何？
6. 当前结果支持什么论文论点？
7. 当前结果不支持什么过强结论？
8. 如果目标是 SCI 一区，还需要补什么？
```

从当前版本开始，实验002的论文汇报统一只使用四个分类指标：

```text
Acc
Pre
Recall
F1
```

当前最重要的结论是：

```text
Experiment 002 已经形成一条较完整的内部实验证据链。
当前 proposed model 建议定义为 direct calibrated F2MI。
该模型在当前内部候选模型中取得了最好的四指标综合表现：
Acc 0.9010，Pre 0.9062，Recall 0.9009，F1 0.9035。
但是 seed robustness 显示最强结果存在 seed sensitivity，因此不能写成完全稳定的 0.90+ F1 模型。
```

建议论文主张：

```text
Direct calibrated fractal-fractional multimodal interaction achieved the best observed four-metric classification profile among internal candidates, while independent seed repeats preserved a modest F1 trend over the contour-auxiliary baseline but showed that the magnitude of the best result was seed-sensitive.
```

不建议论文主张：

```text
The proposed model robustly outperformed all methods on every possible endpoint.
The proposed model is fully seed-stable at F1 > 0.90.
The proposed model is already clinically validated.
```

## 1. 研究背景

腺样体肥大是儿童鼻咽气道阻塞、睡眠呼吸障碍和耳鼻喉相关症状的重要影像学评估对象。临床上常使用侧位 X-ray 图像及相关形态测量，例如 A/N ratio、PAS 或类似距离比例，来判断肥大程度。

传统临床测量存在几个问题：

```text
1. 需要人工定位关键点、参考线和解剖边界。
2. 不同标注者之间可能存在差异。
3. 单一比例指标难以完整表达腺样体轮廓、气道形态和局部复杂结构。
4. 如果只依赖 A/N 或 PAS，模型结构创新空间会被临床强变量压缩。
```

因此，本研究不应写成“替代 A/N 或 PAS 的简单分类器”，而应写成：

```text
利用训练阶段的解剖结构监督，让模型从原始侧位 X-ray 图像中学习轮廓、关键点、形态和分形/分数阶几何表征，并验证这些结构表征能否改善 Acc、Pre、Recall 和 F1。
```

## 2. 研究问题

核心研究问题建议写成：

```text
Can anatomy-guided fractal-fractional multimodal interaction improve adenoid hypertrophy classification from lateral X-ray images when only the raw image is used at test time?
```

拆解为四个子问题：

```text
Q1. 训练阶段加入 y/e 轮廓监督是否优于 matched image-only baseline？
Q2. 简单融合预测轮廓特征是否足够？
Q3. fractal / fractional contour modeling 是否比普通结构融合更有方法学价值？
Q4. direct calibrated F2MI 是否能在 Acc、Pre、Recall、F1 上优于内部候选模型？
```

## 3. 数据集和评价协议

当前 Experiment 002 使用 ZYQ 数据集中的 image-contour matched 子集。

| Item | Value |
|---|---:|
| Matched samples | 667 |
| Normal | 324 |
| Hypertrophy | 343 |
| Unique sample_id | 663 |
| Cross validation | 5-fold StratifiedGroupKFold |
| Test-time input | raw lateral X-ray image only |
| Training supervision | classification label, y/e contour mask, 128-point contour, landmarks/morphology where used |

论文中必须强调公平性原则：

```text
人工轮廓、关键点、参考线、A/N、PAS 等结构信息只用于训练监督、模型设计或分析。
测试阶段 proposed model 不输入人工轮廓、人工关键点或人工临床测量。
```

## 4. Proposed Model

论文中的 proposed model 建议定义为：

```text
Direct calibrated F2MI model
```

完整名称：

```text
Direct calibrated fractal-fractional multimodal interaction model
```

实验编号：

```text
002002003012 direct F2MI calibrated model
```

来源：

```text
002002003010003 final module ablation
variant = no_residual
seed = 20260704
post-hoc validation-fitted Platt calibration
```

模型组件：

```text
image encoder
predicted contour branch
128-point contour representation
fractal token guidance
fractional contour operator
landmark/morphology feature integration
multimodal interaction router
direct F2MI classification logit
validation-fitted Platt calibration
```

它不是：

```text
baseline-preserving residual F2MI
```

原因是最终模块消融显示，`no_residual_platt` 在当前四个论文指标上比 `full010_platt` 更适合作为最终候选模型。

## 5. Baseline 和内部对照

主 baseline：

```text
002002002 contour auxiliary ResNet18
```

原因：

```text
1. 它是 matched 667 samples 上的强 baseline。
2. 它已经使用训练阶段 y/e contour auxiliary supervision。
3. 它比普通 image-only baseline 更公平、更强。
```

Image-only baseline：

```text
002002001 matched image-only ResNet18
```

内部强对照包括：

```text
002002002 contour auxiliary baseline
002002003009003 recall-preserving adaptive F2MI
002002003009004 calibrated-router F2MI
002002003010 residual F2MI
002002003010002 full010 Platt
002002003011 continued-training control
002002003012 direct calibrated F2MI
```

外部 literature comparator 仍需补充：

```text
recent deep learning methods for adenoid hypertrophy or lateral X-ray classification
medical image CNN/Transformer baselines if no direct adenoid comparator exists
clinical measurement baselines such as A/N ratio and PAS
```

## 6. 各模型 Performance 总表

本节补充各个主要模型的 performance 指标。所有表格均只使用：

```text
Acc
Pre
Recall
F1
```

### 6.1 主开发路线

下表是实验002主要开发路线的 5-fold StratifiedGroupKFold 结果，格式为 `mean +/- sd`。这一表适合用来说明模型路线如何逐步演化，但不应和后面的 pooled final-candidate 表混成同一个统计口径。

| Experiment | Model | Role | Acc | Pre | Recall | F1 | Manuscript use |
|---|---|---|---:|---:|---:|---:|---|
| 002002001 | matched image-only ResNet18 | image-only baseline | 0.8378 +/- 0.0481 | 0.8818 +/- 0.0518 | 0.7997 +/- 0.1227 | 0.8321 +/- 0.0638 | main baseline |
| 002002002 | contour auxiliary ResNet18 | main strong comparator | 0.8593 +/- 0.0250 | 0.8845 +/- 0.0363 | 0.8383 +/- 0.0451 | 0.8596 +/- 0.0226 | main comparator |
| 002002003001 | direct predicted contour feature fusion | direct fusion control | 0.8468 +/- 0.0327 | 0.8651 +/- 0.0831 | 0.8451 +/- 0.0934 | 0.8484 +/- 0.0392 | weak fusion control |
| 002002003002 | raw F2CG-Net | gated fractal-fractional contour fusion | 0.8379 +/- 0.0325 | 0.9147 +/- 0.0484 | 0.7584 +/- 0.1163 | 0.8230 +/- 0.0541 | historical fractal-fractional model |
| 002002003009 | original F2MI-Transformer | fractal-guided multimodal interaction | 0.8574 +/- 0.0295 | 0.8717 +/- 0.0528 | 0.8516 +/- 0.0783 | 0.8584 +/- 0.0352 | F2MI baseline |
| 002002003009002 | sample-adaptive router F2MI | sample-adaptive routing repair | 0.8474 +/- 0.0271 | 0.8911 +/- 0.0530 | 0.8070 +/- 0.0936 | 0.8424 +/- 0.0373 | mechanism repair, not final |
| 002002003009003 | recall-preserving adaptive F2MI | recall-oriented candidate | 0.8573 +/- 0.0264 | 0.8538 +/- 0.0309 | 0.8700 +/- 0.0599 | 0.8609 +/- 0.0329 | candidate route |
| 002002003009004 | calibrated-router F2MI | high-recall route | 0.8174 +/- 0.0790 | 0.7786 +/- 0.1192 | 0.9277 +/- 0.0545 | 0.8410 +/- 0.0650 | retired / supplementary |

### 6.2 F2CG 模块消融

下表是 F2CG 模块消融的 5-fold 结果，主要用于说明 fractal / fractional 模块分别带来的结构信息贡献。

| Variant | Gated | Fractal | Fractional | Acc | Pre | Recall | F1 | Interpretation |
|---|---|---|---|---:|---:|---:|---:|---|
| gated-only | yes | no | no | 0.8500 +/- 0.0600 | 0.8991 +/- 0.0488 | 0.8077 +/- 0.1682 | 0.8398 +/- 0.0805 | precision improves, recall unstable |
| fractal-only | no | yes | no | 0.8438 +/- 0.0270 | 0.8981 +/- 0.0537 | 0.7917 +/- 0.1150 | 0.8348 +/- 0.0489 | fractal signal exists but F1 is limited |
| fractional-only | no | no | yes | 0.8592 +/- 0.0146 | 0.8811 +/- 0.0361 | 0.8418 +/- 0.0549 | 0.8593 +/- 0.0204 | strongest default-threshold ablation |
| gated + fractal | yes | yes | no | 0.8454 +/- 0.0325 | 0.8876 +/- 0.0687 | 0.8133 +/- 0.1160 | 0.8416 +/- 0.0406 | fractal adds structure but does not lift F1 |
| gated + fractional | yes | no | yes | 0.8405 +/- 0.0497 | 0.8562 +/- 0.0821 | 0.8355 +/- 0.0903 | 0.8412 +/- 0.0583 | weaker than fractional-only |
| fractal + fractional without gates | no | yes | yes | 0.8504 +/- 0.0305 | 0.8754 +/- 0.0465 | 0.8326 +/- 0.0789 | 0.8502 +/- 0.0331 | modules carry signal even without gates |

### 6.3 Final Candidate 和内部强对照

下表是 final proposed candidate 与主要内部 comparators 的 pooled 667-sample 对比。当前论文主模型应以这一表为核心性能表。

| Model | n | Acc | Pre | Recall | F1 | Rank note |
|---|---:|---:|---:|---:|---:|---|
| direct_f2mi_calibrated | 667 | 0.9010 | 0.9062 | 0.9009 | 0.9035 | best Acc and F1 |
| baseline002002002 | 667 | 0.8591 | 0.8831 | 0.8367 | 0.8593 | main contour-auxiliary baseline |
| full010_platt | 667 | 0.8786 | 0.8722 | 0.8950 | 0.8835 | previous residual-fusion proposal |
| control011 | 667 | 0.8651 | 0.8799 | 0.8542 | 0.8669 | continued-training control |
| no_fractal_platt | 667 | 0.8726 | 0.8707 | 0.8834 | 0.8770 | direct F2MI without fractal token |
| no_fractional_platt | 667 | 0.8741 | 0.8798 | 0.8746 | 0.8772 | direct F2MI without fractional operator |
| recall009003 | 667 | 0.8576 | 0.8543 | 0.8717 | 0.8629 | recall-preserving candidate |
| calibrated_router009004 | 667 | 0.8171 | 0.7650 | 0.9300 | 0.8395 | high recall but weak precision/F1 |
| residual010_raw | 667 | 0.8726 | 0.8468 | 0.9184 | 0.8811 | high recall raw residual route |

相对主 baseline `baseline002002002`，final proposed model 的提升为：

| Metric | direct_f2mi_calibrated - baseline002002002 |
|---|---:|
| Acc | +0.0420 |
| Pre | +0.0231 |
| Recall | +0.0641 |
| F1 | +0.0442 |

### 6.4 Seed Robustness

下表用于回答 direct F2MI 的最强结果是否稳定。结论是：两个 repeat seed 均保留了轻度优于 baseline 的 F1，但没有复现 original candidate 的 0.9035 F1。

| Model | n | Acc | Pre | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| direct_f2mi_calibrated_original | 667 | 0.9010 | 0.9062 | 0.9009 | 0.9035 |
| baseline002002002 | 667 | 0.8591 | 0.8831 | 0.8367 | 0.8593 |
| seed_20260711_platt | 667 | 0.8651 | 0.8604 | 0.8805 | 0.8703 |
| seed_20260712_platt | 667 | 0.8726 | 0.8909 | 0.8571 | 0.8737 |

### 6.5 如何阅读这些表

```text
1. 002002001 -> 002002002：说明训练阶段 contour supervision 有价值。
2. 002002003001：说明简单拼接 predicted contour feature 不够。
3. 002002003002 / 002002003005：说明 fractal / fractional 模块有结构信息，但早期 fusion 方式不足。
4. 002002003009 系列：说明 F2MI 路线可以提高 recall，但需要更稳的 final decision architecture。
5. 002002003010 / 011 / 012：说明 direct calibrated F2MI 是目前最适合作为 proposed model 的路线。
6. seed robustness：说明 final result 有 promising performance，但不能过度声称完全 seed-stable。
```

## 7. 实验逻辑链

建议论文结果部分按照以下证据链组织。

Step 1：Contour supervision helps

```text
002002001 matched image-only baseline
vs
002002002 contour auxiliary baseline
```

结论：

```text
训练阶段加入 y/e contour auxiliary supervision 可以提升四指标分类表现。
```

Step 2：Direct contour feature fusion is not enough

```text
002002002 contour auxiliary baseline
vs
002002003001 direct predicted contour feature fusion
```

结论：

```text
简单拼接预测 contour 特征不能稳定超过 contour auxiliary baseline。
```

Step 3：Fractal/fractional modeling is methodologically meaningful

证据来源：

```text
002002003002 F2CG-Net
002002003003 probability calibration
002002003005 F2CG module ablation
002002003010003 final module ablation
```

结论：

```text
fractal/fractional contour representation 对四指标分类表现具有方法学贡献，但需要配合合适的 calibration 和 fusion 设计。
```

Step 4：Direct calibrated F2MI gives best observed four-metric profile

核心结果：

| Model | Acc | Pre | Recall | F1 |
|---|---:|---:|---:|---:|
| direct_f2mi_calibrated | 0.9010 | 0.9062 | 0.9009 | 0.9035 |
| baseline002002002 | 0.8591 | 0.8831 | 0.8367 | 0.8593 |

相对 baseline：

| Metric | Improvement |
|---|---:|
| Acc | +0.0420 |
| Pre | +0.0231 |
| Recall | +0.0641 |
| F1 | +0.0442 |

Bootstrap：

```text
Acc    p = 0.000
Pre    p = 0.134
Recall p = 0.000
F1     p = 0.000
```

Step 5：Continued-training control

| Model | Acc | Pre | Recall | F1 |
|---|---:|---:|---:|---:|
| control011 | 0.8651 | 0.8799 | 0.8542 | 0.8669 |
| direct_f2mi_calibrated | 0.9010 | 0.9062 | 0.9009 | 0.9035 |

结论：

```text
direct F2MI 的最优 observed performance 不能简单解释为 longer training 或 checkpoint difference。
```

Step 6：Seed robustness boundary

| Model | Acc | Pre | Recall | F1 |
|---|---:|---:|---:|---:|
| direct_f2mi_calibrated_original | 0.9010 | 0.9062 | 0.9009 | 0.9035 |
| baseline002002002 | 0.8591 | 0.8831 | 0.8367 | 0.8593 |
| seed_20260711_platt | 0.8651 | 0.8604 | 0.8805 | 0.8703 |
| seed_20260712_platt | 0.8726 | 0.8909 | 0.8571 | 0.8737 |

结论：

```text
两个独立 seed repeat 都保留了相对 baseline 的轻度 F1 提升。
但两个 repeat 都没有复现 original direct F2MI 的 0.9035 F1。
因此 original direct F2MI 不能写成完全 seed-stable 的 0.90+ F1 结果。
```

## 8. 当前结果支持的论点

当前结果可以支持：

```text
1. Contour auxiliary supervision improves the matched image-only baseline.
2. Simple direct contour feature fusion is insufficient.
3. Fractal/fractional contour modeling and calibration are meaningful.
4. Direct calibrated F2MI achieved the best observed four-metric profile among internal candidates.
5. Continued-training control did not reproduce the full observed gain.
6. Seed repeats preserve a modest F1 trend, but the strongest gain is seed-sensitive.
```

当前结果不支持：

```text
1. Proposed model robustly outperforms all methods on every possible endpoint.
2. Proposed model is fully seed-stable at F1 > 0.90.
3. Proposed model is already clinically validated.
4. Proposed model is superior to A/N or PAS unless clinical baselines are formally analyzed.
5. Proposed model is superior to external literature comparators until the literature table is completed.
```

## 9. 建议论文结构

Abstract 逻辑：

```text
Background:
Manual measurements such as A/N ratio and PAS are useful but depend on anatomical landmark interpretation.

Objective:
To develop and evaluate a structure-guided fractal-fractional multimodal interaction model for adenoid hypertrophy classification from lateral X-ray images.

Methods:
Use 667 matched ZYQ lateral X-ray samples with y/e contour annotations. Train image-only, contour auxiliary, F2CG, F2MI, residual F2MI, direct calibrated F2MI, ablations, controls, and seed repeats. Evaluate with 5-fold StratifiedGroupKFold and paired clustered bootstrap.

Results:
The best observed direct calibrated F2MI achieved Acc 0.9010, Pre 0.9062, Recall 0.9009, and F1 0.9035, outperforming the contour-auxiliary baseline on three of the four paper-facing metrics. Seed repeats preserved modest F1 trends but did not reproduce the full magnitude of the best result.

Conclusion:
Fractal-fractional multimodal interaction provides promising structure-guided decision evidence, but external validation and robustness improvements are required before strong clinical claims.
```

Methods 建议小节：

```text
Dataset and annotations
Task definition
Preprocessing and 128-point contour resampling
Image-only and contour auxiliary baselines
Fractal token representation
Fractional contour operator
Multimodal interaction module
Direct calibrated F2MI classifier
Calibration strategy
Statistical analysis
```

Results 建议顺序：

```text
1. Dataset and cohort summary
2. Baseline comparison
3. Main internal comparator table
4. Paired bootstrap analysis
5. Ablation analysis
6. Seed robustness analysis
7. Error case review without adding extra headline metrics
```

Discussion 建议包含：

```text
1. Why contour supervision helps.
2. Why direct contour concatenation is insufficient.
3. Why fractal/fractional interaction may improve the decision boundary.
4. Why seed sensitivity matters and how to interpret it.
5. Clinical meaning and limitations.
```

## 10. 建议表格和图

主文表格：

```text
Table 1. Dataset and annotation summary
Table 2. Main baseline and internal comparator performance using Acc, Pre, Recall, F1
Table 3. Paired bootstrap comparison against contour auxiliary baseline using Acc, Pre, Recall, F1
Table 4. Module ablation using Acc, Pre, Recall, F1
Table 5. Seed robustness using Acc, Pre, Recall, F1
```

补充表：

```text
Supplementary Table S1. All image-only baselines
Supplementary Table S2. F2CG calibration results under four-metric reporting
Supplementary Table S3. Route status by experiment ID
```

建议图：

```text
Figure 1. Overall pipeline: raw X-ray -> contour prediction -> fractal/fractional contour representation -> F2MI -> calibrated classification
Figure 2. Model architecture of direct calibrated F2MI
Figure 3. Bar plot of Acc, Pre, Recall, and F1
Figure 4. Ablation plot for fractal/fractional/direct residual variants
Figure 5. Seed robustness plot
Figure 6. Example correct/incorrect cases with predicted contour and decision scores
```

## 11. SCI 一区定位

当前支持水平：

```text
SCI paper draft: yes
Q1-oriented manuscript: possible
Stable Q1 evidence package: not yet fully sufficient
```

优点：

```text
实验链条完整。
有强 baseline。
有内部候选模型对照。
有 bootstrap。
有 continued-training control。
有 ablation。
有 seed robustness。
```

不足：

```text
没有外部验证。
external literature comparator 尚未系统整理。
proposed model 最强结果 seed-sensitive。
clinical measurement baseline 尚未正式纳入主结果。
```

最佳当前论文策略：

```text
honest strong-method paper
```

而不是：

```text
overclaiming clinical replacement paper
```

如果要增强一区说服力，优先补充：

```text
1. External validation.
2. Clinical baseline: A/N, PAS, distance-ratio model, clinical-only logistic regression.
3. Clinical + model combined analysis.
4. Multi-seed ensemble or validation-selected seed aggregation.
5. Error case review with clinical interpretation.
```

## 12. 协作者任务列表

Clinical collaborators：

```text
1. 确认标签定义和肥大/非肥大标准。
2. 确认 A/N、PAS 或其他距离比例指标应如何计算。
3. 评估错误病例是否具有临床边界性。
4. 决定论文中应强调 screening、辅助诊断还是自动测量。
```

Machine learning collaborators：

```text
1. 审查 direct calibrated F2MI 的模型定义是否足够 solid。
2. 审查 seed sensitivity 是否需要用 ensemble 解决。
3. 检查 calibration 是否存在 data leakage。
4. 完善 external comparator。
5. 优化模型图和公式表达。
```

Statistical collaborators：

```text
1. 审查 paired clustered bootstrap 是否合适。
2. 建议多重比较校正策略。
3. 帮助设计 external validation 或 locked holdout analysis。
4. 帮助定义主终点和次终点，但论文性能汇报保持四指标口径。
```

Writing collaborators：

```text
1. 将 contribution 写得强但不过度。
2. 将 seed sensitivity 写成透明稳健性分析，而不是负面失败。
3. 整理 related work。
4. 对齐目标期刊风格。
```

## 13. 推荐论文叙事

最推荐的论文叙事是：

```text
This study does not simply propose another image classifier.
It studies whether anatomical contour supervision, fractal/fractional contour geometry, and multimodal interaction can improve adenoid hypertrophy classification from raw lateral X-ray images.

The direct calibrated F2MI model achieved the best observed four-metric profile in the internal comparison.

However, independent seed repeats showed that the magnitude of the best observed result was seed-sensitive. Therefore, the model should be interpreted as a promising structure-guided fractal-fractional framework rather than a fully locked clinical system.
```

## 14. 关键文件

Main experiment folder:

```text
experiments/002_image_contour_landmark_guided_fusion_classification/
```

Direct F2MI candidate:

```text
002002_image_contour_prediction/002002003012_direct_f2mi_calibrated_model/
```

Main direct F2MI analysis:

```text
002002_image_contour_prediction/002002003012_direct_f2mi_calibrated_model/analysis/012001_out/analysis_report.md
```

Seed robustness analysis:

```text
002002_image_contour_prediction/002002003012_direct_f2mi_calibrated_model/analysis/012002_out/analysis_report.md
```

Paper result tables:

```text
paper_tables/table_1_main_cv_results.csv
paper_tables/table_5_module_ablation_cv_results.csv
paper_tables/table_7_direct_f2mi_internal_sota_comparison.csv
paper_tables/table_8_direct_f2mi_paired_bootstrap_vs_sota.csv
paper_tables/table_10_direct_f2mi_seed_robustness_metrics.csv
paper_tables/table_11_direct_f2mi_seed_robustness_bootstrap.csv
```

## 15. 给协作者的最短结论

```text
我们已经有一套完整的内部实验链条，支持 direct calibrated F2MI 是当前 best observed four-metric model。
它相对强 contour auxiliary baseline 明显提升 Acc、Recall 和 F1，Pre 也略有提升。
但两个独立 seed repeat 没有复现 0.9035 F1，只保留了轻度优于 baseline 的趋势。
所以这篇文章可以写，但需要透明报告 robustness，并最好补 external validation 或 clinical baseline 对照来增强 Q1 说服力。
```
