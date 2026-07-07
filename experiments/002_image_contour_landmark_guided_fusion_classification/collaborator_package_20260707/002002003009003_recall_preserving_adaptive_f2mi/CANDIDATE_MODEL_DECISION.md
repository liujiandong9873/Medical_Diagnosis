# 002002003009003 Candidate Model Decision

决策日期: 2026-07-06

## 当前决策

将 `002002003009003_recall_preserving_adaptive_f2mi` 定位为:

```text
recall-preserving sample-adaptive F2MI candidate
```

即:

```text
召回保持型样本自适应 F2MI 候选模型
```

同时，`002002003009004_calibrated_router_calibrated_router_f2mi` 不再作为主线继续优化。

## 决策依据

### 009003 的核心结果

| Model | Acc | Pre | Recall | F1 |
| --- | --- | --- | --- | --- |
| 002002002 contour auxiliary | 0.8593 | 0.8845 | 0.8383 | 0.8596 |
| 002002003009 original F2MI | 0.8574 | 0.8717 | 0.8516 | 0.8584 |
| 002002003009002 sample-adaptive router | 0.8474 | 0.8911 | 0.8070 | 0.8424 |
| 002002003009003 recall-preserving adaptive F2MI | 0.8573 | 0.8538 | 0.8700 | 0.8609 |

009003 的主要价值:

- 修复了 009002 的低 recall 问题;
- fixed 0.5 threshold 下 F1 回到 baseline 水平附近;
- recall 高于 002002002、原始 009 和 009002;
- sample-adaptive router 仍然存在 fold 内样本级变化。

009003 的限制:

- precision 有一定下降;
- paired bootstrap 未显示 009003 显著优于 002002002 或原始 009;
- 因此不能声称整体性能显著优于 baseline。

### 009004 的结果


实际结果:

| Model | Acc | Pre | Recall | F1 |
| --- | --- | --- | --- | --- |
| 009003 | 0.8573 | 0.8538 | 0.8700 | 0.8609 |
| 009004 | 0.8174 | 0.7786 | 0.9277 | 0.8410 |

009004 的失败模式:

```text
```

具体表现:

- recall 最高，但 precision 明显下降;
- F1 低于 009003;
- fold 间稳定性更差。

因此:

```text
不继续沿 009004 调参。
```

## 论文中的建议定位

009003 不适合作为“显著优于 baseline 的最终模型”来写。

更合适的表述是:

```text
009003 is a recall-preserving sample-adaptive F2MI variant.
It demonstrates that the proposed fractal/fractional multimodal interaction route can be adjusted toward a clinically safer high-recall operating point while maintaining F1 near the contour-auxiliary baseline.
```

中文表述:

```text
009003 是召回保持型样本自适应 F2MI 变体。
它证明了 fractal/fractional multimodal interaction 架构可以被调整到更偏临床筛查安全性的高召回工作点，同时保持接近 contour auxiliary baseline 的 F1。
```

## 允许声称的内容

可以较稳妥地声称:

- 009003 修复了 009002 的 low-recall 问题;
- 009003 在 fixed 0.5 threshold 下达到了更高 recall;
- 009003 的 F1 与 002002002 / 原始 009 基本持平;
- sample-adaptive router 机制仍然存在;

不建议声称:

- 009003 显著优于 002002002;
- 009003 显著优于原始 009;
- 009003 是整体最优模型;
- 009004 是成功模型。

## 下一步工作

建议下一步不再继续训练 009004。

优先做:

```text
整理 009003 作为 candidate model 的最终报告与论文材料。
```

具体包括:

1. 将 009003 写入实验 002 总结文件，标记为 recall-preserving candidate。
2. 整理一张总表，包含 002002002、009、009002、009003、009004。
3. 整理 paired bootstrap 结果，明确 009003 的优势和限制。
4. 整理 false negative / false positive 的样本级对比。
5. 整理 router mechanism 结果，说明 sample-adaptive routing 的存在和局限。
6. 准备论文表述时，避免使用 significant improvement 这类过强措辞。

## 当前路线状态

```text
Main candidate: 002002003009003 recall-preserving adaptive F2MI
Retired route: 002002003009004 calibrated-router F2MI
Next priority: candidate model reporting and manuscript framing
```
