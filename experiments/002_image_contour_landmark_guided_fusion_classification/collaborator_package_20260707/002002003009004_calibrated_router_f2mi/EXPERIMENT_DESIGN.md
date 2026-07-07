# 002002003009004 Implementation Design

## Source Script

建议从以下脚本复制:

```text
002002003009003_recall_preserving_adaptive_f2mi/scripts/run_009003.py
```

目标脚本:

```text
002002003009004_calibrated_router_calibrated_router_f2mi/scripts/run_009004.py
```

## Constants

需要修改:

```python
CHECKPOINT_DIR = MODEL_ROOT / "Generated_Models" / "002002003009004_calibrated_router_calibrated_router_f2mi"
EXPERIMENT_ID = "002002003009004"
EXPERIMENT_LABEL = "calibrated-router F2MI"
```

## RunConfig Additions

在 `RunConfig` 中增加:

```python
lambda_rank: float
lambda_contour_floor: float
contour_router_min: float
monitor_f1_weight: float
monitor_recall_weight: float
```

保留:

```python
lambda_router_div: float = 0.0
lambda_router_entropy: float = 0.0
```

## CLI Defaults

新增或修改默认参数:

```text
--monitor val_joint_score
--monitor-f1-weight 0.30
--monitor-recall-weight 0.20
--lambda-rank 0.02
--lambda-contour-floor 0.001
--contour-router-min 0.10
--lambda-router-div 0.0
--lambda-router-entropy 0.0
```

## Pairwise Ranking Loss

新增函数:

```python
def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positives = logits[labels > 0.5]
    negatives = logits[labels <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.new_tensor(0.0)
    diff = positives[:, None] - negatives[None, :]
    return F.softplus(-diff).mean()
```

设计解释:

- batch 内同时存在 positive / negative 时计算;
- 使用 logits 而不是 probabilities;
- 权重必须小，避免破坏 BCE 训练。

## Contour Floor Loss

在 `compute_losses` 中，从 `extras["router_weights"]` 取 contour branch:

```python
router_weights = extras["router_weights"]
router_contour = router_weights[:, 1]
contour_floor_loss = F.relu(config.contour_router_min - router_contour).pow(2).mean()
```

加入总损失:

```python
rank_loss = pairwise_rank_loss(logits, labels)

total = (
    cls_loss
    + config.lambda_mask * mask_loss
    + config.lambda_points * points_loss
    + config.lambda_landmark * landmark_loss
    + config.lambda_morphology * morphology_loss
    + config.lambda_fractal * fractal_loss
    + config.lambda_rank * rank_loss
    + config.lambda_contour_floor * contour_floor_loss
    + config.lambda_router_div * router_div_loss
    + config.lambda_router_entropy * router_entropy_loss
)
```

记录到 `loss_parts`:

```python
"rank_loss": rank_loss,
"contour_floor_loss": contour_floor_loss,
```

## Joint Monitor

在 `run_epoch` 返回 metrics 后，增加:

```python
metrics["joint_score"] = (
    + config.monitor_f1_weight * metrics["f1"]
    + config.monitor_recall_weight * metrics["recall"]
)
```

训练时 `row` 中会自动有:

```text
val_joint_score
test_joint_score
train_joint_score
```

默认:

```text
config.monitor = "val_joint_score"
```

训练日志建议打印:

```text
```

## Metrics Summary Additions

在 `summarize_metrics` 中增加:

```text
joint_score
rank_loss
contour_floor_loss
router_batch_std
router_entropy
router branch std
```

## Smoke Test Requirements

smoke test 输出应包含:

```text
rank_loss
contour_floor_loss
router_weights_mean
router_weights_std
router_entropy_mean
loss
cls_loss
```

## Expected Behavior

训练中预期:

- early epochs 可能 recall 波动较大;
- joint score 应避免只选 recall 极高但 precision 很差的点;
- router_contour 不应长期低于 0.10;
- router_image 可以仍为主导，但不应吞掉 contour 分支;

## Stop Criteria

如果训练完成后:

```text
```

则 009004 失败。

如果:

```text
F1 >= 0.860
Recall >= 0.850
Pre >= 0.860
```

则进入 `002002003009004001` 统计验证。

## Required Output Files

训练输出:

```text
outputs/cv_metrics_by_fold.csv
outputs/cv_predictions.csv
outputs/cv_summary.csv
outputs/training_history.csv
outputs/training_progress.md
outputs/run_summary.md
outputs/folds/f*/metrics.csv
outputs/folds/f*/predictions.csv
outputs/folds/f*/history.csv
```

结果解释:

```text
outputs/training_result_analysis.md
```

统计分析:

```text
analysis/009004001_out/
```
