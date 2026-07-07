# 002002002: Image + Contour Auxiliary Supervision

Created date: 2026-07-02

## Purpose

`002002002` is the second sub-experiment in `002002 Image + Contour Prediction`.

The core question is:

```text
Compared with the matched image-only baseline, does adding y/e contour auxiliary supervision improve adenoid hypertrophy classification?
```

This experiment does not feed manual contours into the model at test time. Manual `y/e` contours are used only as training supervision.

The fair comparison is:

```text
002002001: image-only matched baseline
002002002: image classification + contour auxiliary supervision
```

## Baseline To Beat

The direct baseline is:

```text
002002001_matched_image_only_baseline
```

Baseline result:

```text
accuracy         = 0.8378 +/- 0.0481
F1               = 0.8321 +/- 0.0638
```

`002002002` must reuse the same matched samples and fold assignment from `002002001`.

## Data

Source dataset:

```text
Dataset/ZYQ_Dataset
```

Model input:

```text
X-ray image only
```

Training supervision:

```text
normal/hypertrophy classification label
y contour mask
e contour mask
```

Forbidden as model input:

```text
manual y/e contour coordinates
manual b/Y/B landmarks
A(mm), N, A/N, PAS(mm)
```

Manual contours are allowed only for constructing the auxiliary contour loss during training.

## Matched Subset

Reuse from `002002001`:

```text
Total matched image-contour samples: 667
Unique sample_id count: 663
Normal samples: 324
Hypertrophy samples: 343
Required annotations: y/e linestrip contours
Splitter: StratifiedGroupKFold grouped by sample_id
Folds: 5
```

The following files must be copied or reused:

```text
../002002001_matched_image_only_baseline/outputs/metadata.csv
../002002001_matched_image_only_baseline/outputs/folds.csv
../002002001_matched_image_only_baseline/outputs/outer_splits.csv
```

## Target Design

For the first version, use contour heatmap/mask supervision instead of direct 128-point coordinate regression.

Reason:

```text
Contour masks are more stable for CNN encoder-decoder training.
They support Dice/BCE losses naturally.
They allow visual QA and contour-quality metrics.
They are easier to fuse later in 002002003001 and 002002003002.
```

Target channels:

```text
channel 0: y contour mask
channel 1: e contour mask
```

Default target size:

```text
224 x 224
```

Default line width:

```text
3 pixels
```

## Model Design

First implementation should use:

```text
Shared ResNet18 encoder
  + classification head
  + lightweight contour decoder
```

Outputs:

```text
classification logit
2-channel contour logits
```

Loss:

```text
L_total = L_cls + lambda_contour * L_contour
```

Where:

```text
L_cls = BCEWithLogitsLoss for normal/hypertrophy classification
L_contour = BCEWithLogitsLoss + Dice loss for y/e contour masks
```

Initial recommendation:

```text
lambda_contour = 0.3
```

The value can be tuned later in an ablation experiment.

## Evaluation

Classification metrics:

```text
accuracy
precision
F1
loss
```

Contour auxiliary metrics:

```text
y Dice
e Dice
mean Dice
y IoU
e IoU
mean IoU
```

Primary conclusion should be based on classification improvement over `002002001`.

## Preparation Outputs

The preparation script should generate:

```text
metadata.csv
folds.csv
outer_splits.csv
contour_targets_224_uint8.npz
contour_target_audit.csv
contour_target_manifest.json
run_summary.md
previews/
```

## Current Status

```text
Step 1: experiment design README created
Step 2: contour target preparation script completed
Step 3: contour target generation completed
Step 4: training script completed and smoke-tested
Step 5: training completed; result interpretation pending
```

## Preparation Completion Record

Completed date: 2026-07-02

Preparation script:

```text
scripts/prepare_002002002_contour_targets.py
```

Preparation command:

```text
python scripts/prepare_002002002_contour_targets.py --target-size 224 --line-width 3 --preview-count 12
```

Generated target:

```text
contour_targets_224_uint8.npz
shape = [667, 2, 224, 224]
dtype = uint8
channels = y, e
```

Quality check:

```text
empty y masks = 0
empty e masks = 0
y mask pixels mean = 80.03
e mask pixels mean = 101.77
y length mean = 24.59
e length mean = 31.80
```

Visual QA:

```text
12 overlay previews generated in outputs/previews/
Red = y contour
Blue = e contour
```

Preparation status:

```text
002002002 is ready for formal 5-fold training.
```

## Training Script Completion Record

Completed date: 2026-07-02

Training script:

```text
scripts/run_002002002_contour_auxiliary.py
```

Model:

```text
Shared ResNet18 encoder
classification head
lightweight 2-channel contour decoder
```

Loss:

```text
L_total = L_cls + 0.3 * L_contour
L_cls = BCEWithLogitsLoss
L_contour = contour BCEWithLogitsLoss + Dice loss
```

Smoke test:

```text
image_shape = [4, 3, 224, 224]
mask_shape = [4, 2, 224, 224]
cls_logits_shape = [4]
contour_logits_shape = [4, 2, 224, 224]
status = ok
```

## Training Completion Record

Completed date: 2026-07-02

Formal training command:

```text
python scripts/run_002002002_contour_auxiliary.py --device cuda --epochs 80 --patience 10 --batch-size 24 --lr 1e-4 --weight-decay 1e-4 --lambda-contour 0.3 --allow-random-fallback
```

Cross-validation classification result:

```text
accuracy         = 0.8593 +/- 0.0250
precision        = 0.8845 +/- 0.0363
F1               = 0.8596 +/- 0.0226
```

Auxiliary contour result:

```text
mean Dice        = 0.1720 +/- 0.0604
y Dice           = 0.1417 +/- 0.0750
e Dice           = 0.2023 +/- 0.0564
mean IoU         = 0.0955 +/- 0.0356
```

Direct comparison with `002002001`:

```text


002002001 F1           = 0.8321
002002002 F1           = 0.8596
Delta F1               = +0.0275
```

Initial interpretation:

```text
Compared with the matched image-only ResNet18 baseline, adding y/e contour auxiliary supervision improved classification performance.
The contour decoder only achieved modest Dice because the supervision target is a sparse contour line, but the auxiliary task still appears to regularize the image encoder toward more discriminative anatomy-aware features.
```
