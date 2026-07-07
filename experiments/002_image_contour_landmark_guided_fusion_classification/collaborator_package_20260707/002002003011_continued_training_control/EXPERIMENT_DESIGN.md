# 002002003011 Experiment Design

## Background

`002002003010` improves the paper narrative because its calibrated operating point (`010_platt_calibration_0_5`) improves F1 and recall over the strong `002002002` contour-auxiliary baseline. However, `002002003010` is initialized from `002002002` checkpoints and then trained further with a larger multimodal residual architecture.

This creates a potential alternative explanation:

```text
Maybe the gain comes from continued fine-tuning of the baseline checkpoint, not from the fractal-fractional residual F2MI architecture.
```

`002002003011` is designed to test this alternative explanation directly.

## Hypothesis

Primary hypothesis:

```text
Continued fine-tuning of the unchanged 002002002 contour-auxiliary baseline will not reproduce the F1/recall gain obtained by 002002003010/010002.
```

If supported, this strengthens the architectural claim for `002002003010`.

## Data

The experiment reuses the same prepared data from `002002002`:

```text
metadata.csv
folds.csv
outer_splits.csv
contour_targets_224_uint8.npz
contour_target_manifest.json
```

The matched sample set contains:

```text
667 image-contour samples
663 unique sample_id values
324 normal samples
343 hypertrophy samples
5 outer folds
```

## Model

Architecture is intentionally unchanged from `002002002`:

```text
X-ray image
  -> ResNet18 encoder
  -> classification head
  -> contour decoder
```

Loss:

```text
L_total = L_cls + 0.3 * L_contour
L_cls = BCEWithLogitsLoss
L_contour = contour BCEWithLogitsLoss + Dice loss
```

No additional modules are allowed in this control experiment.

## Initialization

For each outer fold:

```text
fold 0 loads contour_aux_resnet18_fold0.pt
fold 1 loads contour_aux_resnet18_fold1.pt
fold 2 loads contour_aux_resnet18_fold2.pt
fold 3 loads contour_aux_resnet18_fold3.pt
fold 4 loads contour_aux_resnet18_fold4.pt
```

The loaded checkpoint is evaluated as epoch 0 and kept as the initial best candidate. This avoids falsely worsening the control if continued training overfits.

## Evaluation

Primary comparison:

```text
002002002 contour auxiliary baseline
002002003011 continued-training control
002002003010 fixed 0.5
002002003010002 Platt-calibrated 010
```

Main metrics:

```text
Acc
Precision
Recall
F1
```

Decision-level interpretation:

```text
Does 011 reduce false negatives like 010?
Does 011 improve F1 like 010?
```

## Expected Interpretation

If `011` is close to `002002002`, then `010` cannot be reduced to simple continued training.

If `011` approaches or exceeds `010`, then the paper should weaken the architectural claim and describe `010` more carefully as a training/initialization/fine-tuning strategy rather than a clearly superior fractal-fractional architecture.
