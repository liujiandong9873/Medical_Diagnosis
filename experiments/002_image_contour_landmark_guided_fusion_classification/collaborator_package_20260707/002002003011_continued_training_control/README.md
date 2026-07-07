# 002002003011: Continued-Training Control

Created date: 2026-07-06

## Purpose

This experiment is a control for `002002003010 baseline-preserving residual F2MI`.

The core question is:

```text
Can the improvement observed in 002002003010/010002 be explained simply by continuing to train the strong 002002002 contour-auxiliary baseline?
```

## Control Logic

`002002003011` keeps the baseline architecture unchanged:

```text
ResNet18 shared encoder
classification head
2-channel y/e contour decoder
```

It initializes each outer fold from the corresponding trained `002002002` checkpoint:

```text
Models_saved/Generated_Models/002002002_contour_auxiliary_resnet18/contour_aux_resnet18_fold{fold}.pt
```

Then it continues fine-tuning on exactly the same training data, same contour mask target, same outer folds, and same inference input.

## What Is Controlled

Same as `002002002`:

```text
input at inference = raw X-ray image only
training supervision = hypertrophy label + y/e contour mask
model architecture = contour-auxiliary ResNet18
sample set = 667 matched image-contour samples
outer CV = same 5 StratifiedGroupKFold splits
```

Different from `002002003010`:

```text
no fractal tokens
no fractional contour operator
no landmark tokens
no morphology tokens
no F2MI router
no residual logit branch
```

## Main Test

If `002002003011` reaches performance close to `002002003010/010002`, then the proposed-model gain may be explainable by continued fine-tuning.

If `002002003011` remains close to `002002002`, while `002002003010/010002` keeps higher F1/recall, then the result supports a model-architecture contribution from the fractal-fractional residual F2MI path.

## Training Defaults

The script keeps the initial source checkpoint as epoch 0 and includes it as a candidate best state.

Default continued-training setting:

```text
epochs = 40
patience = 8
lr = 1e-5
monitor = val_f1
batch_size = 24
lambda_contour = 0.3
```

## Command

```text
python scripts/run_011.py --device cuda --epochs 40 --patience 8 --batch-size 24 --lr 1e-5 --weight-decay 1e-4 --lambda-contour 0.3 --monitor val_f1 --allow-random-fallback
```

## Outputs

```text
outputs/cv_predictions.csv
outputs/cv_metrics_by_fold.csv
outputs/cv_summary.csv
outputs/training_history.csv
outputs/training_progress.md
outputs/folds/f{fold}/metrics.csv
outputs/folds/f{fold}/predictions.csv
outputs/folds/f{fold}/history.csv
Models_saved/Generated_Models/002002003011_continued_training_control/002002003011_fold{fold}.pt
```
