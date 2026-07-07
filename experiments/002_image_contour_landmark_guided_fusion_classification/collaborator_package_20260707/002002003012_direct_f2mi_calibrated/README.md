# 002002003012: Direct F2MI Calibrated Model

Created date: 2026-07-07

## Purpose

This experiment formalizes the strongest calibrated model found in the final module ablation:

```text
direct F2MI calibrated model = 010003 no_residual variant + validation-fitted Platt calibration
```

The model is promoted from an ablation variant to a formal proposed-model candidate because it achieved the best current four-metric profile under:

```text
Acc
Pre
Recall
F1
```

## Background

The previous proposed route was:

```text
002002003010 baseline-preserving residual F2MI + Platt calibration
```

The final module ablation showed that removing the residual baseline-preserving decision equation and using the F2MI branch directly produced stronger calibrated classification performance.

Therefore, this experiment changes the paper-facing proposed model candidate from residual F2MI to direct calibrated F2MI.

## Model Definition

The model keeps:

```text
image encoder
predicted contour branch
landmark and morphology features
fractal token guidance
fractional contour operation
multimodal interaction router
validation-fitted Platt calibration
```

The model removes:

```text
final_logit = baseline_logit + residual_scale * residual_logit
```

Instead, the F2MI branch logit is used as the classification logit before Platt calibration.

## Source Outputs

The first formal analysis reuses the completed trained outputs from:

```text
../002002003010003_final_module_ablation/outputs/no_residual/cv_predictions.csv
```

The output is calibrated fold-by-fold with validation-fitted Platt scaling, exactly as in the final module ablation.

## Main Comparators

Primary comparators:

```text
002002002 contour auxiliary baseline
002002003010002 full010 Platt
002002003011 continued-training control
010003 no_fractal Platt
010003 no_fractional Platt
```

Internal comparators:

```text
002002003009003 recall-preserving adaptive F2MI
002002003009004 calibrated-router F2MI
002002003010 raw residual F2MI
```

## Main Metrics

All paper-facing reports use only:

```text
Acc
Pre
Recall
F1
```

## Statistics

```text
paired clustered bootstrap by sample_id
default bootstrap iterations = 2000
target = direct_f2mi_calibrated
metrics = Acc, Pre, Recall, F1
```

## Analysis

Run:

```text
python analysis/run_012001.py
```

Main output folder:

```text
analysis/012001_out/
```

Expected files:

```text
model_predictions_test.csv
model_metrics.csv
performance_deltas_vs_sota.csv
paired_bootstrap_vs_sota.csv
decision_counts_vs_sota.csv
analysis_report.md
```

## Paper Use

The intended manuscript claim is:

```text
Direct calibrated F2MI achieved the strongest four-metric classification profile among internal candidates.
```

The claim should not be written as:

```text
The direct calibrated F2MI model is globally superior on every possible endpoint.
```

## Seed Robustness

Formal repeat training is organized as:

```text
002002003012002 seed robustness repeat
```

Training script:

```text
scripts/run_012_direct_f2mi_seed.py
```

Example:

```text
python scripts/run_012_direct_f2mi_seed.py --seed 20260711 --device cuda --epochs 80 --patience 10 --batch-size 12 --lr 1e-4 --weight-decay 1e-4 --allow-random-fallback
```

Each seed writes to an independent folder:

```text
outputs/seed_20260711/
outputs/seed_20260712/
outputs/seed_20260713/
```

Seed robustness analysis:

```text
python analysis/012002_seed_robustness/run_012002.py
```
