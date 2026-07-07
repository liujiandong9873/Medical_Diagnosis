# 002002003012 Experiment Design

## Research Question

Can a direct fractal-fractional multimodal interaction classifier, after validation-fitted probability calibration, outperform previous internal candidate models under the four paper-facing metrics?

```text
Acc
Pre
Recall
F1
```

## Hypothesis

The direct F2MI calibrated model will provide a stronger balanced classifier than previous internal candidates because it lets the multimodal interaction branch define the final decision boundary directly, instead of constraining it to be a small residual correction of the contour-auxiliary baseline.

Expected advantages:

```text
higher accuracy
higher precision
higher recall
higher F1
```

## Dataset and Splits

The experiment uses the same matched ZYQ classification cohort used throughout experiment 002:

```text
667 matched test rows across 5 outer folds
positive class = adenoid hypertrophy
negative class = non-hypertrophy
paired sample_id-aware statistical comparison
```

The model follows the same data inputs as the 010003 ablation:

```text
medical image
predicted contour data
128-point resampled contour representation
landmark-derived features
morphology/clinical geometric features
```

## Model

Target model:

```text
002002003012 direct F2MI calibrated model
```

Implementation source:

```text
002002003010003 no_residual variant
```

Formal model components:

```text
image feature encoder
contour feature branch
fractal token guidance
fractional contour operation
landmark/morphology feature integration
multimodal interaction router
direct F2MI classification logit
validation-fitted Platt calibration
```

The removed component is:

```text
baseline-preserving residual fusion
```

## Comparators

Main baseline:

```text
002002002 contour auxiliary baseline
```

Previous proposed model:

```text
002002003010002 full010 Platt
```

Training-length control:

```text
002002003011 continued-training control
```

Module comparators:

```text
010003 no_fractal Platt
010003 no_fractional Platt
```

Internal candidate routes:

```text
009003 recall-preserving adaptive F2MI
009004 calibrated-router F2MI
010 raw residual F2MI
```

## Primary Endpoints

Only the following endpoints are used for paper-facing reporting:

```text
Acc
Pre
Recall
F1
```

## Statistical Test

Use paired clustered bootstrap:

```text
cluster key = sample_id
iterations = 2000
target = direct_f2mi_calibrated
reference = each comparator
```

Report:

```text
target minus reference metric difference
95% bootstrap CI
two-sided bootstrap p value
```

## Decision Criteria

The proposed model is considered a stronger final candidate if it shows:

```text
best or near-best Acc
best or near-best Pre
best or near-best Recall
best or near-best F1
statistically supported Acc or F1 improvement over key comparators
```

## Expected Manuscript Role

This experiment should provide the final performance table for the paper-facing proposed model.

The strongest claim can be:

```text
The direct calibrated F2MI model achieved the best four-metric profile among internal candidate models.
```

The claim should be avoided:

```text
The direct calibrated F2MI model is globally superior on every possible endpoint.
```
