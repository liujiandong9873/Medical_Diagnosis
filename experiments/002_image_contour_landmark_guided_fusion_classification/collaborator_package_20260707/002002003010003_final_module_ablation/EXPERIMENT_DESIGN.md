# 002002003010003 Experiment Design

## Background

The current proposed model is `002002003010 baseline-preserving residual F2MI`, and the preferred paper operating point is `010_platt_calibration_0_5`.

The locked result shows:

```text
010 Platt improves recall and F1 over the 002002002 contour-auxiliary baseline.
The 002002003011 continued-training control does not reproduce this gain.
```

The remaining question for a stronger SCI/Q1 manuscript is whether the named architecture modules actually contribute to the final performance.

## Hypothesis

The final calibrated proposed model benefits from three structural components:

```text
fractal token guidance
fractional contour operation
baseline-preserving residual fusion
```

Removing any one of these components should weaken the recall/F1 or false-negative reduction profile.

## Variant Definitions

### Full

The original `002002003010` architecture:

```text
contour prediction
fractal token encoder
fractional contour operator
image/contour/landmark/morphology interaction router
baseline logit + residual_scale * residual logit
```

This variant is kept for reproducibility but does not need to be rerun first because the completed `002002003010` result is already available.

### No Fractal

Ablation:

```text
replace fractal box-counting/multifractal tokens with learned non-fractal query tokens
remove fractal token consistency loss
keep contour, landmark, morphology, router, and residual fusion
```

Interpretation:

```text
If performance decreases, the fractal token representation contributes information beyond ordinary learned query tokens.
```

### No Fractional

Ablation:

```text
skip fractional contour transform
feed raw predicted contour points as both raw and transformed contour inputs
keep fractal token, router, and residual fusion
```

Interpretation:

```text
If performance decreases, the fractional contour operator contributes to shape-sequence modeling.
```

### No Residual

Ablation:

```text
do not use final_logit = baseline_logit + residual_scale * residual_logit
use the F2MI branch logit directly as the classification logit
keep baseline classifier as an auxiliary head
remove residual L2 constraint from the final branch
```

Interpretation:

```text
If performance decreases, the baseline-preserving residual design is necessary for keeping the strong baseline decision surface while allowing F2MI correction.
```

## Fairness Rules

All variants must reuse:

```text
same matched sample set
same outer_splits.csv
same contour masks
same 128-point y/e contour resampling
same landmark and morphology annotation parsing
same 002002002 checkpoint initialization
same optimizer, epoch, patience, and batch size
```

All variants must be calibrated using validation folds before test evaluation.

## Decision Criteria

A module is considered useful if removing it causes one or more of the following:

```text
lower F1
lower recall
more false negatives
loss of significant recall/F1 advantage over baseline
inferior paired bootstrap comparison against Full 010 Platt
```

## Paper Use

The ablation table should support this manuscript argument:

```text
The proposed model's improvement is not only a calibration artifact or a continued-training artifact. The final recall/F1 gain depends on the combination of fractal token guidance, fractional contour modeling, and baseline-preserving residual fusion.
```
