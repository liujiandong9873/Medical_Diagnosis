# 002002003012002 Seed Robustness Interpretation

Updated date: 2026-07-07

## Purpose

This experiment tests whether the strong `direct_f2mi_calibrated` result is a stable model-architecture effect or a post-hoc candidate selected from a favorable seed.

## Completed Runs

The original candidate came from:

```text
010003 no_residual + validation-fitted Platt calibration
formal name: direct_f2mi_calibrated_original
```

Two independent repeat seeds were trained under the formal `002002003012` folder:

```text
seed_20260711
seed_20260712
```

Both repeats used:

```text
same 667 samples
same 5 outer folds
same 002002002 baseline checkpoint initialization
same direct F2MI architecture
same validation-fitted Platt calibration
same paired cluster bootstrap analysis
```

## Main Results

| Model | Acc | Pre | Recall | F1 |
| --- | --- | --- | --- | --- |
| direct_f2mi_calibrated_original | 0.9010 | 0.9062 | 0.9009 | 0.9035 |
| baseline002002002 | 0.8591 | 0.8831 | 0.8367 | 0.8593 |
| seed_20260711_platt | 0.8651 | 0.8604 | 0.8805 | 0.8703 |
| seed_20260712_platt | 0.8726 | 0.8909 | 0.8571 | 0.8737 |

## Interpretation

The seed robustness result is mixed.

The positive part:

```text
Both repeated seeds keep F1 numerically above the 002002002 baseline.
Both repeated seeds keep Acc numerically above the 002002002 baseline.
```

The caution:

```text
Neither repeated seed reproduces the original 0.9035 F1.
Both repeated seeds are statistically worse than the original direct F2MI candidate in Acc and F1.
The strong original candidate therefore cannot be presented as a fully seed-stable 0.90+ F1 result.
```

## Bootstrap Summary

Against baseline:

```text
seed_20260711_platt:
  Acc    +0.0060, p=0.692
  Pre    -0.0227, p=0.236
  Recall +0.0437, p=0.016
  F1     +0.0110, p=0.395

seed_20260712_platt:
  Acc    +0.0135, p=0.350
  Pre    +0.0078, p=0.754
  Recall +0.0204, p=0.292
  F1     +0.0144, p=0.280
```

Against the original direct F2MI:

```text
seed_20260711_platt:
  Acc -0.0360, p=0.004
  F1  -0.0332, p=0.004

seed_20260712_platt:
  Acc -0.0285, p=0.011
  F1  -0.0298, p=0.005
```

## Paper Consequence

The original wording:

```text
Direct calibrated F2MI is a stable final proposed model with strong 0.90+ F1 performance.
```

is not supported by the seed robustness results.

The safer wording is:

```text
Direct calibrated F2MI achieved the best observed four-metric profile, and independent seed repeats preserved a modest F1 trend over the baseline, but the magnitude of the best result was seed-sensitive.
```

## Recommendation

For a stronger SCI/Q1 manuscript, do not rely only on `direct_f2mi_calibrated_original` as the central evidence.

Better options:

```text
1. Present direct F2MI as the best observed candidate, with seed robustness reported transparently as supplementary evidence.
2. Treat the repeated-seed mean as the conservative performance estimate.
3. Consider using an ensemble or validation-selected seed aggregation if the paper needs more stable performance above baseline.
4. Add external validation or a locked holdout test before making a strong final-model claim.
```

## Files

```text
analysis/012002_out/model_metrics.csv
analysis/012002_out/paired_bootstrap.csv
analysis/012002_out/decision_counts.csv
analysis/012002_out/analysis_report.md
paper_tables/table_10_direct_f2mi_seed_robustness_metrics.csv
paper_tables/table_11_direct_f2mi_seed_robustness_bootstrap.csv
paper_tables/table_12_direct_f2mi_seed_robustness_decision_counts.csv
```
