# 002002003010003 Ablation Interpretation

Updated date: 2026-07-07

## Main Finding

The final module ablation produced an unexpected but important result:

```text
The no_residual variant outperformed the previous Full 010 Platt model after validation-fitted Platt calibration.
```

This means the baseline-preserving residual fusion design should not be claimed as the final best architecture.

## Calibrated Test Metrics

| Model | Acc | Pre | Recall | F1 |
| --- | --- | --- | --- | --- |
| baseline002002002 | 0.8591 | 0.8831 | 0.8367 | 0.8593 |
| full010_platt | 0.8786 | 0.8722 | 0.8950 | 0.8835 |
| no_fractal_platt | 0.8726 | 0.8707 | 0.8834 | 0.8770 |
| no_fractional_platt | 0.8741 | 0.8798 | 0.8746 | 0.8772 |
| no_residual_platt | 0.9010 | 0.9062 | 0.9009 | 0.9035 |

## Full 010 Platt vs no_residual Platt

The bootstrap table is reported as `full010_platt - no_residual_platt`.

| Metric | Difference | 95% CI | p |
| --- | --- | --- | --- |
| Acc | -0.0225 | [-0.0423, -0.0015] | 0.042 |
| Precision | -0.0340 | [-0.0613, -0.0076] | 0.008 |
| Recall | -0.0058 | [-0.0350, 0.0211] | 0.726 |
| F1 | -0.0201 | [-0.0398, -0.0001] | 0.048 |

Interpretation:

```text
Recall is similar.
```

Decision counts:

```text
full010_platt only correct: 19
no_residual_platt only correct: 34
net correct gain for full010_platt: -15
```

So `no_residual_platt` reduces both false positives and false negatives compared with the previous full model operating point.

## Implication For Manuscript

The previous claim:

```text
baseline-preserving residual fusion is necessary for the final proposed model
```

is not supported by the current ablation.

The stronger claim should be revised to:

```text
A fractal-fractional multimodal interaction branch with validation-fitted probability calibration provides the strongest balanced classification performance. The residual baseline-preserving formulation improves recall over the baseline but is not the optimal final decision head.
```

## Recommended Next Experiment

Create a new candidate experiment based on `no_residual`:

```text
002002003012 direct F2MI calibrated model
```

Purpose:

```text
Promote no_residual from an ablation variant to a formal proposed-model candidate, then validate it against baseline, full010, and continued-training control with final statistics.
```

Minimum next steps:

```text
1. Rename or formalize no_residual as candidate model 002002003012.
2. Run a dedicated analysis where no_residual_platt is the primary target rather than only an ablation reference.
3. Compare no_residual_platt vs baseline002002002, control011, full010_platt, no_fractal_platt, and no_fractional_platt.
4. Update the paper narrative around direct calibrated F2MI instead of baseline-preserving residual F2MI.
```
