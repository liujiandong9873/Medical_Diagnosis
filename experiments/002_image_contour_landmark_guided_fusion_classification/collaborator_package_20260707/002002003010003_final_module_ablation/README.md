# 002002003010003: Final Module Ablation

Created date: 2026-07-06

## Purpose

This experiment organizes the final module ablation around the current proposed model:

```text
002002003010 baseline-preserving residual F2MI + Platt calibration
```

The goal is to test whether the final performance argument is supported by the model architecture itself, not only by calibration, extra fine-tuning, or threshold choice.

## Main Question

```text
Do fractal tokens, fractional contour operation, residual fusion, and calibration each contribute to the final recall/F1 improvement?
```

## Variants

The training script supports four variants:

| Variant | Meaning | Purpose |
| --- | --- | --- |
| `full` | Full 010 architecture rerun | Reproducibility control |
| `no_fractal` | Replace fractal tokens with learned non-fractal query tokens | Test box-counting/multifractal token contribution |
| `no_fractional` | Use raw contour points instead of fractional contour transform | Test fractional operator contribution |
| `no_residual` | Use F2MI branch logit directly instead of baseline + residual logit | Test baseline-preserving residual fusion contribution |

The primary ablation variants are:

```text
no_fractal
no_fractional
no_residual
```

## Fixed Experimental Conditions

All variants must use:

```text
same 667 matched samples
same 5 outer folds
same 002002002 baseline checkpoint initialization
same training schedule
same validation-based model selection
same Platt calibration analysis after training
same paired cluster bootstrap statistics
```

This keeps the comparison focused on module removal.

## Training Script

```text
scripts/run_010003_ablation.py
```

Example smoke test:

```text
python scripts/run_010003_ablation.py --variant no_fractal --smoke-test --device cuda --allow-random-fallback
```

Example training command:

```text
python scripts/run_010003_ablation.py --variant no_fractal --device cuda --epochs 80 --patience 10 --batch-size 12 --lr 1e-4 --weight-decay 1e-4 --allow-random-fallback
```

## Outputs

Outputs are separated by variant:

```text
outputs/full/
outputs/no_fractal/
outputs/no_fractional/
outputs/no_residual/
```

Checkpoints are also separated by variant:

```text
Models_saved/Generated_Models/002002003010003_final_module_ablation/full/
Models_saved/Generated_Models/002002003010003_final_module_ablation/no_fractal/
Models_saved/Generated_Models/002002003010003_final_module_ablation/no_fractional/
Models_saved/Generated_Models/002002003010003_final_module_ablation/no_residual/
```

## Planned Analysis

After training, each variant should receive the same validation-fitted Platt calibration as the proposed model.

Primary comparisons:

```text
Full 010 Platt vs no_fractal Platt
Full 010 Platt vs no_fractional Platt
Full 010 Platt vs no_residual Platt
Full 010 Platt vs 002002002 baseline
Full 010 Platt vs 002002003011 continued-training control
```

Main metrics:

```text
Acc
Precision
Recall
F1
```

Statistics:

```text
paired cluster bootstrap by sample_id
recommended iterations = 2000
```
