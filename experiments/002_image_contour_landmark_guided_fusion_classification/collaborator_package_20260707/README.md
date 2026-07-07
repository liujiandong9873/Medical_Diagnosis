# Experiment 002 Collaborator Package

Created date: 2026-07-07

## Purpose

This folder is a collaborator-facing package for Experiment 002. It contains:

```text
1. The main collaborator document.
2. Copied code files for every model or model route mentioned in that document.
3. One separate folder per model or experiment route.
```

All files in this package are copies. The original model code remains in its original experiment folder.

## Main Document

```text
00_collaborator_document/
  EXPERIMENT_002_MANUSCRIPT_FRAMEWORK_FOR_COLLABORATORS.md
```

## Dataset Description

```text
01_zyq_dataset_description/
  ZYQ_DATASET_DESCRIPTION.md
  case_examples/
  case_examples_manifest.csv
  generate_zyq_case_examples.py
```

This folder describes the ZYQ dataset used by Experiment 002 and provides ten annotated case figures for collaborators.

## Model Code Folders

| Folder | Corresponding model / route | Included file types |
|---|---|---|
| `002002001_matched_image_only_resnet18/` | matched image-only ResNet18 | training script, README |
| `002002002_contour_auxiliary_resnet18/` | contour auxiliary ResNet18 baseline | contour target preparation, training script, README |
| `002002003001_direct_predicted_contour_feature_fusion/` | direct predicted contour feature fusion | training script, README |
| `002002003002_raw_f2cg_net/` | raw F2CG-Net | training script, README |
| `002002003003_f2cg_probability_calibration/` | F2CG probability calibration route | calibration script, README |
| `002002003004_calibrated_f2cg_statistical_validation/` | calibrated F2CG statistical validation | statistical validation script, README |
| `002002003005_f2cg_module_ablation/` | F2CG module ablation | ablation training script, README |
| `002002003009_original_f2mi_transformer/` | original F2MI-Transformer | training script, analysis script, README |
| `002002003009002_sample_adaptive_router_f2mi/` | sample-adaptive router F2MI | training script, analysis script, README |
| `002002003009003_recall_preserving_adaptive_f2mi/` | recall-preserving adaptive F2MI | training script, analysis script, README, candidate decision note |
| `002002003009004_calibrated_router_f2mi/` | calibrated-router F2MI | training script, README, experiment design |
| `002002003010_residual_f2mi_full010/` | residual F2MI / full010 Platt route | training script, analysis scripts, README, experiment design |
| `002002003010003_final_module_ablation/` | final module ablation: no_fractal, no_fractional, no_residual | ablation training script, analysis script, README, experiment design, interpretation note |
| `002002003011_continued_training_control/` | continued-training control | training script, analysis script, README, experiment design |
| `002002003012_direct_f2mi_calibrated/` | direct calibrated F2MI final candidate and seed robustness | seed training script, analysis scripts, README, experiment design, seed robustness interpretation |

## Notes For Collaborators

```text
1. This package is for reading, review, and handoff.
2. It is not intended to replace the original experiment folders.
3. If code is modified for future experiments, modify the original experiment route or create a new experiment folder, not this package.
4. The main paper-facing metrics are Acc, Pre, Recall, and F1.
```
