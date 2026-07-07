from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


KEYS = ["row_id", "sample_id", "image_file", "outer_fold", "split", "label"]
METRICS = ["auc", "acc", "pre", "recall", "specificity", "f1"]


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    experiment_dir = analysis_dir.parent
    base_dir = experiment_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-predictions", type=Path, default=experiment_dir / "outputs" / "cv_predictions.csv")
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=base_dir / "002002002_image_contour_auxiliary_supervision" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--strategy-predictions",
        type=Path,
        default=base_dir
        / "002002003010_baseline_preserving_residual_f2mi"
        / "analysis"
        / "010002_out"
        / "strategy_test_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "011001_out")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def specificity_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    pred = pred.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "acc": float(accuracy_score(y_true, pred)),
        "pre": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": specificity_score(y_true, pred),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_model_predictions(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {*KEYS, "prob_hypertrophy", "pred_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[df["split"] == "test"][[*KEYS, "prob_hypertrophy", "pred_label"]].copy()
    return df.rename(
        columns={
            "prob_hypertrophy": f"{model_name}_prob",
            "pred_label": f"{model_name}_pred",
        }
    )


def load_strategy_predictions(path: Path, strategy: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {*KEYS, "strategy", "prob_hypertrophy", "pred_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[(df["split"] == "test") & (df["strategy"] == strategy)][[*KEYS, "prob_hypertrophy", "pred_label"]].copy()
    return df.rename(
        columns={
            "prob_hypertrophy": f"{model_name}_prob",
            "pred_label": f"{model_name}_pred",
        }
    )


def merge_predictions(args: argparse.Namespace) -> pd.DataFrame:
    baseline = load_model_predictions(args.baseline_predictions, "baseline")
    target = load_model_predictions(args.target_predictions, "control011")
    fixed010 = load_strategy_predictions(args.strategy_predictions, "010_fixed_0_5", "fixed010")
    platt010 = load_strategy_predictions(args.strategy_predictions, "010_platt_calibration_0_5", "platt010")
    merged = baseline.merge(target, on=KEYS, validate="one_to_one")
    merged = merged.merge(fixed010, on=KEYS, validate="one_to_one")
    merged = merged.merge(platt010, on=KEYS, validate="one_to_one")
    merged["label"] = merged["label"].astype(int)
    for col in ["baseline_pred", "control011_pred", "fixed010_pred", "platt010_pred"]:
        merged[col] = merged[col].astype(int)
    return merged.sort_values(["outer_fold", "row_id"]).reset_index(drop=True)


def summarize_models(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = merged["label"].to_numpy()
    names = [
        ("002002002_baseline", "baseline"),
        ("002002003011_continued_training", "control011"),
        ("002002003010_fixed_0_5", "fixed010"),
        ("002002003010002_platt_0_5", "platt010"),
    ]
    for display_name, prefix in names:
        metrics = compute_metrics(y, merged[f"{prefix}_prob"].to_numpy(), merged[f"{prefix}_pred"].to_numpy())
        rows.append({"model": display_name, "n": int(len(merged)), **metrics})
    return pd.DataFrame(rows)


def decision_counts(merged: pd.DataFrame, target_prefix: str, reference_prefix: str) -> dict[str, int]:
    target_correct = merged[f"{target_prefix}_pred"].astype(int).eq(merged["label"].astype(int))
    ref_correct = merged[f"{reference_prefix}_pred"].astype(int).eq(merged["label"].astype(int))
    target_fn = ((merged["label"] == 1) & (merged[f"{target_prefix}_pred"] == 0)).sum()
    ref_fn = ((merged["label"] == 1) & (merged[f"{reference_prefix}_pred"] == 0)).sum()
    target_fp = ((merged["label"] == 0) & (merged[f"{target_prefix}_pred"] == 1)).sum()
    ref_fp = ((merged["label"] == 0) & (merged[f"{reference_prefix}_pred"] == 1)).sum()
    return {
        "target": target_prefix,
        "reference": reference_prefix,
        "both_correct": int((target_correct & ref_correct).sum()),
        "target_only_correct": int((target_correct & ~ref_correct).sum()),
        "reference_only_correct": int((~target_correct & ref_correct).sum()),
        "both_wrong": int((~target_correct & ~ref_correct).sum()),
        "net_correct_gain": int((target_correct & ~ref_correct).sum() - (~target_correct & ref_correct).sum()),
        "fn_diff_target_minus_reference": int(target_fn - ref_fn),
        "fp_diff_target_minus_reference": int(target_fp - ref_fp),
    }


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.asarray(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    return np.concatenate([np.asarray(groups[sid], dtype=int) for sid in sampled_ids])


def paired_bootstrap(
    merged: pd.DataFrame,
    target_prefix: str,
    reference_prefix: str,
    n_iters: int,
    seed: int,
) -> pd.DataFrame:
    y = merged["label"].to_numpy()
    target_metrics = compute_metrics(y, merged[f"{target_prefix}_prob"].to_numpy(), merged[f"{target_prefix}_pred"].to_numpy())
    ref_metrics = compute_metrics(y, merged[f"{reference_prefix}_prob"].to_numpy(), merged[f"{reference_prefix}_pred"].to_numpy())
    observed = {metric: target_metrics[metric] - ref_metrics[metric] for metric in METRICS}
    rng = np.random.default_rng(seed)
    boot_diffs = {metric: [] for metric in METRICS}
    for _ in range(n_iters):
        idx = sample_cluster_indices(merged, rng)
        boot = merged.iloc[idx]
        y_boot = boot["label"].to_numpy()
        if len(np.unique(y_boot)) < 2:
            continue
        target_boot = compute_metrics(
            y_boot,
            boot[f"{target_prefix}_prob"].to_numpy(),
            boot[f"{target_prefix}_pred"].to_numpy(),
        )
        ref_boot = compute_metrics(
            y_boot,
            boot[f"{reference_prefix}_prob"].to_numpy(),
            boot[f"{reference_prefix}_pred"].to_numpy(),
        )
        for metric in METRICS:
            value = target_boot[metric] - ref_boot[metric]
            if not np.isnan(value):
                boot_diffs[metric].append(value)
    rows = []
    for metric in METRICS:
        values = np.asarray(boot_diffs[metric], dtype=float)
        p_lower = float(np.mean(values <= 0.0))
        p_upper = float(np.mean(values >= 0.0))
        rows.append(
            {
                "target": target_prefix,
                "reference": reference_prefix,
                "metric": metric,
                "diff_target_minus_reference": observed[metric],
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_p_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
                "bootstrap_n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def metric_line(boot: pd.DataFrame, target: str, reference: str, metric: str) -> str:
    row = boot[(boot["target"] == target) & (boot["reference"] == reference) & (boot["metric"] == metric)].iloc[0]
    return (
        f"{row['diff_target_minus_reference']:+.4f} "
        f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], p={row['bootstrap_p_two_sided']:.3f}"
    )


def write_report(output_dir: Path, summary: pd.DataFrame, decisions: pd.DataFrame, boot: pd.DataFrame, n_iters: int) -> None:
    rows = {row["model"]: row for _, row in summary.iterrows()}
    base = rows["002002002_baseline"]
    control = rows["002002003011_continued_training"]
    platt = rows["002002003010002_platt_0_5"]
    lines = [
        "# 002002003011001 Continued-Training Control Analysis",
        "",
        "## Scope",
        "",
        "- Target: `002002003011 continued-training control`",
        "- Comparator 1: `002002002 contour auxiliary baseline`",
        "- Comparator 2: `002002003010002 Platt-calibrated 010`",
        f"- Paired cluster bootstrap iterations: {n_iters}",
        "",
        "## Pooled Test Metrics",
        "",
        "| Model | AUC | Acc | Pre | Recall | Specificity | F1 | FN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['auc']:.4f} | {row['acc']:.4f} | {row['pre']:.4f} | "
            f"{row['recall']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | "
            f"{int(row['fn'])} | {int(row['fp'])} |"
        )
    lines += [
        "",
        "## Key Deltas",
        "",
        "| Comparison | AUC | Acc | Pre | Recall | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| 011 - baseline | {control['auc'] - base['auc']:+.4f} | {control['acc'] - base['acc']:+.4f} | "
            f"{control['pre'] - base['pre']:+.4f} | {control['recall'] - base['recall']:+.4f} | "
            f"{control['specificity'] - base['specificity']:+.4f} | {control['f1'] - base['f1']:+.4f} |"
        ),
        (
            f"| Platt 010 - 011 | {platt['auc'] - control['auc']:+.4f} | {platt['acc'] - control['acc']:+.4f} | "
            f"{platt['pre'] - control['pre']:+.4f} | {platt['recall'] - control['recall']:+.4f} | "
            f"{platt['specificity'] - control['specificity']:+.4f} | {platt['f1'] - control['f1']:+.4f} |"
        ),
        "",
        "## Bootstrap",
        "",
        "| Comparison | Recall diff | F1 diff | Precision diff | Specificity diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, reference, label in [
        ("control011", "baseline", "011 - baseline"),
        ("platt010", "control011", "Platt 010 - 011"),
        ("platt010", "baseline", "Platt 010 - baseline"),
    ]:
        lines.append(
            f"| {label} | {metric_line(boot, target, reference, 'recall')} | "
            f"{metric_line(boot, target, reference, 'f1')} | "
            f"{metric_line(boot, target, reference, 'pre')} | "
            f"{metric_line(boot, target, reference, 'specificity')} |"
        )
    lines += [
        "",
        "## Decision Counts",
        "",
        "| Target | Reference | Target-only correct | Reference-only correct | Net correct gain | FN diff | FP diff |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in decisions.iterrows():
        lines.append(
            f"| {row['target']} | {row['reference']} | {int(row['target_only_correct'])} | "
            f"{int(row['reference_only_correct'])} | {int(row['net_correct_gain'])} | "
            f"{int(row['fn_diff_target_minus_reference'])} | {int(row['fp_diff_target_minus_reference'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if control["f1"] < platt["f1"] and control["recall"] < platt["recall"]:
        lines.append(
            "The continued-training control improves only modestly over the baseline and does not reproduce the "
            "Platt-calibrated 010 recall/F1 profile. This supports the claim that the 010 gain is not explained "
            "by extra fine-tuning alone."
        )
    else:
        lines.append(
            "The continued-training control approaches the proposed model. The manuscript should therefore weaken "
            "the architecture-specific claim and discuss fine-tuning as a possible contributor."
        )
    lines += [
        "",
        "## Files",
        "",
        "- `merged_predictions.csv`",
        "- `pooled_model_metrics.csv`",
        "- `decision_counts.csv`",
        "- `paired_bootstrap.csv`",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_predictions(args)
    summary = summarize_models(merged)
    decisions = pd.DataFrame(
        [
            decision_counts(merged, "control011", "baseline"),
            decision_counts(merged, "platt010", "control011"),
            decision_counts(merged, "platt010", "baseline"),
        ]
    )
    boot = pd.concat(
        [
            paired_bootstrap(merged, "control011", "baseline", args.bootstrap_iters, args.seed),
            paired_bootstrap(merged, "platt010", "control011", args.bootstrap_iters, args.seed + 1009),
            paired_bootstrap(merged, "platt010", "baseline", args.bootstrap_iters, args.seed + 2018),
        ],
        ignore_index=True,
    )
    merged.to_csv(args.output_dir / "merged_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "pooled_model_metrics.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(args.output_dir / "decision_counts.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(args.output_dir / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir, summary, decisions, boot, args.bootstrap_iters)
    print(summary[["model", "auc", "acc", "pre", "recall", "specificity", "f1", "fn", "fp"]].to_string(index=False))


if __name__ == "__main__":
    main()
