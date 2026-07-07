from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


KEYS = ["row_id", "sample_id", "image_file", "outer_fold", "split", "label"]
ROUTER_COLS = ["router_image", "router_contour", "router_landmark", "router_morphology"]
METRICS = [
    "auc",
    "average_precision",
    "brier",
    "nll",
    "acc",
    "balanced_acc",
    "pre",
    "recall",
    "specificity",
    "f1",
]
MAIN_METRICS = ["auc", "acc", "pre", "recall", "specificity", "f1"]
TARGET = "residual010"
MODELS = [
    ("contour002002002", "002002002 contour auxiliary"),
    ("recall009003", "002002003009003 recall-preserving adaptive F2MI"),
    ("auc009004", "002002003009004 AUC-preserving calibrated-router F2MI"),
    ("residual010_baseline", "002002003010 baseline head"),
    ("residual010", "002002003010 final residual F2MI"),
]
COMPARATORS = [
    ("contour002002002", "002002002 contour auxiliary"),
    ("recall009003", "009003 recall-preserving adaptive F2MI"),
    ("auc009004", "009004 AUC-preserving calibrated-router F2MI"),
    ("residual010_baseline", "010 baseline head"),
]


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    experiment_dir = analysis_dir.parent
    base_dir = experiment_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-predictions", type=Path, default=experiment_dir / "outputs" / "cv_predictions.csv")
    parser.add_argument(
        "--contour-predictions",
        type=Path,
        default=base_dir / "002002002_image_contour_auxiliary_supervision" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--recall009003-predictions",
        type=Path,
        default=base_dir / "002002003009003_recall_preserving_adaptive_f2mi" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--auc009004-predictions",
        type=Path,
        default=base_dir / "002002003009004_auc_preserving_calibrated_router_f2mi" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "010001_out")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def clip_prob(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(prob.astype(float), eps, 1.0 - eps)


def specificity_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    pred = pred.astype(int)
    prob = prob.astype(float)
    out = {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "average_precision": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, prob)),
        "nll": float(log_loss(y_true, clip_prob(prob), labels=[0, 1])),
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "pre": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": specificity_score(y_true, pred),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def load_simple_predictions(path: Path, prefix: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {*KEYS, "prob_hypertrophy", "pred_label"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    data = data[[*KEYS, "prob_hypertrophy", "pred_label"]].copy()
    data = data.rename(columns={"prob_hypertrophy": f"{prefix}_prob", "pred_label": f"{prefix}_pred"})
    data["label"] = data["label"].astype(int)
    data[f"{prefix}_pred"] = data[f"{prefix}_pred"].astype(int)
    return data


def load_target_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        *KEYS,
        "prob_hypertrophy",
        "pred_label",
        "baseline_prob_hypertrophy",
        "baseline_pred_label",
        "residual_logit",
        "final_logit",
        *ROUTER_COLS,
        "router_entropy",
        "fractional_alpha_y",
        "fractional_alpha_e",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    keep = [
        *KEYS,
        "prob_hypertrophy",
        "pred_label",
        "baseline_prob_hypertrophy",
        "baseline_pred_label",
        "residual_logit",
        "final_logit",
        *ROUTER_COLS,
        "router_entropy",
        "fractional_alpha_y",
        "fractional_alpha_e",
    ]
    data = data[keep].copy()
    rename = {
        "prob_hypertrophy": "residual010_prob",
        "pred_label": "residual010_pred",
        "baseline_prob_hypertrophy": "residual010_baseline_prob",
        "baseline_pred_label": "residual010_baseline_pred",
        "residual_logit": "residual010_residual_logit",
        "final_logit": "residual010_final_logit",
        "router_entropy": "residual010_router_entropy",
        "fractional_alpha_y": "residual010_fractional_alpha_y",
        "fractional_alpha_e": "residual010_fractional_alpha_e",
    }
    rename.update({col: f"residual010_{col}" for col in ROUTER_COLS})
    data = data.rename(columns=rename)
    data["label"] = data["label"].astype(int)
    data["residual010_pred"] = data["residual010_pred"].astype(int)
    data["residual010_baseline_pred"] = data["residual010_baseline_pred"].astype(int)
    data["residual010_residual_abs_logit"] = data["residual010_residual_logit"].abs()
    data["residual010_prob_delta_vs_baseline"] = data["residual010_prob"] - data["residual010_baseline_prob"]
    data["residual010_alpha_mean"] = data[
        ["residual010_fractional_alpha_y", "residual010_fractional_alpha_e"]
    ].mean(axis=1)
    data["residual010_alpha_abs_diff"] = (
        data["residual010_fractional_alpha_y"] - data["residual010_fractional_alpha_e"]
    ).abs()
    return data


def load_all_predictions(args: argparse.Namespace) -> pd.DataFrame:
    contour = load_simple_predictions(args.contour_predictions, "contour002002002")
    recall009003 = load_simple_predictions(args.recall009003_predictions, "recall009003")
    auc009004 = load_simple_predictions(args.auc009004_predictions, "auc009004")
    target = load_target_predictions(args.target_predictions)
    merged = contour.merge(recall009003, on=KEYS, validate="one_to_one")
    merged = merged.merge(auc009004, on=KEYS, validate="one_to_one")
    merged = merged.merge(target, on=KEYS, validate="one_to_one")
    for prefix, _ in MODELS:
        merged[f"{prefix}_correct"] = (merged[f"{prefix}_pred"] == merged["label"]).astype(int)
        merged[f"{prefix}_confidence"] = (merged[f"{prefix}_prob"].astype(float) - 0.5).abs()
        merged[f"{prefix}_false_negative"] = ((merged["label"] == 1) & (merged[f"{prefix}_pred"] == 0)).astype(int)
        merged[f"{prefix}_false_positive"] = ((merged["label"] == 0) & (merged[f"{prefix}_pred"] == 1)).astype(int)
    merged["residual010_error_type"] = "true_negative"
    merged.loc[(merged["label"] == 1) & (merged["residual010_pred"] == 1), "residual010_error_type"] = "true_positive"
    merged.loc[(merged["label"] == 1) & (merged["residual010_pred"] == 0), "residual010_error_type"] = "false_negative"
    merged.loc[(merged["label"] == 0) & (merged["residual010_pred"] == 1), "residual010_error_type"] = "false_positive"
    merged["residual010_changed_baseline_decision"] = (
        merged["residual010_pred"] != merged["residual010_baseline_pred"]
    ).astype(int)
    for comparator, _ in COMPARATORS:
        merged[f"residual010_vs_{comparator}"] = "same_correctness"
        merged.loc[
            (merged["residual010_correct"] == 1) & (merged[f"{comparator}_correct"] == 0),
            f"residual010_vs_{comparator}",
        ] = "target_only_correct"
        merged.loc[
            (merged["residual010_correct"] == 0) & (merged[f"{comparator}_correct"] == 1),
            f"residual010_vs_{comparator}",
        ] = f"{comparator}_only_correct"
        merged.loc[
            (merged["residual010_correct"] == 0) & (merged[f"{comparator}_correct"] == 0),
            f"residual010_vs_{comparator}",
        ] = "both_wrong"
    return merged.sort_values(["split", "outer_fold", "row_id"]).reset_index(drop=True)


def model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    y = df["label"].to_numpy()
    rows = []
    for prefix, label in MODELS:
        row = {
            "model": prefix,
            "model_label": label,
            "n": int(len(df)),
            "unique_sample_id": int(df["sample_id"].nunique()),
            "positive_n": int(y.sum()),
            "negative_n": int((y == 0).sum()),
        }
        row.update(compute_metrics(y, df[f"{prefix}_prob"].to_numpy(), df[f"{prefix}_pred"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def model_comparison_by_fold(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, fold_df in df.groupby("outer_fold"):
        y = fold_df["label"].to_numpy()
        row = {"outer_fold": int(fold), "n": int(len(fold_df)), "positive_n": int(y.sum())}
        row["negative_n"] = int((y == 0).sum())
        for prefix, _ in MODELS:
            metrics = compute_metrics(y, fold_df[f"{prefix}_prob"].to_numpy(), fold_df[f"{prefix}_pred"].to_numpy())
            for metric in [*METRICS, "tn", "fp", "fn", "tp"]:
                row[f"{prefix}_{metric}"] = metrics[metric]
        for comparator, _ in COMPARATORS:
            for metric in MAIN_METRICS:
                row[f"residual010_minus_{comparator}_{metric}"] = (
                    row[f"residual010_{metric}"] - row[f"{comparator}_{metric}"]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.asarray(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    return np.concatenate([np.asarray(groups[sid], dtype=int) for sid in sampled_ids])


def paired_bootstrap(df: pd.DataFrame, comparator: str, n_iters: int, seed: int) -> pd.DataFrame:
    y = df["label"].to_numpy()
    target_metrics = compute_metrics(y, df[f"{TARGET}_prob"].to_numpy(), df[f"{TARGET}_pred"].to_numpy())
    comparator_metrics = compute_metrics(y, df[f"{comparator}_prob"].to_numpy(), df[f"{comparator}_pred"].to_numpy())
    observed = {metric: target_metrics[metric] - comparator_metrics[metric] for metric in METRICS}
    rng = np.random.default_rng(seed)
    boot_diffs = {metric: [] for metric in METRICS}
    for _ in range(n_iters):
        idx = sample_cluster_indices(df, rng)
        boot = df.iloc[idx]
        y_boot = boot["label"].to_numpy()
        if len(np.unique(y_boot)) < 2:
            continue
        target_boot = compute_metrics(y_boot, boot[f"{TARGET}_prob"].to_numpy(), boot[f"{TARGET}_pred"].to_numpy())
        comparator_boot = compute_metrics(
            y_boot,
            boot[f"{comparator}_prob"].to_numpy(),
            boot[f"{comparator}_pred"].to_numpy(),
        )
        for metric in METRICS:
            value = target_boot[metric] - comparator_boot[metric]
            if not np.isnan(value):
                boot_diffs[metric].append(value)
    rows = []
    for metric in METRICS:
        values = np.asarray(boot_diffs[metric], dtype=float)
        p_lower = float(np.mean(values <= 0.0))
        p_upper = float(np.mean(values >= 0.0))
        rows.append(
            {
                "target": TARGET,
                "comparator": comparator,
                "metric": metric,
                "diff_target_minus_comparator": observed[metric],
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_p_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
                "bootstrap_n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def decision_counts(df: pd.DataFrame, comparator: str) -> pd.DataFrame:
    target_correct = df[f"{TARGET}_correct"].astype(bool)
    comparator_correct = df[f"{comparator}_correct"].astype(bool)
    target_fn = ((df["label"] == 1) & (df[f"{TARGET}_pred"] == 0)).sum()
    comparator_fn = ((df["label"] == 1) & (df[f"{comparator}_pred"] == 0)).sum()
    target_fp = ((df["label"] == 0) & (df[f"{TARGET}_pred"] == 1)).sum()
    comparator_fp = ((df["label"] == 0) & (df[f"{comparator}_pred"] == 1)).sum()
    return pd.DataFrame(
        [
            {"category": "both_correct", "count": int((target_correct & comparator_correct).sum())},
            {"category": "target_only_correct", "count": int((target_correct & ~comparator_correct).sum())},
            {"category": f"{comparator}_only_correct", "count": int((~target_correct & comparator_correct).sum())},
            {"category": "both_wrong", "count": int((~target_correct & ~comparator_correct).sum())},
            {
                "category": f"net_correct_gain_target_minus_{comparator}",
                "count": int((target_correct & ~comparator_correct).sum() - (~target_correct & comparator_correct).sum()),
            },
            {"category": f"target_fn_minus_{comparator}_fn", "count": int(target_fn - comparator_fn)},
            {"category": f"target_fp_minus_{comparator}_fp", "count": int(target_fp - comparator_fp)},
        ]
    )


def summarize_numeric(df: pd.DataFrame, cols: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        out = df[cols].agg(["mean", "std", "min", "median", "max"]).T.reset_index()
        return out.rename(columns={"index": "field"})
    grouped = df.groupby(group_cols)[cols].agg(["count", "mean", "std", "min", "median", "max"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def metric_value(comparison: pd.DataFrame, model: str, metric: str) -> float:
    return float(comparison[comparison["model"] == model][metric].iloc[0])


def boot_text(boot: pd.DataFrame, metric: str) -> str:
    row = boot[boot["metric"] == metric].iloc[0]
    return (
        f"{row['diff_target_minus_comparator']:+.4f} "
        f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], p={row['bootstrap_p_two_sided']:.3f}"
    )


def decision_value(decision: pd.DataFrame, category: str) -> int:
    row = decision[decision["category"] == category]
    return int(row["count"].iloc[0]) if len(row) else 0


def write_report(
    output_dir: Path,
    test_df: pd.DataFrame,
    comparison: pd.DataFrame,
    by_fold: pd.DataFrame,
    boot_tables: dict[str, pd.DataFrame],
    decision_tables: dict[str, pd.DataFrame],
    residual_summary: pd.DataFrame,
    residual_by_error: pd.DataFrame,
    router_by_fold: pd.DataFrame,
) -> None:
    lines = [
        "# 002002003010001 Paired Statistical and Residual Analysis",
        "",
        "## Scope",
        "",
        "- Target: `002002003010 baseline-preserving residual F2MI`",
        "- Main comparator: `002002002 contour auxiliary`",
        "- Additional comparators: `009003`, `009004`, and `010 baseline head`",
        "- Split: paired test predictions only",
        "- Bootstrap: clustered paired bootstrap by sample_id",
        f"- Test rows: {len(test_df)}",
        f"- Unique sample_id: {test_df['sample_id'].nunique()}",
        "",
        "## Pooled Test Comparison",
        "",
        "| Model | AUC | Acc | Pre | Recall | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, label in MODELS:
        lines.append(
            f"| {label} | {metric_value(comparison, model, 'auc'):.4f} | "
            f"{metric_value(comparison, model, 'acc'):.4f} | {metric_value(comparison, model, 'pre'):.4f} | "
            f"{metric_value(comparison, model, 'recall'):.4f} | {metric_value(comparison, model, 'specificity'):.4f} | "
            f"{metric_value(comparison, model, 'f1'):.4f} |"
        )

    lines += [
        "",
        "## Paired Bootstrap Summary",
        "",
        "| Comparator | AUC diff | Acc diff | Pre diff | Recall diff | Specificity diff | F1 diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparator, label in COMPARATORS:
        boot = boot_tables[comparator]
        lines.append(
            f"| {label} | {boot_text(boot, 'auc')} | {boot_text(boot, 'acc')} | "
            f"{boot_text(boot, 'pre')} | {boot_text(boot, 'recall')} | "
            f"{boot_text(boot, 'specificity')} | {boot_text(boot, 'f1')} |"
        )

    lines += [
        "",
        "## Decision Counts",
        "",
        "| Comparator | Both correct | 010 only correct | Comparator only correct | Both wrong | Net correct gain | FN diff | FP diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparator, label in COMPARATORS:
        table = decision_tables[comparator]
        lines.append(
            f"| {label} | {decision_value(table, 'both_correct')} | "
            f"{decision_value(table, 'target_only_correct')} | "
            f"{decision_value(table, f'{comparator}_only_correct')} | {decision_value(table, 'both_wrong')} | "
            f"{decision_value(table, f'net_correct_gain_target_minus_{comparator}')} | "
            f"{decision_value(table, f'target_fn_minus_{comparator}_fn')} | "
            f"{decision_value(table, f'target_fp_minus_{comparator}_fp')} |"
        )

    changed_count = int(test_df["residual010_changed_baseline_decision"].sum())
    mean_residual = float(test_df["residual010_residual_abs_logit"].mean())
    mean_prob_delta = float(test_df["residual010_prob_delta_vs_baseline"].abs().mean())

    lines += [
        "",
        "## Residual Mechanism",
        "",
        f"- Final-vs-baseline changed decisions: {changed_count} / {len(test_df)}",
        f"- Mean absolute residual logit: {mean_residual:.4f}",
        f"- Mean absolute probability delta vs baseline head: {mean_prob_delta:.6f}",
        "",
        "### Residual Summary",
        "",
        "| Field | Mean | Std | Min | Median | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in residual_summary.iterrows():
        lines.append(
            f"| {row['field']} | {row['mean']:.6f} | {row['std']:.6f} | {row['min']:.6f} | "
            f"{row['median']:.6f} | {row['max']:.6f} |"
        )

    lines += [
        "",
        "### Residual by 010 Error Type",
        "",
        "| Error type | Count | Residual abs mean | Prob delta mean |",
        "|---|---:|---:|---:|",
    ]
    for _, row in residual_by_error.iterrows():
        lines.append(
            f"| {row['residual010_error_type']} | {int(row['residual010_residual_abs_logit_count'])} | "
            f"{row['residual010_residual_abs_logit_mean']:.4f} | "
            f"{row['residual010_prob_delta_vs_baseline_mean']:.6f} |"
        )

    lines += [
        "",
        "## Router by Fold",
        "",
        "| Fold | Image | Contour | Landmark | Morphology | Residual abs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in router_by_fold.iterrows():
        lines.append(
            f"| {int(row['outer_fold'])} | {row['residual010_router_image_mean']:.4f} | "
            f"{row['residual010_router_contour_mean']:.4f} | {row['residual010_router_landmark_mean']:.4f} | "
            f"{row['residual010_router_morphology_mean']:.4f} | {row['residual010_residual_abs_logit_mean']:.4f} |"
        )

    vs_contour = boot_tables["contour002002002"]
    auc_row = vs_contour[vs_contour["metric"] == "auc"].iloc[0]
    f1_row = vs_contour[vs_contour["metric"] == "f1"].iloc[0]
    recall_row = vs_contour[vs_contour["metric"] == "recall"].iloc[0]
    pre_row = vs_contour[vs_contour["metric"] == "pre"].iloc[0]
    specificity_row = vs_contour[vs_contour["metric"] == "specificity"].iloc[0]

    lines += [
        "",
        "## Interpretation",
        "",
        "Against `002002002`, 010 shows higher pooled AUC, accuracy, recall, and F1, with lower precision and specificity.",
        "",
        f"- AUC vs 002002002: {boot_text(vs_contour, 'auc')}",
        f"- Recall vs 002002002: {boot_text(vs_contour, 'recall')}",
        f"- F1 vs 002002002: {boot_text(vs_contour, 'f1')}",
        f"- Precision vs 002002002: {boot_text(vs_contour, 'pre')}",
        f"- Specificity vs 002002002: {boot_text(vs_contour, 'specificity')}",
        "",
    ]

    if auc_row["ci95_low"] > 0 and f1_row["ci95_low"] > 0:
        lines.append("The paired bootstrap supports 010 as a stronger candidate than 002002002 for AUC and F1.")
    elif recall_row["ci95_low"] > 0 and f1_row["ci95_low"] > 0:
        lines.append("The paired bootstrap supports 010 as a recall/F1-improved candidate, although AUC superiority is not fully established.")
    else:
        lines.append("The paired bootstrap does not yet fully support a strong superiority claim over 002002002.")

    if pre_row["ci95_high"] < 0 or specificity_row["ci95_high"] < 0:
        lines.append("However, the gain is accompanied by a statistically supported decrease in precision and/or specificity.")

    if changed_count == 0:
        lines.append(
            "Final and baseline-head labels are identical at the default 0.5 threshold; the observed gain should be framed as baseline-preserving joint F2MI fine-tuning rather than direct residual relabeling."
        )
    else:
        lines.append(
            "Final and baseline-head labels differ for some samples, so residual correction contributes directly to decisions in addition to joint fine-tuning."
        )

    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = load_all_predictions(args)
    validation_df = merged[merged["split"] == "validation"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    comparison = model_comparison(test_df)
    by_fold = model_comparison_by_fold(test_df)
    residual_cols = [
        "residual010_residual_logit",
        "residual010_residual_abs_logit",
        "residual010_prob_delta_vs_baseline",
        "residual010_router_entropy",
        "residual010_alpha_mean",
        "residual010_alpha_abs_diff",
    ]
    residual_summary = summarize_numeric(test_df, residual_cols)
    residual_by_error = summarize_numeric(
        test_df,
        ["residual010_residual_abs_logit", "residual010_prob_delta_vs_baseline"],
        ["residual010_error_type"],
    )
    router_by_fold = summarize_numeric(
        test_df,
        [
            "residual010_router_image",
            "residual010_router_contour",
            "residual010_router_landmark",
            "residual010_router_morphology",
            "residual010_residual_abs_logit",
        ],
        ["outer_fold"],
    )
    router_by_error = summarize_numeric(
        test_df,
        [
            "residual010_router_image",
            "residual010_router_contour",
            "residual010_router_landmark",
            "residual010_router_morphology",
            "residual010_residual_abs_logit",
        ],
        ["residual010_error_type"],
    )
    boot_tables = {
        comparator: paired_bootstrap(test_df, comparator, args.bootstrap_iters, args.seed + index * 1009)
        for index, (comparator, _) in enumerate(COMPARATORS)
    }
    decision_tables = {comparator: decision_counts(test_df, comparator) for comparator, _ in COMPARATORS}

    merged.to_csv(args.output_dir / "merged_predictions_all_splits.csv", index=False, encoding="utf-8-sig")
    validation_df.to_csv(args.output_dir / "merged_predictions_validation.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(args.output_dir / "merged_predictions_test.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(args.output_dir / "model_comparison_test.csv", index=False, encoding="utf-8-sig")
    by_fold.to_csv(args.output_dir / "model_comparison_by_fold_test.csv", index=False, encoding="utf-8-sig")
    residual_summary.to_csv(args.output_dir / "residual_summary.csv", index=False, encoding="utf-8-sig")
    residual_by_error.to_csv(args.output_dir / "residual_by_error_type.csv", index=False, encoding="utf-8-sig")
    router_by_fold.to_csv(args.output_dir / "router_by_fold.csv", index=False, encoding="utf-8-sig")
    router_by_error.to_csv(args.output_dir / "router_by_error_type.csv", index=False, encoding="utf-8-sig")
    for comparator, table in boot_tables.items():
        table.to_csv(args.output_dir / f"boot_vs_{comparator}.csv", index=False, encoding="utf-8-sig")
    for comparator, table in decision_tables.items():
        table.to_csv(args.output_dir / f"decisions_vs_{comparator}.csv", index=False, encoding="utf-8-sig")

    write_report(
        args.output_dir,
        test_df,
        comparison,
        by_fold,
        boot_tables,
        decision_tables,
        residual_summary,
        residual_by_error,
        router_by_fold,
    )
    print(f"Wrote analysis outputs to {args.output_dir}")
    print(comparison[["model", "auc", "acc", "pre", "recall", "specificity", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()

