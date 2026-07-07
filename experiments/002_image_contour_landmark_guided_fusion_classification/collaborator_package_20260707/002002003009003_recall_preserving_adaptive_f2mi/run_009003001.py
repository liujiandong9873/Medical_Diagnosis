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
TARGET_ROUTER_COLS = [f"recall009003_{col}" for col in ROUTER_COLS]
TARGET_EXTRA_COLS = [
    *TARGET_ROUTER_COLS,
    "recall009003_router_entropy",
    "recall009003_fractional_alpha_y",
    "recall009003_fractional_alpha_e",
]
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
MODELS = [
    ("contour002002002", "002002002 contour auxiliary"),
    ("f2mi009", "002002003009 original F2MI"),
    ("adaptive009002", "002002003009002 sample-adaptive F2MI"),
    ("recall009003", "002002003009003 recall-preserving adaptive F2MI"),
]
COMPARATORS = [
    ("contour002002002", "002002002"),
    ("f2mi009", "002002003009"),
    ("adaptive009002", "002002003009002"),
]


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    experiment_dir = analysis_dir.parent
    base_dir = experiment_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-predictions",
        type=Path,
        default=experiment_dir / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--contour-predictions",
        type=Path,
        default=base_dir / "002002002_image_contour_auxiliary_supervision" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--f2mi-predictions",
        type=Path,
        default=base_dir / "002002003009_fractal_guided_multimodal_interaction" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--adaptive009002-predictions",
        type=Path,
        default=base_dir / "002002003009002_sample_adaptive_router_f2mi" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "009003001_out")
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


def metrics_at_threshold(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    return compute_metrics(y_true, prob, (prob >= threshold).astype(int))


def load_model_predictions(path: Path, prefix: str, include_target_extras: bool = False) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {*KEYS, "prob_hypertrophy", "pred_label"}
    if include_target_extras:
        required.update(ROUTER_COLS)
        required.update({"router_entropy", "fractional_alpha_y", "fractional_alpha_e"})
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    keep = [*KEYS, "prob_hypertrophy", "pred_label"]
    if include_target_extras:
        keep += [*ROUTER_COLS, "router_entropy", "fractional_alpha_y", "fractional_alpha_e"]
    data = data[keep].copy()
    rename = {"prob_hypertrophy": f"{prefix}_prob", "pred_label": f"{prefix}_pred"}
    if include_target_extras:
        rename.update({col: f"{prefix}_{col}" for col in ROUTER_COLS})
        rename.update(
            {
                "router_entropy": f"{prefix}_router_entropy",
                "fractional_alpha_y": f"{prefix}_fractional_alpha_y",
                "fractional_alpha_e": f"{prefix}_fractional_alpha_e",
            }
        )
    data = data.rename(columns=rename)
    data["label"] = data["label"].astype(int)
    data[f"{prefix}_pred"] = data[f"{prefix}_pred"].astype(int)
    return data


def load_all_predictions(args: argparse.Namespace) -> pd.DataFrame:
    contour = load_model_predictions(args.contour_predictions, "contour002002002")
    f2mi = load_model_predictions(args.f2mi_predictions, "f2mi009")
    adaptive = load_model_predictions(args.adaptive009002_predictions, "adaptive009002")
    target = load_model_predictions(args.target_predictions, "recall009003", include_target_extras=True)
    merged = contour.merge(f2mi, on=KEYS, validate="one_to_one")
    merged = merged.merge(adaptive, on=KEYS, validate="one_to_one")
    merged = merged.merge(target, on=KEYS, validate="one_to_one")
    for prefix, _ in MODELS:
        merged[f"{prefix}_correct"] = (merged[f"{prefix}_pred"] == merged["label"]).astype(int)
        merged[f"{prefix}_confidence"] = (merged[f"{prefix}_prob"].astype(float) - 0.5).abs()
    merged["recall009003_false_negative"] = ((merged["label"] == 1) & (merged["recall009003_pred"] == 0)).astype(int)
    merged["recall009003_false_positive"] = ((merged["label"] == 0) & (merged["recall009003_pred"] == 1)).astype(int)
    merged["recall009003_error_type"] = "true_negative"
    merged.loc[(merged["label"] == 1) & (merged["recall009003_pred"] == 1), "recall009003_error_type"] = "true_positive"
    merged.loc[(merged["label"] == 1) & (merged["recall009003_pred"] == 0), "recall009003_error_type"] = "false_negative"
    merged.loc[(merged["label"] == 0) & (merged["recall009003_pred"] == 1), "recall009003_error_type"] = "false_positive"
    merged["recall009003_alpha_mean"] = merged[
        ["recall009003_fractional_alpha_y", "recall009003_fractional_alpha_e"]
    ].mean(axis=1)
    merged["recall009003_alpha_abs_diff"] = (
        merged["recall009003_fractional_alpha_y"] - merged["recall009003_fractional_alpha_e"]
    ).abs()
    for comparator, _ in COMPARATORS:
        merged[f"recall009003_vs_{comparator}"] = "same_decision"
        merged.loc[
            (merged["recall009003_correct"] == 1) & (merged[f"{comparator}_correct"] == 0),
            f"recall009003_vs_{comparator}",
        ] = "target_only_correct"
        merged.loc[
            (merged["recall009003_correct"] == 0) & (merged[f"{comparator}_correct"] == 1),
            f"recall009003_vs_{comparator}",
        ] = f"{comparator}_only_correct"
        merged.loc[
            (merged["recall009003_correct"] == 0) & (merged[f"{comparator}_correct"] == 0),
            f"recall009003_vs_{comparator}",
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
            for metric in METRICS:
                row[f"{prefix}_{metric}"] = metrics[metric]
            for count_col in ["tn", "fp", "fn", "tp"]:
                row[f"{prefix}_{count_col}"] = metrics[count_col]
        for comparator, _ in COMPARATORS:
            for metric in MAIN_METRICS:
                row[f"recall009003_minus_{comparator}_{metric}"] = (
                    row[f"recall009003_{metric}"] - row[f"{comparator}_{metric}"]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.asarray(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    return np.concatenate([np.asarray(groups[sid], dtype=int) for sid in sampled_ids])


def paired_bootstrap(df: pd.DataFrame, comparator: str, n_iters: int, seed: int) -> pd.DataFrame:
    target = "recall009003"
    y = df["label"].to_numpy()
    target_metrics = compute_metrics(y, df[f"{target}_prob"].to_numpy(), df[f"{target}_pred"].to_numpy())
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
        target_boot = compute_metrics(y_boot, boot[f"{target}_prob"].to_numpy(), boot[f"{target}_pred"].to_numpy())
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
                "target": target,
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
    target_correct = df["recall009003_correct"].astype(bool)
    comparator_correct = df[f"{comparator}_correct"].astype(bool)
    target_fn = ((df["label"] == 1) & (df["recall009003_pred"] == 0)).sum()
    comparator_fn = ((df["label"] == 1) & (df[f"{comparator}_pred"] == 0)).sum()
    target_fp = ((df["label"] == 0) & (df["recall009003_pred"] == 1)).sum()
    comparator_fp = ((df["label"] == 0) & (df[f"{comparator}_pred"] == 1)).sum()
    rows = [
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
    return pd.DataFrame(rows)


def summarize_numeric(df: pd.DataFrame, cols: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        out = df[cols].agg(["mean", "std", "min", "median", "max"]).T.reset_index()
        return out.rename(columns={"index": "field"})
    grouped = df.groupby(group_cols)[cols].agg(["count", "mean", "std", "min", "median", "max"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def router_correlations(df: pd.DataFrame) -> pd.DataFrame:
    corr_df = df[
        [
            *TARGET_EXTRA_COLS,
            "recall009003_alpha_mean",
            "recall009003_alpha_abs_diff",
            "label",
            "recall009003_prob",
            "recall009003_confidence",
            "recall009003_correct",
            "recall009003_false_negative",
            "recall009003_false_positive",
            "contour002002002_correct",
            "f2mi009_correct",
            "adaptive009002_correct",
            "contour002002002_prob",
            "f2mi009_prob",
            "adaptive009002_prob",
        ]
    ].copy()
    corr = corr_df.corr(method="spearman")
    sources = [*TARGET_EXTRA_COLS, "recall009003_alpha_mean", "recall009003_alpha_abs_diff"]
    targets = [
        "label",
        "recall009003_prob",
        "recall009003_confidence",
        "recall009003_correct",
        "recall009003_false_negative",
        "recall009003_false_positive",
        "contour002002002_correct",
        "f2mi009_correct",
        "adaptive009002_correct",
        "contour002002002_prob",
        "f2mi009_prob",
        "adaptive009002_prob",
    ]
    rows = []
    for source in sources:
        for target in targets:
            rows.append({"source": source, "target": target, "spearman_r": float(corr.loc[source, target])})
    return pd.DataFrame(rows)


def threshold_grid(probs: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 181)
    unique_probs = np.unique(np.round(probs.astype(float), 4))
    candidates = np.unique(np.concatenate([grid, unique_probs, np.asarray([0.5])]))
    return candidates[(candidates > 0.0) & (candidates < 1.0)]


def threshold_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "f1":
        return metrics["f1"]
    if objective == "balanced_acc":
        return metrics["balanced_acc"]
    if objective == "youden":
        return metrics["recall"] + metrics["specificity"] - 1.0
    raise ValueError(f"Unknown objective: {objective}")


def select_threshold(y_val: np.ndarray, prob_val: np.ndarray, objective: str) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = metrics_at_threshold(y_val, prob_val, 0.5)
    best_key = (
        threshold_score(best_metrics, objective),
        best_metrics["f1"],
        best_metrics["balanced_acc"],
        -abs(best_threshold - 0.5),
    )
    for threshold in threshold_grid(prob_val):
        metrics = metrics_at_threshold(y_val, prob_val, float(threshold))
        key = (
            threshold_score(metrics, objective),
            metrics["f1"],
            metrics["balanced_acc"],
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def threshold_tuning(validation_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, val_fold in validation_df.groupby("outer_fold"):
        test_fold = test_df[test_df["outer_fold"] == fold].copy()
        y_val = val_fold["label"].to_numpy()
        prob_val = val_fold["recall009003_prob"].to_numpy()
        y_test = test_fold["label"].to_numpy()
        prob_test = test_fold["recall009003_prob"].to_numpy()
        fixed_val = metrics_at_threshold(y_val, prob_val, 0.5)
        fixed_test = metrics_at_threshold(y_test, prob_test, 0.5)
        row = {"outer_fold": int(fold), "objective": "fixed_0_5", "threshold": 0.5}
        row["val_objective_score"] = fixed_val["f1"]
        row.update({f"val_{key}": value for key, value in fixed_val.items()})
        row.update({f"test_{key}": value for key, value in fixed_test.items()})
        rows.append(row)
        for objective in ["f1", "balanced_acc", "youden"]:
            threshold, val_metrics = select_threshold(y_val, prob_val, objective)
            test_metrics = metrics_at_threshold(y_test, prob_test, threshold)
            row = {
                "outer_fold": int(fold),
                "objective": objective,
                "threshold": threshold,
                "val_objective_score": threshold_score(val_metrics, objective),
            }
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row.update({f"test_{key}": value for key, value in test_metrics.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_summary(thresholds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["threshold", "test_auc", "test_acc", "test_pre", "test_recall", "test_specificity", "test_f1"]
    grouped = thresholds.groupby("objective")[metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def metric_value(comparison: pd.DataFrame, model: str, metric: str) -> float:
    return float(comparison[comparison["model"] == model][metric].iloc[0])


def write_report(
    output_dir: Path,
    test_df: pd.DataFrame,
    comparison: pd.DataFrame,
    fold_comparison: pd.DataFrame,
    boot_tables: dict[str, pd.DataFrame],
    decision_tables: dict[str, pd.DataFrame],
    router_by_error: pd.DataFrame,
    router_by_fold: pd.DataFrame,
    correlations: pd.DataFrame,
    threshold_sum: pd.DataFrame,
) -> None:
    def boot_line(comparator: str, metric: str) -> str:
        row = boot_tables[comparator][boot_tables[comparator]["metric"] == metric].iloc[0]
        return (
            f"{row['diff_target_minus_comparator']:+.4f} "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], p={row['bootstrap_p_two_sided']:.3f}"
        )

    def decision_value(comparator: str, category: str) -> int:
        table = decision_tables[comparator]
        match = table[table["category"] == category]
        return int(match["count"].iloc[0]) if len(match) else 0

    def fold_mean(model: str, metric: str) -> float:
        return float(fold_comparison[f"{model}_{metric}"].astype(float).mean())

    router_std_cols = [f"{col}_std" for col in TARGET_ROUTER_COLS]
    router_std_mean = router_by_fold[router_std_cols].mean().to_dict()
    top_corr = correlations.copy()
    top_corr["abs_r"] = top_corr["spearman_r"].abs()
    top_corr = top_corr.sort_values("abs_r", ascending=False).head(12)
    fixed_summary = threshold_sum[threshold_sum["objective"] == "fixed_0_5"].iloc[0]
    f1_summary = threshold_sum[threshold_sum["objective"] == "f1"].iloc[0]
    f1_gain = float(f1_summary["test_f1_mean"] - fixed_summary["test_f1_mean"])
    recall_gain = float(f1_summary["test_recall_mean"] - fixed_summary["test_recall_mean"])

    sig_positive = []
    sig_negative = []
    for comparator, label in COMPARATORS:
        table = boot_tables[comparator]
        for metric in ["auc", "acc", "pre", "recall", "f1"]:
            row = table[table["metric"] == metric].iloc[0]
            if row["ci95_low"] > 0:
                sig_positive.append(f"{metric} vs {label}")
            if row["ci95_high"] < 0:
                sig_negative.append(f"{metric} vs {label}")

    lines = [
        "# 002002003009003001 Paired Statistical and Error Analysis",
        "",
        "## Scope",
        "",
        "- Target: `002002003009003 recall-preserving adaptive F2MI`",
        "- Comparators: `002002002`, original `002002003009`, and `002002003009002`",
        "- Main comparison split: test only",
        "- Bootstrap: clustered paired bootstrap by sample_id",
        "- Threshold tuning: selected on validation split and applied to the paired test fold",
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
            f"{metric_value(comparison, model, 'recall'):.4f} | "
            f"{metric_value(comparison, model, 'specificity'):.4f} | {metric_value(comparison, model, 'f1'):.4f} |"
        )

    lines += [
        "",
        "## Fold-Mean Test Comparison",
        "",
        "| Model | AUC mean | Acc mean | Pre mean | Recall mean | F1 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, label in MODELS:
        lines.append(
            f"| {label} | {fold_mean(model, 'auc'):.4f} | {fold_mean(model, 'acc'):.4f} | "
            f"{fold_mean(model, 'pre'):.4f} | {fold_mean(model, 'recall'):.4f} | {fold_mean(model, 'f1'):.4f} |"
        )

    lines += ["", "## Paired Bootstrap", ""]
    for comparator, label in COMPARATORS:
        lines += [
            f"Target minus {label}:",
            "",
            "| Metric | Difference [95% CI], p |",
            "|---|---:|",
        ]
        for metric in ["auc", "acc", "pre", "recall", "f1"]:
            lines.append(f"| {metric} | {boot_line(comparator, metric)} |")
        lines.append("")

    lines += ["## Decision Counts", ""]
    for comparator, label in COMPARATORS:
        lines += [
            f"Compared with {label}:",
            "",
            f"- Both correct: {decision_value(comparator, 'both_correct')}",
            f"- Target only correct: {decision_value(comparator, 'target_only_correct')}",
            f"- Comparator only correct: {decision_value(comparator, f'{comparator}_only_correct')}",
            f"- Both wrong: {decision_value(comparator, 'both_wrong')}",
            f"- Net correct gain: {decision_value(comparator, f'net_correct_gain_target_minus_{comparator}')}",
            f"- Target FN minus comparator FN: {decision_value(comparator, f'target_fn_minus_{comparator}_fn')}",
            f"- Target FP minus comparator FP: {decision_value(comparator, f'target_fp_minus_{comparator}_fp')}",
            "",
        ]

    lines += [
        "## Router Mechanism",
        "",
        "Mean within-fold router std:",
        "",
    ]
    for key, value in router_std_mean.items():
        lines.append(f"- {key}: {value:.4f}")
    lines += [
        "",
        f"- False negatives: {int(test_df['recall009003_false_negative'].sum())}",
        f"- False positives: {int(test_df['recall009003_false_positive'].sum())}",
        "- Router/error type summary is saved in `router_by_error_type.csv`.",
        "",
        "Top Spearman correlations involving target router/alpha variables:",
        "",
        "| Source | Target | Spearman r |",
        "|---|---|---:|",
    ]
    for _, row in top_corr.iterrows():
        lines.append(f"| {row['source']} | {row['target']} | {row['spearman_r']:.4f} |")

    lines += [
        "",
        "## Validation-Based Threshold Tuning",
        "",
        "| Objective | Threshold mean | Test Acc | Test Pre | Test Recall | Test F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in threshold_sum.sort_values("objective").iterrows():
        lines.append(
            f"| {row['objective']} | {row['threshold_mean']:.4f} | {row['test_acc_mean']:.4f} | "
            f"{row['test_pre_mean']:.4f} | {row['test_recall_mean']:.4f} | {row['test_f1_mean']:.4f} |"
        )
    lines += [
        "",
        "Threshold interpretation:",
        "",
        f"- F1-tuned threshold mean test F1 change vs fixed 0.5: {f1_gain:+.4f}",
        f"- F1-tuned threshold mean test recall change vs fixed 0.5: {recall_gain:+.4f}",
        "",
        "## Overall Decision",
        "",
    ]
    if sig_positive:
        lines.append(f"Significant positive bootstrap findings: {', '.join(sig_positive)}.")
    else:
        lines.append("No main metric shows a clearly significant positive bootstrap advantage across all comparators.")
    if sig_negative:
        lines.append(f"Significant negative bootstrap findings: {', '.join(sig_negative)}.")
    else:
        lines.append("No main metric shows a clearly significant negative disadvantage by CI crossing entirely below zero.")
    lines += [
        "",
        "Interpretation:",
        "",
        "```text",
        "009003 is a useful recall-preserving candidate and fixes the main weakness of 009002.",
        "However, its advantage is mainly in recall/F1, while AUC and precision are not improved.",
        "It should be treated as a candidate model route pending manuscript-level statistical framing, not as a settled final model.",
        "```",
        "",
        "Recommended next step:",
        "",
        "```text",
        "Decide whether to present 009003 as a recall-preserving variant, or design 009004 focused on preserving AUC while keeping the F1 gain.",
        "```",
        "",
        "## Output Files",
        "",
        "- `merged_predictions.csv`",
        "- `model_test.csv`",
        "- `model_by_fold.csv`",
        "- `boot_vs_002002002.csv`",
        "- `boot_vs_009.csv`",
        "- `boot_vs_009002.csv`",
        "- `decisions_vs_002002002.csv`",
        "- `decisions_vs_009.csv`",
        "- `decisions_vs_009002.csv`",
        "- `router_by_error_type.csv`",
        "- `router_by_fold.csv`",
        "- `router_correlations.csv`",
        "- `threshold_tuning_by_fold.csv`",
        "- `threshold_tuning_summary.csv`",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = load_all_predictions(args)
    validation_df = merged[merged["split"] == "validation"].copy()
    test_df = merged[merged["split"] == "test"].copy()
    if validation_df.empty or test_df.empty:
        raise ValueError("Both validation and test predictions are required.")

    comparison = model_comparison(test_df)
    fold_comparison = model_comparison_by_fold(test_df)
    boot_tables = {
        comparator: paired_bootstrap(test_df, comparator, args.bootstrap_iters, args.seed + idx)
        for idx, (comparator, _) in enumerate(COMPARATORS)
    }
    decision_tables = {comparator: decision_counts(test_df, comparator) for comparator, _ in COMPARATORS}

    router_cols = [
        *TARGET_EXTRA_COLS,
        "recall009003_alpha_mean",
        "recall009003_alpha_abs_diff",
        "recall009003_prob",
        "recall009003_confidence",
    ]
    router_overall = summarize_numeric(test_df, router_cols)
    router_by_error = summarize_numeric(test_df, router_cols, ["recall009003_error_type"])
    router_by_fold = summarize_numeric(test_df, router_cols, ["outer_fold"])
    router_by_label_correctness = summarize_numeric(test_df, router_cols, ["label", "recall009003_correct"])
    router_by_vs = {
        comparator: summarize_numeric(test_df, router_cols, [f"recall009003_vs_{comparator}"])
        for comparator, _ in COMPARATORS
    }
    correlations = router_correlations(test_df)
    thresholds = threshold_tuning(validation_df, test_df)
    threshold_sum = threshold_summary(thresholds)

    false_negatives = test_df[test_df["recall009003_false_negative"] == 1].sort_values(
        ["outer_fold", "recall009003_prob", "row_id"]
    )
    false_positives = test_df[test_df["recall009003_false_positive"] == 1].sort_values(
        ["outer_fold", "recall009003_prob", "row_id"],
        ascending=[True, False, True],
    )

    short_labels = {"002002002": "002002002", "002002003009": "009", "002002003009002": "009002"}
    merged.to_csv(args.output_dir / "merged_predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(args.output_dir / "model_test.csv", index=False, encoding="utf-8-sig")
    fold_comparison.to_csv(args.output_dir / "model_by_fold.csv", index=False, encoding="utf-8-sig")
    for comparator, label in COMPARATORS:
        short_label = short_labels[label]
        boot_tables[comparator].to_csv(
            args.output_dir / f"boot_vs_{short_label}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        decision_tables[comparator].to_csv(
            args.output_dir / f"decisions_vs_{short_label}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        router_by_vs[comparator].to_csv(
            args.output_dir / f"router_vs_{short_label}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    router_overall.to_csv(args.output_dir / "router_overall_summary.csv", index=False, encoding="utf-8-sig")
    router_by_error.to_csv(args.output_dir / "router_by_error_type.csv", index=False, encoding="utf-8-sig")
    router_by_fold.to_csv(args.output_dir / "router_by_fold.csv", index=False, encoding="utf-8-sig")
    router_by_label_correctness.to_csv(
        args.output_dir / "router_by_label_correctness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlations.to_csv(args.output_dir / "router_correlations.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(args.output_dir / "threshold_tuning_by_fold.csv", index=False, encoding="utf-8-sig")
    threshold_sum.to_csv(args.output_dir / "threshold_tuning_summary.csv", index=False, encoding="utf-8-sig")
    false_negatives.to_csv(args.output_dir / "target_false_negative_cases.csv", index=False, encoding="utf-8-sig")
    false_positives.to_csv(args.output_dir / "target_false_positive_cases.csv", index=False, encoding="utf-8-sig")

    write_report(
        args.output_dir,
        test_df,
        comparison,
        fold_comparison,
        boot_tables,
        decision_tables,
        router_by_error,
        router_by_fold,
        correlations,
        threshold_sum,
    )
    print(f"Saved analysis outputs to {args.output_dir}")
    print(comparison[["model", "auc", "acc", "pre", "recall", "f1"]].to_string(index=False))
    print(threshold_sum[["objective", "threshold_mean", "test_recall_mean", "test_f1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
