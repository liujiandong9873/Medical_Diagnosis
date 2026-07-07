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
ADAPTIVE_ROUTER_COLS = [f"adaptive_{col}" for col in ROUTER_COLS]
ADAPTIVE_EXTRA_COLS = [
    *ADAPTIVE_ROUTER_COLS,
    "adaptive_router_entropy",
    "adaptive_fractional_alpha_y",
    "adaptive_fractional_alpha_e",
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
    ("contour", "002002002 contour auxiliary"),
    ("f2mi", "002002003009 original F2MI"),
    ("adaptive", "002002003009002 sample-adaptive F2MI"),
]


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    experiment_dir = analysis_dir.parent
    base_dir = experiment_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adaptive-predictions",
        type=Path,
        default=experiment_dir / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--f2mi-predictions",
        type=Path,
        default=base_dir / "002002003009_fractal_guided_multimodal_interaction" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--contour-predictions",
        type=Path,
        default=base_dir / "002002002_image_contour_auxiliary_supervision" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "009002001_outputs")
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


def load_model_predictions(path: Path, prefix: str, include_adaptive_extras: bool = False) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {*KEYS, "prob_hypertrophy", "pred_label"}
    if include_adaptive_extras:
        required.update(ROUTER_COLS)
        required.update({"router_entropy", "fractional_alpha_y", "fractional_alpha_e"})
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    keep = [*KEYS, "prob_hypertrophy", "pred_label"]
    if include_adaptive_extras:
        keep += [*ROUTER_COLS, "router_entropy", "fractional_alpha_y", "fractional_alpha_e"]
    data = data[keep].copy()
    rename = {
        "prob_hypertrophy": f"{prefix}_prob",
        "pred_label": f"{prefix}_pred",
    }
    if include_adaptive_extras:
        rename.update({col: f"adaptive_{col}" for col in ROUTER_COLS})
        rename.update(
            {
                "router_entropy": "adaptive_router_entropy",
                "fractional_alpha_y": "adaptive_fractional_alpha_y",
                "fractional_alpha_e": "adaptive_fractional_alpha_e",
            }
        )
    data = data.rename(columns=rename)
    data["label"] = data["label"].astype(int)
    data[f"{prefix}_pred"] = data[f"{prefix}_pred"].astype(int)
    return data


def load_all_predictions(args: argparse.Namespace) -> pd.DataFrame:
    contour = load_model_predictions(args.contour_predictions, "contour")
    f2mi = load_model_predictions(args.f2mi_predictions, "f2mi")
    adaptive = load_model_predictions(args.adaptive_predictions, "adaptive", include_adaptive_extras=True)

    merged = contour.merge(f2mi, on=KEYS, validate="one_to_one")
    merged = merged.merge(adaptive, on=KEYS, validate="one_to_one")
    for prefix, _ in MODELS:
        merged[f"{prefix}_correct"] = (merged[f"{prefix}_pred"] == merged["label"]).astype(int)
        merged[f"{prefix}_confidence"] = (merged[f"{prefix}_prob"].astype(float) - 0.5).abs()
    merged["adaptive_false_negative"] = ((merged["label"] == 1) & (merged["adaptive_pred"] == 0)).astype(int)
    merged["adaptive_false_positive"] = ((merged["label"] == 0) & (merged["adaptive_pred"] == 1)).astype(int)
    merged["adaptive_error_type"] = "true_negative"
    merged.loc[(merged["label"] == 1) & (merged["adaptive_pred"] == 1), "adaptive_error_type"] = "true_positive"
    merged.loc[(merged["label"] == 1) & (merged["adaptive_pred"] == 0), "adaptive_error_type"] = "false_negative"
    merged.loc[(merged["label"] == 0) & (merged["adaptive_pred"] == 1), "adaptive_error_type"] = "false_positive"
    merged["adaptive_alpha_mean"] = merged[["adaptive_fractional_alpha_y", "adaptive_fractional_alpha_e"]].mean(axis=1)
    merged["adaptive_alpha_abs_diff"] = (
        merged["adaptive_fractional_alpha_y"] - merged["adaptive_fractional_alpha_e"]
    ).abs()
    for comparator in ["contour", "f2mi"]:
        merged[f"adaptive_vs_{comparator}"] = "same_decision"
        merged.loc[
            (merged["adaptive_correct"] == 1) & (merged[f"{comparator}_correct"] == 0),
            f"adaptive_vs_{comparator}",
        ] = "adaptive_only_correct"
        merged.loc[
            (merged["adaptive_correct"] == 0) & (merged[f"{comparator}_correct"] == 1),
            f"adaptive_vs_{comparator}",
        ] = f"{comparator}_only_correct"
        merged.loc[
            (merged["adaptive_correct"] == 0) & (merged[f"{comparator}_correct"] == 0),
            f"adaptive_vs_{comparator}",
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
        row = {
            "outer_fold": int(fold),
            "n": int(len(fold_df)),
            "positive_n": int(y.sum()),
            "negative_n": int((y == 0).sum()),
        }
        for prefix, _ in MODELS:
            metrics = compute_metrics(y, fold_df[f"{prefix}_prob"].to_numpy(), fold_df[f"{prefix}_pred"].to_numpy())
            for metric in METRICS:
                row[f"{prefix}_{metric}"] = metrics[metric]
            for count_col in ["tn", "fp", "fn", "tp"]:
                row[f"{prefix}_{count_col}"] = metrics[count_col]
        for metric in MAIN_METRICS:
            row[f"adaptive_minus_contour_{metric}"] = row[f"adaptive_{metric}"] - row[f"contour_{metric}"]
            row[f"adaptive_minus_f2mi_{metric}"] = row[f"adaptive_{metric}"] - row[f"f2mi_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.asarray(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    return np.concatenate([np.asarray(groups[sid], dtype=int) for sid in sampled_ids])


def paired_bootstrap(
    df: pd.DataFrame,
    target: str,
    comparator: str,
    n_iters: int,
    seed: int,
) -> pd.DataFrame:
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
    adaptive_correct = df["adaptive_correct"].astype(bool)
    comparator_correct = df[f"{comparator}_correct"].astype(bool)
    rows = [
        {"category": "both_correct", "count": int((adaptive_correct & comparator_correct).sum())},
        {"category": "adaptive_only_correct", "count": int((adaptive_correct & ~comparator_correct).sum())},
        {"category": f"{comparator}_only_correct", "count": int((~adaptive_correct & comparator_correct).sum())},
        {"category": "both_wrong", "count": int((~adaptive_correct & ~comparator_correct).sum())},
        {
            "category": f"net_correct_gain_adaptive_minus_{comparator}",
            "count": int((adaptive_correct & ~comparator_correct).sum() - (~adaptive_correct & comparator_correct).sum()),
        },
        {
            "category": f"adaptive_fn_minus_{comparator}_fn",
            "count": int(df["adaptive_false_negative"].sum() - (((df["label"] == 1) & (df[f"{comparator}_pred"] == 0)).sum())),
        },
        {
            "category": f"adaptive_fp_minus_{comparator}_fp",
            "count": int(df["adaptive_false_positive"].sum() - (((df["label"] == 0) & (df[f"{comparator}_pred"] == 1)).sum())),
        },
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
            *ADAPTIVE_EXTRA_COLS,
            "adaptive_alpha_mean",
            "adaptive_alpha_abs_diff",
            "label",
            "adaptive_prob",
            "adaptive_confidence",
            "adaptive_correct",
            "adaptive_false_negative",
            "adaptive_false_positive",
            "contour_correct",
            "f2mi_correct",
            "contour_prob",
            "f2mi_prob",
        ]
    ].copy()
    corr = corr_df.corr(method="spearman")
    sources = [
        *ADAPTIVE_EXTRA_COLS,
        "adaptive_alpha_mean",
        "adaptive_alpha_abs_diff",
    ]
    targets = [
        "label",
        "adaptive_prob",
        "adaptive_confidence",
        "adaptive_correct",
        "adaptive_false_negative",
        "adaptive_false_positive",
        "contour_correct",
        "f2mi_correct",
        "contour_prob",
        "f2mi_prob",
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
        prob_val = val_fold["adaptive_prob"].to_numpy()
        y_test = test_fold["label"].to_numpy()
        prob_test = test_fold["adaptive_prob"].to_numpy()

        fixed_val = metrics_at_threshold(y_val, prob_val, 0.5)
        fixed_test = metrics_at_threshold(y_test, prob_test, 0.5)
        row = {
            "outer_fold": int(fold),
            "objective": "fixed_0_5",
            "threshold": 0.5,
            "val_objective_score": fixed_val["f1"],
        }
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


def write_report(
    output_dir: Path,
    test_df: pd.DataFrame,
    comparison: pd.DataFrame,
    fold_comparison: pd.DataFrame,
    boot_vs_contour: pd.DataFrame,
    boot_vs_f2mi: pd.DataFrame,
    decisions_vs_contour: pd.DataFrame,
    decisions_vs_f2mi: pd.DataFrame,
    router_by_error: pd.DataFrame,
    router_by_fold: pd.DataFrame,
    correlations: pd.DataFrame,
    threshold_by_fold: pd.DataFrame,
    threshold_sum: pd.DataFrame,
) -> None:
    def model_row(prefix: str) -> pd.Series:
        return comparison[comparison["model"] == prefix].iloc[0]

    def fold_mean(prefix: str, metric: str) -> tuple[float, float]:
        values = fold_comparison[f"{prefix}_{metric}"].astype(float)
        return float(values.mean()), float(values.std(ddof=1))

    def boot_line(bootstrap: pd.DataFrame, metric: str) -> str:
        row = bootstrap[bootstrap["metric"] == metric].iloc[0]
        return (
            f"{row['diff_target_minus_comparator']:+.4f} "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], p={row['bootstrap_p_two_sided']:.3f}"
        )

    def decision_value(table: pd.DataFrame, category: str) -> int:
        match = table[table["category"] == category]
        return int(match["count"].iloc[0]) if len(match) else 0

    contour = model_row("contour")
    f2mi = model_row("f2mi")
    adaptive = model_row("adaptive")
    fixed_summary = threshold_sum[threshold_sum["objective"] == "fixed_0_5"].iloc[0]
    f1_summary = threshold_sum[threshold_sum["objective"] == "f1"].iloc[0]
    bal_summary = threshold_sum[threshold_sum["objective"] == "balanced_acc"].iloc[0]

    router_std_cols = [f"{col}_std" for col in ADAPTIVE_ROUTER_COLS]
    router_std_mean = router_by_fold[router_std_cols].mean().to_dict()
    max_router_std = max(router_std_mean.values())

    fn_count = int(test_df["adaptive_false_negative"].sum())
    fp_count = int(test_df["adaptive_false_positive"].sum())
    threshold_f1_gain = float(f1_summary["test_f1_mean"] - fixed_summary["test_f1_mean"])
    threshold_recall_gain = float(f1_summary["test_recall_mean"] - fixed_summary["test_recall_mean"])
    f1_vs_contour = float(f1_summary["test_f1_mean"] - fold_mean("contour", "f1")[0])

    if threshold_f1_gain >= 0.01 or threshold_recall_gain >= 0.03:
        threshold_decision = (
            "validation-based threshold tuning can recover part of the recall/F1 loss, "
            "so the 0.5 decision threshold is one contributor."
        )
    else:
        threshold_decision = (
            "validation-based threshold tuning does not materially recover F1/recall, "
            "so the issue is mainly architectural or training-objective related."
        )

    if f1_vs_contour >= 0:
        next_decision = (
            "009002 can be kept as a candidate if reported with validation-tuned thresholds, "
            "but fixed-threshold and tuned-threshold results must both be disclosed."
        )
    else:
        next_decision = (
            "009002 should not be used as the final main model. The next experiment should keep "
            "sample-adaptive router inputs but redesign the loss/checkpoint rule to preserve recall."
        )

    if max_router_std > 0.01:
        router_decision = "sample-adaptive routing is active; router weights vary within fold."
    else:
        router_decision = "router still behaves like a fold-level prior; sample-adaptive routing is not established."

    top_corr = correlations.copy()
    top_corr["abs_r"] = top_corr["spearman_r"].abs()
    top_corr = top_corr.sort_values("abs_r", ascending=False).head(12)

    lines = [
        "# 002002003009002001 Router Error and Threshold Analysis",
        "",
        "## Scope",
        "",
        "- Target: `002002003009002 sample-adaptive-router F2MI`",
        "- Comparators: `002002002 contour auxiliary` and `002002003009 original F2MI`",
        "- Main comparison split: test only",
        "- Threshold tuning: threshold selected on validation split within each fold, then applied to the same fold's test split",
        f"- Test rows: {len(test_df)}",
        f"- Unique sample_id: {test_df['sample_id'].nunique()}",
        "",
        "## Pooled Test Comparison",
        "",
        "| Model | AUC | Acc | Pre | Recall | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [contour, f2mi, adaptive]:
        lines.append(
            f"| {row['model_label']} | {row['auc']:.4f} | {row['acc']:.4f} | {row['pre']:.4f} | "
            f"{row['recall']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} |"
        )

    lines += [
        "",
        "## Fold-Mean Test Comparison",
        "",
        "| Model | AUC mean | Acc mean | Pre mean | Recall mean | F1 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for prefix, label in MODELS:
        auc_mean, _ = fold_mean(prefix, "auc")
        acc_mean, _ = fold_mean(prefix, "acc")
        pre_mean, _ = fold_mean(prefix, "pre")
        recall_mean, _ = fold_mean(prefix, "recall")
        f1_mean, _ = fold_mean(prefix, "f1")
        lines.append(
            f"| {label} | {auc_mean:.4f} | {acc_mean:.4f} | {pre_mean:.4f} | {recall_mean:.4f} | {f1_mean:.4f} |"
        )

    lines += [
        "",
        "## Paired Bootstrap",
        "",
        "Adaptive minus contour auxiliary:",
        "",
        "| Metric | Difference [95% CI], p |",
        "|---|---:|",
    ]
    for metric in ["auc", "acc", "pre", "recall", "f1"]:
        lines.append(f"| {metric} | {boot_line(boot_vs_contour, metric)} |")

    lines += [
        "",
        "Adaptive minus original F2MI:",
        "",
        "| Metric | Difference [95% CI], p |",
        "|---|---:|",
    ]
    for metric in ["auc", "acc", "pre", "recall", "f1"]:
        lines.append(f"| {metric} | {boot_line(boot_vs_f2mi, metric)} |")

    lines += [
        "",
        "## Decision Counts",
        "",
        "Compared with contour auxiliary:",
        "",
        f"- Both correct: {decision_value(decisions_vs_contour, 'both_correct')}",
        f"- Adaptive only correct: {decision_value(decisions_vs_contour, 'adaptive_only_correct')}",
        f"- Contour only correct: {decision_value(decisions_vs_contour, 'contour_only_correct')}",
        f"- Both wrong: {decision_value(decisions_vs_contour, 'both_wrong')}",
        f"- Net correct gain: {decision_value(decisions_vs_contour, 'net_correct_gain_adaptive_minus_contour')}",
        f"- Adaptive FN minus contour FN: {decision_value(decisions_vs_contour, 'adaptive_fn_minus_contour_fn')}",
        f"- Adaptive FP minus contour FP: {decision_value(decisions_vs_contour, 'adaptive_fp_minus_contour_fp')}",
        "",
        "Compared with original F2MI:",
        "",
        f"- Both correct: {decision_value(decisions_vs_f2mi, 'both_correct')}",
        f"- Adaptive only correct: {decision_value(decisions_vs_f2mi, 'adaptive_only_correct')}",
        f"- Original F2MI only correct: {decision_value(decisions_vs_f2mi, 'f2mi_only_correct')}",
        f"- Both wrong: {decision_value(decisions_vs_f2mi, 'both_wrong')}",
        f"- Net correct gain: {decision_value(decisions_vs_f2mi, 'net_correct_gain_adaptive_minus_f2mi')}",
        f"- Adaptive FN minus original F2MI FN: {decision_value(decisions_vs_f2mi, 'adaptive_fn_minus_f2mi_fn')}",
        f"- Adaptive FP minus original F2MI FP: {decision_value(decisions_vs_f2mi, 'adaptive_fp_minus_f2mi_fp')}",
        "",
        "## Router Mechanism",
        "",
        router_decision,
        "",
        "Mean within-fold router std:",
        "",
    ]
    for key, value in router_std_mean.items():
        lines.append(f"- {key}: {value:.4f}")

    lines += [
        "",
        "Adaptive error counts:",
        "",
        f"- False negatives: {fn_count}",
        f"- False positives: {fp_count}",
        "",
        "Router/error-type summary is saved in `router_by_error_type.csv`.",
        "",
        "Top Spearman correlations involving router/alpha variables:",
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
        f"- F1-tuned threshold mean test F1 change vs fixed 0.5: {threshold_f1_gain:+.4f}",
        f"- F1-tuned threshold mean test recall change vs fixed 0.5: {threshold_recall_gain:+.4f}",
        f"- F1-tuned threshold mean test F1 change vs contour fold-mean F1: {f1_vs_contour:+.4f}",
        f"- Decision: {threshold_decision}",
        "",
        "## Overall Decision",
        "",
        next_decision,
        "",
        "Recommended next experiment:",
        "",
        "```text",
        "002002003009003 Recall-preserving Sample-Adaptive F2MI",
        "```",
        "",
        "Design direction:",
        "",
        "- keep the 009002 sample-adaptive router input;",
        "- remove or greatly reduce router entropy/diversity losses;",
        "- keep router std as an analysis metric instead of forcing it through the loss;",
        "- consider checkpoint selection by validation F1 or a joint AUC/F1 criterion;",
        "- report both fixed 0.5 threshold and validation-tuned threshold.",
        "",
        "## Output Files",
        "",
        "- `merged_predictions_all_splits.csv`",
        "- `model_comparison_test.csv`",
        "- `model_comparison_by_fold_test.csv`",
        "- `paired_bootstrap_vs_002002002.csv`",
        "- `paired_bootstrap_vs_002002003009.csv`",
        "- `decision_counts_vs_002002002.csv`",
        "- `decision_counts_vs_002002003009.csv`",
        "- `router_by_error_type.csv`",
        "- `router_by_fold.csv`",
        "- `router_correlations.csv`",
        "- `threshold_tuning_by_fold.csv`",
        "- `threshold_tuning_summary.csv`",
        "- `adaptive_false_negative_cases.csv`",
        "- `adaptive_false_positive_cases.csv`",
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
    boot_vs_contour = paired_bootstrap(test_df, "adaptive", "contour", args.bootstrap_iters, args.seed)
    boot_vs_f2mi = paired_bootstrap(test_df, "adaptive", "f2mi", args.bootstrap_iters, args.seed + 1)
    decisions_vs_contour = decision_counts(test_df, "contour")
    decisions_vs_f2mi = decision_counts(test_df, "f2mi")

    router_cols = [
        *ADAPTIVE_EXTRA_COLS,
        "adaptive_alpha_mean",
        "adaptive_alpha_abs_diff",
        "adaptive_prob",
        "adaptive_confidence",
    ]
    router_overall = summarize_numeric(test_df, router_cols)
    router_by_error = summarize_numeric(test_df, router_cols, ["adaptive_error_type"])
    router_by_fold = summarize_numeric(test_df, router_cols, ["outer_fold"])
    router_by_label_correctness = summarize_numeric(test_df, router_cols, ["label", "adaptive_correct"])
    router_by_vs_contour = summarize_numeric(test_df, router_cols, ["adaptive_vs_contour"])
    router_by_vs_f2mi = summarize_numeric(test_df, router_cols, ["adaptive_vs_f2mi"])
    correlations = router_correlations(test_df)
    thresholds = threshold_tuning(validation_df, test_df)
    threshold_sum = threshold_summary(thresholds)

    false_negatives = test_df[test_df["adaptive_false_negative"] == 1].sort_values(
        ["outer_fold", "adaptive_prob", "row_id"]
    )
    false_positives = test_df[test_df["adaptive_false_positive"] == 1].sort_values(
        ["outer_fold", "adaptive_prob", "row_id"],
        ascending=[True, False, True],
    )
    adaptive_only_vs_contour = test_df[test_df["adaptive_vs_contour"] == "adaptive_only_correct"].copy()
    contour_only_vs_adaptive = test_df[test_df["adaptive_vs_contour"] == "contour_only_correct"].copy()
    adaptive_only_vs_f2mi = test_df[test_df["adaptive_vs_f2mi"] == "adaptive_only_correct"].copy()
    f2mi_only_vs_adaptive = test_df[test_df["adaptive_vs_f2mi"] == "f2mi_only_correct"].copy()

    merged.to_csv(args.output_dir / "merged_predictions_all_splits.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(args.output_dir / "model_comparison_test.csv", index=False, encoding="utf-8-sig")
    fold_comparison.to_csv(args.output_dir / "model_comparison_by_fold_test.csv", index=False, encoding="utf-8-sig")
    boot_vs_contour.to_csv(args.output_dir / "paired_bootstrap_vs_002002002.csv", index=False, encoding="utf-8-sig")
    boot_vs_f2mi.to_csv(args.output_dir / "paired_bootstrap_vs_002002003009.csv", index=False, encoding="utf-8-sig")
    decisions_vs_contour.to_csv(args.output_dir / "decision_counts_vs_002002002.csv", index=False, encoding="utf-8-sig")
    decisions_vs_f2mi.to_csv(args.output_dir / "decision_counts_vs_002002003009.csv", index=False, encoding="utf-8-sig")
    router_overall.to_csv(args.output_dir / "router_overall_summary.csv", index=False, encoding="utf-8-sig")
    router_by_error.to_csv(args.output_dir / "router_by_error_type.csv", index=False, encoding="utf-8-sig")
    router_by_fold.to_csv(args.output_dir / "router_by_fold.csv", index=False, encoding="utf-8-sig")
    router_by_label_correctness.to_csv(
        args.output_dir / "router_by_label_correctness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    router_by_vs_contour.to_csv(args.output_dir / "router_by_vs_contour.csv", index=False, encoding="utf-8-sig")
    router_by_vs_f2mi.to_csv(args.output_dir / "router_by_vs_f2mi.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(args.output_dir / "router_correlations.csv", index=False, encoding="utf-8-sig")
    thresholds.to_csv(args.output_dir / "threshold_tuning_by_fold.csv", index=False, encoding="utf-8-sig")
    threshold_sum.to_csv(args.output_dir / "threshold_tuning_summary.csv", index=False, encoding="utf-8-sig")
    false_negatives.to_csv(args.output_dir / "adaptive_false_negative_cases.csv", index=False, encoding="utf-8-sig")
    false_positives.to_csv(args.output_dir / "adaptive_false_positive_cases.csv", index=False, encoding="utf-8-sig")
    adaptive_only_vs_contour.to_csv(args.output_dir / "adaptive_only_correct_vs_contour.csv", index=False, encoding="utf-8-sig")
    contour_only_vs_adaptive.to_csv(args.output_dir / "contour_only_correct_vs_adaptive.csv", index=False, encoding="utf-8-sig")
    adaptive_only_vs_f2mi.to_csv(args.output_dir / "adaptive_only_correct_vs_f2mi.csv", index=False, encoding="utf-8-sig")
    f2mi_only_vs_adaptive.to_csv(args.output_dir / "f2mi_only_correct_vs_adaptive.csv", index=False, encoding="utf-8-sig")

    write_report(
        args.output_dir,
        test_df,
        comparison,
        fold_comparison,
        boot_vs_contour,
        boot_vs_f2mi,
        decisions_vs_contour,
        decisions_vs_f2mi,
        router_by_error,
        router_by_fold,
        correlations,
        thresholds,
        threshold_sum,
    )
    print(f"Saved analysis outputs to {args.output_dir}")
    print(comparison[["model", "auc", "acc", "pre", "recall", "f1"]].to_string(index=False))
    print(threshold_sum[["objective", "threshold_mean", "test_recall_mean", "test_f1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()

