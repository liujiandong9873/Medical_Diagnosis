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

ROUTER_COLS = ["router_image", "router_contour", "router_landmark", "router_morphology"]
ALPHA_COLS = ["fractional_alpha_y", "fractional_alpha_e"]


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    experiment_dir = analysis_dir.parent
    base_dir = experiment_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--f2mi-predictions",
        type=Path,
        default=experiment_dir / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--contour-aux-predictions",
        type=Path,
        default=base_dir
        / "002002002_image_contour_auxiliary_supervision"
        / "outputs"
        / "cv_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "009001_outputs")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def clip_prob(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(prob.astype(float), eps, 1.0 - eps)


def specificity_score(y_true: np.ndarray, pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out = {
        "auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
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


def load_paired_predictions(f2mi_path: Path, contour_aux_path: Path) -> pd.DataFrame:
    f2mi = pd.read_csv(f2mi_path)
    contour = pd.read_csv(contour_aux_path)

    f2mi = f2mi[f2mi["split"] == "test"].copy()
    contour = contour[contour["split"] == "test"].copy()

    required_f2mi = {
        "row_id",
        "sample_id",
        "image_file",
        "outer_fold",
        "label",
        "prob_hypertrophy",
        "pred_label",
        *ROUTER_COLS,
        *ALPHA_COLS,
    }
    required_contour = {
        "row_id",
        "sample_id",
        "image_file",
        "outer_fold",
        "label",
        "prob_hypertrophy",
        "pred_label",
    }
    missing_f2mi = sorted(required_f2mi - set(f2mi.columns))
    missing_contour = sorted(required_contour - set(contour.columns))
    if missing_f2mi:
        raise ValueError(f"Missing F2MI columns: {missing_f2mi}")
    if missing_contour:
        raise ValueError(f"Missing contour-aux columns: {missing_contour}")

    keep_f2mi = [
        "row_id",
        "sample_id",
        "image_file",
        "outer_fold",
        "label",
        "prob_hypertrophy",
        "pred_label",
        *ROUTER_COLS,
        *ALPHA_COLS,
    ]
    keep_contour = [
        "row_id",
        "sample_id",
        "image_file",
        "outer_fold",
        "label",
        "prob_hypertrophy",
        "pred_label",
    ]
    paired = contour[keep_contour].merge(
        f2mi[keep_f2mi],
        on=["row_id", "sample_id", "image_file", "outer_fold", "label"],
        suffixes=("_contour_aux", "_f2mi"),
        validate="one_to_one",
    )
    paired = paired.rename(
        columns={
            "prob_hypertrophy_contour_aux": "contour_aux_prob",
            "pred_label_contour_aux": "contour_aux_pred",
            "prob_hypertrophy_f2mi": "f2mi_prob",
            "pred_label_f2mi": "f2mi_pred",
        }
    )
    paired["label"] = paired["label"].astype(int)
    paired["contour_aux_pred"] = paired["contour_aux_pred"].astype(int)
    paired["f2mi_pred"] = paired["f2mi_pred"].astype(int)
    paired["contour_aux_correct"] = (paired["contour_aux_pred"] == paired["label"]).astype(int)
    paired["f2mi_correct"] = (paired["f2mi_pred"] == paired["label"]).astype(int)
    paired["f2mi_confidence"] = (paired["f2mi_prob"].astype(float) - 0.5).abs()
    paired["contour_aux_confidence"] = (paired["contour_aux_prob"].astype(float) - 0.5).abs()
    paired["alpha_mean"] = paired[ALPHA_COLS].mean(axis=1)
    paired["alpha_abs_diff"] = (paired["fractional_alpha_y"] - paired["fractional_alpha_e"]).abs()

    if len(paired) != len(f2mi) or len(paired) != len(contour):
        raise ValueError(f"Paired row count mismatch: paired={len(paired)}, f2mi={len(f2mi)}, contour={len(contour)}")
    return paired.sort_values(["outer_fold", "row_id"]).reset_index(drop=True)


def model_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    y = paired["label"].to_numpy()
    rows = []
    for model, label, prob_col, pred_col in [
        ("002002002_contour_aux", "002002002 contour auxiliary", "contour_aux_prob", "contour_aux_pred"),
        ("002002003009_f2mi", "002002003009 F2MI-Transformer", "f2mi_prob", "f2mi_pred"),
    ]:
        row = {
            "model": model,
            "model_label": label,
            "n": int(len(paired)),
            "unique_sample_id": int(paired["sample_id"].nunique()),
            "positive_n": int(y.sum()),
            "negative_n": int((y == 0).sum()),
        }
        row.update(compute_metrics(y, paired[prob_col].to_numpy(), paired[pred_col].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def model_comparison_by_fold(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, df in paired.groupby("outer_fold"):
        y = df["label"].to_numpy()
        for model, label, prob_col, pred_col in [
            ("002002002_contour_aux", "002002002 contour auxiliary", "contour_aux_prob", "contour_aux_pred"),
            ("002002003009_f2mi", "002002003009 F2MI-Transformer", "f2mi_prob", "f2mi_pred"),
        ]:
            row = {
                "outer_fold": int(fold),
                "model": model,
                "model_label": label,
                "n": int(len(df)),
                "positive_n": int(y.sum()),
                "negative_n": int((y == 0).sum()),
            }
            row.update(compute_metrics(y, df[prob_col].to_numpy(), df[pred_col].to_numpy()))
            rows.append(row)
    fold_table = pd.DataFrame(rows)
    pivot_rows = []
    for fold, df in fold_table.groupby("outer_fold"):
        contour = df[df["model"] == "002002002_contour_aux"].iloc[0]
        f2mi = df[df["model"] == "002002003009_f2mi"].iloc[0]
        row = {"outer_fold": int(fold)}
        for metric in ["auc", "acc", "balanced_acc", "pre", "recall", "specificity", "f1"]:
            row[f"contour_aux_{metric}"] = float(contour[metric])
            row[f"f2mi_{metric}"] = float(f2mi[metric])
            row[f"diff_f2mi_minus_contour_aux_{metric}"] = float(f2mi[metric] - contour[metric])
        pivot_rows.append(row)
    return pd.DataFrame(pivot_rows)


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.asarray(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    return np.concatenate([np.asarray(groups[sid], dtype=int) for sid in sampled_ids])


def paired_bootstrap(paired: pd.DataFrame, n_iters: int, seed: int) -> pd.DataFrame:
    y = paired["label"].to_numpy()
    contour_metrics = compute_metrics(y, paired["contour_aux_prob"].to_numpy(), paired["contour_aux_pred"].to_numpy())
    f2mi_metrics = compute_metrics(y, paired["f2mi_prob"].to_numpy(), paired["f2mi_pred"].to_numpy())
    observed = {metric: f2mi_metrics[metric] - contour_metrics[metric] for metric in METRICS}

    rng = np.random.default_rng(seed)
    boot_diffs = {metric: [] for metric in METRICS}
    for _ in range(n_iters):
        idx = sample_cluster_indices(paired, rng)
        boot = paired.iloc[idx]
        y_boot = boot["label"].to_numpy()
        if len(np.unique(y_boot)) < 2:
            continue
        contour = compute_metrics(
            y_boot,
            boot["contour_aux_prob"].to_numpy(),
            boot["contour_aux_pred"].to_numpy(),
        )
        f2mi = compute_metrics(
            y_boot,
            boot["f2mi_prob"].to_numpy(),
            boot["f2mi_pred"].to_numpy(),
        )
        for metric in METRICS:
            boot_diffs[metric].append(f2mi[metric] - contour[metric])

    rows = []
    for metric in METRICS:
        values = np.asarray(boot_diffs[metric], dtype=float)
        p_lower = float(np.mean(values <= 0.0))
        p_upper = float(np.mean(values >= 0.0))
        p_two_sided = min(1.0, 2.0 * min(p_lower, p_upper))
        rows.append(
            {
                "target": "002002003009_f2mi",
                "comparator": "002002002_contour_aux",
                "metric": metric,
                "diff_target_minus_comparator": observed[metric],
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_p_two_sided": p_two_sided,
                "bootstrap_n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def decision_counts(paired: pd.DataFrame) -> pd.DataFrame:
    f2mi_correct = paired["f2mi_correct"].astype(bool)
    contour_correct = paired["contour_aux_correct"].astype(bool)
    rows = [
        {"category": "both_correct", "count": int((f2mi_correct & contour_correct).sum())},
        {"category": "f2mi_only_correct", "count": int((f2mi_correct & ~contour_correct).sum())},
        {"category": "contour_aux_only_correct", "count": int((~f2mi_correct & contour_correct).sum())},
        {"category": "both_wrong", "count": int((~f2mi_correct & ~contour_correct).sum())},
        {
            "category": "net_correct_gain_f2mi_minus_contour_aux",
            "count": int((f2mi_correct & ~contour_correct).sum() - ((~f2mi_correct & contour_correct).sum())),
        },
    ]
    return pd.DataFrame(rows)


def describe_columns(df: pd.DataFrame, cols: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        desc = df[cols].agg(["mean", "std", "min", "median", "max"]).T.reset_index()
        return desc.rename(columns={"index": "field"})
    grouped = df.groupby(group_cols)[cols].agg(["mean", "std", "min", "median", "max"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.to_flat_index()]
    return grouped.reset_index()


def router_correlations(paired: pd.DataFrame) -> pd.DataFrame:
    corr_fields = [
        *ROUTER_COLS,
        *ALPHA_COLS,
        "alpha_mean",
        "alpha_abs_diff",
        "label",
        "f2mi_prob",
        "f2mi_confidence",
        "f2mi_correct",
        "contour_aux_correct",
    ]
    corr = paired[corr_fields].corr(method="spearman")
    rows = []
    for source in [*ROUTER_COLS, *ALPHA_COLS, "alpha_mean", "alpha_abs_diff"]:
        for target in ["label", "f2mi_prob", "f2mi_confidence", "f2mi_correct", "contour_aux_correct", *ALPHA_COLS]:
            if source == target:
                continue
            rows.append({"source": source, "target": target, "spearman_r": float(corr.loc[source, target])})
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    paired: pd.DataFrame,
    comparison: pd.DataFrame,
    fold_table: pd.DataFrame,
    bootstrap: pd.DataFrame,
    decisions: pd.DataFrame,
    router_by_fold: pd.DataFrame,
    router_within_fold: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    contour = comparison[comparison["model"] == "002002002_contour_aux"].iloc[0]
    f2mi = comparison[comparison["model"] == "002002003009_f2mi"].iloc[0]

    def metric_line(metric: str) -> str:
        boot = bootstrap[bootstrap["metric"] == metric].iloc[0]
        return (
            f"| {metric} | {contour[metric]:.4f} | {f2mi[metric]:.4f} | "
            f"{boot['diff_target_minus_comparator']:+.4f} | "
            f"[{boot['ci95_low']:.4f}, {boot['ci95_high']:.4f}] | "
            f"{boot['bootstrap_p_two_sided']:.3f} |"
        )

    decision_map = dict(zip(decisions["category"], decisions["count"]))
    router_stds = router_within_fold[[f"{col}_std" for col in ROUTER_COLS]].mean().to_dict()
    max_router_std = max(router_stds.values())

    if max_router_std < 0.01:
        router_decision = "router 主要表现为 fold-level modality prior，而不是明确的样本级动态路由。"
    else:
        router_decision = "router 存在一定样本级变化，但仍需结合 label/correctness 相关性判断其意义。"

    sig_metrics = bootstrap[
        (bootstrap["ci95_low"] > 0.0) | (bootstrap["ci95_high"] < 0.0)
    ]["metric"].tolist()
    if sig_metrics:
        stat_decision = f"paired bootstrap 显示以下指标差异 CI 不跨 0: {', '.join(sig_metrics)}。"
    else:
        stat_decision = "paired bootstrap 未发现主指标差异 CI 不跨 0；不能宣称 F2MI 显著优于 contour auxiliary。"

    lines = [
        "# 002002003009001 Result Analysis and Statistical Validation",
        "",
        "## Analysis Scope",
        "",
        "- Target model: `002002003009 F2MI-Transformer`",
        "- Comparator: `002002002 contour auxiliary ResNet18`",
        "- Split: test only",
        f"- Paired predictions: {len(paired)}",
        f"- Unique sample_id: {paired['sample_id'].nunique()}",
        "- Bootstrap: clustered paired bootstrap by sample_id, 2000 iterations",
        "",
        "## Pooled Model Comparison",
        "",
        "| Metric | 002002002 contour auxiliary | 002002003009 F2MI | Diff | 95% CI | p |",
        "|---|---:|---:|---:|---:|---:|",
        metric_line("auc"),
        metric_line("acc"),
        metric_line("pre"),
        metric_line("recall"),
        metric_line("specificity"),
        metric_line("f1"),
        "",
        "## Decision Counts",
        "",
        f"- Both correct: {decision_map.get('both_correct', 0)}",
        f"- F2MI only correct: {decision_map.get('f2mi_only_correct', 0)}",
        f"- Contour auxiliary only correct: {decision_map.get('contour_aux_only_correct', 0)}",
        f"- Both wrong: {decision_map.get('both_wrong', 0)}",
        f"- Net correct gain: {decision_map.get('net_correct_gain_f2mi_minus_contour_aux', 0)}",
        "",
        "## Fold-Level Summary",
        "",
        "| Fold | F2MI AUC | Contour Aux AUC | AUC Diff | F2MI F1 | Contour Aux F1 | F1 Diff |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in fold_table.iterrows():
        lines.append(
            f"| {int(row['outer_fold'])} | {row['f2mi_auc']:.4f} | {row['contour_aux_auc']:.4f} | "
            f"{row['diff_f2mi_minus_contour_aux_auc']:+.4f} | {row['f2mi_f1']:.4f} | "
            f"{row['contour_aux_f1']:.4f} | {row['diff_f2mi_minus_contour_aux_f1']:+.4f} |"
        )

    lines += [
        "",
        "## Router and Alpha Interpretation",
        "",
        router_decision,
        "",
        "Mean within-fold router std:",
        "",
    ]
    for key, value in router_stds.items():
        lines.append(f"- {key}: {value:.6f}")

    key_corr = correlations[
        correlations["target"].isin(["label", "f2mi_correct", "f2mi_confidence"])
    ].copy()
    key_corr["abs_r"] = key_corr["spearman_r"].abs()
    key_corr = key_corr.sort_values("abs_r", ascending=False).head(10)
    lines += [
        "",
        "Top router/alpha correlations with label, correctness, or confidence:",
        "",
        "| Source | Target | Spearman r |",
        "|---|---|---:|",
    ]
    for _, row in key_corr.iterrows():
        lines.append(f"| {row['source']} | {row['target']} | {row['spearman_r']:.4f} |")

    lines += [
        "",
        "## Route Decision",
        "",
        stat_decision,
        "",
        "Current recommendation:",
        "",
        "```text",
        "Do not immediately retrain or replace the route.",
        "First use this analysis as the gate for 002002003009002.",
        "If continuing F2MI, the next architectural target should be sample-adaptive routing, not a larger model.",
        "```",
        "",
        "## Output Files",
        "",
        "- `model_comparison.csv`",
        "- `model_comparison_by_fold.csv`",
        "- `paired_bootstrap.csv`",
        "- `paired_decision_counts.csv`",
        "- `paired_predictions.csv`",
        "- `f2mi_only_correct_cases.csv`",
        "- `contour_aux_only_correct_cases.csv`",
        "- `both_wrong_cases.csv`",
        "- `router_overall_summary.csv`",
        "- `router_by_fold.csv`",
        "- `router_by_label.csv`",
        "- `router_by_correctness.csv`",
        "- `router_within_fold_std.csv`",
        "- `router_alpha_correlations.csv`",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired = load_paired_predictions(args.f2mi_predictions, args.contour_aux_predictions)
    comparison = model_comparison(paired)
    fold_table = model_comparison_by_fold(paired)
    bootstrap = paired_bootstrap(paired, args.bootstrap_iters, args.seed)
    decisions = decision_counts(paired)

    f2mi_only = paired[(paired["f2mi_correct"] == 1) & (paired["contour_aux_correct"] == 0)].copy()
    contour_only = paired[(paired["f2mi_correct"] == 0) & (paired["contour_aux_correct"] == 1)].copy()
    both_wrong = paired[(paired["f2mi_correct"] == 0) & (paired["contour_aux_correct"] == 0)].copy()

    router_overall = describe_columns(paired, [*ROUTER_COLS, *ALPHA_COLS, "alpha_mean", "alpha_abs_diff"])
    router_by_fold = describe_columns(paired, [*ROUTER_COLS, *ALPHA_COLS, "alpha_mean", "alpha_abs_diff"], ["outer_fold"])
    router_by_label = describe_columns(paired, [*ROUTER_COLS, *ALPHA_COLS, "alpha_mean", "alpha_abs_diff"], ["label"])
    router_by_correctness = describe_columns(
        paired,
        [*ROUTER_COLS, *ALPHA_COLS, "alpha_mean", "alpha_abs_diff"],
        ["f2mi_correct"],
    )
    router_within_fold = paired.groupby("outer_fold")[[*ROUTER_COLS, *ALPHA_COLS]].std().reset_index()
    router_within_fold.columns = [
        "outer_fold",
        *[f"{col}_std" for col in [*ROUTER_COLS, *ALPHA_COLS]],
    ]
    correlations = router_correlations(paired)

    paired.to_csv(args.output_dir / "paired_predictions.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    fold_table.to_csv(args.output_dir / "model_comparison_by_fold.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(args.output_dir / "paired_decision_counts.csv", index=False, encoding="utf-8-sig")
    f2mi_only.to_csv(args.output_dir / "f2mi_only_correct_cases.csv", index=False, encoding="utf-8-sig")
    contour_only.to_csv(args.output_dir / "contour_aux_only_correct_cases.csv", index=False, encoding="utf-8-sig")
    both_wrong.to_csv(args.output_dir / "both_wrong_cases.csv", index=False, encoding="utf-8-sig")
    router_overall.to_csv(args.output_dir / "router_overall_summary.csv", index=False, encoding="utf-8-sig")
    router_by_fold.to_csv(args.output_dir / "router_by_fold.csv", index=False, encoding="utf-8-sig")
    router_by_label.to_csv(args.output_dir / "router_by_label.csv", index=False, encoding="utf-8-sig")
    router_by_correctness.to_csv(args.output_dir / "router_by_correctness.csv", index=False, encoding="utf-8-sig")
    router_within_fold.to_csv(args.output_dir / "router_within_fold_std.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(args.output_dir / "router_alpha_correlations.csv", index=False, encoding="utf-8-sig")

    write_report(
        args.output_dir,
        paired,
        comparison,
        fold_table,
        bootstrap,
        decisions,
        router_by_fold,
        router_within_fold,
        correlations,
    )
    print(f"Saved 002002003009001 analysis outputs to: {args.output_dir}")
    print(comparison[["model", "auc", "acc", "pre", "recall", "f1"]].to_string(index=False))
    print(bootstrap[bootstrap["metric"].isin(["auc", "acc", "pre", "recall", "f1"])].to_string(index=False))


if __name__ == "__main__":
    main()
