from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


MODEL_ORDER = [
    "002002002_contour_aux",
    "f2cg_raw_default",
    "f2cg_beta_default",
    "f2cg_isotonic_default",
]

MODEL_LABELS = {
    "002002002_contour_aux": "002002002 contour auxiliary",
    "f2cg_raw_default": "F2CG raw + 0.5",
    "f2cg_beta_default": "F2CG beta-calibrated + 0.5",
    "f2cg_isotonic_default": "F2CG isotonic-calibrated + 0.5",
}

METRICS = [
    "auc",
    "average_precision",
    "brier",
    "nll",
    "accuracy",
    "balanced_acc",
    "precision",
    "recall",
    "specificity",
    "f1",
]

DIFF_PAIRS = [
    ("f2cg_beta_default", "002002002_contour_aux"),
    ("f2cg_isotonic_default", "002002002_contour_aux"),
    ("f2cg_raw_default", "002002002_contour_aux"),
    ("f2cg_beta_default", "f2cg_raw_default"),
    ("f2cg_isotonic_default", "f2cg_raw_default"),
    ("f2cg_isotonic_default", "f2cg_beta_default"),
]


def parse_args() -> argparse.Namespace:
    exp_dir = Path(__file__).resolve().parents[1]
    base_dir = exp_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contour-aux-predictions",
        type=Path,
        default=base_dir
        / "002002002_image_contour_auxiliary_supervision"
        / "outputs"
        / "cv_predictions.csv",
    )
    parser.add_argument(
        "--calibrated-f2cg-predictions",
        type=Path,
        default=base_dir
        / "002002003003_f2cg_probability_calibration"
        / "outputs"
        / "calibrated_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=exp_dir / "outputs")
    parser.add_argument("--figure-dir", type=Path, default=exp_dir / "figures")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260703)
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
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": specificity_score(y_true, pred),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def calibration_ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(prob[mask])) - float(np.mean(y_true[mask])))
    return float(ece)


def load_analysis_table(contour_path: Path, calibrated_path: Path) -> pd.DataFrame:
    contour = pd.read_csv(contour_path)
    contour = contour[contour["split"] == "test"].copy()
    contour["model"] = "002002002_contour_aux"
    contour["prob"] = contour["prob_hypertrophy"].astype(float)
    contour["pred"] = contour["pred_label"].astype(int)

    f2cg = pd.read_csv(calibrated_path)
    f2cg = f2cg[f2cg["split"] == "test"].copy()
    f2cg = f2cg[f2cg["method"].isin(["raw", "beta", "isotonic"])].copy()
    method_to_model = {
        "raw": "f2cg_raw_default",
        "beta": "f2cg_beta_default",
        "isotonic": "f2cg_isotonic_default",
    }
    f2cg["model"] = f2cg["method"].map(method_to_model)
    f2cg["prob"] = f2cg["calibrated_prob"].astype(float)
    f2cg["pred"] = (f2cg["prob"] >= 0.5).astype(int)

    cols = ["row_id", "sample_id", "image_file", "outer_fold", "label", "model", "prob", "pred"]
    combined = pd.concat([contour[cols], f2cg[cols]], ignore_index=True)
    combined["label"] = combined["label"].astype(int)
    combined["row_id"] = combined["row_id"].astype(int)

    counts = combined.groupby("model")["row_id"].nunique().to_dict()
    expected = counts.get("002002002_contour_aux")
    for model in MODEL_ORDER:
        if counts.get(model) != expected:
            raise ValueError(f"Model {model} has {counts.get(model)} rows, expected {expected}.")
    label_check = combined.pivot_table(index="row_id", columns="model", values="label", aggfunc="first")
    if label_check.nunique(axis=1).max() != 1:
        raise ValueError("Labels are not aligned across paired model predictions.")
    return combined


def pivot_predictions(combined: pd.DataFrame) -> pd.DataFrame:
    meta = combined.drop_duplicates("row_id")[["row_id", "sample_id", "image_file", "outer_fold", "label"]]
    prob = combined.pivot(index="row_id", columns="model", values="prob").add_suffix("_prob")
    pred = combined.pivot(index="row_id", columns="model", values="pred").add_suffix("_pred")
    table = meta.set_index("row_id").join(prob).join(pred).reset_index()
    return table


def metrics_table(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        df = combined[combined["model"] == model].copy()
        row = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "n": int(len(df)),
            "positive_n": int(df["label"].sum()),
            "negative_n": int((df["label"] == 0).sum()),
        }
        row.update(compute_metrics(df["label"].to_numpy(), df["prob"].to_numpy(), df["pred"].to_numpy()))
        row["ece_10"] = calibration_ece(df["label"].to_numpy(), df["prob"].to_numpy(), 10)
        rows.append(row)
    return pd.DataFrame(rows)


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.array(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    index_parts = [np.asarray(groups[sid], dtype=int) for sid in sampled_ids]
    return np.concatenate(index_parts)


def bootstrap_metrics(table: pd.DataFrame, n_iters: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    diff_rows = []
    for i in range(n_iters):
        idx = sample_cluster_indices(table, rng)
        boot = table.iloc[idx].copy()
        y = boot["label"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        iter_metrics: dict[str, dict[str, float]] = {}
        for model in MODEL_ORDER:
            prob = boot[f"{model}_prob"].to_numpy()
            pred = boot[f"{model}_pred"].to_numpy().astype(int)
            model_metrics = compute_metrics(y, prob, pred)
            iter_metrics[model] = model_metrics
            for metric in METRICS:
                rows.append(
                    {
                        "bootstrap_iter": i,
                        "model": model,
                        "metric": metric,
                        "value": model_metrics[metric],
                    }
                )
        for model_a, model_b in DIFF_PAIRS:
            for metric in METRICS:
                diff_rows.append(
                    {
                        "bootstrap_iter": i,
                        "model_a": model_a,
                        "model_b": model_b,
                        "comparison": f"{model_a} - {model_b}",
                        "metric": metric,
                        "diff": iter_metrics[model_a][metric] - iter_metrics[model_b][metric],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(diff_rows)


def summarize_bootstrap(observed: pd.DataFrame, boot: pd.DataFrame, diff_boot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_long = observed.melt(
        id_vars=["model", "model_label"],
        value_vars=METRICS,
        var_name="metric",
        value_name="observed",
    )
    ci = (
        boot.groupby(["model", "metric"])["value"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
    )
    metric_summary = obs_long.merge(ci, on=["model", "metric"], how="left")
    metric_summary["model_label"] = metric_summary["model"].map(MODEL_LABELS)
    metric_summary = metric_summary[["model", "model_label", "metric", "observed", "ci_low", "ci_high"]]

    observed_map = obs_long.set_index(["model", "metric"])["observed"].to_dict()
    diff_rows = []
    for (model_a, model_b, metric), group in diff_boot.groupby(["model_a", "model_b", "metric"]):
        observed_diff = observed_map[(model_a, metric)] - observed_map[(model_b, metric)]
        diffs = group["diff"].to_numpy()
        p_two = 2.0 * min(float(np.mean(diffs <= 0.0)), float(np.mean(diffs >= 0.0)))
        p_two = min(p_two, 1.0)
        diff_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "comparison": f"{model_a} - {model_b}",
                "metric": metric,
                "observed_diff": observed_diff,
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "p_two_sided": p_two,
                "bootstrap_n": int(len(diffs)),
            }
        )
    diff_summary = pd.DataFrame(diff_rows)
    return metric_summary, diff_summary


def calibration_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if high == 1.0:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        rows.append(
            {
                "bin": idx,
                "low": low,
                "high": high,
                "n": int(mask.sum()),
                "mean_probability": float(np.mean(prob[mask])) if mask.any() else np.nan,
                "observed_rate": float(np.mean(y_true[mask])) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_roc_pr(combined: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=170)
    colors = {
        "002002002_contour_aux": "#1f77b4",
        "f2cg_raw_default": "#7f7f7f",
        "f2cg_beta_default": "#2ca02c",
        "f2cg_isotonic_default": "#d62728",
    }
    for model in MODEL_ORDER:
        df = combined[combined["model"] == model]
        y = df["label"].to_numpy()
        prob = df["prob"].to_numpy()
        fpr, tpr, _ = roc_curve(y, prob)
        precision, recall, _ = precision_recall_curve(y, prob)
        axes[0].plot(fpr, tpr, linewidth=2, color=colors[model], label=f"{MODEL_LABELS[model]} ({roc_auc_score(y, prob):.3f})")
        axes[1].plot(recall, precision, linewidth=2, color=colors[model], label=f"{MODEL_LABELS[model]} ({average_precision_score(y, prob):.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curves")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "roc_pr_model_comparison.png")
    plt.close(fig)


def save_calibration(combined: pd.DataFrame, figure_dir: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=170)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="perfect")
    bin_rows = []
    colors = {
        "002002002_contour_aux": "#1f77b4",
        "f2cg_raw_default": "#7f7f7f",
        "f2cg_beta_default": "#2ca02c",
        "f2cg_isotonic_default": "#d62728",
    }
    for model in MODEL_ORDER:
        df = combined[combined["model"] == model]
        bins = calibration_bins(df["label"].to_numpy(), df["prob"].to_numpy())
        bins["model"] = model
        bin_rows.append(bins)
        visible = bins[bins["n"] > 0]
        ax.plot(
            visible["mean_probability"],
            visible["observed_rate"],
            marker="o",
            linewidth=2,
            color=colors[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed hypertrophy rate")
    ax.set_title("Calibration curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_model_comparison.png")
    plt.close(fig)
    return pd.concat(bin_rows, ignore_index=True)


def save_metric_bars(summary: pd.DataFrame, figure_dir: Path) -> None:
    metrics = ["balanced_acc", "f1", "recall", "specificity"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=170)
    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    for i, metric in enumerate(metrics):
        values = [float(summary.loc[summary["model"] == model, metric].iloc[0]) for model in MODEL_ORDER]
        ax.bar(x + (i - 1.5) * width, values, width=width, label=metric, color=colors[i])
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_ylim(0.72, 0.94)
    ax.set_ylabel("Metric")
    ax.set_title("Decision metrics at model-specific default thresholds")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "decision_metric_bars.png")
    plt.close(fig)


def save_confusion_matrices(summary: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=170)
    for ax, model in zip(axes, MODEL_ORDER):
        row = summary[summary["model"] == model].iloc[0]
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(MODEL_LABELS[model], fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7)
        ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=7)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrices.png")
    plt.close(fig)


def fmt_ci(row: pd.Series) -> str:
    return f"{row['observed']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"


def write_report(
    output_dir: Path,
    figure_dir: Path,
    metric_summary: pd.DataFrame,
    diff_summary: pd.DataFrame,
    observed: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    key_metrics = ["auc", "balanced_acc", "recall", "specificity", "f1", "brier", "nll"]
    table_rows = []
    for model in MODEL_ORDER:
        row = {"model": MODEL_LABELS[model]}
        for metric in key_metrics:
            mrow = metric_summary[(metric_summary["model"] == model) & (metric_summary["metric"] == metric)].iloc[0]
            row[metric] = fmt_ci(mrow)
        table_rows.append(row)
    key_table = pd.DataFrame(table_rows)

    key_diffs = diff_summary[
        diff_summary["metric"].isin(["auc", "balanced_acc", "f1", "recall", "specificity", "brier", "nll"])
    ].copy()
    key_diffs["model_a_label"] = key_diffs["model_a"].map(MODEL_LABELS)
    key_diffs["model_b_label"] = key_diffs["model_b"].map(MODEL_LABELS)
    key_diffs = key_diffs[
        [
            "model_a_label",
            "model_b_label",
            "metric",
            "observed_diff",
            "ci_low",
            "ci_high",
            "p_two_sided",
        ]
    ]

    beta = observed[observed["model"] == "f2cg_beta_default"].iloc[0]
    iso = observed[observed["model"] == "f2cg_isotonic_default"].iloc[0]
    aux = observed[observed["model"] == "002002002_contour_aux"].iloc[0]
    lines = [
        "# 002002003004 Calibrated F2CG Statistical Validation",
        "",
        "## Input",
        "",
        f"- 002002002 predictions: `{args.contour_aux_predictions}`",
        f"- Calibrated F2CG predictions: `{args.calibrated_f2cg_predictions}`",
        f"- Bootstrap: clustered paired bootstrap by `sample_id`, n={args.bootstrap_iters}, seed={args.seed}",
        "",
        "## Main Metric Table",
        "",
        key_table.to_markdown(index=False),
        "",
        "## Paired Bootstrap Differences",
        "",
        key_diffs.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Main Interpretation",
        "",
        (
            f"- Beta-calibrated F2CG reaches Balanced Acc {beta['balanced_acc']:.4f} "
            f"and F1 {beta['f1']:.4f}, compared with 002002002 Balanced Acc {aux['balanced_acc']:.4f} "
            f"and F1 {aux['f1']:.4f}."
        ),
        (
            f"- Isotonic-calibrated F2CG reaches Balanced Acc {iso['balanced_acc']:.4f} "
            f"and F1 {iso['f1']:.4f}."
        ),
        "- Bootstrap p-values should be interpreted as exploratory because the calibration procedure was developed after observing earlier results.",
        "- For the manuscript, beta calibration is the more reproducible parametric primary option; isotonic calibration is useful as a sensitivity analysis.",
        "",
        "## Figures",
        "",
        f"- ROC/PR comparison: `{figure_dir / 'roc_pr_model_comparison.png'}`",
        f"- Calibration comparison: `{figure_dir / 'calibration_model_comparison.png'}`",
        f"- Decision metric bars: `{figure_dir / 'decision_metric_bars.png'}`",
        f"- Confusion matrices: `{figure_dir / 'confusion_matrices.png'}`",
        "",
    ]
    (output_dir / "statistical_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    combined = load_analysis_table(args.contour_aux_predictions, args.calibrated_f2cg_predictions)
    paired_table = pivot_predictions(combined)
    observed = metrics_table(combined)
    boot, diff_boot = bootstrap_metrics(paired_table, args.bootstrap_iters, args.seed)
    metric_summary, diff_summary = summarize_bootstrap(observed, boot, diff_boot)
    calib_bins = save_calibration(combined, args.figure_dir)
    save_roc_pr(combined, args.figure_dir)
    save_metric_bars(observed, args.figure_dir)
    save_confusion_matrices(observed, args.figure_dir)

    combined.to_csv(args.output_dir / "paired_model_predictions_long.csv", index=False, encoding="utf-8-sig")
    paired_table.to_csv(args.output_dir / "paired_model_predictions_wide.csv", index=False, encoding="utf-8-sig")
    observed.to_csv(args.output_dir / "observed_metrics.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(args.output_dir / "bootstrap_metric_samples.csv", index=False, encoding="utf-8-sig")
    diff_boot.to_csv(args.output_dir / "paired_bootstrap_difference_samples.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(args.output_dir / "bootstrap_metric_ci.csv", index=False, encoding="utf-8-sig")
    diff_summary.to_csv(args.output_dir / "paired_bootstrap_difference_tests.csv", index=False, encoding="utf-8-sig")
    calib_bins.to_csv(args.output_dir / "calibration_bins.csv", index=False, encoding="utf-8-sig")

    input_summary = {
        "contour_aux_predictions": str(args.contour_aux_predictions),
        "calibrated_f2cg_predictions": str(args.calibrated_f2cg_predictions),
        "paired_rows": int(len(paired_table)),
        "unique_sample_id": int(paired_table["sample_id"].nunique()),
        "positive_n": int(paired_table["label"].sum()),
        "negative_n": int((paired_table["label"] == 0).sum()),
        "bootstrap_iters": int(args.bootstrap_iters),
        "seed": int(args.seed),
        "bootstrap_unit": "sample_id clustered paired bootstrap",
    }
    (args.output_dir / "input_summary.json").write_text(
        json.dumps(input_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir, args.figure_dir, metric_summary, diff_summary, observed, args)

    print("Observed metrics:")
    print(
        observed[
            [
                "model",
                "auc",
                "average_precision",
                "brier",
                "nll",
                "balanced_acc",
                "precision",
                "recall",
                "specificity",
                "f1",
            ]
        ].to_string(index=False)
    )
    print("")
    print("Key paired differences:")
    key = diff_summary[
        diff_summary["metric"].isin(["balanced_acc", "f1", "recall", "specificity", "auc", "brier"])
    ].copy()
    key = key[key["model_a"].isin(["f2cg_beta_default", "f2cg_isotonic_default"])]
    print(key.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")
    print(f"Wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
