from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


BASELINES = {
    "002002001 image-only": {
        "auc": 0.9205,
        "balanced_acc": 0.8366,
        "recall": 0.7997,
        "specificity": 0.8735,
        "f1": 0.8321,
    },
    "002002002 contour auxiliary": {
        "auc": 0.9366,
        "balanced_acc": 0.8609,
        "recall": 0.8383,
        "specificity": 0.8834,
        "f1": 0.8596,
    },
    "002002003001 direct contour fusion": {
        "auc": 0.9236,
        "balanced_acc": 0.8495,
        "recall": 0.8451,
        "specificity": 0.8540,
        "f1": 0.8484,
    },
}

F2CG_FOLD_MEAN_AUC = 0.9456
F2CG_FOLD_AUC_STD = 0.0138


@dataclass
class MethodResult:
    method: str
    fold: int
    validation_prob: np.ndarray
    test_prob: np.ndarray
    parameters: dict[str, float | str]


def parse_args() -> argparse.Namespace:
    exp_dir = Path(__file__).resolve().parents[1]
    pred_path = (
        exp_dir.parent
        / "002002003002_f2cg_gated_fractal_fractional_fusion"
        / "outputs"
        / "cv_predictions.csv"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=pred_path)
    parser.add_argument("--output-dir", type=Path, default=exp_dir / "outputs")
    parser.add_argument("--figure-dir", type=Path, default=exp_dir / "figures")
    parser.add_argument("--recall-target", type=float, default=0.85)
    return parser.parse_args()


def clip_prob(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(prob.astype(float), eps, 1.0 - eps)


def logit(prob: np.ndarray) -> np.ndarray:
    p = clip_prob(prob)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        if not np.any(mask):
            continue
        confidence = float(np.mean(prob[mask]))
        observed = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(confidence - observed)
    return float(ece)


def calibration_bins(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if high == 1.0:
            mask = (prob >= low) & (prob <= high)
        else:
            mask = (prob >= low) & (prob < high)
        if np.any(mask):
            rows.append(
                {
                    "bin": idx,
                    "low": low,
                    "high": high,
                    "n": int(np.sum(mask)),
                    "mean_probability": float(np.mean(prob[mask])),
                    "observed_rate": float(np.mean(y_true[mask])),
                }
            )
        else:
            rows.append(
                {
                    "bin": idx,
                    "low": low,
                    "high": high,
                    "n": 0,
                    "mean_probability": float("nan"),
                    "observed_rate": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def probability_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "brier": brier_score(y_true, prob),
        "nll": float(log_loss(y_true, clip_prob(prob), labels=[0, 1])),
        "ece_10": expected_calibration_error(y_true, prob, n_bins=10),
        "prob_mean": float(np.mean(prob)),
        "prob_std": float(np.std(prob)),
    }


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return classification_metrics_from_pred(y_true, prob, pred, threshold)


def classification_metrics_from_pred(
    y_true: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    threshold: float = float("nan"),
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "youden": float(recall + specificity - 1.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "positive_n": int(np.sum(y_true == 1)),
        "negative_n": int(np.sum(y_true == 0)),
    }


def threshold_grid(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    rows = [classification_metrics(y_true, prob, threshold) for threshold in np.linspace(0, 1, 1001)]
    return pd.DataFrame(rows)


def select_threshold(grid: pd.DataFrame, rule: str, recall_target: float) -> float:
    if rule == "youden":
        row = grid.sort_values(["youden", "balanced_acc", "f1", "threshold"], ascending=False).iloc[0]
    elif rule == "balanced_acc":
        row = grid.sort_values(["balanced_acc", "youden", "f1", "threshold"], ascending=False).iloc[0]
    elif rule == "f1":
        row = grid.sort_values(["f1", "balanced_acc", "youden", "threshold"], ascending=False).iloc[0]
    elif rule == "recall_target":
        candidates = grid[grid["recall"] >= recall_target]
        if candidates.empty:
            row = grid.sort_values(["recall", "balanced_acc", "f1", "threshold"], ascending=False).iloc[0]
        else:
            row = candidates.sort_values(
                ["specificity", "balanced_acc", "f1", "threshold"],
                ascending=False,
            ).iloc[0]
    else:
        raise ValueError(f"Unknown threshold rule: {rule}")
    return float(row["threshold"])


def fit_temperature(val_y: np.ndarray, val_prob: np.ndarray) -> tuple[float, np.ndarray]:
    val_logits = logit(val_prob)
    candidate_t = np.concatenate(
        [
            np.linspace(0.05, 1.0, 96),
            np.linspace(1.05, 5.0, 80),
            np.linspace(5.1, 20.0, 60),
        ]
    )
    losses = []
    for temp in candidate_t:
        calibrated = sigmoid(val_logits / temp)
        losses.append(log_loss(val_y, clip_prob(calibrated), labels=[0, 1]))
    best_idx = int(np.argmin(losses))
    return float(candidate_t[best_idx]), np.asarray(losses)


def fit_methods_for_fold(fold: int, val_df: pd.DataFrame, test_df: pd.DataFrame) -> list[MethodResult]:
    val_y = val_df["label"].astype(int).to_numpy()
    val_prob = val_df["prob_hypertrophy"].astype(float).to_numpy()
    test_prob = test_df["prob_hypertrophy"].astype(float).to_numpy()
    val_logits = logit(val_prob).reshape(-1, 1)
    test_logits = logit(test_prob).reshape(-1, 1)

    results = [
        MethodResult(
            method="raw",
            fold=fold,
            validation_prob=val_prob,
            test_prob=test_prob,
            parameters={"kind": "identity"},
        )
    ]

    temp, _ = fit_temperature(val_y, val_prob)
    results.append(
        MethodResult(
            method="temperature",
            fold=fold,
            validation_prob=sigmoid(val_logits.ravel() / temp),
            test_prob=sigmoid(test_logits.ravel() / temp),
            parameters={"temperature": temp},
        )
    )

    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    platt.fit(val_logits, val_y)
    results.append(
        MethodResult(
            method="platt",
            fold=fold,
            validation_prob=platt.predict_proba(val_logits)[:, 1],
            test_prob=platt.predict_proba(test_logits)[:, 1],
            parameters={
                "coef": float(platt.coef_[0, 0]),
                "intercept": float(platt.intercept_[0]),
            },
        )
    )

    beta_x_val = np.column_stack([np.log(clip_prob(val_prob)), np.log(1.0 - clip_prob(val_prob))])
    beta_x_test = np.column_stack([np.log(clip_prob(test_prob)), np.log(1.0 - clip_prob(test_prob))])
    beta = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    beta.fit(beta_x_val, val_y)
    results.append(
        MethodResult(
            method="beta",
            fold=fold,
            validation_prob=beta.predict_proba(beta_x_val)[:, 1],
            test_prob=beta.predict_proba(beta_x_test)[:, 1],
            parameters={
                "coef_log_p": float(beta.coef_[0, 0]),
                "coef_log_1_minus_p": float(beta.coef_[0, 1]),
                "intercept": float(beta.intercept_[0]),
            },
        )
    )

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(val_prob, val_y)
    results.append(
        MethodResult(
            method="isotonic",
            fold=fold,
            validation_prob=isotonic.predict(val_prob),
            test_prob=isotonic.predict(test_prob),
            parameters={"kind": "isotonic"},
        )
    )
    return results


def build_calibrated_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    parameter_rows = []
    for fold, val_df in df[df["split"] == "validation"].groupby("outer_fold"):
        test_df = df[(df["split"] == "test") & (df["outer_fold"] == fold)].copy()
        fold = int(fold)
        methods = fit_methods_for_fold(fold, val_df.copy(), test_df)
        for result in methods:
            parameter_rows.append({"method": result.method, "outer_fold": fold, **result.parameters})
            for split, split_df, prob in [
                ("validation", val_df.copy(), result.validation_prob),
                ("test", test_df.copy(), result.test_prob),
            ]:
                out = split_df.copy()
                out["method"] = result.method
                out["calibrated_prob"] = prob
                out["raw_prob"] = out["prob_hypertrophy"].astype(float)
                out["calibrated_pred_0_5"] = (out["calibrated_prob"] >= 0.5).astype(int)
                out["split"] = split
                prediction_rows.append(out)
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(parameter_rows)


def pooled_probability_summary(cal_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, method_df in cal_df[cal_df["split"] == "test"].groupby("method"):
        y = method_df["label"].astype(int).to_numpy()
        prob = method_df["calibrated_prob"].astype(float).to_numpy()
        row = {"method": method}
        row.update(probability_metrics(y, prob))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("brier")


def fold_probability_summary(cal_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, fold), method_df in cal_df[cal_df["split"] == "test"].groupby(["method", "outer_fold"]):
        y = method_df["label"].astype(int).to_numpy()
        prob = method_df["calibrated_prob"].astype(float).to_numpy()
        row = {"method": method, "outer_fold": int(fold)}
        row.update(probability_metrics(y, prob))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "outer_fold"])


def pooled_decision_summary(cal_df: pd.DataFrame, recall_target: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = [
        ("default_0.5", "fixed"),
        ("validation_youden", "youden"),
        ("validation_balanced_acc", "balanced_acc"),
        ("validation_f1", "f1"),
        (f"validation_recall_{recall_target:.2f}", "recall_target"),
    ]
    pooled_rows = []
    fold_rows = []
    all_method_parts = []
    for method in sorted(cal_df["method"].unique()):
        for fold, val_df in cal_df[(cal_df["method"] == method) & (cal_df["split"] == "validation")].groupby("outer_fold"):
            test_df = cal_df[
                (cal_df["method"] == method)
                & (cal_df["split"] == "test")
                & (cal_df["outer_fold"] == fold)
            ].copy()
            if test_df.empty:
                continue
            val_y = val_df["label"].astype(int).to_numpy()
            val_prob = val_df["calibrated_prob"].astype(float).to_numpy()
            test_y = test_df["label"].astype(int).to_numpy()
            test_prob = test_df["calibrated_prob"].astype(float).to_numpy()
            grid = threshold_grid(val_y, val_prob)
            for selection, rule in rules:
                threshold = 0.5 if rule == "fixed" else select_threshold(grid, rule, recall_target)
                metrics = classification_metrics(test_y, test_prob, threshold)
                fold_rows.append(
                    {
                        "method": method,
                        "selection": selection,
                        "outer_fold": int(fold),
                        **metrics,
                    }
                )
                part = test_df[["row_id", "sample_id", "label", "calibrated_prob"]].copy()
                part["method"] = method
                part["selection"] = selection
                part["threshold"] = threshold
                part["pred"] = (part["calibrated_prob"] >= threshold).astype(int)
                all_method_parts.append(part)
    if not all_method_parts:
        raise ValueError("No calibrated decision predictions were produced.")
    all_parts = pd.concat(all_method_parts, ignore_index=True)
    for (method, selection), group in all_parts.groupby(["method", "selection"]):
        y = group["label"].astype(int).to_numpy()
        prob = group["calibrated_prob"].astype(float).to_numpy()
        pred = group["pred"].astype(int).to_numpy()
        row = {"method": method, "selection": selection}
        row.update(probability_metrics(y, prob))
        row.update(classification_metrics_from_pred(y, prob, pred))
        row["threshold_mean"] = float(group["threshold"].mean())
        row["threshold_std"] = float(group["threshold"].std(ddof=0))
        pooled_rows.append(row)
    return (
        pd.DataFrame(pooled_rows).sort_values(["balanced_acc", "f1"], ascending=False),
        pd.DataFrame(fold_rows).sort_values(["method", "selection", "outer_fold"]),
    )


def save_calibration_curves(cal_df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    test_df = cal_df[cal_df["split"] == "test"].copy()
    bin_rows = []
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1, label="perfect")
    colors = {
        "raw": "#1f77b4",
        "temperature": "#ff7f0e",
        "platt": "#2ca02c",
        "beta": "#9467bd",
        "isotonic": "#d62728",
    }
    for method, method_df in test_df.groupby("method"):
        y = method_df["label"].astype(int).to_numpy()
        prob = method_df["calibrated_prob"].astype(float).to_numpy()
        bins = calibration_bins(y, prob)
        bins["method"] = method
        bin_rows.append(bins)
        visible = bins[bins["n"] > 0]
        ax.plot(
            visible["mean_probability"],
            visible["observed_rate"],
            marker="o",
            linewidth=1.8,
            label=method,
            color=colors.get(method),
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed hypertrophy rate")
    ax.set_title("OOF calibration curves")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_curves.png")
    plt.close(fig)
    return pd.concat(bin_rows, ignore_index=True)


def save_metric_bars(prob_summary: pd.DataFrame, decision_summary: pd.DataFrame, fig_dir: Path) -> None:
    method_order = prob_summary.sort_values("brier")["method"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    brier = prob_summary.set_index("method").loc[method_order, "brier"]
    ece = prob_summary.set_index("method").loc[method_order, "ece_10"]
    x = np.arange(len(method_order))
    axes[0].bar(x - 0.18, brier, width=0.36, label="Brier", color="#1f77b4")
    axes[0].bar(x + 0.18, ece, width=0.36, label="ECE", color="#ff7f0e")
    axes[0].set_xticks(x, method_order, rotation=30, ha="right")
    axes[0].set_title("Probability calibration metrics")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    best_decision = (
        decision_summary.sort_values(["balanced_acc", "f1"], ascending=False)
        .drop_duplicates("method")
        .set_index("method")
        .loc[method_order]
    )
    axes[1].bar(x - 0.18, best_decision["balanced_acc"], width=0.36, label="Balanced Acc", color="#2ca02c")
    axes[1].bar(x + 0.18, best_decision["f1"], width=0.36, label="F1", color="#d62728")
    axes[1].axhline(BASELINES["002002002 contour auxiliary"]["balanced_acc"], color="#2ca02c", linestyle="--", linewidth=1)
    axes[1].axhline(BASELINES["002002002 contour auxiliary"]["f1"], color="#d62728", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, method_order, rotation=30, ha="right")
    axes[1].set_ylim(0.80, 0.90)
    axes[1].set_title("Best validation-selected decision metrics")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_metric_bars.png")
    plt.close(fig)


def save_roc_pr(cal_df: pd.DataFrame, fig_dir: Path) -> None:
    test_df = cal_df[cal_df["split"] == "test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    for method, method_df in test_df.groupby("method"):
        y = method_df["label"].astype(int).to_numpy()
        prob = method_df["calibrated_prob"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, prob)
        precision, recall, _ = precision_recall_curve(y, prob)
        axes[0].plot(fpr, tpr, linewidth=1.7, label=f"{method} AUC={roc_auc_score(y, prob):.3f}")
        axes[1].plot(recall, precision, linewidth=1.7, label=f"{method} AP={average_precision_score(y, prob):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("OOF ROC after calibration")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("OOF PR after calibration")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_pr_after_calibration.png")
    plt.close(fig)


def write_report(
    output_dir: Path,
    figure_dir: Path,
    pred_path: Path,
    prob_summary: pd.DataFrame,
    decision_summary: pd.DataFrame,
    parameter_df: pd.DataFrame,
) -> None:
    prob_cols = ["method", "auc", "average_precision", "brier", "nll", "ece_10", "prob_mean", "prob_std"]
    decision_cols = [
        "method",
        "selection",
        "threshold_mean",
        "threshold_std",
        "balanced_acc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    best_prob = prob_summary.sort_values(["brier", "ece_10"]).iloc[0]
    best_decision = decision_summary.sort_values(["balanced_acc", "f1"], ascending=False).iloc[0]
    default_rows = decision_summary[decision_summary["selection"] == "default_0.5"].sort_values("balanced_acc", ascending=False)

    lines = [
        "# 002002003003 F2CG Probability Calibration Report",
        "",
        "## Input",
        "",
        f"- Prediction file: `{pred_path}`",
        "- Calibration is fitted separately for each outer fold using that fold's validation predictions.",
        "- Each fitted calibrator is then fixed and applied to the corresponding fold's test predictions.",
        "",
        "## Probability Calibration Summary",
        "",
        prob_summary[prob_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Best probability calibration by Brier score: `{best_prob['method']}`.",
        "",
        "## Decision Summary",
        "",
        decision_summary[decision_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Default-Threshold Rows",
        "",
        default_rows[decision_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Baseline Comparison",
        "",
        "| Model / setting | AUC | Balanced Acc | Recall | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in BASELINES.items():
        lines.append(
            f"| {name} | {metrics['auc']:.4f} | {metrics['balanced_acc']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['specificity']:.4f} | {metrics['f1']:.4f} |"
        )
    auc_cell = f"{F2CG_FOLD_MEAN_AUC:.4f} fold; {best_decision['auc']:.4f} pooled"
    lines.append(
        f"| 002002003003 best calibrated decision | {auc_cell} | {best_decision['balanced_acc']:.4f} | "
        f"{best_decision['recall']:.4f} | {best_decision['specificity']:.4f} | {best_decision['f1']:.4f} |"
    )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Probability calibration and threshold calibration answer different questions.",
        "- A calibrator can improve Brier/ECE without necessarily improving threshold classification at 0.5.",
        "- The best deployable decision row should be compared with 002002002 contour auxiliary.",
        "- If a calibrated method with default 0.5 matches the fold-wise threshold result, calibration can replace manual threshold tuning.",
        "",
        "## Calibrator Parameters",
        "",
        parameter_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Figures",
        "",
        f"- Calibration curves: `{figure_dir / 'calibration_curves.png'}`",
        f"- Metric bars: `{figure_dir / 'calibration_metric_bars.png'}`",
        f"- ROC/PR curves: `{figure_dir / 'roc_pr_after_calibration.png'}`",
        "",
    ]
    (output_dir / "probability_calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    required = {"split", "label", "prob_hypertrophy", "outer_fold", "row_id", "sample_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    cal_df, parameter_df = build_calibrated_predictions(df)
    prob_summary = pooled_probability_summary(cal_df)
    fold_prob_summary = fold_probability_summary(cal_df)
    decision_summary, fold_decision_summary = pooled_decision_summary(cal_df, args.recall_target)
    calibration_bin_df = save_calibration_curves(cal_df, args.figure_dir)
    save_metric_bars(prob_summary, decision_summary, args.figure_dir)
    save_roc_pr(cal_df, args.figure_dir)

    cal_df.to_csv(args.output_dir / "calibrated_predictions.csv", index=False, encoding="utf-8-sig")
    parameter_df.to_csv(args.output_dir / "calibrator_parameters.csv", index=False, encoding="utf-8-sig")
    prob_summary.to_csv(args.output_dir / "probability_calibration_summary.csv", index=False, encoding="utf-8-sig")
    fold_prob_summary.to_csv(args.output_dir / "probability_calibration_by_fold.csv", index=False, encoding="utf-8-sig")
    decision_summary.to_csv(args.output_dir / "calibrated_decision_summary.csv", index=False, encoding="utf-8-sig")
    fold_decision_summary.to_csv(args.output_dir / "calibrated_decision_by_fold.csv", index=False, encoding="utf-8-sig")
    calibration_bin_df.to_csv(args.output_dir / "calibration_bins.csv", index=False, encoding="utf-8-sig")

    input_summary = {
        "prediction_file": str(args.predictions),
        "total_input_rows": int(len(df)),
        "validation_rows": int((df["split"] == "validation").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "methods": sorted(cal_df["method"].unique().tolist()),
        "baseline_fold_mean_auc": F2CG_FOLD_MEAN_AUC,
        "baseline_fold_auc_std": F2CG_FOLD_AUC_STD,
    }
    (args.output_dir / "input_summary.json").write_text(
        json.dumps(input_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir, args.figure_dir, args.predictions, prob_summary, decision_summary, parameter_df)

    print("Probability calibration summary:")
    print(prob_summary[["method", "auc", "average_precision", "brier", "nll", "ece_10"]].to_string(index=False))
    print("")
    print("Decision summary:")
    print(
        decision_summary[
            [
                "method",
                "selection",
                "threshold_mean",
                "balanced_acc",
                "precision",
                "recall",
                "specificity",
                "f1",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote outputs to {args.output_dir}")
    print(f"Wrote figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
