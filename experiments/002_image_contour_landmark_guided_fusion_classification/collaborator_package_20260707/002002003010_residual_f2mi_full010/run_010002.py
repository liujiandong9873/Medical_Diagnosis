from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "010002_out")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def clip_prob(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(prob.astype(float), eps, 1.0 - eps)


def logit_from_prob(prob: np.ndarray) -> np.ndarray:
    prob = clip_prob(prob)
    return np.log(prob / (1.0 - prob))


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
        "brier": float(brier_score_loss(y_true, clip_prob(prob))),
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


def threshold_grid(prob: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.01, 0.99, 197)
    unique = np.unique(np.round(prob.astype(float), 5))
    candidates = np.unique(np.concatenate([grid, unique, np.asarray([0.5])]))
    return candidates[(candidates > 0.0) & (candidates < 1.0)]


def select_threshold(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    mode: str,
    baseline_metrics: dict[str, float],
    margin: float = 0.0,
) -> tuple[float, dict[str, float], str]:
    rows = []
    for threshold in threshold_grid(prob_val):
        metrics = metrics_at_threshold(y_val, prob_val, float(threshold))
        rows.append((float(threshold), metrics))

    if mode == "max_f1":
        selected = max(rows, key=lambda item: (item[1]["f1"], item[1]["balanced_acc"], -abs(item[0] - 0.5)))
        return selected[0], selected[1], "max_f1"
    if mode == "max_balanced_acc":
        selected = max(rows, key=lambda item: (item[1]["balanced_acc"], item[1]["f1"], -abs(item[0] - 0.5)))
        return selected[0], selected[1], "max_balanced_acc"
    if mode == "youden":
        selected = max(
            rows,
            key=lambda item: (
                item[1]["recall"] + item[1]["specificity"] - 1.0,
                item[1]["f1"],
                -abs(item[0] - 0.5),
            ),
        )
        return selected[0], selected[1], "youden"
    if mode.startswith("specificity_"):
        objective = mode.split("_", 1)[1]
        floor = max(0.0, baseline_metrics["specificity"] - margin)
        feasible = [item for item in rows if item[1]["specificity"] >= floor]
        if not feasible:
            feasible = rows
            note = f"specificity_floor_not_reached_{floor:.4f}"
        else:
            note = f"specificity_floor_{floor:.4f}"
        if objective == "max_f1":
            selected = max(feasible, key=lambda item: (item[1]["f1"], item[1]["recall"], -abs(item[0] - 0.5)))
        elif objective == "max_recall":
            selected = max(feasible, key=lambda item: (item[1]["recall"], item[1]["f1"], item[1]["specificity"]))
        else:
            raise ValueError(f"Unknown constrained objective: {mode}")
        return selected[0], selected[1], note
    if mode.startswith("precision_"):
        objective = mode.split("_", 1)[1]
        floor = max(0.0, baseline_metrics["pre"] - margin)
        feasible = [item for item in rows if item[1]["pre"] >= floor]
        if not feasible:
            feasible = rows
            note = f"precision_floor_not_reached_{floor:.4f}"
        else:
            note = f"precision_floor_{floor:.4f}"
        if objective == "max_f1":
            selected = max(feasible, key=lambda item: (item[1]["f1"], item[1]["recall"], -abs(item[0] - 0.5)))
        elif objective == "max_recall":
            selected = max(feasible, key=lambda item: (item[1]["recall"], item[1]["f1"], item[1]["pre"]))
        else:
            raise ValueError(f"Unknown constrained objective: {mode}")
        return selected[0], selected[1], note
    raise ValueError(f"Unknown threshold mode: {mode}")


def load_predictions(target_path: Path, baseline_path: Path) -> pd.DataFrame:
    target = pd.read_csv(target_path)
    baseline = pd.read_csv(baseline_path)
    target_required = {*KEYS, "prob_hypertrophy", "pred_label", "final_logit"}
    baseline_required = {*KEYS, "prob_hypertrophy", "pred_label"}
    missing_target = sorted(target_required - set(target.columns))
    missing_baseline = sorted(baseline_required - set(baseline.columns))
    if missing_target:
        raise ValueError(f"Missing target columns: {missing_target}")
    if missing_baseline:
        raise ValueError(f"Missing baseline columns: {missing_baseline}")

    target = target[[*KEYS, "prob_hypertrophy", "pred_label", "final_logit"]].copy()
    baseline = baseline[[*KEYS, "prob_hypertrophy", "pred_label"]].copy()
    target = target.rename(
        columns={
            "prob_hypertrophy": "010_prob",
            "pred_label": "010_pred",
            "final_logit": "010_logit",
        }
    )
    baseline = baseline.rename(columns={"prob_hypertrophy": "baseline_prob", "pred_label": "baseline_pred"})
    merged = baseline.merge(target, on=KEYS, validate="one_to_one")
    merged["label"] = merged["label"].astype(int)
    merged["010_pred"] = merged["010_pred"].astype(int)
    merged["baseline_pred"] = merged["baseline_pred"].astype(int)
    return merged.sort_values(["split", "outer_fold", "row_id"]).reset_index(drop=True)


def make_strategy_prediction_rows(
    fold: int,
    strategy: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prob_val: np.ndarray,
    prob_test: np.ndarray,
    threshold: float,
    note: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    val_metrics = metrics_at_threshold(y_val, prob_val, threshold)
    test_metrics = metrics_at_threshold(y_test, prob_test, threshold)
    row = {"outer_fold": int(fold), "strategy": strategy, "threshold": float(threshold), "note": note}
    row.update({f"val_{key}": value for key, value in val_metrics.items()})
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    pred_df = test_df[KEYS].copy()
    pred_df["strategy"] = strategy
    pred_df["prob_hypertrophy"] = prob_test.astype(float)
    pred_df["pred_label"] = (prob_test >= threshold).astype(int)
    pred_df["threshold"] = float(threshold)
    pred_df["note"] = note
    return row, pred_df


def fit_platt(y_val: np.ndarray, prob_val: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y_val)) < 2:
        return None
    x = logit_from_prob(prob_val).reshape(-1, 1)
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(x, y_val.astype(int))
    return model


def apply_platt(model: LogisticRegression | None, prob: np.ndarray) -> np.ndarray:
    if model is None:
        return prob.astype(float)
    x = logit_from_prob(prob).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def fit_isotonic(y_val: np.ndarray, prob_val: np.ndarray) -> IsotonicRegression | None:
    if len(np.unique(y_val)) < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(prob_val.astype(float), y_val.astype(int))
    return model


def apply_isotonic(model: IsotonicRegression | None, prob: np.ndarray) -> np.ndarray:
    if model is None:
        return prob.astype(float)
    return model.predict(prob.astype(float))


def generate_strategy_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    threshold_modes = [
        ("010_fixed_0_5", "fixed", 0.0),
        ("010_val_max_f1", "max_f1", 0.0),
        ("010_val_max_balanced_acc", "max_balanced_acc", 0.0),
        ("010_val_youden", "youden", 0.0),
    ]
    for margin in [0.0, 0.02, 0.05]:
        threshold_modes.extend(
            [
                (f"010_spec_m{margin:.2f}_max_f1", "specificity_max_f1", margin),
                (f"010_spec_m{margin:.2f}_max_recall", "specificity_max_recall", margin),
                (f"010_pre_m{margin:.2f}_max_f1", "precision_max_f1", margin),
                (f"010_pre_m{margin:.2f}_max_recall", "precision_max_recall", margin),
            ]
        )

    for fold in sorted(df["outer_fold"].unique()):
        val_df = df[(df["outer_fold"] == fold) & (df["split"] == "validation")].copy()
        test_df = df[(df["outer_fold"] == fold) & (df["split"] == "test")].copy()
        y_val = val_df["label"].to_numpy()
        prob_val = val_df["010_prob"].to_numpy(dtype=float)
        prob_test = test_df["010_prob"].to_numpy(dtype=float)
        baseline_val_metrics = compute_metrics(
            y_val,
            val_df["baseline_prob"].to_numpy(dtype=float),
            val_df["baseline_pred"].to_numpy(dtype=int),
        )

        for strategy, mode, margin in threshold_modes:
            if mode == "fixed":
                threshold = 0.5
                val_metrics = metrics_at_threshold(y_val, prob_val, threshold)
                note = "fixed_0_5"
            else:
                threshold, val_metrics, note = select_threshold(y_val, prob_val, mode, baseline_val_metrics, margin)
            row, pred_df = make_strategy_prediction_rows(
                fold, strategy, val_df, test_df, prob_val, prob_test, threshold, note
            )
            row["constraint_margin"] = margin
            row.update({f"baseline_val_{key}": value for key, value in baseline_val_metrics.items()})
            rows.append(row)
            predictions.append(pred_df)

        platt = fit_platt(y_val, prob_val)
        platt_val = apply_platt(platt, prob_val)
        platt_test = apply_platt(platt, prob_test)
        row, pred_df = make_strategy_prediction_rows(
            fold, "010_platt_calibration_0_5", val_df, test_df, platt_val, platt_test, 0.5, "val_platt"
        )
        row["constraint_margin"] = 0.0
        row.update({f"baseline_val_{key}": value for key, value in baseline_val_metrics.items()})
        rows.append(row)
        predictions.append(pred_df)

        isotonic = fit_isotonic(y_val, prob_val)
        iso_val = apply_isotonic(isotonic, prob_val)
        iso_test = apply_isotonic(isotonic, prob_test)
        row, pred_df = make_strategy_prediction_rows(
            fold, "010_isotonic_calibration_0_5", val_df, test_df, iso_val, iso_test, 0.5, "val_isotonic"
        )
        row["constraint_margin"] = 0.0
        row.update({f"baseline_val_{key}": value for key, value in baseline_val_metrics.items()})
        rows.append(row)
        predictions.append(pred_df)

    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def summarize_strategy_predictions(strategy_predictions: pd.DataFrame, baseline_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, pred_df in strategy_predictions.groupby("strategy"):
        merged = baseline_test.merge(
            pred_df[[*KEYS, "prob_hypertrophy", "pred_label", "threshold", "note"]],
            on=KEYS,
            validate="one_to_one",
        )
        y = merged["label"].to_numpy()
        row = {
            "strategy": strategy,
            "n": int(len(merged)),
            "threshold_mean": float(merged["threshold"].mean()),
            "threshold_std": float(merged["threshold"].std(ddof=1)),
        }
        row.update(compute_metrics(y, merged["prob_hypertrophy"].to_numpy(), merged["pred_label"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["f1", "recall", "specificity"], ascending=False).reset_index(drop=True)


def baseline_summary(baseline_test: pd.DataFrame) -> dict[str, float]:
    y = baseline_test["label"].to_numpy()
    return compute_metrics(y, baseline_test["baseline_prob"].to_numpy(), baseline_test["baseline_pred"].to_numpy())


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.asarray(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    return np.concatenate([np.asarray(groups[sid], dtype=int) for sid in sampled_ids])


def paired_bootstrap_strategy(
    merged: pd.DataFrame,
    n_iters: int,
    seed: int,
) -> pd.DataFrame:
    y = merged["label"].to_numpy()
    strategy_metrics = compute_metrics(y, merged["prob_hypertrophy"].to_numpy(), merged["pred_label"].to_numpy())
    baseline_metrics = compute_metrics(y, merged["baseline_prob"].to_numpy(), merged["baseline_pred"].to_numpy())
    observed = {metric: strategy_metrics[metric] - baseline_metrics[metric] for metric in METRICS}
    rng = np.random.default_rng(seed)
    boot_diffs = {metric: [] for metric in METRICS}
    for _ in range(n_iters):
        idx = sample_cluster_indices(merged, rng)
        boot = merged.iloc[idx]
        y_boot = boot["label"].to_numpy()
        if len(np.unique(y_boot)) < 2:
            continue
        strategy_boot = compute_metrics(y_boot, boot["prob_hypertrophy"].to_numpy(), boot["pred_label"].to_numpy())
        baseline_boot = compute_metrics(y_boot, boot["baseline_prob"].to_numpy(), boot["baseline_pred"].to_numpy())
        for metric in METRICS:
            value = strategy_boot[metric] - baseline_boot[metric]
            if not np.isnan(value):
                boot_diffs[metric].append(value)
    rows = []
    for metric in METRICS:
        values = np.asarray(boot_diffs[metric], dtype=float)
        p_lower = float(np.mean(values <= 0.0))
        p_upper = float(np.mean(values >= 0.0))
        rows.append(
            {
                "metric": metric,
                "diff_strategy_minus_baseline": observed[metric],
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_p_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
                "bootstrap_n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_key_strategies(
    strategy_predictions: pd.DataFrame,
    baseline_test: pd.DataFrame,
    summary: pd.DataFrame,
    n_iters: int,
    seed: int,
) -> pd.DataFrame:
    selected = [
        "010_fixed_0_5",
        "010_val_max_f1",
        "010_val_max_balanced_acc",
        "010_val_youden",
        "010_spec_m0.02_max_f1",
        "010_spec_m0.02_max_recall",
        "010_pre_m0.02_max_f1",
        "010_pre_m0.02_max_recall",
        "010_platt_calibration_0_5",
        "010_isotonic_calibration_0_5",
    ]
    # Include top three F1 strategies even if they are not in the preset list.
    selected.extend(summary.head(3)["strategy"].tolist())
    selected = list(dict.fromkeys([item for item in selected if item in set(summary["strategy"])]))
    tables = []
    for index, strategy in enumerate(selected):
        pred_df = strategy_predictions[strategy_predictions["strategy"] == strategy].copy()
        merged = baseline_test.merge(
            pred_df[[*KEYS, "prob_hypertrophy", "pred_label", "threshold", "note"]],
            on=KEYS,
            validate="one_to_one",
        )
        boot = paired_bootstrap_strategy(merged, n_iters, seed + index * 1009)
        boot.insert(0, "strategy", strategy)
        tables.append(boot)
    return pd.concat(tables, ignore_index=True)


def decision_counts(strategy_predictions: pd.DataFrame, baseline_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, pred_df in strategy_predictions.groupby("strategy"):
        merged = baseline_test.merge(
            pred_df[[*KEYS, "pred_label"]],
            on=KEYS,
            validate="one_to_one",
        )
        target_correct = merged["pred_label"].astype(int).eq(merged["label"].astype(int))
        baseline_correct = merged["baseline_pred"].astype(int).eq(merged["label"].astype(int))
        target_fn = ((merged["label"] == 1) & (merged["pred_label"] == 0)).sum()
        baseline_fn = ((merged["label"] == 1) & (merged["baseline_pred"] == 0)).sum()
        target_fp = ((merged["label"] == 0) & (merged["pred_label"] == 1)).sum()
        baseline_fp = ((merged["label"] == 0) & (merged["baseline_pred"] == 1)).sum()
        rows.append(
            {
                "strategy": strategy,
                "both_correct": int((target_correct & baseline_correct).sum()),
                "target_only_correct": int((target_correct & ~baseline_correct).sum()),
                "baseline_only_correct": int((~target_correct & baseline_correct).sum()),
                "both_wrong": int((~target_correct & ~baseline_correct).sum()),
                "net_correct_gain": int((target_correct & ~baseline_correct).sum() - (~target_correct & baseline_correct).sum()),
                "fn_diff_target_minus_baseline": int(target_fn - baseline_fn),
                "fp_diff_target_minus_baseline": int(target_fp - baseline_fp),
            }
        )
    return pd.DataFrame(rows)


def metric_line(table: pd.DataFrame, strategy: str, metric: str) -> str:
    row = table[(table["strategy"] == strategy) & (table["metric"] == metric)].iloc[0]
    return (
        f"{row['diff_strategy_minus_baseline']:+.4f} "
        f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], p={row['bootstrap_p_two_sided']:.3f}"
    )


def write_report(
    output_dir: Path,
    base_metrics: dict[str, float],
    fold_table: pd.DataFrame,
    summary: pd.DataFrame,
    boot: pd.DataFrame,
    decisions: pd.DataFrame,
    bootstrap_iters: int,
) -> None:
    best_f1 = summary.iloc[0]
    baseline_f1 = base_metrics["f1"]
    baseline_recall = base_metrics["recall"]
    baseline_specificity = base_metrics["specificity"]
    baseline_pre = base_metrics["pre"]

    def summary_row(strategy: str) -> pd.Series | None:
        rows = summary[summary["strategy"] == strategy]
        return rows.iloc[0] if len(rows) else None

    highlight_names = [
        "010_fixed_0_5",
        "010_val_max_f1",
        "010_spec_m0.02_max_f1",
        "010_spec_m0.02_max_recall",
        "010_pre_m0.02_max_f1",
        "010_pre_m0.02_max_recall",
        "010_platt_calibration_0_5",
        "010_isotonic_calibration_0_5",
    ]

    lines = [
        "# 002002003010002 Threshold and Calibration Analysis",
        "",
        "## Scope",
        "",
        "- Target: existing `002002003010` predictions only",
        "- No model retraining",
        "- Thresholds/calibrators are selected or fitted on validation folds, then applied to paired test folds",
        "- Main comparator: `002002002 contour auxiliary` at default threshold 0.5",
        f"- Bootstrap iterations: {bootstrap_iters}",
        "",
        "## Baseline Test Metrics",
        "",
        "| Model | AUC | Acc | Pre | Recall | Specificity | F1 | FN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 002002002 contour auxiliary | {base_metrics['auc']:.4f} | {base_metrics['acc']:.4f} | "
            f"{base_metrics['pre']:.4f} | {base_metrics['recall']:.4f} | {base_metrics['specificity']:.4f} | "
            f"{base_metrics['f1']:.4f} | {base_metrics['fn']} | {base_metrics['fp']} |"
        ),
        "",
        "## Strategy Test Summary",
        "",
        "| Strategy | Threshold mean | AUC | Acc | Pre | Recall | Specificity | F1 | FN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.head(15).iterrows():
        lines.append(
            f"| {row['strategy']} | {row['threshold_mean']:.4f} | {row['auc']:.4f} | {row['acc']:.4f} | "
            f"{row['pre']:.4f} | {row['recall']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | "
            f"{int(row['fn'])} | {int(row['fp'])} |"
        )

    lines += [
        "",
        "## Highlighted Strategies",
        "",
        "| Strategy | Pre diff | Recall diff | Specificity diff | F1 diff | FN diff | FP diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in highlight_names:
        row = summary_row(strategy)
        if row is None:
            continue
        dec = decisions[decisions["strategy"] == strategy].iloc[0]
        lines.append(
            f"| {strategy} | {row['pre'] - baseline_pre:+.4f} | {row['recall'] - baseline_recall:+.4f} | "
            f"{row['specificity'] - baseline_specificity:+.4f} | {row['f1'] - baseline_f1:+.4f} | "
            f"{int(dec['fn_diff_target_minus_baseline'])} | {int(dec['fp_diff_target_minus_baseline'])} |"
        )

    lines += [
        "",
        "## Bootstrap for Key Strategies",
        "",
        "| Strategy | Recall diff | F1 diff | Pre diff | Specificity diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for strategy in boot["strategy"].drop_duplicates():
        lines.append(
            f"| {strategy} | {metric_line(boot, strategy, 'recall')} | {metric_line(boot, strategy, 'f1')} | "
            f"{metric_line(boot, strategy, 'pre')} | {metric_line(boot, strategy, 'specificity')} |"
        )

    lines += [
        "",
        "## Main Interpretation",
        "",
        (
            f"Best F1 strategy is `{best_f1['strategy']}` with F1={best_f1['f1']:.4f}, "
            f"recall={best_f1['recall']:.4f}, specificity={best_f1['specificity']:.4f}, precision={best_f1['pre']:.4f}."
        ),
        "",
    ]
    if best_f1["f1"] > baseline_f1 and best_f1["specificity"] >= baseline_specificity - 0.02:
        lines.append("This analysis found a threshold/calibration strategy that improves F1 while keeping specificity close to baseline.")
    elif best_f1["f1"] > baseline_f1:
        lines.append("This analysis found higher F1, but the specificity cost remains larger than the desired margin.")
    else:
        lines.append("This analysis did not find a validation-selected strategy that improves F1 over baseline.")

    close_spec = summary[summary["specificity"] >= baseline_specificity - 0.02].copy()
    if len(close_spec):
        close_spec = close_spec.sort_values(["f1", "recall"], ascending=False).iloc[0]
        lines += [
            "",
            "Best strategy under specificity >= baseline - 0.02:",
            "",
            (
                f"- `{close_spec['strategy']}`: F1={close_spec['f1']:.4f}, recall={close_spec['recall']:.4f}, "
                f"specificity={close_spec['specificity']:.4f}, precision={close_spec['pre']:.4f}."
            ),
        ]

    close_pre = summary[summary["pre"] >= baseline_pre - 0.02].copy()
    if len(close_pre):
        close_pre = close_pre.sort_values(["recall", "f1"], ascending=False).iloc[0]
        lines += [
            "",
            "Best strategy under precision >= baseline - 0.02:",
            "",
            (
                f"- `{close_pre['strategy']}`: F1={close_pre['f1']:.4f}, recall={close_pre['recall']:.4f}, "
                f"specificity={close_pre['specificity']:.4f}, precision={close_pre['pre']:.4f}."
            ),
        ]

    lines += [
        "",
        "## Files",
        "",
        "- `strategy_by_fold.csv`",
        "- `strategy_test_summary.csv`",
        "- `strategy_test_predictions.csv`",
        "- `strategy_bootstrap_vs_baseline.csv`",
        "- `strategy_decision_counts_vs_baseline.csv`",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = load_predictions(args.target_predictions, args.baseline_predictions)
    baseline_test = merged[merged["split"] == "test"].copy()

    fold_table, strategy_predictions = generate_strategy_predictions(merged)
    summary = summarize_strategy_predictions(strategy_predictions, baseline_test)
    base_metrics = baseline_summary(baseline_test)
    boot = bootstrap_key_strategies(strategy_predictions, baseline_test, summary, args.bootstrap_iters, args.seed)
    decisions = decision_counts(strategy_predictions, baseline_test)

    merged.to_csv(args.output_dir / "merged_predictions_all_splits.csv", index=False, encoding="utf-8-sig")
    fold_table.to_csv(args.output_dir / "strategy_by_fold.csv", index=False, encoding="utf-8-sig")
    strategy_predictions.to_csv(args.output_dir / "strategy_test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "strategy_test_summary.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(args.output_dir / "strategy_bootstrap_vs_baseline.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(args.output_dir / "strategy_decision_counts_vs_baseline.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([base_metrics]).to_csv(args.output_dir / "baseline_test_metrics.csv", index=False, encoding="utf-8-sig")

    write_report(args.output_dir, base_metrics, fold_table, summary, boot, decisions, args.bootstrap_iters)
    print(f"Wrote 010002 outputs to {args.output_dir}")
    print(summary[["strategy", "threshold_mean", "auc", "acc", "pre", "recall", "specificity", "f1", "fn", "fp"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
