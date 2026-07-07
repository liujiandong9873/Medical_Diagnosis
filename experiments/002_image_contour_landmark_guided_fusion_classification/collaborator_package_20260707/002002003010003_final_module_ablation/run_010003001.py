from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


KEYS = ["row_id", "sample_id", "image_file", "outer_fold", "split", "label"]
TEST_KEYS = ["row_id", "sample_id", "image_file", "outer_fold", "label"]
METRICS = ["auc", "acc", "pre", "recall", "specificity", "f1"]
VARIANTS = ["no_fractal", "no_fractional", "no_residual"]


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    experiment_dir = analysis_dir.parent
    base_dir = experiment_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", type=Path, default=experiment_dir)
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=base_dir / "002002002_image_contour_auxiliary_supervision" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--control011-predictions",
        type=Path,
        default=base_dir / "002002003011_continued_training_control" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument(
        "--full010-strategy-predictions",
        type=Path,
        default=base_dir
        / "002002003010_baseline_preserving_residual_f2mi"
        / "analysis"
        / "010002_out"
        / "strategy_test_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "010003001_out")
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


def load_cv_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {*KEYS, "prob_hypertrophy", "pred_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[[*KEYS, "prob_hypertrophy", "pred_label"]].copy()


def load_test_model(path: Path, model_name: str) -> pd.DataFrame:
    df = load_cv_predictions(path)
    df = df[df["split"] == "test"].copy()
    out = df[TEST_KEYS].copy()
    out["model"] = model_name
    out["prob_hypertrophy"] = df["prob_hypertrophy"].astype(float).to_numpy()
    out["pred_label"] = df["pred_label"].astype(int).to_numpy()
    return out


def load_full010_platt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {*KEYS, "strategy", "prob_hypertrophy", "pred_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[(df["split"] == "test") & (df["strategy"] == "010_platt_calibration_0_5")].copy()
    out = df[TEST_KEYS].copy()
    out["model"] = "full010_platt"
    out["prob_hypertrophy"] = df["prob_hypertrophy"].astype(float).to_numpy()
    out["pred_label"] = df["pred_label"].astype(int).to_numpy()
    return out


def fit_platt(y_val: np.ndarray, prob_val: np.ndarray) -> LogisticRegression | None:
    if len(np.unique(y_val)) < 2:
        return None
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(logit_from_prob(prob_val).reshape(-1, 1), y_val.astype(int))
    return model


def apply_platt(model: LogisticRegression | None, prob: np.ndarray) -> np.ndarray:
    if model is None:
        return prob.astype(float)
    return model.predict_proba(logit_from_prob(prob).reshape(-1, 1))[:, 1]


def load_variant_platt(ablation_dir: Path, variant: str) -> pd.DataFrame | None:
    path = ablation_dir / "outputs" / variant / "cv_predictions.csv"
    if not path.exists():
        return None
    df = load_cv_predictions(path)
    outputs = []
    for fold in sorted(df["outer_fold"].unique()):
        val_df = df[(df["outer_fold"] == fold) & (df["split"] == "validation")].copy()
        test_df = df[(df["outer_fold"] == fold) & (df["split"] == "test")].copy()
        platt = fit_platt(val_df["label"].to_numpy(), val_df["prob_hypertrophy"].to_numpy(dtype=float))
        prob_test = apply_platt(platt, test_df["prob_hypertrophy"].to_numpy(dtype=float))
        out = test_df[TEST_KEYS].copy()
        out["model"] = f"{variant}_platt"
        out["prob_hypertrophy"] = prob_test
        out["pred_label"] = (prob_test >= 0.5).astype(int)
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def summarize(long_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, df in long_predictions.groupby("model", sort=False):
        metrics = compute_metrics(
            df["label"].to_numpy(),
            df["prob_hypertrophy"].to_numpy(dtype=float),
            df["pred_label"].to_numpy(dtype=int),
        )
        rows.append({"model": model, "n": int(len(df)), **metrics})
    return pd.DataFrame(rows)


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.asarray(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    return np.concatenate([np.asarray(groups[sid], dtype=int) for sid in sampled_ids])


def paired_table(long_predictions: pd.DataFrame, target: str, reference: str) -> pd.DataFrame:
    target_df = long_predictions[long_predictions["model"] == target].copy()
    ref_df = long_predictions[long_predictions["model"] == reference].copy()
    merged = target_df.merge(
        ref_df[[*TEST_KEYS, "prob_hypertrophy", "pred_label"]],
        on=TEST_KEYS,
        validate="one_to_one",
        suffixes=("_target", "_reference"),
    )
    return merged


def paired_bootstrap(merged: pd.DataFrame, n_iters: int, seed: int) -> pd.DataFrame:
    y = merged["label"].to_numpy()
    target_metrics = compute_metrics(y, merged["prob_hypertrophy_target"].to_numpy(), merged["pred_label_target"].to_numpy())
    ref_metrics = compute_metrics(y, merged["prob_hypertrophy_reference"].to_numpy(), merged["pred_label_reference"].to_numpy())
    observed = {metric: target_metrics[metric] - ref_metrics[metric] for metric in METRICS}
    rng = np.random.default_rng(seed)
    boot_diffs = {metric: [] for metric in METRICS}
    for _ in range(n_iters):
        idx = sample_cluster_indices(merged, rng)
        boot = merged.iloc[idx]
        y_boot = boot["label"].to_numpy()
        if len(np.unique(y_boot)) < 2:
            continue
        target_boot = compute_metrics(y_boot, boot["prob_hypertrophy_target"].to_numpy(), boot["pred_label_target"].to_numpy())
        ref_boot = compute_metrics(y_boot, boot["prob_hypertrophy_reference"].to_numpy(), boot["pred_label_reference"].to_numpy())
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
                "metric": metric,
                "diff_target_minus_reference": observed[metric],
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
                "bootstrap_p_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
                "bootstrap_n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def decision_counts(merged: pd.DataFrame) -> dict[str, int]:
    target_correct = merged["pred_label_target"].astype(int).eq(merged["label"].astype(int))
    ref_correct = merged["pred_label_reference"].astype(int).eq(merged["label"].astype(int))
    target_fn = ((merged["label"] == 1) & (merged["pred_label_target"] == 0)).sum()
    ref_fn = ((merged["label"] == 1) & (merged["pred_label_reference"] == 0)).sum()
    target_fp = ((merged["label"] == 0) & (merged["pred_label_target"] == 1)).sum()
    ref_fp = ((merged["label"] == 0) & (merged["pred_label_reference"] == 1)).sum()
    return {
        "target_only_correct": int((target_correct & ~ref_correct).sum()),
        "reference_only_correct": int((~target_correct & ref_correct).sum()),
        "net_correct_gain": int((target_correct & ~ref_correct).sum() - (~target_correct & ref_correct).sum()),
        "fn_diff_target_minus_reference": int(target_fn - ref_fn),
        "fp_diff_target_minus_reference": int(target_fp - ref_fp),
    }


def metric_line(boot: pd.DataFrame, metric: str) -> str:
    row = boot[boot["metric"] == metric].iloc[0]
    return (
        f"{row['diff_target_minus_reference']:+.4f} "
        f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], p={row['bootstrap_p_two_sided']:.3f}"
    )


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    boot: pd.DataFrame,
    decisions: pd.DataFrame,
    missing_variants: list[str],
    n_iters: int,
) -> None:
    lines = [
        "# 002002003010003001 Final Module Ablation Analysis",
        "",
        "## Scope",
        "",
        "- Main target: `full010_platt` from completed 010002 analysis",
        "- Ablation variants are calibrated with validation-fitted Platt scaling",
        f"- Paired cluster bootstrap iterations: {n_iters}",
    ]
    if missing_variants:
        lines += ["", "Missing variants skipped:", ""]
        for variant in missing_variants:
            lines.append(f"- `{variant}`")
    lines += [
        "",
        "## Model Metrics",
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
        "## Bootstrap Comparisons",
        "",
        "| Target | Reference | Recall diff | F1 diff | Precision diff | Specificity diff |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (target, reference), table in boot.groupby(["target", "reference"], sort=False):
        lines.append(
            f"| {target} | {reference} | {metric_line(table, 'recall')} | {metric_line(table, 'f1')} | "
            f"{metric_line(table, 'pre')} | {metric_line(table, 'specificity')} |"
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
        "## Files",
        "",
        "- `model_predictions_test.csv`",
        "- `model_metrics.csv`",
        "- `paired_bootstrap.csv`",
        "- `decision_counts.csv`",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = [
        load_test_model(args.baseline_predictions, "baseline002002002"),
        load_full010_platt(args.full010_strategy_predictions),
    ]
    if args.control011_predictions.exists():
        predictions.append(load_test_model(args.control011_predictions, "control011"))

    missing = []
    for variant in VARIANTS:
        variant_df = load_variant_platt(args.ablation_dir, variant)
        if variant_df is None:
            missing.append(variant)
        else:
            predictions.append(variant_df)

    long_predictions = pd.concat(predictions, ignore_index=True)
    summary = summarize(long_predictions)

    comparisons = [("full010_platt", "baseline002002002")]
    if "control011" in set(long_predictions["model"]):
        comparisons.append(("full010_platt", "control011"))
    for variant in VARIANTS:
        model = f"{variant}_platt"
        if model in set(long_predictions["model"]):
            comparisons.append(("full010_platt", model))

    boot_tables = []
    decision_rows = []
    for index, (target, reference) in enumerate(comparisons):
        merged = paired_table(long_predictions, target, reference)
        boot = paired_bootstrap(merged, args.bootstrap_iters, args.seed + index * 1009)
        boot.insert(0, "reference", reference)
        boot.insert(0, "target", target)
        boot_tables.append(boot)
        decision = decision_counts(merged)
        decision.update({"target": target, "reference": reference})
        decision_rows.append(decision)

    boot_all = pd.concat(boot_tables, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)

    long_predictions.to_csv(args.output_dir / "model_predictions_test.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    boot_all.to_csv(args.output_dir / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(args.output_dir / "decision_counts.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir, summary, boot_all, decisions, missing, args.bootstrap_iters)
    print(summary[["model", "auc", "acc", "pre", "recall", "specificity", "f1", "fn", "fp"]].to_string(index=False))
    if missing:
        print(f"Missing variants skipped: {', '.join(missing)}")


if __name__ == "__main__":
    main()
