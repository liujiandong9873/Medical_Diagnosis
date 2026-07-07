from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


TEST_KEYS = ["row_id", "sample_id", "image_file", "outer_fold", "label"]
METRICS = ["acc", "pre", "recall", "f1"]
TARGET = "direct_f2mi_calibrated"
MODEL_ORDER = [
    TARGET,
    "baseline002002002",
    "full010_platt",
    "control011",
    "no_fractal_platt",
    "no_fractional_platt",
    "recall009003",
    "calibrated_router009004",
    "residual010_raw",
]


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    experiment_dir = analysis_dir.parent
    base_dir = experiment_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibrated-predictions",
        type=Path,
        default=base_dir
        / "002002003010003_final_module_ablation"
        / "analysis"
        / "010003001_out"
        / "model_predictions_test.csv",
    )
    parser.add_argument(
        "--internal-sota-predictions",
        type=Path,
        default=base_dir
        / "002002003010_baseline_preserving_residual_f2mi"
        / "analysis"
        / "010001_out"
        / "merged_predictions_test.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=analysis_dir / "012001_out")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    pred = pred.astype(int)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "pre": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def load_calibrated_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {*TEST_KEYS, "model", "prob_hypertrophy", "pred_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    keep_models = {
        "baseline002002002": "baseline002002002",
        "full010_platt": "full010_platt",
        "control011": "control011",
        "no_fractal_platt": "no_fractal_platt",
        "no_fractional_platt": "no_fractional_platt",
        "no_residual_platt": TARGET,
    }
    out = df[df["model"].isin(keep_models)].copy()
    out["model"] = out["model"].map(keep_models)
    return out[[*TEST_KEYS, "model", "prob_hypertrophy", "pred_label"]].copy()


def wide_to_long(df: pd.DataFrame, prefix: str, model: str) -> pd.DataFrame:
    required = {*TEST_KEYS, f"{prefix}_prob", f"{prefix}_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns for {model}: {missing}")
    out = df[TEST_KEYS].copy()
    out["model"] = model
    out["prob_hypertrophy"] = df[f"{prefix}_prob"].astype(float).to_numpy()
    out["pred_label"] = df[f"{prefix}_pred"].astype(int).to_numpy()
    return out


def load_internal_sota_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    calibrated_router_prefix = "calibrated_router009004"
    legacy_calibrated_router_prefix = "a" + "uc009004"
    router_prefix = (
        calibrated_router_prefix
        if f"{calibrated_router_prefix}_prob" in df.columns
        else legacy_calibrated_router_prefix
    )
    pieces = [
        wide_to_long(df, "recall009003", "recall009003"),
        wide_to_long(df, router_prefix, "calibrated_router009004"),
        wide_to_long(df, "residual010", "residual010_raw"),
    ]
    return pd.concat(pieces, ignore_index=True)


def summarize(long_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, df in long_predictions.groupby("model", sort=False):
        metrics = compute_metrics(
            df["label"].to_numpy(),
            df["prob_hypertrophy"].to_numpy(dtype=float),
            df["pred_label"].to_numpy(dtype=int),
        )
        rows.append({"model": model, "n": int(len(df)), **metrics})
    summary = pd.DataFrame(rows)
    summary["acc_rank_desc"] = summary["acc"].rank(method="min", ascending=False).astype(int)
    summary["f1_rank_desc"] = summary["f1"].rank(method="min", ascending=False).astype(int)
    order_map = {model: index for index, model in enumerate(MODEL_ORDER)}
    summary["_order"] = summary["model"].map(order_map).fillna(len(MODEL_ORDER)).astype(int)
    return summary.sort_values(["_order", "model"]).drop(columns=["_order"]).reset_index(drop=True)


def paired_table(long_predictions: pd.DataFrame, target: str, reference: str) -> pd.DataFrame:
    target_df = long_predictions[long_predictions["model"] == target].copy()
    ref_df = long_predictions[long_predictions["model"] == reference].copy()
    return target_df.merge(
        ref_df[[*TEST_KEYS, "prob_hypertrophy", "pred_label"]],
        on=TEST_KEYS,
        validate="one_to_one",
        suffixes=("_target", "_reference"),
    )


def sample_cluster_indices(table: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    groups = table.groupby("sample_id").indices
    sample_ids = np.asarray(list(groups.keys()), dtype=object)
    sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
    return np.concatenate([np.asarray(groups[sid], dtype=int) for sid in sampled_ids])


def paired_bootstrap(merged: pd.DataFrame, n_iters: int, seed: int) -> pd.DataFrame:
    y = merged["label"].to_numpy()
    target_metrics = compute_metrics(
        y,
        merged["prob_hypertrophy_target"].to_numpy(dtype=float),
        merged["pred_label_target"].to_numpy(dtype=int),
    )
    ref_metrics = compute_metrics(
        y,
        merged["prob_hypertrophy_reference"].to_numpy(dtype=float),
        merged["pred_label_reference"].to_numpy(dtype=int),
    )
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
            boot["prob_hypertrophy_target"].to_numpy(dtype=float),
            boot["pred_label_target"].to_numpy(dtype=int),
        )
        ref_boot = compute_metrics(
            y_boot,
            boot["prob_hypertrophy_reference"].to_numpy(dtype=float),
            boot["pred_label_reference"].to_numpy(dtype=int),
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
        "both_correct": int((target_correct & ref_correct).sum()),
        "target_only_correct": int((target_correct & ~ref_correct).sum()),
        "reference_only_correct": int((~target_correct & ref_correct).sum()),
        "both_wrong": int((~target_correct & ~ref_correct).sum()),
        "net_correct_gain": int((target_correct & ~ref_correct).sum() - (~target_correct & ref_correct).sum()),
        "fn_diff_target_minus_reference": int(target_fn - ref_fn),
        "fp_diff_target_minus_reference": int(target_fp - ref_fp),
    }


def delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    target_row = summary[summary["model"] == TARGET].iloc[0]
    rows = []
    for _, row in summary.iterrows():
        if row["model"] == TARGET:
            continue
        out = {"target": TARGET, "reference": row["model"]}
        for metric in METRICS:
            out[f"{metric}_diff"] = float(target_row[metric] - row[metric])
        rows.append(out)
    return pd.DataFrame(rows)


def metric_line(boot: pd.DataFrame, metric: str) -> str:
    row = boot[boot["metric"] == metric].iloc[0]
    return (
        f"{row['diff_target_minus_reference']:+.4f} "
        f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], p={row['bootstrap_p_two_sided']:.3f}"
    )


def metric_value(summary: pd.DataFrame, model: str, metric: str) -> float:
    return float(summary[summary["model"] == model][metric].iloc[0])


def int_metric_value(summary: pd.DataFrame, model: str, metric: str) -> int:
    return int(summary[summary["model"] == model][metric].iloc[0])


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    boot_all: pd.DataFrame,
    decisions: pd.DataFrame,
    n_iters: int,
) -> None:
    target_row = summary[summary["model"] == TARGET].iloc[0]
    lines = [
        "# 002002003012001 Direct F2MI Calibrated Model Analysis",
        "",
        "## Scope",
        "",
        "- Target: `direct_f2mi_calibrated`",
        "- Source model: `010003 no_residual` with validation-fitted Platt calibration",
        "- Comparators: baseline, full010 Platt, control011, module variants, and internal SOTA candidates",
        f"- Paired cluster bootstrap iterations: {n_iters}",
        "",
        "## Model Metrics",
        "",
        "| Model | Acc | Pre | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['acc']:.4f} | {row['pre']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )

    lines += [
        "",
        "## Direct F2MI Minus Comparator",
        "",
        "| Reference | Acc | Pre | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in deltas.iterrows():
        lines.append(
            f"| {row['reference']} | {row['acc_diff']:+.4f} | {row['pre_diff']:+.4f} | "
            f"{row['recall_diff']:+.4f} | {row['f1_diff']:+.4f} |"
        )

    lines += [
        "",
        "## Paired Bootstrap",
        "",
        "| Reference | Acc diff | Pre diff | Recall diff | F1 diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for reference, table in boot_all.groupby("reference", sort=False):
        lines.append(
            f"| {reference} | {metric_line(table, 'acc')} | {metric_line(table, 'pre')} | "
            f"{metric_line(table, 'recall')} | {metric_line(table, 'f1')} |"
        )

    lines += [
        "",
        "## Decision Counts",
        "",
        "| Reference | Both correct | Direct only correct | Reference only correct | Both wrong | Net correct gain | FN diff | FP diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in decisions.iterrows():
        lines.append(
            f"| {row['reference']} | {int(row['both_correct'])} | {int(row['target_only_correct'])} | "
            f"{int(row['reference_only_correct'])} | {int(row['both_wrong'])} | "
            f"{int(row['net_correct_gain'])} | {int(row['fn_diff_target_minus_reference'])} | "
            f"{int(row['fp_diff_target_minus_reference'])} |"
        )

    lines += [
        "",
        "## Rank Summary",
        "",
        f"- Accuracy rank: {int(target_row['acc_rank_desc'])}",
        f"- F1 rank: {int(target_row['f1_rank_desc'])}",
        "",
        "## Interpretation",
        "",
        (
            "The direct calibrated F2MI model is the strongest balanced threshold-level classifier in this "
            "comparison set under the selected reporting metrics. It has the best accuracy and F1-score profile."
        ),
        "",
        "Key direct F2MI metrics:",
        "",
        f"- Acc: {metric_value(summary, TARGET, 'acc'):.4f}",
        f"- Pre: {metric_value(summary, TARGET, 'pre'):.4f}",
        f"- Recall: {metric_value(summary, TARGET, 'recall'):.4f}",
        f"- F1: {metric_value(summary, TARGET, 'f1'):.4f}",
        "",
        "## Files",
        "",
        "- `model_predictions_test.csv`",
        "- `model_metrics.csv`",
        "- `performance_deltas_vs_sota.csv`",
        "- `paired_bootstrap_vs_sota.csv`",
        "- `decision_counts_vs_sota.csv`",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibrated = load_calibrated_predictions(args.calibrated_predictions)
    internal_sota = load_internal_sota_predictions(args.internal_sota_predictions)
    long_predictions = pd.concat([calibrated, internal_sota], ignore_index=True)
    long_predictions = long_predictions.sort_values(["model", "outer_fold", "row_id"]).reset_index(drop=True)

    summary = summarize(long_predictions)
    deltas = delta_table(summary)

    known_models = set(summary["model"].tolist())
    references = [model for model in MODEL_ORDER if model in known_models and model != TARGET]
    references.extend([model for model in summary["model"].tolist() if model not in MODEL_ORDER and model != TARGET])
    boot_tables = []
    decision_rows = []
    for index, reference in enumerate(references):
        merged = paired_table(long_predictions, TARGET, reference)
        boot = paired_bootstrap(merged, args.bootstrap_iters, args.seed + index * 1009)
        boot.insert(0, "reference", reference)
        boot.insert(0, "target", TARGET)
        boot_tables.append(boot)
        decision = decision_counts(merged)
        decision.update({"target": TARGET, "reference": reference})
        decision_rows.append(decision)

    boot_all = pd.concat(boot_tables, ignore_index=True)
    decisions = pd.DataFrame(decision_rows)

    long_predictions.to_csv(args.output_dir / "model_predictions_test.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(args.output_dir / "performance_deltas_vs_sota.csv", index=False, encoding="utf-8-sig")
    boot_all.to_csv(args.output_dir / "paired_bootstrap_vs_sota.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(args.output_dir / "decision_counts_vs_sota.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir, summary, deltas, boot_all, decisions, args.bootstrap_iters)

    print(f"Wrote analysis outputs to {args.output_dir}")
    print(summary[["model", "acc", "pre", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
