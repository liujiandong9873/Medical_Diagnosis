from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


KEYS = ["row_id", "sample_id", "image_file", "outer_fold", "split", "label"]
TEST_KEYS = ["row_id", "sample_id", "image_file", "outer_fold", "label"]
METRICS = ["acc", "pre", "recall", "f1"]
ORIGINAL_MODEL = "direct_f2mi_calibrated_original"


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    experiment_dir = analysis_dir.parents[1]
    base_dir = experiment_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=experiment_dir)
    parser.add_argument(
        "--formal-comparison-predictions",
        type=Path,
        default=experiment_dir / "analysis" / "012001_out" / "model_predictions_test.csv",
    )
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=base_dir / "002002002_image_contour_auxiliary_supervision" / "outputs" / "cv_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "analysis" / "012002_out")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def clip_prob(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(prob.astype(float), eps, 1.0 - eps)


def logit_from_prob(prob: np.ndarray) -> np.ndarray:
    prob = clip_prob(prob)
    return np.log(prob / (1.0 - prob))


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    pred = pred.astype(int)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "pre": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


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


def load_cv_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {*KEYS, "prob_hypertrophy", "pred_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[[*KEYS, "prob_hypertrophy", "pred_label"]].copy()


def load_seed_prediction(seed_dir: Path) -> pd.DataFrame | None:
    path = seed_dir / "cv_predictions.csv"
    if not path.exists():
        return None
    df = load_cv_predictions(path)
    observed_test_folds = sorted(df[df["split"] == "test"]["outer_fold"].unique().tolist())
    if observed_test_folds != [0, 1, 2, 3, 4]:
        return None

    outputs = []
    for fold in observed_test_folds:
        val_df = df[(df["outer_fold"] == fold) & (df["split"] == "validation")].copy()
        test_df = df[(df["outer_fold"] == fold) & (df["split"] == "test")].copy()
        platt = fit_platt(val_df["label"].to_numpy(), val_df["prob_hypertrophy"].to_numpy(dtype=float))
        prob_test = apply_platt(platt, test_df["prob_hypertrophy"].to_numpy(dtype=float))
        out = test_df[TEST_KEYS].copy()
        out["model"] = f"{seed_dir.name}_platt"
        out["prob_hypertrophy"] = prob_test
        out["pred_label"] = (prob_test >= 0.5).astype(int)
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def load_original_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {*TEST_KEYS, "model", "prob_hypertrophy", "pred_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    out = df[df["model"] == "direct_f2mi_calibrated"].copy()
    out["model"] = ORIGINAL_MODEL
    return out[[*TEST_KEYS, "model", "prob_hypertrophy", "pred_label"]].copy()


def load_baseline_predictions(path: Path) -> pd.DataFrame:
    df = load_cv_predictions(path)
    df = df[df["split"] == "test"].copy()
    out = df[TEST_KEYS].copy()
    out["model"] = "baseline002002002"
    out["prob_hypertrophy"] = df["prob_hypertrophy"].astype(float).to_numpy()
    out["pred_label"] = df["pred_label"].astype(int).to_numpy()
    return out


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
    return target_df.merge(
        ref_df[[*TEST_KEYS, "prob_hypertrophy", "pred_label"]],
        on=TEST_KEYS,
        validate="one_to_one",
        suffixes=("_target", "_reference"),
    )


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
        "target_only_correct": int((target_correct & ~ref_correct).sum()),
        "reference_only_correct": int((~target_correct & ref_correct).sum()),
        "net_correct_gain": int((target_correct & ~ref_correct).sum() - (~target_correct & ref_correct).sum()),
        "fn_diff_target_minus_reference": int(target_fn - ref_fn),
        "fp_diff_target_minus_reference": int(target_fp - ref_fp),
    }


def metric_line(table: pd.DataFrame, metric: str) -> str:
    row = table[table["metric"] == metric].iloc[0]
    return (
        f"{row['diff_target_minus_reference']:+.4f} "
        f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}], p={row['bootstrap_p_two_sided']:.3f}"
    )


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    boot_all: pd.DataFrame,
    decisions: pd.DataFrame,
    complete_seed_models: list[str],
    skipped_seed_dirs: list[str],
    n_iters: int,
) -> None:
    lines = [
        "# 002002003012002 Seed Robustness Analysis",
        "",
        "## Scope",
        "",
        "- Target family: direct F2MI with validation-fitted Platt calibration",
        "- Original target: `direct_f2mi_calibrated_original` from 010003 no_residual",
        "- Repeated seed outputs are loaded from `outputs/seed_*`",
        f"- Paired cluster bootstrap iterations: {n_iters}",
        "",
        "## Completed Seed Models",
        "",
    ]
    if complete_seed_models:
        for model in complete_seed_models:
            lines.append(f"- `{model}`")
    else:
        lines.append("- No complete repeated seed model found yet.")
    if skipped_seed_dirs:
        lines += ["", "## Skipped Seed Directories", ""]
        for name in skipped_seed_dirs:
            lines.append(f"- `{name}`")

    lines += [
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

    if len(boot_all):
        lines += [
            "",
            "## Paired Bootstrap",
            "",
            "| Target | Reference | Acc diff | Pre diff | Recall diff | F1 diff |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for (target, reference), table in boot_all.groupby(["target", "reference"], sort=False):
            lines.append(
                f"| {target} | {reference} | {metric_line(table, 'acc')} | "
                f"{metric_line(table, 'pre')} | {metric_line(table, 'recall')} | {metric_line(table, 'f1')} |"
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
        "## Interpretation Rule",
        "",
        "A repeated seed supports the final proposed model if it preserves the main advantage over `baseline002002002`, especially Acc, Pre, Recall, and F1.",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pieces = [load_original_predictions(args.formal_comparison_predictions), load_baseline_predictions(args.baseline_predictions)]
    complete_seed_models = []
    skipped_seed_dirs = []
    for seed_dir in sorted((args.experiment_dir / "outputs").glob("seed_*")):
        seed_pred = load_seed_prediction(seed_dir)
        if seed_pred is None:
            skipped_seed_dirs.append(seed_dir.name)
            continue
        complete_seed_models.append(seed_pred["model"].iloc[0])
        pieces.append(seed_pred)

    long_predictions = pd.concat(pieces, ignore_index=True)
    summary = summarize(long_predictions)

    boot_tables = []
    decision_rows = []
    comparisons = []
    for model in complete_seed_models:
        comparisons.append((model, "baseline002002002"))
        comparisons.append((model, ORIGINAL_MODEL))
    comparisons.append((ORIGINAL_MODEL, "baseline002002002"))

    for index, (target, reference) in enumerate(comparisons):
        merged = paired_table(long_predictions, target, reference)
        boot = paired_bootstrap(merged, args.bootstrap_iters, args.seed + index * 1009)
        boot.insert(0, "reference", reference)
        boot.insert(0, "target", target)
        boot_tables.append(boot)
        decision = decision_counts(merged)
        decision.update({"target": target, "reference": reference})
        decision_rows.append(decision)

    boot_all = pd.concat(boot_tables, ignore_index=True) if boot_tables else pd.DataFrame()
    decisions = pd.DataFrame(decision_rows)

    long_predictions.to_csv(args.output_dir / "model_predictions_test.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    boot_all.to_csv(args.output_dir / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(args.output_dir / "decision_counts.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir, summary, boot_all, decisions, complete_seed_models, skipped_seed_dirs, args.bootstrap_iters)

    print(summary[["model", "acc", "pre", "recall", "f1"]].to_string(index=False))
    if skipped_seed_dirs:
        print(f"Skipped incomplete seed dirs: {', '.join(skipped_seed_dirs)}")


if __name__ == "__main__":
    main()
