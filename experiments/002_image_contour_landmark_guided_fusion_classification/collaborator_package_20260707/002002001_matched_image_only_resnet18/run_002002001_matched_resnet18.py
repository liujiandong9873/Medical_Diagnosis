from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[5]
DATASET_ROOT = ROOT / "Dataset" / "ZYQ_Dataset"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

MODEL_ROOT = ROOT / "Models_saved"
PRETRAINED_CACHE_DIR = MODEL_ROOT / "Downloaded_Pretrained_Models"
CHECKPOINT_DIR = MODEL_ROOT / "Generated_Models" / "002002001_matched_resnet18"

IMAGE_DIR = DATASET_ROOT / "images"
LABEL_DIR = DATASET_ROOT / "labels"
AUDIT_CSV = DATASET_ROOT / "excel_image_label_alias_match_audit.csv"

NORMAL_GROUP = "\u6b63\u5e38"
HYPERTROPHY_GROUP = "\u80a5\u5927"
LEGACY_NORMAL_GROUP = "\u59dd\uff45\u7236"
LEGACY_HYPERTROPHY_GROUP = "\u9472\u30e5\u3047"
GROUP_TO_LABEL = {
    NORMAL_GROUP: 0,
    LEGACY_NORMAL_GROUP: 0,
    HYPERTROPHY_GROUP: 1,
    LEGACY_HYPERTROPHY_GROUP: 1,
}
LABEL_TO_GROUP = {0: "normal", 1: "hypertrophy"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class RunConfig:
    seed: int
    n_splits: int
    image_size: int
    batch_size: int
    epochs: int
    patience: int
    monitor: str
    validation_fraction: float
    lr: float
    weight_decay: float
    pretrained: bool
    amp: bool
    use_pos_weight: bool
    num_workers: int
    device: str


class XrayClassificationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return {
            "image": image,
            "label": label,
            "row_id": int(row["row_id"]),
            "sample_id": str(row["sample_id"]),
            "image_file": str(row["image_file"]),
            "fold": int(row.get("fold", -1)),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_model_storage() -> None:
    PRETRAINED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(PRETRAINED_CACHE_DIR)
    torch.hub.set_dir(str(PRETRAINED_CACHE_DIR))


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def split_files(value) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    for sep in ["|", ";"]:
        if sep in text:
            return [item.strip() for item in text.split(sep) if item.strip()]
    return [text]


def load_label_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="gb18030"))


def label_has_required_contours(label_path: Path) -> tuple[bool, dict[str, bool]]:
    if not label_path.exists():
        return False, {"has_y": False, "has_e": False, "has_b": False, "has_Y": False, "has_B": False}
    data = load_label_json(label_path)
    pairs = {(shape.get("label"), shape.get("shape_type")) for shape in data.get("shapes", [])}
    flags = {
        "has_y": ("y", "linestrip") in pairs,
        "has_e": ("e", "linestrip") in pairs,
        "has_b": ("b", "point") in pairs,
        "has_Y": ("Y", "point") in pairs,
        "has_B": ("B", "line") in pairs,
    }
    return flags["has_y"] and flags["has_e"], flags


def build_metadata() -> pd.DataFrame:
    if not AUDIT_CSV.exists():
        raise FileNotFoundError(f"Missing match audit CSV: {AUDIT_CSV}")

    audit = pd.read_csv(AUDIT_CSV, encoding="utf-8-sig")
    rows = []
    for _, row in audit.iterrows():
        group = str(row.get("image_group", "")).strip()
        if group not in GROUP_TO_LABEL:
            continue
        if not parse_bool(row.get("has_image", False)) or not parse_bool(row.get("has_label", False)):
            continue

        image_files = split_files(row.get("image_files", ""))
        label_files = split_files(row.get("label_files", ""))
        if not image_files or not label_files:
            continue

        selected_label_file = None
        selected_flags = None
        for label_file in label_files:
            label_path = LABEL_DIR / label_file
            has_contours, flags = label_has_required_contours(label_path)
            if has_contours:
                selected_label_file = label_file
                selected_flags = flags
                break
        if selected_label_file is None or selected_flags is None:
            continue

        for image_file in image_files:
            image_path = IMAGE_DIR / image_file
            if not image_path.exists():
                continue
            label_value = GROUP_TO_LABEL[group]
            rows.append(
                {
                    "sample_id": str(row["excel_id"]),
                    "excel_row": int(row["excel_row"]),
                    "image_file": image_file,
                    "image_path": str(image_path),
                    "label_file": selected_label_file,
                    "label_path": str(LABEL_DIR / selected_label_file),
                    "source_group": group,
                    "group": LABEL_TO_GROUP[label_value],
                    "label": label_value,
                    "A_N": row.get("A_N", np.nan),
                    **selected_flags,
                }
            )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise RuntimeError("No labeled image samples with y/e contour annotations were found.")
    metadata = metadata.drop_duplicates(subset=["sample_id", "image_file"]).sort_values(
        ["label", "sample_id", "image_file"]
    )
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=int))
    return metadata.reset_index(drop=True)


def add_group_folds(metadata: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    frame = metadata.copy()
    labels = frame["label"].to_numpy()
    groups = frame["sample_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    frame["fold"] = -1
    for fold, (_, test_index) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        frame.loc[test_index, "fold"] = fold
    if (frame["fold"] < 0).any():
        raise RuntimeError("Some samples were not assigned to a fold.")
    return frame


def split_train_validation(train_pool: pd.DataFrame, fold: int, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = train_pool["label"].to_numpy()
    groups = train_pool["sample_id"].to_numpy()
    val_splits = max(2, int(round(1.0 / config.validation_fraction)))
    splitter = StratifiedGroupKFold(
        n_splits=val_splits,
        shuffle=True,
        random_state=config.seed + 1000 + fold,
    )
    train_index, val_index = next(splitter.split(np.zeros(len(labels)), labels, groups))
    train_frame = train_pool.iloc[train_index].reset_index(drop=True)
    val_frame = train_pool.iloc[val_index].reset_index(drop=True)
    return train_frame, val_frame


def build_outer_split_assignments(fold_frame: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    rows = []
    for fold in range(config.n_splits):
        train_pool = fold_frame[fold_frame["fold"] != fold].reset_index(drop=True)
        test_frame = fold_frame[fold_frame["fold"] == fold].reset_index(drop=True)
        train_frame, val_frame = split_train_validation(train_pool, fold, config)
        for split_name, split_frame in [
            ("train", train_frame),
            ("validation", val_frame),
            ("test", test_frame),
        ]:
            for _, row in split_frame.iterrows():
                rows.append(
                    {
                        "outer_fold": fold,
                        "split": split_name,
                        "row_id": int(row["row_id"]),
                        "sample_id": row["sample_id"],
                        "image_file": row["image_file"],
                        "label_file": row["label_file"],
                        "group": row["group"],
                        "label": int(row["label"]),
                    }
                )
    return pd.DataFrame(rows)


def train_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(pretrained: bool, allow_random_fallback: bool = True) -> nn.Module:
    configure_model_storage()
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except Exception as exc:
            if not allow_random_fallback:
                raise
            print(f"[warn] Could not load ResNet18 ImageNet weights metadata: {exc}")
            weights = None
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        if pretrained and allow_random_fallback:
            print(f"[warn] Could not initialize pretrained ResNet18, falling back to random weights: {exc}")
            model = models.resnet18(weights=None)
        else:
            raise
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def amp_autocast(device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    labels = labels.astype(int)
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    if len(np.unique(labels)) > 1:
        auc = float(roc_auc_score(labels, probs))
    else:
        auc = float("nan")
    return {
        "auc": auc,
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "acc": float(accuracy_score(labels, preds)),
        "pre": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def run_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels = []
    all_probs = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with amp_autocast(device, use_amp):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(labels)
        all_probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_metrics(labels_np.astype(int), probs_np)
    metrics["loss"] = total_loss / max(len(labels_np), 1)
    return metrics


def evaluate_with_ids(model, loader, criterion, device, use_amp: bool, split_name: str, outer_fold: int):
    model.eval()
    total_loss = 0.0
    records = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with amp_autocast(device, use_amp):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            total_loss += float(loss.detach().cpu()) * len(labels_np)
            for row_id, sample_id, image_file, fold, label, prob in zip(
                batch["row_id"],
                batch["sample_id"],
                batch["image_file"],
                batch["fold"],
                labels_np,
                probs,
            ):
                records.append(
                    {
                        "row_id": int(row_id),
                        "sample_id": str(sample_id),
                        "image_file": str(image_file),
                        "outer_fold": int(outer_fold),
                        "source_fold": int(fold),
                        "split": split_name,
                        "label": int(label),
                        "prob_hypertrophy": float(prob),
                        "pred_label": int(prob >= 0.5),
                    }
                )
    labels = np.array([item["label"] for item in records], dtype=int)
    probs = np.array([item["prob_hypertrophy"] for item in records], dtype=float)
    metrics = compute_metrics(labels, probs)
    metrics["loss"] = total_loss / max(len(records), 1)
    return metrics, records


def make_criterion(train_frame: pd.DataFrame, config: RunConfig, device) -> nn.Module:
    if not config.use_pos_weight:
        return nn.BCEWithLogitsLoss()
    labels = train_frame["label"].to_numpy()
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_fold(frame: pd.DataFrame, fold: int, args, config: RunConfig):
    device = torch.device(config.device)
    train_pool = frame[frame["fold"] != fold].reset_index(drop=True)
    test_frame = frame[frame["fold"] == fold].reset_index(drop=True)
    train_frame, val_frame = split_train_validation(train_pool, fold, config)

    train_set = XrayClassificationDataset(train_frame, train_transform(config.image_size))
    val_set = XrayClassificationDataset(val_frame, eval_transform(config.image_size))
    test_set = XrayClassificationDataset(test_frame, eval_transform(config.image_size))

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config.pretrained, allow_random_fallback=args.allow_random_fallback).to(device)
    criterion = make_criterion(train_frame, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(device, config.amp)

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp=config.amp)
        val_metrics, _ = evaluate_with_ids(
            model, val_loader, criterion, device, use_amp=config.amp, split_name="validation", outer_fold=fold
        )
        test_metrics, _ = evaluate_with_ids(
            model, test_loader, criterion, device, use_amp=config.amp, split_name="test", outer_fold=fold
        )
        row = {"fold": fold, "epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        history.append(row)

        monitor_value = val_metrics["auc"] if config.monitor == "val_auc" else val_metrics["f1"]
        if monitor_value > best_score:
            best_score = monitor_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"fold={fold} epoch={epoch:03d} "
            f"train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} "
            f"test_auc={test_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"patience={epochs_without_improvement}/{config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            print(f"fold={fold} early stopping at epoch={epoch} with best_{config.monitor}={best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_val_metrics, val_prediction_records = evaluate_with_ids(
        model, val_loader, criterion, device, use_amp=config.amp, split_name="validation", outer_fold=fold
    )
    best_test_metrics, test_prediction_records = evaluate_with_ids(
        model, test_loader, criterion, device, use_amp=config.amp, split_name="test", outer_fold=fold
    )
    prediction_records = val_prediction_records + test_prediction_records
    best_metrics = {
        "fold": fold,
        "best_epoch": best_epoch,
        "runtime_sec": time.time() - start,
    }
    best_metrics.update({f"val_{key}": value for key, value in best_val_metrics.items()})
    best_metrics.update({f"test_{key}": value for key, value in best_test_metrics.items()})
    best_metrics.update(best_test_metrics)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"matched_resnet18_fold{fold}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "fold": fold,
            "best_epoch": best_epoch,
            "metrics": best_metrics,
        },
        ckpt_path,
    )
    best_metrics["checkpoint"] = str(ckpt_path)
    return best_metrics, prediction_records, history


def summarize_metrics(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    row = {}
    for metric in ["auc", "balanced_acc", "acc", "pre", "recall", "specificity", "f1", "loss"]:
        row[f"{metric}_mean"] = metrics_by_fold[metric].mean()
        row[f"{metric}_std"] = metrics_by_fold[metric].std(ddof=1)
    return pd.DataFrame([row])


def write_summary(config: RunConfig, fold_frame: pd.DataFrame, trained: bool, metrics=None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = fold_frame["group"].value_counts().to_dict()
    lines = [
        "# 002002001 Run Summary",
        "",
        "## Status",
        "",
        "- Backbone selected from 002001: ResNet18",
        "- Selection reason: highest test AUC among 002001 image-only baselines and strongest acc/F1.",
        f"- Prepared matched samples: {len(fold_frame)}",
        f"- Unique sample_id count: {fold_frame['sample_id'].nunique()}",
        f"- Normal samples: {counts.get('normal', 0)}",
        f"- Hypertrophy samples: {counts.get('hypertrophy', 0)}",
        f"- Required annotations: y/e linestrip contours",
        f"- Splitter: StratifiedGroupKFold grouped by sample_id",
        f"- Folds: {config.n_splits}",
        f"- Image size: {config.image_size} x {config.image_size}",
        f"- Trained: {'yes' if trained else 'no'}",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Output Files",
        "",
        "- `metadata.csv`",
        "- `folds.csv`",
        "- `outer_splits.csv`",
        "",
        "## Model Storage",
        "",
        f"- Pretrained weight cache: `{PRETRAINED_CACHE_DIR}`",
        f"- Generated checkpoints: `{CHECKPOINT_DIR}`",
    ]
    if trained:
        lines += [
            "- `cv_predictions.csv`",
            "- `cv_metrics_by_fold.csv`",
            "- `cv_summary.csv`",
            "- `training_history.csv`",
        ]
    if metrics is not None and not metrics.empty:
        lines += [
            "",
            "## Cross-Validation Summary",
            "",
            "| Metric | Mean | Std |",
            "|---|---:|---:|",
        ]
        for metric in ["auc", "balanced_acc", "acc", "pre", "recall", "specificity", "f1", "loss"]:
            lines.append(
                f"| {metric} | {metrics[f'{metric}_mean'].iloc[0]:.4f} | "
                f"{metrics[f'{metric}_std'].iloc[0]:.4f} |"
            )
    (OUTPUT_DIR / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_outputs(config: RunConfig) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata()
    fold_frame = add_group_folds(metadata, config.n_splits, config.seed)
    split_assignments = build_outer_split_assignments(fold_frame, config)
    metadata.to_csv(OUTPUT_DIR / "metadata.csv", index=False, encoding="utf-8-sig")
    fold_frame.to_csv(OUTPUT_DIR / "folds.csv", index=False, encoding="utf-8-sig")
    split_assignments.to_csv(OUTPUT_DIR / "outer_splits.csv", index=False, encoding="utf-8-sig")
    write_summary(config, fold_frame, trained=False)
    return fold_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--allow-random-fallback", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--use-pos-weight", dest="use_pos_weight", action="store_true")
    parser.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.set_defaults(pretrained=True, amp=True, use_pos_weight=True)
    args = parser.parse_args()

    configure_model_storage()
    config = RunConfig(
        seed=args.seed,
        n_splits=args.n_splits,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        monitor="val_auc",
        validation_fraction=args.validation_fraction,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        amp=args.amp,
        use_pos_weight=args.use_pos_weight,
        num_workers=args.num_workers,
        device=args.device,
    )
    set_seed(config.seed)
    fold_frame = prepare_outputs(config)
    print(f"Prepared {len(fold_frame)} matched image-contour samples in {OUTPUT_DIR}")
    print(fold_frame.groupby(["fold", "group"]).size().unstack(fill_value=0).to_string())

    if args.prepare_only:
        return

    selected_folds = args.folds if args.folds is not None and len(args.folds) > 0 else list(range(config.n_splits))
    all_metrics = []
    all_predictions = []
    all_history = []
    for fold in selected_folds:
        fold_metrics, prediction_records, history = train_fold(fold_frame, fold, args, config)
        all_metrics.append(fold_metrics)
        all_predictions.extend(prediction_records)
        all_history.extend(history)

    metrics_by_fold = pd.DataFrame(all_metrics)
    predictions = pd.DataFrame(all_predictions)
    history = pd.DataFrame(all_history)
    summary = summarize_metrics(metrics_by_fold)

    metrics_by_fold.to_csv(OUTPUT_DIR / "cv_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    history.to_csv(OUTPUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False, encoding="utf-8-sig")
    write_summary(config, fold_frame, trained=True, metrics=summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
