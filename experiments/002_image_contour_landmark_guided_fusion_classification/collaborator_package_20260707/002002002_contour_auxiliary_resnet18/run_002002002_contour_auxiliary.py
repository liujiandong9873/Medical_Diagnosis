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
import torch.nn.functional as nnf
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
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

MODEL_ROOT = ROOT / "Models_saved"
PRETRAINED_CACHE_DIR = MODEL_ROOT / "Downloaded_Pretrained_Models"
CHECKPOINT_DIR = MODEL_ROOT / "Generated_Models" / "002002002_contour_auxiliary_resnet18"

TARGET_PATH = OUTPUT_DIR / "contour_targets_224_uint8.npz"
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
FOLDS_PATH = OUTPUT_DIR / "folds.csv"
OUTER_SPLITS_PATH = OUTPUT_DIR / "outer_splits.csv"

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
    lr: float
    weight_decay: float
    lambda_contour: float
    contour_pos_weight: float
    pretrained: bool
    amp: bool
    use_pos_weight: bool
    num_workers: int
    device: str


class XrayContourAuxDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        masks_by_row_id: dict[int, np.ndarray],
        image_size: int,
        training: bool,
    ):
        self.frame = frame.reset_index(drop=True)
        self.masks_by_row_id = masks_by_row_id
        self.image_size = image_size
        self.training = training
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        row_id = int(row["row_id"])
        image = Image.open(row["image_path"]).convert("RGB").resize(
            (self.image_size, self.image_size),
            resample=Image.BILINEAR,
        )
        mask = self.masks_by_row_id[row_id]
        image, mask = self.apply_joint_transform(image, mask)
        image_tensor = tvf.to_tensor(image)
        image_tensor = self.normalize(image_tensor)
        mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": label,
            "row_id": row_id,
            "sample_id": str(row["sample_id"]),
            "image_file": str(row["image_file"]),
            "fold": int(row["fold"]),
        }

    def apply_joint_transform(self, image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
        mask_images = [Image.fromarray(mask[channel], mode="L") for channel in range(mask.shape[0])]
        if self.training:
            angle = random.uniform(-5.0, 5.0)
            translate = (
                int(round(random.uniform(-0.03, 0.03) * self.image_size)),
                int(round(random.uniform(-0.03, 0.03) * self.image_size)),
            )
            scale = random.uniform(0.95, 1.05)
            shear = [0.0, 0.0]
            image = tvf.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask_images = [
                tvf.affine(
                    item,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=shear,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
                for item in mask_images
            ]
            image = tvf.adjust_brightness(image, random.uniform(0.88, 1.12))
            image = tvf.adjust_contrast(image, random.uniform(0.88, 1.12))
        mask_array = np.stack([np.asarray(item, dtype=np.uint8) for item in mask_images], axis=0)
        return image, mask_array


class ResNet18ContourAux(nn.Module):
    def __init__(self, pretrained: bool, allow_random_fallback: bool = True):
        super().__init__()
        configure_model_storage()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except Exception as exc:
                if not allow_random_fallback:
                    raise
                print(f"[warn] Could not load ResNet18 weights metadata: {exc}")
                weights = None
        try:
            resnet = models.resnet18(weights=weights)
        except Exception as exc:
            if pretrained and allow_random_fallback:
                print(f"[warn] Could not initialize pretrained ResNet18; using random weights: {exc}")
                resnet = models.resnet18(weights=None)
            else:
                raise

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.classifier = nn.Linear(512, 1)
        self.decoder = nn.Sequential(
            up_block(512, 256),
            up_block(256, 128),
            up_block(128, 64),
            up_block(64, 32),
            up_block(32, 16),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        pooled = torch.flatten(self.avgpool(features), 1)
        cls_logits = self.classifier(pooled).squeeze(1)
        contour_logits = self.decoder(features)
        return cls_logits, contour_logits


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    for path in [METADATA_PATH, FOLDS_PATH, OUTER_SPLITS_PATH, TARGET_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required 002002002 preparation output: {path}")
    metadata = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
    folds = pd.read_csv(FOLDS_PATH, encoding="utf-8-sig")
    outer_splits = pd.read_csv(OUTER_SPLITS_PATH, encoding="utf-8-sig")
    target_data = np.load(TARGET_PATH, allow_pickle=True)
    masks = target_data["masks"]
    row_ids = target_data["row_ids"].astype(int)
    if masks.shape[0] != len(row_ids):
        raise RuntimeError("Target mask count does not match row_id count.")
    masks_by_row_id = {int(row_id): masks[index] for index, row_id in enumerate(row_ids)}
    missing = sorted(set(metadata["row_id"].astype(int)) - set(masks_by_row_id))
    if missing:
        raise RuntimeError(f"Missing masks for row_ids: {missing[:10]}")
    return metadata, folds, outer_splits, masks_by_row_id


def split_frames(folds: pd.DataFrame, outer_splits: pd.DataFrame, fold: int):
    split_table = outer_splits[outer_splits["outer_fold"] == fold]
    frames = {}
    for split_name in ["train", "validation", "test"]:
        row_ids = split_table.loc[split_table["split"] == split_name, "row_id"].astype(int)
        frame = folds[folds["row_id"].astype(int).isin(set(row_ids))].copy()
        frames[split_name] = frame.reset_index(drop=True)
    return frames["train"], frames["validation"], frames["test"]


def make_criterion(train_frame: pd.DataFrame, config: RunConfig, device) -> nn.Module:
    if not config.use_pos_weight:
        return nn.BCEWithLogitsLoss()
    labels = train_frame["label"].to_numpy()
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def contour_loss_fn(logits: torch.Tensor, targets: torch.Tensor, bce_loss: nn.Module) -> tuple[torch.Tensor, float, float]:
    bce = bce_loss(logits, targets)
    dice = soft_dice_loss(logits, targets)
    return bce + dice, float(bce.detach().cpu()), float(dice.detach().cpu())


def compute_cls_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    labels = labels.astype(int)
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "auc": auc,
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "acc": float(accuracy_score(labels, preds)),
        "pre": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def contour_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    targets = (targets >= 0.5).float()
    dims = (0, 2, 3)
    intersection = torch.sum(preds * targets, dim=dims)
    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return {
        "y_dice": float(dice[0].detach().cpu()),
        "e_dice": float(dice[1].detach().cpu()),
        "mean_dice": float(dice.mean().detach().cpu()),
        "y_iou": float(iou[0].detach().cpu()),
        "e_iou": float(iou[1].detach().cpu()),
        "mean_iou": float(iou.mean().detach().cpu()),
    }


def run_epoch(
    model,
    loader,
    cls_criterion,
    contour_bce,
    optimizer,
    scaler,
    device,
    config: RunConfig,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_cls_loss = 0.0
    total_contour_loss = 0.0
    total_contour_bce = 0.0
    total_contour_dice_loss = 0.0
    all_labels = []
    all_probs = []
    contour_logits_all = []
    contour_targets_all = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with amp_autocast(device, config.amp):
                cls_logits, contour_logits = model(images)
                cls_loss = cls_criterion(cls_logits, labels)
                c_loss, c_bce, c_dice_loss = contour_loss_fn(contour_logits, masks, contour_bce)
                loss = cls_loss + config.lambda_contour * c_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        batch_size = len(labels)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_cls_loss += float(cls_loss.detach().cpu()) * batch_size
        total_contour_loss += float(c_loss.detach().cpu()) * batch_size
        total_contour_bce += c_bce * batch_size
        total_contour_dice_loss += c_dice_loss * batch_size
        all_probs.append(torch.sigmoid(cls_logits.detach()).cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        contour_logits_all.append(contour_logits.detach().cpu())
        contour_targets_all.append(masks.detach().cpu())

    n = max(sum(len(item) for item in all_labels), 1)
    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_cls_metrics(labels_np, probs_np)
    metrics.update(
        {
            "loss": total_loss / n,
            "cls_loss": total_cls_loss / n,
            "contour_loss": total_contour_loss / n,
            "contour_bce": total_contour_bce / n,
            "contour_dice_loss": total_contour_dice_loss / n,
        }
    )
    c_metrics = contour_metrics(torch.cat(contour_logits_all, dim=0), torch.cat(contour_targets_all, dim=0))
    metrics.update(c_metrics)
    return metrics


def evaluate_with_ids(
    model,
    loader,
    cls_criterion,
    contour_bce,
    device,
    config: RunConfig,
    split_name: str,
    outer_fold: int,
):
    model.eval()
    records = []
    total_loss = 0.0
    total_cls_loss = 0.0
    total_contour_loss = 0.0
    total_contour_bce = 0.0
    total_contour_dice_loss = 0.0
    contour_logits_all = []
    contour_targets_all = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with amp_autocast(device, config.amp):
                cls_logits, contour_logits = model(images)
                cls_loss = cls_criterion(cls_logits, labels)
                c_loss, c_bce, c_dice_loss = contour_loss_fn(contour_logits, masks, contour_bce)
                loss = cls_loss + config.lambda_contour * c_loss
            probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            batch_size = len(labels_np)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_cls_loss += float(cls_loss.detach().cpu()) * batch_size
            total_contour_loss += float(c_loss.detach().cpu()) * batch_size
            total_contour_bce += c_bce * batch_size
            total_contour_dice_loss += c_dice_loss * batch_size
            contour_logits_all.append(contour_logits.detach().cpu())
            contour_targets_all.append(masks.detach().cpu())
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
    n = max(len(records), 1)
    metrics = compute_cls_metrics(labels, probs)
    metrics.update(
        {
            "loss": total_loss / n,
            "cls_loss": total_cls_loss / n,
            "contour_loss": total_contour_loss / n,
            "contour_bce": total_contour_bce / n,
            "contour_dice_loss": total_contour_dice_loss / n,
        }
    )
    c_metrics = contour_metrics(torch.cat(contour_logits_all, dim=0), torch.cat(contour_targets_all, dim=0))
    metrics.update(c_metrics)
    return metrics, records


def make_loaders(train_frame, val_frame, test_frame, masks_by_row_id, config: RunConfig, device):
    train_set = XrayContourAuxDataset(train_frame, masks_by_row_id, config.image_size, training=True)
    val_set = XrayContourAuxDataset(val_frame, masks_by_row_id, config.image_size, training=False)
    test_set = XrayContourAuxDataset(test_frame, masks_by_row_id, config.image_size, training=False)
    kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def make_contour_bce(config: RunConfig, device):
    pos_weight = torch.tensor([config.contour_pos_weight, config.contour_pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, 2, 1, 1))


def train_fold(folds, outer_splits, masks_by_row_id, fold: int, args, config: RunConfig):
    device = torch.device(config.device)
    train_frame, val_frame, test_frame = split_frames(folds, outer_splits, fold)
    train_loader, val_loader, test_loader = make_loaders(
        train_frame, val_frame, test_frame, masks_by_row_id, config, device
    )
    model = ResNet18ContourAux(config.pretrained, allow_random_fallback=args.allow_random_fallback).to(device)
    cls_criterion = make_criterion(train_frame, config, device)
    contour_bce = make_contour_bce(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(device, config.amp)

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, cls_criterion, contour_bce, optimizer, scaler, device, config
        )
        val_metrics, _ = evaluate_with_ids(
            model,
            val_loader,
            cls_criterion,
            contour_bce,
            device,
            config,
            split_name="validation",
            outer_fold=fold,
        )
        test_metrics, _ = evaluate_with_ids(
            model,
            test_loader,
            cls_criterion,
            contour_bce,
            device,
            config,
            split_name="test",
            outer_fold=fold,
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
            f"test_auc={test_metrics['auc']:.4f} val_dice={val_metrics['mean_dice']:.4f} "
            f"test_dice={test_metrics['mean_dice']:.4f} patience={epochs_without_improvement}/{config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            print(f"fold={fold} early stopping at epoch={epoch} with best_{config.monitor}={best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_val_metrics, val_prediction_records = evaluate_with_ids(
        model,
        val_loader,
        cls_criterion,
        contour_bce,
        device,
        config,
        split_name="validation",
        outer_fold=fold,
    )
    best_test_metrics, test_prediction_records = evaluate_with_ids(
        model,
        test_loader,
        cls_criterion,
        contour_bce,
        device,
        config,
        split_name="test",
        outer_fold=fold,
    )
    best_metrics = {
        "fold": fold,
        "best_epoch": best_epoch,
        "runtime_sec": time.time() - start,
    }
    best_metrics.update({f"val_{key}": value for key, value in best_val_metrics.items()})
    best_metrics.update({f"test_{key}": value for key, value in best_test_metrics.items()})
    best_metrics.update(best_test_metrics)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"contour_aux_resnet18_fold{fold}.pt"
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
    prediction_records = val_prediction_records + test_prediction_records
    return best_metrics, prediction_records, history


def summarize_metrics(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    row = {}
    metrics = [
        "auc",
        "balanced_acc",
        "acc",
        "pre",
        "recall",
        "specificity",
        "f1",
        "loss",
        "cls_loss",
        "contour_loss",
        "mean_dice",
        "y_dice",
        "e_dice",
        "mean_iou",
        "y_iou",
        "e_iou",
    ]
    for metric in metrics:
        row[f"{metric}_mean"] = metrics_by_fold[metric].mean()
        row[f"{metric}_std"] = metrics_by_fold[metric].std(ddof=1)
    return pd.DataFrame([row])


def write_summary(config: RunConfig, folds: pd.DataFrame, trained: bool, metrics=None) -> None:
    counts = folds["group"].value_counts().to_dict()
    lines = [
        "# 002002002 Run Summary",
        "",
        "## Status",
        "",
        "- Model: ResNet18 shared encoder + classification head + contour decoder",
        "- Direct baseline: 002002001 matched ResNet18 image-only baseline",
        "- Manual contours are used only for training supervision, not as model input.",
        f"- Samples: {len(folds)}",
        f"- Unique sample_id count: {folds['sample_id'].nunique()}",
        f"- Normal samples: {counts.get('normal', 0)}",
        f"- Hypertrophy samples: {counts.get('hypertrophy', 0)}",
        f"- Folds: {config.n_splits}",
        f"- Image/mask size: {config.image_size} x {config.image_size}",
        f"- Trained: {'yes' if trained else 'no'}",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Model Storage",
        "",
        f"- Pretrained weight cache: `{PRETRAINED_CACHE_DIR}`",
        f"- Generated checkpoints: `{CHECKPOINT_DIR}`",
        "",
        "## Output Files",
        "",
        "- `metadata.csv`",
        "- `folds.csv`",
        "- `outer_splits.csv`",
        "- `contour_targets_224_uint8.npz`",
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
        for metric in [
            "auc",
            "balanced_acc",
            "acc",
            "pre",
            "recall",
            "specificity",
            "f1",
            "mean_dice",
            "y_dice",
            "e_dice",
            "mean_iou",
            "loss",
        ]:
            lines.append(
                f"| {metric} | {metrics[f'{metric}_mean'].iloc[0]:.4f} | "
                f"{metrics[f'{metric}_std'].iloc[0]:.4f} |"
            )
    (OUTPUT_DIR / "training_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke_test(folds, outer_splits, masks_by_row_id, args, config: RunConfig) -> None:
    device = torch.device(config.device)
    train_frame, _, _ = split_frames(folds, outer_splits, fold=0)
    smoke_frame = train_frame.head(min(8, len(train_frame))).copy()
    loader = DataLoader(
        XrayContourAuxDataset(smoke_frame, masks_by_row_id, config.image_size, training=True),
        batch_size=min(4, len(smoke_frame)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = ResNet18ContourAux(config.pretrained, allow_random_fallback=args.allow_random_fallback).to(device)
    cls_criterion = make_criterion(train_frame, config, device)
    contour_bce = make_contour_bce(config, device)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    masks = batch["mask"].to(device)
    with amp_autocast(device, config.amp):
        cls_logits, contour_logits = model(images)
        cls_loss = cls_criterion(cls_logits, labels)
        c_loss, c_bce, c_dice_loss = contour_loss_fn(contour_logits, masks, contour_bce)
        loss = cls_loss + config.lambda_contour * c_loss
    print(
        json.dumps(
            {
                "smoke_test": "ok",
                "image_shape": list(images.shape),
                "mask_shape": list(masks.shape),
                "cls_logits_shape": list(cls_logits.shape),
                "contour_logits_shape": list(contour_logits.shape),
                "loss": float(loss.detach().cpu()),
                "cls_loss": float(cls_loss.detach().cpu()),
                "contour_loss": float(c_loss.detach().cpu()),
                "contour_bce": c_bce,
                "contour_dice_loss": c_dice_loss,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-contour", type=float, default=0.3)
    parser.add_argument("--contour-pos-weight", type=float, default=50.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--allow-random-fallback", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--use-pos-weight", dest="use_pos_weight", action="store_true")
    parser.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.set_defaults(pretrained=True, amp=True, use_pos_weight=True)
    args = parser.parse_args()

    config = RunConfig(
        seed=args.seed,
        n_splits=args.n_splits,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        monitor="val_auc",
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_contour=args.lambda_contour,
        contour_pos_weight=args.contour_pos_weight,
        pretrained=args.pretrained,
        amp=args.amp,
        use_pos_weight=args.use_pos_weight,
        num_workers=args.num_workers,
        device=args.device,
    )
    set_seed(config.seed)
    configure_model_storage()
    _, folds, outer_splits, masks_by_row_id = load_inputs()
    write_summary(config, folds, trained=False)
    print(f"Loaded {len(folds)} 002002002 samples from {OUTPUT_DIR}")
    print(folds.groupby(["fold", "group"]).size().unstack(fill_value=0).to_string())

    if args.smoke_test:
        run_smoke_test(folds, outer_splits, masks_by_row_id, args, config)
        return
    if args.prepare_only:
        return

    selected_folds = args.folds if args.folds is not None and len(args.folds) > 0 else list(range(config.n_splits))
    all_metrics = []
    all_predictions = []
    all_history = []
    for fold in selected_folds:
        fold_metrics, prediction_records, history = train_fold(folds, outer_splits, masks_by_row_id, fold, args, config)
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
    write_summary(config, folds, trained=True, metrics=summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
