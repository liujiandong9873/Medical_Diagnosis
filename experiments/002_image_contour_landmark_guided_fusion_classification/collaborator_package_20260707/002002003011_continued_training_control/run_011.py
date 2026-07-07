from __future__ import annotations

import argparse
import json
import os
import random
import shutil
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
FOLD_OUTPUT_DIR = OUTPUT_DIR / "folds"

MODEL_ROOT = ROOT / "Models_saved"
PRETRAINED_CACHE_DIR = MODEL_ROOT / "Downloaded_Pretrained_Models"
SOURCE_OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "002_image_contour_landmark_guided_fusion_classification"
    / "002002_image_contour_prediction"
    / "002002002_image_contour_auxiliary_supervision"
    / "outputs"
)
SOURCE_CHECKPOINT_DIR = MODEL_ROOT / "Generated_Models" / "002002002_contour_auxiliary_resnet18"
CHECKPOINT_DIR = MODEL_ROOT / "Generated_Models" / "002002003011_continued_training_control"

EXPERIMENT_ID = "002002003011"
EXPERIMENT_LABEL = "Continued-training control for contour auxiliary baseline"
TARGET_NAME = "contour_targets_224_uint8.npz"
REQUIRED_INPUTS = [
    "metadata.csv",
    "folds.csv",
    "outer_splits.csv",
    TARGET_NAME,
    "contour_target_manifest.json",
]

TARGET_PATH = OUTPUT_DIR / TARGET_NAME
METADATA_PATH = OUTPUT_DIR / "metadata.csv"
FOLDS_PATH = OUTPUT_DIR / "folds.csv"
OUTER_SPLITS_PATH = OUTPUT_DIR / "outer_splits.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class RunConfig:
    experiment_id: str
    experiment_label: str
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
    load_source_checkpoint: bool
    source_checkpoint_dir: str
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


def prepare_local_inputs() -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in REQUIRED_INPUTS:
        src = SOURCE_OUTPUT_DIR / name
        dst = OUTPUT_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required source file from 002002002: {src}")
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            copied[name] = "copied"
        else:
            copied[name] = "exists"
    return copied


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
    prepare_local_inputs()
    for path in [METADATA_PATH, FOLDS_PATH, OUTER_SPLITS_PATH, TARGET_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required 002002003011 preparation output: {path}")
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


def monitor_score(metrics: dict[str, float], monitor: str) -> float:
    if monitor == "val_auc":
        return metrics["auc"]
    if monitor == "val_f1":
        return metrics["f1"]
    if monitor == "val_balanced_acc":
        return metrics["balanced_acc"]
    raise ValueError(f"Unsupported monitor: {monitor}")


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def load_source_checkpoint(model: nn.Module, fold: int, config: RunConfig) -> dict[str, object]:
    if not config.load_source_checkpoint:
        return {"loaded": False, "checkpoint": "", "reason": "disabled"}
    ckpt_path = Path(config.source_checkpoint_dir) / f"contour_aux_resnet18_fold{fold}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing source 002002002 checkpoint for fold {fold}: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return {
        "loaded": True,
        "checkpoint": str(ckpt_path),
        "source_best_epoch": checkpoint.get("best_epoch", ""),
    }


def train_fold(folds, outer_splits, masks_by_row_id, fold: int, args, config: RunConfig):
    device = torch.device(config.device)
    fold_output_dir = FOLD_OUTPUT_DIR / f"f{fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, val_frame, test_frame = split_frames(folds, outer_splits, fold)
    train_loader, val_loader, test_loader = make_loaders(
        train_frame, val_frame, test_frame, masks_by_row_id, config, device
    )
    model = ResNet18ContourAux(config.pretrained, allow_random_fallback=args.allow_random_fallback).to(device)
    source_init_status = load_source_checkpoint(model, fold, config)
    cls_criterion = make_criterion(train_frame, config, device)
    contour_bce = make_contour_bce(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = make_grad_scaler(device, config.amp)

    initial_val_metrics, _ = evaluate_with_ids(
        model,
        val_loader,
        cls_criterion,
        contour_bce,
        device,
        config,
        split_name="validation",
        outer_fold=fold,
    )
    initial_test_metrics, _ = evaluate_with_ids(
        model,
        test_loader,
        cls_criterion,
        contour_bce,
        device,
        config,
        split_name="test",
        outer_fold=fold,
    )
    best_score = monitor_score(initial_val_metrics, config.monitor)
    best_epoch = 0
    best_state = cpu_state_dict(model)
    epochs_without_improvement = 0
    history = [{"fold": fold, "epoch": 0, "phase": "source_checkpoint"}]
    history[0].update({f"val_{key}": value for key, value in initial_val_metrics.items()})
    history[0].update({f"test_{key}": value for key, value in initial_test_metrics.items()})
    pd.DataFrame(history).to_csv(fold_output_dir / "history_live.csv", index=False, encoding="utf-8-sig")
    print(
        f"fold={fold} epoch=000 source_checkpoint "
        f"val_auc={initial_val_metrics['auc']:.4f} test_auc={initial_test_metrics['auc']:.4f} "
        f"val_f1={initial_val_metrics['f1']:.4f} test_f1={initial_test_metrics['f1']:.4f} "
        f"best_{config.monitor}={best_score:.4f}",
        flush=True,
    )
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
        row = {"fold": fold, "epoch": epoch, "phase": "continued_training"}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row.update({f"test_{key}": value for key, value in test_metrics.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(fold_output_dir / "history_live.csv", index=False, encoding="utf-8-sig")

        monitor_value = monitor_score(val_metrics, config.monitor)
        if monitor_value > best_score:
            best_score = monitor_value
            best_epoch = epoch
            best_state = cpu_state_dict(model)
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
    best_metrics["source_checkpoint_loaded"] = bool(source_init_status.get("loaded", False))
    best_metrics["source_checkpoint"] = str(source_init_status.get("checkpoint", ""))
    best_metrics["source_best_epoch"] = source_init_status.get("source_best_epoch", "")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{config.experiment_id}_fold{fold}.pt"
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
    pd.DataFrame([best_metrics]).to_csv(fold_output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_records).to_csv(fold_output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(fold_output_dir / "history.csv", index=False, encoding="utf-8-sig")
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


def write_summary(config: RunConfig, folds: pd.DataFrame, trained: bool, input_status=None, metrics=None) -> None:
    counts = folds["group"].value_counts().to_dict()
    lines = [
        "# 002002003011 Continued-Training Control Run Summary",
        "",
        "## Status",
        "",
        f"- Experiment: `{config.experiment_id}`",
        f"- Experiment label: {config.experiment_label}",
        "- Model: same ResNet18 shared encoder + classification head + contour decoder as 002002002",
        "- Purpose: test whether extra fine-tuning of the strong contour-auxiliary baseline can explain the 002002003010/010002 gains.",
        "- Initialization: fold-matched 002002002 contour-auxiliary checkpoint.",
        "- Input at inference: raw X-ray image only.",
        "- Training supervision: classification label + y/e contour mask, unchanged from 002002002.",
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
        f"- Source checkpoints: `{SOURCE_CHECKPOINT_DIR}`",
        f"- Generated checkpoints: `{CHECKPOINT_DIR}`",
        "",
        "## Input Reuse",
        "",
        f"- Source outputs: `{SOURCE_OUTPUT_DIR}`",
        "",
        "## Output Files",
        "",
        "- `metadata.csv`",
        "- `folds.csv`",
        "- `outer_splits.csv`",
        "- `contour_targets_224_uint8.npz`",
    ]
    if input_status:
        for name, status in input_status.items():
            lines.append(f"- `{name}`: {status}")
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


def write_training_progress(config: RunConfig, selected_folds: list[int], metrics_by_fold: pd.DataFrame, summary: pd.DataFrame) -> None:
    completed_folds = [int(value) for value in metrics_by_fold["fold"].tolist()]
    remaining_folds = [fold for fold in selected_folds if fold not in completed_folds]
    lines = [
        f"# {config.experiment_id} Training Progress",
        "",
        f"- Status: {'complete' if len(remaining_folds) == 0 else 'running'}",
        f"- Completed folds: {completed_folds}",
        f"- Remaining folds: {remaining_folds}",
        f"- Updated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Fold Metrics",
        "",
        "| Fold | AUC | Acc | Pre | Recall | Specificity | F1 | Best epoch | Source loaded |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in metrics_by_fold.sort_values("fold").iterrows():
        lines.append(
            f"| {int(row['fold'])} | {row['auc']:.4f} | {row['acc']:.4f} | {row['pre']:.4f} | "
            f"{row['recall']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | "
            f"{int(row['best_epoch'])} | {bool(row['source_checkpoint_loaded'])} |"
        )
    lines += [
        "",
        "## Current Summary",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
    ]
    for metric in ["auc", "acc", "pre", "recall", "specificity", "f1", "mean_dice", "loss"]:
        mean_value = summary[f"{metric}_mean"].iloc[0]
        std_value = summary[f"{metric}_std"].iloc[0]
        std_text = "NA" if pd.isna(std_value) else f"{std_value:.4f}"
        lines.append(f"| {metric} | {mean_value:.4f} | {std_text} |")
    (OUTPUT_DIR / "training_progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_incremental_outputs(
    config: RunConfig,
    selected_folds: list[int],
    all_metrics: list[dict],
    all_predictions: list[dict],
    all_history: list[dict],
) -> pd.DataFrame:
    metrics_by_fold = pd.DataFrame(all_metrics)
    predictions = pd.DataFrame(all_predictions)
    history = pd.DataFrame(all_history)
    summary = summarize_metrics(metrics_by_fold)
    metrics_by_fold.to_csv(OUTPUT_DIR / "cv_metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    history.to_csv(OUTPUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "cv_summary.csv", index=False, encoding="utf-8-sig")
    write_training_progress(config, selected_folds, metrics_by_fold, summary)
    return summary


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
    source_init_status = load_source_checkpoint(model, 0, config)
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
                "source_init_status": source_init_status,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-contour", type=float, default=0.3)
    parser.add_argument("--contour-pos-weight", type=float, default=50.0)
    parser.add_argument("--monitor", choices=["val_auc", "val_f1", "val_balanced_acc"], default="val_f1")
    parser.add_argument("--source-checkpoint-dir", default=str(SOURCE_CHECKPOINT_DIR))
    parser.add_argument("--load-source-checkpoint", dest="load_source_checkpoint", action="store_true")
    parser.add_argument("--no-load-source-checkpoint", dest="load_source_checkpoint", action="store_false")
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
    parser.set_defaults(pretrained=True, amp=True, use_pos_weight=True, load_source_checkpoint=True)
    args = parser.parse_args()

    config = RunConfig(
        experiment_id=EXPERIMENT_ID,
        experiment_label=EXPERIMENT_LABEL,
        seed=args.seed,
        n_splits=args.n_splits,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        monitor=args.monitor,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_contour=args.lambda_contour,
        contour_pos_weight=args.contour_pos_weight,
        pretrained=args.pretrained,
        amp=args.amp,
        use_pos_weight=args.use_pos_weight,
        load_source_checkpoint=args.load_source_checkpoint,
        source_checkpoint_dir=args.source_checkpoint_dir,
        num_workers=args.num_workers,
        device=args.device,
    )
    set_seed(config.seed)
    configure_model_storage()
    input_status = prepare_local_inputs()
    _, folds, outer_splits, masks_by_row_id = load_inputs()
    write_summary(config, folds, trained=False, input_status=input_status)
    print(f"Loaded {len(folds)} {config.experiment_id} samples from {OUTPUT_DIR}")
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
        summary = write_incremental_outputs(config, selected_folds, all_metrics, all_predictions, all_history)
        print(f"Saved incremental outputs after fold={fold}", flush=True)

    summary = write_incremental_outputs(config, selected_folds, all_metrics, all_predictions, all_history)
    write_summary(config, folds, trained=True, input_status=input_status, metrics=summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
