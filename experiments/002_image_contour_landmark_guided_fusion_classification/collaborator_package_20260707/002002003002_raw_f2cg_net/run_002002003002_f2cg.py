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
import torch.nn.functional as F
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

SOURCE_OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "002_image_contour_landmark_guided_fusion_classification"
    / "002002_image_contour_prediction"
    / "002002002_image_contour_auxiliary_supervision"
    / "outputs"
)

MODEL_ROOT = ROOT / "Models_saved"
PRETRAINED_CACHE_DIR = MODEL_ROOT / "Downloaded_Pretrained_Models"
CHECKPOINT_DIR = MODEL_ROOT / "Generated_Models" / "002002003002_f2cg_resnet18"

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
    lambda_fractal: float
    contour_pos_weight: float
    fractal_gamma: float
    fractal_embedding_dim: int
    fractional_embedding_dim: int
    fusion_hidden_dim: int
    fusion_dropout: float
    detach_fractal_target: bool
    pretrained: bool
    amp: bool
    use_pos_weight: bool
    num_workers: int
    device: str


class XrayContourDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, masks_by_row_id: dict[int, np.ndarray], image_size: int, training: bool):
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
        image = Image.open(row["image_path"]).convert("RGB").resize((self.image_size, self.image_size))
        mask = self.masks_by_row_id[row_id]
        image, mask = self.apply_joint_transform(image, mask)
        image_tensor = self.normalize(tvf.to_tensor(image))
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
            image = tvf.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask_images = [
                tvf.affine(
                    item,
                    angle=angle,
                    translate=translate,
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
                for item in mask_images
            ]
            image = tvf.adjust_brightness(image, random.uniform(0.88, 1.12))
            image = tvf.adjust_contrast(image, random.uniform(0.88, 1.12))
        return image, np.stack([np.asarray(item, dtype=np.uint8) for item in mask_images], axis=0)


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class GatedFusionBlock(nn.Module):
    def __init__(self, channels: int, initial_alpha: float = -1.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(float(initial_alpha)))

    def forward(self, image_feature: torch.Tensor, contour_feature: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([image_feature, contour_feature], dim=1))
        alpha = torch.sigmoid(self.alpha_logit)
        return image_feature + alpha * gate * contour_feature


class DifferentiableMultifractalLayer(nn.Module):
    def __init__(
        self,
        scales: tuple[int, ...] = (2, 4, 8, 16, 32),
        q_values: tuple[float, ...] = (0.5, 1.0, 2.0),
        gamma: float = 0.05,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.scales = scales
        self.q_values = q_values
        self.gamma = gamma
        self.vector_dim = 2 * len(scales) + 2 + 2 * len(scales) * len(q_values)
        self.projection = nn.Sequential(
            nn.Linear(self.vector_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def extract_scale_features(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-6
        log_counts = []
        q_moments = []
        for scale in self.scales:
            box_sum = F.avg_pool2d(probs, kernel_size=scale, stride=scale) * float(scale * scale)
            occupancy = 1.0 - torch.exp(-self.gamma * box_sum)
            count = occupancy.sum(dim=(2, 3)) + eps
            log_counts.append(torch.log(count))

            mass = box_sum + eps
            mass = mass / mass.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
            for q in self.q_values:
                partition = torch.pow(mass, q).sum(dim=(2, 3)).clamp_min(eps)
                q_moments.append(torch.log(partition))

        log_count_tensor = torch.stack(log_counts, dim=-1)
        q_tensor = torch.stack(q_moments, dim=-1)
        x = torch.log(1.0 / torch.tensor(self.scales, dtype=probs.dtype, device=probs.device))
        x = x - x.mean()
        y = log_count_tensor - log_count_tensor.mean(dim=-1, keepdim=True)
        slope = (y * x.view(1, 1, -1)).sum(dim=-1) / (x.pow(2).sum() + eps)
        scale_sequence = log_count_tensor.mean(dim=1)
        vector = torch.cat(
            [
                log_count_tensor.flatten(1),
                slope,
                q_tensor.flatten(1),
            ],
            dim=1,
        )
        return vector, scale_sequence

    def forward(self, probs: torch.Tensor) -> dict[str, torch.Tensor]:
        scale_vector, scale_sequence = self.extract_scale_features(probs)
        return {
            "scale_vector": scale_vector,
            "scale_sequence": scale_sequence,
            "embedding": self.projection(scale_vector),
        }


class FractionalScaleMemory(nn.Module):
    def __init__(self, num_scales: int, embedding_dim: int = 32):
        super().__init__()
        self.num_scales = num_scales
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.projection = nn.Sequential(
            nn.Linear(num_scales, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def fractional_coefficients(self, dtype, device) -> torch.Tensor:
        alpha = 0.05 + 0.9 * torch.sigmoid(self.alpha_logit)
        coeffs = [torch.ones((), dtype=dtype, device=device)]
        for j in range(1, self.num_scales):
            coeffs.append(coeffs[-1] * (alpha - float(j - 1)) / float(j) * -1.0)
        return torch.stack(coeffs)

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        coeffs = self.fractional_coefficients(sequence.dtype, sequence.device)
        outputs = []
        for k in range(self.num_scales):
            value = 0.0
            for j in range(k + 1):
                value = value + coeffs[j] * sequence[:, k - j]
            outputs.append(value)
        memory = torch.stack(outputs, dim=1)
        alpha = 0.05 + 0.9 * torch.sigmoid(self.alpha_logit)
        return {
            "memory": memory,
            "embedding": self.projection(memory),
            "alpha": alpha,
        }


class F2CGNet(nn.Module):
    def __init__(
        self,
        pretrained: bool,
        fractal_gamma: float,
        fractal_embedding_dim: int,
        fractional_embedding_dim: int,
        fusion_hidden_dim: int,
        fusion_dropout: float,
        allow_random_fallback: bool = True,
    ):
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

        self.dec4 = up_block(512, 256)
        self.dec3 = up_block(256, 128)
        self.dec2 = up_block(128, 64)
        self.dec1 = up_block(64, 32)
        self.dec0 = up_block(32, 16)
        self.contour_head = nn.Conv2d(16, 2, kernel_size=1)

        self.gate4 = GatedFusionBlock(256)
        self.gate3 = GatedFusionBlock(128)
        self.gate2 = GatedFusionBlock(64)

        self.fractal = DifferentiableMultifractalLayer(gamma=fractal_gamma, embedding_dim=fractal_embedding_dim)
        self.fractional = FractionalScaleMemory(
            num_scales=len(self.fractal.scales),
            embedding_dim=fractional_embedding_dim,
        )
        image_dim = 512 + 256 + 128 + 64
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
        )
        final_dim = 256 + fractal_embedding_dim + fractional_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    @staticmethod
    def gap(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        d4 = self.dec4(c5)
        f4 = self.gate4(c4, d4)
        d3 = self.dec3(d4)
        f3 = self.gate3(c3, d3)
        d2 = self.dec2(d3)
        f2 = self.gate2(c2, d2)
        d1 = self.dec1(d2)
        d0 = self.dec0(d1)
        contour_logits = self.contour_head(d0)
        contour_probs = torch.sigmoid(contour_logits)

        fractal_out = self.fractal(contour_probs)
        fractional_out = self.fractional(fractal_out["scale_sequence"])
        image_feature = torch.cat([self.gap(c5), self.gap(f4), self.gap(f3), self.gap(f2)], dim=1)
        fused = torch.cat(
            [
                self.image_projection(image_feature),
                fractal_out["embedding"],
                fractional_out["embedding"],
            ],
            dim=1,
        )
        cls_logits = self.classifier(fused).squeeze(1)
        extras = {
            "fractal_scale_vector": fractal_out["scale_vector"],
            "fractal_scale_sequence": fractal_out["scale_sequence"],
            "fractal_embedding": fractal_out["embedding"],
            "fractional_memory": fractional_out["memory"],
            "fractional_embedding": fractional_out["embedding"],
            "fractional_alpha": fractional_out["alpha"],
        }
        return cls_logits, contour_logits, extras


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    prepare_local_inputs()
    metadata = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
    folds = pd.read_csv(FOLDS_PATH, encoding="utf-8-sig")
    outer_splits = pd.read_csv(OUTER_SPLITS_PATH, encoding="utf-8-sig")
    target_data = np.load(TARGET_PATH, allow_pickle=True)
    masks = target_data["masks"]
    row_ids = target_data["row_ids"].astype(int)
    if masks.shape[0] != len(row_ids):
        raise RuntimeError("Target mask count does not match row_id count.")
    masks_by_row_id = {int(row_id): masks[index] for index, row_id in enumerate(row_ids)}
    return metadata, folds, outer_splits, masks_by_row_id


def split_frames(folds: pd.DataFrame, outer_splits: pd.DataFrame, fold: int):
    split_table = outer_splits[outer_splits["outer_fold"] == fold]
    frames = {}
    for split_name in ["train", "validation", "test"]:
        row_ids = split_table.loc[split_table["split"] == split_name, "row_id"].astype(int)
        frame = folds[folds["row_id"].astype(int).isin(set(row_ids))].copy()
        frames[split_name] = frame.reset_index(drop=True)
    return frames["train"], frames["validation"], frames["test"]


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


def make_cls_criterion(train_frame: pd.DataFrame, config: RunConfig, device) -> nn.Module:
    if not config.use_pos_weight:
        return nn.BCEWithLogitsLoss()
    labels = train_frame["label"].to_numpy()
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_contour_bce(config: RunConfig, device):
    pos_weight = torch.tensor([config.contour_pos_weight, config.contour_pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, 2, 1, 1))


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


def make_loaders(train_frame, val_frame, test_frame, masks_by_row_id, config: RunConfig, device):
    kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        XrayContourDataset(train_frame, masks_by_row_id, config.image_size, training=True),
        shuffle=True,
        **kwargs,
    )
    val_loader = DataLoader(
        XrayContourDataset(val_frame, masks_by_row_id, config.image_size, training=False),
        shuffle=False,
        **kwargs,
    )
    test_loader = DataLoader(
        XrayContourDataset(test_frame, masks_by_row_id, config.image_size, training=False),
        shuffle=False,
        **kwargs,
    )
    return train_loader, val_loader, test_loader


def build_model(config: RunConfig, allow_random_fallback: bool) -> F2CGNet:
    return F2CGNet(
        pretrained=config.pretrained,
        fractal_gamma=config.fractal_gamma,
        fractal_embedding_dim=config.fractal_embedding_dim,
        fractional_embedding_dim=config.fractional_embedding_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        fusion_dropout=config.fusion_dropout,
        allow_random_fallback=allow_random_fallback,
    )


def compute_losses(model, images, labels, masks, cls_criterion, contour_bce, config: RunConfig):
    cls_logits, contour_logits, extras = model(images)
    cls_loss = cls_criterion(cls_logits, labels)
    c_loss, c_bce, c_dice_loss = contour_loss_fn(contour_logits, masks, contour_bce)
    target_vector, _ = model.fractal.extract_scale_features(masks)
    if config.detach_fractal_target:
        target_vector = target_vector.detach()
    fractal_loss = F.smooth_l1_loss(extras["fractal_scale_vector"], target_vector)
    loss = cls_loss + config.lambda_contour * c_loss + config.lambda_fractal * fractal_loss
    return cls_logits, contour_logits, extras, loss, cls_loss, c_loss, c_bce, c_dice_loss, fractal_loss


def run_epoch(model, loader, cls_criterion, contour_bce, optimizer, scaler, device, config: RunConfig):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "cls_loss": 0.0, "contour_loss": 0.0, "contour_bce": 0.0, "contour_dice_loss": 0.0, "fractal_loss": 0.0}
    all_labels, all_probs, contour_logits_all, contour_targets_all = [], [], [], []
    alphas = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with amp_autocast(device, config.amp):
                cls_logits, contour_logits, extras, loss, cls_loss, c_loss, c_bce, c_dice_loss, fractal_loss = compute_losses(
                    model, images, labels, masks, cls_criterion, contour_bce, config
                )
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
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["cls_loss"] += float(cls_loss.detach().cpu()) * batch_size
        totals["contour_loss"] += float(c_loss.detach().cpu()) * batch_size
        totals["contour_bce"] += c_bce * batch_size
        totals["contour_dice_loss"] += c_dice_loss * batch_size
        totals["fractal_loss"] += float(fractal_loss.detach().cpu()) * batch_size
        all_probs.append(torch.sigmoid(cls_logits.detach()).cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        contour_logits_all.append(contour_logits.detach().cpu())
        contour_targets_all.append(masks.detach().cpu())
        alphas.append(float(extras["fractional_alpha"].detach().cpu()))

    n = max(sum(len(item) for item in all_labels), 1)
    labels_np = np.concatenate(all_labels)
    probs_np = np.concatenate(all_probs)
    metrics = compute_cls_metrics(labels_np, probs_np)
    metrics.update({key: value / n for key, value in totals.items()})
    metrics.update(contour_metrics(torch.cat(contour_logits_all, dim=0), torch.cat(contour_targets_all, dim=0)))
    metrics["fractional_alpha"] = float(np.mean(alphas))
    return metrics


def evaluate_with_ids(model, loader, cls_criterion, contour_bce, device, config: RunConfig, split_name: str, outer_fold: int):
    model.eval()
    records = []
    totals = {"loss": 0.0, "cls_loss": 0.0, "contour_loss": 0.0, "contour_bce": 0.0, "contour_dice_loss": 0.0, "fractal_loss": 0.0}
    contour_logits_all, contour_targets_all = [], []
    alphas = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with amp_autocast(device, config.amp):
                cls_logits, contour_logits, extras, loss, cls_loss, c_loss, c_bce, c_dice_loss, fractal_loss = compute_losses(
                    model, images, labels, masks, cls_criterion, contour_bce, config
                )
            probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            batch_size = len(labels_np)
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["cls_loss"] += float(cls_loss.detach().cpu()) * batch_size
            totals["contour_loss"] += float(c_loss.detach().cpu()) * batch_size
            totals["contour_bce"] += c_bce * batch_size
            totals["contour_dice_loss"] += c_dice_loss * batch_size
            totals["fractal_loss"] += float(fractal_loss.detach().cpu()) * batch_size
            contour_logits_all.append(contour_logits.detach().cpu())
            contour_targets_all.append(masks.detach().cpu())
            alphas.append(float(extras["fractional_alpha"].detach().cpu()))
            for row_id, sample_id, image_file, fold, label, prob in zip(
                batch["row_id"], batch["sample_id"], batch["image_file"], batch["fold"], labels_np, probs
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
    metrics.update({key: value / n for key, value in totals.items()})
    metrics.update(contour_metrics(torch.cat(contour_logits_all, dim=0), torch.cat(contour_targets_all, dim=0)))
    metrics["fractional_alpha"] = float(np.mean(alphas))
    return metrics, records


def train_fold(folds, outer_splits, masks_by_row_id, fold: int, args, config: RunConfig):
    device = torch.device(config.device)
    train_frame, val_frame, test_frame = split_frames(folds, outer_splits, fold)
    train_loader, val_loader, test_loader = make_loaders(train_frame, val_frame, test_frame, masks_by_row_id, config, device)
    model = build_model(config, allow_random_fallback=args.allow_random_fallback).to(device)
    cls_criterion = make_cls_criterion(train_frame, config, device)
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
        train_metrics = run_epoch(model, train_loader, cls_criterion, contour_bce, optimizer, scaler, device, config)
        val_metrics, _ = evaluate_with_ids(model, val_loader, cls_criterion, contour_bce, device, config, "validation", fold)
        test_metrics, _ = evaluate_with_ids(model, test_loader, cls_criterion, contour_bce, device, config, "test", fold)
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
            f"alpha={val_metrics['fractional_alpha']:.3f} patience={epochs_without_improvement}/{config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            print(f"fold={fold} early stopping at epoch={epoch} with best_{config.monitor}={best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_val_metrics, val_records = evaluate_with_ids(model, val_loader, cls_criterion, contour_bce, device, config, "validation", fold)
    best_test_metrics, test_records = evaluate_with_ids(model, test_loader, cls_criterion, contour_bce, device, config, "test", fold)
    best_metrics = {"fold": fold, "best_epoch": best_epoch, "runtime_sec": time.time() - start}
    best_metrics.update({f"val_{key}": value for key, value in best_val_metrics.items()})
    best_metrics.update({f"test_{key}": value for key, value in best_test_metrics.items()})
    best_metrics.update(best_test_metrics)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"f2cg_resnet18_fold{fold}.pt"
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
    return best_metrics, val_records + test_records, history


def summarize_metrics(metrics_by_fold: pd.DataFrame) -> pd.DataFrame:
    row = {}
    metrics = [
        "auc", "balanced_acc", "acc", "pre", "recall", "specificity", "f1",
        "loss", "cls_loss", "contour_loss", "fractal_loss",
        "mean_dice", "y_dice", "e_dice", "mean_iou", "fractional_alpha",
    ]
    for metric in metrics:
        row[f"{metric}_mean"] = metrics_by_fold[metric].mean()
        row[f"{metric}_std"] = metrics_by_fold[metric].std(ddof=1)
    return pd.DataFrame([row])


def write_summary(config: RunConfig, folds: pd.DataFrame, trained: bool, input_status=None, metrics=None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = folds["group"].value_counts().to_dict()
    lines = [
        "# 002002003002 Run Summary",
        "",
        "## Status",
        "",
        "- Model: F2CG-Net gated contour fusion + differentiable multifractal layer + fractional scale memory",
        "- Direct baselines: 002002002 contour auxiliary and 002002003001 direct contour fusion",
        "- Manual contours are used only for training supervision and fractal consistency targets.",
        "- Test-time input remains raw X-ray image only.",
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
        "## Input Reuse",
        "",
        f"- Source outputs: `{SOURCE_OUTPUT_DIR}`",
    ]
    if input_status:
        for name, status in input_status.items():
            lines.append(f"- `{name}`: {status}")
    lines += [
        "",
        "## Output Files",
        "",
        "- `metadata.csv`",
        "- `folds.csv`",
        "- `outer_splits.csv`",
        "- `contour_targets_224_uint8.npz`",
        "- `contour_target_manifest.json`",
    ]
    if trained:
        lines += ["- `cv_predictions.csv`", "- `cv_metrics_by_fold.csv`", "- `cv_summary.csv`", "- `training_history.csv`"]
    if metrics is not None and not metrics.empty:
        lines += ["", "## Cross-Validation Summary", "", "| Metric | Mean | Std |", "|---|---:|---:|"]
        for metric in ["auc", "balanced_acc", "acc", "pre", "recall", "specificity", "f1", "mean_dice", "fractal_loss", "fractional_alpha", "loss"]:
            lines.append(f"| {metric} | {metrics[f'{metric}_mean'].iloc[0]:.4f} | {metrics[f'{metric}_std'].iloc[0]:.4f} |")
    (OUTPUT_DIR / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke_test(folds, outer_splits, masks_by_row_id, args, config: RunConfig) -> None:
    device = torch.device(config.device)
    train_frame, _, _ = split_frames(folds, outer_splits, fold=0)
    smoke_frame = train_frame.head(min(8, len(train_frame))).copy()
    loader = DataLoader(
        XrayContourDataset(smoke_frame, masks_by_row_id, config.image_size, training=True),
        batch_size=min(4, len(smoke_frame)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config, allow_random_fallback=args.allow_random_fallback).to(device)
    cls_criterion = make_cls_criterion(train_frame, config, device)
    contour_bce = make_contour_bce(config, device)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    masks = batch["mask"].to(device)
    with amp_autocast(device, config.amp):
        cls_logits, contour_logits, extras, loss, cls_loss, c_loss, c_bce, c_dice_loss, fractal_loss = compute_losses(
            model, images, labels, masks, cls_criterion, contour_bce, config
        )
    print(
        json.dumps(
            {
                "smoke_test": "ok",
                "image_shape": list(images.shape),
                "mask_shape": list(masks.shape),
                "cls_logits_shape": list(cls_logits.shape),
                "contour_logits_shape": list(contour_logits.shape),
                "fractal_scale_vector_shape": list(extras["fractal_scale_vector"].shape),
                "fractal_embedding_shape": list(extras["fractal_embedding"].shape),
                "fractional_memory_shape": list(extras["fractional_memory"].shape),
                "fractional_alpha": float(extras["fractional_alpha"].detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "cls_loss": float(cls_loss.detach().cpu()),
                "contour_loss": float(c_loss.detach().cpu()),
                "fractal_loss": float(fractal_loss.detach().cpu()),
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
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-contour", type=float, default=0.3)
    parser.add_argument("--lambda-fractal", type=float, default=0.05)
    parser.add_argument("--contour-pos-weight", type=float, default=50.0)
    parser.add_argument("--fractal-gamma", type=float, default=0.05)
    parser.add_argument("--fractal-embedding-dim", type=int, default=64)
    parser.add_argument("--fractional-embedding-dim", type=int, default=32)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--allow-random-fallback", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--use-pos-weight", dest="use_pos_weight", action="store_true")
    parser.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false")
    parser.add_argument("--detach-fractal-target", dest="detach_fractal_target", action="store_true")
    parser.add_argument("--no-detach-fractal-target", dest="detach_fractal_target", action="store_false")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.set_defaults(pretrained=True, amp=True, use_pos_weight=True, detach_fractal_target=True)
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
        lambda_fractal=args.lambda_fractal,
        contour_pos_weight=args.contour_pos_weight,
        fractal_gamma=args.fractal_gamma,
        fractal_embedding_dim=args.fractal_embedding_dim,
        fractional_embedding_dim=args.fractional_embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
        detach_fractal_target=args.detach_fractal_target,
        pretrained=args.pretrained,
        amp=args.amp,
        use_pos_weight=args.use_pos_weight,
        num_workers=args.num_workers,
        device=args.device,
    )
    set_seed(config.seed)
    configure_model_storage()
    input_status = prepare_local_inputs()
    _, folds, outer_splits, masks_by_row_id = load_inputs()
    write_summary(config, folds, trained=False, input_status=input_status)
    print(f"Loaded {len(folds)} 002002003002 samples from {OUTPUT_DIR}")
    print(folds.groupby(["fold", "group"]).size().unstack(fill_value=0).to_string())

    if args.smoke_test:
        run_smoke_test(folds, outer_splits, masks_by_row_id, args, config)
        return
    if args.prepare_only:
        return

    selected_folds = args.folds if args.folds is not None and len(args.folds) > 0 else list(range(config.n_splits))
    all_metrics, all_predictions, all_history = [], [], []
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
    write_summary(config, folds, trained=True, input_status=input_status, metrics=summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
